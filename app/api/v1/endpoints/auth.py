"""Google identity, contextual risk, and email step-up authentication."""
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from sqlalchemy import case, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import create_access_token, create_refresh_token
from app.db.session import get_db
from app.models.entities import ContextualLogin, User, UserDeviceProfile
from app.schemas.auth import (AuthTokenResponse, BiometricUnlockRequest,
                              GoogleAuthRequest, GoogleProfile, LogoutRequest,
                              NeedsRegistrationResponse, ResendMfaRequest, UserResponse,
                              VerifyMfaRequest)
from app.services.email import mfa_email_configured, send_mfa_email
from app.services.risk_engine import risk_engine

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger("uvicorn.error")


def _log_risk_assessment(method: str, user_id: uuid.UUID, assessment: dict,
                         *, source: str = "Risk engine") -> None:
    """Print one readable decision block without credentials or raw telemetry."""
    base_risk = assessment.get("base_ml_risk")
    adjustment = assessment.get("device_adjustment")
    lines = [
        "\n" + "🎯 " * 8 + " AUTHENTICATION RISK ASSESSMENT " + "🎯 " * 8,
        f"  • Sign-in method      : {method}",
        f"  • User ID             : {user_id}",
        f"  • Score source        : {source}",
        f"  • Engine base risk    : {base_risk:.2f}" if base_risk is not None
        else "  • Engine base risk    : n/a",
        f"  • Device adjustment   : {adjustment:+.2f}" if adjustment is not None
        else "  • Device adjustment   : n/a",
        f"  • Unified Trust Score : {assessment['uts_score']:.2f}",
        "  • Step-up MFA         : " + (
            "Required (UTS ≥ 0.55)" if assessment["step_up_required"] else "No"
        ),
        "  • Anomaly detected    : " + (
            "Yes" if assessment["anomaly_detected"] else "No"
        ),
        "  • Risk reasons        :",
    ]
    for index, reason in enumerate(
        assessment.get("reasons") or ["No specific risk modifier was reported"], 1
    ):
        lines.append(f"      {index}. {reason}")
    lines.append("--------------------------------------------------------------")
    logger.info("%s", "\n".join(lines))


def _utc_now_naive():
    # Device-profile and MFA columns are existing timezone-naive database columns.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _response(user: User, fingerprint: str, score: float, *, step_up=False,
              anomaly=False, profile: UserDeviceProfile | None = None) -> AuthTokenResponse:
    return AuthTokenResponse(
        access_token="" if step_up else create_access_token(str(user.user_id)),
        refresh_token=None if step_up else create_refresh_token(
            str(user.user_id), fingerprint, profile.refresh_jti
        ),
        role=user.role,
        user=UserResponse.model_validate(user),
        step_up_required=step_up,
        anomaly_detected=anomaly,
        uts_score=score,
    )


async def _device(db: AsyncSession, user: User, fingerprint: str):
    result = await db.execute(select(UserDeviceProfile).where(
        UserDeviceProfile.user_id == user.user_id,
        UserDeviceProfile.device_fingerprint == fingerprint,
    ))
    return result.scalar_one_or_none()


def _audit(db: AsyncSession, user: User, fingerprint: str, ip: str | None,
           geolocation: str | None, score: float, success: bool):
    db.add(ContextualLogin(
        user_id=user.user_id, device_fingerprint=fingerprint,
        ip_address=ip or "127.0.0.1", geolocation=geolocation,
        login_risk_score=score, is_successful=success,
    ))


def _enroll_or_update(db: AsyncSession, user: User, fingerprint: str,
                      profile: UserDeviceProfile | None, raw: dict,
                      *, mfa=False):
    now = _utc_now_naive()
    if profile is None:
        profile = UserDeviceProfile(
            user_id=user.user_id, device_fingerprint=fingerprint,
            trust_state="PROBATIONARY", successful_login_count=1,
            device_model=f"{raw.get('device_brand', '')} {raw.get('device_model', '')}".strip() or "Unknown",
            os_name=f"{raw.get('os_name', '')} {raw.get('os_version', '')}".strip() or "Unknown",
            first_seen_at=now, last_login_at=now,
            last_mfa_completed_at=now if mfa else None,
        )
        db.add(profile)
    else:
        profile.successful_login_count += 1
        profile.last_login_at = now
        if mfa:
            profile.last_mfa_completed_at = now
        if profile.trust_state == "PROBATIONARY" and profile.successful_login_count >= 3:
            profile.trust_state = "TRUSTED"
    profile.refresh_jti = secrets.token_urlsafe(24)
    return profile


def _challenge(user: User, fingerprint: str, raw: dict,
               background_tasks: BackgroundTasks):
    if not mfa_email_configured():
        raise HTTPException(status_code=503, detail="Email MFA is not configured.")
    otp = "".join(str(secrets.randbelow(10)) for _ in range(6))
    user.mfa_code = otp
    user.mfa_expires_at = _utc_now_naive() + timedelta(minutes=10)
    user.mfa_device_fingerprint = fingerprint
    background_tasks.add_task(
        send_mfa_email, to_email=user.email, full_name=user.full_name,
        otp_code=otp,
        device_model=f"{raw.get('device_brand', '')} {raw.get('device_model', '')}".strip() or "Mobile Device",
    )


async def verify_google_id_token(id_token: str) -> tuple[str, str, str]:
    configured_ids = settings.GOOGLE_CLIENT_IDS or settings.GOOGLE_CLIENT_ID
    client_ids = [value.strip() for value in configured_ids.split(",") if value.strip()]
    if not client_ids:
        raise HTTPException(status_code=503, detail="Google Sign-In is not configured.")
    try:
        claims = await run_in_threadpool(
            google_id_token.verify_oauth2_token, id_token, google_requests.Request()
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid Google ID token.") from exc
    if claims.get("aud") not in client_ids or not claims.get("email_verified"):
        raise HTTPException(status_code=401, detail="Unverified Google identity.")
    email, name, sub = claims.get("email"), claims.get("name"), claims.get("sub")
    if not email or not sub:
        raise HTTPException(status_code=401, detail="Google profile is incomplete.")
    return email, name or email, sub


async def find_google_account(db: AsyncSession, email: str, sub: str) -> User | None:
    result = await db.execute(
        select(User)
        .where((User.google_sub == sub) | (User.email == email))
        .order_by(case((User.google_sub == sub, 0), else_=1))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def create_google_account(
    db: AsyncSession, *, email: str, name: str, sub: str, role: str,
    fingerprint: str, raw: dict | None = None, ip_address: str | None = None,
    geolocation: str | None = None, license_number: str | None = None,
) -> AuthTokenResponse:
    user = User(
        email=email, full_name=name, role=role, status="ACTIVE",
        auth_provider="google", google_sub=sub, license_number=license_number,
        is_verified=role == "family" or license_number is not None,
        verification_status="APPROVED",
    )
    db.add(user)
    try:
        await db.flush()
        profile = _enroll_or_update(db, user, fingerprint, None, raw or {})
        _audit(db, user, fingerprint, ip_address, geolocation, 0.10, True)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Google account is already registered.") from exc
    return _response(user, fingerprint, 0.10, profile=profile)


@router.post("/google", response_model=AuthTokenResponse | NeedsRegistrationResponse)
async def google_auth(payload: GoogleAuthRequest, background_tasks: BackgroundTasks,
                      db: AsyncSession = Depends(get_db)):
    email, name, sub = await verify_google_id_token(payload.id_token)

    user = await find_google_account(db, email, sub)
    raw = payload.raw_features or {}
    if user is None:
        if payload.role is not None and payload.role not in ("family", "caregiver"):
            raise HTTPException(status_code=400, detail="Select Family Member or Caregiver to register.")
        return NeedsRegistrationResponse(
            google_profile=GoogleProfile(email=email, name=name, google_id=sub)
        )

    if user.status != "ACTIVE":
        raise HTTPException(status_code=403, detail="Account is inactive.")
    if user.google_sub and user.google_sub != sub:
        raise HTTPException(status_code=401, detail="Google account does not match this user.")
    # A verified Google email migrates an existing password-era account once.
    user.google_sub = sub
    user.auth_provider = "google"
    if name:
        user.full_name = name
    profile = await _device(db, user, payload.device_fingerprint)
    if profile is not None and profile.trust_state == "REVOKED":
        raise HTTPException(status_code=403, detail="Device access has been revoked.")
    assessment = risk_engine.evaluate_risk(raw_features=raw,
                                           ip_address=payload.ip_address,
                                           device_profile=profile)
    _log_risk_assessment("Google sign-in", user.user_id, assessment)
    score = assessment["uts_score"]
    if assessment["step_up_required"]:
        if profile is not None:
            profile.refresh_jti = None
        _challenge(user, payload.device_fingerprint, raw, background_tasks)
        _audit(db, user, payload.device_fingerprint, payload.ip_address,
               payload.geolocation, score, False)
        await db.commit()
        return _response(user, payload.device_fingerprint, score, step_up=True,
                         anomaly=assessment["anomaly_detected"])

    profile = _enroll_or_update(db, user, payload.device_fingerprint, profile, raw)
    _audit(db, user, payload.device_fingerprint, payload.ip_address,
           payload.geolocation, score, True)
    await db.commit()
    return _response(user, payload.device_fingerprint, score, profile=profile)


@router.post("/register-family", response_model=AuthTokenResponse)
async def register_family(payload: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    email, name, sub = await verify_google_id_token(payload.id_token)
    existing = await find_google_account(db, email, sub)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Google account is already registered.")
    return await create_google_account(
        db, email=email, name=name, sub=sub, role="family",
        fingerprint=payload.device_fingerprint, raw=payload.raw_features,
        ip_address=payload.ip_address, geolocation=payload.geolocation,
    )


@router.post("/biometric-unlock", response_model=AuthTokenResponse)
async def biometric_unlock(payload: BiometricUnlockRequest,
                           background_tasks: BackgroundTasks,
                           db: AsyncSession = Depends(get_db)):
    try:
        claims = jwt.decode(payload.refresh_token, settings.SECRET_KEY,
                            algorithms=[settings.ALGORITHM])
        if claims.get("token_type") != "refresh" or claims.get("device_fingerprint") != payload.device_fingerprint:
            raise ValueError("Refresh token is not bound to this device")
        user_id = uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.") from exc
    user = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
    if user is None or user.status != "ACTIVE" or user.auth_provider != "google":
        raise HTTPException(status_code=401, detail="Account is unavailable.")
    profile = await _device(db, user, payload.device_fingerprint)
    if (profile is None or profile.trust_state == "REVOKED"
            or not profile.refresh_jti
            or not secrets.compare_digest(profile.refresh_jti, claims.get("jti", ""))):
        raise HTTPException(status_code=401, detail="Device is not enrolled.")
    assessment = risk_engine.evaluate_risk(raw_features=payload.raw_features,
                                           ip_address=payload.ip_address,
                                           device_profile=profile)
    _log_risk_assessment("Biometric unlock", user.user_id, assessment)
    score = assessment["uts_score"]
    if assessment["step_up_required"]:
        profile.refresh_jti = None
        _challenge(user, payload.device_fingerprint, payload.raw_features, background_tasks)
        _audit(db, user, payload.device_fingerprint, payload.ip_address, None, score, False)
        await db.commit()
        return _response(user, payload.device_fingerprint, score, step_up=True,
                         anomaly=assessment["anomaly_detected"])
    profile = _enroll_or_update(db, user, payload.device_fingerprint, profile, payload.raw_features)
    _audit(db, user, payload.device_fingerprint, payload.ip_address, None, score, True)
    await db.commit()
    return _response(user, payload.device_fingerprint, score, profile=profile)


@router.post("/verify-mfa", response_model=AuthTokenResponse)
async def verify_mfa_challenge(payload: VerifyMfaRequest,
                               db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.email == payload.email))).scalar_one_or_none()
    if user is None or user.auth_provider != "google" or user.status != "ACTIVE":
        raise HTTPException(status_code=401, detail="Invalid MFA challenge.")
    if (not user.mfa_code or not secrets.compare_digest(user.mfa_code, payload.otp_code.strip())
            or user.mfa_device_fingerprint != payload.device_fingerprint):
        raise HTTPException(status_code=400, detail="Invalid verification code.")
    if user.mfa_expires_at is None or _utc_now_naive() > user.mfa_expires_at:
        user.mfa_code = None
        user.mfa_expires_at = None
        user.mfa_device_fingerprint = None
        await db.commit()
        raise HTTPException(status_code=400, detail="Verification code has expired.")
    user.mfa_code = None
    user.mfa_expires_at = None
    user.mfa_device_fingerprint = None
    profile = await _device(db, user, payload.device_fingerprint)
    profile = _enroll_or_update(db, user, payload.device_fingerprint, profile,
                                {"device_model": payload.device_model, "os_name": payload.os_name}, mfa=True)
    _audit(db, user, payload.device_fingerprint, None, None, 0.10, True)
    await db.commit()
    return _response(user, payload.device_fingerprint, 0.10, profile=profile)


@router.post("/resend-mfa")
async def resend_mfa_code(payload: ResendMfaRequest,
                          background_tasks: BackgroundTasks,
                          db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.email == payload.email))).scalar_one_or_none()
    if user is None or not user.mfa_code or not user.mfa_device_fingerprint:
        raise HTTPException(status_code=400, detail="No active MFA challenge.")
    _challenge(user, user.mfa_device_fingerprint, {}, background_tasks)
    await db.commit()
    return {"message": "A fresh verification code has been sent to your email."}


@router.post("/logout")
async def logout(payload: LogoutRequest, db: AsyncSession = Depends(get_db)):
    try:
        claims = jwt.decode(payload.refresh_token, settings.SECRET_KEY,
                            algorithms=[settings.ALGORITHM])
        if claims.get("token_type") != "refresh":
            raise ValueError("Wrong token type")
        user_id = uuid.UUID(claims["sub"])
        fingerprint = claims["device_fingerprint"]
    except (jwt.PyJWTError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid refresh token.") from exc
    result = await db.execute(select(UserDeviceProfile).where(
        UserDeviceProfile.user_id == user_id,
        UserDeviceProfile.device_fingerprint == fingerprint,
    ))
    profile = result.scalar_one_or_none()
    if profile and profile.refresh_jti and secrets.compare_digest(
        profile.refresh_jti, claims.get("jti", "")
    ):
        profile.refresh_jti = None
        await db.commit()
    return {"message": "Session ended."}
