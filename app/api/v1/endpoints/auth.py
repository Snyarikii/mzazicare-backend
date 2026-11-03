"""Google identity, contextual risk, and email step-up authentication."""
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from sqlalchemy import case, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import get_current_access_claims, get_current_active_user
from app.core.security import create_access_token, create_refresh_token
from app.db.session import get_db
from app.models.entities import (BehavioralAuditLog, ContextualLogin, SecurityAlert,
                                 TrustEvaluation, User, UserDeviceProfile)
from app.schemas.auth import (AuthTokenResponse, BiometricUnlockRequest,
                              DeviceSessionResponse, GoogleAuthRequest,
                              GoogleProfile, LogoutRequest,
                              NeedsRegistrationResponse, RefreshTokenRequest,
                              ResendMfaRequest, RevokeSessionsResponse,
                              UserResponse, VerifyMfaRequest)
from app.services.email import mfa_email_configured, send_mfa_email
from app.services.risk_engine import risk_engine
from app.services.token_blacklist import (blacklist_refresh_jti, blacklist_token,
                                          is_token_blacklisted)

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger("uvicorn.error")


def _is_admin_email(email: str | None) -> bool:
    return bool(email and settings.ADMIN_EMAIL and
                email.strip().casefold() == settings.ADMIN_EMAIL.strip().casefold())


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
        f"  • Risk action         : {assessment.get('risk_action', 'UNSPECIFIED')}",
        "  • Step-up MFA         : " + (
            "Required (CHALLENGE)" if assessment["step_up_required"] else "No"
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
              anomaly=False, profile: UserDeviceProfile | None = None,
              auth_status: str = "LOGGED_IN") -> AuthTokenResponse:
    return AuthTokenResponse(
        status=auth_status,
        access_token="" if step_up else create_access_token(
            str(user.user_id), jti=profile.access_jti,
            expires_at=profile.access_expires_at,
        ),
        refresh_token=None if step_up else create_refresh_token(
            str(user.user_id), fingerprint, profile.refresh_jti
        ),
        role=user.role,
        user=UserResponse.model_validate(user),
        step_up_required=step_up,
        anomaly_detected=anomaly,
        uts_score=score,
    )


async def _device(db: AsyncSession, user: User, fingerprint: str, *, for_update=False):
    query = select(UserDeviceProfile).where(
        UserDeviceProfile.user_id == user.user_id,
        UserDeviceProfile.device_fingerprint == fingerprint,
    )
    result = await db.execute(query.with_for_update() if for_update else query)
    return result.scalar_one_or_none()


def _risk_simulation_inputs(request: Request, profile: UserDeviceProfile | None,
                            client_ip: str | None):
    sim_ip = request.headers.get("x-simulate-ip-address") or request.headers.get("x-forwarded-for")
    sim_device_state = request.headers.get("x-simulate-device-state")
    risk_profile = profile
    if sim_device_state == "NEW":
        risk_profile = None
    elif sim_device_state == "DORMANT":
        risk_profile = SimpleNamespace(
            trust_state="TRUSTED",
            last_login_at=_utc_now_naive() - timedelta(days=45),
        )
    if sim_ip or sim_device_state:
        logger.info("Risk simulation headers received: device_state=%s, ip_override=%s",
                    sim_device_state or "NORMAL", bool(sim_ip))
    return sim_ip or client_ip, risk_profile


def _audit(db: AsyncSession, user: User, fingerprint: str, ip: str | None,
           geolocation: str | None, score: float, success: bool,
           assessment: dict | None = None):
    db.add(ContextualLogin(
        user_id=user.user_id, device_fingerprint=fingerprint,
        ip_address=ip or "127.0.0.1", geolocation=geolocation,
        login_risk_score=score, is_successful=success,
    ))
    if assessment is not None:
        action = assessment.get("risk_action") or (
            "DENY" if assessment.get("access_denied") else
            "CHALLENGE" if assessment.get("step_up_required") else "ALLOW"
        )
        evaluation = TrustEvaluation(
            evaluation_id=uuid.uuid4(), user_id=user.user_id,
            perimeter_score=score, behavioral_score=score,
            unified_trust_score=score,
            enforcement_action="STEP_UP_MFA" if action == "CHALLENGE" else action,
        )
        db.add(evaluation)
        if action != "ALLOW":
            db.add(SecurityAlert(
                evaluation_id=evaluation.evaluation_id, user_id=user.user_id,
                alert_type="; ".join(assessment.get("reasons", []))[:50] or "Risk evaluation",
                severity_level="HIGH" if action == "DENY" else "MEDIUM",
                shap_explanation=json.dumps(assessment.get("shap_attributions", [])),
            ))


def _clean_label(value, limit: int = 64) -> str | None:
    text = " ".join(str(value or "").split())
    return text[:limit] if text and text.casefold() not in ("unknown", "none", "null") else None


def _observed_ip(request: Request | None, reported: str | None = None) -> str | None:
    """Prefer the address the server saw; fall back to the client-reported one."""
    if request is not None and request.client and request.client.host:
        return request.client.host[:45]
    return (reported or None) and reported[:45]


def _apply_session_metadata(profile: UserDeviceProfile, raw: dict | None,
                            ip: str | None) -> None:
    """Refresh the labels and presence shown in the Active Devices & Sessions list."""
    raw = raw or {}
    brand = _clean_label(raw.get("device_brand"))
    model = _clean_label(raw.get("device_model"))
    name = _clean_label(raw.get("device_name")) or (
        _clean_label(f"{brand or ''} {model or ''}") if model else None
    )
    identifier = _clean_label(raw.get("model_id")) or model
    if name:
        profile.device_name = name
    if identifier:
        profile.model_identifier = identifier
    if ip:
        profile.last_ip_address = ip
    profile.last_seen_at = datetime.now(timezone.utc)


def _enroll_or_update(db: AsyncSession, user: User, fingerprint: str,
                      profile: UserDeviceProfile | None, raw: dict,
                      *, mfa=False, ip: str | None = None):
    now = _utc_now_naive()
    if profile is None:
        profile = UserDeviceProfile(
            user_id=user.user_id, device_fingerprint=fingerprint,
            trust_state="PROBATIONARY", successful_login_count=1,
            device_model=(f"{raw.get('device_brand') or ''} {raw.get('device_model') or ''}".strip() or "Unknown")[:64],
            os_name=_device_os_label(raw),
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
    _apply_session_metadata(profile, raw, ip)
    profile.refresh_jti = secrets.token_urlsafe(24)
    profile.access_jti = str(uuid.uuid4())
    profile.access_expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return profile


def _device_os_label(raw: dict) -> str:
    """Keep the display label within user_device_profiles.os_name's 32 characters."""
    name = str(raw.get("os_name") or "").strip()
    version = str(raw.get("os_version") or "").strip()
    lower_name = name.casefold()
    if "release-keys" in lower_name or "test-keys" in lower_name:
        name = "Android"
    return (f"{name} {version}".strip() or "Unknown")[:32]


async def _replace_device_session(db: AsyncSession, user: User, fingerprint: str,
                                  profile: UserDeviceProfile | None, raw: dict,
                                  *, mfa=False, ip: str | None = None):
    if profile is not None and profile.access_jti and profile.access_expires_at:
        await blacklist_token(profile.access_jti, str(user.user_id),
                              profile.access_expires_at, db=db)
    return _enroll_or_update(db, user, fingerprint, profile, raw, mfa=mfa, ip=ip)


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
    role = "ADMIN" if _is_admin_email(email) else role
    user = User(
        email=email, full_name=name, role=role, status="ACTIVE",
        auth_provider="google", google_sub=sub, license_number=license_number,
        is_verified=role in ("family", "ADMIN") or license_number is not None,
        verification_status="APPROVED",
    )
    db.add(user)
    try:
        await db.flush()
        profile = await _replace_device_session(db, user, fingerprint, None, raw or {},
                                                ip=ip_address)
        _audit(db, user, fingerprint, ip_address, geolocation, 0.10, True)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Google account is already registered.") from exc
    return _response(user, fingerprint, 0.10, profile=profile)


@router.post("/google", response_model=AuthTokenResponse | NeedsRegistrationResponse)
async def google_auth(payload: GoogleAuthRequest, background_tasks: BackgroundTasks,
                      request: Request, db: AsyncSession = Depends(get_db)):
    email, name, sub = await verify_google_id_token(payload.id_token)

    user = await find_google_account(db, email, sub)
    raw = payload.raw_features or {}
    if user is None:
        if _is_admin_email(email):
            return await create_google_account(
                db, email=email, name=name, sub=sub, role="ADMIN",
                fingerprint=payload.device_fingerprint, raw=raw,
                ip_address=payload.ip_address, geolocation=payload.geolocation,
            )
        if payload.role is not None and payload.role not in ("family", "caregiver"):
            raise HTTPException(status_code=400, detail="Select Family Member or Caregiver to register.")
        return NeedsRegistrationResponse(
            google_profile=GoogleProfile(email=email, name=name, google_id=sub)
        )

    if user.status != "ACTIVE":
        raise HTTPException(status_code=403, detail="Account is inactive.")
    if user.google_sub and user.google_sub != sub:
        raise HTTPException(status_code=401, detail="Google account does not match this user.")
    if _is_admin_email(email) and _is_admin_email(user.email):
        user.role = "ADMIN"
    # A verified Google email migrates an existing password-era account once.
    user.google_sub = sub
    user.auth_provider = "google"
    if name:
        user.full_name = name
    profile = await _device(db, user, payload.device_fingerprint)
    if profile is not None and profile.trust_state == "REVOKED":
        raise HTTPException(status_code=403, detail="Device access has been revoked.")
    risk_ip, risk_profile = _risk_simulation_inputs(request, profile, payload.ip_address)
    assessment = await risk_engine.evaluate_risk_and_revoke(
        raw_features=raw, ip_address=risk_ip, device_profile=risk_profile,
        current_jti=profile.access_jti if profile else None,
        user_id=str(user.user_id),
        expires_at=profile.access_expires_at if profile else None,
    )
    _log_risk_assessment("Google sign-in", user.user_id, assessment)
    score = assessment["uts_score"]
    if assessment.get("access_denied", False):
        if profile is not None:
            profile.refresh_jti = None
            profile.access_jti = None
            profile.access_expires_at = None
        _audit(db, user, payload.device_fingerprint, payload.ip_address,
               payload.geolocation, score, False, assessment)
        await db.commit()
        return JSONResponse(status_code=403, content={
            "status": "ACCESS_DENIED",
            "access_denied": True,
            "detail": {"message": "Sign-in denied by risk assessment."},
        })
    if assessment["step_up_required"]:
        if profile is not None:
            profile.refresh_jti = None
        _challenge(user, payload.device_fingerprint, raw, background_tasks)
        _audit(db, user, payload.device_fingerprint, payload.ip_address,
               payload.geolocation, score, False, assessment)
        await db.commit()
        return _response(user, payload.device_fingerprint, score, step_up=True,
                         anomaly=assessment["anomaly_detected"], auth_status="MFA_REQUIRED")

    profile = await _replace_device_session(db, user, payload.device_fingerprint, profile, raw,
                                            ip=_observed_ip(request, payload.ip_address))
    _audit(db, user, payload.device_fingerprint, payload.ip_address,
           payload.geolocation, score, True, assessment)
    await db.commit()
    return _response(user, payload.device_fingerprint, score, profile=profile,
                     auth_status="SUCCESS")


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
                           request: Request, db: AsyncSession = Depends(get_db)):
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
    risk_ip, risk_profile = _risk_simulation_inputs(request, profile, payload.ip_address)
    assessment = await risk_engine.evaluate_risk_and_revoke(
        raw_features=payload.raw_features, ip_address=risk_ip,
        device_profile=risk_profile, current_jti=profile.access_jti,
        user_id=str(user.user_id), expires_at=profile.access_expires_at,
    )
    _log_risk_assessment("Biometric unlock", user.user_id, assessment)
    score = assessment["uts_score"]
    if assessment.get("access_denied", False):
        profile.refresh_jti = None
        profile.access_jti = None
        profile.access_expires_at = None
        _audit(db, user, payload.device_fingerprint, payload.ip_address,
               None, score, False, assessment)
        await db.commit()
        raise HTTPException(status_code=403, detail="Sign-in denied by risk assessment.")
    if assessment["step_up_required"]:
        profile.refresh_jti = None
        _challenge(user, payload.device_fingerprint, payload.raw_features, background_tasks)
        _audit(db, user, payload.device_fingerprint, payload.ip_address, None, score, False, assessment)
        await db.commit()
        return _response(user, payload.device_fingerprint, score, step_up=True,
                         anomaly=assessment["anomaly_detected"], auth_status="MFA_REQUIRED")
    profile = await _replace_device_session(db, user, payload.device_fingerprint,
                                            profile, payload.raw_features,
                                            ip=_observed_ip(request, payload.ip_address))
    _audit(db, user, payload.device_fingerprint, payload.ip_address, None, score, True, assessment)
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
    profile = await _replace_device_session(
        db, user, payload.device_fingerprint, profile,
        {"device_model": payload.device_model, "device_brand": payload.device_brand,
         "device_name": payload.device_name, "model_id": payload.model_id,
         "os_name": payload.os_name}, mfa=True,
    )
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


@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh_session(payload: RefreshTokenRequest, request: Request,
                          db: AsyncSession = Depends(get_db)):
    try:
        claims = jwt.decode(
            payload.refresh_token, settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"require": ["exp", "sub", "jti", "token_type", "device_fingerprint"]},
        )
        if claims["token_type"] != "refresh":
            raise ValueError("Wrong token type")
        user_id = uuid.UUID(claims["sub"])
        refresh_jti = claims["jti"]
        fingerprint = claims["device_fingerprint"]
        if not isinstance(refresh_jti, str) or not refresh_jti or not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Incomplete refresh token")
    except (jwt.PyJWTError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.") from exc

    if await is_token_blacklisted(refresh_jti, db=db):
        raise HTTPException(status_code=401, detail="Refresh token has been revoked.")

    user = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="Account is unavailable.")

    profile = await _device(db, user, fingerprint, for_update=True)
    if (profile is None or profile.trust_state == "REVOKED" or not profile.refresh_jti
            or not secrets.compare_digest(profile.refresh_jti, refresh_jti)):
        raise HTTPException(status_code=401, detail="Device session is unavailable.")
    header_fingerprint = request.headers.get("x-device-fingerprint")
    if header_fingerprint and not secrets.compare_digest(header_fingerprint, fingerprint):
        raise HTTPException(status_code=401, detail="Device session is unavailable.")

    if profile.access_jti and profile.access_expires_at:
        await blacklist_token(profile.access_jti, str(user_id),
                              profile.access_expires_at, db=db)
    profile.access_jti = str(uuid.uuid4())
    profile.access_expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    profile.refresh_jti = secrets.token_urlsafe(24)

    client_ip = request.client.host if request.client else None
    _apply_session_metadata(profile, None, client_ip)
    device_headers = {
        name: request.headers[name][:256]
        for name in ("user-agent", "x-device-id", "x-device-model", "x-device-os",
                     "x-device-fingerprint") if name in request.headers
    }
    db.add(BehavioralAuditLog(
        user_id=user.user_id, anomaly_score=0.0, is_anomaly=False,
        session_revocation_recommended=False,
        risk_factors=[], telemetry_snapshot={
            "event": "SESSION_REFRESH", "ip_address": client_ip,
            "device_fingerprint": fingerprint, "device_headers": device_headers,
        },
    ))
    await db.commit()
    return _response(user, fingerprint, 0.10, profile=profile)


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
        if profile.access_jti and profile.access_expires_at:
            await blacklist_token(profile.access_jti, str(user_id),
                                  profile.access_expires_at, db=db)
        profile.refresh_jti = None
        profile.access_jti = None
        profile.access_expires_at = None
        await db.commit()
    return {"message": "Session ended."}


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _is_current_session(profile: UserDeviceProfile, access_jti: str) -> bool:
    return bool(profile.access_jti) and secrets.compare_digest(profile.access_jti, access_jti)


async def _active_device_sessions(db: AsyncSession, user_id: uuid.UUID, *, for_update=False):
    """Device profiles that still hold a live refresh session."""
    query = select(UserDeviceProfile).where(
        UserDeviceProfile.user_id == user_id,
        UserDeviceProfile.refresh_jti.is_not(None),
        UserDeviceProfile.trust_state != "REVOKED",
    )
    result = await db.execute(query.with_for_update() if for_update else query)
    return list(result.scalars().all())


def _session_view(profile: UserDeviceProfile, is_current: bool) -> DeviceSessionResponse:
    return DeviceSessionResponse(
        id=profile.id,
        device_name=profile.device_name or profile.device_model or "Unknown device",
        model=profile.model_identifier or profile.device_model,
        ip_address=profile.last_ip_address,
        last_seen=_as_utc(profile.last_seen_at or profile.last_login_at),
        is_current=is_current,
    )


@router.get("/sessions", response_model=list[DeviceSessionResponse])
async def list_active_sessions(request: Request,
                               claims: dict = Depends(get_current_access_claims),
                               current_user: User = Depends(get_current_active_user),
                               db: AsyncSession = Depends(get_db)):
    """List the caller's active refresh sessions, current device first."""
    profiles = await _active_device_sessions(db, current_user.user_id)
    current = next((p for p in profiles if _is_current_session(p, claims["jti"])), None)
    if current is not None:
        # The caller is demonstrably online; keep its presence fresh.
        _apply_session_metadata(current, None, _observed_ip(request))
        await db.commit()
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    others = sorted((p for p in profiles if p is not current),
                    key=lambda p: _as_utc(p.last_seen_at or p.last_login_at) or epoch,
                    reverse=True)
    ordered = ([current] if current is not None else []) + others
    return [_session_view(p, p is current) for p in ordered]


@router.post("/revoke-all-sessions", response_model=RevokeSessionsResponse)
async def revoke_all_other_sessions(request: Request,
                                    claims: dict = Depends(get_current_access_claims),
                                    current_user: User = Depends(get_current_active_user),
                                    db: AsyncSession = Depends(get_db)):
    """Revoke every refresh session except the one behind the caller's bearer token."""
    profiles = await _active_device_sessions(db, current_user.user_id, for_update=True)
    current = next((p for p in profiles if _is_current_session(p, claims["jti"])), None)
    if current is None:
        # Never fall through to revoking everything, including the caller's own device.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Current device session could not be identified. Sign in again.")

    refresh_expiry = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    revoked = []
    for profile in profiles:
        if profile is current:
            continue
        if profile.access_jti and profile.access_expires_at:
            await blacklist_token(profile.access_jti, str(current_user.user_id),
                                  profile.access_expires_at, db=db)
        if profile.refresh_jti:
            await blacklist_refresh_jti(profile.refresh_jti, str(current_user.user_id),
                                        refresh_expiry, db=db)
        profile.refresh_jti = None
        profile.access_jti = None
        profile.access_expires_at = None
        revoked.append(profile)

    client_ip = _observed_ip(request)
    db.add(BehavioralAuditLog(
        user_id=current_user.user_id, anomaly_score=0.0, is_anomaly=False,
        session_revocation_recommended=False, risk_factors=[],
        telemetry_snapshot={
            "event": "SESSIONS_REVOKED_REMOTELY",
            "ip_address": client_ip,
            "initiating_session_id": current.id,
            "initiating_device_fingerprint": current.device_fingerprint,
            "revoked_count": len(revoked),
            "revoked_sessions": [
                {"id": p.id, "device_name": p.device_name or p.device_model,
                 "last_ip_address": p.last_ip_address}
                for p in revoked
            ],
        },
    ))
    await db.commit()
    count = len(revoked)
    logger.info("SESSIONS_REVOKED_REMOTELY user=%s revoked=%d", current_user.user_id, count)
    return RevokeSessionsResponse(
        revoked_count=count,
        message=(f"Signed out {count} other session{'s' if count != 1 else ''}."
                 if count else "No other active sessions."),
    )
