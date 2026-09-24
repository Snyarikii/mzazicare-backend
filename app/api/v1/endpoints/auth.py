import uuid
import random
import string
import json
import secrets
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, get_password_hash, verify_password
from app.db.session import get_db
from app.models.entities import ContextualLogin, User, UserDeviceProfile
from app.schemas.auth import AuthTokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse, RegisterInitiateRequest, VerifyEmailRequest, VerifyMfaRequest, ResendMfaRequest
from app.models.entities import PendingRegistration
from app.services.email import send_verification_email, send_mfa_email
from app.services.risk_engine import risk_engine

router = APIRouter(prefix="/auth", tags=["Authentication"])

def generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(payload: UserRegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check duplicate email
    query = select(User).where(User.email == payload.email)
    result = await db.execute(query)
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    new_user = User(
        email=payload.email,
        full_name=payload.full_name,
        password_hash=get_password_hash(payload.password),
        role=payload.role,
        status="ACTIVE",
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

@router.post("/login", response_model=AuthTokenResponse)
async def login_user(
    payload: UserLoginRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    # 1. Retrieve user & verify password
    query = select(User).where(User.email == payload.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    # 2. Extract Perimeter Telemetry
    raw = payload.raw_features or {}

    print("\n" + "🛡️ " * 15 + " INCOMING PERIMETER TELEMETRY " + "🛡️ " * 15, flush=True)
    print(f"👤 User:              {user.email} (ID: {user.user_id})", flush=True)
    print(f"📱 Hardware ID:       {payload.device_fingerprint}", flush=True)
    print(f"📟 Model & OS:        {raw.get('device_brand')} {raw.get('device_model')} running {raw.get('os_name')} {raw.get('os_version')}", flush=True)
    print(f"🌐 Network:           {raw.get('network_type')} | IP: {payload.ip_address}", flush=True)
    print(f"📍 GPS Location:      Lat: {raw.get('latitude')}, Lng: {raw.get('longitude')} (Accuracy: {raw.get('gps_accuracy_meters')}m)", flush=True)
    print(f"🚨 Mocked GPS:        {raw.get('is_mock_location')}", flush=True)
    print(f"🔋 Battery / State:   {raw.get('battery_level')}% | Charging: {raw.get('is_charging')}", flush=True)
    print(f"⏰ Temporal Context:  Hour {raw.get('local_hour')}:00, DayOfWeek: {raw.get('day_of_week')}, TZ: {raw.get('timezone')}", flush=True)
    print("=" * 72 + "\n", flush=True)

    # 3. Query Device Profile
    device_stmt = select(UserDeviceProfile).where(
        UserDeviceProfile.user_id == user.user_id,
        UserDeviceProfile.device_fingerprint == payload.device_fingerprint,
    )
    device_profile = (await db.execute(device_stmt)).scalar_one_or_none()

    # 4. Evaluate risk with device context
    risk_assessment = risk_engine.evaluate_risk(
        raw_features=raw,
        ip_address=payload.ip_address,
        device_profile=device_profile,
    )

    risk_score = risk_assessment["uts_score"]
    anomaly_detected = risk_assessment["anomaly_detected"]
    step_up_required = risk_assessment["step_up_required"]
    reasons = risk_assessment["reasons"]

    # ML Risk engine results
    print("\n" + "🤖 " * 12 + " ML RISK ENGINE EVALUATION " + "🤖 " * 12, flush=True)
    print(f"🎯 Unified Trust Score (UTS): {risk_score}", flush=True)
    print(f"⚠️  Anomaly Detected:          {anomaly_detected}", flush=True)
    print(f"🔐 Step-up MFA Required:      {step_up_required}", flush=True)
    print(f"📋 Risk Reasoning:           {'; '.join(reasons)}", flush=True)
    print("=" * 66 + "\n", flush=True)

    # =========================================================================
    # BRANCH A: HIGH RISK / ELEVATED TELEMETRY (Step-up MFA Challenge)
    # =========================================================================
    if step_up_required:
        # 1. Generate 6-digit numeric OTP and set 10-min expiration
        otp = "".join(str(secrets.randbelow(10)) for _ in range(6))
        user.mfa_code = otp
        user.mfa_expires_at = datetime.utcnow() + timedelta(minutes=10)

        # 2. Record audit log as pending / unverified
        login_log = ContextualLogin(
            user_id=user.user_id,
            ip_address=payload.ip_address or "127.0.0.1",
            device_fingerprint=payload.device_fingerprint,
            geolocation=payload.geolocation,
            login_risk_score=risk_score,
            is_successful=False,
        )
        db.add(login_log)
        await db.commit()

        # 3. Schedule email dispatch via background threadpool (non-blocking)
        device_label = f"{raw.get('device_brand', '')} {raw.get('device_model', '')}".strip() or "Mobile Device"
        user_name = getattr(user, "full_name", None) or getattr(user, "name", None) or "Caregiver"

        background_tasks.add_task(
            send_mfa_email,
            to_email=user.email,
            full_name=user_name,
            otp_code=otp,
            device_model=device_label,
        )

        print(f"🔑 [MFA CHALLENGE] Dispatched OTP {otp} to {user.email}", flush=True)

        # 4. Return instructions to show Security Verification screen (NO ACCESS TOKEN)
        return AuthTokenResponse(
            access_token="",
            token_type="bearer",
            role=user.role,
            user=UserResponse.model_validate(user),
            step_up_required=True,
            anomaly_detected=anomaly_detected,
            uts_score=risk_score,
        )

    # =========================================================================
    # BRANCH B: LOW RISK (Tier 1 Direct Pass)
    # =========================================================================
    model_name = f"{raw.get('device_brand', '')} {raw.get('device_model', '')}".strip()
    os_info = f"{raw.get('os_name', '')} {raw.get('os_version', '')}".strip()

    if device_profile is None:
        device_profile = UserDeviceProfile(
            user_id=user.user_id,
            device_fingerprint=payload.device_fingerprint,
            trust_state="PROBATIONARY",
            successful_login_count=1,
            device_model=model_name or "Unknown",
            os_name=os_info or "Unknown",
            first_seen_at=datetime.utcnow(),
            last_login_at=datetime.utcnow(),
        )
        db.add(device_profile)
    else:
        device_profile.successful_login_count += 1
        device_profile.last_login_at = datetime.utcnow()

        if device_profile.trust_state == "PROBATIONARY" and device_profile.successful_login_count >= 3:
            device_profile.trust_state = "TRUSTED"
            print(f"⭐ [Device Trust] Device {payload.device_fingerprint[:8]}... PROMOTED to TRUSTED!", flush=True)

    # Record successful contextual login
    login_log = ContextualLogin(
        user_id=user.user_id,
        ip_address=payload.ip_address or "127.0.0.1",
        device_fingerprint=payload.device_fingerprint,
        geolocation=payload.geolocation,
        login_risk_score=risk_score,
        is_successful=True,
    )
    db.add(login_log)
    await db.commit()

    # Issue JWT token
    token = create_access_token(subject=str(user.user_id))

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        user=UserResponse.model_validate(user),
        step_up_required=False,
        anomaly_detected=False,
        uts_score=risk_score,
    )

@router.post("/register-initiate", status_code=status.HTTP_200_OK)
async def register_initiate(
    payload: RegisterInitiateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    # Ensure user is not already registered in primary table
    user_check = await db.execute(select(User).where(User.email == payload.email))
    if user_check.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists."
        )

    # Clear any existing pending registration for this email
    await db.execute(delete(PendingRegistration).where(PendingRegistration.email == payload.email))

    # Generate 6-digit OTP valid for 10mins
    otp = generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    pending_user = PendingRegistration(
        email=payload.email,
        full_name=payload.full_name,
        password_hash=get_password_hash(payload.password),
        role=payload.role,
        otp_code=otp,
        expires_at=expires_at,
    )
    db.add(pending_user)
    await db.commit()

    # Send email in the background
    background_tasks.add_task(
        send_verification_email,
        to_email=payload.email,
        full_name=payload.full_name,
        otp_code=otp
    )
    return {"message": "Verification code sent to email."}

    # # Print OTP to terminal (replace with SMTP/SendGrid in production)
    # print("\n" + "=" * 50)
    # print(f"VERIFICATION CODE FOR {payload.email}: {otp}")
    # print("=" * 50 + "\n")

@router.post("/verify-email", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
async def verify_email(payload: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    query = select(PendingRegistration).where(PendingRegistration.email == payload.email)
    result = await db.execute(query)
    pending = result.scalar_one_or_none()

    if not pending:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No registration pending for this email. Please sign up again."
        )

    # Check if OTP is valid and not expired
    if pending.otp_code != payload.otp_code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code."
        )

    if datetime.now(timezone.utc) > pending.expires_at:
        await db.delete(pending)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification code has expired. Please register again."
        )

    # Insert into active User table
    new_user = User(
        email=pending.email,
        full_name=pending.full_name,
        password_hash=pending.password_hash,
        role=pending.role,
        status="ACTIVE"
    )
    db.add(new_user)

    # Remove from staging
    await db.delete(pending)
    await db.commit()
    await db.refresh(new_user)

    # Issue initial access token so they are immediately logged in
    token = create_access_token(subject=str(new_user.user_id))

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        role=new_user.role,
        user=UserResponse.model_validate(new_user),
        step_up_required=False,
        anomaly_detected=False,
        uts_score=0.10
    )

@router.post("/verify-mfa", response_model=AuthTokenResponse)
async def verify_mfa_challenge(
    payload: VerifyMfaRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Validates the 6-digit Step-Up MFA code.
    On success: binds the device profile to PROBATIONARY/TRUSTED and issues the JWT token.
    """
    # Fetch user by email
    query = select(User).where(User.email == payload.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User account not found."
        )

    # Check if an active MFA challenge exists
    if not user.mfa_code or user.mfa_code != payload.otp_code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code. Please check your email."
        )

    # Check expiration safely (handling timezone-aware or naive DB datetimes)
    now = datetime.now(timezone.utc) if user.mfa_expires_at.tzinfo else datetime.utcnow()
    if now > user.mfa_expires_at:
        user.mfa_code = None
        user.mfa_expires_at = None
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification code has expired. Please sign in again."
        )

    # Clear the OTP code so it cannot be reused
    user.mfa_code = None
    user.mfa_expires_at = None

    # Update or Register the Device Profile (State Machine)
    device_stmt = select(UserDeviceProfile).where(
        UserDeviceProfile.user_id == user.user_id,
        UserDeviceProfile.device_fingerprint == payload.device_fingerprint,
    )
    device_profile = (await db.execute(device_stmt)).scalar_one_or_none()

    now_utc = datetime.utcnow()
    if device_profile is None:
        # First verification on this hardware: enroll as PROBATIONARY
        device_profile = UserDeviceProfile(
            user_id=user.user_id,
            device_fingerprint=payload.device_fingerprint,
            trust_state="PROBATIONARY",
            successful_login_count=1,
            device_model=payload.device_model or "Mobile Device",
            os_name=payload.os_name or "Unknown",
            first_seen_at=now_utc,
            last_login_at=now_utc,
            last_mfa_completed_at=now_utc,
        )
        db.add(device_profile)
    else:
        # Increment verified sessions
        device_profile.successful_login_count += 1
        device_profile.last_login_at = now_utc
        device_profile.last_mfa_completed_at = now_utc

        if device_profile.trust_state == "NEW":
            device_profile.trust_state = "PROBATIONARY"
        elif device_profile.trust_state == "PROBATIONARY":
            device_profile.trust_state = "TRUSTED"
            print(f" [Device Trust] Promoted device {payload.device_fingerprint[:8]}... to TRUSTED!", flush=True)

    # Record successful contextual login autid log
    audit_log = ContextualLogin(
        user_id=user.user_id,
        ip_address="127.0.0.1",
        device_fingerprint=payload.device_fingerprint,
        geolocation=None,
        login_risk_score=0.10,
        is_successful=True,
    )
    db.add(audit_log)
    await db.commit()

    # Issue the JWT Access Token
    token = create_access_token(subject=str(user.user_id))
    print(f" [MFA Passed] Issued JWT token for {user.email} on device {payload.device_fingerprint[:8]}...", flush=True)

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        user=UserResponse.model_validate(user),
        step_up_required=False,
        anomaly_detected=False,
        uts_score=0.10,
    )


@router.post("/resend-mfa")
@router.post("/resend-otp")
async def resend_mfa_code(
    payload: ResendMfaRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Generates a new 6-digit OTP code and emails it to the user.
    """
    query = select(User).where(User.email == payload.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User account not found."
        )

    # Generate fresh 6-digit OTP valid for 10 mins
    otp = "".join(str(secrets.randbelow(10)) for _ in range(6))
    user.mfa_code = otp
    user.mfa_expires_at = datetime.utcnow() + timedelta(minutes=10)
    await db.commit()

    user_name = getattr(user, "full_name", None) or getattr(user, "name", None) or "Caregiver"

    background_tasks.add_task(
        send_mfa_email,
        to_email=user.email,
        full_name=user_name,
        otp_code=otp,
        device_model="Mobile Device"
    )

    print(f"[MFA Resend] New OTP {otp} sent to {user.email}", flush=True)
    return {"message": "A fresh verification code has been sent to your email."}