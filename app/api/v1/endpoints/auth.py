import uuid
import random
import string
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, get_password_hash, verify_password
from app.db.session import get_db
from app.models.entities import ContextualLogin, User
from app.schemas.auth import AuthTokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse, RegisterInitiateRequest, VerifyEmailRequest
from app.models.entities import PendingRegistration
from app.services.email import send_verification_email

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
async def login_user(payload: UserLoginRequest, db: AsyncSession = Depends(get_db)):
    # Retrieve user
    query = select(User).where(User.email == payload.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    # Telemetry Perimeter Check (Anomaly Simulation Hook)
    anomaly_detected = False  # Replace with actual anomaly detection logic
    step_up_required = False
    risk_score = 0.12

    # If login email contains 'anomaly' or non-standard device, escalate risk
    if "anomaly" in payload.email.lower() or payload.device_fingerprint == "unknown-device-sim":
        anomaly_detected = True
        step_up_required = True
        risk_score = 0.78

    # Record contextual login telemetry
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

    token = create_access_token(subject=str(user.user_id))

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        user=UserResponse.model_validate(user),
        step_up_required=step_up_required,
        anomaly_detected=anomaly_detected,
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