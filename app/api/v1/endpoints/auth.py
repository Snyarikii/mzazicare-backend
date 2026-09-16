import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, get_password_hash, verify_password
from app.db.session import get_db
from app.models.entities import ContextualLogin, User
from app.schemas.auth import AuthTokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

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