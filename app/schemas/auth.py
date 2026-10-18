import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, Literal

class UserResponse(BaseModel):
    user_id: uuid.UUID
    email: EmailStr
    full_name: str
    role: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}

class AuthTokenResponse(BaseModel):
    status: Literal["LOGGED_IN"] = "LOGGED_IN"
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    role: str
    user: UserResponse
    step_up_required: bool = False
    anomaly_detected: bool = False
    uts_score: float = 0.10


class GoogleProfile(BaseModel):
    email: EmailStr
    name: str
    sub: str


class NeedsRegistrationResponse(BaseModel):
    status: Literal["NEEDS_REGISTRATION"] = "NEEDS_REGISTRATION"
    google_profile: GoogleProfile

class VerifyMfaRequest(BaseModel):
    email: EmailStr
    otp_code: str
    device_fingerprint: str
    device_model: Optional[str] = None
    os_name: Optional[str] = None

class ResendMfaRequest(BaseModel):
    email: EmailStr


class GoogleAuthRequest(BaseModel):
    id_token: str
    role: Optional[str] = None
    device_fingerprint: str = Field(..., min_length=4)
    ip_address: Optional[str] = "127.0.0.1"
    geolocation: Optional[str] = None
    raw_features: Optional[Dict[str, Any]] = None


class BiometricUnlockRequest(BaseModel):
    refresh_token: str
    device_fingerprint: str = Field(..., min_length=4)
    raw_features: Dict[str, Any]
    ip_address: Optional[str] = "127.0.0.1"


class LogoutRequest(BaseModel):
    refresh_token: str
