import uuid
from datetime import datetime
from pydantic import AliasChoices, BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, Literal

class UserResponse(BaseModel):
    user_id: uuid.UUID
    email: EmailStr
    full_name: str
    role: str
    status: str
    license_number: Optional[str] = None
    is_verified: bool
    verification_status: str
    created_at: datetime

    model_config = {"from_attributes": True}

class AuthTokenResponse(BaseModel):
    status: Literal["LOGGED_IN", "SUCCESS", "MFA_REQUIRED"] = "LOGGED_IN"
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
    google_id: str


class NeedsRegistrationResponse(BaseModel):
    status: Literal["NEEDS_REGISTRATION"] = "NEEDS_REGISTRATION"
    google_profile: GoogleProfile

class VerifyMfaRequest(BaseModel):
    email: EmailStr
    otp_code: str
    device_fingerprint: str
    device_model: Optional[str] = None
    device_brand: Optional[str] = None
    device_name: Optional[str] = None
    model_id: Optional[str] = None
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


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(
        min_length=1, validation_alias=AliasChoices("refreshToken", "refresh_token")
    )


class DeviceSessionResponse(BaseModel):
    id: int
    device_name: str
    model: Optional[str] = None
    ip_address: Optional[str] = None
    last_seen: Optional[datetime] = None
    is_current: bool = False


class RevokeSessionsResponse(BaseModel):
    revoked_count: int
    message: str
