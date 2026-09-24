import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any

class UserRegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=6)
    role: str = Field("caregiver", pattern="^(caregiver|family|admin)$")
    patient_linking_code: Optional[str] = None


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str
    device_fingerprint: str = Field(..., min_length=4)
    ip_address: str | None = "127.0.0.1"
    geolocation: str | None = "Nairobi, KE"
    raw_features: Optional[Dict[str, Any]] = None

class UserResponse(BaseModel):
    user_id: uuid.UUID
    email: EmailStr
    full_name: str
    role: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}

class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user: UserResponse
    step_up_required: bool = False
    anomaly_detected: bool = False
    uts_score: float = 0.10

class RegisterInitiateRequest(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    role: str = "caregiver"

class VerifyEmailRequest(BaseModel):
    email: EmailStr
    otp_code: str

class VerifyMfaRequest(BaseModel):
    email: EmailStr
    otp_code: str
    device_fingerprint: str
    device_model: Optional[str] = None
    os_name: Optional[str] = None

class ResendMfaRequest(BaseModel):
    email: EmailStr
