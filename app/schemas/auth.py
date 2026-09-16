import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field

class UserRegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=6)
    role: str = Field("caregiver", pattern="^(caregiver|family|admin)$")
    patient_linking_code: str | None = None


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str
    device_fingerprint: str = Field(..., min_length=4)
    ip_address: str | None = "127.0.0.1"
    geolocation: str | None = "Nairobi, KE"

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