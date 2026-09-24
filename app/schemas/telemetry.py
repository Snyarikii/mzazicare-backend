import uuid
from pydantic import BaseModel, Field

class TelemetryIngestRequest(BaseModel):
    user_id: uuid.UUID
    event_type: str = Field(..., example="CARE_LOG_RECORDED")
    resource_accessed: str | None = "CARE_RECORDS"
    modification_volume: int = Field(0, ge=0)
    device_fingerprint: str
    ip_address: str = "127.0.0.1"

class TelemetryEvaluationResponse(BaseModel):
    status: str
    unified_trust_score: float
    enforcement_action: str
    message: str


class SessionActivityTelemetry(BaseModel):
    total_logons: int = Field(default=1, ge=0)
    distinct_devices: int = Field(default=1, ge=0)
    activity_span_hours: float = Field(default=1.0, ge=0.0)
    off_hours_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    records_accessed_count: int = Field(default=0, ge=0)
    total_payload_bytes: int = Field(default=0, ge=0)


class BehavioralEvaluationResponse(BaseModel):
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    session_revocation_recommended: bool
    risk_factors: list[str]
