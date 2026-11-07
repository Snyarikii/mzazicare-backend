import uuid
from typing import Literal
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
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    total_logons: int = Field(default=1, ge=0)
    distinct_devices: int = Field(default=1, ge=0)
    activity_span_hours: float = Field(default=1.0, ge=0.0)
    off_hours_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    records_accessed_count: int = Field(default=0, ge=0)
    total_payload_bytes: int = Field(default=0, ge=0)


class BehavioralEvaluationResponse(BaseModel):
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    risk_action: Literal["ALLOW", "CHALLENGE", "DENY"]
    step_up_required: bool
    access_denied: bool
    recommend_revocation: bool
    reasons: list[str]
    is_anomaly: bool
    session_revocation_recommended: bool
    risk_factors: list[str]
    # 2 = per-user rolling baseline model, 1 = population model, 0 = heuristic fallback.
    model_version: int | None = None
    # Earlier telemetry snapshots in the user's 30-day baseline.
    baseline_samples: int | None = None


class AnomalyResetRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=128)


class AnomalyResetResponse(BaseModel):
    status: Literal["RESET"]
    session_id: str
