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