from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.entities import SessionActivityLog, TrustEvaluation
from app.schemas.telemetry import TelemetryEvaluationResponse, TelemetryIngestRequest

router = APIRouter(prefix="/telemetry", tags=["Telemetry & Risk Engine"])

@router.post("/context", response_model=TelemetryEvaluationResponse)
async def ingest_session_telemetry(
    payload: TelemetryIngestRequest, db: AsyncSession = Depends(get_db)
):
    # Log session activity
    activity_log = SessionActivityLog(
        user_id=payload.user_id,
        session_token="session-live-token",
        event_type=payload.event_type,
        resource_accessed=payload.resource_accessed,
        modification_volume=payload.modification_volume,
    )
    db.add(activity_log)

    # Calculate unified trust score based on modification velocity
    if payload.modification_volume > 15:
        # High velocity burst anomaly
        uts = 0.84
        enforcement = "DENY"
        msg = "Excessive deletion/modification burst detected. Revoking session access."
    elif payload.modification_volume > 5:
        uts = 0.58
        enforcement = "STEP_UP_MFA"
        msg = "Mild activity drift detected. Step-up MFA challenge issued."
    else:
        uts = 0.12
        enforcement = "ALLOW"
        msg = "Telemetry within normal behavioral envelope."

    # Record Evaluation
    trust_eval = TrustEvaluation(
        user_id=payload.user_id,
        perimeter_score=0.10,
        behavioral_score=uts,
        unified_trust_score=uts,
        enforcement_action=enforcement,
    )
    db.add(trust_eval)
    await db.commit()

    return TelemetryEvaluationResponse(
        status="PROCESSED",
        unified_trust_score=uts,
        enforcement_action=enforcement,
        message=msg,
    )