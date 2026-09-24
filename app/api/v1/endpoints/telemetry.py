from datetime import datetime, timezone
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.entities import BehavioralAuditLog, SessionActivityLog, TrustEvaluation, User
from app.schemas.telemetry import (BehavioralEvaluationResponse, SessionActivityTelemetry,
                                   TelemetryEvaluationResponse, TelemetryIngestRequest)
from app.services.behavioral_engine import behavioral_engine

router = APIRouter(prefix="/telemetry", tags=["Telemetry & Risk Engine"])
logger = logging.getLogger(__name__)


@router.post("/behavioral-check", response_model=BehavioralEvaluationResponse)
async def behavioral_check(
    payload: SessionActivityTelemetry,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    evaluation = await behavioral_engine.evaluate_session_activity(
        user_id=current_user.user_id,
        metrics=payload.model_dump(),
        db=db,
    )
    audit = BehavioralAuditLog(
        user_id=current_user.user_id,
        anomaly_score=evaluation["anomaly_score"],
        is_anomaly=evaluation["is_anomaly"],
        session_revocation_recommended=evaluation["session_revocation_recommended"],
        risk_factors=evaluation["risk_factors"],
        telemetry_snapshot=payload.model_dump(mode="json"),
    )
    try:
        db.add(audit)
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("Could not persist behavioral audit for user %s", current_user.user_id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Behavioral evaluation could not be recorded.") from exc
    if evaluation["session_revocation_recommended"]:
        logger.critical("Behavioral anomaly recommends session revocation for user %s (score %.4f)",
                        current_user.user_id, evaluation["anomaly_score"])
    return BehavioralEvaluationResponse(**evaluation)

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
