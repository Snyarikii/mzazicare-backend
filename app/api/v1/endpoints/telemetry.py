from datetime import datetime, timezone
import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_access_claims, get_current_active_user
from app.core.rate_limit import limiter
from app.db.session import get_db
from app.models.entities import BehavioralAuditLog, SessionActivityLog, TrustEvaluation, User
from app.schemas.telemetry import (AnomalyResetRequest, AnomalyResetResponse,
                                   BehavioralEvaluationResponse, DataExportEvent, DataExportResponse,
                                   SessionActivityTelemetry,
                                   TelemetryEvaluationResponse, TelemetryIngestRequest)
from app.services.behavioral_engine import EXPORT_FEATURE, _in_savepoint, behavioral_engine
from app.services.token_blacklist import blacklist_token

router = APIRouter(prefix="/telemetry", tags=["Telemetry & Risk Engine"])
logger = logging.getLogger(__name__)
# Shares from the app and screenshots of patient screens, stored in session_activity_logs.
DATA_EXPORT_EVENT = "DATA_EXPORT"


async def session_export_counts(db: AsyncSession, user_id, session_id: str | None) -> tuple[int, int]:
    """Exports in this session since its last risk reset, and the user's exports before them.

    Counted on the server from recorded events, so a modified client cannot under-report them.
    """
    snapshot = BehavioralAuditLog.telemetry_snapshot
    reset_at = select(func.max(BehavioralAuditLog.timestamp)).where(
        BehavioralAuditLog.user_id == user_id,
        snapshot["event"].as_string() == "ANOMALY_RESET",
        snapshot["session_id"].as_string() == (session_id or ""),
    ).scalar_subquery()
    in_session = and_(SessionActivityLog.session_token == (session_id or ""),
                      or_(reset_at.is_(None), SessionActivityLog.timestamp > reset_at))
    statement = select(func.count().filter(in_session), func.count()).where(
        SessionActivityLog.user_id == user_id,
        SessionActivityLog.event_type == DATA_EXPORT_EVENT,
    )
    current, total = (await db.execute(statement)).one()
    current = int(current or 0) if session_id else 0
    return current, int(total or 0) - current


@router.post("/behavioral-check", response_model=BehavioralEvaluationResponse)
async def behavioral_check(
    payload: SessionActivityTelemetry,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    claims: dict = Depends(get_current_access_claims),
):
    metrics = payload.model_dump()
    try:
        exports, earlier_exports = await _in_savepoint(
            db, lambda: session_export_counts(db, current_user.user_id, payload.session_id))
    except Exception as exc:  # Scoring continues without export counts.
        logger.warning("Could not count data exports for user %s: %s", current_user.user_id, exc)
        exports, earlier_exports = 0, None
    metrics[EXPORT_FEATURE] = exports
    metrics["prior_data_exports"] = earlier_exports
    evaluation = await behavioral_engine.evaluate_session_activity(
        user_id=current_user.user_id,
        metrics=metrics,
        db=db,
        current_jti=claims["jti"],
        expires_at=datetime.fromtimestamp(claims["exp"], timezone.utc),
    )
    audit = BehavioralAuditLog(
        user_id=current_user.user_id,
        anomaly_score=evaluation["anomaly_score"],
        is_anomaly=evaluation["is_anomaly"],
        session_revocation_recommended=evaluation["session_revocation_recommended"],
        risk_factors=evaluation["risk_factors"],
        telemetry_snapshot={**payload.model_dump(mode="json"), EXPORT_FEATURE: exports},
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


@router.post("/data-export", response_model=DataExportResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("30/minute")
async def record_data_export(
    request: Request,
    payload: DataExportEvent,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Record that patient data left the app (a share or a screenshot). The next behavioral
    check counts these events on the server."""
    db.add(SessionActivityLog(
        user_id=current_user.user_id,
        session_token=payload.session_id,
        event_type=DATA_EXPORT_EVENT,
        resource_accessed=f"{payload.kind}:{payload.screen}"[:100],
        modification_volume=1,
    ))
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("Could not record data export for user %s", current_user.user_id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="The data export could not be recorded.") from exc
    logger.warning("Data export recorded for user %s: %s on %s", current_user.user_id,
                   payload.kind, payload.screen)
    return DataExportResponse(status="RECORDED", kind=payload.kind)


@router.post("/reset-anomaly", response_model=AnomalyResetResponse)
async def reset_anomaly(
    payload: AnomalyResetRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Behavioral scoring uses client session counters. Record the client's
    # reset acknowledgement before it starts a fresh counter window.
    audit = BehavioralAuditLog(
        user_id=current_user.user_id,
        anomaly_score=0.0,
        is_anomaly=False,
        session_revocation_recommended=False,
        risk_factors=["Client PIN re-verification reported; session counters reset"],
        telemetry_snapshot={"session_id": payload.session_id, "event": "ANOMALY_RESET"},
    )
    try:
        db.add(audit)
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("Could not record anomaly reset for user %s", current_user.user_id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Session risk reset could not be recorded.") from exc
    return AnomalyResetResponse(status="RESET", session_id=payload.session_id)

@router.post("/context", response_model=TelemetryEvaluationResponse)
async def ingest_session_telemetry(
    payload: TelemetryIngestRequest, db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    claims: dict = Depends(get_current_access_claims),
):
    if payload.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Telemetry user does not match the bearer token.")
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

    if enforcement == "DENY":
        await blacklist_token(claims["jti"], str(current_user.user_id),
                              datetime.fromtimestamp(claims["exp"], timezone.utc),
                              revoke_refresh=True)

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
