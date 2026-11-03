"""Administrator security dashboard and caregiver controls."""
import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.entities import (
    BehavioralAuditLog, ContextualLogin, SecurityAlert, SessionActivityLog,
    TrustEvaluation, User, UserDeviceProfile,
)
from app.services.token_blacklist import blacklist_token

router = APIRouter(prefix="/admin", tags=["administration"])


async def require_admin(user: User = Depends(get_current_active_user)) -> User:
    if user.role.upper() != "ADMIN":
        raise HTTPException(status_code=403, detail="Administrator access required.")
    return user


def _attributions(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if isinstance(value, dict):
        value = value.get("shapAttributions", value.get("shap_attributions", []))
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            impact = float(item.get("impact", item.get("value")))
        except (TypeError, ValueError):
            continue
        result.append({"feature": str(item.get("feature", item.get("name", "Unknown feature"))),
                       "impact": impact, "positive": impact > 0})
    return result


async def _alerts(db: AsyncSession) -> list[dict]:
    evaluations = (await db.execute(
        select(TrustEvaluation, User)
        .join(User, User.user_id == TrustEvaluation.user_id)
        .order_by(TrustEvaluation.evaluated_at.desc()).limit(100)
    )).all()
    if not evaluations:
        return []
    eval_ids = [evaluation.evaluation_id for evaluation, _ in evaluations]
    stored_alerts = (await db.execute(
        select(SecurityAlert).where(SecurityAlert.evaluation_id.in_(eval_ids))
    )).scalars().all()
    by_evaluation = {alert.evaluation_id: alert for alert in stored_alerts}
    active_users = set((await db.execute(
        select(UserDeviceProfile.user_id).where(
            UserDeviceProfile.access_jti.is_not(None),
            UserDeviceProfile.access_expires_at > datetime.now(timezone.utc),
        )
    )).scalars().all())
    items = []
    for evaluation, user in evaluations:
        alert = by_evaluation.get(evaluation.evaluation_id)
        risk = (alert.severity_level.upper() if alert else
                "HIGH" if evaluation.enforcement_action == "DENY" else
                "MEDIUM" if evaluation.enforcement_action in ("STEP_UP_MFA", "CHALLENGE") else "LOW")
        items.append({
            "id": str(alert.alert_id if alert else evaluation.evaluation_id),
            "userId": str(user.user_id),
            "role": user.role,
            "timestamp": (alert.triggered_at if alert else evaluation.evaluated_at).isoformat(),
            "behavior": alert.alert_type if alert else evaluation.enforcement_action.replace("_", " ").title(),
            "subtext": f"Trust evaluation · {evaluation.enforcement_action}",
            "utsScore": evaluation.unified_trust_score,
            "riskLevel": "Safe" if risk == "LOW" else risk.title(),
            "status": ("Revoked" if user.status != "ACTIVE" or
                       (risk == "HIGH" and user.user_id not in active_users) else
                       "MFA Prompted" if risk == "MEDIUM" else "Active"),
            "shapAttributions": _attributions(alert.shap_explanation if alert else None),
        })
    return items


@router.get("/security-alerts")
async def security_alerts(db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)):
    return await _alerts(db)


@router.get("/metrics")
async def metrics(db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)):
    active = (await db.execute(
        select(UserDeviceProfile.id).join(User, User.user_id == UserDeviceProfile.user_id)
        .where(User.status == "ACTIVE", UserDeviceProfile.access_jti.is_not(None),
               UserDeviceProfile.access_expires_at > datetime.now(timezone.utc))
    )).scalars().all()
    alerts = await _alerts(db)
    return {
        "activeSessions": len(active),
        "platformTrustScore": round(1 - sum(item["utsScore"] for item in alerts) / len(alerts), 2) if alerts else None,
        "flaggedAnomaliesCount": sum(item["riskLevel"] != "Safe" and item["status"] != "Revoked" for item in alerts),
    }


@router.get("/audit-logs")
async def audit_logs(db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)):
    activities = (await db.execute(
        select(SessionActivityLog, User).join(User, User.user_id == SessionActivityLog.user_id)
        .order_by(SessionActivityLog.timestamp.desc()).limit(100)
    )).all()
    logins = (await db.execute(
        select(ContextualLogin, User).join(User, User.user_id == ContextualLogin.user_id)
        .order_by(ContextualLogin.login_time.desc()).limit(100)
    )).all()
    behavioral = (await db.execute(
        select(BehavioralAuditLog, User).join(User, User.user_id == BehavioralAuditLog.user_id)
        .order_by(BehavioralAuditLog.timestamp.desc()).limit(100)
    )).all()
    rows = [{"id": str(log.log_id), "timestamp": log.timestamp.isoformat(),
             "userId": str(user.user_id), "role": user.role, "action": log.event_type,
             "resourceId": log.resource_accessed, "ipAddress": None}
            for log, user in activities]
    rows += [{"id": str(log.login_id), "timestamp": log.login_time.isoformat(),
              "userId": str(user.user_id), "role": user.role,
              "action": "LOGIN_SUCCESS" if log.is_successful else "LOGIN_FAILED",
              "resourceId": None, "ipAddress": log.ip_address}
             for log, user in logins]
    rows += [{"id": str(log.id), "timestamp": log.timestamp.isoformat(),
              "userId": str(user.user_id), "role": user.role,
              "action": log.telemetry_snapshot.get("event", "BEHAVIORAL_CHECK"),
              "resourceId": log.telemetry_snapshot.get("resource_id", log.telemetry_snapshot.get("session_id")),
              "ipAddress": log.telemetry_snapshot.get("ip_address")}
             for log, user in behavioral]
    return sorted(rows, key=lambda row: row["timestamp"], reverse=True)[:100]


def _caregiver(user: User) -> dict:
    return {"userId": str(user.user_id), "fullName": user.full_name,
            "email": user.email, "status": user.status,
            "isVerified": user.is_verified, "verificationStatus": user.verification_status}


@router.get("/caregivers")
async def caregivers(db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)):
    users = (await db.execute(select(User).where(User.role == "caregiver")
                              .order_by(User.full_name))).scalars().all()
    return [_caregiver(user) for user in users]


async def _target(db: AsyncSession, user_id: uuid.UUID) -> User:
    user = (await db.execute(select(User).where(User.user_id == user_id,
                                                User.role == "caregiver"))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Caregiver not found.")
    return user


async def _revoke(db: AsyncSession, user_id: uuid.UUID) -> None:
    profiles = (await db.execute(select(UserDeviceProfile).where(
        UserDeviceProfile.user_id == user_id).with_for_update())).scalars().all()
    for profile in profiles:
        if profile.access_jti and profile.access_expires_at:
            await blacklist_token(profile.access_jti, str(user_id), profile.access_expires_at,
                                  db=db, revoke_refresh=True)
        profile.access_jti = None
        profile.access_expires_at = None
        profile.refresh_jti = None


def _record_action(db: AsyncSession, admin: User, action: str,
                   target_id: uuid.UUID, request: Request) -> None:
    db.add(BehavioralAuditLog(
        user_id=admin.user_id, anomaly_score=0.0, is_anomaly=False,
        session_revocation_recommended=False, risk_factors=[],
        telemetry_snapshot={"event": action, "resource_id": str(target_id),
                            "ip_address": request.client.host if request.client else None},
    ))


@router.post("/caregivers/{user_id}/revoke-session")
async def revoke_session(user_id: uuid.UUID, request: Request,
                         db: AsyncSession = Depends(get_db),
                         admin: User = Depends(require_admin)):
    await _target(db, user_id)
    await _revoke(db, user_id)
    _record_action(db, admin, "ADMIN_REVOKE_SESSION", user_id, request)
    await db.commit()
    return {"status": "Revoked", "userId": str(user_id)}


class CaregiverStatus(BaseModel):
    status: Literal["ACTIVE", "SUSPENDED"]


@router.post("/caregivers/{user_id}/status")
async def set_caregiver_status(user_id: uuid.UUID, payload: CaregiverStatus,
                               request: Request,
                               db: AsyncSession = Depends(get_db),
                               admin: User = Depends(require_admin)):
    user = await _target(db, user_id)
    if payload.status == "SUSPENDED":
        await _revoke(db, user_id)
    user.status = payload.status
    _record_action(db, admin, f"ADMIN_{payload.status}_CAREGIVER", user_id, request)
    await db.commit()
    return _caregiver(user)
