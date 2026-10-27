from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import cast, exists, select, String
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_verified_caregiver
from app.db.session import get_db
from app.models.entities import CareLog, CareSession, ElderlyProfile, User
from app.schemas.caregiver_activity import CareLogCreate, CareLogResponse

router = APIRouter(prefix="/care-logs", tags=["care logs"])


def log_response(log: CareLog, patient_name: str | None) -> CareLogResponse:
    return CareLogResponse(
        id=log.log_id,
        caregiver_id=log.user_id,
        patient_id=log.patient_id,
        patient_name=patient_name,
        category=log.category,
        notes=log.care_notes or "",
        created_at=log.recorded_at,
    )


@router.get("", response_model=list[CareLogResponse])
async def list_care_logs(
    patient_id: str | None = Query(default=None),
    current_user: User = Depends(get_current_verified_caregiver),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(CareLog, ElderlyProfile.full_name)
        .outerjoin(ElderlyProfile, cast(ElderlyProfile.id, String) == CareLog.patient_id)
        .where(CareLog.user_id == current_user.user_id)
        .order_by(CareLog.recorded_at.desc(), CareLog.log_id.desc())
    )
    if patient_id is not None:
        query = query.where(CareLog.patient_id == patient_id)
    result = await db.execute(query)
    return [log_response(log, name) for log, name in result.all()]


@router.post("", response_model=CareLogResponse, status_code=status.HTTP_201_CREATED)
async def create_care_log(
    payload: CareLogCreate,
    current_user: User = Depends(get_current_verified_caregiver),
    db: AsyncSession = Depends(get_db),
):
    if not payload.patient_id.isdecimal():
        raise HTTPException(status_code=422, detail="patient_id must be an assigned patient ID.")
    elder_id = int(payload.patient_id)
    patient = await db.get(ElderlyProfile, elder_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found.")
    assigned = await db.scalar(select(exists().where(
        CareSession.elderly_id == elder_id,
        CareSession.caregiver_id == current_user.user_id,
        CareSession.status.in_(("ACCEPTED", "IN_PROGRESS", "COMPLETED")),
    )))
    if not assigned:
        raise HTTPException(status_code=403, detail="This patient is not assigned to you.")

    log = CareLog(
        user_id=current_user.user_id,
        patient_id=str(elder_id),
        category=payload.category,
        care_notes=payload.notes,
        task_status="RECORDED",
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log_response(log, patient.full_name)
