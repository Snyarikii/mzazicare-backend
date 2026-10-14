from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import case, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.entities import CareSession, ElderlyProfile, User
from app.schemas.family import (
    AvailableCareSession,
    CareSessionComplete,
    CareSessionResponse,
    CaregiverSessionResponse,
    ElderlySummary,
    PatientRecords,
)

router = APIRouter(prefix="/caregiver", tags=["caregiver sessions"])


def require_caregiver(user: User) -> None:
    if user.role != "caregiver":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Caregiver account required")


def elder_summary(elder: ElderlyProfile) -> ElderlySummary:
    return ElderlySummary(
        id=elder.id,
        elderly_id=elder.id,
        full_name=elder.full_name,
        age=elder.age,
        care_tier=elder.care_tier,
        chronic_conditions=elder.chronic_conditions,
        residence_address=elder.residence_address,
    )


def session_response(session: CareSession, elder: ElderlyProfile) -> CaregiverSessionResponse:
    return CaregiverSessionResponse(
        **CareSessionResponse.model_validate(session).model_dump(),
        session_id=session.id,
        elder=elder_summary(elder),
    )


@router.get("/available-sessions", response_model=list[AvailableCareSession])
async def available_sessions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_caregiver(current_user)
    result = await db.execute(
        select(
            CareSession.id,
            CareSession.elderly_id,
            CareSession.scheduled_date,
            CareSession.service_tier,
            CareSession.special_instructions,
            CareSession.status,
            ElderlyProfile.full_name,
            ElderlyProfile.age,
            ElderlyProfile.care_tier,
            ElderlyProfile.chronic_conditions,
            ElderlyProfile.residence_address,
        )
        .join(ElderlyProfile, CareSession.elderly_id == ElderlyProfile.id)
        .where(CareSession.status == "REQUESTED", CareSession.caregiver_id.is_(None))
        .order_by(CareSession.scheduled_date.asc())
    )
    return [
        AvailableCareSession(
            id=session_id,
            session_id=session_id,
            elderly_id=elderly_id,
            scheduled_date=scheduled_date,
            service_tier=service_tier,
            special_instructions=special_instructions,
            status=session_status,
            elder=ElderlySummary(
                id=elderly_id,
                elderly_id=elderly_id,
                full_name=full_name,
                age=age,
                care_tier=care_tier,
                chronic_conditions=chronic_conditions,
                residence_address=residence_address,
            ),
        )
        for (
            session_id, elderly_id, scheduled_date, service_tier,
            special_instructions, session_status, full_name, age, care_tier,
            chronic_conditions, residence_address,
        ) in result.all()
    ]


@router.post("/sessions/{session_id}/accept", response_model=CaregiverSessionResponse)
async def accept_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_caregiver(current_user)
    result = await db.execute(
        select(CareSession).where(CareSession.id == session_id).with_for_update()
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care session not found")
    if session.status != "REQUESTED" or session.caregiver_id is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This care request has already been accepted by another caregiver")

    session.caregiver_id = current_user.user_id
    session.status = "ACCEPTED"
    await db.commit()
    await db.refresh(session)
    elder = await db.get(ElderlyProfile, session.elderly_id)
    return session_response(session, elder)


@router.get("/my-sessions", response_model=list[CaregiverSessionResponse])
async def my_sessions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_caregiver(current_user)
    result = await db.execute(
        select(CareSession, ElderlyProfile)
        .join(ElderlyProfile, CareSession.elderly_id == ElderlyProfile.id)
        .where(CareSession.caregiver_id == current_user.user_id)
        .order_by(case((CareSession.status == "ACCEPTED", 0), else_=1), CareSession.scheduled_date.desc())
    )
    return [session_response(session, elder) for session, elder in result.all()]


@router.get("/patients/{elderly_id}/records", response_model=PatientRecords)
async def patient_records(
    elderly_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_caregiver(current_user)
    authorized = await db.scalar(
        select(
            exists().where(
                CareSession.elderly_id == elderly_id,
                CareSession.caregiver_id == current_user.user_id,
                CareSession.status.in_(("ACCEPTED", "IN_PROGRESS", "COMPLETED")),
            )
        )
    )
    if not authorized:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Only the assigned caregiver can view this patient's clinical file",
        )

    elder = await db.get(ElderlyProfile, elderly_id)
    if elder is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Elderly profile not found")
    result = await db.execute(
        select(CareSession)
        .where(CareSession.elderly_id == elderly_id, CareSession.status == "COMPLETED")
        .order_by(CareSession.completed_at.asc(), CareSession.id.asc())
    )
    return {"elder": elder, "encounter_history": result.scalars().all()}


@router.post("/sessions/{session_id}/complete", response_model=CaregiverSessionResponse)
async def complete_session(
    session_id: int,
    payload: CareSessionComplete,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_caregiver(current_user)
    result = await db.execute(select(CareSession).where(CareSession.id == session_id).with_for_update())
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care session not found")
    if session.caregiver_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session is assigned to another caregiver")
    if session.status not in {"ACCEPTED", "IN_PROGRESS"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Care session cannot be completed in its current state")

    session.vitals_recorded = payload.vitals_recorded
    session.encounter_notes = payload.encounter_notes
    session.status = "COMPLETED"
    session.completed_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(session)
    elder = await db.get(ElderlyProfile, session.elderly_id)
    return session_response(session, elder)
