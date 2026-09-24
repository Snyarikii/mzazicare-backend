from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.entities import CareSession, User
from app.schemas.family import CareSessionComplete, CareSessionResponse

router = APIRouter(prefix="/caregiver", tags=["caregiver sessions"])


def require_caregiver(user: User) -> None:
    if user.role != "caregiver":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Caregiver account required")


@router.get("/available-sessions", response_model=list[CareSessionResponse])
async def available_sessions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_caregiver(current_user)
    result = await db.execute(
        select(CareSession)
        .where(CareSession.status == "REQUESTED")
        .order_by(CareSession.scheduled_date.asc())
    )
    return result.scalars().all()


@router.post("/sessions/{session_id}/accept", response_model=CareSessionResponse)
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
    if session.status != "REQUESTED":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Care session is no longer available")

    session.caregiver_id = current_user.user_id
    session.status = "ACCEPTED"
    await db.commit()
    await db.refresh(session)
    return session


@router.post("/sessions/{session_id}/complete", response_model=CareSessionResponse)
async def complete_session(
    session_id: int,
    payload: CareSessionComplete,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_caregiver(current_user)
    result = await db.execute(select(CareSession).where(CareSession.id == session_id))
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
    return session
