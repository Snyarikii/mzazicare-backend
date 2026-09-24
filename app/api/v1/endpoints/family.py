from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.entities import CareSession, ElderlyProfile, User
from app.schemas.family import (
    CareSessionCreate,
    CareSessionResponse,
    ElderlyProfileCreate,
    ElderlyProfileResponse,
)

router = APIRouter(prefix="/family", tags=["family care"])


def require_family(user: User) -> None:
    if user.role != "family":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Family account required")


@router.post("/elders", response_model=ElderlyProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_elder(
    payload: ElderlyProfileCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_family(current_user)
    elder = ElderlyProfile(**payload.model_dump(), family_user_id=current_user.user_id)
    db.add(elder)
    await db.commit()
    await db.refresh(elder)
    return elder


@router.get("/elders", response_model=list[ElderlyProfileResponse])
async def list_elders(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_family(current_user)
    result = await db.execute(
        select(ElderlyProfile)
        .where(ElderlyProfile.family_user_id == current_user.user_id)
        .order_by(ElderlyProfile.created_at.desc())
    )
    return result.scalars().all()


@router.post("/bookings", response_model=CareSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_booking(
    payload: CareSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_family(current_user)
    elder_result = await db.execute(
        select(ElderlyProfile).where(
            ElderlyProfile.id == payload.elderly_id,
            ElderlyProfile.family_user_id == current_user.user_id,
        )
    )
    if elder_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Elderly profile not found")

    session = CareSession(
        **payload.model_dump(), family_user_id=current_user.user_id, status="REQUESTED"
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.get("/bookings", response_model=list[CareSessionResponse])
async def list_bookings(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    require_family(current_user)
    result = await db.execute(
        select(CareSession)
        .where(CareSession.family_user_id == current_user.user_id)
        .order_by(CareSession.scheduled_date.desc())
    )
    return result.scalars().all()
