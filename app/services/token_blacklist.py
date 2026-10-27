"""Persistent, idempotent revocation of access-token JWT IDs."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.models.entities import TokenBlacklist, UserDeviceProfile


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


async def blacklist_token(jti: str, user_id: str, expires_at: datetime,
                          db: AsyncSession | None = None, *,
                          revoke_refresh: bool = False) -> None:
    """Persist revocation immediately, or join a caller-supplied transaction."""
    subject = uuid.UUID(str(user_id))
    token_id = str(uuid.UUID(jti))
    values = {
        "jti": token_id,
        "user_id": subject,
        "revoked_at": datetime.now(timezone.utc),
        "expires_at": _utc(expires_at),
    }
    statement = insert(TokenBlacklist).values(**values).on_conflict_do_nothing(
        index_elements=[TokenBlacklist.jti]
    )
    async def persist(session: AsyncSession) -> None:
        await session.execute(statement)
        if revoke_refresh:
            await session.execute(
                update(UserDeviceProfile)
                .where(UserDeviceProfile.user_id == subject,
                       UserDeviceProfile.access_jti == token_id)
                .values(refresh_jti=None, access_jti=None, access_expires_at=None)
            )

    if db is not None:
        await persist(db)
        return
    async with AsyncSessionLocal() as session:
        await persist(session)
        await session.commit()


async def is_token_blacklisted(jti: str, db: AsyncSession | None = None) -> bool:
    """Check unexpired revocations; JWT expiration is verified separately."""
    statement = select(TokenBlacklist.jti).where(
        TokenBlacklist.jti == jti,
        TokenBlacklist.expires_at > datetime.now(timezone.utc),
    ).limit(1)
    if db is not None:
        result = await db.execute(statement)
        return result.scalar_one_or_none() is not None
    async with AsyncSessionLocal() as session:
        result = await session.execute(statement)
        return result.scalar_one_or_none() is not None
