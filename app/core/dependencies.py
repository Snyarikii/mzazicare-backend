"""Authentication dependencies for protected API routes."""
import uuid
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models.entities import User
from app.services.token_blacklist import is_token_blacklisted

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/google")


async def get_current_access_claims(token: str = Depends(oauth2_scheme),
                                    db: AsyncSession = Depends(get_db)) -> dict:
    error = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                          detail="Invalid or expired bearer token.",
                          headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, settings.SECRET_KEY,
                             algorithms=[settings.ALGORITHM],
                             options={"require": ["exp", "sub", "jti", "token_type"]})
        if payload.get("token_type") != "access":
            raise ValueError("Only access tokens authorize API requests")
        uuid.UUID(payload["sub"])
        token_id = uuid.UUID(payload["jti"])
        if token_id.version != 4:
            raise ValueError("Access token jti must be a UUID4")
    except (jwt.PyJWTError, ValueError, TypeError, KeyError, AttributeError):
        raise error
    if await is_token_blacklisted(payload["jti"], db=db):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token has been revoked",
                            headers={"WWW-Authenticate": "Bearer"})
    return payload


async def get_current_active_user(claims: dict = Depends(get_current_access_claims),
                                  db: AsyncSession = Depends(get_db)) -> User:
    error = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                          detail="Invalid or expired bearer token.",
                          headers={"WWW-Authenticate": "Bearer"})
    subject = uuid.UUID(claims["sub"])
    result = await db.execute(select(User).where(User.user_id == subject))
    user = result.scalar_one_or_none()
    if user is None or user.status != "ACTIVE":
        raise error
    return user


def ensure_verified_caregiver(user: User) -> None:
    if (user.role != "caregiver" or not user.is_verified
            or user.verification_status != "APPROVED"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Verified caregiver account required.",
        )


async def get_current_verified_caregiver(
    user: User = Depends(get_current_active_user),
) -> User:
    ensure_verified_caregiver(user)
    return user
