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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")


async def get_current_active_user(token: str = Depends(oauth2_scheme),
                                  db: AsyncSession = Depends(get_db)) -> User:
    error = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                          detail="Invalid or expired bearer token.",
                          headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        subject = uuid.UUID(payload.get("sub", ""))
    except (jwt.PyJWTError, ValueError, TypeError):
        raise error
    result = await db.execute(select(User).where(User.user_id == subject))
    user = result.scalar_one_or_none()
    if user is None or user.status != "ACTIVE":
        raise error
    return user
