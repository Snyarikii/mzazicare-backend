from datetime import datetime, timedelta, timezone
from typing import Any
import jwt
from app.core.config import settings


def create_access_token(
        subject: str | Any, expires_delta: timedelta | None = None
) -> str:
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject), "token_type": "access"}
    return jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )


def create_refresh_token(subject: str, device_fingerprint: str, jti: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"exp": expire, "sub": subject, "token_type": "refresh",
         "device_fingerprint": device_fingerprint, "jti": jti},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
