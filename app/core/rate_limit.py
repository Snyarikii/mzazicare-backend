"""Per-client API rate limiting shared by main.py and the route decorators."""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

# RATE_LIMIT_ENABLED=false turns limiting off (unit tests that call endpoints directly).
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/day", "50/hour"],
    enabled=settings.RATE_LIMIT_ENABLED,
)
