"""Structured logging: JSON lines outside development, readable key-value pairs in development.

Request logs are written by RequestLoggingMiddleware (registered in main.py). Authenticated
routes report the caller through bind_request_user_id(), called from the auth dependency.
"""

import logging
import sys
import time
from contextvars import ContextVar

import structlog

from app.core.config import settings

_DEVELOPMENT_NAMES = {"development", "dev", "local"}

# One mutable dict per request. The middleware creates it; the auth dependency fills in
# user_id. A shared dict (rather than a plain value) still reaches the middleware if the
# dependency runs in a copied context.
_request_context: ContextVar[dict | None] = ContextVar("mzazicare_request_context", default=None)


def is_development() -> bool:
    return settings.ENVIRONMENT.strip().casefold() in _DEVELOPMENT_NAMES


def _console_colors() -> bool:
    """Colour only an interactive terminal; Windows also needs colorama installed."""
    if not sys.stdout.isatty():
        return False
    if sys.platform != "win32":
        return True
    try:
        import colorama  # noqa: F401
    except ImportError:
        return False
    return True


def configure_logging() -> None:
    renderer = (structlog.dev.ConsoleRenderer(colors=_console_colors()) if is_development()
                else structlog.processors.JSONRenderer())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


def bind_request_user_id(user_id) -> None:
    """Record the authenticated caller for the current request's access log."""
    context = _request_context.get()
    if context is not None and user_id is not None:
        context["user_id"] = str(user_id)


class RequestLoggingMiddleware:
    """Pure ASGI middleware that writes one structured log entry per HTTP request.

    Only the path is logged, never the query string, headers or body, so tokens and
    patient data stay out of the logs.
    """

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("mzazicare.http")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        context: dict = {}
        token = _request_context.set(context)
        status_code = 500  # Reported if the app raises before sending a response.
        started = time.perf_counter()

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _request_context.reset(token)
            client = scope.get("client")
            log = (self.logger.error if status_code >= 500 else
                   self.logger.warning if status_code >= 400 else self.logger.info)
            log(
                "http_request",
                method=scope.get("method"),
                path=scope.get("path"),
                status_code=status_code,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                client_ip=client[0] if client else None,
                user_id=context.get("user_id"),
            )
