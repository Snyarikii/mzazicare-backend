import io
import json
import unittest
import uuid
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

import main
from app.core import dependencies
from app.core.config import settings
from app.core.logger import RequestLoggingMiddleware, configure_logging, get_logger
from app.core.security import create_access_token
from app.db.session import get_db


def build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/open")
    async def open_route():
        return {"ok": True}

    @app.get("/me")
    async def me(claims: dict = Depends(dependencies.get_current_access_claims)):
        return {"sub": claims["sub"]}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("unexpected failure")

    async def no_db():
        yield None

    app.dependency_overrides[get_db] = no_db
    app.add_middleware(RequestLoggingMiddleware)
    return app


class RequestLoggingTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(build_app(), raise_server_exceptions=False)

    def request_logs(self, logs):
        return [entry for entry in logs if entry.get("event") == "http_request"]

    def test_logs_request_fields_without_query_string(self):
        with capture_logs() as logs:
            self.client.get("/open?token=secret-value")
        [entry] = self.request_logs(logs)
        self.assertEqual(entry["log_level"], "info")
        self.assertEqual(entry["method"], "GET")
        self.assertEqual(entry["path"], "/open")
        self.assertEqual(entry["status_code"], 200)
        self.assertEqual(entry["client_ip"], "testclient")
        self.assertIsNone(entry["user_id"])
        self.assertIsInstance(entry["duration_ms"], float)
        self.assertNotIn("secret-value", json.dumps(entry))

    def test_authenticated_request_records_user_id(self):
        user_id = str(uuid.uuid4())
        token = create_access_token(user_id)
        with patch.object(dependencies, "is_token_blacklisted", AsyncMock(return_value=False)), \
                capture_logs() as logs:
            response = self.client.get("/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        [entry] = self.request_logs(logs)
        self.assertEqual(entry["user_id"], user_id)

    def test_rejected_token_logs_warning_without_user_id(self):
        with capture_logs() as logs:
            response = self.client.get("/me", headers={"Authorization": "Bearer not-a-jwt"})
        self.assertEqual(response.status_code, 401)
        [entry] = self.request_logs(logs)
        self.assertEqual(entry["log_level"], "warning")
        self.assertIsNone(entry["user_id"])

    def test_unhandled_exception_is_logged_as_500_error(self):
        with capture_logs() as logs:
            response = self.client.get("/boom")
        self.assertEqual(response.status_code, 500)
        [entry] = self.request_logs(logs)
        self.assertEqual(entry["status_code"], 500)
        self.assertEqual(entry["log_level"], "error")

    def test_main_app_registers_middleware_outermost(self):
        self.assertIs(main.app.user_middleware[0].cls, RequestLoggingMiddleware)


class LogFormatTests(unittest.TestCase):
    def tearDown(self):
        configure_logging()

    def render(self, environment):
        stream = io.StringIO()
        with patch.object(settings, "ENVIRONMENT", environment), redirect_stdout(stream):
            configure_logging()
            get_logger("format-test").info("http_request", method="GET", status_code=200)
        return stream.getvalue().strip()

    def test_production_writes_json_with_timestamp_and_level(self):
        entry = json.loads(self.render("production"))
        self.assertEqual(entry["event"], "http_request")
        self.assertEqual(entry["level"], "info")
        self.assertEqual(entry["method"], "GET")
        self.assertTrue(entry["timestamp"].endswith("Z"))

    def test_development_writes_key_value_pairs(self):
        line = self.render("development")
        self.assertIn("http_request", line)
        self.assertIn("method=GET", line)
        self.assertIn("status_code=200", line)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(line)


if __name__ == "__main__":
    unittest.main()
