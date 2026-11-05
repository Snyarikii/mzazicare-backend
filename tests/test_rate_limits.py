import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from app.api.v1.endpoints import auth
from app.core.dependencies import get_current_access_claims, get_current_active_user
from app.core.rate_limit import limiter


class RateLimitTests(unittest.TestCase):
    """Exercise the route limits through HTTP, without a database or Google."""

    def setUp(self):
        self.previously_enabled = limiter.enabled
        limiter.enabled = True
        limiter.reset()
        self.client = TestClient(main.app)

    def tearDown(self):
        limiter.reset()
        limiter.enabled = self.previously_enabled
        main.app.dependency_overrides.clear()

    def statuses(self, path, count, **kwargs):
        return [self.client.post(path, **kwargs).status_code for _ in range(count)]

    def test_google_sign_in_allows_five_per_minute(self):
        rejected = AsyncMock(side_effect=HTTPException(status_code=401, detail="Invalid Google ID token."))
        with patch.object(auth, "verify_google_id_token", rejected):
            codes = self.statuses("/api/v1/auth/google", 6,
                                  json={"id_token": "x", "device_fingerprint": "abcd1234"})
        self.assertEqual(codes, [401] * 5 + [429])
        response = self.client.post("/api/v1/auth/google",
                                    json={"id_token": "x", "device_fingerprint": "abcd1234"})
        self.assertEqual(response.json(), {"error": "Rate limit exceeded: 5 per 1 minute"})

    def test_refresh_allows_twenty_per_minute(self):
        codes = self.statuses("/api/v1/auth/refresh", 21, json={"refresh_token": "not-a-jwt"})
        self.assertEqual(codes, [401] * 20 + [429])

    def test_revoke_all_sessions_allows_ten_per_minute(self):
        main.app.dependency_overrides[get_current_access_claims] = lambda: {"jti": "current"}
        main.app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(user_id=uuid.uuid4())
        with patch.object(auth, "_active_device_sessions", AsyncMock(return_value=[])):
            codes = self.statuses("/api/v1/auth/revoke-all-sessions", 11)
        self.assertEqual(codes, [409] * 10 + [429])


if __name__ == "__main__":
    unittest.main()
