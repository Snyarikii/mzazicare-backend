import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import jwt
from fastapi import HTTPException, Request

from app.api.v1.endpoints import auth
from app.core.config import settings
from app.core.security import create_refresh_token
from app.models.entities import BehavioralAuditLog, User, UserDeviceProfile
from app.schemas.auth import RefreshTokenRequest
from app.core.rate_limit import limiter

# These tests call endpoints directly with hand-built requests, so rate limiting is off.
limiter.enabled = False


class Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeDb:
    def __init__(self, user):
        self.user = user
        self.added = []
        self.commits = 0

    async def execute(self, _statement):
        return Result(self.user)

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        self.commits += 1


def request(headers=None):
    return Request({
        "type": "http", "client": ("203.0.113.10", 12345),
        "headers": [(name.lower().encode(), value.encode())
                    for name, value in (headers or {}).items()],
    })


def session(status="ACTIVE"):
    user_id = uuid.uuid4()
    user = User(
        user_id=user_id, email="member@example.com", full_name="Member",
        role="family", status=status, auth_provider="google", is_verified=True,
        verification_status="APPROVED", created_at=datetime.now(timezone.utc),
    )
    profile = UserDeviceProfile(
        user_id=user_id, device_fingerprint="device-1", trust_state="TRUSTED",
        refresh_jti="refresh-nonce", access_jti=str(uuid.uuid4()),
        access_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    token = create_refresh_token(str(user_id), "device-1", "refresh-nonce")
    return user, profile, token


class RefreshAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_rejects_blacklisted_token(self):
        user, profile, token = session()
        db = FakeDb(user)
        with patch.object(auth, "is_token_blacklisted", new=AsyncMock(return_value=True)):
            with self.assertRaises(HTTPException) as rejected:
                await auth.refresh_session(RefreshTokenRequest(refreshToken=token), request(), db)
        self.assertEqual(rejected.exception.status_code, 401)
        self.assertEqual(db.commits, 0)
        self.assertEqual(db.added, [])

    async def test_refresh_rejects_missing_and_suspended_users(self):
        user, profile, token = session("SUSPENDED")
        for account in (None, user):
            db = FakeDb(account)
            with patch.object(auth, "is_token_blacklisted", new=AsyncMock(return_value=False)):
                with self.assertRaises(HTTPException) as rejected:
                    await auth.refresh_session(RefreshTokenRequest(refresh_token=token), request(), db)
            self.assertEqual(rejected.exception.status_code, 401)
            self.assertEqual(db.commits, 0)

    async def test_refresh_rotates_session_and_audits_ip_and_device_headers(self):
        user, profile, token = session()
        previous_access_jti = profile.access_jti
        previous_access_expires_at = profile.access_expires_at
        db = FakeDb(user)
        headers = {
            "User-Agent": "MzaziCare Android", "X-Device-Model": "Pixel 9",
            "X-Device-Fingerprint": "device-1", "Authorization": "Bearer secret",
        }
        with patch.object(auth, "is_token_blacklisted", new=AsyncMock(return_value=False)) as lookup, \
             patch.object(auth, "_device", new=AsyncMock(return_value=profile)), \
             patch.object(auth, "blacklist_token", new=AsyncMock()) as revoke:
            response = await auth.refresh_session(
                RefreshTokenRequest(refreshToken=token), request(headers), db,
            )
        lookup.assert_awaited_once_with("refresh-nonce", db=db)
        revoke.assert_awaited_once_with(
            previous_access_jti, str(user.user_id), previous_access_expires_at, db=db,
        )
        self.assertEqual(db.commits, 1)
        self.assertNotEqual(profile.access_jti, previous_access_jti)
        self.assertNotEqual(profile.refresh_jti, "refresh-nonce")
        self.assertEqual(jwt.decode(response.access_token, settings.SECRET_KEY,
                                    algorithms=[settings.ALGORITHM])["jti"], profile.access_jti)
        audit = next(item for item in db.added if isinstance(item, BehavioralAuditLog))
        self.assertEqual(audit.telemetry_snapshot["event"], "SESSION_REFRESH")
        self.assertEqual(audit.telemetry_snapshot["ip_address"], "203.0.113.10")
        self.assertEqual(audit.telemetry_snapshot["device_headers"]["x-device-model"], "Pixel 9")
        self.assertNotIn("authorization", audit.telemetry_snapshot["device_headers"])


if __name__ == "__main__":
    unittest.main()
