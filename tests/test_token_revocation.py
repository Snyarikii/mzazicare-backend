import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import jwt
from fastapi import BackgroundTasks, HTTPException, Request
from sqlalchemy.dialects import postgresql

from app.api.v1.endpoints import auth, telemetry
from app.core import dependencies
from app.core.config import settings
from app.core.security import create_access_token, create_refresh_token
from app.models.entities import User, UserDeviceProfile
from app.schemas.auth import GoogleAuthRequest, LogoutRequest
from app.schemas.telemetry import TelemetryIngestRequest
from app.services.behavioral_engine import behavioral_engine
from app.services.risk_engine import risk_engine
from app.services.token_blacklist import blacklist_token, is_token_blacklisted
from app.core.rate_limit import limiter

# These tests call endpoints directly with hand-built requests, so rate limiting is off.
limiter.enabled = False


# The history lookups have their own tests (test_login_history.py). Here the fake database answers
# queries in a fixed order, so the lookups are stubbed: no history, and no challenged-login context.
_history_patches = []


def setUpModule():
    from unittest.mock import AsyncMock, patch
    from app.services import login_history
    for name in ("lookup_history", "latest_challenge_columns"):
        stub = patch.object(login_history, name, new=AsyncMock(return_value=None))
        stub.start()
        _history_patches.append(stub)


def tearDownModule():
    while _history_patches:
        _history_patches.pop().stop()


class Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeDb:
    def __init__(self, result=None):
        self.result = result
        self.statements = []
        self.added = []
        self.commits = 0

    async def execute(self, statement):
        self.statements.append(statement)
        return Result(self.result)

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        self.commits += 1


class TokenRevocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_access_tokens_have_distinct_uuid4_ids_and_expiration(self):
        subject = str(uuid.uuid4())
        first = jwt.decode(create_access_token(subject), settings.SECRET_KEY,
                           algorithms=[settings.ALGORITHM])
        second = jwt.decode(create_access_token(subject), settings.SECRET_KEY,
                            algorithms=[settings.ALGORITHM])
        self.assertEqual(first["sub"], subject)
        self.assertEqual(uuid.UUID(first["jti"]).version, 4)
        self.assertNotEqual(first["jti"], second["jti"])
        self.assertGreater(first["exp"], datetime.now(timezone.utc).timestamp())

    async def test_issued_access_token_matches_its_device_session(self):
        user_id, token_id = uuid.uuid4(), str(uuid.uuid4())
        expires = datetime.now(timezone.utc) + timedelta(hours=2)
        user = User(user_id=user_id, email="member@example.com", full_name="Member",
                    role="family", status="ACTIVE", is_verified=True,
                    verification_status="APPROVED",
                    created_at=datetime.now(timezone.utc))
        profile = UserDeviceProfile(user_id=user_id, device_fingerprint="device-1",
                                    refresh_jti="refresh-nonce", access_jti=token_id,
                                    access_expires_at=expires)
        response = auth._response(user, "device-1", 0.10, profile=profile)
        access_claims = jwt.decode(response.access_token, settings.SECRET_KEY,
                                   algorithms=[settings.ALGORITHM])
        self.assertEqual(access_claims["jti"], profile.access_jti)
        self.assertEqual(access_claims["exp"], int(expires.timestamp()))
        refresh_claims = jwt.decode(response.refresh_token, settings.SECRET_KEY,
                                    algorithms=[settings.ALGORITHM])
        self.assertEqual(refresh_claims["jti"], profile.refresh_jti)

    async def test_blacklist_insert_is_idempotent_and_lookup_is_expiry_bounded(self):
        token_id = str(uuid.uuid4())
        db = FakeDb()
        await blacklist_token(token_id, str(uuid.uuid4()),
                              datetime.now(timezone.utc) + timedelta(hours=2), db=db)
        sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
        self.assertIn("ON CONFLICT (jti) DO NOTHING", sql)
        self.assertEqual(db.commits, 0)

        db.result = token_id
        self.assertTrue(await is_token_blacklisted(token_id, db=db))
        db.result = None
        self.assertFalse(await is_token_blacklisted(token_id, db=db))

        db.statements.clear()
        await blacklist_token(token_id, str(uuid.uuid4()),
                              datetime.now(timezone.utc) + timedelta(hours=2),
                              db=db, revoke_refresh=True)
        self.assertEqual(len(db.statements), 2)
        update_sql = str(db.statements[1].compile(dialect=postgresql.dialect()))
        self.assertIn("UPDATE user_device_profiles", update_sql)
        self.assertIn("access_jti", update_sql)

    async def test_auth_guard_rejects_revoked_and_legacy_tokens(self):
        subject = str(uuid.uuid4())
        token = create_access_token(subject)
        with patch.object(dependencies, "is_token_blacklisted",
                          new=AsyncMock(return_value=True)):
            with self.assertRaises(HTTPException) as rejected:
                await dependencies.get_current_access_claims(token, FakeDb())
        self.assertEqual(rejected.exception.status_code, 401)
        self.assertEqual(rejected.exception.detail, "Token has been revoked")

        legacy = jwt.encode({"sub": subject, "token_type": "access",
                             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                            settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        with self.assertRaises(HTTPException) as invalid:
            await dependencies.get_current_access_claims(legacy, FakeDb())
        self.assertEqual(invalid.exception.status_code, 401)

    async def test_risk_and_behavioral_deny_revoke_current_access_token(self):
        token_id, user_id = str(uuid.uuid4()), str(uuid.uuid4())
        expires = datetime.now(timezone.utc) + timedelta(hours=2)
        with patch.object(risk_engine, "evaluate_risk",
                          return_value={"risk_action": "DENY", "access_denied": True}), \
             patch("app.services.risk_engine.blacklist_token", new_callable=AsyncMock) as revoke:
            result = await risk_engine.evaluate_risk_and_revoke(
                {}, current_jti=token_id, user_id=user_id, expires_at=expires,
            )
        self.assertEqual(result["risk_action"], "DENY")
        revoke.assert_awaited_once_with(token_id, user_id, expires,
                                        revoke_refresh=True)

        metrics = {"records_accessed_count": 45,
                   "total_payload_bytes": 15 * 1024 * 1024}
        with patch.object(behavioral_engine, "model", None), \
             patch("app.services.behavioral_engine.blacklist_token",
                   new_callable=AsyncMock) as revoke:
            result = await behavioral_engine.evaluate_session_activity(
                user_id, metrics, FakeDb(), current_jti=token_id, expires_at=expires,
            )
        self.assertEqual(result["risk_action"], "DENY")
        revoke.assert_awaited_once_with(token_id, user_id, expires,
                                        revoke_refresh=True)

    async def test_google_risk_deny_revokes_existing_device_session(self):
        user_id, token_id = uuid.uuid4(), str(uuid.uuid4())
        expires = datetime.now(timezone.utc) + timedelta(hours=2)
        user = User(user_id=user_id, email="member@example.com", full_name="Member",
                    role="family", status="ACTIVE", auth_provider="google",
                    google_sub="google-sub")
        profile = UserDeviceProfile(user_id=user_id, device_fingerprint="device-1",
                                    trust_state="TRUSTED", refresh_jti="refresh-nonce",
                                    access_jti=token_id, access_expires_at=expires)
        assessment = {"uts_score": 0.95, "base_ml_risk": 0.15,
                      "device_adjustment": 0.0, "risk_action": "DENY",
                      "step_up_required": False, "access_denied": True,
                      "anomaly_detected": True, "reasons": ["Critical threat"]}
        request = Request({"type": "http", "headers": []})
        db = FakeDb()
        with patch.object(auth, "verify_google_id_token",
                          new=AsyncMock(return_value=(user.email, user.full_name,
                                                      user.google_sub))), \
             patch.object(auth, "find_google_account",
                          new=AsyncMock(return_value=user)), \
             patch.object(auth, "_device", new=AsyncMock(return_value=profile)), \
             patch.object(risk_engine, "evaluate_risk", return_value=assessment), \
             patch("app.services.risk_engine.blacklist_token",
                   new_callable=AsyncMock) as revoke:
            response = await auth.google_auth(
                GoogleAuthRequest(id_token="google-id", device_fingerprint="device-1"),
                BackgroundTasks(), request, db,
            )
        self.assertEqual(response.status_code, 403)
        revoke.assert_awaited_once_with(token_id, str(user_id), expires,
                                        revoke_refresh=True)
        self.assertIsNone(profile.access_jti)
        self.assertIsNone(profile.refresh_jti)

    async def test_logout_revokes_access_token_paired_with_refresh_token(self):
        user_id, token_id = uuid.uuid4(), str(uuid.uuid4())
        expires = datetime.now(timezone.utc) + timedelta(hours=2)
        profile = UserDeviceProfile(
            user_id=user_id, device_fingerprint="device-1", trust_state="TRUSTED",
            refresh_jti="refresh-nonce", access_jti=token_id, access_expires_at=expires,
        )
        db = FakeDb(profile)
        request = LogoutRequest(refresh_token=create_refresh_token(
            str(user_id), "device-1", "refresh-nonce"))
        with patch.object(auth, "blacklist_token", new_callable=AsyncMock) as revoke:
            response = await auth.logout(request, db)
        revoke.assert_awaited_once_with(token_id, str(user_id), expires, db=db)
        self.assertEqual(response["message"], "Session ended.")
        self.assertIsNone(profile.access_jti)
        self.assertIsNone(profile.refresh_jti)
        self.assertEqual(db.commits, 1)

    async def test_session_rotation_revokes_previous_device_token(self):
        user_id, previous_jti = uuid.uuid4(), str(uuid.uuid4())
        expires = datetime.now(timezone.utc) + timedelta(hours=1)
        user = User(user_id=user_id, email="member@example.com", full_name="Member",
                    role="family", status="ACTIVE")
        profile = UserDeviceProfile(
            user_id=user_id, device_fingerprint="device-1", trust_state="TRUSTED",
            successful_login_count=3, refresh_jti="old-refresh",
            access_jti=previous_jti, access_expires_at=expires,
        )
        db = FakeDb()
        with patch.object(auth, "blacklist_token", new_callable=AsyncMock) as revoke:
            await auth._replace_device_session(db, user, "device-1", profile, {})
        revoke.assert_awaited_once_with(previous_jti, str(user_id), expires, db=db)
        self.assertNotEqual(profile.access_jti, previous_jti)
        self.assertEqual(uuid.UUID(profile.access_jti).version, 4)

    async def test_legacy_context_deny_also_revokes_and_requires_matching_user(self):
        user_id, token_id = uuid.uuid4(), str(uuid.uuid4())
        user = User(user_id=user_id, email="member@example.com", full_name="Member",
                    role="family", status="ACTIVE")
        claims = {"jti": token_id,
                  "exp": int((datetime.now(timezone.utc) + timedelta(hours=2)).timestamp())}
        payload = TelemetryIngestRequest(
            user_id=user_id, event_type="CARE_LOG_RECORDED",
            modification_volume=16, device_fingerprint="device-1",
        )
        db = FakeDb()
        with patch.object(telemetry, "blacklist_token", new_callable=AsyncMock) as revoke:
            result = await telemetry.ingest_session_telemetry(payload, db, user, claims)
        self.assertEqual(result.enforcement_action, "DENY")
        self.assertEqual(revoke.await_count, 1)
        self.assertTrue(revoke.await_args.kwargs["revoke_refresh"])
        self.assertEqual(db.commits, 1)

        payload.user_id = uuid.uuid4()
        with self.assertRaises(HTTPException) as rejected:
            await telemetry.ingest_session_telemetry(payload, FakeDb(), user, claims)
        self.assertEqual(rejected.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
