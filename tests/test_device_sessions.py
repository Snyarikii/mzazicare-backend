import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, Request
from sqlalchemy.dialects import postgresql

from app.api.v1.endpoints import auth
from app.models.entities import BehavioralAuditLog, User, UserDeviceProfile
from app.services.token_blacklist import blacklist_refresh_jti
from app.core.rate_limit import limiter

# These tests call endpoints directly with hand-built requests, so rate limiting is off.
limiter.enabled = False


class Scalars:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class Result:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return Scalars(self.rows)


class FakeDb:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []
        self.added = []
        self.commits = 0

    async def execute(self, statement):
        self.statements.append(statement)
        return Result(self.rows)

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        self.commits += 1


def http_request(host="198.51.100.7"):
    return Request({"type": "http", "client": (host, 443), "headers": []})


def member():
    return User(user_id=uuid.uuid4(), email="member@example.com", full_name="Member",
                role="family", status="ACTIVE", auth_provider="google",
                created_at=datetime.now(timezone.utc))


def device(user, index, *, seen_minutes_ago, name="Samsung Galaxy A14", model="SM-A145F"):
    return UserDeviceProfile(
        id=index, user_id=user.user_id, device_fingerprint=f"device-{index}",
        trust_state="TRUSTED", refresh_jti=f"refresh-nonce-{index}",
        access_jti=str(uuid.uuid4()),
        access_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        device_name=name, model_identifier=model, device_model=f"samsung {model}",
        last_ip_address=f"10.0.0.{index}",
        last_seen_at=datetime.now(timezone.utc) - timedelta(minutes=seen_minutes_ago),
        last_login_at=datetime.utcnow(),
    )


class DeviceSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_current_device_first_and_refreshes_its_presence(self):
        user = member()
        older = device(user, 1, seen_minutes_ago=600, name="Tecno Spark 20", model="KJ5")
        current = device(user, 2, seen_minutes_ago=90)
        recent = device(user, 3, seen_minutes_ago=5, name="Apple iPhone 13", model="iPhone14,5")
        db = FakeDb([older, current, recent])

        sessions = await auth.list_active_sessions(
            http_request(), {"jti": current.access_jti}, user, db)

        self.assertEqual([s.id for s in sessions], [2, 3, 1])
        self.assertEqual([s.is_current for s in sessions], [True, False, False])
        first = sessions[0]
        self.assertEqual(first.device_name, "Samsung Galaxy A14")
        self.assertEqual(first.model, "SM-A145F")
        self.assertEqual(first.ip_address, "198.51.100.7")
        self.assertLess(datetime.now(timezone.utc) - first.last_seen, timedelta(seconds=5))
        self.assertEqual(sessions[2].ip_address, "10.0.0.1")
        self.assertEqual(db.commits, 1)
        sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
        self.assertIn("refresh_jti IS NOT NULL", sql)
        self.assertIn("trust_state !=", sql)

    async def test_revokes_every_other_session_and_audits(self):
        user = member()
        current = device(user, 1, seen_minutes_ago=0)
        other_a = device(user, 2, seen_minutes_ago=30)
        other_b = device(user, 3, seen_minutes_ago=300)
        kept_access, kept_refresh = current.access_jti, current.refresh_jti
        other_access = {p.id: (p.access_jti, p.access_expires_at) for p in (other_a, other_b)}
        db = FakeDb([current, other_a, other_b])

        with patch.object(auth, "blacklist_token", new=AsyncMock()) as revoke_access, \
             patch.object(auth, "blacklist_refresh_jti", new=AsyncMock(return_value=True)) as revoke_refresh:
            response = await auth.revoke_all_other_sessions(
                http_request(), {"jti": current.access_jti}, user, db)

        self.assertEqual(response.revoked_count, 2)
        self.assertEqual(current.access_jti, kept_access)
        self.assertEqual(current.refresh_jti, kept_refresh)
        for profile in (other_a, other_b):
            self.assertIsNone(profile.refresh_jti)
            self.assertIsNone(profile.access_jti)
            self.assertIsNone(profile.access_expires_at)
            self.assertEqual(profile.trust_state, "TRUSTED")  # devices may sign in again
        revoked_access = {call.args[0] for call in revoke_access.await_args_list}
        self.assertEqual(revoked_access, {jti for jti, _ in other_access.values()})
        self.assertNotIn(kept_access, revoked_access)
        revoked_refresh = {call.args[0] for call in revoke_refresh.await_args_list}
        self.assertEqual(revoked_refresh, {"refresh-nonce-2", "refresh-nonce-3"})
        audit = next(row for row in db.added if isinstance(row, BehavioralAuditLog))
        self.assertEqual(audit.telemetry_snapshot["event"], "SESSIONS_REVOKED_REMOTELY")
        self.assertEqual(audit.telemetry_snapshot["revoked_count"], 2)
        self.assertEqual(audit.telemetry_snapshot["initiating_session_id"], 1)
        self.assertEqual(db.commits, 1)
        sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
        self.assertIn("FOR UPDATE", sql)

    async def test_refuses_when_current_session_is_unknown(self):
        user = member()
        other = device(user, 2, seen_minutes_ago=30)
        db = FakeDb([other])
        with patch.object(auth, "blacklist_token", new=AsyncMock()) as revoke_access:
            with self.assertRaises(HTTPException) as rejected:
                await auth.revoke_all_other_sessions(
                    http_request(), {"jti": str(uuid.uuid4())}, user, db)
        self.assertEqual(rejected.exception.status_code, 409)
        revoke_access.assert_not_awaited()
        self.assertEqual(other.refresh_jti, "refresh-nonce-2")
        self.assertEqual(db.commits, 0)

    async def test_login_records_device_labels_and_observed_ip(self):
        user = member()
        profile = auth._enroll_or_update(
            FakeDb([]), user, "device-9", None,
            {"device_brand": "samsung", "device_model": "SM-A145F",
             "device_name": "Samsung Galaxy A14", "os_name": "Android"},
            ip="203.0.113.4",
        )
        self.assertEqual(profile.device_name, "Samsung Galaxy A14")
        self.assertEqual(profile.model_identifier, "SM-A145F")
        self.assertEqual(profile.last_ip_address, "203.0.113.4")
        self.assertIsNotNone(profile.last_seen_at)

    async def test_refresh_blacklist_entry_uses_opaque_nonce(self):
        db = FakeDb([])
        stored = await blacklist_refresh_jti(
            "a" * 32, str(uuid.uuid4()), datetime.now(timezone.utc), db=db)
        self.assertTrue(stored)
        sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
        self.assertIn("INSERT INTO token_blacklist", sql)
        self.assertIn("ON CONFLICT (jti) DO NOTHING", sql)
        self.assertFalse(await blacklist_refresh_jti(
            "b" * 40, str(uuid.uuid4()), datetime.now(timezone.utc), db=db))


if __name__ == "__main__":
    unittest.main()
