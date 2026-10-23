import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from fastapi import BackgroundTasks, HTTPException

from app.api.v1.endpoints import auth
from app.core.security import create_refresh_token
from app.models.entities import User, UserDeviceProfile
from app.schemas.auth import BiometricUnlockRequest, GoogleAuthRequest, VerifyMfaRequest



class Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeDb:
    def __init__(self, *results):
        self.results = list(results)
        self.added = []
        self.commits = 0

    async def execute(self, statement):
        return Result(self.results.pop(0))

    def add(self, item):
        self.added.append(item)

    async def flush(self):
        for item in self.added:
            if isinstance(item, User):
                item.user_id = uuid.uuid4()
                item.created_at = datetime.now(timezone.utc)

    async def commit(self):
        self.commits += 1


def existing_user():
    return User(user_id=uuid.uuid4(), email="member@example.com", full_name="Member",
                role="family", status="ACTIVE", auth_provider="google",
                google_sub="google-sub", is_verified=True,
                verification_status="APPROVED", created_at=datetime.now(timezone.utc))


class GoogleAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_registration_requires_role(self):
        db = FakeDb(None)
        payload = GoogleAuthRequest(id_token="id", device_fingerprint="device-1")
        with patch.object(auth.settings, "GOOGLE_CLIENT_IDS", "client-id"), \
             patch.object(auth, "run_in_threadpool", new=AsyncMock(return_value={
                 "aud": "client-id", "email_verified": True,
                 "email": "member@example.com", "name": "Member", "sub": "google-sub",
             })):
            response = await auth.google_auth(payload, BackgroundTasks(), db)
        self.assertEqual(response.status, "NEEDS_REGISTRATION")
        self.assertEqual(response.google_profile.email, "member@example.com")
        self.assertEqual(response.google_profile.name, "Member")
        self.assertEqual(response.google_profile.google_id, "google-sub")
        self.assertEqual(db.commits, 0)

    async def test_registration_rejects_invalid_role(self):
        db = FakeDb(None)
        payload = GoogleAuthRequest(id_token="id", role="admin", device_fingerprint="device-1")
        with patch.object(auth.settings, "GOOGLE_CLIENT_IDS", "client-id"), \
             patch.object(auth, "run_in_threadpool", new=AsyncMock(return_value={
                 "aud": "client-id", "email_verified": True,
                 "email": "member@example.com", "name": "Member", "sub": "google-sub",
             })):
            with self.assertRaises(HTTPException) as error:
                await auth.google_auth(payload, BackgroundTasks(), db)
        self.assertEqual(error.exception.status_code, 400)

    async def test_new_google_user_with_role_is_not_written_until_completion(self):
        db = FakeDb(None)
        payload = GoogleAuthRequest(id_token="id", role="family", device_fingerprint="device-1")
        with patch.object(auth.settings, "GOOGLE_CLIENT_IDS", "client-id"), \
             patch.object(auth, "run_in_threadpool", new=AsyncMock(return_value={
                 "aud": "client-id", "email_verified": True,
                 "email": "member@example.com", "name": "Member", "sub": "google-sub",
             })):
            response = await auth.google_auth(payload, BackgroundTasks(), db)
        self.assertEqual(response.status, "NEEDS_REGISTRATION")
        self.assertEqual(response.google_profile.google_id, "google-sub")
        self.assertEqual(db.commits, 0)
        self.assertEqual(db.added, [])

    async def test_family_completion_creates_device_bound_session(self):
        db = FakeDb(None)
        payload = GoogleAuthRequest(id_token="id", role="family", device_fingerprint="device-1")
        with patch.object(auth.settings, "GOOGLE_CLIENT_IDS", "client-id"), \
             patch.object(auth, "run_in_threadpool", new=AsyncMock(return_value={
                 "aud": "client-id", "email_verified": True,
                 "email": "member@example.com", "name": "Member", "sub": "google-sub",
             })):
            response = await auth.register_family(payload, db)
        self.assertEqual(response.status, "LOGGED_IN")
        self.assertTrue(response.access_token and response.refresh_token)
        self.assertTrue(response.user.is_verified)
        self.assertEqual(db.commits, 1)
        self.assertEqual(next(x for x in db.added if isinstance(x, UserDeviceProfile)).trust_state,
                         "PROBATIONARY")

    async def test_verified_caregiver_completion_creates_approved_session(self):
        db = FakeDb()
        response = await auth.create_google_account(
            db, email="stephen@example.com", name="Stephen Nyariki",
            sub="google-sub-caregiver", role="caregiver", fingerprint="device-1",
            license_number="CG-2026-001",
        )
        self.assertEqual(response.role, "caregiver")
        self.assertTrue(response.user.is_verified)
        self.assertEqual(response.user.verification_status, "APPROVED")
        self.assertEqual(response.user.license_number, "CG-2026-001")
        self.assertTrue(response.access_token and response.refresh_token)
        self.assertEqual(db.commits, 1)

    async def test_high_risk_google_login_dispatches_email_without_token(self):
        user = existing_user()
        profile = UserDeviceProfile(user_id=user.user_id, device_fingerprint="device-1",
                                    trust_state="PROBATIONARY", successful_login_count=2)
        db = FakeDb(user, profile)
        background = BackgroundTasks()
        payload = GoogleAuthRequest(id_token="id", device_fingerprint="device-1")
        with patch.object(auth.settings, "GOOGLE_CLIENT_IDS", "client-id"), \
             patch.object(auth, "run_in_threadpool", new=AsyncMock(return_value={
                 "aud": "client-id", "email_verified": True,
                 "email": user.email, "name": user.full_name, "sub": user.google_sub,
             })), \
             patch.object(auth, "mfa_email_configured", return_value=True), \
             patch.object(auth.risk_engine, "evaluate_risk", return_value={
                 "uts_score": 0.55, "step_up_required": True, "anomaly_detected": False,
                 "base_ml_risk": 0.55, "device_adjustment": 0.0,
                 "reasons": ["Probationary device (2/3 verified sessions)"],
             }):
            with self.assertLogs(auth.logger, level="INFO") as risk_logs:
                response = await auth.google_auth(payload, background, db)
        self.assertTrue(response.step_up_required)
        self.assertEqual(response.status, "LOGGED_IN")
        self.assertEqual(response.access_token, "")
        self.assertIsNone(response.refresh_token)
        self.assertEqual(len(background.tasks), 1)
        self.assertEqual(profile.refresh_jti, None)
        logged = "\n".join(risk_logs.output)
        self.assertIn("AUTHENTICATION RISK ASSESSMENT", logged)
        self.assertIn("Sign-in method      : Google sign-in", logged)
        self.assertIn("Unified Trust Score : 0.55", logged)
        self.assertIn("Step-up MFA         : Required (UTS ≥ 0.55)", logged)
        self.assertIn("1. Probationary device", logged)
        self.assertNotIn(user.email, logged)

    async def test_low_risk_login_promotes_device_and_rotates_refresh(self):
        user = existing_user()
        profile = UserDeviceProfile(user_id=user.user_id, device_fingerprint="device-1",
                                    trust_state="PROBATIONARY", successful_login_count=2,
                                    refresh_jti="old-nonce")
        db = FakeDb(user, profile)
        payload = GoogleAuthRequest(id_token="id", device_fingerprint="device-1")
        with patch.object(auth.settings, "GOOGLE_CLIENT_IDS", "client-id"), \
             patch.object(auth, "run_in_threadpool", new=AsyncMock(return_value={
                 "aud": "client-id", "email_verified": True,
                 "email": user.email, "name": user.full_name, "sub": user.google_sub,
             })), \
             patch.object(auth.risk_engine, "evaluate_risk", return_value={
                 "uts_score": 0.54, "step_up_required": False, "anomaly_detected": False,
             }):
            response = await auth.google_auth(payload, BackgroundTasks(), db)
        self.assertFalse(response.step_up_required)
        self.assertTrue(response.access_token and response.refresh_token)
        self.assertEqual(profile.trust_state, "TRUSTED")
        self.assertNotEqual(profile.refresh_jti, "old-nonce")

    async def test_biometric_replay_is_rejected(self):
        user = existing_user()
        token = create_refresh_token(str(user.user_id), "device-1", "old-nonce")
        profile = UserDeviceProfile(user_id=user.user_id, device_fingerprint="device-1",
                                    trust_state="TRUSTED", refresh_jti="new-nonce")
        db = FakeDb(user, profile)
        payload = BiometricUnlockRequest(refresh_token=token, device_fingerprint="device-1",
                                         raw_features={})
        with self.assertRaises(HTTPException) as error:
            await auth.biometric_unlock(payload, BackgroundTasks(), db)
        self.assertEqual(error.exception.status_code, 401)

    async def test_biometric_unlock_logs_risk_and_reasons(self):
        user = existing_user()
        token = create_refresh_token(str(user.user_id), "device-1", "current-nonce")
        profile = UserDeviceProfile(user_id=user.user_id, device_fingerprint="device-1",
                                    trust_state="TRUSTED", successful_login_count=3,
                                    refresh_jti="current-nonce")
        db = FakeDb(user, profile)
        payload = BiometricUnlockRequest(refresh_token=token, device_fingerprint="device-1",
                                         raw_features={})
        with patch.object(auth.risk_engine, "evaluate_risk", return_value={
            "uts_score": 0.15, "base_ml_risk": 0.40, "device_adjustment": -0.25,
            "step_up_required": False, "anomaly_detected": False,
            "reasons": ["Recognized hardware: Trusted device discount applied (-0.25)"],
        }):
            with self.assertLogs(auth.logger, level="INFO") as risk_logs:
                response = await auth.biometric_unlock(payload, BackgroundTasks(), db)
        self.assertFalse(response.step_up_required)
        self.assertEqual(response.uts_score, 0.15)
        logged = "\n".join(risk_logs.output)
        self.assertIn("Sign-in method      : Biometric unlock", logged)
        self.assertIn("Engine base risk    : 0.40", logged)
        self.assertIn("Device adjustment   : -0.25", logged)
        self.assertIn("Unified Trust Score : 0.15", logged)
        self.assertIn("1. Recognized hardware: Trusted device discount", logged)
        self.assertNotIn(token, logged)

    async def test_mfa_completes_and_rotates_refresh(self):
        user = existing_user()
        user.mfa_code = "123456"
        user.mfa_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=5)
        user.mfa_device_fingerprint = "device-1"
        profile = UserDeviceProfile(user_id=user.user_id, device_fingerprint="device-1",
                                    trust_state="PROBATIONARY", successful_login_count=1)
        db = FakeDb(user, profile)
        response = await auth.verify_mfa_challenge(
            VerifyMfaRequest(email=user.email, otp_code="123456",
                             device_fingerprint="device-1"), db)
        self.assertTrue(response.access_token and response.refresh_token)
        self.assertIsNone(user.mfa_code)
        self.assertEqual(profile.successful_login_count, 2)


if __name__ == "__main__":
    unittest.main()
