import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.api.v1.endpoints import caregiver
from app.api.v1.endpoints.caregiver import (
    ExistingLicenseVerificationRequest, LicenseVerificationRequest,
    verify_existing_license, verify_license,
)
from app.core.dependencies import get_current_verified_caregiver


class CaregiverVerificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.user = SimpleNamespace(
            role="caregiver", full_name="Stephen Nyariki", license_number=None,
            is_verified=False, verification_status="PENDING",
        )
        self.db = MagicMock()
        self.db.execute = AsyncMock(return_value=MagicMock())
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()

    async def test_invalid_or_inactive_license_is_rejected(self):
        self.db.execute.return_value.scalar_one_or_none.return_value = None
        with self.assertRaises(HTTPException) as raised:
            await verify_existing_license(ExistingLicenseVerificationRequest(license_number="CG-2026-999"), self.user, self.db)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, "License number is invalid or unregistered.")
        self.db.commit.assert_not_awaited()
        sql = str(self.db.execute.call_args.args[0].compile(dialect=postgresql.dialect(),
                  compile_kwargs={"literal_binds": True}))
        self.assertIn("CG-2026-999", sql)
        self.assertIn("mock_caregiver_registry.status = 'ACTIVE'", sql)
        with self.assertRaises(HTTPException) as blank:
            await verify_existing_license(ExistingLicenseVerificationRequest(license_number=""), self.user, self.db)
        self.assertEqual(blank.exception.status_code, 400)

    async def test_registry_name_mismatch_is_rejected(self):
        self.db.execute.return_value.scalar_one_or_none.return_value = SimpleNamespace(full_name="Another Person")
        with self.assertRaises(HTTPException) as raised:
            await verify_existing_license(ExistingLicenseVerificationRequest(license_number="CG-2026-001"), self.user, self.db)
        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(raised.exception.detail, "License number is registered under a different name.")
        self.assertFalse(self.user.is_verified)
        self.db.commit.assert_not_awaited()

    async def test_matching_name_approves_caregiver(self):
        self.db.execute.return_value.scalar_one_or_none.return_value = SimpleNamespace(full_name="  STEPHEN   nyariki ")
        result = await verify_existing_license(ExistingLicenseVerificationRequest(license_number=" cg-2026-001 "), self.user, self.db)
        self.assertEqual(result["status"], "APPROVED")
        self.assertEqual(self.user.license_number, "CG-2026-001")
        self.assertTrue(self.user.is_verified)
        self.assertEqual(self.user.verification_status, "APPROVED")
        self.db.commit.assert_awaited_once()
        self.db.refresh.assert_awaited_once_with(self.user)

    async def test_access_requires_caregiver_and_approval(self):
        for role, verified, state in [
            ("caregiver", False, "PENDING"),
            ("caregiver", True, "PENDING"),
            ("family", True, "APPROVED"),
        ]:
            account = SimpleNamespace(role=role, is_verified=verified, verification_status=state)
            with self.assertRaises(HTTPException) as raised:
                await get_current_verified_caregiver(account)
            self.assertEqual(raised.exception.status_code, 403)
        self.user.is_verified = True
        self.user.verification_status = "APPROVED"
        self.assertIs(await get_current_verified_caregiver(self.user), self.user)

    async def test_new_registration_rejects_invalid_license_without_writing(self):
        self.db.execute.return_value.scalar_one_or_none.return_value = None
        with patch.object(caregiver, "verify_google_id_token", new=AsyncMock(
            return_value=("stephen@example.com", "Stephen Nyariki", "google-sub")
        )):
            with self.assertRaises(HTTPException) as raised:
                await verify_license(LicenseVerificationRequest(
                    google_id_token="valid-token", license_number="CG-2026-999"
                ), self.db)
        self.assertEqual(raised.exception.status_code, 400)
        self.db.add.assert_not_called()
        self.db.commit.assert_not_awaited()

    async def test_new_registration_creates_user_only_after_name_match(self):
        self.db.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=None)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=SimpleNamespace(full_name="STEPHEN  NYARIKI"))),
        ]
        with patch.object(caregiver, "verify_google_id_token", new=AsyncMock(
            return_value=("stephen@example.com", "Stephen Nyariki", "google-sub")
        )), patch.object(caregiver, "create_google_account", new=AsyncMock(return_value="session")) as create:
            result = await verify_license(LicenseVerificationRequest(
                google_id_token="valid-token", license_number="CG-2026-001",
                device_fingerprint="device-1",
            ), self.db)
        self.assertEqual(result, "session")
        self.assertEqual(create.await_args.kwargs["role"], "caregiver")
        self.assertEqual(create.await_args.kwargs["license_number"], "CG-2026-001")

    async def test_new_registration_name_mismatch_does_not_write_user(self):
        self.db.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=None)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=SimpleNamespace(full_name="Another Person"))),
        ]
        with patch.object(caregiver, "verify_google_id_token", new=AsyncMock(
            return_value=("stephen@example.com", "Stephen Nyariki", "google-sub")
        )):
            with self.assertRaises(HTTPException) as raised:
                await verify_license(LicenseVerificationRequest(
                    google_id_token="valid-token", license_number="CG-2026-001"
                ), self.db)
        self.assertEqual(raised.exception.status_code, 403)
        self.db.add.assert_not_called()
        self.db.commit.assert_not_awaited()
