import unittest
import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from app.api.v1.endpoints import caregiver
from app.api.v1.endpoints.care_logs import create_care_log, list_care_logs
from app.api.v1.endpoints.caregiver import complete_session, get_caregiver_schedule
from app.schemas.caregiver_activity import CareLogCreate
from app.schemas.family import CareSessionComplete


def sql_of(statement):
    return str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


class CaregiverActivityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.user = SimpleNamespace(
            user_id=uuid.uuid4(), role="caregiver", is_verified=True,
            verification_status="APPROVED",
        )
        self.db = MagicMock()
        self.db.execute = AsyncMock()
        self.db.execute.return_value = MagicMock()
        self.db.scalar = AsyncMock()
        self.db.get = AsyncMock()
        self.db.commit = AsyncMock()
        self.db.refresh = AsyncMock()

    async def test_schedule_is_scoped_to_caregiver_and_selected_day(self):
        visit = SimpleNamespace(
            id=12, scheduled_date=datetime(2026, 10, 23, 9, tzinfo=timezone.utc),
            end_time=None, service_tier="Daily Assistance", status="ACCEPTED",
            encounter_notes=None, special_instructions="Bring supplies",
        )
        elder = SimpleNamespace(id=4, full_name="Mary", residence_address="Nairobi")
        self.db.execute.return_value.all.return_value = [(visit, elder)]
        result = await get_caregiver_schedule(date(2026, 10, 23), self.user, self.db)
        self.assertEqual(result[0].status, "UPCOMING")
        self.assertEqual(result[0].patient_name, "Mary")
        self.assertIsNone(result[0].end_time)
        query = sql_of(self.db.execute.call_args.args[0])
        self.assertIn(str(self.user.user_id), query)
        self.assertIn("care_sessions.caregiver_id", query)
        self.assertIn("care_sessions.scheduled_date >=", query)
        self.assertIn("care_sessions.scheduled_date <", query)

    async def test_logs_are_scoped_to_caregiver_and_patient(self):
        self.db.execute.return_value.all.return_value = []
        await list_care_logs("4", self.user, self.db)
        query = sql_of(self.db.execute.call_args.args[0])
        self.assertIn(str(self.user.user_id), query)
        self.assertIn("care_logs.patient_id = '4'", query)
        self.assertIn("care_logs.recorded_at DESC", query)

    async def test_create_rejects_unassigned_patient(self):
        self.db.get.return_value = SimpleNamespace(id=4, full_name="Mary")
        self.db.scalar.return_value = False
        with self.assertRaises(HTTPException) as raised:
            await create_care_log(CareLogCreate(patient_id="4", category="NOTE", notes="Observation"), self.user, self.db)
        self.assertEqual(raised.exception.status_code, 403)
        self.db.add.assert_not_called()
        self.db.commit.assert_not_awaited()
        query = sql_of(self.db.scalar.call_args.args[0])
        self.assertIn(str(self.user.user_id), query)

    async def test_create_writes_valid_log_for_assigned_patient(self):
        self.db.get.return_value = SimpleNamespace(id=4, full_name="Mary")
        self.db.scalar.return_value = True
        payload = CareLogCreate(patient_id="4", category="VITALS", notes=" BP 120/80 ")
        async def fill_defaults(log):
            log.log_id = uuid.uuid4()
            log.recorded_at = datetime.now(timezone.utc)
        self.db.refresh.side_effect = fill_defaults
        created = await create_care_log(payload, self.user, self.db)
        self.assertEqual(created.category, "VITALS")
        self.assertEqual(created.notes, "BP 120/80")
        self.assertEqual(created.caregiver_id, self.user.user_id)
        self.db.commit.assert_awaited_once()

    async def test_completed_visit_creates_care_log(self):
        visit = SimpleNamespace(id=9, elderly_id=4, caregiver_id=self.user.user_id, status="ACCEPTED")
        self.db.execute.return_value.scalar_one_or_none.return_value = visit
        self.db.get.return_value = SimpleNamespace(full_name="Mary")
        with patch.object(caregiver, "session_response", return_value="completed"):
            response = await complete_session(
                9, CareSessionComplete(encounter_notes="Visit finished"), self.user, self.db
            )
        self.assertEqual(response, "completed")
        log = self.db.add.call_args.args[0]
        self.assertEqual(log.category, "VISIT_COMPLETION")
        self.assertEqual(log.patient_id, "4")
        self.assertEqual(log.user_id, self.user.user_id)
        self.db.commit.assert_awaited_once()

    def test_category_and_blank_notes_are_invalid(self):
        for category, notes in [("OTHER", "test"), ("NOTE", "  ")]:
            with self.assertRaises(ValidationError):
                CareLogCreate(patient_id="4", category=category, notes=notes)
