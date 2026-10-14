import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.api.v1.endpoints.caregiver import (
    accept_session,
    available_sessions,
    complete_session,
    my_sessions,
    patient_records,
)
from app.schemas.family import CareSessionComplete, PatientRecords


def query_sql(database):
    statement = database.execute.call_args.args[0]
    return str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


def elder_record():
    return SimpleNamespace(
        id=20, family_user_id=uuid4(), created_at=datetime.now(timezone.utc),
        full_name="Elder", age=75, gender="female", care_tier="Daily Assistance",
        chronic_conditions=["Hypertension"], emergency_contact_name="Family",
        emergency_contact_phone="123", residence_address="Home",
    )


def session_record():
    return SimpleNamespace(
        id=10, elderly_id=20, family_user_id=uuid4(), caregiver_id=None,
        scheduled_date=datetime.now(timezone.utc), service_tier="Daily",
        special_instructions="Bring supplies", status="REQUESTED",
        vitals_recorded=None, encounter_notes=None, completed_at=None,
    )


class CaregiverAccessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.caregiver = SimpleNamespace(role="caregiver", user_id=uuid4())
        self.database = MagicMock()
        self.database.execute = AsyncMock(return_value=MagicMock())
        self.database.scalar = AsyncMock()
        self.database.get = AsyncMock()
        self.database.commit = AsyncMock()
        self.database.refresh = AsyncMock()
        self.database.get.return_value = elder_record()

    async def test_family_cannot_list_available_sessions(self):
        family = SimpleNamespace(role="family", user_id=uuid4())
        with self.assertRaises(HTTPException) as raised:
            await available_sessions(family, self.database)
        self.assertEqual(raised.exception.status_code, 403)
        self.database.execute.assert_not_awaited()

    async def test_available_sessions_include_queue_ids_and_elder_summary(self):
        self.database.execute.return_value.all.return_value = [
            (10, 20, datetime.now(timezone.utc), "Daily", "Bring supplies", "REQUESTED",
             "Elder", 75, "Daily Assistance", ["Hypertension"], "Home")
        ]
        sessions = await available_sessions(self.caregiver, self.database)
        self.assertEqual(
            set(sessions[0].model_dump()),
            {"id", "session_id", "elderly_id", "scheduled_date", "service_tier", "special_instructions", "status", "elder"},
        )
        self.assertEqual(sessions[0].id, sessions[0].session_id)
        self.assertEqual(
            set(sessions[0].elder.model_dump()),
            {"id", "elderly_id", "full_name", "age", "care_tier", "chronic_conditions", "residence_address"},
        )
        statement = query_sql(self.database)
        self.assertIn("care_sessions.status = 'REQUESTED'", statement)
        self.assertIn("care_sessions.caregiver_id IS NULL", statement)
        self.assertNotIn("emergency_contact", statement)

    async def test_accept_rejects_already_assigned_session(self):
        session = session_record()
        session.caregiver_id = uuid4()
        self.database.execute.return_value.scalar_one_or_none.return_value = session
        with self.assertRaises(HTTPException) as raised:
            await accept_session(10, self.caregiver, self.database)
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(raised.exception.detail, "This care request has already been accepted by another caregiver")
        self.database.commit.assert_not_awaited()

        session.caregiver_id = None
        await accept_session(10, self.caregiver, self.database)
        self.assertEqual(session.caregiver_id, self.caregiver.user_id)
        self.assertEqual(session.status, "ACCEPTED")
        self.database.commit.assert_awaited_once()
        self.assertEqual(self.database.get.await_args.args[1], 20)

    async def test_my_sessions_filter_by_caregiver_and_prioritize_accepted(self):
        session = session_record()
        session.caregiver_id = self.caregiver.user_id
        session.status = "ACCEPTED"
        self.database.execute.return_value.all.return_value = [(session, elder_record())]
        sessions = await my_sessions(self.caregiver, self.database)
        self.assertEqual(sessions[0].elder.full_name, "Elder")
        statement = query_sql(self.database)
        self.assertIn(str(self.caregiver.user_id), statement)
        self.assertIn("ACCEPTED", statement)
        self.assertIn("scheduled_date DESC", statement)

    async def test_patient_records_require_assignment_and_include_completed_history(self):
        self.database.scalar.return_value = False
        with self.assertRaises(HTTPException) as raised:
            await patient_records(20, self.caregiver, self.database)
        self.assertEqual(raised.exception.status_code, 403)
        self.database.get.assert_not_awaited()
        authorization = str(
            self.database.scalar.call_args.args[0].compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )
        )
        self.assertIn(str(self.caregiver.user_id), authorization)
        for state in ("ACCEPTED", "IN_PROGRESS", "COMPLETED"):
            self.assertIn(state, authorization)

        elder = elder_record()
        encounter = session_record()
        encounter.status = "COMPLETED"
        encounter.completed_at = datetime.now(timezone.utc)
        encounter.encounter_notes = "Visit"
        encounter.vitals_recorded = {"pulse": 72}
        self.database.scalar.return_value = True
        self.database.get.return_value = elder
        self.database.execute.return_value.scalars.return_value.all.return_value = [encounter]
        records = await patient_records(20, self.caregiver, self.database)
        self.assertIs(records["elder"], elder)
        self.assertEqual(records["encounter_history"], [encounter])
        response = PatientRecords.model_validate(records)
        self.assertEqual(response.elder.chronic_conditions, ["Hypertension"])
        self.assertEqual(response.elder.emergency_contact_phone, "123")
        self.assertEqual(response.encounter_history[0].vitals_recorded, {"pulse": 72})
        self.assertEqual(response.encounter_history[0].id, 10)
        statement = query_sql(self.database)
        self.assertIn("care_sessions.status = 'COMPLETED'", statement)
        self.assertIn("completed_at ASC", statement)

    async def test_only_assigned_caregiver_can_complete_once(self):
        session = session_record()
        session.status = "ACCEPTED"
        session.caregiver_id = uuid4()
        self.database.execute.return_value.scalar_one_or_none.return_value = session
        payload = CareSessionComplete(vitals_recorded={"pulse": 72}, encounter_notes="Visit")
        with self.assertRaises(HTTPException) as raised:
            await complete_session(10, payload, self.caregiver, self.database)
        self.assertEqual(raised.exception.status_code, 403)
        self.database.commit.assert_not_awaited()

        session.caregiver_id = self.caregiver.user_id
        await complete_session(10, payload, self.caregiver, self.database)
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.vitals_recorded, {"pulse": 72})
        self.assertEqual(session.encounter_notes, "Visit")
        self.assertIsInstance(session.completed_at, datetime)
        self.assertIsNotNone(session.completed_at.tzinfo)
        self.database.commit.assert_awaited_once()

        with self.assertRaises(HTTPException) as raised:
            await complete_session(10, payload, self.caregiver, self.database)
        self.assertEqual(raised.exception.status_code, 409)
        self.database.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
