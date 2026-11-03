import unittest
import uuid
from datetime import datetime, timezone

from cryptography.fernet import InvalidToken
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security_crypto import decrypt_field, encrypt_field
from app.db.base import Base
from app.db.encrypted_type import EncryptedJSON
from app.models.entities import CareLog, CareSession, ElderlyProfile
from app.schemas.family import CareSessionResponse, ElderlyProfileResponse


class PhiEncryptionTests(unittest.TestCase):
    def test_legacy_json_rows_are_readable_before_migration(self):
        field = EncryptedJSON()
        self.assertEqual(field.process_result_value(["diabetes"], None), ["diabetes"])
        self.assertEqual(field.process_result_value({"pulse": 72}, None), {"pulse": 72})
        self.assertEqual(field.process_result_value('["diabetes"]', None), ["diabetes"])
        with self.assertRaises(InvalidToken):
            field.process_result_value("gAAAAcorrupt-token", None)

    def test_key_fallback_is_stable_and_invalid_explicit_key_fails(self):
        first = Settings(SECRET_KEY="test-signing-secret", PHI_ENCRYPTION_KEY=None, _env_file=None)
        second = Settings(SECRET_KEY="test-signing-secret", PHI_ENCRYPTION_KEY=None, _env_file=None)
        self.assertEqual(first.PHI_ENCRYPTION_KEY, second.PHI_ENCRYPTION_KEY)
        with self.assertRaises(ValueError):
            Settings(PHI_ENCRYPTION_KEY="invalid", _env_file=None)

    def test_plaintext_fallback_does_not_mask_corrupt_tokens(self):
        self.assertIsNone(encrypt_field(None))
        self.assertIsNone(decrypt_field(None))
        self.assertEqual(decrypt_field("legacy address"), "legacy address")
        self.assertEqual(decrypt_field(encrypt_field("Nairobi")), "Nairobi")
        with self.assertRaises(InvalidToken):
            decrypt_field("gAAAAcorrupt-token")

    def test_database_stores_tokens_and_models_return_plain_values(self):
        engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine, tables=[
            ElderlyProfile.__table__, CareSession.__table__, CareLog.__table__,
        ])
        family_id = uuid.uuid4()
        with Session(engine) as session:
            elder = ElderlyProfile(
                family_user_id=family_id, full_name="Mary", age=78,
                chronic_conditions=["diabetes"],
                emergency_contact_phone="+254700000000",
                residence_address="Nairobi",
            )
            session.add(elder)
            session.flush()
            visit = CareSession(
                elderly_id=elder.id, family_user_id=family_id,
                scheduled_date=datetime.now(timezone.utc),
                service_tier="Daily Assistance",
                special_instructions="Bring medication",
                vitals_recorded={"pulse": 72},
                encounter_notes="Stable",
            )
            log = CareLog(
                user_id=family_id, patient_id=str(elder.id),
                care_notes="Medication given", task_status="RECORDED",
            )
            session.add_all([visit, log])
            session.commit()
            elder_id, visit_id, log_id = elder.id, visit.id, log.log_id

        with engine.connect() as connection:
            raw_elder = connection.execute(text(
                "SELECT chronic_conditions, emergency_contact_phone, residence_address "
                "FROM elderly_profiles WHERE id = :id"
            ), {"id": elder_id}).one()
            raw_visit = connection.execute(text(
                "SELECT special_instructions, vitals_recorded, encounter_notes "
                "FROM care_sessions WHERE id = :id"
            ), {"id": visit_id}).one()
            raw_log = connection.execute(text(
                "SELECT care_notes FROM care_logs WHERE log_id = :id"
            ), {"id": log_id.hex}).scalar_one()
            for value in (*raw_elder, *raw_visit, raw_log):
                self.assertTrue(value.startswith("gAAAA"))
            self.assertNotIn("Nairobi", str(raw_elder))
            self.assertNotIn("Medication", raw_log)

        with Session(engine) as session:
            elder = session.get(ElderlyProfile, elder_id)
            visit = session.get(CareSession, visit_id)
            log = session.get(CareLog, log_id)
            self.assertEqual(elder.chronic_conditions, ["diabetes"])
            self.assertEqual(elder.emergency_contact_phone, "+254700000000")
            self.assertEqual(log.care_notes, "Medication given")
            self.assertEqual(visit.vitals_recorded, {"pulse": 72})
            self.assertEqual(ElderlyProfileResponse.model_validate(elder).residence_address, "Nairobi")
            self.assertEqual(CareSessionResponse.model_validate(visit).special_instructions,
                             "Bring medication")
            elder.residence_address = "Mombasa"
            session.commit()

        with engine.connect() as connection:
            updated = connection.execute(text(
                "SELECT residence_address FROM elderly_profiles WHERE id = :id"
            ), {"id": elder_id}).scalar_one()
            self.assertTrue(updated.startswith("gAAAA"))
            self.assertEqual(decrypt_field(updated), "Mombasa")
