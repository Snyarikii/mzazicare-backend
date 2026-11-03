"""Opt-in migration smoke test against temporary PostgreSQL tables only."""

import os
import unittest
import uuid

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.core.config import settings
from app.core.security_crypto import decrypt_field


@unittest.skipUnless(os.environ.get("PHI_MIGRATION_TEST") == "1", "requires opt-in PostgreSQL")
class PhiMigrationPostgresTests(unittest.TestCase):
    def test_upgrade_and_downgrade_existing_plaintext(self):
        # Loaded by path because Alembic revision filenames are not import packages.
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "20261028_add_field_level_phi_encryption.py"
        spec = importlib.util.spec_from_file_location("phi_revision", path)
        revision = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(revision)

        url = make_url(settings.DATABASE_URL).set(drivername="postgresql+psycopg2")
        engine = create_engine(url)
        log_id = uuid.uuid4()
        with engine.begin() as connection:
            connection.execute(text("""
                CREATE TEMP TABLE elderly_profiles (
                    id integer PRIMARY KEY, chronic_conditions json NOT NULL,
                    emergency_contact_phone varchar(50), residence_address varchar(255)
                ) ON COMMIT DROP
            """))
            connection.execute(text("""
                CREATE TEMP TABLE care_sessions (
                    id integer PRIMARY KEY, special_instructions text,
                    vitals_recorded json, encounter_notes text
                ) ON COMMIT DROP
            """))
            connection.execute(text("""
                CREATE TEMP TABLE care_logs (log_id uuid PRIMARY KEY, care_notes text) ON COMMIT DROP
            """))
            connection.execute(text("""
                INSERT INTO elderly_profiles VALUES
                (1, '["diabetes"]'::json, '+254700000000', 'Nairobi')
            """))
            connection.execute(text("""
                INSERT INTO care_sessions VALUES
                (1, 'Bring medication', '{"pulse": 72}'::json, 'Stable')
            """))
            connection.execute(text("INSERT INTO care_logs VALUES (:id, 'Medication given')"),
                               {"id": log_id})

            migration_context = MigrationContext.configure(connection)
            with Operations.context(migration_context):
                revision.upgrade()

            raw = connection.execute(text("""
                SELECT chronic_conditions, emergency_contact_phone, residence_address
                FROM elderly_profiles WHERE id = 1
            """)).one()
            self.assertTrue(all(value.startswith("gAAAA") for value in raw))
            self.assertEqual(decrypt_field(raw[0]), '["diabetes"]')
            self.assertEqual(decrypt_field(raw[2]), 'Nairobi')
            self.assertEqual(connection.execute(text(
                "SELECT care_notes FROM care_logs WHERE log_id = :id"
            ), {"id": log_id}).scalar_one()[:5], "gAAAA")

            with Operations.context(migration_context):
                revision.downgrade()
            self.assertEqual(connection.execute(text(
                "SELECT residence_address FROM elderly_profiles WHERE id = 1"
            )).scalar_one(), "Nairobi")
            self.assertEqual(connection.execute(text(
                "SELECT chronic_conditions::text FROM elderly_profiles WHERE id = 1"
            )).scalar_one(), '["diabetes"]')
        engine.dispose()
