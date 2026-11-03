"""Add field-level PHI encryption to the existing schema.

Revision ID: 20261028_phi_encryption
Revises:
Create Date: 2026-10-28

The prior schema is managed by create_tables.py and migrations/*.sql. Apply those
schema changes before this revision on an existing database.
"""

from alembic import op
import sqlalchemy as sa

from app.core.security_crypto import decrypt_field, encrypt_field
from app.db.encrypted_type import EncryptedJSON, EncryptedString


revision = "20261028_phi_encryption"
down_revision = None
branch_labels = None
depends_on = None


_COLUMNS = {
    "elderly_profiles": ("id", (
        "chronic_conditions", "emergency_contact_phone", "residence_address",
    )),
    "care_sessions": ("id", (
        "special_instructions", "vitals_recorded", "encounter_notes",
    )),
    "care_logs": ("log_id", ("care_notes",)),
}


def _rewrite_rows(encrypt: bool) -> None:
    connection = op.get_bind()
    for table, (primary_key, columns) in _COLUMNS.items():
        for column in columns:
            rows = connection.execute(sa.text(
                f"SELECT {primary_key}, {column} FROM {table} WHERE {column} IS NOT NULL"
            ))
            for row_id, stored_value in rows:
                if encrypt:
                    if stored_value.startswith("gAAAA"):
                        decrypt_field(stored_value)  # Validate any already-encrypted row.
                        continue
                    replacement = encrypt_field(stored_value)
                else:
                    replacement = decrypt_field(stored_value)
                connection.execute(
                    sa.text(f"UPDATE {table} SET {column} = :value WHERE {primary_key} = :row_id"),
                    {"value": replacement, "row_id": row_id},
                )


def upgrade() -> None:
    op.alter_column("elderly_profiles", "chronic_conditions", existing_type=sa.JSON(),
                    type_=EncryptedJSON(), postgresql_using="chronic_conditions::text")
    op.alter_column("elderly_profiles", "emergency_contact_phone", existing_type=sa.String(50),
                    type_=EncryptedString())
    op.alter_column("elderly_profiles", "residence_address", existing_type=sa.String(255),
                    type_=EncryptedString())
    op.alter_column("care_sessions", "special_instructions", existing_type=sa.Text(),
                    type_=EncryptedString())
    op.alter_column("care_sessions", "vitals_recorded", existing_type=sa.JSON(),
                    type_=EncryptedJSON(), postgresql_using="vitals_recorded::text")
    op.alter_column("care_sessions", "encounter_notes", existing_type=sa.Text(),
                    type_=EncryptedString())
    op.alter_column("care_logs", "care_notes", existing_type=sa.Text(),
                    type_=EncryptedString())
    _rewrite_rows(encrypt=True)


def downgrade() -> None:
    _rewrite_rows(encrypt=False)
    op.alter_column("care_logs", "care_notes", existing_type=EncryptedString(),
                    type_=sa.Text())
    op.alter_column("care_sessions", "encounter_notes", existing_type=EncryptedString(),
                    type_=sa.Text())
    op.alter_column("care_sessions", "vitals_recorded", existing_type=EncryptedJSON(),
                    type_=sa.JSON(), postgresql_using="vitals_recorded::json")
    op.alter_column("care_sessions", "special_instructions", existing_type=EncryptedString(),
                    type_=sa.Text())
    op.alter_column("elderly_profiles", "residence_address", existing_type=EncryptedString(),
                    type_=sa.String(255))
    op.alter_column("elderly_profiles", "emergency_contact_phone", existing_type=EncryptedString(),
                    type_=sa.String(50))
    op.alter_column("elderly_profiles", "chronic_conditions", existing_type=EncryptedJSON(),
                    type_=sa.JSON(), postgresql_using="chronic_conditions::json")
