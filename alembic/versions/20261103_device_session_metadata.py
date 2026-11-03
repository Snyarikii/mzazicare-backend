"""Track device session metadata for the Active Devices & Sessions view.

Revision ID: 20261103_device_sessions
Revises: 20261028_phi_encryption
Create Date: 2026-11-03
"""

from alembic import op


revision = "20261103_device_sessions"
down_revision = "20261028_phi_encryption"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IF NOT EXISTS keeps this safe on databases created by create_tables.py.
    op.execute("ALTER TABLE user_device_profiles ADD COLUMN IF NOT EXISTS device_name VARCHAR(64)")
    op.execute("ALTER TABLE user_device_profiles ADD COLUMN IF NOT EXISTS model_identifier VARCHAR(64)")
    op.execute("ALTER TABLE user_device_profiles ADD COLUMN IF NOT EXISTS last_ip_address VARCHAR(45)")
    op.execute("ALTER TABLE user_device_profiles ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ")


def downgrade() -> None:
    op.execute("ALTER TABLE user_device_profiles DROP COLUMN IF EXISTS last_seen_at")
    op.execute("ALTER TABLE user_device_profiles DROP COLUMN IF EXISTS last_ip_address")
    op.execute("ALTER TABLE user_device_profiles DROP COLUMN IF EXISTS model_identifier")
    op.execute("ALTER TABLE user_device_profiles DROP COLUMN IF EXISTS device_name")
