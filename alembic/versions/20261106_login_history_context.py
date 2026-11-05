"""Store login context for the per-user history features of the version 2 risk model.

Revision ID: 20261106_login_history
Revises: 20261103_device_sessions
Create Date: 2026-11-06
"""

from alembic import op


revision = "20261106_login_history"
down_revision = "20261103_device_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IF NOT EXISTS keeps this safe on databases created by create_tables.py.
    op.execute("ALTER TABLE contextual_logins ADD COLUMN IF NOT EXISTS country VARCHAR(2)")
    op.execute("ALTER TABLE contextual_logins ADD COLUMN IF NOT EXISTS asn INTEGER")
    op.execute("ALTER TABLE contextual_logins ADD COLUMN IF NOT EXISTS user_agent VARCHAR(200)")
    op.execute("ALTER TABLE contextual_logins ADD COLUMN IF NOT EXISTS os_family VARCHAR(16)")
    op.execute("ALTER TABLE contextual_logins ADD COLUMN IF NOT EXISTS device_type VARCHAR(16)")
    # The history lookup filters by user and successful logins.
    op.execute("CREATE INDEX IF NOT EXISTS ix_contextual_logins_user_success "
               "ON contextual_logins (user_id, is_successful)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_contextual_logins_user_success")
    for column in ("device_type", "os_family", "user_agent", "asn", "country"):
        op.execute(f"ALTER TABLE contextual_logins DROP COLUMN IF EXISTS {column}")
