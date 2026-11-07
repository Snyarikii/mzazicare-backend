"""Index session_activity_logs for the server-side data export counts (behavioral model v3).

Revision ID: 20261107_data_exports
Revises: 20261107_behavioral_baselines
Create Date: 2026-11-07
"""

from alembic import op


revision = "20261107_data_exports"
down_revision = "20261107_behavioral_baselines"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Each behavioral check counts one user's DATA_EXPORT events.
    op.execute("CREATE INDEX IF NOT EXISTS ix_session_activity_logs_user_event "
               "ON session_activity_logs (user_id, event_type)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_session_activity_logs_user_event")
