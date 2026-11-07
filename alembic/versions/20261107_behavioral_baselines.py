"""Store per-user rolling baselines for the version 2 behavioral model.

Revision ID: 20261107_behavioral_baselines
Revises: 20261106_login_history
Create Date: 2026-11-07
"""

from alembic import op


revision = "20261107_behavioral_baselines"
down_revision = "20261106_login_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IF NOT EXISTS keeps this safe on databases created by create_tables.py.
    op.execute("ALTER TABLE behavioral_baselines ADD COLUMN IF NOT EXISTS feature_means JSON")
    op.execute("ALTER TABLE behavioral_baselines ADD COLUMN IF NOT EXISTS sample_count INTEGER")
    # The baseline query averages one user's snapshots over the last 30 days.
    op.execute("CREATE INDEX IF NOT EXISTS ix_behavioral_audit_logs_user_time "
               "ON behavioral_audit_logs (user_id, timestamp)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_behavioral_audit_logs_user_time")
    for column in ("sample_count", "feature_means"):
        op.execute(f"ALTER TABLE behavioral_baselines DROP COLUMN IF EXISTS {column}")
