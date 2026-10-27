-- Extend existing tables without changing their primary keys or foreign keys.
ALTER TABLE care_sessions ADD COLUMN IF NOT EXISTS end_time TIMESTAMPTZ;
ALTER TABLE care_logs ADD COLUMN IF NOT EXISTS category VARCHAR(30) NOT NULL DEFAULT 'NOTE';

CREATE INDEX IF NOT EXISTS ix_care_logs_user_recorded_at
    ON care_logs (user_id, recorded_at DESC);
