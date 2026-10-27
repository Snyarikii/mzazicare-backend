-- Apply to existing PostgreSQL databases before enabling access-token revocation.
ALTER TABLE user_device_profiles ADD COLUMN IF NOT EXISTS access_jti VARCHAR(36);
ALTER TABLE user_device_profiles ADD COLUMN IF NOT EXISTS access_expires_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS token_blacklist (
    jti VARCHAR(36) PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    revoked_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_token_blacklist_user_id ON token_blacklist (user_id);
CREATE INDEX IF NOT EXISTS ix_token_blacklist_expires_at ON token_blacklist (expires_at);
