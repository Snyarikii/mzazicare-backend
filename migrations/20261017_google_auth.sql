-- Apply to existing PostgreSQL databases before starting the Google auth API.
ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(20) NOT NULL DEFAULT 'google';
ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS mfa_device_fingerprint VARCHAR(128);
CREATE UNIQUE INDEX IF NOT EXISTS uq_users_google_sub ON users (google_sub);
ALTER TABLE user_device_profiles ADD COLUMN IF NOT EXISTS refresh_jti VARCHAR(64);
