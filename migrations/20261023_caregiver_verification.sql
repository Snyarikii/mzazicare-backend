-- Apply to existing PostgreSQL databases before starting the verification API.
CREATE TABLE IF NOT EXISTS mock_caregiver_registry (
    id SERIAL PRIMARY KEY,
    license_number VARCHAR(100) NOT NULL UNIQUE,
    full_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_mock_caregiver_registry_license_number
    ON mock_caregiver_registry (license_number);

INSERT INTO mock_caregiver_registry (license_number, full_name, status)
VALUES
    ('CG-2026-001', 'Stephen Nyariki', 'ACTIVE'),
    ('CG-2026-002', 'Lydia Naliaka', 'ACTIVE'),
    ('CG-2026-003', 'Mercy Atieno', 'ACTIVE')
ON CONFLICT (license_number) DO NOTHING;

ALTER TABLE users ADD COLUMN IF NOT EXISTS license_number VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN;
ALTER TABLE users ADD COLUMN IF NOT EXISTS verification_status VARCHAR(20);

-- Existing caregivers must verify; family accounts retain immediate access.
UPDATE users SET
    is_verified = CASE WHEN role = 'caregiver' THEN FALSE ELSE TRUE END,
    verification_status = CASE WHEN role = 'caregiver' THEN 'PENDING' ELSE 'APPROVED' END
WHERE is_verified IS NULL OR verification_status IS NULL;

ALTER TABLE users ALTER COLUMN is_verified SET DEFAULT FALSE;
ALTER TABLE users ALTER COLUMN is_verified SET NOT NULL;
ALTER TABLE users ALTER COLUMN verification_status SET DEFAULT 'PENDING';
ALTER TABLE users ALTER COLUMN verification_status SET NOT NULL;
