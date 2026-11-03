# MzaziCare Backend

FastAPI service for MzaziCare authentication, family care bookings, caregiver visits, and session risk telemetry. The API uses PostgreSQL through SQLAlchemy's async engine.

## Local setup

You need Python 3.13 and a running PostgreSQL database. From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source venv/bin/activate` instead.

Create a `.env` file in the repository root with your own connection details and signing key:

```dotenv
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@localhost:5432/mzazicare_db
SECRET_KEY=replace-with-a-long-random-secret
PHI_ENCRYPTION_KEY=replace-with-a-persistent-fernet-key
```

Generate a dedicated Fernet key with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` and store it securely. Use the same key for every API and migration process. For local development only, an unset `PHI_ENCRYPTION_KEY` is deterministically derived from `SECRET_KEY` and logs a warning; changing either key after data is encrypted requires re-encrypting the stored values first. Keep the key in a secret manager for production and out of source control.

The database must exist before initializing tables. For a fresh local database, run:

```powershell
python create_tables.py
```

`create_tables.py` creates tables from the current SQLAlchemy models; it is intended for initial local setup, not schema migrations.

Google sign-in requires `GOOGLE_CLIENT_IDS` (or `GOOGLE_CLIENT_ID` for one ID) in `.env`, set to the allowed OAuth client ID. Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, and `SMTP_PASSWORD` for step-up email delivery. The API refuses MFA challenges when SMTP credentials are missing.

For an existing database, apply [the Google auth migration](migrations/20261017_google_auth.sql) before starting the updated API. It makes `users.password_hash` nullable and adds Google identity, MFA device binding, and refresh rotation fields. New databases created with `create_tables.py` already include these columns.

Apply [the caregiver verification migration](migrations/20261023_caregiver_verification.sql) to existing databases before starting this API. It adds verification fields, marks existing caregivers as pending, approves family accounts, and seeds a mock registry. Fresh databases initialized with `create_tables.py` also receive the sample registry entries.

Apply [the schedule and care logs migration](migrations/20261023_schedule_care_logs.sql) to existing databases before using `GET /api/v1/caregiver/schedule` or `GET/POST /api/v1/care-logs`. It adds an optional visit end time and a category to the existing `care_logs` table. The API accepts care logs only for patients assigned to the signed-in, verified caregiver.
Completing a visit now also creates a `VISIT_COMPLETION` care log. Previously completed visits remain in `care_sessions`; the migration does not duplicate those historical records.

Apply [the token blacklist migration](migrations/20261027_token_blacklist.sql) to existing databases before starting this version of the API. New databases initialized with `create_tables.py` include the table and device-session columns automatically. New access tokens carry a UUID4 `jti` and expiration; protected routes reject revoked tokens. Logout revokes the access token paired with the submitted device refresh token. Behavioral and risk `DENY` decisions revoke both the affected access token and its device refresh token before returning, and a new login rotates the device's previous access token. Access tokens issued before this migration have no `jti` and must sign in again.

## PHI encryption migration

The ORM encrypts elderly profile `chronic_conditions`, `emergency_contact_phone`, and `residence_address`; care session `special_instructions`, `vitals_recorded`, and `encounter_notes`; and care log `care_notes`. The JSON fields continue to appear as lists or dictionaries in API responses. Medication entries are stored in `care_logs.care_notes`; there is no separate medications column.

For an existing database, apply the earlier SQL migrations, configure a persistent `PHI_ENCRYPTION_KEY`, back up the database, and run:

```powershell
python -m alembic upgrade head
```

This Alembic revision starts from the existing schema established by `create_tables.py` and the earlier SQL migrations. It changes affected columns to PostgreSQL `TEXT` and encrypts existing rows in the same transaction. On a fresh database, run `create_tables.py` first, then `alembic upgrade head` to record the revision. Until the migration runs, reads tolerate legacy plaintext; after it runs, raw PostgreSQL rows contain Fernet tokens while authenticated API responses contain decrypted values. Malformed Fernet tokens raise an error rather than leaking ciphertext. Do not change the encryption key without a planned key rotation.

## Run the API

```powershell
.\venv\Scripts\python.exe -X utf8 -m uvicorn main:app --reload --reload-dir app --host 0.0.0.0 --port 8000
```

Run this from the repository root. Using `python -m uvicorn` through the project virtual environment ensures Uvicorn and SQLAlchemy come from the same installation.

- Health check: `http://127.0.0.1:8000/api/v1/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/api/v1/openapi.json`

All application routes are under `/api/v1`. Protected routes require a bearer token from the authentication flow.

## API areas

| Prefix | Purpose |
| --- | --- |
| `/api/v1/auth` | Google sign-in, biometric session re-evaluation, email MFA, and logout |
| `/api/v1/auth/register-family` | Complete a new family registration after Google identity verification |
| `/api/v1/family` | Elder profiles and care visit bookings for family users |
| `/api/v1/caregiver` | Available requests, exclusive acceptance, assigned sessions, patient records, and visit completion |
| `/api/v1/caregivers/verify` | Complete a new caregiver registration after Google ID token and mock license checks |
| `/api/v1/telemetry` | Behavioral and context risk evaluation |
| `/api/v1/patients` | Synthetic patient records used by the existing demo workflow |

Caregiver routes require an approved license verification. Patient records are available only to a caregiver assigned to a qualifying care session. Accepting a request removes it from the unassigned queue; completing a visit saves vitals and encounter notes with the session.

For a new Google account, `/api/v1/auth/google` returns `NEEDS_REGISTRATION` without creating a user. Family accounts are created by `/api/v1/auth/register-family`. Caregiver accounts are created only after `/api/v1/caregivers/verify` validates the Google ID token, active license, and registered name. Existing pending caregiver accounts can use the authenticated `/api/v1/caregiver/verify` route.

## Tests

```powershell
python -X utf8 -m unittest discover -s tests -p "test_*.py"
```

The tests cover caregiver access and behavioral evaluation without requiring a running API server.
The encryption tests also verify raw ciphertext and decrypted model responses. To test the Alembic upgrade and downgrade against temporary tables in a configured PostgreSQL database, set `PHI_MIGRATION_TEST=1` before running the test suite.
