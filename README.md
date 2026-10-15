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
```

The database must exist before initializing tables. For a fresh local database, run:

```powershell
python create_tables.py
```

`create_tables.py` creates tables from the current SQLAlchemy models; it is intended for initial local setup, not schema migrations.

Email delivery is optional for local development. To send verification and MFA codes through SMTP, set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, and `SMTP_PASSWORD` in `.env`. When SMTP credentials are absent, the development mail service prints codes to the server console.

## Run the API

```powershell
uvicorn main:app --reload --port 8000
```

- Health check: `http://127.0.0.1:8000/api/v1/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/api/v1/openapi.json`

All application routes are under `/api/v1`. Protected routes require a bearer token from the authentication flow.

## API areas

| Prefix | Purpose |
| --- | --- |
| `/api/v1/auth` | Registration, email verification, login, and MFA |
| `/api/v1/family` | Elder profiles and care visit bookings for family users |
| `/api/v1/caregiver` | Available requests, exclusive acceptance, assigned sessions, patient records, and visit completion |
| `/api/v1/telemetry` | Behavioral and context risk evaluation |
| `/api/v1/patients` | Synthetic patient records used by the existing demo workflow |

Caregiver patient records are available only to a caregiver assigned to a qualifying care session. Accepting a request removes it from the unassigned queue; completing a visit saves vitals and encounter notes with the session.

## Tests

```powershell
python -m unittest discover -p "test_*.py"
```

The tests cover caregiver access and behavioral evaluation without requiring a running API server.
