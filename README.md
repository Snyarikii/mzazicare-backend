# MzaziCare Backend 🩺🛡️

The secure, Zero-Trust backend service for **MzaziCare**, an adaptive geriatric care and family-to-elderly platform built for the East African healthcare ecosystem. MzaziCare features a dual-layer machine learning security architecture combining perimeter authentication risk scoring with continuous User and Entity Behavior Analytics (UEBA).

---

## 🚀 Key Features

- **Geriatric Care Management**: Endpoints for managing elderly dependent profiles (chronic conditions, care tiers, daily vitals, and emergency contacts) sponsored by family members.
- **Care Session Workflows**: Full lifecycle management for home care bookings—from family requests (`REQUESTED`) to caregiver acceptance (`ACCEPTED`) and clinical encounter logging (`COMPLETED`).
- **Perimeter Risk Engine (`risk_engine.py`)**: A Random Forest classifier evaluated at login that scores environmental/network context to prevent unauthorized access while avoiding false-positive MFA prompts during evening clinical handovers.
- **Continuous UEBA Engine (`behavioral_engine.py`)**: An Isolation Forest model trained on 200k+ enterprise user-day profiles (CMU CERT r4.2 streams) that evaluates continuous session telemetry (logon velocity, multi-device hopping, off-hours activity, and record payload volumes) to enforce real-time session revocation.
- **Data Privacy & Synthetic Fixtures**: Fully instrumented with realistic Kenyan geriatric synthetic profiles marked `synthetic_demo` to ensure zero exposure of real patient PII during development and demonstrations.

---

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python 3.10+)
- **Database & ORM**: PostgreSQL, SQLAlchemy, Pydantic v2
- **Machine Learning**: Scikit-Learn, Joblib, Pandas, NumPy
- **Authentication**: OAuth2 with JWT Bearer tokens & role-based access control (`family`, `caregiver`, `admin`)

---

## 📂 Project Structure

```text
mzazicare-backend/
├── app/
│   ├── api/v1/endpoints/        # Routers: auth, family, caregiver, telemetry, patients
│   ├── core/                    # Config, security, and dependencies
│   ├── db/                      # Database session and base setup
│   ├── ml/                      # Serialized ML joblib artifacts (.joblib)
│   ├── models/                  # SQLAlchemy entities (User, ElderlyProfile, CareSession, BehavioralAuditLog)
│   ├── schemas/                 # Pydantic validation models
│   └── services/                # Business logic & ML inference engines (risk_engine, behavioral_engine)
├── ml/
│   ├── data/                    # Training CSV streams (logon.csv, file.csv)
│   └── train_behavioral_model.py # UEBA Isolation Forest training pipeline
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
└── .env                         # Environment variables

```
## Getting Started
1. Clone and Setup Environment
```text
git clone [https://github.com/Snyarikii/mzazicare-backend.git](https://github.com/Snyarikii/mzazicare-backend.git)
cd mzazicare-backend
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
2. Install Dependencies
```text
pip install -r requirements.txt
```
3. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory:

```text
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/mzazicare
SECRET_KEY=your_secret_jwt_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```
4. Run the Development Server
```text
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API Documentation `http://127.0.0.1:8000/docs`


## 🔒 Security and UEBA Testing
To run a simulation check against the behavioral anomaly detection endpoint

```text
curl -X POST "[http://127.0.0.1:8000/api/v1/telemetry/behavioral-check](http://127.0.0.1:8000/api/v1/telemetry/behavioral-check)" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
      "total_logons": 1,
      "distinct_devices": 1,
      "activity_span_hours": 3.0,
      "off_hours_ratio": 0.0,
      "records_accessed_count": 5,
      "total_payload_bytes": 12000
    }'
```
