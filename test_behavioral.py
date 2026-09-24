import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000/api/v1/telemetry/behavioral-check"
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3OTAxOTkxODUsInN1YiI6IjViZjZkYzZjLWFmN2QtNDc5YS1hMTI4LWM5Nzg4ZTJiYTg5YSJ9.8hDZMGDQwqLv8_Nu1w6T2oOU3C97zJRYhD6Q3cByI9Y"



def send_behavioral_check(payload: dict, label: str):
    print(f"\n🧪 Testing {label}...")
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
            print(f"Status Code: {status}")
            print(f"Response   : {json.dumps(body, indent=2)}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"Connection failed: {e}")


# 1. Normal Shift Simulation
normal_payload = {
    "total_logons": 1,
    "distinct_devices": 1,
    "activity_span_hours": 4.5,
    "off_hours_ratio": 0.0,
    "records_accessed_count": 8,
    "total_payload_bytes": 12400,
}
send_behavioral_check(normal_payload, "Scenario 1: Normal Caregiver Shift")

# 2. Bulk Exfiltration / Anomalous Activity
anomaly_payload = {
    "total_logons": 7,
    "distinct_devices": 4,
    "activity_span_hours": 14.0,
    "off_hours_ratio": 0.85,
    "records_accessed_count": 185,
    "total_payload_bytes": 1450000,
}
send_behavioral_check(anomaly_payload, "Scenario 2: Data Exfiltration / Multi-Device Hopping")