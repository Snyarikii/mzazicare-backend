import os
import math
import ipaddress
from itertools import combinations
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Any
from datetime import datetime

from app.services.token_blacklist import blacklist_token


# Search for the model artifact across standard project locations
POSSIBLE_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "ml_models", "adaptive_risk_model.joblib"),
    os.path.join(os.path.dirname(__file__), "..", "..", "ml", "adaptive_risk_model.joblib"),
    os.path.join(os.path.dirname(__file__), "..", "ml", "adaptive_risk_model.joblib"),
]

# Primary Facility Reference Point (Nairobi, Kenya)
REF_LAT = -1.3032
REF_LNG = 36.8124
KNOWN_MALICIOUS_IPS = {"192.0.2.1", "198.51.100.45", "203.0.113.99"}

# Version 2 model: SHAP is reported per feature group, which keeps exact Shapley values cheap
# (2^6 = 64 model calls instead of 2^13) and reads better on the admin dashboard.
V2_GROUPS = {
    "Device": ["device_type", "os_family"],
    "Time Of Day": ["hour_sin", "hour_cos"],
    "Network Speed": ["rtt_ms"],
    "Location History": ["new_country", "new_asn"],
    "Device History": ["new_user_agent", "new_os", "new_device_type"],
    "Account Activity": ["first_login", "log_prior_logins", "log_hours_since_last"],
}
# A login from a country the account has never used forces at least a CHALLENGE once the account
# has this many earlier successful logins, whatever the device trust (a stolen trusted phone used
# abroad must still be verified). Brand-new accounts are exempt: every login looks "new" to them.
NEW_COUNTRY_MIN_PRIOR_LOGINS = 3

HISTORY_REASONS = {
    "new_country": "Login from a country not seen before for this account",
    "new_asn": "Login from a network not seen before for this account",
    "new_user_agent": "App signature (device or OS version) not seen before for this account",
    "new_os": "Operating system not seen before for this account",
    "new_device_type": "Device type not seen before for this account",
}


def normalise_os_family(raw_features: dict) -> str:
    """Map the app's OS and brand details to the OS families the model was trained on."""
    os_name = str(raw_features.get("os_name", "") or raw_features.get("os_family", "")).lower()
    brand = str(raw_features.get("device_brand", "") or "").lower()
    if (
        "android" in os_name
        or "release-keys" in os_name
        or "test-keys" in os_name
        or brand in ["samsung", "xiaomi", "oppo", "transsion", "tecno", "infinix", "google", "huawei"]
        or os_name.startswith("samsung/")
    ):
        return "Android"
    if "ios" in os_name or "iphone" in os_name or "ipad" in os_name or brand == "apple":
        return "iOS"
    if "windows" in os_name:
        return "Windows"
    if "mac" in os_name or "darwin" in os_name:
        return "MacOS"
    if "linux" in os_name:
        return "Linux"
    return "Other"


class AdaptiveRiskEngine:
    def __init__(self):
        self.model = None
        self.model_version = 1
        self.reference = None
        self.history_features: list[str] = []
        for path in POSSIBLE_MODEL_PATHS:
            if os.path.exists(path):
                try:
                    self.load_artifact(joblib.load(path))
                    print(f"🧠 [Risk Engine] Loaded trained RBA model (version {self.model_version}) from: {path}")
                    break
                except Exception as e:
                    print(f"⚠️ [Risk Engine] Failed to load model at {path}: {e}")

        if not self.model:
            print("⚠️ [Risk Engine] RBA model artifact missing. Operating in baseline heuristic mode.")

    def load_artifact(self, artifact) -> None:
        """Accept the version 1 pipeline or the version 2 bundle saved by ml/train_risk_model_v2.py."""
        if isinstance(artifact, dict) and artifact.get("version") == 2:
            self.model = artifact["pipeline"]
            self.model_version = 2
            self.reference = dict(artifact["reference"])
            self.history_features = list(artifact["history_features"])
        else:
            self.model = artifact
            self.model_version = 1
            self.reference = None
            self.history_features = []

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates geographic distance in kilometers between two GPS points."""
        R = 6371.0  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def _inspect_ip(ip_str: str, is_flagged_ip: bool = False) -> dict:
        """Derives security features from the incoming client IP address."""
        flagged = ip_str in KNOWN_MALICIOUS_IPS or bool(is_flagged_ip)
        if not ip_str or ip_str in ["127.0.0.1", "localhost", "::1"]:
            analysis = {
                "is_loopback": True,
                "is_private": True,
                "is_routable": False,
                "risk_modifier": 0.0,
            }
        else:
            try:
                ip_obj = ipaddress.ip_address(ip_str)
                is_private = ip_obj.is_private
                is_loopback = ip_obj.is_loopback
                analysis = {
                    "is_loopback": is_loopback,
                    "is_private": is_private,
                    "is_routable": not (is_private or is_loopback),
                    "risk_modifier": 0.0,
                }
            except ValueError:
                analysis = {
                    "is_loopback": False,
                    "is_private": False,
                    "is_routable": False,
                    "risk_modifier": 0.20,  # Malformed IP syntax
                }

        if flagged:
            analysis["risk_modifier"] = 0.40
        analysis["is_critical_threat"] = flagged
        return analysis

    def _grouped_shap_attributions(self, input_data: pd.DataFrame) -> list[dict]:
        """Exact interventional Shapley values for the six feature groups of the version 2 model.

        The reference is a routine login (daytime Android on Wi-Fi from a familiar country, network
        and device, about 20 earlier logins). Values explain the model probability before the rules.
        """
        groups = list(V2_GROUPS.items())
        count = len(groups)
        row = input_data.iloc[0]
        batch = pd.DataFrame([
            {column: (row[column] if mask & (1 << index) else self.reference[column])
             for index, (_, columns) in enumerate(groups) for column in columns}
            for mask in range(1 << count)
        ], columns=list(input_data.columns))
        probabilities = self.model.predict_proba(batch)[:, 1]
        factorial = math.factorial
        values = []
        for index, (name, _) in enumerate(groups):
            contribution = 0.0
            others = [position for position in range(count) if position != index]
            for size in range(count):
                weight = factorial(size) * factorial(count - size - 1) / factorial(count)
                for subset in combinations(others, size):
                    mask = sum(1 << position for position in subset)
                    contribution += weight * (probabilities[mask | (1 << index)] - probabilities[mask])
            values.append({"feature": name, "impact": round(float(contribution), 4),
                           "positive": bool(contribution > 0)})
        return values

    def _model_shap_attributions(self, input_data: pd.DataFrame) -> list[dict]:
        """Exact interventional Shapley values for the five model inputs.

        The fixed reference describes a routine Nairobi daytime Wi-Fi session.
        Values explain the model probability, before policy adjustments.
        """
        if self.model is None:
            return []
        columns = list(input_data.columns)
        reference = {"device_type": "mobile", "os_family": "Android",
                     "hour_sin": 0.0, "hour_cos": -1.0, "rtt_ms": 45.0}
        masks = range(1 << len(columns))
        batch = pd.DataFrame([
            {name: input_data.iloc[0][name] if mask & (1 << index) else reference[name]
             for index, name in enumerate(columns)} for mask in masks
        ], columns=columns)
        probabilities = self.model.predict_proba(batch)[:, 1]
        factorial = math.factorial
        count = len(columns)
        values = []
        for index, name in enumerate(columns):
            contribution = 0.0
            other_indices = [position for position in range(count) if position != index]
            for subset_size in range(count):
                weight = factorial(subset_size) * factorial(count - subset_size - 1) / factorial(count)
                for subset in combinations(other_indices, subset_size):
                    mask = sum(1 << position for position in subset)
                    contribution += weight * (probabilities[mask | (1 << index)] - probabilities[mask])
            values.append({"feature": name.replace("_", " ").title(),
                           "impact": round(float(contribution), 4),
                           "positive": bool(contribution > 0)})
        return values

    def evaluate_risk(
            self,
            raw_features: dict, 
            ip_address: Optional[str] = None,
            device_profile: Optional[Any] = None,
            history: Optional[dict] = None,
        ) -> dict:
        """
        Evaluates real-time perimeter telemetry and computes the Unified Trust Score (UTS).
        """
        raw_features = raw_features or {}
        active_ip = ip_address or raw_features.get("ip_address", "127.0.0.1")

        reasons = []

        # 1. Temporal Feature Engineering
        local_hour = int(raw_features.get("local_hour", 12))
        day_of_week = int(raw_features.get("day_of_week", 0))
        hour_sin = np.sin(2 * np.pi * local_hour / 24.0)
        hour_cos = np.cos(2 * np.pi * local_hour / 24.0)

        # 2. OS Family Normalization
        os_family = normalise_os_family(raw_features)

        # 3. Network & RTT Approximation
        network_type = str(raw_features.get("network_type", "UNKNOWN")).upper()
        if "WIFI" in network_type:
            estimated_rtt = 45.0
        elif "CELLULAR" in network_type:
            estimated_rtt = 95.0
        else:
            estimated_rtt = 180.0

        # 4. IP Context Analysis
        ip_analysis = self._inspect_ip(active_ip, raw_features.get("is_flagged_ip", False))
        is_critical_threat = ip_analysis["is_critical_threat"]
        if is_critical_threat:
            reasons.append("CRITICAL: IP address detected on high-risk intelligence blacklist")

        # 5. Geolocation / Spatial Check
        latitude = raw_features.get("latitude")
        longitude = raw_features.get("longitude")
        country = "KE"  # Baseline deployment region

        if latitude is not None and longitude is not None:
            dist_km = self._haversine_distance(REF_LAT, REF_LNG, float(latitude), float(longitude))
            if dist_km > 75.0:
                reasons.append(f"Spatial anomaly: Device is {dist_km:.1f} km from primary care facility")
        else:
            dist_km = 0.0

        # 6. ML Model Inference
        model_row = {
            "device_type": "mobile",
            "os_family": os_family,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "rtt_ms": estimated_rtt,
        }
        if self.model_version == 2:
            # Without history (lookup failed or unavailable) use the routine-login reference values,
            # so a missing history neither raises nor lowers the risk.
            source = history if history else self.reference
            for feature in self.history_features:
                model_row[feature] = float(source.get(feature, self.reference[feature]))
            if history:
                if history.get("first_login"):
                    reasons.append("First recorded login for this account")
                for feature, text in HISTORY_REASONS.items():
                    if history.get(feature) and not history.get("first_login"):
                        reasons.append(text)
        input_data = pd.DataFrame([model_row])

        # Print exact feature values passed to the model
        print("\n" + "🔬 " * 8 + " EXACT MODEL INPUT VECTOR " + "🔬 " * 8, flush=True)
        print(f"  • raw local_hour     : {local_hour} (derived: sin={hour_sin:.4f}, cos={hour_cos:.4f})", flush=True)
        print(f"  • os_family          : {os_family} (from raw: '{raw_features.get('os_name')}')", flush=True)
        print(f"  • device_type        : mobile", flush=True)
        print(f"  • network / rtt_ms   : {estimated_rtt}ms (from network: '{network_type}')", flush=True)
        print(f"  • spatial distance   : {dist_km:.2f} km from hospital", flush=True)
        print("-" * 62, flush=True)

        shap_attributions = []
        if self.model:
            try:
                probabilities = self.model.predict_proba(input_data)[0]
                base_ml_risk = float(probabilities[1])
                try:
                    shap_attributions = (self._grouped_shap_attributions(input_data)
                                         if self.model_version == 2
                                         else self._model_shap_attributions(input_data))
                except Exception as exc:
                    print(f"[Risk Engine] Attribution unavailable: {exc}")
                print("\n" + "🔍 " * 8 + " MODEL DIAGNOSTICS " + "🔍 " * 8, flush=True)
                print(f"Model Classes: {getattr(self.model, 'classes_', 'N/A')}", flush=True)
                print(f"Raw Probabilities Array: {probabilities}", flush=True)
                if hasattr(self.model, "feature_names_in_") and hasattr(self.model, "feature_importances_"):
                    importances = dict(zip(self.model.feature_names_in_, [round(x, 4) for x in self.model.feature_importances_]))
                    print(f"Feature Importances: {importances}", flush=True)
                print("=" * 60 + "\n", flush=True)
            except Exception as e:
                print(f"⚠️ [Risk Engine] Inference failed, using fallback score: {e}")
                base_ml_risk = 0.15
        else:
            base_ml_risk = 0.15  

        # Stateful Device Profile Trust Adjustments

        # The former evening cap (17:00-22:59 -> at most 0.40) was removed: it offset the version 1
        # model's hour-of-day bias, and with the history model it would hide evening takeovers.
        if 0 <= local_hour < 5:
            base_ml_risk = max(base_ml_risk, 0.65)
            reasons.append("Clinical shift Context: Early morning high-risk window applied")
        
        device_adjustment = 0.0
        dormant_trusted_device = False
        
        if device_profile is None or device_profile.trust_state == "NEW":
            device_adjustment = +0.10
            reasons.append("Unrecognized hardware: First-time device enrollment penalty (+0.10)")
        elif device_profile.trust_state == "PROBATIONARY":
            device_adjustment = 0.00
            reasons.append(f"Probationary device ({device_profile.successful_login_count}/3 verified sessions)")
        elif device_profile.trust_state == "TRUSTED":
            # Apply trust discount only if last login was within 30 days
            days_inactive = (datetime.utcnow() - device_profile.last_login_at).days if device_profile.last_login_at else 0
            if days_inactive <=30:
                device_adjustment = -0.25
                reasons.append("Recognized hardware: Trusted device discount applied (-0.25)")
            else:
                device_adjustment = +0.35
                dormant_trusted_device = True
                reasons.append("Trusted device dormant >30 days: Inactivity step-up penalty (+0.35)")

        # combination of statistical risk with device trust modifier
        preliminary_score = base_ml_risk + device_adjustment + ip_analysis["risk_modifier"]
        if dormant_trusted_device and preliminary_score < 0.50:
            preliminary_score = 0.50
            reasons.append("Dormant trusted device: Minimum MFA challenge threshold applied")

        if (self.model_version == 2 and history and history.get("new_country")
                and not history.get("first_login")
                and math.expm1(float(history.get("log_prior_logins", 0.0))) + 1e-9 >= NEW_COUNTRY_MIN_PRIOR_LOGINS):
            preliminary_score = max(preliminary_score, 0.50)
            reasons.append("New country for an established account: step-up verification required")

        is_mock = raw_features.get("is_mock_location", False)
        is_physical = raw_features.get("is_physical_device", True)

        if is_mock:
            preliminary_score = max(preliminary_score, 0.95)
            is_critical_threat = True
            reasons.append("CRITICAL THREAT: GPS mock provider / location spoofing active")

        if not is_physical:
            preliminary_score = max(preliminary_score, 0.90)
            is_critical_threat = True
            reasons.append("CRITICAL THREAT: Virtual environment / emulator detected")

        final_uts = round(min(max(preliminary_score, 0.0), 1.0), 2)

        if is_critical_threat or final_uts >= 0.85:
            risk_action = "DENY"
        elif final_uts >= 0.50:
            risk_action = "CHALLENGE"
        else:
            risk_action = "ALLOW"

        anomaly_detected = final_uts >= 0.80

        if not reasons:
            reasons.append("Telemetry profile matches trusted caregiver baseline")

        return {
            "uts_score": final_uts,
            "base_ml_risk":round(base_ml_risk, 2),
            "device_adjustment": device_adjustment,
            "anomaly_detected": anomaly_detected,
            "risk_action": risk_action,
            "step_up_required": risk_action == "CHALLENGE",
            "access_denied": risk_action == "DENY",
            "reasons": reasons,
            "shap_attributions": shap_attributions,
            "model_version": self.model_version,
            "history_features": history if self.model_version == 2 else None,
        }

    async def evaluate_risk_and_revoke(
            self, raw_features: dict, ip_address: Optional[str] = None,
            device_profile: Optional[Any] = None, history: Optional[dict] = None, *,
            current_jti: str | None = None,
            user_id: str | None = None, expires_at: datetime | None = None,
        ) -> dict:
        assessment = self.evaluate_risk(
            raw_features=raw_features, ip_address=ip_address,
            device_profile=device_profile, history=history,
        )
        if (assessment.get("access_denied", False) and current_jti and user_id
                and expires_at is not None):
            await blacklist_token(current_jti, user_id, expires_at,
                                  revoke_refresh=True)
        return assessment


risk_engine = AdaptiveRiskEngine()
