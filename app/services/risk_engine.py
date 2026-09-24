import os
import math
import ipaddress
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Any
from datetime import datetime


# Search for the model artifact across standard project locations
POSSIBLE_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "ml_models", "adaptive_risk_model.joblib"),
    os.path.join(os.path.dirname(__file__), "..", "..", "ml", "adaptive_risk_model.joblib"),
    os.path.join(os.path.dirname(__file__), "..", "ml", "adaptive_risk_model.joblib"),
]

# Primary Facility Reference Point (Nairobi, Kenya)
REF_LAT = -1.3032
REF_LNG = 36.8124


class AdaptiveRiskEngine:
    def __init__(self):
        self.model = None
        for path in POSSIBLE_MODEL_PATHS:
            if os.path.exists(path):
                try:
                    self.model = joblib.load(path)
                    print(f"🧠 [Risk Engine] Loaded trained RBA model from: {path}")
                    break
                except Exception as e:
                    print(f"⚠️ [Risk Engine] Failed to load model at {path}: {e}")

        if not self.model:
            print("⚠️ [Risk Engine] RBA model artifact missing. Operating in baseline heuristic mode.")

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
    def _inspect_ip(ip_str: str) -> dict:
        """Derives security features from the incoming client IP address."""
        if not ip_str or ip_str in ["127.0.0.1", "localhost", "::1"]:
            return {
                "is_loopback": True,
                "is_private": True,
                "is_routable": False,
                "risk_modifier": 0.0,
            }

        try:
            ip_obj = ipaddress.ip_address(ip_str)
            is_private = ip_obj.is_private
            is_loopback = ip_obj.is_loopback
            is_routable = not (is_private or is_loopback)
        except ValueError:
            return {
                "is_loopback": False,
                "is_private": False,
                "is_routable": False,
                "risk_modifier": 0.20,  # Malformed IP syntax
            }

        return {
            "is_loopback": is_loopback,
            "is_private": is_private,
            "is_routable": is_routable,
            "risk_modifier": 0.0,
        }

    def evaluate_risk(
            self,
            raw_features: dict, 
            ip_address: Optional[str] = None,
            device_profile: Optional[Any] = None,
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
        os_name = str(raw_features.get("os_name", "") or raw_features.get("os_family", "")).lower()
        brand = str(raw_features.get("device_brand", "") or "").lower()
        if (
            "android" in os_name
            or "release-keys" in os_name
            or "test-keys" in os_name
            or brand in ["samsung", "xiaomi", "oppo", "transsion", "tecno", "infinix", "google", "huawei"]
            or os_name.startswith("samsung/")            
        ):
            os_family = "Android"
        elif "ios" in os_name or "iphone" in os_name or "ipad" in os_name or brand == "apple":
            os_family = "iOS"
        elif "windows" in os_name:
            os_family = "Windows"
        elif "mac" in os_name or "darwin" in os_name:
            os_family = "macOS"
        elif "linux"  in os_name:
            os_family = "Linux"
        else:
            os_family = "Other"

        # 3. Network & RTT Approximation
        network_type = str(raw_features.get("network_type", "UNKNOWN")).upper()
        if "WIFI" in network_type:
            estimated_rtt = 45.0
        elif "CELLULAR" in network_type:
            estimated_rtt = 95.0
        else:
            estimated_rtt = 180.0

        # 4. IP Context Analysis
        ip_analysis = self._inspect_ip(active_ip)

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
        input_data = pd.DataFrame([{
            "device_type": "mobile",
            "os_family": os_family,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "rtt_ms": estimated_rtt,
        }])

        # Print exact feature values passed to the model
        print("\n" + "🔬 " * 8 + " EXACT MODEL INPUT VECTOR " + "🔬 " * 8, flush=True)
        print(f"  • raw local_hour     : {local_hour} (derived: sin={hour_sin:.4f}, cos={hour_cos:.4f})", flush=True)
        print(f"  • os_family          : {os_family} (from raw: '{raw_features.get('os_name')}')", flush=True)
        print(f"  • device_type        : mobile", flush=True)
        print(f"  • network / rtt_ms   : {estimated_rtt}ms (from network: '{network_type}')", flush=True)
        print(f"  • spatial distance   : {dist_km:.2f} km from hospital", flush=True)
        print("-" * 62, flush=True)

        if self.model:
            try:
                probabilities = self.model.predict_proba(input_data)[0]
                base_ml_risk = float(probabilities[1])
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

        if 17 <= local_hour <= 22 and base_ml_risk > 0.40:
            base_ml_risk = 0.40
            reasons.append("Clinical shift Context: Evening handover window dampening applied")
        elif 0 <= local_hour < 5:
            base_ml_risk = max(base_ml_risk, 0.80)
            reasons.append("Clinical shift Context: Early morning high-risk window applied")
        
        device_adjustment = 0.0
        
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
                device_adjustment = 0.00
                reasons.append("Trusted device dormant >30 days: Discount paused")

        # combination of statistical risk with device trust modifier
        preliminary_score = base_ml_risk + device_adjustment

        is_mock = raw_features.get("is_mock_location", False)
        is_physical = raw_features.get("is_physical_device", True)

        if is_mock:
            preliminary_score = max(preliminary_score, 0.92)
            reasons.append("CRITICAL: GPS mock provider / spoofing active")

        if not is_physical:
            preliminary_score = max(preliminary_score, 0.85)
            reasons.append("SECURITY: Virtual environment / emulator detected")

        final_uts = round(min(max(preliminary_score, 0.0), 1.0), 2)

        step_up_required = final_uts >= 0.55
        anomaly_detected = final_uts >= 0.80

        if not reasons:
            reasons.append("Telemetry profile matches trusted caregiver baseline")

        return {
            "uts_score": final_uts,
            "base_ml_risk":round(base_ml_risk, 2),
            "device_adjustment": device_adjustment,
            "anomaly_detected": anomaly_detected,
            "step_up_required": step_up_required,
            "reasons": reasons,
        }


risk_engine = AdaptiveRiskEngine()