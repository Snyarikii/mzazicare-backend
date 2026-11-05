import contextlib
import io
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from app.services.risk_engine import KNOWN_MALICIOUS_IPS, risk_engine


class FixedRiskModel:
    """Stands in for the trained model so a test controls the base probability exactly."""

    def __init__(self, risk):
        self.risk = risk

    def predict_proba(self, input_data):
        return [[1.0 - self.risk, self.risk]]


class RiskEngineDecisionTests(unittest.TestCase):
    def evaluate(self, device_profile=None, **features):
        with contextlib.redirect_stdout(io.StringIO()):
            return risk_engine.evaluate_risk(features, device_profile=device_profile)

    def test_model_loads_and_four_decision_tiers(self):
        self.assertIsNotNone(risk_engine.model)
        cases = [
            ("daytime", 10, "102.222.140.1", False, "ALLOW", False, False),
            ("early morning", 2, "102.222.140.1", False, "CHALLENGE", True, False),
            ("mock location", 10, "102.222.140.1", True, "DENY", False, True),
            ("flagged IP", 10, "192.0.2.1", False, "DENY", False, True),
        ]
        for name, hour, ip, mock, action, step_up, denied in cases:
            with self.subTest(name=name):
                result = self.evaluate(local_hour=hour, ip_address=ip,
                                       is_physical_device=True, is_mock_location=mock)
                self.assertEqual(result["risk_action"], action)
                self.assertIs(result["step_up_required"], step_up)
                self.assertIs(result["access_denied"], denied)
                self.assertIsInstance(result["base_ml_risk"], float)
                self.assertIsInstance(result["uts_score"], float)
                self.assertIsInstance(result["reasons"], list)
                if name == "early morning":
                    self.assertEqual(result["base_ml_risk"], 0.65)
                    self.assertEqual(result["uts_score"], 0.75)
                if name == "flagged IP":
                    self.assertIn("CRITICAL: IP address detected on high-risk intelligence blacklist",
                                  result["reasons"])

    def test_all_blacklisted_ips_and_explicit_flag_deny(self):
        for ip in (*KNOWN_MALICIOUS_IPS, "102.222.140.1"):
            with self.subTest(ip=ip):
                features = {"local_hour": 10, "ip_address": ip}
                if ip not in KNOWN_MALICIOUS_IPS:
                    features["is_flagged_ip"] = True
                result = self.evaluate(**features)
                self.assertEqual(result["risk_action"], "DENY")
                self.assertTrue(result["access_denied"])
                self.assertEqual(risk_engine._inspect_ip(ip, features.get("is_flagged_ip", False))["risk_modifier"], 0.40)

    def test_virtual_device_is_denied(self):
        result = self.evaluate(local_hour=10, ip_address="102.222.140.1",
                               is_physical_device=False)
        self.assertEqual(result["risk_action"], "DENY")
        self.assertTrue(result["access_denied"])
        self.assertFalse(result["step_up_required"])
        self.assertGreaterEqual(result["uts_score"], 0.90)
        self.assertIn("CRITICAL THREAT: Virtual environment / emulator detected",
                      result["reasons"])

    def test_dormant_trusted_device_requires_mfa(self):
        profile = SimpleNamespace(trust_state="TRUSTED",
                                  last_login_at=datetime.utcnow() - timedelta(days=45))
        result = self.evaluate(device_profile=profile, local_hour=10,
                               ip_address="102.222.140.1",
                               is_physical_device=True, is_mock_location=False)
        self.assertEqual(result["device_adjustment"], 0.35)
        self.assertGreaterEqual(result["uts_score"], 0.50)
        self.assertEqual(result["risk_action"], "CHALLENGE")
        self.assertTrue(result["step_up_required"])
        self.assertFalse(result["access_denied"])
        self.assertIn("Trusted device dormant >30 days: Inactivity step-up penalty (+0.35)",
                      result["reasons"])

    def test_low_model_risk_dormant_device_still_requires_mfa(self):
        # Pin the model probability: this test checks the 0.50 floor for dormant trusted devices,
        # which only applies when the model's risk is low (0.05 + 0.35 = 0.40 is raised to 0.50).
        profile = SimpleNamespace(trust_state="TRUSTED",
                                  last_login_at=datetime.utcnow() - timedelta(days=45))
        with patch.object(risk_engine, "model", FixedRiskModel(0.05)):
            result = self.evaluate(device_profile=profile, local_hour=10,
                                   os_name="iOS", network_type="WIFI",
                                   ip_address="102.222.140.1",
                                   is_physical_device=True, is_mock_location=False)
        self.assertEqual(result["device_adjustment"], 0.35)
        self.assertEqual(result["uts_score"], 0.50)
        self.assertEqual(result["risk_action"], "CHALLENGE")
        self.assertTrue(result["step_up_required"])
        self.assertIn("Dormant trusted device: Minimum MFA challenge threshold applied",
                      result["reasons"])

    def test_action_threshold_boundaries(self):
        for model_risk, expected_score, expected_action in [
            (0.39, 0.49, "ALLOW"),
            (0.40, 0.50, "CHALLENGE"),
            (0.74, 0.84, "CHALLENGE"),
            (0.75, 0.85, "DENY"),
        ]:
            with self.subTest(score=expected_score):
                with patch.object(risk_engine, "model", FixedRiskModel(model_risk)):
                    result = self.evaluate(local_hour=10, ip_address="102.222.140.1")
                self.assertEqual(result["uts_score"], expected_score)
                self.assertEqual(result["risk_action"], expected_action)
                self.assertEqual(result["step_up_required"], expected_action == "CHALLENGE")
                self.assertEqual(result["access_denied"], expected_action == "DENY")


if __name__ == "__main__":
    unittest.main()
