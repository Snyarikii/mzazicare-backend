import gzip
import math
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi import Request
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.entities import ContextualLogin
from app.services import login_history
from app.services.login_history import AsnLookup, LoginContext, features_from_summary
from app.services.risk_engine import AdaptiveRiskEngine, V2_GROUPS


def http_request(headers=None, client=("10.0.0.5", 1234)):
    return Request({
        "type": "http", "client": client,
        "headers": [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()],
    })


def context(**values):
    base = {"ip": "41.90.1.1", "country": "KE", "asn": 33771, "user_agent": "MzaziCare (samsung SM-A145F; Android 15)",
            "os_family": "Android"}
    base.update(values)
    return LoginContext(**base)


class HeaderTests(unittest.TestCase):
    def test_cloudflare_ip_and_country_are_preferred(self):
        request = http_request({"CF-Connecting-IP": "41.90.1.1", "X-Forwarded-For": "8.8.8.8, 10.0.0.1",
                                "CF-IPCountry": "ke"})
        self.assertEqual(login_history.client_ip(request), "41.90.1.1")
        self.assertEqual(login_history.client_country(request), ("KE", False))

    def test_forwarded_then_socket_address_are_fallbacks(self):
        self.assertEqual(login_history.client_ip(http_request({"X-Forwarded-For": "8.8.8.8, 10.0.0.1"})), "8.8.8.8")
        self.assertEqual(login_history.client_ip(http_request()), "10.0.0.5")

    def test_unknown_or_malformed_country_is_none(self):
        self.assertEqual(login_history.client_country(http_request({"CF-IPCountry": "XX"})), (None, False))
        self.assertEqual(login_history.client_country(http_request({"CF-IPCountry": "Kenya"})), (None, False))
        self.assertEqual(login_history.client_country(http_request()), (None, False))

    def test_simulated_country_overrides_cloudflare(self):
        request = http_request({"CF-IPCountry": "KE", "X-Simulate-Country": "ru"})
        self.assertEqual(login_history.client_country(request), ("RU", True))

    def test_app_signature_normalises_android_build_strings(self):
        signature = login_history.app_signature({
            "device_brand": "samsung", "device_model": "SM-A145F",
            "os_name": "samsung/a14nnxx/a14:15/AP3A:user/release-keys", "os_version": "15"})
        self.assertEqual(signature, "MzaziCare (samsung SM-A145F; Android 15)")
        self.assertIsNone(login_history.app_signature({}))


class AsnLookupTests(unittest.TestCase):
    def test_ranges_for_ipv4_and_ipv6_and_unrouted_space(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "asn.tsv.gz")
            with gzip.open(path, "wt") as handle:
                handle.write("41.90.0.0\t41.90.255.255\t33771\tKE\tSAFARICOM-LIMITED\n")
                handle.write("41.91.0.0\t41.91.255.255\t0\tNone\tNot routed\n")
                handle.write("2c0f:fe38::\t2c0f:fe38:ffff:ffff:ffff:ffff:ffff:ffff\t33771\tKE\tSAFARICOM-LIMITED\n")
            lookup = AsnLookup(path)
            self.assertEqual(lookup.lookup("41.90.12.7"), 33771)
            self.assertEqual(lookup.lookup("2c0f:fe38:2::1"), 33771)
            self.assertIsNone(lookup.lookup("41.91.0.1"))       # AS 0 = not routed
            self.assertIsNone(lookup.lookup("8.8.8.8"))         # outside every range
            self.assertIsNone(lookup.lookup("192.168.1.10"))    # private address
            self.assertIsNone(lookup.lookup("not-an-ip"))

    def test_missing_table_disables_network_history(self):
        self.assertIsNone(AsnLookup("/nonexistent/asn.tsv.gz").lookup("41.90.12.7"))


class FeatureTests(unittest.TestCase):
    now = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)

    def summary(self, **values):
        base = {"prior": 12, "last_login": self.now - timedelta(hours=24), "countries_recorded": 12,
                "country_seen": 1, "asns_recorded": 12, "asn_seen": 1, "agents_recorded": 12, "agent_seen": 1,
                "os_recorded": 12, "os_seen": 1, "devices_recorded": 12, "devices_seen": 1, "device_seen": 1}
        base.update(values)
        return base

    def test_first_login_is_new_in_every_known_respect(self):
        features = features_from_summary({"prior": 0}, context(asn=None), self.now)
        self.assertEqual(features["first_login"], 1.0)
        self.assertEqual(features["new_country"], 1.0)
        self.assertEqual(features["new_asn"], 0.0)              # unknown network is never "new"
        self.assertEqual(features["log_prior_logins"], 0.0)

    def test_familiar_login_has_no_novelty(self):
        features = features_from_summary(self.summary(), context(), self.now)
        self.assertEqual(features["first_login"], 0.0)
        self.assertEqual(sum(features[name] for name in ("new_country", "new_asn", "new_user_agent",
                                                          "new_os", "new_device_type")), 0.0)
        self.assertAlmostEqual(features["log_prior_logins"], math.log1p(12))
        self.assertAlmostEqual(features["log_hours_since_last"], math.log1p(24))

    def test_new_country_and_network_are_flagged(self):
        features = features_from_summary(self.summary(country_seen=0, asn_seen=0), context(), self.now)
        self.assertEqual(features["new_country"], 1.0)
        self.assertEqual(features["new_asn"], 1.0)

    def test_unknown_country_and_unrecorded_history_carry_no_information(self):
        self.assertEqual(features_from_summary(self.summary(country_seen=0), context(country=None),
                                               self.now)["new_country"], 0.0)
        # Logins recorded before the context columns existed: counted, but nothing to compare with.
        legacy = self.summary(countries_recorded=0, country_seen=0)
        self.assertEqual(features_from_summary(legacy, context(), self.now)["new_country"], 0.0)


class AsyncSessionAdapter:
    """Runs the module's real queries on a synchronous SQLite session."""

    def __init__(self, session):
        self.session = session

    async def execute(self, statement):
        return self.session.execute(statement)


class DatabaseQueryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        ContextualLogin.__table__.create(self.engine)
        self.session = Session(self.engine)
        self.db = AsyncSessionAdapter(self.session)
        self.user_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        rows = [  # two successful Kenyan logins, one legacy row, one failed foreign attempt
            dict(country="KE", asn=33771, user_agent="MzaziCare (a)", os_family="Android", device_type="mobile",
                 is_successful=True, login_time=now - timedelta(days=3)),
            dict(country="KE", asn=33771, user_agent="MzaziCare (a)", os_family="Android", device_type="mobile",
                 is_successful=True, login_time=now - timedelta(hours=5)),
            dict(is_successful=True, login_time=now - timedelta(days=30)),
            dict(country="RU", asn=12389, user_agent="MzaziCare (b)", os_family="Android", device_type="mobile",
                 is_successful=False, login_time=now - timedelta(hours=1)),
        ]
        for row in rows:
            self.session.add(ContextualLogin(user_id=self.user_id, device_fingerprint="device-1",
                                             ip_address="41.90.1.1", login_risk_score=0.1, **row))
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    async def test_history_counts_only_successful_logins(self):
        familiar = await login_history.lookup_history(self.db, self.user_id, context(user_agent="MzaziCare (a)"))
        self.assertEqual(familiar["first_login"], 0.0)
        self.assertAlmostEqual(familiar["log_prior_logins"], math.log1p(3))
        self.assertEqual(familiar["new_country"], 0.0)
        self.assertAlmostEqual(familiar["log_hours_since_last"], math.log1p(5), places=1)
        # The failed Russian attempt must not make Russia look familiar.
        foreign = await login_history.lookup_history(
            self.db, self.user_id, context(country="RU", asn=12389, user_agent="MzaziCare (b)"))
        self.assertEqual(foreign["new_country"], 1.0)
        self.assertEqual(foreign["new_asn"], 1.0)
        self.assertEqual(foreign["new_user_agent"], 1.0)

    async def test_unknown_user_is_a_first_login(self):
        features = await login_history.lookup_history(self.db, uuid.uuid4(), context())
        self.assertEqual(features["first_login"], 1.0)

    async def test_challenged_login_context_is_recovered(self):
        columns = await login_history.latest_challenge_columns(self.db, self.user_id, "device-1")
        self.assertEqual(columns["country"], "RU")
        self.assertEqual(columns["asn"], 12389)
        self.assertIsNone(await login_history.latest_challenge_columns(self.db, self.user_id, "other-device"))

    async def test_lookup_failure_returns_no_history(self):
        class BrokenDb:
            async def execute(self, statement):
                raise RuntimeError("database unavailable")
        self.assertIsNone(await login_history.lookup_history(BrokenDb(), self.user_id, context()))


def tiny_v2_artifact():
    """A small version 2 model in which a new country drives the risk, saved in the deployed format."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    rng = np.random.default_rng(0)
    n = 2000
    history = ["first_login", "log_prior_logins", "log_hours_since_last", "new_country", "new_asn",
               "new_user_agent", "new_os", "new_device_type"]
    X = pd.DataFrame({"device_type": "mobile", "os_family": rng.choice(["Android", "iOS"], n),
                      "hour_sin": rng.uniform(-1, 1, n), "hour_cos": rng.uniform(-1, 1, n),
                      "rtt_ms": rng.uniform(20, 300, n), "first_login": 0.0,
                      "log_prior_logins": rng.uniform(0, 5, n), "log_hours_since_last": rng.uniform(0, 6, n),
                      "new_country": rng.integers(0, 2, n).astype(float), "new_asn": 0.0,
                      "new_user_agent": 0.0, "new_os": 0.0, "new_device_type": 0.0})
    y = X["new_country"].astype(int)
    numeric = ["hour_sin", "hour_cos", "rtt_ms"] + history
    pipeline = Pipeline([("preprocessor", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["device_type", "os_family"]),
        ("num", StandardScaler(), numeric)])),
        ("classifier", RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0))]).fit(X, y)
    reference = {"device_type": "mobile", "os_family": "Android", "hour_sin": 0.0, "hour_cos": -1.0,
                 "rtt_ms": 45.0, "first_login": 0.0, "log_prior_logins": math.log1p(20),
                 "log_hours_since_last": math.log1p(24), "new_country": 0.0, "new_asn": 0.0,
                 "new_user_agent": 0.0, "new_os": 0.0, "new_device_type": 0.0}
    return {"pipeline": pipeline, "version": 2, "history_features": history, "reference": reference}


class VersionTwoEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AdaptiveRiskEngine.__new__(AdaptiveRiskEngine)
        cls.engine.load_artifact(tiny_v2_artifact())

    def evaluate(self, history=None, **features):
        import contextlib, io
        raw = {"local_hour": 10, "os_name": "Android", "network_type": "WIFI",
               "is_physical_device": True, "is_mock_location": False, **features}
        trusted = SimpleNamespace(trust_state="TRUSTED",
                                  last_login_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1))
        with contextlib.redirect_stdout(io.StringIO()):
            return self.engine.evaluate_risk(raw, ip_address="41.90.1.1", device_profile=trusted, history=history)

    def familiar(self, **changes):
        history = {"first_login": 0.0, "log_prior_logins": math.log1p(12), "log_hours_since_last": math.log1p(24),
                   "new_country": 0.0, "new_asn": 0.0, "new_user_agent": 0.0, "new_os": 0.0, "new_device_type": 0.0}
        history.update(changes)
        return history

    def test_new_country_raises_risk_and_is_explained(self):
        familiar = self.evaluate(self.familiar())
        foreign = self.evaluate(self.familiar(new_country=1.0))
        self.assertGreater(foreign["base_ml_risk"], familiar["base_ml_risk"])
        self.assertIn("Login from a country not seen before for this account", foreign["reasons"])
        self.assertEqual(foreign["model_version"], 2)

    def test_grouped_shap_explains_six_groups_and_sums_to_the_difference(self):
        result = self.evaluate(self.familiar(new_country=1.0))
        groups = {item["feature"] for item in result["shap_attributions"]}
        self.assertEqual(groups, set(V2_GROUPS))
        location = next(item for item in result["shap_attributions"] if item["feature"] == "Location History")
        self.assertGreater(location["impact"], 0)
        # Shapley values add up to the model's probability minus the reference login's probability.
        reference = pd.DataFrame([self.engine.reference])
        baseline = float(self.engine.model.predict_proba(reference)[0][1])
        total = sum(item["impact"] for item in result["shap_attributions"])
        self.assertAlmostEqual(total, result["base_ml_risk"] - baseline, delta=0.011)

    def test_missing_history_uses_the_routine_reference(self):
        result = self.evaluate(None)
        self.assertNotIn("Login from a country not seen before for this account", result["reasons"])
        self.assertEqual(result["history_features"], None)

    def test_new_country_on_trusted_device_is_challenged_for_established_accounts(self):
        result = self.evaluate(self.familiar(new_country=1.0, log_prior_logins=math.log1p(3)))
        self.assertEqual(result["risk_action"], "CHALLENGE")
        self.assertIn("New country for an established account: step-up verification required", result["reasons"])

    def test_new_country_rule_skips_young_accounts_and_familiar_countries(self):
        young = self.evaluate(self.familiar(new_country=1.0, log_prior_logins=math.log1p(2)))
        self.assertNotIn("New country for an established account: step-up verification required", young["reasons"])
        familiar = self.evaluate(self.familiar())
        self.assertEqual(familiar["risk_action"], "ALLOW")

    def test_evening_logins_are_no_longer_capped(self):
        evening = self.evaluate(self.familiar(new_country=1.0), local_hour=20)
        self.assertNotIn("Clinical shift Context: Evening handover window dampening applied", evening["reasons"])


if __name__ == "__main__":
    unittest.main()
