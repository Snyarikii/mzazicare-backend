"""Data exports (shares and screenshots) as a behavioral signal, model version 3."""
import math
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

import main
from app.api.v1.endpoints import telemetry
from app.core.dependencies import get_current_access_claims, get_current_active_user
from app.core.rate_limit import limiter
from app.db.session import get_db
from app.models.entities import BehavioralAuditLog, SessionActivityLog
from app.schemas.telemetry import SessionActivityTelemetry
from app.services.behavioral_engine import (BASELINE_COUNTERS, BehavioralAnomalyEngine,
                                            first_data_export, relative_features)

QUIET = {"total_logons": 1, "distinct_devices": 1, "activity_span_hours": 1.0, "off_hours_ratio": 0.0,
         "records_accessed_count": 2, "total_payload_bytes": 2000, "data_exports": 0}
V3_FEATURES = BASELINE_COUNTERS + [f"relative_{n}" for n in BASELINE_COUNTERS] + ["first_data_export"]


class Pipeline:
    """Records the frame it scores and always reports a comfortably normal session."""

    def __init__(self):
        self.frames = []
        self.named_steps = {"isolation_forest": type("Forest", (), {"offset_": 0.0})()}

    def decision_function(self, frame):
        self.frames.append(frame.copy())
        return [0.3]

    def predict(self, frame):
        return [1]


def v3_engine(pipeline=None):
    engine = BehavioralAnomalyEngine.__new__(BehavioralAnomalyEngine)
    engine.configure({"pipeline": pipeline or Pipeline(), "version": 3, "features": V3_FEATURES,
                      "base_features": BASELINE_COUNTERS, "flag_features": ["first_data_export"]})
    return engine


def baseline(samples=20, **overrides):
    return {"means": {**{k: float(v) for k, v in QUIET.items()}, **overrides}, "samples": samples}


class Result:
    def __init__(self, row):
        self.row = row

    def one(self):
        return self.row


class FakeDb:
    def __init__(self, rows=()):
        self.rows, self.statements, self.added, self.commits = list(rows), [], [], 0

    async def execute(self, statement):
        self.statements.append(statement)
        return self.rows.pop(0)

    def add(self, item):
        self.added.append(item)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass


class ExportFeatureTests(unittest.TestCase):
    def test_first_export_needs_history_and_no_earlier_exports(self):
        one = {**QUIET, "data_exports": 1}
        self.assertEqual(first_data_export(one, 0, baseline()), 1.0)
        self.assertEqual(first_data_export(one, 3, baseline()), 0.0)
        self.assertEqual(first_data_export(one, 0, baseline(samples=4)), 0.0)
        self.assertEqual(first_data_export(QUIET, 0, baseline()), 0.0)

    def test_export_relative_feature_uses_the_user_norm(self):
        value = relative_features({**QUIET, "data_exports": 6}, baseline(data_exports=0.5), 5, ["data_exports"])
        self.assertAlmostEqual(value["relative_data_exports"], math.log1p(6) - math.log1p(0.5))


class EngineExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_version_three_scores_fifteen_features(self):
        pipeline = Pipeline()
        await v3_engine(pipeline).evaluate_session_activity(
            "user", {**QUIET, "data_exports": 2, "prior_data_exports": 0}, None, baseline=baseline())
        frame = pipeline.frames[0]
        self.assertEqual(list(frame.columns), V3_FEATURES)
        self.assertEqual(frame["first_data_export"].iloc[0], 1.0)

    async def test_first_export_requires_step_up(self):
        result = await v3_engine().evaluate_session_activity(
            "user", {**QUIET, "data_exports": 1, "prior_data_exports": 0}, None, baseline=baseline())
        self.assertEqual(result["risk_action"], "CHALLENGE")
        self.assertIn("First time this user has shared or captured patient data", result["reasons"])
        self.assertIn("Patient data left the app 1 time this session (share or screenshot)", result["reasons"])

    async def test_occasional_export_by_a_user_who_exports_is_allowed(self):
        result = await v3_engine().evaluate_session_activity(
            "user", {**QUIET, "data_exports": 1, "prior_data_exports": 12}, None,
            baseline=baseline(data_exports=1.0))
        self.assertEqual(result["risk_action"], "ALLOW")

    async def test_export_burst_against_the_user_norm_requires_step_up(self):
        result = await v3_engine().evaluate_session_activity(
            "user", {**QUIET, "data_exports": 6, "prior_data_exports": 4}, None,
            baseline=baseline(data_exports=0.2))
        self.assertEqual(result["risk_action"], "CHALLENGE")
        self.assertIn("Patient data is being shared or captured more than this user normally does",
                      result["reasons"])

    async def test_burst_that_matches_the_user_norm_is_allowed(self):
        result = await v3_engine().evaluate_session_activity(
            "user", {**QUIET, "data_exports": 6, "prior_data_exports": 300}, None,
            baseline=baseline(data_exports=5.0))
        self.assertEqual(result["risk_action"], "ALLOW")

    async def test_no_rule_without_a_baseline(self):
        result = await v3_engine().evaluate_session_activity(
            "user", {**QUIET, "data_exports": 6, "prior_data_exports": 0}, None, baseline=baseline(samples=2))
        self.assertEqual(result["risk_action"], "ALLOW")


class ExportCountTests(unittest.IsolatedAsyncioTestCase):
    async def test_counts_come_from_one_query_scoped_to_the_session_since_its_reset(self):
        db = FakeDb([Result((3, 10))])
        current, earlier = await telemetry.session_export_counts(db, uuid.uuid4(), "session-1234")
        self.assertEqual((current, earlier), (3, 7))
        sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
        self.assertIn("count(*) FILTER (WHERE session_activity_logs.session_token =", sql)
        self.assertIn("max(behavioral_audit_logs.timestamp)", sql)
        self.assertIn("session_activity_logs.event_type =", sql)

    async def test_without_a_session_every_export_is_earlier(self):
        db = FakeDb([Result((0, 4))])
        self.assertEqual(await telemetry.session_export_counts(db, uuid.uuid4(), None), (0, 4))

    async def test_behavioral_check_scores_server_counts_and_stores_them(self):
        user = SimpleNamespace(user_id=uuid.uuid4())
        db = FakeDb()
        payload = SessionActivityTelemetry(session_id="session-1234", records_accessed_count=2)
        evaluation = {"anomaly_score": 0.6, "risk_action": "CHALLENGE", "step_up_required": True,
                      "access_denied": False, "recommend_revocation": False, "reasons": ["x"],
                      "is_anomaly": True, "session_revocation_recommended": False, "risk_factors": ["x"]}
        expires = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())
        with patch.object(telemetry, "session_export_counts", AsyncMock(return_value=(2, 0))), \
             patch.object(telemetry.behavioral_engine, "evaluate_session_activity",
                          AsyncMock(return_value=evaluation)) as evaluate:
            await telemetry.behavioral_check(payload, db, user, {"jti": "j", "exp": expires})
        metrics = evaluate.await_args.kwargs["metrics"]
        self.assertEqual((metrics["data_exports"], metrics["prior_data_exports"]), (2, 0))
        audit = next(item for item in db.added if isinstance(item, BehavioralAuditLog))
        self.assertEqual(audit.telemetry_snapshot["data_exports"], 2)

    async def test_count_failure_does_not_block_the_check(self):
        user = SimpleNamespace(user_id=uuid.uuid4())
        db = FakeDb()
        expires = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())
        engine = v3_engine()
        with patch.object(telemetry, "session_export_counts", AsyncMock(side_effect=RuntimeError("down"))), \
             patch.object(telemetry, "behavioral_engine", engine):
            response = await telemetry.behavioral_check(
                SessionActivityTelemetry(session_id="session-1234"), db, user, {"jti": "j", "exp": expires})
        self.assertEqual(response.risk_action, "ALLOW")


class DataExportEndpointTests(unittest.TestCase):
    def setUp(self):
        self.previously_enabled = limiter.enabled
        limiter.enabled = True
        limiter.reset()
        self.db = FakeDb()
        self.user = SimpleNamespace(user_id=uuid.uuid4())
        main.app.dependency_overrides[get_db] = lambda: self.db
        main.app.dependency_overrides[get_current_active_user] = lambda: self.user
        main.app.dependency_overrides[get_current_access_claims] = lambda: {"jti": "j"}
        self.client = TestClient(main.app)

    def tearDown(self):
        limiter.reset()
        limiter.enabled = self.previously_enabled
        main.app.dependency_overrides.clear()

    def test_records_a_share_event(self):
        response = self.client.post("/api/v1/telemetry/data-export",
                                    json={"session_id": "session-1234", "kind": "SHARE", "screen": "CareLogs"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"status": "RECORDED", "kind": "SHARE"})
        event = self.db.added[0]
        self.assertIsInstance(event, SessionActivityLog)
        self.assertEqual((event.user_id, event.session_token, event.event_type, event.resource_accessed),
                         (self.user.user_id, "session-1234", "DATA_EXPORT", "SHARE:CareLogs"))
        self.assertEqual(self.db.commits, 1)

    def test_rejects_unknown_kinds_and_unsafe_screen_names(self):
        for body in ({"session_id": "session-1234", "kind": "PRINT", "screen": "CareLogs"},
                     {"session_id": "session-1234", "kind": "SHARE", "screen": "<script>"},
                     {"session_id": "short", "kind": "SHARE", "screen": "CareLogs"}):
            self.assertEqual(self.client.post("/api/v1/telemetry/data-export", json=body).status_code, 422)

    def test_allows_thirty_per_minute(self):
        body = {"session_id": "session-1234", "kind": "SCREENSHOT", "screen": "App"}
        codes = [self.client.post("/api/v1/telemetry/data-export", json=body).status_code for _ in range(31)]
        self.assertEqual(codes, [201] * 30 + [429])


if __name__ == "__main__":
    unittest.main()
