"""Per-user rolling baseline for the version 2 behavioral model."""
import importlib.util
import math
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
from sqlalchemy.dialects import postgresql

from app.models.entities import BehavioralBaseline
from app.services import behavioral_engine as be
from app.services.behavioral_engine import (BASE_FEATURES, RELATIVE_FEATURES, BehavioralAnomalyEngine,
                                            load_user_baseline, relative_features)

ROOT = Path(__file__).resolve().parents[1]
QUIET = {"total_logons": 1, "distinct_devices": 1, "activity_span_hours": 1.0, "off_hours_ratio": 0.0,
         "records_accessed_count": 2, "total_payload_bytes": 2000}


def load_training_v2():
    spec = importlib.util.spec_from_file_location("train_behavioral_model_v2",
                                                  ROOT / "ml" / "train_behavioral_model_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingPipeline:
    """Stands in for the Isolation Forest pipeline and records the frame it receives."""

    def __init__(self, decision=0.2, prediction=1):
        self.decision, self.prediction, self.frames = decision, prediction, []
        self.named_steps = {"isolation_forest": type("Forest", (), {"offset_": 0.0})()}

    def decision_function(self, frame):
        self.frames.append(frame.copy())
        return np.array([self.decision])

    def predict(self, frame):
        return np.array([self.prediction])


class Result:
    def __init__(self, row=None, scalar=None):
        self.row, self.scalar = row, scalar

    def one(self):
        return self.row

    def scalar_one_or_none(self):
        return self.scalar


class FakeDb:
    def __init__(self, rows):
        self.rows, self.statements, self.added = list(rows), [], []

    async def execute(self, statement):
        self.statements.append(statement)
        return self.rows.pop(0)

    def add(self, item):
        self.added.append(item)


def v2_engine(pipeline):
    engine = BehavioralAnomalyEngine.__new__(BehavioralAnomalyEngine)
    engine.configure({"pipeline": pipeline, "version": 2, "features": BASE_FEATURES + RELATIVE_FEATURES,
                      "window_days": 30, "min_history": 5})
    return engine


def baseline(samples=20, **overrides):
    return {"means": {**{k: float(v) for k, v in QUIET.items()}, **overrides}, "samples": samples}


class RelativeFeatureTests(unittest.TestCase):
    def test_no_baseline_or_short_history_gives_zero_deviation(self):
        for value in (None, {"means": {}, "samples": 0}, baseline(samples=4)):
            features = relative_features({**QUIET, "records_accessed_count": 30}, value)
            self.assertEqual(set(features), set(RELATIVE_FEATURES))
            self.assertTrue(all(v == 0.0 for v in features.values()))

    def test_relative_feature_is_log_ratio_to_user_mean(self):
        features = relative_features({**QUIET, "records_accessed_count": 20}, baseline())
        self.assertAlmostEqual(features["relative_records_accessed_count"], math.log1p(20) - math.log1p(2))
        self.assertAlmostEqual(features["relative_total_logons"], 0.0)

    def test_runtime_formula_matches_training_features(self):
        training = load_training_v2()
        days = pd.date_range("2010-01-04", periods=40, freq="D").date
        rng = np.random.default_rng(7)
        rows = {name: rng.integers(0, 12, len(days)).astype(float) for name in BASE_FEATURES}
        rows["off_hours_ratio"] = rng.random(len(days))
        matrix = pd.DataFrame(rows, index=pd.MultiIndex.from_arrays([["U1"] * len(days), days]))
        trained = training.add_relative_features(matrix)
        self.assertEqual(training.FEATURES, BASE_FEATURES + RELATIVE_FEATURES)
        for position in (0, 4, 5, 12, 39):
            previous = matrix.iloc[max(0, position - 30):position]
            value = baseline(samples=len(previous), **previous.mean().to_dict()) if len(previous) else None
            runtime = relative_features(matrix.iloc[position].to_dict(), value)
            for name in RELATIVE_FEATURES:
                self.assertAlmostEqual(runtime[name], trained.iloc[position][name], places=9,
                                       msg=f"day {position} {name}")


class BaselineQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_baseline_is_one_aggregate_over_the_users_recent_snapshots(self):
        user_id = uuid.uuid4()
        db = FakeDb([Result(row=(12, 1.0, 1.0, 2.0, 0.1, 3.0, 4096.0))])
        value = await load_user_baseline(db, user_id)
        self.assertEqual(value["samples"], 12)
        self.assertEqual(value["means"]["records_accessed_count"], 3.0)
        sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
        self.assertIn("avg(CAST(behavioral_audit_logs.telemetry_snapshot ->>", sql)
        self.assertIn("behavioral_audit_logs.timestamp >=", sql)
        self.assertIn("session_revocation_recommended IS false", sql)

    async def test_user_without_snapshots_has_empty_baseline(self):
        db = FakeDb([Result(row=(0, None, None, None, None, None, None))])
        self.assertEqual(await load_user_baseline(db, uuid.uuid4()), {"means": {}, "samples": 0})


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_version_two_scores_raw_and_relative_features(self):
        pipeline = RecordingPipeline()
        engine = v2_engine(pipeline)
        result = await engine.evaluate_session_activity(
            "user", {**QUIET, "records_accessed_count": 20}, None, baseline=baseline())
        frame = pipeline.frames[0]
        self.assertEqual(list(frame.columns), BASE_FEATURES + RELATIVE_FEATURES)
        self.assertAlmostEqual(frame["relative_records_accessed_count"].iloc[0], math.log1p(20) - math.log1p(2))
        self.assertEqual(result["model_version"], 2)
        self.assertEqual(result["baseline_samples"], 20)
        self.assertIn("Patient records accessed are well above this user's 30-day norm", result["reasons"])

    async def test_same_activity_is_not_unusual_for_a_user_who_always_does_it(self):
        engine = v2_engine(RecordingPipeline())
        result = await engine.evaluate_session_activity(
            "user", {**QUIET, "records_accessed_count": 20}, None,
            baseline=baseline(records_accessed_count=18.0))
        self.assertNotIn("Patient records accessed are well above this user's 30-day norm", result["reasons"])

    async def test_baseline_is_loaded_and_stored_when_not_supplied(self):
        user_id = uuid.uuid4()
        db = FakeDb([Result(row=(9, 1.0, 1.0, 1.0, 0.0, 2.0, 2000.0)), Result(scalar=None)])
        result = await v2_engine(RecordingPipeline()).evaluate_session_activity(user_id, QUIET, db)
        self.assertEqual(result["baseline_samples"], 9)
        stored = next(item for item in db.added if isinstance(item, BehavioralBaseline))
        self.assertEqual(stored.user_id, user_id)
        self.assertEqual(stored.sample_count, 9)
        self.assertEqual(stored.avg_access_frequency, 2.0)
        self.assertEqual(stored.feature_means["total_payload_bytes"], 2000.0)

    async def test_existing_baseline_row_is_updated(self):
        row = BehavioralBaseline(user_id=uuid.uuid4(), avg_access_frequency=0.0,
                                 out_of_hours_ratio=0.0, anomaly_threshold=0.5)
        db = FakeDb([Result(row=(15, 1.0, 1.0, 1.0, 0.2, 4.0, 2000.0)), Result(scalar=row)])
        await v2_engine(RecordingPipeline()).evaluate_session_activity(row.user_id, QUIET, db)
        self.assertEqual(db.added, [])
        self.assertEqual((row.sample_count, row.avg_access_frequency, row.out_of_hours_ratio), (15, 4.0, 0.2))

    async def test_baseline_lookup_failure_still_scores_without_deviation(self):
        class BrokenDb:
            async def execute(self, statement):
                raise RuntimeError("database unavailable")

        pipeline = RecordingPipeline()
        result = await v2_engine(pipeline).evaluate_session_activity("user", QUIET, BrokenDb())
        self.assertEqual(result["risk_action"], "ALLOW")
        self.assertTrue((pipeline.frames[0][RELATIVE_FEATURES] == 0).all(axis=None))

    async def test_baseline_work_runs_in_savepoints_that_absorb_failures(self):
        class SavepointDb(FakeDb):
            def __init__(self, rows):
                super().__init__(rows)
                self.savepoints, self.flushes = 0, 0

            def begin_nested(self):
                outer = self

                class Savepoint:
                    async def __aenter__(self):
                        outer.savepoints += 1

                    async def __aexit__(self, *exc):
                        return False
                return Savepoint()

            async def flush(self):
                self.flushes += 1
                if self.flushes == 2:  # e.g. the baseline columns are not migrated yet
                    raise RuntimeError("column feature_means does not exist")

        db = SavepointDb([Result(row=(9, 1.0, 1.0, 1.0, 0.0, 2.0, 2000.0)), Result(scalar=None)])
        result = await v2_engine(RecordingPipeline()).evaluate_session_activity(uuid.uuid4(), QUIET, db)
        self.assertEqual(db.savepoints, 2)
        self.assertEqual(result["risk_action"], "ALLOW")
        self.assertEqual(result["baseline_samples"], 9)

    async def test_exfiltration_rule_still_denies_and_revokes(self):
        engine = v2_engine(RecordingPipeline())
        with patch.object(be, "blacklist_token", new_callable=AsyncMock) as revoke:
            from datetime import datetime, timedelta, timezone
            expires = datetime.now(timezone.utc) + timedelta(hours=1)
            result = await engine.evaluate_session_activity(
                "user", {**QUIET, "records_accessed_count": 40}, None, baseline=baseline(),
                current_jti="jti", expires_at=expires)
        self.assertEqual(result["risk_action"], "DENY")
        revoke.assert_awaited_once()

    async def test_version_one_pipeline_still_loads(self):
        pipeline = RecordingPipeline()
        engine = BehavioralAnomalyEngine.__new__(BehavioralAnomalyEngine)
        engine.configure(pipeline)
        result = await engine.evaluate_session_activity("user", QUIET, None)
        self.assertEqual(list(pipeline.frames[0].columns), BASE_FEATURES)
        self.assertEqual(result["model_version"], 1)
        self.assertFalse(engine.uses_baseline)

    async def test_deployed_artifact_finds_a_spike_more_unusual_than_the_same_routine_activity(self):
        engine = BehavioralAnomalyEngine()
        if not engine.uses_baseline:
            self.skipTest("Deployed behavioral artifact has no per-user baseline (version 1)")

        class Recorder:
            """Pass-through to the real pipeline that keeps its decision values."""
            def __init__(self, pipeline):
                self.pipeline, self.named_steps, self.decisions = pipeline, pipeline.named_steps, []

            def decision_function(self, frame):
                values = self.pipeline.decision_function(frame)
                self.decisions.append(float(values[0]))
                return values

            def predict(self, frame):
                return self.pipeline.predict(frame)

        engine.model = Recorder(engine.model)
        spike = {**QUIET, "records_accessed_count": 30, "total_payload_bytes": 60000}
        await engine.evaluate_session_activity("user", spike, None, baseline=baseline())
        await engine.evaluate_session_activity(
            "user", spike, None, baseline=baseline(records_accessed_count=30.0, total_payload_bytes=60000.0))
        unusual, routine = engine.model.decisions  # Lower decision values are more anomalous.
        self.assertLess(unusual, routine)

if __name__ == "__main__":
    unittest.main()
