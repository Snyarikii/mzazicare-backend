"""Continuous session behavioral anomaly inference.

Model version 2 compares each telemetry window with the user's own rolling baseline: the mean of
their earlier behavioral snapshots over the last 30 days. The six raw counters are kept and six
relative features are added, relative_<name> = log1p(value) - log1p(baseline mean). Without at
least five earlier snapshots the relative features are 0, exactly as in training
(ml/train_behavioral_model_v2.py). A version 1 artifact (six population features) still loads.

Model version 3 adds a seventh counter, data_exports: shares from the app and screenshots of patient
screens in the current session, counted on the server. It gets its own relative feature and a
first_data_export flag (the user's first export after at least five baseline snapshots). In
training, CERT USB connections stand in for exports (ml/train_behavioral_model_v3.py).
"""
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import BehavioralAuditLog, BehavioralBaseline
from app.services.token_blacklist import blacklist_token

BASE_FEATURES = ["total_logons", "distinct_devices", "activity_span_hours", "off_hours_ratio",
                 "records_accessed_count", "total_payload_bytes"]
FEATURES = BASE_FEATURES  # Version 1 feature list, kept for existing imports.
RELATIVE_FEATURES = [f"relative_{name}" for name in BASE_FEATURES]
EXPORT_FEATURE = "data_exports"
# Counters averaged into every baseline. Snapshots from before exports were counted have none,
# which is the same as zero exports.
BASELINE_COUNTERS = BASE_FEATURES + [EXPORT_FEATURE]
MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "behavioral_anomaly_model.joblib"
DEFAULT_WINDOW_DAYS = 30
DEFAULT_MIN_HISTORY = 5
# A relative feature at or above log(3) means the value is at least about three times the user's norm.
RELATIVE_REASON_THRESHOLD = float(np.log(3.0))
RELATIVE_REASONS = {
    "records_accessed_count": ("Patient records accessed are well above this user's 30-day norm", 5),
    "total_payload_bytes": ("Data volume is well above this user's 30-day norm", 1024 * 1024),
    "off_hours_ratio": ("Off-hours activity is unusual for this user", 0.25),
    "distinct_devices": ("More devices than this user normally uses", 2),
    "activity_span_hours": ("Activity span is much longer than this user's norm", 8),
    "total_logons": ("Sign-in count is well above this user's 30-day norm", 5),
    EXPORT_FEATURE: ("Patient data is being shared or captured more than this user normally does", 2),
}
# Export policy rule: a first-ever export after an established history, or a burst of at least
# EXPORT_SPIKE_MIN exports that is at least three times the user's norm, always needs step-up
# verification. On CERT r4.2 (USB as the stand-in) each condition fires on about 0.01% of
# user-days, a quarter of them malicious, against a 0.3% base rate.
EXPORT_RULE_FLOOR = 0.50
EXPORT_SPIKE_MIN = 5
EMPTY_BASELINE = {"means": {}, "samples": 0}
logger = logging.getLogger(__name__)


def relative_features(vector: dict, baseline: dict | None, min_history: int = DEFAULT_MIN_HISTORY,
                      features: list[str] | None = None) -> dict:
    """log1p(value) - log1p(baseline mean); 0 for every feature without a usable baseline."""
    features = features or BASE_FEATURES
    baseline = baseline or EMPTY_BASELINE
    means = dict(baseline.get("means") or {})
    if means and means.get(EXPORT_FEATURE) is None:
        means[EXPORT_FEATURE] = 0.0  # A baseline from before exports were counted means none.
    usable = int(baseline.get("samples") or 0) >= min_history and all(
        means.get(name) is not None for name in features)
    if not usable:
        return {f"relative_{name}": 0.0 for name in features}
    return {f"relative_{name}": float(np.log1p(max(vector[name], 0.0)) - np.log1p(max(float(means[name]), 0.0)))
            for name in features}


def first_data_export(vector: dict, prior_exports: float | None, baseline: dict | None,
                      min_history: int = DEFAULT_MIN_HISTORY) -> float:
    """1.0 when this is the user's first export after an established, export-free history."""
    samples = int((baseline or EMPTY_BASELINE).get("samples") or 0)
    return float(vector.get(EXPORT_FEATURE, 0.0) > 0 and float(prior_exports or 0) == 0 and samples >= min_history)


async def load_user_baseline(db: AsyncSession, user_id: Any, *, window_days: int = DEFAULT_WINDOW_DAYS,
                             now: datetime | None = None) -> dict:
    """Mean of the user's earlier telemetry snapshots in the window, from one aggregate query.

    Snapshots that recommended session revocation are left out so an attacker's activity does
    not become the account's new normal. Reset events carry no counters and are not counted.
    """
    since = (now or datetime.now(timezone.utc)) - timedelta(days=window_days)
    snapshot = BehavioralAuditLog.telemetry_snapshot
    columns = [func.avg(snapshot[name].as_float()).label(name) for name in BASE_FEATURES]
    columns.append(func.avg(func.coalesce(snapshot[EXPORT_FEATURE].as_float(), 0.0)).label(EXPORT_FEATURE))
    statement = select(func.count().label("samples"), *columns).where(
        BehavioralAuditLog.user_id == user_id,
        BehavioralAuditLog.timestamp >= since,
        BehavioralAuditLog.session_revocation_recommended.is_(False),
        snapshot["total_logons"].as_float().isnot(None),
    )
    row = (await db.execute(statement)).one()
    samples = int(row[0] or 0)
    if not samples:
        return dict(EMPTY_BASELINE)
    means = {name: (None if value is None else float(value)) for name, value in zip(BASELINE_COUNTERS, row[1:])}
    return {"means": means, "samples": samples}


async def store_user_baseline(db: AsyncSession, user_id: Any, baseline: dict, threshold: float) -> None:
    """Keep behavioral_baselines in step with the baseline used for the latest evaluation."""
    if not baseline.get("samples"):
        return
    means = baseline["means"]
    row = (await db.execute(select(BehavioralBaseline).where(BehavioralBaseline.user_id == user_id))).scalar_one_or_none()
    if row is None:
        row = BehavioralBaseline(user_id=user_id)
        db.add(row)
    row.avg_access_frequency = float(means.get("records_accessed_count") or 0.0)
    row.out_of_hours_ratio = float(means.get("off_hours_ratio") or 0.0)
    row.anomaly_threshold = float(threshold)
    row.feature_means = {name: means.get(name) for name in BASELINE_COUNTERS}
    row.sample_count = int(baseline["samples"])
    row.last_updated = datetime.now(timezone.utc)


async def _in_savepoint(db: AsyncSession, operation):
    """Run a baseline read or write in a SAVEPOINT so a failure there (for example a database
    that has not been migrated yet) cannot abort the request's audit insert."""
    begin_nested = getattr(db, "begin_nested", None)
    if begin_nested is None:
        return await operation()
    async with begin_nested():
        result = await operation()
        flush = getattr(db, "flush", None)
        if flush is not None:
            await flush()
        return result


class BehavioralAnomalyEngine:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.configure(None)
        if model_path.is_file():
            try:
                self.configure(joblib.load(model_path))
            except Exception as exc:
                print(f"[Behavioral Engine] Failed to load model {model_path}: {exc}; using heuristic fallback.")
        else:
            print(f"[Behavioral Engine] Model artifact absent at {model_path}; using heuristic fallback.")

    def configure(self, artifact: Any) -> None:
        """Load a version 1 pipeline or bundle, or a version 2 or 3 bundle."""
        bundle = artifact if isinstance(artifact, dict) else {"pipeline": artifact}
        self.model = bundle.get("pipeline")
        self.version = int(bundle.get("version", 1)) if self.model is not None else 1
        self.window_days = int(bundle.get("window_days", DEFAULT_WINDOW_DAYS))
        self.min_history = int(bundle.get("min_history", DEFAULT_MIN_HISTORY))
        default_base = BASELINE_COUNTERS if self.version >= 3 else BASE_FEATURES
        self.base_features = list(bundle.get("base_features") or default_base)
        self.flag_features = list(bundle.get("flag_features") or (
            ["first_data_export"] if self.version >= 3 else []))
        default_features = (self.base_features + [f"relative_{n}" for n in self.base_features]
                            + self.flag_features) if self.version >= 2 else BASE_FEATURES
        self.features = list(bundle.get("features") or default_features)

    @property
    def uses_baseline(self) -> bool:
        return self.model is not None and self.version >= 2

    async def evaluate_session_activity(self, user_id: Any, metrics: dict,
                                        db: AsyncSession | None, *, current_jti: str | None = None,
                                        expires_at: datetime | None = None,
                                        baseline: dict | None = None) -> dict:
        vector = {name: float(metrics.get(name, 0) or 0) for name in BASELINE_COUNTERS}
        if self.uses_baseline and baseline is None and db is not None:
            try:
                baseline = await _in_savepoint(
                    db, lambda: load_user_baseline(db, user_id, window_days=self.window_days))
            except Exception as exc:  # Never block scoring on the baseline lookup.
                logger.warning("Behavioral baseline lookup failed for %s: %s", user_id, exc)
                baseline = None
        relative = (relative_features(vector, baseline, self.min_history, self.base_features)
                    if self.uses_baseline else {})
        has_baseline = bool(baseline) and int(baseline.get("samples") or 0) >= self.min_history
        first_export = first_data_export(vector, metrics.get("prior_data_exports"), baseline, self.min_history)
        flags = {"first_data_export": first_export} if "first_data_export" in self.flag_features else {}
        export_relative = relative.get(f"relative_{EXPORT_FEATURE}")
        if export_relative is None:
            export_relative = relative_features(vector, baseline, self.min_history, [EXPORT_FEATURE])[
                f"relative_{EXPORT_FEATURE}"]
        export_spike = (vector[EXPORT_FEATURE] >= EXPORT_SPIKE_MIN and has_baseline
                        and export_relative >= RELATIVE_REASON_THRESHOLD)

        is_critical_exfiltration = (
            vector["records_accessed_count"] >= 35
            or vector["total_payload_bytes"] >= 10 * 1024 * 1024
        )
        frame = pd.DataFrame([{**vector, **relative, **flags}])
        if self.model is None:
            raw_score = self._fallback_score(vector)
        else:
            try:
                frame = frame[self.features]
                decision = float(self.model.decision_function(frame)[0])
                prediction = int(self.model.predict(frame)[0])
                threshold = float(getattr(self.model.named_steps.get("isolation_forest"), "offset_", 0.0)) \
                    if hasattr(self.model, "named_steps") else 0.0
                # Calibrate the decision boundary to 0.65 and smoothly map distance to [0, 1].
                logit_065 = np.log(0.65 / 0.35)
                raw_score = float(1.0 / (1.0 + np.exp(-np.clip(logit_065 + (threshold - decision) * 12.0, -30, 30))))
                if prediction == -1:
                    raw_score = max(raw_score, 0.65)
            except Exception as exc:
                print(f"[Behavioral Engine] Inference failed: {exc}; using heuristic fallback.")
                raw_score = self._fallback_score(vector)

        if raw_score <= 0.40:
            calibrated_score = raw_score
        elif raw_score < 0.70:
            calibrated_score = 0.50 + (raw_score - 0.40) * 0.80
        else:
            calibrated_score = 0.75 + (raw_score - 0.70) * 1.00
        if first_export or export_spike:
            calibrated_score = max(calibrated_score, EXPORT_RULE_FLOOR)
        if is_critical_exfiltration:
            calibrated_score = max(calibrated_score, 0.85)
        final_score = round(min(max(calibrated_score, 0.0), 1.0), 2)

        if final_score >= 0.75:
            risk_action = "DENY"
        elif final_score >= 0.50:
            risk_action = "CHALLENGE"
        else:
            risk_action = "ALLOW"

        factors: list[str] = []
        if vector["records_accessed_count"] >= 35:
            factors.append("Critical exfiltration: 35 or more patient records accessed")
        if vector["records_accessed_count"] > 100:
            factors.append("Abnormal data query volume (>100 records)")
        if vector["off_hours_ratio"] >= 0.5:
            factors.append("Off-hours EHR query velocity")
        if vector["distinct_devices"] >= 4:
            factors.append("Multi-device hopping detected")
        if vector["total_payload_bytes"] >= 10 * 1024 * 1024:
            factors.append("Critical exfiltration: Data payload reached 10 MB")
        if vector["activity_span_hours"] >= 16:
            factors.append("Unusually long activity span")
        exports = int(vector[EXPORT_FEATURE])
        if exports:
            factors.append(f"Patient data left the app {exports} time{'s' if exports != 1 else ''} "
                           "this session (share or screenshot)")
        if first_export:
            factors.append("First time this user has shared or captured patient data")
        if has_baseline:
            for name, (reason, minimum) in RELATIVE_REASONS.items():
                value = export_relative if name == EXPORT_FEATURE else relative.get(f"relative_{name}", 0.0)
                if value >= RELATIVE_REASON_THRESHOLD and vector[name] >= minimum:
                    factors.append(reason)
        if risk_action != "ALLOW" and not factors:
            factors.append("Session activity deviates from the learned behavioral baseline")

        if self.uses_baseline and db is not None and baseline:
            try:
                await _in_savepoint(db, lambda: store_user_baseline(db, user_id, baseline, 0.50))
            except Exception as exc:
                logger.warning("Behavioral baseline could not be stored for %s: %s", user_id, exc)

        step_up_required = risk_action == "CHALLENGE"
        access_denied = risk_action == "DENY"
        if access_denied and current_jti and expires_at is not None:
            await blacklist_token(current_jti, str(user_id), expires_at,
                                  revoke_refresh=True)
        return {
            "anomaly_score": final_score,
            "risk_action": risk_action,
            "step_up_required": step_up_required,
            "access_denied": access_denied,
            "recommend_revocation": access_denied,
            "reasons": factors,
            # Retain the fields consumed by existing telemetry audit and clients.
            "is_anomaly": risk_action != "ALLOW",
            "session_revocation_recommended": access_denied,
            "risk_factors": factors,
            "model_version": self.version if self.model is not None else 0,
            "baseline_samples": int(baseline.get("samples") or 0) if baseline else 0,
        }

    @staticmethod
    def _fallback_score(values: dict[str, float]) -> float:
        return float(np.clip(max(
            values["off_hours_ratio"],
            min(values["records_accessed_count"] / 200.0, 1.0),
            min(values["distinct_devices"] / 6.0, 1.0),
            min(values["total_payload_bytes"] / 20_000_000.0, 1.0),
            min(values["activity_span_hours"] / 24.0, 1.0),
            min(values["total_logons"] / 100.0, 1.0),
        ), 0.0, 1.0))


behavioral_engine = BehavioralAnomalyEngine()
