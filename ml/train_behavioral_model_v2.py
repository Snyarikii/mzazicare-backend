"""Train the behavioural Isolation Forest with a per-user rolling baseline (model version 2).

Version 1 scored each user-day against the whole population, so a user who is always busy looked
risky and a quiet user's sudden spike looked normal. Version 2 keeps the six population features
and adds six user-relative features: how far today's value sits above or below that user's own
average over their previous 30 active days.

    relative_<feature> = log1p(today) - log1p(mean of the user's previous 30 days)

The current day is excluded from its own baseline, so every feature uses only past data and the
same values can be computed at runtime from earlier telemetry. Users with fewer than five earlier
days have no baseline yet and their relative features are 0 (no deviation).

    python ml/train_behavioral_model_v2.py

Writes app/ml/behavioral_anomaly_model.joblib and archives an existing version 1 artifact to
ml/artifacts/behavioral_anomaly_model_v1.joblib.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = ROOT / "ml" / "train_behavioral_model.py"
OUTPUT_PATH = ROOT / "app" / "ml" / "behavioral_anomaly_model.joblib"
ARCHIVE_PATH = ROOT / "ml" / "artifacts" / "behavioral_anomaly_model_v1.joblib"

BASE_FEATURES = ["total_logons", "distinct_devices", "activity_span_hours", "off_hours_ratio",
                 "records_accessed_count", "total_payload_bytes"]
RELATIVE_FEATURES = [f"relative_{name}" for name in BASE_FEATURES]
FEATURES = BASE_FEATURES + RELATIVE_FEATURES
WINDOW_DAYS = 30          # baseline window: the user's previous 30 active days
MIN_HISTORY = 5           # days needed before a baseline is used
N_ESTIMATORS, CONTAMINATION, RANDOM_STATE = 150, 0.03, 42


def load_v1_module():
    spec = importlib.util.spec_from_file_location("train_behavioral_model", V1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_relative_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Add baseline means and relative features to a (user, date)-indexed user-day matrix.

    Returns a frame in the input's index order with BASE_FEATURES, RELATIVE_FEATURES,
    baseline_<feature> means (NaN when there is no baseline) and baseline_samples.
    """
    frame = matrix[BASE_FEATURES].astype(float).copy()
    users = frame.index.get_level_values(0).astype(str)
    dates = pd.to_datetime(pd.Index(frame.index.get_level_values(1)))
    work = frame.reset_index(drop=True)
    work["_user"], work["_date"], work["_row"] = users, dates, np.arange(len(work))
    work = work.sort_values(["_user", "_date"], kind="mergesort")
    grouped = work.groupby("_user", sort=False)
    work["baseline_samples"] = grouped.cumcount().clip(upper=WINDOW_DAYS)
    for name in BASE_FEATURES:
        previous = grouped[name].shift(1)
        mean = previous.groupby(work["_user"], sort=False).rolling(
            WINDOW_DAYS, min_periods=MIN_HISTORY).mean().reset_index(level=0, drop=True)
        work[f"baseline_{name}"] = mean
        work[f"relative_{name}"] = (np.log1p(work[name]) - np.log1p(mean)).fillna(0.0)
    work = work.sort_values("_row")
    out = work.drop(columns=["_user", "_date", "_row"])
    out.index = frame.index
    return out


def build_training_frame() -> pd.DataFrame:
    v1 = load_v1_module()
    print("Building user-day features from", v1.LOGON_PATH.parent)
    return add_relative_features(v1.build_user_day_matrix())


def make_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("isolation_forest", IsolationForest(
        n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1))])


def archive_v1_artifact() -> None:
    if not OUTPUT_PATH.is_file() or ARCHIVE_PATH.is_file():
        return
    try:
        current = joblib.load(OUTPUT_PATH)
    except Exception:
        return
    if not (isinstance(current, dict) and current.get("version") == 2):
        ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_PATH, ARCHIVE_PATH)
        print(f"Archived version 1 artifact: {ARCHIVE_PATH}")


def run_training() -> None:
    frame = build_training_frame()
    with_baseline = int((frame["baseline_samples"] >= MIN_HISTORY).sum())
    print(f"Total user-day profiles: {len(frame):,}  with a usable baseline: {with_baseline:,} "
          f"({with_baseline / len(frame):.1%})")
    pipeline = make_pipeline().fit(frame[FEATURES])
    flagged = int((pipeline.predict(frame[FEATURES]) == -1).sum())
    print(f"Flagged anomalies: {flagged:,} ({flagged / len(frame):.2%})")
    print("Relative feature quantiles (1%, 50%, 99%):")
    print(frame[RELATIVE_FEATURES].quantile([0.01, 0.5, 0.99]).round(3).to_string())
    archive_v1_artifact()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline, "version": 2, "features": FEATURES,
        "base_features": BASE_FEATURES, "relative_features": RELATIVE_FEATURES,
        "window_days": WINDOW_DAYS, "min_history": MIN_HISTORY,
        "relative_formula": "log1p(value) - log1p(baseline mean); 0 without a baseline",
    }, OUTPUT_PATH)
    print(f"Saved model: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_training()
