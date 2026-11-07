"""Train the behavioural Isolation Forest with data-export features (model version 3).

Version 3 = version 2 (six population features plus the per-user rolling baseline) plus a
seventh counter, data_exports: the number of times patient data left the app.

In MzaziCare a data export is a share from the app or a screenshot of a patient screen. The CERT
r4.2 dataset has no mobile app, so its closest equivalent is used for training: a removable-drive
(USB) connection in device.csv, the moment data leaves the organisation in CERT's scenarios.

    data_exports            exports that day (CERT: USB connections)
    relative_data_exports   log1p(today) - log1p(the user's mean over their previous 30 days)
    first_data_export       1 when the user exports for the first time after at least five
                            earlier days without any export, otherwise 0

    python ml/train_behavioral_model_v3.py

Needs ml/data/logon.csv, file.csv and device.csv. Writes app/ml/behavioral_anomaly_model.joblib
and archives the artifact it replaces under ml/artifacts/.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2_SCRIPT = ROOT / "ml" / "train_behavioral_model_v2.py"
DEVICE_PATH = ROOT / "ml" / "data" / "device.csv"
OUTPUT_PATH = ROOT / "app" / "ml" / "behavioral_anomaly_model.joblib"
ARCHIVE_DIR = ROOT / "ml" / "artifacts"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = load_module(V2_SCRIPT, "train_behavioral_model_v2")
EXPORT_FEATURE = "data_exports"
BASE_FEATURES = v2.BASE_FEATURES + [EXPORT_FEATURE]
RELATIVE_FEATURES = [f"relative_{name}" for name in BASE_FEATURES]
FLAG_FEATURES = ["first_data_export"]
FEATURES = BASE_FEATURES + RELATIVE_FEATURES + FLAG_FEATURES
WINDOW_DAYS, MIN_HISTORY = v2.WINDOW_DAYS, v2.MIN_HISTORY
make_pipeline = v2.make_pipeline


def device_exports(path: Path = DEVICE_PATH) -> pd.Series:
    """USB connections per (user, date) from CERT device.csv."""
    if not path.is_file():
        raise FileNotFoundError(f"Required training data not found: {path}")
    parts = []
    for chunk in pd.read_csv(path, usecols=["date", "user", "activity"], chunksize=250_000):
        chunk = chunk[chunk["activity"].astype(str).str.lower().eq("connect")]
        # CERT r4.2 dates are month-first.
        stamp = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        chunk = chunk.assign(date_only=stamp.dt.date).dropna(subset=["date_only", "user"])
        parts.append(chunk.groupby(["user", "date_only"]).size())
    counts = pd.concat(parts).groupby(level=[0, 1]).sum()
    counts.index = counts.index.set_names(["user", "date_only"])
    return counts.rename(EXPORT_FEATURE).astype(float)


def add_export_column(matrix: pd.DataFrame, exports: pd.Series) -> pd.DataFrame:
    """Join export counts onto the user-day matrix; days with only USB activity are added."""
    matrix = matrix.copy()
    matrix.index = matrix.index.set_names(["user", "date_only"])
    index = matrix.index.union(exports.index)
    out = matrix.reindex(index).fillna(0.0)
    out[EXPORT_FEATURE] = exports.reindex(index).fillna(0.0)
    return out


def add_relative_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Baseline means, relative features and first_data_export for a (user, date) matrix that
    already has the seven BASE_FEATURES. Returned in the input's index order."""
    frame = matrix[BASE_FEATURES].astype(float).copy()
    users = frame.index.get_level_values(0).astype(str)
    dates = pd.to_datetime(pd.Index(frame.index.get_level_values(1)))
    work = frame.reset_index(drop=True)
    work["_user"], work["_date"], work["_row"] = users, dates, np.arange(len(work))
    work = work.sort_values(["_user", "_date"], kind="mergesort")
    grouped = work.groupby("_user", sort=False)
    prior_days = grouped.cumcount()
    work["baseline_samples"] = prior_days.clip(upper=WINDOW_DAYS)
    for name in BASE_FEATURES:
        previous = grouped[name].shift(1)
        mean = previous.groupby(work["_user"], sort=False).rolling(
            WINDOW_DAYS, min_periods=MIN_HISTORY).mean().reset_index(level=0, drop=True)
        work[f"baseline_{name}"] = mean
        work[f"relative_{name}"] = (np.log1p(work[name]) - np.log1p(mean)).fillna(0.0)
    earlier_exports = grouped[EXPORT_FEATURE].transform(lambda s: s.shift(1).fillna(0).cumsum())
    work["prior_data_exports"] = earlier_exports
    work["first_data_export"] = ((work[EXPORT_FEATURE] > 0) & (earlier_exports == 0)
                                 & (prior_days >= MIN_HISTORY)).astype(float)
    work = work.sort_values("_row")
    out = work.drop(columns=["_user", "_date", "_row"])
    out.index = frame.index
    return out


def build_training_frame() -> pd.DataFrame:
    v1 = v2.load_v1_module()
    print("Building user-day features from", v1.LOGON_PATH.parent)
    matrix = add_export_column(v1.build_user_day_matrix(), device_exports())
    return add_relative_features(matrix)


def archive_current_artifact() -> None:
    if not OUTPUT_PATH.is_file():
        return
    try:
        current = joblib.load(OUTPUT_PATH)
    except Exception:
        return
    version = current.get("version", 1) if isinstance(current, dict) else 1
    if version >= 3:
        return
    target = ARCHIVE_DIR / f"behavioral_anomaly_model_v{version}.joblib"
    if not target.is_file():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_PATH, target)
        print(f"Archived version {version} artifact: {target}")


def run_training() -> None:
    frame = build_training_frame()
    print(f"Total user-day profiles: {len(frame):,}  with exports: {int((frame[EXPORT_FEATURE] > 0).sum()):,}  "
          f"first exports: {int(frame['first_data_export'].sum()):,}")
    pipeline = make_pipeline().fit(frame[FEATURES])
    flagged = int((pipeline.predict(frame[FEATURES]) == -1).sum())
    print(f"Flagged anomalies: {flagged:,} ({flagged / len(frame):.2%})")
    archive_current_artifact()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline, "version": 3, "features": FEATURES,
        "base_features": BASE_FEATURES, "relative_features": RELATIVE_FEATURES,
        "flag_features": FLAG_FEATURES, "window_days": WINDOW_DAYS, "min_history": MIN_HISTORY,
        "relative_formula": "log1p(value) - log1p(baseline mean); 0 without a baseline",
        "export_proxy": "CERT r4.2 device.csv Connect events stand in for app shares and screenshots",
    }, OUTPUT_PATH)
    print(f"Saved model: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_training()
