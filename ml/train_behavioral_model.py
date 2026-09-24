"""Train a user-day behavioral anomaly model from CMU CERT r4.2 streams."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LOGON_PATH = ROOT / "ml" / "data" / "logon.csv"
FILE_PATH = ROOT / "ml" / "data" / "file.csv"
OUTPUT_PATH = ROOT / "app" / "ml" / "behavioral_anomaly_model.joblib"
FEATURES = ["total_logons", "distinct_devices", "activity_span_hours", "off_hours_ratio",
            "records_accessed_count", "total_payload_bytes"]
KEYS = ["user", "date_only"]


def _event_aggregates(path: Path, stream: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required training data not found: {path}")
    wanted = ["date", "user", "pc", "activity"] if stream == "logon" else ["date", "user", "pc", "content"]
    partials = []
    for chunk in pd.read_csv(path, usecols=wanted, dtype={"user": "category", "pc": "category"},
                             chunksize=250_000, low_memory=True):
        chunk["timestamp"] = pd.to_datetime(chunk["date"], errors="coerce", dayfirst=True, utc=True)
        chunk.dropna(subset=["timestamp", "user"], inplace=True)
        chunk["date_only"] = chunk["timestamp"].dt.date
        chunk["hour"] = chunk["timestamp"].dt.hour
        chunk["off_hours"] = ((chunk["hour"] < 6) | (chunk["hour"] >= 19)).astype("int8")
        if stream == "logon":
            chunk = chunk[chunk["activity"].astype(str).str.lower().eq("logon")].copy()
            if chunk.empty:
                continue
            agg = chunk.groupby(KEYS, observed=True, sort=False).agg(
                total_logons=("timestamp", "size"), logon_devices=("pc", "nunique"),
                logon_min_hour=("hour", "min"), logon_max_hour=("hour", "max"),
                logon_offhours=("off_hours", "sum"), logon_actions=("timestamp", "size"),
                logon_endpoints=("pc", lambda s: frozenset(s.dropna().astype(str))))
        else:
            # Derive byte counts and discard large content strings before any grouping.
            chunk["file_size_bytes"] = chunk["content"].fillna("").astype(str).str.len()
            chunk.drop(columns=["content"], inplace=True)
            agg = chunk.groupby(KEYS, observed=True, sort=False).agg(
                records_accessed_count=("file_size_bytes", "size"),
                total_payload_bytes=("file_size_bytes", "sum"),
                file_min_hour=("hour", "min"), file_max_hour=("hour", "max"),
                file_offhours=("off_hours", "sum"), file_actions=("timestamp", "size"),
                file_endpoints=("pc", lambda s: frozenset(s.dropna().astype(str))))
        partials.append(agg)
    if not partials:
        return pd.DataFrame()
    raw = pd.concat(partials)
    numeric = raw.select_dtypes(include="number").groupby(level=[0, 1], sort=False).sum()
    for col in ["logon_min_hour", "file_min_hour"]:
        if col in raw:
            numeric[col] = raw[col].groupby(level=[0, 1], sort=False).min()
    for col in ["logon_max_hour", "file_max_hour"]:
        if col in raw:
            numeric[col] = raw[col].groupby(level=[0, 1], sort=False).max()
    for col in ["logon_endpoints", "file_endpoints"]:
        if col in raw:
            numeric[col] = raw[col].groupby(level=[0, 1], sort=False).agg(
                lambda groups: frozenset().union(*groups))
    combined = numeric
    return combined


def build_user_day_matrix() -> pd.DataFrame:
    # Process streams by chunk and aggregate counts/endpoint diversity without loading the CSVs whole.
    logons = _event_aggregates(LOGON_PATH, "logon")
    files = _event_aggregates(FILE_PATH, "file")
    if logons.empty:
        raise ValueError("logon.csv did not contain valid Logon events")
    index = logons.index.union(files.index if not files.empty else logons.index)
    result = pd.DataFrame(index=index)
    result["total_logons"] = logons["total_logons"]
    result["distinct_devices"] = logons["logon_devices"]
    result["logon_min_hour"] = logons["logon_min_hour"]
    result["logon_max_hour"] = logons["logon_max_hour"]
    result["logon_endpoints"] = logons["logon_endpoints"]
    # Combined-action off-hours ratio; file count/after-hours counts are joined below.
    result["logon_offhours"] = logons["logon_offhours"]
    result["logon_actions"] = logons["logon_actions"]
    if not files.empty:
        for col in ("records_accessed_count", "total_payload_bytes", "file_offhours", "file_actions", "file_min_hour", "file_max_hour", "file_endpoints"):
            if col in files:
                result[col] = files[col]
    for col in ["total_logons", "logon_devices", "logon_offhours", "logon_actions", "records_accessed_count",
                "total_payload_bytes", "file_offhours", "file_actions", "file_min_hour", "file_max_hour"]:
        if col not in result:
            result[col] = 0
    result[["total_logons", "records_accessed_count", "total_payload_bytes"]] = result[
        ["total_logons", "records_accessed_count", "total_payload_bytes"]].fillna(0)
    def endpoint_count(row):
        left, right = row.get("logon_endpoints", frozenset()), row.get("file_endpoints", frozenset())
        return len((left if isinstance(left, frozenset) else frozenset()) |
                   (right if isinstance(right, frozenset) else frozenset()))
    result["distinct_devices"] = result.apply(endpoint_count, axis=1)
    result["off_hours_ratio"] = (result["logon_offhours"].fillna(0) + result["file_offhours"].fillna(0)) / (
        result["logon_actions"].fillna(0) + result["file_actions"].fillna(0))
    result["off_hours_ratio"] = result["off_hours_ratio"].replace([np.inf, -np.inf], 0).fillna(0)
    # Activity span: hourly extrema across streams (hour-of-day as requested).
    mins = pd.concat([result["logon_min_hour"], result["file_min_hour"]], axis=1)
    maxs = pd.concat([result["logon_max_hour"], result["file_max_hour"]], axis=1)
    result["activity_span_hours"] = (maxs.max(axis=1) - mins.min(axis=1)).fillna(0).clip(lower=0)
    return result[FEATURES].replace([np.inf, -np.inf], 0).fillna(0).astype(float)


def run_training() -> None:
    profiles = build_user_day_matrix()
    if len(profiles) < 2:
        raise ValueError("At least two User-Day profiles are required.")
    pipeline = Pipeline([("scaler", StandardScaler()), ("isolation_forest", IsolationForest(
        n_estimators=150, contamination=0.03, random_state=42, n_jobs=-1))])
    pipeline.fit(profiles[FEATURES])
    predictions = pipeline.predict(profiles[FEATURES])
    flagged = int((predictions == -1).sum())
    print(f"Total User-Day profiles: {len(profiles):,}")
    print(f"Flagged anomalies: {flagged:,} ({flagged / len(profiles):.2%})")
    print("Feature quantiles (0%, 25%, 50%, 75%, 100%):")
    print(profiles[FEATURES].quantile([0, .25, .5, .75, 1]).to_string())
    # Isolation Forest has no native feature_importances_; report permutation-free feature ranges
    # and the model's per-feature training spread as transparent baseline diagnostics.
    print("Feature importance: IsolationForest does not expose impurity importances; feature spread:")
    print(profiles[FEATURES].std().sort_values(ascending=False).to_string())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": FEATURES}, OUTPUT_PATH)
    print(f"Saved model: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_training()
