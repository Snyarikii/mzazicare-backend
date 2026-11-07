"""Evaluate the behavioral Isolation Forest against the CERT r4.2 insider answer key.

The script rebuilds the same user-day features as the training scripts, fits the same Isolation
Forest configuration, and compares its flags with the CERT ground truth. By default it evaluates
both feature sets side by side:

    v1  six population features (ml/train_behavioral_model.py)
    v2  the six population features plus six per-user rolling-baseline features
        (ml/train_behavioral_model_v2.py)

    python scripts/evaluate_ueba_iforest.py --answers path/to/answers [--model v1|v2|both]

--answers may be the extracted CERT "answers" folder or its insiders.csv file. When the
per-incident detail files (answers/r4.2-*/...) are present, a user-day is labelled malicious
when one of that insider's recorded malicious events falls on that day. Otherwise the
insider's start-to-end window from insiders.csv is used (--label-mode window).

Definitions (user-day level, "malicious" is the positive class):
    FAR (False Acceptance Rate) = malicious user-days not flagged / all malicious user-days
    FRR (False Rejection Rate)  = legitimate user-days flagged / all legitimate user-days

Outputs reports/iforest_evaluation.json and prints the same figures.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = ROOT / "ml" / "train_behavioral_model.py"
TRAINING_SCRIPT_V2 = ROOT / "ml" / "train_behavioral_model_v2.py"
DEPLOYED_MODEL = ROOT / "app" / "ml" / "behavioral_anomaly_model.joblib"
REPORT_PATH = ROOT / "reports" / "iforest_evaluation.json"

# Same configuration as ml/train_behavioral_model.py.
N_ESTIMATORS, CONTAMINATION, RANDOM_STATE = 150, 0.03, 42
CERT_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def load_training_module(path: Path = TRAINING_SCRIPT):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_cert_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format=CERT_DATE_FORMAT, errors="coerce")
    missing = parsed.isna() & values.notna()
    if missing.any():  # Some answer files omit seconds or use another layout.
        parsed[missing] = pd.to_datetime(values[missing], errors="coerce")
    return parsed


def find_insiders_csv(answers: Path) -> Path:
    if answers.is_file():
        return answers
    candidates = sorted(answers.rglob("insiders.csv"))
    if not candidates:
        raise FileNotFoundError(f"insiders.csv was not found under {answers}")
    return candidates[0]


def load_insiders(answers: Path, release: str) -> pd.DataFrame:
    path = find_insiders_csv(answers)
    insiders = pd.read_csv(path)
    insiders.columns = [column.strip().lower() for column in insiders.columns]
    required = {"dataset", "user", "start", "end"}
    if not required.issubset(insiders.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}; found {list(insiders.columns)}")
    wanted = insiders["dataset"].astype(str).str.strip().str.replace("r", "", regex=False)
    insiders = insiders[wanted == release.replace("r", "")].copy()
    if insiders.empty:
        raise ValueError(f"No insiders for release {release} in {path}")
    insiders["start"] = parse_cert_dates(insiders["start"].astype(str))
    insiders["end"] = parse_cert_dates(insiders["end"].astype(str))
    insiders["user"] = insiders["user"].astype(str).str.strip()
    return insiders


def event_days_from_details(answers: Path, insiders: pd.DataFrame) -> set[tuple[str, object]] | None:
    """User-days with at least one recorded malicious event, or None if details are missing."""
    if answers.is_file() or "details" not in insiders.columns:
        return None
    days: set[tuple[str, object]] = set()
    found_any = False
    for _, row in insiders.iterrows():
        matches = list(answers.rglob(str(row["details"]).strip()))
        if not matches:
            continue
        found_any = True
        for line in matches[0].read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split(",")
            if len(fields) < 3:
                continue
            stamp = parse_cert_dates(pd.Series([fields[2].strip()])).iloc[0]
            if pd.notna(stamp):
                days.add((row["user"], stamp.date()))
    return days if found_any else None


def label_user_days(index: pd.MultiIndex, insiders: pd.DataFrame, answers: Path,
                    mode: str) -> tuple[np.ndarray, str]:
    users = index.get_level_values(0).astype(str)
    dates = pd.to_datetime(pd.Index(index.get_level_values(1)))
    if mode in ("auto", "events"):
        event_days = event_days_from_details(answers, insiders)
        if event_days is not None:
            labels = np.fromiter(((u, d.date()) in event_days for u, d in zip(users, dates)),
                                 dtype=bool, count=len(index))
            return labels.astype(int), "events"
        if mode == "events":
            raise FileNotFoundError("Per-incident detail files were not found under --answers.")
    labels = np.zeros(len(index), dtype=int)
    for _, row in insiders.iterrows():
        start, end = row["start"].normalize(), row["end"].normalize()
        labels |= ((users == row["user"]) & (dates >= start) & (dates <= end)).astype(int)
    return labels, "window"


def make_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("isolation_forest", IsolationForest(
        n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1))])


def runtime_scores(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Vectorised copy of BehavioralAnomalyEngine.evaluate_session_activity's scoring."""
    decision = pipeline.decision_function(features)
    prediction = pipeline.predict(features)
    threshold = float(pipeline.named_steps["isolation_forest"].offset_)
    logit = np.log(0.65 / 0.35) + (threshold - decision) * 12.0
    raw = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    raw = np.where(prediction == -1, np.maximum(raw, 0.65), raw)
    calibrated = np.where(raw <= 0.40, raw,
                          np.where(raw < 0.70, 0.50 + (raw - 0.40) * 0.80, 0.75 + (raw - 0.70)))
    critical = (features["records_accessed_count"].to_numpy() >= 35) | \
               (features["total_payload_bytes"].to_numpy() >= 10 * 1024 * 1024)
    calibrated = np.where(critical, np.maximum(calibrated, 0.85), calibrated)
    return np.round(np.clip(calibrated, 0.0, 1.0), 2)


def check_runtime_parity(pipeline: Pipeline, features: pd.DataFrame, samples: int = 300,
                         frame: pd.DataFrame | None = None, bundle: dict | None = None) -> str:
    """Confirm the vectorised scores match the deployed engine code on a random sample.

    For version 2 the engine receives only the raw six counters and the user's baseline means,
    and recomputes the relative features itself, so this also checks the feature formula.
    """
    sys.path.insert(0, str(ROOT))
    try:
        import asyncio
        from app.services.behavioral_engine import BASE_FEATURES, BehavioralAnomalyEngine
    except Exception as exc:  # The app package needs the backend's dependencies.
        return f"skipped ({type(exc).__name__}: {exc})"
    engine = BehavioralAnomalyEngine.__new__(BehavioralAnomalyEngine)
    engine.configure(bundle if bundle is not None else {"pipeline": pipeline})
    sample = features.sample(min(samples, len(features)), random_state=RANDOM_STATE)
    expected = runtime_scores(pipeline, sample)

    def baseline_for(position):
        if frame is None:
            return None
        row = frame.loc[position]
        means = {name: row[f"baseline_{name}"] for name in BASE_FEATURES}
        if any(pd.isna(value) for value in means.values()):
            return {"means": {}, "samples": 0}
        return {"means": {k: float(v) for k, v in means.items()}, "samples": int(row["baseline_samples"])}

    async def score_all():
        return [(await engine.evaluate_session_activity(
                    "eval", {name: row[name] for name in BASE_FEATURES}, None,
                    baseline=baseline_for(position)))["anomaly_score"]
                for position, row in sample.iterrows()]

    actual = np.array(asyncio.run(score_all()))
    mismatches = int((np.abs(actual - expected) > 1e-9).sum())
    return "matched" if mismatches == 0 else f"{mismatches} of {len(sample)} scores differed"


def rates(y_true: np.ndarray, flagged: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (flagged == 1)).sum())
    fn = int(((y_true == 1) & (flagged == 0)).sum())
    fp = int(((y_true == 0) & (flagged == 1)).sum())
    tn = int(((y_true == 0) & (flagged == 0)).sum())
    positives, negatives = tp + fn, fp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    flag_rate = (tp + fp) / len(y_true) if len(y_true) else 0.0
    return {
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "FAR": fn / positives if positives else None,
        "FRR": fp / negatives if negatives else None,
        "precision": precision,
        "recall_detection_rate": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "flag_rate": flag_rate,
        # A detector that flags user-days at random with the same rate.
        "random_baseline_same_flag_rate": {"FAR": 1 - flag_rate, "FRR": flag_rate},
    }


def user_level(index: pd.MultiIndex, y_true: np.ndarray, flagged: np.ndarray) -> dict:
    frame = pd.DataFrame({"user": index.get_level_values(0).astype(str), "y": y_true, "flag": flagged})
    insiders = frame[frame["y"] == 1]["user"].unique()
    detected = frame[(frame["y"] == 1) & (frame["flag"] == 1)]["user"].unique()
    legit_users = frame.groupby("user")["y"].max()
    legit_users = legit_users[legit_users == 0].index
    legit_flagged = frame[frame["user"].isin(legit_users) & (frame["flag"] == 1)]["user"].unique()
    return {
        "insider_users": int(len(insiders)),
        "insider_users_detected_on_a_malicious_day": int(len(detected)),
        "legitimate_users": int(len(legit_users)),
        "legitimate_users_flagged_at_least_once": int(len(legit_flagged)),
    }


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def evaluate(name: str, pipeline: Pipeline, features: pd.DataFrame, y_true: np.ndarray) -> dict:
    anomaly_score = -pipeline.decision_function(features)
    model_flags = (pipeline.predict(features) == -1).astype(int)
    runtime = runtime_scores(pipeline, features)
    result = {
        "evaluated_user_days": int(len(features)),
        "malicious_user_days": int(y_true.sum()),
        "threshold_free": {
            "roc_auc": float(roc_auc_score(y_true, anomaly_score)) if 0 < y_true.sum() < len(y_true) else None,
            "average_precision": float(average_precision_score(y_true, anomaly_score)) if y_true.sum() else None,
        },
        "isolation_forest_flag": rates(y_true, model_flags),
        "runtime_challenge_or_deny": rates(y_true, (runtime >= 0.50).astype(int)),
        "runtime_deny_only": rates(y_true, (runtime >= 0.75).astype(int)),
        "user_level_isolation_forest_flag": user_level(features.index, y_true, model_flags),
    }
    flag = result["isolation_forest_flag"]
    print(f"\n[{name}] user-days={len(features):,}  malicious={int(y_true.sum()):,}")
    if not y_true.sum():
        print("  No malicious user-days in this partition: FAR is undefined (n/a); FRR is still reported.")
    print(f"  Isolation Forest flag  FAR={fmt(flag['FAR'])}  FRR={fmt(flag['FRR'])}  "
          f"precision={fmt(flag['precision'])}  recall={fmt(flag['recall_detection_rate'])}  F1={fmt(flag['f1'])}")
    print(f"  Confusion matrix       {flag['confusion_matrix']}")
    run = result["runtime_challenge_or_deny"]
    print(f"  Runtime CHALLENGE/DENY FAR={fmt(run['FAR'])}  FRR={fmt(run['FRR'])}")
    base = flag["random_baseline_same_flag_rate"]
    print(f"  Random baseline (same flag rate {fmt(flag['flag_rate'])}) FAR={fmt(base['FAR'])}  FRR={fmt(base['FRR'])}")
    free = result["threshold_free"]
    print(f"  ROC-AUC={fmt(free['roc_auc'])}  average precision={fmt(free['average_precision'])}")
    return result


def latency_ms(pipeline: Pipeline, features: pd.DataFrame, runs: int = 500) -> dict:
    row = features.iloc[[0]]
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        pipeline.decision_function(row)
        timings.append((time.perf_counter() - started) * 1000)
    return {"runs": runs, "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95))}


def evaluate_model(name: str, features: pd.DataFrame, y_true: np.ndarray, split: float,
                   frame: pd.DataFrame | None = None, bundle: dict | None = None) -> dict:
    """Transductive and temporal hold-out evaluation of one feature set."""
    result: dict = {"features": list(features.columns)}
    # 1. Transductive: fitted on every user-day, as the training scripts do.
    pipeline = make_pipeline().fit(features)
    result["transductive"] = evaluate(f"{name} transductive (fit and evaluate on all user-days)",
                                      pipeline, features, y_true)
    if bundle is not None:
        bundle = {**bundle, "pipeline": pipeline}
    result["runtime_parity_check"] = check_runtime_parity(pipeline, features, frame=frame, bundle=bundle)
    print(f"  Runtime engine parity check ({name}):", result["runtime_parity_check"])
    result["inference_latency_single_user_day"] = latency_ms(pipeline, features)

    # 2. Temporal hold-out: fit on earlier days, evaluate on later unseen days.
    day_values = pd.to_datetime(pd.Index(features.index.get_level_values(1)))
    days = np.sort(day_values.unique())
    cutoff = days[int(len(days) * split)]
    fit_mask, test_mask = day_values < cutoff, day_values >= cutoff
    held_out = make_pipeline().fit(features[fit_mask])
    result["temporal_holdout"] = {"cutoff_day": str(pd.Timestamp(cutoff).date()),
                                  "fit_user_days": int(fit_mask.sum()),
                                  "malicious_user_days_in_fit": int(y_true[fit_mask].sum()),
                                  **evaluate(f"{name} temporal hold-out (test from {pd.Timestamp(cutoff).date()})",
                                             held_out, features[test_mask], y_true[test_mask])}
    return result


def summary_row(name: str, block: dict) -> str:
    flag = block["isolation_forest_flag"]
    users = block["user_level_isolation_forest_flag"]
    free = block["threshold_free"]
    return (f"  {name:<16} AUC={fmt(free['roc_auc'])}  AP={fmt(free['average_precision'])}  "
            f"FAR={fmt(flag['FAR'])} (random {fmt(flag['random_baseline_same_flag_rate']['FAR'])})  "
            f"FRR={fmt(flag['FRR'])}  insiders {users['insider_users_detected_on_a_malicious_day']}/"
            f"{users['insider_users']}  legitimate users flagged {users['legitimate_users_flagged_at_least_once']}/"
            f"{users['legitimate_users']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, required=True,
                        help="CERT answers folder (with insiders.csv) or the insiders.csv file")
    parser.add_argument("--release", default="4.2", help="CERT release to evaluate (default 4.2)")
    parser.add_argument("--label-mode", choices=["auto", "events", "window"], default="auto")
    parser.add_argument("--temporal-split", type=float, default=0.70,
                        help="Fraction of calendar days used for fitting in the temporal hold-out")
    parser.add_argument("--model", choices=["v1", "v2", "both"], default="both",
                        help="Feature set to evaluate (default both, side by side)")
    args = parser.parse_args()

    training = load_training_module(TRAINING_SCRIPT)
    v2 = load_training_module(TRAINING_SCRIPT_V2)
    print("Building user-day features from", training.LOGON_PATH.parent)
    matrix = training.build_user_day_matrix()
    frame = v2.add_relative_features(matrix)
    insiders = load_insiders(args.answers, args.release)
    y_true, label_source = label_user_days(frame.index, insiders, args.answers, args.label_mode)
    print(f"Labels: {label_source} mode, {len(insiders)} insider records, "
          f"{int(y_true.sum()):,} malicious of {len(y_true):,} user-days")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
        "data": {"logon": str(training.LOGON_PATH), "file": str(training.FILE_PATH),
                 "user_days": int(len(frame)), "users": int(frame.index.get_level_values(0).nunique()),
                 "first_day": str(min(frame.index.get_level_values(1))),
                 "last_day": str(max(frame.index.get_level_values(1))),
                 "user_days_with_baseline": int((frame["baseline_samples"] >= v2.MIN_HISTORY).sum()),
                 "label_source": label_source, "insider_records": int(len(insiders)),
                 "malicious_user_days": int(y_true.sum())},
        "model": {"n_estimators": N_ESTIMATORS, "contamination": CONTAMINATION,
                  "random_state": RANDOM_STATE, "scaler": "StandardScaler",
                  "baseline_window_days": v2.WINDOW_DAYS, "baseline_min_history": v2.MIN_HISTORY},
        "definitions": {"FAR": "malicious user-days not flagged / malicious user-days",
                        "FRR": "legitimate user-days flagged / legitimate user-days"},
    }
    v2_bundle = {"version": 2, "features": v2.FEATURES, "base_features": v2.BASE_FEATURES,
                 "relative_features": v2.RELATIVE_FEATURES, "window_days": v2.WINDOW_DAYS,
                 "min_history": v2.MIN_HISTORY}
    if args.model in ("v1", "both"):
        print("\n=== v1: population features ===")
        report["v1"] = evaluate_model("v1", frame[training.FEATURES], y_true, args.temporal_split)
    if args.model in ("v2", "both"):
        print("\n=== v2: population + per-user rolling baseline ===")
        report["v2"] = evaluate_model("v2", frame[v2.FEATURES], y_true, args.temporal_split,
                                      frame=frame, bundle=v2_bundle)

    # The model currently deployed in app/ml, if present.
    if DEPLOYED_MODEL.is_file():
        import joblib
        artifact = joblib.load(DEPLOYED_MODEL)
        deployed = artifact.get("pipeline") if isinstance(artifact, dict) else artifact
        version = artifact.get("version", 1) if isinstance(artifact, dict) else 1
        columns = v2.FEATURES if version == 2 else training.FEATURES
        report["deployed_artifact_on_all_user_days"] = {"version": version, **evaluate(
            f"Deployed artifact app/ml/behavioral_anomaly_model.joblib (version {version})",
            deployed, frame[columns], y_true)}

    print("\n=== Summary ===")
    for tag, title in (("transductive", "All user-days"), ("temporal_holdout", "Temporal hold-out")):
        print(title)
        for name in ("v1", "v2"):
            if name in report:
                print(summary_row(name, report[name][tag]))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
