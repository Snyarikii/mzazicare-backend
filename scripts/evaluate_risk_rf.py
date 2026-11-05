"""Evaluate the perimeter Random Forest risk model on the RBA dataset.

Reproduces the data sampling, features and 80/20 stratified split of ml/train_risk_model.py,
then reports Precision, Recall, F1, ROC-AUC and the confusion matrix on the held-out 20%, at the
default 0.5 threshold and at a threshold tuned on validation data drawn from the training rows.

    python scripts/evaluate_risk_rf.py
    python scripts/evaluate_risk_rf.py --sampling random --seed 7

Models compared on the same held-out rows:
  * deployed  - app/ml/adaptive_risk_model.joblib, if present (trained on the same 80% split)
  * retrained - the training configuration fitted again, to confirm reproducibility
  * baselines - majority class and logistic regression with the same preprocessing

Outputs:
  reports/rf_evaluation.json         all metrics, counts and environment details
  reports/figures/rf_roc_curve.png   ROC curve (300 dpi)
  reports/rba_sample_rows.csv        three raw rows of the fields the model uses (for Chapter 3)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = ROOT / "ml" / "train_risk_model.py"
DEPLOYED_MODEL = ROOT / "app" / "ml" / "adaptive_risk_model.joblib"
REPORTS = ROOT / "reports"
DEFAULT_DATA = [ROOT / "ml" / "data" / "rba-dataset.csv", ROOT / "ml" / "rba-dataset.csv"]
RAW_COLUMNS = ["Login Timestamp", "Device Type", "OS Name and Version", "Round-Trip Time [ms]",
               "Is Attack IP", "Is Account Takeover"]
CATEGORICAL, NUMERICAL = ["device_type", "os_family"], ["hour_sin", "hour_cos", "rtt_ms"]
SEED = 42


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_risk_model", TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_attack(frame: pd.DataFrame) -> pd.Series:
    flag = lambda column: frame[column].astype(str).str.lower().isin(["true", "1", "t"])
    return flag("Is Account Takeover") | flag("Is Attack IP")


def random_sample(path: Path, normal: int, attack: int, fraction: float, seed: int, chunksize: int = 200_000):
    """One pass over the whole file: keep every attack row and a random fraction of legitimate rows."""
    rng = np.random.default_rng(seed)
    attacks, normals, rows_read = [], [], 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        rows_read += len(chunk)
        mask = is_attack(chunk)
        attacks.append(chunk[mask])
        legit = chunk[~mask]
        normals.append(legit[rng.random(len(legit)) < fraction])
    attacks = pd.concat(attacks, ignore_index=True)
    normals = pd.concat(normals, ignore_index=True)
    if len(normals) < normal:
        raise ValueError(f"Only {len(normals)} legitimate rows sampled; raise --random-fraction.")
    attacks = attacks.sample(min(attack, len(attacks)), random_state=seed)
    normals = normals.sample(normal, random_state=seed)
    return pd.concat([normals, attacks], ignore_index=True), rows_read


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERICAL),
    ])


def training_pipeline() -> Pipeline:
    # Same configuration as ml/train_risk_model.py.
    return Pipeline([("preprocessor", preprocessor()), ("classifier", RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=SEED, n_jobs=-1))])


def scores(y_true, probability, threshold: float = 0.5) -> dict:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    single_class = len(np.unique(y_true)) < 2
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": None if single_class else float(roc_auc_score(y_true, probability)),
        "average_precision": None if single_class else float(average_precision_score(y_true, probability)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def tuned_threshold(make_model, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Choose the F1-maximising threshold on a validation split of the training data only.

    The held-out test set is never used to pick the threshold, so the tuned results stay honest.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                  random_state=SEED, stratify=y_train)
    probability = make_model().fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probability)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def hour_from_cyclical(sin_values, cos_values) -> np.ndarray:
    angle = np.arctan2(sin_values, cos_values)
    return np.mod(np.rint(angle * 24 / (2 * np.pi)), 24).astype(int)


def policy_simulation(probability: np.ndarray, hours: np.ndarray, y_true: np.ndarray) -> dict:
    """Apply the risk engine's time-of-day and device-trust adjustments to the model probability.

    Mirrors AdaptiveRiskEngine.evaluate_risk for a non-flagged IP on a physical device with real
    GPS, so only the model score, the hour and the device trust state vary. Compared with a static
    policy that challenges every sign-in.
    """
    base = probability.copy()
    evening = (hours >= 17) & (hours <= 22) & (base > 0.40)
    base[evening] = 0.40
    night = hours < 5
    base[night] = np.maximum(base[night], 0.65)
    adjustments = {"NEW": 0.10, "PROBATIONARY": 0.0, "TRUSTED (active within 30 days)": -0.25}
    result = {}
    for state, adjustment in adjustments.items():
        score = np.clip(np.round(base + adjustment, 2), 0, 1)
        step_up = score >= 0.50
        legit, attack = y_true == 0, y_true == 1
        result[state] = {
            "legitimate_sign_ins_challenged_or_denied": float(step_up[legit].mean()) if legit.any() else None,
            "attack_sign_ins_challenged_or_denied": float(step_up[attack].mean()) if attack.any() else None,
            "attack_sign_ins_denied": float((score >= 0.85)[attack].mean()) if attack.any() else None,
        }
    result["static_mfa_baseline"] = {"legitimate_sign_ins_challenged_or_denied": 1.0,
                                     "attack_sign_ins_challenged_or_denied": 1.0}
    return result


def latency_ms(model: Pipeline, row: pd.DataFrame, runs: int = 500) -> dict:
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        model.predict_proba(row)
        timings.append((time.perf_counter() - started) * 1000)
    return {"runs": runs, "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95))}


def plot_roc(curves: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for label, (y_true, probability, auc) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, probability)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate (legitimate sign-ins flagged)")
    ax.set_ylabel("True positive rate (attack sign-ins flagged)")
    ax.set_title("Perimeter risk model: ROC curve on the 20% hold-out")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="Path to rba-dataset.csv")
    parser.add_argument("--sampling", choices=["training", "random"], default="training",
                        help="training: the training script's own sample (a seeded random sample of the "
                             "whole file); random: an independent random sample drawn with --seed")
    parser.add_argument("--normal", type=int, default=15000)
    parser.add_argument("--attack", type=int, default=3000)
    parser.add_argument("--random-fraction", type=float, default=0.002,
                        help="Share of legitimate rows kept during the random-sampling pass")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seed for --sampling random (the training sample uses 42)")
    args = parser.parse_args()

    data_path = args.data or next((p for p in DEFAULT_DATA if p.is_file()), None)
    if data_path is None or not data_path.is_file():
        raise FileNotFoundError("rba-dataset.csv not found; pass --data PATH")
    training = load_training_module()
    print(f"Loading {data_path} ({args.sampling} sampling)")

    rows_read = None
    if args.sampling == "training":
        X, y = training.load_rba_csv(str(data_path), target_normal=args.normal, target_attack=args.attack)
        raw_sample = next(pd.read_csv(data_path, chunksize=3, low_memory=False))
    else:
        raw, rows_read = random_sample(data_path, args.normal, args.attack, args.random_fraction, args.seed)
        X, y = training.engineer_features(raw)
        raw_sample = raw.head(3)
    y = pd.Series(np.asarray(y), name="label")
    print(f"Sample: {len(y):,} sign-ins ({int((y == 0).sum()):,} legitimate, {int(y.sum()):,} attack)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"Split: {len(y_train):,} training / {len(y_test):,} held-out (stratified, seed {SEED})")

    results, curves, probabilities = {}, {}, {}
    retrained = training_pipeline().fit(X_train, y_train)
    probabilities["retrained_random_forest"] = retrained.predict_proba(X_test)[:, 1]

    if DEPLOYED_MODEL.is_file():
        deployed = joblib.load(DEPLOYED_MODEL)
        probabilities["deployed_random_forest"] = deployed.predict_proba(X_test)[:, 1]
        agreement = float(np.mean((probabilities["deployed_random_forest"] >= 0.5) ==
                                  (probabilities["retrained_random_forest"] >= 0.5)))
        results["deployed_vs_retrained_prediction_agreement"] = agreement
    else:
        deployed = None
        print("Deployed model not found; reporting the retrained model only.")

    for name, classifier in (("baseline_majority_class", DummyClassifier(strategy="most_frequent")),
                             ("baseline_logistic_regression", LogisticRegression(max_iter=1000))):
        model = Pipeline([("preprocessor", preprocessor()), ("classifier", clone(classifier))])
        probabilities[name] = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]

    y_true = y_test.to_numpy()
    for name, probability in probabilities.items():
        results[name] = scores(y_true, probability)
        metrics = results[name]
        auc = metrics["roc_auc"]
        print(f"\n{name}")
        print(f"  Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
              f"F1={metrics['f1']:.4f}  ROC-AUC={'n/a' if auc is None else f'{auc:.4f}'}")
        cm = metrics["confusion_matrix"]
        print("  Confusion matrix (rows = actual, columns = predicted)")
        print(f"                 Legitimate  Attack")
        print(f"    Legitimate   {cm['TN']:>10,}  {cm['FP']:>6,}")
        print(f"    Attack       {cm['FN']:>10,}  {cm['TP']:>6,}")
        if name in ("deployed_random_forest", "retrained_random_forest", "baseline_logistic_regression") and auc:
            curves[name.replace("_", " ").capitalize()] = (y_true, probability, auc)

    # Operating point: the 0.5 cut-off suits balanced classes; pick one on validation data instead.
    tuned = {"selected_on": "25% validation split of the training rows (F1-maximising)"}
    makers = {"random_forest": training_pipeline,
              "logistic_regression": lambda: Pipeline([("preprocessor", preprocessor()),
                                                       ("classifier", LogisticRegression(max_iter=1000))])}
    for family, make_model in makers.items():
        threshold = tuned_threshold(make_model, X_train, y_train)
        targets = (["retrained_random_forest", "deployed_random_forest"] if family == "random_forest"
                   else ["baseline_logistic_regression"])
        for name in targets:
            if name in probabilities:
                tuned[name] = scores(y_true, probabilities[name], threshold)
                m = tuned[name]
                print(f"\n{name} at tuned threshold {threshold:.3f}: Precision={m['precision']:.4f}  "
                      f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}  {m['confusion_matrix']}")
    results["tuned_threshold"] = tuned

    evaluated = "deployed_random_forest" if deployed is not None else "retrained_random_forest"
    hours = hour_from_cyclical(X_test["hour_sin"].to_numpy(), X_test["hour_cos"].to_numpy())
    results["engine_policy_simulation"] = {"model": evaluated,
                                           **policy_simulation(probabilities[evaluated], hours, y_true)}
    print(f"\nRisk-engine policy simulation on the held-out sign-ins ({evaluated}):")
    for state, values in results["engine_policy_simulation"].items():
        if isinstance(values, dict):
            print(f"  {state:32} legitimate challenged/denied="
                  f"{values['legitimate_sign_ins_challenged_or_denied']:.4f}  attacks challenged/denied="
                  f"{values['attack_sign_ins_challenged_or_denied']:.4f}")

    final_model = deployed if deployed is not None else retrained
    results["inference_latency_single_sign_in"] = latency_ms(final_model, X_test.iloc[[0]])
    names = final_model.named_steps["preprocessor"].get_feature_names_out()
    results["feature_importance"] = dict(sorted(
        zip(map(str, names), map(float, final_model.named_steps["classifier"].feature_importances_)),
        key=lambda item: -item[1]))

    roc_path = REPORTS / "figures" / "rf_roc_curve.png"
    plot_roc(curves, roc_path)
    sample_path = REPORTS / "rba_sample_rows.csv"
    raw_sample.columns = raw_sample.columns.str.strip()
    raw_sample[[c for c in RAW_COLUMNS if c in raw_sample.columns]].to_csv(sample_path, index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "platform": platform.platform()},
        "data": {"path": str(data_path), "sampling": args.sampling,
                 "seed": 42 if args.sampling == "training" else args.seed, "rows_read": rows_read,
                 "sample_size": int(len(y)), "legitimate": int((y == 0).sum()), "attack": int(y.sum()),
                 "label": "Is Account Takeover OR Is Attack IP",
                 "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
                 "test_legitimate": int((y_test == 0).sum()), "test_attack": int(y_test.sum())},
        "model": {"type": "RandomForestClassifier", "n_estimators": 100, "max_depth": 8,
                  "random_state": SEED, "features": CATEGORICAL + NUMERICAL,
                  "preprocessing": "OneHotEncoder(categorical) + StandardScaler(numerical)"},
        "results": results,
        "outputs": {"roc_curve": str(roc_path), "sample_rows": str(sample_path)},
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "rf_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {REPORTS / 'rf_evaluation.json'}, {roc_path} and {sample_path}")


if __name__ == "__main__":
    main()
