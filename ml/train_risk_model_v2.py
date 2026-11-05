"""Perimeter risk model, version 2: per-user history ("novelty") features. Offline experiment.

Version 1 describes each login in isolation (device type, OS, hour, round-trip time). Risk-based
authentication research on this dataset (Wiefling et al., 2022) finds the signal in comparing a
login with the same user's earlier logins. This script adds, for every login, using ONLY that
user's earlier logins in the file:

    first_login              1 if the user has no earlier login
    log_prior_logins         log(1 + number of earlier logins)
    log_hours_since_last     log(1 + hours since the previous login; 0 for a first login)
    new_country              country not seen before for this user
    new_asn                  network (autonomous system) not seen before for this user
    new_user_agent           user-agent string not seen before for this user
    new_os                   operating system not seen before for this user
    new_device_type          device type not seen before for this user

The exact IP address is deliberately NOT used: mobile networks reassign addresses often, so
"new IP" would fire constantly for legitimate Kenyan users. The country and network are stable.

One streaming pass draws two independent random samples (seed 42 for training and testing,
seed 7 for an independent check), trains Random Forests with and without the history features
on the SAME rows, plus a logistic-regression baseline, and writes:

    ml/artifacts/adaptive_risk_model_v2.joblib     (the live API in app/ml is NOT touched)
    reports/rf_v2_evaluation.json
    reports/figures/rf_v2_roc_curve.png

    python ml/train_risk_model_v2.py
    python ml/train_risk_model_v2.py --label ato     (account takeovers only; 5-fold cross-validation)

With --label ato the positives are only the logins marked Is Account Takeover, which is what
risk-based authentication is designed to stop. Rows from attack IPs that are not account takeovers
are left out, because they are neither clear positives nor clear negatives. There are few account
takeovers, so every one is used and results come from stratified 5-fold cross-validation (each
login is scored by a model that never saw it). Outputs: reports/rf_v2_ato_evaluation.json and
reports/figures/rf_v2_ato_roc_curve.png; no model artefact is saved in this mode.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ml" / "data" / "rba-dataset.csv"
V1_SCRIPT = ROOT / "ml" / "train_risk_model.py"
ARTIFACT = ROOT / "ml" / "artifacts" / "adaptive_risk_model_v2.joblib"
REPORTS = ROOT / "reports"
SEED, CHECK_SEED = 42, 7

BASE_CAT = ["device_type", "os_family"]
BASE_NUM = ["hour_sin", "hour_cos", "rtt_ms"]
HISTORY = ["first_login", "log_prior_logins", "log_hours_since_last", "new_country", "new_asn",
           "new_user_agent", "new_os", "new_device_type"]
RAW = ["Login Timestamp", "User ID", "Round-Trip Time [ms]", "Country", "ASN", "User Agent String",
       "OS Name and Version", "Device Type", "Is Attack IP", "Is Account Takeover"]


def load_v1():
    spec = importlib.util.spec_from_file_location("train_risk_model", V1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flag(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().isin(["true", "1", "t"]).to_numpy()


def history_pass(path: Path, fraction: float, label_mode: str = "attack_or_ato", chunksize: int = 500_000):
    """Stream the file once in order, computing history features for every row and keeping the
    candidate rows for both samples (all attack rows plus a random share of legitimate rows)."""
    rng_a, rng_b = np.random.default_rng(SEED), np.random.default_rng(CHECK_SEED)
    seen: set[int] = set()            # hashes of (user, kind, value) already observed
    users: dict = {}                  # user -> [earlier login count, last login time in seconds]
    kept, rows, out_of_order, last_t = [], 0, 0, -math.inf
    started = time.time()

    header = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    missing = [c for c in RAW if c not in header]
    if missing:
        raise ValueError(f"Missing columns {missing}; found {header}")

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False,
                             usecols=lambda c: c.strip() in RAW):
        chunk.columns = chunk.columns.str.strip()
        n = len(chunk)
        ato, attack_ip = flag(chunk["Is Account Takeover"]), flag(chunk["Is Attack IP"])
        if label_mode == "ato":
            label = ato
            eligible = ~(attack_ip & ~ato)      # leave out attack-IP rows that are not takeovers
        else:
            label = ato | attack_ip
            eligible = np.ones(n, dtype=bool)
        in_a = label | ((rng_a.random(n) < fraction) & eligible)
        in_b = label | ((rng_b.random(n) < fraction) & eligible)
        keep = in_a | in_b

        ts = pd.to_datetime(chunk["Login Timestamp"], errors="coerce")
        tsec = ts.to_numpy(dtype="datetime64[s]").astype("int64")
        tsec = np.where(ts.isna().to_numpy(), -1, tsec)
        uid = chunk["User ID"].to_numpy()
        country = chunk["Country"].astype(str).to_numpy()
        asn = chunk["ASN"].astype(str).to_numpy()
        agent = chunk["User Agent String"].astype(str).to_numpy()
        os_name = chunk["OS Name and Version"].astype(str).str.split().str[0].fillna("").to_numpy()
        device = chunk["Device Type"].astype(str).to_numpy()

        features = np.zeros((n, len(HISTORY)), dtype=np.float32)
        for i in range(n):
            u, t = uid[i], tsec[i]
            if t >= 0:
                if t < last_t:
                    out_of_order += 1
                last_t = t
            state = users.get(u)
            if state is None:
                prior, gap = 0, 0.0
                state = [0, t]
                users[u] = state
            else:
                prior = state[0]
                gap = max((t - state[1]) / 3600.0, 0.0) if t >= 0 and state[1] >= 0 else 0.0
            keys = (hash((u, 0, country[i])), hash((u, 1, asn[i])), hash((u, 2, agent[i])),
                    hash((u, 3, os_name[i])), hash((u, 4, device[i])))
            if keep[i]:
                row = features[i]
                row[0] = 1.0 if prior == 0 else 0.0
                row[1] = math.log1p(prior)
                row[2] = math.log1p(gap)
                for k, key in enumerate(keys):
                    row[3 + k] = 0.0 if key in seen else 1.0
            for key in keys:
                seen.add(key)
            state[0] = prior + 1
            if t >= 0:
                state[1] = t

        if keep.any():
            part = chunk.loc[keep, RAW].copy()
            part[HISTORY] = features[keep]
            part["label"] = label[keep].astype(int)
            part["in_a"], part["in_b"] = in_a[keep], in_b[keep]
            part["row_id"] = np.arange(rows, rows + n)[keep]
            kept.append(part)
        rows += n
        print(f"   {rows:>12,} rows  {len(users):>10,} users  {time.time() - started:7.0f} s", flush=True)

    frame = pd.concat(kept, ignore_index=True)
    return frame, {"rows_read": rows, "users": len(users), "out_of_order_timestamps": out_of_order,
                   "pass_seconds": round(time.time() - started, 1)}


def draw(frame: pd.DataFrame, column: str, n_normal: int, n_attack: int, seed: int) -> pd.DataFrame:
    pool = frame[frame[column]]
    attacks = pool[pool["label"] == 1]
    normals = pool[pool["label"] == 0]
    if len(normals) < n_normal:
        raise ValueError(f"Only {len(normals)} legitimate candidates; raise --random-fraction")
    n_attack = len(attacks) if n_attack <= 0 else min(n_attack, len(attacks))
    return pd.concat([normals.sample(n_normal, random_state=seed),
                      attacks.sample(n_attack, random_state=seed)], ignore_index=True)


def to_features(v1, sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X_base, _ = v1.engineer_features(sample)
    X = pd.concat([X_base.reset_index(drop=True), sample[HISTORY].reset_index(drop=True)], axis=1)
    return X, sample["label"].reset_index(drop=True)


def pipeline(numeric: list[str], classifier) -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), BASE_CAT),
                             ("num", StandardScaler(), numeric)])
    return Pipeline([("preprocessor", pre), ("classifier", classifier)])


def forest():
    return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED, n_jobs=-1)


MODELS = {
    "rf_v1_features": lambda: pipeline(BASE_NUM, forest()),
    "rf_v2_history_features": lambda: pipeline(BASE_NUM + HISTORY, forest()),
    "logreg_v2_history_features": lambda: pipeline(BASE_NUM + HISTORY, LogisticRegression(max_iter=2000)),
    "majority_class": lambda: pipeline(BASE_NUM, DummyClassifier(strategy="most_frequent")),
}


def scores(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(float(threshold), 4),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def tuned_threshold(make, X, y) -> float:
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    p = make().fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))]) if len(thresholds) else 0.5


def cross_validated(make, X, y, use_tuned: bool, folds: int = 5):
    """Out-of-fold probabilities and predictions; each fold's threshold is tuned on its own training rows."""
    p = np.zeros(len(y)); predicted = np.zeros(len(y), dtype=int); thresholds = []
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        X_tr, y_tr = X.iloc[train], y.iloc[train]
        threshold = tuned_threshold(make, X_tr, y_tr) if use_tuned else 0.5
        p[test] = make().fit(X_tr, y_tr).predict_proba(X.iloc[test])[:, 1]
        predicted[test] = (p[test] >= threshold).astype(int)
        thresholds.append(round(threshold, 4))
    return p, predicted, thresholds


def counts(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}


def run_account_takeover(frame, info, args, v1):
    sample = draw(frame, "in_a", args.normal, 0, SEED)          # every account takeover
    X, y = to_features(v1, sample)
    print(f"Account-takeover sample: {len(y):,} logins, {int(y.sum()):,} takeovers "
          f"(base rate {y.mean():.4f}); stratified 5-fold cross-validation")
    results, curves = {}, {}
    for name, make in MODELS.items():
        use_tuned = name != "majority_class"
        p, predicted, thresholds = cross_validated(make, X, y, use_tuned)
        results[name] = {"roc_auc": float(roc_auc_score(y, p)),
                         "average_precision": float(average_precision_score(y, p)),
                         "at_0.5": counts(y, (p >= 0.5).astype(int)),
                         "at_tuned_threshold": {**counts(y, predicted), "fold_thresholds": thresholds}}
        if name != "majority_class":
            curves[name] = (y.to_numpy(), p, results[name]["roc_auc"])
        r, t = results[name], results[name]["at_tuned_threshold"]
        print(f"\n{name}\n  AUC={r['roc_auc']:.4f}  AP={r['average_precision']:.4f}  F1@0.5={r['at_0.5']['f1']:.4f}  "
              f"F1@tuned={t['f1']:.4f}  P={t['precision']:.3f} R={t['recall']:.3f} FPR={t['false_positive_rate']:.4f}  "
              f"{t['confusion_matrix']}")
    final = MODELS["rf_v2_history_features"]().fit(X, y)
    names = final.named_steps["preprocessor"].get_feature_names_out()
    results["rf_v2_history_features"]["feature_importance"] = dict(sorted(
        zip(map(str, names), map(float, final.named_steps["classifier"].feature_importances_)),
        key=lambda item: -item[1]))
    takeovers = sample[sample["label"] == 1]
    profile = {feature: float(takeovers[feature].mean()) for feature in HISTORY}
    legit = sample[sample["label"] == 0]
    profile_legit = {feature: float(legit[feature].mean()) for feature in HISTORY}
    plot_roc(curves, REPORTS / "figures" / "rf_v2_ato_roc_curve.png",
             "Account-takeover detection: ROC from 5-fold cross-validation")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                              "pandas": pd.__version__, "platform": platform.platform()},
              "data": {**info, "path": str(args.data), "random_fraction": args.random_fraction,
                       "label": "Is Account Takeover only (attack-IP rows that are not takeovers excluded)",
                       "logins": int(len(y)), "takeovers": int(y.sum()), "legitimate": int((y == 0).sum())},
              "evaluation": "stratified 5-fold cross-validation, seed 42; thresholds tuned within each training fold",
              "feature_means": {"takeovers": profile, "legitimate": profile_legit},
              "results": results}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "rf_v2_ato_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nMean history features, takeovers vs legitimate:")
    for feature in HISTORY:
        print(f"  {feature:22} {profile[feature]:8.3f}   {profile_legit[feature]:8.3f}")
    print(f"\nSaved {REPORTS / 'rf_v2_ato_evaluation.json'} and {REPORTS / 'figures' / 'rf_v2_ato_roc_curve.png'}")


def plot_roc(curves, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"rf_v1_features": "Random Forest, login features only",
              "rf_v2_history_features": "Random Forest, with per-user history",
              "logreg_v2_history_features": "Logistic regression, with per-user history"}
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for name, (yt, p, auc) in curves.items():
        fpr, tpr, _ = roc_curve(yt, p)
        ax.plot(fpr, tpr, linewidth=2, label=f"{labels[name]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate (legitimate sign-ins flagged)")
    ax.set_ylabel("True positive rate (attack sign-ins flagged)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def evaluate(name, model, threshold, X, y):
    p = model.predict_proba(X)[:, 1]
    single = len(np.unique(y)) < 2
    return p, {"roc_auc": None if single else float(roc_auc_score(y, p)),
               "average_precision": None if single else float(average_precision_score(y, p)),
               "at_0.5": scores(y, p, 0.5), "at_tuned_threshold": scores(y, p, threshold)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--normal", type=int, default=15000)
    parser.add_argument("--attack", type=int, default=3000)
    parser.add_argument("--random-fraction", type=float, default=0.002)
    parser.add_argument("--label", choices=["attack_or_ato", "ato"], default="attack_or_ato",
                        help="attack_or_ato: attack IP or account takeover (default); ato: account takeovers only")
    args = parser.parse_args()

    v1 = load_v1()
    print(f"Streaming {args.data} once to build per-user history ...")
    frame, info = history_pass(args.data, args.random_fraction, args.label)
    if info["out_of_order_timestamps"]:
        print(f"WARNING: {info['out_of_order_timestamps']:,} rows are earlier than the row before them; "
              "history assumes the file is in time order.")
    if args.label == "ato":
        run_account_takeover(frame, info, args, v1)
        return

    sample_a = draw(frame, "in_a", args.normal, args.attack, SEED)
    X, y = to_features(v1, sample_a)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    train_ids = set(sample_a["row_id"].iloc[X_train.index])

    sample_b = draw(frame[~frame["row_id"].isin(train_ids)], "in_b", args.normal, args.attack, CHECK_SEED)
    X_b, y_b = to_features(v1, sample_b)
    print(f"Sample A: {len(y):,} logins ({int(y.sum()):,} attack); test {len(y_test):,}. "
          f"Independent sample B: {len(y_b):,} logins ({int(y_b.sum()):,} attack), none used in training.")

    results, curves = {}, {}
    for name, make in MODELS.items():
        threshold = tuned_threshold(make, X_train, y_train) if name != "majority_class" else 0.5
        model = make().fit(X_train, y_train)
        p_test, on_test = evaluate(name, model, threshold, X_test, y_test)
        _, on_b = evaluate(name, model, threshold, X_b, y_b)
        results[name] = {"held_out_test": on_test, "independent_sample_b": on_b}
        if name != "majority_class":
            curves[name] = (y_test.to_numpy(), p_test, on_test["roc_auc"])
        t = on_test["at_tuned_threshold"]
        print(f"\n{name}\n  held-out:   AUC={on_test['roc_auc']:.4f}  AP={on_test['average_precision']:.4f}  "
              f"F1@0.5={on_test['at_0.5']['f1']:.4f}  F1@tuned({t['threshold']:.3f})={t['f1']:.4f}  "
              f"P={t['precision']:.3f} R={t['recall']:.3f} FPR={t['false_positive_rate']:.3f}")
        tb = on_b["at_tuned_threshold"]
        print(f"  sample B:   AUC={on_b['roc_auc']:.4f}  F1@tuned={tb['f1']:.4f}  P={tb['precision']:.3f} "
              f"R={tb['recall']:.3f} FPR={tb['false_positive_rate']:.3f}")
        if name == "rf_v2_history_features":
            final = model
            names = model.named_steps["preprocessor"].get_feature_names_out()
            results[name]["feature_importance"] = dict(sorted(
                zip(map(str, names), map(float, model.named_steps["classifier"].feature_importances_)),
                key=lambda item: -item[1]))
            row = X_test.iloc[[0]]
            timings = []
            for _ in range(300):
                started = time.perf_counter(); model.predict_proba(row)
                timings.append((time.perf_counter() - started) * 1000)
            results[name]["latency_ms"] = {"median": float(np.median(timings)),
                                           "p95": float(np.percentile(timings, 95))}

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "categorical": BASE_CAT, "numeric": BASE_NUM + HISTORY,
                 "history_features": HISTORY}, ARTIFACT)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    labels = {"rf_v1_features": "Random Forest, login features only",
              "rf_v2_history_features": "Random Forest, with per-user history",
              "logreg_v2_history_features": "Logistic regression, with per-user history"}
    for name, (yt, p, auc) in curves.items():
        fpr, tpr, _ = roc_curve(yt, p)
        ax.plot(fpr, tpr, linewidth=2, label=f"{labels[name]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate (legitimate sign-ins flagged)")
    ax.set_ylabel("True positive rate (attack sign-ins flagged)")
    ax.set_title("Perimeter risk model with per-user history: ROC on the 20% hold-out")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "rf_v2_roc_curve.png", dpi=300)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                              "pandas": pd.__version__, "platform": platform.platform()},
              "data": {**info, "path": str(args.data), "random_fraction": args.random_fraction,
                       "label": "Is Account Takeover OR Is Attack IP",
                       "sample_a": {"seed": SEED, "logins": int(len(y)), "attack": int(y.sum()),
                                    "train": int(len(y_train)), "test": int(len(y_test))},
                       "sample_b": {"seed": CHECK_SEED, "logins": int(len(y_b)), "attack": int(y_b.sum()),
                                    "overlap_with_training": 0}},
              "features": {"login": BASE_CAT + BASE_NUM, "history": HISTORY,
                           "excluded": "exact IP address (mobile networks reassign addresses often)"},
              "model": {"type": "RandomForestClassifier", "n_estimators": 100, "max_depth": 8, "random_state": SEED},
              "threshold_selection": "F1-maximising on a 25% validation split of the training rows only",
              "results": results}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "rf_v2_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {ARTIFACT}, {REPORTS / 'rf_v2_evaluation.json'} and {REPORTS / 'figures' / 'rf_v2_roc_curve.png'}")


if __name__ == "__main__":
    main()
