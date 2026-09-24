"""Continuous session behavioral anomaly inference."""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

FEATURES = ["total_logons", "distinct_devices", "activity_span_hours", "off_hours_ratio",
            "records_accessed_count", "total_payload_bytes"]
MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "behavioral_anomaly_model.joblib"


class BehavioralAnomalyEngine:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model: Any = None
        if model_path.is_file():
            try:
                artifact = joblib.load(model_path)
                self.model = artifact.get("pipeline") if isinstance(artifact, dict) else artifact
            except Exception as exc:
                print(f"[Behavioral Engine] Failed to load model {model_path}: {exc}; using heuristic fallback.")
        else:
            print(f"[Behavioral Engine] Model artifact absent at {model_path}; using heuristic fallback.")

    async def evaluate_session_activity(self, user_id: Any, metrics: dict,
                                        db: AsyncSession) -> dict:
        # Reserved parameters make the service compatible with per-caregiver baseline persistence.
        del user_id, db
        vector = {name: float(metrics.get(name, 0) or 0) for name in FEATURES}
        frame = pd.DataFrame([vector], columns=FEATURES)
        if self.model is None:
            score = self._fallback_score(vector)
            is_anomaly = score >= 0.65
        else:
            try:
                decision = float(self.model.decision_function(frame)[0])
                prediction = int(self.model.predict(frame)[0])
                threshold = float(getattr(self.model.named_steps.get("isolation_forest"), "offset_", 0.0)) \
                    if hasattr(self.model, "named_steps") else 0.0
                # Calibrate the decision boundary to 0.65 and smoothly map distance to [0, 1].
                logit_065 = np.log(0.65 / 0.35)
                score = float(1.0 / (1.0 + np.exp(-np.clip(logit_065 + (threshold - decision) * 12.0, -30, 30))))
                is_anomaly = prediction == -1 or score >= 0.65
                if prediction == -1:
                    score = max(score, 0.65)
            except Exception as exc:
                print(f"[Behavioral Engine] Inference failed: {exc}; using heuristic fallback.")
                score = self._fallback_score(vector)
                is_anomaly = score >= 0.65

        factors: list[str] = []
        if vector["records_accessed_count"] > 100:
            factors.append("Abnormal data query volume (>100 records)")
        if vector["off_hours_ratio"] >= 0.5:
            factors.append("Off-hours EHR query velocity")
        if vector["distinct_devices"] >= 4:
            factors.append("Multi-device hopping detected")
        if vector["total_payload_bytes"] >= 10_000_000:
            factors.append("Unusually large data payload volume")
        if vector["activity_span_hours"] >= 16:
            factors.append("Unusually long activity span")
        if is_anomaly and not factors:
            factors.append("Session activity deviates from the learned behavioral baseline")
        score = float(np.clip(score, 0.0, 1.0))
        return {
            "anomaly_score": round(score, 4),
            "is_anomaly": bool(is_anomaly),
            "session_revocation_recommended": bool(score >= 0.75),
            "risk_factors": factors,
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
