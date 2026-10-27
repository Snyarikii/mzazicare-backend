"""Continuous session behavioral anomaly inference."""
from pathlib import Path
from typing import Any
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.token_blacklist import blacklist_token

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
                                        db: AsyncSession, *, current_jti: str | None = None,
                                        expires_at: datetime | None = None) -> dict:
        # Reserved parameters make the service compatible with per-caregiver baseline persistence.
        del db
        vector = {name: float(metrics.get(name, 0) or 0) for name in FEATURES}
        is_critical_exfiltration = (
            vector["records_accessed_count"] >= 35
            or vector["total_payload_bytes"] >= 10 * 1024 * 1024
        )
        frame = pd.DataFrame([vector], columns=FEATURES)
        if self.model is None:
            raw_score = self._fallback_score(vector)
        else:
            try:
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
        if risk_action != "ALLOW" and not factors:
            factors.append("Session activity deviates from the learned behavioral baseline")
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
