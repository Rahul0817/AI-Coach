"""Model loading and inference.

The trained artifact is loaded **once** at process start and held in memory.
Loading joblib per request would add tens of milliseconds and, worse, would let
a mid-deploy artifact swap serve two different model versions within a single
user session.

Every prediction carries its ``model_version``, so a stored result can always be
traced back to the exact pipeline that produced it.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.exceptions import ModelUnavailableError
from app.core.logging import get_logger
from ml.features import FEATURE_ORDER, engineer_features

logger = get_logger(__name__)

#: Hard bounds applied to every probability before it leaves the service.
#:
#: Isotonic calibration is a step function, so extreme inputs land in the
#: terminal bin and come back as exactly 0.0 or 1.0. Rendering that to a user as
#: "100% risk of PCOS" would be both statistically indefensible — no screening
#: questionnaire is ever certain — and clinically irresponsible. Clamping is a
#: deliberate product safeguard, not a numerical hack, and it is applied at the
#: single point where scores exit the model so no caller can bypass it.
MIN_RISK_SCORE = 0.02
MAX_RISK_SCORE = 0.97


class RiskPredictor:
    """Thread-safe singleton wrapper around the trained sklearn pipeline."""

    _instance: "RiskPredictor | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._artifact: dict[str, Any] | None = None
        self._explainer: Any = None
        self._explainer_lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def instance(cls) -> "RiskPredictor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def artifact_path(self) -> Path:
        base = Path(settings.ml_artifact_dir)
        if not base.is_absolute():
            # Resolve relative to backend/ so the path is identical whether the
            # server is started from the repo root or from backend/.
            base = Path(__file__).resolve().parents[2] / base
        return base / settings.ml_model_file

    def load(self) -> bool:
        """Load the artifact from disk. Returns ``True`` on success.

        A missing model is *not* fatal to the whole application: chat, tracking
        and analytics all keep working, and only the prediction endpoints
        return 503. Coupling every feature to one artifact would be a bad
        availability trade.
        """
        import joblib

        path = self.artifact_path
        if not path.exists():
            logger.warning(
                "model artifact not found; prediction endpoints will return 503",
                extra={"path": str(path)},
            )
            self._artifact = None
            return False

        try:
            artifact = joblib.load(path)
        except Exception as exc:
            logger.error("failed to load model artifact", extra={"error": str(exc)})
            self._artifact = None
            return False

        expected = artifact.get("features")
        if expected != FEATURE_ORDER:
            # Guards against deploying code and artifact from different commits.
            logger.error(
                "model feature schema does not match the running code",
                extra={"artifact_features": expected, "code_features": FEATURE_ORDER},
            )
            self._artifact = None
            return False

        self._artifact = artifact
        self._explainer = None  # invalidate any explainer built for the old model
        logger.info(
            "model loaded",
            extra={
                "model": artifact["model_name"],
                "version": artifact["model_version"],
                "roc_auc": artifact["metrics"]["roc_auc"],
            },
        )
        return True

    @property
    def is_ready(self) -> bool:
        return self._artifact is not None

    def _require(self) -> dict[str, Any]:
        if self._artifact is None:
            raise ModelUnavailableError(
                "The risk model is not loaded. Run `python -m ml.train` and "
                "restart the service."
            )
        return self._artifact

    # ------------------------------------------------------------ inference
    def build_frame(self, payload: dict) -> pd.DataFrame:
        """Convert an API payload into a one-row frame with the exact schema.

        Column *order* is enforced here because the sklearn ColumnTransformer
        selects by name but the downstream estimator is positional.
        """
        features = engineer_features(payload)
        return pd.DataFrame([[features[name] for name in FEATURE_ORDER]],
                            columns=FEATURE_ORDER)

    def predict(self, payload: dict) -> dict[str, Any]:
        """Return the calibrated probability plus provenance metadata."""
        artifact = self._require()
        frame = self.build_frame(payload)

        raw = float(artifact["model"].predict_proba(frame)[0, 1])
        proba = self.clamp(raw)
        # Confidence is distance from the decision boundary, rescaled to 0–1.
        # A 0.5 output means the model genuinely cannot tell, and the UI should
        # say so rather than presenting a coin flip as an answer. It is computed
        # from the *clamped* score so a saturated bin cannot report certainty.
        confidence = round(min(1.0, abs(proba - 0.5) * 2), 4)

        return {
            "risk_score": round(proba, 4),
            "raw_score": round(raw, 4),
            "confidence": confidence,
            "model_name": artifact["model_name"],
            "model_version": artifact["model_version"],
            "features": frame.iloc[0].to_dict(),
        }

    @staticmethod
    def clamp(score: float) -> float:
        """Constrain a probability to the reportable range. See the constants."""
        return max(MIN_RISK_SCORE, min(MAX_RISK_SCORE, score))

    def predict_batch(self, payloads: list[dict]) -> list[float]:
        """Vectorised scoring, used by tests and offline evaluation."""
        artifact = self._require()
        rows = [engineer_features(p) for p in payloads]
        frame = pd.DataFrame(
            [[r[name] for name in FEATURE_ORDER] for r in rows],
            columns=FEATURE_ORDER,
        )
        return [
            round(self.clamp(float(p)), 4)
            for p in artifact["model"].predict_proba(frame)[:, 1]
        ]

    # ------------------------------------------------------------- metadata
    def metadata(self) -> dict[str, Any]:
        artifact = self._require()
        return {
            "model_name": artifact["model_name"],
            "model_version": artifact["model_version"],
            "trained_at": artifact["trained_at"],
            "n_training_samples": artifact["n_training_samples"],
            "features": artifact["features"],
            "metrics": artifact["metrics"],
            "all_model_scores": artifact["comparison"],
        }

    def roc_curve(self) -> list[dict[str, float]]:
        return self._require()["roc_curve"]

    def confusion_matrix(self) -> list[list[int]]:
        return self._require()["confusion_matrix"]

    def global_importance(self) -> dict[str, float]:
        return self._require()["permutation_importance"]

    def feature_medians(self) -> dict[str, float]:
        return self._require()["feature_medians"]

    @property
    def pipeline(self) -> Any:
        return self._require()["model"]


def get_predictor() -> RiskPredictor:
    """FastAPI dependency returning the shared predictor."""
    return RiskPredictor.instance()
