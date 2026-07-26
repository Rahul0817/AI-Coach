"""End-to-end training pipeline for the PCOS risk model.

Run with::

    python -m ml.train                 # from the backend/ directory
    python -m ml.train --rows 12000 --seed 7

Stages
------
1. **Load** the dataset (real CSV if present, otherwise the documented
   synthetic cohort).
2. **Engineer** the derived features shared with inference.
3. **Split** stratified into train/test so the class balance is preserved.
4. **Build** a scikit-learn ``Pipeline`` per candidate model. Preprocessing
   lives *inside* the pipeline, which is what guarantees the exact same
   imputation and scaling at serving time — fitting a scaler outside the
   pipeline is the classic source of train/serve skew.
5. **Cross-validate** all candidates on the training split only.
6. **Select** the best by mean cross-validated ROC-AUC. AUC is the right
   selection metric here because the classes are imbalanced (~25% positive)
   and accuracy would reward a model that just predicts the majority class.
7. **Calibrate** the winner's probabilities — a risk score shown to a user as
   a percentage must actually mean something, and raw tree-ensemble outputs
   are typically poorly calibrated.
8. **Evaluate** once on the held-out test set: accuracy, precision, recall,
   F1, ROC-AUC, PR-AUC, confusion matrix.
9. **Persist** the fitted pipeline plus full metadata to a joblib artifact.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Allow `python ml/train.py` as well as `python -m ml.train`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.dataset import TARGET_COLUMN, load_dataset  # noqa: E402
from ml.features import FEATURE_ORDER, engineer_frame  # noqa: E402

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_FILENAME = "pcos_risk_model.joblib"
METRICS_FILENAME = "training_report.json"

MODEL_VERSION = "1.2.0"
RANDOM_STATE = 42

#: Continuous columns get median imputation + standardisation. Binary columns
#: only need most-frequent imputation — scaling a 0/1 flag adds nothing and
#: makes coefficients harder to read.
CONTINUOUS_FEATURES = [
    "age", "bmi", "cycle_length_days", "exercise_hours_per_week",
    "sleep_hours", "stress_level", "activity_score",
    "androgenic_symptom_count", "metabolic_load", "cycle_deviation",
    "lifestyle_score",
]
BINARY_COLUMNS = [f for f in FEATURE_ORDER if f not in CONTINUOUS_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Column-wise preprocessing, fitted as part of every candidate pipeline."""
    return ColumnTransformer(
        transformers=[
            (
                "continuous",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                CONTINUOUS_FEATURES,
            ),
            (
                "binary",
                SimpleImputer(strategy="most_frequent"),
                BINARY_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    """The three candidates required by the brief, each fully pipelined.

    ``class_weight='balanced'`` / ``scale_pos_weight`` matter: without them a
    model minimising log-loss on a 3:1 imbalance learns to under-predict the
    minority class, which for a *risk screening* tool is the expensive error.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        C=0.8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=12,
                        min_samples_split=8,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=3,
                        gamma=0.1,
                        reg_lambda=1.5,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Every metric the brief asks for, plus two that matter in practice.

    ``pr_auc`` is more informative than ROC-AUC under class imbalance, and
    ``brier_score`` measures probability *calibration* — whether "70% risk"
    really happens 70% of the time. A risk percentage shown to a user is
    meaningless without it.
    """
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
    }


def permutation_importance_summary(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, *, n_repeats: int = 5
) -> dict[str, float]:
    """Model-agnostic global importance via permutation.

    Preferred over ``feature_importances_`` because impurity-based importance
    is biased toward high-cardinality continuous features, and because this
    works identically for logistic regression and for the tree ensembles.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    scores = {
        feature: round(float(mean), 5)
        for feature, mean in zip(X.columns, result.importances_mean)
    }
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def _background_sample(
    X_train: pd.DataFrame, y_train: np.ndarray, *, size: int, seed: int
) -> list[dict[str, float]]:
    """Class-stratified background rows for the SHAP masker.

    Stratifying matters: a background drawn only from the majority class would
    bias every explanation toward "you look unusual", inflating attributions
    for anyone with a positive-class profile.
    """
    rng = np.random.default_rng(seed)
    per_class = max(1, size // 2)
    chosen: list[int] = []
    for label in (0, 1):
        idx = np.flatnonzero(y_train == label)
        take = min(per_class, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    sample = X_train.iloc[sorted(chosen)]
    # Imputed here so the background never contains NaNs, which SHAP's
    # permutation masker cannot handle.
    sample = sample.fillna(X_train.median(numeric_only=True))
    return sample.round(4).to_dict(orient="records")


def train(rows: int = 6000, seed: int = RANDOM_STATE, force_synthetic: bool = False) -> dict[str, Any]:
    """Run the full pipeline and write the artifact. Returns the report dict."""
    started = time.perf_counter()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- 1. load ----
    bundle = load_dataset(force_synthetic=force_synthetic, n=rows, seed=seed)
    print(f"[1/9] dataset: {bundle.source} | rows={bundle.n_rows} "
          f"| positive_rate={bundle.positive_rate:.3f}")

    # ------------------------------------------------------- 2. engineer ---
    X = engineer_frame(bundle.frame)
    y = bundle.frame[TARGET_COLUMN].astype(int).to_numpy()
    print(f"[2/9] engineered {X.shape[1]} features "
          f"({int(X.isna().sum().sum())} missing cells pending imputation)")

    # ----------------------------------------------------------- 3. split --
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    print(f"[3/9] split: train={len(X_train)} test={len(X_test)}")

    # --------------------------------------------- 4/5. build + cross-val --
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    scale_pos_weight = round(negative / max(1, positive), 3)
    models = candidate_models(scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    comparison: dict[str, dict[str, float]] = {}
    for name, pipeline in models.items():
        auc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        comparison[name] = {
            "cv_roc_auc_mean": round(float(auc_scores.mean()), 4),
            "cv_roc_auc_std": round(float(auc_scores.std()), 4),
            "cv_f1_mean": round(float(f1_scores.mean()), 4),
        }
        print(f"[5/9] {name:<22} cv_roc_auc={auc_scores.mean():.4f} "
              f"(±{auc_scores.std():.4f})  cv_f1={f1_scores.mean():.4f}")

    # ---------------------------------------------------------- 6. select --
    best_name = max(comparison, key=lambda k: comparison[k]["cv_roc_auc_mean"])
    print(f"[6/9] selected '{best_name}' on mean cross-validated ROC-AUC")

    # ------------------------------------------------------- 7. calibrate --
    # Fit the winner, then wrap it in isotonic calibration using internal CV so
    # the reported percentage is trustworthy rather than merely well-ranked.
    best_pipeline = models[best_name]
    best_pipeline.fit(X_train, y_train)
    uncalibrated_proba = best_pipeline.predict_proba(X_test)[:, 1]

    calibrated = CalibratedClassifierCV(
        estimator=models[best_name].__class__(steps=best_pipeline.steps),
        method="isotonic",
        cv=3,
    )
    calibrated.fit(X_train, y_train)
    print(f"[7/9] calibrated probabilities "
          f"(brier before={brier_score_loss(y_test, uncalibrated_proba):.4f})")

    # -------------------------------------------------------- 8. evaluate --
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate(y_test, y_pred, y_proba)
    matrix = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    # Down-sample the ROC curve so the artifact stays small but still plottable.
    step = max(1, len(fpr) // 100)
    roc_points = [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr[::step], tpr[::step])
    ]

    print(f"[8/9] holdout: acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
          f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
          f"auc={metrics['roc_auc']:.4f} brier={metrics['brier_score']:.4f}")

    importances = permutation_importance_summary(calibrated, X_test, y_test)

    # Also evaluate every candidate on the holdout for the comparison table.
    for name, pipeline in models.items():
        if name != best_name:
            pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        comparison[name].update(
            {f"test_{k}": v for k, v in evaluate(y_test, pred, proba).items()}
        )

    # --------------------------------------------------------- 9. persist --
    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "model": calibrated,
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "features": FEATURE_ORDER,
        "continuous_features": CONTINUOUS_FEATURES,
        "binary_features": BINARY_COLUMNS,
        "trained_at": trained_at,
        "metrics": metrics,
        "confusion_matrix": matrix,
        "roc_curve": roc_points,
        "comparison": comparison,
        "permutation_importance": importances,
        "n_training_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        "data_source": bundle.source,
        "data_description": bundle.description,
        "scale_pos_weight": scale_pos_weight,
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": platform.python_version(),
        "random_state": seed,
        # Training-set medians, used by the explainer as the "typical user"
        # baseline when phrasing a SHAP contribution.
        "feature_medians": {
            col: round(float(X_train[col].median()), 4) for col in FEATURE_ORDER
        },
        # A stratified slice of the training data. SHAP needs a background
        # distribution to answer "compared to whom?"; shipping it inside the
        # artifact means the explainer never has to re-read the dataset at
        # serving time, and the baseline can never drift from the fitted model.
        "background_sample": _background_sample(X_train, y_train, size=120, seed=seed),
    }

    model_path = ARTIFACT_DIR / MODEL_FILENAME
    joblib.dump(artifact, model_path, compress=3)

    # The report is a human-readable companion to the artifact; the model
    # object and the bulky background sample stay in the joblib file only.
    report = {
        k: v for k, v in artifact.items()
        if k not in {"model", "background_sample", "roc_curve"}
    }
    report["roc_curve_points"] = len(roc_points)
    report["duration_seconds"] = round(time.perf_counter() - started, 2)
    report["artifact_path"] = str(model_path)
    report["artifact_size_kb"] = round(model_path.stat().st_size / 1024, 1)
    (ARTIFACT_DIR / METRICS_FILENAME).write_text(json.dumps(report, indent=2))

    print(f"[9/9] saved {model_path} ({report['artifact_size_kb']} KB) "
          f"in {report['duration_seconds']}s")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Oviora PCOS risk model.")
    parser.add_argument("--rows", type=int, default=6000,
                        help="Synthetic cohort size (ignored when a real CSV exists).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--force-synthetic", action="store_true",
                        help="Ignore ml/data/pcos_dataset.csv even if present.")
    args = parser.parse_args()
    train(rows=args.rows, seed=args.seed, force_synthetic=args.force_synthetic)


if __name__ == "__main__":
    main()
