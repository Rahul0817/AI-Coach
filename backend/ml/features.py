"""Canonical feature schema for the PCOS risk model.

This module is the single source of truth for *what the model sees*. The API
schema, the training pipeline, the SHAP explainer and the inference service all
import from here, so a feature can never be renamed in one place and silently
mismatched in another — the classic way ML systems break in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Ordered feature list. **Order matters**: scikit-learn pipelines are
#: positional, so a DataFrame must be built with exactly these columns in
#: exactly this sequence before calling ``predict``.
FEATURE_ORDER: list[str] = [
    "age",
    "bmi",
    "cycle_length_days",
    "cycle_irregularity",
    "weight_gain",
    "hair_growth",
    "skin_darkening",
    "hair_loss",
    "pimples",
    "fast_food",
    "exercise_hours_per_week",
    "sleep_hours",
    "stress_level",
    "family_history",
    "activity_score",
    # ---- engineered features (derived, never supplied by the client) ----
    "androgenic_symptom_count",
    "metabolic_load",
    "cycle_deviation",
    "lifestyle_score",
]

#: Features the API accepts directly; the rest are engineered from these.
RAW_INPUT_FEATURES: list[str] = FEATURE_ORDER[:15]

#: Binary yes/no indicators — used to keep the generator and validators honest.
BINARY_FEATURES: frozenset[str] = frozenset(
    {
        "cycle_irregularity",
        "weight_gain",
        "hair_growth",
        "skin_darkening",
        "hair_loss",
        "pimples",
        "fast_food",
        "family_history",
    }
)

#: Hyperandrogenism markers. Their *count* is a stronger signal than any one
#: of them alone, which is exactly what feature engineering should capture.
ANDROGENIC_MARKERS: list[str] = [
    "hair_growth",
    "skin_darkening",
    "hair_loss",
    "pimples",
]

#: Maps the API's activity enum onto an ordinal the model can use.
ACTIVITY_SCORE_MAP: dict[str, int] = {
    "sedentary": 0,
    "light": 1,
    "moderate": 2,
    "active": 3,
    "very_active": 4,
}

#: Typical mid-point of a normal menstrual cycle, used to compute deviation.
REFERENCE_CYCLE_LENGTH: int = 28


@dataclass(frozen=True)
class FeatureSpec:
    """Human-facing metadata for one feature.

    The explainer turns SHAP numbers into sentences using these, so a user sees
    "a 42-day average cycle pushed the estimate up" rather than
    ``cycle_length_days: +0.18``.
    """

    name: str
    display_name: str
    unit: str = ""
    higher_is_risk: bool = True
    increase_text: str = ""
    decrease_text: str = ""
    modifiable: bool = True


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "age": FeatureSpec(
        "age", "Age", "years", higher_is_risk=False, modifiable=False,
        increase_text="Your age group is associated with a higher reported rate of PCOS symptoms.",
        decrease_text="Your age group is associated with a lower reported rate of PCOS symptoms.",
    ),
    "bmi": FeatureSpec(
        "bmi", "Body Mass Index", "kg/m²",
        increase_text="A higher BMI is linked with insulin resistance, which commonly accompanies PCOS.",
        decrease_text="Your BMI sits in a range associated with lower metabolic strain.",
    ),
    "cycle_length_days": FeatureSpec(
        "cycle_length_days", "Average cycle length", "days",
        increase_text="Longer-than-typical cycles suggest irregular or absent ovulation.",
        decrease_text="Your cycle length falls within the commonly cited 21–35 day range.",
    ),
    "cycle_irregularity": FeatureSpec(
        "cycle_irregularity", "Irregular periods",
        increase_text="Self-reported cycle irregularity is one of the strongest signals in the model.",
        decrease_text="Regular periods lower the estimated likelihood.",
    ),
    "weight_gain": FeatureSpec(
        "weight_gain", "Recent weight gain",
        increase_text="Unexplained weight gain often accompanies the insulin resistance seen in PCOS.",
        decrease_text="Stable weight is a reassuring sign.",
    ),
    "hair_growth": FeatureSpec(
        "hair_growth", "Excess hair growth", modifiable=False,
        increase_text="Hirsutism reflects elevated androgen activity, a core PCOS feature.",
        decrease_text="No reported excess hair growth reduces the estimate.",
    ),
    "skin_darkening": FeatureSpec(
        "skin_darkening", "Skin darkening", modifiable=False,
        increase_text="Darkened skin folds (acanthosis nigricans) are a visible marker of insulin resistance.",
        decrease_text="No reported skin darkening reduces the estimate.",
    ),
    "hair_loss": FeatureSpec(
        "hair_loss", "Scalp hair thinning", modifiable=False,
        increase_text="Androgenic hair thinning points to elevated androgen levels.",
        decrease_text="No reported hair thinning reduces the estimate.",
    ),
    "pimples": FeatureSpec(
        "pimples", "Persistent acne",
        increase_text="Adult acne that resists treatment can reflect hormonal imbalance.",
        decrease_text="Clear skin reduces the estimate slightly.",
    ),
    "fast_food": FeatureSpec(
        "fast_food", "Frequent fast food",
        increase_text="A high-glycaemic diet worsens insulin resistance.",
        decrease_text="A lower-glycaemic diet supports steadier insulin levels.",
    ),
    "exercise_hours_per_week": FeatureSpec(
        "exercise_hours_per_week", "Weekly exercise", "hours", higher_is_risk=False,
        increase_text="Limited physical activity reduces insulin sensitivity.",
        decrease_text="Regular exercise improves insulin sensitivity and is protective.",
    ),
    "sleep_hours": FeatureSpec(
        "sleep_hours", "Nightly sleep", "hours", higher_is_risk=False,
        increase_text="Short sleep raises cortisol and disrupts glucose handling.",
        decrease_text="Adequate sleep supports hormonal balance.",
    ),
    "stress_level": FeatureSpec(
        "stress_level", "Stress level", "1–5",
        increase_text="Sustained stress elevates cortisol, which can disturb ovulation.",
        decrease_text="Lower stress supports a more regular cycle.",
    ),
    "family_history": FeatureSpec(
        "family_history", "Family history of PCOS", modifiable=False,
        increase_text="PCOS has a strong heritable component.",
        decrease_text="No known family history lowers the estimate.",
    ),
    "activity_score": FeatureSpec(
        "activity_score", "Overall activity level", "0–4", higher_is_risk=False,
        increase_text="A largely sedentary routine contributes to metabolic risk.",
        decrease_text="An active routine is protective.",
    ),
    "androgenic_symptom_count": FeatureSpec(
        "androgenic_symptom_count", "Number of androgenic symptoms", "0–4",
        modifiable=False,
        increase_text="Several hyperandrogenism markers appearing together is a strong combined signal.",
        decrease_text="Few androgenic markers present.",
    ),
    "metabolic_load": FeatureSpec(
        "metabolic_load", "Metabolic load index", "0–1",
        increase_text="Your combination of BMI, diet and activity indicates elevated metabolic strain.",
        decrease_text="Your metabolic indicators are in a favourable range.",
    ),
    "cycle_deviation": FeatureSpec(
        "cycle_deviation", "Deviation from a 28-day cycle", "days",
        increase_text="Your cycle length departs substantially from the typical 28-day reference.",
        decrease_text="Your cycle length sits close to the 28-day reference.",
    ),
    "lifestyle_score": FeatureSpec(
        "lifestyle_score", "Lifestyle score", "0–1", higher_is_risk=False,
        increase_text="Sleep, exercise and stress together indicate room for improvement.",
        decrease_text="Your sleep, exercise and stress balance is working in your favour.",
    ),
}


@dataclass
class EngineeredRow:
    """Container returned by :func:`engineer_features`."""

    values: dict[str, float] = field(default_factory=dict)

    def ordered(self) -> list[float]:
        return [float(self.values[name]) for name in FEATURE_ORDER]


def engineer_features(raw: dict) -> dict[str, float]:
    """Turn a validated API payload into the full model feature vector.

    Four derived features are added. Each encodes clinical structure that a
    tree model would otherwise have to rediscover from limited data:

    ``androgenic_symptom_count``
        Hyperandrogenism is diagnosed on a *pattern* of markers, so their sum
        carries more information than the individual flags.
    ``metabolic_load``
        Compresses BMI, diet quality and inactivity into one 0–1 index —
        essentially a hand-built interaction term.
    ``cycle_deviation``
        Absolute distance from a 28-day cycle. Both 19-day and 45-day cycles
        are abnormal, which a raw length cannot express monotonically.
    ``lifestyle_score``
        The modifiable side of risk (sleep, exercise, stress), which is what
        the coaching agents can actually act on.
    """
    out: dict[str, float] = {}

    for key in RAW_INPUT_FEATURES:
        if key == "activity_score":
            continue
        out[key] = float(raw[key])

    activity = raw.get("activity_level", "moderate")
    activity_value = (
        activity.value if hasattr(activity, "value") else str(activity)
    )
    out["activity_score"] = float(ACTIVITY_SCORE_MAP.get(activity_value, 2))

    # --- androgenic symptom burden (0–4) ---
    out["androgenic_symptom_count"] = float(
        sum(int(raw[m]) for m in ANDROGENIC_MARKERS)
    )

    # --- metabolic load (0–1): BMI dominates, diet and inactivity modulate ---
    bmi_component = min(1.0, max(0.0, (out["bmi"] - 18.5) / 21.5))  # 18.5→40
    diet_component = out["fast_food"]
    inactivity = 1.0 - (out["activity_score"] / 4.0)
    out["metabolic_load"] = round(
        0.55 * bmi_component + 0.20 * diet_component + 0.25 * inactivity, 4
    )

    # --- absolute deviation from a reference cycle ---
    out["cycle_deviation"] = float(
        abs(out["cycle_length_days"] - REFERENCE_CYCLE_LENGTH)
    )

    # --- lifestyle score (0–1, higher is better) ---
    sleep_component = min(1.0, out["sleep_hours"] / 8.0)
    exercise_component = min(1.0, out["exercise_hours_per_week"] / 5.0)
    stress_component = 1.0 - ((out["stress_level"] - 1) / 4.0)
    out["lifestyle_score"] = round(
        0.35 * sleep_component + 0.35 * exercise_component + 0.30 * stress_component,
        4,
    )

    return {name: out[name] for name in FEATURE_ORDER}


def engineer_frame(frame):  # type: ignore[no-untyped-def]
    """Vectorised twin of :func:`engineer_features` for training-time use.

    Training applies this to thousands of rows, where a Python loop would be
    wasteful; inference applies :func:`engineer_features` to one dict. The two
    **must** produce identical values — training/serving skew is the single
    most common cause of a model that scores well offline and fails in
    production — so ``tests/unit/test_features.py`` asserts they agree on
    randomised inputs. Treat that test as part of this function's contract.

    ``frame`` is a ``pandas.DataFrame``; the import is local so that
    ``ml.features`` stays importable in environments without pandas.
    """
    import numpy as np
    import pandas as pd

    out = pd.DataFrame(index=frame.index)

    for key in RAW_INPUT_FEATURES:
        if key == "activity_score":
            continue
        out[key] = frame[key].astype(float)

    out["activity_score"] = (
        frame["activity_level"].map(ACTIVITY_SCORE_MAP).fillna(2).astype(float)
    )

    out["androgenic_symptom_count"] = (
        frame[ANDROGENIC_MARKERS].astype(float).sum(axis=1)
    )

    bmi_component = ((out["bmi"] - 18.5) / 21.5).clip(0.0, 1.0)
    diet_component = out["fast_food"]
    inactivity = 1.0 - (out["activity_score"] / 4.0)
    out["metabolic_load"] = (
        0.55 * bmi_component + 0.20 * diet_component + 0.25 * inactivity
    ).round(4)

    out["cycle_deviation"] = (
        (out["cycle_length_days"] - REFERENCE_CYCLE_LENGTH).abs()
    )

    sleep_component = (out["sleep_hours"] / 8.0).clip(upper=1.0)
    exercise_component = (out["exercise_hours_per_week"] / 5.0).clip(upper=1.0)
    stress_component = 1.0 - ((out["stress_level"] - 1) / 4.0)
    out["lifestyle_score"] = (
        0.35 * sleep_component + 0.35 * exercise_component + 0.30 * stress_component
    ).round(4)

    # NaNs are preserved deliberately — the sklearn pipeline's imputer is the
    # one place missing values get filled, so the same rule applies at
    # training and inference time.
    return out[FEATURE_ORDER].replace([np.inf, -np.inf], np.nan)
