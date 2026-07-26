"""Unit tests for ML feature engineering.

The headline test here is :meth:`TestTrainServeParity.test_scalar_and_vectorised_agree`.
Training uses a vectorised pandas implementation and inference uses a scalar
one; if they ever diverge, the model scores well offline and behaves
differently in production — a failure mode that is silent, dangerous in a
health context, and notoriously hard to notice. This pins them together.
"""

from __future__ import annotations

import random

import pytest

from ml.features import (
    ACTIVITY_SCORE_MAP,
    ANDROGENIC_MARKERS,
    FEATURE_ORDER,
    FEATURE_SPECS,
    RAW_INPUT_FEATURES,
    REFERENCE_CYCLE_LENGTH,
    engineer_features,
)

pytestmark = pytest.mark.unit


def _random_payload(rng: random.Random) -> dict:
    return {
        "age": rng.randint(15, 48),
        "bmi": round(rng.uniform(15.0, 45.0), 1),
        "cycle_length_days": rng.randint(14, 90),
        "cycle_irregularity": rng.randint(0, 1),
        "weight_gain": rng.randint(0, 1),
        "hair_growth": rng.randint(0, 1),
        "skin_darkening": rng.randint(0, 1),
        "hair_loss": rng.randint(0, 1),
        "pimples": rng.randint(0, 1),
        "fast_food": rng.randint(0, 1),
        "exercise_hours_per_week": round(rng.uniform(0, 15), 1),
        "sleep_hours": round(rng.uniform(3, 11), 1),
        "stress_level": rng.randint(1, 5),
        "family_history": rng.randint(0, 1),
        "activity_level": rng.choice(list(ACTIVITY_SCORE_MAP)),
    }


class TestSchemaIntegrity:
    def test_every_feature_has_a_spec(self) -> None:
        """The explainer renders text from FEATURE_SPECS; a missing entry would
        crash an explanation rather than degrade it."""
        assert set(FEATURE_ORDER) == set(FEATURE_SPECS)

    def test_feature_order_has_no_duplicates(self) -> None:
        assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER))

    def test_raw_inputs_are_a_prefix_of_the_full_order(self) -> None:
        assert FEATURE_ORDER[: len(RAW_INPUT_FEATURES)] == RAW_INPUT_FEATURES


class TestEngineering:
    def test_output_matches_declared_order(self) -> None:
        rng = random.Random(1)
        result = engineer_features(_random_payload(rng))
        assert list(result) == FEATURE_ORDER

    def test_androgenic_count_sums_the_markers(self) -> None:
        payload = _random_payload(random.Random(2))
        payload.update(dict.fromkeys(ANDROGENIC_MARKERS, 1))
        assert engineer_features(payload)["androgenic_symptom_count"] == 4

        payload.update(dict.fromkeys(ANDROGENIC_MARKERS, 0))
        assert engineer_features(payload)["androgenic_symptom_count"] == 0

    def test_cycle_deviation_is_symmetric(self) -> None:
        """Both short and long cycles are abnormal, so deviation must be
        absolute — a raw length cannot express that monotonically."""
        base = _random_payload(random.Random(3))
        short = {**base, "cycle_length_days": REFERENCE_CYCLE_LENGTH - 9}
        long = {**base, "cycle_length_days": REFERENCE_CYCLE_LENGTH + 9}
        assert (
            engineer_features(short)["cycle_deviation"]
            == engineer_features(long)["cycle_deviation"]
            == 9
        )

    def test_metabolic_load_bounded_and_directional(self) -> None:
        base = _random_payload(random.Random(4))
        worst = {**base, "bmi": 40.0, "fast_food": 1, "activity_level": "sedentary"}
        best = {**base, "bmi": 19.0, "fast_food": 0, "activity_level": "very_active"}

        high = engineer_features(worst)["metabolic_load"]
        low = engineer_features(best)["metabolic_load"]
        assert 0.0 <= low < high <= 1.0

    def test_lifestyle_score_bounded_and_directional(self) -> None:
        """Higher is better for this one, which the explainer relies on."""
        base = _random_payload(random.Random(5))
        good = {
            **base,
            "sleep_hours": 8.0,
            "exercise_hours_per_week": 6.0,
            "stress_level": 1,
        }
        poor = {
            **base,
            "sleep_hours": 4.0,
            "exercise_hours_per_week": 0.0,
            "stress_level": 5,
        }

        assert engineer_features(good)["lifestyle_score"] == pytest.approx(1.0)
        assert engineer_features(poor)["lifestyle_score"] < 0.3

    def test_activity_level_accepts_enum_or_string(self) -> None:
        from app.models.enums import ActivityLevel

        base = _random_payload(random.Random(6))
        as_string = engineer_features({**base, "activity_level": "active"})
        as_enum = engineer_features({**base, "activity_level": ActivityLevel.ACTIVE})
        assert as_string == as_enum

    def test_unknown_activity_falls_back_to_moderate(self) -> None:
        base = _random_payload(random.Random(7))
        result = engineer_features({**base, "activity_level": "teleporting"})
        assert result["activity_score"] == ACTIVITY_SCORE_MAP["moderate"]


class TestTrainServeParity:
    def test_scalar_and_vectorised_agree(self) -> None:
        """Guards against training/serving skew.

        `engineer_features` (inference, one dict) and `engineer_frame`
        (training, a DataFrame) are separate implementations for performance
        reasons. They must produce identical numbers, or the model is scored on
        different features than it was trained on.
        """
        pd = pytest.importorskip("pandas")
        from ml.features import engineer_frame

        rng = random.Random(99)
        payloads = [_random_payload(rng) for _ in range(200)]

        vectorised = engineer_frame(pd.DataFrame(payloads))

        for index, payload in enumerate(payloads):
            scalar = engineer_features(payload)
            for feature in FEATURE_ORDER:
                assert scalar[feature] == pytest.approx(
                    float(vectorised.iloc[index][feature]), rel=1e-9, abs=1e-9
                ), f"mismatch on '{feature}' for row {index}"

    def test_vectorised_preserves_missing_values(self) -> None:
        """NaNs must survive to the pipeline's imputer, which is the single
        place missing values are handled at both train and serve time."""
        pd = pytest.importorskip("pandas")
        import numpy as np

        from ml.features import engineer_frame

        frame = pd.DataFrame([_random_payload(random.Random(11))])
        frame.loc[0, "bmi"] = np.nan
        assert bool(engineer_frame(frame)["bmi"].isna().iloc[0])
