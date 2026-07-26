"""Dataset acquisition for the PCOS risk model.

Provenance — read this before quoting any metric
------------------------------------------------
Real PCOS clinical datasets (e.g. the widely used Kerala fertility-centre
dataset) are governed by licences that forbid redistribution inside an
application repository. So this module works in two modes:

1. **Real data (preferred).** Drop a CSV at ``ml/data/pcos_dataset.csv`` with
   the columns listed in :data:`EXPECTED_COLUMNS` and it is used verbatim.
   ``DATA_SOURCE`` in the saved artifact then reads ``real``.

2. **Synthetic cohort (default).** When no CSV is present, :func:`generate_cohort`
   simulates one from an explicit causal model: a latent PCOS state is sampled
   at a realistic prevalence, and each observable is then drawn *conditional on*
   that state using published symptom-prevalence ranges. Label noise and
   confounding are injected deliberately so the problem is non-trivial.

**The honest caveat:** metrics obtained on synthetic data measure whether the
pipeline recovers the generative structure — they are *not* clinical validation
and must never be presented as such. The value of this module is that the whole
train/evaluate/explain/serve path is real and runs end to end; swapping in a
licensed dataset is a one-file change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ml.features import ACTIVITY_SCORE_MAP, RAW_INPUT_FEATURES

DATA_DIR = Path(__file__).parent / "data"
REAL_DATASET_PATH = DATA_DIR / "pcos_dataset.csv"
SYNTHETIC_DATASET_PATH = DATA_DIR / "pcos_synthetic_cohort.csv"

TARGET_COLUMN = "pcos"

EXPECTED_COLUMNS: list[str] = [*RAW_INPUT_FEATURES[:14], "activity_level", TARGET_COLUMN]

#: Roughly the mid-point of commonly cited PCOS prevalence estimates in
#: reproductive-age women (literature ranges ~8–13% depending on criteria).
BASE_PREVALENCE = 0.12

#: Deliberate case enrichment. People who install a PCOS management app are not
#: a random population sample — they are overwhelmingly women who already
#: suspect or have PCOS. Simulating that selection bias gives a ~30% positive
#: rate, which both matches the deployment population and avoids the extreme
#: imbalance that would make holdout metrics unstable at this sample size.
#: If you swap in a population-representative dataset, drop this to 1.0.
COHORT_ENRICHMENT = 2.2


@dataclass(frozen=True)
class DatasetBundle:
    """A loaded dataset plus the provenance metadata saved into the artifact."""

    frame: pd.DataFrame
    source: Literal["real", "synthetic"]
    n_rows: int
    positive_rate: float
    description: str


# --------------------------------------------------------------------------
# Conditional symptom probabilities: P(symptom = 1 | PCOS status)
# Values are drawn from the ranges reported across PCOS symptom-prevalence
# literature and rounded conservatively. They encode *association*, not
# causation, and exist only to make the synthetic cohort structurally realistic.
# --------------------------------------------------------------------------
SYMPTOM_PROBABILITIES: dict[str, tuple[float, float]] = {
    # feature:            (P | no PCOS, P | PCOS)
    "cycle_irregularity": (0.12, 0.82),
    "weight_gain": (0.22, 0.63),
    "hair_growth": (0.10, 0.68),
    "skin_darkening": (0.07, 0.42),
    "hair_loss": (0.14, 0.48),
    "pimples": (0.28, 0.61),
    "family_history": (0.09, 0.35),
    "fast_food": (0.38, 0.52),
}


def generate_cohort(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Simulate a cohort from a latent-variable causal model.

    Generative process, in order:

    1. Sample age, then the latent PCOS state with an age-modulated prevalence
       (symptomatic presentation peaks in the early-to-mid reproductive years).
    2. Sample BMI from a status-conditional distribution — the PCOS arm is
       shifted higher and has a fatter right tail.
    3. Sample each binary symptom from :data:`SYMPTOM_PROBABILITIES`.
    4. Sample lifestyle variables, letting BMI feed back into diet/exercise so
       the features are *correlated with each other*, not just with the label.
       This is what stops a model from trivially separating the classes.
    5. Sample cycle length from a mixture: regular cycles are tight around 28
       days; irregular cycles are long-tailed toward oligomenorrhoea.
    6. Flip ~4% of labels to represent diagnostic uncertainty and self-report
       error, which caps achievable accuracy at a believable level.
    """
    rng = np.random.default_rng(seed)

    # ---- 1. age and latent status -----------------------------------------
    age = np.clip(rng.normal(27, 6.5, n), 15, 48).round().astype(int)
    # Prevalence peaks around 25 and tapers at both ends of the range.
    age_modifier = 1.0 + 0.45 * np.exp(-((age - 25) ** 2) / (2 * 8.0**2)) - 0.20
    prevalence = np.clip(BASE_PREVALENCE * age_modifier * COHORT_ENRICHMENT, 0.03, 0.45)
    pcos = rng.binomial(1, prevalence).astype(int)

    # ---- 2. BMI, conditional on status ------------------------------------
    bmi = np.where(
        pcos == 1,
        rng.normal(28.6, 5.4, n),  # higher mean, wider spread
        rng.normal(23.8, 3.9, n),
    )
    bmi = np.clip(bmi, 15.0, 48.0).round(1)

    # ---- 3. binary symptoms, conditional on status ------------------------
    symptoms: dict[str, np.ndarray] = {}
    for feature, (p_neg, p_pos) in SYMPTOM_PROBABILITIES.items():
        probability = np.where(pcos == 1, p_pos, p_neg)
        symptoms[feature] = rng.binomial(1, probability).astype(int)

    # Fast food is also driven by BMI, creating an honest confounder: the model
    # must learn that diet alone is weak evidence once BMI is accounted for.
    bmi_pressure = np.clip((bmi - 22) / 20, 0, 0.35)
    symptoms["fast_food"] = np.clip(
        rng.binomial(1, np.clip(0.30 + bmi_pressure, 0, 0.95)), 0, 1
    ).astype(int)

    # ---- 4. lifestyle, partly driven by BMI -------------------------------
    exercise = np.clip(rng.gamma(2.0, 1.4, n) - (bmi - 24) * 0.08, 0.0, 20.0).round(1)
    activity_index = np.clip(np.digitize(exercise, [0.5, 2.0, 4.0, 7.0]), 0, 4).astype(
        int
    )
    reverse_map = {v: k for k, v in ACTIVITY_SCORE_MAP.items()}
    activity_level = np.array([reverse_map[int(i)] for i in activity_index])

    sleep = np.clip(rng.normal(7.1, 1.2, n) - pcos * 0.45, 3.0, 11.0).round(1)
    stress = np.clip(rng.normal(2.9 + pcos * 0.55, 1.05, n), 1, 5).round().astype(int)

    # ---- 5. cycle length: mixture conditioned on irregularity -------------
    irregular = symptoms["cycle_irregularity"]
    regular_cycles = rng.normal(28.2, 2.1, n)
    # Long-tailed toward oligomenorrhoea (>35 days), which is the PCOS pattern.
    irregular_cycles = 30.0 + rng.gamma(3.0, 5.5, n)
    cycle_length = np.where(irregular == 1, irregular_cycles, regular_cycles)
    # A minority of irregular cycles are short rather than long.
    short_mask = (irregular == 1) & (rng.random(n) < 0.18)
    cycle_length[short_mask] = rng.normal(19.0, 2.0, short_mask.sum())
    cycle_length = np.clip(cycle_length, 12, 120).round().astype(int)

    frame = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "cycle_length_days": cycle_length,
            "cycle_irregularity": symptoms["cycle_irregularity"],
            "weight_gain": symptoms["weight_gain"],
            "hair_growth": symptoms["hair_growth"],
            "skin_darkening": symptoms["skin_darkening"],
            "hair_loss": symptoms["hair_loss"],
            "pimples": symptoms["pimples"],
            "fast_food": symptoms["fast_food"],
            "exercise_hours_per_week": exercise,
            "sleep_hours": sleep,
            "stress_level": stress,
            "family_history": symptoms["family_history"],
            "activity_level": activity_level,
            TARGET_COLUMN: pcos,
        }
    )

    # ---- 6. label noise ---------------------------------------------------
    # Real PCOS diagnosis is criteria-dependent and contested; a model that hits
    # 99% on this data would be a sign of leakage, not of skill.
    flip_mask = rng.random(n) < 0.04
    frame.loc[flip_mask, TARGET_COLUMN] = 1 - frame.loc[flip_mask, TARGET_COLUMN]

    return frame


def inject_missing_values(
    frame: pd.DataFrame, rate: float = 0.03, seed: int = 42
) -> pd.DataFrame:
    """Blank a small share of continuous cells.

    Real intake forms are incomplete. Injecting missingness forces the pipeline
    to carry a real imputation step rather than one that is never exercised.
    """
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for column in ("bmi", "sleep_hours", "exercise_hours_per_week", "cycle_length_days"):
        mask = rng.random(len(out)) < rate
        out.loc[mask, column] = np.nan
    return out


def load_dataset(
    *, force_synthetic: bool = False, n: int = 6000, seed: int = 42
) -> DatasetBundle:
    """Return the training dataset, preferring a real CSV when one is present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if REAL_DATASET_PATH.exists() and not force_synthetic:
        frame = pd.read_csv(REAL_DATASET_PATH)
        missing = set(EXPECTED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{REAL_DATASET_PATH} is missing required columns: {sorted(missing)}"
            )
        frame = frame[EXPECTED_COLUMNS]
        return DatasetBundle(
            frame=frame,
            source="real",
            n_rows=len(frame),
            positive_rate=float(frame[TARGET_COLUMN].mean()),
            description=f"Real dataset loaded from {REAL_DATASET_PATH.name}",
        )

    frame = inject_missing_values(generate_cohort(n=n, seed=seed), seed=seed)
    # Persisted so training runs are reproducible and the data is inspectable.
    frame.to_csv(SYNTHETIC_DATASET_PATH, index=False)
    return DatasetBundle(
        frame=frame,
        source="synthetic",
        n_rows=len(frame),
        positive_rate=float(frame[TARGET_COLUMN].mean()),
        description=(
            "Synthetic cohort simulated from a documented latent-variable causal "
            "model with literature-informed conditional symptom prevalences, "
            "correlated lifestyle confounders and 4% label noise. NOT clinical data."
        ),
    )


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    bundle = load_dataset()
    print(f"source          : {bundle.source}")
    print(f"rows            : {bundle.n_rows}")
    print(f"positive rate   : {bundle.positive_rate:.3f}")
    print(f"missing cells   : {int(bundle.frame.isna().sum().sum())}")
    print(bundle.frame.head())
