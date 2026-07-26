"""Unit tests for domain calculations: BMI, streaks, scoring, OCR parsing."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.ai.multimodal.nutrition_db import find_all_foods, lookup_food, score_meal
from app.ai.multimodal.ocr import parse_biomarkers
from app.models.enums import BiomarkerFlag, RiskBand
from app.models.health import Prediction
from app.models.user import Profile
from app.services.tracking_service import HabitService

pytestmark = pytest.mark.unit


class TestProfileCalculations:
    def test_bmi_computed_correctly(self) -> None:
        profile = Profile(height_cm=165.0, weight_kg=72.5)
        # 72.5 / 1.65² = 26.63
        assert profile.bmi == 26.6

    def test_bmi_none_without_measurements(self) -> None:
        assert Profile(height_cm=165.0, weight_kg=None).bmi is None
        assert Profile(height_cm=None, weight_kg=70.0).bmi is None

    def test_bmi_zero_height_does_not_divide_by_zero(self) -> None:
        assert Profile(height_cm=0, weight_kg=70.0).bmi is None

    @pytest.mark.parametrize(
        ("weight", "expected"),
        [(45.0, "underweight"), (60.0, "healthy"), (75.0, "overweight"), (95.0, "obese")],
    )
    def test_bmi_categories(self, weight: float, expected: str) -> None:
        assert Profile(height_cm=165.0, weight_kg=weight).bmi_category == expected

    def test_age_from_date_of_birth(self) -> None:
        today = date.today()
        profile = Profile(date_of_birth=today.replace(year=today.year - 24))
        assert profile.age == 24

    def test_age_accounts_for_birthday_not_yet_reached(self) -> None:
        today = date.today()
        # A birthday one day in the future means they are still 23.
        tomorrow = today + timedelta(days=1)
        profile = Profile(
            date_of_birth=date(today.year - 24, tomorrow.month, tomorrow.day)
        )
        assert profile.age == 23


class TestRiskBanding:
    @pytest.mark.parametrize(
        ("score", "band"),
        [
            (0.0, RiskBand.LOW), (0.34, RiskBand.LOW),
            (0.35, RiskBand.MODERATE), (0.64, RiskBand.MODERATE),
            (0.65, RiskBand.HIGH), (1.0, RiskBand.HIGH),
        ],
    )
    def test_band_boundaries(self, score: float, band: RiskBand) -> None:
        assert Prediction.band_for(score) is band


class TestHabitStreaks:
    def test_streak_counts_consecutive_days(self) -> None:
        today = date.today()
        days = {today - timedelta(days=n) for n in range(5)}
        assert HabitService._current_streak(days, today) == 5

    def test_yesterday_keeps_the_streak_alive(self) -> None:
        """The day is not over. Resetting a 40-day streak at midnight, before
        the user has had a chance to log, is how habit trackers lose users."""
        today = date.today()
        days = {today - timedelta(days=n) for n in range(1, 6)}
        assert HabitService._current_streak(days, today) == 5

    def test_gap_breaks_the_streak(self) -> None:
        today = date.today()
        days = {today, today - timedelta(days=1), today - timedelta(days=4)}
        assert HabitService._current_streak(days, today) == 2

    def test_stale_streak_is_zero(self) -> None:
        today = date.today()
        days = {today - timedelta(days=n) for n in range(5, 10)}
        assert HabitService._current_streak(days, today) == 0

    def test_no_entries_is_zero(self) -> None:
        assert HabitService._current_streak(set(), date.today()) == 0

    def test_longest_streak_finds_the_best_run(self) -> None:
        start = date(2026, 1, 1)
        days = {start + timedelta(days=n) for n in range(3)}
        days |= {start + timedelta(days=n) for n in range(10, 17)}
        assert HabitService._longest_streak(days) == 7


class TestNutritionScoring:
    def test_finds_multiple_foods_in_a_description(self) -> None:
        found = {f.key for f in find_all_foods("two rotis, dal and a bowl of curd")}
        assert {"roti", "dal", "curd"} <= found

    def test_specific_match_beats_generic(self) -> None:
        """'brown rice' must not resolve to plain 'rice'."""
        assert lookup_food("brown rice").key == "brown_rice"
        assert lookup_food("chicken curry").key == "chicken_curry"

    def test_overlapping_names_not_double_counted(self) -> None:
        """'chicken curry' must not also register as plain 'chicken', which
        would double the meal's protein."""
        keys = [f.key for f in find_all_foods("chicken curry with rice")]
        assert "chicken_breast" not in keys
        assert "chicken_curry" in keys

    def test_balanced_meal_outscores_refined_one(self) -> None:
        from app.ai.multimodal.nutrition_db import FOODS

        wholesome = [(FOODS["dal"], 200.0), (FOODS["roti"], 80.0),
                     (FOODS["salad"], 100.0)]
        refined = [(FOODS["white_rice"], 250.0), (FOODS["soft_drink"], 330.0)]

        good, _ = score_meal(wholesome)
        poor, _ = score_meal(refined)
        assert good > poor
        assert 0 <= poor <= 100 and 0 <= good <= 100

    def test_empty_meal_is_neutral(self) -> None:
        score, verdict = score_meal([])
        assert score == 50
        assert "not enough information" in verdict.lower()


class TestBiomarkerParsing:
    SAMPLE = """
    Test Name                Result      Unit        Biological Ref Range
    Testosterone, Total      82.4        ng/dL       15 - 70
    SHBG                     16.2        nmol/L      18 - 144
    Fasting Insulin          28.6        uIU/mL      2.6 - 24.9
    HbA1c                    5.9         %           4.0 - 5.6
    TSH                      2.1         uIU/mL      0.4 - 4.0
    Triglycerides            188         mg/dL       < 150
    """

    def test_parses_every_marker(self) -> None:
        parsed = {b.key for b in parse_biomarkers(self.SAMPLE)}
        assert {
            "total_testosterone", "shbg", "fasting_insulin",
            "hba1c", "tsh", "triglycerides",
        } <= parsed

    def test_flags_are_correct(self) -> None:
        flags = {b.key: b.flag for b in parse_biomarkers(self.SAMPLE)}
        assert flags["total_testosterone"] is BiomarkerFlag.HIGH
        assert flags["shbg"] is BiomarkerFlag.LOW
        assert flags["tsh"] is BiomarkerFlag.NORMAL

    def test_upper_bound_only_range_parsed(self) -> None:
        marker = next(
            b for b in parse_biomarkers(self.SAMPLE) if b.key == "triglycerides"
        )
        assert marker.reference_high == 150
        assert marker.flag is BiomarkerFlag.HIGH

    def test_reference_range_converts_with_the_value(self) -> None:
        """Regression: converting the value but not its range produced wrong
        flags. 1.8 nmol/L is normal against a 0.5–2.4 nmol/L range and must not
        be judged against that range after conversion to ng/dL."""
        parsed = parse_biomarkers("Total Testosterone 1.8 nmol/L 0.5 - 2.4")
        marker = parsed[0]
        assert marker.unit == "ng/dL"
        assert marker.flag is BiomarkerFlag.NORMAL
        assert marker.reference_high == pytest.approx(69.2, abs=0.5)

    def test_micro_prefix_variants_normalised(self) -> None:
        parsed = parse_biomarkers("Fasting Insulin 28.6 uIU/mL 2.6 - 24.9")
        assert parsed[0].unit == "µIU/mL"

    def test_unparseable_text_yields_nothing(self) -> None:
        assert parse_biomarkers("Dear patient, your appointment is on Tuesday.") == []

    def test_every_marker_gets_an_explanation(self) -> None:
        for marker in parse_biomarkers(self.SAMPLE):
            assert marker.interpretation
            assert "reference ranges differ" in marker.interpretation.lower()
