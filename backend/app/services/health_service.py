"""Cycle tracking, symptom logging and PCOS risk assessment."""

from __future__ import annotations

import statistics
import uuid
from datetime import date, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.ml.explainer import RiskExplainer, get_explainer
from app.ml.predictor import RiskPredictor, get_predictor
from app.models.health import CycleLog, Prediction, SymptomLog
from app.repositories.health import (
    CycleRepository,
    PredictionRepository,
    SymptomRepository,
)
from app.repositories.user import ProfileRepository
from app.schemas.common import MEDICAL_DISCLAIMER
from app.schemas.health import (
    CycleCreate,
    CycleInsights,
    CycleUpdate,
    RiskAssessmentRequest,
    SymptomCreate,
    SymptomSummary,
)

logger = get_logger(__name__)

#: The band widely cited as a typical adult cycle length.
NORMAL_CYCLE_RANGE = (21, 35)
#: Cycles needed before a regularity verdict is meaningful. One or two cycles
#: cannot distinguish a pattern from an anomaly.
MIN_CYCLES_FOR_PATTERN = 3


class CycleService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.cycles = CycleRepository(session)
        self.profiles = ProfileRepository(session)

    async def log_cycle(self, user_id: uuid.UUID, payload: CycleCreate) -> CycleLog:
        """Record a new cycle and back-fill the previous cycle's length.

        Cycle length is only knowable once the *next* cycle begins, so logging
        a new start date is what completes the record before it.
        """
        if await self.cycles.find_by_start(user_id, payload.start_date):
            raise ConflictError(
                f"A cycle starting {payload.start_date.isoformat()} is already logged."
            )

        cycle = await self.cycles.create(
            user_id=user_id,
            start_date=payload.start_date,
            end_date=payload.end_date,
            period_length_days=payload.period_length_days,
            flow_intensity=payload.flow_intensity.value,
            pain_level=payload.pain_level,
            notes=payload.notes,
        )

        previous = await self.cycles.previous_before(user_id, payload.start_date)
        if previous is not None:
            length = (payload.start_date - previous.start_date).days
            if 10 <= length <= 180:
                previous.cycle_length_days = length
                await self.session.flush()

        await self._refresh_profile_averages(user_id)
        await self.session.commit()
        return cycle

    async def update_cycle(
        self, user_id: uuid.UUID, cycle_id: uuid.UUID, payload: CycleUpdate
    ) -> CycleLog:
        cycle = await self.cycles.get_for_user(cycle_id, user_id)
        updates = payload.model_dump(exclude_unset=True)
        if "flow_intensity" in updates and updates["flow_intensity"] is not None:
            updates["flow_intensity"] = updates["flow_intensity"].value
        await self.cycles.set_fields(cycle, **updates)
        await self.session.commit()
        return cycle

    async def delete_cycle(self, user_id: uuid.UUID, cycle_id: uuid.UUID) -> None:
        await self.cycles.delete_for_user(cycle_id, user_id)
        await self._refresh_profile_averages(user_id)
        await self.session.commit()

    async def list_cycles(self, user_id: uuid.UUID, limit: int = 24) -> list[CycleLog]:
        return await self.cycles.list_ordered(user_id, limit=limit)

    async def _refresh_profile_averages(self, user_id: uuid.UUID) -> None:
        """Keep the profile's cached averages in step with the logs.

        Denormalised deliberately: the AI prompt builder reads these on every
        chat turn, and recomputing an aggregate per message would be wasteful.
        """
        cycles = await self.cycles.list_ordered(user_id, limit=12)
        lengths = [c.cycle_length_days for c in cycles if c.cycle_length_days]
        periods = [c.period_length_days for c in cycles if c.period_length_days]

        profile = await self.profiles.get_or_create(user_id)
        if lengths:
            profile.average_cycle_length = round(statistics.mean(lengths))
        if periods:
            profile.average_period_length = round(statistics.mean(periods))
        await self.session.flush()

    async def insights(self, user_id: uuid.UUID) -> CycleInsights:
        """Regularity statistics and a next-period estimate."""
        cycles = await self.cycles.list_ordered(user_id, limit=24)
        lengths = [c.cycle_length_days for c in cycles if c.cycle_length_days]
        periods = [c.period_length_days for c in cycles if c.period_length_days]

        average = round(statistics.mean(lengths), 1) if lengths else None
        # Population standard deviation: these are all the observed cycles, not
        # a sample drawn from a larger set.
        std_dev = round(statistics.pstdev(lengths), 1) if len(lengths) >= 2 else None
        irregular = sum(
            1 for length in lengths
            if not NORMAL_CYCLE_RANGE[0] <= length <= NORMAL_CYCLE_RANGE[1]
        )

        label = self._regularity_label(lengths, std_dev)

        predicted_start: date | None = None
        fertile_window: tuple[date, date] | None = None
        days_until: int | None = None

        if cycles and average:
            last_start = cycles[0].start_date
            predicted_start = last_start + timedelta(days=round(average))
            days_until = (predicted_start - date.today()).days
            # The luteal phase is the stable part of the cycle (~14 days), so
            # ovulation is estimated backwards from the predicted next period
            # rather than forwards from the last one.
            ovulation = predicted_start - timedelta(days=14)
            fertile_window = (
                ovulation - timedelta(days=5),  # sperm viability
                ovulation + timedelta(days=1),
            )

        return CycleInsights(
            total_cycles=len(cycles),
            average_cycle_length=average,
            shortest_cycle=min(lengths) if lengths else None,
            longest_cycle=max(lengths) if lengths else None,
            cycle_length_std_dev=std_dev,
            irregular_cycle_count=irregular,
            regularity_label=label,
            predicted_next_start=predicted_start,
            predicted_fertile_window=fertile_window,
            days_until_next=days_until,
            average_period_length=(
                round(statistics.mean(periods), 1) if periods else None
            ),
        )

    @staticmethod
    def _regularity_label(lengths: list[int], std_dev: float | None) -> str:
        """Describe regularity from variance, not from any single cycle.

        Variance is the clinically interesting signal: consistently 38-day
        cycles are 'long but predictable', whereas cycles swinging between 24
        and 50 days are the pattern that matters in PCOS — even though both can
        share the same mean.
        """
        if len(lengths) < MIN_CYCLES_FOR_PATTERN:
            return "Not enough data yet — log at least three cycles."
        if std_dev is None:
            return "Not enough data yet."
        mean = statistics.mean(lengths)
        in_range = NORMAL_CYCLE_RANGE[0] <= mean <= NORMAL_CYCLE_RANGE[1]

        if std_dev <= 3 and in_range:
            return "Regular"
        if std_dev <= 3:
            return "Predictable but outside the typical 21–35 day range"
        if std_dev <= 7:
            return "Somewhat irregular"
        return "Highly irregular"


class SymptomService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.symptoms = SymptomRepository(session)

    async def log(self, user_id: uuid.UUID, payload: SymptomCreate) -> SymptomLog:
        entry = await self.symptoms.upsert(
            user_id=user_id,
            logged_on=payload.logged_on,
            symptom=payload.symptom.value,
            severity=payload.severity,
            notes=payload.notes,
        )
        await self.session.commit()
        return entry

    async def list_range(
        self, user_id: uuid.UUID, start: date, end: date
    ) -> list[SymptomLog]:
        return await self.symptoms.list_between(user_id, start, end)

    async def delete(self, user_id: uuid.UUID, symptom_id: uuid.UUID) -> None:
        await self.symptoms.delete_for_user(symptom_id, user_id)
        await self.session.commit()

    async def summary(
        self, user_id: uuid.UUID, days: int = 90
    ) -> list[SymptomSummary]:
        """Per-symptom frequency, severity and direction of travel."""
        end = date.today()
        start = end - timedelta(days=days)
        rows = await self.symptoms.frequency_summary(user_id, start, end)

        # Compare the two halves of the window to describe a trend.
        midpoint = start + timedelta(days=days // 2)
        recent = await self.symptoms.list_between(user_id, midpoint, end)
        earlier = await self.symptoms.list_between(user_id, start, midpoint)

        def mean_severity(entries: list[SymptomLog], symptom: str) -> float | None:
            values = [e.severity for e in entries if e.symptom == symptom]
            return statistics.mean(values) if values else None

        summaries: list[SymptomSummary] = []
        for row in rows:
            recent_mean = mean_severity(recent, row["symptom"])
            earlier_mean = mean_severity(earlier, row["symptom"])
            if recent_mean is None or earlier_mean is None:
                trend = "stable"
            elif recent_mean > earlier_mean + 0.4:
                trend = "increasing"
            elif recent_mean < earlier_mean - 0.4:
                trend = "decreasing"
            else:
                trend = "stable"

            summaries.append(
                SymptomSummary(
                    symptom=row["symptom"],
                    occurrences=row["occurrences"],
                    average_severity=row["average_severity"],
                    last_logged=row["last_logged"],
                    trend=trend,
                )
            )
        return summaries


class PredictionService:
    """Runs the ML model and persists the assessment with its explanation."""

    def __init__(
        self,
        session: AsyncSession,
        predictor: RiskPredictor | None = None,
        explainer: RiskExplainer | None = None,
    ) -> None:
        self.session = session
        self.predictions = PredictionRepository(session)
        self.predictor = predictor or get_predictor()
        self.explainer = explainer or get_explainer()

    async def assess(
        self, user_id: uuid.UUID, payload: RiskAssessmentRequest
    ) -> Prediction:
        features = payload.model_dump()
        features["activity_level"] = payload.activity_level.value

        result = self.predictor.predict(features)
        frame = self.predictor.build_frame(features)
        factors = self.explainer.explain(frame, result["risk_score"])

        band = Prediction.band_for(result["risk_score"])
        summary = self.explainer.summarise(result["risk_score"], band.value, factors)

        prediction = await self.predictions.create(
            user_id=user_id,
            model_name=result["model_name"],
            model_version=result["model_version"],
            features=features,
            risk_score=result["risk_score"],
            risk_band=band.value,
            confidence=result["confidence"],
            explanation=factors,
            summary=summary,
        )
        await self.session.commit()

        logger.info(
            "risk assessment stored",
            extra={
                "user_id": str(user_id),
                "risk_band": band.value,
                "model": result["model_name"],
            },
        )
        return prediction

    def response_payload(self, prediction: Prediction) -> dict:
        """Shape a stored prediction into the API response body."""
        return {
            "id": prediction.id,
            "risk_score": prediction.risk_score,
            "risk_percentage": round(prediction.risk_score * 100, 1),
            "risk_band": prediction.risk_band,
            "confidence": prediction.confidence,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "summary": prediction.summary or "",
            "top_factors": prediction.explanation,
            "recommendations": self.explainer.recommendations(
                prediction.risk_band, prediction.explanation
            ),
            "disclaimer": MEDICAL_DISCLAIMER,
            "created_at": prediction.created_at,
        }

    async def latest(self, user_id: uuid.UUID) -> Prediction:
        prediction = await self.predictions.latest(user_id)
        if prediction is None:
            raise NotFoundError(
                "No risk assessment has been completed yet for this account."
            )
        return prediction

    async def history(self, user_id: uuid.UUID, limit: int = 20) -> list[Prediction]:
        return await self.predictions.history(user_id, limit=limit)
