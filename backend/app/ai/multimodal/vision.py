"""Food image analysis.

Two providers behind one interface
----------------------------------
``OpenAIVisionProvider``
    Sends the image to a multimodal model, which returns a structured list of
    identified foods and portion estimates. This is real image understanding.

``DescriptionVisionProvider``
    The no-credentials fallback. **It does not look at pixels**, and it does not
    pretend to: it works from the caption or filename the user supplies and
    matches that against the local food database.

That distinction is stated plainly in the response (``analysis_method``) and in
the UI, because a fabricated "I can see grilled chicken" from a system with no
vision capability would be a lie to the user about their own health data. A
fallback that says "tell me what's on the plate and I'll do the nutrition
maths" is honest and still genuinely useful — the nutrition database, portion
maths, PCOS scoring and swap suggestions are identical on both paths, and that
is where most of the value actually is.
"""

from __future__ import annotations

import base64
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.ai.multimodal.nutrition_db import (
    FoodItem,
    find_all_foods,
    healthier_alternatives,
    lookup_food,
    score_meal,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import MealType

logger = get_logger(__name__)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@dataclass(slots=True)
class DetectedFood:
    name: str
    confidence: float
    grams: float
    item: FoodItem | None = None


@dataclass(slots=True)
class VisionAnalysis:
    detected: list[DetectedFood]
    analysis_method: str
    notes: list[str] = field(default_factory=list)


class VisionProvider(ABC):
    name = "base"

    @abstractmethod
    async def analyse(
        self, image_bytes: bytes, content_type: str, caption: str | None
    ) -> VisionAnalysis: ...


class DescriptionVisionProvider(VisionProvider):
    """Text-driven fallback. Explicit about not seeing the image."""

    name = "description"

    async def analyse(
        self, image_bytes: bytes, content_type: str, caption: str | None
    ) -> VisionAnalysis:
        notes = [
            "This deployment has no vision model configured, so the photo "
            "itself was not analysed. The nutrition breakdown below is based "
            "on the description provided."
        ]
        if not caption:
            return VisionAnalysis(
                detected=[],
                analysis_method="description_no_input",
                notes=[
                    *notes,
                    "No description was given. Tell me what is on the plate — "
                    "for example 'two rotis, dal and a bowl of curd' — and I "
                    "will work out the nutrition.",
                ],
            )

        matches = find_all_foods(caption)
        detected = [
            DetectedFood(
                name=item.name,
                # Confidence reflects a text match, not image recognition.
                confidence=0.75,
                grams=_infer_portion(caption, item),
                item=item,
            )
            for item in matches
        ]
        if not detected:
            notes.append(
                "None of the foods in that description matched the nutrition "
                "database. Try naming the main components individually."
            )
        return VisionAnalysis(
            detected=detected, analysis_method="description_match", notes=notes
        )


class OpenAIVisionProvider(VisionProvider):
    """Multimodal analysis via the OpenAI vision-capable chat endpoint."""

    name = "openai_vision"

    PROMPT = (
        "You are a nutrition analyst. Identify every distinct food in this "
        "photograph and estimate the portion in grams.\n\n"
        "Respond with ONLY a JSON array, no prose, no code fences. Each element "
        'must be: {"name": "<food>", "grams": <number>, "confidence": <0-1>}\n'
        "Use simple food names (for example 'white rice', 'dal', 'roti', "
        "'grilled chicken'). If the image contains no food, return []."
    )

    async def analyse(
        self, image_bytes: bytes, content_type: str, caption: str | None
    ) -> VisionAnalysis:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=60.0,
        )
        encoded = base64.b64encode(image_bytes).decode("ascii")
        user_content: list[dict] = [
            {"type": "text", "text": self.PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{content_type};base64,{encoded}"},
            },
        ]
        if caption:
            user_content.append({"type": "text", "text": f"The user adds: {caption}"})

        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.1,
                max_tokens=800,
            )
            raw = response.choices[0].message.content or "[]"
        except Exception as exc:
            logger.error("vision analysis failed", extra={"error": str(exc)})
            # Fall back rather than fail the request outright — a text-based
            # answer beats an error page.
            return await DescriptionVisionProvider().analyse(
                image_bytes, content_type, caption
            )

        parsed = _parse_json_array(raw)
        detected: list[DetectedFood] = []
        for entry in parsed:
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            item = lookup_food(name)
            detected.append(
                DetectedFood(
                    name=item.name if item else name.title(),
                    confidence=float(entry.get("confidence", 0.7)),
                    grams=float(entry.get("grams", item.typical_grams if item else 100)),
                    item=item,
                )
            )

        notes: list[str] = []
        unmatched = [d.name for d in detected if d.item is None]
        if unmatched:
            notes.append(
                "These items were identified but are not in the nutrition "
                f"database, so they are excluded from the totals: "
                f"{', '.join(unmatched)}."
            )
        return VisionAnalysis(
            detected=detected, analysis_method="vision_model", notes=notes
        )


_PORTION_WORDS = {
    "half": 0.5,
    "quarter": 0.25,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "a": 1,
    "an": 1,
    "single": 1,
    "double": 2,
    "couple": 2,
    "few": 3,
}
_GRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(g|gram|grams|gm|ml)\b", re.I)


def _infer_portion(caption: str, item: FoodItem) -> float:
    """Estimate the portion mass from wording, falling back to a typical serving."""
    lowered = caption.lower()

    if match := _GRAM_RE.search(lowered):
        return float(match.group(1))

    # Look for a count immediately before the food's name: "two rotis".
    for alias in [item.name.lower(), *[a.lower() for a in item.aliases]]:
        index = lowered.find(alias)
        if index <= 0:
            continue
        preceding = lowered[max(0, index - 24) : index].split()
        if not preceding:
            continue
        last = preceding[-1].strip(",.")
        if last.isdigit():
            return item.typical_grams * int(last)
        if last in _PORTION_WORDS:
            return item.typical_grams * _PORTION_WORDS[last]

    return item.typical_grams


def _parse_json_array(raw: str) -> list[dict]:
    """Parse a JSON array from a model response, tolerating code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        # Last resort: grab the outermost bracketed span.
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                pass
    logger.warning("could not parse vision response as JSON")
    return []


def get_vision_provider() -> VisionProvider:
    """Select the vision backend from configuration."""
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAIVisionProvider()
    return DescriptionVisionProvider()


def build_analysis_payload(analysis: VisionAnalysis, image_url: str | None) -> dict:
    """Turn a :class:`VisionAnalysis` into the API response body.

    All nutrition maths, scoring and swap logic lives here rather than in the
    providers, so both the vision and description paths produce identical
    output shapes and identical numbers for the same identified foods.
    """
    from app.schemas.common import MEDICAL_DISCLAIMER

    known = [d for d in analysis.detected if d.item is not None]
    weighted = [(d.item, d.grams) for d in known if d.item]

    items: list[dict] = []
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fibre": 0.0}

    for detected in known:
        item = detected.item
        assert item is not None  # guaranteed by the filter above
        factor = detected.grams / 100.0
        entry = {
            "name": item.name,
            "confidence": round(detected.confidence, 2),
            "estimated_grams": round(detected.grams, 1),
            "calories": round(item.calories * factor, 1),
            "protein_g": round(item.protein_g * factor, 1),
            "carbs_g": round(item.carbs_g * factor, 1),
            "fat_g": round(item.fat_g * factor, 1),
            "fibre_g": round(item.fibre_g * factor, 1),
            "glycemic_index": item.glycemic_index,
        }
        items.append(entry)
        totals["calories"] += entry["calories"]
        totals["protein"] += entry["protein_g"]
        totals["carbs"] += entry["carbs_g"]
        totals["fat"] += entry["fat_g"]
        totals["fibre"] += entry["fibre_g"]

    score, verdict = score_meal(weighted)

    # Carb-weighted mean GI: a spoon of a high-GI condiment should not drag the
    # average of a large low-GI meal upward.
    carb_weight = sum(i["carbs_g"] for i in items)
    if carb_weight > 0:
        average_gi = round(
            sum(i["glycemic_index"] * i["carbs_g"] for i in items) / carb_weight
        )
    else:
        average_gi = 0

    swaps: list[dict] = []
    for detected in sorted(known, key=lambda d: d.item.pcos_score if d.item else 100):
        item = detected.item
        assert item is not None
        for alternative in healthier_alternatives(item)[:1]:
            factor = detected.grams / 100.0
            swaps.append(
                {
                    "replace": item.name,
                    "with_alternative": alternative.name,
                    "reason": (
                        f"{alternative.name} has a glycaemic index of "
                        f"{alternative.glycemic_index} versus "
                        f"{item.glycemic_index}, and "
                        f"{alternative.fibre_g:.1f}g of fibre per 100g versus "
                        f"{item.fibre_g:.1f}g — a gentler blood-sugar response."
                    ),
                    "calories_saved": round(
                        (item.calories - alternative.calories) * factor, 1
                    ),
                }
            )
        if len(swaps) >= 3:
            break

    assessment_parts = [verdict]
    if totals["protein"] < 15:
        assessment_parts.append(
            f"Protein is on the low side at {totals['protein']:.0f}g. Aim for "
            f"20–30g per meal to steady blood sugar and stay full longer."
        )
    if totals["fibre"] < 5 and totals["carbs"] > 25:
        assessment_parts.append(
            "There is little fibre here relative to the carbohydrate. Adding a "
            "vegetable or a legume side would flatten the glucose response."
        )
    if average_gi > 65:
        assessment_parts.append(
            f"The carbohydrate-weighted glycaemic index is {average_gi}, which "
            f"is high. Pairing these carbs with protein or fat lowers it."
        )
    assessment_parts.extend(analysis.notes)

    return {
        "image_url": image_url,
        "detected_items": items,
        "total_calories": round(totals["calories"], 1),
        "total_protein_g": round(totals["protein"], 1),
        "total_carbs_g": round(totals["carbs"], 1),
        "total_fat_g": round(totals["fat"], 1),
        "total_fibre_g": round(totals["fibre"], 1),
        "average_glycemic_index": average_gi,
        "pcos_score": score,
        "pcos_verdict": verdict,
        "assessment": " ".join(assessment_parts),
        "healthier_swaps": swaps,
        "suggested_meal_type": _suggest_meal_type(totals["calories"]).value,
        "analysis_method": analysis.analysis_method,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


def _suggest_meal_type(calories: float) -> MealType:
    """Guess which meal slot this belongs in, from its energy content."""
    if calories < 200:
        return MealType.SNACK
    if calories < 450:
        return MealType.BREAKFAST
    if calories < 750:
        return MealType.LUNCH
    return MealType.DINNER
