"""Extracting durable facts from conversational text.

The product requirement is concrete: a user who says "I'm 22 years old" must
not be asked their age again three messages later, or tomorrow.

Two extraction strategies, used together:

**Deterministic patterns (this module).** Fast, free, testable, and — crucially
— *predictable*. A regex that captures "I am 22" either fires or does not, and
its behaviour can be pinned by a unit test. For the handful of high-value facts
that personalise nearly every answer (age, height, weight, diet, diagnosis
status, goals), this is the right tool.

**LLM extraction (optional, in the manager).** Handles everything the patterns
miss. It costs a call, so it runs on a schedule rather than every turn.

Why not LLM-only? Because memory is a *correctness* feature. A model that
occasionally hallucinates a user's age into long-term storage produces answers
that are confidently wrong for the rest of the account's life. Deterministic
extraction for the critical fields, with the model as a supplement, puts the
reliability where it matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Numeric bounds. Anything outside these is a misparse (a year, a phone
#: number, a typo) and is dropped rather than stored.
AGE_RANGE = (10, 100)
HEIGHT_CM_RANGE = (90.0, 250.0)
WEIGHT_KG_RANGE = (25.0, 350.0)
CYCLE_RANGE = (15, 120)


@dataclass(slots=True)
class ExtractedFacts:
    """Facts recovered from one user message."""

    values: dict[str, Any]
    #: Which pattern produced each value — useful for debugging and for
    #: explaining to a user why the assistant "knows" something.
    provenance: dict[str, str]

    def __bool__(self) -> bool:
        return bool(self.values)


# Ordered most-specific first so "I am 22 years old" is not caught by a looser
# pattern that would read "22" as something else.
_AGE_PATTERNS = [
    re.compile(r"\bi(?:'m| am)\s+(\d{1,3})\s*(?:years?\s*old|yo|yrs?)\b", re.I),
    re.compile(r"\bi(?:'m| am)\s+(\d{1,3})\b(?!\s*(?:kg|kgs|cm|lbs|pounds|%|days?))", re.I),
    re.compile(r"\bmy age is\s+(\d{1,3})\b", re.I),
    re.compile(r"\baged?\s+(\d{1,3})\b", re.I),
    re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|yo)\b", re.I),
]

_WEIGHT_PATTERNS = [
    re.compile(r"\bi weigh\s+(\d{2,3}(?:\.\d)?)\s*(kg|kgs|kilos?|lbs?|pounds?)?\b", re.I),
    re.compile(r"\bmy weight is\s+(\d{2,3}(?:\.\d)?)\s*(kg|kgs|kilos?|lbs?|pounds?)?\b", re.I),
    re.compile(r"\b(\d{2,3}(?:\.\d)?)\s*(kg|kgs|kilos?|lbs?|pounds?)\b", re.I),
]

_HEIGHT_PATTERNS = [
    re.compile(r"\bi(?:'m| am)\s+(\d{3})\s*cm\b", re.I),
    re.compile(r"\bmy height is\s+(\d{3})\s*cm\b", re.I),
    re.compile(r"\b(\d{3})\s*cm\s*tall\b", re.I),
    # Imperial, e.g. 5'4" or 5 ft 4
    re.compile(r"\b(\d)\s*(?:'|ft|feet)\s*(\d{1,2})\s*(?:\"|in|inches)?\b", re.I),
]

_CYCLE_PATTERNS = [
    re.compile(r"\bmy cycles? (?:are|is|last)\s+(?:about\s+|around\s+)?(\d{2,3})\s*days?\b", re.I),
    re.compile(r"\b(\d{2,3})[- ]day cycles?\b", re.I),
    re.compile(r"\bcycle length (?:is|of)\s+(\d{2,3})\b", re.I),
]

# The ``i'm …`` prefix may be separated from the diet word by other clauses
# ("I'm 165 cm tall and vegetarian"), so the pattern spans to the end of the
# sentence rather than requiring adjacency. ``[^.!?]*`` keeps it inside one
# clause so it cannot reach across into an unrelated sentence.
_DIET_PATTERNS = {
    "vegan": re.compile(
        r"\b(?:i(?:'m| am)\b[^.!?]*?\bvegan\b|vegan diet)", re.I
    ),
    "eggetarian": re.compile(
        r"\b(?:i(?:'m| am)\b[^.!?]*?\beggetarian\b|eggetarian diet|\beggetarian\b)", re.I
    ),
    "pescatarian": re.compile(
        r"\b(?:i(?:'m| am)\b[^.!?]*?\bpescatarian\b|\bpescatarian\b)", re.I
    ),
    "vegetarian": re.compile(
        r"\b(?:i(?:'m| am)\b[^.!?]*?\bvegetarian\b|vegetarian diet|veg only)", re.I
    ),
}

_DIAGNOSIS_PATTERNS = {
    "diagnosed": re.compile(
        r"\bi(?:'ve| have)?\s*(?:been\s+)?diagnosed with pcos\b|"
        r"\bmy (?:doctor|gynae?c?o?l?o?g?i?s?t?|gp) (?:said|says|confirmed) i have pcos\b|"
        r"\bi have pcos\b",
        re.I,
    ),
    "suspected": re.compile(
        r"\bi think i (?:might |may )?have pcos\b|\bsuspect(?:ed)? pcos\b|"
        r"\bwaiting (?:for|on) (?:a )?(?:pcos )?diagnosis\b",
        re.I,
    ),
}

_GOAL_PATTERNS = [
    re.compile(r"\bi want to\s+(.{4,80}?)(?:[.!?]|$)", re.I),
    re.compile(r"\bmy goal is (?:to\s+)?(.{4,80}?)(?:[.!?]|$)", re.I),
    re.compile(r"\bi(?:'m| am) trying to\s+(.{4,80}?)(?:[.!?]|$)", re.I),
]

_ALLERGY_PATTERN = re.compile(
    r"\bi(?:'m| am)?\s*(?:allergic to|can't eat|cannot eat|intolerant to)\s+"
    r"([a-z ,and]{3,60}?)(?:[.!?]|$)",
    re.I,
)

#: Sentences that mention a fact but negate or hypothesise it. Extracting from
#: "I'm not vegetarian" or "if I were 30" would store the opposite of the truth.
_NEGATION_RE = re.compile(
    r"\b(?:not|isn't|aren't|no longer|used to be|if i (?:were|was)|"
    r"pretend|imagine|suppose|hypothetically|my (?:friend|sister|mother|mom|cousin))\b",
    re.I,
)


def _is_negated(text: str, span: tuple[int, int]) -> bool:
    """Check for a negation cue in the clause containing ``span``.

    Scoped to the clause rather than the whole message so "I'm 24. I'm not
    vegetarian." still yields the age.
    """
    start = max(0, span[0] - 60)
    clause = text[start : span[1]]
    # Only look back to the nearest clause boundary.
    for delimiter in (". ", "! ", "? ", "; "):
        index = clause.rfind(delimiter)
        if index != -1:
            clause = clause[index + len(delimiter) :]
    return bool(_NEGATION_RE.search(clause))


def extract_facts(text: str) -> ExtractedFacts:
    """Recover durable facts from a single user message."""
    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    def record(key: str, value: Any, source: str) -> None:
        if key not in values:
            values[key] = value
            provenance[key] = source

    # ---- age ----
    for pattern in _AGE_PATTERNS:
        match = pattern.search(text)
        if match and not _is_negated(text, match.span()):
            age = int(match.group(1))
            if AGE_RANGE[0] <= age <= AGE_RANGE[1]:
                record("age", age, "age_pattern")
                break

    # ---- weight ----
    for pattern in _WEIGHT_PATTERNS:
        match = pattern.search(text)
        if match and not _is_negated(text, match.span()):
            amount = float(match.group(1))
            unit = (match.group(2) or "kg").lower()
            if unit.startswith(("lb", "pound")):
                amount = round(amount * 0.453592, 1)
            if WEIGHT_KG_RANGE[0] <= amount <= WEIGHT_KG_RANGE[1]:
                record("weight_kg", amount, "weight_pattern")
                break

    # ---- height ----
    for pattern in _HEIGHT_PATTERNS:
        match = pattern.search(text)
        if match and not _is_negated(text, match.span()):
            if len(match.groups()) == 2 and match.group(2) is not None:
                feet, inches = int(match.group(1)), int(match.group(2))
                if 3 <= feet <= 7 and 0 <= inches <= 11:
                    record("height_cm", round(feet * 30.48 + inches * 2.54, 1),
                           "height_imperial")
                    break
            else:
                centimetres = float(match.group(1))
                if HEIGHT_CM_RANGE[0] <= centimetres <= HEIGHT_CM_RANGE[1]:
                    record("height_cm", centimetres, "height_metric")
                    break

    # ---- cycle length ----
    for pattern in _CYCLE_PATTERNS:
        match = pattern.search(text)
        if match and not _is_negated(text, match.span()):
            days = int(match.group(1))
            if CYCLE_RANGE[0] <= days <= CYCLE_RANGE[1]:
                record("average_cycle_length", days, "cycle_pattern")
                break

    # ---- dietary preference ----
    for label, pattern in _DIET_PATTERNS.items():
        match = pattern.search(text)
        if match and not _is_negated(text, match.span()):
            record("dietary_preference", label, "diet_pattern")
            break

    # ---- diagnosis status ----
    for label, pattern in _DIAGNOSIS_PATTERNS.items():
        match = pattern.search(text)
        if match and not _is_negated(text, match.span()):
            record("diagnosis_status", label, "diagnosis_pattern")
            break

    # ---- allergies ----
    match = _ALLERGY_PATTERN.search(text)
    if match and not _is_negated(text, match.span()):
        items = [
            item.strip()
            for item in re.split(r",| and ", match.group(1))
            if item.strip() and len(item.strip()) > 1
        ]
        if items:
            record("allergies", items[:10], "allergy_pattern")

    # ---- goal ----
    for pattern in _GOAL_PATTERNS:
        match = pattern.search(text)
        if match and not _is_negated(text, match.span()):
            goal = " ".join(match.group(1).split())
            # Filter out conversational noise like "I want to know more".
            if len(goal) >= 8 and not goal.lower().startswith(
                ("know", "ask", "understand", "hear", "see if", "check")
            ):
                record("primary_goal", goal[:200], "goal_pattern")
                break

    return ExtractedFacts(values=values, provenance=provenance)
