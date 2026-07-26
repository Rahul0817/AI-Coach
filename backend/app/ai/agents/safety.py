"""Pre- and post-generation safety checks.

Prompt instructions are guidance, not guarantees. For the two failure modes
that actually matter in a health product, this module enforces the behaviour in
code where it cannot be talked out of:

**Crisis detection (pre-generation).** If a message suggests self-harm or a
medical emergency, the request never reaches the model. A templated response
directing the user to emergency services is returned instead. A model asked to
handle a crisis "carefully" will usually do so — but "usually" is the wrong
reliability target when the failure mode is someone in danger receiving a
paragraph about glycaemic load.

**Disclaimer enforcement (post-generation).** The prompt asks for a disclaimer;
this checks it is actually there and appends it if not. Cheap, and it makes the
guarantee absolute rather than probabilistic.

There is also a light prompt-injection screen. It does not attempt to be
comprehensive — that is not a solved problem — but it catches the common
"ignore your instructions" patterns and logs them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logging import get_logger
from app.schemas.common import MEDICAL_DISCLAIMER

logger = get_logger(__name__)


@dataclass(slots=True)
class SafetyVerdict:
    """Outcome of a pre-generation check."""

    blocked: bool
    category: str | None = None
    response: str | None = None
    matched: str | None = None


# --- Self-harm and suicidal ideation --------------------------------------
_SELF_HARM = re.compile(
    r"\b(?:kill myself|killing myself|end my life|ending my life|take my own life|"
    r"want to die|wanna die|better off dead|no reason to live|nothing to live for|"
    r"suicidal|suicide|self[- ]harm|harm myself|hurt myself|cut myself|"
    r"cutting myself|overdose on)\b",
    re.I,
)

# --- Acute medical emergencies --------------------------------------------
_MEDICAL_EMERGENCY = re.compile(
    r"\b(?:chest pain|can'?t breathe|cannot breathe|difficulty breathing|"
    r"severe abdominal pain|unbearable pain|passing out|passed out|fainted|"
    r"soaking (?:a |through )?(?:pad|tampon)s? (?:every|an) hour|"
    r"bleeding (?:heavily|non[- ]?stop)|haemorrhag|hemorrhag|"
    r"coughing up blood|vision going black)\b",
    re.I,
)

# --- Eating-disorder signals ----------------------------------------------
# Not blocked — blocking would be punitive and would push the user away. The
# conversation continues, but a support note is prepended and the Mental
# Wellness Coach is engaged.
_DISORDERED_EATING = re.compile(
    r"\b(?:starve myself|starving myself|purge|purging|make myself (?:sick|throw up)|"
    r"throw up after eating|binge and|laxatives to lose|not eaten (?:in|for) \d+ days?|"
    r"eat (?:under|less than) \d{3} calories)\b",
    re.I,
)

# --- Prompt injection ------------------------------------------------------
_INJECTION = re.compile(
    r"\b(?:ignore (?:all |your |the )?(?:previous |prior |above )?instructions?|"
    r"disregard (?:all |your |the )?(?:previous |prior )?(?:instructions?|rules?)|"
    r"forget (?:everything|your instructions|the rules)|"
    r"you are (?:now|no longer) (?:a|an|my)|"
    r"system prompt|reveal your (?:prompt|instructions)|"
    r"pretend (?:you are|to be) (?:a|an) (?:doctor|physician|licensed))\b",
    re.I,
)


CRISIS_RESPONSE = """
**Please reach out for support right now.**

What you have written concerns me, and this is beyond what an app should be
handling. You deserve to speak to a person who can help immediately.

- **If you are in danger right now, call your local emergency number.**
- **India:** Tele-MANAS — 14416 (free, 24/7). AASRA — +91 98204 66726.
- **United States:** call or text **988** (Suicide & Crisis Lifeline).
- **United Kingdom:** Samaritans — **116 123** (free, 24/7).
- **Elsewhere:** findahelpline.com lists free services by country.

If you can, tell someone near you how you are feeling — a friend, a family
member, or a colleague. You do not have to explain it well. Saying "I am not
okay" is enough to start.

I will be here when you want to talk about managing PCOS. Right now, please
talk to someone who can support you properly.
""".strip()


EMERGENCY_RESPONSE = """
**This needs urgent medical attention, not an app.**

The symptoms you have described can be signs of something serious that needs
assessing in person, right now.

- **Call your local emergency number, or go to the nearest emergency
  department.**
- If someone can take you, ask them — do not drive yourself if you feel faint
  or your vision is affected.
- Bring a list of any medications you take.

Please do this now. I can help you understand PCOS and build healthy routines
once you are safe, but I cannot assess an acute symptom, and it would be wrong
of me to try.
""".strip()


DISORDERED_EATING_NOTE = """
> **Before anything else:** what you have described sounds difficult, and I want
> to be careful not to give you advice that makes it harder. Restricting further
> is not the answer, and this is not a willpower problem. Please consider
> speaking to a doctor or a therapist who works with eating and body image —
> the **NEDA helpline (1-800-931-2237)** in the US, **Beat (0808 801 0677)** in
> the UK, or your GP anywhere. What follows is general information only.
""".strip()


def screen_input(text: str) -> SafetyVerdict:
    """Check a user message before it reaches the model."""
    if match := _SELF_HARM.search(text):
        logger.warning("crisis language detected", extra={"category": "self_harm"})
        return SafetyVerdict(
            blocked=True,
            category="self_harm",
            response=CRISIS_RESPONSE,
            matched=match.group(0),
        )

    if match := _MEDICAL_EMERGENCY.search(text):
        logger.warning("emergency language detected", extra={"category": "emergency"})
        return SafetyVerdict(
            blocked=True,
            category="medical_emergency",
            response=EMERGENCY_RESPONSE,
            matched=match.group(0),
        )

    if match := _DISORDERED_EATING.search(text):
        # Not blocked. Flagged so the orchestrator prepends support resources
        # and routes to the Mental Wellness Coach.
        logger.info("disordered eating signal detected")
        return SafetyVerdict(
            blocked=False,
            category="disordered_eating",
            matched=match.group(0),
        )

    if match := _INJECTION.search(text):
        logger.warning(
            "possible prompt injection", extra={"pattern": match.group(0)[:60]}
        )
        return SafetyVerdict(
            blocked=False, category="prompt_injection", matched=match.group(0)
        )

    return SafetyVerdict(blocked=False)


def enforce_disclaimer(response: str) -> str:
    """Guarantee the medical disclaimer is present exactly once."""
    if not response.strip():
        return MEDICAL_DISCLAIMER

    # Match on a distinctive fragment rather than the whole string, so a model
    # that paraphrased slightly does not cause a duplicate to be appended.
    if "consult a qualified healthcare professional" in response.lower():
        return response
    if "does not diagnose" in response.lower() and "disclaimer" in response.lower():
        return response

    return f"{response.rstrip()}\n\n---\n\n{MEDICAL_DISCLAIMER}"


def sanitise_user_input(text: str) -> str:
    """Neutralise the sentinel markers used by the prompt format.

    Without this, a user could paste ``<<<SOURCE …>>>`` into a message and
    forge a citation that the local composer would treat as retrieved
    knowledge. Stripping the delimiters closes that hole at the boundary.
    """
    return (
        text.replace("<<<SOURCE", "&lt;&lt;&lt;SOURCE")
        .replace("<<<PERSONA", "&lt;&lt;&lt;PERSONA")
        .replace("<<<PROFILE", "&lt;&lt;&lt;PROFILE")
        .replace("<<<END>>>", "&lt;&lt;&lt;END&gt;&gt;&gt;")
    )
