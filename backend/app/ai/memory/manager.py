"""Conversation memory: the three-tier architecture.

A language model has no memory between calls. Everything the assistant "knows"
must be reconstructed and re-sent on every single turn. Doing that naively —
replaying the entire thread — hits the context limit, and cost grows
quadratically with conversation length.

Oviora uses three tiers, each solving a different problem:

**Tier 1 — Working memory (Redis, verbatim).**
The last N turns, exactly as written. This is what makes immediate follow-ups
("and for dinner?") work. Redis because it is read on every message and must be
fast; Postgres holds the same data durably, so a cache flush costs nothing but
a rehydration.

**Tier 2 — Rolling summary (Postgres, compressed).**
Once a thread exceeds the summary trigger, older turns are compressed into a
running summary stored on the conversation row. The thread can then continue
indefinitely at bounded prompt cost. The summary is regenerated incrementally,
not from scratch, so cost stays constant per turn.

**Tier 3 — Long-term facts (Postgres, structured).**
Durable attributes — age, height, dietary preference, goals — extracted into
``Profile.ai_memory`` as structured JSON. This is what survives across
conversations and across sessions, and it is what makes "I'm 22" on Monday
inform "suggest my breakfast" on Friday.

Tier 3 is deliberately *structured* rather than a text blob. Structured facts
can be validated, displayed to the user, edited, and deleted individually —
which matters when the data is health information and the user has a right to
correct it.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm.base import GenerationConfig, LLMMessage, LLMProvider
from app.ai.memory.extractor import extract_facts
from app.core.config import settings
from app.core.logging import get_logger
from app.core.redis_client import CacheService, get_cache
from app.models.chat import Conversation
from app.models.enums import MessageRole
from app.repositories.chat import MessageRepository
from app.repositories.user import ProfileRepository

logger = get_logger(__name__)

#: Fields the extractor may write straight onto the profile, because they map
#: to real columns. Anything else lands in the free-form ``ai_memory`` blob.
PROFILE_FIELDS = {
    "height_cm",
    "weight_kg",
    "average_cycle_length",
    "dietary_preference",
    "diagnosis_status",
    "primary_goal",
    "allergies",
}


class ConversationMemory:
    """Assembles prompt context and persists what a turn taught us."""

    def __init__(
        self,
        session: AsyncSession,
        llm: LLMProvider,
        cache: CacheService | None = None,
    ) -> None:
        self.session = session
        self.llm = llm
        self.cache = cache or get_cache()
        self.messages = MessageRepository(session)
        self.profiles = ProfileRepository(session)

    # ------------------------------------------------------------ tier 1 ---
    @staticmethod
    def _cache_key(conversation_id: uuid.UUID) -> str:
        return f"memory:conversation:{conversation_id}"

    async def working_window(
        self, conversation_id: uuid.UUID, limit: int | None = None
    ) -> list[LLMMessage]:
        """Recent turns for the prompt, from Redis with a Postgres fallback."""
        window = limit or settings.memory_window_messages
        key = self._cache_key(conversation_id)

        cached = await self.cache.recent(key, window)
        if cached:
            return [
                LLMMessage(role=item["role"], content=item["content"])
                for item in cached
                if item.get("role") in {"user", "assistant"}
            ]

        # Cache miss (cold start, eviction, or a Redis restart). Rebuild from
        # the durable store and repopulate, so the miss costs one query.
        rows = await self.messages.recent_window(conversation_id, window)
        for row in rows:
            await self.cache.push_recent(
                key,
                {"role": row.role, "content": row.content},
                window,
                settings.memory_ttl_seconds,
            )
        return [
            LLMMessage(role=row.role, content=row.content)  # type: ignore[arg-type]
            for row in rows
            if row.role in {MessageRole.USER.value, MessageRole.ASSISTANT.value}
        ]

    async def remember_turn(
        self, conversation_id: uuid.UUID, role: str, content: str
    ) -> None:
        """Append a turn to working memory."""
        await self.cache.push_recent(
            self._cache_key(conversation_id),
            {"role": role, "content": content},
            settings.memory_window_messages,
            settings.memory_ttl_seconds,
        )

    async def forget_conversation(self, conversation_id: uuid.UUID) -> None:
        await self.cache.delete(self._cache_key(conversation_id))

    # ------------------------------------------------------------ tier 2 ---
    async def maybe_summarise(self, conversation: Conversation) -> str | None:
        """Compress aged-out turns into the rolling summary when due.

        Runs only when the thread has grown past the trigger and there are
        unsummarised turns beyond the verbatim window, so the cost is amortised
        across many messages rather than paid every turn.
        """
        if conversation.message_count < settings.memory_summary_trigger:
            return conversation.summary

        cutoff = conversation.message_count - settings.memory_window_messages
        if cutoff <= conversation.summarised_up_to:
            return conversation.summary

        pending = await self.messages.messages_before(conversation.id, cutoff)
        new_material = [m for m in pending if m.sequence > conversation.summarised_up_to]
        if not new_material:
            return conversation.summary

        transcript = "\n".join(f"{m.role}: {m.content[:600]}" for m in new_material)
        instruction = (
            "You are maintaining a running summary of a PCOS coaching "
            "conversation. Merge the new exchange into the existing summary. "
            "Preserve every concrete fact about the user — age, measurements, "
            "symptoms, diagnoses, preferences, goals, and any commitments they "
            "made. Drop pleasantries. Return only the updated summary, at most "
            "200 words.\n\n"
            f"EXISTING SUMMARY:\n{conversation.summary or '(none yet)'}\n\n"
            f"NEW EXCHANGE:\n{transcript}"
        )

        try:
            response = await self.llm.complete(
                [LLMMessage(role="user", content=instruction)],
                GenerationConfig(temperature=0.2, max_tokens=400),
            )
            summary = response.content.strip()
        except Exception as exc:
            logger.warning("summarisation failed", extra={"error": str(exc)})
            return conversation.summary

        if not summary:
            return conversation.summary

        conversation.summary = summary
        conversation.summarised_up_to = cutoff
        await self.session.flush()
        logger.info(
            "conversation summarised",
            extra={"conversation_id": str(conversation.id), "up_to": cutoff},
        )
        return summary

    # ------------------------------------------------------------ tier 3 ---
    async def learn_from_message(self, user_id: uuid.UUID, text: str) -> dict[str, Any]:
        """Extract durable facts from a user turn and persist them.

        Returns the facts that were newly learned, which the API echoes back so
        the UI can show a subtle "I've noted that you're 22" confirmation —
        memory the user cannot see is memory the user cannot correct.
        """
        facts = extract_facts(text)
        if not facts:
            return {}

        profile = await self.profiles.get_or_create(user_id)
        learned: dict[str, Any] = {}
        blob_updates: dict[str, Any] = {}

        for key, value in facts.values.items():
            if key in PROFILE_FIELDS:
                current = getattr(profile, key, None)
                # Only write when the value actually changes, so an unchanged
                # restatement does not churn the row on every message.
                if current != value and value is not None:
                    setattr(profile, key, value)
                    learned[key] = value
            elif key == "age":
                # Age is derived from date_of_birth and is never written
                # directly — a stored age silently rots as time passes. It goes
                # into the memory blob, where it is timestamped by context.
                if profile.date_of_birth is None:
                    blob_updates["age"] = value
                    learned["age"] = value
            else:
                blob_updates[key] = value
                learned[key] = value

        if blob_updates:
            await self.profiles.merge_ai_memory(profile, blob_updates)
        if learned:
            await self.session.flush()
            logger.info(
                "long-term memory updated",
                extra={"user_id": str(user_id), "fields": list(learned)},
            )
        return learned

    # -------------------------------------------------------- prompt context
    async def build_profile_context(self, user_id: uuid.UUID) -> dict[str, Any]:
        """Everything the prompt builder needs to personalise a response."""
        profile = await self.profiles.get_by_user(user_id)
        if profile is None:
            return {}

        remembered = dict(profile.ai_memory or {})
        # A real date of birth always beats a remembered age.
        age = profile.age if profile.date_of_birth else remembered.pop("age", None)

        return {
            "age": age,
            "bmi": profile.bmi,
            "bmi_category": profile.bmi_category,
            "weight_kg": profile.weight_kg,
            "height_cm": profile.height_cm,
            "diagnosis_status": profile.diagnosis_status,
            "dietary_preference": profile.dietary_preference,
            "activity_level": profile.activity_level,
            "primary_goal": profile.primary_goal,
            "allergies": profile.allergies,
            "medical_conditions": profile.medical_conditions,
            "medications": profile.medications,
            "average_cycle_length": profile.average_cycle_length,
            "remembered_facts": remembered,
        }

    async def clear_long_term(self, user_id: uuid.UUID) -> None:
        """Wipe the free-form memory blob at the user's request."""
        profile = await self.profiles.get_by_user(user_id)
        if profile is not None:
            profile.ai_memory = {}
            await self.session.flush()
