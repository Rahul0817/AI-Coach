"""Chat, agent-routing and RAG-citation schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.models.enums import AgentName, MessageRole
from app.schemas.common import ORMModel

#: Hard cap on a single user turn. Prevents both prompt-injection padding and
#: runaway token spend from a pasted document.
MAX_MESSAGE_CHARS = 4000


class Citation(BaseModel):
    """A retrieved knowledge chunk that grounded part of an answer."""

    title: str
    source: str
    snippet: str
    score: float = Field(..., ge=0, le=1)
    category: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: uuid.UUID | None = Field(
        None, description="Omit to start a new conversation."
    )
    #: Force a specific specialist instead of letting the router decide.
    agent: AgentName | None = None
    stream: bool = False

    @field_validator("message")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        return cleaned


class RoutingDecision(BaseModel):
    """Why the router picked the agent it picked — surfaced for transparency."""

    agent: AgentName
    confidence: float = Field(..., ge=0, le=1)
    reason: str
    matched_signals: list[str] = Field(default_factory=list)


class MessageResponseSchema(ORMModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    sequence: int
    role: MessageRole
    content: str
    agent: str | None = None
    routing_confidence: float | None = None
    sources: list[Citation] = Field(default_factory=list)
    tokens_used: int | None = None
    latency_ms: int | None = None
    model: str | None = None
    created_at: datetime


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message: MessageResponseSchema
    routing: RoutingDecision
    used_rag: bool
    #: Facts the memory layer learned from this turn (e.g. ``{"age": 22}``).
    memory_updates: dict = Field(default_factory=dict)
    suggested_followups: list[str] = Field(default_factory=list)


class ConversationResponse(ORMModel):
    id: uuid.UUID
    title: str
    is_pinned: bool
    is_archived: bool
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message_preview: str | None = None


class ConversationDetail(ConversationResponse):
    messages: list[MessageResponseSchema] = Field(default_factory=list)
    summary: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=160)
    is_pinned: bool | None = None
    is_archived: bool | None = None


class AgentInfo(BaseModel):
    """Directory entry describing one specialist, shown in the UI switcher."""

    name: AgentName
    display_name: str
    description: str
    icon: str
    colour: str
    capabilities: list[str]
    example_prompts: list[str]


class StreamChunk(BaseModel):
    """One server-sent event frame during a streaming completion."""

    type: str  # "meta" | "token" | "sources" | "done" | "error"
    content: str | None = None
    data: dict | None = None
