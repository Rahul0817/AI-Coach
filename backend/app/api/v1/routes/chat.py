"""Chat, streaming and the agent directory."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Query, Request, status
from fastapi.responses import StreamingResponse

from app.ai.agents.prompts import AGENT_DEFINITIONS
from app.api.deps import (
    Cache,
    ChatServiceDep,
    CurrentUser,
    CurrentUserId,
    DbSession,
)
from app.core.exceptions import AuthenticationError
from app.schemas.chat import (
    AgentInfo,
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationResponse,
    ConversationUpdate,
    MessageResponseSchema,
)
from app.schemas.common import ErrorResponse, MessageResponse
from app.services.auth_service import resolve_current_user
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["AI Chat"])


@router.get(
    "/agents",
    response_model=list[AgentInfo],
    summary="List the specialist agents",
)
async def list_agents() -> list[AgentInfo]:
    """Return the agent directory shown in the UI's specialist switcher.

    Unauthenticated: the landing page advertises the agents before sign-up.
    """
    return [
        AgentInfo(
            name=definition.name,
            display_name=definition.display_name,
            description=definition.description,
            icon=definition.icon,
            colour=definition.colour,
            capabilities=definition.capabilities,
            example_prompts=definition.example_prompts,
        )
        for definition in AGENT_DEFINITIONS.values()
    ]


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a message",
    responses={
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def send_message(
    payload: ChatRequest, user_id: CurrentUserId, service: ChatServiceDep
) -> ChatResponse:
    """Send one message and receive a complete grounded response.

    The reply includes which specialist answered and why, the knowledge sources
    it cited, and any facts the memory layer learned from the message — memory
    the user cannot see is memory they cannot correct.
    """
    result = await service.send(user_id, payload)
    return ChatResponse(
        conversation_id=result["conversation_id"],
        message=MessageResponseSchema.model_validate(result["message"]),
        routing=result["routing"],
        used_rag=result["used_rag"],
        memory_updates=result["memory_updates"],
        suggested_followups=result["suggested_followups"],
    )


@router.get(
    "/stream",
    summary="Stream a response (Server-Sent Events)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "SSE frames: `start`, `meta`, `sources`, `token`, `done`, `error`."
            ),
        }
    },
)
async def stream_message(
    request: Request,
    session: DbSession,
    cache: Cache,
    message: str = Query(..., min_length=1, max_length=4000),
    conversation_id: uuid.UUID | None = Query(None),
    agent: str | None = Query(None),
    token: str = Query(
        ...,
        description=(
            "Access token. Passed as a query parameter because the browser "
            "EventSource API cannot set an Authorization header."
        ),
    ),
) -> StreamingResponse:
    """Stream the assistant's reply token by token.

    **Why the token is a query parameter.** ``EventSource`` — the browser API
    for SSE — supports no custom headers, so the standard bearer scheme is not
    available. The token is therefore passed in the URL. This is a real
    trade-off: URLs land in server logs and browser history. It is mitigated by
    using only short-lived access tokens (30 minutes) here, never refresh
    tokens, and by the access log deliberately recording only the path. The
    alternative — a WebSocket, which does support an auth handshake — is the
    right answer if this ever needs to carry more than one-way text.
    """
    from app.models.enums import AgentName

    try:
        user = await resolve_current_user(session, token, cache)
    except AuthenticationError:
        # SSE clients handle a normal error response poorly, so the failure is
        # delivered as a single well-formed error frame the UI already knows
        # how to render.
        async def error_stream() -> AsyncIterator[str]:
            yield _sse({"type": "error", "content": "Your session has expired. "
                                                    "Please sign in again."})

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    forced_agent: AgentName | None = None
    if agent:
        try:
            forced_agent = AgentName(agent)
        except ValueError:
            forced_agent = None  # unknown name → fall back to routing

    chat_service = ChatService(session)
    payload = ChatRequest(
        message=message,
        conversation_id=conversation_id,
        agent=forced_agent,
        stream=True,
    )

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for frame in chat_service.stream(user.id, payload):
                # Stop generating if the browser navigated away — otherwise the
                # model keeps producing tokens (and billing) for nobody.
                if await request.is_disconnected():
                    break
                yield _sse(frame)
        except Exception as exc:  # noqa: BLE001 - must not break the stream
            yield _sse({"type": "error", "content": str(exc)})
        finally:
            yield "event: close\ndata: {}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Without this, nginx buffers the response and the stream arrives
            # all at once — which defeats the entire point of streaming.
            "X-Accel-Buffering": "no",
        },
    )


def _sse(frame: dict) -> str:
    """Encode one frame in the Server-Sent Events wire format."""
    return f"data: {json.dumps(frame, default=str)}\n\n"


# ------------------------------------------------------------ conversations
@router.get(
    "/conversations",
    response_model=list[ConversationResponse],
    summary="List conversations",
)
async def list_conversations(
    user_id: CurrentUserId,
    service: ChatServiceDep,
    include_archived: bool = Query(False),
) -> list[ConversationResponse]:
    """Sidebar list: pinned first, then most recently updated."""
    rows = await service.list_conversations(user_id, include_archived)
    return [ConversationResponse(**row) for row in rows]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get a conversation with its messages",
    responses={404: {"model": ErrorResponse}},
)
async def get_conversation(
    conversation_id: uuid.UUID, user_id: CurrentUserId, service: ChatServiceDep
) -> ConversationDetail:
    conversation = await service.get_conversation(user_id, conversation_id)
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        is_pinned=conversation.is_pinned,
        is_archived=conversation.is_archived,
        message_count=conversation.message_count,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        summary=conversation.summary,
        messages=[
            MessageResponseSchema.model_validate(m) for m in conversation.messages
        ],
    )


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Rename, pin or archive a conversation",
)
async def update_conversation(
    conversation_id: uuid.UUID,
    payload: ConversationUpdate,
    user_id: CurrentUserId,
    service: ChatServiceDep,
) -> ConversationResponse:
    conversation = await service.update_conversation(user_id, conversation_id, payload)
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        is_pinned=conversation.is_pinned,
        is_archived=conversation.is_archived,
        message_count=conversation.message_count,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.delete(
    "/conversations/{conversation_id}",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a conversation",
)
async def delete_conversation(
    conversation_id: uuid.UUID, user_id: CurrentUserId, service: ChatServiceDep
) -> MessageResponse:
    """Delete a thread and drop its working memory from the cache."""
    await service.delete_conversation(user_id, conversation_id)
    return MessageResponse(message="Conversation deleted.")
