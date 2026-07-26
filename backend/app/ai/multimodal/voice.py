"""Voice input and output.

Architecture note — where speech processing belongs
---------------------------------------------------
The best implementation of browser voice chat is often *not* server-side. The
Web Speech API gives every modern browser free, low-latency, streaming speech
recognition and synthesis with no upload, no per-minute cost, and — importantly
for a health product — **no audio ever leaving the user's device**.

So Oviora's default voice path runs entirely in the browser (see
``frontend/src/hooks/useVoice.ts``). This module exists for the cases the
browser path cannot serve:

* Safari and older browsers with no or partial ``SpeechRecognition`` support.
* Users who want higher-accuracy transcription of accented speech, where
  Whisper is meaningfully better than the built-in engine.
* Recorded audio uploaded as a file rather than spoken live.

``/voice/capabilities`` tells the frontend which paths are available so it can
choose without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.core.exceptions import ExternalServiceError, UnsupportedMediaError
from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_AUDIO_TYPES = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/x-wav",
    "audio/mpeg", "audio/mp4", "audio/m4a", "audio/flac",
}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # matches the Whisper API limit

#: Voices exposed by the OpenAI speech endpoint.
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
DEFAULT_VOICE = "nova"


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    language: str
    duration_seconds: float | None
    confidence: float | None
    method: str


def server_stt_available() -> bool:
    return bool(settings.openai_api_key)


def server_tts_available() -> bool:
    return bool(settings.openai_api_key)


def capabilities() -> dict:
    """Report which voice paths this deployment supports.

    The frontend calls this once on load. Without it, the client would have to
    attempt a server request and interpret a failure, which is slower and
    produces a worse first-run experience.
    """
    return {
        "server_speech_to_text": server_stt_available(),
        "server_text_to_speech": server_tts_available(),
        # Always true — this is a browser capability, not a server one. The
        # client still feature-detects, but this tells it the server is happy
        # for it to be the primary path.
        "browser_speech_recommended": True,
        "voices": AVAILABLE_VOICES if server_tts_available() else [],
        "max_audio_bytes": MAX_AUDIO_BYTES,
        "supported_audio_types": sorted(SUPPORTED_AUDIO_TYPES),
        "note": (
            "Browser-native speech runs on-device: no audio is uploaded and "
            "there is no per-minute cost. Server transcription is offered as a "
            "fallback for browsers without Web Speech support."
            if not server_stt_available()
            else
            "Both browser-native and server-side speech are available. The "
            "browser path keeps audio on-device; the server path uses Whisper "
            "for higher accuracy on accented or noisy speech."
        ),
    }


async def transcribe(
    audio: bytes, filename: str, content_type: str, language: str = "en"
) -> TranscriptionResult:
    """Transcribe uploaded audio via Whisper."""
    if content_type not in SUPPORTED_AUDIO_TYPES:
        raise UnsupportedMediaError(
            f"'{content_type}' is not a supported audio format. Supported: "
            f"{', '.join(sorted(SUPPORTED_AUDIO_TYPES))}."
        )
    if len(audio) > MAX_AUDIO_BYTES:
        raise UnsupportedMediaError(
            f"Audio exceeds the {MAX_AUDIO_BYTES // (1024 * 1024)}MB limit."
        )
    if not server_stt_available():
        raise ExternalServiceError(
            "Server-side transcription is not configured on this deployment. "
            "Use your browser's built-in voice input instead — it runs "
            "on-device and keeps your audio private."
        )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=120.0,
    )
    try:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, audio, content_type),
            language=language,
            response_format="verbose_json",
        )
    except Exception as exc:
        logger.error("transcription failed", extra={"error": str(exc)})
        raise ExternalServiceError(
            "Could not transcribe that audio. Please try again."
        ) from exc

    return TranscriptionResult(
        text=getattr(response, "text", "").strip(),
        language=getattr(response, "language", language),
        duration_seconds=getattr(response, "duration", None),
        # Whisper does not return a calibrated confidence, and inventing one
        # would be misleading. The UI shows nothing rather than a fake number.
        confidence=None,
        method="whisper",
    )


async def synthesise(text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> bytes:
    """Render text to speech, returning MP3 bytes."""
    if not server_tts_available():
        raise ExternalServiceError(
            "Server-side speech synthesis is not configured on this "
            "deployment. The app uses your browser's built-in voices instead."
        )
    if voice not in AVAILABLE_VOICES:
        voice = DEFAULT_VOICE

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=120.0,
    )
    try:
        response = await client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=strip_markdown(text)[:4000],
            speed=max(0.5, min(2.0, speed)),
        )
        return response.content
    except Exception as exc:
        logger.error("speech synthesis failed", extra={"error": str(exc)})
        raise ExternalServiceError("Could not generate audio.") from exc


def strip_markdown(text: str) -> str:
    """Flatten Markdown so it is not read aloud literally.

    Without this a synthesiser articulates "asterisk asterisk Important
    asterisk asterisk", and the horizontal rule before the disclaimer becomes a
    string of dashes. Speech needs prose, not markup.
    """
    import re

    out = text
    out = re.sub(r"```[\s\S]*?```", " (code block omitted) ", out)
    out = re.sub(r"`([^`]+)`", r"\1", out)
    out = re.sub(r"^\s{0,3}#{1,6}\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"\*([^*]+)\*", r"\1", out)
    out = re.sub(r"__([^_]+)__", r"\1", out)
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)   # links → link text
    out = re.sub(r"^\s{0,3}[-*+]\s+", "", out, flags=re.MULTILINE)
    out = re.sub(r"^\s{0,3}>\s?", "", out, flags=re.MULTILINE)
    out = re.sub(r"^\s{0,3}[-*_]{3,}\s*$", "", out, flags=re.MULTILINE)
    out = re.sub(r"\|", " ", out)                         # table pipes
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
