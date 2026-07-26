"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Browser-native speech recognition and synthesis.
 *
 * This is the **primary** voice path, not a fallback. The Web Speech API runs
 * on the user's device: no audio upload, no per-minute cost, and lower latency
 * than a round trip to a transcription service. For a product handling health
 * information, "the recording never leaves your device" is a meaningful
 * property, not just an optimisation.
 *
 * The server-side Whisper endpoint exists for browsers without support (mainly
 * Firefox) — see `/api/v1/voice/capabilities`.
 */

// The API is still vendor-prefixed in Chromium and absent from the standard
// TypeScript DOM lib, so it is declared locally.
interface SpeechRecognitionEventLike extends Event {
  results: {
    length: number;
    item(index: number): { 0: { transcript: string }; isFinal: boolean };
    [index: number]: { 0: { transcript: string }; isFinal: boolean };
  };
  resultIndex: number;
}

interface SpeechRecognitionLike extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: Event & { error?: string }) => void) | null;
  onend: (() => void) | null;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

function getRecognitionConstructor(): SpeechRecognitionConstructor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function useSpeechRecognition(onFinal: (text: string) => void) {
  const [isListening, setIsListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [supported, setSupported] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  // Held in a ref so restarting recognition does not need a new callback
  // identity, which would tear down and rebuild the recogniser on every render.
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  useEffect(() => {
    setSupported(getRecognitionConstructor() !== null);
  }, []);

  const start = useCallback(() => {
    const Constructor = getRecognitionConstructor();
    if (!Constructor) {
      setError(
        "Your browser does not support voice input. Chrome, Edge and Safari do.",
      );
      return;
    }

    const recognition = new Constructor();
    recognition.continuous = true;
    // Interim results are what make dictation feel responsive — the user sees
    // words appear as they speak rather than after a pause.
    recognition.interimResults = true;
    recognition.lang = navigator.language || "en-US";

    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (!result) continue;
        const transcript = result[0].transcript;
        if (result.isFinal) finalText += transcript;
        else interimText += transcript;
      }
      setInterim(interimText);
      if (finalText.trim()) {
        onFinalRef.current(finalText.trim());
        setInterim("");
      }
    };

    recognition.onerror = (event) => {
      const code = (event as { error?: string }).error;
      if (code === "not-allowed") {
        setError("Microphone access was denied. Enable it in your browser settings.");
      } else if (code === "no-speech") {
        setError("No speech detected. Try again.");
      } else if (code !== "aborted") {
        setError("Voice input stopped unexpectedly.");
      }
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      setInterim("");
    };

    recognitionRef.current = recognition;
    setError(null);
    setIsListening(true);
    recognition.start();
  }, []);

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setIsListening(false);
    setInterim("");
  }, []);

  // Stop the microphone if the component unmounts mid-session, otherwise the
  // browser keeps the recording indicator lit after navigation.
  useEffect(() => () => recognitionRef.current?.abort(), []);

  return { isListening, interim, supported, error, start, stop };
}

/** Strip Markdown so a synthesiser reads prose, not punctuation. */
function toSpeakableText(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, " code block omitted. ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^#{1,6}\s*/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*>\s?/gm, "")
    .replace(/^\s*[-*_]{3,}\s*$/gm, "")
    .replace(/\|/g, " ")
    .replace(/\n{2,}/g, ". ")
    .trim();
}

export function useSpeechSynthesis() {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [supported, setSupported] = useState(false);

  useEffect(() => {
    setSupported(typeof window !== "undefined" && "speechSynthesis" in window);
  }, []);

  const speak = useCallback((text: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(toSpeakableText(text));
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    // Prefer a female English voice, matching the product's presentation.
    // Voice lists load asynchronously, so an empty list on first call is normal.
    const voices = window.speechSynthesis.getVoices();
    const preferred =
      voices.find((v) => /female|samantha|zira|aria/i.test(v.name) && v.lang.startsWith("en")) ??
      voices.find((v) => v.lang.startsWith("en"));
    if (preferred) utterance.voice = preferred;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  }, []);

  const stop = useCallback(() => {
    if (typeof window === "undefined") return;
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  useEffect(
    () => () => {
      if (typeof window !== "undefined") window.speechSynthesis.cancel();
    },
    [],
  );

  return { isSpeaking, supported, speak, stop };
}
