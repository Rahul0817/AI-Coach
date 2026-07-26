"use client";

import { useCallback, useRef, useState } from "react";

import { api, tokens } from "@/lib/api";
import type { Citation, ChatMessage, StreamFrame } from "@/types/api";

/**
 * Streaming chat over Server-Sent Events.
 *
 * Implemented with `fetch` + a `ReadableStream` reader rather than the browser
 * `EventSource` API, for two reasons:
 *
 * 1. `EventSource` cannot set an Authorization header, and cannot be aborted
 *    cleanly mid-stream.
 * 2. `AbortController` lets the user stop a long generation, which stops the
 *    server generating too — the backend checks `request.is_disconnected()`
 *    and breaks out, so a cancelled response stops costing money.
 *
 * The token still travels as a query parameter because the backend's SSE route
 * is designed to be reachable by `EventSource` clients as well; see the route
 * docstring for that trade-off.
 */

export interface StreamingState {
  content: string;
  agent: string | null;
  confidence: number | null;
  sources: Citation[];
  memoryUpdates: Record<string, unknown>;
  isStreaming: boolean;
  error: string | null;
}

const EMPTY: StreamingState = {
  content: "",
  agent: null,
  confidence: null,
  sources: [],
  memoryUpdates: {},
  isStreaming: false,
  error: null,
};

export function useChatStream() {
  const [state, setState] = useState<StreamingState>(EMPTY);
  const abortRef = useRef<AbortController | null>(null);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState((previous) => ({ ...previous, isStreaming: false }));
  }, []);

  const send = useCallback(
    async (
      message: string,
      conversationId: string | null,
      agent: string | null,
      onComplete: (
        finalContent: string,
        conversationId: string,
        sources: Citation[],
        agent: string | null,
      ) => void,
    ) => {
      // Cancel any generation still in flight before starting another.
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState({ ...EMPTY, isStreaming: true });

      const params = new URLSearchParams({
        message,
        token: tokens.access ?? "",
      });
      if (conversationId) params.set("conversation_id", conversationId);
      if (agent) params.set("agent", agent);

      let resolvedConversationId = conversationId ?? "";
      let collected = "";
      let collectedSources: Citation[] = [];
      let resolvedAgent: string | null = null;

      try {
        const response = await fetch(
          `${api.baseUrl}/api/v1/chat/stream?${params.toString()}`,
          { signal: controller.signal, headers: { Accept: "text/event-stream" } },
        );

        if (!response.ok || !response.body) {
          throw new Error(`Stream failed with status ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        // SSE frames are newline-delimited but a network chunk can split one in
        // half, so partial data is buffered until a full frame arrives.
        let buffer = "";

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6).trim();
            if (!payload) continue;

            let frame: StreamFrame;
            try {
              frame = JSON.parse(payload) as StreamFrame;
            } catch {
              continue; // Ignore a malformed frame rather than killing the stream.
            }

            switch (frame.type) {
              case "start":
                resolvedConversationId = frame.data.conversation_id;
                break;
              case "meta":
                resolvedAgent = frame.data.agent;
                setState((previous) => ({
                  ...previous,
                  agent: frame.data.agent,
                  confidence: frame.data.confidence,
                  memoryUpdates: frame.data.memory_updates ?? {},
                }));
                break;
              case "sources":
                collectedSources = frame.data.sources;
                setState((previous) => ({
                  ...previous,
                  sources: frame.data.sources,
                }));
                break;
              case "token":
                collected += frame.content;
                setState((previous) => ({ ...previous, content: collected }));
                break;
              case "done":
                collected = frame.data.content || collected;
                if (frame.data.sources) collectedSources = frame.data.sources;
                setState((previous) => ({ ...previous, content: collected }));
                break;
              case "error":
                setState((previous) => ({
                  ...previous,
                  error: frame.content,
                  isStreaming: false,
                }));
                break;
            }
          }
        }

        setState((previous) => ({ ...previous, isStreaming: false }));
        onComplete(collected, resolvedConversationId, collectedSources, resolvedAgent);
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          // User-initiated stop. Keep whatever was generated so far.
          if (collected) {
            onComplete(
              collected,
              resolvedConversationId,
              collectedSources,
              resolvedAgent,
            );
          }
          return;
        }
        setState((previous) => ({
          ...previous,
          isStreaming: false,
          error:
            "The connection was interrupted. Please check your network and try again.",
        }));
      } finally {
        abortRef.current = null;
      }
    },
    [],
  );

  const reset = useCallback(() => setState(EMPTY), []);

  return { ...state, send, stop, reset };
}

/** Non-streaming send, used as a fallback and by tests. */
export async function sendMessage(
  message: string,
  conversationId: string | null,
  agent: string | null,
) {
  return api.post<{
    conversation_id: string;
    message: ChatMessage;
    routing: { agent: string; confidence: number; reason: string };
    used_rag: boolean;
    memory_updates: Record<string, unknown>;
    suggested_followups: string[];
  }>("/api/v1/chat", {
    message,
    conversation_id: conversationId,
    agent,
  });
}
