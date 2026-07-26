"use client";

import { AlertCircle, ArrowUp, Mic, MicOff, Sparkles, Square } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { ChatBubble, TypingIndicator } from "@/components/chat/ChatMessage";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { Badge, Button, Card } from "@/components/ui/primitives";
import { useChatStream } from "@/hooks/useChat";
import { useSpeechRecognition } from "@/hooks/useVoice";
import type { Citation } from "@/types/api";

interface LocalMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  agent?: string | null;
  confidence?: number | null;
  sources?: Citation[];
}

const STARTERS = [
  "What actually causes PCOS?",
  "What should I eat for breakfast?",
  "My cycles are 45 days apart — is that normal?",
  "Build me a 4-day workout plan",
];

const MAX_CHARS = 4000;

export default function ChatPage() {
  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [input, setInput] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);

  const stream = useChatStream();
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const voice = useSpeechRecognition(
    useCallback((text: string) => {
      setInput((previous) => (previous ? `${previous} ${text}` : text));
    }, []),
  );

  // Follow the conversation as it grows. `behavior: smooth` on every token
  // would fight the user if they scrolled up, so it only runs when a message
  // is added or streaming ends.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, stream.isStreaming]);

  // Grow the textarea with its content, up to a cap.
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [input]);

  const submit = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || stream.isStreaming) return;

      setMessages((previous) => [
        ...previous,
        { id: `u-${Date.now()}`, role: "user", content: trimmed },
      ]);
      setInput("");
      if (voice.isListening) voice.stop();

      void stream.send(
        trimmed,
        conversationId,
        null,
        (finalContent, newConversationId, sources, agent) => {
          setConversationId(newConversationId || null);
          setMessages((previous) => [
            ...previous,
            {
              id: `a-${Date.now()}`,
              role: "assistant",
              content: finalContent,
              agent,
              sources,
            },
          ]);
          stream.reset();
        },
      );
    },
    [conversationId, stream, voice],
  );

  const showStreamingBubble = stream.isStreaming && stream.content.length > 0;
  const showTypingIndicator = stream.isStreaming && stream.content.length === 0;
  const isEmpty = messages.length === 0 && !stream.isStreaming;

  return (
    <div className="flex h-dvh flex-col">
      <header className="shrink-0 border-b border-border/60 bg-white/60 backdrop-blur-xl dark:bg-background/70">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
          <Link href="/" className="flex items-center gap-2 focus-ring rounded-lg">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-brand">
              <Sparkles className="h-4 w-4 text-white" />
            </span>
            <span className="font-semibold">Oviora</span>
          </Link>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">
                Dashboard
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <main
        id="main"
        className="scrollbar-slim flex-1 overflow-y-auto"
        aria-live="polite"
        aria-busy={stream.isStreaming}
      >
        <div className="mx-auto max-w-3xl px-4 py-6">
          {isEmpty ? (
            <div className="flex flex-col items-center pt-10 text-center sm:pt-16">
              <span className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-brand">
                <Sparkles className="h-7 w-7 text-white" />
              </span>
              <h1 className="text-2xl font-semibold sm:text-3xl">
                How can I help with your PCOS today?
              </h1>
              <p className="mt-2 max-w-md text-sm text-muted-foreground">
                Ask anything. Your question is routed to the right specialist,
                and answers are grounded in curated guidance.
              </p>

              <div className="mt-8 grid w-full gap-2.5 sm:grid-cols-2">
                {STARTERS.map((starter) => (
                  <button
                    key={starter}
                    onClick={() => submit(starter)}
                    className="glass-card focus-ring rounded-xl px-4 py-3 text-left text-sm transition-shadow hover:shadow-glass-lg"
                  >
                    {starter}
                  </button>
                ))}
              </div>

              <Card className="mt-8 border-amber-300/50 bg-amber-50/70 text-left dark:border-amber-400/20 dark:bg-amber-500/5">
                <div className="flex gap-2.5">
                  <AlertCircle className="h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
                  <p className="text-xs leading-relaxed text-amber-800/90 dark:text-amber-200/80">
                    Oviora provides education and lifestyle guidance only. It
                    does not diagnose conditions or replace your doctor. If
                    something is urgent, contact a healthcare professional or
                    emergency services.
                  </p>
                </div>
              </Card>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message) => (
                <ChatBubble
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  agent={message.agent}
                  confidence={message.confidence}
                  sources={message.sources}
                />
              ))}

              {showStreamingBubble && (
                <ChatBubble
                  role="assistant"
                  content={stream.content}
                  agent={stream.agent}
                  confidence={stream.confidence}
                  sources={stream.sources}
                  isStreaming
                />
              )}
              {showTypingIndicator && <TypingIndicator agent={stream.agent} />}

              {stream.error && (
                <Card className="border-destructive/40 bg-destructive/5">
                  <p className="text-sm text-destructive">{stream.error}</p>
                </Card>
              )}
            </div>
          )}

          <div ref={bottomRef} className="h-4" />
        </div>
      </main>

      <footer className="shrink-0 border-t border-border/60 bg-white/70 backdrop-blur-xl dark:bg-background/80">
        <div className="mx-auto max-w-3xl px-4 py-3">
          {/* Memory confirmations. Showing what was learned is what makes the
              memory correctable rather than uncanny. */}
          {Object.keys(stream.memoryUpdates).length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {Object.entries(stream.memoryUpdates).map(([key, value]) => (
                <Badge key={key} tone="brand" className="text-[11px]">
                  Noted: {key.replace(/_/g, " ")} ={" "}
                  {Array.isArray(value) ? value.join(", ") : String(value)}
                </Badge>
              ))}
            </div>
          )}

          {voice.isListening && (
            <p className="mb-2 text-xs text-primary">
              Listening… {voice.interim && <span className="italic">{voice.interim}</span>}
            </p>
          )}
          {voice.error && (
            <p className="mb-2 text-xs text-destructive">{voice.error}</p>
          )}

          <form
            onSubmit={(event) => {
              event.preventDefault();
              submit(input);
            }}
            className="glass-card flex items-end gap-2 rounded-2xl p-2"
          >
            <label htmlFor="chat-input" className="sr-only">
              Message Oviora
            </label>
            <textarea
              id="chat-input"
              ref={textareaRef}
              rows={1}
              value={input}
              maxLength={MAX_CHARS}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                // Enter sends, Shift+Enter adds a newline — the convention
                // every chat interface uses.
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  submit(input);
                }
              }}
              placeholder="Ask about symptoms, food, exercise, your cycle…"
              className="max-h-[200px] flex-1 resize-none bg-transparent px-2 py-2.5 text-[15px] outline-none placeholder:text-muted-foreground"
            />

            {voice.supported && (
              <Button
                type="button"
                variant={voice.isListening ? "destructive" : "ghost"}
                size="icon"
                onClick={() => (voice.isListening ? voice.stop() : voice.start())}
                aria-label={voice.isListening ? "Stop voice input" : "Start voice input"}
              >
                {voice.isListening ? (
                  <MicOff className="h-5 w-5" />
                ) : (
                  <Mic className="h-5 w-5" />
                )}
              </Button>
            )}

            {stream.isStreaming ? (
              <Button
                type="button"
                variant="secondary"
                size="icon"
                onClick={stream.stop}
                aria-label="Stop generating"
              >
                <Square className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim()}
                aria-label="Send message"
              >
                <ArrowUp className="h-5 w-5" />
              </Button>
            )}
          </form>

          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            Oviora can make mistakes and does not diagnose. Always confirm health
            decisions with a qualified clinician.
          </p>
        </div>
      </footer>
    </div>
  );
}
