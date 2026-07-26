"use client";

import { motion } from "framer-motion";
import { BookOpen, Copy, Check, Volume2, VolumeX } from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { Badge, Button } from "@/components/ui/primitives";
import { useSpeechSynthesis } from "@/hooks/useVoice";
import { cn } from "@/lib/utils";
import type { Citation } from "@/types/api";

const AGENT_COLOURS: Record<string, string> = {
  health_expert: "#8b5cf6",
  nutrition_coach: "#ec4899",
  fitness_coach: "#f59e0b",
  mental_wellness_coach: "#06b6d4",
  blood_report_analyzer: "#10b981",
  food_analyzer: "#f43f5e",
  habit_coach: "#a855f7",
  cycle_tracker_assistant: "#d946ef",
};

function agentLabel(agent: string): string {
  return agent
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export interface ChatBubbleProps {
  role: "user" | "assistant";
  content: string;
  agent?: string | null;
  confidence?: number | null;
  sources?: Citation[];
  isStreaming?: boolean;
}

export function ChatBubble({
  role,
  content,
  agent,
  confidence,
  sources = [],
  isStreaming = false,
}: ChatBubbleProps) {
  const [copied, setCopied] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const { isSpeaking, supported: canSpeak, speak, stop } = useSpeechSynthesis();

  const isUser = role === "user";
  const colour = agent ? (AGENT_COLOURS[agent] ?? "#8b5cf6") : "#8b5cf6";

  async function copy() {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={cn("flex w-full gap-3", isUser && "flex-row-reverse")}
    >
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-semibold text-white",
          isUser ? "bg-muted-foreground" : "",
        )}
        style={!isUser ? { background: colour } : undefined}
        aria-hidden
      >
        {isUser ? "You" : "AI"}
      </div>

      <div className={cn("min-w-0 max-w-[85%]", isUser && "flex flex-col items-end")}>
        {!isUser && agent && (
          <div className="mb-1.5 flex items-center gap-2">
            <span className="text-xs font-medium" style={{ color: colour }}>
              {agentLabel(agent)}
            </span>
            {/* Routing confidence is surfaced rather than hidden: a user should
                be able to see when the assistant was unsure which specialist
                to use. */}
            {typeof confidence === "number" && confidence < 0.5 && (
              <Badge tone="neutral" className="text-[10px]">
                low routing confidence
              </Badge>
            )}
          </div>
        )}

        <div
          className={cn(
            "rounded-2xl px-4 py-3",
            isUser
              ? "bg-gradient-brand text-white"
              : "glass-card border border-border/60",
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
              {content}
            </p>
          ) : (
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
              {isStreaming && (
                <span
                  className="ml-0.5 inline-block h-4 w-[2px] animate-pulse-soft bg-primary align-middle"
                  aria-label="Assistant is typing"
                />
              )}
            </div>
          )}
        </div>

        {!isUser && !isStreaming && content && (
          <div className="mt-2 flex flex-wrap items-center gap-1">
            <Button variant="ghost" size="sm" onClick={copy} aria-label="Copy response">
              {copied ? (
                <Check className="h-3.5 w-3.5" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
              <span className="text-xs">{copied ? "Copied" : "Copy"}</span>
            </Button>

            {canSpeak && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => (isSpeaking ? stop() : speak(content))}
                aria-label={isSpeaking ? "Stop reading" : "Read aloud"}
              >
                {isSpeaking ? (
                  <VolumeX className="h-3.5 w-3.5" />
                ) : (
                  <Volume2 className="h-3.5 w-3.5" />
                )}
                <span className="text-xs">{isSpeaking ? "Stop" : "Listen"}</span>
              </Button>
            )}

            {sources.length > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowSources((open) => !open)}
                aria-expanded={showSources}
              >
                <BookOpen className="h-3.5 w-3.5" />
                <span className="text-xs">
                  {sources.length} source{sources.length === 1 ? "" : "s"}
                </span>
              </Button>
            )}
          </div>
        )}

        {/* Citations are collapsed by default — they matter for trust but
            would otherwise dominate the reading experience. */}
        {showSources && sources.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mt-2 space-y-2 overflow-hidden"
          >
            {sources.map((source, index) => (
              <div
                key={`${source.title}-${index}`}
                className="rounded-xl border border-border bg-muted/40 p-3 text-xs"
              >
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="font-medium">{source.title}</span>
                  <Badge tone="brand" className="text-[10px]">
                    {Math.round(source.score * 100)}% match
                  </Badge>
                </div>
                <p className="leading-relaxed text-muted-foreground">
                  {source.snippet}
                </p>
                <p className="mt-1.5 text-[11px] italic text-muted-foreground/70">
                  {source.source}
                </p>
              </div>
            ))}
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

/** Shown while waiting for the first token. */
export function TypingIndicator({ agent }: { agent?: string | null }) {
  const colour = agent ? (AGENT_COLOURS[agent] ?? "#8b5cf6") : "#8b5cf6";
  return (
    <div className="flex gap-3">
      <div
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-semibold text-white"
        style={{ background: colour }}
        aria-hidden
      >
        AI
      </div>
      <div className="glass-card flex items-center gap-1.5 rounded-2xl px-4 py-3.5">
        <span className="sr-only">Assistant is thinking</span>
        {[0, 1, 2].map((index) => (
          <span
            key={index}
            className="h-2 w-2 animate-pulse-soft rounded-full bg-primary/60"
            style={{ animationDelay: `${index * 0.15}s` }}
            aria-hidden
          />
        ))}
      </div>
    </div>
  );
}
