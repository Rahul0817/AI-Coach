import {
  Activity,
  Brain,
  Camera,
  FileText,
  HeartHandshake,
  Mic,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import Link from "next/link";

import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { Badge, Button, Card } from "@/components/ui/primitives";

/**
 * Landing page.
 *
 * Rendered on the server with no client JavaScript beyond the theme toggle —
 * it is static content, and shipping a React bundle to render it would slow
 * the one page most likely to be a visitor's first impression.
 */

const FEATURES = [
  {
    icon: Brain,
    title: "Eight specialist agents",
    body: "A router sends each question to the right expert — nutrition, fitness, cycles, mental wellbeing, lab reports and more — instead of one generalist chatbot.",
  },
  {
    icon: ShieldCheck,
    title: "Grounded in real guidance",
    body: "Answers are retrieved from a curated PCOS knowledge base and cited. When nothing relevant is found, Oviora says so rather than inventing an answer.",
  },
  {
    icon: Activity,
    title: "Explainable risk assessment",
    body: "A calibrated model estimates risk from your symptoms, and SHAP shows exactly which answers moved the number and by how much.",
  },
  {
    icon: FileText,
    title: "Understand your blood work",
    body: "Upload a lab report. Oviora extracts each biomarker, explains what it measures, and helps you prepare questions for your doctor.",
  },
  {
    icon: Camera,
    title: "Meal analysis",
    body: "Photograph a meal for a macro breakdown, a glycaemic-load score and one realistic swap — no shame, no banned foods.",
  },
  {
    icon: Mic,
    title: "Talk, don't type",
    body: "Voice input runs on your device. The recording never leaves your browser.",
  },
];

const AGENTS = [
  { name: "Health Expert", colour: "#8b5cf6" },
  { name: "Nutrition Coach", colour: "#ec4899" },
  { name: "Fitness Coach", colour: "#f59e0b" },
  { name: "Mental Wellness Coach", colour: "#06b6d4" },
  { name: "Blood Report Analyzer", colour: "#10b981" },
  { name: "Food Analyzer", colour: "#f43f5e" },
  { name: "Habit Coach", colour: "#a855f7" },
  { name: "Cycle Tracker Assistant", colour: "#d946ef" },
];

export default function LandingPage() {
  return (
    <div className="min-h-dvh">
      <header className="sticky top-0 z-40 border-b border-white/30 bg-white/60 backdrop-blur-xl dark:border-white/10 dark:bg-background/70">
        <nav className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
          <Link href="/" className="flex items-center gap-2 focus-ring rounded-lg">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-brand">
              <Sparkles className="h-5 w-5 text-white" />
            </span>
            <span className="text-lg font-semibold tracking-tight">Oviora</span>
          </Link>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Link href="/login">
              <Button variant="ghost" size="sm">
                Sign in
              </Button>
            </Link>
            <Link href="/register">
              <Button size="sm">Get started</Button>
            </Link>
          </div>
        </nav>
      </header>

      <main id="main">
        {/* ------------------------------------------------------- hero -- */}
        <section className="mx-auto max-w-6xl px-5 py-16 sm:py-24">
          <div className="mx-auto max-w-3xl text-center">
            <Badge tone="brand" className="mb-5">
              <Sparkles className="h-3 w-3" />
              Educational guidance, never a diagnosis
            </Badge>

            <h1 className="text-balance text-4xl font-bold leading-tight tracking-tight sm:text-6xl">
              Your personal AI companion for{" "}
              <span className="brand-text">managing PCOS</span>
            </h1>

            <p className="mx-auto mt-6 max-w-2xl text-pretty text-lg text-muted-foreground">
              PCOS affects up to one in eight women, and most wait years for
              answers. Oviora gives you a knowledgeable companion that remembers
              your history, explains what is happening in plain language, and
              helps you walk into an appointment prepared.
            </p>

            <div className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Link href="/register">
                <Button size="lg" className="w-full sm:w-auto">
                  Start free
                </Button>
              </Link>
              <Link href="/chat">
                <Button variant="outline" size="lg" className="w-full sm:w-auto">
                  Try the assistant
                </Button>
              </Link>
            </div>

            <p className="mt-5 text-xs text-muted-foreground">
              No credit card required · Your data stays yours · Export or delete
              anytime
            </p>
          </div>
        </section>

        {/* --------------------------------------------------- agents -- */}
        <section className="mx-auto max-w-6xl px-5 pb-16">
          <Card className="p-6 sm:p-8">
            <h2 className="text-center text-sm font-medium uppercase tracking-wider text-muted-foreground">
              One question, the right specialist
            </h2>
            <div className="mt-6 flex flex-wrap items-center justify-center gap-2.5">
              {AGENTS.map((agent) => (
                <span
                  key={agent.name}
                  className="rounded-full border px-3.5 py-1.5 text-sm font-medium"
                  style={{
                    borderColor: `${agent.colour}40`,
                    backgroundColor: `${agent.colour}12`,
                    color: agent.colour,
                  }}
                >
                  {agent.name}
                </span>
              ))}
            </div>
            <p className="mt-6 text-center text-sm text-muted-foreground">
              A routing agent reads each message and dispatches it. Ask about
              breakfast and the Nutrition Coach answers; mention a 45-day cycle
              and the Cycle Tracker Assistant takes over.
            </p>
          </Card>
        </section>

        {/* -------------------------------------------------- features -- */}
        <section className="mx-auto max-w-6xl px-5 pb-20">
          <div className="mb-10 text-center">
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
              Built like a product, not a demo
            </h2>
            <p className="mx-auto mt-3 max-w-2xl text-muted-foreground">
              Every answer is grounded, every prediction is explained, and every
              health claim carries a disclaimer enforced in code.
            </p>
          </div>

          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((feature) => (
              <Card key={feature.title} className="transition-shadow hover:shadow-glass-lg">
                <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-brand">
                  <feature.icon className="h-5 w-5 text-white" />
                </div>
                <h3 className="mb-1.5 font-semibold">{feature.title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {feature.body}
                </p>
              </Card>
            ))}
          </div>
        </section>

        {/* ------------------------------------------------- disclaimer -- */}
        <section className="mx-auto max-w-4xl px-5 pb-20">
          <Card className="border-amber-300/50 bg-amber-50/70 dark:border-amber-400/20 dark:bg-amber-500/5">
            <div className="flex gap-3">
              <HeartHandshake className="h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" />
              <div className="text-sm">
                <p className="font-semibold text-amber-900 dark:text-amber-200">
                  Oviora does not diagnose any condition.
                </p>
                <p className="mt-1.5 leading-relaxed text-amber-800/90 dark:text-amber-200/80">
                  It provides educational information, lifestyle guidance and a
                  screening-level risk estimate. PCOS is diagnosed by a clinician
                  after examination and testing, and several other conditions
                  produce overlapping symptoms. Please speak to a gynaecologist
                  or endocrinologist about anything that concerns you.
                </p>
              </div>
            </div>
          </Card>
        </section>
      </main>

      <footer className="border-t border-border py-8">
        <div className="mx-auto max-w-6xl px-5 text-center text-sm text-muted-foreground">
          <p>
            <span className="font-semibold text-foreground">Oviora AI</span> ·
            Your Personal AI Companion for Managing PCOS
          </p>
          <p className="mt-2 text-xs">
            Educational software. Not a medical device. Not a substitute for
            professional medical advice.
          </p>
        </div>
      </footer>
    </div>
  );
}
