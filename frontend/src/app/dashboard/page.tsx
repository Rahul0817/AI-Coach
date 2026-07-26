"use client";

import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowRight,
  Droplet,
  Flame,
  Lightbulb,
  Moon,
  Scale,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  Dumbbell,
} from "lucide-react";
import Link from "next/link";
import type { ComponentType } from "react";

import {
  ActivityBars,
  MacroDoughnut,
  TrendChart,
} from "@/components/dashboard/TrendChart";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import {
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  Progress,
  Skeleton,
} from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { cn, formatValue, riskBandColour } from "@/lib/utils";
import type { Dashboard, MetricCard as MetricCardType } from "@/types/api";

const ICONS: Record<string, ComponentType<{ className?: string }>> = {
  scale: Scale,
  activity: Activity,
  moon: Moon,
  flame: Flame,
  drumstick: Flame,
  droplet: Droplet,
  dumbbell: Dumbbell,
  target: Target,
};

function MetricTile({ metric }: { metric: MetricCardType }) {
  const Icon = ICONS[metric.icon] ?? Activity;
  const improving = metric.direction === "up";
  const declining = metric.direction === "down";

  return (
    <Card className="flex flex-col gap-3">
      <div className="flex items-start justify-between">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10">
          <Icon className="h-4.5 w-4.5 text-primary" />
        </div>
        {metric.change_percentage !== null && (
          <Badge tone={improving ? "success" : declining ? "warning" : "neutral"}>
            {improving ? (
              <TrendingUp className="h-3 w-3" />
            ) : declining ? (
              <TrendingDown className="h-3 w-3" />
            ) : null}
            {Math.abs(metric.change_percentage).toFixed(1)}%
          </Badge>
        )}
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground">{metric.label}</p>
        <p className="mt-0.5 text-2xl font-semibold tabular-nums">
          {formatValue(metric.value)}
          {metric.unit && (
            <span className="ml-1 text-sm font-normal text-muted-foreground">
              {metric.unit}
            </span>
          )}
        </p>
        {metric.secondary_value && (
          <p className="text-xs text-muted-foreground">
            {metric.secondary_label}: {metric.secondary_value}
          </p>
        )}
      </div>

      {metric.progress_percentage !== null && (
        <div>
          <Progress
            value={metric.progress_percentage}
            tone={metric.progress_percentage >= 100 ? "success" : "brand"}
          />
          {metric.goal !== null && (
            <p className="mt-1.5 text-[11px] text-muted-foreground">
              {Math.round(metric.progress_percentage)}% of {formatValue(metric.goal, 0)}{" "}
              {metric.unit}
            </p>
          )}
        </div>
      )}
    </Card>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 8 }).map((_, index) => (
          <Card key={index} className="space-y-3">
            <Skeleton className="h-9 w-9 rounded-xl" />
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-7 w-24" />
          </Card>
        ))}
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <Skeleton className="mb-4 h-4 w-32" />
          <Skeleton className="h-[220px] rounded-xl" />
        </Card>
        <Card>
          <Skeleton className="mb-4 h-4 w-32" />
          <Skeleton className="h-[220px] rounded-xl" />
        </Card>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["dashboard"],
    queryFn: () => api.get<Dashboard>("/api/v1/analytics/dashboard?days=30"),
  });

  return (
    <div className="min-h-dvh">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-white/60 backdrop-blur-xl dark:bg-background/70">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-3">
          <Link href="/" className="flex items-center gap-2 focus-ring rounded-lg">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-brand">
              <Sparkles className="h-4 w-4 text-white" />
            </span>
            <span className="font-semibold">Oviora</span>
          </Link>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Link href="/chat">
              <Button size="sm">
                Ask the AI
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <main id="main" className="mx-auto max-w-7xl px-5 py-8">
        {isLoading && <DashboardSkeleton />}

        {isError && (
          <Card className="border-destructive/40 bg-destructive/5">
            <EmptyState
              title="Could not load your dashboard"
              description={
                (error as Error)?.message ??
                "Something went wrong. Please try again."
              }
              action={
                <Button onClick={() => window.location.reload()}>Retry</Button>
              }
            />
          </Card>
        )}

        {data && (
          <div className="space-y-6">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
                  {data.greeting}
                </h1>
                <p className="mt-1 text-sm text-muted-foreground">
                  Here is how the last 30 days are looking.
                </p>
              </div>
              {data.latest_risk && (
                <Card className="px-4 py-3">
                  <p className="text-xs text-muted-foreground">
                    Latest risk estimate
                  </p>
                  <p
                    className={cn(
                      "text-lg font-semibold",
                      riskBandColour(data.latest_risk.risk_band),
                    )}
                  >
                    {data.latest_risk.risk_percentage}%{" "}
                    <span className="text-sm font-normal capitalize">
                      ({data.latest_risk.risk_band})
                    </span>
                  </p>
                </Card>
              )}
            </div>

            {/* --------------------------------------------- KPI tiles -- */}
            <section aria-label="Key metrics">
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {data.metrics.map((metric) => (
                  <MetricTile key={metric.key} metric={metric} />
                ))}
              </div>
            </section>

            {/* ---------------------------------------------- insights -- */}
            {data.insights.length > 0 && (
              <section aria-label="AI insights">
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {data.insights.map((insight) => (
                    <Card
                      key={insight.title}
                      className={cn(
                        "border-l-4",
                        insight.severity === "positive" && "border-l-emerald-500",
                        insight.severity === "attention" && "border-l-amber-500",
                        insight.severity === "info" && "border-l-primary",
                      )}
                    >
                      <div className="mb-2 flex items-start gap-2">
                        <Lightbulb
                          className={cn(
                            "mt-0.5 h-4 w-4 shrink-0",
                            insight.severity === "positive" && "text-emerald-500",
                            insight.severity === "attention" && "text-amber-500",
                            insight.severity === "info" && "text-primary",
                          )}
                        />
                        <h3 className="text-sm font-semibold">{insight.title}</h3>
                      </div>
                      <p className="text-sm leading-relaxed text-muted-foreground">
                        {insight.body}
                      </p>
                      {insight.action_url && insight.action_label && (
                        <Link href={insight.action_url}>
                          <Button variant="ghost" size="sm" className="mt-2 -ml-3">
                            {insight.action_label}
                            <ArrowRight className="h-3.5 w-3.5" />
                          </Button>
                        </Link>
                      )}
                    </Card>
                  ))}
                </div>
              </section>
            )}

            {/* ------------------------------------------------ charts -- */}
            <section aria-label="Trends" className="grid gap-5 lg:grid-cols-2">
              <Card>
                <CardHeader
                  title="Weight"
                  description={
                    data.weight_trend.average
                      ? `Averaging ${data.weight_trend.average} kg`
                      : "Log your weight to see a trend"
                  }
                />
                <TrendChart series={data.weight_trend} colour="#a855f7" />
              </Card>

              <Card>
                <CardHeader
                  title="Sleep"
                  description={
                    data.sleep_trend.average
                      ? `Averaging ${data.sleep_trend.average} hours`
                      : "Log your sleep to see a trend"
                  }
                />
                <TrendChart series={data.sleep_trend} colour="#06b6d4" />
              </Card>

              <Card>
                <CardHeader
                  title="Calories"
                  description="Daily intake over the last 30 days"
                />
                <TrendChart series={data.calorie_trend} colour="#f59e0b" />
              </Card>

              <Card>
                <CardHeader
                  title="Macro split"
                  description="Share of calories from each macronutrient"
                />
                <MacroDoughnut split={data.macro_split} />
              </Card>

              <Card>
                <CardHeader
                  title="Active minutes"
                  description="Last two weeks"
                />
                <ActivityBars series={data.workout_minutes_trend} />
              </Card>

              <Card>
                <CardHeader
                  title="Mood"
                  description="Daily check-ins, scored 1 to 5"
                />
                <TrendChart series={data.mood_trend} colour="#ec4899" />
              </Card>
            </section>

            {/* ------------------------------------------ cycle + habits -- */}
            <section className="grid gap-5 lg:grid-cols-2">
              <Card>
                <CardHeader
                  title="Cycle"
                  description={data.cycle_summary.regularity}
                />
                {data.cycle_summary.total_cycles === 0 ? (
                  <EmptyState
                    title="No cycles logged yet"
                    description="Three months of cycle data is the single most useful thing you can bring to an appointment."
                  />
                ) : (
                  <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                    <div>
                      <p className="text-xs text-muted-foreground">Average length</p>
                      <p className="text-xl font-semibold tabular-nums">
                        {formatValue(data.cycle_summary.average_length)}
                        <span className="ml-1 text-sm font-normal text-muted-foreground">
                          days
                        </span>
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Variability</p>
                      <p className="text-xl font-semibold tabular-nums">
                        ±{formatValue(data.cycle_summary.variability)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Next expected</p>
                      <p className="text-xl font-semibold tabular-nums">
                        {data.cycle_summary.days_until_next !== null
                          ? `${data.cycle_summary.days_until_next}d`
                          : "—"}
                      </p>
                    </div>
                  </div>
                )}
                <p className="mt-4 text-[11px] leading-relaxed text-muted-foreground">
                  Predictions are estimates and are especially unreliable in PCOS.
                  Never use them as contraception.
                </p>
              </Card>

              <Card>
                <CardHeader title="Habits" description="Streaks and 30-day consistency" />
                {data.habit_completion.length === 0 ? (
                  <EmptyState
                    title="No habits yet"
                    description="Start with one habit smaller than feels worthwhile. That is the one that survives."
                  />
                ) : (
                  <ul className="space-y-3">
                    {data.habit_completion.map((habit) => (
                      <li key={habit.name} className="flex items-center gap-3">
                        <span
                          className="h-2.5 w-2.5 shrink-0 rounded-full"
                          style={{ background: habit.colour }}
                          aria-hidden
                        />
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center justify-between gap-2">
                            <span className="truncate text-sm font-medium">
                              {habit.name}
                            </span>
                            <span className="shrink-0 text-xs text-muted-foreground">
                              {habit.current_streak}d streak
                            </span>
                          </div>
                          <Progress
                            value={habit.completion_rate_30d}
                            className="mt-1.5 h-1.5"
                          />
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
