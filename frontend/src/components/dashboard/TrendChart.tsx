"use client";

import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  ArcElement,
  BarElement,
} from "chart.js";
import { Bar, Doughnut, Line } from "react-chartjs-2";
import { useEffect, useState } from "react";

import type { TimeSeries } from "@/types/api";

// Chart.js is tree-shakeable but requires explicit registration; importing the
// whole `chart.js/auto` bundle would add well over 100 KB of unused code.
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Filler,
  Tooltip,
  Legend,
);

/** Read a themed CSS variable so charts follow light/dark mode. */
function useThemeColours() {
  const [colours, setColours] = useState({
    text: "#6b7280",
    grid: "rgba(148,163,184,0.15)",
  });

  useEffect(() => {
    const update = () => {
      const isDark = document.documentElement.classList.contains("dark");
      setColours({
        text: isDark ? "#a1a1aa" : "#6b7280",
        grid: isDark ? "rgba(148,163,184,0.12)" : "rgba(148,163,184,0.18)",
      });
    };
    update();
    // Re-read when the theme class changes, so charts do not stay light-themed
    // after a toggle.
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, []);

  return colours;
}

export function TrendChart({
  series,
  colour = "#a855f7",
  height = 220,
}: {
  series: TimeSeries;
  colour?: string;
  height?: number;
}) {
  const themeColours = useThemeColours();

  const hasData = series.points.some((point) => point.value !== null);
  if (!hasData) {
    return (
      <div
        className="flex items-center justify-center rounded-xl bg-muted/40 text-sm text-muted-foreground"
        style={{ height }}
      >
        No {series.metric.replace(/_/g, " ")} logged yet
      </div>
    );
  }

  return (
    <div style={{ height }}>
      <Line
        data={{
          labels: series.points.map((point) =>
            new Date(point.label).toLocaleDateString(undefined, {
              day: "numeric",
              month: "short",
            }),
          ),
          datasets: [
            {
              label: `${series.metric} (${series.unit})`,
              data: series.points.map((point) => point.value),
              borderColor: colour,
              backgroundColor: `${colour}20`,
              borderWidth: 2.5,
              fill: true,
              tension: 0.35,
              pointRadius: 0,
              pointHoverRadius: 5,
              pointHoverBackgroundColor: colour,
              // Draw a gap where data is missing rather than interpolating a
              // straight line through it. A line implies measurements that were
              // never taken.
              spanGaps: false,
            },
          ],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: "rgba(24,16,38,0.92)",
              padding: 10,
              cornerRadius: 8,
              displayColors: false,
              callbacks: {
                label: (context) =>
                  context.parsed.y === null
                    ? "Not logged"
                    : `${context.parsed.y} ${series.unit}`,
              },
            },
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: {
                color: themeColours.text,
                maxTicksLimit: 6,
                font: { size: 11 },
              },
              border: { display: false },
            },
            y: {
              grid: { color: themeColours.grid },
              ticks: { color: themeColours.text, font: { size: 11 } },
              border: { display: false },
            },
          },
        }}
      />
    </div>
  );
}

export function MacroDoughnut({ split }: { split: Record<string, number> }) {
  const themeColours = useThemeColours();
  const total = Object.values(split).reduce((sum, value) => sum + value, 0);

  if (total === 0) {
    return (
      <div className="flex h-[220px] items-center justify-center rounded-xl bg-muted/40 text-sm text-muted-foreground">
        Log a meal to see your macro split
      </div>
    );
  }

  return (
    <div style={{ height: 220 }}>
      <Doughnut
        data={{
          labels: ["Protein", "Carbohydrate", "Fat"],
          datasets: [
            {
              data: [split.protein ?? 0, split.carbs ?? 0, split.fat ?? 0],
              backgroundColor: ["#a855f7", "#f472b6", "#f59e0b"],
              borderWidth: 0,
              // A doughnut rather than a pie: the hole makes the arc lengths
              // easier to compare than wedge areas.
              hoverOffset: 6,
            },
          ],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          cutout: "62%",
          plugins: {
            legend: {
              position: "bottom",
              labels: {
                color: themeColours.text,
                boxWidth: 10,
                boxHeight: 10,
                usePointStyle: true,
                font: { size: 12 },
              },
            },
            tooltip: {
              backgroundColor: "rgba(24,16,38,0.92)",
              callbacks: {
                label: (context) => `${context.label}: ${context.parsed}% of calories`,
              },
            },
          },
        }}
      />
    </div>
  );
}

export function ActivityBars({ series }: { series: TimeSeries }) {
  const themeColours = useThemeColours();
  const recent = series.points.slice(-14);

  return (
    <div style={{ height: 200 }}>
      <Bar
        data={{
          labels: recent.map((point) =>
            new Date(point.label).toLocaleDateString(undefined, {
              weekday: "narrow",
            }),
          ),
          datasets: [
            {
              label: "Active minutes",
              data: recent.map((point) => point.value ?? 0),
              backgroundColor: "#a855f7",
              borderRadius: 6,
              maxBarThickness: 26,
            },
          ],
        }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: "rgba(24,16,38,0.92)" },
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: { color: themeColours.text, font: { size: 11 } },
              border: { display: false },
            },
            y: {
              grid: { color: themeColours.grid },
              ticks: { color: themeColours.text, font: { size: 11 } },
              border: { display: false },
              beginAtZero: true,
            },
          },
        }}
      />
    </div>
  );
}
