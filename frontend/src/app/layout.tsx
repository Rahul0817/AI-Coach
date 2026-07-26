import type { Metadata, Viewport } from "next";

import { Providers } from "@/components/layout/Providers";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Oviora AI — Your Personal AI Companion for Managing PCOS",
    template: "%s · Oviora AI",
  },
  description:
    "An AI copilot for PCOS: conversational guidance from eight specialist " +
    "agents, an explainable risk model, lab-report parsing and full health " +
    "tracking. Educational guidance only — Oviora does not diagnose.",
  keywords: ["PCOS", "polycystic ovary syndrome", "health tracking", "AI health"],
  authors: [{ name: "Oviora AI" }],
  openGraph: {
    title: "Oviora AI — Your Personal AI Companion for Managing PCOS",
    description:
      "Chat with specialist AI agents, track your cycle and understand your " +
      "health — grounded in curated PCOS guidance.",
    type: "website",
  },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#faf5ff" },
    { media: "(prefers-color-scheme: dark)", color: "#140f1c" },
  ],
  width: "device-width",
  initialScale: 1,
};

/**
 * Applied before first paint to avoid a flash of the wrong theme.
 *
 * This has to be a blocking inline script: any React-based approach runs after
 * hydration, by which point the browser has already painted. Wrapped in
 * try/catch because localStorage throws in private browsing on some engines,
 * and a theme preference is never worth a blank page.
 */
const themeScript = `
(function () {
  try {
    var stored = localStorage.getItem("oviora.theme");
    var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    if (stored === "dark" || (!stored && prefersDark)) {
      document.documentElement.classList.add("dark");
    }
  } catch (e) {}
})();
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-dvh bg-gradient-soft dark:bg-background">
        {/* Skip link: the first tab stop on every page, so keyboard and screen
            reader users are not forced through the whole navigation. */}
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-primary focus:px-4 focus:py-2 focus:text-white"
        >
          Skip to main content
        </a>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
