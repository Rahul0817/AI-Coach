"use client";

import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/primitives";

const STORAGE_KEY = "oviora.theme";

/**
 * Dark-mode toggle.
 *
 * The initial theme is applied by a blocking inline script in the document
 * head (see `layout.tsx`), not here. Doing it in a React effect would paint
 * the light theme first and then swap — the "flash of wrong theme" that makes
 * an app feel unpolished, and which is genuinely unpleasant at night.
 */
export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggle() {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
    window.localStorage.setItem(STORAGE_KEY, next ? "dark" : "light");
  }

  // Render a stable placeholder until mounted so server and client markup
  // match and React does not log a hydration mismatch.
  if (!mounted) return <div className="h-10 w-10" aria-hidden />;

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggle}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
    </Button>
  );
}
