"use client";

import { QueryClient } from "@tanstack/react-query";

/**
 * React Query defaults tuned for a health dashboard.
 *
 * `staleTime` of 30s stops the dashboard re-fetching on every tab focus —
 * health data changes on the scale of hours, not seconds, and a spinner on
 * every window switch reads as jank.
 *
 * Retries are disabled for 4xx: a 401 or a 422 will fail identically three
 * more times, and retrying just delays the error the user needs to see.
 */
export function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        gcTime: 5 * 60_000,
        refetchOnWindowFocus: false,
        retry: (failureCount, error) => {
          const status = (error as { status?: number })?.status ?? 0;
          if (status >= 400 && status < 500) return false;
          return failureCount < 2;
        },
      },
      mutations: { retry: false },
    },
  });
}
