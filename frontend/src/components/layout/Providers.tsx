"use client";

import { QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

import { createQueryClient } from "@/lib/query";

/**
 * Client-side providers.
 *
 * The QueryClient is created inside `useState` rather than at module scope.
 * At module scope it would be shared across requests during server rendering,
 * leaking one user's cached data into another's response — a real data-privacy
 * bug in a health application, not a theoretical one.
 */
export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(createQueryClient);

  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}
