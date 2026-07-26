/**
 * Typed API client.
 *
 * Two things this handles that a bare `fetch` does not:
 *
 * **Transparent token refresh.** Access tokens last 30 minutes. Rather than
 * bouncing the user to the login screen when one expires mid-session, a 401
 * triggers a refresh and the original request is replayed once. Concurrent
 * 401s share a single in-flight refresh promise, so ten parallel dashboard
 * requests produce one refresh call rather than ten — which matters because
 * refresh tokens rotate, and ten concurrent rotations would invalidate each
 * other and log the user out.
 *
 * **A uniform error type.** The backend returns one error shape everywhere, so
 * every failure surfaces as an `ApiError` with a readable message instead of
 * a raw Response the caller has to interrogate.
 */

import type { ApiError } from "@/types/api";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

const ACCESS_TOKEN_KEY = "oviora.access_token";
const REFRESH_TOKEN_KEY = "oviora.refresh_token";

export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly details?: Record<string, string>,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiRequestError";
  }

  /** Field-level messages, for rendering inline validation errors. */
  get fieldErrors(): Record<string, string> {
    return this.details ?? {};
  }
}

// --------------------------------------------------------------- token store
// localStorage rather than an httpOnly cookie is a deliberate trade-off for
// this deployment: the SPA and API are separate origins, and cookie-based auth
// across origins needs CSRF protection plus SameSite=None, which is a larger
// surface than it sounds. The mitigation is short-lived access tokens and
// server-side revocation on logout. For a production healthcare deployment,
// httpOnly cookies with a same-origin BFF proxy is the stronger design.

export const tokens = {
  get access(): string | null {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(ACCESS_TOKEN_KEY);
  },
  get refresh(): string | null {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(REFRESH_TOKEN_KEY);
  },
  set(access: string, refresh: string) {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(ACCESS_TOKEN_KEY, access);
    window.localStorage.setItem(REFRESH_TOKEN_KEY, refresh);
  },
  clear() {
    if (typeof window === "undefined") return;
    window.localStorage.removeItem(ACCESS_TOKEN_KEY);
    window.localStorage.removeItem(REFRESH_TOKEN_KEY);
  },
};

/** Shared in-flight refresh, so concurrent 401s trigger exactly one rotation. */
let refreshInFlight: Promise<boolean> | null = null;

async function refreshAccessToken(): Promise<boolean> {
  if (refreshInFlight) return refreshInFlight;

  refreshInFlight = (async () => {
    const refresh = tokens.refresh;
    if (!refresh) return false;
    try {
      const response = await fetch(`${BASE_URL}/api/v1/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refresh }),
      });
      if (!response.ok) {
        tokens.clear();
        return false;
      }
      const data = await response.json();
      tokens.set(data.access_token, data.refresh_token);
      return true;
    } catch {
      tokens.clear();
      return false;
    } finally {
      refreshInFlight = null;
    }
  })();

  return refreshInFlight;
}

interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
  /** Skip the Authorization header (login, register, agent directory). */
  anonymous?: boolean;
  /** Internal: prevents infinite retry loops. */
  _retried?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, anonymous, _retried, headers, ...rest } = options;

  const finalHeaders: Record<string, string> = {
    ...(body !== undefined && !(body instanceof FormData)
      ? { "Content-Type": "application/json" }
      : {}),
    ...((headers as Record<string, string>) ?? {}),
  };

  if (!anonymous) {
    const token = tokens.access;
    if (token) finalHeaders.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    ...rest,
    headers: finalHeaders,
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  });

  if (response.status === 401 && !anonymous && !_retried) {
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      return request<T>(path, { ...options, _retried: true });
    }
  }

  if (!response.ok) {
    let payload: ApiError = {
      error: `http_${response.status}`,
      message: response.statusText || "Request failed.",
    };
    try {
      payload = (await response.json()) as ApiError;
    } catch {
      // A non-JSON error body (a proxy 502 page, say). Keep the default.
    }
    throw new ApiRequestError(
      payload.message,
      response.status,
      payload.error,
      payload.details,
      payload.request_id,
    );
  }

  if (response.status === 204) return undefined as T;

  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    return (await response.blob()) as T;
  }
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string, options?: RequestOptions) =>
    request<T>(path, { ...options, method: "GET" }),
  post: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>(path, { ...options, method: "POST", body }),
  put: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>(path, { ...options, method: "PUT", body }),
  patch: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>(path, { ...options, method: "PATCH", body }),
  delete: <T>(path: string, options?: RequestOptions) =>
    request<T>(path, { ...options, method: "DELETE" }),
  baseUrl: BASE_URL,
};
