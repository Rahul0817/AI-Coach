"use client";

import { Check, Sparkles, X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { Button, Card, Input, Label } from "@/components/ui/primitives";
import { ApiRequestError, api, tokens } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { TokenPair } from "@/types/api";

/**
 * The password rules mirror the server's `validate_password_strength`.
 *
 * Duplicating them client-side is for feedback, not enforcement — the server
 * is still the authority, and the API rejects a weak password regardless of
 * what the browser allowed. Showing the rules live means a user is never
 * surprised by a rejection after filling in the whole form.
 */
const RULES = [
  { label: "At least 10 characters", test: (v: string) => v.length >= 10 },
  { label: "A lowercase letter", test: (v: string) => /[a-z]/.test(v) },
  { label: "An uppercase letter", test: (v: string) => /[A-Z]/.test(v) },
  { label: "A number or symbol", test: (v: string) => /[\d\W]/.test(v) },
];

export default function RegisterPage() {
  const router = useRouter();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  const passed = useMemo(
    () => RULES.map((rule) => rule.test(password)),
    [password],
  );
  // The server requires three of four character classes plus the length rule.
  const strongEnough =
    passed[0] === true && passed.slice(1).filter(Boolean).length >= 2;

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setFieldErrors({});
    try {
      const result = await api.post<TokenPair>(
        "/api/v1/auth/register",
        { email, password, full_name: fullName },
        { anonymous: true },
      );
      tokens.set(result.access_token, result.refresh_token);
      router.push("/dashboard");
    } catch (caught) {
      if (caught instanceof ApiRequestError) {
        setError(caught.message);
        setFieldErrors(caught.fieldErrors);
      } else {
        setError("Could not create your account. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main id="main" className="flex min-h-dvh items-center justify-center px-5 py-12">
      <div className="w-full max-w-md">
        <Link
          href="/"
          className="mb-8 flex items-center justify-center gap-2 focus-ring rounded-lg"
        >
          <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-brand">
            <Sparkles className="h-5 w-5 text-white" />
          </span>
          <span className="text-xl font-semibold">Oviora</span>
        </Link>

        <Card className="p-7">
          <h1 className="text-xl font-semibold">Create your account</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Free, and your data stays yours — export or delete it whenever you
            like.
          </p>

          <form onSubmit={onSubmit} className="mt-6 space-y-4" noValidate>
            <div>
              <Label htmlFor="full_name">Full name</Label>
              <Input
                id="full_name"
                autoComplete="name"
                required
                value={fullName}
                onChange={(event) => setFullName(event.target.value)}
                placeholder="Asha Rao"
                error={fieldErrors.full_name}
              />
            </div>

            <div>
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@example.com"
                error={fieldErrors.email}
              />
            </div>

            <div>
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete="new-password"
                required
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="••••••••••"
                error={fieldErrors.password}
              />
              {password.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {RULES.map((rule, index) => (
                    <li
                      key={rule.label}
                      className={cn(
                        "flex items-center gap-1.5 text-xs",
                        passed[index]
                          ? "text-emerald-600 dark:text-emerald-400"
                          : "text-muted-foreground",
                      )}
                    >
                      {passed[index] ? (
                        <Check className="h-3 w-3" />
                      ) : (
                        <X className="h-3 w-3" />
                      )}
                      {rule.label}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}

            <Button
              type="submit"
              className="w-full"
              loading={loading}
              disabled={!strongEnough}
            >
              Create account
            </Button>
          </form>

          <p className="mt-5 text-center text-sm text-muted-foreground">
            Already have an account?{" "}
            <Link href="/login" className="font-medium text-primary hover:underline">
              Sign in
            </Link>
          </p>

          <p className="mt-4 text-center text-[11px] leading-relaxed text-muted-foreground">
            Oviora provides educational guidance only and does not diagnose any
            medical condition.
          </p>
        </Card>
      </div>
    </main>
  );
}
