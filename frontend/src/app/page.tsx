"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  type RatedMovie,
  type RecommendResponse,
  type UserSummary,
  getUserHistory,
  listUsers,
  recommend,
} from "@/lib/api";
import { ContextControls, type Context } from "@/components/ContextControls";
import { RecommendationCard } from "@/components/RecommendationCard";

export default function HomePage() {
  const [users, setUsers] = useState<UserSummary[]>([]);
  const [userId, setUserId] = useState<number | null>(null);
  const [history, setHistory] = useState<RatedMovie[]>([]);
  const [context, setContext] = useState<Context>({
    dayOfWeek: "Friday",
    timeOfDay: "evening",
    device: "TV",
  });
  const [known, setKnown] = useState<RecommendResponse | null>(null);
  const [cold, setCold] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listUsers(50)
      .then((u) => {
        setUsers(u);
        if (u.length > 0) setUserId(u[0].user_id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (userId === null) return;
    getUserHistory(userId, 8)
      .then(setHistory)
      .catch((e) => setError(String(e)));
  }, [userId]);

  const recentMovieIds = useMemo(() => history.map((h) => h.movie_id), [history]);

  async function runComparison() {
    if (userId === null) return;
    setLoading(true);
    setError(null);
    try {
      const [k, c] = await Promise.all([
        recommend({
          user_id: userId,
          day_of_week: context.dayOfWeek,
          time_of_day: context.timeOfDay,
          device: context.device,
          k: 8,
        }),
        recommend({
          recent_movie_ids: recentMovieIds,
          day_of_week: context.dayOfWeek,
          time_of_day: context.timeOfDay,
          device: context.device,
          k: 8,
        }),
      ]);
      setKnown(k);
      setCold(c);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (userId !== null && recentMovieIds.length > 0) {
      // Fetching in response to user/context changes, not setting state
      // synchronously in the effect body itself — the lint rule can't see
      // through the function call, so it's suppressed here deliberately.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      runComparison();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, recentMovieIds.join(","), context.dayOfWeek, context.timeOfDay]);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <section className="mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-neutral-50">
          A context-aware two-tower recommender
        </h1>
        <p className="mt-2 max-w-3xl text-sm text-neutral-400">
          Built on NVIDIA Merlin. Pick a user below and watch recommendations react to
          request-time context (day of week, time of day) and to a cold-start simulation —
          the same user&apos;s recent activity, scored with no learned identity at all. See
          the <Link href="/architecture" className="underline hover:text-neutral-200">
            Architecture
          </Link>{" "}
          page for how this is built, the real ablation numbers, and credits.
        </p>
      </section>

      <section className="mb-8 rounded-xl border border-neutral-800 bg-neutral-900/40 p-5">
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <label className="flex flex-col gap-1 text-xs text-neutral-400">
            User
            <select
              className="min-w-[220px] rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100"
              value={userId ?? ""}
              onChange={(e) => setUserId(Number(e.target.value))}
            >
              {users.map((u) => (
                <option key={u.user_id} value={u.user_id}>
                  user #{u.user_id} ({u.num_ratings} ratings)
                </option>
              ))}
            </select>
          </label>
          <div className="flex-1">
            <ContextControls value={context} onChange={setContext} />
          </div>
        </div>

        {history.length > 0 && (
          <div>
            <p className="mb-2 text-xs uppercase tracking-wide text-neutral-500">
              Recent activity used as the &quot;user-take&quot; feature
            </p>
            <div className="flex flex-wrap gap-2">
              {history.map((h) => (
                <span
                  key={h.movie_id}
                  className="rounded-full border border-neutral-700 bg-neutral-900 px-3 py-1 text-xs text-neutral-300"
                >
                  {h.title} · {h.rating}★
                </span>
              ))}
            </div>
          </div>
        )}
      </section>

      {error && (
        <p className="mb-6 rounded-md border border-red-900 bg-red-950/50 p-3 text-sm text-red-300">
          {error}
        </p>
      )}

      <section className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div>
          <h2 className="mb-1 text-sm font-semibold text-neutral-200">
            Personalized (known user)
          </h2>
          <p className="mb-3 text-xs text-neutral-500">
            Uses this user&apos;s real, trained identity embedding + context.
            {known && <> Latency: {known.latency_ms.toFixed(2)}ms.</>}
          </p>
          <ul className="flex flex-col gap-2">
            {known?.recommendations.map((item, i) => (
              <RecommendationCard key={item.movie_id} item={item} rank={i + 1} />
            ))}
          </ul>
        </div>
        <div>
          <h2 className="mb-1 text-sm font-semibold text-neutral-200">
            Cold start (same activity, no identity)
          </h2>
          <p className="mb-3 text-xs text-neutral-500">
            Same recent activity and context, but with the user id forced to
            &quot;unknown&quot; — what a brand-new user would see.
            {cold && <> Latency: {cold.latency_ms.toFixed(2)}ms.</>}
          </p>
          <ul className="flex flex-col gap-2">
            {cold?.recommendations.map((item, i) => (
              <RecommendationCard key={item.movie_id} item={item} rank={i + 1} />
            ))}
          </ul>
        </div>
      </section>

      {loading && <p className="mt-6 text-sm text-neutral-500">Loading recommendations…</p>}

      {known && cold && (
        <p className="mt-8 text-xs text-neutral-500">
          Curious why these two lists are often this close? That&apos;s a real, reproducible
          finding from the notebooks, not a demo artifact — see ADR-004 in{" "}
          <code className="text-neutral-400">docs/ARCHITECTURE_DECISIONS.md</code> and the{" "}
          <Link href="/architecture" className="underline hover:text-neutral-300">
            ablation chart
          </Link>
          .
        </p>
      )}
    </div>
  );
}
