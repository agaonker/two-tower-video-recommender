"use client";

import { useState } from "react";
import type { MovieMeta, RecommendResponse } from "@/lib/api";
import { recommend } from "@/lib/api";
import { ContextControls, type Context } from "@/components/ContextControls";
import { MovieSearchPicker } from "@/components/MovieSearchPicker";
import { RecommendationCard } from "@/components/RecommendationCard";

export default function ColdStartPage() {
  const [selected, setSelected] = useState<MovieMeta[]>([]);
  const [context, setContext] = useState<Context>({
    dayOfWeek: "Friday",
    timeOfDay: "evening",
    device: "Mobile",
  });
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (selected.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      setResult(
        await recommend({
          recent_movie_ids: selected.map((m) => m.movie_id),
          day_of_week: context.dayOfWeek,
          time_of_day: context.timeOfDay,
          device: context.device,
          k: 10,
        }),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="text-2xl font-semibold tracking-tight text-neutral-50">
        Cold-start simulator
      </h1>
      <p className="mt-2 text-sm text-neutral-400">
        This is what the system does for a brand-new, never-before-seen user: no
        trained identity embedding, no rating history — just whatever you tell it
        you&apos;ve recently watched plus the request-time context. Pick a few
        titles below as if you just signed up.
      </p>

      <div className="mt-6 rounded-xl border border-neutral-800 bg-neutral-900/40 p-5">
        <p className="mb-2 text-xs uppercase tracking-wide text-neutral-500">
          What have you recently watched?
        </p>
        <MovieSearchPicker selected={selected} onChange={setSelected} />

        <p className="mt-5 mb-2 text-xs uppercase tracking-wide text-neutral-500">
          Context
        </p>
        <ContextControls value={context} onChange={setContext} />

        <button
          type="button"
          onClick={run}
          disabled={selected.length === 0 || loading}
          className="mt-5 rounded-md bg-amber-600 px-4 py-2 text-sm font-medium text-neutral-950 transition-colors hover:bg-amber-500 disabled:cursor-not-allowed disabled:bg-neutral-700 disabled:text-neutral-400"
        >
          {loading ? "Scoring…" : "Get recommendations"}
        </button>
      </div>

      {error && (
        <p className="mt-6 rounded-md border border-red-900 bg-red-950/50 p-3 text-sm text-red-300">
          {error}
        </p>
      )}

      {result && (
        <section className="mt-8">
          <p className="mb-3 text-xs text-neutral-500">
            Mode: <code className="text-neutral-300">{result.mode}</code> · Latency:{" "}
            {result.latency_ms.toFixed(2)}ms
            {result.note && <> · {result.note}</>}
          </p>
          <ul className="flex flex-col gap-2">
            {result.recommendations.map((item, i) => (
              <RecommendationCard key={item.movie_id} item={item} rank={i + 1} />
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
