"use client";

import { useEffect, useState } from "react";
import type { EvalMetrics } from "@/lib/api";
import { getEvalMetrics } from "@/lib/api";

const LABELS: Record<string, string> = {
  plain: "Plain two-tower (ids + genres only)",
  enriched: "Enriched (context + user-take + text embeddings)",
  enriched_cold_start: "Enriched, identity forced to “unknown”",
};

function Bar({ label, value, max }: { label: string; value: number; max: number }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="mb-3">
      <div className="mb-1 flex justify-between text-xs text-neutral-400">
        <span>{label}</span>
        <span className="font-mono text-neutral-300">{value.toFixed(4)}</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-neutral-800">
        <div className="h-full rounded-full bg-amber-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function AblationChart() {
  const [metrics, setMetrics] = useState<EvalMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEvalMetrics().then(setMetrics).catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <p className="text-sm text-red-300">
        Could not load eval metrics from the backend ({error}). Is it running?
      </p>
    );
  }
  if (!metrics) return <p className="text-sm text-neutral-500">Loading…</p>;

  const keys = Object.keys(metrics);
  const maxRecall = Math.max(...keys.map((k) => metrics[k].recall_at_k));
  const maxNdcg = Math.max(...keys.map((k) => metrics[k].ndcg_at_k));

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
      <div>
        <h3 className="mb-3 text-sm font-semibold text-neutral-200">Recall@10</h3>
        {keys.map((k) => (
          <Bar key={k} label={LABELS[k] ?? k} value={metrics[k].recall_at_k} max={maxRecall} />
        ))}
      </div>
      <div>
        <h3 className="mb-3 text-sm font-semibold text-neutral-200">NDCG@10</h3>
        {keys.map((k) => (
          <Bar key={k} label={LABELS[k] ?? k} value={metrics[k].ndcg_at_k} max={maxNdcg} />
        ))}
      </div>
      <p className="sm:col-span-2 text-xs text-neutral-500">
        n = {metrics[keys[0]]?.n} held-out users, leave-one-out split. See notebook 4 for the
        full seed-robustness check — these differences are small enough, at this dataset size,
        that a single run shouldn&apos;t be over-read.
      </p>
    </div>
  );
}
