import type { RecommendationItem } from "@/lib/api";

function scoreToPercent(score: number): number {
  // Dot products of L2-normalized vectors live in [-1, 1]; map to a 0-100 bar.
  return Math.round(((score + 1) / 2) * 100);
}

export function RecommendationCard({
  item,
  rank,
}: {
  item: RecommendationItem;
  rank: number;
}) {
  const pct = scoreToPercent(item.score);
  return (
    <li className="flex items-center gap-3 rounded-lg border border-neutral-800 bg-neutral-900/60 p-3">
      <span className="w-5 shrink-0 text-right text-sm text-neutral-500">{rank}</span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-neutral-100">{item.title ?? "Unknown title"}</p>
        <p className="truncate text-xs text-neutral-500">{item.genres ?? ""}</p>
        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-neutral-800">
          <div
            className="h-full rounded-full bg-amber-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
      <span className="shrink-0 font-mono text-xs text-neutral-400">{item.score.toFixed(3)}</span>
    </li>
  );
}
