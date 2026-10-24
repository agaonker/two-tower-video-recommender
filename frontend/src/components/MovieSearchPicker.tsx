"use client";

import { useState } from "react";
import type { MovieMeta } from "@/lib/api";
import { searchMovies } from "@/lib/api";

export function MovieSearchPicker({
  selected,
  onChange,
}: {
  selected: MovieMeta[];
  onChange: (next: MovieMeta[]) => void;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MovieMeta[]>([]);
  const [searching, setSearching] = useState(false);

  async function onSearch(q: string) {
    setQuery(q);
    if (q.trim().length < 2) {
      setResults([]);
      return;
    }
    setSearching(true);
    try {
      setResults(await searchMovies(q, 8));
    } finally {
      setSearching(false);
    }
  }

  function add(movie: MovieMeta) {
    if (selected.some((m) => m.movie_id === movie.movie_id)) return;
    onChange([...selected, movie]);
    setQuery("");
    setResults([]);
  }

  function remove(movieId: number) {
    onChange(selected.filter((m) => m.movie_id !== movieId));
  }

  return (
    <div>
      <input
        className="w-full rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-600"
        placeholder="Search for a movie you've recently watched…"
        value={query}
        onChange={(e) => onSearch(e.target.value)}
      />
      {searching && <p className="mt-1 text-xs text-neutral-500">Searching…</p>}
      {results.length > 0 && (
        <ul className="mt-2 flex flex-col gap-1 rounded-md border border-neutral-800 bg-neutral-900 p-1">
          {results.map((m) => (
            <li key={m.movie_id}>
              <button
                type="button"
                onClick={() => add(m)}
                className="w-full rounded-sm px-2 py-1.5 text-left text-sm text-neutral-200 hover:bg-neutral-800"
              >
                {m.title} <span className="text-neutral-500">· {m.genres}</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      {selected.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {selected.map((m) => (
            <span
              key={m.movie_id}
              className="flex items-center gap-1.5 rounded-full border border-amber-800 bg-amber-950/40 px-3 py-1 text-xs text-amber-200"
            >
              {m.title}
              <button
                type="button"
                onClick={() => remove(m.movie_id)}
                className="text-amber-500 hover:text-amber-300"
                aria-label={`remove ${m.title}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
