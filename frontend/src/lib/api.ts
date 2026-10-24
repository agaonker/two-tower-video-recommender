const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type UserSummary = {
  user_id: number;
  num_ratings: number;
};

export type RatedMovie = {
  movie_id: number;
  title: string | null;
  genres: string | null;
  rating: number;
};

export type MovieMeta = {
  movie_id: number;
  title: string;
  genres: string;
};

export type RecommendationItem = {
  movie_id: number;
  title: string | null;
  genres: string | null;
  score: number;
};

export type RecommendResponse = {
  mode: "known_user" | "cold_start";
  recommendations: RecommendationItem[];
  latency_ms: number;
  context_used: { day_of_week: string; time_of_day: string };
  note: string | null;
};

export type RecommendRequest = {
  user_id?: number;
  recent_movie_ids?: number[];
  day_of_week?: string;
  time_of_day?: string;
  device?: string;
  k?: number;
};

export type EvalMetrics = Record<string, { recall_at_k: number; ndcg_at_k: number; n: number }>;

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function listUsers(limit = 50): Promise<UserSummary[]> {
  return getJson<UserSummary[]>(`/users?limit=${limit}`);
}

export async function getUserHistory(userId: number, limit = 10): Promise<RatedMovie[]> {
  return getJson<RatedMovie[]>(`/users/${userId}?limit=${limit}`);
}

export async function searchMovies(query: string, limit = 20): Promise<MovieMeta[]> {
  if (!query.trim()) return [];
  return getJson<MovieMeta[]>(`/movies/search?q=${encodeURIComponent(query)}&limit=${limit}`);
}

export async function getEvalMetrics(): Promise<EvalMetrics> {
  return getJson<EvalMetrics>("/eval/metrics");
}

export async function recommend(req: RecommendRequest): Promise<RecommendResponse> {
  const res = await fetch(`${API_URL}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`POST /recommend failed: ${res.status} ${body}`);
  }
  return res.json() as Promise<RecommendResponse>;
}

export const DAYS_OF_WEEK = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
];

export const TIMES_OF_DAY = ["morning", "afternoon", "evening", "night"] as const;

export const DEVICES = ["TV", "Mobile", "Tablet", "Laptop"] as const;
