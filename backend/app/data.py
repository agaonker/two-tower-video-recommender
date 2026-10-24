"""Read-only access to the raw MovieLens files, for user history lookups and
movie metadata display. No model logic lives here — see reclib.py for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MovieLensData:
    ratings: pd.DataFrame
    movies: pd.DataFrame


def load_movielens(data_dir: Path) -> MovieLensData:
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv").set_index("movieId")
    return MovieLensData(ratings=ratings, movies=movies)


def day_and_time_of_day_now() -> tuple[str, str]:
    import datetime

    now = datetime.datetime.now()
    day = now.strftime("%A")
    hour = now.hour
    if 5 <= hour < 12:
        tod = "morning"
    elif 12 <= hour < 17:
        tod = "afternoon"
    elif 17 <= hour < 22:
        tod = "evening"
    else:
        tod = "night"
    return day, tod


def movie_meta(data: MovieLensData, movie_id: int) -> dict | None:
    if movie_id not in data.movies.index:
        return None
    row = data.movies.loc[movie_id]
    return {"movie_id": movie_id, "title": row["title"], "genres": row["genres"]}


def sample_users(data: MovieLensData, limit: int = 50) -> list[dict]:
    counts = data.ratings.groupby("userId").size().sort_values(ascending=False)
    return [{"user_id": int(uid), "num_ratings": int(n)} for uid, n in counts.head(limit).items()]


def user_recent_ratings(data: MovieLensData, user_id: int, limit: int = 10) -> list[dict]:
    rows = (
        data.ratings[data.ratings["userId"] == user_id]
        .sort_values("timestamp", ascending=False)
        .head(limit)
    )
    out = []
    for _, row in rows.iterrows():
        meta = movie_meta(data, int(row["movieId"])) or {}
        out.append(
            {
                "movie_id": int(row["movieId"]),
                "title": meta.get("title"),
                "genres": meta.get("genres"),
                "rating": float(row["rating"]),
            }
        )
    return out


def search_movies(data: MovieLensData, query: str, limit: int = 20) -> list[dict]:
    if not query.strip():
        return []
    matches = data.movies[data.movies["title"].str.contains(query, case=False, na=False)]
    return [
        {"movie_id": int(movie_id), "title": row["title"], "genres": row["genres"]}
        for movie_id, row in matches.head(limit).iterrows()
    ]


def user_recent_movie_ids(data: MovieLensData, user_id: int, limit: int) -> list[int]:
    rows = (
        data.ratings[data.ratings["userId"] == user_id]
        .sort_values("timestamp", ascending=False)
        .head(limit)
    )
    return rows["movieId"].astype(int).tolist()
