from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app import data, reclib

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "ml-latest-small"

# "pedagogical" (default) serves notebooks/04's PyTorch-trained artifacts —
# full feature set (context, user-take, cold-start). "merlin" serves
# training/modal_app.py's real NVTabular + Merlin Models run — a reduced
# model (just user_id/item_id, no cold-start mechanism) kept in a separate
# artifacts/merlin/ directory precisely so switching this can't regress the
# working demo. See docs/PLAN_LIVE_MERLIN_SERVING.md.
ArtifactSource = Literal["pedagogical", "merlin"]
ARTIFACT_SOURCE: ArtifactSource = (
    "merlin" if os.environ.get("ARTIFACT_SOURCE") == "merlin" else "pedagogical"
)
ARTIFACTS_DIR = (
    REPO_ROOT / "artifacts" / "merlin" if ARTIFACT_SOURCE == "merlin" else REPO_ROOT / "artifacts"
)

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _state["bundle"] = (
        reclib.load_merlin_artifacts(ARTIFACTS_DIR)
        if ARTIFACT_SOURCE == "merlin"
        else reclib.load_artifacts(ARTIFACTS_DIR)
    )
    _state["ml_data"] = data.load_movielens(DATA_DIR)
    yield
    _state.clear()


app = FastAPI(
    title="Two-Tower Video Recommender API",
    description=(
        "Serves recommendations from a precomputed query tower + item "
        "embeddings. CPU-only, no ML framework installed — see "
        "backend/app/reclib.py and docs/ARCHITECTURE_DECISIONS.md (ADR-007)."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_loaded() -> tuple[
    reclib.ArtifactBundle | reclib.MerlinArtifactBundle, data.MovieLensData
]:
    bundle = _state.get("bundle")
    ml_data = _state.get("ml_data")
    if bundle is None or ml_data is None:
        raise HTTPException(status_code=503, detail="artifacts not loaded yet")
    assert isinstance(bundle, (reclib.ArtifactBundle, reclib.MerlinArtifactBundle))
    assert isinstance(ml_data, data.MovieLensData)
    return bundle, ml_data


class UserSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: int
    num_ratings: int


class RatedMovie(BaseModel):
    model_config = ConfigDict(extra="forbid")
    movie_id: int
    title: str | None
    genres: str | None
    rating: float


class MovieMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    movie_id: int
    title: str
    genres: str


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: int | None = None
    recent_movie_ids: list[int] | None = Field(
        default=None,
        description="Required when user_id is omitted — the cold-start path.",
    )
    day_of_week: str | None = Field(
        default=None, description="e.g. 'Monday'. Defaults to the server's current day."
    )
    time_of_day: str | None = Field(
        default=None,
        description="One of morning/afternoon/evening/night. Defaults to the server's current time.",
    )
    device: str | None = Field(
        default=None,
        description=(
            "UI-only. MovieLens has no device field, so this is not a modeled "
            "signal — included to illustrate where real device context would "
            "plug into this same architecture. See the README's note on "
            "simulated context."
        ),
    )
    k: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _one_identity_source(self) -> RecommendRequest:
        if self.user_id is None and not self.recent_movie_ids:
            raise ValueError("provide either user_id or a non-empty recent_movie_ids")
        return self


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    movie_id: int
    title: str | None
    genres: str | None
    score: float


class RecommendResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["known_user", "cold_start"]
    source: ArtifactSource
    recommendations: list[RecommendationItem]
    latency_ms: float
    context_used: dict[str, str]
    note: str | None = None


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": "bundle" in _state}


@app.get("/users", response_model=list[UserSummary])
def list_users(limit: int = 50) -> list[UserSummary]:
    _, ml = _require_loaded()
    return [UserSummary(**u) for u in data.sample_users(ml, limit=limit)]


@app.get("/users/{user_id}", response_model=list[RatedMovie])
def user_history(user_id: int, limit: int = 10) -> list[RatedMovie]:
    _, ml = _require_loaded()
    history = data.user_recent_ratings(ml, user_id, limit=limit)
    if not history:
        raise HTTPException(status_code=404, detail=f"no ratings found for user_id={user_id}")
    return [RatedMovie(**h) for h in history]


@app.get("/movies/search", response_model=list[MovieMeta])
def movie_search(q: str, limit: int = 20) -> list[MovieMeta]:
    _, ml = _require_loaded()
    return [MovieMeta(**m) for m in data.search_movies(ml, q, limit=limit)]


@app.get("/movies/{movie_id}", response_model=MovieMeta)
def movie_detail(movie_id: int) -> MovieMeta:
    _, ml = _require_loaded()
    meta = data.movie_meta(ml, movie_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"no movie with movie_id={movie_id}")
    return MovieMeta(**meta)


@app.get("/eval/metrics")
def eval_metrics() -> dict:
    import json

    path = ARTIFACTS_DIR / "eval_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="eval_metrics.json not found in artifacts/")
    with open(path) as f:
        return json.load(f)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    bundle, ml = _require_loaded()
    started = time.perf_counter()

    default_day, default_tod = data.day_and_time_of_day_now()
    day_of_week = req.day_of_week or default_day
    time_of_day = req.time_of_day or default_tod

    if isinstance(bundle, reclib.MerlinArtifactBundle):
        # This artifact source's model only has {user_id, item_id} — no
        # context, no user-take, so no cold-start mechanism at all (see
        # docs/PLAN_LIVE_MERLIN_SERVING.md). Reject rather than silently
        # produce something misleading.
        if req.user_id is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "cold-start is not supported for the 'merlin' artifact source — "
                    "this model has no context or user-take feature to fall back on. "
                    "Use ARTIFACT_SOURCE=pedagogical for the cold-start demo."
                ),
            )
        user_idx = bundle.user_vocab.get(str(req.user_id), reclib.UNK)
        if user_idx == reclib.UNK:
            raise HTTPException(
                status_code=404, detail=f"unknown user_id={req.user_id} (not in training vocab)"
            )
        recent_ids = data.user_recent_movie_ids(ml, req.user_id, limit=10)
        user_vec = reclib.merlin_query_embedding(bundle, user_idx)
        mode: Literal["known_user", "cold_start"] = "known_user"
        note = (
            "Served by the 'merlin' artifact source — real NVTabular + Merlin "
            "Models training, but a reduced model (no context, no user-take). "
            "day_of_week/time_of_day below were not used by this model."
        )
    else:
        day_idx = bundle.day_vocab.get(day_of_week, reclib.UNK)
        tod_idx = bundle.tod_vocab.get(time_of_day, reclib.UNK)
        note = None
        if req.user_id is not None:
            mode = "known_user"
            user_idx = bundle.user_vocab.get(str(req.user_id), reclib.UNK)
            if user_idx == reclib.UNK:
                raise HTTPException(
                    status_code=404, detail=f"unknown user_id={req.user_id} (not in training vocab)"
                )
            recent_ids = data.user_recent_movie_ids(ml, req.user_id, limit=bundle.user_take_k)
        else:
            mode = "cold_start"
            user_idx = reclib.UNK
            recent_ids = req.recent_movie_ids or []
            note = (
                "No user_id was given, so this request used only the recent-movies "
                "list and context — the same mechanism that lets a brand-new user "
                "get recommendations. See docs/ARCHITECTURE_DECISIONS.md (ADR-004)."
            )
        user_take = reclib.build_user_take(bundle, recent_ids)
        user_vec = reclib.query_embedding(bundle, user_idx, day_idx, tod_idx, user_take)

    exclude = set(recent_ids)
    top = reclib.recommend(
        bundle.item_embeddings, bundle.item_movie_ids, user_vec, k=req.k, exclude_movie_ids=exclude
    )

    recs = []
    for movie_id, score in top:
        meta = data.movie_meta(ml, movie_id) or {}
        recs.append(
            RecommendationItem(
                movie_id=movie_id, title=meta.get("title"), genres=meta.get("genres"), score=score
            )
        )

    latency_ms = (time.perf_counter() - started) * 1000
    return RecommendResponse(
        mode=mode,
        source=ARTIFACT_SOURCE,
        recommendations=recs,
        latency_ms=latency_ms,
        context_used={"day_of_week": day_of_week, "time_of_day": time_of_day},
        note=note,
    )
