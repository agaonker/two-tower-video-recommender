"""Pure-numpy serving: the query tower's forward pass, re-implemented from
exported weights, with no training-framework dependency at all.

Why this exists instead of just loading a PyTorch/TF model: see ADR-007 in
docs/ARCHITECTURE_DECISIONS.md. The short version is that the libraries used
to *train* this model (whether the pedagogical PyTorch version in
notebooks/04, or the real NVTabular/Merlin Models stack in
training/modal_app.py) don't need to be installed wherever this model is
*served*. At serving time, all that's actually needed is: the item tower's
precomputed output for every item, the query tower's weights, and the same
vocab mapping used to fit those weights. Everything here operates on that
minimal, framework-free artifact set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArtifactBundle:
    item_movie_ids: (
        np.ndarray
    )  # (n_movies_with_embedding,) raw MovieLens movieId, in embedding order
    item_embeddings: np.ndarray  # (n_movies_with_embedding, tower_out_dim)
    user_embedding: np.ndarray  # (n_users, emb_user)
    movie_embedding: np.ndarray  # (n_movies, emb_movie) — shared table, indexed by movie_idx
    day_embedding: np.ndarray  # (n_days, emb_day)
    tod_embedding: np.ndarray  # (n_tods, emb_tod)
    mlp_0_weight: np.ndarray
    mlp_0_bias: np.ndarray
    mlp_2_weight: np.ndarray
    mlp_2_bias: np.ndarray
    user_vocab: dict[str, int]  # raw userId (str) -> user_idx
    movie_vocab: dict[str, int]  # raw movieId (str) -> movie_idx
    day_vocab: dict[str, int]  # day name -> day_idx
    tod_vocab: dict[str, int]  # time-of-day bucket -> tod_idx
    user_take_k: int


def load_artifacts(artifacts_dir: Path) -> ArtifactBundle:
    qt = np.load(artifacts_dir / "query_tower_weights.npz")
    ie = np.load(artifacts_dir / "item_embeddings.npz")
    with open(artifacts_dir / "feature_vocab.json") as f:
        vocab = json.load(f)

    return ArtifactBundle(
        item_movie_ids=ie["movie_ids"],
        item_embeddings=ie["embeddings"],
        user_embedding=qt["user_embedding"],
        movie_embedding=qt["movie_embedding"],
        day_embedding=qt["day_embedding"],
        tod_embedding=qt["tod_embedding"],
        mlp_0_weight=qt["mlp_0_weight"],
        mlp_0_bias=qt["mlp_0_bias"],
        mlp_2_weight=qt["mlp_2_weight"],
        mlp_2_bias=qt["mlp_2_bias"],
        user_vocab=vocab["user_vocab"],
        movie_vocab=vocab["movie_vocab"],
        day_vocab=vocab["day_vocab"],
        tod_vocab=vocab["tod_vocab"],
        user_take_k=vocab["user_take_k"],
    )


UNK = 0


def _masked_mean_1d(vectors: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """vectors: (L, D) rows already gathered for this one request; idxs: (L,)
    the raw indices used to gather them, so padding (idx == UNK) can be
    excluded from the mean rather than diluting it."""
    mask = (idxs != UNK).astype(vectors.dtype)
    if mask.sum() == 0:
        return np.zeros(vectors.shape[1], dtype=vectors.dtype)
    return (vectors * mask[:, None]).sum(axis=0) / mask.sum()


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def query_embedding(
    bundle: ArtifactBundle,
    user_idx: int,
    day_idx: int,
    tod_idx: int,
    user_take_idxs: list[int],
) -> np.ndarray:
    """Re-implements QueryTower.forward from notebooks/04 in plain numpy.

    Matches, in order: user embedding lookup, user-take mean-pool (over the
    *shared* movie embedding table), day/time-of-day embedding lookups,
    concat, Linear -> ReLU -> Linear, L2-normalize.
    """
    u = bundle.user_embedding[user_idx]

    take_idxs = np.asarray(user_take_idxs, dtype=np.int64)
    take_vectors = bundle.movie_embedding[take_idxs]
    take_pooled = _masked_mean_1d(take_vectors, take_idxs)

    d = bundle.day_embedding[day_idx]
    t = bundle.tod_embedding[tod_idx]

    x = np.concatenate([u, take_pooled, d, t])
    h = np.maximum(0.0, x @ bundle.mlp_0_weight.T + bundle.mlp_0_bias)
    out = h @ bundle.mlp_2_weight.T + bundle.mlp_2_bias
    return _l2_normalize(out)


def recommend(
    item_embeddings: np.ndarray,
    item_movie_ids: np.ndarray,
    user_vec: np.ndarray,
    k: int,
    exclude_movie_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Exhaustive dot-product scoring against every precomputed item
    embedding — the "small enough catalog, score everything" approach this
    whole project is built around. See the scale-tradeoffs table in the
    README for when this specific choice stops being the right one.

    Takes the raw arrays rather than a whole bundle so it works the same way
    for either artifact source (pedagogical or merlin — see
    MerlinArtifactBundle below)."""
    scores = item_embeddings @ user_vec  # (n_movies,)
    order = np.argsort(-scores)
    results: list[tuple[int, float]] = []
    for idx in order:
        movie_id = int(item_movie_ids[idx])
        if movie_id < 0:
            continue  # reserved UNK slot, not a real movie
        if exclude_movie_ids and movie_id in exclude_movie_ids:
            continue
        results.append((movie_id, float(scores[idx])))
        if len(results) >= k:
            break
    return results


def build_user_take(bundle: ArtifactBundle, recent_movie_ids: list[int]) -> list[int]:
    """Most-recent-first, truncated/padded to user_take_k — mirrors the
    construction in notebooks/02, just applied at request time instead of to
    historical training rows."""
    idxs = [bundle.movie_vocab.get(str(mid), UNK) for mid in recent_movie_ids]
    idxs = idxs[: bundle.user_take_k]
    idxs += [UNK] * (bundle.user_take_k - len(idxs))
    return idxs


@dataclass(frozen=True)
class MerlinArtifactBundle:
    """The real NVTabular + Merlin Models training path (training/modal_app.py),
    exported to the same kind of numpy-only artifact set as ArtifactBundle
    above — but a deliberately smaller model. See
    docs/PLAN_LIVE_MERLIN_SERVING.md: this one only has {user_id, item_id}
    plus the cold-start id-dropout trick — no context, no user-take, no
    genres, no text embeddings (those are real follow-up work, not done
    here). That means there's no cold-start *mechanism* for this bundle —
    only a known user_id can be scored; see main.py's handling.

    Also: these weights are TensorFlow's convention (kernel shape
    (in_dim, out_dim), y = x @ kernel + bias) — NOT PyTorch's
    (out_dim, in_dim) convention that ArtifactBundle's weights use. Do not
    reuse query_embedding() above for this bundle.
    """

    item_movie_ids: np.ndarray  # (n_movies,) raw MovieLens movieId, in embedding order
    item_embeddings: np.ndarray  # (n_movies, tower_out_dim)
    user_embedding: np.ndarray  # (n_users, emb_user) — TF Embedding table
    mlp_0_weight: np.ndarray  # (in_dim, hidden_dim) — TF convention
    mlp_0_bias: np.ndarray
    mlp_2_weight: np.ndarray  # (hidden_dim, tower_out_dim) — TF convention
    mlp_2_bias: np.ndarray
    user_vocab: dict[str, int]  # raw userId (str) -> user_idx


def load_merlin_artifacts(artifacts_dir: Path) -> MerlinArtifactBundle:
    qt = np.load(artifacts_dir / "query_tower_weights.npz")
    ie = np.load(artifacts_dir / "item_embeddings.npz")
    with open(artifacts_dir / "feature_vocab.json") as f:
        vocab = json.load(f)

    return MerlinArtifactBundle(
        item_movie_ids=ie["movie_ids"],
        item_embeddings=ie["embeddings"],
        user_embedding=qt["user_embedding"],
        mlp_0_weight=qt["mlp_0_weight"],
        mlp_0_bias=qt["mlp_0_bias"],
        mlp_2_weight=qt["mlp_2_weight"],
        mlp_2_bias=qt["mlp_2_bias"],
        user_vocab=vocab["user_vocab"],
    )


def merlin_query_embedding(bundle: MerlinArtifactBundle, user_idx: int) -> np.ndarray:
    """Re-implements this model's query tower forward pass: user embedding
    lookup -> Dense(relu) -> Dense -> L2-normalize. No context, no
    user-take — this bundle's model doesn't have them (see the class
    docstring). TF weight convention: y = x @ kernel + bias, no transpose
    (contrast with query_embedding() above, which transposes for PyTorch's
    convention)."""
    u = bundle.user_embedding[user_idx]
    h = np.maximum(0.0, u @ bundle.mlp_0_weight + bundle.mlp_0_bias)
    out = h @ bundle.mlp_2_weight + bundle.mlp_2_bias
    return _l2_normalize(out)
