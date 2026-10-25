# Plan: wire the real-Merlin model into live, context-aware serving

**Status: Phase B done (2026-10-24).** The real-Merlin model is now live and
servable — `backend/`'s `ARTIFACT_SOURCE=merlin` env var switches the API to
`artifacts/merlin/`, with `backend/app/reclib.py`'s `MerlinArtifactBundle` +
`merlin_query_embedding()` reimplementing its forward pass in plain numpy,
same as the pedagogical path. Verified end to end: booted the real server
with `ARTIFACT_SOURCE=merlin`, hit `/recommend` for known users (real,
non-degenerate recommendations) and confirmed cold-start requests correctly
return `400` (this model has no cold-start mechanism — see the scope table
below). 14/14 backend tests pass (9 pedagogical + 5 merlin). Phase A
(restoring genres/text-embeddings/user-take parity) remains **not done** —
see below, still real follow-up work, not started.

**Decided scope** (2026-10-24): start with today's reduced model (skip
Phase A — genres/text-embeddings/user-take parity deferred, not forgotten;
see the table below for exactly what's missing). Keep serving numpy-only
(Phase B, option 1) — `reclib.py` gets rewritten to match Merlin's real
auto-built architecture rather than bringing TensorFlow into `backend/`.
Phase B is now complete (see below); Phase A and Phase C remain open.

## The scope question, upfront

The real Merlin model trained so far is **not** feature-complete. During the
debugging session, three things were either dropped or never wired in:

| Feature | Pedagogical model (serves the demo today) | Real Merlin model (trained, not served) |
|---|---|---|
| user/item ids | yes | yes |
| day/time-of-day context | yes | yes |
| genres | yes | **no** — dropped after a CPU-mode NVTabular crash |
| item text embeddings | yes | **no** — computed in `train()` but never merged into the training dataframe |
| user-take sequence | yes | **no** — sketched (`WorkflowNode`) but never wired into the workflow or schema |
| cold-start dropout trick | yes, and it works (has user-take to fall back on) | applied to the data, but with nothing to compensate with — currently a weaker, context-only fallback |

So "wire the real model into serving" is really two separable pieces of work:

- **A. Restore feature parity** in `training/modal_app.py` — real engineering,
  not just plumbing. Each of the three missing features has a different fix,
  below.
- **B. Export + serve** whatever model results — the TF/numpy serving
  question discussed last session.

Doing B against today's reduced model is faster and cheaper, but means the
"real Merlin" demo path would visibly do less than the pedagogical one —
worth being upfront about if that's user-facing. Doing A first is more
correct but is realistically more GPU iterations against real error output,
same as the first debugging session.

## Phase A — restore feature parity (if we want it before serving)

**Genres.** The CPU-mode NVTabular crash was specifically about
`merlin.io.Dataset`'s "experimental, extremely limited support" when backed
by plain pandas. The real fix isn't dropping the feature — it's using
NVTabular's GPU-native path: build a `cudf.DataFrame` instead of a pandas
one before constructing the `Dataset`. `cudf` should already be present in
the `merlin-tensorflow` NGC image (it's RAPIDS-based); needs confirming.
This exercises NVTabular's actual intended (GPU) code path for list/multi-hot
columns instead of the admittedly-fragile CPU fallback we hit.

**Item text embeddings.** These are fixed-length (384-dim) float vectors, not
variable-length lists — a different code path from genres, and likely not
affected by the same CPU-mode limitation. Plan: merge
`movies["text_embedding"]` into `ratings`/`train_df` (it currently isn't),
tag the column(s) `Tags.CONTINUOUS` in the schema (NVTabular/Merlin Models
support dense vector continuous features as item-tower input), and confirm
`TwoTowerModel` picks them up automatically via schema tags the same way it
already does for the categorical columns.

**User-take sequence.** Two real options:
1. Finish the `Groupby` + shift/truncate NVTabular op chain sketched (but
   never completed) in the current file — produces a genuine list column,
   which risks the *same* CPU-mode limitation that broke genres, so this
   should wait until the cudf fix above is confirmed working.
2. Side-step list columns entirely: compute `take_1..take_5` as five
   separate scalar categorical columns in pandas (same way notebook 2 does
   it), each `Categorify`'d independently. Simpler, avoids NVTabular list
   handling altogether, but loses the mean-pooling semantics the pedagogical
   model uses — would need either a small custom Merlin Models block to pool
   them, or accepting 5 separate embeddings concatenated instead of pooled
   (a real, if minor, architecture difference worth noting plainly rather
   than quietly calling it equivalent).

Each of these is a genuine iteration loop against a real GPU, same shape as
last session: change code, run `train()`, read the actual error, fix, repeat.
Realistic estimate: several more paid runs, likely a session of its own.

## Phase B — export + serve whatever model results — ✅ DONE

1. ✅ **Cheap research pass, no GPU needed.** Added `inspect_architecture()`
   to `training/modal_app.py` (builds the schema + `TwoTowerModel` on a tiny
   synthetic dataset, no `.fit()`, no GPU). Found, by actually reading the
   object rather than guessing: the real accessor is
   `model.retrieval_block.query_block()`/`.item_block()` (not
   `model.query_encoder`, which doesn't exist); the query tower's embedding
   table is reachable via `query_block().layers[0].layers[0].layers[0].embedding_tables["userId_for_training"]`
   — a name-keyed dict, needed because raw positional weight order flips
   between the query and item tower; the two Dense layers are
   `query_block().layers[0].layers[1].layers` (in order); and
   `cat_features.op.categories[name]` is a **parquet file path**, not an
   in-memory dict — read it, and the row's pandas index is the
   Categorify-assigned integer id. Also confirmed, unexpectedly:
   `TwoTowerModel` only wires in schema columns tagged `USER_ID`/`ITEM_ID` —
   day_of_week/time_of_day were in the schema the whole time but **never
   actually used by the model**, which is why they're gone from
   `train()` now (see the updated scope table above).
2. ✅ **Added `MerlinArtifactBundle` + `merlin_query_embedding()`** to
   `backend/app/reclib.py` — a separate bundle/forward-pass from the
   pedagogical `ArtifactBundle`/`query_embedding()`, since the real model has
   fewer inputs (no user-take, no context) and uses TensorFlow's weight
   convention (`y = x @ kernel + bias`, not PyTorch's transposed one).
   `recommend()` (the dot-product scorer) was generalized to take raw arrays
   instead of a whole bundle, so both paths share it unchanged.
3. ✅ **Fixed `train()`'s export section** using the confirmed attribute
   path and vocab-extraction mechanism from step 1.
4. ✅ **`feature_vocab.json`'s export** now matches the actual reduced input
   set (`user_vocab`, `movie_vocab`, `embedding_dims` — no day/tod/user-take
   vocabs, since this model doesn't have them).
5. ✅ **Re-ran `train()` end to end** (3 iterations: first hit a
   `np.savez` pickling error because `model.item_embeddings()` returns a
   lazy `merlin.io.Dataset`, not a plain array — fixed by `.to_ddf().compute()`;
   second iteration had item embeddings contaminated with `title`/`genres`
   text columns from an under-specified exclude filter — fixed). Final run:
   clean, `recall_at_10 ≈ 0.185`, `ndcg_at_10 ≈ 0.103`. Downloaded via
   `modal volume get`, sanity-checked locally (non-degenerate recs across 5
   real users, correct 404 handling for unknown users) before touching the
   backend at all.
6. ✅ **`ARTIFACT_SOURCE` env var** added to `backend/app/main.py`
   (`pedagogical` default, `merlin` opt-in), reading from `artifacts/merlin/`
   — the committed `artifacts/` the pedagogical model serves was never
   touched. Cold-start requests against the `merlin` source correctly return
   `400` (this model has no cold-start mechanism) rather than silently
   degrading. Verified by booting the real server with
   `ARTIFACT_SOURCE=merlin` and hitting every path with curl.

## Phase C — small frontend touch (optional)

A toggle or label on `/architecture` showing which model is currently being
served (pedagogical vs. real-Merlin) — mostly storytelling value once both
exist side by side.

## Cost/risk honesty

Phase A alone is realistically comparable in size to the entire first
debugging session (multiple paid GPU iterations per feature, 3 features).
Phase B is smaller but not trivial — step 2 (rewriting `reclib.py` against
Merlin's real architecture) is the part most likely to need its own iteration
loop. This is genuinely a multi-session piece of work, not an afternoon fix.
