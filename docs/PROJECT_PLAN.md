# Project Plan & Status

A complete record of what this project set out to build, what's actually done
(verified by running it, not just written), and what's explicitly deferred.
Checkboxes reflect real status as of the last working session.

## 1. Data

- [x] Source MovieLens from Hugging Face Hub, not a raw GroupLens download —
      used `johnidouglas/movie_lens_small_latest` (`ml-latest-small`: 100,836
      ratings, 9,742 movies, 610 users, with real tags and TMDb/IMDb links).
      Confirmed exact match to canonical MovieLens file structure.
- [x] Committed to `data/raw/ml-latest-small/` (no Git LFS needed, a few MB).

## 2. Notebooks (the learning spine)

- [x] `01_explore_movielens.ipynb` — sparsity, real context signal (day/hour
      from timestamps), tag coverage (~16% of movies), genre distribution.
      Executed clean, real output.
- [x] `02_nvtabular_features.ipynb` — Categorify, day/time-of-day derivation,
      the "user-take" sequence feature, leave-one-out split, the cold-start
      training trick (15% of rows get user-id dropped). Pandas
      reimplementation, not the real NVTabular library — see ADR-007.
      Executed clean, writes `data/processed/`.
- [x] `03_item_text_embeddings.ipynb` — `all-MiniLM-L6-v2` sentence embeddings
      per movie (tags, falling back to title/genres). Sanity-checked with a
      nearest-neighbor query (Toy Story → Toy Story 2/3, Bug's Life — sensible).
      Executed clean, writes `data/processed/item_text_embeddings.npz`.
- [x] `04_two_tower_cold_start.ipynb` — plain-PyTorch two-tower model, trained
      `plain` vs. `enriched` vs. `enriched_cold_start` variants, same eval
      protocol. Re-ran the ablation across 4 random seeds after the first
      result looked too surprising to trust blind (cold-start matched or beat
      the full model consistently — written up as a real, scale-dependent
      finding, not asserted). **Writes the `artifacts/` actually committed to
      this repo and used by `backend/`.**

## 3. Architecture decision journal

- [x] `docs/ARCHITECTURE_DECISIONS.md` scaffolded with 7 ADR entries.
- [x] ADR-007 (the dependency-hell / numpy-numba-ABI finding) pre-filled —
      factual, not a judgment call.
- [ ] ADR-001 through ADR-006 are intentionally left as open questions for
      you to answer after reading the notebooks — that's the mechanism for
      "forming your own opinions," not an oversight.

## 4. Backend (`backend/`, FastAPI)

- [x] `reclib.py` — pure-numpy reimplementation of the query tower's forward
      pass (no ML framework dependency at serving time). Exhaustive
      dot-product scoring over precomputed item embeddings.
- [x] Endpoints: `/health`, `/users`, `/users/{id}`, `/movies/{id}`,
      `/movies/search`, `POST /recommend` (handles both known-user and
      cold-start modes via one endpoint), `/eval/metrics`.
- [x] 9 pytest tests passing; ruff + mypy clean.
- [x] Manually verified: booted the real server, hit every endpoint with
      curl, confirmed context changes actually change rankings.

## 5. Frontend (`frontend/`, Next.js)

- [x] `/` — user picker, context controls (day/time-of-day; device is an
      explicitly-labeled non-functional illustration), side-by-side
      known-user vs. cold-start comparison for the same user.
- [x] `/cold-start` — movie search picker, build a synthetic "recent
      activity" profile with no identity, get live recommendations.
- [x] `/architecture` — both Mermaid diagrams rendered live, the ablation
      chart pulling real numbers from `GET /eval/metrics`, the scale-tradeoffs
      table, credits.
- [x] Verified via headless browser against the production build (`next
      start`) — picked a user, watched recs update, ran the cold-start flow,
      screenshotted the architecture page to confirm diagrams/chart render.
      (Note: `next dev`'s HMR websocket didn't hydrate correctly under the
      headless test browser specifically — not an app bug, doesn't affect
      normal browser use.)
- [x] `npm run lint` and `npm run build` clean.

## 6. Documentation

- [x] `README.md` — full rewrite: project framed as my own work (Joyn
      credited as inspiration, not the organizing narrative), both Mermaid
      diagrams, scale-tradeoffs table, and a complete step-by-step
      "Getting started, end to end" walkthrough (prerequisites → install →
      run backend → run frontend → optional notebooks → optional full
      regen → optional Modal production path, with accurate status labels
      on each).

## 7. Production-fidelity training (`training/modal_app.py`) — the real Merlin stack on a Modal GPU

This was debugged live against a real Modal GPU account, iterating on actual
error output rather than guessed API calls. In order, what broke and what
fixed it:

- [x] **numpy/numba ABI conflict**: `pip_install("sentence-transformers", ...)`
      on top of the pinned NGC image upgraded numpy underneath the image's
      pre-compiled `numba` binary, breaking `merlin.models.tf` entirely.
      **Fix**: removed sentence-transformers from the Merlin image; the
      training job now loads the text embeddings notebook 3 already computed
      instead of recomputing them in a container that shouldn't need them.
- [x] **`numba.cuda` segfault with a real GPU attached**: traced via a
      cheap CPU-only validation function, then confirmed the crash was
      GPU-driver-related, not GPU-absence-related. **Fix**: `NUMBA_DISABLE_CUDA=1`
      — TensorFlow uses its own CUDA bindings, so this doesn't cost GPU access
      for actual training.
- [x] **Missing `genres` column**: wasn't merged from `movies` into `ratings`
      before building the NVTabular dataset. **Fixed** with a merge.
- [x] **`TwoTowerModel()` API**: needs an explicit `query_tower=` Block
      argument, not just schema + embedding_dim. **Fixed**, mirrored the
      pedagogical model's 64-wide hidden layer.
- [x] **Multi-hot `genres` column crashing `model.evaluate()`**: traced to
      NVTabular's own documented CPU-mode limitation ("experimental...
      extremely limited support"). **Fix**: dropped `genres` from the real
      Merlin workflow for now (the pedagogical model still covers it).
- [x] **Full successful run achieved**: `workflow.fit()` →
      `workflow.transform()` → `model.fit()` → `model.evaluate()` all
      completed. Real numbers: `recall_at_10 ≈ 0.16–0.18`,
      `ndcg_at_10 ≈ 0.08–0.09` — notably better than the pedagogical model's
      ~0.015/0.005. **This is the proof that the real NVTabular + Merlin
      Models stack actually works end-to-end in this environment.**
- [x] **Exported the trained model into `backend/`-servable artifacts.**
      Full plan and execution log: `docs/PLAN_LIVE_MERLIN_SERVING.md`. Short
      version: scoped to a reduced model first (just `{user_id, item_id}` —
      see that doc's scope table for exactly what's deferred), kept serving
      numpy-only. A cheap, GPU-free `inspect_architecture()` research pass
      found the real attribute path (`model.retrieval_block.query_block()`,
      not `model.query_encoder`) and the real vocab-extraction mechanism
      (a parquet file path, not a dict) — confirmed by reading the actual
      object, not guessed. The subsequent real GPU run succeeded on the
      first attempt at weight extraction (shapes matched exactly what the
      research pass predicted); two further iterations fixed an
      `item_embeddings()` serialization issue (it returns a lazy
      `merlin.io.Dataset`, not a plain array) and a column-filtering bug.
      Final artifacts live in `artifacts/merlin/` (committed, 2.4MB),
      selectable via `backend/`'s `ARTIFACT_SOURCE=merlin` env var — the
      default `artifacts/` the pedagogical model serves was never touched.
      14/14 backend tests pass. Cold-start requests against this source
      correctly return `400` (this reduced model has no cold-start
      mechanism) rather than degrading silently.

**Current state**: the demo serves the pedagogical artifacts by default
(full feature set: context, user-take, cold-start) and can be switched to
the real-Merlin ones (`ARTIFACT_SOURCE=merlin`, known-users only, no
cold-start) — both real, both tested, kept deliberately separate.

## 8. Not done

- [x] ~~Git commits~~ — done, pushed to `agaonker/two-tower-video-recommender`.
- [x] ~~Pushed to GitHub~~ — done.
- [x] ~~Real-Merlin model wired into live context-aware serving~~ — done for
      the reduced model (see §7); full feature parity is still open, below.
- [ ] ADR-001 through ADR-006 — yours to fill in.
- [ ] Phase A of `docs/PLAN_LIVE_MERLIN_SERVING.md`: restoring genres, item
      text embeddings, and the user-take sequence feature to the real-Merlin
      model (currently reduced to just user/item ids). Real engineering, not
      plumbing — see that doc for the three different fixes needed.
- [ ] Phase C of the same doc: a frontend toggle/label showing which
      artifact source is currently being served.
- [ ] Optional nice-to-haves mentioned in the README but not built: TMDb
      poster integration, a Modal scheduled retrain.
