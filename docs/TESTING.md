# Testing this yourself

Every command below is copy-paste-able from the repo root
(`/Users/ashish/ws/recommendation-system-merlin`), in order. Nothing here
needs a GPU or a Modal account — both artifact sets (`artifacts/` and
`artifacts/merlin/`) are already committed.

## 1. Install

```bash
uv sync                       # Python env: backend, notebooks, training client
cd frontend && npm install && cd ..
```

## 2. Automated checks (fast, no servers needed)

```bash
# Backend: type-check, lint, format-check, full test suite (14 tests —
# 9 against the pedagogical model, 5 against the real-Merlin one)
uv run mypy backend/app/ backend/tests/
uv run ruff check backend/ training/
uv run ruff format backend/ training/ --check
uv run pytest backend/tests/ -v

# Frontend: lint + production build
cd frontend
npm run lint
npm run build
cd ..
```

Expect: mypy "Success", ruff "All checks passed", pytest "14 passed",
frontend lint "No issues found", build ends with a route table (`○ /`,
`○ /architecture`, `○ /cold-start`).

## 3. Run the default demo (pedagogical model — full feature set)

Two terminals:

```bash
# Terminal 1 — backend on :8000
cd backend
uv run uvicorn app.main:app --reload
```

```bash
# Terminal 2 — frontend on :3000
cd frontend
npm run dev
```

Open `http://localhost:3000`:
- **`/`** — pick a user from the dropdown, watch their real rating history
  appear, compare "Personalized" vs. "Cold start (same activity, no
  identity)" side by side. Change the day-of-week/time-of-day controls and
  confirm the recommendation lists actually change.
- **`/cold-start`** — search for a movie (try "Toy Story"), add 2-3 titles,
  hit "Get recommendations" with no user selected.
- **`/architecture`** — both diagrams should render (Mermaid), and the
  ablation chart at the bottom should show real bars (not "Loading…" stuck,
  not an error).

Or skip the UI and hit the API directly:

```bash
curl -s http://localhost:8000/health
curl -s "http://localhost:8000/users?limit=3"

# known user
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 414, "k": 5}'

# cold start
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"recent_movie_ids": [1, 2, 3], "day_of_week": "Friday", "time_of_day": "evening", "k": 5}'

curl -s http://localhost:8000/eval/metrics
```

Stop both with Ctrl-C when done.

## 4. Run the real-Merlin model instead (reduced feature set)

Same frontend, different backend env var. Stop the Terminal 1 backend
(Ctrl-C) and restart it with:

```bash
cd backend
ARTIFACT_SOURCE=merlin uv run uvicorn app.main:app --reload
```

The frontend in Terminal 2 doesn't need restarting — it just talks to
whatever the backend is currently serving.

What to expect, and why it's different from §3:
- `/` and `/cold-start` will still load, but **cold-start requests will
  fail with a 400** — this reduced model (just `{user_id, item_id}`, no
  context, no user-take) has no cold-start mechanism. That's correct
  behavior, not a bug — see `docs/PLAN_LIVE_MERLIN_SERVING.md`.
- Known-user recommendations on `/` will work and will differ from §3's
  (different model, different training run).
- Changing day-of-week/time-of-day will have **no effect** on results —
  this model doesn't use context at all (confirmed empirically, see the
  same doc).

Direct API check:

```bash
curl -s http://localhost:8000/health

# known user — works
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 414, "k": 5}'
# response should include "source":"merlin"

# cold start — should 400
curl -s -w "\nHTTP %{http_code}\n" -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"recent_movie_ids": [1, 2, 3], "k": 5}'

curl -s http://localhost:8000/eval/metrics
# recall_at_10 ≈ 0.185, ndcg_at_10 ≈ 0.103 — different numbers from the
# pedagogical model's eval_metrics.json, because it's a different model
```

## 5. (Optional) Regenerate the pedagogical artifacts yourself

No GPU needed — this runs locally in plain PyTorch/pandas, a few minutes:

```bash
for nb in notebooks/0{1,2,3,4}_*.ipynb; do
  uv run jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

Overwrites `data/processed/` and `artifacts/` (not `artifacts/merlin/`).

## 6. (Optional, costs real money) Retrain the real-Merlin model yourself

Needs a Modal account (`uv run modal setup` if you don't have one configured).

```bash
cd training
uv run modal run modal_app.py::validate             # cheap sanity check first
uv run modal run modal_app.py::inspect_architecture  # cheap, CPU-only, no training
uv run modal run modal_app.py::train                 # the real thing — GPU, costs money
cd ..
mkdir -p artifacts/merlin
uv run modal volume get two-tower-video-recommender-artifacts merlin ./artifacts/merlin --force
# the volume nests one extra "merlin/" level — flatten it:
mv artifacts/merlin/merlin/* artifacts/merlin/ && rmdir artifacts/merlin/merlin
```
