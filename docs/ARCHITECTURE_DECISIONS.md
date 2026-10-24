# Architecture Decision Journal

This is a log of my own architectural reasoning while building this project, not a
list of conclusions handed to me. Each entry below is raised by a specific notebook
or design choice in the repo; I write the answer myself after actually seeing the
numbers (ablation results, latency, etc.), not before.

Template used for each entry:

```
## ADR-00X: <decision>
**Context:** what problem forced a choice
**Options considered:**
**My decision:**
**Why:**
**What would flip this decision, and at what scale/condition:**
```

---

## ADR-001: Two-tower vs. matrix factorization

**Context:** notebook `01_explore_movielens.ipynb` — at what point does context,
item/user metadata, and the cold-start problem actually justify a two-tower neural
model over classic matrix factorization, for *this* dataset's scale (~610 users,
~9,700 movies, ~100K ratings)?

**Options considered:** _TODO — fill in after notebook 1_

**My decision:** _TODO_

**Why:** _TODO_

**What would flip this decision, and at what scale/condition:** _TODO_

---

## ADR-002: NVTabular workflow vs. a hand-rolled pandas/sklearn pipeline

**Context:** notebook `02_nvtabular_features.ipynb` — what did the NVTabular
`Workflow` abstraction actually buy, concretely, over just writing the same
Categorify / feature-derivation logic by hand in pandas?

**Options considered:** _TODO_

**My decision:** _TODO_

**Why:** _TODO_

**What would flip this decision, and at what scale/condition:** _TODO_

---

## ADR-003: In-batch negative sampling vs. an explicit negative sampler

**Context:** notebook `04_two_tower_cold_start.ipynb` — did shuffled in-batch
negatives give good enough negative examples at this dataset's size, or were there
visible artifacts (e.g. popularity bias, too-easy negatives)?

**Options considered:** _TODO_

**My decision:** _TODO_

**Why:** _TODO_

**What would flip this decision, and at what scale/condition:** _TODO_

---

## ADR-004: The cold-start trick (dropping user-ID for a fraction of training rows)

**Context:** notebook `04_two_tower_cold_start.ipynb` — how much did this move the
cold-start Recall@K number in the ablation, and was the added training complexity
worth it?

**Options considered:** _TODO_

**My decision:** _TODO_

**Why:** _TODO_

**What would flip this decision, and at what scale/condition:** _TODO_

---

## ADR-005: Exhaustive dot-product scoring vs. an ANN index (FAISS/ScaNN)

**Context:** serving design — this project scores every item in the catalog
exhaustively against the user embedding. See the scale-tradeoffs table in the
README for the production/hyperscale comparison.

**Options considered:** _TODO_

**My decision:** _TODO — write down the catalog size at which you'd personally
switch to an ANN index, and why._

**Why:** _TODO_

**What would flip this decision, and at what scale/condition:** _TODO_

---

## ADR-006: Plain TF-in-FastAPI serving vs. Triton Inference Server

**Context:** the source article runs Triton alongside FastAPI for the query tower.
This project simplifies to plain TF/Keras inference in-process, for demo
simplicity. Use this entry to record when *you'd* reverse that decision in a real
system (QPS, multi-model serving, autoscaling, etc.)

**Options considered:** _TODO_

**My decision:** _TODO_

**Why:** _TODO_

**What would flip this decision, and at what scale/condition:** _TODO_

---

## ADR-007: Pedagogical reimplementation locally vs. real Merlin library calls in notebooks

**Context:** the plan called for notebooks 2 and 4 to run NVTabular's `Workflow` and
Merlin Models' `TwoTowerModel` directly, on a CPU subsample, for fast local
iteration. In practice: `nvtabular`/`merlin-models` (last released ~Oct–Dec 2023)
pin `pandas<1.6` and `tensorflow<2.13`, which has no PyPI wheel for Apple Silicon.
Attempting to install the real packages locally (even pinning numpy/dask/distributed
to their contemporaneous 2023 versions) hit successive binary/API incompatibilities —
numpy ABI mismatch, then a dask/pandas accessor-API mismatch, then a dask/Python 3.11
`inspect` incompatibility. This is the concrete manifestation of the staleness risk
already flagged for the Merlin project as a whole.

**Options considered:**
1. Keep fighting the pin-hunting locally (diminishing returns — each fix surfaces a
   new incompatibility layer).
2. Run every notebook's Merlin-dependent cells remotely via small Modal functions,
   round-tripping on every iteration.
3. Make notebooks 2 & 4 pedagogical reimplementations in plain modern pandas/PyTorch
   (no version conflicts, fast, true to the concepts), and reserve real
   `nvtabular`/`merlin.models` library calls for the one place that already runs in
   NVIDIA's own pinned container: the Modal training job.

**My decision:** option 3 — this entry is pre-filled by Claude (not left as a TODO
for you) because it's a factual compatibility finding, not a judgment call; the
other entries above are still yours to reason through yourself.

**Why:** it keeps the local learning loop fast and dependency-hell-free without
giving up any fidelity in the *production* artifact — the committed model weights
and embeddings are still produced by the real Merlin libraries, inside the exact
container NVIDIA ships for this reason. The notebooks are explicit about which cells
are "the real thing" (none, locally) vs. "a faithful mirror of `training/modal_app.py`."

**What would flip this decision, and at what scale/condition:** if Merlin ever ships
a maintained release with modern, unpinned dependencies, or if this project moved to
developing inside a Linux x86_64 devcontainer matching the NGC image, the real
libraries could run locally again and this reimplementation layer could be deleted.
