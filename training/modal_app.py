"""The production-fidelity training path: the real NVTabular + Merlin Models
stack, running inside NVIDIA's own pinned container, on a Modal GPU.

This mirrors notebooks/02-04 (Categorify, day/time-of-day context, frozen
text embeddings, in-batch negative sampling, the cold-start user-id-dropout
training trick — "user-take" sequences are not yet wired in here, see below)
but calls the real `nvtabular.Workflow` and `merlin.models.tf.TwoTowerModel`
instead of the pandas/PyTorch reimplementation — see ADR-007 in
docs/ARCHITECTURE_DECISIONS.md for why that split exists.

STATUS (see docs/PROJECT_PLAN.md §7 for the full debugging history): this has
been run for real against a Modal GPU, with the training + evaluation path
fully working — `workflow.fit()`, `workflow.transform()`, `model.fit()`, and
`model.evaluate()` all complete, producing real numbers (recall_at_10 ≈
0.16-0.18, ndcg_at_10 ≈ 0.08-0.09, notably better than the pedagogical
model). Along the way: fixed a numpy/numba ABI conflict (don't
pip_install sentence-transformers into this image — load notebook 3's
precomputed text embeddings instead), fixed a `numba.cuda` segfault
(`NUMBA_DISABLE_CUDA=1`), fixed a missing `genres` merge, fixed the
`TwoTowerModel(query_tower=...)` call signature, and dropped multi-hot
`genres` from this path (NVTabular's CPU-mode Dataset has "extremely limited
support" for list columns, confirmed empirically).

NOT YET DONE: exporting the trained model into backend/-servable artifacts.
`model.item_embeddings()` precomputes cleanly (items are static), but the
real point of this project is context-aware *query* embeddings, which can't
be precomputed the same way without freezing context — see
docs/PROJECT_PLAN.md §7 for the two real options (TF SavedModel live
serving, or reverse-engineering Merlin's auto-built layer shapes) and why
neither has been attempted yet. The `train()` function below stops with a
diagnostic `RuntimeError` at the export step rather than writing wrong
artifacts.

Usage:
    modal run modal_app.py::validate   # cheap, CPU-only import/compat check
    modal run modal_app.py::train      # the real thing — costs GPU money
"""

from __future__ import annotations

import json

import modal

app = modal.App("two-tower-video-recommender-training")

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/merlin/merlin-tensorflow:23.08",
        add_python=None,  # the NGC image ships its own pinned Python + TF + Merlin stack
    )
    # Deliberately NOT pip_install-ing sentence-transformers/huggingface_hub
    # here: confirmed (see ADR-007 follow-up) that doing so upgrades numpy
    # underneath this image's pinned, pre-compiled numba binary and breaks
    # merlin.models.tf entirely. Text embedding computation doesn't need
    # NVTabular or Merlin Models — it's pure sentence-transformers — so it
    # doesn't belong sharing an image with them. Instead, this job loads the
    # embeddings notebooks/03 already computed, same as a real pipeline would
    # treat "text embedding service" and "recommender training" as separate
    # systems with a data contract between them, not one monolith.
    #
    # Separately: with a real GPU attached, `from merlin.models.tf import
    # TwoTowerModel` segfaults — traced (see docs/ARCHITECTURE_DECISIONS.md,
    # ADR-007 follow-up) to `numba.cuda`'s driver probing inside
    # merlin.core.compat, which this Oct-2023 container's bundled numba
    # doesn't handle cleanly against Modal's current GPU driver. TensorFlow
    # uses its own separate CUDA bindings, so this is a stray capability
    # check, not something training actually needs — disabling numba's CUDA
    # path specifically avoids the crash without losing GPU access for the
    # real training step.
    .env({"NUMBA_DISABLE_CUDA": "1"})
    # add_local_* must be the last step in the chain (Modal requirement).
    .add_local_dir("../data/raw/ml-latest-small", remote_path="/data/ml-latest-small")
    .add_local_file(
        "../data/processed/item_text_embeddings.npz",
        remote_path="/data/item_text_embeddings.npz",
    )
)

artifacts_volume = modal.Volume.from_name(
    "two-tower-video-recommender-artifacts", create_if_missing=True
)

K_USER_TAKE = 5
COLD_START_DROP_FRACTION = 0.15
MAX_GENRES = 6
EMB_DIMS = {"user": 16, "movie": 16, "genre": 8, "day": 4, "tod": 4}
TEXT_PROJ_DIM = 16
TOWER_OUT_DIM = 32
K_EVAL = 10


@app.function(image=image)  # CPU only — confirmed this image no longer needs a GPU just to import
def validate() -> None:
    """Confirms the image builds and the exact API surface `train()` depends
    on actually exists on this pinned container, before paying for a GPU run.
    Run this first: `modal run training/modal_app.py::validate`."""
    import os

    import numpy as np
    import tensorflow as tf
    from merlin.models.tf import TwoTowerModel
    from nvtabular import Workflow

    print("tensorflow:", tf.__version__)
    print("TwoTowerModel:", TwoTowerModel)
    print("Workflow:", Workflow)
    print("mounted data dir contents:", os.listdir("/data/ml-latest-small"))

    text_npz = np.load("/data/item_text_embeddings.npz")
    print("text embeddings shape:", text_npz["embeddings"].shape)


@app.function(image=image, gpu="A10G", volumes={"/artifacts": artifacts_volume}, timeout=3600)
def train() -> dict:
    def checkpoint(msg: str) -> None:
        # The previous runs died with a bare SIGSEGV and no Python traceback —
        # checkpoints are the only way to tell which stage actually crashed.
        # Started with one per import, since it crashed before even the
        # first one *after* imports — this pins down which import it is.
        print(f"[checkpoint] {msg}", flush=True)

    checkpoint("starting imports")
    import numpy as np

    checkpoint("imported numpy")
    import pandas as pd

    checkpoint("imported pandas")
    import tensorflow as tf

    checkpoint("imported tensorflow")
    import merlin.models.tf as mm
    from merlin.models.tf import TwoTowerModel

    checkpoint("imported merlin.models.tf.TwoTowerModel")
    from merlin.models.tf.metrics.topk import NDCGAt, RecallAt

    checkpoint("imported merlin.models.tf.metrics.topk")
    from merlin.schema import Tags

    checkpoint("imported merlin.schema.Tags")
    from nvtabular import Workflow

    checkpoint("imported nvtabular.Workflow")
    from nvtabular import ops as nvt_ops

    checkpoint("imported nvtabular.ops")
    from nvtabular.workflow.node import WorkflowNode

    checkpoint("imported nvtabular.workflow.node.WorkflowNode")
    import merlin.io

    checkpoint("imported merlin.io — all imports done")

    data_dir = "/data/ml-latest-small"
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    checkpoint("loaded raw csvs")

    # ---- context + cold-start columns (same derivation as notebooks/02) ----
    ts = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["day_of_week"] = ts.dt.day_name()

    def time_of_day(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 22:
            return "evening"
        return "night"

    ratings["time_of_day"] = ts.dt.hour.map(time_of_day)
    ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    rng = np.random.default_rng(42)
    drop_mask = rng.random(len(ratings)) < COLD_START_DROP_FRACTION
    ratings["userId_for_training"] = ratings["userId"].where(~drop_mask, 0)

    # leave-one-out split, same protocol as notebooks/02
    rank_from_end = ratings.groupby("userId").cumcount(ascending=False)
    train_df = ratings[rank_from_end >= 2].copy()
    test_df = ratings[rank_from_end == 0].copy()
    checkpoint(f"split: train={len(train_df)} test={len(test_df)}")

    # ---- item text embeddings: loaded, not recomputed (see the `image`
    # comment above for why sentence-transformers doesn't run in this image) ----
    text_npz = np.load("/data/item_text_embeddings.npz")
    text_emb_by_movie_id = dict(zip(text_npz["movie_ids"].tolist(), text_npz["embeddings"]))
    movies["text_embedding"] = movies["movieId"].map(
        lambda mid: text_emb_by_movie_id.get(mid, np.zeros(text_npz["embeddings"].shape[1], "float32"))
    )
    checkpoint("loaded text embeddings")

    # ---- NVTabular workflow: the real thing, not the notebooks/02 pandas mirror ----
    #
    # `genres` (multi-hot, pipe-delimited) is deliberately left out here. The
    # first real run got all the way through model.fit() and most of
    # model.evaluate() before crashing on "Failed to convert a NumPy array to
    # a Tensor (Unsupported object type float)" — traced to the genres list
    # column, and NVTabular's own CPU-mode Dataset warning
    # ("experimental... extremely limited support") strongly suggests
    # multi-hot/list columns are one of the gaps in that mode. The
    # pedagogical PyTorch model already covers genre handling; cutting it
    # here trades feature parity for an actually-completing real-Merlin run,
    # which is the more important thing to prove out first. Worth revisiting
    # with a GPU-mode (cudf) Dataset if genres matter enough to chase further.
    cat_features = ["userId_for_training", "movieId", "day_of_week", "time_of_day"] >> nvt_ops.Categorify()
    user_take: WorkflowNode = (
        ["userId", "movieId", "timestamp"]
        >> nvt_ops.Groupby(
            groupby_cols=["userId"],
            sort_cols=["timestamp"],
            aggs={"movieId": ["list"]},
            name_sep="_",
        )
    )
    # NOTE: NVTabular's Groupby produces one row per user (full session list);
    # turning that into a *leading* (shift-by-one, truncated-to-K) feature per
    # interaction row — exactly what notebooks/02's `build_user_take` does —
    # needs a custom LambdaOp over the grouped list column. Sketching the
    # intent here; the precise NVTabular 23.08 call for a rolling/leading
    # window needs to be confirmed against the real Workflow.fit() output
    # before trusting it, per the module docstring above.
    workflow = Workflow(cat_features)
    train_ds = merlin.io.Dataset(train_df)
    checkpoint("built Workflow graph + Dataset, about to fit")
    workflow.fit(train_ds)
    checkpoint("workflow.fit() done")
    train_transformed = workflow.transform(train_ds)
    checkpoint("transformed train set")
    test_transformed = workflow.transform(merlin.io.Dataset(test_df))
    checkpoint("transformed test set")

    schema = train_transformed.schema
    schema["userId_for_training"] = schema["userId_for_training"].with_tags(
        [Tags.USER_ID, Tags.CATEGORICAL]
    )
    schema["movieId"] = schema["movieId"].with_tags([Tags.ITEM_ID, Tags.CATEGORICAL])

    # ---- model: Merlin Models' built-in two-tower, in-batch negatives by default ----
    # `query_tower` is the MLP stack on the user side (required — there's no
    # schema-only default); mirrors this project's pandas/PyTorch model (one
    # 64-wide hidden layer -> TOWER_OUT_DIM). `item_tower` defaults to a
    # matching MLPBlock automatically when left unset.
    model = TwoTowerModel(schema, query_tower=mm.MLPBlock([64, TOWER_OUT_DIM]))
    model.compile(optimizer="adam", run_eagerly=False, metrics=[RecallAt(K_EVAL), NDCGAt(K_EVAL)])
    checkpoint("model compiled, about to fit")
    model.fit(train_transformed, epochs=10, batch_size=256)
    checkpoint("model.fit() done")

    eval_metrics = model.evaluate(test_transformed, batch_size=256, return_dict=True)
    checkpoint(f"evaluated: {eval_metrics}")

    # ---- export: same artifact shapes/keys as notebooks/04's export_artifacts ----
    item_embeddings = model.item_embeddings(merlin.io.Dataset(movies), batch_size=512)
    checkpoint("computed item embeddings")

    # `model.query_encoder` doesn't exist on this version's RetrievalModel —
    # discover the real attribute name instead of guessing again.
    candidates = [a for a in dir(model) if any(k in a.lower() for k in ("quer", "tower", "encoder"))]
    checkpoint(f"candidate query-tower attributes on model: {candidates}")
    query_tower = None
    for name in candidates:
        try:
            obj = getattr(model, name)
            if hasattr(obj, "get_weights"):
                checkpoint(f"  {name}: has get_weights(), {len(obj.get_weights())} arrays")
            else:
                checkpoint(f"  {name}: {type(obj)}, no get_weights()")
        except Exception as e:  # noqa: BLE001 — diagnostic only, want to see every candidate
            checkpoint(f"  {name}: raised {e!r}")
    if query_tower is None:
        raise RuntimeError(
            "see checkpoint log above for candidate attribute names; "
            "pick the right one and fix the query_tower_weights line below"
        )
    query_tower_weights = query_tower.get_weights()

    np.savez(
        "/artifacts/item_embeddings.npz",
        movie_ids=movies["movieId"].to_numpy(),
        embeddings=np.asarray(item_embeddings),
    )

    # reclib.py (the numpy-only serving path) expects named arrays, not the
    # positional arr_0/arr_1/... savez gives by default. `query_tower_weights`
    # is a flat list from Keras's `get_weights()`, in layer-definition order —
    # the mapping below is my best-documented guess at that order for
    # TwoTowerModel's query encoder (embedding tables first, then the two
    # Dense layers' kernel+bias), but it has not been checked against a real
    # trained model. Before trusting this: run this job once, print
    # `[w.shape for w in query_tower_weights]` and `model.query_encoder.summary()`,
    # and fix this mapping to match reality.
    names = [
        "user_embedding",
        "movie_embedding",
        "day_embedding",
        "tod_embedding",
        "mlp_0_weight",
        "mlp_0_bias",
        "mlp_2_weight",
        "mlp_2_bias",
    ]
    if len(names) != len(query_tower_weights):
        raise RuntimeError(
            f"expected {len(names)} query encoder weight arrays, got "
            f"{len(query_tower_weights)} — the layer-order mapping above needs "
            "fixing before artifacts can be trusted; see the comment just above."
        )
    np.savez("/artifacts/query_tower_weights.npz", **dict(zip(names, query_tower_weights)))

    # Same caveat as the weight-name mapping above: NVTabular's Categorify
    # fits vocabularies internally and exposes them via the workflow's
    # output node stats — `nvtabular.ops.get_embedding_sizes(workflow)` gives
    # sizes, but pulling the actual value->index mapping back out as a plain
    # dict (to match feature_vocab.json's shape from notebooks/04) needs
    # confirming against the real `workflow.fit()` output, not assumed here.
    feature_vocab = {
        "user_vocab": cat_features.op.categories["userId_for_training"],
        "movie_vocab": cat_features.op.categories["movieId"],
        "day_vocab": cat_features.op.categories["day_of_week"],
        "tod_vocab": cat_features.op.categories["time_of_day"],
        "user_take_k": K_USER_TAKE,
        "embedding_dims": {**EMB_DIMS, "tower_out": TOWER_OUT_DIM},
    }
    with open("/artifacts/feature_vocab.json", "w") as f:
        json.dump(feature_vocab, f)

    with open("/artifacts/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    artifacts_volume.commit()
    return eval_metrics


@app.local_entrypoint()
def main() -> None:
    metrics = train.remote()
    print("training complete, eval metrics:", metrics)
    print(
        "download artifacts with: modal volume get "
        "two-tower-video-recommender-artifacts / ../artifacts"
    )
