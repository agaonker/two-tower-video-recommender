"""The production-fidelity training path: the real NVTabular + Merlin Models
stack, running inside NVIDIA's own pinned container, on a Modal GPU.

Full history in docs/PROJECT_PLAN.md §7 and docs/PLAN_LIVE_MERLIN_SERVING.md.
Short version: this trains and evaluates successfully (recall_at_10 ≈
0.16-0.18, ndcg_at_10 ≈ 0.08-0.09, beating the pedagogical model), and now
exports query-tower weights into `artifacts/merlin/` for numpy-only serving
(see backend/app/reclib.py) — NOT the committed `artifacts/` the pedagogical
model serves; see backend's `ARTIFACT_SOURCE` setting to switch between them.

SCOPE (deliberately reduced — see docs/PLAN_LIVE_MERLIN_SERVING.md): this
model only uses `{user_id, item_id}` plus the cold-start id-dropout trick.
Confirmed via `inspect_architecture()`: Merlin Models' `TwoTowerModel` only
wires in schema columns tagged `USER_ID`/`ITEM_ID` — day/time-of-day context,
item text embeddings, genres, and the user-take sequence feature are all
real follow-up work (Phase A in the plan doc), not done here. This also means
cold-start for this artifact source only has the id-dropout fallback, with no
user-take signal to compensate — weaker than the pedagogical model's version.

Debugging history along the way: fixed a numpy/numba ABI conflict (don't
pip_install sentence-transformers into this image), a `numba.cuda` segfault
(`NUMBA_DISABLE_CUDA=1`), a missing `genres` merge (since removed along with
genres itself — NVTabular's CPU-mode Dataset has "extremely limited support"
for list columns), the `TwoTowerModel(query_tower=...)` call signature, and —
via the cheap CPU-only `inspect_architecture()` function below — the real
attribute path to the query tower's weights (`model.retrieval_block.query_block()`,
not `model.query_encoder`) and the real vocab-extraction mechanism
(`cat_features.op.categories[name]` is a parquet file path, not a dict).

Usage:
    modal run modal_app.py::validate             # cheap, CPU-only import/compat check
    modal run modal_app.py::inspect_architecture  # cheap, CPU-only architecture research
    modal run modal_app.py::train                 # the real thing — costs GPU money
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

COLD_START_DROP_FRACTION = 0.15
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


@app.function(image=image)  # CPU only, no training — pure architecture discovery
def inspect_architecture() -> None:
    """Step 1 of docs/PLAN_LIVE_MERLIN_SERVING.md: find the real attribute
    path to TwoTowerModel's query tower and its actual layer shapes, by
    building the model on a tiny synthetic dataset and introspecting it —
    instead of guessing again on a paid GPU run. No `.fit()`, no GPU, just
    enough of a forward pass to force Keras to materialize the layers.

    Run this before touching train()'s export section:
    `modal run modal_app.py::inspect_architecture`
    """
    import merlin.io
    import merlin.models.tf as mm
    import numpy as np
    import pandas as pd
    from merlin.models.tf import TwoTowerModel
    from merlin.schema import Tags
    from nvtabular import Workflow
    from nvtabular import ops as nvt_ops

    rng = np.random.default_rng(0)
    n_users, n_movies, n_rows = 8, 12, 200
    synthetic = pd.DataFrame(
        {
            "userId": rng.integers(1, n_users + 1, n_rows),
            "movieId": rng.integers(1, n_movies + 1, n_rows),
            "timestamp": rng.integers(1_600_000_000, 1_700_000_000, n_rows),
        }
    )
    ts = pd.to_datetime(synthetic["timestamp"], unit="s")
    synthetic["day_of_week"] = ts.dt.day_name()
    synthetic["time_of_day"] = (ts.dt.hour % 24 // 6).map(
        dict(enumerate(["night", "morning", "afternoon", "evening"]))
    )
    synthetic["userId_for_training"] = synthetic["userId"]  # no cold-start dropout needed here

    # Only userId_for_training + movieId — a first run of this function (with
    # day_of_week/time_of_day also included) confirmed TwoTowerModel ignores
    # any schema column not tagged USER_ID/ITEM_ID: `model.predict()` raised
    # "Model expected features: {'movieId', 'userId_for_training'}" even
    # though both context columns were present in the input schema. Matching
    # that real (context-less) feature set here instead of the aspirational one.
    cat_features = ["userId_for_training", "movieId"] >> nvt_ops.Categorify()
    workflow = Workflow(cat_features)
    ds = merlin.io.Dataset(synthetic)
    workflow.fit(ds)
    transformed = workflow.transform(ds)

    schema = transformed.schema
    schema["userId_for_training"] = schema["userId_for_training"].with_tags(
        [Tags.USER_ID, Tags.CATEGORICAL]
    )
    schema["movieId"] = schema["movieId"].with_tags([Tags.ITEM_ID, Tags.CATEGORICAL])

    model = TwoTowerModel(schema, query_tower=mm.MLPBlock([64, TOWER_OUT_DIM]))
    model.compile(optimizer="adam", run_eagerly=True)  # eager: easier to introspect after
    model.predict(transformed, batch_size=32)  # forward pass only — materializes the layers

    print("=" * 80)
    print("model.summary():")
    model.summary(expand_nested=True, show_trainable=True)

    print("=" * 80)
    print("Top-level attributes with 'quer'/'tower'/'item'/'retrieval' in the name:")
    for name in dir(model):
        if any(k in name.lower() for k in ("quer", "tower", "item", "retriev", "encoder", "block")):
            try:
                obj = getattr(model, name)
            except Exception as e:  # noqa: BLE001 — want to see every candidate, not stop at first
                print(f"  {name}: raised {e!r}")
                continue
            print(f"  {name}: {type(obj)}")

    print("=" * 80)
    print("model.layers (top level):")
    for layer in model.layers:
        print(f"  {layer.name}: {type(layer).__name__}")
        if hasattr(layer, "get_weights"):
            shapes = [w.shape for w in layer.get_weights()]
            if shapes:
                print(f"    weights: {shapes}")

    print("=" * 80)
    print("model.retrieval_block (TwoTowerBlock) — looking for clean query/item accessors:")
    rb = model.retrieval_block
    for name in dir(rb):
        if name.startswith("_"):
            continue
        if any(k in name.lower() for k in ("quer", "item", "tower")):
            try:
                obj = getattr(rb, name)
            except Exception as e:  # noqa: BLE001 — diagnostic, want every candidate
                print(f"  {name}: raised {e!r}")
                continue
            print(f"  {name}: {type(obj)}")
            if callable(obj) and not isinstance(obj, type):
                try:
                    result = obj()
                    print(f"    called {name}(): {type(result)}")
                    if hasattr(result, "get_weights"):
                        print(
                            f"    {name}().get_weights() shapes: {[w.shape for w in result.get_weights()]}"
                        )
                except Exception as e:  # noqa: BLE001 — diagnostic
                    print(f"    called {name}(): raised {e!r}")

    print("=" * 80)
    print("Surgical walk of query_block()'s own sublayer tree — avoid recursive")
    print("get_weights() through shared ModelContext, target each leaf layer directly:")

    def walk(layer, depth: int = 0) -> None:
        indent = "  " * depth
        own_weights = getattr(layer, "weights", [])
        # `layer.weights` on a container still recurses through Keras's
        # tracking, so the only way to see what THIS layer owns directly
        # (not its children) is to diff against the union of its children's.
        children = list(getattr(layer, "layers", []))
        child_var_ids = {id(w) for c in children for w in getattr(c, "weights", [])}
        direct = [w for w in own_weights if id(w) not in child_var_ids]
        direct_shapes = [tuple(w.shape) for w in direct]
        print(
            f"{indent}{type(layer).__name__} (name={getattr(layer, 'name', '?')})"
            f"{'  OWN WEIGHTS: ' + str(direct_shapes) if direct_shapes else ''}"
        )
        for child in children:
            walk(child, depth + 1)

    print("--- query_block() ---")
    walk(rb.query_block())
    print("--- item_block() ---")
    walk(rb.item_block())

    print("=" * 80)
    print("Checking cat_features.op.categories[...] for vocab extraction:")
    path = cat_features.op.categories["userId_for_training"]
    print(f"  categories path: {path}")
    try:
        vocab_df = pd.read_parquet(path)
        print(f"  read_parquet columns: {list(vocab_df.columns)}")
        print(f"  read_parquet head:\n{vocab_df.head(10)}")
    except Exception as e:  # noqa: BLE001 — diagnostic
        print(f"  read_parquet raised: {e!r}")
    try:
        print("  .vocabs attribute:", cat_features.op.vocabs)
    except Exception as e:  # noqa: BLE001 — diagnostic
        print(f"  .vocabs raised: {e!r}")

    print("=" * 80)
    print("Finding the query tower's EmbeddingFeatures by-name table mapping")
    print("(positional order flipped between query_block and item_block above —")
    print(" need to extract by feature name, not position):")
    query_embed_layer = rb.query_block().layers[0].layers[0].layers[0]
    print(f"query_embed_layer: {type(query_embed_layer).__name__}")
    for name in dir(query_embed_layer):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(query_embed_layer, name)
        except Exception:  # noqa: BLE001, S112 — diagnostic, skip whatever doesn't resolve
            continue
        if isinstance(obj, dict) and obj:
            print(f"  .{name} (dict): {list(obj.keys())}")
            for k, v in obj.items():
                print(
                    f"    {k}: {type(v).__name__}"
                    f"{', weights=' + str([tuple(w.shape) for w in v.weights]) if hasattr(v, 'weights') else ''}"
                )


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
    import merlin.io

    checkpoint("imported merlin.io — all imports done")

    data_dir = "/data/ml-latest-small"
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    checkpoint("loaded raw csvs")

    # Scope note (docs/PLAN_LIVE_MERLIN_SERVING.md): this model is
    # deliberately feature-reduced — just {user_id, item_id} plus the
    # cold-start id-dropout trick below. day_of_week/time_of_day context,
    # item text embeddings, genres, and the user-take sequence feature are
    # all Phase A follow-up work (restoring parity with the pedagogical
    # model), not done here. Confirmed via inspect_architecture(): Merlin
    # Models' TwoTowerModel only wires in schema columns tagged USER_ID or
    # ITEM_ID — adding more categorical columns to the schema without a
    # tower tag does NOT make the model use them, so there was never a
    # reason to compute them in this reduced version.
    ratings = ratings.sort_values(["userId", "timestamp"]).reset_index(drop=True)

    rng = np.random.default_rng(42)
    drop_mask = rng.random(len(ratings)) < COLD_START_DROP_FRACTION
    ratings["userId_for_training"] = ratings["userId"].where(~drop_mask, 0)

    # leave-one-out split, same protocol as notebooks/02
    rank_from_end = ratings.groupby("userId").cumcount(ascending=False)
    train_df = ratings[rank_from_end >= 2].copy()
    test_df = ratings[rank_from_end == 0].copy()
    checkpoint(f"split: train={len(train_df)} test={len(test_df)}")

    cat_features = ["userId_for_training", "movieId"] >> nvt_ops.Categorify()
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

    # ---- export: writes to artifacts/merlin/, NOT the committed artifacts/
    # the pedagogical model serves (docs/PLAN_LIVE_MERLIN_SERVING.md Step 4)
    # — both become separately servable via backend's ARTIFACT_SOURCE
    # setting, so this can't regress the working demo.
    import os

    out_dir = "/artifacts/merlin"
    os.makedirs(out_dir, exist_ok=True)

    item_embeddings_raw = model.item_embeddings(merlin.io.Dataset(movies), batch_size=512)
    checkpoint(
        f"computed item embeddings: type={type(item_embeddings_raw)}, "
        f"has_compute={hasattr(item_embeddings_raw, 'compute')}, "
        f"has_to_ddf={hasattr(item_embeddings_raw, 'to_ddf')}"
    )
    # `model.item_embeddings()` doesn't return a plain numpy array (confirmed:
    # `np.asarray(...)` on it previously raised a pickling error on a
    # "data_iterator" local object, meaning it's some lazy Dataset-like
    # wrapper, not materialized data). Handle the shapes this plausibly is,
    # with checkpoints at each attempt so a failure here still tells us
    # something concrete instead of another blind guess.
    item_embeddings_df = item_embeddings_raw
    if hasattr(item_embeddings_df, "to_ddf"):
        item_embeddings_df = item_embeddings_df.to_ddf()
        checkpoint(f"  .to_ddf() -> {type(item_embeddings_df)}")
    if hasattr(item_embeddings_df, "compute"):
        item_embeddings_df = item_embeddings_df.compute()
        checkpoint(f"  .compute() -> {type(item_embeddings_df)}")
    item_movie_ids = None
    if hasattr(item_embeddings_df, "columns"):
        checkpoint(f"  columns: {list(item_embeddings_df.columns)}")
        checkpoint(f"  dtypes: {dict(item_embeddings_df.dtypes)}")
        id_cols = [c for c in item_embeddings_df.columns if c in ("movieId", "item_id")]
        # The dataframe carries the original `movies` metadata columns
        # (title, genres, ...) alongside the embedding dims — exclude all of
        # them by name, not just the id column, or they silently end up
        # concatenated into the "embeddings" array (confirmed: a first
        # attempt here only excluded movieId and produced a (9742, 34) array
        # instead of (9742, 32), with 2 text columns mixed in).
        exclude_cols = set(movies.columns) | {"item_id"}
        numeric_cols = [c for c in item_embeddings_df.columns if c not in exclude_cols]
        item_embeddings = item_embeddings_df[numeric_cols].to_numpy()
        # Row order through a batched/dask computation isn't guaranteed to
        # match the input `movies` dataframe's order — use the id column
        # that traveled WITH each embedding row if one exists, rather than
        # assuming alignment with the separate `movies` dataframe.
        if id_cols:
            item_movie_ids = item_embeddings_df[id_cols[0]].to_numpy()
            checkpoint(f"  using '{id_cols[0]}' column for movie_ids (guaranteed row-aligned)")
    else:
        item_embeddings = np.asarray(item_embeddings_df)
    if item_movie_ids is None:
        checkpoint(
            "  WARNING: no id column found alongside embeddings — "
            "falling back to movies['movieId'] order, unverified alignment"
        )
        item_movie_ids = movies["movieId"].to_numpy()
    checkpoint(f"  final item_embeddings shape: {np.asarray(item_embeddings).shape}")

    # Confirmed via inspect_architecture() (see docs/PLAN_LIVE_MERLIN_SERVING.md
    # Step 1 — model.query_encoder doesn't exist; model.query_embeddings is a
    # method, not a weights-bearing layer). The real accessor is
    # model.retrieval_block.query_block(), and its EmbeddingFeatures layer's
    # `.embedding_tables` dict is keyed by feature name — required because
    # raw positional weight order flips between the query and item tower
    # (confirmed empirically, not assumed).
    query_tower = model.retrieval_block.query_block()
    # TowerBlock -> SequentialBlock -> ParallelBlock -> EmbeddingFeatures
    embedding_layer = query_tower.layers[0].layers[0].layers[0]
    user_embedding = embedding_layer.embedding_tables["userId_for_training"].get_weights()[0]
    # TowerBlock -> SequentialBlock -> [Dense, Dense] (the mm.MLPBlock([64, TOWER_OUT_DIM]) passed in)
    dense_0, dense_1 = query_tower.layers[0].layers[1].layers
    mlp_0_weight, mlp_0_bias = dense_0.get_weights()
    mlp_2_weight, mlp_2_bias = dense_1.get_weights()
    checkpoint(
        f"extracted query tower weights: user_embedding={user_embedding.shape}, "
        f"mlp_0={mlp_0_weight.shape}/{mlp_0_bias.shape}, mlp_2={mlp_2_weight.shape}/{mlp_2_bias.shape}"
    )

    np.savez(
        f"{out_dir}/item_embeddings.npz",
        movie_ids=item_movie_ids,
        embeddings=np.asarray(item_embeddings),
    )
    # NOTE for backend/app/reclib.py: these are TensorFlow's weight
    # convention — kernel shape (in_dim, out_dim), y = x @ kernel + bias —
    # NOT the pedagogical PyTorch model's (out_dim, in_dim), y = x @ W.T + b.
    # A separate merlin-specific forward function is needed, not a reuse of
    # the pedagogical one.
    np.savez(
        f"{out_dir}/query_tower_weights.npz",
        user_embedding=user_embedding,
        mlp_0_weight=mlp_0_weight,
        mlp_0_bias=mlp_0_bias,
        mlp_2_weight=mlp_2_weight,
        mlp_2_bias=mlp_2_bias,
    )

    # Confirmed via inspect_architecture(): cat_features.op.categories[name]
    # is a parquet file PATH, not an in-memory dict. The row's pandas index
    # is the Categorify-assigned integer id (categories are frequency-sorted
    # descending, with a few low indices reserved before the first real one
    # — confirmed by reading the actual parquet, not assumed).
    def read_vocab(column: str) -> dict:
        vocab_df = pd.read_parquet(cat_features.op.categories[column])
        return {str(row[column]): int(idx) for idx, row in vocab_df.iterrows()}

    feature_vocab = {
        "user_vocab": read_vocab("userId_for_training"),
        "movie_vocab": read_vocab("movieId"),
        "embedding_dims": {"user": int(user_embedding.shape[1]), "tower_out": TOWER_OUT_DIM},
    }
    with open(f"{out_dir}/feature_vocab.json", "w") as f:
        json.dump(feature_vocab, f)

    with open(f"{out_dir}/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    artifacts_volume.commit()
    return eval_metrics


@app.local_entrypoint()
def main() -> None:
    metrics = train.remote()
    print("training complete, eval metrics:", metrics)
    print(
        "download artifacts with: modal volume get "
        "two-tower-video-recommender-artifacts merlin ./artifacts/merlin"
    )
