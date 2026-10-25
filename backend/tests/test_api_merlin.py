"""Exercises the 'merlin' ARTIFACT_SOURCE path — the real NVTabular + Merlin
Models model (training/modal_app.py), as opposed to the pedagogical default.
See docs/PLAN_LIVE_MERLIN_SERVING.md.

ARTIFACT_SOURCE/ARTIFACTS_DIR are read as module globals at request/lifespan
time, not baked in at import time, so monkeypatching them on the already-
imported `app.main` module before opening a fresh TestClient is enough to
switch which artifacts get loaded — no subprocess or reimport needed.
"""

from collections.abc import Iterator
from pathlib import Path

import app.main as main_module
import pytest
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def merlin_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    merlin_dir = main_module.REPO_ROOT / "artifacts" / "merlin"
    if not (merlin_dir / "query_tower_weights.npz").exists():
        pytest.skip("artifacts/merlin/ not present — run training/modal_app.py::train first")
    monkeypatch.setattr(main_module, "ARTIFACT_SOURCE", "merlin")
    monkeypatch.setattr(main_module, "ARTIFACTS_DIR", merlin_dir)
    with TestClient(app) as client:
        yield client


def test_merlin_known_user(merlin_client: TestClient) -> None:
    user_id = merlin_client.get("/users").json()[0]["user_id"]
    resp = merlin_client.post("/recommend", json={"user_id": user_id, "k": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "known_user"
    assert body["source"] == "merlin"
    movie_ids = [r["movie_id"] for r in body["recommendations"]]
    assert len(movie_ids) == 10
    assert len(set(movie_ids)) == 10


def test_merlin_cold_start_rejected(merlin_client: TestClient) -> None:
    resp = merlin_client.post("/recommend", json={"recent_movie_ids": [1, 2, 3], "k": 5})
    assert resp.status_code == 400
    assert "cold-start is not supported" in resp.json()["detail"]


def test_merlin_unknown_user_404(merlin_client: TestClient) -> None:
    resp = merlin_client.post("/recommend", json={"user_id": 999999999, "k": 5})
    assert resp.status_code == 404


def test_merlin_eval_metrics(merlin_client: TestClient) -> None:
    resp = merlin_client.get("/eval/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "recall_at_10" in body


def test_merlin_artifacts_dir_points_at_subdir() -> None:
    # Sanity: the default (pedagogical) ARTIFACTS_DIR should NOT be the
    # merlin subdirectory — confirms the two artifact sets really are kept
    # separate, not accidentally aliased.
    assert main_module.ARTIFACTS_DIR != Path(main_module.REPO_ROOT) / "artifacts" / "merlin"
