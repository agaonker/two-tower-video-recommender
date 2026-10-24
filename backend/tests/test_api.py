from app.main import app
from fastapi.testclient import TestClient


def test_health() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


def test_users_list() -> None:
    with TestClient(app) as client:
        resp = client.get("/users")
        assert resp.status_code == 200
        users = resp.json()
        assert len(users) > 0
        assert "user_id" in users[0]


def test_recommend_known_user() -> None:
    with TestClient(app) as client:
        user_id = client.get("/users").json()[0]["user_id"]
        resp = client.post("/recommend", json={"user_id": user_id, "k": 10})
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "known_user"
        movie_ids = [r["movie_id"] for r in body["recommendations"]]
        assert len(movie_ids) == 10
        assert len(set(movie_ids)) == 10  # all distinct


def test_recommend_cold_start() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/recommend",
            json={"recent_movie_ids": [1, 2, 3], "k": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "cold_start"
        assert body["note"] is not None
        assert len(body["recommendations"]) == 5


def test_recommend_requires_identity() -> None:
    with TestClient(app) as client:
        resp = client.post("/recommend", json={"k": 5})
        assert resp.status_code == 422


def test_recommend_context_changes_results() -> None:
    with TestClient(app) as client:
        user_id = client.get("/users").json()[0]["user_id"]
        morning = client.post(
            "/recommend",
            json={"user_id": user_id, "day_of_week": "Monday", "time_of_day": "morning", "k": 10},
        ).json()
        night = client.post(
            "/recommend",
            json={"user_id": user_id, "day_of_week": "Saturday", "time_of_day": "night", "k": 10},
        ).json()
        morning_ids = [r["movie_id"] for r in morning["recommendations"]]
        night_ids = [r["movie_id"] for r in night["recommendations"]]
        assert morning_ids != night_ids


def test_unknown_user_404() -> None:
    with TestClient(app) as client:
        resp = client.post("/recommend", json={"user_id": 999999999, "k": 5})
        assert resp.status_code == 404


def test_movie_search() -> None:
    with TestClient(app) as client:
        resp = client.get("/movies/search", params={"q": "Toy Story"})
        assert resp.status_code == 200
        titles = [m["title"] for m in resp.json()]
        assert any("Toy Story" in t for t in titles)


def test_eval_metrics() -> None:
    with TestClient(app) as client:
        resp = client.get("/eval/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "enriched" in body
