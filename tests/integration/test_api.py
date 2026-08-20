from fastapi.testclient import TestClient

from rookie_lab.serving.api import app


def test_health_and_skills() -> None:
    client = TestClient(app)
    health = client.get("/health", headers={"X-Request-ID": "integration-test"})
    skills = client.get("/skills")
    assert health.status_code == 200
    assert health.headers["X-Request-ID"] == "integration-test"
    assert health.json()["secrets_required"] is False
    assert skills.status_code == 200
    assert skills.json()["active_nodes"] == 114


def test_invalid_metric_name() -> None:
    assert TestClient(app).get("/metrics/../secret").status_code in {400, 404}


def test_prediction_payload_validation() -> None:
    response = TestClient(app).post(
        "/predict/churn",
        json={
            "age": 8,
            "tenure_days": -1,
            "region": "north",
            "channel": "web",
            "visits_30d": 1,
            "support_tickets": 0,
        },
    )
    assert response.status_code == 422


def test_run_query_endpoints_are_read_only() -> None:
    client = TestClient(app)
    assert client.get("/runs").status_code == 200
    assert client.get("/runs/demo-001/artifacts").status_code == 200
    assert client.get("/runs/../secret/artifacts").status_code in {400, 404}
