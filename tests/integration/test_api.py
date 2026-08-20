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
