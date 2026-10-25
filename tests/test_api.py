import pytest
from fastapi.testclient import TestClient

from app.train import train


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    metrics = train(str(out))
    assert metrics["accuracy"] >= 0.9
    import app.main as main

    main.MODEL_PATH = str(out / "model.joblib")
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_setosa(client):
    r = client.post("/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2})
    assert r.status_code == 200
    assert r.json()["prediction"] == "setosa"


def test_predict_validation(client):
    r = client.post("/predict", json={"sepal_length": -1})
    assert r.status_code == 422


def test_metrics_exposed(client):
    client.get("/health")
    body = client.get("/metrics").text
    assert "http_requests_total" in body
