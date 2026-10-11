from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age_years": 150,
    "size_cm": 30,
    "estimated_value_usd": 500,
    "organic": 0,
    "has_inscription": 1,
    "excavation_find": 1,
    "natural_specimen": 0,
    "artwork": 0,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["models"]["v1"]["loaded"] is True
    assert body["models"]["v2"]["loaded"] is True


def test_v1_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["model_version"] == "v1"
    assert isinstance(body["prediction"], str)
    assert body["request_id"]


def test_v2_prediction():
    response = client.post("/v2/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["model_version"] == "v2"
    assert isinstance(body["prediction"], str)


def test_v1_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"


def test_v2_model_info():
    response = client.get("/v2/model-info")
    assert response.status_code == 200
    assert response.json()["model_version"] == "v2"


def test_reject_negative_age():
    payload = {**VALID_PAYLOAD, "age_years": -5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_reject_invalid_binary_flag():
    payload = {**VALID_PAYLOAD, "artwork": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_reject_missing_field():
    payload = VALID_PAYLOAD.copy()
    del payload["size_cm"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_reject_extra_field():
    payload = {**VALID_PAYLOAD, "unknown": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_reject_malformed_json():
    response = client.post(
        "/predict",
        content='{"age_years": ',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422

