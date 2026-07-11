def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("Heart disease", "No heart disease")
    assert body["model_name"] == "logistic_regression"


def test_predict_optional_fields_can_be_omitted(client, valid_payload):
    del valid_payload["ca"]
    del valid_payload["thal"]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_missing_required_field(client, valid_payload):
    del valid_payload["age"]
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_predict_rejects_out_of_range_values(client, valid_payload):
    for field, bad_value in [("cp", 9), ("age", -5), ("sex", 2), ("oldpeak", 99)]:
        payload = {**valid_payload, field: bad_value}
        assert client.post("/predict", json=payload).status_code == 422, field


def test_metrics_endpoint(client, valid_payload):
    client.post("/predict", json=valid_payload)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_predictions_total" in response.text
