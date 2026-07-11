import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import CLEAN_DATA_PATH
from src.data import split_features_target
from src.pipeline import build_pipeline


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    """The committed cleaned dataset (no network access needed)."""
    if not CLEAN_DATA_PATH.exists():
        pytest.skip("cleaned dataset missing: run python -m scripts.download_data")
    return pd.read_csv(CLEAN_DATA_PATH)


@pytest.fixture(scope="session")
def trained_pipeline(dataset):
    X, y = split_features_target(dataset)
    pipeline = build_pipeline(LogisticRegression(max_iter=2000, random_state=0))
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, trained_pipeline):
    """A directory holding a serialized model + metadata, as the API expects."""
    path = tmp_path_factory.mktemp("model")
    joblib.dump(trained_pipeline, path / "model.joblib")
    (path / "metadata.json").write_text(
        json.dumps({"model_name": "logistic_regression", "trained_at": "test"})
    )
    return path


@pytest.fixture
def client(model_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.setenv("MODEL_PATH", str(model_dir / "model.joblib"))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(model_dir / "metadata.json"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    return {
        "age": 57,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 241,
        "fbs": 0,
        "restecg": 1,
        "thalach": 123,
        "exang": 1,
        "oldpeak": 0.2,
        "slope": 2,
        "ca": 0,
        "thal": 7,
    }
