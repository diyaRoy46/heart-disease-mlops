import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import split_features_target
from src.pipeline import build_pipeline, build_preprocessor


def test_preprocessor_removes_missing_values(dataset):
    X, _ = split_features_target(dataset)
    assert X.isna().any().any(), "dataset should contain missing ca/thal values"
    transformed = build_preprocessor().fit_transform(X)
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()


def test_pipeline_fits_and_predicts(dataset):
    X, y = split_features_target(dataset)
    pipeline = build_pipeline(LogisticRegression(max_iter=2000))
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X.head(5))
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(pipeline.predict(X.head(5))) <= {0, 1}


def test_pipeline_handles_unseen_category(dataset, trained_pipeline):
    X, _ = split_features_target(dataset)
    weird = X.head(1).copy()
    weird["thal"] = 99.0  # category never seen in training
    prediction = trained_pipeline.predict(weird)
    assert prediction[0] in (0, 1)
