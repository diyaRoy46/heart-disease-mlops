from src.data import split_features_target
from src.train import evaluate, model_specs, train_all


def test_model_specs_has_at_least_two_models():
    assert len(model_specs()) >= 2
    assert len(model_specs(quick=True)) >= 2


def test_evaluate_returns_bounded_metrics(dataset, trained_pipeline):
    X, y = split_features_target(dataset)
    metrics = evaluate(trained_pipeline, X, y)
    expected = {"test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"}
    assert set(metrics) == expected
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_quick_training_run(tmp_path, monkeypatch, dataset):
    db_path = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    results = train_all(quick=True)

    assert len(results) >= 2
    for res in results.values():
        assert res["metrics"]["test_roc_auc"] > 0.8, "model should clearly beat chance"
        assert res["run_id"]
    # runs were actually recorded in the tracking store
    assert db_path.exists()
