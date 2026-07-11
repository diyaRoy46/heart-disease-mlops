"""Train, tune and compare classifiers with MLflow experiment tracking.

For each candidate model a GridSearchCV (5-fold stratified, ROC-AUC) is run
inside an MLflow run that logs parameters, cross-validation and test metrics,
a confusion matrix and ROC curve, and the fitted pipeline. The best model by
test ROC-AUC is exported to models/model.joblib for serving.

Usage (from the project root):
    python -m src.train            # full grids
    python -m src.train --quick    # reduced grids (CI / smoke tests)
"""

import argparse
import json
import logging
import os
import platform
from datetime import UTC, datetime

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)

from src.config import (
    DEFAULT_TRACKING_URI,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from src.data import load_dataset, split_features_target
from src.pipeline import build_pipeline

logger = logging.getLogger(__name__)


def model_specs(quick: bool = False) -> dict:
    """Candidate models and their hyperparameter grids (pipeline step prefix)."""
    if quick:
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                {"classifier__C": [1.0]},
            ),
            "random_forest": (
                RandomForestClassifier(random_state=RANDOM_STATE),
                {"classifier__n_estimators": [100], "classifier__max_depth": [8]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [4, 8, None],
                "classifier__min_samples_leaf": [1, 3],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        ),
    }


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_plots(pipeline, X_test, y_test, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, display_labels=["No disease", "Disease"], ax=ax
    )
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title(f"ROC curve — {name}")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)


def train_all(quick: bool = False) -> dict:
    df = load_dataset()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    results = {}
    for name, (estimator, grid) in model_specs(quick).items():
        with mlflow.start_run(run_name=name) as run:
            logger.info("Tuning %s (%d candidates)", name, len(ParameterGrid(grid)))
            search = GridSearchCV(
                build_pipeline(estimator), grid, cv=cv, scoring="roc_auc", n_jobs=-1
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_

            metrics = evaluate(best, X_test, y_test)
            metrics["cv_roc_auc_mean"] = search.best_score_
            metrics["cv_roc_auc_std"] = search.cv_results_["std_test_score"][
                search.best_index_
            ]

            mlflow.set_tag("model_family", name)
            mlflow.log_param("model", type(estimator).__name__)
            mlflow.log_params(search.best_params_)
            mlflow.log_param("cv_folds", cv.get_n_splits())
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_metrics(metrics)
            log_plots(best, X_test, y_test, name)
            mlflow.sklearn.log_model(
                best,
                name="model",
                input_example=X_test.iloc[:3],
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            results[name] = {
                "pipeline": best,
                "metrics": metrics,
                "best_params": search.best_params_,
                "run_id": run.info.run_id,
            }
            logger.info(
                "%s: test ROC-AUC %.3f, accuracy %.3f",
                name,
                metrics["test_roc_auc"],
                metrics["test_accuracy"],
            )

    plot_roc_comparison(results, X_test, y_test)
    return results


def plot_roc_comparison(results: dict, X_test, y_test) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, res in results.items():
        RocCurveDisplay.from_estimator(res["pipeline"], X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("ROC curves — held-out test set")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_roc_curves.png", dpi=150)
    plt.close(fig)


def export_best(results: dict) -> str:
    """Persist the winning pipeline and its metadata; return its name."""
    best_name = max(results, key=lambda n: results[n]["metrics"]["test_roc_auc"])
    best = results[best_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_PATH)
    metadata = {
        "model_name": best_name,
        "best_params": best["best_params"],
        "metrics": {k: round(v, 4) for k, v in best["metrics"].items()},
        "mlflow_run_id": best["run_id"],
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    comparison = pd.DataFrame(
        {name: res["metrics"] for name, res in results.items()}
    ).T.round(4)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_comparison.md").write_text(
        "# Model comparison (held-out test set)\n\n"
        + comparison.to_markdown()
        + f"\n\nExported best model: **{best_name}** -> `{MODEL_PATH.name}`\n"
    )
    print(comparison.to_string())
    print(f"\nBest model: {best_name} -> {MODEL_PATH}")
    return best_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="reduced grids for CI/smoke runs"
    )
    args = parser.parse_args()
    results = train_all(quick=args.quick)
    export_best(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
