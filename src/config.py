"""Central configuration: paths, dataset schema and feature groups."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "processed.cleveland.data"
CLEAN_DATA_PATH = DATA_DIR / "heart.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# UCI Heart Disease (Cleveland) — https://archive.ics.uci.edu/dataset/45/heart+disease
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Column order of the raw file (it ships without a header row).
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

TARGET = "target"

# `num` is 0 (no disease) to 4 (severe); the standard formulation is binary.
RAW_TARGET = "num"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.2

MLFLOW_EXPERIMENT = "heart-disease-classification"
# MLflow 3.x deprecates the ./mlruns file store; default to a local sqlite DB.
# Override with the MLFLOW_TRACKING_URI environment variable.
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
