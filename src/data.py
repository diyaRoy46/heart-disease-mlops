"""Dataset acquisition and cleaning for the UCI Heart Disease (Cleveland) data."""

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    ALL_FEATURES,
    CLEAN_DATA_PATH,
    COLUMN_NAMES,
    DATA_URL,
    RAW_DATA_PATH,
    RAW_TARGET,
    TARGET,
)

logger = logging.getLogger(__name__)


def download_raw(dest: Path = RAW_DATA_PATH, url: str = DATA_URL) -> Path:
    """Download the raw Cleveland data file from the UCI repository."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the headerless raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize the target and drop exact duplicate rows.

    Missing values (only `ca` and `thal` have them) are intentionally kept:
    imputation lives inside the sklearn pipeline so it is fitted on training
    folds only and reused at inference time.
    """
    df = df.copy()
    df[TARGET] = (df[RAW_TARGET] > 0).astype(int)
    df = df.drop(columns=[RAW_TARGET])
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_dataset(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset, building it from the raw source if absent."""
    if path.exists():
        return pd.read_csv(path)
    if not RAW_DATA_PATH.exists():
        download_raw()
    df = clean(load_raw())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote cleaned dataset to %s (%d rows)", path, len(df))
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[ALL_FEATURES], df[TARGET]
