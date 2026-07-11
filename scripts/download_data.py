"""Download and clean the UCI Heart Disease dataset.

Usage (from the project root):
    python -m scripts.download_data
"""

import logging

from src.config import CLEAN_DATA_PATH
from src.data import clean, download_raw, load_raw

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    download_raw()
    df = clean(load_raw())
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Saved cleaned dataset: {CLEAN_DATA_PATH} ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"Class balance:\n{df['target'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
