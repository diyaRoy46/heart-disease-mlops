import pandas as pd

from src.config import ALL_FEATURES, COLUMN_NAMES, TARGET
from src.data import clean, load_raw


def _raw_like(num_values):
    """Minimal raw-schema frame with a given `num` column."""
    rows = len(num_values)
    data = {col: list(range(rows)) for col in COLUMN_NAMES}
    data["num"] = num_values
    return pd.DataFrame(data)


def test_clean_binarizes_target():
    df = clean(_raw_like([0, 1, 2, 3, 4]))
    assert list(df[TARGET]) == [0, 1, 1, 1, 1]
    assert "num" not in df.columns


def test_clean_drops_duplicates():
    raw = _raw_like([0, 0])
    raw.iloc[1] = raw.iloc[0]
    assert len(clean(raw)) == 1


def test_load_raw_parses_question_marks(tmp_path):
    row = ["63.0", "1.0", "1.0", "145.0", "233.0", "1.0", "2.0",
           "150.0", "0.0", "2.3", "3.0", "?", "?", "0"]
    path = tmp_path / "raw.data"
    path.write_text(",".join(row) + "\n")
    df = load_raw(path)
    assert df.loc[0, "ca"] != df.loc[0, "ca"]  # NaN
    assert df.loc[0, "thal"] != df.loc[0, "thal"]  # NaN
    assert df.loc[0, "age"] == 63.0


def test_committed_dataset_schema(dataset):
    assert set(dataset.columns) == set(ALL_FEATURES + [TARGET])
    assert dataset[TARGET].isin([0, 1]).all()
    assert len(dataset) >= 290  # 303 rows minus any future dedup
    assert dataset[TARGET].nunique() == 2
