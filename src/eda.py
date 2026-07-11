"""Exploratory data analysis: saves figures and a summary to reports/.

Usage (from the project root):
    python -m src.eda
"""

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET,
)
from src.data import load_dataset, load_raw

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")

TARGET_LABELS = {0: "No disease", 1: "Disease"}


def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET].map(TARGET_LABELS).value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / len(df):.1%})", ha="center")
    ax.set_title("Class distribution")
    ax.set_ylabel("Patients")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_class_distribution.png", dpi=150)
    plt.close(fig)


def plot_missing_values(raw: pd.DataFrame) -> pd.Series:
    missing = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing.index, y=missing.values, hue=missing.index, legend=False, ax=ax)
    ax.set_title("Missing values per column (raw data)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_missing_values.png", dpi=150)
    plt.close(fig)
    return missing[missing > 0]


def plot_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES, strict=False):
        sns.histplot(df, x=col, hue=TARGET, kde=True, element="step", ax=ax)
        ax.set_title(col)
    axes.flat[-1].axis("off")
    fig.suptitle("Numeric feature distributions by target", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_histograms.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation heatmap (incl. target)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    return corr


def plot_numeric_by_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(18, 4))
    labeled = df.assign(status=df[TARGET].map(TARGET_LABELS))
    for ax, col in zip(axes, NUMERIC_FEATURES, strict=True):
        sns.boxplot(labeled, x="status", y=col, hue="status", legend=False, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Numeric features vs. heart disease", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_numeric_by_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_categorical_by_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    labeled = df.assign(status=df[TARGET].map(TARGET_LABELS))
    for ax, col in zip(axes.flat, CATEGORICAL_FEATURES, strict=True):
        sns.countplot(labeled, x=col, hue="status", ax=ax)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Categorical features vs. heart disease", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_categorical_by_target.png", dpi=150)
    plt.close(fig)


def write_summary(df: pd.DataFrame, missing: pd.Series, corr: pd.DataFrame) -> None:
    target_corr = (
        corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3)
    )
    lines = [
        "# EDA summary — UCI Heart Disease (Cleveland)",
        "",
        f"- Rows: {len(df)}, features: {df.shape[1] - 1}",
        f"- Class balance: {df[TARGET].value_counts().to_dict()} "
        "(0 = no disease, 1 = disease)",
        f"- Missing values (raw): {missing.to_dict() or 'none'} — imputed in the pipeline",
        "",
        "## Correlation with target (sorted by |r|)",
        "",
        target_corr.to_markdown(),
        "",
        "## Descriptive statistics",
        "",
        df.describe().round(2).T.to_markdown(),
        "",
    ]
    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines))


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    if not RAW_DATA_PATH.exists():
        from src.data import download_raw

        download_raw()
    raw = load_raw()

    plot_class_distribution(df)
    missing = plot_missing_values(raw)
    plot_histograms(df)
    corr = plot_correlation_heatmap(df)
    plot_numeric_by_target(df)
    plot_categorical_by_target(df)
    write_summary(df, missing, corr)
    print(f"Figures written to {FIGURES_DIR}")
    print(f"Summary written to {REPORTS_DIR / 'eda_summary.md'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
