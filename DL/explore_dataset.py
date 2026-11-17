# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "matplotlib>=3.8"]
# ///
"""Exploratory analysis of star_classification.csv.

Reports dataset size, feature dtypes, missing-value ratios, and saves
two figures to figures/: numeric feature distributions (histograms)
and boxplot-based outlier detection.

Run: uv run explore_dataset.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
from plot_style import CLASS_COLORS, CLASS_ORDER, INK, MUTED, SERIES_1, SURFACE

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

# Physically meaningful numeric features (ID/metadata columns excluded).
NUMERIC_FEATURES = ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]
# SDSS uses -9999 as an error sentinel in photometric magnitudes.
SENTINEL = -9999.0

plot_style.apply()


def report_overview(df: pd.DataFrame) -> None:
    print("=== Dataset size ===")
    print(f"rows: {len(df):,}  columns: {df.shape[1]}")
    print(f"memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print("\n=== Features and dtypes ===")
    for col, dtype in df.dtypes.items():
        print(f"{col:>12}  {dtype}")

    print("\n=== Class distribution ===")
    print(df["class"].value_counts().to_string())

    print("\n=== Missing values (NaN) ===")
    na_ratio = df.isna().mean()
    if na_ratio.sum() == 0:
        print("no NaN in any column")
    else:
        print(na_ratio[na_ratio > 0].to_string())

    print(f"\n=== Sentinel values ({SENTINEL:g}) ===")
    for col in NUMERIC_FEATURES:
        n = int((df[col] == SENTINEL).sum())
        if n:
            rows = df.index[df[col] == SENTINEL].tolist()
            print(f"{col}: {n} rows (index {rows})")


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric features with the -9999 sentinel treated as missing."""
    feats = df[NUMERIC_FEATURES].copy()
    return feats.mask(feats == SENTINEL)


def plot_distributions(feats: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        ax.hist(feats[col].dropna(), bins=60, color=SERIES_1, edgecolor=SURFACE, linewidth=0.3)
        ax.set_title(col, color=INK, fontsize=11)
        ax.tick_params(labelsize=8)
    fig.suptitle(
        "Numeric feature distributions (sentinel -9999 excluded)",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / "numeric_distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def iqr_outliers(s: pd.Series) -> tuple[int, float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum()), lo, hi


def plot_boxplots(feats: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    print("\n=== IQR outliers (1.5*IQR rule, sentinel excluded) ===")
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        s = feats[col].dropna()
        n_out, lo, hi = iqr_outliers(s)
        pct = n_out / len(s) * 100
        print(f"{col:>9}: {n_out:>6,} ({pct:5.2f}%)  fence [{lo:.3f}, {hi:.3f}]")
        ax.boxplot(
            s,
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": SERIES_1, "edgecolor": INK, "linewidth": 0.8},
            medianprops={"color": SURFACE, "linewidth": 1.2},
            whiskerprops={"color": MUTED, "linewidth": 0.8},
            capprops={"color": MUTED, "linewidth": 0.8},
            flierprops={
                "marker": ".",
                "markersize": 2.5,
                "markerfacecolor": MUTED,
                "markeredgecolor": "none",
                "alpha": 0.4,
            },
        )
        ax.set_title(f"{col}  ·  {n_out:,} outliers ({pct:.1f}%)", color=INK, fontsize=10)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)
    fig.suptitle(
        "Boxplot outlier detection, 1.5×IQR (sentinel -9999 excluded)",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / "outliers_boxplot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def outliers_by_class(feats: pd.DataFrame, classes: pd.Series) -> None:
    """Cross-tabulate IQR outliers with the class label.

    Verifies whether "outliers" concentrate in one class (signal)
    or spread like the base rate (noise candidates).
    """
    base = classes.value_counts(normalize=True)
    print("\n=== IQR outliers by class ===")
    print(f"{'':>9}  {'total':>7}  " + "  ".join(f"{c:>13}" for c in CLASS_ORDER))
    base_row = "  ".join(f"{base[c]*100:>12.1f}%" for c in CLASS_ORDER)
    print(f"{'baseline':>9}  {'':>7}  {base_row}")
    for col in NUMERIC_FEATURES:
        s = feats[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        n = int(mask.sum())
        if n == 0:
            print(f"{col:>9}  {0:>7,}")
            continue
        counts = classes[mask].value_counts()
        row = "  ".join(
            f"{counts.get(c, 0):>5,} ({counts.get(c, 0) / n * 100:4.1f}%)" for c in CLASS_ORDER
        )
        print(f"{col:>9}  {n:>7,}  {row}")


def plot_redshift_by_class(feats: pd.DataFrame, classes: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [feats.loc[classes == c, "redshift"].dropna() for c in CLASS_ORDER]
    boxes = ax.boxplot(
        data,
        tick_labels=CLASS_ORDER,
        widths=0.45,
        patch_artist=True,
        medianprops={"color": SURFACE, "linewidth": 1.2},
        whiskerprops={"color": MUTED, "linewidth": 0.8},
        capprops={"color": MUTED, "linewidth": 0.8},
        flierprops={
            "marker": ".",
            "markersize": 2.5,
            "markerfacecolor": MUTED,
            "markeredgecolor": "none",
            "alpha": 0.4,
        },
    )
    for patch, c in zip(boxes["boxes"], CLASS_ORDER):
        patch.set(facecolor=CLASS_COLORS[c], edgecolor=INK, linewidth=0.8)
    ax.set_ylabel("redshift")
    ax.set_title("Redshift by class", color=INK, fontsize=13)
    fig.tight_layout()
    out = FIGURES / "redshift_by_class.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "star_classification.csv")

    report_overview(df)
    feats = clean_features(df)

    print("\n=== Numeric feature summary (sentinel excluded) ===")
    print(feats.describe().T.to_string(float_format=lambda x: f"{x:.4f}"))

    outliers_by_class(feats, df["class"])

    dist = plot_distributions(feats)
    box = plot_boxplots(feats)
    rbc = plot_redshift_by_class(feats, df["class"])
    print(f"\nsaved: {dist.relative_to(ROOT)}")
    print(f"saved: {box.relative_to(ROOT)}")
    print(f"saved: {rbc.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
