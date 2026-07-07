# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "matplotlib>=3.8"]
# ///
"""Class-wise distributions of photometric magnitudes and color indices.

Compares GALAXY / STAR / QSO as density histograms (density=True so the
imbalanced class sizes don't hide shape differences) and saves:
  figures/magnitudes_by_class.png
  figures/color_indices_by_class.png

Run: uv run compare_class_distributions.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
from plot_style import CLASS_COLORS, CLASS_ORDER, INK
from preprocess import COLOR_INDICES, PHOTOMETRIC, TARGET, add_color_indices, load_clean

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

plot_style.apply()


def plot_by_class(df: pd.DataFrame, columns: list[str], title: str, filename: str) -> Path:
    ncols = len(columns)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.6))
    for ax, col in zip(np.atleast_1d(axes), columns):
        lo, hi = df[col].quantile([0.001, 0.999])
        bins = np.linspace(lo, hi, 70)
        for cls in CLASS_ORDER:
            ax.hist(
                df.loc[df[TARGET] == cls, col].clip(lo, hi),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=CLASS_COLORS[cls],
                label=cls,
            )
        ax.set_title(col, color=INK, fontsize=11)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", ncols=3, frameon=False, fontsize=10,
        bbox_to_anchor=(0.99, 1.0),
    )
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGURES / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def print_medians(df: pd.DataFrame, columns: list[str], title: str) -> None:
    med = df.groupby(TARGET)[columns].median().reindex(CLASS_ORDER)
    print(f"\n=== {title} (median by class) ===")
    print(med.to_string(float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    df = add_color_indices(load_clean())

    print_medians(df, PHOTOMETRIC, "magnitudes")
    print_medians(df, list(COLOR_INDICES), "color indices")

    mag = plot_by_class(
        df, PHOTOMETRIC, "Photometric magnitude distributions by class",
        "magnitudes_by_class.png",
    )
    ci = plot_by_class(
        df, list(COLOR_INDICES), "Color index distributions by class",
        "color_indices_by_class.png",
    )
    print(f"\nsaved: {mag.relative_to(ROOT)}")
    print(f"saved: {ci.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
