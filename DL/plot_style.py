"""Shared chart chrome and class color mapping for project figures.

Values follow the dataviz reference palette (light mode).
"""

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"  # categorical slot 1 (blue)

# Fixed class → color mapping (categorical slots 1-3), never reordered.
CLASS_ORDER = ["GALAXY", "STAR", "QSO"]
CLASS_COLORS = {"GALAXY": "#2a78d6", "STAR": "#1baf7a", "QSO": "#eda100"}


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
