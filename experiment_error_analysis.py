# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "scikit-learn>=1.4", "matplotlib>=3.8", "torch>=2.2"]
# ///
"""Error analysis of the final model on the test set.

Trains the final configuration (MLP (64, 32), 10 features) and inspects
where the test-set misclassifications live in feature space (redshift,
magnitude), to estimate how much headroom remains toward higher
accuracy. Saves figures/error_analysis.png.

Run: uv run experiment_error_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap

import plot_style
from plot_style import CLASS_ORDER, GRID, INK, MUTED, SERIES_1, SURFACE
from experiment_torch import macro_f1, make_data, train_one
from preprocess import COLOR_INDICES, FEATURES, TARGET, add_color_indices, load_clean

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
SEED = 0
ERROR_RED = "#e34948"  # categorical slot 6, fixed meaning here: misclassified

# Sequential blue ramp (steps 100->700) for the confusion heatmap.
BLUES = LinearSegmentedColormap.from_list(
    "seq-blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
)


def rebuild_test_frame(pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Re-derive the raw (unscaled) test rows; the split is deterministic."""
    from sklearn.model_selection import train_test_split

    df = add_color_indices(load_clean())
    y_all = df[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    features = [*FEATURES, *COLOR_INDICES]
    _, X_test, _, y_test = train_test_split(
        df[features], y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    out = X_test.copy()
    out["true"] = np.array(classes)[y_test]
    out["pred"] = np.array(classes)[pred]
    out["correct"] = out["true"] == out["pred"]
    return out


def main() -> None:
    from sklearn.metrics import confusion_matrix

    plot_style.apply()
    FIGURES.mkdir(exist_ok=True)

    tensors, classes = make_data(color=True)
    model, val_f1, _ = train_one(tensors, use_weight=False, seed=SEED)
    test_f1, pred = macro_f1(model, *tensors["test"])
    print(f"model (seed {SEED})  val macro-F1 {val_f1:.4f}  test macro-F1 {test_f1:.4f}")

    t = rebuild_test_frame(pred, classes)
    errors = t[~t["correct"]]
    print(f"\ntest errors: {len(errors):,} / {len(t):,} ({len(errors) / len(t) * 100:.2f}%)")

    print("\n=== error counts by true -> pred ===")
    pair_counts = errors.groupby(["true", "pred"]).size().sort_values(ascending=False)
    print(pair_counts.to_string())

    print("\n=== where the errors live (median [10%, 90%]) ===")
    def spread(s: pd.Series) -> str:
        q = s.quantile([0.1, 0.5, 0.9])
        return f"{q[0.5]:7.3f}  [{q[0.1]:7.3f}, {q[0.9]:7.3f}]"

    for cls in CLASS_ORDER:
        ok = t[(t["true"] == cls) & t["correct"]]
        bad = t[(t["true"] == cls) & ~t["correct"]]
        if len(bad) == 0:
            continue
        print(f"{cls} (errors {len(bad):,} / {len(ok) + len(bad):,}):")
        for col in ("redshift", "r"):
            print(f"  {col:>8}  correct {spread(ok[col])}   error {spread(bad[col])}")

    # How faint are the errors? Compare against the overall r distribution.
    r75 = t["r"].quantile(0.75)
    frac_faint = (errors["r"] > r75).mean()
    print(f"\nerrors fainter than the overall r 75th percentile ({r75:.2f}): {frac_faint * 100:.1f}%")

    galaxy_qso = errors[(errors["true"] == "GALAXY") & (errors["pred"] == "QSO")]
    qso_galaxy = errors[(errors["true"] == "QSO") & (errors["pred"] == "GALAXY")]
    gg = len(galaxy_qso) + len(qso_galaxy)
    print(f"GALAXY<->QSO confusion share of all errors: {gg / len(errors) * 100:.1f}%")

    # --- figure: 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    cm = confusion_matrix(t["true"], t["pred"], labels=CLASS_ORDER)
    ax.imshow(cm, cmap=BLUES, norm=plt.matplotlib.colors.LogNorm())
    for i in range(3):
        for j in range(3):
            frac = cm[i, j] / cm[i].sum()
            color = SURFACE if frac > 0.5 else INK
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10, color=color)
    ax.set_xticks(range(3), CLASS_ORDER)
    ax.set_yticks(range(3), CLASS_ORDER)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test, log color scale)", color=INK, fontsize=11)
    ax.grid(False)

    def compare(ax, sub_ok, sub_bad, col, title, bins):
        ax.hist(sub_ok[col], bins=bins, density=True, histtype="step",
                linewidth=1.8, color=SERIES_1, label="correct")
        ax.hist(sub_bad[col], bins=bins, density=True, histtype="step",
                linewidth=1.8, color=ERROR_RED, label="misclassified")
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel(col)
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=9)

    qso = t[t["true"] == "QSO"]
    compare(
        axes[0, 1], qso[qso["correct"]], qso[~qso["correct"]], "redshift",
        "QSO: redshift of correct vs misclassified", np.linspace(0, 4, 60),
    )
    gal = t[t["true"] == "GALAXY"]
    compare(
        axes[1, 0], gal[gal["correct"]], gal[~gal["correct"]], "redshift",
        "GALAXY: redshift of correct vs misclassified", np.linspace(0, 2, 60),
    )
    compare(
        axes[1, 1], t[t["correct"]], errors, "r",
        "All classes: r magnitude of correct vs misclassified",
        np.linspace(t["r"].quantile(0.001), t["r"].quantile(0.999), 60),
    )

    fig.suptitle("Where the final model's test errors live", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURES / "error_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
