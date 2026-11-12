# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26"]
# ///
"""Generate scientific publication-quality visualization figures for TWEANN EDL.

Generates:
1. figures/evolution_curves.png: 2-panel telemetry of fitness and topology complexity evolution.
2. figures/confusion_matrix.png: Normalized confusion matrix heatmap on held-out test set.
3. figures/dl_vs_edl_comparison.png: Side-by-side metric comparison (Pure ES vs DL MLP vs TWEANN EDL).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_BEST = "#0969da"      # Blue
COLOR_MEAN = "#8c959f"      # Gray
COLOR_VAL_F1 = "#1a7f37"    # Green
COLOR_VAL_ACC = "#8250df"   # Purple
COLOR_PREV = "#8c959f"      # Gray
COLOR_DL = "#cf222e"        # Red
COLOR_EDL = "#0969da"       # Blue


def plot_evolution_curves():
    hist_path = MODELS_DIR / "history.json"
    if not hist_path.exists():
        print("history.json not found, skipping evolution curve plot.")
        return

    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    gens = [h["generation"] for h in history]
    train_best = [h["train_best_fitness"] for h in history]
    train_mean = [h["train_mean_fitness"] for h in history]

    val_gens = [h["generation"] for h in history if h.get("val_macro_f1") is not None]
    val_f1 = [h["val_macro_f1"] for h in history if h.get("val_macro_f1") is not None]
    val_acc = [h["val_accuracy"] for h in history if h.get("val_accuracy") is not None]

    best_params = [h.get("best_num_params", 0) for h in history]
    mean_params = [h.get("mean_num_params", 0) for h in history]

    has_topology_stats = any(p > 0 for p in best_params)
    if has_topology_stats:
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(8.5, 6.8), dpi=300, gridspec_kw={"height_ratios": [2.2, 1.0]}, sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax2 = None

    ax.plot(gens, train_best, label="Train Island Best (Macro-F1)", color=COLOR_BEST, linewidth=2)
    ax.plot(gens, train_mean, label="Train Island Mean", color=COLOR_MEAN, linestyle="--", linewidth=1.5, alpha=0.8)
    if val_gens:
        ax.plot(val_gens, val_f1, label="Validation Split Macro-F1", color=COLOR_VAL_F1, marker="o", markersize=3.5, linewidth=2)
        ax.plot(val_gens, val_acc, label="Validation Split Accuracy", color=COLOR_VAL_ACC, marker="s", markersize=3.0, linewidth=1.8, linestyle=":")

    ax.set_title("Topology & Weight Evolving Neural Network (TWEANN) Convergence", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Metric / Fitness Score", fontsize=11)
    ax.set_ylim([0.90, 0.982])
    ax.set_xlim([1, max(gens)])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#d0d7de", loc="lower right", fontsize=9.5)

    if ax2 is not None:
        ax2.plot(gens, best_params, label="Champion Topology Parameter Count", color="#bf8700", linewidth=2)
        ax2.plot(gens, mean_params, label="Population Mean Parameter Count", color=COLOR_MEAN, linestyle="--", linewidth=1.5)
        ax2.set_xlabel("Generation", fontsize=11)
        ax2.set_ylabel("Parameters", fontsize=10.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(frameon=True, facecolor="white", edgecolor="#d0d7de", loc="upper left", fontsize=9)
    else:
        ax.set_xlabel("Generation", fontsize=11)

    plt.tight_layout()
    out_file = FIGURES_DIR / "evolution_curves.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_confusion_matrix():
    metrics_path = MODELS_DIR / "test_metrics.json"
    meta_path = MODELS_DIR / "metadata.json"
    if not metrics_path.exists() or not meta_path.exists():
        print("test_metrics.json or metadata.json not found, skipping confusion matrix.")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta["classes"]
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (Normalized)", fontsize=10)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, labelpad=8)
    ax.set_title("TWEANN EDL Test Set Confusion Matrix", fontsize=12, fontweight="bold", pad=12)

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            text = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    out_file = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_comparison():
    edl_metrics_path = MODELS_DIR / "test_metrics.json"
    if not edl_metrics_path.exists():
        return

    with open(edl_metrics_path, "r", encoding="utf-8") as f:
        edl_metrics = json.load(f)

    prev_edl_vals = [96.48, 95.99]
    dl_mlp_vals = [97.37, 97.00]
    imp_edl_vals = [edl_metrics["accuracy"] * 100, edl_metrics["macro_f1"] * 100]

    labels = ["Test Accuracy (%)", "Test Macro-F1 (×100)"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=300)
    rects1 = ax.bar(x - width, prev_edl_vals, width, label="Initial EDL (Pure ES, Fixed 32-16)", color=COLOR_PREV, alpha=0.85)
    rects2 = ax.bar(x, dl_mlp_vals, width, label="DL Baseline MLP (Backprop, Fixed 64-32)", color=COLOR_DL, alpha=0.9)
    rects3 = ax.bar(x + width, imp_edl_vals, width, label="TWEANN EDL (Topology + Weight Evolved)", color=COLOR_EDL, alpha=0.95)

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Performance Comparison: Pure ES vs. DL MLP vs. TWEANN EDL", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([94.0, 99.0])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#d0d7de", fontsize=9.5, loc="lower right")

    for rects in (rects1, rects2, rects3):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    plt.tight_layout()
    out_file = FIGURES_DIR / "dl_vs_edl_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_training_time_comparison():
    models = [
        "EDL Pure ES\n(190 gens)",
        "EDL TWEANN\n(Per Island, 45 gens)",
        "DL Single MLP\n(1 Seed Mean)",
        "EDL Memetic\n(Fixed 64-32, 8 Islands)",
        "DL 3-Seed MLP\n(Seeds 0,1,2)",
        "DL Full Ensemble\n(MLP×3+XGB+HistGB)",
        "EDL TWEANN\n(All 8 Islands, 45 gens)",
    ]
    times = [13.55, 11.78, 24.16, 63.61, 72.49, 81.35, 94.22]
    colors = [
        "#8c959f",
        "#2da44e",
        "#cf222e",
        "#54aeff",
        "#cf222e",
        "#a40e26",
        "#0969da",
    ]

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=300)
    y = np.arange(len(models))
    bars = ax.barh(y, times, color=colors, height=0.62, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Wall-Clock Training Time (seconds, Apple Silicon CPU)", fontsize=10.5)
    ax.set_title("Training Time Comparison: Deep Learning (DL) vs. Evolutionary DL (EDL)", fontsize=11.0, fontweight="bold", pad=12)
    ax.set_xlim([0, 110])
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.2f} s",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    out_file = FIGURES_DIR / "training_time_comparison.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_evolution_curves()
    plot_confusion_matrix()
    plot_comparison()
    plot_training_time_comparison()
