# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "scikit-learn>=1.4", "matplotlib>=3.8", "torch>=2.2"]
# ///
"""Majority-vote ensemble of the final model configuration.

Trains 10 copies of the adopted model (PyTorch MLP (64, 32), 10
features incl. color indices, no class weight) with freshly drawn
random seeds, combines them by hard majority voting, and evaluates on
the held-out test set. Saves figures/ensemble_comparison.png.

Seeds are random on every run (they are printed for the record), so
exact numbers vary slightly between runs.

Run: uv run experiment_ensemble.py
"""

import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

import plot_style
from plot_style import INK, MUTED, SERIES_1
from experiment_torch import macro_f1, make_data, train_one

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

N_MODELS = 10
N_CLASSES = 3


def predict(model, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(X).argmax(dim=1).numpy()


def majority_vote(preds: np.ndarray) -> np.ndarray:
    """Plurality vote over axis 0; ties resolve to the lowest class index."""
    counts = np.stack([(preds == c).sum(axis=0) for c in range(N_CLASSES)])
    return counts.argmax(axis=0)


def main() -> None:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    plot_style.apply()
    FIGURES.mkdir(exist_ok=True)

    seeds = random.sample(range(1_000_000), N_MODELS)
    print(f"seeds: {seeds}")

    tensors, classes = make_data(color=True)
    X_val, y_val = tensors["val"]
    X_test, y_test = tensors["test"]

    val_preds, test_preds, test_f1s = [], [], []
    for k, seed in enumerate(seeds, 1):
        t0 = perf_counter()
        model, val_f1, best_epoch = train_one(tensors, use_weight=False, seed=seed)
        val_preds.append(predict(model, X_val))
        pred = predict(model, X_test)
        test_preds.append(pred)
        f1 = f1_score(y_test.numpy(), pred, average="macro")
        test_f1s.append(f1)
        print(
            f"model {k:>2}/{N_MODELS} (seed {seed:>6})  val {val_f1:.4f}"
            f"  test {f1:.4f}  epoch {best_epoch}  ({perf_counter() - t0:.0f}s)"
        )

    test_f1s = np.array(test_f1s)
    print(f"\nindividual test macro-F1: {test_f1s.mean():.4f} ± {test_f1s.std():.4f}"
          f"  [{test_f1s.min():.4f}, {test_f1s.max():.4f}]")

    ens_val = majority_vote(np.stack(val_preds))
    ens_test = majority_vote(np.stack(test_preds))
    ens_val_f1 = f1_score(y_val.numpy(), ens_val, average="macro")
    ens_f1 = f1_score(y_test.numpy(), ens_test, average="macro")
    ens_acc = accuracy_score(y_test.numpy(), ens_test)
    print(f"ensemble (majority vote)  val macro-F1 {ens_val_f1:.4f}")
    print(f"ensemble (majority vote)  test macro-F1 {ens_f1:.4f}  accuracy {ens_acc:.4f}")
    print("\n=== ensemble per-class test report ===")
    print(classification_report(y_test.numpy(), ens_test, target_names=classes, digits=4))

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(
        test_f1s, np.ones_like(test_f1s), "o", markersize=8,
        color=MUTED, alpha=0.6, label=f"individual models (n={N_MODELS})",
    )
    ax.plot([ens_f1], [0], "D", markersize=10, color=SERIES_1, label="majority-vote ensemble")
    ax.annotate(
        f"{ens_f1:.4f}", (ens_f1, 0), textcoords="offset points",
        xytext=(0, 12), ha="center", fontsize=9, color=INK,
    )
    ax.set_yticks([1, 0], ["individual", "ensemble"])
    ax.set_ylim(-0.7, 1.7)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlabel("test macro-F1")
    ax.set_title("Majority-vote ensemble vs individual models", color=INK, fontsize=13)
    fig.tight_layout()
    out = FIGURES / "ensemble_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
