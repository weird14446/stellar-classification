# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "scikit-learn>=1.4", "matplotlib>=3.8"]
# ///
"""Hidden-layer depth comparison for the MLP classifier.

Trains logistic regression (0 hidden layers) and MLPs with 1/2/3
hidden layers on the preprocess.py pipeline, compares validation
macro-F1 (3 seeds for MLPs), and saves figures/depth_comparison.png.
The 20% test split is held out untouched for the final model.

Run: uv run experiment_depth.py
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style
from plot_style import INK, MUTED, SERIES_1
from preprocess import FEATURES, TARGET, build_preprocessor, load_clean

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

# name -> hidden_layer_sizes (None = logistic regression baseline)
CONFIGS: dict[str, tuple[int, ...] | None] = {
    "LogReg (0 hidden)": None,
    "MLP 1 (64)": (64,),
    "MLP 2 (64, 32)": (64, 32),
    "MLP 3 (64, 32, 16)": (64, 32, 16),
}
SEEDS = [0, 1, 2]

plot_style.apply()


def make_model(hidden: tuple[int, ...] | None, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    if hidden is None:
        return LogisticRegression(max_iter=1000, random_state=seed)
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        batch_size=512,
        early_stopping=True,
        n_iter_no_change=10,
        max_iter=200,
        random_state=seed,
    )


def main() -> None:
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import train_test_split

    FIGURES.mkdir(exist_ok=True)
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]

    # Hold out the final test set, then carve validation from the rest.
    X_trainval, _X_test, y_trainval, _y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )
    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train)
    Xt_val = pre.transform(X_val)
    print(f"train {Xt_train.shape}  val {Xt_val.shape}  (test 20,000 held out)")

    rows = []
    val_preds = {}
    for name, hidden in CONFIGS.items():
        seeds = [0] if hidden is None else SEEDS  # LogReg is deterministic
        f1s, accs, times, epochs = [], [], [], []
        for seed in seeds:
            clf = make_model(hidden, seed)
            t0 = perf_counter()
            clf.fit(Xt_train, y_train)
            times.append(perf_counter() - t0)
            pred = clf.predict(Xt_val)
            f1s.append(f1_score(y_val, pred, average="macro"))
            accs.append(accuracy_score(y_val, pred))
            epochs.append(getattr(clf, "n_iter_", np.nan))
            val_preds[name] = pred  # keep last seed's predictions
        rows.append(
            {
                "config": name,
                "val_macro_f1": np.mean(f1s),
                "f1_std": np.std(f1s),
                "val_accuracy": np.mean(accs),
                "fit_seconds": np.mean(times),
                "epochs": np.mean(epochs),
            }
        )
        print(
            f"{name:>20}  macro-F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
            f"  acc {np.mean(accs):.4f}  {np.mean(times):.1f}s  epochs {np.mean(epochs):.0f}"
        )

    results = pd.DataFrame(rows)
    best = results.loc[results["val_macro_f1"].idxmax()]
    print(f"\nbest: {best['config']}")
    print("\n=== per-class report (best config, last seed) ===")
    print(classification_report(y_val, val_preds[best["config"]], digits=4))

    # Dot plot (non-zero baseline is fine for dots, unlike bars).
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ypos = np.arange(len(results))[::-1]
    ax.errorbar(
        results["val_macro_f1"], ypos, xerr=results["f1_std"],
        fmt="o", markersize=9, color=SERIES_1, ecolor=MUTED, capsize=3, linestyle="none",
    )
    for _, r, yp in zip(range(len(results)), results.itertuples(), ypos):
        ax.annotate(
            f"{r.val_macro_f1:.4f}", (r.val_macro_f1, yp),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK,
        )
    ax.set_yticks(ypos, results["config"])
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlabel("validation macro-F1 (mean ± std over seeds)")
    ax.set_title("Hidden-layer depth comparison", color=INK, fontsize=13)
    fig.tight_layout()
    out = FIGURES / "depth_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
