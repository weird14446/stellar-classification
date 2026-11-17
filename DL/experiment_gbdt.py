# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "scikit-learn>=1.4", "matplotlib>=3.8", "torch>=2.2", "xgboost>=2.0"]
# ///
"""Gradient boosting vs MLP, and heterogeneous soft-voting ensembles.

Trains XGBoost and HistGradientBoosting on the final feature set
(10 features incl. color indices), compares them with the adopted MLP
(3 seeds, averaged softmax) on validation macro-F1, then evaluates the
best variant on the held-out test set. Saves figures/gbdt_ensemble.png.

torch and xgboost each bundle their own OpenMP runtime and segfault
when used in one process on macOS, so each model family is trained in
its own subprocess (this file re-invokes itself with a stage argument)
and probabilities are exchanged via .npy files.

Run: uv run experiment_gbdt.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
CACHE = Path(tempfile.gettempdir()) / "stellar-gbdt-probs"
MLP_SEEDS = [0, 1, 2]
SPLITS = ("val", "test")


def stage_mlp() -> None:
    import torch

    from experiment_torch import make_data, train_one

    tensors, classes = make_data(color=True)
    np.save(CACHE / "classes.npy", np.array(classes))
    for split in SPLITS:
        np.save(CACHE / f"y_{split}.npy", tensors[split][1].numpy())

    probs = {split: [] for split in SPLITS}
    for seed in MLP_SEEDS:
        model, val_f1, _ = train_one(tensors, use_weight=False, seed=seed)
        model.eval()
        with torch.no_grad():
            for split in SPLITS:
                probs[split].append(torch.softmax(model(tensors[split][0]), dim=1).numpy())
        print(f"  MLP seed {seed}: val macro-F1 {val_f1:.4f}", flush=True)
    for split in SPLITS:
        np.save(CACHE / f"mlp_{split}.npy", np.mean(probs[split], axis=0))


def stage_xgb() -> None:
    from xgboost import XGBClassifier

    from preprocess import make_splits

    splits, _ = make_splits(color_indices=True)
    clf = XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        early_stopping_rounds=100, eval_metric="mlogloss",
        n_jobs=-1, random_state=42,
    )
    clf.fit(*splits["train"], eval_set=[splits["val"]], verbose=False)
    print(f"  XGBoost: {clf.best_iteration + 1} trees", flush=True)
    for split in SPLITS:
        np.save(CACHE / f"xgb_{split}.npy", clf.predict_proba(splits[split][0]))


def stage_histgb() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from preprocess import make_splits

    splits, _ = make_splits(color_indices=True)
    clf = HistGradientBoostingClassifier(
        max_iter=2000, learning_rate=0.05, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=50, random_state=42,
    )
    clf.fit(*splits["train"])
    print(f"  HistGB: {clf.n_iter_} trees", flush=True)
    for split in SPLITS:
        np.save(CACHE / f"histgb_{split}.npy", clf.predict_proba(splits[split][0]))


def combine() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    import plot_style
    from plot_style import INK, SERIES_1

    plot_style.apply()
    classes = list(np.load(CACHE / "classes.npy"))
    y = {split: np.load(CACHE / f"y_{split}.npy") for split in SPLITS}
    single = {
        name: {split: np.load(CACHE / f"{key}_{split}.npy") for split in SPLITS}
        for name, key in [("MLP x3 (soft)", "mlp"), ("XGBoost", "xgb"), ("HistGB", "histgb")]
    }
    probs = dict(single)
    probs["MLP + XGB"] = {
        s: (single["MLP x3 (soft)"][s] + single["XGBoost"][s]) / 2 for s in SPLITS
    }
    probs["MLP + XGB + HistGB"] = {
        s: sum(p[s] for p in single.values()) / 3 for s in SPLITS
    }

    rows = []
    for name, p in probs.items():
        val_f1 = f1_score(y["val"], p["val"].argmax(axis=1), average="macro")
        rows.append({"config": name, "val_macro_f1": val_f1})
        print(f"{name:>20}  val macro-F1 {val_f1:.4f}")
    results = pd.DataFrame(rows)

    best_name = results.loc[results["val_macro_f1"].idxmax(), "config"]
    pred_test = probs[best_name]["test"].argmax(axis=1)
    test_f1 = f1_score(y["test"], pred_test, average="macro")
    test_acc = accuracy_score(y["test"], pred_test)
    print(f"\nbest on validation: {best_name}")
    print(f"test macro-F1 {test_f1:.4f}  accuracy {test_acc:.4f}")
    print("\n=== per-class test report (best) ===")
    print(classification_report(y["test"], pred_test, target_names=classes, digits=4))

    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ypos = np.arange(len(results))[::-1]
    ax.plot(results["val_macro_f1"], ypos, "o", markersize=9, color=SERIES_1)
    for r, yp in zip(results.itertuples(), ypos):
        ax.annotate(
            f"{r.val_macro_f1:.4f}", (r.val_macro_f1, yp),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK,
        )
    ax.set_yticks(ypos, results["config"])
    ax.set_ylim(-0.5, len(results) - 0.5 + 0.6)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlabel("validation macro-F1")
    ax.set_title("GBDT vs MLP and soft-voting ensembles", color=INK, fontsize=13)
    fig.tight_layout()
    out = FIGURES / "gbdt_ensemble.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out.relative_to(ROOT)}")


STAGES = {"mlp": stage_mlp, "xgb": stage_xgb, "histgb": stage_histgb}


def main() -> None:
    if len(sys.argv) > 1:
        CACHE.mkdir(exist_ok=True)
        STAGES[sys.argv[1]]()
        return
    t0 = perf_counter()
    for stage in STAGES:
        print(f"training {stage} (subprocess) ...", flush=True)
        subprocess.run([sys.executable, __file__, stage], check=True)
    print(f"total training {perf_counter() - t0:.0f}s\n")
    combine()


if __name__ == "__main__":
    main()
