# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "scikit-learn>=1.4", "matplotlib>=3.8", "torch>=2.2"]
# ///
"""PyTorch MLP ablation: class weight and color-index features.

Reimplements the adopted 2-hidden-layer (64, 32) MLP in PyTorch and
compares four variants (baseline / +class weight / +color indices /
+both) on validation macro-F1 (3 seeds each), then evaluates the best
variant on the held-out test set. Saves figures/torch_improvements.png.

Run: uv run experiment_torch.py
"""

import copy
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import plot_style
from plot_style import INK, MUTED, SERIES_1
from preprocess import COLOR_INDICES, FEATURES, TARGET, add_color_indices, build_preprocessor, load_clean

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

CONFIGS = {
    "baseline (6 feat)": {"color": False, "weight": False},
    "+ class weight": {"color": False, "weight": True},
    "+ color indices": {"color": True, "weight": False},
    "+ both": {"color": True, "weight": True},
}
SEEDS = [0, 1, 2]
HIDDEN = (64, 32)
BATCH = 512
MAX_EPOCHS = 200
PATIENCE = 15
DEVICE = "cpu"  # net is tiny; CPU is fast and reproducible


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_data(color: bool):
    """Split, scale (fit on train only), and return tensors + class names."""
    from sklearn.model_selection import train_test_split

    df = load_clean()
    if color:
        df = add_color_indices(df)
    features = [*FEATURES, *COLOR_INDICES] if color else FEATURES
    classes = sorted(df[TARGET].unique())  # GALAXY, QSO, STAR
    y_all = df[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df[features], y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )
    pre = build_preprocessor(color_indices=color)
    splits = {
        "train": (pre.fit_transform(X_train), y_train),
        "val": (pre.transform(X_val), y_val),
        "test": (pre.transform(X_test), y_test),
    }
    tensors = {
        k: (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        for k, (X, y) in splits.items()
    }
    return tensors, classes


def macro_f1(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, np.ndarray]:
    from sklearn.metrics import f1_score

    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1).numpy()
    return f1_score(y.numpy(), pred, average="macro"), pred


def train_one(tensors, use_weight: bool, seed: int) -> tuple[nn.Module, float, int]:
    torch.manual_seed(seed)
    X_train, y_train = tensors["train"]
    X_val, y_val = tensors["val"]

    model = MLP(X_train.shape[1]).to(DEVICE)
    if use_weight:
        counts = torch.bincount(y_train)
        weight = len(y_train) / (len(counts) * counts.float())
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_f1, best_state, best_epoch, stall = 0.0, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        val_f1, _ = macro_f1(model, X_val, y_val)
        if val_f1 > best_f1 + 1e-4:
            best_f1, best_epoch, stall = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stall += 1
            if stall >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_f1, best_epoch


def main() -> None:
    from sklearn.metrics import accuracy_score, classification_report, recall_score

    plot_style.apply()
    FIGURES.mkdir(exist_ok=True)

    data = {color: make_data(color) for color in (False, True)}
    rows, models = [], {}
    for name, cfg in CONFIGS.items():
        tensors, classes = data[cfg["color"]]
        qso_idx = classes.index("QSO")
        f1s, recalls, epochs = [], [], []
        t0 = perf_counter()
        for seed in SEEDS:
            model, val_f1, best_epoch = train_one(tensors, cfg["weight"], seed)
            _, pred = macro_f1(model, *tensors["val"])
            f1s.append(val_f1)
            recalls.append(
                recall_score(tensors["val"][1].numpy(), pred, labels=[qso_idx], average=None)[0]
            )
            epochs.append(best_epoch)
            models[(name, seed)] = model
        rows.append(
            {
                "config": name,
                "val_macro_f1": np.mean(f1s),
                "f1_std": np.std(f1s),
                "qso_recall": np.mean(recalls),
                "epochs": np.mean(epochs),
            }
        )
        print(
            f"{name:>20}  macro-F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
            f"  QSO recall {np.mean(recalls):.4f}  best epoch {np.mean(epochs):.0f}"
            f"  ({perf_counter() - t0:.0f}s)"
        )

    results = pd.DataFrame(rows)
    best_name = results.loc[results["val_macro_f1"].idxmax(), "config"]
    print(f"\nbest on validation: {best_name}")

    # Final held-out test evaluation of the best variant (all seeds).
    tensors, classes = data[CONFIGS[best_name]["color"]]
    X_test, y_test = tensors["test"]
    test_f1s, test_accs = [], []
    for seed in SEEDS:
        f1, pred = macro_f1(models[(best_name, seed)], X_test, y_test)
        test_f1s.append(f1)
        test_accs.append(accuracy_score(y_test.numpy(), pred))
    print(
        f"\n=== test set ({len(y_test):,} rows, {best_name}) ===\n"
        f"macro-F1 {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}"
        f"  accuracy {np.mean(test_accs):.4f}"
    )
    mid_seed = SEEDS[int(np.argsort(test_f1s)[len(test_f1s) // 2])]
    _, pred = macro_f1(models[(best_name, mid_seed)], X_test, y_test)
    print(f"\n=== per-class test report (median seed {mid_seed}) ===")
    print(classification_report(y_test.numpy(), pred, target_names=classes, digits=4))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ypos = np.arange(len(results))[::-1]
    ax.errorbar(
        results["val_macro_f1"], ypos, xerr=results["f1_std"],
        fmt="o", markersize=9, color=SERIES_1, ecolor=MUTED, capsize=3, linestyle="none",
    )
    for r, yp in zip(results.itertuples(), ypos):
        ax.annotate(
            f"{r.val_macro_f1:.4f}", (r.val_macro_f1, yp),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK,
        )
    ax.set_yticks(ypos, results["config"])
    ax.set_ylim(-0.5, len(results) - 0.5 + 0.6)  # headroom so labels clear the title
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlabel("validation macro-F1 (mean ± std over seeds)")
    ax.set_title("PyTorch MLP (64, 32) — class weight & color indices", color=INK, fontsize=13)
    fig.tight_layout()
    out = FIGURES / "torch_improvements.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
