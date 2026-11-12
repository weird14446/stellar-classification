# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "scikit-learn>=1.4", "matplotlib>=3.8", "torch>=2.2", "xgboost>=2.0", "joblib>=1.3"]
# ///
"""Train and save the final model: MLP+XGB+HistGB soft-voting ensemble.

Trains the adopted configuration (10 features incl. color indices) and
saves everything needed for inference into --outdir (default models/):

  preprocessor.joblib   fitted scaling pipeline
  mlp_seed<k>.pt        MLP (64, 32) state dicts
  xgboost.json          XGBoost booster
  histgb.joblib         HistGradientBoosting
  metadata.json         classes, feature list, seeds, test metrics

Like experiment_gbdt.py, torch and xgboost segfault when mixed in one
process on macOS (bundled-OpenMP conflict), so each model family runs
in its own subprocess and probabilities pass through <outdir>/cache/.

Run: uv run train_final.py [--seeds 0,1,2] [--outdir models]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPLITS = ("val", "test")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("stage", nargs="?", choices=["mlp", "xgb", "histgb"])
    p.add_argument("--seeds", default="0,1,2", help="comma-separated MLP seeds")
    p.add_argument("--outdir", default="models")
    return p.parse_args()


def stage_mlp(outdir: Path, cache: Path, seeds: list[int]) -> None:
    import torch

    from experiment_torch import make_data, train_one

    tensors, classes = make_data(color=True)
    np.save(cache / "classes.npy", np.array(classes))
    for split in SPLITS:
        np.save(cache / f"y_{split}.npy", tensors[split][1].numpy())

    probs = {split: [] for split in SPLITS}
    for seed in seeds:
        model, val_f1, _ = train_one(tensors, use_weight=False, seed=seed)
        torch.save(model.state_dict(), outdir / f"mlp_seed{seed}.pt")
        model.eval()
        with torch.no_grad():
            for split in SPLITS:
                probs[split].append(torch.softmax(model(tensors[split][0]), dim=1).numpy())
        print(f"  MLP seed {seed}: val macro-F1 {val_f1:.4f}", flush=True)
    for split in SPLITS:
        np.save(cache / f"mlp_{split}.npy", np.mean(probs[split], axis=0))


def stage_xgb(outdir: Path, cache: Path) -> None:
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
    clf.save_model(outdir / "xgboost.json")
    print(f"  XGBoost: {clf.best_iteration + 1} trees", flush=True)
    for split in SPLITS:
        np.save(cache / f"xgb_{split}.npy", clf.predict_proba(splits[split][0]))


def stage_histgb(outdir: Path, cache: Path) -> None:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    from preprocess import make_splits

    splits, _, pre = make_splits(color_indices=True, return_preprocessor=True)
    joblib.dump(pre, outdir / "preprocessor.joblib")
    clf = HistGradientBoostingClassifier(
        max_iter=2000, learning_rate=0.05, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=50, random_state=42,
    )
    clf.fit(*splits["train"])
    joblib.dump(clf, outdir / "histgb.joblib")
    print(f"  HistGB: {clf.n_iter_} trees", flush=True)
    for split in SPLITS:
        np.save(cache / f"histgb_{split}.npy", clf.predict_proba(splits[split][0]))


def finalize(outdir: Path, cache: Path, seeds: list[int]) -> None:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    from preprocess import COLOR_INDICES, FEATURES

    classes = [str(c) for c in np.load(cache / "classes.npy")]
    metrics = {}
    for split in SPLITS:
        y = np.load(cache / f"y_{split}.npy")
        prob = np.mean(
            [np.load(cache / f"{m}_{split}.npy") for m in ("mlp", "xgb", "histgb")], axis=0
        )
        pred = prob.argmax(axis=1)
        metrics[split] = {
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
        }
        print(f"{split}: {metrics[split]}")
        if split == "test":
            print(classification_report(y, pred, target_names=classes, digits=4))

    meta = {
        "model": "soft-voting ensemble: MLP(64,32) x len(seeds) + XGBoost + HistGB",
        "classes": classes,
        "features": [*FEATURES, *COLOR_INDICES],
        "mlp_seeds": seeds,
        "metrics": metrics,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"saved model artifacts to {outdir}/")


def main() -> None:
    args = parse_args()
    outdir = (ROOT / args.outdir).resolve()
    cache = outdir / "cache"
    seeds = [int(s) for s in args.seeds.split(",")]
    cache.mkdir(parents=True, exist_ok=True)

    if args.stage == "mlp":
        stage_mlp(outdir, cache, seeds)
    elif args.stage == "xgb":
        stage_xgb(outdir, cache)
    elif args.stage == "histgb":
        stage_histgb(outdir, cache)
    else:
        for stage in ("mlp", "xgb", "histgb"):
            print(f"training {stage} (subprocess) ...", flush=True)
            subprocess.run(
                [sys.executable, __file__, stage, "--seeds", args.seeds, "--outdir", args.outdir],
                check=True,
            )
        finalize(outdir, cache, seeds)


if __name__ == "__main__":
    main()
