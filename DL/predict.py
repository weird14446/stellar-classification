# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "scikit-learn>=1.4", "torch>=2.2", "xgboost>=2.0", "joblib>=1.3"]
# ///
"""Inference with the saved final ensemble (see train_final.py).

Reads a CSV with the raw SDSS columns (u, g, r, i, z, redshift), adds
color indices, applies the saved preprocessor, averages the softmax /
probability outputs of the saved MLPs + XGBoost + HistGB, and writes
<input>_predictions.csv with a pred_class column and per-class
probabilities. Rows containing the -9999 sentinel are dropped with a
warning. If the input has a `class` column, accuracy is reported.

Without an input file it runs a self-check: rebuilds the canonical
held-out test split and verifies the metrics stored in metadata.json.

torch and xgboost segfault when mixed in one process on macOS, so each
model family predicts in its own subprocess (same pattern as training).

Run: uv run predict.py [input.csv] [--models models] [-o out.csv]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", help="CSV with raw feature columns; omit for self-check")
    p.add_argument("--stage", choices=["mlp", "xgb", "histgb"], help=argparse.SUPPRESS)
    p.add_argument("--models", default="models")
    p.add_argument("-o", "--output", help="output CSV path")
    return p.parse_args()


def load_features(models: Path, cache: Path) -> np.ndarray:
    """Scaled feature matrix for the cleaned input parked in the cache."""
    import joblib

    from preprocess import add_color_indices

    df = add_color_indices(pd.read_csv(cache / "input.csv"))
    pre = joblib.load(models / "preprocessor.joblib")
    return pre.transform(df)


def stage_mlp(models: Path, cache: Path) -> None:
    import torch

    from model import MLP

    meta = json.loads((models / "metadata.json").read_text())
    X = torch.tensor(load_features(models, cache), dtype=torch.float32)
    probs = []
    for seed in meta["mlp_seeds"]:
        model = MLP(X.shape[1])
        model.load_state_dict(torch.load(models / f"mlp_seed{seed}.pt", weights_only=True))
        model.eval()
        with torch.no_grad():
            probs.append(torch.softmax(model(X), dim=1).numpy())
    np.save(cache / "mlp.npy", np.mean(probs, axis=0))


def stage_xgb(models: Path, cache: Path) -> None:
    from xgboost import XGBClassifier

    clf = XGBClassifier()
    clf.load_model(models / "xgboost.json")
    np.save(cache / "xgb.npy", clf.predict_proba(load_features(models, cache)))


def stage_histgb(models: Path, cache: Path) -> None:
    import joblib

    clf = joblib.load(models / "histgb.joblib")
    np.save(cache / "histgb.npy", clf.predict_proba(load_features(models, cache)))


STAGES = {"mlp": stage_mlp, "xgb": stage_xgb, "histgb": stage_histgb}


def self_check_frame() -> pd.DataFrame:
    """Raw rows of the canonical held-out test split, with labels."""
    from sklearn.model_selection import train_test_split

    from preprocess import TARGET, load_clean

    df = load_clean()
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    return test_df.reset_index(drop=True)


def main() -> None:
    from preprocess import FEATURES, SENTINEL, TARGET

    args = parse_args()
    models = (ROOT / args.models).resolve()
    cache = models / "cache_predict"
    if args.stage:
        STAGES[args.stage](models, cache)
        return

    meta = json.loads((models / "metadata.json").read_text())
    if args.input:
        df = pd.read_csv(args.input)
        bad = (df[FEATURES] == SENTINEL).any(axis=1)
        if bad.any():
            print(f"warning: dropping {bad.sum()} row(s) with sentinel {SENTINEL:g}")
            df = df.loc[~bad].reset_index(drop=True)
        out_path = Path(args.output) if args.output else Path(f"{Path(args.input).stem}_predictions.csv")
    else:
        print("no input given — running self-check on the held-out test split")
        df = self_check_frame()
        out_path = None

    cache.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache / "input.csv", index=False)
    for stage in STAGES:
        subprocess.run(
            [sys.executable, __file__, "--stage", stage, "--models", args.models],
            check=True,
        )

    prob = np.mean([np.load(cache / f"{m}.npy") for m in STAGES], axis=0)
    classes = np.array(meta["classes"])
    result = df.copy()
    result["pred_class"] = classes[prob.argmax(axis=1)]
    for i, c in enumerate(classes):
        result[f"prob_{c}"] = prob[:, i].round(4)

    if TARGET in result.columns:
        from sklearn.metrics import accuracy_score, f1_score

        acc = accuracy_score(result[TARGET], result["pred_class"])
        f1 = f1_score(result[TARGET], result["pred_class"], average="macro")
        print(f"accuracy {acc:.4f}  macro-F1 {f1:.4f}  ({len(result):,} rows)")
        if not args.input:
            stored = meta["metrics"]["test"]
            print(f"stored test metrics: {stored}")
            assert abs(acc - stored["accuracy"]) < 1e-3, "self-check mismatch"
            print("self-check OK")

    if out_path:
        result.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(result):,} rows)")


if __name__ == "__main__":
    main()
