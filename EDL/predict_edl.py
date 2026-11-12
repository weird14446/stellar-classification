# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pandas>=2.0", "scikit-learn>=1.4", "torch>=2.2", "joblib>=1.3"]
# ///
"""Inference CLI for Topology & Weight Evolving Neural Network (EDL / TWEANN).

Reads raw SDSS data, applies feature transformation and scaling,
and performs prediction using the evolved neural network topology and weights
(and Heterogeneous Hall-of-Fame ensemble if available).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for p in [str(ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
import pandas as pd
import torch

try:
    from EDL.evaluate_edl import load_champion_model, load_hof_models
    from EDL.models import HallOfFameEnsemble
    from EDL.preprocess import add_color_indices, make_splits
except ModuleNotFoundError:
    from evaluate_edl import load_champion_model, load_hof_models
    from models import HallOfFameEnsemble
    from preprocess import add_color_indices, make_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with EDL model")
    parser.add_argument("input", nargs="?", help="Input CSV path (omit for self-check)")
    parser.add_argument("-o", "--output", help="Output predictions CSV path")
    parser.add_argument("--models", default=str(ROOT / "models"), help="Path to models directory")
    return parser.parse_args()


def predict():
    args = parse_args()
    models_dir = Path(args.models)

    meta_file = models_dir / "metadata.json"
    weights_file = models_dir / "edl_model.pt"
    mult_file = models_dir / "edl_hof_multipliers.npy"
    pre_file = models_dir / "preprocessor.joblib"

    if not (meta_file.exists() and weights_file.exists() and pre_file.exists()):
        sys.exit(f"Error: Required model artifacts missing in {models_dir}")

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    classes = meta["classes"]

    champion = load_champion_model(models_dir, meta)
    elites = load_hof_models(models_dir, champion)
    mults = np.load(mult_file) if mult_file.exists() else None
    predictor = HallOfFameEnsemble(elites, class_multipliers=mults) if len(elites) > 1 else champion

    if args.input is None:
        print("No input provided. Running self-check on held-out test split...")
        splits, _ = make_splits(color_indices=True)
        X_test, y_test = splits["test"]
        probs = predictor.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)
        acc = (preds == y_test).mean()
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, preds, average="macro")
        print(f"Champion Evolved Topology: {champion.topology.summary_str()}")
        print("Self-check results on 20,000 test rows:")
        print(f"  Test Accuracy: {acc*100:.2f}% | Test Macro-F1: {f1:.4f}")
        print("Self-check OK!")
        return

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Error: File not found: {in_path}")

    df_raw = pd.read_csv(in_path)
    print(f"Loaded {len(df_raw)} rows from {in_path}")

    sentinel_cols = [c for c in ["u", "g", "r", "i", "z"] if c in df_raw.columns]
    bad = (df_raw[sentinel_cols] == -9999).any(axis=1) if sentinel_cols else pd.Series(False, index=df_raw.index)
    if bad.any():
        print(f"Dropping {bad.sum()} rows containing -9999 sentinel values.")
        df_clean = df_raw.loc[~bad].reset_index(drop=True)
    else:
        df_clean = df_raw.copy()

    pre = joblib.load(pre_file)
    df_fe = add_color_indices(df_clean)
    X_scaled = pre.transform(df_fe)

    probs = predictor.predict_proba(X_scaled)
    preds = np.argmax(probs, axis=1)

    out_df = df_clean.copy()
    out_df["pred_class"] = [classes[p] for p in preds]
    for i, c in enumerate(classes):
        out_df[f"prob_{c}"] = probs[:, i].round(4)

    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_edl_pred.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")

    if "class" in df_clean.columns:
        from sklearn.metrics import accuracy_score, f1_score
        y_true = df_clean["class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average="macro")
        print("Ground Truth Evaluation on input:")
        print(f"  Accuracy: {acc*100:.2f}% | Macro-F1: {f1:.4f}")


if __name__ == "__main__":
    predict()
