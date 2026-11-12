# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.12", "scikit-learn>=1.4", "pandas>=2.0", "joblib>=1.3"]
# ///
"""CLI inference and 20,000-sample self-check for GravNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from GravNet.preprocess import FEATURES, add_color_indices, make_splits
from GravNet.physics_model import GravNet, GravNetMultiverse

MODELS_DIR = ROOT / "models"


def run_self_check() -> None:
    splits, classes = make_splits(color_indices=True)
    X_test, y_test = splits["test"]

    single = GravNet.from_checkpoint(joblib.load(MODELS_DIR / "gravnet_single.joblib"))
    multiverse = GravNetMultiverse.from_checkpoint(joblib.load(MODELS_DIR / "gravnet_multiverse.joblib"))

    pred_s = single.predict(X_test)
    pred_m = multiverse.predict(X_test)

    acc_s = accuracy_score(y_test, pred_s)
    f1_s = f1_score(y_test, pred_s, average="macro")
    acc_m = accuracy_score(y_test, pred_m)
    f1_m = f1_score(y_test, pred_m, average="macro")

    print("=== GravNet Self-Check on 20,000 Held-Out Test Celestial Objects ===")
    print(f"  Classes: {classes}")
    print(f"  [Single GravNet Universe]   Test Accuracy: {acc_s*100:.2f}% | Macro-F1: {f1_s:.4f}")
    print(f"  [3-Universe Multiverse]     Test Accuracy: {acc_m*100:.2f}% | Macro-F1: {f1_m:.4f}")


def predict_csv(input_csv: Path, output_csv: Path | None, mode: str = "multiverse") -> None:
    bundle = joblib.load(MODELS_DIR / "preprocessor.joblib")
    preprocessor = bundle["preprocessor"]
    classes = bundle["classes"]

    df_raw = pd.read_csv(input_csv)
    missing = [c for c in FEATURES if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required SDSS columns: {missing}")

    df_feat = add_color_indices(df_raw[FEATURES])
    X = preprocessor.transform(df_feat)

    single = GravNet.from_checkpoint(joblib.load(MODELS_DIR / "gravnet_single.joblib"))
    lag_info = single.lagrange_saddle_metrics(X)

    if mode == "single":
        probs = lag_info["probs"]
    else:
        multiverse = GravNetMultiverse.from_checkpoint(joblib.load(MODELS_DIR / "gravnet_multiverse.joblib"))
        probs = multiverse.predict_proba(X)

    pred_idx = np.argmax(probs, axis=1)
    out = df_raw.copy()
    out["predicted_class"] = [classes[i] for i in pred_idx]
    for i, c in enumerate(classes):
        out[f"prob_{c}"] = np.round(probs[:, i], 6)
    out["lagrange_barrier_gap"] = np.round(lag_info["barrier_gap"], 6)

    if output_csv is not None:
        out.to_csv(output_csv, index=False)
        print(f"Saved {len(out)} predictions to {output_csv}")
    else:
        print(out[["predicted_class", *[f"prob_{c}" for c in classes], "lagrange_barrier_gap"]].head(10).to_string())


def main():
    parser = argparse.ArgumentParser(description="GravNet Celestial Orbital Predictor")
    parser.add_argument("--input", type=Path, default=None, help="Input CSV with u,g,r,i,z,redshift")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path")
    parser.add_argument("--mode", choices=["single", "multiverse"], default="multiverse")
    args = parser.parse_args()

    if args.input is None:
        run_self_check()
    else:
        predict_csv(args.input, args.output, mode=args.mode)


if __name__ == "__main__":
    main()
