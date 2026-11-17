# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.12", "scikit-learn>=1.4", "pandas>=2.0", "joblib>=1.3"]
# ///
"""Evaluate GravNet on the 20,000-row held-out test set & analyze Lagrange L1 saddle points."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from GravNet.preprocess import FEATURES, TARGET, add_color_indices, load_clean
from GravNet.physics_model import GravNet, GravNetMultiverse

MODELS_DIR = ROOT / "models"


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "per_class": {c: report[c] for c in classes},
    }


def main():
    pre_bundle = joblib.load(MODELS_DIR / "preprocessor.joblib")
    preprocessor = pre_bundle["preprocessor"]
    classes = pre_bundle["classes"]

    # Reconstruct exact test split alongside raw unscaled redshift for astrophysical analysis
    df = add_color_indices(load_clean())
    feature_cols = [c for c in df.columns if c != TARGET]
    y_all = df[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    _, df_test, _, y_test = train_test_split(
        df[feature_cols], y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_test = preprocessor.transform(df_test)
    raw_redshift_test = df_test["redshift"].to_numpy()

    single_model = GravNet.from_checkpoint(joblib.load(MODELS_DIR / "gravnet_single.joblib"))
    multiverse = GravNetMultiverse.from_checkpoint(joblib.load(MODELS_DIR / "gravnet_multiverse.joblib"))
    history = json.loads((MODELS_DIR / "history.json").read_text(encoding="utf-8"))

    # 1. Standard Classification Metrics
    single_preds = single_model.predict(X_test)
    multi_probs = multiverse.predict_proba(X_test)
    multi_preds = np.argmax(multi_probs, axis=1)

    single_metrics = compute_metrics_dict(y_test, single_preds, classes)
    multi_metrics = compute_metrics_dict(y_test, multi_preds, classes)

    # 2. Lagrange L1 Saddle-Point Physical Uncertainty Analysis
    lag_info = single_model.lagrange_saddle_metrics(X_test)
    barrier_gap = lag_info["barrier_gap"]
    residual_ke = lag_info["residual_ke"]
    correct = (single_preds == y_test).astype(np.float64)

    # Selective coverage curve: keep top k% highest Lagrange barrier_gap (deepest captures)
    order_desc = np.argsort(-barrier_gap)
    coverages = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    selective_curve = []
    for cov in coverages:
        k = max(1, int(round(cov * len(y_test))))
        sub_idx = order_desc[:k]
        selective_curve.append(
            {
                "coverage": cov,
                "min_barrier_gap": float(barrier_gap[sub_idx[-1]]),
                "accuracy": float(np.mean(correct[sub_idx])),
                "macro_f1": float(f1_score(y_test[sub_idx], single_preds[sub_idx], average="macro")),
            }
        )

    # Compare bottom 5% (Lagrange L1 Ridge Objects) vs top 95% (Deep Gravitational Basin Objects)
    q05 = float(np.quantile(barrier_gap, 0.05))
    ridge_mask = barrier_gap <= q05
    basin_mask = ~ridge_mask

    qso_idx = classes.index("QSO")
    qso_ridge_z = raw_redshift_test[(y_test == qso_idx) & ridge_mask]
    qso_basin_z = raw_redshift_test[(y_test == qso_idx) & basin_mask]

    lagrange_summary = {
        "q05_barrier_threshold": q05,
        "deep_basin_95pct_accuracy": float(np.mean(correct[basin_mask])),
        "lagrange_ridge_5pct_accuracy": float(np.mean(correct[ridge_mask])),
        "deep_basin_mean_residual_ke": float(np.mean(residual_ke[basin_mask])),
        "lagrange_ridge_mean_residual_ke": float(np.mean(residual_ke[ridge_mask])),
        "qso_median_redshift_deep_basin": float(np.median(qso_basin_z)) if len(qso_basin_z) else 0.0,
        "qso_median_redshift_lagrange_ridge": float(np.median(qso_ridge_z)) if len(qso_ridge_z) else 0.0,
        "selective_coverage_curve": selective_curve,
    }

    output = {
        "single_universe": single_metrics,
        "multiverse_3": multi_metrics,
        "lagrange_l1_analysis": lagrange_summary,
        "timing": {
            "single_train_sec": history["single_universe_train_sec"],
            "multiverse_train_sec": history["multiverse_train_sec"],
            "single_inference_20k_ms": history["inference_20k_ms"]["single_universe"]["median_ms"],
            "multiverse_inference_20k_ms": history["inference_20k_ms"]["multiverse_3"]["median_ms"],
        },
        "benchmarks_reference": {
            "DL_Single_MLP": {"accuracy": 0.9737, "macro_f1": 0.9700, "train_sec": 24.16, "inf_ms": 2.83},
            "DL_3Seed_MLP": {"accuracy": 0.9742, "macro_f1": 0.9705, "train_sec": 72.49, "inf_ms": 8.49},
            "DL_Full_GBDT_MLP": {"accuracy": 0.9788, "macro_f1": 0.9755, "train_sec": 81.35, "inf_ms": 121.86},
            "EDL_Pure_ES": {"accuracy": 0.9648, "macro_f1": 0.9599, "train_sec": 13.55, "inf_ms": 1.42},
            "EDL_TWEANN_Single": {"accuracy": 0.9743, "macro_f1": 0.9706, "train_sec": 94.22, "inf_ms": 7.08},
            "EDL_TWEANN_Top4": {"accuracy": 0.9745, "macro_f1": 0.9709, "train_sec": 94.22, "inf_ms": 27.26},
        },
    }

    (MODELS_DIR / "test_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=== GravNet 20,000-Sample Held-Out Test Evaluation ===")
    print(
        f"  Single Universe (Seed 42)   -> Acc: {single_metrics['accuracy']*100:.2f}% | "
        f"Macro-F1: {single_metrics['macro_f1']:.4f} | "
        f"Train: {output['timing']['single_train_sec']:.2f}s | "
        f"Inf (20k): {output['timing']['single_inference_20k_ms']:.2f}ms"
    )
    print(
        f"  3-Universe Multiverse       -> Acc: {multi_metrics['accuracy']*100:.2f}% | "
        f"Macro-F1: {multi_metrics['macro_f1']:.4f} | "
        f"Train: {output['timing']['multiverse_train_sec']:.2f}s | "
        f"Inf (20k): {output['timing']['multiverse_inference_20k_ms']:.2f}ms"
    )
    print("\n=== Lagrange L1 Saddle-Point Physical Uncertainty ===")
    print(f"  Deep Basin (95% objects) Acc:     {lagrange_summary['deep_basin_95pct_accuracy']*100:.2f}%")
    print(f"  Lagrange L1 Ridge (5% objects):   {lagrange_summary['lagrange_ridge_5pct_accuracy']*100:.2f}%")
    print(f"  QSO Median Redshift (Deep Basin): z = {lagrange_summary['qso_median_redshift_deep_basin']:.3f}")
    print(f"  QSO Median Redshift (L1 Ridge):   z = {lagrange_summary['qso_median_redshift_lagrange_ridge']:.3f}")


if __name__ == "__main__":
    main()
