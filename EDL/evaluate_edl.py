# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pandas>=2.0", "scikit-learn>=1.4", "torch>=2.2", "joblib>=1.3"]
# ///
"""Evaluation script for Topology & Weight Evolving Neural Network (EDL) on the test split.

Evaluates:
1. Single Best Champion Evolved Network (`edl_model.pt`) with its evolved topology
2. Heterogeneous Topology Hall-of-Fame Ensemble (`edl_hof_elites.pt`)
3. Direct comparison against DL Single MLP (64, 32) and DL GBDT+MLP Ensemble.
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

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from EDL.models import EvolvableMLP, HallOfFameEnsemble, TopologyGenome
    from EDL.preprocess import make_splits
except ModuleNotFoundError:
    from models import EvolvableMLP, HallOfFameEnsemble, TopologyGenome
    from preprocess import make_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EDL Model")
    parser.add_argument("--models", default=str(ROOT / "models"), help="Path to EDL models directory")
    return parser.parse_args()


def load_champion_model(models_dir: Path, meta: dict) -> EvolvableMLP:
    ckpt = torch.load(models_dir / "edl_model.pt", weights_only=False)
    if isinstance(ckpt, dict) and "topology" in ckpt and "state_dict" in ckpt:
        topo = TopologyGenome.from_dict(ckpt["topology"])
        model = EvolvableMLP(topology=topo)
        model.load_state_dict(ckpt["state_dict"])
    else:
        topo = TopologyGenome.from_dict(meta.get("topology", meta))
        model = EvolvableMLP(topology=topo)
        model.load_state_dict(ckpt)
    model.eval()
    return model


def load_hof_models(models_dir: Path, champion: EvolvableMLP) -> list[EvolvableMLP]:
    hof_pt = models_dir / "edl_hof_elites.pt"
    if hof_pt.exists():
        bundle = torch.load(hof_pt, weights_only=False)
        elites = []
        for item in bundle:
            topo = TopologyGenome.from_dict(item["topology"])
            m = EvolvableMLP(topology=topo)
            m.load_state_dict(item["state_dict"])
            m.eval()
            if "weight" in item:
                m._ensemble_weight = float(item["weight"])
            elites.append(m)
        return elites
    return [champion]


def evaluate():
    args = parse_args()
    models_dir = Path(args.models)

    if not (models_dir / "edl_model.pt").exists():
        raise FileNotFoundError(f"Model checkpoint not found in {models_dir}")

    meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    classes = meta["classes"]
    champion = load_champion_model(models_dir, meta)
    elites = load_hof_models(models_dir, champion)

    print("=" * 76)
    print("  EVALUATION: TOPOLOGY & WEIGHT EVOLVING NEURAL NETWORK (EDL / TWEANN)")
    print(f"  Model Source: {models_dir}")
    print(f"  Champion Evolved Topology: {champion.topology.summary_str()} ({champion.num_params:,} params)")
    print(f"  Algorithm: {meta.get('algo', 'memetic').upper()} (Islands: {meta.get('pop_size', 'N/A')}, Best Gen: {meta.get('best_generation', 'N/A')})")
    print("=" * 76)

    if len(elites) > 1:
        print("\nEvolved Heterogeneous Topologies in Hall-of-Fame Ensemble:")
        for idx, m in enumerate(elites):
            print(f"  Island {idx}: {m.topology.summary_str():<46} ({m.num_params:,} params)")

    # 1. Load data
    print("\n[1/3] Loading held-out test split (20,000 samples)...")
    splits, _ = make_splits(color_indices=True)
    X_test, y_test = splits["test"]
    X_test = np.asarray(X_test, dtype=np.float32)

    # 2. Evaluate single champion & heterogeneous Hall-of-Fame ensemble
    print("[2/3] Evaluating Single Evolved Champion & Heterogeneous Topology Ensemble...")
    single_probs = champion.predict_proba(X_test)
    single_preds = np.argmax(single_probs, axis=1)
    single_acc = accuracy_score(y_test, single_preds)
    single_f1 = f1_score(y_test, single_preds, average="macro")

    mult_path = models_dir / "edl_hof_multipliers.npy"
    mults = np.load(mult_path) if mult_path.exists() else None
    hof_ens = HallOfFameEnsemble(elites, class_multipliers=mults)
    probs = hof_ens.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")
    cm = confusion_matrix(y_test, preds)

    print("\n[3/3] Held-out Test Set Results (20,000 rows):")
    print("-" * 76)
    print(f"  Single Evolved Champion   -> Accuracy: {single_acc * 100:.2f}% | Macro-F1: {single_f1:.4f}")
    print(f"  Heterogeneous HoF Ensemble -> Accuracy: {acc * 100:.2f}% | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")
    print("-" * 76)

    print("\nDetailed Per-Class Classification Report (EDL Heterogeneous HoF):")
    report_dict = classification_report(y_test, preds, target_names=classes, output_dict=True)
    print(classification_report(y_test, preds, target_names=classes, digits=4))

    print("Confusion Matrix (Counts):")
    print(f"{'True \\ Pred':>12} | " + " | ".join(f"{c:>8}" for c in classes))
    print("-" * 45)
    for i, c in enumerate(classes):
        row_str = " | ".join(f"{cm[i, j]:>8d}" for j in range(len(classes)))
        print(f"{c:>12} | {row_str}")

    print("\nConfusion Matrix (Normalized Row-wise Recall %):")
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    for i, c in enumerate(classes):
        row_str = " | ".join(f"{cm_norm[i, j]:>7.1f}%" for j in range(len(classes)))
        print(f"{c:>12} | {row_str}")

    print("\n" + "=" * 76)
    print("  BENCHMARK COMPARISON: DL vs. TOPOLOGY & WEIGHT EVOLVING EDL (TWEANN)")
    print("=" * 76)
    print(f"  {'Model / Paradigm':<40} | {'Test Accuracy':<15} | {'Test Macro-F1':<15}")
    print("  " + "-" * 72)
    print(f"  {'EDL Initial (Fixed 32-16, Pure ES)':<40} | {'96.48%':<15} | {'0.9599':<15}")
    print(f"  {'DL Single MLP (Fixed 64-32, Backprop)':<40} | {'97.37%':<15} | {'0.9700':<15}")
    print(f"  {'EDL Evolved Single Champion (TWEANN)':<40} | {f'{single_acc*100:.2f}%':<15} | {f'{single_f1:.4f}':<15}")
    print(f"  {'EDL Heterogeneous Topology HoF Ensemble':<40} | {f'{acc*100:.2f}%':<15} | {f'{macro_f1:.4f}':<15}")
    print(f"  {'DL Heterogeneous Ensemble (+GBDT)':<40} | {'97.88%':<15} | {'0.9755':<15}")
    print("=" * 76)

    metrics_out = {
        "champion_topology": champion.topology.summary_str(),
        "single_accuracy": round(single_acc, 4),
        "single_macro_f1": round(single_f1, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            c: {
                "precision": round(report_dict[c]["precision"], 4),
                "recall": round(report_dict[c]["recall"], 4),
                "f1-score": round(report_dict[c]["f1-score"], 4),
                "support": int(report_dict[c]["support"]),
            }
            for c in classes
        },
    }
    with open(models_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nSaved test metrics to {models_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    evaluate()
