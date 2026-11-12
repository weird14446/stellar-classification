# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.12", "scikit-learn>=1.4", "joblib>=1.3", "sympy>=1.12"]
# ///
"""Train GravNet (Single Universe & 3-Universe Multiverse) with 100% Backprop-Free Physics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from GravNet.preprocess import make_splits
from GravNet.physics_model import (
    GravNet,
    GravNetMultiverse,
    fast_macro_f1,
    verify_physics_rigor,
)

MODELS_DIR = ROOT / "models"


def train_single_universe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    epochs: int = 18,
    orbital_steps: int = 10,
    fourier_dim: int = 2304,
    n_1d_freqs: int = 44,
    sub_halos_per_class: int = 6,
    ridge_lambda: float = 2.5,
    verbose: bool = True,
):
    t_start = time.perf_counter()
    model = GravNet(
        in_dim=X_train.shape[1],
        n_classes=3,
        phase_dim=6,
        sub_halos_per_class=sub_halos_per_class,
        n_1d_freqs=n_1d_freqs,
        fourier_dim=fourier_dim,
        orbital_steps=orbital_steps,
        dt=0.11,
        damping=0.95,
        softening=0.90,
        confinement=0.04,
        inv_temp=2.6,
        seed=seed,
    )

    # Stage 1: Lens projection + K-Means/SVD Proto-Halo Condensation + SPD Cholesky Launch
    t_s1 = time.perf_counter()
    Phi_train = model.lens.transform(X_train)
    Phi_val = model.lens.transform(X_val)
    model.initialize_proto_halos_and_lens(
        X_train, y_train, Phi_train=Phi_train, ridge_lambda=ridge_lambda
    )
    stage1_sec = time.perf_counter() - t_s1

    val_probs_s1 = model.predict_proba_from_phi(Phi_val)
    val_preds_s1 = np.argmax(val_probs_s1, axis=1)
    best_val_f1 = fast_macro_f1(y_val, val_preds_s1)
    best_val_acc = float(np.mean(val_preds_s1 == y_val))
    best_state = model.copy_mutable_weights()

    if verbose:
        print(
            f"[Seed {seed:3d}] Stage 1 Proto-Halo & Cholesky ({stage1_sec:.2f}s) -> "
            f"Val Acc: {best_val_acc*100:.2f}% | Val Macro-F1: {best_val_f1:.4f}"
        )

    epoch_logs = [
        {
            "epoch": 0,
            "train_nll": None,
            "val_acc": best_val_acc,
            "val_macro_f1": best_val_f1,
        }
    ]

    # Stage 2: Contrastive Spacetime Sculpting + Direct Terminal Tidal Hebbian Plasticity
    t_s2 = time.perf_counter()
    rng = np.random.default_rng(seed)
    for ep in range(1, epochs + 1):
        cosine = 0.5 * (1.0 + np.cos(np.pi * (ep - 1) / max(epochs, 1)))
        nll = model.sculpt_spacetime_epoch(
            Phi_train,
            y_train,
            batch_size=2048,
            lr_vel=0.042 * cosine + 0.003,
            lr_fiber=0.014 * cosine + 0.001,
            lr_mass=0.028 * cosine,
            lr_halo=0.012 * cosine,
            weight_decay=1e-5,
            rng=rng,
        )
        val_probs = model.predict_proba_from_phi(Phi_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = float(np.mean(val_preds == y_val))
        val_f1 = fast_macro_f1(y_val, val_preds)

        epoch_logs.append(
            {
                "epoch": ep,
                "train_nll": float(nll),
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_state = model.copy_mutable_weights()

        if verbose and (ep in (1, 3, 6, 10, 14, epochs)):
            print(
                f"  Epoch {ep:2d}/{epochs} | NLL: {nll:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}"
            )

    model.load_mutable_weights(best_state)
    stage2_sec = time.perf_counter() - t_s2

    # Stage 3: Grand-Canonical Chemical Potential Calibration across Lagrange L1 Ridges
    t_s3 = time.perf_counter()
    cal_f1 = model.calibrate_chemical_potentials(Phi_val, y_val, n_trials=140, seed=seed)
    val_preds_cal = np.argmax(model.predict_proba_from_phi(Phi_val), axis=1)
    cal_acc = float(np.mean(val_preds_cal == y_val))
    stage3_sec = time.perf_counter() - t_s3
    total_sec = time.perf_counter() - t_start

    if verbose:
        print(
            f"[Seed {seed:3d}] Stage 3 Lagrange Calibration ({stage3_sec:.2f}s) -> "
            f"Val Acc: {cal_acc*100:.2f}% | Val Macro-F1: {cal_f1:.4f} | Total: {total_sec:.2f}s\n"
        )

    timing = {
        "stage1_condensation_sec": stage1_sec,
        "stage2_sculpting_sec": stage2_sec,
        "stage3_calibration_sec": stage3_sec,
        "total_train_sec": total_sec,
    }
    return model, epoch_logs, timing, Phi_val


def benchmark_inference_ms(predictor, X_test: np.ndarray, n_runs: int = 15) -> Dict[str, float]:
    # Warm-up runs
    for _ in range(2):
        _ = predictor.predict_proba(X_test)
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = predictor.predict_proba(X_test)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "median_ms": float(np.median(times_ms)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train GravNet Astrophysical Orbital Network")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--orbital-steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 271])
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Running Symbolic & Numerical Physics Rigor Self-Verification...")
    physics_checks = verify_physics_rigor()
    for k, v in physics_checks.items():
        print(f"   {k}: {v:.3e}")

    print("\n2. Preparing Deterministic SDSS-17 Splits (63,999 / 16,000 / 20,000)...")
    splits, classes, preprocessor = make_splits(color_indices=True, return_preprocessor=True)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    joblib.dump({"preprocessor": preprocessor, "classes": classes}, MODELS_DIR / "preprocessor.joblib")

    print("\n3. Training GravNet Universes (100% Backpropagation-Free)...")
    universes = []
    all_logs = {}
    all_timings = {}
    primary_phi_val = None

    t_multiverse_start = time.perf_counter()
    for i, sd in enumerate(args.seeds):
        u_model, ep_logs, u_time, phi_v = train_single_universe(
            X_train,
            y_train,
            X_val,
            y_val,
            seed=sd,
            epochs=args.epochs,
            orbital_steps=args.orbital_steps,
            verbose=True,
        )
        universes.append(u_model)
        all_logs[str(sd)] = ep_logs
        all_timings[str(sd)] = u_time
        if i == 0:
            primary_phi_val = phi_v
    multiverse_train_sec = time.perf_counter() - t_multiverse_start

    primary_model = universes[0]
    multiverse = GravNetMultiverse(universes)

    # Evaluate Orbital Step Progression T = 0..16 on Validation & Test for Primary Universe
    step_grid = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16]
    Phi_test_primary = primary_model.lens.transform(X_test)
    orbital_progression = []
    for st in step_grid:
        vp = np.argmax(primary_model.predict_proba_from_phi(primary_phi_val, steps=st), axis=1)
        tp = np.argmax(primary_model.predict_proba_from_phi(Phi_test_primary, steps=st), axis=1)
        orbital_progression.append(
            {
                "steps": st,
                "val_acc": float(np.mean(vp == y_val)),
                "val_macro_f1": fast_macro_f1(y_val, vp),
                "test_acc": float(np.mean(tp == y_test)),
                "test_macro_f1": fast_macro_f1(y_test, tp),
            }
        )

    print("4. Measuring 20,000-Sample Inference Latency...")
    single_inf = benchmark_inference_ms(primary_model, X_test, n_runs=15)
    multi_inf = benchmark_inference_ms(multiverse, X_test, n_runs=10)
    print(f"   Single GravNet Inference (20k):     {single_inf['median_ms']:.2f} ms (mean {single_inf['mean_ms']:.2f} ms)")
    print(f"   3-Universe Multiverse Inference:    {multi_inf['median_ms']:.2f} ms (mean {multi_inf['mean_ms']:.2f} ms)")

    # Quick Test Summary Preview
    single_test_preds = primary_model.predict(X_test)
    multi_test_preds = multiverse.predict(X_test)
    print(
        f"\n[Test Result] Single GravNet Universe (seed={args.seeds[0]}): "
        f"Acc = {np.mean(single_test_preds == y_test)*100:.2f}% | "
        f"Macro-F1 = {fast_macro_f1(y_test, single_test_preds):.4f} | "
        f"Train = {all_timings[str(args.seeds[0])]['total_train_sec']:.2f}s"
    )
    print(
        f"[Test Result] 3-Universe GravNet Multiverse:       "
        f"Acc = {np.mean(multi_test_preds == y_test)*100:.2f}% | "
        f"Macro-F1 = {fast_macro_f1(y_test, multi_test_preds):.4f} | "
        f"Train = {multiverse_train_sec:.2f}s"
    )

    joblib.dump(primary_model.to_checkpoint(), MODELS_DIR / "gravnet_single.joblib")
    joblib.dump(multiverse.to_checkpoint(), MODELS_DIR / "gravnet_multiverse.joblib")

    history = {
        "classes": classes,
        "seeds": args.seeds,
        "num_parameters_per_universe": primary_model.num_parameters,
        "num_plummer_attractors_per_universe": primary_model.num_attractors,
        "lens_basis_dim": primary_model.lens.out_dim,
        "physics_verification": physics_checks,
        "single_universe_train_sec": all_timings[str(args.seeds[0])]["total_train_sec"],
        "multiverse_train_sec": multiverse_train_sec,
        "per_seed_timings": all_timings,
        "inference_20k_ms": {
            "single_universe": single_inf,
            "multiverse_3": multi_inf,
        },
        "epoch_logs": all_logs,
        "orbital_step_progression": orbital_progression,
    }
    (MODELS_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nSaved checkpoints and telemetry to {MODELS_DIR}")


if __name__ == "__main__":
    main()
