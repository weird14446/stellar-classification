# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pandas>=2.0", "scikit-learn>=1.4", "torch>=2.2", "joblib>=1.3"]
# ///
"""Training script for Topology & Weight Evolving Neural Networks (EDL) on SDSS data.

Supports three neuroevolution paradigms:
1. memetic (default): Topology & Weight Evolving Island Memetic Neuroevolution (TWEANN):
   - Simultaneously evolves neural network topology (number of hidden layers, neurons per
     layer, activation functions, direct input-to-hidden skip connections, and synaptic
     sparsity) AND network weights across P islands.
   - Uses Net2Net / Lamarckian structural mutations with speciation protection.
   - Direct Evolution Strategy (ES) optimization of 6 boundary genes on discrete Macro-F1.
   - Heterogeneous Topology Hall-of-Fame Ensemble combining diverse evolved architectures.
2. es: Enhanced Natural Evolution Strategy (Orthogonal Antithetic Sampling + Composite
   Smooth/Hard Macro-F1 Fitness + Decoupled Weight Decay).
3. ga: Genetic Algorithm with BLX-alpha crossover, adaptive Gaussian mutation, and elitism.

Saves the best evolved model, heterogeneous Hall-of-Fame elites, preprocessor, and telemetry.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for p in [str(ROOT), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
import torch

try:
    from EDL.evolution import (
        GeneticAlgorithm,
        HallOfFame,
        MemeticEvolution,
        NaturalEvolutionStrategy,
        composite_macro_f1_fitness,
        evolve_ensemble_multipliers,
        fast_macro_f1,
    )
    from EDL.models import EvolvableMLP, HallOfFameEnsemble, TopologyGenome
    from EDL.preprocess import make_splits
except ModuleNotFoundError:
    from evolution import (
        GeneticAlgorithm,
        HallOfFame,
        MemeticEvolution,
        NaturalEvolutionStrategy,
        composite_macro_f1_fitness,
        evolve_ensemble_multipliers,
        fast_macro_f1,
    )
    from models import EvolvableMLP, HallOfFameEnsemble, TopologyGenome
    from preprocess import make_splits

DEFAULT_OUTDIR = ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Topology & Weight Evolving Neural Network (EDL)")
    parser.add_argument(
        "--algo",
        choices=["memetic", "es", "ga"],
        default="memetic",
        help="Evolutionary algorithm: memetic (TWEANN topology+weight island hybrid), es (Orthogonal NES), ga (GA)",
    )
    parser.add_argument("--pop-size", type=int, default=None, help="Population / Island count")
    parser.add_argument("--generations", type=int, default=None, help="Number of generations")
    parser.add_argument("--subsample", type=int, default=8000, help="Train batch size per generation (for ES/GA)")
    parser.add_argument("--eval-interval", type=int, default=1, help="Validation eval interval")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--hidden", type=str, default="64,32", help="Initial seed hidden layer sizes (for ES/GA)")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate for ES (default: 0.03)")
    parser.add_argument("--sigma", type=float, default=0.05, help="Initial perturbation sigma (default: 0.05)")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate for GA (default: 0.1)")
    parser.add_argument("--hof-size", type=int, default=8, help="Hall-of-Fame ensemble size (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR), help="Output directory")
    return parser.parse_args()


def train():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.pop_size is None:
        args.pop_size = 8 if args.algo == "memetic" else 80
    if args.generations is None:
        args.generations = 45 if args.algo == "memetic" else 250

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 76)
    print("  EVOLUTIONARY DEEP LEARNING (EDL) - TOPOLOGY & WEIGHT EVOLUTION (TWEANN)")
    print(f"  Algorithm: {args.algo.upper()} | Population/Islands: {args.pop_size} | Generations: {args.generations}")
    print("=" * 76)

    # 1. Load preprocessed splits
    print("[1/4] Preparing dataset splits and preprocessor...")
    splits, classes, preprocessor = make_splits(color_indices=True, return_preprocessor=True)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    n_train = len(X_train)
    n_classes = len(classes)
    in_dim = X_train.shape[1]

    joblib.dump(preprocessor, outdir / "preprocessor.joblib")
    print(f"      Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"      Features: {in_dim} | Classes: {classes}")

    history = []
    topology_mutations_log = []
    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_model: EvolvableMLP | None = None
    best_gen = 0
    stall = 0

    start_time = time.perf_counter()

    # 2 & 3. Run Evolutionary Optimization
    if args.algo == "memetic":
        optimizer = MemeticEvolution(
            in_dim=in_dim,
            n_classes=n_classes,
            pop_size=args.pop_size,
            mutation_std=0.008,
            seed=args.seed,
        )
        print("[2/4] Initialized Diverse Island Seed Topologies:")
        for idx, m in enumerate(optimizer.models):
            print(f"      Island {idx}: {m.topology.summary_str()} ({m.num_params:,} params)")

        print("\n[3/4] Launching Topology & Weight Evolution (Net2Net Mutations + Boundary ES)...")
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train).long()

        for gen in range(1, args.generations + 1):
            gen_start = time.perf_counter()

            # Step A: Lamarckian local refinement across all islands
            optimizer.lamarckian_epoch_step(X_train_t, y_train_t, batch_size=512, gen=gen)

            # Step B: Derivative-free ES evolution of 6 boundary genes (scale & bias)
            sub_idx = rng.choice(n_train, size=min(16000, n_train), replace=False)
            X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]

            train_f1s = np.zeros(optimizer.pop_size, dtype=np.float32)
            val_f1s = np.zeros(optimizer.pop_size, dtype=np.float32)
            val_accs = np.zeros(optimizer.pop_size, dtype=np.float32)

            for i in range(optimizer.pop_size):
                tr_f1 = optimizer.evolve_boundary_genes_es(
                    optimizer.models[i], X_sub, y_sub, n_directions=24
                )
                train_f1s[i] = tr_f1

                unpacked = optimizer.models[i].unpack_layers_numpy()
                val_logits = EvolvableMLP.forward_numpy(X_val, unpacked)
                val_preds = np.argmax(val_logits, axis=1)
                vf1 = fast_macro_f1(y_val, val_preds, n_classes=n_classes)
                vacc = float((val_preds == y_val).mean())
                val_f1s[i] = vf1
                val_accs[i] = vacc

            optimizer.fitnesses = val_f1s.copy()

            # Step C: Darwinian Structural Mutation + Homologous Crossover + Weight Mutation
            events = optimizer.darwinian_topology_and_weight_step(
                X_train_t, y_train_t, X_sub, y_sub, X_val, y_val, gen=gen
            )
            for ev in events:
                print(f"    [Topology Evolved @ Gen {gen:2d}] {ev}")
                topology_mutations_log.append({"generation": gen, "event": ev})

            val_f1s = optimizer.fitnesses.copy()
            gen_best_idx = int(np.argmax(val_f1s))
            gen_val_f1 = float(val_f1s[gen_best_idx])
            champ_island_model = optimizer.models[gen_best_idx]
            gen_val_acc = float(
                (np.argmax(EvolvableMLP.forward_numpy(X_val, champ_island_model.unpack_layers_numpy()), axis=1) == y_val).mean()
            )
            gen_train_best = float(np.max(train_f1s))
            gen_train_mean = float(np.mean(train_f1s))
            mean_params = int(np.mean([m.num_params for m in optimizer.models]))

            if gen_val_f1 > best_val_f1 + 1e-5:
                best_val_f1 = gen_val_f1
                best_val_acc = gen_val_acc
                best_model = champ_island_model.clone()
                best_gen = gen
                stall = 0
                flag = f" * (Best: {champ_island_model.topology.summary_str()})"
            else:
                stall += 1
                flag = ""

            gen_time = (time.perf_counter() - gen_start) * 1000
            print(
                f"  Gen {gen:3d}/{args.generations:3d} | "
                f"Train F1: {gen_train_best:.4f} (mean: {gen_train_mean:.4f}) | "
                f"Val F1: {gen_val_f1:.4f} | Val Acc: {gen_val_acc:.4f} | {gen_time:6.1f}ms{flag}"
            )

            history.append({
                "generation": gen,
                "train_best_fitness": gen_train_best,
                "train_mean_fitness": gen_train_mean,
                "val_macro_f1": gen_val_f1,
                "val_accuracy": gen_val_acc,
                "best_topology": champ_island_model.topology.summary_str(),
                "best_num_params": champ_island_model.num_params,
                "mean_num_params": mean_params,
            })

            if stall >= args.patience:
                print(f"  --> Early stopping at generation {gen}")
                break

        elite_models = [m.clone() for m in optimizer.island_best_models]
        print("\nFinal Evolved Island Topologies:")
        for idx, m in enumerate(elite_models):
            print(
                f"  Island {idx} Champion: {m.topology.summary_str():<46} | "
                f"Params: {m.num_params:5,d} | Active Synapses: {m.active_synapse_ratio()*100:5.1f}% | "
                f"Val F1: {optimizer.island_best_f1[idx]:.4f}"
            )

    else:
        hidden_dims = [int(h.strip()) for h in args.hidden.split(",") if h.strip()]
        template_model = EvolvableMLP(in_dim=in_dim, hidden_dims=hidden_dims, n_classes=n_classes)
        param_dim = template_model.num_params
        hof = HallOfFame(max_size=args.hof_size)
        print(f"[2/4] Initialized EvolvableMLP: {template_model.topology.summary_str()} ({param_dim} params)")
        print(f"[3/4] Launching Pure Derivative-Free {args.algo.upper()} with Composite Smooth+Hard Macro-F1...")

        if args.algo == "ga":
            optimizer = GeneticAlgorithm(
                param_dim=param_dim,
                pop_size=args.pop_size,
                elite_count=max(3, args.pop_size // 15),
                crossover_rate=0.8,
                mutation_rate=args.mutation_rate,
                mutation_std=args.sigma,
                mutation_decay=0.994,
                seed=args.seed,
            )
        else:
            optimizer = NaturalEvolutionStrategy(
                param_dim=param_dim,
                pop_size=args.pop_size,
                sigma=args.sigma,
                lr=args.lr,
                decay=0.996,
                weight_decay=1e-4,
                seed=args.seed,
            )

        for gen in range(1, args.generations + 1):
            gen_start = time.perf_counter()
            batch_idx = rng.choice(n_train, size=min(args.subsample, n_train), replace=False)
            X_sub, y_sub = X_train[batch_idx], y_train[batch_idx]

            if args.algo == "ga":
                comp_fits = np.zeros(optimizer.pop_size, dtype=np.float32)
                hard_f1s = np.zeros(optimizer.pop_size, dtype=np.float32)
                for i in range(optimizer.pop_size):
                    unpacked = template_model.unpack_layers_numpy(optimizer.population[i])
                    logits = EvolvableMLP.forward_numpy(X_sub, unpacked)
                    c_fit, h_f1 = composite_macro_f1_fitness(y_sub, logits, n_classes=n_classes)
                    comp_fits[i] = c_fit
                    hard_f1s[i] = h_f1
                optimizer.fitnesses = comp_fits
                best_idx = int(np.argmax(hard_f1s))
                gen_best_fitness = float(hard_f1s[best_idx])
                gen_mean_fitness = float(np.mean(hard_f1s))
                cand_param = optimizer.population[best_idx].copy()
                optimizer.step()
            else:
                population, _ = optimizer.sample_population()
                comp_fits = np.zeros(optimizer.pop_size, dtype=np.float32)
                hard_f1s = np.zeros(optimizer.pop_size, dtype=np.float32)
                for i in range(optimizer.pop_size):
                    unpacked = template_model.unpack_layers_numpy(population[i])
                    logits = EvolvableMLP.forward_numpy(X_sub, unpacked)
                    c_fit, h_f1 = composite_macro_f1_fitness(y_sub, logits, n_classes=n_classes)
                    comp_fits[i] = c_fit
                    hard_f1s[i] = h_f1
                gen_best_fitness = float(np.max(hard_f1s))
                gen_mean_fitness = float(np.mean(hard_f1s))
                optimizer.update(comp_fits)
                cand_param = optimizer.mu.copy()

            is_eval_gen = (gen % args.eval_interval == 0) or (gen == 1) or (gen == args.generations)
            val_f1, val_acc = 0.0, 0.0
            if is_eval_gen:
                unpacked = template_model.unpack_layers_numpy(cand_param)
                val_logits = EvolvableMLP.forward_numpy(X_val, unpacked)
                val_preds = np.argmax(val_logits, axis=1)
                val_f1 = float(fast_macro_f1(y_val, val_preds, n_classes=n_classes))
                val_acc = float((val_preds == y_val).mean())
                cand_model = template_model.clone().set_flat_params(cand_param)
                hof.update(val_f1, cand_model)

                if val_f1 > best_val_f1 + 1e-4:
                    best_val_f1 = val_f1
                    best_val_acc = val_acc
                    best_model = cand_model.clone()
                    best_gen = gen
                    stall = 0
                    flag = " * (Best)"
                else:
                    stall += 1
                    flag = ""

                gen_time = (time.perf_counter() - gen_start) * 1000
                if gen % 5 == 0 or gen == 1 or flag:
                    print(
                        f"  Gen {gen:3d}/{args.generations:3d} | "
                        f"Train F1: {gen_best_fitness:.4f} (mean: {gen_mean_fitness:.4f}) | "
                        f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | {gen_time:5.1f}ms{flag}"
                    )

            history.append({
                "generation": gen,
                "train_best_fitness": gen_best_fitness,
                "train_mean_fitness": gen_mean_fitness,
                "val_macro_f1": val_f1 if is_eval_gen else None,
                "val_accuracy": val_acc if is_eval_gen else None,
                "best_topology": template_model.topology.summary_str(),
                "best_num_params": param_dim,
                "mean_num_params": param_dim,
            })

        elite_models = hof.get_elite_models()

    # Evaluate & Evolve Heterogeneous Hall-of-Fame Ensemble Multipliers
    if len(elite_models) > 1:
        if args.algo == "memetic":
            f1_arr = np.asarray(optimizer.island_best_f1, dtype=np.float64)
            top_k = min(4, len(elite_models))
            top_idx = set(np.argsort(f1_arr)[-top_k:].tolist())
            raw_w = np.array(
                [np.exp((f1_arr[i] - np.max(f1_arr)) * 800.0) if i in top_idx else 0.0 for i in range(len(elite_models))],
                dtype=np.float64,
            )
            raw_w /= np.sum(raw_w)
            for idx, m in enumerate(elite_models):
                m._ensemble_weight = float(raw_w[idx])
            best_mult = np.ones(n_classes, dtype=np.float32)
            hof_ens = HallOfFameEnsemble(elite_models, class_multipliers=best_mult, model_weights=raw_w)
            hof_val_preds = hof_ens.predict(X_val)
            hof_val_f1 = float(fast_macro_f1(y_val, hof_val_preds, n_classes=n_classes))
            hof_val_acc = float((hof_val_preds == y_val).mean())
        else:
            for m in elite_models:
                m._ensemble_weight = 1.0 / len(elite_models)
            base_ens = HallOfFameEnsemble(elite_models)
            raw_val_probs = base_ens.predict_proba(X_val)
            best_mult, hof_val_f1 = evolve_ensemble_multipliers(raw_val_probs, y_val, seed=args.seed)
            hof_ens = HallOfFameEnsemble(elite_models, class_multipliers=best_mult)
            hof_val_preds = hof_ens.predict(X_val)
            hof_val_acc = float((hof_val_preds == y_val).mean())
    else:
        best_mult = np.ones(n_classes, dtype=np.float32)
        hof_val_f1 = best_val_f1
        hof_val_acc = best_val_acc

    total_time = time.perf_counter() - start_time
    print(f"\n[4/4] Evolution finished in {total_time:.2f}s!")
    print(
        f"      Single Best Champion ({best_model.topology.summary_str()}) "
        f"Val Macro-F1: {best_val_f1:.4f} (Val Acc: {best_val_acc:.4f}) at Gen {best_gen}"
    )
    print(
        f"      Heterogeneous Topology Hall-of-Fame Ensemble ({len(elite_models)} islands) "
        f"Val Macro-F1: {hof_val_f1:.4f} (Val Acc: {hof_val_acc:.4f})"
    )

    # Save champion model & heterogeneous Hall-of-Fame ensemble
    torch.save(
        {
            "topology": best_model.topology.to_dict(),
            "state_dict": best_model.state_dict(),
        },
        outdir / "edl_model.pt",
    )
    np.save(outdir / "edl_weights.npy", best_model.get_flat_params())

    hof_bundle = [
        {
            "topology": m.topology.to_dict(),
            "state_dict": m.state_dict(),
            "num_params": m.num_params,
            "summary": m.topology.summary_str(),
            "weight": float(getattr(m, "_ensemble_weight", 1.0 / max(1, len(elite_models)))),
        }
        for m in elite_models
    ]
    torch.save(hof_bundle, outdir / "edl_hof_elites.pt")
    np.save(outdir / "edl_hof_multipliers.npy", best_mult)

    with open(outdir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    meta = {
        "algo": args.algo,
        "classes": classes,
        "in_dim": in_dim,
        "hidden_dims": list(best_model.topology.hidden_dims),
        "activations": list(best_model.topology.activations),
        "input_skips": list(best_model.topology.input_skips),
        "topology": best_model.topology.to_dict(),
        "topology_summary": best_model.topology.summary_str(),
        "param_dim": best_model.num_params,
        "pop_size": args.pop_size,
        "hof_size": len(elite_models),
        "hof_topologies": [m.topology.summary_str() for m in elite_models],
        "hof_multipliers": [round(float(x), 4) for x in best_mult],
        "topology_mutations": topology_mutations_log,
        "best_generation": best_gen,
        "best_val_macro_f1": round(best_val_f1, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "hof_val_macro_f1": round(float(hof_val_f1), 4),
        "hof_val_accuracy": round(float(hof_val_acc), 4),
        "total_time_seconds": round(total_time, 2),
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Artifacts saved to: {outdir}/")


if __name__ == "__main__":
    train()
