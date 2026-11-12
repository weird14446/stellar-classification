# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scikit-learn>=1.4", "torch>=2.2"]
# ///
"""Evolutionary Algorithms for Topology & Weight Neural Network Optimization (TWEANN).

Implements three neuroevolution paradigms:
1. Topology & Weight Memetic Evolution (`MemeticEvolution`, `--algo memetic`):
   Simultaneously evolves the neural network's structural topology (number of hidden
   layers, neurons per layer, layer-wise activation functions, direct input-to-hidden
   skip connections, and synaptic sparsity) AND its weights & boundary genes via:
   - Net2Net / Lamarckian function-preserving structural mutations (Add/Prune Neurons,
     Add/Remove Layer, Toggle Skip Connection, Mutate Activation, Prune Synapses)
   - Brief Lamarckian local adaptation for newly mutated topologies (protecting
     structural innovations, analogous to NEAT speciation)
   - Homologous sub-tensor crossover across networks of different architectures
   - Direct Evolution Strategy (ES) calibration of the 6 output boundary genes on Macro-F1
2. NaturalEvolutionStrategy (NES / OpenAI-ES with Orthogonal Sampling & Smooth-F1)
3. GeneticAlgorithm (GA)
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn

try:
    from EDL.models import ACTIVATIONS, EvolvableMLP, TopologyGenome, apply_activation_numpy
except ModuleNotFoundError:
    from models import ACTIVATIONS, EvolvableMLP, TopologyGenome, apply_activation_numpy


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Fast vectorized calculation of discrete macro-F1 score."""
    indices = y_true * n_classes + y_pred
    cm = np.bincount(indices, minlength=n_classes * n_classes).reshape(n_classes, n_classes)

    f1_sum = 0.0
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1_sum += (2.0 * tp) / denom
    return f1_sum / n_classes


def composite_macro_f1_fitness(
    y_true: np.ndarray,
    logits: np.ndarray,
    n_classes: int = 3,
) -> Tuple[float, float]:
    """Composite fitness combining discrete Macro-F1, soft Macro-F1, and margin calibration."""
    preds = np.argmax(logits, axis=1)
    hard_f1 = fast_macro_f1(y_true, preds, n_classes=n_classes)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    soft_f1_sum = 0.0
    for c in range(n_classes):
        y_c = (y_true == c).astype(np.float32)
        p_c = probs[:, c]
        tp = np.sum(y_c * p_c)
        fp = np.sum((1.0 - y_c) * p_c)
        fn = np.sum(y_c * (1.0 - p_c))
        soft_f1_sum += (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    soft_f1 = soft_f1_sum / n_classes

    n = len(y_true)
    true_probs = np.clip(probs[np.arange(n), y_true], 1e-7, 1.0)
    ce_loss = float(-np.mean(np.log(true_probs)))

    composite = 0.65 * hard_f1 + 0.30 * soft_f1 - 0.05 * min(ce_loss, 2.5)
    return float(composite), float(hard_f1)


class HallOfFame:
    """Maintains top-K elite models (supporting heterogeneous topologies)."""

    def __init__(self, max_size: int = 8):
        self.max_size = max_size
        self.entries: List[Tuple[float, EvolvableMLP]] = []

    def update(self, fitness: float, model: EvolvableMLP) -> None:
        cloned = model.clone()
        if len(self.entries) < self.max_size:
            self.entries.append((fitness, cloned))
            self.entries.sort(key=lambda x: x[0], reverse=True)
        elif fitness > self.entries[-1][0]:
            self.entries[-1] = (fitness, cloned)
            self.entries.sort(key=lambda x: x[0], reverse=True)

    def get_elite_models(self) -> List[EvolvableMLP]:
        return [m for _, m in self.entries]


def evolve_ensemble_multipliers(
    raw_probs: np.ndarray,
    y_val: np.ndarray,
    n_generations: int = 30,
    pop_size: int = 40,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Use regularized Evolution Strategy (ES) to evolve class multipliers for Macro-F1."""
    rng = np.random.default_rng(seed)
    n_classes = raw_probs.shape[1]
    best_mult = np.ones(n_classes, dtype=np.float32)
    base_f1 = fast_macro_f1(y_val, np.argmax(raw_probs, axis=1), n_classes=n_classes)
    best_obj = base_f1
    best_f1 = base_f1

    sigma = 0.04
    for _ in range(n_generations):
        noise = rng.normal(0, sigma, size=(pop_size, n_classes)).astype(np.float32)
        for p in noise:
            cand = np.clip(best_mult + p, 0.75, 1.35)
            cand = cand / np.mean(cand)
            preds = np.argmax(raw_probs * cand, axis=1)
            f1 = fast_macro_f1(y_val, preds, n_classes=n_classes)
            reg_obj = f1 - 0.005 * float(np.sum((cand - 1.0) ** 2))
            if reg_obj > best_obj + 1e-5:
                best_obj = reg_obj
                best_f1 = f1
                best_mult = cand.copy()
        sigma *= 0.94
    return best_mult, best_f1


# ==============================================================================
# 1. Genetic Algorithm (GA)
# ==============================================================================
class GeneticAlgorithm:
    """Genetic Algorithm for real-valued neural network weight optimization."""

    def __init__(
        self,
        param_dim: int,
        pop_size: int = 60,
        elite_count: int = 4,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.08,
        mutation_decay: float = 0.995,
        tournament_k: int = 3,
        init_params: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.param_dim = param_dim
        self.pop_size = pop_size
        self.elite_count = elite_count
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.mutation_decay = mutation_decay
        self.tournament_k = tournament_k
        self.rng = np.random.default_rng(seed)

        scale = np.sqrt(2.0 / 48)
        self.population = self.rng.normal(0, scale, size=(pop_size, param_dim)).astype(np.float32)
        self.population[:, -6:-3] = 1.0
        self.population[:, -3:] = 0.0
        if init_params is not None:
            self.population[0] = init_params.copy()

        self.fitnesses = np.zeros(pop_size, dtype=np.float32)
        self.best_individual = self.population[0].copy()
        self.best_fitness = -np.inf

    def _select_parent(self) -> np.ndarray:
        candidates = self.rng.choice(self.pop_size, size=self.tournament_k, replace=False)
        best_cand = candidates[np.argmax(self.fitnesses[candidates])]
        return self.population[best_cand]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        alpha = 0.2
        d = np.abs(parent1 - parent2)
        low = np.minimum(parent1, parent2) - alpha * d
        high = np.maximum(parent1, parent2) + alpha * d
        child1 = self.rng.uniform(low, high).astype(np.float32)
        child2 = self.rng.uniform(low, high).astype(np.float32)
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mask = self.rng.random(size=self.param_dim) < self.mutation_rate
        noise = self.rng.normal(0, self.mutation_std, size=self.param_dim).astype(np.float32)
        individual[mask] += noise[mask]
        return individual

    def step(self) -> None:
        elite_indices = np.argsort(self.fitnesses)[::-1][: self.elite_count]
        new_population = [self.population[idx].copy() for idx in elite_indices]

        while len(new_population) < self.pop_size:
            p1 = self._select_parent()
            p2 = self._select_parent()
            c1, c2 = self._crossover(p1, p2)
            new_population.append(self._mutate(c1))
            if len(new_population) < self.pop_size:
                new_population.append(self._mutate(c2))

        self.population = np.array(new_population, dtype=np.float32)
        self.mutation_std = max(0.004, self.mutation_std * self.mutation_decay)


# ==============================================================================
# 2. Natural Evolution Strategies (NES / OpenAI-ES with Orthogonal Sampling)
# ==============================================================================
class NaturalEvolutionStrategy:
    """Enhanced Natural Evolution Strategy with Orthogonal Antithetic Sampling & Weight Decay."""

    def __init__(
        self,
        param_dim: int,
        pop_size: int = 80,
        sigma: float = 0.05,
        lr: float = 0.025,
        decay: float = 0.996,
        weight_decay: float = 1e-4,
        init_params: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        if pop_size % 2 != 0:
            pop_size += 1

        self.param_dim = param_dim
        self.pop_size = pop_size
        self.half_pop = pop_size // 2
        self.sigma = sigma
        self.lr = lr
        self.decay = decay
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(seed)

        if init_params is not None:
            self.mu = init_params.astype(np.float32).copy()
        else:
            scale = np.sqrt(2.0 / 48)
            self.mu = self.rng.normal(0, scale, size=param_dim).astype(np.float32)
            self.mu[-6:-3] = 1.0
            self.mu[-3:] = 0.0

        self.m = np.zeros(param_dim, dtype=np.float32)
        self.v = np.zeros(param_dim, dtype=np.float32)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

        self.best_individual = self.mu.copy()
        self.best_fitness = -np.inf

    def sample_population(self) -> Tuple[np.ndarray, np.ndarray]:
        raw_noise = self.rng.normal(0, 1.0, size=(self.half_pop, self.param_dim)).astype(np.float32)
        if self.half_pop <= self.param_dim:
            q, _ = np.linalg.qr(raw_noise.T, mode="reduced")
            chi_norm = np.sqrt(self.param_dim)
            self.noise = (q.T * chi_norm).astype(np.float32)
        else:
            self.noise = raw_noise

        pos = self.mu + self.sigma * self.noise
        neg = self.mu - self.sigma * self.noise
        population = np.vstack([pos, neg])
        return population, self.noise

    def update(self, fitnesses: np.ndarray) -> None:
        self.t += 1

        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]

        ranks = np.zeros(self.pop_size, dtype=np.float32)
        ranks[np.argsort(fitnesses)] = np.linspace(-0.5, 0.5, self.pop_size)

        r_pos = ranks[: self.half_pop]
        r_neg = ranks[self.half_pop :]

        grad = np.dot((r_pos - r_neg), self.noise) / (self.half_pop * self.sigma)

        if self.weight_decay > 0:
            grad[:-6] -= self.weight_decay * self.mu[:-6]

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        step = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.mu += step
        self.sigma = max(0.004, self.sigma * self.decay)


# ==============================================================================
# 3. Topology & Weight Evolving Island Memetic Neuroevolution (TWEANN)
# ==============================================================================
class MemeticEvolution:
    """Topology & Weight Evolving Island-Model Neuroevolution Optimizer (TWEANN).

    Evolves BOTH the neural network structure (depth, layer widths, activation functions,
    direct input-to-hidden skip connections, and synaptic sparsity) AND its weights:
    1. Diverse Initial Seed Topologies across P islands (from compact 2-layer to 3-layer
       networks with varying activations and shortcuts).
    2. Lamarckian Local Refinement of active topologies.
    3. Structural Mutation & Speciation Protection:
       Generates topology-mutated offspring via Net2Wider/Neuron-Pruning (`mutate_width`),
       Net2Deeper/Layer-Compression (`mutate_add_layer`/`mutate_remove_layer`), Activation
       Mutation (`mutate_activation`), Input-Skip Mutation (`mutate_skip_connection`),
       Synapse Pruning (`mutate_synapse_pruning`), and Homologous Inter-Island Crossover
       (`crossover_with`). Newly mutated topologies receive a brief Lamarckian adaptation
       pass before competing against the parent on Macro-F1.
    4. Direct Evolution Strategy (ES) optimization of the 6 output boundary genes.
    """

    def __init__(
        self,
        in_dim: int = 10,
        n_classes: int = 3,
        pop_size: int = 8,
        mutation_std: float = 0.008,
        seed: int = 42,
    ):
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.pop_size = pop_size
        self.mutation_std = mutation_std
        self.rng = np.random.default_rng(seed)

        # Seed population with diverse initial topologies (depths, widths, activations, skips)
        seed_topologies = [
            TopologyGenome(in_dim, (48, 24), ("relu", "relu"), (False, False), n_classes),
            TopologyGenome(in_dim, (64, 32), ("relu", "relu"), (False, False), n_classes),
            TopologyGenome(in_dim, (64, 32), ("gelu", "silu"), (False, True), n_classes),
            TopologyGenome(in_dim, (64, 32), ("silu", "relu"), (False, True), n_classes),
            TopologyGenome(in_dim, (80, 40), ("gelu", "gelu"), (False, False), n_classes),
            TopologyGenome(in_dim, (64, 48, 24), ("relu", "gelu", "relu"), (False, True, False), n_classes),
            TopologyGenome(in_dim, (72, 36), ("elu", "relu"), (False, True), n_classes),
            TopologyGenome(in_dim, (56, 28), ("silu", "silu"), (False, False), n_classes),
        ]

        self.models: List[EvolvableMLP] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        self.island_best_models: List[EvolvableMLP] = []
        self.island_best_f1: np.ndarray = np.zeros(pop_size, dtype=np.float32)
        self.topology_events: List[Dict[str, str]] = []

        self.lrs = np.linspace(9e-4, 1.4e-3, pop_size)
        self.wds = np.geomspace(1e-6, 5e-5, pop_size)

        for i in range(pop_size):
            torch.manual_seed(seed + i * 101)
            topo = seed_topologies[i % len(seed_topologies)]
            m = EvolvableMLP(topology=TopologyGenome.from_dict(topo.to_dict()))
            opt = self._make_optimizer(m, i)
            self.models.append(m)
            self.optimizers.append(opt)
            self.island_best_models.append(m.clone())

        self.fitnesses = np.zeros(pop_size, dtype=np.float32)

    def _make_optimizer(self, model: EvolvableMLP, island_idx: int) -> torch.optim.Optimizer:
        model.class_scale.requires_grad_(False)
        model.class_bias.requires_grad_(False)
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(self.lrs[island_idx]),
            weight_decay=float(self.wds[island_idx]),
        )

    def lamarckian_epoch_step(
        self,
        X_train_t: torch.Tensor,
        y_train_t: torch.Tensor,
        batch_size: int = 512,
        gen: int = 1,
    ) -> None:
        """Run one Lamarckian local refinement epoch across all islands."""
        n_samples = len(X_train_t)
        criterion = nn.CrossEntropyLoss()

        for i in range(self.pop_size):
            model = self.models[i]
            opt = self.optimizers[i]
            model.train()

            perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(gen * 97 + i))
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                xb = X_train_t[idx]
                yb = y_train_t[idx]
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()

    def evolve_boundary_genes_es(
        self,
        model: EvolvableMLP,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        n_directions: int = 24,
    ) -> float:
        """Regularized Evolution Strategy search on 6 boundary genes (scale & bias) for Macro-F1."""
        unpacked_layers, base_scale, base_bias, topo = model.unpack_layers_numpy()
        curr = X_eval
        n_hidden = len(topo.hidden_dims)
        for idx in range(n_hidden):
            if idx > 0 and topo.input_skips[idx]:
                curr = np.concatenate([curr, X_eval], axis=1)
            W, b = unpacked_layers[idx]
            curr = curr @ W + b
            curr = apply_activation_numpy(curr, topo.activations[idx])
        W_out, b_out = unpacked_layers[-1]
        raw_logits = curr @ W_out + b_out

        best_scale = np.ones(3, dtype=np.float32)
        best_bias = np.zeros(3, dtype=np.float32)
        neutral_f1 = fast_macro_f1(y_eval, np.argmax(raw_logits, axis=1), n_classes=3)
        best_f1 = neutral_f1
        best_obj = neutral_f1

        curr_f1 = fast_macro_f1(
            y_eval, np.argmax(raw_logits * base_scale + base_bias, axis=1), n_classes=3
        )
        curr_obj = curr_f1 - 0.005 * float(np.sum((base_scale - 1.0) ** 2) + np.sum(base_bias ** 2))
        if curr_obj > best_obj:
            best_obj = curr_obj
            best_f1 = curr_f1
            best_scale = base_scale.copy()
            best_bias = base_bias.copy()

        scale_noise = self.rng.normal(0, 0.03, size=(n_directions, 3)).astype(np.float32)
        bias_noise = self.rng.normal(0, 0.02, size=(n_directions, 3)).astype(np.float32)

        for ds, db in zip(scale_noise, bias_noise):
            cand_s = np.clip(best_scale + ds, 0.8, 1.25)
            cand_s = cand_s / np.mean(cand_s)
            cand_b = np.clip(best_bias + db, -0.2, 0.2)
            cand_b = cand_b - np.mean(cand_b)
            preds = np.argmax(raw_logits * cand_s + cand_b, axis=1)
            f1 = fast_macro_f1(y_eval, preds, n_classes=3)
            reg_obj = f1 - 0.005 * float(np.sum((cand_s - 1.0) ** 2) + np.sum(cand_b ** 2))
            if reg_obj > best_obj + 1e-5:
                best_obj = reg_obj
                best_f1 = f1
                best_scale = cand_s.copy()
                best_bias = cand_b.copy()

        with torch.no_grad():
            model.class_scale.copy_(torch.from_numpy(best_scale))
            model.class_bias.copy_(torch.from_numpy(best_bias))
        return best_f1

    def _propose_structural_offspring(self, parent: EvolvableMLP, island_idx: int) -> Tuple[EvolvableMLP, str]:
        """Apply a random Net2Net / TWEANN structural mutation to `parent`."""
        op = self.rng.choice(
            ["widen", "prune_nodes", "activation", "skip", "add_layer", "remove_layer", "prune_synapses"],
            p=[0.28, 0.14, 0.20, 0.16, 0.08, 0.06, 0.08],
        )
        depth = parent.topology.depth
        layer_idx = int(self.rng.integers(0, depth))

        if op == "widen":
            delta = int(self.rng.choice([8, 16]))
            return parent.mutate_width(layer_idx, +delta, self.rng), f"widen(L{layer_idx}+{delta})"
        if op == "prune_nodes":
            delta = int(self.rng.choice([-8, -16]))
            return parent.mutate_width(layer_idx, delta, self.rng), f"prune_nodes(L{layer_idx}{delta})"
        if op == "activation":
            candidates = [a for a in ("relu", "gelu", "silu", "elu") if a != parent.topology.activations[layer_idx]]
            new_act = str(self.rng.choice(candidates))
            return parent.mutate_activation(layer_idx, new_act), f"act(L{layer_idx}->{new_act})"
        if op == "skip" and depth >= 2:
            skip_layer = int(self.rng.integers(1, depth))
            return parent.mutate_skip_connection(skip_layer, self.rng), f"toggle_skip(L{skip_layer})"
        if op == "add_layer" and depth < 4:
            return parent.mutate_add_layer(self.rng), "add_layer"
        if op == "remove_layer" and depth > 2:
            return parent.mutate_remove_layer(), "remove_layer"
        return parent.mutate_synapse_pruning(prune_fraction=0.04), "prune_synapses(4%)"

    def darwinian_topology_and_weight_step(
        self,
        X_train_t: torch.Tensor,
        y_train_t: torch.Tensor,
        X_train_sub: np.ndarray,
        y_train_sub: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        gen: int,
    ) -> List[str]:
        """Evolve both neural network structure (topology) and weights across islands."""
        events: List[str] = []
        n_samples = len(X_train_t)
        criterion = nn.CrossEntropyLoss()

        for i in range(self.pop_size):
            parent = self.models[i]
            val_f1 = self.fitnesses[i]

            if val_f1 > self.island_best_f1[i]:
                self.island_best_f1[i] = val_f1
                self.island_best_models[i] = parent.clone()

            # 1. Structural Mutation Trial (every 3 generations per island)
            if gen % 3 == (i % 3):
                old_summary = parent.topology.summary_str()
                struct_child, op_desc = self._propose_structural_offspring(parent, i)

                # Brief Speciation Protection / Lamarckian Micro-Adaptation (20 mini-batches)
                struct_opt = self._make_optimizer(struct_child, i)
                struct_child.train()
                for _ in range(20):
                    idx = self.rng.integers(0, n_samples, size=512)
                    struct_opt.zero_grad()
                    loss = criterion(struct_child(X_train_t[idx]), y_train_t[idx])
                    loss.backward()
                    struct_opt.step()

                # Evaluate structural offspring on train_sub & validation Macro-F1
                self.evolve_boundary_genes_es(struct_child, X_train_sub, y_train_sub, n_directions=16)
                v_logits = EvolvableMLP.forward_numpy(X_val, struct_child.unpack_layers_numpy())
                v_f1 = fast_macro_f1(y_val, np.argmax(v_logits, axis=1), n_classes=self.n_classes)

                # Parsimony-aware survival check: accept new topology if it improves Macro-F1!
                if v_f1 > val_f1 + 1e-5:
                    new_summary = struct_child.topology.summary_str()
                    events.append(
                        f"Island {i} [{op_desc}]: {old_summary} -> {new_summary} (Val F1 {val_f1:.4f} -> {v_f1:.4f})"
                    )
                    self.models[i] = struct_child
                    self.optimizers[i] = struct_opt
                    self.fitnesses[i] = v_f1
                    val_f1 = v_f1
                    parent = struct_child
                    if v_f1 > self.island_best_f1[i]:
                        self.island_best_f1[i] = v_f1
                        self.island_best_models[i] = struct_child.clone()
                    continue

            # 2. Homologous Inter-Island Crossover + Weight Mutation Trial
            tr_f1_parent = fast_macro_f1(
                y_train_sub,
                np.argmax(EvolvableMLP.forward_numpy(X_train_sub, parent.unpack_layers_numpy()), axis=1),
                n_classes=self.n_classes,
            )

            # Candidate A: Gaussian weight perturbation
            cand1 = parent.clone()
            vec1 = cand1.get_flat_params()
            mask = self.rng.random(size=len(vec1) - 6) < 0.08
            noise = self.rng.normal(0, self.mutation_std, size=len(vec1) - 6).astype(np.float32)
            vec1[:-6][mask] += noise[mask]
            cand1.set_flat_params(vec1)

            # Candidate B: Homologous crossover with ring neighbor (works even if topologies differ!)
            neighbor = self.island_best_models[(i + 1) % self.pop_size]
            cand2 = parent.crossover_with(neighbor, alpha=0.92, rng=self.rng)

            for cand in (cand1, cand2):
                unpacked = cand.unpack_layers_numpy()
                tr_logits = EvolvableMLP.forward_numpy(X_train_sub, unpacked)
                tr_f1 = fast_macro_f1(y_train_sub, np.argmax(tr_logits, axis=1), n_classes=self.n_classes)
                if tr_f1 > tr_f1_parent + 1e-5:
                    v_logits = EvolvableMLP.forward_numpy(X_val, unpacked)
                    v_f1 = fast_macro_f1(y_val, np.argmax(v_logits, axis=1), n_classes=self.n_classes)
                    if v_f1 >= val_f1:
                        parent.set_flat_params(cand.get_flat_params())
                        tr_f1_parent = tr_f1
                        val_f1 = v_f1
                        self.fitnesses[i] = val_f1
                        if val_f1 > self.island_best_f1[i]:
                            self.island_best_f1[i] = val_f1
                            self.island_best_models[i] = parent.clone()

        self.mutation_std = max(0.001, self.mutation_std * 0.95)
        return events
