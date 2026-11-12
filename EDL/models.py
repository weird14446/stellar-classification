# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "torch>=2.2", "scipy>=1.11"]
# ///
"""Topology & Weight Evolving Neural Networks (TWEANN / Evolutionary NAS) for EDL.

Provides:
- TopologyGenome: Encodes neural network architecture genes:
  * Number of hidden layers (depth: 1 to 4)
  * Layer widths (neurons per layer: 16 to 128)
  * Per-layer activation functions ('relu', 'gelu', 'silu', 'elu', 'tanh')
  * Direct input-to-hidden skip connections (DenseNet/ResNet-style shortcuts)
  * Active synaptic sparsity ratio
- EvolvableMLP: Dynamic PyTorch & NumPy neural network supporting:
  * Lamarckian / Net2Net structural mutations (adding/pruning neurons, adding/removing
    layers, toggling skip connections, mutating activations, synaptic pruning/sprouting)
    that preserve learned representations across topology changes.
  * Homologous Topology & Weight Crossover between networks of different shapes!
- HallOfFameEnsemble: Heterogeneous topology ensemble combining diverse evolved
  neural network architectures and weights.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ACTIVATIONS = ("relu", "gelu", "silu", "elu", "tanh")


def apply_activation_torch(x: torch.Tensor, act_name: str) -> torch.Tensor:
    if act_name == "relu":
        return F.relu(x)
    if act_name == "gelu":
        return F.gelu(x)
    if act_name == "silu":
        return F.silu(x)
    if act_name == "elu":
        return F.elu(x)
    if act_name == "tanh":
        return torch.tanh(x)
    return F.relu(x)


def apply_activation_numpy(x: np.ndarray, act_name: str) -> np.ndarray:
    if act_name == "relu":
        return np.maximum(x, 0.0)
    if act_name == "gelu":
        # Fast tanh approximation of GELU
        return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * (x ** 3))))
    if act_name == "silu":
        # Swish: x * sigmoid(x)
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
        return x * sig
    if act_name == "elu":
        return np.where(x > 0.0, x, np.expm1(np.clip(x, -20.0, 0.0)))
    if act_name == "tanh":
        return np.tanh(x)
    return np.maximum(x, 0.0)


@dataclass
class TopologyGenome:
    """Structural chromosome encoding the neural network topology."""

    in_dim: int = 10
    hidden_dims: Tuple[int, ...] = (64, 32)
    activations: Tuple[str, ...] = ("relu", "relu")
    input_skips: Tuple[bool, ...] = (False, False)
    n_classes: int = 3

    def __post_init__(self):
        self.hidden_dims = tuple(int(h) for h in self.hidden_dims)
        n_layers = len(self.hidden_dims)
        if len(self.activations) != n_layers:
            acts = list(self.activations)[:n_layers]
            while len(acts) < n_layers:
                acts.append("relu")
            self.activations = tuple(acts)
        else:
            self.activations = tuple(str(a) for a in self.activations)

        if len(self.input_skips) != n_layers:
            skips = list(self.input_skips)[:n_layers]
            while len(skips) < n_layers:
                skips.append(False)
            skips[0] = False  # Layer 0 already receives raw input directly
            self.input_skips = tuple(bool(s) for s in skips)
        else:
            skips = [bool(s) for s in self.input_skips]
            if skips:
                skips[0] = False
            self.input_skips = tuple(skips)

    @property
    def depth(self) -> int:
        return len(self.hidden_dims)

    def layer_in_dim(self, layer_idx: int) -> int:
        if layer_idx == 0:
            return self.in_dim
        base = self.hidden_dims[layer_idx - 1]
        return base + (self.in_dim if self.input_skips[layer_idx] else 0)

    def summary_str(self) -> str:
        parts = [f"{self.in_dim}"]
        for idx, (h, act, skip) in enumerate(zip(self.hidden_dims, self.activations, self.input_skips)):
            skip_tag = "+skip" if (idx > 0 and skip) else ""
            parts.append(f"[{h}{skip_tag}:{act}]")
        parts.append(f"{self.n_classes}")
        return " -> ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "in_dim": self.in_dim,
            "hidden_dims": list(self.hidden_dims),
            "activations": list(self.activations),
            "input_skips": list(self.input_skips),
            "n_classes": self.n_classes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TopologyGenome":
        return cls(
            in_dim=d.get("in_dim", 10),
            hidden_dims=tuple(d.get("hidden_dims", [64, 32])),
            activations=tuple(d.get("activations", ["relu"] * len(d.get("hidden_dims", [64, 32])))),
            input_skips=tuple(d.get("input_skips", [False] * len(d.get("hidden_dims", [64, 32])))),
            n_classes=d.get("n_classes", 3),
        )


class EvolvableMLP(nn.Module):
    """Topology- and Weight-Evolvable Neural Network (TWEANN).

    Supports dynamic hidden depths, layer widths, per-layer nonlinear activations,
    evolvable input-shortcut connections, synaptic pruning masks, and output boundary
    calibration genes (`class_scale` and `class_bias`).
    """

    def __init__(
        self,
        in_dim: int = 10,
        hidden_dims: Sequence[int] = (64, 32),
        n_classes: int = 3,
        activations: Optional[Sequence[str]] = None,
        input_skips: Optional[Sequence[bool]] = None,
        topology: Optional[TopologyGenome] = None,
    ):
        super().__init__()
        if topology is not None:
            self.topology = topology
        else:
            h_tuple = tuple(int(h) for h in hidden_dims)
            acts = tuple(activations) if activations is not None else tuple("relu" for _ in h_tuple)
            skips = tuple(input_skips) if input_skips is not None else tuple(False for _ in h_tuple)
            self.topology = TopologyGenome(
                in_dim=in_dim,
                hidden_dims=h_tuple,
                activations=acts,
                input_skips=skips,
                n_classes=n_classes,
            )

        self.in_dim = self.topology.in_dim
        self.hidden_dims = self.topology.hidden_dims
        self.n_classes = self.topology.n_classes

        # Build linear layers according to topology (including input skip dimensions)
        self.layers = nn.ModuleList()
        for idx, h_out in enumerate(self.topology.hidden_dims):
            h_in = self.topology.layer_in_dim(idx)
            self.layers.append(nn.Linear(h_in, h_out))
        out_in = self.topology.hidden_dims[-1] if self.topology.hidden_dims else self.in_dim
        self.out_layer = nn.Linear(out_in, self.n_classes)

        # Evolvable class-wise logit multiplier (scale) and shift (bias)
        self.class_scale = nn.Parameter(torch.ones(self.n_classes, dtype=torch.float32))
        self.class_bias = nn.Parameter(torch.zeros(self.n_classes, dtype=torch.float32))

        self._refresh_param_Metadata()

    def _refresh_param_Metadata(self) -> None:
        self.param_shapes: list[Tuple[str, Tuple[int, ...], int]] = []
        total = 0
        for name, param in self.named_parameters():
            num = param.numel()
            self.param_shapes.append((name, tuple(param.shape), num))
            total += num
        self._num_params = total

    @property
    def num_params(self) -> int:
        return self._num_params

    def active_synapse_ratio(self, threshold: float = 1e-4) -> float:
        """Fraction of synaptic weights with magnitude above threshold."""
        total_w = 0
        active_w = 0
        with torch.no_grad():
            for layer in [*self.layers, self.out_layer]:
                w = layer.weight.detach().abs()
                total_w += w.numel()
                active_w += int((w > threshold).sum().item())
        return active_w / max(1, total_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward pass through evolved topology."""
        curr = x
        for idx, layer in enumerate(self.layers):
            if idx > 0 and self.topology.input_skips[idx]:
                curr = torch.cat([curr, x], dim=1)
            curr = layer(curr)
            curr = apply_activation_torch(curr, self.topology.activations[idx])
        logits = self.out_layer(curr)
        return logits * self.class_scale + self.class_bias

    def get_flat_params(self) -> np.ndarray:
        params = [p.detach().cpu().numpy().flatten() for p in self.parameters()]
        return np.concatenate(params).astype(np.float32)

    def set_flat_params(self, flat_vector: np.ndarray) -> "EvolvableMLP":
        if len(flat_vector) != self._num_params:
            raise ValueError(
                f"Vector length mismatch: expected {self._num_params}, got {len(flat_vector)}"
            )
        offset = 0
        with torch.no_grad():
            for param in self.parameters():
                num = param.numel()
                slice_data = flat_vector[offset : offset + num].reshape(param.shape)
                param.copy_(torch.from_numpy(slice_data))
                offset += num
        return self

    def unpack_layers_numpy(
        self, flat_vector: Optional[np.ndarray] = None
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, TopologyGenome]:
        """Unpack flat vector into (layers, class_scale, class_bias, topology) for NumPy forward."""
        if flat_vector is None:
            flat_vector = self.get_flat_params()
        layers = []
        class_scale = np.ones(self.n_classes, dtype=np.float32)
        class_bias = np.zeros(self.n_classes, dtype=np.float32)
        offset = 0
        for name, shape, num in self.param_shapes:
            if name == "class_scale":
                class_scale = flat_vector[offset : offset + num].reshape(shape)
                offset += num
            elif name == "class_bias":
                class_bias = flat_vector[offset : offset + num].reshape(shape)
                offset += num
            elif "weight" in name:
                w = flat_vector[offset : offset + num].reshape(shape).T
                offset += num
            elif "bias" in name:
                b = flat_vector[offset : offset + num].reshape(shape)
                offset += num
                layers.append((w, b))
        return layers, class_scale, class_bias, self.topology

    @staticmethod
    def forward_numpy(
        X: np.ndarray,
        unpacked: Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, TopologyGenome],
    ) -> np.ndarray:
        """Ultra-fast NumPy forward pass supporting arbitrary evolved topologies."""
        layers, class_scale, class_bias, topo = unpacked
        curr = X
        n_hidden = len(topo.hidden_dims)
        for i in range(n_hidden):
            if i > 0 and topo.input_skips[i]:
                curr = np.concatenate([curr, X], axis=1)
            W, b = layers[i]
            curr = curr @ W + b
            curr = apply_activation_numpy(curr, topo.activations[i])
        W_out, b_out = layers[-1]
        logits = curr @ W_out + b_out
        return logits * class_scale + class_bias

    @staticmethod
    def softmax_numpy(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            tX = torch.as_tensor(X, dtype=torch.float32)
            logits = self.forward(tX)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def clone(self) -> "EvolvableMLP":
        new_model = EvolvableMLP(topology=TopologyGenome.from_dict(self.topology.to_dict()))
        new_model.set_flat_params(self.get_flat_params())
        return new_model

    # ==========================================================================
    # Structural / Topology Evolution Operators (Lamarckian Weight-Preserving)
    # ==========================================================================
    def mutate_width(self, layer_idx: int, delta_nodes: int, rng: np.random.Generator) -> "EvolvableMLP":
        """Grow or prune neurons in `layer_idx` while preserving existing learned representations."""
        old_dims = list(self.topology.hidden_dims)
        old_w = old_dims[layer_idx]
        new_w = int(np.clip(old_w + delta_nodes, 16, 128))
        if new_w == old_w:
            return self.clone()

        new_dims = list(old_dims)
        new_dims[layer_idx] = new_w
        new_topo = TopologyGenome(
            in_dim=self.in_dim,
            hidden_dims=tuple(new_dims),
            activations=self.topology.activations,
            input_skips=self.topology.input_skips,
            n_classes=self.n_classes,
        )
        child = EvolvableMLP(topology=new_topo)

        with torch.no_grad():
            child.class_scale.copy_(self.class_scale)
            child.class_bias.copy_(self.class_bias)

            # Copy unaffected layers directly
            for idx in range(len(old_dims)):
                if idx != layer_idx and idx != layer_idx + 1:
                    child.layers[idx].weight.copy_(self.layers[idx].weight)
                    child.layers[idx].bias.copy_(self.layers[idx].bias)
            if layer_idx < len(old_dims) - 1:
                child.out_layer.weight.copy_(self.out_layer.weight)
                child.out_layer.bias.copy_(self.out_layer.bias)

            curr_layer_old = self.layers[layer_idx]
            curr_layer_new = child.layers[layer_idx]
            next_layer_old = self.layers[layer_idx + 1] if layer_idx + 1 < len(old_dims) else self.out_layer
            next_layer_new = child.layers[layer_idx + 1] if layer_idx + 1 < len(old_dims) else child.out_layer

            if new_w > old_w:
                # Net2Wider: keep all old neurons, initialize new neurons with small noise
                curr_layer_new.weight[:old_w, :].copy_(curr_layer_old.weight)
                curr_layer_new.bias[:old_w].copy_(curr_layer_old.bias)
                extra = new_w - old_w
                donor_idx = rng.choice(old_w, size=extra, replace=True)
                noise_w = torch.from_numpy(
                    rng.normal(0, 0.02, size=(extra, curr_layer_old.weight.shape[1])).astype(np.float32)
                )
                curr_layer_new.weight[old_w:, :].copy_(curr_layer_old.weight[donor_idx] + noise_w)
                curr_layer_new.bias[old_w:].copy_(curr_layer_old.bias[donor_idx])

                # Next layer input has grown from old_w (+ skip) to new_w (+ skip)
                next_layer_new.bias.copy_(next_layer_old.bias)
                # Copy weights corresponding to old_w
                next_layer_new.weight[:, :old_w].copy_(next_layer_old.weight[:, :old_w])
                # Small outgoing weights for newly added neurons
                next_layer_new.weight[:, old_w:new_w].copy_(
                    torch.from_numpy(
                        rng.normal(0, 0.01, size=(next_layer_new.weight.shape[0], extra)).astype(np.float32)
                    )
                )
                # If next layer has input_skip, copy the trailing skip columns!
                if layer_idx + 1 < len(old_dims) and self.topology.input_skips[layer_idx + 1]:
                    next_layer_new.weight[:, new_w:].copy_(next_layer_old.weight[:, old_w:])
            else:
                # Importance-based Neuron Pruning: keep top `new_w` neurons by L2 weight norm
                in_norm = curr_layer_old.weight.norm(dim=1)
                out_norm = next_layer_old.weight[:, :old_w].norm(dim=0)
                importance = in_norm * out_norm
                keep_idx = torch.topk(importance, k=new_w, largest=True).indices.sort().values

                curr_layer_new.weight.copy_(curr_layer_old.weight[keep_idx, :])
                curr_layer_new.bias.copy_(curr_layer_old.bias[keep_idx])

                next_layer_new.bias.copy_(next_layer_old.bias)
                next_layer_new.weight[:, :new_w].copy_(next_layer_old.weight[:, keep_idx])
                if layer_idx + 1 < len(old_dims) and self.topology.input_skips[layer_idx + 1]:
                    next_layer_new.weight[:, new_w:].copy_(next_layer_old.weight[:, old_w:])

        return child

    def mutate_activation(self, layer_idx: int, new_act: str) -> "EvolvableMLP":
        """Mutate the nonlinear activation function of `layer_idx`."""
        acts = list(self.topology.activations)
        acts[layer_idx] = new_act
        new_topo = TopologyGenome(
            in_dim=self.in_dim,
            hidden_dims=self.topology.hidden_dims,
            activations=tuple(acts),
            input_skips=self.topology.input_skips,
            n_classes=self.n_classes,
        )
        child = EvolvableMLP(topology=new_topo)
        child.set_flat_params(self.get_flat_params())
        return child

    def mutate_skip_connection(self, layer_idx: int, rng: np.random.Generator) -> "EvolvableMLP":
        """Toggle direct input skip connection into hidden layer `layer_idx` (>= 1)."""
        if layer_idx <= 0 or layer_idx >= self.topology.depth:
            return self.clone()

        new_skips = list(self.topology.input_skips)
        enabling = not new_skips[layer_idx]
        new_skips[layer_idx] = enabling

        new_topo = TopologyGenome(
            in_dim=self.in_dim,
            hidden_dims=self.topology.hidden_dims,
            activations=self.topology.activations,
            input_skips=tuple(new_skips),
            n_classes=self.n_classes,
        )
        child = EvolvableMLP(topology=new_topo)

        with torch.no_grad():
            child.class_scale.copy_(self.class_scale)
            child.class_bias.copy_(self.class_bias)
            child.out_layer.weight.copy_(self.out_layer.weight)
            child.out_layer.bias.copy_(self.out_layer.bias)

            for idx in range(self.topology.depth):
                if idx != layer_idx:
                    child.layers[idx].weight.copy_(self.layers[idx].weight)
                    child.layers[idx].bias.copy_(self.layers[idx].bias)
                else:
                    prev_h = self.topology.hidden_dims[idx - 1]
                    child.layers[idx].bias.copy_(self.layers[idx].bias)
                    child.layers[idx].weight[:, :prev_h].copy_(self.layers[idx].weight[:, :prev_h])
                    if enabling:
                        # Initialize new input-shortcut weights near zero so function is preserved
                        skip_w = torch.from_numpy(
                            rng.normal(0, 0.015, size=(self.topology.hidden_dims[idx], self.in_dim)).astype(np.float32)
                        )
                        child.layers[idx].weight[:, prev_h:].copy_(skip_w)
        return child

    def mutate_add_layer(self, rng: np.random.Generator) -> "EvolvableMLP":
        """Net2Deeper: Insert a new hidden layer initialized near identity to preserve function."""
        if self.topology.depth >= 4:
            return self.clone()

        old_dims = list(self.topology.hidden_dims)
        old_acts = list(self.topology.activations)
        old_skips = list(self.topology.input_skips)

        # Insert after the last hidden layer with matching width
        new_w = old_dims[-1]
        new_dims = old_dims + [new_w]
        new_acts = old_acts + [str(rng.choice(["relu", "gelu", "silu", "elu"]))]
        new_skips = old_skips + [bool(rng.random() < 0.5)]

        new_topo = TopologyGenome(
            in_dim=self.in_dim,
            hidden_dims=tuple(new_dims),
            activations=tuple(new_acts),
            input_skips=tuple(new_skips),
            n_classes=self.n_classes,
        )
        child = EvolvableMLP(topology=new_topo)

        with torch.no_grad():
            child.class_scale.copy_(self.class_scale)
            child.class_bias.copy_(self.class_bias)
            for idx in range(len(old_dims)):
                child.layers[idx].weight.copy_(self.layers[idx].weight)
                child.layers[idx].bias.copy_(self.layers[idx].bias)

            # Initialize newly inserted layer as near-identity on the first `new_w` inputs
            inserted = child.layers[-1]
            inserted.weight.zero_()
            inserted.bias.zero_()
            eye = torch.eye(new_w, dtype=torch.float32)
            noise = torch.from_numpy(rng.normal(0, 0.02, size=inserted.weight.shape).astype(np.float32))
            inserted.weight[:, :new_w].copy_(eye)
            inserted.weight.add_(noise)

            child.out_layer.weight.copy_(self.out_layer.weight)
            child.out_layer.bias.copy_(self.out_layer.bias)

        return child

    def mutate_remove_layer(self) -> "EvolvableMLP":
        """Remove the last hidden layer when depth > 1 via linear projection."""
        if self.topology.depth <= 1:
            return self.clone()

        old_dims = list(self.topology.hidden_dims)
        new_dims = old_dims[:-1]
        new_acts = self.topology.activations[:-1]
        new_skips = self.topology.input_skips[:-1]

        new_topo = TopologyGenome(
            in_dim=self.in_dim,
            hidden_dims=tuple(new_dims),
            activations=new_acts,
            input_skips=new_skips,
            n_classes=self.n_classes,
        )
        child = EvolvableMLP(topology=new_topo)

        with torch.no_grad():
            child.class_scale.copy_(self.class_scale)
            child.class_bias.copy_(self.class_bias)
            for idx in range(len(new_dims)):
                child.layers[idx].weight.copy_(self.layers[idx].weight)
                child.layers[idx].bias.copy_(self.layers[idx].bias)

            # Approximate collapsed layer pair: W_merged = W_out @ W_last[:, :prev_h]
            prev_h = new_dims[-1]
            w_last = self.layers[-1].weight[:, :prev_h]  # (last_h, prev_h)
            b_last = self.layers[-1].bias                # (last_h,)
            w_out = self.out_layer.weight                # (n_classes, last_h)
            b_out = self.out_layer.bias                  # (n_classes,)

            child.out_layer.weight.copy_(w_out @ w_last)
            child.out_layer.bias.copy_(b_out + w_out @ b_last)

        return child

    def mutate_synapse_pruning(self, prune_fraction: float = 0.05) -> "EvolvableMLP":
        """Prune weakest synaptic weights to induce sparse connectivity."""
        child = self.clone()
        with torch.no_grad():
            for layer in child.layers:
                w = layer.weight.abs()
                cutoff = torch.quantile(w.flatten(), prune_fraction)
                mask = (w >= cutoff).float()
                layer.weight.mul_(mask)
        return child

    def crossover_with(
        self, other: "EvolvableMLP", alpha: float = 0.85, rng: Optional[np.random.Generator] = None
    ) -> "EvolvableMLP":
        """Homologous Topology & Weight Crossover between two potentially different architectures.

        Inherits structural topology from `self` (dominant parent) with occasional gene
        transfer from `other`, and blends all overlapping weight sub-tensors!
        """
        child = self.clone()
        with torch.no_grad():
            child.class_scale.copy_(alpha * self.class_scale + (1.0 - alpha) * other.class_scale)
            child.class_bias.copy_(alpha * self.class_bias + (1.0 - alpha) * other.class_bias)

            min_depth = min(len(self.layers), len(other.layers))
            for idx in range(min_depth):
                w1 = child.layers[idx].weight
                w2 = other.layers[idx].weight
                r = min(w1.shape[0], w2.shape[0])
                c = min(w1.shape[1], w2.shape[1])
                w1[:r, :c].copy_(alpha * w1[:r, :c] + (1.0 - alpha) * w2[:r, :c])

                b1 = child.layers[idx].bias
                b2 = other.layers[idx].bias
                b1[:r].copy_(alpha * b1[:r] + (1.0 - alpha) * b2[:r])

            ow1 = child.out_layer.weight
            ow2 = other.out_layer.weight
            oc = min(ow1.shape[1], ow2.shape[1])
            ow1[:, :oc].copy_(alpha * ow1[:, :oc] + (1.0 - alpha) * ow2[:, :oc])
            child.out_layer.bias.copy_(alpha * child.out_layer.bias + (1.0 - alpha) * other.out_layer.bias)

        return child


class HallOfFameEnsemble:
    """Heterogeneous Topology & Weight Ensemble of diverse evolved island champions.

    Each member in the ensemble can have its own distinct evolved neural network
    architecture (depth, layer widths, activations, skip connections) and weights!
    """

    def __init__(
        self,
        elite_models: Sequence[EvolvableMLP],
        class_multipliers: Optional[np.ndarray] = None,
        model_weights: Optional[Sequence[float]] = None,
    ):
        self.elite_models = list(elite_models)
        self.unpacked_elites = [m.unpack_layers_numpy() for m in self.elite_models]
        n_classes = self.elite_models[0].n_classes if self.elite_models else 3
        if class_multipliers is not None:
            self.class_multipliers = np.asarray(class_multipliers, dtype=np.float32)
        else:
            self.class_multipliers = np.ones(n_classes, dtype=np.float32)

        if model_weights is not None:
            w = np.asarray(model_weights, dtype=np.float64)
        elif self.elite_models and all(hasattr(m, "_ensemble_weight") for m in self.elite_models):
            w = np.asarray([ getattr(m, "_ensemble_weight") for m in self.elite_models ], dtype=np.float64)
        else:
            w = np.ones(max(1, len(self.elite_models)), dtype=np.float64)
        self.model_weights = w / max(1e-12, float(np.sum(w)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_f32 = np.asarray(X, dtype=np.float32)
        n_classes = len(self.class_multipliers)
        prob_sum = np.zeros((len(X_f32), n_classes), dtype=np.float64)
        for w, m in zip(self.model_weights, self.elite_models):
            if w <= 1e-8:
                continue
            prob_sum += w * m.predict_proba(X_f32)
        probs = prob_sum.astype(np.float32)
        scaled = probs * self.class_multipliers
        return scaled / np.sum(scaled, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    base = EvolvableMLP(
        in_dim=10,
        hidden_dims=(64, 32),
        activations=("gelu", "silu"),
        input_skips=(False, True),
    )
    print(f"Base topology: {base.topology.summary_str()} | params: {base.num_params}")
    dummy_x = np.random.randn(6, 10).astype(np.float32)
    t_out = base(torch.from_numpy(dummy_x)).detach().numpy()
    n_out = EvolvableMLP.forward_numpy(dummy_x, base.unpack_layers_numpy())
    diff = np.max(np.abs(t_out - n_out))
    print(f"PyTorch vs NumPy max diff (with GELU, SiLU, input_skip): {diff:.2e}")
    assert diff < 1e-4

    # Test structural mutations
    m_wide = base.mutate_width(0, +16, rng)
    print(f"After mutate_width(+16): {m_wide.topology.summary_str()} | params: {m_wide.num_params}")
    m_deep = m_wide.mutate_add_layer(rng)
    print(f"After mutate_add_layer:  {m_deep.topology.summary_str()} | params: {m_deep.num_params}")
    m_cross = m_deep.crossover_with(base, alpha=0.8, rng=rng)
    print(f"After homologous crossover: {m_cross.topology.summary_str()} | params: {m_cross.num_params}")
