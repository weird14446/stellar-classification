# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.12", "scikit-learn>=1.4", "sympy>=1.12"]
# ///
"""GravNet: Astrophysical Potential Dynamical Network (100% Backpropagation-Free).

Models celestial object classification as an N-body gravitational orbital capture process
in a d-dimensional fibered phase space (q, p) in R^d x R^d (d = 2 + d_s):

1. Multi-Resolution Spectral-Astrophysical Lens Phi(x):
   Combines log-spaced 1D Doppler/spectral harmonic wavelets (resolving fine z ~ 0 stellar
   boundaries), quadratic SED curvature interactions, and multiscale 10D Bochner Random
   Fourier modes.
2. Cosmic-Origin Launch & Fibered Phase Space:
   At t = 0, every celestial particle launches strictly from the unclassified cosmic origin
   q_orb(0) = (0, 0) in the 2D Equilateral Supercluster Plane (90 deg, 210 deg, 330 deg),
   with initial spectral fiber coordinate q_spec(0) = Phi(x) W_{q,spec} and ballistic
   Doppler-spectral velocity kick p(0) = Phi(x) W_p.
3. Multi-Halo Plummer Gravitational Wells {mu_{c,m}, G_{c,m}}:
   K = 6 condensed sub-halos per class (18 massive Plummer cores) exert exact closed-form
   gravitational attraction F_grav(q) = -grad_q U(q) with softening epsilon > 0 and cosmic
   harmonic confinement kappa > 0.
4. Symplectic Dissipative Leapfrog (Strang Splitting):
   Integrates Hamiltonian dynamics with Hubble drag gamma > 0, strictly guaranteeing
   monotonic Lyapunov energy dissipation dH/dt = -gamma ||p||_2^2 <= 0.
5. Zero-BPTT Physical Learning:
   - Stage 1: K-Means + SVD proto-halo condensation & SPD Cholesky ballistic calibration
   - Stage 2: Contrastive Spacetime Mass/Halo Sculpting + Direct Terminal Tidal Hebbian Plasticity
   - Stage 3: Grand-Canonical Chemical Potential (nu_c) calibration across Lagrange L1 ridges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.cluster import KMeans


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Compute exact Macro-F1 score in vectorized NumPy."""
    f1_sum = 0.0
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1_sum += (2.0 * tp) / denom
    return float(f1_sum / n_classes)


class SpectralAstrophysicalLens:
    """Multi-Resolution Spectral-Astrophysical Optical Lens Phi(x).

    Resolves both broadband photometric SED curvature (u, g, r, i, z and colors)
    and fine relativistic/Doppler redshift scales near z ~ 0 without backpropagation:
      - Log-spaced 1D harmonic wavelets per observable (omega in [0.25, 120.0])
      - Pairwise quadratic SED curvature terms x_i * x_j
      - Anisotropic multiscale 10D Bochner Random Fourier features + redshift-coupled modes
    """

    def __init__(
        self,
        in_dim: int = 10,
        n_1d_freqs: int = 44,
        fourier_dim: int = 2304,
        bandwidths: Tuple[float, ...] = (0.25, 0.60, 1.40, 3.20),
        seed: int = 42,
    ):
        self.in_dim = int(in_dim)
        self.n_1d_freqs = int(n_1d_freqs)
        self.fourier_dim = int(fourier_dim)
        self.bandwidths = tuple(float(b) for b in bandwidths)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)

        # 1. Per-feature geometric frequency ladder (critical for z ~ 0 stellar vs extragalactic split)
        self.freqs_1d = np.geomspace(0.25, 120.0, self.n_1d_freqs).astype(np.float32)

        # 2. Anisotropic multiscale 10D Random Fourier Features (Bochner's theorem)
        feature_scales = np.ones((self.in_dim, 1), dtype=np.float32)
        if self.in_dim >= 10:
            feature_scales[5:9, 0] = 1.5   # color indices u-g, g-r, r-i, i-z
            feature_scales[9, 0] = 6.0     # standardized log1p(redshift) fine resolution

        n_scales = len(self.bandwidths)
        per_scale = self.fourier_dim // n_scales
        omegas = []
        for idx, bw in enumerate(self.bandwidths):
            m_k = per_scale if idx < n_scales - 1 else (self.fourier_dim - per_scale * (n_scales - 1))
            w_k = rng.normal(0.0, 1.0 / bw, size=(self.in_dim, m_k)).astype(np.float32) * feature_scales
            omegas.append(w_k)

        # Redshift-color coupled high-frequency modes
        n_coupled = self.fourier_dim // 4
        w_coup = rng.normal(0.0, 1.2, size=(self.in_dim, n_coupled)).astype(np.float32)
        if self.in_dim >= 10:
            w_coup[9, :] *= rng.choice([8.0, 25.0, 60.0], size=n_coupled).astype(np.float32)

        self.omega = np.concatenate([*omegas, w_coup], axis=1)
        total_rff = self.omega.shape[1]
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, total_rff)).astype(np.float32)

        iu, ju = np.triu_indices(self.in_dim)
        self.quad_i = iu
        self.quad_j = ju
        self.out_dim = 1 + self.in_dim + len(iu) + (2 * self.in_dim * self.n_1d_freqs) + total_rff

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map X in R^{N x in_dim} -> Phi(X) in R^{N x out_dim}."""
        X_clean = np.asarray(X, dtype=np.float32)
        X_soft = 15.0 * np.tanh(X_clean / 15.0)
        n = X_soft.shape[0]

        ones = np.ones((n, 1), dtype=np.float32)
        quad = (X_soft[:, self.quad_i] * X_soft[:, self.quad_j]) * np.float32(0.25)

        angles_1d = X_soft[:, :, None] * self.freqs_1d[None, None, :]
        scale_1d = np.float32(1.0 / np.sqrt(self.n_1d_freqs))
        sin_1d = (np.sin(angles_1d) * scale_1d).reshape(n, -1)
        cos_1d = (np.cos(angles_1d) * scale_1d).reshape(n, -1)

        proj = X_soft @ self.omega + self.phase
        rff = np.sqrt(2.0 / self.omega.shape[1], dtype=np.float32) * np.cos(proj)
        return np.concatenate([ones, X_soft, quad, sin_1d, cos_1d, rff], axis=1)

    def to_state(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "n_1d_freqs": self.n_1d_freqs,
            "fourier_dim": self.fourier_dim,
            "bandwidths": list(self.bandwidths),
            "seed": self.seed,
            "freqs_1d": self.freqs_1d,
            "omega": self.omega,
            "phase": self.phase,
        }

    @classmethod
    def from_state(cls, state: dict) -> "SpectralAstrophysicalLens":
        obj = cls(
            in_dim=int(state["in_dim"]),
            n_1d_freqs=int(state["n_1d_freqs"]),
            fourier_dim=int(state["fourier_dim"]),
            bandwidths=tuple(state["bandwidths"]),
            seed=int(state["seed"]),
        )
        obj.freqs_1d = np.asarray(state["freqs_1d"], dtype=np.float32)
        obj.omega = np.asarray(state["omega"], dtype=np.float32)
        obj.phase = np.asarray(state["phase"], dtype=np.float32)
        return obj


@dataclass
class OrbitTelemetry:
    """Full orbital trajectory telemetry across Leapfrog steps t = 0..T."""

    q_traj: np.ndarray       # (T + 1, N, d)
    p_traj: np.ndarray       # (T + 1, N, d)
    kinetic: np.ndarray      # (T + 1, N)
    potential: np.ndarray    # (T + 1, N)
    hamiltonian: np.ndarray  # (T + 1, N)


class GravNet:
    """Astrophysical Potential Dynamical Network with Multi-Halo Plummer Gravity."""

    SUPERCLUSTER_RADIUS = 3.2

    def __init__(
        self,
        in_dim: int = 10,
        n_classes: int = 3,
        phase_dim: int = 6,
        sub_halos_per_class: int = 6,
        n_1d_freqs: int = 44,
        fourier_dim: int = 2304,
        orbital_steps: int = 10,
        dt: float = 0.11,
        damping: float = 0.95,
        softening: float = 0.90,
        confinement: float = 0.04,
        inv_temp: float = 2.6,
        seed: int = 42,
    ):
        assert phase_dim >= 3, "phase_dim must be >= 3 (2D supercluster plane + >=1D spectral fiber)"
        assert dt > 0.0 and damping > 0.0 and softening > 0.0 and confinement > 0.0

        self.in_dim = int(in_dim)
        self.n_classes = int(n_classes)
        self.phase_dim = int(phase_dim)
        self.K = int(sub_halos_per_class)
        self.orbital_steps = int(orbital_steps)
        self.dt = float(dt)
        self.damping = float(damping)
        self.softening = float(softening)
        self.confinement = float(confinement)
        self.inv_temp = float(inv_temp)
        self.seed = int(seed)

        self.lens = SpectralAstrophysicalLens(
            in_dim=self.in_dim,
            n_1d_freqs=n_1d_freqs,
            fourier_dim=fourier_dim,
            seed=self.seed,
        )
        M = self.lens.out_dim

        rng = np.random.default_rng(self.seed)
        # Cosmic-Origin Launch Constraint: W_q[:, :2] == 0 so q_orb(0) = (0, 0) always!
        self.W_q = np.zeros((M, self.phase_dim), dtype=np.float64)
        self.W_q[:, 2:] = rng.normal(0.0, 0.01, size=(M, self.phase_dim - 2))
        self.W_p = np.zeros((M, self.phase_dim), dtype=np.float64)

        # Equilateral 120-deg Supercluster Centers in (q_1, q_2): 90 deg, 210 deg, 330 deg
        angles = [np.pi / 2.0, 7.0 * np.pi / 6.0, 11.0 * np.pi / 6.0]
        self.supercluster_centers = np.zeros((self.n_classes, self.phase_dim), dtype=np.float64)
        for c, ang in enumerate(angles[: self.n_classes]):
            self.supercluster_centers[c, 0] = self.SUPERCLUSTER_RADIUS * np.cos(ang)
            self.supercluster_centers[c, 1] = self.SUPERCLUSTER_RADIUS * np.sin(ang)

        self.mu = np.repeat(self.supercluster_centers[:, None, :], self.K, axis=1)
        self.mu += rng.normal(0.0, 0.30, size=(self.n_classes, self.K, self.phase_dim))
        self.G = np.full((self.n_classes, self.K), 2.4 / self.K, dtype=np.float64)
        self.chem_pot = np.zeros(self.n_classes, dtype=np.float64)

    @property
    def num_attractors(self) -> int:
        return self.n_classes * self.K

    @property
    def num_parameters(self) -> int:
        # Trainable physical parameters: W_q[:, 2:], W_p, mu, G, chem_pot, inv_temp
        return (
            self.W_q[:, 2:].size
            + self.W_p.size
            + self.mu.size
            + self.G.size
            + self.chem_pot.size
            + 1
        )

    def _damped_ballistic_reach(self, steps: Optional[int] = None) -> float:
        """Exact displacement of a free unit-velocity particle under Leapfrog Hubble drag."""
        T = self.orbital_steps if steps is None else int(steps)
        eta_d = float(np.exp(-0.5 * self.damping * self.dt))
        pos, vel = 0.0, 1.0
        for _ in range(T):
            v_half = eta_d * vel
            pos += self.dt * v_half
            vel = eta_d * v_half
        return max(pos, 1e-6)

    # ------------------------------------------------------------------
    # Exact Analytical Gravitational Physics (Zero Autograd)
    # ------------------------------------------------------------------
    def class_plummer_potentials(self, q: np.ndarray) -> np.ndarray:
        """Compute per-class Plummer gravitational potentials U_c(q) < 0 for q in R^{N x d}."""
        diff = q[:, None, None, :] - self.mu[None, :, :, :]  # (N, C, K, d)
        dist_sq = np.sum(diff * diff, axis=-1) + (self.softening ** 2)  # (N, C, K)
        return -np.sum(self.G[None, :, :] / np.sqrt(dist_sq), axis=-1)  # (N, C)

    def total_potential(self, q: np.ndarray) -> np.ndarray:
        """Total potential energy U(q) = sum_c U_c(q) + 0.5 * kappa * ||q||_2^2."""
        u_class = self.class_plummer_potentials(q)
        u_conf = 0.5 * self.confinement * np.sum(q * q, axis=-1)
        return np.sum(u_class, axis=-1) + u_conf

    def gravitational_force(self, q: np.ndarray) -> np.ndarray:
        """Closed-form gravitational force F_grav(q) = -grad_q U(q) in R^{N x d}."""
        diff = q[:, None, None, :] - self.mu[None, :, :, :]  # (N, C, K, d)
        dist_sq = np.sum(diff * diff, axis=-1) + (self.softening ** 2)  # (N, C, K)
        coeff = -(self.G[None, :, :] * (dist_sq ** -1.5))[:, :, :, None]  # (N, C, K, 1)
        f_plummer = np.sum(coeff * diff, axis=(1, 2))  # (N, d)
        return f_plummer - self.confinement * q

    def launch_particles(self, Phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map lens basis Phi(X) -> initial phase state (q0, p0) with q_orb(0) == (0, 0)."""
        Phi64 = np.asarray(Phi, dtype=np.float64)
        q0 = Phi64 @ self.W_q
        q0[:, :2] = 0.0  # Enforce strict cosmic-origin launch in 2D Supercluster Plane
        p0 = Phi64 @ self.W_p
        return q0, p0

    def simulate_orbit(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        steps: Optional[int] = None,
        record_trajectory: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[OrbitTelemetry]]:
        """Symplectic Dissipative Leapfrog (Strang-split) orbital integration."""
        T = self.orbital_steps if steps is None else int(steps)
        q = np.array(q0, dtype=np.float64, copy=True)
        p = np.array(p0, dtype=np.float64, copy=True)

        if T == 0:
            if not record_trajectory:
                return q, p, None
            kin = 0.5 * np.sum(p * p, axis=-1)[None, :]
            pot = self.total_potential(q)[None, :]
            return q, p, OrbitTelemetry(q[None], p[None], kin, pot, kin + pot)

        dt = self.dt
        eta_d = float(np.exp(-0.5 * self.damping * dt))

        if record_trajectory:
            N, d = q.shape
            q_hist = np.zeros((T + 1, N, d), dtype=np.float64)
            p_hist = np.zeros((T + 1, N, d), dtype=np.float64)
            kin_hist = np.zeros((T + 1, N), dtype=np.float64)
            pot_hist = np.zeros((T + 1, N), dtype=np.float64)
            q_hist[0], p_hist[0] = q, p
            kin_hist[0] = 0.5 * np.sum(p * p, axis=-1)
            pot_hist[0] = self.total_potential(q)

        f_curr = self.gravitational_force(q)
        for t in range(1, T + 1):
            p_half = eta_d * p + 0.5 * dt * f_curr
            q = q + dt * p_half
            f_curr = self.gravitational_force(q)
            p = eta_d * p_half + 0.5 * dt * f_curr

            if record_trajectory:
                q_hist[t] = q
                p_hist[t] = p
                kin_hist[t] = 0.5 * np.sum(p * p, axis=-1)
                pot_hist[t] = self.total_potential(q)

        telemetry = (
            OrbitTelemetry(q_hist, p_hist, kin_hist, pot_hist, kin_hist + pot_hist)
            if record_trajectory
            else None
        )
        return q, p, telemetry

    def boltzmann_capture_probs(self, q_T: np.ndarray) -> np.ndarray:
        """Canonical Boltzmann-Gibbs capture distribution across class gravitational wells."""
        u_class = self.class_plummer_potentials(q_T)
        logits = -self.inv_temp * u_class + self.chem_pot[None, :]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def lagrange_saddle_metrics(
        self, X: np.ndarray, steps: Optional[int] = None, record_trajectory: bool = False
    ) -> Dict[str, object]:
        """Compute terminal state, capture probabilities, and Lagrange L1 potential barrier."""
        Phi = self.lens.transform(X)
        q0, p0 = self.launch_particles(Phi)
        q_T, p_T, tel = self.simulate_orbit(q0, p0, steps=steps, record_trajectory=record_trajectory)
        u_class = self.class_plummer_potentials(q_T)
        eff_energy = u_class - (self.chem_pot[None, :] / max(self.inv_temp, 1e-6))
        sorted_e = np.sort(eff_energy, axis=1)
        barrier_gap = sorted_e[:, 1] - sorted_e[:, 0]  # Delta E_{L1} >= 0
        residual_ke = 0.5 * np.sum(p_T * p_T, axis=-1)
        probs = self.boltzmann_capture_probs(q_T)
        return {
            "q0": q0,
            "p0": p0,
            "q_T": q_T,
            "p_T": p_T,
            "u_class": u_class,
            "eff_energy": eff_energy,
            "barrier_gap": barrier_gap,
            "residual_ke": residual_ke,
            "probs": probs,
            "telemetry": tel,
        }

    def predict_proba_from_phi(self, Phi: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        q0, p0 = self.launch_particles(Phi)
        q_T, _, _ = self.simulate_orbit(q0, p0, steps=steps, record_trajectory=False)
        return self.boltzmann_capture_probs(q_T).astype(np.float32)

    def predict_proba(self, X: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        Phi = self.lens.transform(X)
        return self.predict_proba_from_phi(Phi, steps=steps)

    def predict(self, X: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        return np.argmax(self.predict_proba(X, steps=steps), axis=1)

    # ------------------------------------------------------------------
    # Backprop-Free Physical Learning (Stages 1, 2, 3)
    # ------------------------------------------------------------------
    def initialize_proto_halos_and_lens(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        Phi_train: Optional[np.ndarray] = None,
        ridge_lambda: float = 2.5,
    ) -> None:
        """Stage 1: Condense K astrophysical sub-halos per class & solve SPD Cholesky launch."""
        if Phi_train is None:
            Phi_train = self.lens.transform(X_train)
        Phi64 = np.asarray(Phi_train, dtype=np.float64)
        N = len(X_train)
        d_spec = self.phase_dim - 2

        target_halo_pos = np.zeros((N, self.phase_dim), dtype=np.float64)

        for c in range(self.n_classes):
            mask_c = np.where(y_train == c)[0]
            X_c = np.clip(X_train[mask_c], -10.0, 10.0)
            km = KMeans(n_clusters=self.K, n_init=5, random_state=self.seed + c)
            sub_lbl = km.fit_predict(X_c)

            centers_centered = km.cluster_centers_ - np.mean(km.cluster_centers_, axis=0, keepdims=True)
            u_svd, s_svd, _ = np.linalg.svd(centers_centered, full_matrices=False)
            take = min(d_spec, u_svd.shape[1])
            fiber_coords = np.zeros((self.K, d_spec), dtype=np.float64)
            fiber_coords[:, :take] = u_svd[:, :take] * np.minimum(s_svd[:take], 2.0)
            max_norm = float(np.max(np.linalg.norm(fiber_coords, axis=1))) + 1e-6
            fiber_coords = 0.85 * (fiber_coords / max_norm)

            for m in range(self.K):
                ang_m = 2.0 * np.pi * m / self.K
                halo = self.supercluster_centers[c].copy()
                halo[0] += 0.45 * np.cos(ang_m)
                halo[1] += 0.45 * np.sin(ang_m)
                halo[2:] = fiber_coords[m]
                self.mu[c, m] = halo

                sub_idx = mask_c[sub_lbl == m]
                target_halo_pos[sub_idx] = halo
                frac = max(0.06, len(sub_idx) / max(1, len(mask_c)))
                self.G[c, m] = 2.4 * frac

        # SPD Cholesky solve: (Phi^T Phi + lambda I) W = Phi^T Y_target
        gram = Phi64.T @ Phi64
        gram[np.diag_indices_from(gram)] += ridge_lambda
        cf, low = cho_factor(gram, lower=True, check_finite=False)

        # Launch strictly at cosmic origin (0, 0) in 2D plane; embed in spectral fiber 2..d
        target_q0 = np.zeros_like(target_halo_pos)
        target_q0[:, 2:] = target_halo_pos[:, 2:]
        self.W_q = np.zeros((Phi64.shape[1], self.phase_dim), dtype=np.float64)
        self.W_q[:, 2:] = cho_solve((cf, low), Phi64.T @ target_q0[:, 2:], check_finite=False)

        # Initial ballistic momentum kick calibrated to damped Leapfrog reach
        reach = self._damped_ballistic_reach(self.orbital_steps)
        target_p0 = ((target_halo_pos - target_q0) / reach) * 0.90
        self.W_p = cho_solve((cf, low), Phi64.T @ target_p0, check_finite=False)

    def sculpt_spacetime_epoch(
        self,
        Phi_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int = 2048,
        lr_vel: float = 0.040,
        lr_fiber: float = 0.014,
        lr_mass: float = 0.028,
        lr_halo: float = 0.012,
        weight_decay: float = 1e-5,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Stage 2: Zero-BPTT Contrastive Spacetime Sculpting + Direct Tidal Hebbian Plasticity."""
        if rng is None:
            rng = np.random.default_rng(self.seed)

        N = len(y_train)
        perm = rng.permutation(N)
        total_nll = 0.0

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            Phi_b = Phi_train[idx].astype(np.float64)
            y_b = y_train[idx]
            B = len(idx)

            # 1. Pure forward physical orbital integration (ZERO autograd / ZERO BPTT)
            q0, p0 = self.launch_particles(Phi_b)
            q_T, _, _ = self.simulate_orbit(q0, p0, record_trajectory=False)

            # 2. Canonical Boltzmann capture probabilities & signed contrastive residual
            probs = self.boltzmann_capture_probs(q_T)
            total_nll -= float(np.sum(np.log(np.clip(probs[np.arange(B), y_b], 1e-12, 1.0))))
            residual = np.eye(self.n_classes, dtype=np.float64)[y_b] - probs  # (B, C)

            diff_to_halo = self.mu[None, :, :, :] - q_T[:, None, None, :]  # (B, C, K, d)
            dist_sq = np.sum(diff_to_halo * diff_to_halo, axis=-1) + (self.softening ** 2)
            inv_dist = 1.0 / np.sqrt(dist_sq)
            inv_dist3 = dist_sq ** (-1.5)

            # 3. Contrastive Plummer Mass Sculpting (Delta G_{c,m})
            dG = (self.inv_temp / B) * np.sum(residual[:, :, None] * inv_dist, axis=0)
            self.G = np.clip(self.G + lr_mass * dG, 0.06, 6.0)

            # 4. Contrastive Attractor Drift (Delta mu_{c,m}) with ternary anchor spring
            weight_ck = self.inv_temp * residual[:, :, None] * self.G[None, :, :] * inv_dist3
            d_mu = -(1.0 / B) * np.sum(weight_ck[:, :, :, None] * diff_to_halo, axis=0)
            anchor_pull = 0.025 * (self.supercluster_centers[:, None, :] - self.mu)
            self.mu = self.mu + lr_halo * (d_mu + anchor_pull)

            # 5. Direct Terminal Tidal Force Vector -> Local Hebbian Lens Plasticity
            f_tidal = np.sum(weight_ck[:, :, :, None] * diff_to_halo, axis=(1, 2))  # (B, d)
            dWp = (Phi_b.T @ f_tidal) / B - weight_decay * self.W_p
            self.W_p += lr_vel * dWp

            dWq_spec = (Phi_b.T @ f_tidal[:, 2:]) / B - weight_decay * self.W_q[:, 2:]
            self.W_q[:, 2:] += lr_fiber * dWq_spec
            self.W_q[:, :2] = 0.0  # Preserve exact cosmic-origin launch

        return total_nll / N

    def calibrate_chemical_potentials(
        self,
        Phi_eval: np.ndarray,
        y_eval: np.ndarray,
        n_trials: int = 120,
        seed: int = 42,
    ) -> float:
        """Stage 3: Derivative-free calibration of grand-canonical chemical potentials nu_c."""
        q0, p0 = self.launch_particles(Phi_eval)
        q_T, _, _ = self.simulate_orbit(q0, p0, record_trajectory=False)
        u_class = self.class_plummer_potentials(q_T)

        best_nu = self.chem_pot.copy()
        best_temp = self.inv_temp

        def eval_f1(nu_vec: np.ndarray, temp_val: float) -> float:
            logits = -temp_val * u_class + nu_vec[None, :]
            preds = np.argmax(logits, axis=1)
            reg = 0.0015 * float(np.sum(nu_vec ** 2))
            return fast_macro_f1(y_eval, preds, n_classes=self.n_classes) - reg

        best_score = eval_f1(best_nu, best_temp)
        rng = np.random.default_rng(seed)

        for step in range(n_trials):
            scale = 0.10 * (0.97 ** step)
            cand_nu = best_nu + rng.normal(0.0, scale, size=self.n_classes)
            cand_nu -= np.mean(cand_nu)
            cand_temp = float(np.clip(best_temp + rng.normal(0.0, 0.08), 0.8, 6.0))
            sc = eval_f1(cand_nu, cand_temp)
            if sc > best_score + 1e-6:
                best_score = sc
                best_nu = cand_nu
                best_temp = cand_temp

        self.chem_pot = best_nu
        self.inv_temp = best_temp
        logits = -self.inv_temp * u_class + self.chem_pot[None, :]
        return fast_macro_f1(y_eval, np.argmax(logits, axis=1), n_classes=self.n_classes)

    def copy_mutable_weights(self) -> dict:
        return {
            "W_q": self.W_q.copy(),
            "W_p": self.W_p.copy(),
            "mu": self.mu.copy(),
            "G": self.G.copy(),
            "chem_pot": self.chem_pot.copy(),
            "inv_temp": float(self.inv_temp),
        }

    def load_mutable_weights(self, state: dict) -> None:
        self.W_q = state["W_q"].copy()
        self.W_p = state["W_p"].copy()
        self.mu = state["mu"].copy()
        self.G = state["G"].copy()
        self.chem_pot = state["chem_pot"].copy()
        self.inv_temp = float(state["inv_temp"])

    def to_checkpoint(self) -> dict:
        return {
            "config": {
                "in_dim": self.in_dim,
                "n_classes": self.n_classes,
                "phase_dim": self.phase_dim,
                "sub_halos_per_class": self.K,
                "n_1d_freqs": self.lens.n_1d_freqs,
                "fourier_dim": self.lens.fourier_dim,
                "orbital_steps": self.orbital_steps,
                "dt": self.dt,
                "damping": self.damping,
                "softening": self.softening,
                "confinement": self.confinement,
                "inv_temp": self.inv_temp,
                "seed": self.seed,
            },
            "lens": self.lens.to_state(),
            "W_q": self.W_q,
            "W_p": self.W_p,
            "mu": self.mu,
            "G": self.G,
            "chem_pot": self.chem_pot,
            "supercluster_centers": self.supercluster_centers,
        }

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "GravNet":
        cfg = ckpt["config"]
        model = cls(**cfg)
        model.lens = SpectralAstrophysicalLens.from_state(ckpt["lens"])
        model.W_q = np.asarray(ckpt["W_q"], dtype=np.float64)
        model.W_p = np.asarray(ckpt["W_p"], dtype=np.float64)
        model.mu = np.asarray(ckpt["mu"], dtype=np.float64)
        model.G = np.asarray(ckpt["G"], dtype=np.float64)
        model.chem_pot = np.asarray(ckpt["chem_pot"], dtype=np.float64)
        model.supercluster_centers = np.asarray(ckpt["supercluster_centers"], dtype=np.float64)
        return model


class GravNetMultiverse:
    """Ensemble of independent GravNet universes averaging canonical Boltzmann capture probs."""

    def __init__(self, universes: List[GravNet]):
        assert len(universes) >= 1
        self.universes = universes

    def predict_proba(self, X: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        probs = [u.predict_proba(X, steps=steps) for u in self.universes]
        return np.mean(probs, axis=0)

    def predict(self, X: np.ndarray, steps: Optional[int] = None) -> np.ndarray:
        return np.argmax(self.predict_proba(X, steps=steps), axis=1)

    def to_checkpoint(self) -> dict:
        return {"universes": [u.to_checkpoint() for u in self.universes]}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "GravNetMultiverse":
        return cls([GravNet.from_checkpoint(c) for c in ckpt["universes"]])


def verify_physics_rigor() -> Dict[str, float]:
    """Symbolic (SymPy) and numerical self-verification of Plummer force and Lyapunov dH/dt <= 0."""
    import sympy as sp

    # 1. Exact Symbolic Verification via SymPy
    q1, q2, m1, m2 = sp.symbols("q1 q2 m1 m2", real=True)
    G_s, eps_s, kap_s, gam_s = sp.symbols("G eps kappa gamma", real=True, positive=True)
    p1, p2 = sp.symbols("p1 p2", real=True)

    r2 = (q1 - m1) ** 2 + (q2 - m2) ** 2 + eps_s ** 2
    U_sym = -G_s / sp.sqrt(r2) + sp.Rational(1, 2) * kap_s * (q1 ** 2 + q2 ** 2)
    F1_exact = -sp.diff(U_sym, q1)
    F1_closed = -G_s * (q1 - m1) / (r2 ** sp.Rational(3, 2)) - kap_s * q1
    sym_force_res = sp.simplify(F1_exact - F1_closed)
    assert sym_force_res == 0, f"Symbolic force residual non-zero: {sym_force_res}"

    H_sym = sp.Rational(1, 2) * (p1 ** 2 + p2 ** 2) + U_sym
    dH_dt = sp.simplify(
        sp.diff(H_sym, q1) * p1
        + sp.diff(H_sym, q2) * p2
        + sp.diff(H_sym, p1) * (-sp.diff(U_sym, q1) - gam_s * p1)
        + sp.diff(H_sym, p2) * (-sp.diff(U_sym, q2) - gam_s * p2)
        + gam_s * (p1 ** 2 + p2 ** 2)
    )
    assert dH_dt == 0, f"Symbolic Lyapunov residual non-zero: {dH_dt}"

    # 2. Numerical Finite-Difference & Trajectory Verification
    model = GravNet(phase_dim=6, sub_halos_per_class=6, orbital_steps=20, dt=0.08, damping=0.95)
    rng = np.random.default_rng(123)
    q_test = rng.normal(0.0, 2.0, size=(8, model.phase_dim))
    p_test = rng.normal(0.0, 1.2, size=(8, model.phase_dim))

    f_analytic = model.gravitational_force(q_test)
    f_numeric = np.zeros_like(f_analytic)
    eps_fd = 1e-6
    for j in range(model.phase_dim):
        dq = np.zeros_like(q_test)
        dq[:, j] = eps_fd
        u_plus = model.total_potential(q_test + dq)
        u_minus = model.total_potential(q_test - dq)
        f_numeric[:, j] = -(u_plus - u_minus) / (2.0 * eps_fd)

    max_force_err = float(np.max(np.abs(f_analytic - f_numeric)))
    assert max_force_err < 1e-8, f"Force gradient mismatch: {max_force_err}"

    _, _, tel = model.simulate_orbit(q_test, p_test, steps=20, record_trajectory=True)
    dH = np.diff(tel.hamiltonian, axis=0)
    max_energy_increase = float(np.max(dH))
    mean_energy_drop = float(np.mean(tel.hamiltonian[-1] - tel.hamiltonian[0]))
    assert max_energy_increase < 1e-6, f"Lyapunov violation: {max_energy_increase}"

    return {
        "symbolic_force_residual": 0.0,
        "symbolic_lyapunov_residual": 0.0,
        "max_force_gradient_error": max_force_err,
        "max_step_energy_increase": max_energy_increase,
        "mean_total_energy_drop": mean_energy_drop,
        "final_rms_momentum": float(np.sqrt(np.mean(tel.p_traj[-1] ** 2))),
    }


if __name__ == "__main__":
    stats = verify_physics_rigor()
    print("=== GravNet Symbolic & Numerical Physics Verification ===")
    for k, v in stats.items():
        print(f"  {k}: {v:.3e}")
    print("All analytical force & Lyapunov dissipation checks PASSED!")
