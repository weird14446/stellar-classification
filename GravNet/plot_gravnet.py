# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.12", "scikit-learn>=1.4", "pandas>=2.0", "matplotlib>=3.8", "joblib>=1.3"]
# ///
"""Generate publication-quality scientific visualizations for GravNet."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from GravNet.preprocess import TARGET, add_color_indices, load_clean
from GravNet.physics_model import GravNet

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"

CLASS_COLORS = {
    "GALAXY": "#1f77b4",  # Sapphire Blue
    "QSO": "#d62728",     # Crimson Quasar Red
    "STAR": "#2ca02c",    # Stellar Emerald Green
}


def plot_potential_and_orbits(model: GravNet, X_test: np.ndarray, y_test: np.ndarray, classes: list[str]) -> None:
    """Figure 1: 2D Spacetime Plummer Potential Contour + Trajectories + Terminal Capture Cloud."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=300)

    # Compute 2D potential slice U(q1, q2, 0, 0, 0, 0)
    grid_lim = 4.9
    g1 = np.linspace(-grid_lim, grid_lim, 220)
    g2 = np.linspace(-grid_lim, grid_lim, 220)
    Q1, Q2 = np.meshgrid(g1, g2)
    pts = np.zeros((Q1.size, model.phase_dim), dtype=np.float64)
    pts[:, 0] = Q1.ravel()
    pts[:, 1] = Q2.ravel()
    U_grid = model.total_potential(pts).reshape(Q1.shape)

    # Coarse quiver grid for gravitational force field F_grav(q1, q2, 0)
    q1_c = np.linspace(-4.4, 4.4, 17)
    q2_c = np.linspace(-4.4, 4.4, 17)
    Q1c, Q2c = np.meshgrid(q1_c, q2_c)
    pts_c = np.zeros((Q1c.size, model.phase_dim), dtype=np.float64)
    pts_c[:, 0] = Q1c.ravel()
    pts_c[:, 1] = Q2c.ravel()
    F_c = model.gravitational_force(pts_c)
    F1 = F_c[:, 0].reshape(Q1c.shape)
    F2 = F_c[:, 1].reshape(Q2c.shape)
    F_mag = np.hypot(F1, F2) + 1e-8

    ax = axes[0]
    cf = ax.contourf(Q1, Q2, U_grid, levels=28, cmap="bone", alpha=0.88)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"Gravitational Potential $U(q_1, q_2, \mathbf{0})$ [arb. energy]", fontsize=9.5)

    ax.quiver(
        Q1c, Q2c, F1 / F_mag, F2 / F_mag,
        color="white", alpha=0.26, scale=26, width=0.0028, headwidth=3.5
    )

    # Simulate trajectories for representative confident + Lagrange boundary objects
    lag = model.lagrange_saddle_metrics(X_test[:2500], record_trajectory=True)
    q_traj = lag["telemetry"].q_traj  # (T+1, N, d)
    barrier = lag["barrier_gap"]
    y_sub = y_test[:2500]

    rng = np.random.default_rng(42)
    for c_idx, cname in enumerate(classes):
        cls_mask = y_sub == c_idx
        cls_thresh = np.quantile(barrier[cls_mask], 0.55)
        cand = np.where(cls_mask & (barrier >= cls_thresh))[0]
        chosen = rng.choice(cand, size=min(8, len(cand)), replace=False)
        for j, idx in enumerate(chosen):
            orb = q_traj[:, idx, :2]
            ax.plot(
                orb[:, 0], orb[:, 1],
                color=CLASS_COLORS[cname], lw=1.65, alpha=0.88,
                label=f"{cname} Orbit" if j == 0 else None,
            )
            ax.scatter(orb[-1, 0], orb[-1, 1], color=CLASS_COLORS[cname], edgecolor="white", s=34, zorder=6)

    # Highlight 2 Lagrange L1 boundary trajectories
    amb_cand = np.argsort(barrier)[:2]
    for k, idx in enumerate(amb_cand):
        orb = q_traj[:, idx, :2]
        ax.plot(
            orb[:, 0], orb[:, 1],
            color="#ffbf00", lw=2.1, ls="--", zorder=7,
            label=r"Lagrange $L_1$ Boundary Orbit" if k == 0 else None,
        )
        ax.scatter(orb[-1, 0], orb[-1, 1], color="#ffbf00", edgecolor="black", marker="D", s=46, zorder=8)

    # Plot 18 Plummer Halo Cores & Cosmic Origin
    for c_idx, cname in enumerate(classes):
        halos_2d = model.mu[c_idx, :, :2]
        masses = model.G[c_idx, :]
        ax.scatter(
            halos_2d[:, 0], halos_2d[:, 1],
            s=95 * masses + 45, marker="*", color=CLASS_COLORS[cname],
            edgecolor="white", linewidth=0.9, zorder=9,
            label=f"{cname} Plummer Cores ($K=6$)" if c_idx == 0 else None,
        )

    ax.scatter(
        [0.0], [0.0], s=130, marker="P", color="white", edgecolor="black", linewidth=1.2, zorder=10,
        label=r"Cosmic Launch Origin $\mathbf{q}_{\mathrm{orb}}(0)=(0,0)$"
    )
    ax.set_title("(a) Plummer Potential Landscape & Orbital Trajectories", fontsize=11.5, fontweight="bold")
    ax.set_xlabel(r"Orbital Plane Coordinate $q_1$ [dimensionless]", fontsize=10)
    ax.set_ylabel(r"Orbital Plane Coordinate $q_2$ [dimensionless]", fontsize=10)
    ax.set_xlim(-grid_lim, grid_lim)
    ax.set_ylim(-grid_lim, grid_lim)
    ax.legend(loc="lower left", fontsize=7.6, framealpha=0.90)

    # Panel (b): Terminal Phase-Space Cloud of 2,500 Test Celestial Objects
    ax2 = axes[1]
    ax2.contour(Q1, Q2, U_grid, levels=16, colors="#adb5bd", linewidths=0.65, alpha=0.65)
    qT_2d = q_traj[-1, :, :2]
    for c_idx, cname in enumerate(classes):
        m = y_sub == c_idx
        ax2.scatter(
            qT_2d[m, 0], qT_2d[m, 1],
            s=9, alpha=0.52, color=CLASS_COLORS[cname], label=f"{cname} ($n={int(np.sum(m))}$)"
        )
        sc = model.supercluster_centers[c_idx, :2]
        ax2.annotate(
            cname, xy=(sc[0], sc[1]), xytext=(sc[0] * 1.16, sc[1] * 1.14),
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=CLASS_COLORS[cname],
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=CLASS_COLORS[cname], alpha=0.88),
        )

    # Mark approximate Lagrange L1 midpoints between supercluster pairs
    sc2 = model.supercluster_centers[:, :2]
    for i in range(3):
        mid = 0.5 * (sc2[i] + sc2[(i + 1) % 3])
        ax2.scatter([mid[0]], [mid[1]], marker="X", s=75, color="#ff9f1c", edgecolor="black", zorder=8,
                    label=r"Lagrange $L_1$ Ridge Zone" if i == 0 else None)

    ax2.set_title(r"(b) Terminal Basin Capture State $\mathbf{q}_{\mathrm{orb}}(T)$ ($N=2,500$)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel(r"Terminal Coordinate $q_1(T)$ [dimensionless]", fontsize=10)
    ax2.set_ylabel(r"Terminal Coordinate $q_2(T)$ [dimensionless]", fontsize=10)
    ax2.set_xlim(-grid_lim, grid_lim)
    ax2.set_ylim(-grid_lim, grid_lim)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="lower left", fontsize=8.2, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "potential_and_orbits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lyapunov_and_ablation(model: GravNet, X_test: np.ndarray, history: dict) -> None:
    """Figure 2: Monotonic Lyapunov Energy Decay & Accuracy vs. Orbital Steps T."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), dpi=300)

    # Panel (a): Kinetic, Potential, and Total Hamiltonian across t = 0..14
    T_plot = 14
    lag = model.lagrange_saddle_metrics(X_test[:3000], steps=T_plot, record_trajectory=True)
    tel = lag["telemetry"]
    steps_arr = np.arange(T_plot + 1)

    for arr, label, color, ls in [
        (tel.hamiltonian, r"Total Hamiltonian $H(t) = K(t) + U(t)$", "#0f2c59", "-"),
        (tel.kinetic, r"Kinetic Energy $K(t) = \frac{1}{2}\|\mathbf{p}(t)\|_2^2$", "#d62728", "--"),
        (tel.potential, r"Plummer Potential $U(\mathbf{q}(t))$", "#1f77b4", "-."),
    ]:
        mean_v = np.mean(arr, axis=1)
        q25 = np.percentile(arr, 25, axis=1)
        q75 = np.percentile(arr, 75, axis=1)
        axes[0].plot(steps_arr, mean_v, color=color, lw=2.3, ls=ls, marker="o", markersize=4, label=label)
        axes[0].fill_between(steps_arr, q25, q75, color=color, alpha=0.14)

    axes[0].set_title(r"(a) Lyapunov Energy Dissipation ($\frac{dH}{dt} = -\gamma\|\mathbf{p}\|_2^2 \leq 0$)", fontsize=11.2, fontweight="bold")
    axes[0].set_xlabel(r"Symplectic Leapfrog Step $t$ [steps]", fontsize=10)
    axes[0].set_ylabel(r"Mechanical Energy [dimensionless units]", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(loc="upper right", fontsize=8.4, framealpha=0.92)

    # Panel (b): Accuracy & Macro-F1 across Orbital Steps T
    prog = history["orbital_step_progression"]
    st_x = [p["steps"] for p in prog]
    acc_y = [p["test_acc"] * 100.0 for p in prog]
    f1_y = [p["test_macro_f1"] * 100.0 for p in prog]

    ax2 = axes[1]
    ax2.plot(st_x, acc_y, color="#0969da", lw=2.4, marker="s", markersize=5.5, label="Test Accuracy [%]")
    ax2.plot(st_x, f1_y, color="#1a7f37", lw=2.2, ls="--", marker="^", markersize=5.5, label=r"Test Macro-$F_1 \times 100$ [%]")
    ax2.axvline(model.orbital_steps, color="#cf222e", ls=":", lw=1.6, label=f"Operating Step $T={model.orbital_steps}$")

    # Annotate T=0 cosmic origin vs T=10 captured state
    ax2.annotate(
        f"$T=0$ Origin\n({acc_y[0]:.1f}%)",
        xy=(st_x[0], acc_y[0]), xytext=(1.8, acc_y[0] + 6.0),
        arrowprops=dict(arrowstyle="->", color="#57606a", lw=1.1),
        fontsize=8.8, fontweight="bold", color="#57606a"
    )
    idx_op = st_x.index(model.orbital_steps)
    ax2.annotate(
        f"Captured Equilibrium\n({acc_y[idx_op]:.2f}%)",
        xy=(model.orbital_steps, acc_y[idx_op]), xytext=(model.orbital_steps - 4.8, acc_y[idx_op] - 11.5),
        arrowprops=dict(arrowstyle="->", color="#0969da", lw=1.2),
        fontsize=8.8, fontweight="bold", color="#0f2c59"
    )

    ax2.set_title("(b) Classification Performance vs. Orbital Integration Steps", fontsize=11.2, fontweight="bold")
    ax2.set_xlabel(r"Integration Steps $T$ [steps]", fontsize=10)
    ax2.set_ylabel("Score [%]", fontsize=10)
    ax2.set_ylim(max(20.0, min(f1_y) - 5.0), 100.5)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="lower right", fontsize=8.6, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lyapunov_and_orbital_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lagrange_uncertainty(
    model: GravNet, X_test: np.ndarray, y_test: np.ndarray, raw_z: np.ndarray, classes: list[str], metrics_json: dict
) -> None:
    """Figure 3: Selective Accuracy Curve & Low-Redshift QSO Isolation at Lagrange L1."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=300)

    sel = metrics_json["lagrange_l1_analysis"]["selective_coverage_curve"]
    cov_pct = [s["coverage"] * 100.0 for s in sel]
    acc_pct = [s["accuracy"] * 100.0 for s in sel]
    f1_pct = [s["macro_f1"] * 100.0 for s in sel]

    ax1 = axes[0]
    ax1.plot(cov_pct, acc_pct, color="#0f2c59", lw=2.4, marker="o", label="Selective Test Accuracy [%]")
    ax1.plot(cov_pct, f1_pct, color="#1a7f37", lw=2.1, ls="--", marker="s", label=r"Selective Macro-$F_1 \times 100$ [%]")
    ax1.invert_xaxis()  # 100% coverage -> 50% highest-barrier core
    ax1.set_title(r"(a) Selective Accuracy by Lagrange $L_1$ Potential Barrier $\Delta E_{L_1}$", fontsize=11.0, fontweight="bold")
    ax1.set_xlabel(r"Retained Sample Coverage (Filtered by $\Delta E_{L_1}$ Threshold) [%]", fontsize=9.8)
    ax1.set_ylabel("Classification Metric [%]", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="lower right", fontsize=8.6, framealpha=0.92)

    # Panel (b): Redshift distribution of QSOs in Deep Basin vs Lagrange L1 Ridge
    lag = model.lagrange_saddle_metrics(X_test)
    barrier = lag["barrier_gap"]
    q08 = np.quantile(barrier, 0.08)
    qso_idx = classes.index("QSO")

    z_basin = np.clip(raw_z[(y_test == qso_idx) & (barrier > q08)], 0.0, 4.0)
    z_ridge = np.clip(raw_z[(y_test == qso_idx) & (barrier <= q08)], 0.0, 4.0)

    ax2 = axes[1]
    bins = np.linspace(0.0, 3.8, 36)
    ax2.hist(
        z_basin, bins=bins, density=True, alpha=0.55, color="#1f77b4", edgecolor="white",
        label=f"Deep Plummer Basin QSOs (median $z={np.median(z_basin):.2f}$)"
    )
    ax2.hist(
        z_ridge, bins=bins, density=True, alpha=0.68, color="#d62728", edgecolor="white",
        label=f"Lagrange $L_1$ Ridge QSOs (median $z={np.median(z_ridge):.2f}$)"
    )
    ax2.axvline(0.5, color="#9a6700", ls="--", lw=1.6, label=r"Host-Galaxy Overlap Regime ($z < 0.5$)")
    ax2.set_title(r"(b) Physical Redshift Distribution of $L_1$-Boundary vs. Basin QSOs", fontsize=11.0, fontweight="bold")
    ax2.set_xlabel(r"Spectroscopic Redshift $z$ [dimensionless]", fontsize=10)
    ax2.set_ylabel("Probability Density [per unit $z$]", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lagrange_uncertainty_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paradigm_comparison(metrics_json: dict, classes: list[str]) -> None:
    """Figure 4: 3-Paradigm Accuracy/Speed Comparison & GravNet Normalized Confusion Matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), dpi=300)

    models_lbl = [
        "EDL Pure ES\n(Fixed 32-16)",
        "DL Single MLP\n(Backprop)",
        "DL 3-Seed MLP\n(Backprop)",
        "EDL TWEANN\n(Top-4 Islands)",
        "GravNet Single\n(18 Halos, No-BP)",
        "GravNet Multiverse\n(3 Univ., No-BP)",
    ]
    accs = [
        96.48,
        97.37,
        97.42,
        97.45,
        metrics_json["single_universe"]["accuracy"] * 100.0,
        metrics_json["multiverse_3"]["accuracy"] * 100.0,
    ]
    train_s = [
        13.55,
        24.16,
        72.49,
        94.22,
        metrics_json["timing"]["single_train_sec"],
        metrics_json["timing"]["multiverse_train_sec"],
    ]
    bar_colors = ["#8c959f", "#4a90e2", "#1f6feb", "#2da44e", "#e85d04", "#9d0208"]

    ax1 = axes[0]
    y_pos = np.arange(len(models_lbl))
    bars = ax1.barh(y_pos, accs, color=bar_colors, height=0.60, edgecolor="black", linewidth=0.6, alpha=0.90)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models_lbl, fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlim(96.0, max(accs) + 0.45)
    ax1.set_xlabel("20,000-Sample Held-Out Test Accuracy [%]", fontsize=10)
    ax1.set_title("(a) Three-Paradigm Neural Benchmark (Accuracy & Train Time)", fontsize=11.0, fontweight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.45)

    for bar, acc_v, t_v in zip(bars, accs, train_s):
        ax1.text(
            acc_v + 0.025,
            bar.get_y() + bar.get_height() / 2.0,
            f"{acc_v:.2f}% ({t_v:.1f}s)",
            va="center",
            ha="left",
            fontsize=8.3,
            fontweight="bold",
            color="#1f2328",
        )

    # Panel (b): Row-Normalized Confusion Matrix of GravNet Multiverse
    cm = np.array(metrics_json["multiverse_3"]["confusion_matrix"], dtype=np.float64)
    cm_norm = cm / np.sum(cm, axis=1, keepdims=True) * 100.0

    ax2 = axes[1]
    im = ax2.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalized Recall [%]", fontsize=9.5)

    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_xticklabels(classes, fontsize=9.5, fontweight="bold")
    ax2.set_yticklabels(classes, fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Predicted Celestial Class", fontsize=10)
    ax2.set_ylabel("True Spectroscopic Class", fontsize=10)
    ax2.set_title("(b) GravNet Confusion Matrix (20,000 Test Objects)", fontsize=11.0, fontweight="bold")

    for i in range(len(classes)):
        for j in range(len(classes)):
            val_pct = cm_norm[i, j]
            cnt = int(cm[i, j])
            txt_col = "white" if val_pct > 50.0 else "#1f2328"
            ax2.text(
                j, i, f"{val_pct:.2f}%\n({cnt:,})",
                ha="center", va="center", color=txt_col, fontsize=8.8, fontweight="bold"
            )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paradigm_3way_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODELS_DIR / "preprocessor.joblib")
    preprocessor = bundle["preprocessor"]
    classes = bundle["classes"]

    df = add_color_indices(load_clean())
    feature_cols = [c for c in df.columns if c != TARGET]
    y_all = df[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    _, df_test, _, y_test = train_test_split(
        df[feature_cols], y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_test = preprocessor.transform(df_test)
    raw_z = df_test["redshift"].to_numpy()

    single_model = GravNet.from_checkpoint(joblib.load(MODELS_DIR / "gravnet_single.joblib"))
    history = json.loads((MODELS_DIR / "history.json").read_text(encoding="utf-8"))
    metrics_json = json.loads((MODELS_DIR / "test_metrics.json").read_text(encoding="utf-8"))

    print("Generating GravNet scientific figures (300 DPI)...")
    plot_potential_and_orbits(single_model, X_test, y_test, classes)
    plot_lyapunov_and_ablation(single_model, X_test, history)
    plot_lagrange_uncertainty(single_model, X_test, y_test, raw_z, classes, metrics_json)
    plot_paradigm_comparison(metrics_json, classes)
    print(f"All 4 figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
