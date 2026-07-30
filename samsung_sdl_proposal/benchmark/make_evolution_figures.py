"""Evolution of the LEARNED map per budget checkpoint, per policy.

Re-runs one representative campaign (seed 1) per policy with a snapshot hook,
and renders what the model believes at 30 / 70 / 110 / 150 electrical tests:
  fig4_map_evolution.png    -- predicted mean log10(J) on the 2-D slice
  fig5_trust_evolution.png  -- predicted aleatoric std (the "trust map")
Rightmost column = ground truth on the same slice. Dots = electrical tests
executed so far (7-D positions projected onto the slice axes).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_benchmark as rb
from synth import structure, fingerprint, outcome_mean, D_NM, T_ANN, HIDDEN_LABEL_WZ

NAVY, TEAL, RED, GRAY = "#1B2A4A", "#2A9D8F", "#B43E3E", "#8A93A2"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": "#C9CFD8", "axes.labelcolor": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
})

CHECKPOINTS = [30, 70, 110, 150]
POLICIES = [("epistemic", "info-gain, epistemic-aware (ours)"),
            ("us_total", "uncertainty sampling (total)"),
            ("random", "random")]
SEED = 1
NG = 90  # slice grid resolution


def slice_features(ng=NG):
    """Model features + truth on the (thickness, anneal T) slice, others mid."""
    u2, u5 = np.meshgrid(np.linspace(0, 1, ng), np.linspace(0, 1, ng))
    U = np.full((ng * ng, 7), 0.5)
    U[:, 2] = u2.ravel()
    U[:, 5] = u5.ravel()
    st = structure(U, np.zeros(ng * ng))
    s = fingerprint(st, rng=None)                    # noise-free fingerprint
    X = np.hstack([s, U[:, 5:7]])
    m = outcome_mean(U, st)
    d = D_NM[0] + (D_NM[1] - D_NM[0]) * u2
    t = T_ANN[0] + (T_ANN[1] - T_ANN[0]) * u5
    truth = {k: v.reshape(ng, ng) for k, v in m.items()}
    return X, d, t, truth


def collect(policy):
    """Run one campaign, snapshotting slice predictions at each round."""
    Xg, d, t, truth = slice_features()
    snaps = {}

    def on_round(n, model, rows, cand):
        mu, _, sal, _ = model.predict(Xg)
        uniq = sorted(set(rows))
        snaps[n] = dict(
            mu=mu.reshape(NG, NG), sal=sal.reshape(NG, NG),
            qd=cand["struct"]["d_nm"][uniq],
            qt=300 + 400 * cand["U"][uniq, 5])

    rb.run_policy(policy, SEED, on_round=on_round)
    return snaps


def render(all_snaps, field, truth_img, cmap, vmin, vmax, cbar_label,
           title, fname, band_contour, d, t, truth_wz):
    nrow, ncol = len(POLICIES), len(CHECKPOINTS) + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.75 * ncol, 2.55 * nrow),
                             constrained_layout=True, sharex=True, sharey=True)
    for r, (pol, name) in enumerate(POLICIES):
        snaps = all_snaps[pol]
        for c, n in enumerate(CHECKPOINTS):
            ax = axes[r, c]
            im = ax.pcolormesh(d, t, snaps[n][field], cmap=cmap,
                               vmin=vmin, vmax=vmax, shading="auto")
            if band_contour:
                ax.contour(d, t, truth_wz, levels=[HIDDEN_LABEL_WZ],
                           colors=[NAVY], linewidths=1.1, linestyles="--")
            k = min(len(snaps[n]["qd"]), n)
            ax.scatter(snaps[n]["qd"], snaps[n]["qt"], s=7, c="white",
                       edgecolors=NAVY, linewidths=0.4, alpha=0.9)
            if r == 0:
                ax.set_title(f"after {n} tests")
            if c == 0:
                ax.set_ylabel(f"{name}\nanneal T (°C)", fontsize=9.5)
        ax = axes[r, ncol - 1]
        ax.pcolormesh(d, t, truth_img, cmap=cmap, vmin=vmin, vmax=vmax,
                      shading="auto")
        if band_contour:
            ax.contour(d, t, truth_wz, levels=[HIDDEN_LABEL_WZ],
                       colors=[NAVY], linewidths=1.1, linestyles="--")
        if r == 0:
            ax.set_title("GROUND TRUTH")
    for ax in axes[-1]:
        ax.set_xlabel("thickness (nm)")
    fig.colorbar(im, ax=axes, shrink=0.75, label=cbar_label)
    fig.suptitle(title, color=NAVY, fontweight="bold", fontsize=13)
    fig.savefig(fname, dpi=180)
    plt.close(fig)
    print("wrote", fname)


def main():
    _, d, t, truth = slice_features()
    all_snaps = {}
    for pol, _ in POLICIES:
        print("running", pol, "...")
        all_snaps[pol] = collect(pol)

    render(all_snaps, "mu", truth["logj"], "magma",
           float(truth["logj"].min()), float(truth["logj"].max()),
           "log$_{10}$ J (A/cm$^2$)",
           "What the model BELIEVES the leakage map is — one campaign (seed 1), "
           "dots = electrical tests so far (7-D positions projected)",
           "fig4_map_evolution.png", False, d, t, truth["w_z"])

    render(all_snaps, "sal", 1.5 * truth["w_z"], "Reds", 0.0, 1.6,
           "aleatoric std (decades)",
           "What the model BELIEVES the trust map is — where fast measurements "
           "cannot predict leakage (dashed = true hidden zone)",
           "fig5_trust_evolution.png", True, d, t, truth["w_z"])


if __name__ == "__main__":
    main()
