"""Figures for the in-silico SDL benchmark (proposal-quality styling)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from synth import (structure, outcome_mean, D_NM, T_ANN,
                   HIDDEN_LABEL_WZ)

NAVY, TEAL, AMBER, GRAY, RED = "#1B2A4A", "#2A9D8F", "#E09A2B", "#8A93A2", "#B43E3E"
POLICY_STYLE = {
    "random":    (GRAY,  "--", "random"),
    "sobol":     ("#6B7C93", ":", "space-filling (Sobol)"),
    "us_total":  (RED,   "-.", "uncertainty sampling (total)"),
    "epistemic": (TEAL,  "-",  "info-gain, epistemic-aware (ours)"),
}
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": "#C9CFD8", "axes.labelcolor": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
})


def slice_grid(n=220):
    """2-D slice over (thickness, anneal T) with all other inputs at mid."""
    u2, u5 = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    U = np.full((n * n, 7), 0.5)
    U[:, 2] = u2.ravel()
    U[:, 5] = u5.ravel()
    st = structure(U, np.zeros(n * n))
    m = outcome_mean(U, st)
    d = D_NM[0] + (D_NM[1] - D_NM[0]) * u2
    t = T_ANN[0] + (T_ANN[1] - T_ANN[0]) * u5
    return d, t, {k: v.reshape(n, n) for k, v in m.items()}


def fig_ground_truth():
    d, t, m = slice_grid()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)

    im0 = axes[0].pcolormesh(d, t, m["k"], cmap="viridis", shading="auto")
    axes[0].set_title("dielectric constant k (true mean)")
    fig.colorbar(im0, ax=axes[0], label="k")

    im1 = axes[1].pcolormesh(d, t, m["logj"], cmap="magma", shading="auto")
    axes[1].set_title("log$_{10}$ leakage J (true mean)")
    fig.colorbar(im1, ax=axes[1], label="log$_{10}$ J (A/cm$^2$)")

    sal = 1.5 * m["w_z"]
    im2 = axes[2].pcolormesh(d, t, sal, cmap="Reds", shading="auto")
    axes[2].contour(d, t, m["w_z"], levels=[HIDDEN_LABEL_WZ], colors=[NAVY],
                    linewidths=1.6, linestyles="--")
    axes[2].set_title("irreducible spread of log J\n(hidden-variable regime)")
    fig.colorbar(im2, ax=axes[2], label="aleatoric std (decades)")

    for ax in axes:
        ax.set_xlabel("thickness (nm)")
    axes[0].set_ylabel("anneal temperature (°C)")
    fig.suptitle("Synthetic ground truth — 2-D slice of the 7-D space "
                 "(physics-inspired, NOT a material model)",
                 color=NAVY, fontweight="bold")
    fig.savefig("fig1_ground_truth.png", dpi=200)
    plt.close(fig)


def fig_sampling(results):
    """Where each policy spent its unique escalations (seed 0)."""
    d, t, m = slice_grid()
    pols = ["random", "us_total", "epistemic"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    for ax, pol in zip(axes, pols):
        ax.pcolormesh(d, t, 1.5 * m["w_z"], cmap="Reds", shading="auto",
                      alpha=0.55, vmin=0, vmax=1.6)
        ax.contour(d, t, m["w_z"], levels=[HIDDEN_LABEL_WZ], colors=[NAVY],
                   linewidths=1.4, linestyles="--")
        q = results[pol][0]["queries"]
        wz = np.array(q["w_z"])
        inside = wz > HIDDEN_LABEL_WZ
        ax.scatter(np.array(q["d_nm"])[~inside], np.array(q["t_ann"])[~inside],
                   s=26, c=TEAL, edgecolors="white", linewidths=0.5, zorder=3)
        ax.scatter(np.array(q["d_nm"])[inside], np.array(q["t_ann"])[inside],
                   s=26, c=RED, edgecolors="white", linewidths=0.5, zorder=3)
        fr = np.mean([r["frac_hidden"][-1] for r in results[pol]])
        col, _, name = POLICY_STYLE[pol]
        ax.set_title(f"{name}\n{fr*100:.0f}% of budget in unlearnable zone")
        ax.set_xlabel("thickness (nm)")
    axes[0].set_ylabel("anneal temperature (°C)")
    fig.suptitle("Where each policy spends the expensive electrical tests "
                 "(projection; red shading = hidden-variable zone)",
                 color=NAVY, fontweight="bold")
    fig.savefig("fig2_sampling.png", dpi=200)
    plt.close(fig)


def _curves(results, key):
    out = {}
    for pol, runs in results.items():
        arr = np.array([r[key] for r in runs])
        out[pol] = (np.array(runs[0]["n"]), arr.mean(0),
                    arr.std(0) / np.sqrt(len(runs)))
    return out


def fig_learning(results):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    for key, ax, title, ylab in [
            ("rmse_learn", axes[0],
             "map accuracy in the learnable region",
             "RMSE of log$_{10}$ J prediction (decades)"),
            ("auc", axes[1],
             "detection of the hidden-variable zone",
             "ROC-AUC (predicted aleatoric std)")]:
        for pol, (n, mu, se) in _curves(results, key).items():
            col, lsty, name = POLICY_STYLE[pol]
            ax.plot(n, mu, lsty, color=col, lw=2.2, label=name)
            ax.fill_between(n, mu - se, mu + se, color=col, alpha=0.15)
        ax.set_xlabel("electrical-test budget (measurements)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[1].axhline(0.5, color=GRAY, lw=1, ls=":")
    axes[1].text(0.02, 0.505, "chance", transform=axes[1].get_yaxis_transform(),
                 color=GRAY, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9.5)
    fig.suptitle("Learning curves — mean ± s.e. over 6 seeded campaigns",
                 color=NAVY, fontweight="bold")
    fig.savefig("fig3_learning.png", dpi=200)
    plt.close(fig)


def fig_slide(results):
    """Composite single-slide figure: truth map | budget use | learning curve."""
    d, t, m = slice_grid()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)

    ax = axes[0]
    im = ax.pcolormesh(d, t, 1.5 * m["w_z"], cmap="Reds", shading="auto")
    ax.contour(d, t, m["w_z"], levels=[HIDDEN_LABEL_WZ], colors=[NAVY],
               linewidths=1.6, linestyles="--")
    ax.set_xlabel("thickness (nm)")
    ax.set_ylabel("anneal temperature (°C)")
    ax.set_title("synthetic landscape:\na zone no fast measurement can resolve")
    fig.colorbar(im, ax=ax, label="irreducible spread (decades)")

    ax = axes[1]
    ax.pcolormesh(d, t, 1.5 * m["w_z"], cmap="Reds", shading="auto",
                  alpha=0.35, vmin=0, vmax=2.2)
    for pol in ["us_total", "epistemic"]:
        col, _, name = POLICY_STYLE[pol]
        q = results[pol][0]["queries"]
        ax.scatter(q["d_nm"], q["t_ann"], s=30, c=col, alpha=0.9,
                   edgecolors="white", linewidths=0.5,
                   label=f"{name.split(' (')[0]}")
    ax.contour(d, t, m["w_z"], levels=[HIDDEN_LABEL_WZ], colors=[NAVY],
               linewidths=1.4, linestyles="--")
    fr_us = np.mean([r["frac_hidden"][-1] for r in results["us_total"]])
    fr_ep = np.mean([r["frac_hidden"][-1] for r in results["epistemic"]])
    ax.set_xlabel("thickness (nm)")
    ax.set_title(f"budget spent in unlearnable zone:\n"
                 f"{fr_us*100:.0f}% (uncert. sampling) vs {fr_ep*100:.0f}% (ours)")
    leg = ax.legend(frameon=True, framealpha=0.9, fontsize=9, loc="lower left", scatterpoints=1, markerscale=1.4)

    ax = axes[2]
    for pol, (n, mu, se) in _curves(results, "rmse_learn").items():
        col, lsty, name = POLICY_STYLE[pol]
        ax.plot(n, mu, lsty, color=col, lw=2.2, label=name)
        ax.fill_between(n, mu - se, mu + se, color=col, alpha=0.15)
    ax.set_xlabel("electrical-test budget")
    ax.set_ylabel("map RMSE (decades of J)")
    ax.set_title("same budget, better map")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.savefig("fig_slide_summary.png", dpi=220)
    plt.close(fig)


def headline(results):
    """Budget multiplier: measurements needed to reach ours' final RMSE."""
    c = _curves(results, "rmse_learn")
    n_ep, mu_ep, _ = c["epistemic"]
    target = mu_ep[-1]
    lines = [f"target RMSE (ours at {int(n_ep[-1])} tests): {target:.3f}"]
    for pol in ["random", "sobol", "us_total"]:
        n, mu, _ = c[pol]
        hit = n[mu <= target]
        lines.append(f"{pol}: reaches target at "
                     + (f"{int(hit[0])} tests" if len(hit) else
                        f">{int(n[-1])} tests (never within budget), "
                        f"final RMSE {mu[-1]:.3f}"))
    au = _curves(results, "auc")
    lines.append("final AUC: " + ", ".join(
        f"{p}={au[p][1][-1]:.2f}" for p in POLICY_STYLE))
    lines.append("final frac_hidden: " + ", ".join(
        f"{p}={np.mean([r['frac_hidden'][-1] for r in results[p]]):.2f}"
        for p in POLICY_STYLE))
    txt = "\n".join(lines)
    print(txt)
    with open("headline.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    with open("results.json") as f:
        results = json.load(f)
    fig_ground_truth()
    fig_sampling(results)
    fig_learning(results)
    fig_slide(results)
    headline(results)
    print("figures written")
