"""
Figure 1G — Yield distribution (flow-optimized conditions).

Shows: histogram of yield %, mean/median lines, and batch vs flow yield comparison
if batch yield data is present.

Jupyter usage:
    from fig1g_yield_distribution import make_figure
    fig = make_figure()
    fig.savefig("fig1g.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_records, get_yields


def _get_batch_yields(records):
    yields = []
    for r in records:
        y = r.get("batch_baseline", {}).get("yield_percent")
        try:
            v = float(y)
            if 0 <= v <= 100:
                yields.append(v)
        except (TypeError, ValueError):
            pass
    return yields


def make_figure(
    records_dir=None,
    bins: int = 20,
    show_batch_comparison: bool = True,
    figsize: tuple = (6, 4),
) -> plt.Figure:
    records = load_records(records_dir) if records_dir else load_records()
    flow_yields = get_yields(records)
    batch_yields = _get_batch_yields(records) if show_batch_comparison else []

    fig, ax = plt.subplots(figsize=figsize)

    # Flow yield histogram
    ax.hist(flow_yields, bins=bins, color="#2E86AB", alpha=0.85,
            edgecolor="white", linewidth=0.5, label=f"Flow (n={len(flow_yields)})", zorder=3)

    if batch_yields:
        ax.hist(batch_yields, bins=bins, color="#F18F01", alpha=0.55,
                edgecolor="white", linewidth=0.5, label=f"Batch (n={len(batch_yields)})", zorder=2)

    # Mean / median lines
    flow_mean = np.mean(flow_yields)
    flow_med = np.median(flow_yields)
    ax.axvline(flow_mean, color="#1a5276", lw=1.5, ls="--",
               label=f"Flow mean: {flow_mean:.0f}%")
    ax.axvline(flow_med, color="#117a65", lw=1.5, ls=":",
               label=f"Flow median: {flow_med:.0f}%")

    ax.set_xlabel("Yield (%)", fontsize=10, labelpad=6)
    ax.set_ylabel("Number of Papers", fontsize=10, labelpad=6)
    ax.set_title("Yield Distribution (Flow-Optimized)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
