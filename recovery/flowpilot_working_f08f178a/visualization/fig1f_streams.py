"""
Figure 1F — Number of inlet streams per reactor (process complexity indicator).

Jupyter usage:
    from fig1f_streams import make_figure
    fig = make_figure()
    fig.savefig("fig1f.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_records, get_pump_inlets


def make_figure(records_dir=None, figsize: tuple = (5, 3.5)) -> plt.Figure:
    records = load_records(records_dir) if records_dir else load_records()
    counts = get_pump_inlets(records)

    if not counts:
        raise ValueError("No pump inlet data found in records.")

    max_inlets = max(counts.keys())
    xs = list(range(1, max_inlets + 1))
    ys = [counts.get(x, 0) for x in xs]

    # Color gradient: complexity increases with more inlets
    cmap = plt.cm.get_cmap("Blues")
    norm_vals = [(x - 1) / max(max_inlets - 1, 1) for x in xs]
    colors = [cmap(0.35 + 0.6 * n) for n in norm_vals]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(xs, ys, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

    for bar, val in zip(bars, ys):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ys) * 0.01,
                    str(val), ha="center", va="bottom", fontsize=8.5, color="#444444")

    ax.set_xlabel("Number of Inlet Streams", fontsize=10, labelpad=6)
    ax.set_ylabel("Number of Papers", fontsize=10, labelpad=6)
    ax.set_title("Process Complexity — Inlet Streams", fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(ys) * 1.15)

    total = sum(ys)
    ax.text(0.98, 0.97, f"n = {total} papers", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#888888", style="italic")

    # Annotation arrow for most common
    most_common = xs[ys.index(max(ys))]
    ax.annotate(
        f"Most common:\n{most_common} streams",
        xy=(most_common, max(ys)),
        xytext=(most_common + 0.8, max(ys) * 0.85),
        fontsize=7.5, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
