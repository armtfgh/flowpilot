"""
Figure 1D — Bond types formed (synthetic scope).

Jupyter usage:
    from fig1d_bond_types import make_figure
    fig = make_figure()
    fig.savefig("fig1d.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from data_loader import load_records, get_bond_types

PALETTE = [
    "#3D405B", "#81B29A", "#F2CC8F", "#E07A5F", "#118AB2",
    "#06D6A0", "#FFD166", "#CCCCCC",
]


def make_figure(records_dir=None, figsize: tuple = (5, 4)) -> plt.Figure:
    records = load_records(records_dir) if records_dir else load_records()
    from data_loader import get_classified_counts
    llm_counts = get_classified_counts("bond_type", records)
    counts = llm_counts if llm_counts else get_bond_types(records)

    known = {k: v for k, v in counts.items() if k != "Other / Unknown"}
    other = counts.get("Other / Unknown", 0)
    top = sorted(known.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in top] + ["Other / Unknown"]
    values = [t[1] for t in top] + [other]
    colors = PALETTE[:len(labels) - 1] + ["#DDDDDD"]

    labels, values, colors = labels[::-1], values[::-1], colors[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=8, color="#444444")

    ax.set_xlabel("Number of Papers", fontsize=10, labelpad=6)
    ax.set_title("Bond Types Formed", fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, max(values) * 1.15)

    total = sum(counts.values())
    ax.text(0.98, 0.02, f"n = {total} papers", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#888888", style="italic")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
