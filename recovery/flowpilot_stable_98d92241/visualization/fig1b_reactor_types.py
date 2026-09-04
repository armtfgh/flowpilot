"""
Figure 1B — Reactor type distribution (horizontal bars).

Jupyter usage:
    from fig1b_reactor_types import make_figure
    fig = make_figure()
    fig.savefig("fig1b.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
from data_loader import load_records, get_reactor_types

PALETTE = [
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
    "#457B9D", "#A8DADC", "#1D3557", "#CCCCCC",
]


def make_figure(records_dir=None, top_n: int = 8, figsize: tuple = (6, 4), use_llm_cache: bool = True) -> plt.Figure:
    records = load_records(records_dir) if records_dir else load_records()
    if use_llm_cache:
        from data_loader import get_classified_counts
        llm_counts = get_classified_counts("reactor_type", records)
        counts = llm_counts if llm_counts else get_reactor_types(records)
    else:
        counts = get_reactor_types(records)

    known = {k: v for k, v in counts.items() if k != "Other / Unknown"}
    other = counts.get("Other / Unknown", 0)
    top = sorted(known.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [t[0] for t in top] + ["Other / Unknown"]
    values = [t[1] for t in top] + [other]
    colors = PALETTE[:len(labels) - 1] + ["#CCCCCC"]

    labels, values, colors = labels[::-1], values[::-1], colors[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6, height=0.65)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=8, color="#444444",
        )

    ax.set_xlabel("Number of Papers", fontsize=10, labelpad=6)
    ax.set_title("Reactor Type Distribution", fontsize=11, fontweight="bold", pad=10)
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
