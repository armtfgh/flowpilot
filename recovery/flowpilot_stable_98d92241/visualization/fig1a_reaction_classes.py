"""
Figure 1A — Reaction class distribution (top 10, horizontal bars).

Jupyter usage:
    from fig1a_reaction_classes import make_figure
    fig = make_figure()
    fig.savefig("fig1a.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from data_loader import load_records, get_reaction_classes

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#44BBA4", "#E94F37", "#393E41", "#6B4226", "#A8DADC",
]
FONT = "DejaVu Sans"


def make_figure(
    records_dir=None,
    top_n: int = 10,
    figsize: tuple = (6, 4),
    show_count_label: bool = True,
    use_llm_cache: bool = True,
) -> plt.Figure:
    """
    Parameters
    ----------
    records_dir : path-like, optional
        Override default records directory.
    top_n : int
        How many classes to show.
    figsize : tuple
        Figure size in inches.
    show_count_label : bool
        Annotate bars with count numbers.
    use_llm_cache : bool
        Use LLM classifications if available (run llm_classifier.classify_all() first).
    """
    records = load_records(records_dir) if records_dir else load_records()

    if use_llm_cache:
        from data_loader import get_classified_counts
        llm_counts = get_classified_counts("chemistry_class", records)
        counts = llm_counts if llm_counts else get_reaction_classes(records)
    else:
        counts = get_reaction_classes(records)

    # Drop "Other / Unknown" from top-N ranking, keep at bottom separately
    known = {k: v for k, v in counts.items() if k != "Other / Unknown"}
    other = counts.get("Other / Unknown", 0)

    top = sorted(known.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [t[0] for t in top] + ["Other / Unknown"]
    values = [t[1] for t in top] + [other]
    colors = PALETTE[:len(labels) - 1] + ["#CCCCCC"]

    # Reverse for horizontal bar (largest on top)
    labels, values, colors = labels[::-1], values[::-1], colors[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6, height=0.65)

    if show_count_label:
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center", ha="left",
                fontsize=8, color="#444444",
            )

    ax.set_xlabel("Number of Papers", fontsize=10, labelpad=6)
    ax.set_title("Reaction Class Distribution", fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))

    total = sum(counts.values())
    ax.text(
        0.98, 0.02, f"n = {total} papers",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#888888", style="italic",
    )

    fig.tight_layout()
    return fig


# ── Notebook quick-run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig = make_figure()
    plt.show()
