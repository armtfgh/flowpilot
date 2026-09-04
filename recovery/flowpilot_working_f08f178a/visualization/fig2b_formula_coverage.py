"""
Figure 2B — Quantitative formula coverage per rule category.

Shows % of rules per category that contain a quantitative formula,
rendered as a lollipop chart (clean, publication-quality).

Jupyter usage:
    from fig2b_formula_coverage import make_figure
    fig = make_figure()
    fig.savefig("fig2b.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict
from rule_classifier import load_rules
from fig2a_rule_landscape import CATEGORY_RENAME


def make_figure(figsize: tuple = (7, 5.5)) -> plt.Figure:
    rules = load_rules()

    total_per_cat   = defaultdict(int)
    formula_per_cat = defaultdict(int)
    for r in rules:
        cat = r["category"]
        total_per_cat[cat] += 1
        if r.get("quantitative", "").strip():
            formula_per_cat[cat] += 1

    # Sort by % descending
    cats = sorted(total_per_cat.keys(),
                  key=lambda c: formula_per_cat[c] / total_per_cat[c],
                  reverse=True)
    labels  = [CATEGORY_RENAME.get(c, c.replace("_", " ").title()) for c in cats]
    pct     = [100 * formula_per_cat[c] / total_per_cat[c] for c in cats]
    totals  = [total_per_cat[c] for c in cats]

    # Color by percentage (green → blue gradient)
    cmap   = plt.get_cmap("RdYlGn")
    colors = [cmap(p / 100) for p in pct]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")

    y_pos = np.arange(len(cats))

    # Lollipop stems
    ax.hlines(y_pos, 0, pct, color="#DDDDDD", linewidth=1.8, zorder=2)

    # Dots
    scatter = ax.scatter(pct, y_pos, c=pct, cmap="RdYlGn", s=120,
                         vmin=0, vmax=100, zorder=3, edgecolors="white", linewidths=0.8)

    # Percentage labels
    for i, (p, n) in enumerate(zip(pct, totals)):
        ax.text(p + 0.8, i, f"{p:.0f}%  (n={n})",
                va="center", ha="left", fontsize=8.2, color="#333333")

    # 100% reference line
    ax.axvline(100, color="#BBBBBB", lw=1.2, ls="--", alpha=0.7, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Rules with Quantitative Formula (%)", fontsize=11, labelpad=8, color="#222222")
    ax.set_title("Equation Coverage by Knowledge Category",
                 fontsize=13, fontweight="bold", color="#111111", pad=14)
    ax.set_xlim(0, 118)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="y", length=0, colors="#333333")
    ax.tick_params(axis="x", colors="#666666", labelsize=9)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.7, zorder=0)

    # Alternating rows
    for i in range(len(cats)):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.45, color="#F0F4F8", zorder=0, alpha=0.55)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01, aspect=20)
    cbar.set_label("Coverage %", fontsize=8.5, color="#555555")
    cbar.ax.tick_params(labelsize=7.5, colors="#555555")
    cbar.outline.set_edgecolor("#CCCCCC")

    fig.tight_layout(pad=1.5)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
