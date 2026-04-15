"""
Figure 2A — Rule landscape: stacked horizontal bars showing
rule count per category, split by severity (hard_rule / guideline / tip).

Jupyter usage:
    from fig2a_rule_landscape import make_figure
    fig = make_figure()
    fig.savefig("fig2a.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
from rule_classifier import load_rules

# ── Palette ───────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "hard_rule":  "#1B4F72",   # deep navy
    "guideline":  "#2E86C1",   # medium blue
    "tip":        "#AED6F1",   # light blue
    "safety":     "#C0392B",   # red (distinct)
}
SEVERITY_LABELS = {
    "hard_rule": "Hard Rule",
    "guideline": "Guideline",
    "tip":       "Tip",
    "safety":    "Safety",
}

CATEGORY_RENAME = {
    "residence_time":    "Residence Time",
    "reactor_design":    "Reactor Design",
    "heat_transfer":     "Heat Transfer",
    "mass_transfer":     "Mass Transfer",
    "mixing":            "Mixing",
    "pressure":          "Pressure",
    "safety":            "Safety",
    "materials":         "Materials",
    "catalyst":          "Catalyst",
    "scale_up":          "Scale-Up",
    "solvent":           "Solvent",
    "photochemistry":    "Photochemistry",
    "general":           "General",
    "temperature":       "Temperature",
    "coupling_reactions":"Coupling Reactions",
    "concentration":     "Concentration",
    "carbene_transfer":  "Carbene Transfer",
}


def make_figure(figsize: tuple = (8, 6)) -> plt.Figure:
    rules = load_rules()

    # Count per (category, severity)
    counts = defaultdict(lambda: defaultdict(int))
    for r in rules:
        counts[r["category"]][r["severity"]] += 1

    # Sort categories by total rules descending
    cats_sorted = sorted(counts.keys(), key=lambda c: sum(counts[c].values()), reverse=True)
    labels = [CATEGORY_RENAME.get(c, c.replace("_", " ").title()) for c in cats_sorted]
    severities = ["hard_rule", "guideline", "tip", "safety"]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")

    # Draw stacked bars
    lefts = np.zeros(len(cats_sorted))
    bar_handles = []
    for sev in severities:
        vals = np.array([counts[c].get(sev, 0) for c in cats_sorted])
        if vals.sum() == 0:
            continue
        bars = ax.barh(
            labels, vals, left=lefts,
            color=SEVERITY_COLORS[sev],
            edgecolor="white", linewidth=0.5,
            height=0.68, label=SEVERITY_LABELS[sev],
        )
        lefts += vals
        bar_handles.append(
            mpatches.Patch(color=SEVERITY_COLORS[sev], label=SEVERITY_LABELS[sev])
        )

    # Total count labels at end of each bar
    totals = [sum(counts[c].values()) for c in cats_sorted]
    for i, (total, label) in enumerate(zip(totals, labels)):
        ax.text(total + 1.5, i, str(total),
                va="center", ha="left", fontsize=8, color="#333333", fontweight="bold")

    # Styling
    ax.set_xlabel("Number of Rules", fontsize=11, labelpad=8, color="#222222")
    ax.set_title("Engineering Knowledge Base — Rule Landscape",
                 fontsize=13, fontweight="bold", color="#111111", pad=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="y", labelsize=9.5, colors="#333333", length=0)
    ax.tick_params(axis="x", labelsize=9, colors="#666666")
    ax.set_xlim(0, max(totals) * 1.12)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7, zorder=0)

    # Subtle alternating row backgrounds
    for i in range(len(cats_sorted)):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.45, color="#F0F4F8", zorder=0, alpha=0.6)

    # Legend
    legend = ax.legend(
        handles=bar_handles[::-1],
        loc="lower right", fontsize=9,
        framealpha=0.9, edgecolor="#DDDDDD",
        title="Severity", title_fontsize=9,
    )

    # Annotation: total rules
    total_rules = len(rules)
    ax.text(0.98, 0.98, f"Total: {total_rules} rules",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#888888", style="italic")

    fig.tight_layout(pad=1.5)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
