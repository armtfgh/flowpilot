"""
Figure 2C — Coverage heatmap: Rule Category × Chemistry Class.

Requires rule_classifier.classify_all_rules() to have been run.
Each cell = number of rules applicable to that chemistry class.

Jupyter usage:
    from fig2c_coverage_heatmap import make_figure
    fig = make_figure()
    fig.savefig("fig2c.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict
from rule_classifier import load_rules, load_rule_classifications, CHEMISTRY_CLASSES
from fig2a_rule_landscape import CATEGORY_RENAME

# Shorten chemistry class labels for axis readability
CHEM_SHORT = {
    "Photoredox Catalysis":          "Photoredox",
    "Heterogeneous Photocatalysis":  "Heterog. Photo.",
    "Photocycloaddition":            "Photocycloadd.",
    "Thermal Synthesis":             "Thermal",
    "Cross-Coupling":                "Cross-Coupling",
    "Hydrogenation":                 "Hydrogenation",
    "Oxidation / Reduction":         "Oxid./Red.",
    "Electrochemistry":              "Electrochemistry",
    "Biocatalysis":                  "Biocatalysis",
    "Polymer Synthesis":             "Polymerization",
    "Organocatalysis":               "Organocatalysis",
    "Precipitation / Crystallization": "Precipitation",
}


def make_figure(figsize: tuple = (11, 7)) -> plt.Figure:
    rules = load_rules()
    cache = load_rule_classifications()

    if not cache:
        raise RuntimeError("Run rule_classifier.classify_all_rules() first.")

    # Build matrix: rows = categories, cols = chemistry classes
    # Determine category order (by total rule count)
    from collections import Counter
    cat_totals = Counter(r["category"] for r in rules)
    cats_ordered = [c for c, _ in cat_totals.most_common()]

    # Filter to non-trivial categories (≥3 rules)
    cats_ordered = [c for c in cats_ordered if cat_totals[c] >= 3]
    chem_ordered = CHEMISTRY_CLASSES

    matrix = np.zeros((len(cats_ordered), len(chem_ordered)), dtype=int)

    for rule in rules:
        clf = cache.get(rule["rule_id"], {})
        applicable = clf.get("applicable_chemistry_classes", [])
        cat = rule["category"]
        if cat not in cats_ordered:
            continue
        r_idx = cats_ordered.index(cat)
        for chem in applicable:
            if chem in chem_ordered:
                c_idx = chem_ordered.index(chem)
                matrix[r_idx, c_idx] += 1

    row_labels = [CATEGORY_RENAME.get(c, c.replace("_"," ").title()) for c in cats_ordered]
    col_labels = [CHEM_SHORT.get(c, c) for c in chem_ordered]

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")

    # Custom colormap: white → teal → deep navy
    colors_list = ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    cmap = mcolors.LinearSegmentedColormap.from_list("flora_blue", colors_list)

    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=0, vmax=matrix.max())

    # Cell annotations
    for i in range(len(cats_ordered)):
        for j in range(len(chem_ordered)):
            val = matrix[i, j]
            if val == 0:
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=8, color="#CCCCCC")
            else:
                text_color = "white" if val > matrix.max() * 0.55 else "#1A1A2E"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    # Axes
    ax.set_xticks(range(len(chem_ordered)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right",
                       fontsize=9.5, color="#222222")
    ax.set_yticks(range(len(cats_ordered)))
    ax.set_yticklabels(row_labels, fontsize=9.5, color="#222222")

    # Grid lines between cells
    for x in np.arange(-0.5, len(chem_ordered), 1):
        ax.axvline(x, color="white", linewidth=1.2)
    for y in np.arange(-0.5, len(cats_ordered), 1):
        ax.axhline(y, color="white", linewidth=1.2)

    ax.set_title("Fundamentals Coverage: Rule Category × Chemistry Class",
                 fontsize=13, fontweight="bold", color="#111111", pad=16)
    ax.set_xlabel("Chemistry Class", fontsize=11, labelpad=10, color="#222222")
    ax.set_ylabel("Rule Category", fontsize=11, labelpad=10, color="#222222")
    ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02, aspect=25)
    cbar.set_label("Applicable Rules", fontsize=9.5, color="#444444", labelpad=8)
    cbar.ax.tick_params(labelsize=8.5, colors="#555555")
    cbar.outline.set_edgecolor("#CCCCCC")

    # Row totals (bar on right)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    row_totals = matrix.sum(axis=1)
    for i, tot in enumerate(row_totals):
        ax2.text(len(chem_ordered) + 0.3, i,
                 f"Σ={tot}", va="center", ha="left",
                 fontsize=7.5, color="#888888")
    ax2.set_yticks([])
    ax2.spines[:].set_visible(False)

    fig.tight_layout(pad=1.5)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
