"""
Figure 2D — Engineering concept network.

Nodes  = engineering concepts/dimensionless numbers extracted by LLM.
         Sized by frequency (how many rules mention them).
         Colored by their most associated rule category.
Edges  = two concepts appear together in the same rule.
         Thickness ∝ co-occurrence frequency.

Requires rule_classifier.classify_all_rules() to have been run.

Jupyter usage:
    from fig2d_concept_network import make_figure
    fig = make_figure()
    fig.savefig("fig2d.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from collections import defaultdict, Counter

try:
    import networkx as nx
except ImportError:
    raise ImportError("pip install networkx")

from rule_classifier import load_rules, load_rule_classifications
from fig2a_rule_landscape import CATEGORY_RENAME

# Category → color mapping (match fig2a palette family)
CATEGORY_COLORS = {
    "residence_time":    "#1A5276",
    "reactor_design":    "#2874A6",
    "heat_transfer":     "#E67E22",
    "mass_transfer":     "#F39C12",
    "mixing":            "#8E44AD",
    "pressure":          "#C0392B",
    "safety":            "#E74C3C",
    "materials":         "#27AE60",
    "catalyst":          "#1E8449",
    "scale_up":          "#117A65",
    "solvent":           "#48C9B0",
    "photochemistry":    "#7D3C98",
    "general":           "#717D7E",
    "temperature":       "#D35400",
    "coupling_reactions":"#2E86C1",
    "concentration":     "#1ABC9C",
    "carbene_transfer":  "#F1C40F",
}
DEFAULT_COLOR = "#95A5A6"


def make_figure(
    min_node_freq: int = 4,
    min_edge_weight: int = 3,
    figsize: tuple = (13, 10),
    seed: int = 7,
) -> plt.Figure:
    """
    Parameters
    ----------
    min_node_freq   : Minimum times a concept must appear to be shown.
    min_edge_weight : Minimum co-occurrences for an edge.
    figsize         : Figure size.
    seed            : Layout seed for reproducibility.
    """
    rules = load_rules()
    cache = load_rule_classifications()

    if not cache:
        raise RuntimeError("Run rule_classifier.classify_all_rules() first.")

    # ── Collect concepts per rule ──────────────────────────────────────────────
    concept_freq   = Counter()
    concept_to_cat = defaultdict(Counter)  # concept → {category: count}
    edge_weights   = defaultdict(int)

    for rule in rules:
        clf      = cache.get(rule["rule_id"], {})
        concepts = clf.get("key_concepts", [])
        cat      = rule.get("category", "general")

        # Normalize concept strings (lowercase strip)
        concepts = [c.strip() for c in concepts if c.strip()]

        for c in concepts:
            concept_freq[c] += 1
            concept_to_cat[c][cat] += 1

        # Edges: all pairs within same rule
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                key = tuple(sorted([concepts[i], concepts[j]]))
                edge_weights[key] += 1

    # ── Filter & build graph ───────────────────────────────────────────────────
    valid_nodes = {c for c, f in concept_freq.items() if f >= min_node_freq}

    G = nx.Graph()
    for c in valid_nodes:
        dominant_cat = concept_to_cat[c].most_common(1)[0][0]
        G.add_node(c, freq=concept_freq[c], category=dominant_cat)

    for (a, b), w in edge_weights.items():
        if a in valid_nodes and b in valid_nodes and w >= min_edge_weight:
            G.add_edge(a, b, weight=w)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # ── Layout ─────────────────────────────────────────────────────────────────
    np.random.seed(seed)
    pos = nx.spring_layout(G, k=2.2, iterations=250, seed=seed, weight="weight")

    # ── Draw ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    all_w = [G[u][v]["weight"] for u, v in G.edges()] or [1]
    max_w = max(all_w)
    max_freq = max(G.nodes[n]["freq"] for n in G.nodes())

    # ── Edges ─────────────────────────────────────────────────────────────────
    for u, v in G.edges():
        w = G[u][v]["weight"]
        # Color edges by dominant category of source node
        cat_u = G.nodes[u]["category"]
        edge_color = CATEGORY_COLORS.get(cat_u, DEFAULT_COLOR)
        lw    = 0.4 + 4.0 * (w / max_w)
        alpha = 0.2 + 0.6 * (w / max_w)
        xu, yu = pos[u]; xv, yv = pos[v]
        ax.plot([xu, xv], [yu, yv], color=edge_color, lw=lw,
                alpha=alpha, zorder=1, solid_capstyle="round")

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for node in G.nodes():
        data  = G.nodes[node]
        freq  = data["freq"]
        cat   = data["category"]
        color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        x, y  = pos[node]

        size  = 80 + 1200 * (freq / max_freq) ** 0.6
        ax.scatter(x, y, s=size, c=color, zorder=3,
                   edgecolors="white", linewidths=0.7, alpha=0.93)

        # Shorter label if long
        label = node if len(node) <= 22 else node[:20] + "…"
        fs    = 6.5 + 3.5 * (freq / max_freq) ** 0.5
        txt   = ax.text(x, y, label, ha="center", va="center",
                        fontsize=min(fs, 9.5), color="white",
                        fontweight="bold", zorder=4)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground=color)])

        # Frequency badge below node
        ax.text(x, y - 0.06 - 0.03 * (freq / max_freq), f"{freq}×",
                ha="center", va="top", fontsize=5.5,
                color="#AAAAAA", alpha=0.8, zorder=4)

    # ── Legend ─────────────────────────────────────────────────────────────────
    seen_cats = {G.nodes[n]["category"] for n in G.nodes()}
    handles   = [
        mpatches.Patch(
            color=CATEGORY_COLORS.get(c, DEFAULT_COLOR),
            label=CATEGORY_RENAME.get(c, c.replace("_", " ").title())
        )
        for c in sorted(seen_cats)
    ]
    leg = ax.legend(
        handles=handles, loc="lower left",
        fontsize=7.5, ncol=2,
        framealpha=0.2, facecolor="#1C2128",
        edgecolor="#444444", labelcolor="white",
        title="Rule Category", title_fontsize=8,
    )
    leg.get_title().set_color("#CCCCCC")

    # Stats
    ax.text(0.99, 0.01,
            f"{len(G.nodes)} concepts · {len(G.edges)} connections · {len(rules)} rules",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#666666", style="italic")

    ax.set_title("Engineering Concept Network — FLORA Knowledge Base",
                 fontsize=14, fontweight="bold", color="white", pad=16)
    ax.axis("off")
    fig.tight_layout(pad=1.5)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
