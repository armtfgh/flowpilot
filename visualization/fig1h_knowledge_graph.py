"""
Figure 1H — Multi-layer knowledge graph of the FLORA literature corpus.

Five node categories:
  • Chemistry Class   (circle,   blue family)
  • Reactor Type      (circle,   orange family)
  • Reactor Material  (diamond,  green family)
  • Bond Type         (circle,   red family)
  • Light Source      (circle,   purple family)

Edges connect features that co-occur in the same paper.
Requires: llm_classifier.classify_all() to have been run first.

Jupyter usage:
    from fig1h_knowledge_graph import make_figure
    fig = make_figure()
    fig.savefig("fig1h.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import numpy as np
from collections import defaultdict, Counter

try:
    import networkx as nx
except ImportError:
    raise ImportError("pip install networkx")

from data_loader import load_records, RECORDS_DIR

# ── Category config ───────────────────────────────────────────────────────────
CATEGORIES = {
    "chemistry_class":   {"color": "#1B6CA8", "label": "Chemistry Class",   "marker": "o"},
    "reactor_type":      {"color": "#E07B39", "label": "Reactor Type",      "marker": "o"},
    "reactor_material":  {"color": "#2A9D5C", "label": "Reactor Material",  "marker": "D"},
    "bond_type":         {"color": "#C0392B", "label": "Bond Type",         "marker": "o"},
    "light_source":      {"color": "#7B2D8B", "label": "Light Source",      "marker": "o"},
}

EDGE_COLORS = {
    ("chemistry_class",  "reactor_type"):     "#C8882A",
    ("chemistry_class",  "reactor_material"): "#1B8A6B",
    ("chemistry_class",  "bond_type"):        "#8B2323",
    ("chemistry_class",  "light_source"):     "#6A1A7A",
    ("reactor_type",     "reactor_material"): "#3A7D3A",
    ("reactor_type",     "bond_type"):        "#A04020",
    ("reactor_type",     "light_source"):     "#5A2070",
    ("reactor_material", "bond_type"):        "#804040",
    ("reactor_material", "light_source"):     "#4A3060",
    ("bond_type",        "light_source"):     "#702060",
}


def _get_edge_color(cat_a: str, cat_b: str) -> str:
    key = tuple(sorted([cat_a, cat_b]))
    return EDGE_COLORS.get(key, "#AAAAAA")


def _get_record_id(record: dict) -> str:
    src = record.get("source_pdf", "")
    return Path(src).stem if src else ""


def _load_paper_features(records: list[dict]) -> list[dict]:
    """Load LLM-classified features per paper."""
    try:
        from llm_classifier import load_classifications
        cache = load_classifications()
    except ImportError:
        cache = {}

    if not cache:
        raise RuntimeError(
            "No LLM classifications found. Run llm_classifier.classify_all() first."
        )

    papers = []
    for r in records:
        rec_id = _get_record_id(r)
        clf = cache.get(rec_id)
        if clf:
            papers.append(clf)
    return papers


def make_figure(
    records_dir=None,
    min_edge_weight: int = 3,
    exclude_other: bool = True,
    figsize: tuple = (14, 11),
    seed: int = 42,
    node_scale: float = 1.0,
) -> plt.Figure:
    """
    Parameters
    ----------
    min_edge_weight : int
        Minimum co-occurrences to draw an edge (lower = more edges).
    exclude_other : bool
        Drop nodes labelled "Other", "Other / Multiple", "Other / Unknown".
    figsize : tuple
        Figure size in inches.
    seed : int
        Random seed for layout reproducibility.
    node_scale : float
        Multiply all node sizes by this factor.
    """
    records = load_records(records_dir) if records_dir else load_records()
    papers = _load_paper_features(records)

    # ── Count node frequencies ─────────────────────────────────────────────────
    node_counts: dict[tuple[str, str], int] = defaultdict(int)
    for p in papers:
        for cat in CATEGORIES:
            val = p.get(cat, "")
            if val:
                node_counts[(cat, val)] += 1

    # ── Count edge weights (co-occurrence across ALL category pairs) ───────────
    edge_weights: dict[tuple, int] = defaultdict(int)
    for p in papers:
        vals = {cat: p.get(cat, "") for cat in CATEGORIES}
        cats = list(CATEGORIES.keys())
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                ca, cb = cats[i], cats[j]
                va, vb = vals[ca], vals[cb]
                if va and vb:
                    key = tuple(sorted([(ca, va), (cb, vb)]))
                    edge_weights[key] += 1

    # ── Build graph ────────────────────────────────────────────────────────────
    G = nx.Graph()

    def _is_other(val: str) -> bool:
        return val.lower().startswith("other") if val else True

    for (cat, val), count in node_counts.items():
        if exclude_other and _is_other(val):
            continue
        G.add_node((cat, val), category=cat, label=val, count=count)

    for ((ca, va), (cb, vb)), w in edge_weights.items():
        if w < min_edge_weight:
            continue
        n_a, n_b = (ca, va), (cb, vb)
        if n_a not in G.nodes or n_b not in G.nodes:
            continue
        G.add_edge(n_a, n_b, weight=w)

    if len(G.nodes) == 0:
        raise ValueError("Graph has no nodes after filtering.")

    print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # ── Layout ─────────────────────────────────────────────────────────────────
    # Use spring layout with higher k for more spread, more iterations for quality
    np.random.seed(seed)
    pos = nx.spring_layout(G, k=2.8, iterations=200, seed=seed, weight="weight")

    # ── Draw ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0F1117")
    ax.set_facecolor("#0F1117")

    # ── Edges ─────────────────────────────────────────────────────────────────
    all_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(all_weights) if all_weights else 1

    for u, v in G.edges():
        w = G[u][v]["weight"]
        cat_u = G.nodes[u]["category"]
        cat_v = G.nodes[v]["category"]
        color = _get_edge_color(cat_u, cat_v)
        lw = 0.4 + 3.5 * (w / max_w)
        alpha = 0.25 + 0.55 * (w / max_w)
        xu, yu = pos[u]
        xv, yv = pos[v]
        ax.plot([xu, xv], [yu, yv], color=color, lw=lw, alpha=alpha, zorder=1, solid_capstyle="round")

    # ── Nodes ─────────────────────────────────────────────────────────────────
    max_count = max(d["count"] for _, d in G.nodes(data=True))

    for node, data in G.nodes(data=True):
        cat = data["category"]
        label = data["label"]
        count = data["count"]
        cfg = CATEGORIES[cat]
        x, y = pos[node]

        size = node_scale * (120 + 700 * (count / max_count))
        marker = cfg["marker"]
        color = cfg["color"]

        ax.scatter(x, y, s=size, c=color, marker=marker, zorder=3,
                   edgecolors="white", linewidths=0.6, alpha=0.92)

        # Label with count badge
        fontsize = 6.5 + 2.5 * (count / max_count)
        txt = ax.text(x, y, label, ha="center", va="center",
                      fontsize=min(fontsize, 8.5), color="white",
                      fontweight="bold", zorder=4, wrap=False)
        txt.set_path_effects([
            pe.withStroke(linewidth=2.5, foreground=color)
        ])

        # Count badge (below node)
        ax.text(x, y - 0.08 - 0.02 * (count / max_count), f"n={count}",
                ha="center", va="top", fontsize=5.5, color="#CCCCCC",
                alpha=0.75, zorder=4)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_handles = []
    for cat, cfg in CATEGORIES.items():
        patch = mpatches.Patch(color=cfg["color"], label=cfg["label"])
        legend_handles.append(patch)

    # Edge weight legend
    for w_pct, label in [(0.2, f"weak (≥{min_edge_weight})"),
                          (0.6, "moderate"),
                          (1.0, f"strong (≥{int(max_w*0.8)})")]:
        lw_ex = 0.4 + 3.5 * w_pct
        legend_handles.append(
            mlines.Line2D([], [], color="#AAAAAA", lw=lw_ex, label=label, alpha=0.8)
        )

    leg = ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        framealpha=0.25,
        facecolor="#1A1D26",
        edgecolor="#555555",
        labelcolor="white",
        title="Node category / Edge weight",
        title_fontsize=8,
    )
    leg.get_title().set_color("white")

    # ── Stats annotation ───────────────────────────────────────────────────────
    n_papers = len(papers)
    ax.text(0.99, 0.01,
            f"{len(G.nodes)} nodes · {len(G.edges)} edges · {n_papers} papers",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#888888", style="italic")

    ax.set_title("FLORA Corpus Knowledge Graph",
                 fontsize=15, fontweight="bold", color="white", pad=16)
    ax.axis("off")
    fig.tight_layout(pad=1.5)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
