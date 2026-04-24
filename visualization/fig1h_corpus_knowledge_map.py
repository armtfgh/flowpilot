"""
Figure 1H (alternative) — FLORA corpus knowledge map built directly from paper records.

Why this version exists:
- The older knowledge graph depends on an external LLM classification cache.
- It mixes many sparse labels into one force-directed graph, which becomes hard to read.
- This script instead builds a layered concept map directly from extracted records.

Layout:
    Chemistry Class  ->  Reactor Type  ->  Solvent  ->  Reactor Material

Nodes:
    Top concepts in each category, normalized and sorted by paper frequency.

Edges:
    Weighted co-occurrence counts between adjacent columns across papers.

Jupyter usage:
    from fig1h_corpus_knowledge_map import make_figure
    fig = make_figure()
    fig.savefig("fig1h_corpus_knowledge_map.png", dpi=300, bbox_inches="tight")
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from data_loader import (
    RECORDS_DIR,
    REACTION_CLASS_MAP,
    REACTOR_MATERIAL_MAP,
    REACTOR_TYPE_MAP,
    load_records,
)


BG = "#FBFBFA"
TEXT_DARK = "#111827"
TEXT_DIM = "#4B5563"
GRID = "#E5E7EB"
EDGE = "#CBD5E1"

CATEGORY_STYLE = {
    "chemistry": {"label": "Chemistry Class", "color": "#2563EB"},
    "reactor": {"label": "Reactor Type", "color": "#F59E0B"},
    "solvent": {"label": "Solvent", "color": "#10B981"},
    "material": {"label": "Reactor Material", "color": "#EF4444"},
}

SOLVENT_MAP = {
    "DMF": [r"(^|[^a-z])dmf([^a-z]|$)", r"dimethylformamide", r"n,n-dimethylformamide"],
    "DMSO": [r"(^|[^a-z])dmso([^a-z]|$)", r"dimethyl sulfoxide"],
    "MeCN": [r"mecn", r"acetonitrile", r"ch3cn"],
    "MeOH": [r"meoh", r"methanol", r"ch3oh"],
    "EtOH": [r"etoh", r"ethanol"],
    "THF": [r"(^|[^a-z])thf([^a-z]|$)", r"tetrahydrofuran"],
    "DCM": [r"(^|[^a-z])dcm([^a-z]|$)", r"ch2cl2", r"dichloromethane"],
    "Toluene": [r"toluene"],
    "Water": [r"(^|[^a-z])h2o([^a-z]|$)", r"water", r"aqueous"],
    "DMA": [r"(^|[^a-z])dma([^a-z]|$)", r"dimethylacetamide"],
    "Diglyme": [r"diglyme"],
    "Other": [],
}


def _normalize(value: str, mapping: dict[str, list[str]], default: str = "Other") -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return default
    for label, patterns in mapping.items():
        if label == default:
            continue
        for pat in patterns:
            if re.search(pat, raw, re.IGNORECASE):
                return label
    return default


def _get_nested(record: dict, *path: str) -> str:
    cur = record
    for key in path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
    return cur if isinstance(cur, str) else ""


def _extract_paper_features(record: dict) -> dict[str, str]:
    chemistry = _normalize(
        _get_nested(record, "chemistry", "reaction_class"), REACTION_CLASS_MAP
    )
    reactor = _normalize(
        _get_nested(record, "reactor", "type"), REACTOR_TYPE_MAP
    )
    material = _normalize(
        _get_nested(record, "reactor", "material"), REACTOR_MATERIAL_MAP
    )
    solvent_raw = (
        _get_nested(record, "flow_optimized", "solvent")
        or _get_nested(record, "batch_baseline", "solvent")
        or _get_nested(record, "metadata", "solvent")
    )
    solvent = _normalize(solvent_raw, SOLVENT_MAP)
    return {
        "chemistry": chemistry,
        "reactor": reactor,
        "solvent": solvent,
        "material": material,
    }


def _select_top_labels(features: list[dict[str, str]], category: str, top_n: int) -> list[str]:
    counts = Counter(
        f[category] for f in features if f.get(category) and f[category] != "Other / Unknown"
    )
    if category == "solvent":
        counts = Counter(f[category] for f in features if f.get(category) and f[category] != "Other")
    labels = [label for label, _ in counts.most_common(top_n)]
    return labels


def _fold_label(value: str, category: str, allowed: set[str]) -> str | None:
    if not value:
        return None
    other_label = "Other" if category == "solvent" else "Other / Unknown"
    if value in allowed:
        return value
    if value == other_label:
        return None
    return "Other"


def _prepare_graph_data(records: list[dict], top_n: int = 7) -> tuple[dict, dict, int]:
    features = [_extract_paper_features(r) for r in records if isinstance(r, dict)]
    total_papers = len(features)

    top_labels = {
        "chemistry": set(_select_top_labels(features, "chemistry", top_n)),
        "reactor": set(_select_top_labels(features, "reactor", top_n)),
        "solvent": set(_select_top_labels(features, "solvent", top_n)),
        "material": set(_select_top_labels(features, "material", top_n)),
    }

    node_counts: dict[str, Counter] = {cat: Counter() for cat in CATEGORY_STYLE}
    edge_counts: dict[tuple[str, str], Counter] = {
        ("chemistry", "reactor"): Counter(),
        ("reactor", "solvent"): Counter(),
        ("solvent", "material"): Counter(),
    }

    for feat in features:
        folded = {}
        for cat in CATEGORY_STYLE:
            val = _fold_label(feat[cat], cat, top_labels[cat])
            folded[cat] = val
            if val:
                node_counts[cat][val] += 1

        for left, right in edge_counts:
            a, b = folded[left], folded[right]
            if a and b:
                edge_counts[(left, right)][(a, b)] += 1

    return node_counts, edge_counts, total_papers


def _compute_positions(node_counts: dict[str, Counter]) -> dict[tuple[str, str], tuple[float, float]]:
    x_positions = {
        "chemistry": 0.12,
        "reactor": 0.38,
        "solvent": 0.64,
        "material": 0.90,
    }
    pos: dict[tuple[str, str], tuple[float, float]] = {}
    for cat, counter in node_counts.items():
        labels = [label for label, _ in counter.most_common()]
        if not labels:
            continue
        ys = [0.88 - i * (0.76 / max(len(labels) - 1, 1)) for i in range(len(labels))]
        if len(labels) == 1:
            ys = [0.50]
        for label, y in zip(labels, ys):
            pos[(cat, label)] = (x_positions[cat], y)
    return pos


def _draw_edge(
    ax: plt.Axes,
    src: tuple[float, float],
    dst: tuple[float, float],
    color: str,
    weight: int,
    max_weight: int,
) -> None:
    if max_weight <= 0:
        max_weight = 1
    lw = 0.5 + 6.0 * (weight / max_weight)
    alpha = 0.15 + 0.45 * (weight / max_weight)
    rad = 0.10 if dst[0] - src[0] > 0.2 else 0.06
    patch = FancyArrowPatch(
        src,
        dst,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-",
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=1,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(patch)


def make_figure(
    records_dir: Path | None = None,
    top_n_per_category: int = 7,
    min_edge_weight: int = 3,
    figsize: tuple[float, float] = (15, 9),
) -> plt.Figure:
    records = load_records(records_dir or RECORDS_DIR)
    node_counts, edge_counts, total_papers = _prepare_graph_data(records, top_n_per_category)
    pos = _compute_positions(node_counts)

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.02, 0.98)
    ax.axis("off")

    max_node_count = max((count for counter in node_counts.values() for count in counter.values()), default=1)
    max_edge_weight = max(
        (count for counter in edge_counts.values() for count in counter.values()),
        default=1,
    )

    # Draw column guides and titles.
    for cat, cfg in CATEGORY_STYLE.items():
        x = pos[(cat, next(iter(node_counts[cat])))] [0] if node_counts[cat] else 0.5
        ax.text(
            x,
            0.955,
            cfg["label"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=cfg["color"],
        )
        ax.plot([x, x], [0.10, 0.92], color=GRID, lw=1.0, zorder=0)

    # Draw edges first.
    for (left, right), counter in edge_counts.items():
        color = CATEGORY_STYLE[left]["color"]
        for (a, b), weight in counter.items():
            if weight < min_edge_weight:
                continue
            src = pos.get((left, a))
            dst = pos.get((right, b))
            if src and dst:
                _draw_edge(ax, src, dst, color, weight, max_edge_weight)

    # Draw nodes.
    box_w = 0.16
    box_h_base = 0.045
    for cat, counter in node_counts.items():
        color = CATEGORY_STYLE[cat]["color"]
        for label, count in counter.most_common():
            x, y = pos[(cat, label)]
            box_h = box_h_base + 0.010 * (count / max_node_count)
            rect = FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.005,rounding_size=0.012",
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y + 0.007,
                label,
                ha="center",
                va="center",
                fontsize=8.7,
                color=TEXT_DARK,
                fontweight="bold",
                zorder=4,
            )
            ax.text(
                x,
                y - 0.010,
                f"{count} papers",
                ha="center",
                va="center",
                fontsize=7.4,
                color=TEXT_DIM,
                zorder=4,
            )

    title = "FLORA Literature Corpus Knowledge Map"
    subtitle = (
        "Directly built from extracted paper metadata. Nodes show the most common concepts; "
        "edges show how often adjacent concepts co-occur in the same paper."
    )
    ax.text(0.03, 0.985, title, ha="left", va="top", fontsize=16, fontweight="bold", color=TEXT_DARK)
    ax.text(0.03, 0.958, subtitle, ha="left", va="top", fontsize=9.5, color=TEXT_DIM)

    legend_y = 0.055
    ax.text(
        0.03,
        legend_y,
        f"{total_papers} papers loaded  |  top {top_n_per_category} concepts per category  |  edges shown for ≥ {min_edge_weight} co-occurrences",
        ha="left",
        va="center",
        fontsize=8.5,
        color=TEXT_DIM,
    )

    return fig


if __name__ == "__main__":
    fig = make_figure()
    out_png = Path(__file__).resolve().parent.parent / "outputs" / "fig1h_corpus_knowledge_map.png"
    out_svg = Path(__file__).resolve().parent.parent / "outputs" / "fig1h_corpus_knowledge_map.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")
