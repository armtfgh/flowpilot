"""
Figure 3A — FlowPilot 3-Tier RAG vs Standard RAG architecture comparison.

Pure matplotlib illustration — no data or API calls needed.

Jupyter usage:
    from fig3a_rag_architecture import make_figure
    fig = make_figure()
    fig.savefig("fig3a.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette ─────────────────────────────────────────────────────────────
BG          = "#FFFFFF"
NEUTRAL     = "#D0D7DE"
TEXT_LIGHT  = "#111827"
TEXT_DIM    = "#4B5563"

C_INPUT     = "#F3F4F6"   # input/output boxes
C_STD       = "#EAF2FA"   # standard RAG boxes
C_TIER1     = "#EAF7EF"   # tier 1 — query enrichment (green)
C_TIER2     = "#EAF2FF"   # tier 2 — filtered retrieval (blue)
C_TIER3     = "#FFF1E8"   # tier 3 — field reranking (orange)
C_OUTPUT    = "#F3ECFF"   # output box (purple)

TIER_COLORS = {
    "Tier 1\nQuery Enrichment":   ("#2EA04326", "#3FB950"),
    "Tier 2\nFiltered Retrieval": ("#388BFD26", "#58A6FF"),
    "Tier 3\nField Reranking":    ("#F7853126", "#F78166"),
}


def _box(ax, x, y, w, h, label, sublabel="", color=C_STD,
         text_color=TEXT_LIGHT, fontsize=8.5, radius=0.012):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=color, edgecolor=NEUTRAL,
        linewidth=0.8, zorder=3,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + h * 0.12, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold",
                zorder=4, linespacing=1.3)
        ax.text(x, y - h * 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=TEXT_DIM, zorder=4,
                style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold",
                zorder=4, linespacing=1.3)


def _arrow(ax, x1, y1, x2, y2, color="#555C64", lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10),
                zorder=2)


def _tier_bg(ax, x, y_top, y_bot, w, label, border_color, bg_color):
    """Draw a transparent tier background rectangle with label on left."""
    pad = 0.012
    rect = FancyBboxPatch(
        (x - w / 2 - pad, y_bot - pad),
        w + 2 * pad, (y_top - y_bot) + 2 * pad,
        boxstyle="round,pad=0.008",
        facecolor=bg_color, edgecolor=border_color,
        linewidth=1.0, linestyle="--", zorder=1, alpha=0.55,
    )
    ax.add_patch(rect)
    ax.text(x - w / 2 - pad - 0.008, (y_top + y_bot) / 2, label,
            ha="right", va="center", fontsize=7, color=border_color,
            fontweight="bold", rotation=90, zorder=4)


def make_figure(figsize=(13, 8)) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor(BG)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")

    # ── LEFT: Standard RAG ────────────────────────────────────────────────────
    ax = axes[0]
    ax.text(0, 1.02, "Standard RAG", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT_LIGHT)
    ax.text(0, 0.97, "(Semantic Search Only)", ha="center", va="bottom",
            fontsize=8.5, color=TEXT_DIM, style="italic")

    std_boxes = [
        (0, 0.88, "Batch Protocol",       "",                         C_INPUT),
        (0, 0.73, "LLM Query Summary",    "Generic text description", C_STD),
        (0, 0.57, "Text Embedding",       "text-embedding-3-small",   C_STD),
        (0, 0.41, "Cosine Similarity",    "ChromaDB vector search",   C_STD),
        (0, 0.25, "Sort by Semantic\nScore Only", "",                 C_STD),
        (0, 0.10, "Return Top-K\nAnalogies",  "",                     C_OUTPUT),
    ]
    BW, BH = 0.42, 0.075
    for (x, y, lbl, sub, col) in std_boxes:
        _box(ax, x, y, BW, BH, lbl, sub, color=col)
    for i in range(len(std_boxes) - 1):
        _arrow(ax, std_boxes[i][0], std_boxes[i][1] - BH / 2,
               std_boxes[i + 1][0], std_boxes[i + 1][1] + BH / 2)

    # ── RIGHT: FlowPilot 3-Tier RAG ───────────────────────────────────────────
    ax = axes[1]
    ax.text(0, 1.02, "FlowPilot — 3-Tier RAG", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT_LIGHT)
    ax.text(0, 0.97, "(Chemistry-Aware Retrieval)", ha="center", va="bottom",
            fontsize=8.5, color=TEXT_DIM, style="italic")

    BW2 = 0.44

    # Input
    _box(ax, 0, 0.93, BW2, 0.065, "Batch Protocol", "", C_INPUT)

    # ── Tier 1: Query Enrichment ──────────────────────────────────────────────
    _tier_bg(ax, 0, 0.875, 0.685, BW2 + 0.04,
             "Tier 1", "#3FB950", "#2EA04310")
    _box(ax, 0, 0.845, BW2, 0.065,
         "Chemistry Reasoning Agent",
         "Mechanism · Photocatalyst · Intermediates", C_TIER1)
    _box(ax, 0, 0.755, BW2, 0.065,
         "Plan-Aware Rich Query",
         "mechanism + photocatalyst + wavelength + keywords + similar classes",
         C_TIER1)
    _box(ax, 0, 0.695, BW2, 0.052,
         "Text Embedding",
         "text-embedding-3-small", C_TIER1)
    _arrow(ax, 0, 0.93 - 0.065 / 2, 0, 0.845 + 0.065 / 2)
    _arrow(ax, 0, 0.845 - 0.065 / 2, 0, 0.755 + 0.065 / 2)
    _arrow(ax, 0, 0.755 - 0.065 / 2, 0, 0.695 + 0.052 / 2)

    # ── Tier 2: 3-step Filtered Retrieval ─────────────────────────────────────
    _tier_bg(ax, 0, 0.655, 0.435, BW2 + 0.04,
             "Tier 2", "#58A6FF", "#388BFD10")

    step_y = [0.620, 0.535, 0.455]
    step_labels = [
        ("Step 1: Hard Metadata Filter",
         "mechanism_type + phase_regime  ·  pairs only"),
        ("Step 2: Relax Filters",
         "pairs only  ·  (if Step 1 < 3 results)"),
        ("Step 3: No Filters",
         "all records  ·  (last resort fallback)"),
    ]
    for i, (y, (lbl, sub)) in enumerate(zip(step_y, step_labels)):
        _box(ax, 0, y, BW2, 0.062, lbl, sub, C_TIER2)
        if i < 2:
            # fallback arrow on side
            ax.annotate("", xy=(-BW2 / 2 - 0.01, step_y[i + 1] + 0.031),
                        xytext=(-BW2 / 2 - 0.01, y - 0.031),
                        arrowprops=dict(arrowstyle="-|>", color="#388BFD",
                                        lw=0.9, mutation_scale=8,
                                        connectionstyle="arc3,rad=0.0"),
                        zorder=2)
            ax.text(-BW2 / 2 - 0.055, (y + step_y[i + 1]) / 2,
                    "< 3 hits", ha="center", va="center",
                    fontsize=6.5, color="#58A6FF", style="italic")

    _arrow(ax, 0, 0.695 - 0.052 / 2, 0, step_y[0] + 0.062 / 2)

    # ── Tier 3: Field Reranking ───────────────────────────────────────────────
    _tier_bg(ax, 0, 0.395, 0.215, BW2 + 0.04,
             "Tier 3", "#F78166", "#F7853110")

    _box(ax, 0, 0.360, BW2, 0.065,
         "Field Similarity Scoring",
         "photocatalyst class · solvent · wavelength · temperature · concentration",
         C_TIER3)
    _box(ax, 0, 0.275, BW2, 0.062,
         "Weighted Final Score",
         "0.6 × semantic  +  0.4 × field", C_TIER3)

    _arrow(ax, 0, step_y[-1] - 0.062 / 2, 0, 0.360 + 0.065 / 2)
    _arrow(ax, 0, 0.360 - 0.065 / 2, 0, 0.275 + 0.062 / 2)

    # Output
    _box(ax, 0, 0.185, BW2, 0.062,
         "Return Top-K Analogies", "", C_OUTPUT)
    _arrow(ax, 0, 0.275 - 0.062 / 2, 0, 0.185 + 0.062 / 2)

    # ── Tier legend (right side) ───────────────────────────────────────────────
    tier_legend = [
        ("Tier 1: Query Enrichment",   "#3FB950"),
        ("Tier 2: Filtered Retrieval", "#58A6FF"),
        ("Tier 3: Field Reranking",    "#F78166"),
    ]
    for i, (lbl, col) in enumerate(tier_legend):
        y_l = 0.14 - i * 0.045
        axes[1].add_patch(mpatches.Rectangle(
            (0.27, y_l - 0.012), 0.022, 0.024,
            facecolor=col, alpha=0.5, transform=axes[1].transData, zorder=5,
        ))
        axes[1].text(0.30, y_l, lbl, va="center", ha="left",
                     fontsize=7, color=col, zorder=5)

    # ── Divider line ──────────────────────────────────────────────────────────
    fig.add_artist(plt.Line2D(
        [0.505, 0.505], [0.06, 0.96],
        transform=fig.transFigure,
        color=NEUTRAL, lw=1.0, linestyle="--",
    ))

    fig.suptitle("Retrieval Architecture: Standard RAG vs FlowPilot 3-Tier RAG",
                 fontsize=13, fontweight="bold", color=TEXT_LIGHT, y=1.01)
    fig.tight_layout(pad=1.2)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
