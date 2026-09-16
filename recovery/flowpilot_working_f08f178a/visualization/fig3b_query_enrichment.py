"""
Figure 3B — Query enrichment: naive text vs FlowPilot plan-aware query.

Finds real records with rich field coverage from ChromaDB, then reconstructs
both query types and visualizes:
  - Top   : organized example cards for naive vs FlowPilot query construction
  - Bottom: bar chart of information richness (# meaningful chemical terms)

No API calls needed — uses existing ChromaDB metadata.

Jupyter usage:
    from fig3b_query_enrichment import make_figure
    fig = make_figure()
    fig.savefig("fig3b.png", dpi=300, bbox_inches="tight")
"""

import sys
import re
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

BG = "#FFFFFF"
TEXT_LIGHT = "#111827"
TEXT_DIM = "#4B5563"
TEXT_SOFT = "#374151"
PANEL_BG = "#F8FAFC"
PANEL_EDGE = "#D0D7DE"
SOFT_BG = "#F3F4F6"
GRID = "#E5E7EB"

# Term type highlight colors
TERM_COLORS = {
    "mechanism": "#3FB950",
    "photocatalyst": "#F78166",
    "wavelength": "#79C0FF",
    "solvent": "#D2A8FF",
    "temperature": "#FFA657",
    "keyword": "#56D364",
    "reaction": "#58A6FF",
    "bond": "#FF7B72",
    "intermediate": "#E3B341",
}


def _load_rich_records(n=3):
    """Pull n records from ChromaDB with the richest field coverage."""
    from flora_translate.vector_store import VectorStore

    store = VectorStore()
    result = store.collection.get(include=["metadatas", "documents"])
    ids = result["ids"]
    metas = result["metadatas"]
    docs = result["documents"]

    def richness(meta):
        return sum([
            bool(meta.get("photocatalyst", "").strip()),
            bool(meta.get("wavelength_nm") and meta["wavelength_nm"] > 0),
            bool(meta.get("mechanism_type", "").strip()),
            bool(meta.get("solvent", "").strip()),
            bool(meta.get("chemistry_class", "").strip()),
            bool(meta.get("reactor_type", "").strip()),
        ])

    scored = sorted(zip(ids, metas, docs), key=lambda item: richness(item[1]), reverse=True)
    return scored[:n]


def _build_naive_query(meta: dict, doc: str) -> str:
    """Simulate a naive query: just reaction name + class from metadata."""
    parts = []
    cls = meta.get("chemistry_class", "").strip()
    mech = meta.get("mechanism_type", "").strip()
    if cls:
        parts.append(f"{cls} reaction.")
    if mech:
        parts.append(f"Mechanism: {mech}.")
    first_sent = doc.split(".")[0].strip() + "." if doc else ""
    if first_sent:
        parts.append(first_sent)
    return " ".join(parts) if parts else "Flow chemistry reaction."


def _build_flora_query(meta: dict, doc: str) -> list[tuple[str, str | None]]:
    """
    Build FlowPilot-style enriched query as a list of (text_fragment, term_type).
    term_type maps to TERM_COLORS, or None for plain text.
    """
    fragments = []

    cls = meta.get("chemistry_class", "").strip()
    mech = meta.get("mechanism_type", "").strip()
    pc = meta.get("photocatalyst", "").strip()
    wl = meta.get("wavelength_nm")
    sol = meta.get("solvent", "").strip()
    rt = meta.get("reactor_type", "").strip()

    if cls:
        fragments.append((f"Reaction class: {cls}. ", "reaction"))
    if mech:
        fragments.append((f"Mechanism: {mech}. ", "mechanism"))
    if pc:
        fragments.append((f"Photocatalyst: {pc}. ", "photocatalyst"))
    if wl and wl > 0:
        fragments.append((f"Wavelength: {wl:.0f} nm. ", "wavelength"))
    if sol:
        fragments.append((f"Solvent: {sol}. ", "solvent"))
    if rt:
        fragments.append((f"Reactor: {rt}. ", "reaction"))

    sentences = [s.strip() for s in doc.split(".") if len(s.strip()) > 15]
    for sentence in sentences[:2]:
        fragments.append((sentence + ". ", None))

    return fragments


def _count_info_terms(query: str) -> int:
    """Count distinct chemically meaningful terms in a query string."""
    keywords = [
        r"\b(photocatalysis|photoredox|mechanism|photocatalyst|sensitizer|"
        r"wavelength|nm|solvent|temperature|°C|reaction|class|bond|intermediate|"
        r"residence|flow rate|concentration|reactor|coil|microreactor|LED|UV|"
        r"radical|ionic|thermal|electrochemical|biocatalytic|singlet|triplet)\b"
    ]
    matches = set(re.findall(keywords[0], query, re.IGNORECASE))
    return len(matches)


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _wrap_text(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        _clean_text(text),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _flora_fields(meta: dict) -> list[tuple[str, str, str]]:
    rows = []
    cls = meta.get("chemistry_class", "").strip()
    mech = meta.get("mechanism_type", "").strip()
    pc = meta.get("photocatalyst", "").strip()
    wl = meta.get("wavelength_nm")
    sol = meta.get("solvent", "").strip()
    rt = meta.get("reactor_type", "").strip()

    if cls:
        rows.append(("Reaction class", cls, "reaction"))
    if mech:
        rows.append(("Mechanism", mech, "mechanism"))
    if pc:
        rows.append(("Photocatalyst", pc, "photocatalyst"))
    if wl and wl > 0:
        rows.append(("Wavelength", f"{wl:.0f} nm", "wavelength"))
    if sol:
        rows.append(("Solvent", sol, "solvent"))
    if rt:
        rows.append(("Reactor", rt, "reaction"))
    return rows


def _context_summary(doc: str, n_sentences: int = 2) -> str:
    sentences = [s.strip() for s in doc.split(".") if len(s.strip()) > 15]
    selected = sentences[:n_sentences]
    return ". ".join(selected) + ("." if selected else "")


def _draw_badge(ax, x, y, value: int, accent: str):
    badge = FancyBboxPatch(
        (x, y), 0.18, 0.10,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor="#FFFFFF", edgecolor=PANEL_EDGE, linewidth=0.8, zorder=4,
    )
    ax.add_patch(badge)
    ax.text(x + 0.09, y + 0.064, f"{value}", ha="center", va="center",
            fontsize=11, color=accent, fontweight="bold", zorder=5)
    ax.text(x + 0.09, y + 0.028, "chem. terms", ha="center", va="center",
            fontsize=6.8, color=TEXT_DIM, zorder=5)


def _draw_query_box(ax, x0, y0, w, h, title, content,
                    variant="naive", accent="#58A6FF", bg=PANEL_BG):
    """Draw a styled query card with controlled wrapping."""
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor=bg, edgecolor=PANEL_EDGE, linewidth=0.9, zorder=2,
    )
    ax.add_patch(box)

    title_bar = FancyBboxPatch(
        (x0, y0 + h - 0.115), w, 0.115,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        facecolor=accent, edgecolor="none", zorder=3, alpha=0.18,
    )
    ax.add_patch(title_bar)
    ax.text(x0 + 0.035, y0 + h - 0.058, title,
            ha="left", va="center", fontsize=10, color=TEXT_LIGHT,
            fontweight="bold", zorder=4)
    ax.text(x0 + w - 0.03, y0 + h - 0.058,
            "plan-aware" if variant == "flora" else "baseline",
            ha="right", va="center", fontsize=7.2,
            color=accent, fontweight="bold", zorder=4)

    left = x0 + 0.04
    right = x0 + w - 0.04
    body_top = y0 + h - 0.155

    if variant == "flora":
        ax.text(left, body_top, "Structured enrichment", ha="left", va="top",
                fontsize=7.0, color=TEXT_DIM, fontweight="bold", zorder=4)
        label_x = left
        value_x = x0 + 0.34
        y = body_top - 0.07
        row_gap = 0.088
        line_gap = 0.043

        for label, value, key in content["fields"]:
            wrapped = _wrap_text(value, width=36)[:2]
            ax.text(label_x, y, label.upper(), ha="left", va="top",
                    fontsize=6.5, color=TERM_COLORS[key], fontweight="bold",
                    zorder=4)
            for line_idx, line in enumerate(wrapped):
                ax.text(value_x, y - line_idx * line_gap, line,
                        ha="left", va="top", fontsize=8.1,
                        color=TEXT_LIGHT if line_idx == 0 else TEXT_SOFT,
                        zorder=4)
            sep_y = y - max(0, len(wrapped) - 1) * line_gap - 0.05
            ax.plot([left, right], [sep_y, sep_y],
                    color=GRID, lw=0.8, zorder=3)
            y -= row_gap + max(0, len(wrapped) - 1) * line_gap

        y -= 0.008
        ax.text(left, y, "Context summary", ha="left", va="top",
                fontsize=7.0, color=TEXT_DIM, fontweight="bold", zorder=4)
        y -= 0.058
        for line in _wrap_text(content["context"], width=68)[:3]:
            ax.text(left, y, line, ha="left", va="top",
                    fontsize=7.8, color=TEXT_SOFT, zorder=4)
            y -= 0.042
    else:
        ax.text(left, body_top, "Raw query text", ha="left", va="top",
                fontsize=7.0, color=TEXT_DIM, fontweight="bold", zorder=4)
        y = body_top - 0.065
        for line in _wrap_text(content["text"], width=74)[:6]:
            ax.text(left, y, line, ha="left", va="top",
                    fontsize=8.0, color=TEXT_SOFT, zorder=4,
                    fontfamily="monospace")
            y -= 0.046
        ax.text(left, y - 0.01, "Minimal chemistry-specific structure retained.",
                ha="left", va="top", fontsize=7.2, color=TEXT_DIM,
                style="italic", zorder=4)

    _draw_badge(ax, x0 + w - 0.215, y0 + 0.04, content["term_count"], accent)


def _draw_term_legend(ax, items):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panel = FancyBboxPatch(
        (0.0, 0.05), 1.0, 0.90,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor=SOFT_BG, edgecolor=PANEL_EDGE, linewidth=0.8, zorder=1,
    )
    ax.add_patch(panel)
    ax.text(0.03, 0.72, "Highlighted fields in FlowPilot query",
            ha="left", va="center", fontsize=8.5,
            color=TEXT_LIGHT, fontweight="bold", zorder=2)
    ax.text(0.97, 0.72, "Each field is injected explicitly before retrieval",
            ha="right", va="center", fontsize=7.2,
            color=TEXT_DIM, zorder=2)

    chip_w = 0.175
    x_positions = [0.03, 0.22, 0.41, 0.60, 0.79]
    for (label, key), x in zip(items, x_positions):
        chip = FancyBboxPatch(
            (x, 0.22), chip_w, 0.28,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor="#FFFFFF", edgecolor=PANEL_EDGE, linewidth=0.8, zorder=2,
        )
        ax.add_patch(chip)
        swatch = FancyBboxPatch(
            (x + 0.014, 0.275), 0.022, 0.17,
            boxstyle="round,pad=0.0,rounding_size=0.01",
            facecolor=TERM_COLORS[key], edgecolor="none", zorder=3,
        )
        ax.add_patch(swatch)
        ax.text(x + 0.048, 0.36, label, ha="left", va="center",
                fontsize=7.6, color=TEXT_LIGHT, zorder=3)


def make_figure(figsize=(13, 8.6)) -> plt.Figure:
    records = _load_rich_records(n=2)

    fig = plt.figure(figsize=figsize, facecolor=BG)
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.0, 1.0, 0.22, 0.86],
        hspace=0.18, wspace=0.12,
        left=0.05, right=0.97, top=0.92, bottom=0.09,
    )

    example_axes = [
        (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])),
        (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])),
    ]
    ax_legend = fig.add_subplot(gs[2, :])
    ax_bar = fig.add_subplot(gs[3, :])

    for ax_pair in example_axes:
        for ax in ax_pair:
            ax.set_facecolor(BG)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

    ax_legend.set_facecolor(BG)
    ax_legend.axis("off")
    ax_bar.set_facecolor(BG)
    for spine in ax_bar.spines.values():
        spine.set_color(PANEL_EDGE)
    ax_bar.tick_params(colors=TEXT_DIM, labelsize=8)

    fig.suptitle("Query Enrichment: Naive Text vs FlowPilot Plan-Aware Query",
                 fontsize=14, fontweight="bold", color=TEXT_LIGHT, y=0.965)

    naive_counts = []
    flora_counts = []
    example_labels = []

    for idx, (rid, meta, doc) in enumerate(records):
        naive_text = _build_naive_query(meta, doc)
        flora_fragments = _build_flora_query(meta, doc)
        flora_text = " ".join(fragment for fragment, _ in flora_fragments)

        naive_n = _count_info_terms(naive_text)
        flora_n = _count_info_terms(flora_text)
        naive_counts.append(naive_n)
        flora_counts.append(flora_n)

        cls = meta.get("chemistry_class", "").strip() or f"Record {idx + 1}"
        short_label = textwrap.shorten(cls, width=30, placeholder="...")
        example_labels.append(short_label)

        ax_naive, ax_flora = example_axes[idx]
        ax_naive.text(0.03, 1.02, f"Example {idx + 1}  |  {short_label}",
                      ha="left", va="bottom", fontsize=8.2,
                      color=TEXT_DIM, fontweight="bold",
                      transform=ax_naive.transAxes)

        _draw_query_box(
            ax_naive, 0.03, 0.05, 0.94, 0.88,
            "Naive Query",
            {"text": naive_text, "term_count": naive_n},
            variant="naive",
            accent="#58A6FF",
            bg=PANEL_BG,
        )
        _draw_query_box(
            ax_flora, 0.03, 0.05, 0.94, 0.88,
            "FlowPilot Query",
            {
                "fields": _flora_fields(meta),
                "context": _context_summary(doc),
                "term_count": flora_n,
            },
            variant="flora",
            accent="#3FB950",
            bg=PANEL_BG,
        )

    legend_items = [
        ("Mechanism", "mechanism"),
        ("Photocatalyst", "photocatalyst"),
        ("Wavelength", "wavelength"),
        ("Solvent", "solvent"),
        ("Reaction class", "reaction"),
    ]
    _draw_term_legend(ax_legend, legend_items)

    x = np.arange(len(records))
    bw = 0.32
    bars_naive = ax_bar.bar(
        x - bw / 2, naive_counts, bw,
        color="#1C3A5E", edgecolor="#58A6FF",
        linewidth=0.9, label="Naive Query", zorder=3,
    )
    bars_flora = ax_bar.bar(
        x + bw / 2, flora_counts, bw,
        color="#1A4731", edgecolor="#3FB950",
        linewidth=0.9, label="FlowPilot Query", zorder=3,
    )

    for bar, value in zip(list(bars_naive) + list(bars_flora), naive_counts + flora_counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.18, str(value),
                    ha="center", va="bottom", fontsize=9,
                    color=TEXT_LIGHT, fontweight="bold")

    for i in range(len(records)):
        mult = flora_counts[i] / max(naive_counts[i], 1)
        ax_bar.text(i, max(flora_counts[i], naive_counts[i]) + 1.0,
                    f"{mult:.1f}x richer",
                    ha="center", va="bottom", fontsize=8,
                    color="#F78166", fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f"Example {i + 1}\n({example_labels[i]})" for i in range(len(records))],
        fontsize=8.6, color=TEXT_DIM,
    )
    ax_bar.set_ylabel("Chemical Information Terms", fontsize=9,
                      color=TEXT_DIM, labelpad=6)
    ax_bar.set_title("Query Information Richness Comparison",
                     fontsize=10.5, fontweight="bold", color=TEXT_LIGHT, pad=8)
    ax_bar.tick_params(axis="x", length=0)
    ax_bar.set_ylim(0, max(flora_counts + naive_counts) * 1.35 + 1.2)
    ax_bar.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax_bar.legend(fontsize=8.5, framealpha=0.35,
                  facecolor="#FFFFFF", edgecolor=PANEL_EDGE,
                  labelcolor=TEXT_LIGHT, loc="upper right")
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
