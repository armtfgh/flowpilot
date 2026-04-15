"""
Figure 1E — Light source types (highlights photochemistry coverage).

Shows a split: photochemical papers (with light source) vs thermal/dark.

Jupyter usage:
    from fig1e_light_sources import make_figure
    fig = make_figure()
    fig.savefig("fig1e.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_loader import load_records, get_light_sources

PALETTE = {
    "LED (blue/violet)":  "#5E60CE",
    "LED (white/other)":  "#48CAE4",
    "UV Lamp":            "#F72585",
    "Xenon Lamp":         "#FF9E00",
    "Solar Simulator":    "#FFD60A",
    "Microwave":          "#80B918",
    "None / Dark":        "#DDDDDD",
}


def make_figure(records_dir=None, figsize: tuple = (6, 4)) -> plt.Figure:
    records = load_records(records_dir) if records_dir else load_records()
    from data_loader import get_classified_counts
    llm_counts = get_classified_counts("light_source", records)
    counts = llm_counts if llm_counts else get_light_sources(records)

    # Separate photochem vs dark
    photo_counts = {k: v for k, v in counts.items() if k != "None / Dark"}
    dark = counts.get("None / Dark", 0)
    total = sum(counts.values())
    photo_total = sum(photo_counts.values())

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)
    ax_bar = fig.add_subplot(gs[0])
    ax_pie = fig.add_subplot(gs[1])

    # ── Left: light source breakdown (photo only) ─────────────────────────────
    photo_sorted = sorted(photo_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [t[0] for t in photo_sorted]
    values = [t[1] for t in photo_sorted]
    colors = [PALETTE.get(l, "#AAAAAA") for l in labels]

    labels_r, values_r, colors_r = labels[::-1], values[::-1], colors[::-1]
    bars = ax_bar.barh(labels_r, values_r, color=colors_r, edgecolor="white", height=0.6)
    for bar, val in zip(bars, values_r):
        ax_bar.text(bar.get_width() + max(values_r) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", ha="left", fontsize=8, color="#444444")

    ax_bar.set_xlabel("Number of Papers", fontsize=9, labelpad=5)
    ax_bar.set_title("Light Source Types\n(photochem papers only)", fontsize=9, fontweight="bold")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(labelsize=8)
    ax_bar.set_xlim(0, max(values_r) * 1.2)

    # ── Right: photochem vs dark pie ──────────────────────────────────────────
    pie_vals = [photo_total, dark]
    pie_labels = [f"Photochem\n({photo_total})", f"Thermal/Dark\n({dark})"]
    pie_colors = ["#5E60CE", "#DDDDDD"]
    wedges, _, autotexts = ax_pie.pie(
        pie_vals,
        labels=None,
        colors=pie_colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.2),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")

    ax_pie.legend(wedges, pie_labels, loc="lower center",
                  bbox_to_anchor=(0.5, -0.18), fontsize=7.5, frameon=False, ncol=1)
    ax_pie.set_title("Corpus Split", fontsize=9, fontweight="bold")

    fig.suptitle("Photochemical Coverage in Dataset", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = make_figure()
    plt.show()
