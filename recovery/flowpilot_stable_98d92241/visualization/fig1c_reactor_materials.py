"""
Figure 1C — Reactor material breakdown (donut chart + bar).

Two styles available: 'donut' or 'bar'.

Jupyter usage:
    from fig1c_reactor_materials import make_figure
    fig = make_figure(style="donut")
    fig.savefig("fig1c.png", dpi=300, bbox_inches="tight")
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_records, get_reactor_materials

PALETTE = [
    "#E63946", "#457B9D", "#1D3557", "#A8DADC", "#F1FAEE",
    "#2A9D8F", "#E9C46A", "#F4A261", "#CCCCCC",
]


def make_figure(
    records_dir=None,
    style: str = "donut",   # "donut" or "bar"
    top_n: int = 7,
    figsize: tuple = (5, 5),
) -> plt.Figure:
    records = load_records(records_dir) if records_dir else load_records()
    if True:  # always try LLM cache first
        from data_loader import get_classified_counts
        llm_counts = get_classified_counts("reactor_material", records)
        counts = llm_counts if llm_counts else get_reactor_materials(records)
    else:
        counts = get_reactor_materials(records)

    known = {k: v for k, v in counts.items() if k != "Other / Unknown"}
    other = counts.get("Other / Unknown", 0)
    top = sorted(known.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [t[0] for t in top] + ["Other / Unknown"]
    values = [t[1] for t in top] + [other]
    colors = PALETTE[:len(labels) - 1] + ["#DDDDDD"]

    fig, ax = plt.subplots(figsize=figsize)

    if style == "donut":
        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors,
            autopct=lambda p: f"{p:.0f}%" if p > 4 else "",
            startangle=140,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.2),
            pctdistance=0.78,
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color("white")
            at.set_fontweight("bold")

        ax.legend(
            wedges, [f"{l} ({v})" for l, v in zip(labels, values)],
            loc="center left", bbox_to_anchor=(0.95, 0.5),
            fontsize=8, frameon=False,
        )
        total = sum(values)
        ax.text(0, 0, f"{total}\npapers", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#333333")
        ax.set_title("Reactor Materials", fontsize=11, fontweight="bold", pad=14)

    else:  # bar
        labels_r, values_r, colors_r = labels[::-1], values[::-1], colors[::-1]
        bars = ax.barh(labels_r, values_r, color=colors_r, edgecolor="white", height=0.65)
        for bar, val in zip(bars, values_r):
            ax.text(bar.get_width() + max(values_r) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", ha="left", fontsize=8, color="#444444")
        ax.set_xlabel("Number of Papers", fontsize=10)
        ax.set_title("Reactor Materials", fontsize=11, fontweight="bold", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, max(values_r) * 1.15)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = make_figure(style="donut")
    plt.show()
