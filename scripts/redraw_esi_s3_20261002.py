#!/usr/bin/env python3
"""Re-export Figure S3 from its original CSVs without changing the ESI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "visualization/panel_data_exports"
STEM = ROOT / "manuscript/Figure_S3_corrected_20261002"
AUDIT = ROOT / "manuscript/figure_s3_layout_check_20261002"
INK, GRID = "#17212B", "#D9DEE5"
BLUE, TEAL, ORANGE = "#2468A2", "#16847A", "#D17A22"
PURPLE, GREEN = "#7A5AA6", "#4C956C"
FILES = [
    "fig1a_reaction_classes.csv", "fig1b_reactor_types.csv",
    "fig1c_reactor_materials.csv", "fig1d_bond_types.csv",
    "fig1f_inlet_streams.csv", "fig1g_batch_yields_raw.csv",
    "fig1g_flow_yields_raw.csv",
]


def clean(ax, axis):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(color=GRID, labelcolor=INK, length=3, pad=6)
    ax.grid(axis=axis, color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.labelsize": 14, "xtick.labelsize": 12,
        "ytick.labelsize": 13, "text.color": INK,
        "axes.labelcolor": INK, "figure.facecolor": "white",
        "axes.facecolor": "white", "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": True,
    })
    frames = [pd.read_csv(SOURCE / name) for name in FILES]
    reaction, reactor, material, bond, streams = frames[:5]
    batch, flow = [d.iloc[:, -1].dropna().astype(float) for d in frames[5:]]
    assert (int(streams["count"].sum()), len(batch), len(flow)) == (154, 135, 285)
    fig = plt.figure(figsize=(13, 14.4), dpi=120)
    titles = ["Reaction class", "Reactor type", "Reactor material",
              "Bond / transformation", "Inlet streams", "Available yield distributions"]
    header_y = [0.975, 0.660, 0.345]
    axes_y = [0.715, 0.400, 0.085]
    axes, headers = [], []
    for i, title in enumerate(titles):
        row, col = divmod(i, 2)
        x = 0.025 + 0.51 * col
        letter = fig.text(x, header_y[row], "abcdef"[i], fontsize=19,
                          fontweight="bold", va="top")
        heading = fig.text(x + 0.035, header_y[row] - 0.001, title,
                           fontsize=15, fontweight="bold", va="top")
        denominator = ("n = 464 records" if i < 4 else "n = 154 annotated records"
                       if i == 4 else "Batch n = 135; flow n = 285")
        note = fig.text(x + 0.035, header_y[row] - 0.024, denominator,
                        fontsize=12.5, va="top", color="#5B6775")
        headers.append([letter, heading, note])
        # Each column reserves a full label gutter independent of the bars.
        axes.append(fig.add_axes([0.205 + 0.51 * col, axes_y[row], 0.265, 0.213]))
    plotted_counts = {}
    for ax, frame, color, n, letter in zip(
        axes[:4], frames[:4], [BLUE, TEAL, ORANGE, PURPLE], [8, 8, 7, 7], "abcd"
    ):
        data = frame.sort_values("count", ascending=False).head(n).sort_values("count")
        bars = ax.barh(np.arange(len(data)), data["count"], color=color, alpha=0.9, height=0.74)
        ax.set_yticks(np.arange(len(data)), [fill(str(v), 23) for v in data.iloc[:, 0]])
        ax.set_xlim(0, data["count"].max() * 1.16)
        ax.set_ylim(-0.65, len(data) - 0.35)
        ax.set_xlabel("Records", labelpad=8)
        ticks = MaxNLocator(nbins=5, integer=True).tick_values(*ax.get_xlim())
        ax.set_xticks([v for v in ticks if 0 <= v <= ax.get_xlim()[1]])
        ax.bar_label(bars, fontsize=12, padding=5)
        clean(ax, "x")
        plotted_counts[letter] = dict(zip(data.iloc[:, 0], data["count"].astype(int)))
    ax = axes[4]
    data = streams.sort_values(streams.columns[0])
    bars = ax.bar(data.iloc[:, 0].astype(str), data["count"], color=GREEN, width=0.72)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of inlet streams", labelpad=8)
    ax.set_ylabel("Records", labelpad=10)
    ax.bar_label(bars, fontsize=12, padding=5)
    clean(ax, "y")
    ax = axes[5]
    bins = np.arange(0, 105, 5)
    ax.hist(batch, bins=bins, density=True, alpha=0.45, color=ORANGE, label="Batch")
    ax.hist(flow, bins=bins, density=True, alpha=0.55, color=BLUE, label="Flow")
    ax.axvline(batch.median(), color=ORANGE, linestyle="--", linewidth=1.4)
    ax.axvline(flow.median(), color=BLUE, linestyle="--", linewidth=1.4)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.070)
    ax.set_xlabel("Yield (%)", labelpad=8)
    ax.set_ylabel("Density", labelpad=10)
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    clean(ax, "y")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = [Bbox.union([ax.get_tightbbox(renderer)] + [t.get_window_extent(renderer) for t in head])
               for ax, head in zip(axes, headers)]
    gaps = [extents[i + 1].x0 - extents[i].x1 for i in (0, 2, 4)]
    assert min(gaps) > 10, f"Insufficient inter-panel gap: {gaps}"
    collisions = []
    for i, first in enumerate(extents):
        for j, second in enumerate(extents[i + 1:], i + 1):
            if first.overlaps(second):
                collisions.append([i, j])
    assert not collisions, f"Panel bounding boxes overlap: {collisions}"
    text_boxes = []
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            bounds = t.get_window_extent(renderer)
            if bounds.width and bounds.height:
                text_boxes.append((t.get_text(), bounds))
    outside = [text for text, bounds in text_boxes if
               bounds.x0 < 0 or bounds.y0 < 0 or
               bounds.x1 > fig.bbox.width or bounds.y1 > fig.bbox.height]
    assert not outside, f"Text outside canvas: {outside}"
    AUDIT.mkdir(exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(STEM.with_suffix("." + suffix), dpi=600, facecolor="white")
    fig.savefig(AUDIT / "preview.png", dpi=130, facecolor="white")
    report = {
        "source_data_unchanged": True,
        "source_sha256": {n: hashlib.sha256((SOURCE / n).read_bytes()).hexdigest() for n in FILES},
        "categories_and_counts": plotted_counts,
        "denominators": {"corpus": 464, "inlet_streams": 154, "batch_yields": 135, "flow_yields": 285},
        "inter_panel_gap_pixels_at_120dpi": gaps,
        "overlapping_panels": collisions,
        "text_outside_canvas": outside,
        "png_dimensions": [7800, 8640], "png_dpi": 600,
        "word_documents_modified": False,
    }
    (AUDIT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
