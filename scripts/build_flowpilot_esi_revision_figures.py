#!/usr/bin/env python3
"""Build the reviewed FlowPilot ESI figures and their source-data package."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from textwrap import fill

import cairosvg
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "deliverables" / "flowpilot_esi_revision_20260902"
FIG_DIR = OUT / "figures"
DATA_DIR = OUT / "source_data"
DOC_DIR = OUT / "documentation"
PANEL = ROOT / "visualization" / "panel_data_exports"
BENCH = ROOT / "deliverables" / "manuscript_benchmark_visualizations_20260825"

INK = "#17212B"
MUTED = "#5B6775"
GRID = "#D9DEE5"
LIGHT = "#F5F7FA"
BLUE = "#2468A2"
BLUE_L = "#DCEAF5"
TEAL = "#16847A"
TEAL_L = "#DCEFEB"
ORANGE = "#D17A22"
ORANGE_L = "#F8E7D5"
RED = "#C44E52"
RED_L = "#F6DEDF"
PURPLE = "#7A5AA6"
PURPLE_L = "#E9E2F1"
GREEN = "#4C956C"
GREEN_L = "#E0EFE6"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> None:
    for path in (FIG_DIR, DATA_DIR, DOC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, number: int) -> None:
    stem = FIG_DIR / f"Figure_S{number:02d}"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax, text: str, x: float = -0.07, y: float = 1.04) -> None:
    ax.text(x, y, text, transform=ax.transAxes, va="top", ha="left", fontsize=10.5, fontweight="bold", color=INK)


def clean(ax, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(color=GRID, labelcolor=INK, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)


def box(ax, x, y, w, h, text, *, fc=LIGHT, ec=GRID, fs=6.5, bold=False, radius=0.02, lw=1.0) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, start, end, *, color=MUTED, lw=1.0, style="-|>") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=8, linewidth=lw, color=color))


def figure_s1() -> None:
    fig = plt.figure(figsize=(7.4, 5.05))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 0.9], hspace=0.16)

    ax = fig.add_subplot(gs[0])
    panel_label(ax, "a", x=-0.015, y=1.02)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Architecture and authority boundaries", pad=3, fontweight="bold", color=INK)
    stages = [
        ("1", "Standardized\nintake", BLUE_L, BLUE),
        ("2", "BatchRecord", BLUE_L, BLUE),
        ("3", "ChemistryPlan\n(upstream)", PURPLE_L, PURPLE),
        ("4", "Plan-aware\nretrieval", PURPLE_L, PURPLE),
        ("5", "Engineering +\ndesign search", TEAL_L, TEAL),
        ("6", "FlowProposal\n(downstream)", PURPLE_L, PURPLE),
        ("7", "Council +\ncandidate gates", ORANGE_L, ORANGE),
        ("8", "Evidence +\ninventory", ORANGE_L, ORANGE),
        ("9", "FinalDesign\nContract", TEAL_L, TEAL),
        ("10", "Artifacts +\nautosave", GREEN_L, GREEN),
    ]
    xs = np.linspace(0.06, 0.62, 5)
    for row in range(2):
        y = 0.72 if row == 0 else 0.33
        subset = stages[row * 5 : row * 5 + 5]
        for i, (num, label, fc, ec) in enumerate(subset):
            x = xs[i] if row == 0 else xs[4 - i]
            box(ax, x, y, 0.104, 0.23, f"{num}\n{label}", fc=fc, ec=ec, fs=5.4, bold=True)
            if i < 4:
                x2 = xs[i + 1] if row == 0 else xs[4 - (i + 1)]
                arrow(ax, (x + (0.055 if row == 0 else -0.055), y), (x2 - (0.055 if row == 0 else -0.055), y), color=ec)
    arrow(ax, (xs[-1], 0.60), (xs[-1], 0.45), color=TEAL)

    ax.text(0.77, 0.91, "Authority order", ha="center", fontsize=6.8, fontweight="bold", color=INK)
    authority = [
        ("Measured evidence", "#E5E7EB"),
        ("Safety + inventory", "#ECEFF1"),
        ("Confirmed protocol", "#F1F3F5"),
        ("Chemist hypotheses", "#F5F6F7"),
        ("Model inference", "#F8F9FA"),
    ]
    for i, (label, fc) in enumerate(authority):
        y = 0.80 - i * 0.105
        box(ax, 0.77, y, 0.16, 0.083, label, fc=fc, ec=GRID, fs=5.0, bold=i < 2, radius=0.005)
    ax.text(0.77, 0.23, "Higher authority constrains\nlower-authority proposals", ha="center", fontsize=4.8, color=MUTED)

    box(ax, 0.93, 0.70, 0.11, 0.15, "EXECUTABLE", fc=GREEN_L, ec=GREEN, fs=5.2, bold=True)
    box(ax, 0.93, 0.42, 0.11, 0.18, "BLOCKED /\nCONFIRMATION", fc=RED_L, ec=RED, fs=4.9, bold=True)
    arrow(ax, (0.855, 0.52), (0.87, 0.69), color=GREEN)
    arrow(ax, (0.855, 0.49), (0.87, 0.43), color=RED)
    ax.text(0.48, 0.055, "Model confidence cannot override deterministic feasibility gates.", ha="center", fontsize=6.0, color=INK, fontweight="bold")

    ax = fig.add_subplot(gs[1])
    panel_label(ax, "b", x=-0.015, y=1.02)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Typed contracts and numerical closure", pad=3, fontweight="bold", color=INK)
    chain = ["DesignInput\nPackage", "BatchRecord", "ChemistryPlan", "FlowProposal", "Design\nCalculations", "Process\nTopology", "FinalDesign\nContract"]
    xs = np.linspace(0.07, 0.93, len(chain))
    for i, (x, label) in enumerate(zip(xs, chain)):
        fc, ec = (GREEN_L, GREEN) if i == len(chain) - 1 else (TEAL_L, TEAL)
        box(ax, x, 0.70, 0.112, 0.18, label, fc=fc, ec=ec, fs=5.4, bold=True)
        if i < len(chain) - 1:
            arrow(ax, (x + 0.058, 0.70), (xs[i + 1] - 0.058, 0.70), color=TEAL)
            ax.text((x + xs[i + 1]) / 2, 0.79, "check", fontsize=4.6, color=TEAL, ha="center")
    loop = [
        (0.17, "Bounded\ncandidate edit"),
        (0.39, "Recompute\ndependencies"),
        (0.61, "Assign exact\nhardware"),
        (0.83, "Validate or\nblock"),
    ]
    for i, (x, label) in enumerate(loop):
        box(ax, x, 0.27, 0.17, 0.17, label, fc=ORANGE_L if i == 0 else TEAL_L, ec=ORANGE if i == 0 else TEAL, fs=5.6, bold=True)
        if i < len(loop) - 1:
            arrow(ax, (x + 0.09, 0.27), (loop[i + 1][0] - 0.09, 0.27), color=TEAL)
    arrow(ax, (0.83, 0.17), (0.17, 0.17), color=MUTED, style="-")
    arrow(ax, (0.17, 0.17), (0.17, 0.18), color=MUTED)
    ax.text(0.50, 0.045, "Council candidates are not executable; only a closed FinalDesignContract publishes run parameters.", ha="center", fontsize=5.7, color=MUTED)
    save(fig, 1)


def figure_s2() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 4.15), gridspec_kw={"wspace": 0.16})
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    panel_label(ax, "a", x=-0.02, y=1.02)
    ax.set_title("Standardized\nintake", fontweight="bold", color=INK, pad=2, fontsize=8.0)
    box(ax, 0.18, 0.86, 0.25, 0.12, "Free-text\nprotocol", fc=BLUE_L, ec=BLUE, fs=5.6, bold=True)
    box(ax, 0.52, 0.86, 0.25, 0.12, "Intake LLM\nextracts content", fc=PURPLE_L, ec=PURPLE, fs=5.4, bold=True)
    arrow(ax, (0.31, 0.86), (0.39, 0.86), color=BLUE)
    questions = ["Q-BATCH-001", "Q-OBJ-001", "Q-CHEM-001", "Q-HIST-001", "Q-INV-001", "Q-CONSTR-001", "Q-HYP-001", "Q-PREF-001"]
    y = 0.70
    for i, q in enumerate(questions):
        required = i < 3
        ax.add_patch(patches.Rectangle((0.08, y - i * 0.065), 0.55, 0.050, facecolor=LIGHT, edgecolor=GRID, linewidth=0.7))
        ax.text(0.10, y + 0.025 - i * 0.065, q, fontsize=4.8, va="center", color=INK)
        ax.text(0.61, y + 0.025 - i * 0.065, "answer" if required else "answer / unavailable", fontsize=4.2, va="center", ha="right", color=GREEN if required else MUTED)
    box(ax, 0.78, 0.55, 0.25, 0.15, "Deterministic\nreadiness gate", fc=ORANGE_L, ec=ORANGE, fs=5.4, bold=True)
    arrow(ax, (0.64, 0.56), (0.65, 0.56), color=ORANGE)
    box(ax, 0.50, 0.12, 0.66, 0.13, "Frozen DesignInputPackage\nflowpilot_intake_v1.0", fc=GREEN_L, ec=GREEN, fs=5.5, bold=True)
    arrow(ax, (0.78, 0.47), (0.57, 0.20), color=GREEN)
    ax.text(0.50, 0.015, "Fixed IDs; the LLM cannot bypass readiness.", ha="center", fontsize=4.7, color=MUTED)

    ax = axes[1]
    panel_label(ax, "b", x=-0.02, y=1.02)
    ax.set_title("GUI and single-source\nrendering", fontweight="bold", color=INK, pad=2, fontsize=8.0)
    steps = ["Paste", "Analyze", "Resolve", "Inventory", "Freeze", "Run", "Review"]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, (x, label) in enumerate(zip(xs, steps)):
        box(ax, x, 0.84, 0.105, 0.10, label, fc=BLUE_L if i < 5 else TEAL_L, ec=BLUE if i < 5 else TEAL, fs=4.6, bold=True)
        if i < len(steps) - 1:
            arrow(ax, (x + 0.055, 0.84), (xs[i + 1] - 0.055, 0.84), color=MUTED)
    box(ax, 0.50, 0.61, 0.53, 0.14, "FinalDesignContract\nflowpilot_final_design_v2.0", fc=GREEN_L, ec=GREEN, fs=5.4, bold=True)
    views = ["Summary", "Engineering", "Diagram", "Streams", "Equipment", "Safety", "Raw JSON"]
    for i, label in enumerate(views):
        col, row = i % 4, i // 4
        x, y = 0.14 + col * 0.24, 0.39 - row * 0.17
        box(ax, x, y, 0.20, 0.10, label, fc=LIGHT, ec=GREEN, fs=4.7)
        arrow(ax, (0.50, 0.53), (x, y + 0.055), color=GREEN, lw=0.7)
    box(ax, 0.50, 0.08, 0.70, 0.10, "Blocked: requirements + reasons; parameters withheld", fc=RED_L, ec=RED, fs=4.7, bold=True)
    ax.text(0.50, 0.005, "All executable numerical views use the same post-validation contract.", ha="center", fontsize=4.5, color=MUTED)

    ax = axes[2]
    panel_label(ax, "c", x=-0.02, y=1.02)
    ax.set_title("Autosaved\nprovenance", fontweight="bold", color=INK, pad=2, fontsize=8.0)
    artifacts = [
        "input.txt +\nintake_package.json",
        "result.json",
        "final_design.json",
        "topology.json +\ninventory_allocation.json",
        "instrument_manifest.json",
        "process.svg/png +\nrender_manifest.json",
        "summary.json",
    ]
    ys = np.linspace(0.88, 0.18, len(artifacts))
    for i, (y, label) in enumerate(zip(ys, artifacts)):
        box(ax, 0.46, y, 0.64, 0.080, label, fc=GREEN_L if i >= 2 else BLUE_L, ec=GREEN if i >= 2 else BLUE, fs=4.7, bold=i in (2, 3))
        if i < len(artifacts) - 1:
            arrow(ax, (0.46, y - 0.045), (0.46, ys[i + 1] + 0.045), color=MUTED)
    box(ax, 0.83, 0.49, 0.27, 0.21, "Conditional diagnostics\nrequirements topology\ndiagnostic render", fc=LIGHT, ec=MUTED, fs=4.4)
    arrow(ax, (0.79, 0.55), (0.69, 0.55), color=MUTED, style="-")
    ax.text(0.50, 0.035, "Exact replay also requires prompt, model, decoding, software, and environment records.", ha="center", fontsize=4.2, color=MUTED, wrap=True)
    save(fig, 2)


def _top_barh(ax, frame: pd.DataFrame, label: str, value: str, title: str, color: str, n: int = 8) -> None:
    data = frame.sort_values(value, ascending=False).head(n).sort_values(value)
    ax.barh(np.arange(len(data)), data[value], color=color, alpha=0.9)
    ax.set_yticks(np.arange(len(data)), [fill(str(v), 23) for v in data[label]])
    ax.set_title(title, fontweight="bold", color=INK)
    ax.set_xlabel("Records")
    ax.bar_label(ax.containers[0], fontsize=6.0, padding=2)
    clean(ax, "x")


def figure_s3() -> None:
    files = [
        "fig1a_reaction_classes.csv",
        "fig1b_reactor_types.csv",
        "fig1c_reactor_materials.csv",
        "fig1d_bond_types.csv",
        "fig1f_inlet_streams.csv",
        "fig1g_batch_yields_raw.csv",
        "fig1g_flow_yields_raw.csv",
    ]
    for name in files:
        shutil.copy2(PANEL / name, DATA_DIR / f"S3_{name}")
    reaction, reactor, material, bond, streams = [pd.read_csv(PANEL / name) for name in files[:5]]
    batch = pd.read_csv(PANEL / files[5]).iloc[:, -1].dropna().astype(float)
    flow = pd.read_csv(PANEL / files[6]).iloc[:, -1].dropna().astype(float)
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.7))
    plt.subplots_adjust(hspace=0.45, wspace=0.43)
    _top_barh(axes[0, 0], reaction, "category", "count", "Reaction class (n = 464)", BLUE)
    _top_barh(axes[0, 1], reactor, reactor.columns[0], "count", "Reactor type (n = 464)", TEAL)
    _top_barh(axes[1, 0], material, material.columns[0], "count", "Reactor material (n = 464)", ORANGE, 7)
    _top_barh(axes[1, 1], bond, bond.columns[0], "count", "Bond / transformation (n = 464)", PURPLE, 7)
    for ax, label in zip(axes.flat[:4], "abcd"):
        panel_label(ax, label, x=-0.20 if ax in axes[:, 0] else -0.18, y=1.07)
    ax = axes[2, 0]
    panel_label(ax, "e", x=-0.20, y=1.07)
    xcol, ycol = streams.columns[0], "count"
    data = streams.sort_values(xcol)
    ax.bar(data[xcol].astype(str), data[ycol], color=GREEN)
    ax.set_title("Inlet streams (n = 154)", fontweight="bold", color=INK)
    ax.set_xlabel("Number of inlet streams")
    ax.set_ylabel("Records")
    ax.bar_label(ax.containers[0], fontsize=6.0, padding=2)
    clean(ax, "y")
    ax = axes[2, 1]
    panel_label(ax, "f", x=-0.18, y=1.07)
    bins = np.arange(0, 105, 5)
    ax.hist(batch, bins=bins, density=True, alpha=0.45, color=ORANGE, label=f"Batch (n = {len(batch)})")
    ax.hist(flow, bins=bins, density=True, alpha=0.55, color=BLUE, label=f"Flow (n = {len(flow)})")
    ax.axvline(batch.median(), color=ORANGE, linestyle="--", linewidth=1.1)
    ax.axvline(flow.median(), color=BLUE, linestyle="--", linewidth=1.1)
    ax.set_title("Available yield distributions", fontweight="bold", color=INK)
    ax.set_xlabel("Yield (%)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left")
    clean(ax, "y")
    save(fig, 3)


def figure_s4() -> None:
    names = ["fig2a_rule_landscape.csv", "fig2b_formula_coverage.csv", "fig2c_coverage_heatmap_matrix.csv", "fig2d_concept_network_nodes.csv", "fig2d_concept_network_edges.csv"]
    for name in names:
        shutil.copy2(PANEL / name, DATA_DIR / f"S4_{name}")
    landscape, formula, heat, nodes, edges = [pd.read_csv(PANEL / name) for name in names]
    totals = landscape.groupby(["category_key", "category_label"], as_index=False)["count"].sum().sort_values("count", ascending=False).head(14)
    keep = totals["category_key"].tolist()
    pivot = landscape[landscape["category_key"].isin(keep)].pivot(index="category_label", columns="severity", values="count").fillna(0)
    pivot = pivot.reindex(totals["category_label"].tolist()[::-1])
    # Dedicated header bands keep letters and legends out of the data axes.
    fig = plt.figure(figsize=(11.5, 12.5))
    headers = []
    for letter, title, x, y in (
        ("a", "Largest rule categories", 0.025, 0.962),
        ("b", "Quantitative-expression coverage", 0.535, 0.962),
        ("c", "Associations by chemistry class", 0.025, 0.535),
        ("d", "Engineering concept co-occurrence", 0.535, 0.535),
    ):
        badge = fig.text(x, y, letter, fontsize=18, fontweight="bold", color=INK, va="top")
        title_artist = fig.text(x + 0.036, y - 0.001, title, fontsize=14,
                                fontweight="bold", color=INK, va="top")
        headers.append((badge, title_artist))
    ax = fig.add_axes([0.165, 0.605, 0.31, 0.29])
    landscape_ax = ax
    left = np.zeros(len(pivot))
    colors = {"hard_rule": RED, "guideline": BLUE, "tip": GREEN, "safety": ORANGE}
    for severity in [s for s in colors if s in pivot]:
        vals = pivot[severity].values
        ax.barh(np.arange(len(pivot)), vals, left=left, label=severity.replace("_", " "), color=colors[severity])
        left += vals
    ax.set_yticks(np.arange(len(pivot)), pivot.index, fontsize=12)
    ax.set_xlabel("Number of rules", fontsize=12, labelpad=9)
    ax.tick_params(axis="x", labelsize=12)
    severity_legend = fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=4,
                                loc="upper left", bbox_to_anchor=(0.16, 0.928),
                                fontsize=11, columnspacing=0.9, handlelength=1.1,
                                handletextpad=0.4, borderaxespad=0)
    clean(ax, "x")
    ax = fig.add_axes([0.665, 0.605, 0.31, 0.29])
    top = formula.sort_values("total_rules", ascending=False).head(14).sort_values("percent_with_formula")
    ax.barh(np.arange(len(top)), top["percent_with_formula"], color=TEAL)
    ax.set_yticks(np.arange(len(top)), top["category_label"], fontsize=12)
    ax.set_xlim(80, 101)
    ax.set_xlabel("Rules with detected expressions (%)", fontsize=12, labelpad=9)
    ax.tick_params(axis="x", labelsize=12)
    clean(ax, "x")
    ax = fig.add_axes([0.165, 0.205, 0.28, 0.27])
    colorbar_ax = fig.add_axes([0.459, 0.225, 0.012, 0.23])
    h = heat.sort_values("row_total", ascending=False).head(13).set_index("category_label")
    values = h.drop(columns=["category_key", "row_total"], errors="ignore")
    sns.heatmap(np.log1p(values), ax=ax, cmap="YlGnBu", cbar_ax=colorbar_ax,
                cbar_kws={"label": "log(1 + association count)"},
                linewidths=0.4, linecolor="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    colorbar_ax.tick_params(labelsize=11)
    colorbar_ax.set_ylabel("log(1 + association count)", fontsize=12, labelpad=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax = fig.add_axes([0.535, 0.195, 0.45, 0.29])
    top_nodes = nodes.nlargest(12, "frequency").copy()
    node_set = set(top_nodes["concept"])
    top_edges = edges[edges["source_concept"].isin(node_set) & edges["target_concept"].isin(node_set)].nlargest(22, "weight")
    graph = nx.Graph()
    for _, row in top_nodes.iterrows():
        graph.add_node(row["concept"], frequency=float(row["frequency"]), category=row["dominant_category_label"])
    for _, row in top_edges.iterrows():
        graph.add_edge(row["source_concept"], row["target_concept"], weight=float(row["weight"]))
    order = list(top_nodes.sort_values("frequency", ascending=False)["concept"])
    graph = nx.Graph(graph.subgraph(order))
    angles = np.linspace(0, 2 * np.pi, len(order), endpoint=False) + np.pi / 12
    pos = {node: np.array([0.86 * np.cos(angle), 0.86 * np.sin(angle)])
           for node, angle in zip(order, angles)}
    freqs = np.array([graph.nodes[n]["frequency"] for n in graph.nodes])
    sizes = 35 + 420 * np.sqrt(freqs / freqs.max())
    category_colors = {cat: [BLUE, TEAL, ORANGE, PURPLE, GREEN, RED][i % 6] for i, cat in enumerate(sorted({graph.nodes[n]["category"] for n in graph.nodes}))}
    node_colors = [category_colors[graph.nodes[n]["category"]] for n in graph.nodes]
    max_weight = max([graph.edges[e]["weight"] for e in graph.edges] or [1])
    widths = [0.25 + 2.4 * graph.edges[e]["weight"] / max_weight for e in graph.edges]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, edge_color=GRID, alpha=0.85)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes, node_color=node_colors, edgecolors="white", linewidths=0.7)
    network_labels = []
    for side in (-1, 1):
        side_nodes = sorted((n for n in order if np.sign(pos[n][0]) == side), key=lambda n: pos[n][1])
        for node, y in zip(side_nodes, np.linspace(-1.16, 1.16, len(side_nodes))):
            x = side * 1.18
            ax.plot([pos[node][0], side * 1.03, x - side * 0.05],
                    [pos[node][1], y, y], color=MUTED, linewidth=0.65, zorder=0)
            label = "Light penetration\ndepth" if node == "light penetration depth" else fill(node.capitalize(), 15)
            network_labels.append(ax.text(x, y, label, ha="left" if side > 0 else "right",
                                           va="center", fontsize=12, linespacing=1.08, color=INK))
    ax.set_xlim(-2.55, 2.55)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=color, markeredgecolor="white", label=category)
               for category, color in category_colors.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.76, 0.19),
               ncol=2, frameon=False, fontsize=11, columnspacing=1.1, handletextpad=0.3,
               title="Node color: dominant rule category", title_fontsize=12)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_boxes = [artist.get_window_extent(renderer) for artist in network_labels]
    node_boxes = []
    from matplotlib.transforms import Bbox
    for node, size in zip(graph.nodes, sizes):
        x, y = ax.transData.transform(pos[node])
        radius = np.sqrt(size) * fig.dpi / 72 / 2
        node_boxes.append(Bbox.from_extents(x - radius, y - radius, x + radius, y + radius))
    checks = {
        "legend_outside_bar_axes": not severity_legend.get_window_extent(renderer).overlaps(landscape_ax.get_window_extent(renderer)),
        "panel_letters_clear_of_titles": all(not letter.get_window_extent(renderer).overlaps(title.get_window_extent(renderer)) for letter, title in headers),
        "network_labels_do_not_overlap": not any(a.overlaps(b) for i, a in enumerate(label_boxes) for b in label_boxes[i + 1:]),
        "network_labels_clear_of_nodes": not any(a.overlaps(b) for a in label_boxes for b in node_boxes),
        "network_labels_have_no_boxes": all(artist.get_bbox_patch() is None for artist in network_labels),
        "network_labels_inside_canvas": all(fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1) for box in label_boxes),
    }
    (DOC_DIR / "Figure_S4_layout_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    if not all(checks.values()):
        raise ValueError(f"Figure S4 layout check failed: {checks}")
    (DOC_DIR / "Figure_S4d_computational_basis.md").write_text(
        "# Figure S4(d) computational basis\n\n"
        "Concept frequency is the number of rules whose cached classifier assigned the normalized concept. "
        "An edge weight is the number of rules in which the two concepts co-occur. The panel displays the "
        "12 most frequent concepts and the 22 highest-weight edges connecting those concepts. Node area is "
        "35 + 420*sqrt(f_i/f_max) points squared; node color is the dominant rule-category label; edge width "
        "is 0.25 + 2.4*(w_ij/w_max) points. A fixed circular layout is used only for legibility.\n",
        encoding="utf-8",
    )
    save(fig, 4)


def figure_s5() -> None:
    from flora_translate import config

    rows = [
        ("Semantic similarity", float(config.W_SEMANTIC), "final score"),
        ("Field similarity", float(config.W_FIELD), "final score"),
        ("Photocatalyst", float(config.W_PHOTOCATALYST), "field score"),
        ("Solvent", float(config.W_SOLVENT), "field score"),
        ("Wavelength", float(config.W_WAVELENGTH), "field score"),
        ("Temperature", float(config.W_TEMPERATURE), "field score"),
        ("Concentration", float(config.W_CONCENTRATION), "field score"),
    ]
    pd.DataFrame(rows, columns=["component", "weight", "scope"]).to_csv(DATA_DIR / "S5_retrieval_weights.csv", index=False)
    pd.DataFrame(
        [(1, "paired records", "mechanism + phase", "<3 hits"), (2, "paired records", "filters relaxed", "0 hits"), (3, "all records", "no metadata filters", "exclude hidden IDs, then rank")],
        columns=["tier", "scope", "filters", "transition"],
    ).to_csv(DATA_DIR / "S5_retrieval_tiers.csv", index=False)
    fig = plt.figure(figsize=(11.5, 10.0))
    text_artists = []
    weight_labels, weight_bars = [], []

    def label(ax, x, y, text, *, fs=12, color=INK, bold=False, ha="left", va="center"):
        artist = ax.text(x, y, text, fontsize=fs, color=color,
                         fontweight="bold" if bold else "normal", ha=ha, va=va,
                         linespacing=1.4)
        text_artists.append(artist)
        return artist

    def canvas(rect):
        ax = fig.add_axes(rect)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        return ax

    for letter, title, y in (
        ("a", "Build the search query", 0.974),
        ("b", "Broaden the search only when needed", 0.673),
        ("c", "Score, filter and rank the candidates", 0.356),
    ):
        text_artists.append(fig.text(0.025, y, letter, fontsize=18, fontweight="bold", color=INK, va="top"))
        text_artists.append(fig.text(0.062, y - 0.001, title, fontsize=15, fontweight="bold", color=INK, va="top"))
    section_rules = []
    for y in (0.705, 0.388):
        rule = plt.Line2D([0.04, 0.97], [y, y], transform=fig.transFigure, color=GRID, linewidth=0.8)
        fig.add_artist(rule)
        section_rules.append(rule)

    ax = canvas([0.04, 0.727, 0.93, 0.20])
    label(ax, 0, 0.88, "Chemistry + batch context", fs=13, bold=True)
    label(ax, 0, 0.63, "Reaction class / mechanism / bond", fs=11.5)
    label(ax, 0, 0.43, "Catalyst / solvent / temperature / wavelength", fs=11.5)
    label(ax, 0, 0.23, "Intermediate / retrieval keywords", fs=11.5)
    label(ax, 0, 0.04, "ChemistryPlan + BatchRecord", fs=11, color=MUTED)
    arrow(ax, (0.37, 0.58), (0.43, 0.58), color=BLUE, lw=1.3)
    label(ax, 0.445, 0.68, "Plan-aware query", fs=14, color=BLUE, bold=True)
    label(ax, 0.445, 0.43, "One enriched\nsearch string", fs=12, color=MUTED)
    arrow(ax, (0.645, 0.68), (0.73, 0.68), color=BLUE, lw=1.3)
    label(ax, 0.75, 0.83, "Semantic retrieval", fs=13, bold=True)
    label(ax, 0.75, 0.65, "Embedding provider", fs=11.5, color=MUTED)
    # The exception path is subordinate to the normal retrieval route.
    arrow(ax, (0.72, 0.60), (0.72, 0.24), color=ORANGE, lw=1.0)
    label(ax, 0.75, 0.43, "If provider unavailable", fs=11, color=ORANGE)
    label(ax, 0.75, 0.22, "Lexical fallback", fs=13, color=ORANGE, bold=True)
    label(ax, 0.75, 0.03, "Same query and filters", fs=11, color=MUTED)

    ax = canvas([0.04, 0.407, 0.93, 0.205])
    tiers = [(0.13, "TIER 1", "Paired records", "Mechanism + phase filters", BLUE),
             (0.49, "TIER 2", "Paired records", "Metadata filters relaxed", TEAL),
             (0.85, "TIER 3", "All records", "No metadata filters", ORANGE)]
    for x, tier, scope, filters, color in tiers:
        ax.plot([x - 0.12, x + 0.12], [0.97, 0.97], color=color, linewidth=2)
        label(ax, x, 0.83, tier, fs=11, color=color, bold=True, ha="center")
        label(ax, x, 0.63, scope, fs=14, bold=True, ha="center")
        label(ax, x, 0.43, filters, fs=11.5, color=MUTED, ha="center")
        arrow(ax, (x, 0.32), (x, 0.10), color=color, lw=1.0)
    for start, end, condition in ((0.26, 0.36, "< 3 hits"), (0.62, 0.72, "0 hits")):
        arrow(ax, (start, 0.63), (end, 0.63), color=MUTED, lw=1.1)
        label(ax, (start + end) / 2, 0.80, condition, fs=11, ha="center", color=MUTED)
    label(ax, 0.15, 0.24, "3 or more hits", fs=10.5, color=MUTED)
    label(ax, 0.51, 0.24, "1 or more hits", fs=10.5, color=MUTED)
    label(ax, 0.87, 0.24, "Available hits", fs=10.5, color=MUTED)
    ax.plot([0.13, 0.85], [0.09, 0.09], color=GRID, linewidth=1.2)
    label(ax, 0.49, -0.015, "Candidates from the selected tier", fs=12, bold=True, ha="center")

    ax = canvas([0.04, 0.094, 0.93, 0.205])
    label(ax, 0, 0.98, "Similarity term", fs=12, bold=True)
    label(ax, 0, 0.74, r"$s_{\mathrm{semantic}}=\max(0,\,1-d^2/2)$", fs=17)
    label(ax, 0, 0.53, "d: returned retrieval distance", fs=11, color=MUTED)
    label(ax, 0, 0.29,
          rf"$S={config.W_SEMANTIC:.2f}\,s_{{\mathrm{{semantic}}}}+{config.W_FIELD:.2f}\,s_{{\mathrm{{field}}}}$",
          fs=18, color=BLUE)
    label(ax, 0, 0.07, "Field similarity = sum of weighted matches", fs=11, color=MUTED)
    label(ax, 0.58, 0.98, "Field-score weights", fs=12, bold=True)
    field = [(c, w) for c, w, scope in rows if scope == "field score"]
    for (name, weight), y in zip(field, np.linspace(0.79, 0.08, len(field))):
        label(ax, 0.58, y, name, fs=12)
        bar = patches.Rectangle((0.79, y - 0.026), weight / max(w for _, w in field) * 0.15,
                                0.052, facecolor=TEAL, edgecolor="none")
        ax.add_patch(bar)
        weight_bars.append(bar)
        weight_labels.append(label(ax, 0.965, y, f"{weight:.2f}", fs=12, color=TEAL, bold=True))

    ax = canvas([0.04, 0.014, 0.93, 0.048])
    ax.plot([0, 1], [1, 1], color=GRID, linewidth=0.8)
    label(ax, 0, 0.49, "Exclude specified source IDs", fs=12, bold=True)
    label(ax, 0, 0.04, "When exclusion IDs are provided; applies to every tier", fs=10.5, color=MUTED)
    arrow(ax, (0.46, 0.49), (0.53, 0.49), color=MUTED, lw=1.1)
    label(ax, 0.55, 0.49, "Sort by S, descending", fs=12)
    arrow(ax, (0.78, 0.49), (0.85, 0.49), color=MUTED, lw=1.1)
    label(ax, 0.87, 0.49, "Return top-k", fs=12, color=TEAL, bold=True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [artist.get_window_extent(renderer) for artist in text_artists]
    collisions = [(text_artists[i].get_text(), text_artists[j].get_text())
                  for i, first in enumerate(bounds) for j in range(i + 1, len(bounds)) if first.overlaps(bounds[j])]
    checks = {
        "no_text_overlaps": not collisions,
        "text_clear_of_section_rules": not any(b.overlaps(line.get_window_extent(renderer)) for b in bounds for line in section_rules),
        "all_text_inside_canvas": all(fig.bbox.contains(b.x0, b.y0) and fig.bbox.contains(b.x1, b.y1) for b in bounds),
        "weight_values_clear_of_bars": not any(label.get_window_extent(renderer).overlaps(bar.get_window_extent(renderer)) for label in weight_labels for bar in weight_bars),
        "minimum_font_size_at_least_10_5_pt": min(a.get_fontsize() for a in text_artists) >= 10.5,
        "field_weights_sum_to_one": bool(np.isclose(sum(w for _, w in field), 1)),
        "final_weights_sum_to_one": bool(np.isclose(config.W_SEMANTIC + config.W_FIELD, 1)),
    }
    (DOC_DIR / "Figure_S5_layout_checks.json").write_text(json.dumps({"checks": checks, "text_collisions": collisions}, indent=2) + "\n")
    if not all(checks.values()):
        raise ValueError(f"Figure S5 layout checks failed: {checks}; collisions={collisions}")
    save(fig, 5)


def figure_s6() -> None:
    names = ["fig3c_retrieval_pairs_raw.csv", "fig3c_component_summary.csv", "fig3c_summary_metrics.csv", "fig3d_family_match_rates.csv", "fig3d_demo_retrieval_table.csv"]
    for name in names:
        shutil.copy2(PANEL / name, DATA_DIR / f"S6_{name}")
    pairs, components, metrics, families, demo = [pd.read_csv(PANEL / name) for name in names]
    # Check the frozen exports rather than silently fabricating zero rank changes.
    rank_delta = pd.to_numeric(pairs["rank_delta"], errors="raise")
    assert rank_delta.notna().all()
    assert np.array_equal(rank_delta, pairs["rank_sem"] - pairs["rank_flora"])
    pct = float(100 * rank_delta.ne(0).mean())
    assert np.isclose(pct, metrics.set_index("metric").loc["pct_reranked", "value"])
    component_columns = [("fs_pc", "q_has_pc"), ("fs_sol", "q_has_sol"), ("fs_wl", "q_has_wl")]
    for (_, row), (score, available) in zip(components.iterrows(), component_columns):
        values = pairs.loc[pairs[available], score]
        assert len(values) == row["n_values"]
        assert np.isclose(values.mean(), row["mean_score"])
        assert np.isclose(100 * values.gt(0).mean(), row["percent_nonzero"])

    with mpl.rc_context({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 12, "legend.fontsize": 12}):
        fig = plt.figure(figsize=(14, 11.5), facecolor="white")
        headers = []

        def heading(letter, title, x, y):
            headers.append(fig.text(x, y, letter, fontsize=19, fontweight="bold", color=INK, va="top"))
            headers.append(fig.text(x + 0.032, y - 0.002, title, fontsize=15, fontweight="bold", color=INK, va="top"))

        heading("a", "Photocatalyst-family alignment", 0.045, 0.975)
        heading("b", "How far did ranks change?", 0.55, 0.975)
        heading("c", "Field-score contributions", 0.045, 0.555)
        heading("d", "One iridium query: top five results", 0.55, 0.555)
        heading("e", "Keep the reference source out of retrieval", 0.045, 0.235)

        # Shared horizontal scale makes paired differences visible without a legend on the data.
        ax_a = fig.add_axes([0.17, 0.655, 0.31, 0.24])
        y = np.arange(len(families))
        semantic = families["semantic_match_rate_pct"].to_numpy()
        reranked = families["flowpilot_match_rate_pct"].to_numpy()
        ax_a.hlines(y, semantic, reranked, color=GRID, linewidth=3, zorder=2)
        ax_a.scatter(semantic, y, color=MUTED, s=65, zorder=3, label="Semantic only")
        ax_a.scatter(reranked, y, color=BLUE, s=65, zorder=3, label="FlowPilot")
        for yy, s, r in zip(y, semantic, reranked):
            ax_a.annotate(f"{s:.1f}", (s, yy), xytext=(-9, 0), textcoords="offset points",
                          ha="right", va="center", fontsize=11, color=MUTED)
            ax_a.annotate(f"{r:.1f}", (r, yy), xytext=(9, 0), textcoords="offset points",
                          ha="left", va="center", fontsize=11, color=BLUE)
        ax_a.set_yticks(y, [f"{r.family}\n(n = {r.n_queries} queries)" for r in families.itertuples()])
        ax_a.set_ylim(len(y) - 0.4, -0.65)
        ax_a.set_xlim(0, 118)
        ax_a.set_xticks([0, 25, 50, 75, 100])
        ax_a.set_xlabel("Top-5 family match rate (%)", labelpad=10)
        clean(ax_a, "x")
        legend = fig.legend(*ax_a.get_legend_handles_labels(), loc="upper left",
                            bbox_to_anchor=(0.16, 0.943), frameon=False, ncol=2,
                            columnspacing=1.5, handletextpad=0.5)

        ax_b = fig.add_axes([0.62, 0.655, 0.35, 0.24])
        counts = rank_delta.value_counts().sort_index()
        colors = [MUTED if d == 0 else TEAL if d > 0 else ORANGE for d in counts.index]
        ax_b.bar(counts.index, counts.values, width=0.8, color=colors, zorder=3)
        ax_b.set_yscale("log")
        ax_b.set_ylim(0.8, 2000)
        ax_b.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
        ax_b.minorticks_off()
        ax_b.set_xlim(-20, 20)
        ax_b.set_xticks([-20, -10, 0, 10, 20])
        ax_b.set_xlabel("Rank change: original rank - reranked rank", labelpad=10)
        ax_b.set_ylabel("Query-result pairs (log scale)", labelpad=9)
        clean(ax_b, "y")
        fig.text(0.62, 0.928, f"{pct:.1f}% changed rank  |  n = {len(pairs):,} pairs", fontsize=12, color=INK)
        fig.text(0.62, 0.583, "Negative: demoted", fontsize=11, color=ORANGE)
        fig.text(0.97, 0.583, "Positive: promoted", fontsize=11, color=TEAL, ha="right")

        # Different units have independent, explicitly labeled axes; no dual-axis bars.
        ax_c1 = fig.add_axes([0.17, 0.337, 0.14, 0.15])
        ax_c2 = fig.add_axes([0.35, 0.337, 0.13, 0.15])
        labels = ["Photocatalyst", "Solvent", "Wavelength"]
        yy = np.arange(len(components))
        ax_c1.barh(yy, components["mean_score"], height=0.40, color=BLUE, zorder=3)
        ax_c2.barh(yy, components["percent_nonzero"], height=0.40, color=TEAL, zorder=3)
        ax_c1.set_yticks(yy, [f"{name}\n(n = {int(n)})" for name, n in zip(labels, components["n_values"])])
        ax_c2.set_yticks(yy, [])
        ax_c1.set_xlim(0, 0.22)
        ax_c1.set_xticks([0, 0.1, 0.2], ["0", "0.10", "0.20"])
        ax_c2.set_xlim(0, 117)
        ax_c2.set_xticks([0, 50, 100])
        for ax, column, delta, fmt in [(ax_c1, "mean_score", 0.005, ".3f"),
                                       (ax_c2, "percent_nonzero", 3, ".1f")]:
            ax.set_ylim(2.65, -0.65)
            clean(ax, "x")
            for j, value in enumerate(components[column]):
                ax.text(value + delta, j, format(value, fmt), va="center", fontsize=11, color=INK)
        fig.text(0.17, 0.505, "Mean score", fontsize=12, fontweight="bold", color=BLUE)
        fig.text(0.35, 0.505, "Nonzero (%)", fontsize=12, fontweight="bold", color=TEAL)
        fig.text(0.17, 0.288, "n = pairs with the query field available", fontsize=11, color=MUTED)

        ax_d = fig.add_axes([0.58, 0.31, 0.39, 0.20])
        ax_d.set_xlim(0.4, 5.6)
        ax_d.set_ylim(0, 1)
        ax_d.axis("off")
        for rank in range(1, 6):
            ax_d.text(rank, 0.98, f"Rank {rank}", ha="center", va="top", fontsize=11, color=MUTED)
        methods = [("semantic_only", "Semantic only", 0.78, 0.59),
                   ("flowpilot", "FlowPilot reranked", 0.36, 0.17)]
        symbols = {"Iridium": (BLUE, "Ir"), "Ruthenium": (ORANGE, "Ru"), "": (MUTED, "?")}
        for method, label, label_y, dot_y in methods:
            subset = demo[demo["method"] == method].sort_values("rank")
            assert subset["rank"].tolist() == [1, 2, 3, 4, 5]
            ax_d.text(0.5, label_y, label, ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
            for item in subset.itertuples():
                family = "" if pd.isna(item.result_photocatalyst_family) else item.result_photocatalyst_family
                color, symbol = symbols[family]
                ax_d.scatter(item.rank, dot_y, s=520, facecolors=color if family else "white",
                             edgecolors=color, linewidths=1.5, zorder=3)
                ax_d.text(item.rank, dot_y, symbol, color="white" if family else MUTED,
                          ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)
        fig.text(0.58, 0.288, "Ir: iridium   Ru: ruthenium   ?: unassigned family", fontsize=11, color=MUTED)

        ax_e = fig.add_axes([0.045, 0.062, 0.925, 0.133])
        ax_e.set_xlim(0, 1)
        ax_e.set_ylim(0, 1)
        ax_e.axis("off")
        steps = [(0.11, "Retrieve candidates", "Build the candidate pool", BLUE),
                 (0.37, "Exclude reference", "Match normalized source IDs\nto the held-out case manifest", ORANGE),
                 (0.63, "Rank remaining", "Sort by final retrieval score", TEAL),
                 (0.89, "Return top-k", "Pass evidence to the designer", BLUE)]
        for i, (x, title, subtitle, color) in enumerate(steps):
            ax_e.scatter(x, 0.80, s=420, color=color, zorder=3)
            ax_e.text(x, 0.80, str(i + 1), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
            ax_e.text(x, 0.51, title, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
            ax_e.text(x, 0.20, subtitle, ha="center", va="center", fontsize=11, color=MUTED, linespacing=1.4)
            if i < len(steps) - 1:
                arrow(ax_e, (x + 0.028, 0.80), (steps[i + 1][0] - 0.028, 0.80), color=GRID)
        fig.text(0.045, 0.025, "Scope: metadata alignment, not reaction success. Source exclusion does not establish absence from model pretraining.",
                 fontsize=11, color=MUTED)
        for y_line in (0.566, 0.25):
            fig.add_artist(plt.Line2D([0.045, 0.97], [y_line, y_line], transform=fig.transFigure, color=GRID, linewidth=0.9))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_boxes = [(t, t.get_window_extent(renderer)) for t in fig.findobj(mpl.text.Text)
                      if t.get_visible() and t.get_text()]
        outside = [t.get_text() for t, b in text_boxes if not fig.bbox.contains(b.x0, b.y0)
                   or not fig.bbox.contains(b.x1, b.y1)]
        overlaps = [[t1.get_text(), t2.get_text()] for i, (t1, b1) in enumerate(text_boxes)
                    for t2, b2 in text_boxes[i + 1:] if b1.overlaps(b2)]
        checks = {"all_text_inside_canvas": not outside, "text_outside_canvas": outside,
                  "text_overlap_pairs": overlaps,
                  "legend_outside_data": not legend.get_window_extent(renderer).overlaps(ax_a.bbox),
                  "rank_delta_matches_raw_ranks": True, "component_summaries_match_raw_pairs": True,
                  "changed_pairs": int(rank_delta.ne(0).sum()), "total_pairs": len(pairs),
                  "pct_changed": pct, "missing_demo_family_fields": int(demo["result_photocatalyst_family"].isna().sum()),
                  "histogram_count_sum": int(counts.sum()), "count_axis_scale": "logarithmic"}
        (DOC_DIR / "Figure_S6_layout_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        assert not outside, outside
        assert not overlaps, overlaps
        assert checks["legend_outside_data"]
        save(fig, 6)


def figure_s7() -> None:
    from scripts.esi_benchmark_readability import figure_s7 as render
    render(BENCH, OUT)


def figure_s8() -> None:
    src = BENCH / "figures_revised" / "main" / "raw" / "figs5-2.csv"
    shutil.copy2(src, DATA_DIR / "S8_criterion_gain.csv")
    data = pd.read_csv(src)
    cases = ["CuAAC", "Photochemical oxidation", "Hydrogenolysis"]
    models = ["Qwen3.6-27B", "Qwen3.8-27B", "GPT-4o", "Claude Sonnet 4.6", "Claude Opus 4.6"]
    criteria = [f"UO-{i:02d}" for i in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14]]
    cmap = LinearSegmentedColormap.from_list("gain", ["#B84A45", "#F7F7F7", "#0B6B4F"])
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.4), sharex=True)
    for label, case, ax in zip("abc", cases, axes):
        subset = data[data["case"] == case]
        matrix = subset.pivot(index="model", columns="criterion_id", values="mean_paired_delta").reindex(index=models, columns=criteria)
        sns.heatmap(matrix, ax=ax, cmap=cmap, center=0, vmin=-0.15, vmax=0.65, annot=True, fmt="+.2f", annot_kws={"fontsize": 5.1}, cbar=False, linewidths=0.4, linecolor="white")
        ax.set_title(case, loc="left", fontweight="bold", color=INK, fontsize=8.2, pad=4)
        ax.set_ylabel(""); ax.set_xlabel("")
        ax.set_yticklabels([m.replace("Claude ", "").replace("-27B", "") for m in models], rotation=0, fontsize=5.8)
        panel_label(ax, label, x=-0.095, y=1.08)
        ax.tick_params(axis="x", labelrotation=0, labelsize=5.8)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=-0.15, vmax=0.65))
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035, pad=0.08, aspect=40)
    cbar.set_label("Benchmark score difference: FlowPilot - one-shot", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8)
    plt.subplots_adjust(hspace=0.37, left=0.17, right=0.98, bottom=0.15, top=0.97)
    save(fig, 8)


def _revise_existing_svg(source: Path, target: Path, replacements: dict[str, str]) -> None:
    text = source.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=text.encode(), write_to=str(target.with_suffix(".pdf")))
    cairosvg.svg2png(bytestring=text.encode(), write_to=str(target.with_suffix(".png")), output_width=6200)


def figures_s9_s10() -> None:
    from scripts.esi_benchmark_readability import figure_s9, figure_s10
    figure_s9(BENCH, OUT)
    figure_s10(BENCH, OUT)


def main() -> None:
    configure()
    ensure_dirs()
    figure_s1()
    figure_s2()
    figure_s3()
    figure_s4()
    figure_s5()
    figure_s6()
    figure_s7()
    figure_s8()
    figures_s9_s10()
    print(OUT)


if __name__ == "__main__":
    main()
