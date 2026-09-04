#!/usr/bin/env python3
"""Build the reviewed FlowPilot ESI figures and their source-data package."""

from __future__ import annotations

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
    fig = plt.figure(figsize=(7.4, 7.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], hspace=0.38, wspace=0.31)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a", x=-0.19, y=1.07)
    left = np.zeros(len(pivot))
    colors = {"hard_rule": RED, "guideline": BLUE, "tip": GREEN, "safety": ORANGE}
    for severity in [s for s in colors if s in pivot]:
        vals = pivot[severity].values
        ax.barh(np.arange(len(pivot)), vals, left=left, label=severity.replace("_", " "), color=colors[severity])
        left += vals
    ax.set_yticks(np.arange(len(pivot)), [fill(v, 19) for v in pivot.index])
    ax.set_xlabel("Rules")
    ax.set_title("Largest rule categories", fontweight="bold", color=INK, loc="left", pad=13)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.02), columnspacing=0.7, handlelength=1.2)
    clean(ax, "x")
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b", x=-0.18, y=1.07)
    top = formula.sort_values("total_rules", ascending=False).head(14).sort_values("percent_with_formula")
    ax.barh(np.arange(len(top)), top["percent_with_formula"], color=TEAL)
    ax.set_yticks(np.arange(len(top)), [fill(v, 19) for v in top["category_label"]])
    ax.set_xlim(80, 101)
    ax.set_xlabel("Rules with detected quantitative expression (%)")
    ax.set_title("Quantitative-expression coverage", fontweight="bold", color=INK)
    clean(ax, "x")
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c", x=-0.12, y=1.15)
    h = heat.sort_values("row_total", ascending=False).head(13).set_index("category_label")
    values = h.drop(columns=["category_key", "row_total"], errors="ignore")
    sns.heatmap(np.log1p(values), ax=ax, cmap="YlGnBu", cbar_kws={"label": "log(1 + association count)", "shrink": 0.78}, linewidths=0.25, linecolor="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=48, ha="right", fontsize=5.6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5.9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Rule-category associations by chemistry class", fontweight="bold", color=INK, pad=7)
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d", x=-0.19, y=1.06)
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
    pos = nx.circular_layout(order, scale=1.0)
    freqs = np.array([graph.nodes[n]["frequency"] for n in graph.nodes])
    sizes = 35 + 420 * np.sqrt(freqs / freqs.max())
    category_colors = {cat: [BLUE, TEAL, ORANGE, PURPLE, GREEN, RED][i % 6] for i, cat in enumerate(sorted({graph.nodes[n]["category"] for n in graph.nodes}))}
    node_colors = [category_colors[graph.nodes[n]["category"]] for n in graph.nodes]
    max_weight = max([graph.edges[e]["weight"] for e in graph.edges] or [1])
    widths = [0.25 + 2.4 * graph.edges[e]["weight"] / max_weight for e in graph.edges]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, edge_color=GRID, alpha=0.85)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes, node_color=node_colors, edgecolors="white", linewidths=0.7)
    label_pos = {node: (xy[0] * 1.24, xy[1] * 1.24) for node, xy in pos.items()}
    artists = nx.draw_networkx_labels(graph, label_pos, labels={n: fill(n, 13) for n in graph.nodes}, ax=ax, font_size=4.7, font_color=INK)
    for artist in artists.values():
        artist.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35})
    ax.set_xlim(-1.50, 1.50)
    ax.set_ylim(-1.44, 1.44)
    ax.set_title("Engineering concept co-occurrence", fontweight="bold", color=INK, pad=7)
    ax.axis("off")
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
    fig = plt.figure(figsize=(7.4, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.14], width_ratios=[1.05, 0.95], hspace=0.30, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "a", x=-0.02, y=1.03)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Plan-aware query construction and provider fallback", fontweight="bold", color=INK)
    items = ["Reaction class", "Mechanism", "Catalyst", "Solvent", "T / wavelength", "Intermediate", "Bond", "Keywords"]
    for i, item in enumerate(items):
        row, col = divmod(i, 4)
        box(ax, 0.08 + col * 0.11, 0.73 - row * 0.31, 0.10, 0.18, item, fc=BLUE_L, ec=BLUE, fs=5.4, bold=True)
    arrow(ax, (0.45, 0.58), (0.54, 0.58), color=BLUE, lw=1.3)
    box(ax, 0.64, 0.58, 0.19, 0.31, "Plan-aware\nrich query", fc=TEAL_L, ec=TEAL, fs=7.6, bold=True)
    arrow(ax, (0.74, 0.58), (0.80, 0.58), color=TEAL, lw=1.3)
    box(ax, 0.89, 0.74, 0.18, 0.17, "Embedding\nprovider", fc=PURPLE_L, ec=PURPLE, fs=6.8, bold=True)
    box(ax, 0.89, 0.37, 0.18, 0.17, "Deterministic\nlexical fallback", fc=ORANGE_L, ec=ORANGE, fs=6.4, bold=True)
    arrow(ax, (0.89, 0.65), (0.89, 0.47), color=RED)
    ax.text(0.91, 0.56, "provider failure", fontsize=5.7, color=RED)
    ax.text(0.31, 0.08, "ChemistryPlan + BatchRecord", fontsize=6.5, color=MUTED, ha="center")
    ax.text(0.89, 0.08, "Same query and metadata filters", fontsize=6.5, color=MUTED, ha="center")
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "b", x=-0.11, y=1.04)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Three-stage retrieval scope", fontweight="bold", color=INK)
    tiers = [(0.82, "Tier 1\nPairs only\nmechanism + phase", GREEN_L, GREEN), (0.54, "Tier 2\nPairs only\nfilters relaxed", BLUE_L, BLUE), (0.26, "Tier 3\nAll records\nno metadata filters", ORANGE_L, ORANGE)]
    for i, (y, text, fc, ec) in enumerate(tiers):
        box(ax, 0.35, y, 0.55, 0.19, text, fc=fc, ec=ec, fs=6.6, bold=True)
        if i < 2:
            arrow(ax, (0.35, y - 0.105), (0.35, tiers[i + 1][0] + 0.105), color=ec)
            ax.text(0.68, (y + tiers[i + 1][0]) / 2, "<3 hits" if i == 0 else "0 hits", va="center", fontsize=5.8, color=ec, fontweight="bold")
    box(ax, 0.82, 0.26, 0.23, 0.19, "Exclude hidden\nsource IDs", fc=RED_L, ec=RED, fs=6.2, bold=True)
    arrow(ax, (0.64, 0.26), (0.695, 0.26), color=RED)
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c", x=-0.14, y=1.04)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Reranking calculation", fontweight="bold", color=INK)
    box(ax, 0.50, 0.88, 0.78, 0.14, "semantic = max(0, 1 - L2^2 / 2)", fc=PURPLE_L, ec=PURPLE, fs=6.6, bold=True)
    field = [(c, w) for c, w, scope in rows if scope == "field score"]
    for (name, weight), y in zip(field, np.linspace(0.68, 0.35, len(field))):
        ax.text(0.06, y, name, va="center", fontsize=6.3, color=INK)
        ax.add_patch(patches.Rectangle((0.37, y - 0.025), weight / 0.30 * 0.38, 0.05, facecolor=TEAL, edgecolor="none"))
        ax.text(0.82, y, f"{weight:.2f}", va="center", ha="right", fontsize=6.2, color=TEAL, fontweight="bold")
    box(ax, 0.50, 0.17, 0.84, 0.15, "final = 0.60 x semantic + 0.40 x field", fc=GREEN_L, ec=GREEN, fs=6.7, bold=True)
    ax.text(0.50, 0.035, "Sort descending; return top-k", ha="center", fontsize=6.4, color=MUTED)
    save(fig, 5)


def figure_s6() -> None:
    names = ["fig3c_retrieval_pairs_raw.csv", "fig3c_component_summary.csv", "fig3c_summary_metrics.csv", "fig3d_family_match_rates.csv", "fig3d_demo_retrieval_table.csv"]
    for name in names:
        shutil.copy2(PANEL / name, DATA_DIR / f"S6_{name}")
    pairs, components, metrics, families, demo = [pd.read_csv(PANEL / name) for name in names]
    rank_col = next((c for c in pairs.columns if "rank" in c.lower() and ("change" in c.lower() or "delta" in c.lower())), None)
    rank_delta = pairs[rank_col].fillna(0).astype(float) if rank_col else pd.Series(np.zeros(len(pairs)))
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.94, 0.94, 0.72], hspace=0.50, wspace=0.38)
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a", x=-0.17, y=1.08)
    x = np.arange(len(families)); width = 0.36
    ax.bar(x - width / 2, families["semantic_match_rate_pct"], width, color=MUTED, label="Semantic only")
    ax.bar(x + width / 2, families["flowpilot_match_rate_pct"], width, color=BLUE, label="FlowPilot")
    ax.set_xticks(x, families["family"], rotation=27, ha="right")
    ax.set_ylim(0, 125); ax.set_ylabel("Top-5 family match rate (%)")
    ax.set_title("Photocatalyst-family alignment", fontweight="bold", color=INK, loc="left", pad=9)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    clean(ax, "y")
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, "b", x=-0.17, y=1.08)
    bins = np.arange(math.floor(rank_delta.min()) - 0.5, math.ceil(rank_delta.max()) + 1.5, 1)
    ax.hist(rank_delta, bins=bins, color=TEAL, edgecolor="white")
    ax.axvline(0, color=INK, linewidth=0.9)
    pct = float(metrics.loc[metrics["metric"] == "pct_reranked", "value"].iloc[0])
    ax.text(0.97, 0.92, f"{pct:.1f}% changed rank\nn = {len(pairs):,} pairs", transform=ax.transAxes, ha="right", va="top", fontsize=6.6, bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, edgecolor=GRID))
    ax.set_xlabel("Rank change after field reranking"); ax.set_ylabel("Query-result pairs")
    ax.set_title("Reranking displacement", fontweight="bold", color=INK); clean(ax, "y")
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "c", x=-0.17, y=1.08)
    comp = components.copy(); x = np.arange(len(comp))
    ax.bar(x - 0.18, comp["mean_score"], 0.36, color=PURPLE, label="Mean score")
    ax2 = ax.twinx(); ax2.bar(x + 0.18, comp["percent_nonzero"], 0.36, color=ORANGE, label="Nonzero rate")
    ax.set_xticks(x, [fill(v.replace(" Match", ""), 12) for v in comp["component"]], rotation=18, ha="right")
    ax.set_ylabel("Mean component score", labelpad=4); ax2.set_ylabel("Nonzero observations (%)", labelpad=7)
    ax.set_ylim(0, max(0.18, float(comp["mean_score"].max()) * 1.22)); ax2.set_ylim(0, 105)
    ax.set_title("Available field-score components", fontweight="bold", color=INK, loc="left", pad=9)
    ax.legend(handles=[patches.Patch(color=PURPLE, label="Mean score"), patches.Patch(color=ORANGE, label="Nonzero rate")], frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    clean(ax, "y"); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_color(GRID)
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "d", x=-0.17, y=1.08)
    ax.set_title("Representative iridium query: top five", fontweight="bold", color=INK)
    methods = [("semantic_only", "Semantic only"), ("flowpilot", "FlowPilot reranked")]
    color_map = {"Iridium": BLUE, "Ruthenium": RED, "Organic dye": ORANGE, "": MUTED}
    for row_index, (method, label) in enumerate(methods):
        subset = demo[demo["method"] == method].sort_values("rank").head(5); y = 1 - row_index
        ax.text(0.01, y, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.2, fontweight="bold")
        for _, item in subset.iterrows():
            rank = int(item["rank"]); family = str(item.get("result_photocatalyst_family") or "")
            if family == "nan": family = ""
            ax.scatter(rank, y, s=70, c=color_map.get(family, MUTED), marker="o" if family == "Iridium" else "X", edgecolors="white", linewidths=0.6, zorder=3)
            ax.text(rank, y - 0.21, family or "unknown", ha="center", va="top", fontsize=5.0, color=MUTED, rotation=18)
    ax.set_xlim(0.35, 5.65); ax.set_ylim(-0.55, 1.55); ax.set_xticks(range(1, 6)); ax.set_yticks([]); ax.set_xlabel("Rank")
    clean(ax, "x")
    ax = fig.add_subplot(gs[2, :]); panel_label(ax, "e", x=-0.02, y=1.10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Leave-one-source-out and deterministic retrieval controls", fontweight="bold", color=INK)
    controls = [(0.10, "Hidden source ID\nfrom case manifest", BLUE_L, BLUE), (0.31, "Retrieve candidate\npool", TEAL_L, TEAL), (0.52, "Normalize and exclude\nmatching IDs", RED_L, RED), (0.73, "Sort remaining\nfinal scores", ORANGE_L, ORANGE), (0.92, "Return\ntop-k", GREEN_L, GREEN)]
    for i, (x, label, fc, ec) in enumerate(controls):
        w = 0.17 if i < 4 else 0.11
        box(ax, x, 0.52, w, 0.32, label, fc=fc, ec=ec, fs=6.0, bold=True)
        if i < len(controls) - 1:
            next_x = controls[i + 1][0]; next_w = 0.17 if i + 1 < 4 else 0.11
            arrow(ax, (x + w / 2 + 0.004, 0.52), (next_x - next_w / 2 - 0.004, 0.52), color=MUTED)
    ax.text(0.50, 0.08, "Automated controls cover hidden-ID exclusion and embedding-provider lexical fallback.", ha="center", fontsize=6.3, color=MUTED)
    save(fig, 6)


def figure_s7() -> None:
    src = BENCH / "figures_revised" / "main" / "raw" / "figs5-1.csv"
    shutil.copy2(src, DATA_DIR / "S7_case_scores.csv")
    data = pd.read_csv(src)
    cases = ["CuAAC", "Photochemical oxidation", "Hydrogenolysis"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 4.15), sharex=True)
    for label, case, ax in zip("abc", cases, axes):
        subset = data[data["case"] == case].sort_values("mean")
        y = np.arange(len(subset))
        colors = [TEAL if a == "FlowPilot" else RED for a in subset["architecture"]]
        for xx, yy, err, color in zip(subset["mean"], y, subset["sample_sd"], colors):
            ax.errorbar(xx, yy, xerr=err, fmt="none", ecolor=color, elinewidth=1.0, capsize=2, zorder=1)
        ax.scatter(subset["mean"], y, c=colors, s=20, edgecolors="white", linewidths=0.5, zorder=2)
        short_labels = [str(v).replace(" | FlowPilot", " | FP").replace(" | One-shot", " | OS").replace("Claude ", "") for v in subset["display_label"]]
        ax.set_yticks(y, short_labels, fontsize=5.3)
        ax.set_xlim(0.55, 1.10); ax.set_title(case, fontweight="bold", color=INK, fontsize=8.2)
        panel_label(ax, label, x=-0.19, y=1.08)
        ax.set_xlabel("Benchmark score")
        for yy, value in zip(y, subset["mean"]):
            ax.text(1.085, yy, f"{value:.2f}", ha="right", va="center", fontsize=4.8, color=INK)
        clean(ax, "x")
    fig.legend(handles=[patches.Patch(color=RED, label="One-shot"), patches.Patch(color=TEAL, label="FlowPilot")], frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    plt.subplots_adjust(wspace=0.62, top=0.84, left=0.15, right=0.98, bottom=0.12)
    save(fig, 7)


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
    s9 = BENCH / "figures_revised" / "fig08a_campaign_error_map.svg"
    s10 = BENCH / "figures_revised" / "fig18a_flowpilot_campaign_cost_efficiency.svg"
    _revise_existing_svg(s9, FIG_DIR / "Figure_S09.svg", {">A  One-shot<": ">a  One-shot<", ">B  FlowPilot<": ">b  FlowPilot<"})
    _revise_existing_svg(s10, FIG_DIR / "Figure_S10.svg", {">A  Token use<": ">a  Token use<", ">B  Generation cost<": ">b  Generation cost<", ">C  Observed runtime<": ">c  Observed runtime<", ">D1  CuAAC<": ">d1  CuAAC<", ">D2  Photochemical oxidation<": ">d2  Photochemical oxidation<", ">D3  Hydrogenolysis<": ">d3  Hydrogenolysis<"})
    shutil.copy2(BENCH / "figures_revised" / "main" / "raw" / "figs5-3_error_map.csv", DATA_DIR / "S9_error_map.csv")
    shutil.copy2(BENCH / "figures_revised" / "main" / "raw" / "figs5-3_error_details.csv", DATA_DIR / "S9_error_details.csv")
    shutil.copy2(BENCH / "figures_revised" / "main" / "raw" / "figs5-4_campaigns.csv", DATA_DIR / "S10_campaigns.csv")
    shutil.copy2(BENCH / "figures_revised" / "main" / "raw" / "figs5-4_resource_summary.csv", DATA_DIR / "S10_resource_summary.csv")


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
