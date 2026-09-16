"""Build publication-ready FlowPilot ESI Figures S1-S10 and source package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "deliverables" / "flowpilot_esi_figures_s1_s10_20260811"
FIG_DIR = OUT / "figures"
DATA_DIR = OUT / "source_data"
META_DIR = OUT / "documentation"

PANEL_DATA = ROOT / "visualization" / "panel_data_exports"
CASE_ROOT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "khu_three_protocols_canonical_20260810_161346"
    / "case_01_photoredox_giese_aerobic_oxidation"
)
INTAKE_PATH = CASE_ROOT / "intake_package.json"
RESULT_PATH = CASE_ROOT / "full_result.json"
EVENTS_PATH = CASE_ROOT / "llm_events.jsonl"

# Accessible, colorblind-safe palette with restrained saturation.
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
YELLOW = "#E0AE35"
YELLOW_L = "#F8EFCF"

COLORS = [BLUE, TEAL, ORANGE, PURPLE, GREEN, RED, YELLOW]


CAPTIONS: dict[int, str] = {
    1: (
        "Extended FlowPilot architecture and authority boundaries. (a) The current "
        "pipeline converts a standardized intake package into a chemistry plan, "
        "retrieves literature analogies, performs deterministic engineering and "
        "design-space calculations, generates and audits candidates through the "
        "multi-agent council, reconciles the selected design against laboratory "
        "inventory, validates a single final-design contract, and renders topology "
        "and provenance artifacts. (b) Evidence authority is ordered from measured "
        "experimental evidence to model inference. Higher-authority information may "
        "constrain or override lower-authority suggestions. (c) Deterministic final "
        "gates separate executable screening designs from inventory-confirmation and "
        "blocked diagnostic outputs. LLM modules interpret chemistry; deterministic "
        "modules own numerical closure and feasibility."
    ),
    2: (
        "Typed contracts and field ownership. (a) Major Pydantic data objects passed "
        "through FlowPilot and their field counts in the current code. (b) Primary "
        "ownership of representative design fields. A filled cell denotes the module "
        "that creates, computes, constrains, or publishes the field. (c) Candidate "
        "revisions are not published directly: editable fields are recomputed, "
        "inventory-reconciled, and checked before entering the final contract."
    ),
    3: (
        "Reproducible standardized intake. (a) Fixed question bank with stable IDs, "
        "required/optional status, and the design context populated by each answer. "
        "(b) Readiness state machine. Batch protocol and objective must be answered; "
        "history, inventory, limits, and hypotheses must be answered or explicitly "
        "marked unavailable. (c) Pending question IDs are a deterministic function of "
        "package state, while the LLM is limited to extracting content and cannot "
        "invent new IDs."
    ),
    4: (
        "Frozen DesignInputPackage and evidence propagation. (a) Section-level status "
        "for the representative photoredox case used in this package. (b) Authority-"
        "labeled influence matrix showing where protocol facts, measured evidence, "
        "inventory, operating limits, hypotheses, and preferences enter the pipeline. "
        "(c) The package is serialized with schema version and content hash before "
        "design, providing a reproducible boundary between chemist input and model "
        "inference. Empty historical data are recorded as unavailable rather than "
        "silently omitted."
    ),
    5: (
        "GUI workflow and single-source result rendering. (a) User workflow from "
        "protocol intake and inventory selection to design execution and review. "
        "(b) Ten result tabs in the standardized-intake design view. All numerical "
        "tabs are rebuilt from the post-validation FinalDesignContract; supporting "
        "traces remain diagnostic. (c) Executable and blocked paths use different "
        "rendering rules: blocked runs expose requirements topology and reconciliation "
        "reasons but withhold run parameters."
    ),
    6: (
        "Run-level provenance and artifact integrity. (a) Artifact classes stored for "
        "the representative photoredox run. (b) Provenance chain from frozen input and "
        "model events through raw and canonical results, deterministic audit, topology, "
        "and checksums. (c) All nine final validation checks passed in the selected "
        "executable example. (d) Distribution of 57 recorded LLM events by pipeline "
        "component. Artifact counts describe this stored case and are not architecture "
        "requirements."
    ),
    7: (
        "Composition of the frozen flow-chemistry corpus (n = 464 classified records). "
        "Distributions are shown for (a) reaction class, (b) reactor type, (c) reactor "
        "material, (d) bond type, (e) number of inlet streams, and (f) available batch "
        "and optimized-flow yields. Unknown/other categories are retained to expose "
        "metadata incompleteness. Percentages use the available denominator for each "
        "field; yield distributions therefore do not imply complete yield coverage."
    ),
    8: (
        "Engineering rule-base structure (2,537 rules). (a) Counts by category and "
        "severity for the 14 largest categories. (b) Fraction of rules containing a "
        "machine-detected quantitative expression; this is an expression-coverage "
        "indicator, not proof that every expression is an independently validated "
        "equation. (c) Rule-category associations across chemistry classes. Cell "
        "values are association counts and may exceed category totals because one rule "
        "can map to multiple chemistry classes. (d) Co-occurrence network of the most "
        "frequent engineering concepts, with node size proportional to frequency and "
        "edge width proportional to co-occurrence."
    ),
    9: (
        "Current plan-aware retrieval workflow. (a) ChemistryPlan fields enrich the "
        "query before embedding or deterministic lexical retrieval. (b) Retrieval "
        "starts with mechanism/phase filters on paired records, relaxes filters when "
        "fewer than three hits are found, and finally searches all records when needed. "
        "(c) Candidate records are reranked by 0.60 semantic similarity and 0.40 field "
        "similarity. Field similarity combines photocatalyst, solvent, wavelength, "
        "temperature, and concentration terms using the current configuration. "
        "Leave-one-source-out evaluation excludes hidden reference record IDs after "
        "candidate construction and before top-k selection."
    ),
    10: (
        "Retrieval benchmark and leakage controls. (a) Top-5 photocatalyst-family match "
        "rates for semantic retrieval and FlowPilot reranking. (b) Distribution of "
        "rank changes across 1,600 frozen query-result pairs; 25.1% changed rank. "
        "(c) Mean and nonzero rates for available field-score components. (d) One "
        "iridium-query example showing the family composition of the top five before "
        "and after reranking. (e) Leave-one-source-out control and deterministic test "
        "coverage. These retrieval metrics measure metadata alignment, not downstream "
        "chemical yield or design optimality."
    ),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.1,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_dirs() -> None:
    for path in (OUT, FIG_DIR, DATA_DIR, META_DIR):
        path.mkdir(parents=True, exist_ok=True)


def panel_label(ax, label: str, *, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )


def clean_ax(ax, *, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(color=GRID, labelcolor=INK, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = LIGHT,
    ec: str = GRID,
    fontsize: float = 7.4,
    weight: str = "normal",
    color: str = INK,
    radius: float = 0.02,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        zorder=zorder + 1,
        linespacing=1.15,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    lw: float = 1.1,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def save_figure(fig: plt.Figure, number: int) -> None:
    stem = f"Figure_S{number:02d}"
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=400, facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = DATA_DIR / name
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_source(name: str) -> Path:
    source = PANEL_DATA / name
    destination = DATA_DIR / name
    shutil.copy2(source, destination)
    return destination


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_field_count(model_name: str) -> int:
    from flora_translate import schemas

    model = getattr(schemas, model_name)
    return len(model.model_fields)


def figure_s1() -> None:
    modules = [
        ("Standardized\nintake", "Human + LLM", BLUE_L, BLUE),
        ("Batch parser", "LLM", PURPLE_L, PURPLE),
        ("Chemistry\nanalysis", "LLM + rules", PURPLE_L, PURPLE),
        ("Plan-aware\nretrieval", "Code + embeddings", TEAL_L, TEAL),
        ("9-step calculator\n+ design space", "Deterministic", TEAL_L, TEAL),
        ("Flow proposal", "LLM", PURPLE_L, PURPLE),
        ("7-agent\ncouncil", "LLM + gates", ORANGE_L, ORANGE),
        ("Evidence + inventory\nreconciliation", "Deterministic", TEAL_L, TEAL),
        ("Final validation\n+ contract", "Deterministic", TEAL_L, TEAL),
        ("Topology +\nautosave", "Deterministic", GREEN_L, GREEN),
    ]
    write_csv(
        "S1_pipeline_modules.csv",
        [
            {"order": i + 1, "module": m[0].replace("\n", " "), "execution_type": m[1]}
            for i, m in enumerate(modules)
        ],
    )
    authority = [
        ("Measured experimental evidence", 5, BLUE),
        ("Hard inventory and safety constraints", 4, RED),
        ("Batch protocol facts", 3, TEAL),
        ("Chemist hypotheses", 2, ORANGE),
        ("Model inference", 1, PURPLE),
    ]
    write_csv(
        "S1_authority_order.csv",
        [{"authority": a, "rank": rank} for a, rank, _ in authority],
    )

    fig = plt.figure(figsize=(7.4, 7.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 0.88], hspace=0.22, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a", x=-0.025, y=1.02)
    ax.set_title("Current end-to-end execution path", fontweight="bold", color=INK, pad=3)
    xs = np.linspace(0.07, 0.93, 5)
    ys = [0.72, 0.30]
    positions: list[tuple[float, float]] = []
    for row, y in enumerate(ys):
        row_xs = xs if row == 0 else xs[::-1]
        for x in row_xs:
            positions.append((float(x), y))
    for i, ((title, kind, fc, ec), (x, y)) in enumerate(zip(modules, positions)):
        box(ax, x, y, 0.155, 0.22, f"{i+1}\n{title}\n{kind}", fc=fc, ec=ec, fontsize=6.8, weight="bold")
        if i < len(modules) - 1:
            nx_, ny_ = positions[i + 1]
            if abs(y - ny_) < 0.01:
                direction = 1 if nx_ > x else -1
                arrow(ax, (x + direction * 0.08, y), (nx_ - direction * 0.08, ny_), color=ec)
            else:
                arrow(ax, (x, y - 0.12), (nx_, ny_ + 0.12), color=ec, connectionstyle="arc3,rad=-0.16")
    legend = [
        patches.Patch(facecolor=PURPLE_L, edgecolor=PURPLE, label="LLM interpretation"),
        patches.Patch(facecolor=TEAL_L, edgecolor=TEAL, label="Deterministic computation"),
        patches.Patch(facecolor=BLUE_L, edgecolor=BLUE, label="Human/intake boundary"),
        patches.Patch(facecolor=GREEN_L, edgecolor=GREEN, label="Published artifacts"),
    ]
    ax.legend(handles=legend, ncol=4, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.02))

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "b", x=-0.13, y=1.04)
    ax.set_title("Information authority", fontweight="bold", color=INK, pad=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for idx, (label, rank, col) in enumerate(authority):
        y = 5.35 - idx * 1.03
        width = 0.42 + rank * 0.08
        box(ax, 0.48, y, width, 0.63, f"{rank}  {label}", fc=mpl.colors.to_rgba(col, 0.12), ec=col, fontsize=7.1, weight="bold")
        if idx < len(authority) - 1:
            arrow(ax, (0.48, y - 0.34), (0.48, y - 0.67), color=MUTED)
    ax.text(0.48, 0.12, "Higher-authority evidence constrains lower-authority proposals", ha="center", va="bottom", fontsize=6.8, color=MUTED)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c", x=-0.10, y=1.04)
    ax.set_title("Final deterministic disposition", fontweight="bold", color=INK, pad=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.5, 0.82, 0.50, 0.14, "Final-design contract\n+ validation checks", fc=TEAL_L, ec=TEAL, weight="bold")
    arrow(ax, (0.5, 0.75), (0.5, 0.63), color=TEAL)
    box(ax, 0.5, 0.55, 0.40, 0.13, "All critical gates pass?", fc=LIGHT, ec=MUTED, weight="bold")
    arrow(ax, (0.32, 0.49), (0.18, 0.36), color=GREEN)
    arrow(ax, (0.68, 0.49), (0.82, 0.36), color=RED)
    ax.text(0.22, 0.44, "YES", color=GREEN, fontsize=7, fontweight="bold")
    ax.text(0.74, 0.44, "NO", color=RED, fontsize=7, fontweight="bold")
    box(ax, 0.18, 0.25, 0.30, 0.20, "SCREEN\nExecutable topology\n+ run parameters", fc=GREEN_L, ec=GREEN, weight="bold")
    box(ax, 0.82, 0.25, 0.30, 0.20, "BLOCKED / CONFIRM\nRequirements topology\n+ reasons only", fc=RED_L, ec=RED, weight="bold")
    ax.text(0.5, 0.035, "Model confidence never overrides deterministic feasibility", ha="center", fontsize=6.8, color=MUTED)
    fig.suptitle("Figure S1 | FlowPilot architecture, authority, and final disposition", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 1)


def figure_s2() -> None:
    model_names = [
        "BatchRecord",
        "ChemistryPlan",
        "FlowProposal",
        "DesignCalculations",
        "ProcessTopology",
        "DesignInputPackage",
    ]
    counts = {}
    for name in model_names:
        try:
            counts[name] = model_field_count(name)
        except (AttributeError, TypeError):
            counts[name] = 0
    # FinalDesignContract is currently a deterministic dictionary contract.
    contracts = [
        ("DesignInputPackage", counts.get("DesignInputPackage", 0), "intake"),
        ("BatchRecord", counts.get("BatchRecord", 0), "parser"),
        ("ChemistryPlan", counts.get("ChemistryPlan", 0), "chemistry"),
        ("FlowProposal", counts.get("FlowProposal", 0), "translation"),
        ("DesignCalculations", counts.get("DesignCalculations", 0), "calculator"),
        ("ProcessTopology", counts.get("ProcessTopology", 0), "compiler"),
        ("FinalDesignContract", 5, "publisher sections"),
    ]
    write_csv(
        "S2_contracts.csv",
        [{"order": i + 1, "contract": n, "field_count": c, "created_by": owner} for i, (n, c, owner) in enumerate(contracts)],
    )
    fields = [
        "protocol facts",
        "mechanism / stages",
        "retrieval analogies",
        "residence time",
        "reactor geometry",
        "gas basis",
        "temperature / BPR",
        "inventory IDs",
        "candidate score",
        "disposition",
        "topology",
    ]
    owners = ["Intake", "Chemistry", "Retriever", "Calculator", "Council", "Inventory", "Final contract"]
    matrix = np.zeros((len(fields), len(owners)))
    ownership = {
        "protocol facts": [0, 6],
        "mechanism / stages": [1, 4, 6],
        "retrieval analogies": [2, 4, 6],
        "residence time": [3, 4, 5, 6],
        "reactor geometry": [3, 4, 5, 6],
        "gas basis": [3, 4, 5, 6],
        "temperature / BPR": [3, 4, 5, 6],
        "inventory IDs": [5, 6],
        "candidate score": [4, 6],
        "disposition": [5, 6],
        "topology": [5, 6],
    }
    for i, field in enumerate(fields):
        for j in ownership[field]:
            matrix[i, j] = 1
    write_csv(
        "S2_field_ownership.csv",
        [
            {"field": field, **{owner: int(matrix[i, j]) for j, owner in enumerate(owners)}}
            for i, field in enumerate(fields)
        ],
    )

    fig = plt.figure(figsize=(7.4, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.22], width_ratios=[1.5, 0.8], hspace=0.30, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "a", x=-0.025, y=1.02)
    ax.set_title("Typed objects passed through the pipeline", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xs = np.linspace(0.07, 0.93, len(contracts))
    for i, ((name, count, owner), x) in enumerate(zip(contracts, xs)):
        fc, ec = (PURPLE_L, PURPLE) if owner in {"chemistry", "translation"} else (TEAL_L, TEAL)
        if owner == "intake":
            fc, ec = BLUE_L, BLUE
        if owner == "publisher sections":
            fc, ec = GREEN_L, GREEN
        count_label = f"{count} fields" if count else "typed object"
        if name == "FinalDesignContract":
            count_label = "5 sections"
        box(ax, float(x), 0.54, 0.12, 0.38, f"{name}\n\n{count_label}\n{owner}", fc=fc, ec=ec, fontsize=6.1, weight="bold")
        if i < len(contracts) - 1:
            arrow(ax, (x + 0.062, 0.54), (xs[i + 1] - 0.062, 0.54), color=MUTED)
    ax.text(0.5, 0.15, "Typed validation occurs at module boundaries; the final published contract is rebuilt after validation", ha="center", fontsize=7, color=MUTED)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "b", x=-0.10, y=1.03)
    ax.set_title("Representative field ownership", fontweight="bold", color=INK)
    cmap = mpl.colors.ListedColormap(["#FFFFFF", BLUE])
    sns.heatmap(matrix, ax=ax, cmap=cmap, cbar=False, linewidths=0.6, linecolor=GRID, xticklabels=owners, yticklabels=fields, square=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=38, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for i in range(len(fields)):
        for j in range(len(owners)):
            if matrix[i, j]:
                ax.text(j + 0.5, i + 0.5, "●", ha="center", va="center", color="white", fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c", x=-0.18, y=1.03)
    ax.set_title("Revision closure", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        (0.86, "Specialist proposes\nbounded edit", PURPLE_L, PURPLE),
        (0.65, "Recompute all\ndependent quantities", TEAL_L, TEAL),
        (0.44, "Reconcile exact\ninventory hardware", ORANGE_L, ORANGE),
        (0.23, "Validate arithmetic,\ngas, topology, limits", TEAL_L, TEAL),
        (0.05, "Publish or block", GREEN_L, GREEN),
    ]
    for i, (y, label, fc, ec) in enumerate(steps):
        box(ax, 0.5, y, 0.76, 0.13, label, fc=fc, ec=ec, fontsize=7.2, weight="bold")
        if i < len(steps) - 1:
            arrow(ax, (0.5, y - 0.07), (0.5, steps[i + 1][0] + 0.07), color=MUTED)
    fig.suptitle("Figure S2 | Typed contracts and field ownership", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 2)


def figure_s3() -> None:
    from flora_translate.intake_agent import QUESTION_BANK, REQUIRED_READINESS_IDS

    questions = []
    for qid, q in QUESTION_BANK.items():
        questions.append(
            {
                "question_id": qid,
                "section": q.section,
                "required_for_readiness": qid in REQUIRED_READINESS_IDS,
                "expected_format": q.expected_format,
                "why_needed": q.why_needed,
            }
        )
    write_csv("S3_question_bank.csv", questions)

    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 0.85], hspace=0.28, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "a", x=-0.025, y=1.02)
    ax.set_title("Fixed intake question bank", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    section_colors = [BLUE, BLUE, TEAL, ORANGE, RED, PURPLE, GREEN]
    for i, (row, col) in enumerate(zip(questions, section_colors)):
        y = 0.90 - i * 0.125
        ax.add_patch(FancyBboxPatch((0.04, y - 0.045), 0.92, 0.09, boxstyle="round,pad=0.006,rounding_size=0.012", facecolor=mpl.colors.to_rgba(col, 0.09), edgecolor=col, linewidth=0.9))
        ax.text(0.065, y, row["question_id"], va="center", ha="left", fontsize=7.4, fontweight="bold", color=col)
        ax.text(0.225, y + 0.012, row["section"].replace("_", " ").title(), va="center", ha="left", fontsize=7.3, fontweight="bold", color=INK)
        ax.text(0.225, y - 0.020, fill(row["why_needed"], 75), va="center", ha="left", fontsize=6.2, color=MUTED)
        req = "REQUIRED" if row["required_for_readiness"] else "OPTIONAL"
        ax.text(0.93, y, req, va="center", ha="right", fontsize=6.2, fontweight="bold", color=RED if req == "REQUIRED" else GREEN)
    ax.text(0.5, 0.015, "The LLM may extract answers and select missing questions, but it cannot create new question identifiers", ha="center", fontsize=6.8, color=MUTED)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "b", x=-0.12, y=1.04)
    ax.set_title("Readiness state machine", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.18, 0.72, 0.28, 0.18, "Raw protocol\nor saved package", fc=BLUE_L, ec=BLUE, weight="bold")
    box(ax, 0.52, 0.72, 0.28, 0.18, "Evaluate six\nrequired IDs", fc=LIGHT, ec=MUTED, weight="bold")
    arrow(ax, (0.33, 0.72), (0.37, 0.72), color=MUTED)
    box(ax, 0.52, 0.34, 0.30, 0.18, "Missing IDs\nAsk in fixed order", fc=ORANGE_L, ec=ORANGE, weight="bold")
    box(ax, 0.85, 0.72, 0.25, 0.18, "Ready = true\nFreeze package", fc=GREEN_L, ec=GREEN, weight="bold")
    arrow(ax, (0.66, 0.72), (0.72, 0.72), color=GREEN)
    arrow(ax, (0.52, 0.62), (0.52, 0.44), color=ORANGE)
    arrow(ax, (0.64, 0.34), (0.75, 0.62), color=MUTED, connectionstyle="arc3,rad=-0.25")
    ax.text(0.70, 0.77, "none missing", color=GREEN, fontsize=6.3)
    ax.text(0.55, 0.52, "one or more", color=ORANGE, fontsize=6.3)
    ax.text(0.5, 0.06, "History, inventory, limits, and hypotheses may be explicitly unavailable", ha="center", fontsize=6.4, color=MUTED)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c", x=-0.12, y=1.04)
    ax.set_title("Deterministic pending IDs", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    states = [
        (0.78, "State A", "protocol only", ["OBJ", "HIST", "INV", "CONSTR", "HYP"]),
        (0.50, "State B", "same protocol + objective", ["HIST", "INV", "CONSTR", "HYP"]),
        (0.22, "State C", "all required resolved", []),
    ]
    for y, name, desc, ids in states:
        fc, ec = (GREEN_L, GREEN) if not ids else (LIGHT, BLUE)
        box(ax, 0.22, y, 0.32, 0.18, f"{name}\n{desc}", fc=fc, ec=ec, fontsize=6.8, weight="bold")
        ax.text(0.44, y, "→", ha="center", va="center", color=MUTED, fontsize=12)
        text_ids = "ready" if not ids else "Q-" + " / Q-".join(ids)
        box(ax, 0.72, y, 0.45, 0.18, text_ids, fc=fc, ec=ec, fontsize=6.3, weight="bold")
    fig.suptitle("Figure S3 | Reproducible standardized intake", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 3)


def figure_s4() -> None:
    package = json.loads(INTAKE_PATH.read_text())
    shutil.copy2(INTAKE_PATH, DATA_DIR / "S4_representative_intake_package.json")
    sections = [
        ("Raw protocol", package.get("raw_protocol"), "protocol facts", BLUE),
        ("Objective", package.get("objective"), "chemist intent", BLUE),
        ("Historical data", package.get("historical_data"), "measured evidence", TEAL),
        ("Inventory", package.get("inventory_constraints"), "hard constraint", RED),
        ("Operating limits", package.get("operating_limits"), "hard constraint", RED),
        ("Hypotheses", package.get("hypotheses"), "hypothesis", ORANGE),
        ("Output preferences", package.get("output_preferences"), "presentation", GREEN),
    ]
    section_rows = []
    for label, value, authority, _ in sections:
        if value is None or value == [] or value == {} or value == "":
            status = "unavailable/empty"
            size = 0
        else:
            status = "present"
            size = len(json.dumps(value, default=str))
        section_rows.append({"section": label, "status": status, "serialized_characters": size, "authority": authority})
    write_csv("S4_package_sections.csv", section_rows)

    modules = ["Upstream chemistry", "Retrieval", "Calculator", "Council", "Inventory compiler", "Final contract"]
    labels = [s[0] for s in sections]
    influence = np.array(
        [
            [1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 1],
        ]
    )
    write_csv(
        "S4_influence_matrix.csv",
        [{"section": label, **{module: int(influence[i, j]) for j, module in enumerate(modules)}} for i, label in enumerate(labels)],
    )
    pkg_hash = sha256(INTAKE_PATH)
    (DATA_DIR / "S4_package_hash.txt").write_text(f"sha256  {pkg_hash}\n")

    fig = plt.figure(figsize=(7.4, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.03, 0.97], width_ratios=[1.05, 0.95], hspace=0.52, wspace=0.24)
    ax = fig.add_subplot(gs[:, 0])
    panel_label(ax, "a", x=-0.14, y=1.02)
    ax.set_title("Representative frozen package", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, ((label, _, authority, col), row) in enumerate(zip(sections, section_rows)):
        y = 0.89 - i * 0.122
        fc = mpl.colors.to_rgba(col, 0.10) if row["status"] == "present" else LIGHT
        ec = col if row["status"] == "present" else GRID
        box(ax, 0.50, y, 0.86, 0.095, f"{label}  |  {row['status']}\n{authority}  ·  {row['serialized_characters']:,} serialized characters", fc=fc, ec=ec, fontsize=6.8, weight="bold" if row["status"] == "present" else "normal")
    ax.text(0.5, 0.025, f"Schema: {package.get('schema_version')}\nReady for design: {package.get('ready_for_design')}", ha="center", fontsize=6.8, color=MUTED)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b", x=-0.14, y=1.05)
    ax.set_title("Where intake context is consumed", fontweight="bold", color=INK)
    cmap = mpl.colors.ListedColormap(["#FFFFFF", TEAL])
    sns.heatmap(influence, ax=ax, cmap=cmap, cbar=False, linewidths=0.5, linecolor=GRID, xticklabels=modules, yticklabels=labels)
    ax.set_xticklabels([fill(module, 14) for module in modules], rotation=28, ha="right", fontsize=5.9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6.2)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c", x=-0.14, y=1.05)
    ax.set_title("Frozen boundary and provenance", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.5, 0.78, 0.78, 0.16, "Chemist answers + extracted protocol", fc=BLUE_L, ec=BLUE, weight="bold")
    arrow(ax, (0.5, 0.69), (0.5, 0.57), color=BLUE)
    box(ax, 0.5, 0.48, 0.78, 0.18, f"DesignInputPackage\n{package.get('schema_version')}", fc=TEAL_L, ec=TEAL, weight="bold")
    arrow(ax, (0.5, 0.38), (0.5, 0.26), color=TEAL)
    box(ax, 0.5, 0.17, 0.78, 0.16, f"SHA-256\n{pkg_hash[:16]}…", fc=LIGHT, ec=MUTED, fontsize=7.0, weight="bold")
    ax.text(0.5, 0.01, "The frozen package is attached to the result JSON", ha="center", fontsize=6.5, color=MUTED)
    fig.suptitle("Figure S4 | Frozen intake package and evidence propagation", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 4)


def figure_s5() -> None:
    tabs = [
        "Summary",
        "Engineering Design",
        "Process Diagram",
        "Chemistry Plan & Recipe",
        "Stream Assignments",
        "Council Deliberation",
        "Council Report",
        "Experiment Loop",
        "Raw JSON",
        "Equipment & Inventory",
    ]
    tab_source = [
        "FinalDesignContract",
        "FinalDesignContract",
        "ProcessTopology + contract",
        "ChemistryPlan + proposal",
        "FinalDesignContract",
        "Council logs",
        "Council report",
        "Historical experiments",
        "Complete result",
        "Inventory allocation + contract",
    ]
    write_csv("S5_gui_tabs.csv", [{"order": i + 1, "tab": t, "primary_source": s} for i, (t, s) in enumerate(zip(tabs, tab_source))])

    fig = plt.figure(figsize=(7.4, 6.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18], hspace=0.26, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "a", x=-0.025, y=1.04)
    ax.set_title("Standardized-intake GUI workflow", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        (0.12, "1", "Paste protocol\nAnalyze intake", BLUE_L, BLUE),
        (0.37, "2", "Resolve fixed IDs\nFreeze package", BLUE_L, BLUE),
        (0.62, "3", "Select/import\ninventory JSON", ORANGE_L, ORANGE),
        (0.87, "4", "Run design\nReview disposition", GREEN_L, GREEN),
    ]
    for i, (x, n, label, fc, ec) in enumerate(steps):
        box(ax, x, 0.53, 0.20, 0.34, f"{n}\n{label}", fc=fc, ec=ec, fontsize=7.4, weight="bold")
        if i < len(steps) - 1:
            arrow(ax, (x + 0.105, 0.53), (steps[i + 1][0] - 0.105, 0.53), color=MUTED)
    ax.text(0.5, 0.16, "Run button remains blocked until intake readiness is true; imported inventory supplies hard constraints", ha="center", fontsize=6.8, color=MUTED)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "b", x=-0.12, y=1.03)
    ax.set_title("Ten result views, one authoritative contract", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (tab, source) in enumerate(zip(tabs, tab_source)):
        row, col = divmod(i, 2)
        x = 0.27 + col * 0.48
        y = 0.88 - row * 0.17
        authoritative = "FinalDesignContract" in source or "contract" in source.lower()
        fc, ec = (GREEN_L, GREEN) if authoritative else (LIGHT, GRID)
        box(ax, x, y, 0.43, 0.12, f"{tab}\n{source}", fc=fc, ec=ec, fontsize=6.1, weight="bold" if authoritative else "normal")
    ax.text(0.5, 0.02, "Green views contain final numerical outputs", ha="center", fontsize=6.5, color=GREEN)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c", x=-0.12, y=1.03)
    ax.set_title("Disposition controls what is shown", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.5, 0.88, 0.62, 0.13, "Rebuild FinalDesignContract on render", fc=TEAL_L, ec=TEAL, weight="bold")
    arrow(ax, (0.5, 0.81), (0.5, 0.69), color=TEAL)
    box(ax, 0.5, 0.61, 0.42, 0.13, "status = executable?", fc=LIGHT, ec=MUTED, weight="bold")
    arrow(ax, (0.34, 0.55), (0.20, 0.41), color=GREEN)
    arrow(ax, (0.66, 0.55), (0.80, 0.41), color=RED)
    box(ax, 0.20, 0.28, 0.34, 0.24, "EXECUTABLE\nFinal parameters\nExecutable topology\nInstrument manifest", fc=GREEN_L, ec=GREEN, fontsize=7.0, weight="bold")
    box(ax, 0.80, 0.28, 0.34, 0.24, "BLOCKED\nNo run parameters\nRequirements topology\nReconciliation reasons", fc=RED_L, ec=RED, fontsize=7.0, weight="bold")
    ax.text(0.5, 0.045, "Diagnostic intermediate values are not promoted to final tabs", ha="center", fontsize=6.5, color=MUTED)
    fig.suptitle("Figure S5 | GUI workflow and single-source rendering", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 5)


def load_events() -> list[dict[str, Any]]:
    events = []
    for line in EVENTS_PATH.read_text().splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
        except json.JSONDecodeError:
            continue
    return events


def figure_s6() -> None:
    result = json.loads(RESULT_PATH.read_text())
    events = load_events()
    files = [p for p in CASE_ROOT.rglob("*") if p.is_file()]
    categories = Counter()
    for p in files:
        name = p.name.lower()
        if "topology" in name or name.endswith((".png", ".svg")):
            cat = "topology / render"
        elif "result" in name or "design" in name or "summary" in name:
            cat = "design results"
        elif "inventory" in name or "manifest" in name:
            cat = "inventory / manifest"
        elif "intake" in name or "protocol" in name or name == "input.txt":
            cat = "inputs"
        elif "log" in name or "events" in name:
            cat = "logs / model events"
        elif "audit" in name or "checksum" in name:
            cat = "audit / integrity"
        else:
            cat = "other metadata"
        categories[cat] += 1
    write_csv("S6_artifact_counts.csv", [{"artifact_class": k, "file_count": v} for k, v in categories.items()])
    event_counts = Counter(e.get("api_name", "unknown") for e in events)
    write_csv("S6_llm_event_counts.csv", [{"component": k, "event_count": v} for k, v in event_counts.most_common()])
    checks = (result.get("final_validation") or {}).get("checks") or {}
    write_csv("S6_final_validation_checks.csv", [{"check": k, "passed": bool(v)} for k, v in checks.items()])

    fig = plt.figure(figsize=(7.4, 6.7))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a", x=-0.15, y=1.05)
    ax.set_title("Stored artifact classes", fontweight="bold", color=INK)
    cats = [k for k, _ in categories.most_common()]
    vals = [categories[k] for k in cats]
    y = np.arange(len(cats))
    ax.barh(y, vals, color=COLORS[: len(cats)], edgecolor="none")
    ax.set_yticks(y, [c.replace(" / ", " /\n") for c in cats])
    ax.invert_yaxis()
    ax.set_xlabel("Files in representative case")
    ax.bar_label(ax.containers[0], padding=2, fontsize=6.8, color=INK)
    clean_ax(ax, grid_axis="x")

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b", x=-0.13, y=1.05)
    ax.set_title("Provenance chain", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    chain = [
        (0.88, "Protocol + frozen intake", BLUE_L, BLUE),
        (0.72, "LLM event log (57 events)", PURPLE_L, PURPLE),
        (0.56, "Raw result + proposal", ORANGE_L, ORANGE),
        (0.40, "Canonical reconciliation", TEAL_L, TEAL),
        (0.24, "Audit + final contract", GREEN_L, GREEN),
        (0.08, "PNG / SVG / JSON + checksums", LIGHT, MUTED),
    ]
    for i, (yy, label, fc, ec) in enumerate(chain):
        box(ax, 0.5, yy, 0.75, 0.105, label, fc=fc, ec=ec, fontsize=7.0, weight="bold")
        if i < len(chain) - 1:
            arrow(ax, (0.5, yy - 0.055), (0.5, chain[i + 1][0] + 0.055), color=MUTED)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c", x=-0.15, y=1.05)
    ax.set_title("Final deterministic checks", fontweight="bold", color=INK)
    check_names = [k.replace("_", " ") for k in checks]
    pass_values = [1 if checks[k] else 0 for k in checks]
    y = np.arange(len(check_names))
    colors = [GREEN if value else RED for value in pass_values]
    ax.barh(y, pass_values, color=colors, height=0.58)
    ax.set_yticks(y, [fill(name, 28) for name in check_names])
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 1], ["fail", "pass"])
    ax.invert_yaxis()
    clean_ax(ax, grid_axis="x")
    ax.text(1.03, len(check_names) - 0.5, f"{sum(pass_values)}/{len(pass_values)}", ha="center", va="bottom", color=GREEN, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d", x=-0.13, y=1.05)
    ax.set_title("Recorded LLM events", fontweight="bold", color=INK)
    names = [k for k, _ in event_counts.most_common()]
    values = [event_counts[k] for k in names]
    if len(names) > 8:
        other = sum(values[7:])
        names = names[:7] + ["other"]
        values = values[:7] + [other]
    y = np.arange(len(names))
    ax.barh(y, values, color=[PURPLE if "agent" in name or "council" in name else BLUE for name in names])
    ax.set_yticks(y, [name.replace("_", " ") for name in names])
    ax.invert_yaxis()
    ax.set_xlabel("Events")
    ax.bar_label(ax.containers[0], padding=2, fontsize=6.8)
    clean_ax(ax, grid_axis="x")
    fig.suptitle("Figure S6 | Run provenance, validation, and artifacts", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 6)


def horizontal_top_categories(ax, df: pd.DataFrame, label_col: str, value_col: str, *, top_n: int, title: str, color: str) -> None:
    data = df.sort_values(value_col, ascending=False).head(top_n).sort_values(value_col)
    ax.barh(np.arange(len(data)), data[value_col], color=color, alpha=0.90)
    ax.set_yticks(np.arange(len(data)), [fill(str(v), 24) for v in data[label_col]])
    ax.set_title(title, fontweight="bold", color=INK)
    ax.set_xlabel("Records")
    ax.bar_label(ax.containers[0], padding=2, fontsize=6.2)
    clean_ax(ax, grid_axis="x")


def figure_s7() -> None:
    names = [
        "fig1a_reaction_classes.csv",
        "fig1b_reactor_types.csv",
        "fig1c_reactor_materials.csv",
        "fig1d_bond_types.csv",
        "fig1f_inlet_streams.csv",
        "fig1g_flow_yields_raw.csv",
        "fig1g_batch_yields_raw.csv",
    ]
    for name in names:
        copy_source(name)
    reaction = pd.read_csv(PANEL_DATA / names[0])
    reactor = pd.read_csv(PANEL_DATA / names[1])
    material = pd.read_csv(PANEL_DATA / names[2])
    bond = pd.read_csv(PANEL_DATA / names[3])
    streams = pd.read_csv(PANEL_DATA / names[4])
    flow_yield = pd.read_csv(PANEL_DATA / names[5]).iloc[:, -1].dropna().astype(float)
    batch_yield = pd.read_csv(PANEL_DATA / names[6]).iloc[:, -1].dropna().astype(float)

    fig, axes = plt.subplots(3, 2, figsize=(7.4, 9.0))
    plt.subplots_adjust(hspace=0.45, wspace=0.42)
    horizontal_top_categories(axes[0, 0], reaction, "category", "count", top_n=8, title="Reaction classes", color=BLUE)
    panel_label(axes[0, 0], "a", x=-0.20, y=1.07)
    horizontal_top_categories(axes[0, 1], reactor, reactor.columns[0], "count", top_n=8, title="Reactor types", color=TEAL)
    panel_label(axes[0, 1], "b", x=-0.18, y=1.07)
    horizontal_top_categories(axes[1, 0], material, material.columns[0], "count", top_n=7, title="Reactor materials", color=ORANGE)
    panel_label(axes[1, 0], "c", x=-0.20, y=1.07)
    horizontal_top_categories(axes[1, 1], bond, bond.columns[0], "count", top_n=7, title="Bond classes", color=PURPLE)
    panel_label(axes[1, 1], "d", x=-0.18, y=1.07)

    ax = axes[2, 0]
    panel_label(ax, "e", x=-0.20, y=1.07)
    x_col = streams.columns[0]
    y_col = "count" if "count" in streams else streams.columns[1]
    data = streams.sort_values(x_col)
    ax.bar(data[x_col].astype(str), data[y_col], color=GREEN)
    ax.set_title("Number of inlet streams", fontweight="bold", color=INK)
    ax.set_xlabel("Inlet streams")
    ax.set_ylabel("Records with available field")
    ax.bar_label(ax.containers[0], padding=2, fontsize=6.2)
    clean_ax(ax, grid_axis="y")

    ax = axes[2, 1]
    panel_label(ax, "f", x=-0.18, y=1.07)
    bins = np.arange(0, 105, 5)
    ax.hist(batch_yield, bins=bins, density=True, alpha=0.45, color=ORANGE, label=f"Batch (n={len(batch_yield)})")
    ax.hist(flow_yield, bins=bins, density=True, alpha=0.55, color=BLUE, label=f"Flow (n={len(flow_yield)})")
    ax.axvline(batch_yield.median(), color=ORANGE, linestyle="--", linewidth=1.1)
    ax.axvline(flow_yield.median(), color=BLUE, linestyle="--", linewidth=1.1)
    ax.set_title("Available yield distributions", fontweight="bold", color=INK)
    ax.set_xlabel("Yield (%)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left")
    clean_ax(ax, grid_axis="y")
    fig.suptitle("Figure S7 | Frozen literature-corpus composition (n = 464)", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 7)


def figure_s8() -> None:
    for name in (
        "fig2a_rule_landscape.csv",
        "fig2b_formula_coverage.csv",
        "fig2c_coverage_heatmap_matrix.csv",
        "fig2d_concept_network_nodes.csv",
        "fig2d_concept_network_edges.csv",
    ):
        copy_source(name)
    landscape = pd.read_csv(PANEL_DATA / "fig2a_rule_landscape.csv")
    formula = pd.read_csv(PANEL_DATA / "fig2b_formula_coverage.csv")
    heat = pd.read_csv(PANEL_DATA / "fig2c_coverage_heatmap_matrix.csv")
    nodes = pd.read_csv(PANEL_DATA / "fig2d_concept_network_nodes.csv")
    edges = pd.read_csv(PANEL_DATA / "fig2d_concept_network_edges.csv")

    totals = landscape.groupby(["category_key", "category_label"], as_index=False)["count"].sum().sort_values("count", ascending=False).head(14)
    keep = totals["category_key"].tolist()
    pivot = landscape[landscape["category_key"].isin(keep)].pivot(index="category_label", columns="severity", values="count").fillna(0)
    order = totals.set_index("category_key").loc[keep]["category_label"].tolist()[::-1]
    pivot = pivot.reindex(order)
    severity_order = [c for c in ["hard_rule", "guideline", "tip", "safety"] if c in pivot]

    fig = plt.figure(figsize=(7.4, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08], hspace=0.34, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a", x=-0.18, y=1.06)
    left = np.zeros(len(pivot))
    sev_colors = {"hard_rule": RED, "guideline": BLUE, "tip": GREEN, "safety": ORANGE}
    for sev in severity_order:
        vals = pivot[sev].values
        ax.barh(np.arange(len(pivot)), vals, left=left, label=sev.replace("_", " "), color=sev_colors[sev])
        left += vals
    ax.set_yticks(np.arange(len(pivot)), [fill(v, 20) for v in pivot.index])
    ax.set_xlabel("Rules")
    ax.set_title("Largest rule categories", fontweight="bold", color=INK)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    clean_ax(ax, grid_axis="x")

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b", x=-0.17, y=1.06)
    top_formula = formula.sort_values("total_rules", ascending=False).head(14).sort_values("percent_with_formula")
    ax.barh(np.arange(len(top_formula)), top_formula["percent_with_formula"], color=TEAL)
    ax.set_yticks(np.arange(len(top_formula)), [fill(v, 20) for v in top_formula["category_label"]])
    ax.set_xlim(80, 101)
    ax.set_xlabel("Rules with detected quantitative expression (%)")
    ax.set_title("Quantitative-expression coverage", fontweight="bold", color=INK)
    clean_ax(ax, grid_axis="x")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c", x=-0.18, y=1.04)
    heat_top = heat.sort_values("row_total", ascending=False).head(13).set_index("category_label")
    heat_values = heat_top.drop(columns=["category_key", "row_total"], errors="ignore")
    sns.heatmap(np.log1p(heat_values), ax=ax, cmap="YlGnBu", cbar_kws={"label": "log(1 + association count)"}, linewidths=0.25, linecolor="white")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=48, ha="right", fontsize=5.7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6.0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Rule-category associations by chemistry class", fontweight="bold", color=INK)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d", x=-0.16, y=1.04)
    top_nodes = nodes.nlargest(12, "frequency").copy()
    node_set = set(top_nodes["concept"])
    top_edges = edges[edges["source_concept"].isin(node_set) & edges["target_concept"].isin(node_set)].nlargest(22, "weight")
    graph = nx.Graph()
    for _, row in top_nodes.iterrows():
        graph.add_node(row["concept"], frequency=float(row["frequency"]), category=row["dominant_category_label"])
    for _, row in top_edges.iterrows():
        graph.add_edge(row["source_concept"], row["target_concept"], weight=float(row["weight"]))
    # A fixed circular layout keeps concept labels legible, including disconnected nodes.
    ordered_nodes = list(top_nodes.sort_values("frequency", ascending=False)["concept"])
    graph = nx.Graph(graph.subgraph(ordered_nodes))
    pos = nx.circular_layout(ordered_nodes, scale=1.0)
    freqs = np.array([graph.nodes[n]["frequency"] for n in graph.nodes])
    sizes = 35 + 420 * np.sqrt(freqs / freqs.max())
    category_order = {cat: COLORS[i % len(COLORS)] for i, cat in enumerate(sorted({graph.nodes[n]["category"] for n in graph.nodes}))}
    node_colors = [category_order[graph.nodes[n]["category"]] for n in graph.nodes]
    weights = [0.25 + 2.4 * graph.edges[e]["weight"] / max(1, top_edges["weight"].max()) for e in graph.edges]
    nx.draw_networkx_edges(graph, pos, ax=ax, width=weights, edge_color=GRID, alpha=0.8)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes, node_color=node_colors, edgecolors="white", linewidths=0.7)
    label_pos = {node: (xy[0] * 1.22, xy[1] * 1.22) for node, xy in pos.items()}
    labels = {n: fill(n, 13) for n in graph.nodes}
    label_artists = nx.draw_networkx_labels(graph, label_pos, labels=labels, ax=ax, font_size=4.7, font_color=INK)
    for artist in label_artists.values():
        artist.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35})
    ax.set_xlim(-1.48, 1.48)
    ax.set_ylim(-1.42, 1.42)
    ax.set_title("Engineering concept co-occurrence", fontweight="bold", color=INK)
    ax.axis("off")
    fig.suptitle("Figure S8 | Engineering rule-base structure (2,537 rules)", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 8)


def figure_s9() -> None:
    from flora_translate import config

    weights = [
        ("Semantic similarity", float(config.W_SEMANTIC), "final score"),
        ("Field similarity", float(config.W_FIELD), "final score"),
        ("Photocatalyst", float(config.W_PHOTOCATALYST), "field score"),
        ("Solvent", float(config.W_SOLVENT), "field score"),
        ("Wavelength", float(config.W_WAVELENGTH), "field score"),
        ("Temperature", float(config.W_TEMPERATURE), "field score"),
        ("Concentration", float(config.W_CONCENTRATION), "field score"),
    ]
    write_csv("S9_retrieval_weights.csv", [{"component": c, "weight": w, "scope": s} for c, w, s in weights])
    fallback_rows = [
        {"tier": 1, "scope": "paired records", "filters": "mechanism + phase", "trigger_to_next": "<3 hits"},
        {"tier": 2, "scope": "paired records", "filters": "none", "trigger_to_next": "0 hits"},
        {"tier": 3, "scope": "all records", "filters": "none", "trigger_to_next": "return empty"},
    ]
    write_csv("S9_retrieval_fallbacks.csv", fallback_rows)

    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08], width_ratios=[1.15, 0.85], hspace=0.28, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    panel_label(ax, "a", x=-0.025, y=1.03)
    ax.set_title("Plan-aware query construction and provider fallback", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    left_items = ["Reaction class", "Mechanism", "Catalyst", "Solvent", "T / λ", "Intermediate", "Bond", "Keywords"]
    for i, item in enumerate(left_items):
        row, col = divmod(i, 4)
        x = 0.08 + col * 0.11
        y = 0.72 - row * 0.30
        box(ax, x, y, 0.095, 0.18, item, fc=BLUE_L, ec=BLUE, fontsize=6.0, weight="bold")
    arrow(ax, (0.45, 0.57), (0.54, 0.57), color=BLUE, lw=1.4)
    box(ax, 0.64, 0.57, 0.19, 0.32, "Plan-aware\nrich query", fc=TEAL_L, ec=TEAL, fontsize=8.0, weight="bold")
    arrow(ax, (0.74, 0.57), (0.80, 0.57), color=TEAL, lw=1.4)
    box(ax, 0.89, 0.72, 0.18, 0.18, "Embedding\nprovider", fc=PURPLE_L, ec=PURPLE, fontsize=7.2, weight="bold")
    box(ax, 0.89, 0.36, 0.18, 0.18, "Deterministic\nlexical fallback", fc=ORANGE_L, ec=ORANGE, fontsize=7.0, weight="bold")
    arrow(ax, (0.89, 0.62), (0.89, 0.46), color=RED)
    ax.text(0.91, 0.54, "provider failure", fontsize=6.0, color=RED, ha="left")
    ax.text(0.31, 0.08, "ChemistryPlan + BatchRecord", fontsize=6.8, color=MUTED, ha="center")
    ax.text(0.89, 0.08, "Same query text and metadata filters", fontsize=6.8, color=MUTED, ha="center")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "b", x=-0.11, y=1.04)
    ax.set_title("Three-stage retrieval scope", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    tiers = [
        (0.82, "1", "Pairs only\nmechanism + phase", GREEN_L, GREEN, "< 3 hits"),
        (0.52, "2", "Pairs only\nfilters relaxed", BLUE_L, BLUE, "0 hits"),
        (0.22, "3", "All records\nno filters", ORANGE_L, ORANGE, "no records"),
    ]
    for i, (y, num, label, fc, ec, trigger) in enumerate(tiers):
        box(ax, 0.43, y, 0.62, 0.20, f"Tier {num}\n{label}", fc=fc, ec=ec, fontsize=7.4, weight="bold")
        if i < len(tiers) - 1:
            arrow(ax, (0.43, y - 0.11), (0.43, tiers[i + 1][0] + 0.11), color=ec)
            ax.text(0.76, (y + tiers[i + 1][0]) / 2, trigger, va="center", fontsize=6.2, color=ec, fontweight="bold")
    box(ax, 0.82, 0.22, 0.24, 0.20, "Exclude hidden\nsource IDs", fc=RED_L, ec=RED, fontsize=6.8, weight="bold")
    arrow(ax, (0.65, 0.22), (0.69, 0.22), color=RED)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c", x=-0.15, y=1.04)
    ax.set_title("Reranking calculation", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.5, 0.88, 0.74, 0.14, "semantic = max(0, 1 − L2²/2)", fc=PURPLE_L, ec=PURPLE, fontsize=7.0, weight="bold")
    field_weights = [(c, w) for c, w, scope in weights if scope == "field score"]
    y_positions = np.linspace(0.68, 0.35, len(field_weights))
    for (name, weight), y in zip(field_weights, y_positions):
        ax.text(0.08, y, name, va="center", fontsize=6.6, color=INK)
        ax.add_patch(patches.Rectangle((0.35, y - 0.025), weight / max(w for _, w in field_weights) * 0.42, 0.05, facecolor=TEAL, edgecolor="none"))
        ax.text(0.80, y, f"{weight:.2f}", va="center", ha="right", fontsize=6.5, color=TEAL, fontweight="bold")
    box(ax, 0.5, 0.18, 0.80, 0.15, "final = 0.60 × semantic + 0.40 × field", fc=GREEN_L, ec=GREEN, fontsize=7.4, weight="bold")
    ax.text(0.5, 0.04, "Sort descending → return top-k", ha="center", fontsize=6.7, color=MUTED)
    fig.suptitle("Figure S9 | Current plan-aware retrieval workflow", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 9)


def figure_s10() -> None:
    for name in (
        "fig3c_retrieval_pairs_raw.csv",
        "fig3c_component_summary.csv",
        "fig3c_summary_metrics.csv",
        "fig3d_family_match_rates.csv",
        "fig3d_demo_retrieval_table.csv",
    ):
        copy_source(name)
    pairs = pd.read_csv(PANEL_DATA / "fig3c_retrieval_pairs_raw.csv")
    components = pd.read_csv(PANEL_DATA / "fig3c_component_summary.csv")
    metrics = pd.read_csv(PANEL_DATA / "fig3c_summary_metrics.csv")
    families = pd.read_csv(PANEL_DATA / "fig3d_family_match_rates.csv")
    demo = pd.read_csv(PANEL_DATA / "fig3d_demo_retrieval_table.csv")

    # Determine rank delta column robustly from frozen export.
    rank_col = next((c for c in pairs.columns if "rank" in c.lower() and ("change" in c.lower() or "delta" in c.lower())), None)
    if rank_col is None:
        numeric = pairs.select_dtypes(include=[np.number]).columns
        rank_col = next((c for c in numeric if "rank" in c.lower()), None)
    if rank_col is None:
        rank_delta = pd.Series(np.zeros(len(pairs)))
    else:
        rank_delta = pairs[rank_col].fillna(0).astype(float)

    fig = plt.figure(figsize=(7.4, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.92, 0.90, 0.78], hspace=0.43, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a", x=-0.16, y=1.06)
    x = np.arange(len(families))
    width = 0.36
    ax.bar(x - width / 2, families["semantic_match_rate_pct"], width, color=MUTED, label="Semantic only")
    ax.bar(x + width / 2, families["flowpilot_match_rate_pct"], width, color=BLUE, label="FlowPilot")
    ax.set_xticks(x, families["family"], rotation=28, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Top-5 family match rate (%)")
    ax.set_title("Photocatalyst-family alignment", fontweight="bold", color=INK)
    ax.legend(frameon=False, loc="upper left")
    clean_ax(ax, grid_axis="y")

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b", x=-0.16, y=1.06)
    bins = np.arange(math.floor(rank_delta.min()) - 0.5, math.ceil(rank_delta.max()) + 1.5, 1)
    if len(bins) < 3:
        bins = np.arange(-1.5, 2.5, 1)
    ax.hist(rank_delta, bins=bins, color=TEAL, edgecolor="white")
    ax.axvline(0, color=INK, linewidth=0.9)
    pct = float(metrics.loc[metrics["metric"] == "pct_reranked", "value"].iloc[0])
    ax.text(0.97, 0.92, f"{pct:.1f}% changed rank\nn = {len(pairs):,} pairs", transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=INK, bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, edgecolor=GRID))
    ax.set_xlabel("Rank change after field reranking")
    ax.set_ylabel("Query–result pairs")
    ax.set_title("Reranking displacement", fontweight="bold", color=INK)
    clean_ax(ax, grid_axis="y")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c", x=-0.16, y=1.06)
    comp = components.copy()
    x = np.arange(len(comp))
    ax.bar(x - 0.18, comp["mean_score"], 0.36, color=PURPLE, label="Mean score")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, comp["percent_nonzero"], 0.36, color=ORANGE, label="Nonzero rate")
    ax.set_xticks(x, [fill(v.replace(" Match", ""), 13) for v in comp["component"]], rotation=20, ha="right")
    ax.set_ylabel("Mean component score")
    ax2.set_ylabel("Nonzero observations (%)")
    ax.set_title("Available field-score components", fontweight="bold", color=INK)
    clean_ax(ax, grid_axis="y")
    ax2.spines[["top"]].set_visible(False)
    ax2.spines["right"].set_color(GRID)
    handles = [patches.Patch(color=PURPLE, label="Mean score"), patches.Patch(color=ORANGE, label="Nonzero rate")]
    ax.legend(handles=handles, frameon=False, loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d", x=-0.16, y=1.06)
    ax.set_title("Representative iridium query: top five", fontweight="bold", color=INK)
    methods = ["semantic_only", "flowpilot"]
    method_labels = ["Semantic only", "FlowPilot reranked"]
    family_colors = {"Iridium": BLUE, "Ruthenium": RED, "Organic dye": ORANGE, "": MUTED}
    for row_idx, (method, label) in enumerate(zip(methods, method_labels)):
        subset = demo[demo["method"] == method].sort_values("rank").head(5)
        y = 1 - row_idx
        ax.text(-0.45, y, label, ha="right", va="center", fontsize=7.0, fontweight="bold", color=INK)
        for _, item in subset.iterrows():
            rank = int(item["rank"])
            family = str(item.get("result_photocatalyst_family") or "")
            if family == "nan":
                family = ""
            color = family_colors.get(family, MUTED)
            marker = "o" if family == "Iridium" else "X"
            ax.scatter(rank, y, s=90, c=color, marker=marker, edgecolors="white", linewidths=0.7, zorder=3)
            ax.text(rank, y - 0.22, family or "unknown", ha="center", va="top", fontsize=5.4, color=MUTED, rotation=20)
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xticks(range(1, 6))
    ax.set_yticks([])
    ax.set_xlabel("Rank")
    clean_ax(ax, grid_axis="x")

    ax = fig.add_subplot(gs[2, :])
    panel_label(ax, "e", x=-0.025, y=1.08)
    ax.set_title("Leave-one-source-out and deterministic retrieval controls", fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    controls = [
        (0.10, "Hidden source ID\nfrom case manifest", BLUE_L, BLUE),
        (0.32, "Retrieve candidate\npool", TEAL_L, TEAL),
        (0.54, "Normalize + exclude\nmatching IDs", RED_L, RED),
        (0.76, "Sort remaining\nfinal scores", ORANGE_L, ORANGE),
        (0.93, "Return\ntop-k", GREEN_L, GREEN),
    ]
    for i, (x_, label, fc, ec) in enumerate(controls):
        width_ = 0.16 if i < 4 else 0.11
        box(ax, x_, 0.52, width_, 0.36, label, fc=fc, ec=ec, fontsize=6.6, weight="bold")
        if i < len(controls) - 1:
            next_x = controls[i + 1][0]
            arrow(ax, (x_ + width_ / 2 + 0.005, 0.52), (next_x - (0.08 if i + 1 < 4 else 0.055) - 0.005, 0.52), color=MUTED)
    ax.text(0.5, 0.12, "Automated tests cover hidden-ID exclusion and embedding-provider lexical fallback", ha="center", fontsize=7.0, color=MUTED)
    fig.suptitle("Figure S10 | Retrieval benchmark and leakage controls", fontsize=12, fontweight="bold", color=INK, y=0.995)
    save_figure(fig, 10)


def write_documentation() -> None:
    lines = ["# FlowPilot ESI Figures S1-S10\n", "Generated from the current repository and frozen manuscript data exports.\n", "## Contents\n"]
    for n in range(1, 11):
        lines.append(f"- `figures/Figure_S{n:02d}.png` (400 dpi), `.pdf`, and `.svg`")
    lines += [
        "\n## Source data\n",
        "Each figure has one or more CSV/JSON files in `source_data/`. Files copied from the manuscript visualization cache retain their original names; new files begin with the corresponding `S#` prefix.",
        "\n## Interpretation boundaries\n",
        "- S7 uses the frozen classified corpus (`n=464`), not every raw JSON file currently present in the records directory.",
        "- S8 quantitative-expression coverage is machine-detected expression coverage, not independent equation validation.",
        "- S10 evaluates retrieval metadata alignment and rank behavior, not chemical yield or design optimality.",
        "- S6 is one representative executable case and does not imply all runs contain the same artifact/event counts.",
        "\n## Reproduction\n",
        "Run: `/home/amirreza/anaconda3/envs/flent/bin/python scripts/build_esi_figures_s1_s10.py` from the project root.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    caption_lines = ["# Supporting Figure Captions\n"]
    for n in range(1, 11):
        caption_lines.append(f"## Figure S{n}\n\n{CAPTIONS[n]}\n")
    (META_DIR / "captions.md").write_text("\n".join(caption_lines), encoding="utf-8")
    methods = """# Methods Notes for Figures S1-S10

## Workflow figures (S1-S6)

Architecture, schema, intake, GUI, and provenance figures were generated from the current Python source and one frozen executable photoredox run. Pydantic field counts were introspected at figure-generation time. Question IDs and readiness rules were read from `flora_translate/intake_agent.py`. The representative artifact and validation counts were read from the canonical photoredox case under `outputs/benchmarks/khu_three_protocols_canonical_20260810_161346`.

## Corpus figure (S7)

The figure uses the frozen manuscript panel exports in `visualization/panel_data_exports`. The classified corpus contains 464 records. Field-specific denominators vary because incomplete metadata are retained. Yield panels use only records with numeric yields between 0 and 100 percent.

## Rule-base figure (S8)

Rule counts derive from the 2,537-entry current rule store and its frozen classification exports. Severity counts are mutually exclusive within category. Chemistry-class association counts are non-exclusive. Quantitative-expression coverage was generated by the existing expression detector and must not be interpreted as independent verification of equation correctness.

## Retrieval figures (S9-S10)

S9 reads current retrieval weights from `flora_translate/config.py` and depicts the execution logic in `flora_translate/retriever.py`. S10 uses the frozen 1,600-pair retrieval benchmark exports. Photocatalyst-family matching is reported for families with available query annotations. Rank-change and field-component summaries use the exported benchmark values. Hidden source IDs are excluded using the current leave-one-source-out mechanism before final top-k selection.
"""
    (META_DIR / "methods_notes.md").write_text(methods, encoding="utf-8")


def write_manifest_and_checksums() -> None:
    source_files = [
        ROOT / "flora_translate" / "intake_agent.py",
        ROOT / "flora_translate" / "schemas.py",
        ROOT / "flora_translate" / "retriever.py",
        ROOT / "flora_translate" / "config.py",
        ROOT / "flora_translate" / "main.py",
        ROOT / "pages" / "flora_design_unified.py",
        ROOT / "flora_fundamentals" / "data" / "rules.json",
        ROOT / "visualization" / "cache" / "classifications.json",
        ROOT / "visualization" / "cache" / "rule_classifications.json",
        INTAKE_PATH,
        RESULT_PATH,
        EVENTS_PATH,
    ]
    manifest = {
        "schema_version": "flowpilot_esi_s1_s10_v1.0",
        "corpus_classified_records": 464,
        "engineering_rules": 2537,
        "retrieval_pairs": 1600,
        "figures": [
            {
                "figure": f"S{n}",
                "png": f"figures/Figure_S{n:02d}.png",
                "pdf": f"figures/Figure_S{n:02d}.pdf",
                "svg": f"figures/Figure_S{n:02d}.svg",
            }
            for n in range(1, 11)
        ],
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in source_files if p.is_file()},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "checksums.sha256":
            rows.append(f"{sha256(p)}  {p.relative_to(OUT)}")
    (OUT / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_style()
    figure_s1()
    figure_s2()
    figure_s3()
    figure_s4()
    figure_s5()
    figure_s6()
    figure_s7()
    figure_s8()
    figure_s9()
    figure_s10()
    write_documentation()
    # Store the exact generator in the package after all outputs are built.
    shutil.copy2(Path(__file__), OUT / "build_esi_figures_s1_s10.py")
    write_manifest_and_checksums()
    print(OUT)


if __name__ == "__main__":
    main()
