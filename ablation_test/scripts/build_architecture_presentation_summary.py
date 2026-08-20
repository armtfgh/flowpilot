from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


MODEL_ORDER = ("Qwen 27B", "GPT-5.6 Terra")
ARCH_ORDER = ("One-shot", "Full FlowPilot")
COLORS = {
    "one_shot": "#4C78A8",
    "flowpilot": "#138A72",
    "ink": "#17212B",
    "muted": "#5F6B76",
    "grid": "#D9DEE3",
    "soft_blue": "#EAF2F8",
    "soft_green": "#E7F5F1",
    "soft_gold": "#FFF3D6",
    "soft_gray": "#F3F5F7",
    "danger": "#B5483A",
}
DIMENSIONS = (
    ("Inventory compliance", 0.30, "Listed reactor, pump, BPR, ratings, light source"),
    ("Numerical closure", 0.30, "V = Q x tau, stream sums, gas correction and equivalents"),
    ("Chemistry fidelity", 0.20, "Required chemistry, operating window and gas identity"),
    ("Process completeness", 0.20, "Required topology, feeds, preparation, work-up and safety"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AEB7BF")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["ink"])


def _outcome_overview(
    matrix: list[dict[str, Any]],
    uplift: list[dict[str, Any]],
    figures: Path,
) -> None:
    lookup = {(row["model"], row["architecture"]): row for row in matrix}
    uplift_lookup = {row["model"]: row for row in uplift}
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.08, 0.92), hspace=0.34, wspace=0.20)

    for index, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(grid[0, index])
        rows = [lookup[(model, architecture)] for architecture in ARCH_ORDER]
        bars = ax.bar(
            np.arange(2),
            [100 * float(row["feasible_executable_rate"]) for row in rows],
            width=0.62,
            color=(COLORS["one_shot"], COLORS["flowpilot"]),
        )
        ax.bar_label(
            bars,
            labels=[f"{row['feasible_executable_n']}/{row['feasible_n']}" for row in rows],
            padding=5,
            fontsize=13,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax.set_xticks(np.arange(2), ARCH_ORDER, fontsize=12)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Executable feasible designs (%)", fontsize=11)
        ax.set_title(model, fontsize=17, fontweight="bold", color=COLORS["ink"], pad=12)
        _style_axis(ax)
        effect = uplift_lookup[model]
        ax.text(
            0.5,
            0.88,
            f"FlowPilot uplift: +{100 * float(effect['executable_rate_difference']):.1f} pp",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=COLORS["flowpilot"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORS["soft_green"], edgecolor="none"),
        )

    ax_wtl = fig.add_subplot(grid[1, 0])
    x = np.arange(len(MODEL_ORDER))
    wins = [int(uplift_lookup[model]["flowpilot_wins"]) for model in MODEL_ORDER]
    ties = [int(uplift_lookup[model]["ties"]) for model in MODEL_ORDER]
    losses = [int(uplift_lookup[model]["one_shot_wins"]) for model in MODEL_ORDER]
    ax_wtl.bar(x, wins, color=COLORS["flowpilot"], label="FlowPilot wins")
    ax_wtl.bar(x, ties, bottom=wins, color="#B8C0C7", label="Ties")
    ax_wtl.bar(x, losses, bottom=np.array(wins) + np.array(ties), color=COLORS["one_shot"], label="One-shot wins")
    for idx, (win, tie, loss) in enumerate(zip(wins, ties, losses)):
        ax_wtl.text(idx, win / 2, str(win), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        if tie:
            ax_wtl.text(idx, win + tie / 2, str(tie), ha="center", va="center", color=COLORS["ink"], fontsize=12, fontweight="bold")
        if loss:
            ax_wtl.text(idx, win + tie + loss / 2, str(loss), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax_wtl.set_xticks(x, MODEL_ORDER, fontsize=11)
    ax_wtl.set_ylim(0, 16.2)
    ax_wtl.set_ylabel("Paired feasible outcomes (n=15)")
    ax_wtl.set_title("Paired architecture outcomes", fontsize=15, fontweight="bold", color=COLORS["ink"])
    ax_wtl.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _style_axis(ax_wtl)

    ax_note = fig.add_subplot(grid[1, 1])
    ax_note.axis("off")
    notes = (
        ("Matched comparison", "Same protocol, objective, inventory and repeat input within each model."),
        ("Strict primary endpoint", "A feasible output counts only when every critical engineering check passes."),
        ("Safety behavior", "All four conditions correctly blocked 15/15 deliberately infeasible cases."),
        ("Scope", "Evidence supports constrained design executability, not wet-lab yield or global optimality."),
    )
    for idx, (heading, body) in enumerate(notes):
        top = 0.97 - idx * 0.235
        patch = FancyBboxPatch(
            (0.01, top - 0.18), 0.98, 0.18,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            transform=ax_note.transAxes,
            facecolor=COLORS["soft_gray"] if idx != 1 else COLORS["soft_green"],
            edgecolor="none",
        )
        ax_note.add_patch(patch)
        ax_note.text(0.04, top - 0.045, heading, transform=ax_note.transAxes, fontsize=12.5, fontweight="bold", color=COLORS["ink"], va="top")
        ax_note.text(0.04, top - 0.105, body, transform=ax_note.transAxes, fontsize=10.5, color=COLORS["muted"], va="top", wrap=True)

    fig.suptitle("FlowPilot architecture improves executable batch-to-flow designs", fontsize=23, fontweight="bold", color=COLORS["ink"], y=0.985)
    fig.text(0.5, 0.012, "Five chemistry families x three repeats; retrospective paired architecture pilot.", ha="center", fontsize=9.5, color=COLORS["muted"])
    _save(fig, figures, "01_results_overview")


def _architecture_effects(uplift: list[dict[str, Any]], figures: Path) -> None:
    lookup = {row["model"]: row for row in uplift}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), gridspec_kw={"wspace": 0.28})

    estimates = np.array([100 * float(lookup[model]["executable_rate_difference"]) for model in MODEL_ORDER])
    lower = estimates - np.array([100 * float(lookup[model]["executable_ci_low"]) for model in MODEL_ORDER])
    upper = np.array([100 * float(lookup[model]["executable_ci_high"]) for model in MODEL_ORDER]) - estimates
    axes[0].errorbar(
        np.arange(2), estimates, yerr=[lower, upper], fmt="o", markersize=12,
        color=COLORS["flowpilot"], ecolor=COLORS["ink"], elinewidth=2, capsize=7,
    )
    axes[0].axhline(0, color=COLORS["muted"], linewidth=1)
    axes[0].set_xticks(np.arange(2), MODEL_ORDER, fontsize=11)
    axes[0].set_ylim(-5, 110)
    axes[0].set_ylabel("FlowPilot - one-shot executable rate (pp)")
    axes[0].set_title("Executability uplift", fontsize=16, fontweight="bold")
    for idx, value in enumerate(estimates):
        axes[0].annotate(f"+{value:.1f} pp", (idx, value), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=12, fontweight="bold", color=COLORS["flowpilot"])
    _style_axis(axes[0])

    quality = np.array([float(lookup[model]["quality_score_difference"]) for model in MODEL_ORDER])
    quality_low = quality - np.array([float(lookup[model]["quality_ci_low"]) for model in MODEL_ORDER])
    quality_high = np.array([float(lookup[model]["quality_ci_high"]) for model in MODEL_ORDER]) - quality
    axes[1].errorbar(
        np.arange(2), quality, yerr=[quality_low, quality_high], fmt="o", markersize=12,
        color="#D18B21", ecolor=COLORS["ink"], elinewidth=2, capsize=7,
    )
    axes[1].axhline(0, color=COLORS["muted"], linewidth=1)
    axes[1].set_xticks(np.arange(2), MODEL_ORDER, fontsize=11)
    axes[1].set_ylabel("FlowPilot - one-shot quality score (points)")
    axes[1].set_title("Secondary weighted-quality difference", fontsize=16, fontweight="bold")
    for idx, value in enumerate(quality):
        axes[1].annotate(f"+{value:.1f}", (idx, value), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=12, fontweight="bold", color="#A66B12")
    _style_axis(axes[1])

    fig.suptitle("Paired architecture effects with 95% family-cluster intervals", fontsize=22, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.025, "Intervals resample five chemistry families; repeated calls are not treated as independent protocols.", ha="center", color=COLORS["muted"], fontsize=10)
    _save(fig, figures, "02_architecture_effects")


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str, body: str, color: str) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor=color,
        edgecolor="#BAC3CB",
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(x + 0.025, y + height - 0.045, title, transform=ax.transAxes, fontsize=13, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(x + 0.025, y + height - 0.105, body, transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"], va="top", linespacing=1.35)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color="#77838E"))


def _score_calculation_figure(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    fig.suptitle("How the architecture benchmark is scored", fontsize=24, fontweight="bold", color=COLORS["ink"], y=0.975)
    ax.text(0.5, 0.925, "Primary claims use deterministic gates; the weighted score is secondary.", transform=ax.transAxes, ha="center", fontsize=12, color=COLORS["muted"])

    _box(ax, (0.03, 0.61), 0.20, 0.23, "1  Frozen input", "Protocol + objective\nInventory + hard constraints\nExpected SCREEN or BLOCK", COLORS["soft_blue"])
    _box(ax, (0.29, 0.61), 0.27, 0.23, "2  Deterministic checks", "Inventory match and equipment ratings\nNumerical closure and gas calculations\nChemistry window and required topology", COLORS["soft_gold"])
    _box(ax, (0.62, 0.61), 0.34, 0.23, "3  Strict endpoint", "FEASIBLE: correct SCREEN + structured proposal\n+ every critical check passes + no unlisted equipment\n\nINFEASIBLE: correct BLOCK", COLORS["soft_green"])
    _arrow(ax, (0.23, 0.725), (0.29, 0.725))
    _arrow(ax, (0.56, 0.725), (0.62, 0.725))

    ax.text(0.03, 0.535, "Secondary quality score for feasible outputs", transform=ax.transAxes, fontsize=15, fontweight="bold", color=COLORS["ink"])
    x_positions = (0.03, 0.275, 0.52, 0.765)
    colors = (COLORS["soft_blue"], COLORS["soft_gold"], COLORS["soft_green"], COLORS["soft_gray"])
    compact_dimensions = (
        ("30%  Inventory\ncompliance", "Listed hardware\nOperating limits\nNo assumed equipment"),
        ("30%  Numerical\nclosure", "V-Q-tau closure\nGas pressure correction\nStoichiometric closure"),
        ("20%  Chemistry\nfidelity", "Reaction requirements\nOperating window\nGas identity"),
        ("20%  Process\ncompleteness", "Required topology\nFeed preparation\nWork-up and safety"),
    )
    for (title, body), x, color in zip(compact_dimensions, x_positions, colors):
        _box(ax, (x, 0.295), 0.205, 0.195, title, body, color)
    ax.text(0.5, 0.255, "Dimension score = 100 x passed applicable checks / applicable checks", transform=ax.transAxes, ha="center", fontsize=12, color=COLORS["ink"])
    ax.text(0.5, 0.205, "Quality = 0.30 x Inventory + 0.30 x Numerical + 0.20 x Chemistry + 0.20 x Process", transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold", color="#A66B12", bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF8E8", edgecolor="#E4C078"))

    ax.text(0.03, 0.115, "Paired ranking", transform=ax.transAxes, fontsize=13, fontweight="bold", color=COLORS["ink"])
    ax.text(0.16, 0.115, "Executable beats non-executable  |  If both execute, >5 quality points determines a win  |  If neither executes, fewer critical failures wins  |  Otherwise tie", transform=ax.transAxes, fontsize=10.5, color=COLORS["muted"], va="center")
    ax.text(0.5, 0.045, "Confidence intervals use 10,000 bootstrap resamples of chemistry families, not individual repeats.", transform=ax.transAxes, ha="center", fontsize=10, color=COLORS["muted"])
    _save(fig, figures, "03_score_calculation")


def _quality_breakdown(matrix: list[dict[str, Any]], figures: Path) -> None:
    lookup = {(row["model"], row["architecture"]): row for row in matrix}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.7), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    one = [float(lookup[(model, "One-shot")]["mean_feasible_quality"]) for model in MODEL_ORDER]
    flow = [float(lookup[(model, "Full FlowPilot")]["mean_feasible_quality"]) for model in MODEL_ORDER]
    b1 = axes[0].bar(x - width / 2, one, width, color=COLORS["one_shot"], label="One-shot")
    b2 = axes[0].bar(x + width / 2, flow, width, color=COLORS["flowpilot"], label="Full FlowPilot")
    axes[0].bar_label(b1, fmt="%.1f", padding=3, fontsize=11)
    axes[0].bar_label(b2, fmt="%.1f", padding=3, fontsize=11)
    axes[0].set_xticks(x, MODEL_ORDER, fontsize=11)
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("Mean feasible quality score")
    axes[0].set_title("Observed secondary score", fontsize=16, fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper center")
    _style_axis(axes[0])

    weights = [100 * item[1] for item in DIMENSIONS]
    labels = [item[0].replace(" ", "\n", 1) for item in DIMENSIONS]
    bars = axes[1].bar(np.arange(4), weights, color=("#5E81AC", "#D49A35", "#48A58C", "#89939D"), width=0.65)
    axes[1].bar_label(bars, labels=[f"{value:.0f}%" for value in weights], padding=4, fontsize=11, fontweight="bold")
    axes[1].set_xticks(np.arange(4), labels, fontsize=10)
    axes[1].set_ylim(0, 38)
    axes[1].set_ylabel("Weight in quality score (%)")
    axes[1].set_title("Frozen dimension weights", fontsize=16, fontweight="bold")
    _style_axis(axes[1])

    fig.suptitle("Secondary quality score: outcome and composition", fontsize=22, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.02, "Quality scores include all feasible-scenario outputs; executability remains the strict primary endpoint.", ha="center", fontsize=10, color=COLORS["muted"])
    _save(fig, figures, "04_quality_score_breakdown")


def _weight_sensitivity(qwen_summary: dict[str, Any], gpt_summary: dict[str, Any], figures: Path) -> None:
    summaries = {"Qwen 27B": qwen_summary, "GPT-5.6 Terra": gpt_summary}
    schemes = [item["weight_scheme"] for item in qwen_summary["weight_sensitivity"]]
    labels = [item.replace("_", " ").title() for item in schemes]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, model in zip(axes, MODEL_ORDER):
        rows = {row["weight_scheme"]: row for row in summaries[model]["weight_sensitivity"]}
        wins = [int(rows[key]["flowpilot_wins"]) for key in schemes]
        ties = [int(rows[key]["ties"]) for key in schemes]
        losses = [int(rows[key]["one_shot_wins"]) for key in schemes]
        y = np.arange(len(schemes))
        ax.barh(y, wins, color=COLORS["flowpilot"], label="FlowPilot wins")
        ax.barh(y, ties, left=wins, color="#B8C0C7", label="Ties")
        ax.barh(y, losses, left=np.array(wins) + np.array(ties), color=COLORS["one_shot"], label="One-shot wins")
        ax.set_yticks(y, labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 15.5)
        ax.set_xlabel("Paired feasible outcomes (n=15)")
        ax.set_title(model, fontsize=16, fontweight="bold")
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1.0, -0.13))
    fig.suptitle("Pairwise conclusion is unchanged across six weighting schemes", fontsize=22, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.02, "Sensitivity analysis changes quality weights only; the strict executability gate is unchanged.", ha="center", fontsize=10, color=COLORS["muted"])
    _save(fig, figures, "05_weight_sensitivity")


def _write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output)}")
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(source: Path, qwen_source: Path, gpt_source: Path, output: Path) -> None:
    summary = _read_json(source / "summary.json")
    audit = summary["comparability_audit"]
    if not audit.get("passed"):
        raise ValueError("Source comparability audit did not pass")
    matrix = summary["architecture_results"]
    uplift = summary["architecture_uplift"]
    qwen_summary = _read_json(qwen_source / "summary.json")
    gpt_summary = _read_json(gpt_source / "summary.json")

    figures = output / "figures"
    tables = output / "tables"
    frozen = output / "frozen"
    for directory in (figures, tables, frozen):
        directory.mkdir(parents=True, exist_ok=True)

    _outcome_overview(matrix, uplift, figures)
    _architecture_effects(uplift, figures)
    _score_calculation_figure(figures)
    _quality_breakdown(matrix, figures)
    _weight_sensitivity(qwen_summary, gpt_summary, figures)

    _write_csv(tables / "model_architecture_results.csv", matrix)
    _write_csv(tables / "paired_architecture_effects.csv", uplift)
    _write_csv(
        tables / "score_dimensions.csv",
        [
            {"dimension": title, "weight": weight, "meaning": meaning}
            for title, weight, meaning in DIMENSIONS
        ],
    )
    shutil.copy2(source / "frozen" / "comparability_audit.json", frozen / "comparability_audit.json")
    shutil.copy2(qwen_source / "frozen" / "scoring_spec.json", frozen / "scoring_spec.json")

    package_summary = {
        "schema_version": "flowpilot_architecture_presentation_summary_v1.0",
        "source_comparison": str(source),
        "source_qwen_study": str(qwen_source),
        "source_gpt_study": str(gpt_source),
        "comparability_audit": audit,
        "results": matrix,
        "effects": uplift,
        "primary_endpoint": "feasible executable design rate",
        "secondary_endpoint": "weighted feasible-output quality score",
        "claim_boundary": [
            "The comparison evaluates constrained design executability and quality.",
            "It does not evaluate wet-lab yield or global design optimality.",
            "The scoring study is a retrospective paired architecture pilot.",
            "Five chemistry families, rather than repeated calls, are the independent bootstrap clusters.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(package_summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        """# FlowPilot vs One-Shot: Qwen and GPT Presentation Summary

This package merges the matched Qwen 27B and GPT-5.6 Terra architecture
comparisons into slide-ready figures. Within each model comparison, one-shot
and full FlowPilot received identical protocol, objective, inventory, and
repeat inputs.

## Headline results

| Base model | One-shot executable | FlowPilot executable | Uplift | FlowPilot / tie / one-shot wins |
|---|---:|---:|---:|---:|
| Qwen 27B | 3/15 (20.0%) | 15/15 (100%) | +80.0 pp | 12 / 3 / 0 |
| GPT-5.6 Terra | 2/15 (13.3%) | 15/15 (100%) | +86.7 pp | 13 / 2 / 0 |

All four conditions correctly blocked 15/15 deliberately infeasible cases.

## Figure guide

- `01_results_overview`: main presentation result.
- `02_architecture_effects`: paired uplift and family-cluster intervals.
- `03_score_calculation`: deterministic endpoint and secondary score workflow.
- `04_quality_score_breakdown`: observed scores and frozen weights.
- `05_weight_sensitivity`: paired conclusions under six weight schemes.

## Interpretation boundary

Executability is the primary endpoint. A feasible output is executable only
when it makes the correct SCREEN decision, contains a structured proposal,
passes every applicable critical deterministic check, and does not assume
unlisted equipment. The weighted quality score is secondary.

This is a retrospective paired architecture pilot. It supports claims about
constraint-compliant design executability and repeatable architecture behavior;
it does not establish wet-lab yield superiority or global design optimality.
""",
        encoding="utf-8",
    )
    _write_checksums(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qwen-source", type=Path, required=True)
    parser.add_argument("--gpt-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.qwen_source, args.gpt_source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
