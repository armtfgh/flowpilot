from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARCHITECTURES = ("one_shot", "flowpilot")
ARCH_NAMES = {
    "one_shot": "One-shot",
    "flowpilot": "Full FlowPilot",
}
COLORS = {
    "one_shot": "#4C78A8",
    "flowpilot": "#2A9D8F",
}
DIMENSION_LABELS = {
    "inventory_compliance": "Inventory",
    "numerical_closure": "Numerical",
    "chemistry_fidelity": "Chemistry",
    "process_completeness": "Process",
}


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
    fig.savefig(directory / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _scoring_hash(package: Path) -> str:
    return hashlib.sha256(
        (package / "frozen" / "scoring_spec.json").read_bytes()
    ).hexdigest()


def _input_map(rows: list[dict[str, str]]) -> dict[tuple[str, int], str]:
    output: dict[tuple[str, int], str] = {}
    for row in rows:
        key = (row["scenario_id"], int(row["repeat"]))
        digest = row["design_input_sha256"]
        previous = output.setdefault(key, digest)
        if previous != digest:
            raise ValueError(f"Architecture input mismatch within package for {key}")
    return output


def validate_cross_model_identity(
    qwen_package: Path,
    gpt_package: Path,
) -> dict[str, Any]:
    qwen_summary = _read_json(qwen_package / "summary.json")
    gpt_summary = _read_json(gpt_package / "summary.json")
    qwen_rows = _read_csv(qwen_package / "tables" / "cell_level_scores.csv")
    gpt_rows = _read_csv(gpt_package / "tables" / "cell_level_scores.csv")
    qwen_inputs = _input_map(qwen_rows)
    gpt_inputs = _input_map(gpt_rows)
    checks = {
        "qwen_input_identity": bool(qwen_summary["input_identity_passed"]),
        "gpt_input_identity": bool(gpt_summary["input_identity_passed"]),
        "cell_count_equal": qwen_summary["cell_count"] == gpt_summary["cell_count"] == 60,
        "paired_count_equal": (
            qwen_summary["paired_feasible_count"]
            == gpt_summary["paired_feasible_count"]
            == 15
        ),
        "scenario_repeat_keys_equal": set(qwen_inputs) == set(gpt_inputs),
        "cross_model_input_hashes_equal": qwen_inputs == gpt_inputs,
        "scoring_spec_hash_equal": _scoring_hash(qwen_package)
        == _scoring_hash(gpt_package),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"Cross-model comparability validation failed: {failed}")
    return {
        "passed": True,
        "checks": checks,
        "scoring_spec_sha256": _scoring_hash(qwen_package),
        "matched_scenario_repeat_count": len(qwen_inputs),
    }


def _architecture_rows(
    summaries: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for display_model, summary in summaries:
        for architecture in ARCHITECTURES:
            values = summary["architectures"][architecture]
            rows.append(
                {
                    "model": display_model,
                    "architecture": ARCH_NAMES[architecture],
                    "feasible_executable_n": values["feasible_executable_n"],
                    "feasible_n": values["feasible_n"],
                    "feasible_executable_rate": (
                        values["feasible_executable_n"] / values["feasible_n"]
                    ),
                    "infeasible_correct_block_n": values["infeasible_correct_block_n"],
                    "infeasible_n": values["infeasible_n"],
                    "infeasible_correct_block_rate": (
                        values["infeasible_correct_block_n"] / values["infeasible_n"]
                    ),
                    "mean_feasible_quality": values["mean_feasible_quality"],
                }
            )
    return rows


def _uplift_rows(
    summaries: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for display_model, summary in summaries:
        paired = summary["paired_feasible"]
        effect = paired["executable_rate_difference"]
        quality = paired["quality_score_difference"]
        counts = paired["counts"]
        rows.append(
            {
                "model": display_model,
                "executable_rate_difference": effect["estimate"],
                "executable_ci_low": effect["ci_low"],
                "executable_ci_high": effect["ci_high"],
                "quality_score_difference": quality["estimate"],
                "quality_ci_low": quality["ci_low"],
                "quality_ci_high": quality["ci_high"],
                "flowpilot_wins": counts.get("flowpilot", 0),
                "ties": counts.get("tie", 0),
                "one_shot_wins": counts.get("one_shot", 0),
            }
        )
    return rows


def _dimension_rows(
    summaries: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for display_model, summary in summaries:
        for item in summary["dimension_effects"]:
            rows.append({"model": display_model, **item})
    return rows


def _architecture_matrix_figure(
    rows: list[dict[str, Any]], figures: Path
) -> None:
    models = list(dict.fromkeys(row["model"] for row in rows))
    x = np.arange(len(models))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for offset, architecture_key in (
        (-width / 2, "One-shot"),
        (width / 2, "Full FlowPilot"),
    ):
        selected = [
            next(
                row
                for row in rows
                if row["model"] == model and row["architecture"] == architecture_key
            )
            for model in models
        ]
        color_key = (
            "one_shot" if architecture_key == "One-shot" else "flowpilot"
        )
        executable = axes[0].bar(
            x + offset,
            [100 * row["feasible_executable_rate"] for row in selected],
            width,
            color=COLORS[color_key],
            label=architecture_key,
        )
        blocked = axes[1].bar(
            x + offset,
            [100 * row["infeasible_correct_block_rate"] for row in selected],
            width,
            color=COLORS[color_key],
            label=architecture_key,
        )
        axes[0].bar_label(
            executable,
            labels=[
                f"{row['feasible_executable_n']}/{row['feasible_n']}"
                for row in selected
            ],
            padding=3,
        )
        axes[1].bar_label(
            blocked,
            labels=[
                f"{row['infeasible_correct_block_n']}/{row['infeasible_n']}"
                for row in selected
            ],
            padding=3,
        )
    for ax, title in zip(
        axes,
        ("Feasible executable designs", "Infeasible correctly blocked"),
    ):
        ax.set_xticks(x, models)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Rate (%)")
        ax.set_title(title)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[1].legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
    )
    fig.suptitle("Same-input model × architecture outcome matrix", fontsize=16)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _save(fig, figures, "01_model_architecture_matrix")


def _uplift_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    models = [row["model"] for row in rows]
    estimates = [100 * row["executable_rate_difference"] for row in rows]
    lower = [
        100 * (row["executable_rate_difference"] - row["executable_ci_low"])
        for row in rows
    ]
    upper = [
        100 * (row["executable_ci_high"] - row["executable_rate_difference"])
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(models))
    ax.errorbar(
        x,
        estimates,
        yerr=[lower, upper],
        fmt="o",
        color="#2A9D8F",
        ecolor="#333333",
        markersize=9,
        capsize=5,
    )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(x, models)
    ax.set_ylabel("FlowPilot minus one-shot executable rate (percentage points)")
    ax.set_title("Architecture uplift with 95% family-cluster intervals")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "02_architecture_uplift")


def _win_tie_loss_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    models = [row["model"] for row in rows]
    wins = [row["flowpilot_wins"] for row in rows]
    ties = [row["ties"] for row in rows]
    losses = [row["one_shot_wins"] for row in rows]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x, wins, color=COLORS["flowpilot"], label="FlowPilot wins")
    ax.bar(x, ties, bottom=wins, color="#B8B8B8", label="Ties")
    ax.bar(
        x,
        losses,
        bottom=np.array(wins) + np.array(ties),
        color=COLORS["one_shot"],
        label="One-shot wins",
    )
    ax.set_xticks(x, models)
    ax.set_ylim(0, 16)
    ax.set_ylabel("Paired feasible outcomes (n=15)")
    ax.set_title("Paired architecture outcomes by base model")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "03_cross_model_win_tie_loss")


def _dimension_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    models = list(dict.fromkeys(row["model"] for row in rows))
    dimensions = list(DIMENSION_LABELS)
    x = np.arange(len(dimensions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    for index, model in enumerate(models):
        selected = {
            row["dimension"]: row for row in rows if row["model"] == model
        }
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            [selected[item]["mean_difference"] for item in dimensions],
            width,
            label=model,
            color=("#457B9D", "#2A9D8F")[index % 2],
        )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(x, [DIMENSION_LABELS[item] for item in dimensions])
    ax.set_ylabel("FlowPilot minus one-shot score (points)")
    ax.set_title("Architecture benefit by deterministic outcome dimension")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    _save(fig, figures, "04_dimension_uplift_by_model")


def _write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output)}")
    (output / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_comparison(
    qwen_package: Path,
    gpt_package: Path,
    output: Path,
) -> dict[str, Any]:
    audit = validate_cross_model_identity(qwen_package, gpt_package)
    qwen_summary = _read_json(qwen_package / "summary.json")
    gpt_summary = _read_json(gpt_package / "summary.json")
    summaries = [
        ("Qwen 27B", qwen_summary),
        (gpt_summary["model"], gpt_summary),
    ]
    architecture_rows = _architecture_rows(summaries)
    uplift_rows = _uplift_rows(summaries)
    dimension_rows = _dimension_rows(summaries)
    tables = output / "tables"
    figures = output / "figures"
    frozen = output / "frozen"
    for directory in (tables, figures, frozen):
        directory.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "model_architecture_matrix.csv", architecture_rows)
    _write_csv(tables / "architecture_uplift.csv", uplift_rows)
    _write_csv(tables / "dimension_uplift.csv", dimension_rows)
    _architecture_matrix_figure(architecture_rows, figures)
    _uplift_figure(uplift_rows, figures)
    _win_tie_loss_figure(uplift_rows, figures)
    _dimension_figure(dimension_rows, figures)
    summary = {
        "schema_version": "flowpilot_qwen_gpt_architecture_comparison_v1.0",
        "comparability_audit": audit,
        "models": [item[0] for item in summaries],
        "architecture_results": architecture_rows,
        "architecture_uplift": uplift_rows,
        "interpretation_boundary": [
            "The comparison measures constrained design executability and quality.",
            "It does not measure experimental yield or global design optimality.",
            "The five chemistry families are the independent bootstrap clusters.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (frozen / "comparability_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(
        f"""# Qwen and GPT Architecture Comparison

This package compares one-shot translation with full FlowPilot while holding
the base model constant within each comparison.

The comparability audit passed all checks: both model studies contain the same
60-cell structure, the same 30 scenario-repeat input hashes, and the same frozen
scoring specification.

| Model | FlowPilot wins | Ties | One-shot wins | Executability uplift |
|---|---:|---:|---:|---:|
| {uplift_rows[0]["model"]} | {uplift_rows[0]["flowpilot_wins"]} | {uplift_rows[0]["ties"]} | {uplift_rows[0]["one_shot_wins"]} | {100 * uplift_rows[0]["executable_rate_difference"]:.1f} pp |
| {uplift_rows[1]["model"]} | {uplift_rows[1]["flowpilot_wins"]} | {uplift_rows[1]["ties"]} | {uplift_rows[1]["one_shot_wins"]} | {100 * uplift_rows[1]["executable_rate_difference"]:.1f} pp |

The architecture effect must be interpreted separately for each base model.
Cross-model differences do not isolate model quality unless architecture is
also held constant.
""",
        encoding="utf-8",
    )
    _write_checksums(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-package", type=Path, required=True)
    parser.add_argument("--gpt-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_comparison(
        args.qwen_package,
        args.gpt_package,
        args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
