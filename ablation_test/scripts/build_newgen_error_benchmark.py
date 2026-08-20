"""Build the error-first NewGen comparison from authoritative benchmark runs."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from ablation_test.src.cases import load_cases_from_path
from ablation_test.src.error_audit import CRITERIA, audit_result, load_expectations


SOURCE_MANIFEST = (
    ROOT
    / "deliverables"
    / "flowpilot_newgen_qwen_openai_20260813"
    / "source_data"
    / "run_metrics.csv"
)
EXPECTATIONS_PATH = (
    ROOT / "ablation_test" / "benchmarks" / "newgen_error_expectations_v1.json"
)
OUTPUT = ROOT / "deliverables" / "flowpilot_newgen_error_benchmark_20260814"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
AUDITS = OUTPUT / "audits"

CASE_PATHS = (
    ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_cuaac" / "case.json",
    ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_hydrogenolysis" / "case.json",
    ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_multistep" / "case.json",
)
CASE_IDS = {
    "CuAAC": "cuaac_adsc_200900726",
    "Hydrogenolysis": "hydrogenolysis_oprd_9b00416",
    "Two-stage amidation": "multistep_amidation_c5ra20838f",
}
ARCH_COLORS = {"One-shot": "#C44E52", "FlowPilot": "#237A57"}
MODEL_LABELS = {"Qwen3.6-27B": "Qwen 27B", "GPT-5.4": "GPT-5.4"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, name: str) -> None:
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(FIGURES / f"{name}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D8DCE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def _load_cases() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for path in CASE_PATHS:
        for case in load_cases_from_path(path):
            cases[case.case_id] = case
    return cases


def _audit_runs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = _load_cases()
    expectations = load_expectations(EXPECTATIONS_PATH)
    manifest_rows = list(csv.DictReader(SOURCE_MANIFEST.open(encoding="utf-8")))
    summaries: list[dict[str, Any]] = []
    criteria_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        case_id = CASE_IDS[row["case"]]
        run_dir = Path(row["run_directory"])
        result = _read_json(run_dir / "result.json")
        audit = audit_result(cases[case_id], result, expectations[case_id])
        model = MODEL_LABELS.get(row["model"], row["model"])
        run_id = f"{model.replace(' ', '_').replace('.', '')}_{row['architecture']}_{row['case']}".lower().replace("-", "_").replace(" ", "_")
        _write_json(AUDITS / f"{run_id}.json", audit)
        summaries.append({
            "model": model,
            "architecture": row["architecture"],
            "case": row["case"],
            "case_id": case_id,
            "applicable_criteria": audit["applicable_criteria"],
            "total_errors": audit["total_errors"],
            "critical_errors": audit["critical_errors"],
            "error_rate_pct": round(100.0 * audit["error_rate"], 2),
            "critical_error_free": audit["critical_error_free"],
            "failed_criterion_ids": ";".join(audit["failed_criterion_ids"]),
            "run_directory": str(run_dir),
            "audit_file": str(AUDITS / f"{run_id}.json"),
        })
        for criterion in audit["criteria"]:
            criteria_rows.append({
                "model": model,
                "architecture": row["architecture"],
                "case": row["case"],
                **criterion,
            })
    return summaries, criteria_rows


def _campaign_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["model"], row["architecture"])].append(row)
    output: list[dict[str, Any]] = []
    for (model, architecture), rows in grouped.items():
        applicable = sum(int(row["applicable_criteria"]) for row in rows)
        errors = sum(int(row["total_errors"]) for row in rows)
        critical = sum(int(row["critical_errors"]) for row in rows)
        error_free = sum(bool(row["critical_error_free"]) for row in rows)
        output.append({
            "model": model,
            "architecture": architecture,
            "cases": len(rows),
            "applicable_checks": applicable,
            "total_errors": errors,
            "critical_errors": critical,
            "error_rate_pct": round(100.0 * errors / applicable, 2),
            "critical_error_free_runs": error_free,
            "critical_error_free_rate_pct": round(100.0 * error_free / len(rows), 2),
        })
    return sorted(output, key=lambda row: (row["model"], row["architecture"]), reverse=True)


def _plot_campaign_errors(campaigns: list[dict[str, Any]]) -> None:
    models = ["Qwen 27B", "GPT-5.4"]
    architectures = ["One-shot", "FlowPilot"]
    lookup = {(row["model"], row["architecture"]): row for row in campaigns}
    x = np.arange(len(models))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for index, architecture in enumerate(architectures):
        values = [lookup[(model, architecture)]["total_errors"] for model in models]
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=architecture,
            color=ARCH_COLORS[architecture],
        )
        ax.bar_label(bars, padding=4, fontsize=11, fontweight="bold")
    ax.set_xticks(x, models)
    ax.set_ylabel("Deterministic criterion failures (lower is better)")
    fig.text(0.105, 0.95, "Error-first architecture comparison", fontsize=15, fontweight="bold")
    fig.text(0.105, 0.905, "Three paired held-out cases per model; 20 fixed criteria; N/A excluded", color="#4D5562")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(row["total_errors"] for row in campaigns) + 2)
    _style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_figure(fig, "campaign_total_errors")


def _plot_paired_errors(run_rows: list[dict[str, Any]]) -> None:
    models = ["Qwen 27B", "GPT-5.4"]
    cases = ["CuAAC", "Hydrogenolysis", "Two-stage amidation"]
    lookup = {(row["model"], row["architecture"], row["case"]): row for row in run_rows}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharey=True)
    x = np.arange(len(cases))
    width = 0.34
    for ax, model in zip(axes, models):
        for index, architecture in enumerate(("One-shot", "FlowPilot")):
            values = [lookup[(model, architecture, case)]["total_errors"] for case in cases]
            bars = ax.bar(x + (index - 0.5) * width, values, width, color=ARCH_COLORS[architecture], label=architecture)
            ax.bar_label(bars, padding=3, fontsize=9, fontweight="bold")
        ax.set_xticks(x, ["CuAAC", "H2\npacked bed", "Two-stage\namidation"])
        ax.set_title(model, fontweight="bold")
        _style_axis(ax)
    axes[0].set_ylabel("Criterion failures per output")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Paired error counts by chemistry", x=0.07, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, "paired_case_error_counts")


def _plot_error_heatmap(criteria_rows: list[dict[str, Any]]) -> None:
    order = []
    for model in ("Qwen 27B", "GPT-5.4"):
        for architecture in ("One-shot", "FlowPilot"):
            for case in ("CuAAC", "Hydrogenolysis", "Two-stage amidation"):
                order.append((model, architecture, case))
    criterion_ids = [row[0] for row in CRITERIA]
    lookup = {
        (row["model"], row["architecture"], row["case"], row["criterion_id"]): row["status"]
        for row in criteria_rows
    }
    codes = {"PASS": 0, "NOT_APPLICABLE": 1, "FAIL": 2}
    matrix = np.array([
        [codes[lookup[(*run, cid)]] for cid in criterion_ids]
        for run in order
    ])
    labels = [f"{model} | {architecture} | {case}" for model, architecture, case in order]
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.imshow(matrix, cmap=ListedColormap(["#2F8F62", "#D9DCE1", "#C94747"]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(criterion_ids)), criterion_ids, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Every error is traceable to a fixed criterion", loc="left", fontsize=15, fontweight="bold")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x] == 2:
                ax.text(x, y, "x", ha="center", va="center", color="white", fontweight="bold")
    ax.tick_params(length=0)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for color, label in (("#2F8F62", "PASS"), ("#D9DCE1", "N/A"), ("#C94747", "FAIL"))
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.07))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    _save_figure(fig, "criterion_error_heatmap")


def _plot_domain_errors(criteria_rows: list[dict[str, Any]]) -> None:
    domains = ["chemistry", "process", "source_fidelity", "completeness", "calculation", "inventory"]
    campaigns = [(model, architecture) for model in ("Qwen 27B", "GPT-5.4") for architecture in ("One-shot", "FlowPilot")]
    values = defaultdict(int)
    for row in criteria_rows:
        if row["status"] == "FAIL":
            values[(row["model"], row["architecture"], row["domain"])] += 1
    colors = ["#725A9A", "#D17C35", "#5A78A6", "#9A9FA8", "#C94747", "#2B8A62"]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    bottom = np.zeros(len(campaigns))
    labels = [f"{model}\n{architecture}" for model, architecture in campaigns]
    for domain, color in zip(domains, colors):
        counts = np.array([values[(*campaign, domain)] for campaign in campaigns])
        ax.bar(labels, counts, bottom=bottom, color=color, label=domain.replace("_", " ").title())
        bottom += counts
    for index, total in enumerate(bottom):
        ax.text(index, total + 0.15, f"{int(total)}", ha="center", fontweight="bold")
    ax.set_ylabel("Criterion failures across three cases")
    ax.set_title("Where the deterministic errors occurred", loc="left", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_ylim(0, max(bottom) + 2)
    _style_axis(ax)
    fig.tight_layout()
    _save_figure(fig, "error_domains_stacked")


def _plot_methodology() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.axis("off")
    ax.text(0.03, 0.93, "NewGen error-first benchmark", fontsize=22, fontweight="bold", color="#20242A")
    ax.text(0.03, 0.875, "Raw inconsistencies and violations are the endpoint; weighted QA scores are not used.", fontsize=12, color="#555D68")
    boxes = [
        (0.03, "1  Freeze", "Same protocol, inventory,\n20 criteria and tolerances"),
        (0.275, "2  Generate", "One-shot and FlowPilot\nreceive the same design input"),
        (0.52, "3  Recalculate", "Python recomputes flows,\nV/Q, gas basis and limits"),
        (0.765, "4  Count", "PASS / FAIL / N/A\nOne FAIL = one error"),
    ]
    for x, title, body in boxes:
        patch = plt.Rectangle((x, 0.57), 0.205, 0.20, facecolor="#F5F6F8", edgecolor="#B8BEC7", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + 0.018, 0.72, title, fontsize=12, fontweight="bold", color="#20242A")
        ax.text(x + 0.018, 0.64, body, fontsize=10.5, color="#4B535E", va="center")
    for x in (0.245, 0.49, 0.735):
        ax.annotate("", xy=(x + 0.02, 0.67), xytext=(x - 0.015, 0.67), arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 1.6})

    ax.text(0.03, 0.47, "Primary endpoint", fontsize=12, fontweight="bold")
    ax.text(0.03, 0.42, "Critical-error-free output: no FAIL on a criterion marked critical.", fontsize=11)
    ax.text(0.03, 0.34, "Secondary endpoints", fontsize=12, fontweight="bold")
    ax.text(0.03, 0.29, "Total errors = count(FAIL)    |    Error rate = FAIL / applicable criteria    |    N/A is excluded", fontsize=11)
    ax.text(0.03, 0.19, "Audit domains", fontsize=12, fontweight="bold")
    ax.text(0.03, 0.14, "Chemistry  |  Process order  |  Protocol fidelity  |  Completeness  |  Numerical closure  |  Inventory", fontsize=11)
    ax.text(0.03, 0.055, "Important: source-distance and wet-lab yield accuracy remain separate analyses; a valid design need not copy the published flow condition.", fontsize=10.5, color="#555D68")
    fig.tight_layout()
    _save_figure(fig, "error_scoring_methodology")


def _write_report(run_rows: list[dict[str, Any]], campaigns: list[dict[str, Any]]) -> None:
    lookup = {(row["model"], row["architecture"]): row for row in campaigns}
    lines = [
        "# FlowPilot NewGen Error-First Benchmark",
        "",
        "## Purpose",
        "",
        "This report replaces the weighted quality score as the primary comparison. Each output is audited with the same 20 frozen questions. Python deterministically recalculates numerical closure and checks structured protocol, topology, safety, and inventory evidence. A failed applicable criterion counts as one error; `NOT_APPLICABLE` is excluded.",
        "",
        "## Headline Results",
        "",
        "| Model | Architecture | Errors | Applicable checks | Error rate | Critical-error-free runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("Qwen 27B", "GPT-5.4"):
        for architecture in ("One-shot", "FlowPilot"):
            row = lookup[(model, architecture)]
            lines.append(
                f"| {model} | {architecture} | {row['total_errors']} | {row['applicable_checks']} | {row['error_rate_pct']:.2f}% | {row['critical_error_free_runs']}/{row['cases']} |"
            )
    lines.extend([
        "",
        f"Qwen one-shot produced {lookup[('Qwen 27B', 'One-shot')]['total_errors']} errors; Qwen FlowPilot produced {lookup[('Qwen 27B', 'FlowPilot')]['total_errors']}. GPT-5.4 one-shot produced {lookup[('GPT-5.4', 'One-shot')]['total_errors']} errors; GPT-5.4 FlowPilot produced {lookup[('GPT-5.4', 'FlowPilot')]['total_errors']}.",
        "",
        "FlowPilot was error-free under this frozen deterministic rubric in all 6 model-case runs. The one-shot outputs were also error-free for CuAAC, so the result is not that one-shot always fails. The architecture difference appeared in the gas-liquid-solid and multistage cases.",
        "",
        "## Observed Errors",
        "",
    ])
    for row in run_rows:
        if not row["total_errors"]:
            continue
        lines.append(f"### {row['model']} {row['architecture']}: {row['case']}")
        lines.append("")
        lines.append(f"Errors: **{row['total_errors']}** (`{row['failed_criterion_ids']}`).")
        lines.append("")
        audit = _read_json(Path(row["audit_file"]))
        for criterion in audit["criteria"]:
            if criterion["status"] == "FAIL":
                lines.append(f"- `{criterion['criterion_id']}`: {criterion['observed']}")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "The strongest architecture effect is numerical and operational closure. The most severe one-shot example was Qwen hydrogenolysis: the reported pressure-corrected hydrogen flow was inconsistent with the ideal-gas conversion, the hydrogen-equivalent calculation did not close, both residence-time bases were wrong, and the liquid flow was below the declared pump minimum. Both one-shot models also made a total-residence-time error in the two-stage case.",
        "",
        "These data support the claim that deterministic engineering realization and inventory validation reduce detectable design errors. They do not establish superior reaction yield prediction, broad chemical generalization, or statistical significance because this package contains three cases and one run per model-architecture cell.",
        "",
        "## Reproducibility",
        "",
        "- Evaluator: `ablation_test/src/error_audit.py`",
        "- Fixed expectations: `ablation_test/benchmarks/newgen_error_expectations_v1.json`",
        "- Builder: `ablation_test/scripts/build_newgen_error_benchmark.py`",
        "- Per-run audit JSONs: `audits/`",
        "- Machine-readable tables: `tables/`",
        "- Figures: `figures/`",
        "",
        "The expectation file is retrospective for this current package but is derived only from the public protocol, hard constraints, inventory, and held-out source oracle. It must be frozen unchanged before any confirmatory reruns or additional models are tested.",
    ])
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _checksums() -> None:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(OUTPUT)}")
    (OUTPUT / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    for path in (OUTPUT, FIGURES, TABLES, AUDITS):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXPECTATIONS_PATH, OUTPUT / "frozen_error_expectations.json")

    run_rows, criteria_rows = _audit_runs()
    campaign_rows = _campaign_rows(run_rows)
    _write_csv(TABLES / "run_error_summary.csv", run_rows)
    _write_csv(TABLES / "criterion_audit.csv", criteria_rows)
    _write_csv(TABLES / "campaign_error_summary.csv", campaign_rows)
    _write_csv(
        TABLES / "criterion_dictionary.csv",
        [
            {"criterion_id": cid, "domain": domain, "critical": critical, "question": question}
            for cid, domain, critical, question in CRITERIA
        ],
    )
    _plot_campaign_errors(campaign_rows)
    _plot_paired_errors(run_rows)
    _plot_error_heatmap(criteria_rows)
    _plot_domain_errors(criteria_rows)
    _plot_methodology()
    _write_report(run_rows, campaign_rows)
    summary = {
        "schema_version": "flowpilot_newgen_error_benchmark_v1.0",
        "primary_endpoint": "critical_error_free_output",
        "error_definition": "one failed applicable universal criterion",
        "weighted_score_used": False,
        "run_count": len(run_rows),
        "campaigns": campaign_rows,
    }
    _write_json(OUTPUT / "summary.json", summary)
    _checksums()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
