from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ablation_test.src.cases import ROOT
from ablation_test.src.metrics import DEPLOYMENT_CAPS_V2


CONFIG_PATH = ROOT / "configs" / "cross_model_benchmark.json"
SCORER_PATH = ROOT / "src" / "metrics.py"
DIMENSIONS = [
    "formal_validity",
    "engineering_integrity",
    "process_completeness",
    "safety_adequacy",
    "evidence_provenance",
    "decision_assurance",
    "actionability_calibration",
]
DIMENSION_LABELS = {
    "formal_validity": "Formal validity",
    "engineering_integrity": "Engineering",
    "process_completeness": "Completeness",
    "safety_adequacy": "Safety",
    "evidence_provenance": "Evidence",
    "decision_assurance": "Assurance",
    "actionability_calibration": "Calibration",
}
COLORS = {
    "qwen_one_shot": "#7b8794",
    "qwen_full": "#24796f",
    "gpt4o_one_shot": "#c47b35",
    "claude_one_shot": "#9a5d87",
    "gpt4o_full": "#245a9b",
    "claude_sonnet5_one_shot": "#9a5d87",
    "gpt56_terra_one_shot": "#c47b35",
}
FALLBACK_COLORS = ["#24796f", "#9a5d87", "#c47b35", "#245a9b", "#7b8794"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}_{key}" if prefix else key, item, output)
    elif not isinstance(value, (list, tuple)):
        output[prefix] = value


def load_results(experiment: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(experiment.glob("cross_model/**/run_summary.json")):
        run_dir = summary_path.parent
        parts = run_dir.relative_to(experiment / "cross_model").parts
        if len(parts) < 3:
            continue
        condition_id = parts[0]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata_payload = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        metadata = metadata_payload.get("metadata", metadata_payload)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "run_dir": str(run_dir.resolve()),
            "status": summary.get("status", "unknown"),
            "runtime_s": summary.get(
                "runtime_total_s", summary.get("runtime_s", 0)
            ),
            "llm_call_count": summary.get("llm_call_count", 0),
            "total_tokens": (summary.get("token_totals") or {}).get(
                "total_tokens", 0
            ),
            "repeat": int(parts[2].replace("repeat_", "")),
            "malformed_output": (
                result.get("output_validity") == "malformed_json"
            ),
        }
        _flatten("", metadata, row)
        _flatten("", metrics, row)
        rows.append(row)
    frame = pd.DataFrame(rows)
    formal = frame[
        "quality_assurance_dimensions_v2_formal_validity"
    ]
    frame["schema_neutral_quality_v2"] = (
        frame["quality_assurance_score_v2"] - 0.10 * formal
    ) / 0.90
    frame["valid_json"] = ~frame["malformed_output"]
    return frame


def matched_results(
    frame: pd.DataFrame,
    condition_ids: list[str],
) -> pd.DataFrame:
    completed = frame[frame["status"] == "completed"].copy()
    if completed.empty:
        return completed
    counts = (
        completed.groupby(["case_id", "repeat"])["condition_id"]
        .nunique()
        .reset_index(name="condition_count")
    )
    complete_keys = counts[counts["condition_count"] == len(condition_ids)][
        ["case_id", "repeat"]
    ]
    matched = completed.merge(complete_keys, on=["case_id", "repeat"], how="inner")
    return matched[matched["condition_id"].isin(condition_ids)].copy()


def summarize(frame: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    return (
        frame.groupby("condition_id", as_index=False)
        .agg(
            quality_mean=("quality_assurance_score_v2", "mean"),
            quality_std=("quality_assurance_score_v2", "std"),
            schema_neutral_quality_mean=("schema_neutral_quality_v2", "mean"),
            schema_neutral_quality_std=("schema_neutral_quality_v2", "std"),
            readiness_mean=("deployment_readiness_score_v2", "mean"),
            readiness_std=("deployment_readiness_score_v2", "std"),
            deployment_ready_rate=("deployment_ready_v2", "mean"),
            schema_valid_rate=("schema_valid", "mean"),
            valid_json_rate=("valid_json", "mean"),
            malformed_output_rate=("malformed_output", "mean"),
            gas_bookkeeping_rate=("gas_bookkeeping_complete", "mean"),
            geometry_consistency_rate=("geometry_consistent_10pct", "mean"),
            pump_feasibility_rate=("inventory_pump_feasible", "mean"),
            tubing_feasibility_rate=("inventory_tubing_feasible", "mean"),
            exact_reactor_match_rate=("inventory_exact_reactor_match", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_llm_calls=("llm_call_count", "mean"),
            mean_tokens=("total_tokens", "mean"),
            runs=("case_id", "size"),
            cases=("case_id", "nunique"),
        )
        .assign(condition_label=lambda data: data["condition_id"].map(labels))
    )


def paired_comparisons(
    frame: pd.DataFrame,
    comparisons: list[dict[str, str]],
    labels: dict[str, str],
    score_column: str = "quality_assurance_score_v2",
) -> pd.DataFrame:
    case_means = (
        frame.groupby(["case_id", "condition_id"], as_index=False)[
            score_column
        ]
        .mean()
    )
    pivot = case_means.pivot(
        index="case_id",
        columns="condition_id",
        values=score_column,
    )
    rng = np.random.default_rng(20260729)
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        treatment = comparison["treatment"]
        comparator = comparison["comparator"]
        if treatment not in pivot or comparator not in pivot:
            continue
        values = (pivot[treatment] - pivot[comparator]).dropna().to_numpy()
        bootstrap = rng.choice(values, size=(20000, len(values)), replace=True).mean(
            axis=1
        )
        rows.append(
            {
                "comparison_id": comparison["comparison_id"],
                "treatment": treatment,
                "treatment_label": labels[treatment],
                "comparator": comparator,
                "comparator_label": labels[comparator],
                "mean_delta": float(values.mean()),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "wins": int((values > 1e-12).sum()),
                "losses": int((values < -1e-12).sum()),
                "ties": int((np.abs(values) <= 1e-12).sum()),
                "cases": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def model_provenance(experiment: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    api_rows: list[dict[str, Any]] = []
    for condition_dir in sorted((experiment / "cross_model").iterdir()):
        if not condition_dir.is_dir():
            continue
        model_counts: dict[tuple[str, str], int] = {}
        api_counts: dict[tuple[str, str, str], int] = {}
        run_count = 0
        for event_path in sorted(condition_dir.glob("**/llm_events.jsonl")):
            run_count += 1
            for line in event_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                provider = str(event.get("provider", ""))
                model = str(event.get("model", ""))
                api_name = str(event.get("api_name", ""))
                model_key = (provider, model)
                api_key = (provider, model, api_name)
                model_counts[model_key] = model_counts.get(model_key, 0) + 1
                api_counts[api_key] = api_counts.get(api_key, 0) + 1
        for (provider, model), event_count in model_counts.items():
            model_rows.append(
                {
                    "condition_id": condition_dir.name,
                    "provider": provider,
                    "model": model,
                    "event_count": event_count,
                    "run_count": run_count,
                }
            )
        for (provider, model, api_name), event_count in api_counts.items():
            api_rows.append(
                {
                    "condition_id": condition_dir.name,
                    "provider": provider,
                    "model": model,
                    "api_name": api_name,
                    "event_count": event_count,
                }
            )
    return pd.DataFrame(model_rows), pd.DataFrame(api_rows)


def _savefig(figures: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(
        figures / f"{name}.png",
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        figures / f"{name}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def make_figures(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    paired_schema_neutral: pd.DataFrame,
    labels: dict[str, str],
    order: list[str],
    figures: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = [
        COLORS.get(value, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])
        for index, value in enumerate(order)
    ]
    ordered_summary = summary.set_index("condition_id").reindex(order).reset_index()

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=frame,
        x="condition_id",
        y="quality_assurance_score_v2",
        hue="condition_id",
        order=order,
        hue_order=order,
        palette=palette,
        legend=False,
        errorbar=("ci", 95),
    )
    plt.xticks(range(len(order)), [labels[value] for value in order], rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.ylabel("Quality and Assurance Score v2")
    plt.title("Matched cross-model design quality")
    _savefig(figures, "01_quality_comparison")

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=frame,
        x="condition_id",
        y="deployment_readiness_score_v2",
        hue="condition_id",
        order=order,
        hue_order=order,
        palette=palette,
        legend=False,
        errorbar=("ci", 95),
    )
    plt.xticks(range(len(order)), [labels[value] for value in order], rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.ylabel("Deployment readiness v2")
    plt.title("Readiness after hard engineering and safety gates")
    _savefig(figures, "02_deployment_readiness")

    dimension_columns = [
        f"quality_assurance_dimensions_v2_{dimension}" for dimension in DIMENSIONS
    ]
    dimensions = (
        frame.groupby("condition_id")[dimension_columns].mean().reindex(order)
    )
    dimensions.index = [labels[value] for value in dimensions.index]
    dimensions.columns = [DIMENSION_LABELS[value] for value in DIMENSIONS]
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        dimensions,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.8,
        linecolor="white",
    )
    plt.title("Quality dimension decomposition")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=25, ha="right")
    _savefig(figures, "03_quality_dimensions")

    case_means = (
        frame.groupby(["case_id", "condition_id"])[
            "quality_assurance_score_v2"
        ]
        .mean()
        .unstack()
        .reindex(columns=order)
    )
    case_means.columns = [labels[value] for value in case_means.columns]
    plt.figure(figsize=(13, 8))
    sns.heatmap(
        case_means,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.6,
        linecolor="white",
    )
    plt.title("Protocol-level mean quality across repetitions")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=25, ha="right")
    _savefig(figures, "04_case_quality_heatmap")

    if not paired.empty:
        plot = paired.sort_values("mean_delta")
        means = plot["mean_delta"].to_numpy()
        lower = means - plot["bootstrap_ci_low"].to_numpy()
        upper = plot["bootstrap_ci_high"].to_numpy() - means
        y = np.arange(len(plot))
        plt.figure(figsize=(11, 6))
        plt.errorbar(
            means,
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color="#24796f",
            ecolor="#66727c",
            capsize=4,
        )
        plt.axvline(0, color="#202020", linewidth=1)
        plt.yticks(
            y,
            [
                f"{row.treatment_label} minus {row.comparator_label}"
                for row in plot.itertuples()
            ],
        )
        plt.xlabel("Mean paired quality difference, 95% case bootstrap CI")
        plt.title("Predefined paired comparisons")
        _savefig(figures, "05_paired_quality_effects")

    checks = [
        "schema_valid_rate",
        "geometry_consistency_rate",
        "gas_bookkeeping_rate",
        "pump_feasibility_rate",
        "tubing_feasibility_rate",
        "exact_reactor_match_rate",
        "deployment_ready_rate",
    ]
    check_labels = [
        "Schema valid",
        "Geometry consistent",
        "Gas bookkeeping",
        "Pump feasible",
        "Tubing feasible",
        "Exact reactor",
        "Deployment ready",
    ]
    hard_checks = ordered_summary.set_index("condition_id")[checks]
    hard_checks.index = [labels[value] for value in hard_checks.index]
    hard_checks.columns = check_labels
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        hard_checks,
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.8,
        linecolor="white",
    )
    plt.title("Hard engineering and deployment checks")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=25, ha="right")
    _savefig(figures, "06_hard_check_rates")

    gate_columns = [
        f"deployment_gate_flags_v2_{reason}" for reason in DEPLOYMENT_CAPS_V2
    ]
    gates = frame.groupby("condition_id")[gate_columns].mean().reindex(order)
    gates.index = [labels[value] for value in gates.index]
    gates.columns = [reason.replace("_", " ") for reason in DEPLOYMENT_CAPS_V2]
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        gates,
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        cmap=sns.light_palette("#b44743", as_cmap=True),
        linewidths=0.8,
        linecolor="white",
    )
    plt.title("Deployment-gate activation rates")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=30, ha="right")
    _savefig(figures, "07_deployment_gate_rates")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, column, title, ylabel in (
        (axes[0], "mean_runtime_s", "Runtime", "Seconds per design"),
        (axes[1], "mean_llm_calls", "Model calls", "Calls per design"),
        (axes[2], "mean_tokens", "Token volume", "Tokens per design"),
    ):
        ax.bar(
            range(len(order)),
            ordered_summary[column],
            color=palette,
        )
        ax.set_xticks(
            range(len(order)),
            [labels[value] for value in order],
            rotation=25,
            ha="right",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    fig.suptitle("Execution cost and complexity", weight="bold")
    _savefig(figures, "08_execution_cost")

    architecture_rows = {}
    for condition_id in order:
        label = labels[condition_id]
        architecture_rows[label] = {
            "One-shot": float(
                ordered_summary.loc[
                    ordered_summary["condition_id"] == condition_id,
                    "quality_mean",
                ].iloc[0]
            )
            if "one_shot" in condition_id
            else np.nan,
            "Full FlowPilot": float(
                ordered_summary.loc[
                    ordered_summary["condition_id"] == condition_id,
                    "quality_mean",
                ].iloc[0]
            )
            if "full" in condition_id
            else np.nan,
        }
    architecture_matrix = pd.DataFrame.from_dict(
        architecture_rows, orient="index"
    )
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        architecture_matrix,
        annot=True,
        fmt=".3f",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=1,
        linecolor="white",
    )
    plt.title("Model by architecture quality matrix")
    plt.xlabel("")
    plt.ylabel("")
    _savefig(figures, "09_model_architecture_matrix")

    if not paired.empty:
        comparison_deltas = {}
        for comparison in paired.itertuples():
            comparison_deltas[
                f"{comparison.treatment_label}\nminus {comparison.comparator_label}"
            ] = (
                case_means[comparison.treatment_label]
                - case_means[comparison.comparator_label]
            )
        delta_frame = pd.DataFrame(comparison_deltas)
        plt.figure(figsize=(12, 7))
        sns.heatmap(
            delta_frame,
            annot=True,
            fmt="+.3f",
            center=0,
            cmap="RdYlGn",
            linewidths=0.6,
            linecolor="white",
        )
        plt.title("Protocol-level paired quality differences")
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=20, ha="right")
        _savefig(figures, "10_pairwise_differences_by_protocol")

    validity = ordered_summary.set_index("condition_id")[
        ["valid_json_rate", "schema_valid_rate"]
    ]
    validity.index = [labels[value] for value in validity.index]
    validity.columns = ["Valid JSON", "Strict FlowProposal schema"]
    plt.figure(figsize=(9, 5))
    sns.heatmap(
        validity,
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.8,
        linecolor="white",
    )
    plt.title("Output contract validity")
    plt.xlabel("")
    plt.ylabel("")
    _savefig(figures, "11_output_contract_validity")

    if not paired_schema_neutral.empty:
        plot = paired_schema_neutral.sort_values("mean_delta")
        means = plot["mean_delta"].to_numpy()
        lower = means - plot["bootstrap_ci_low"].to_numpy()
        upper = plot["bootstrap_ci_high"].to_numpy() - means
        y = np.arange(len(plot))
        plt.figure(figsize=(11, 6))
        plt.errorbar(
            means,
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color="#245a9b",
            ecolor="#66727c",
            capsize=4,
        )
        plt.axvline(0, color="#202020", linewidth=1)
        plt.yticks(
            y,
            [
                f"{row.treatment_label} minus {row.comparator_label}"
                for row in plot.itertuples()
            ],
        )
        plt.xlabel(
            "Mean paired schema-neutral quality difference, "
            "95% case bootstrap CI"
        )
        plt.title("Schema-neutral sensitivity analysis")
        _savefig(figures, "12_schema_neutral_pairwise_effects")


def write_report(
    output: Path,
    experiment: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    paired_schema_neutral: pd.DataFrame,
    scorer_hash: str,
    config: dict[str, Any],
) -> None:
    case_count = int(frame["case_id"].nunique())
    condition_count = int(frame["condition_id"].nunique())
    repeat_count = int(frame["repeat"].nunique())
    lines = [
        "# FlowPilot Reduced Frontier Benchmark",
        "",
        "## Design",
        "",
        f"- {case_count} stratified non-THQ batch-to-flow protocols.",
        f"- {condition_count} conditions with {repeat_count} repetitions each.",
        "- Matched protocol text and fixed condition assignment are used across models.",
        "- Temperature 0 is used where accepted. Claude Sonnet 5 and GPT-5.6 Terra require provider-default temperature; GPT-5.6 receives the fixed benchmark seed.",
        "- Qwen Full routes upstream, translation, engineering calculations, council, revision, and formatting through the complete FlowPilot workflow.",
        "- Commercial comparators are general one-shot systems and do not receive FlowPilot's deterministic engineering or council modules.",
        "- Shared OpenAI embeddings used by FlowPilot retrieval do not generate designs.",
        "- Quality and Assurance Score v2 was frozen before this run and does not receive the condition label.",
        f"- Frozen scorer SHA-256: `{scorer_hash}`.",
        "",
        "## Results",
        "",
        "| Condition | Quality v2 | Schema-neutral | Readiness v2 | Ready rate | Valid JSON | Schema valid | Runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("quality_mean", ascending=False).itertuples():
        lines.append(
            f"| {row.condition_label} | {row.quality_mean:.3f} | "
            f"{row.schema_neutral_quality_mean:.3f} | "
            f"{row.readiness_mean:.3f} | {row.deployment_ready_rate:.1%} | "
            f"{row.valid_json_rate:.1%} | {row.schema_valid_rate:.1%} | "
            f"{int(row.runs)} |"
        )
    lines.extend(
        [
            "",
            "## Predefined Paired Comparisons",
            "",
            "| Comparison | Mean delta | 95% case-bootstrap CI | W-L-T |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in paired.itertuples():
        lines.append(
            f"| {row.treatment_label} minus {row.comparator_label} | "
            f"{row.mean_delta:+.3f} | [{row.bootstrap_ci_low:+.3f}, "
            f"{row.bootstrap_ci_high:+.3f}] | "
            f"{row.wins}-{row.losses}-{row.ties} |"
        )
    lines.extend(
        [
            "",
            "## Schema-Neutral Sensitivity",
            "",
            "The formal-validity dimension (10% weight) is removed and the remaining dimensions are renormalized.",
            "",
            "| Comparison | Mean delta | 95% case-bootstrap CI | W-L-T |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in paired_schema_neutral.itertuples():
        lines.append(
            f"| {row.treatment_label} minus {row.comparator_label} | "
            f"{row.mean_delta:+.3f} | [{row.bootstrap_ci_low:+.3f}, "
            f"{row.bootstrap_ci_high:+.3f}] | "
            f"{row.wins}-{row.losses}-{row.ties} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Qwen Full versus either commercial one-shot condition is a system-level comparison, not a pure base-model comparison.",
            "- Claude Sonnet 5 versus GPT-5.6 Terra compares one-shot model behavior under the same prompt contract.",
            "- Quality and deployment readiness are separate outcomes; a screened design is not immediately executable.",
            "- The score rewards visible calculation, provenance, and independent audit evidence. This is appropriate for assurance evaluation but structurally favors systems that emit such evidence.",
            "- Five Claude Sonnet 5 responses reached the output-token limit with malformed or empty JSON and are scored as formal-validity failures without retrying.",
            "- Valid one-shot JSON still failed strict FlowProposal field typing; the resulting readiness cap measures contract noncompliance, not automatic chemical invalidity.",
            "- The schema-neutral sensitivity removes the formal-validity dimension; it does not remove engineering, safety, evidence, or assurance requirements.",
            "",
            "## Limitations",
            "",
            f"- {case_count} protocols are deliberately cost-controlled and do not establish wet-lab yield superiority.",
            "- Provider execution may vary because current Claude and GPT models do not accept temperature zero.",
            "- Qwen Full receives architecture and deterministic tooling unavailable to the commercial one-shot baselines.",
            "- Local-model throughput and frontier API latency are hardware/provider dependent.",
            "- Hidden literature references are machine-extracted and reference agreement is secondary.",
            "",
            f"Raw prompts, completions, snapshots, metrics, and checksums: `{experiment}`.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the matched cross-model benchmark report."
    )
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Condition config used to execute the benchmark.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    condition_ids = list(config["conditions"])
    labels = {
        key: value["label"] for key, value in config["conditions"].items()
    }
    frame = load_results(args.experiment)
    matched = matched_results(frame, condition_ids)
    expected_cases = len(config["profiles"]["publication"]["case_ids"])
    expected = (
        len(condition_ids)
        * expected_cases
        * config["repeats"]["publication"]
    )
    if len(matched) != expected:
        raise RuntimeError(
            f"Expected {expected} matched completed runs, found {len(matched)}."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    tables = args.output / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    summary = summarize(matched, labels)
    paired = paired_comparisons(
        matched,
        config["primary_comparisons"],
        labels,
    )
    paired_schema_neutral = paired_comparisons(
        matched,
        config["primary_comparisons"],
        labels,
        score_column="schema_neutral_quality_v2",
    )
    matched.to_csv(tables / "all_matched_runs.csv", index=False)
    summary.to_csv(tables / "condition_summary.csv", index=False)
    paired.to_csv(tables / "paired_comparisons.csv", index=False)
    paired_schema_neutral.to_csv(
        tables / "paired_schema_neutral_comparisons.csv",
        index=False,
    )

    dimension_columns = [
        f"quality_assurance_dimensions_v2_{dimension}" for dimension in DIMENSIONS
    ]
    matched.groupby("condition_id")[dimension_columns].mean().reset_index().to_csv(
        tables / "quality_dimensions.csv",
        index=False,
    )
    case_summary = (
        matched.groupby(["case_id", "condition_id"])
        .agg(
            quality_mean=("quality_assurance_score_v2", "mean"),
            quality_std=("quality_assurance_score_v2", "std"),
            readiness_mean=("deployment_readiness_score_v2", "mean"),
            ready_rate=("deployment_ready_v2", "mean"),
        )
        .reset_index()
    )
    case_summary.to_csv(tables / "case_condition_summary.csv", index=False)
    provenance, agent_calls = model_provenance(args.experiment)
    provenance.to_csv(tables / "model_provenance.csv", index=False)
    agent_calls.to_csv(tables / "agent_call_counts.csv", index=False)

    scorer_hash = _sha256(SCORER_PATH)
    plan = json.loads(
        (args.experiment / "execution_plan.json").read_text(encoding="utf-8")
    )
    planned_hash = plan["frozen_scorer"]["sha256"]
    if planned_hash != scorer_hash:
        raise RuntimeError(
            "Scorer changed after execution: "
            f"planned={planned_hash}, current={scorer_hash}"
        )
    summary_payload = {
        "schema_version": "flowpilot_cross_model_report_v1.0",
        "matched_runs": int(len(matched)),
        "cases": int(matched["case_id"].nunique()),
        "repeats": int(matched["repeat"].nunique()),
        "conditions": condition_ids,
        "winner_by_quality": summary.sort_values(
            "quality_mean", ascending=False
        ).iloc[0]["condition_id"],
        "winner_by_readiness": summary.sort_values(
            "readiness_mean", ascending=False
        ).iloc[0]["condition_id"],
        "scorer_sha256": scorer_hash,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    make_figures(
        matched,
        summary,
        paired,
        paired_schema_neutral,
        labels,
        condition_ids,
        figures,
    )
    write_report(
        args.output,
        args.experiment,
        matched,
        summary,
        paired,
        paired_schema_neutral,
        scorer_hash,
        config,
    )
    checksums = []
    for path in sorted(args.output.glob("**/*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(f"{_sha256(path)}  {path.relative_to(args.output)}")
    (args.output / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "matched_runs": len(matched),
                "figures": len(list(figures.glob("*.png"))),
                "tables": len(list(tables.glob("*.csv"))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
