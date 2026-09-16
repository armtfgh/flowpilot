from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

from ablation_test.src.cases import ROOT
from ablation_test.src.metrics import DEPLOYMENT_CAPS_V2
from ablation_test.src.paths import resolve_artifact_path


CONDITION_ORDER = [
    "qwen_one_shot",
    "qwen_full",
    "gpt4o_one_shot",
    "claude_one_shot",
    "gpt4o_full",
]
LABELS = {
    "qwen_one_shot": "Qwen 27B one-shot",
    "qwen_full": "Qwen 27B + Full FlowPilot",
    "gpt4o_one_shot": "GPT-4o one-shot",
    "claude_one_shot": "Claude Sonnet 4.6 one-shot",
    "gpt4o_full": "GPT-4o + Full FlowPilot",
}
SHORT_LABELS = {
    "qwen_one_shot": "Qwen one-shot",
    "qwen_full": "Qwen Full",
    "gpt4o_one_shot": "GPT-4o one-shot",
    "claude_one_shot": "Claude one-shot",
    "gpt4o_full": "GPT-4o Full",
}
COLORS = {
    "qwen_one_shot": "#7b8794",
    "qwen_full": "#24796f",
    "gpt4o_one_shot": "#c47b35",
    "claude_one_shot": "#9a5d87",
    "gpt4o_full": "#245a9b",
}
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
    "process_completeness": "Process completeness",
    "safety_adequacy": "Safety",
    "evidence_provenance": "Evidence",
    "decision_assurance": "Decision assurance",
    "actionability_calibration": "Calibration",
}
NUMERIC_FIELDS = {
    "residence_time_min": ("Residence time", "relative", 0.10),
    "flow_rate_mL_min": ("Liquid flow", "relative", 0.10),
    "temperature_C": ("Temperature", "absolute", 2.0),
    "concentration_M": ("Concentration", "relative", 0.10),
    "BPR_bar": ("BPR", "absolute", 0.5),
    "reactor_volume_mL": ("Reactor volume", "relative", 0.10),
    "tubing_ID_mm": ("Tubing ID", "absolute", 0.10),
}
CATEGORICAL_FIELDS = {
    "reactor_type": "Reactor type",
    "tubing_material": "Tubing material",
    "residence_time_basis": "Residence-time basis",
    "mixer_type": "Mixer type",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proposal(result: dict[str, Any]) -> dict[str, Any]:
    proposal = result.get("proposal")
    if isinstance(proposal, dict):
        return proposal
    candidate = result.get("final_design_candidate")
    if isinstance(candidate, dict) and isinstance(candidate.get("proposal"), dict):
        return candidate["proposal"]
    return {}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _aggregate_gas(proposal: dict[str, Any]) -> tuple[float, float, float]:
    stp = 0.0
    actual = 0.0
    equivalents = 0.0
    streams = proposal.get("streams") or []
    if isinstance(streams, dict):
        streams = list(streams.values())
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        phase = str(stream.get("phase", "")).lower()
        gas_stp = _number(stream.get("gas_flow_sccm")) or 0.0
        gas_actual = _number(stream.get("gas_flow_actual_mL_min")) or 0.0
        contents = " ".join(map(str, stream.get("contents") or [])).lower()
        is_gas = (
            phase == "gas"
            or gas_stp > 0
            or gas_actual > 0
            or any(
                token in contents
                for token in ("oxygen", "hydrogen", "chlorine", "ozone", "air")
            )
        )
        if not is_gas:
            continue
        stp += gas_stp
        actual += gas_actual
        equivalents += (
            _number(stream.get("molar_equiv"))
            or _number(stream.get("equivalents"))
            or 0.0
        )
    return stp, actual, equivalents


def enrich_raw_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    categorical: dict[str, list[Any]] = {
        field: [] for field in CATEGORICAL_FIELDS
    }
    gas_stp: list[float] = []
    gas_actual: list[float] = []
    gas_equiv: list[float] = []
    for run_dir in enriched["run_dir"]:
        result = json.loads(
            (resolve_artifact_path(str(run_dir)) / "result.json").read_text(
                encoding="utf-8"
            )
        )
        proposal = _proposal(result)
        for field in CATEGORICAL_FIELDS:
            categorical[field].append(str(proposal.get(field) or "").strip())
        stp, actual, equivalents = _aggregate_gas(proposal)
        gas_stp.append(stp)
        gas_actual.append(actual)
        gas_equiv.append(equivalents)
    for field, values in categorical.items():
        enriched[field] = values
    enriched["gas_flow_stp_total"] = gas_stp
    enriched["gas_flow_actual_total"] = gas_actual
    enriched["gas_equiv_total"] = gas_equiv
    return enriched


def within_tolerance(
    values: list[float] | np.ndarray,
    mode: str,
    tolerance: float,
) -> tuple[bool, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return False, float("nan")
    spread = float(np.ptp(array))
    if mode == "absolute":
        return spread <= tolerance, spread / tolerance if tolerance else float("inf")
    median = float(np.median(np.abs(array)))
    normalized = spread / median if median > 0 else float(spread > 0)
    return normalized <= tolerance, normalized / tolerance if tolerance else float("inf")


def _modal_agreement(values: pd.Series) -> float:
    cleaned = values.fillna("").astype(str)
    if cleaned.empty:
        return 0.0
    return float(cleaned.value_counts(dropna=False).iloc[0] / len(cleaned))


def build_reproducibility(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for (condition_id, case_id), group in frame.groupby(
        ["condition_id", "case_id"]
    ):
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "condition_label": LABELS[condition_id],
            "case_id": case_id,
            "quality_mean": group["quality_assurance_score_v2"].mean(),
            "quality_sd": group["quality_assurance_score_v2"].std(ddof=1),
            "quality_min": group["quality_assurance_score_v2"].min(),
            "readiness_mean": group["deployment_readiness_score_v2"].mean(),
            "readiness_sd": group["deployment_readiness_score_v2"].std(ddof=1),
            "deployment_decision_agreement": float(
                group["deployment_ready_v2"].nunique() == 1
            ),
            "deployment_ready_all_repeats": bool(
                group["deployment_ready_v2"].all()
            ),
            "schema_valid_all_repeats": bool(group["schema_valid"].all()),
        }
        numeric_passes: list[bool] = []
        for field, (label, mode, tolerance) in NUMERIC_FIELDS.items():
            values = pd.to_numeric(
                group[f"proposal_summary_{field}"],
                errors="coerce",
            ).dropna()
            passed, tolerance_units = within_tolerance(
                values.to_numpy(),
                mode,
                tolerance,
            )
            numeric_passes.append(passed)
            parameter_rows.append(
                {
                    "condition_id": condition_id,
                    "condition_label": LABELS[condition_id],
                    "case_id": case_id,
                    "parameter": field,
                    "parameter_label": label,
                    "within_tolerance": passed,
                    "spread_in_tolerance_units": tolerance_units,
                    "observed_min": values.min() if not values.empty else np.nan,
                    "observed_max": values.max() if not values.empty else np.nan,
                }
            )

        if bool(group["gas_required_by_case"].iloc[0]):
            for field, label in (
                ("gas_flow_stp_total", "Gas flow at STP"),
                ("gas_flow_actual_total", "Gas flow in channel"),
                ("gas_equiv_total", "Gas equivalents"),
            ):
                passed, tolerance_units = within_tolerance(
                    group[field].to_numpy(),
                    "relative",
                    0.10,
                )
                numeric_passes.append(passed)
                parameter_rows.append(
                    {
                        "condition_id": condition_id,
                        "condition_label": LABELS[condition_id],
                        "case_id": case_id,
                        "parameter": field,
                        "parameter_label": label,
                        "within_tolerance": passed,
                        "spread_in_tolerance_units": tolerance_units,
                        "observed_min": group[field].min(),
                        "observed_max": group[field].max(),
                    }
                )

        categorical_agreements = {
            field: _modal_agreement(group[field])
            for field in CATEGORICAL_FIELDS
        }
        for field, agreement in categorical_agreements.items():
            row[f"{field}_agreement"] = agreement
        row["mean_categorical_agreement"] = float(
            np.mean(list(categorical_agreements.values()))
        )
        row["all_numeric_within_tolerance"] = bool(all(numeric_passes))

        hard_pass = (
            group["geometry_consistent_10pct"].astype(bool)
            & group["inventory_pump_feasible"].astype(bool)
            & group["inventory_tubing_feasible"].astype(bool)
            & group["inventory_exact_reactor_match"].astype(bool)
            & group["gas_bookkeeping_complete"].astype(bool)
            & (group["topology_coverage"] >= 0.50)
            & (group["safety_coverage"] >= 0.50)
        )
        row["hard_engineering_pass_rate"] = float(hard_pass.mean())
        row["hard_engineering_all_repeats"] = bool(hard_pass.all())
        row["functional_reproducibility"] = bool(
            hard_pass.all() and all(numeric_passes)
        )
        row["strict_reproducibility"] = bool(
            row["functional_reproducibility"] and group["schema_valid"].all()
        )
        row["reliable_translation"] = bool(
            row["strict_reproducibility"]
            and (group["quality_assurance_score_v2"] >= 0.70).all()
        )
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(parameter_rows)


def cluster_bootstrap_summary(
    frame: pd.DataFrame,
    value_column: str,
    seed: int = 20260729,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    case_means = (
        frame.groupby(["condition_id", "case_id"])[value_column]
        .mean()
        .reset_index()
    )
    for condition_id, group in case_means.groupby("condition_id"):
        values = group[value_column].to_numpy()
        bootstrap = rng.choice(
            values,
            size=(20000, len(values)),
            replace=True,
        ).mean(axis=1)
        rows.append(
            {
                "condition_id": condition_id,
                "condition_label": LABELS[condition_id],
                "mean": float(values.mean()),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "case_count": len(values),
            }
        )
    return pd.DataFrame(rows)


def build_condition_overview(
    frame: pd.DataFrame,
    reproducibility: pd.DataFrame,
) -> pd.DataFrame:
    dimension_columns = {
        dimension: f"quality_assurance_dimensions_v2_{dimension}"
        for dimension in DIMENSIONS
    }
    quality = frame.groupby("condition_id").agg(
        quality_mean=("quality_assurance_score_v2", "mean"),
        quality_sd_all_runs=("quality_assurance_score_v2", "std"),
        quality_10th_percentile=("quality_assurance_score_v2", lambda x: x.quantile(0.10)),
        readiness_mean=("deployment_readiness_score_v2", "mean"),
        deployment_ready_rate=("deployment_ready_v2", "mean"),
        schema_valid_rate=("schema_valid", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
        mean_llm_calls=("llm_call_count", "mean"),
        mean_tokens=("total_tokens", "mean"),
    )
    dimension_means = frame.groupby("condition_id")[
        list(dimension_columns.values())
    ].mean()
    direct = (
        0.10 * dimension_means[dimension_columns["formal_validity"]]
        + 0.25 * dimension_means[dimension_columns["engineering_integrity"]]
        + 0.15 * dimension_means[dimension_columns["process_completeness"]]
        + 0.15 * dimension_means[dimension_columns["safety_adequacy"]]
    ) / 0.65
    assurance = (
        0.10 * dimension_means[dimension_columns["evidence_provenance"]]
        + 0.20 * dimension_means[dimension_columns["decision_assurance"]]
        + 0.05 * dimension_means[dimension_columns["actionability_calibration"]]
    ) / 0.35
    schema_neutral = (
        frame["quality_assurance_score_v2"]
        - 0.10
        * frame["quality_assurance_dimensions_v2_formal_validity"]
    ) / 0.90
    frame_with_sensitivity = frame.assign(schema_neutral_quality=schema_neutral)
    quality["schema_neutral_quality"] = frame_with_sensitivity.groupby(
        "condition_id"
    )["schema_neutral_quality"].mean()

    readiness_without_schema: list[float] = []
    for _, run in frame.iterrows():
        caps = [
            cap
            for reason, cap in DEPLOYMENT_CAPS_V2.items()
            if reason != "schema_invalid"
            and bool(run[f"deployment_gate_flags_v2_{reason}"])
        ]
        readiness_without_schema.append(
            min(
                float(run["quality_assurance_score_v2"]),
                min(caps, default=1.0),
            )
        )
    quality["readiness_without_schema_gate"] = (
        frame.assign(readiness_without_schema=readiness_without_schema)
        .groupby("condition_id")["readiness_without_schema"]
        .mean()
    )
    quality["direct_design_score"] = direct
    quality["assurance_score"] = assurance

    repeat = reproducibility.groupby("condition_id").agg(
        mean_within_protocol_quality_sd=("quality_sd", "mean"),
        mean_within_protocol_readiness_sd=("readiness_sd", "mean"),
        deployment_decision_agreement=("deployment_decision_agreement", "mean"),
        numeric_repeatability_rate=("all_numeric_within_tolerance", "mean"),
        categorical_agreement=("mean_categorical_agreement", "mean"),
        functional_reproducibility_rate=("functional_reproducibility", "mean"),
        strict_reproducibility_rate=("strict_reproducibility", "mean"),
        reliable_translation_rate=("reliable_translation", "mean"),
    )
    overview = quality.join(repeat).reset_index()
    overview["condition_label"] = overview["condition_id"].map(LABELS)
    return overview


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


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("condition_id").reindex(CONDITION_ORDER).reset_index()


def _bar_with_ci(
    summary: pd.DataFrame,
    ylabel: str,
    title: str,
    figures: Path,
    name: str,
) -> None:
    ordered = _ordered(summary)
    x = np.arange(len(ordered))
    means = ordered["mean"].to_numpy()
    errors = np.vstack(
        [
            means - ordered["ci_low"].to_numpy(),
            ordered["ci_high"].to_numpy() - means,
        ]
    )
    plt.figure(figsize=(11, 6))
    plt.bar(
        x,
        means,
        color=[COLORS[value] for value in CONDITION_ORDER],
        yerr=errors,
        capsize=5,
    )
    plt.xticks(
        x,
        [SHORT_LABELS[value] for value in CONDITION_ORDER],
        rotation=20,
        ha="right",
    )
    plt.ylim(0, 1)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.title(title, weight="bold")
    _savefig(figures, name)


def figure_study_design(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor=color,
                edgecolor="#30363b",
                linewidth=1.2,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#4a5359",
            )
        )

    box(0.3, 3.0, 2.1, 2.0, "12 non-THQ\nbatch protocols\n3 repeats each", "#e9eef1")
    conditions = [
        ("Qwen\none-shot", COLORS["qwen_one_shot"]),
        ("Qwen\nFull FlowPilot", COLORS["qwen_full"]),
        ("GPT-4o\none-shot", COLORS["gpt4o_one_shot"]),
        ("Claude\none-shot", COLORS["claude_one_shot"]),
        ("GPT-4o\nFull FlowPilot", COLORS["gpt4o_full"]),
    ]
    y_values = [6.4, 4.9, 3.4, 1.9, 0.4]
    for (text, color), y in zip(conditions, y_values):
        box(3.3, y, 2.2, 1.0, text, color)
        arrow(2.4, 4.0, 3.3, y + 0.5)

    box(
        6.5,
        2.6,
        2.5,
        2.8,
        "Saved evidence\n\nProposal JSON\nEngineering checks\nAgent logs\nCouncil snapshots\nRuntime and tokens",
        "#f4f6f7",
    )
    for y in y_values:
        arrow(5.5, y + 0.5, 6.5, 4.0)
    box(
        10.0,
        3.0,
        2.1,
        2.0,
        "Frozen, label-blind\nQuality and\nAssurance Score v2",
        "#dfece8",
    )
    arrow(9.0, 4.0, 10.0, 4.0)
    box(
        13.0,
        4.4,
        1.7,
        1.5,
        "Quality\nand paired\neffects",
        "#e5edf7",
    )
    box(
        13.0,
        2.1,
        1.7,
        1.5,
        "Readiness,\ngates, and\nrepeatability",
        "#f6eadd",
    )
    arrow(12.1, 4.0, 13.0, 5.15)
    arrow(12.1, 4.0, 13.0, 2.85)
    ax.text(
        7.5,
        7.75,
        "Matched FlowPilot cross-model and repeatability study",
        ha="center",
        va="top",
        fontsize=17,
        weight="bold",
    )
    _savefig(figures, "01_study_design")


def make_figures(
    frame: pd.DataFrame,
    overview: pd.DataFrame,
    reproducibility: pd.DataFrame,
    parameter_repeatability: pd.DataFrame,
    paired: pd.DataFrame,
    figures: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    figure_study_design(figures)

    quality_ci = cluster_bootstrap_summary(
        frame,
        "quality_assurance_score_v2",
    )
    _bar_with_ci(
        quality_ci,
        "Quality and Assurance Score v2",
        "Mean quality with protocol-clustered 95% bootstrap intervals",
        figures,
        "02_quality_clustered_ci",
    )

    ordered = _ordered(overview)
    x = np.arange(len(ordered))
    width = 0.37
    plt.figure(figsize=(11, 6))
    plt.bar(
        x - width / 2,
        ordered["direct_design_score"],
        width,
        color="#24796f",
        label="Direct design quality",
    )
    plt.bar(
        x + width / 2,
        ordered["assurance_score"],
        width,
        color="#66727c",
        label="Evidence and assurance",
    )
    plt.xticks(
        x,
        [SHORT_LABELS[value] for value in CONDITION_ORDER],
        rotation=20,
        ha="right",
    )
    plt.ylim(0, 1)
    plt.ylabel("Normalized component score")
    plt.title("Direct design quality versus assurance evidence", weight="bold")
    plt.legend(frameon=False)
    _savefig(figures, "03_direct_quality_vs_assurance")

    dimension_columns = [
        f"quality_assurance_dimensions_v2_{dimension}"
        for dimension in DIMENSIONS
    ]
    dimensions = (
        frame.groupby("condition_id")[dimension_columns]
        .mean()
        .reindex(CONDITION_ORDER)
    )
    dimensions.index = [SHORT_LABELS[value] for value in dimensions.index]
    dimensions.columns = [DIMENSION_LABELS[value] for value in DIMENSIONS]
    plt.figure(figsize=(12.5, 6))
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
    plt.title("Score decomposition by measured dimension", weight="bold")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=25, ha="right")
    _savefig(figures, "04_quality_dimensions")

    if not paired.empty:
        plot = paired.sort_values("mean_delta")
        y = np.arange(len(plot))
        means = plot["mean_delta"].to_numpy()
        errors = np.vstack(
            [
                means - plot["bootstrap_ci_low"].to_numpy(),
                plot["bootstrap_ci_high"].to_numpy() - means,
            ]
        )
        plt.figure(figsize=(11, 6))
        plt.errorbar(
            means,
            y,
            xerr=errors,
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
        plt.xlabel("Mean paired quality difference, 95% case-bootstrap CI")
        plt.title("Predefined paired comparisons", weight="bold")
        _savefig(figures, "05_paired_quality_effects")

    sensitivity = ordered[
        [
            "condition_id",
            "quality_mean",
            "schema_neutral_quality",
            "readiness_without_schema_gate",
        ]
    ].melt(
        id_vars="condition_id",
        var_name="analysis",
        value_name="score",
    )
    analysis_labels = {
        "quality_mean": "Frozen quality",
        "schema_neutral_quality": "Quality without formal-validity component",
        "readiness_without_schema_gate": "Readiness without schema cap",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    widths = 0.24
    for index, analysis in enumerate(analysis_labels):
        subset = sensitivity[sensitivity["analysis"] == analysis].set_index(
            "condition_id"
        ).reindex(CONDITION_ORDER)
        ax.bar(
            x + (index - 1) * widths,
            subset["score"],
            widths,
            label=analysis_labels[analysis],
        )
    ax.set_xticks(
        x,
        [SHORT_LABELS[value] for value in CONDITION_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Schema-neutral sensitivity analysis", weight="bold")
    ax.legend(frameon=False, fontsize=9)
    _savefig(figures, "06_schema_neutral_sensitivity")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x - width / 2,
        ordered["readiness_mean"],
        width,
        color=[COLORS[value] for value in CONDITION_ORDER],
        label="Mean readiness",
    )
    ax.bar(
        x + width / 2,
        ordered["deployment_ready_rate"],
        width,
        color="#b7c0c7",
        label="Immediately ready rate",
    )
    ax.set_xticks(
        x,
        [SHORT_LABELS[value] for value in CONDITION_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score or fraction")
    ax.set_title("Deployment readiness and immediate executability", weight="bold")
    ax.legend(frameon=False)
    _savefig(figures, "07_readiness_and_ready_rate")

    check_columns = [
        "schema_valid",
        "geometry_consistent_10pct",
        "gas_bookkeeping_complete",
        "inventory_pump_feasible",
        "inventory_tubing_feasible",
        "inventory_exact_reactor_match",
        "deployment_ready_v2",
    ]
    check_labels = [
        "Schema valid",
        "Geometry",
        "Gas bookkeeping",
        "Pump feasible",
        "Tubing feasible",
        "Exact reactor",
        "Deployment ready",
    ]
    checks = frame.groupby("condition_id")[check_columns].mean().reindex(
        CONDITION_ORDER
    )
    checks.index = [SHORT_LABELS[value] for value in checks.index]
    checks.columns = check_labels
    plt.figure(figsize=(12.5, 6))
    sns.heatmap(
        checks,
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.8,
        linecolor="white",
    )
    plt.title("Hard engineering and deployment checks", weight="bold")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=25, ha="right")
    _savefig(figures, "08_hard_check_rates")

    gate_columns = [
        f"deployment_gate_flags_v2_{reason}" for reason in DEPLOYMENT_CAPS_V2
    ]
    gates = frame.groupby("condition_id")[gate_columns].mean().reindex(
        CONDITION_ORDER
    )
    gates.index = [SHORT_LABELS[value] for value in gates.index]
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
    plt.title("Why designs were not immediately executable", weight="bold")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=30, ha="right")
    _savefig(figures, "09_deployment_gate_rates")

    case_quality = (
        frame.groupby(["case_id", "condition_id"])[
            "quality_assurance_score_v2"
        ]
        .mean()
        .unstack()
        .reindex(columns=CONDITION_ORDER)
    )
    case_quality.columns = [SHORT_LABELS[value] for value in case_quality.columns]
    plt.figure(figsize=(13, 8))
    sns.heatmap(
        case_quality,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.6,
        linecolor="white",
    )
    plt.title("Protocol-level mean quality over three repeats", weight="bold")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=25, ha="right")
    _savefig(figures, "10_case_quality_heatmap")

    rep_ordered = _ordered(overview)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x,
        rep_ordered["mean_within_protocol_quality_sd"],
        color=[COLORS[value] for value in CONDITION_ORDER],
    )
    ax.set_xticks(
        x,
        [SHORT_LABELS[value] for value in CONDITION_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Mean within-protocol quality SD")
    ax.set_title("Run-to-run quality variability; lower is better", weight="bold")
    _savefig(figures, "11_quality_repeatability")

    parameter_rates = (
        parameter_repeatability.groupby(["condition_id", "parameter_label"])[
            "within_tolerance"
        ]
        .mean()
        .unstack()
        .reindex(CONDITION_ORDER)
    )
    parameter_rates.index = [
        SHORT_LABELS[value] for value in parameter_rates.index
    ]
    preferred_order = [
        value
        for value in (
            "Residence time",
            "Liquid flow",
            "Temperature",
            "Concentration",
            "BPR",
            "Reactor volume",
            "Tubing ID",
            "Gas flow at STP",
            "Gas flow in channel",
            "Gas equivalents",
        )
        if value in parameter_rates.columns
    ]
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        parameter_rates.reindex(columns=preferred_order),
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.8,
        linecolor="white",
    )
    plt.title("Parameter repeatability within predefined tolerances", weight="bold")
    plt.xlabel("Gas columns include only protocols requiring a gas feed")
    plt.ylabel("")
    plt.xticks(rotation=25, ha="right")
    _savefig(figures, "12_parameter_repeatability")

    reproducibility_measures = [
        ("numeric_repeatability_rate", "All numeric parameters stable"),
        ("functional_reproducibility_rate", "Stable and engineering-feasible"),
        ("strict_reproducibility_rate", "Plus valid machine contract"),
        ("reliable_translation_rate", "Plus quality at least 0.70"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.19
    for index, (column, label) in enumerate(reproducibility_measures):
        ax.bar(
            x + (index - 1.5) * width,
            rep_ordered[column],
            width,
            label=label,
        )
    ax.set_xticks(
        x,
        [SHORT_LABELS[value] for value in CONDITION_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of protocols")
    ax.set_title("Functional reproducibility across three repeats", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    _savefig(figures, "13_functional_reproducibility")

    plt.figure(figsize=(10, 7))
    for row in rep_ordered.itertuples():
        plt.scatter(
            row.mean_within_protocol_quality_sd,
            row.quality_mean,
            s=150 + 500 * row.deployment_ready_rate,
            color=COLORS[row.condition_id],
            edgecolor="white",
            linewidth=1,
        )
        plt.annotate(
            SHORT_LABELS[row.condition_id],
            (row.mean_within_protocol_quality_sd, row.quality_mean),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    plt.xlabel("Mean within-protocol quality SD; lower is better")
    plt.ylabel("Mean quality; higher is better")
    plt.ylim(0.45, 1.0)
    plt.title("Quality-repeatability tradeoff", weight="bold")
    _savefig(figures, "14_quality_repeatability_tradeoff")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, column, title, ylabel in (
        (axes[0], "mean_runtime_s", "Runtime", "Seconds per run"),
        (axes[1], "mean_llm_calls", "Generative calls", "Calls per run"),
        (axes[2], "mean_tokens", "Token volume", "Tokens per run"),
    ):
        ax.bar(
            x,
            rep_ordered[column],
            color=[COLORS[value] for value in CONDITION_ORDER],
        )
        ax.set_xticks(
            x,
            [SHORT_LABELS[value] for value in CONDITION_ORDER],
            rotation=25,
            ha="right",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    fig.suptitle("Execution cost and complexity", weight="bold")
    _savefig(figures, "15_execution_cost")

    qwen_case = (
        frame[frame["condition_id"].isin(["qwen_one_shot", "qwen_full"])]
        .groupby(["case_id", "condition_id"])["quality_assurance_score_v2"]
        .mean()
        .unstack()
    )
    gain = (qwen_case["qwen_full"] - qwen_case["qwen_one_shot"]).sort_values()
    plt.figure(figsize=(10, 7))
    plt.barh(
        gain.index,
        gain.values,
        color=["#24796f" if value >= 0 else "#b44743" for value in gain],
    )
    plt.axvline(0, color="#202020", linewidth=1)
    plt.xlabel("Qwen Full minus Qwen one-shot quality")
    plt.title("Qwen architecture gain for every protocol", weight="bold")
    _savefig(figures, "16_qwen_gain_by_protocol")

    weights = pd.Series(
        {
            "Formal validity": 0.10,
            "Engineering integrity": 0.25,
            "Process completeness": 0.15,
            "Safety adequacy": 0.15,
            "Evidence provenance": 0.10,
            "Decision assurance": 0.20,
            "Actionability/calibration": 0.05,
        }
    ).sort_values()
    plt.figure(figsize=(9, 6))
    plt.barh(
        weights.index,
        weights.values,
        color=[
            "#24796f" if value in (
                "Formal validity",
                "Engineering integrity",
                "Process completeness",
                "Safety adequacy",
            ) else "#66727c"
            for value in weights.index
        ],
    )
    plt.xlabel("Weight in Quality and Assurance Score v2")
    plt.title("Frozen score anatomy: 65% direct design, 35% assurance", weight="bold")
    _savefig(figures, "17_score_weights")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_documentation(
    output: Path,
    experiment: Path,
    overview: pd.DataFrame,
    paired: pd.DataFrame,
    scorer_hash: str,
) -> None:
    by_id = overview.set_index("condition_id")
    lines = [
        "# FlowPilot Ablation and Repeatability Study",
        "",
        "## Purpose",
        "",
        "This package explains the completed FlowPilot ablation benchmark for a "
        "collaborator who did not build the software. It separates four questions:",
        "",
        "1. Does the full architecture improve design quality over a general one-shot LLM?",
        "2. Are the proposed designs internally consistent and compatible with laboratory hardware?",
        "3. Are cautious, audited designs actually ready for immediate execution?",
        "4. Does the same protocol produce functionally similar results across repeated runs?",
        "",
        "The benchmark evaluates engineering decision support. It does not establish "
        "superior wet-lab yield prediction.",
        "",
        "## Study Design",
        "",
        "- 12 literature-derived, non-THQ batch protocols.",
        "- 5 model/architecture conditions.",
        "- 3 independent runs per protocol and condition.",
        "- 180 completed runs with no failed cells.",
        "- Temperature zero and predetermined seeds.",
        "- The hidden literature flow result was excluded from model prompts and retrieval.",
        "- The frozen scorer did not receive the architecture label.",
        f"- Scorer SHA-256: `{scorer_hash}`.",
        "",
        "![Study design](figures/01_study_design.png)",
        "",
        "## Conditions",
        "",
        "| Condition | What it contains |",
        "|---|---|",
        "| Qwen 27B one-shot | One general prompt and one Qwen response |",
        "| Qwen 27B + Full FlowPilot | Lightweight Qwen upstream adapter, retrieval, deterministic engineering, inventory enforcement, Qwen council, skeptic, revision, and validation |",
        "| GPT-4o one-shot | One general prompt and one GPT-4o response |",
        "| Claude Sonnet 4.6 one-shot | One general prompt and one Claude response |",
        "| GPT-4o + Full FlowPilot | Full-schema GPT-4o upstream, retrieval, deterministic engineering, inventory enforcement, GPT-4o council, skeptic, revision, and validation |",
        "",
        "Both full conditions used the same shared OpenAI embedding service for "
        "retrieval. Embeddings did not generate any design text.",
        "",
        "## Main Results",
        "",
        "| Condition | Quality | Readiness | Ready rate | Quality SD | Reliable translation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition_id in sorted(
        CONDITION_ORDER,
        key=lambda value: by_id.loc[value, "quality_mean"],
        reverse=True,
    ):
        row = by_id.loc[condition_id]
        lines.append(
            f"| {LABELS[condition_id]} | {_fmt(row['quality_mean'])} | "
            f"{_fmt(row['readiness_mean'])} | "
            f"{row['deployment_ready_rate']:.1%} | "
            f"{_fmt(row['mean_within_protocol_quality_sd'], 4)} | "
            f"{row['reliable_translation_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "![Quality comparison](figures/02_quality_clustered_ci.png)",
            "",
            "### Primary interpretation",
            "",
            f"- Qwen Full achieved the highest mean quality "
            f"({_fmt(by_id.loc['qwen_full', 'quality_mean'])}).",
            f"- GPT-4o Full achieved the highest mean readiness "
            f"({_fmt(by_id.loc['gpt4o_full', 'readiness_mean'])}) and ready rate "
            f"({by_id.loc['gpt4o_full', 'deployment_ready_rate']:.1%}).",
            "- Qwen Full beat Qwen one-shot on every protocol.",
            "- Full FlowPilot improves traceability, structured validity, inventory compliance, gas bookkeeping, and independent review.",
            "- Full FlowPilot should still be supervised by a chemist because many outputs correctly requested experimental screening.",
            "",
            "## What The Score Measures",
            "",
            "The Quality and Assurance Score v2 is a weighted output-based score:",
            "",
            "```text",
            "0.10 Formal validity",
            "+ 0.25 Engineering integrity",
            "+ 0.15 Process completeness",
            "+ 0.15 Safety adequacy",
            "+ 0.10 Evidence provenance",
            "+ 0.20 Decision assurance",
            "+ 0.05 Actionability and calibration",
            "```",
            "",
            "The first four dimensions contribute 65% and measure direct design quality. "
            "The remaining 35% measures evidence, independent review, and calibrated actionability.",
            "",
            "![Score weights](figures/17_score_weights.png)",
            "",
            "![Score dimensions](figures/04_quality_dimensions.png)",
            "",
            "### Direct quality versus assurance",
            "",
            "One-shot systems received substantial engineering, process, and safety "
            "credit. They were not assigned zero. Their main deficits were strict "
            "machine-contract validity, retrieved-source provenance, deterministic "
            "calculation traces, inventory provenance, council deliberation, and final audit.",
            "",
            "![Direct versus assurance](figures/03_direct_quality_vs_assurance.png)",
            "",
            "## Paired Architecture Effects",
            "",
            "| Comparison | Mean difference | 95% case-bootstrap interval | W-L-T |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in paired.itertuples():
        lines.append(
            f"| {row.treatment_label} minus {row.comparator_label} | "
            f"{row.mean_delta:+.3f} | [{row.bootstrap_ci_low:+.3f}, "
            f"{row.bootstrap_ci_high:+.3f}] | "
            f"{int(row.wins)}-{int(row.losses)}-{int(row.ties)} |"
        )
    lines.extend(
        [
            "",
            "![Paired effects](figures/05_paired_quality_effects.png)",
            "",
            "Paired differences were calculated after averaging the three repetitions "
            "within each protocol. Confidence intervals used 20,000 protocol-level "
            "bootstrap samples, avoiding treatment of repeated runs as independent chemistries.",
            "",
            "## Schema-Neutral Sensitivity",
            "",
            "Every general one-shot output was parseable JSON but failed at least one "
            "strict FlowProposal field type. This set formal validity to zero and capped "
            "readiness at 0.35. To test whether the conclusion depended on that penalty, "
            "we recalculated quality without formal validity and readiness without the schema cap.",
            "",
            "![Schema sensitivity](figures/06_schema_neutral_sensitivity.png)",
            "",
            f"Without the formal-validity component, Qwen Full remained at "
            f"{_fmt(by_id.loc['qwen_full', 'schema_neutral_quality'])}, compared with "
            f"{_fmt(by_id.loc['qwen_one_shot', 'schema_neutral_quality'])} for Qwen "
            f"one-shot, {_fmt(by_id.loc['gpt4o_one_shot', 'schema_neutral_quality'])} "
            f"for GPT-4o one-shot, and "
            f"{_fmt(by_id.loc['claude_one_shot', 'schema_neutral_quality'])} for Claude one-shot.",
            "",
            "The architecture advantage therefore does not depend only on schema compliance. "
            "Nevertheless, a future confirmatory benchmark should include an exact-schema "
            "structured one-shot baseline.",
            "",
            "## Readiness Is Not Quality",
            "",
            "Quality rewards a well-supported and correctly cautious recommendation. "
            "Readiness asks whether the design can be executed immediately. Hard defects cap "
            "readiness even when the output is otherwise well documented.",
            "",
            "![Readiness](figures/07_readiness_and_ready_rate.png)",
            "",
            "![Hard checks](figures/08_hard_check_rates.png)",
            "",
            "![Deployment gates](figures/09_deployment_gate_rates.png)",
            "",
            "A screened design is not an experimental failure. It is a recommendation "
            "that additional validation is required before deployment.",
            "",
            "## Protocol-Level Behavior",
            "",
            "![Case quality](figures/10_case_quality_heatmap.png)",
            "",
            "![Qwen gain](figures/16_qwen_gain_by_protocol.png)",
            "",
            "Qwen Full improved quality over Qwen one-shot for all 12 protocols. "
            "The magnitude varied by chemistry, which is expected because gas-liquid, "
            "heterogeneous, hazardous, and multistep cases exercise different modules.",
            "",
            "## Repeatability And Reproducibility",
            "",
            "The current three repeated runs measure run-to-run repeatability under the "
            "same software and provider environment. Reproducibility across another server, "
            "software version, or date requires a separate rerun.",
            "",
            "A repeated answer is not automatically useful. A one-shot model can repeat the "
            "same non-deployable design. We therefore report several layers:",
            "",
            "- Numeric repeatability: all relevant parameters remain inside predefined tolerances.",
            "- Functional reproducibility: numeric repeatability plus geometry, pump, tubing, exact reactor inventory, gas, topology, and safety checks pass in every repeat.",
            "- Strict reproducibility: functional reproducibility plus valid structured output.",
            "- Reliable translation: strict reproducibility plus quality at least 0.70 in every repeat.",
            "",
            "Parameter tolerances were residence time, liquid flow, concentration, reactor "
            "volume, and gas quantities within 10%; temperature within 2 C; BPR within "
            "0.5 bar; and tubing ID within 0.10 mm. Gas-column percentages use only "
            "the protocols that require a gas feed.",
            "",
            "![Quality repeatability](figures/11_quality_repeatability.png)",
            "",
            "![Parameter repeatability](figures/12_parameter_repeatability.png)",
            "",
            "![Functional reproducibility](figures/13_functional_reproducibility.png)",
            "",
            "![Tradeoff](figures/14_quality_repeatability_tradeoff.png)",
            "",
            "### Repeatability interpretation",
            "",
            f"- Qwen Full had the lowest mean within-protocol quality SD "
            f"({_fmt(by_id.loc['qwen_full', 'mean_within_protocol_quality_sd'], 4)}).",
            f"- Qwen one-shot had the highest all-parameter numeric repeatability "
            f"({by_id.loc['qwen_one_shot', 'numeric_repeatability_rate']:.1%}), showing "
            f"that exact numeric stability does not by itself establish design quality.",
            f"- Qwen Full deployment decisions agreed across all three repeats for "
            f"{by_id.loc['qwen_full', 'deployment_decision_agreement']:.1%} of protocols.",
            f"- GPT-4o Full decision agreement was "
            f"{by_id.loc['gpt4o_full', 'deployment_decision_agreement']:.1%}.",
            "- Three repetitions are preliminary. A strong reproducibility claim should use at least 10 repeats on a stratified protocol subset.",
            "",
            "## Execution Cost",
            "",
            "![Execution cost](figures/15_execution_cost.png)",
            "",
            "Full FlowPilot requires substantially more calls and runtime because it "
            "performs retrieval, candidate generation, specialist scoring, skeptical audit, "
            "revision, and final validation. The Qwen local pipeline reduces commercial API "
            "dependence but is slower on the tested local server.",
            "",
            "## What Can Be Claimed",
            "",
            "> Under the frozen Quality and Assurance Score v2, Full FlowPilot produced "
            "more structured, engineering-consistent, inventory-aware, traceable, and "
            "safety-audited batch-to-flow designs than general one-shot prompting across "
            "12 matched protocols and three repetitions.",
            "",
            "## What Cannot Yet Be Claimed",
            "",
            "- Superior wet-lab yield prediction.",
            "- Identification of globally optimal operating conditions.",
            "- Safe autonomous operation without chemist review.",
            "- General superiority over every possible structured frontier-model prompt.",
            "- Cross-server or cross-version reproducibility from the current three-repeat study.",
            "",
            "## Recommended Next Validation",
            "",
            "1. Add an exact-schema structured one-shot baseline.",
            "2. Select six stratified protocols and run 10 repeats per condition.",
            "3. Repeat the frozen benchmark on another date or model deployment.",
            "4. Conduct blinded expert scoring without architecture labels.",
            "5. Compare predicted conditions with wet-lab conversion, yield, operability, and safety observations.",
            "",
            "## Files In This Package",
            "",
            "- `tables/condition_overview.csv`: headline quality, readiness, sensitivity, cost, and repeatability metrics.",
            "- `tables/reproducibility_by_case.csv`: protocol-condition repeatability outcomes.",
            "- `tables/parameter_repeatability.csv`: parameter-level spread and tolerance results.",
            "- `tables/quality_dimensions.csv`: score decomposition.",
            "- `tables/paired_comparisons.csv`: predefined paired effects.",
            "- `tables/model_provenance.csv`: provider/model evidence from call logs.",
            "- `tables/data_dictionary.csv`: definitions for derived variables.",
            "- `figures/`: PNG and vector PDF versions of every figure.",
            f"- Raw run directory: `{experiment}`.",
        ]
    )
    (output / "COWORKER_GUIDE.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_data_dictionary(output: Path) -> None:
    rows = [
        ("quality_mean", "Mean Quality and Assurance Score v2 over 36 runs."),
        ("readiness_mean", "Quality after the most restrictive deployment cap."),
        ("deployment_ready_rate", "Fraction of runs with no deployment gate."),
        ("schema_neutral_quality", "Quality after removing formal-validity contribution and renormalizing remaining weights."),
        ("readiness_without_schema_gate", "Readiness recalculated without the schema-invalid cap."),
        ("direct_design_score", "Normalized 65% block: formal validity, engineering, process, and safety."),
        ("assurance_score", "Normalized 35% block: evidence, decision assurance, and calibration."),
        ("mean_within_protocol_quality_sd", "Mean standard deviation across three repeats, calculated separately per protocol."),
        ("numeric_repeatability_rate", "Fraction of protocols for which every relevant numeric parameter stayed within tolerance."),
        ("categorical_agreement", "Mean modal agreement across reactor type, material, residence-time basis, and mixer."),
        ("functional_reproducibility_rate", "Numeric repeatability plus engineering, gas, topology, and safety checks passed in every repeat."),
        ("strict_reproducibility_rate", "Functional reproducibility plus valid structured output in every repeat."),
        ("reliable_translation_rate", "Strict reproducibility plus quality at least 0.70 in every repeat."),
    ]
    pd.DataFrame(rows, columns=["field", "definition"]).to_csv(
        output / "tables" / "data_dictionary.csv",
        index=False,
    )


def write_figure_guide(output: Path) -> None:
    figures = [
        (
            "01_study_design",
            "Benchmark structure",
            "Shows the 12 protocols, five matched conditions, saved evidence, frozen scorer, and two output families.",
            "This is a matched output-based evaluation; hidden references and architecture labels do not enter scoring.",
        ),
        (
            "02_quality_clustered_ci",
            "Primary quality result",
            "Mean Quality and Assurance Score v2 with protocol-clustered bootstrap intervals.",
            "Both full pipelines exceed all one-shot conditions; Qwen Full has the highest mean quality.",
        ),
        (
            "03_direct_quality_vs_assurance",
            "Where the advantage comes from",
            "Separates normalized direct design quality from evidence and assurance.",
            "Full FlowPilot leads on both, so its advantage is not only additional logging.",
        ),
        (
            "04_quality_dimensions",
            "Score decomposition",
            "Displays all seven measured dimensions for each condition.",
            "One-shot models receive real engineering, process, and safety credit but lack formal validity and audit evidence.",
        ),
        (
            "05_paired_quality_effects",
            "Matched architecture effects",
            "Shows protocol-level paired differences and 95% bootstrap intervals.",
            "Qwen Full beats each one-shot comparator for all 12 protocols.",
        ),
        (
            "06_schema_neutral_sensitivity",
            "Fairness check",
            "Removes the formal-validity component and schema readiness cap.",
            "The full-pipeline advantage remains, although the one-shot gap becomes smaller.",
        ),
        (
            "07_readiness_and_ready_rate",
            "Quality versus executability",
            "Compares mean capped readiness with the fraction of immediately executable runs.",
            "High assurance does not imply that every design is ready for unsupervised wet-lab execution.",
        ),
        (
            "08_hard_check_rates",
            "Engineering pass rates",
            "Shows schema, geometry, gas, pump, tubing, reactor, and deployment checks.",
            "Full pipelines consistently satisfy machine and inventory checks.",
        ),
        (
            "09_deployment_gate_rates",
            "Reasons for screening",
            "Shows how often each hard deployment gate activates.",
            "One-shot readiness is dominated by schema invalidity; full-pipeline readiness is mainly limited by explicit screening.",
        ),
        (
            "10_case_quality_heatmap",
            "Protocol-level performance",
            "Shows mean quality for each protocol and condition.",
            "The architecture effect is broad rather than driven by one chemistry.",
        ),
        (
            "11_quality_repeatability",
            "Run-to-run score stability",
            "Mean within-protocol quality standard deviation over three runs.",
            "Qwen Full has the lowest quality variability; lower is better.",
        ),
        (
            "12_parameter_repeatability",
            "Parameter stability",
            "Fraction of protocols whose repeated numerical values remain within predefined tolerances.",
            "One-shot Qwen is highly repetitive numerically, but gas-flow stability is weaker for the full systems; gas columns use only gas-required protocols.",
        ),
        (
            "13_functional_reproducibility",
            "Useful reproducibility",
            "Adds engineering feasibility, exact inventory, valid structure, and minimum quality to raw parameter stability.",
            "Both full pipelines retain 58.3% reliable translation; repetitive one-shot values often fail functional requirements.",
        ),
        (
            "14_quality_repeatability_tradeoff",
            "Joint performance",
            "Plots average quality against within-protocol variability.",
            "The desirable region is high and left; Qwen Full occupies the strongest position.",
        ),
        (
            "15_execution_cost",
            "Resource tradeoff",
            "Compares runtime, generative calls, and token volume.",
            "Full FlowPilot is substantially more expensive; local Qwen reduces frontier generation dependence but was slower on the tested server.",
        ),
        (
            "16_qwen_gain_by_protocol",
            "Matched Qwen architecture gain",
            "Subtracts Qwen one-shot quality from Qwen Full for each protocol.",
            "All values are positive, but the magnitude varies by chemistry.",
        ),
        (
            "17_score_weights",
            "Scoring transparency",
            "Displays the fixed weights used by Quality and Assurance Score v2.",
            "Direct design contributes 65%; evidence, review, and calibration contribute 35%.",
        ),
    ]
    lines = [
        "# Figure Reading Guide",
        "",
        "Use this document as a caption and interpretation reference. Each figure is "
        "available as PNG for presentations and PDF for manuscripts.",
        "",
        "| Figure | Purpose | What it shows | Main takeaway |",
        "|---|---|---|---|",
    ]
    for figure_id, purpose, description, takeaway in figures:
        lines.append(
            f"| `{figure_id}` | {purpose} | {description} | {takeaway} |"
        )
    lines.extend(
        [
            "",
            "## Reading Rules",
            "",
            "- Quality, readiness, and repeatability are different outcomes.",
            "- A lower variability value is favorable only when the repeated design is also valid and feasible.",
            "- A deployment-ready rate of zero does not mean that every chemical suggestion is useless; it means every run triggered at least one hard gate.",
            "- The one-shot schema penalty is shown transparently and tested with a schema-neutral sensitivity analysis.",
            "- Three repeats provide preliminary repeatability evidence, not cross-server reproducibility.",
        ]
    )
    (output / "FIGURE_GUIDE.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a coworker-facing FlowPilot ablation report."
    )
    parser.add_argument("--benchmark-package", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.benchmark_package / "tables" / "all_matched_runs.csv"
    frame = pd.read_csv(source)
    if len(frame) != 180:
        raise RuntimeError(f"Expected 180 matched runs, found {len(frame)}")
    if set(frame["condition_id"]) != set(CONDITION_ORDER):
        raise RuntimeError("Condition set does not match the frozen comparison")
    frame = enrich_raw_outputs(frame)
    reproducibility, parameter_repeatability = build_reproducibility(frame)
    overview = build_condition_overview(frame, reproducibility)
    paired = pd.read_csv(
        args.benchmark_package / "tables" / "paired_comparisons.csv"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    tables = args.output / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    overview.to_csv(tables / "condition_overview.csv", index=False)
    reproducibility.to_csv(tables / "reproducibility_by_case.csv", index=False)
    parameter_repeatability.to_csv(
        tables / "parameter_repeatability.csv",
        index=False,
    )
    frame.groupby("condition_id")[
        [
            f"quality_assurance_dimensions_v2_{dimension}"
            for dimension in DIMENSIONS
        ]
    ].mean().reset_index().to_csv(
        tables / "quality_dimensions.csv",
        index=False,
    )
    paired.to_csv(tables / "paired_comparisons.csv", index=False)
    for filename in ("model_provenance.csv", "agent_call_counts.csv"):
        pd.read_csv(args.benchmark_package / "tables" / filename).to_csv(
            tables / filename,
            index=False,
        )
    write_data_dictionary(args.output)
    write_figure_guide(args.output)

    make_figures(
        frame,
        overview,
        reproducibility,
        parameter_repeatability,
        paired,
        figures,
    )
    scorer_path = ROOT / "src" / "metrics.py"
    scorer_hash = _sha256(scorer_path)
    write_documentation(
        args.output,
        args.experiment,
        overview,
        paired,
        scorer_hash,
    )

    summary = {
        "schema_version": "flowpilot_coworker_report_v1.0",
        "source_benchmark": str(args.benchmark_package.resolve()),
        "source_experiment": str(args.experiment.resolve()),
        "matched_runs": int(len(frame)),
        "protocols": int(frame["case_id"].nunique()),
        "repeats": int(frame["repeat"].nunique()),
        "conditions": CONDITION_ORDER,
        "quality_winner": overview.sort_values(
            "quality_mean",
            ascending=False,
        ).iloc[0]["condition_id"],
        "readiness_winner": overview.sort_values(
            "readiness_mean",
            ascending=False,
        ).iloc[0]["condition_id"],
        "repeatability_winner": overview.sort_values(
            "mean_within_protocol_quality_sd",
        ).iloc[0]["condition_id"],
        "scorer_sha256": scorer_hash,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    checksums: list[str] = []
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
                "figures": len(list(figures.glob("*.png"))),
                "tables": len(list(tables.glob("*.csv"))),
                "matched_runs": len(frame),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
