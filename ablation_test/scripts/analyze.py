from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ablation_test.src.cases import ROOT, load_cases
from ablation_test.src.metrics import (
    DEPLOYMENT_CAPS_V2,
    ENGINEERING_V2_WEIGHTS,
    QUALITY_ASSURANCE_V2_WEIGHTS,
    score_run,
)
from ablation_test.src.runner import write_checksums
from ablation_test.src.paths import (
    EXPERT_SCORING_ROOT,
    FIGURES_ROOT,
    REPORTS_ROOT,
    RUNS_ROOT,
    TABLES_ROOT,
)


TABLES = TABLES_ROOT
FIGURES = FIGURES_ROOT
REPORTS = REPORTS_ROOT
EXPERT = EXPERT_SCORING_ROOT

VARIANT_ORDER = [
    "general_one_shot",
    "structured_single_agent",
    "no_retrieval",
    "no_engineering",
    "no_council",
    "no_inventory",
    "full",
]
VARIANT_LABELS = {
    "general_one_shot": "General one-shot",
    "structured_single_agent": "Structured single agent",
    "no_retrieval": "No retrieval",
    "no_engineering": "No engineering",
    "no_council": "No council",
    "no_inventory": "No inventory",
    "full": "Full FlowPilot",
}
CASE_LABELS = {
    "aerobic_oxidation_fmoc_methionine": "Aerobic oxidation",
    "hydrogenolysis_azetidinol": "Hydrogenolysis",
    "knoevenagel_benzaldehyde_malononitrile": "Knoevenagel",
    "photochemical_benzylic_bromination": "Photochemical bromination",
    "snar_4_fluoronitrobenzene_piperazine": "SNAr",
    "two_step_oxidative_amidation": "Two-step amidation",
}
QUALITY_V2_DIMENSIONS = {
    "formal_validity": "Formal validity",
    "engineering_integrity": "Engineering integrity",
    "process_completeness": "Process completeness",
    "safety_adequacy": "Safety adequacy",
    "evidence_provenance": "Evidence provenance",
    "decision_assurance": "Decision assurance",
    "actionability_calibration": "Actionability & calibration",
}
QUALITY_V2_WEIGHT_RANGES = {
    "formal_validity": (0.08, 0.15),
    "engineering_integrity": (0.20, 0.35),
    "process_completeness": (0.10, 0.20),
    "safety_adequacy": (0.12, 0.25),
    "evidence_provenance": (0.05, 0.15),
    "decision_assurance": (0.10, 0.25),
    "actionability_calibration": (0.03, 0.10),
}


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}_{key}" if prefix else key, item, output)
    elif not isinstance(value, (list, tuple)):
        output[prefix] = value


def _load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(RUNS_ROOT.glob("**/run_summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics_path = run_dir / "metrics.json"
        metadata_path = run_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = raw.get("metadata", raw)
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        row: dict[str, Any] = {
            "run_dir": str(run_dir),
            "arm": (
                "architecture"
                if "architecture" in run_dir.parts
                else "portability"
                if "portability" in run_dir.parts
                else "unknown"
            ),
            "status": summary.get("status", "unknown"),
            "runtime_s": summary.get("runtime_total_s", summary.get("runtime_s")),
            "llm_call_count": summary.get("llm_call_count", 0),
            "total_tokens": (summary.get("token_totals") or {}).get("total_tokens", 0),
        }
        _flatten("", metadata, row)
        _flatten("", metrics, row)
        rows.append(row)
    return rows


def _rescore_existing_runs(cases) -> None:
    by_id = {case.case_id: case for case in cases}
    for result_path in sorted(RUNS_ROOT.glob("**/result.json")):
        run_dir = result_path.parent
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = metadata_payload.get("metadata", metadata_payload)
        case = by_id.get(metadata.get("case_id"))
        if case is None:
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = score_run(case, result, run_dir)
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "completed":
                summary["external_metrics"] = metrics
                summary_path.write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        write_checksums(run_dir)


def _savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()


def _empty_figure(name: str, title: str, note: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.axis("off")
    plt.text(0.5, 0.60, title, ha="center", va="center", fontsize=16, weight="bold")
    plt.text(0.5, 0.42, note, ha="center", va="center", fontsize=11, wrap=True)
    _savefig(name)


def _primary_architecture(completed: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    architecture = completed[completed["arm"] == "architecture"].copy()
    if architecture.empty:
        return architecture, ""
    bundle_rank = []
    for bundle_name, group in architecture.groupby("bundle_name"):
        counts = group.groupby("variant")["case_id"].nunique()
        variants_with_two_cases = counts[counts >= 2]
        bundle_rank.append(
            (
                bundle_name,
                int("full" in variants_with_two_cases),
                len(variants_with_two_cases),
                int(variants_with_two_cases.min())
                if not variants_with_two_cases.empty
                else 0,
                int(counts.sum()),
            )
        )
    primary_bundle = max(
        bundle_rank,
        key=lambda item: (item[1], item[2], item[3], item[4]),
    )[0]
    architecture = architecture[architecture["bundle_name"] == primary_bundle]
    architecture = architecture.sort_values("run_dir").drop_duplicates(
        ["variant", "case_id"],
        keep="last",
    )
    variant_counts = architecture.groupby("variant")["case_id"].nunique()
    if not variant_counts.empty:
        coverage_floor = max(2, math.ceil(0.5 * variant_counts.max()))
        eligible_variants = set(
            variant_counts[variant_counts >= coverage_floor].index
        )
        architecture = architecture[architecture["variant"].isin(eligible_variants)]
        case_sets = [
            set(group["case_id"])
            for _, group in architecture.groupby("variant")
        ]
        common_cases = set.intersection(*case_sets) if case_sets else set()
        if common_cases:
            architecture = architecture[architecture["case_id"].isin(common_cases)]
    return architecture, primary_bundle


def _label_variants(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    labeled["variant_label"] = labeled["variant"].map(VARIANT_LABELS).fillna(
        labeled["variant"]
    )
    return labeled


def _suite_figures(cases) -> None:
    categories = Counter(case.category for case in cases)
    plt.figure(figsize=(10, 5.5))
    order = [name for name, _ in categories.most_common()]
    sns.barplot(x=[categories[name] for name in order], y=order, color="#2f6f6d")
    plt.title("Benchmark Case Coverage")
    plt.xlabel("Number of cases")
    plt.ylabel("")
    _savefig("01_case_category_coverage")

    literature = [case for case in cases if case.reference_flow]
    fields = ("residence_time_min", "flow_rate_mL_min", "temperature_C", "reactor_volume_mL")
    data = []
    for case in literature:
        for field in fields:
            value = case.reference_flow.get(field)
            if isinstance(value, (int, float)) and value > 0:
                data.append({"case_id": case.case_id, "field": field, "value": value})
    if data:
        frame = pd.DataFrame(data)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, field in zip(axes.flat, fields):
            subset = frame[frame["field"] == field]
            sns.stripplot(data=subset, x="value", ax=ax, color="#9b3a3a", size=7)
            if field != "temperature_C":
                ax.set_xscale("log")
            ax.set_title(field.replace("_", " "))
            ax.set_ylabel("")
        fig.suptitle("Hidden Reference Design Range", y=1.01, fontsize=15)
        _savefig("02_reference_design_range")

    hazard_counts = Counter()
    topology_counts = Counter()
    for case in cases:
        hazard_counts.update(case.expected_features.get("hazards", []))
        topology_counts.update(case.expected_features.get("topology", []))

    for name, counts, title, color in (
        ("03_hazard_coverage", hazard_counts, "Required Hazard Recognition Coverage", "#a64545"),
        ("04_topology_requirement_coverage", topology_counts, "Required Unit-Operation Coverage", "#3d668f"),
    ):
        common = counts.most_common(18)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=[count for _, count in common],
            y=[label.replace("_", " ") for label, _ in common],
            color=color,
        )
        plt.title(title)
        plt.xlabel("Cases requiring feature")
        plt.ylabel("")
        _savefig(name)


def _result_figures(frame: pd.DataFrame) -> None:
    completed = frame[frame["status"] == "completed"].copy()
    if completed.empty:
        for name, title in (
            ("05_architecture_composite", "Architecture Composite Score"),
            ("06_architecture_metric_heatmap", "Architecture Metric Heatmap"),
            ("07_model_portability", "Model Portability"),
            ("08_case_variant_heatmap", "Per-Case Architecture Scores"),
            ("09_runtime_by_variant", "Runtime by Variant"),
            ("10_token_cost_by_variant", "Token Use by Variant"),
            ("11_geometry_consistency", "Geometry Consistency"),
            ("12_inventory_feasibility", "Inventory Feasibility"),
            ("13_safety_topology", "Safety and Topology Coverage"),
            ("14_reference_accuracy", "Hidden-Reference Accuracy"),
            ("15_gas_bookkeeping", "Gas Bookkeeping"),
            ("16_score_vs_calls", "Score vs LLM Calls"),
            ("17_metric_correlation", "Metric Correlation"),
        ):
            _empty_figure(name, title, "No completed result cells were available when analysis ran.")
        return

    score = "deterministic_composite_score"
    architecture, primary_bundle = _primary_architecture(completed)
    architecture = _label_variants(architecture)

    if score in architecture:
        plt.figure(figsize=(10, 5.5))
        order = (
            architecture.groupby("variant_label")[score]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        sns.barplot(
            data=architecture,
            x=score,
            y="variant_label",
            order=order,
            color="#286f6b",
            errorbar=None,
        )
        plt.xlim(0, 1)
        plt.title(f"Architecture Composite Score ({primary_bundle}, matched model)")
        plt.xlabel("Deterministic composite (0-1)")
        plt.ylabel("")
        _savefig("05_architecture_composite")

    metric_cols = [
        "schema_valid",
        "numeric_completeness",
        "geometry_consistent_10pct",
        "inventory_pump_feasible",
        "inventory_tubing_feasible",
        "topology_coverage",
        "safety_coverage",
        "gas_bookkeeping_complete",
    ]
    available = [column for column in metric_cols if column in architecture]
    if available:
        numeric_metrics = architecture[["variant_label", *available]].copy()
        for column in available:
            numeric_metrics[column] = pd.to_numeric(
                numeric_metrics[column], errors="coerce"
            )
        heat = numeric_metrics.groupby("variant_label")[available].mean()
        plt.figure(figsize=(11, max(4, 0.55 * len(heat))))
        sns.heatmap(heat, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="RdYlGn", cbar_kws={"label": "Mean score"})
        plt.title("Architecture Metric Heatmap")
        plt.xlabel("")
        plt.ylabel("")
        _savefig("06_architecture_metric_heatmap")

    portability = completed[
        (completed["arm"] == "portability") & (completed["variant"] == "full")
    ]
    if not portability.empty and score in portability:
        plt.figure(figsize=(9, 5))
        order = portability.groupby("bundle_name")[score].mean().sort_values(ascending=False).index
        sns.barplot(data=portability, x=score, y="bundle_name", order=order, color="#536f9e", errorbar=None)
        plt.xlim(0, 1)
        plt.title("Full-Pipeline Model Portability")
        plt.xlabel("Deterministic composite (0-1)")
        plt.ylabel("")
        _savefig("07_model_portability")
    else:
        _empty_figure("07_model_portability", "Model Portability", "No completed full-pipeline portability cells.")

    if score in architecture and {"case_id", "variant"}.issubset(architecture):
        pivot = architecture.pivot_table(
            index="case_id",
            columns="variant_label",
            values=score,
            aggfunc="mean",
        )
        plt.figure(figsize=(12, max(5, 0.45 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="RdYlGn")
        plt.title("Per-Case Architecture Scores")
        plt.xlabel("")
        plt.ylabel("")
        _savefig("08_case_variant_heatmap")

    for name, column, title, xlabel, color in (
        ("09_runtime_by_variant", "runtime_s", "Runtime by Variant", "Seconds", "#7a5c91"),
        ("10_token_cost_by_variant", "total_tokens", "Token Use by Variant", "Tokens", "#8c6b32"),
    ):
        if column in architecture and architecture[column].notna().any():
            plt.figure(figsize=(10, 5.5))
            sns.barplot(
                data=architecture,
                x=column,
                y="variant_label",
                color=color,
                errorbar=None,
            )
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("")
            _savefig(name)
        else:
            _empty_figure(name, title, f"No {xlabel.lower()} telemetry available.")

    binary_panels = (
        ("11_geometry_consistency", "geometry_consistent_10pct", "Geometry Consistency"),
        ("12_inventory_feasibility", "inventory_pump_feasible", "Pump Feasibility"),
        ("15_gas_bookkeeping", "gas_bookkeeping_complete", "Gas Bookkeeping Completeness"),
    )
    for name, column, title in binary_panels:
        if column in architecture:
            binary = architecture[["variant_label", column]].copy()
            binary[column] = pd.to_numeric(binary[column], errors="coerce")
            values = (
                binary.groupby("variant_label")[column]
                .mean()
                .sort_values(ascending=False)
            )
            plt.figure(figsize=(9, 5))
            sns.barplot(x=values.values, y=values.index, color="#467c68")
            plt.xlim(0, 1)
            plt.title(title)
            plt.xlabel("Passing fraction")
            plt.ylabel("")
            _savefig(name)

    coverage_cols = [column for column in ("safety_coverage", "topology_coverage") if column in architecture]
    if coverage_cols:
        melted = architecture.melt(
            id_vars=["variant_label"],
            value_vars=coverage_cols,
            var_name="metric",
            value_name="score",
        )
        plt.figure(figsize=(10, 5.5))
        sns.barplot(
            data=melted,
            x="score",
            y="variant_label",
            hue="metric",
            errorbar=None,
        )
        plt.xlim(0, 1)
        plt.title("Safety and Topology Coverage")
        plt.xlabel("Coverage")
        plt.ylabel("")
        _savefig("13_safety_topology")

    if "reference_accuracy_score" in architecture and architecture["reference_accuracy_score"].notna().any():
        ref = architecture[architecture["reference_accuracy_score"].notna()]
        plt.figure(figsize=(10, 5.5))
        sns.boxplot(
            data=ref,
            x="reference_accuracy_score",
            y="variant_label",
            color="#c2d6cf",
        )
        sns.stripplot(
            data=ref,
            x="reference_accuracy_score",
            y="variant_label",
            color="#244d47",
            size=4,
        )
        plt.xlim(0, 1)
        plt.title("Hidden-Reference Accuracy")
        plt.xlabel("Reference similarity score")
        plt.ylabel("")
        _savefig("14_reference_accuracy")
    else:
        _empty_figure("14_reference_accuracy", "Hidden-Reference Accuracy", "No checkable hidden-reference values.")

    if score in architecture and "llm_call_count" in architecture:
        plt.figure(figsize=(8, 5.5))
        sns.scatterplot(
            data=architecture,
            x="llm_call_count",
            y=score,
            hue="variant_label",
            style="bundle_name" if "bundle_name" in completed else None,
            s=80,
        )
        plt.ylim(0, 1)
        plt.title("Quality-Cost Tradeoff")
        plt.xlabel("LLM calls")
        plt.ylabel("Deterministic composite")
        _savefig("16_score_vs_calls")

    correlation_cols = [
        column
        for column in (
            score,
            "numeric_completeness",
            "topology_coverage",
            "safety_coverage",
            "reference_accuracy_score",
            "runtime_s",
            "llm_call_count",
            "total_tokens",
        )
        if column in architecture and pd.api.types.is_numeric_dtype(architecture[column])
    ]
    if len(correlation_cols) >= 2:
        corr = architecture[correlation_cols].corr(numeric_only=True)
        plt.figure(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title("Metric Correlation")
        _savefig("17_metric_correlation")


def _status_figure(frame: pd.DataFrame) -> None:
    if frame.empty:
        _empty_figure("18_run_status", "Run Status", "No run summaries found.")
        return
    counts = frame["status"].value_counts()
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=counts.values, y=counts.index, color="#6d7480")
    plt.title("Run Status")
    plt.xlabel("Cells")
    plt.ylabel("")
    _savefig("18_run_status")


def _architecture_evidence_figures(frame: pd.DataFrame) -> None:
    modules = [
        "Structured schema",
        "Chemistry upstream",
        "Retrieval",
        "Engineering",
        "Inventory",
        "Council",
    ]
    presence = {
        "general_one_shot": [0, 0, 0, 0, 0, 0],
        "structured_single_agent": [1, 0, 0, 0, 0, 0],
        "no_retrieval": [1, 1, 0, 1, 1, 1],
        "no_engineering": [1, 1, 1, 0, 0, 0],
        "no_council": [1, 1, 1, 1, 1, 0],
        "no_inventory": [1, 1, 1, 1, 0, 1],
        "full": [1, 1, 1, 1, 1, 1],
    }
    matrix = pd.DataFrame(
        [presence[variant] for variant in VARIANT_ORDER],
        index=[VARIANT_LABELS[variant] for variant in VARIANT_ORDER],
        columns=modules,
    )
    plt.figure(figsize=(11.5, 5.5))
    ax = sns.heatmap(
        matrix,
        annot=np.where(matrix.values == 1, "ACTIVE", "REMOVED"),
        fmt="",
        cmap=sns.color_palette(["#d8dadd", "#2f766d"], as_cmap=True),
        cbar=False,
        linewidths=1,
        linecolor="white",
    )
    ax.set_title("FlowPilot Ablation Architecture Map")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    _savefig("20_ablation_architecture_map")

    completed = frame[frame["status"] == "completed"].copy()
    architecture, primary_bundle = _primary_architecture(completed)
    if primary_bundle:
        observed = frame[
            (frame["arm"] == "architecture")
            & (frame["bundle_name"] == primary_bundle)
        ].copy()
        observed = observed.sort_values("run_dir").drop_duplicates(
            ["variant", "case_id"],
            keep="last",
        )
        case_order = list(dict.fromkeys(observed["case_id"].dropna().tolist()))
        status_values = {"completed": 2, "failed": 1, "blocked_missing_credential": 1}
        coverage = pd.DataFrame(
            0,
            index=[VARIANT_LABELS[variant] for variant in VARIANT_ORDER],
            columns=case_order,
            dtype=int,
        )
        for _, row in observed.iterrows():
            variant = row.get("variant")
            case_id = row.get("case_id")
            if variant in VARIANT_LABELS and case_id in coverage:
                coverage.loc[VARIANT_LABELS[variant], case_id] = status_values.get(
                    row.get("status"),
                    1,
                )
        coverage = coverage.rename(columns=CASE_LABELS)
        annotations = np.full(coverage.shape, "NOT RUN", dtype=object)
        annotations[coverage.values == 1] = "FAILED"
        annotations[coverage.values == 2] = "DONE"
        plt.figure(figsize=(max(12, 1.15 * len(case_order)), 5.8))
        ax = sns.heatmap(
            coverage,
            annot=annotations,
            fmt="",
            cmap=sns.color_palette(["#e4e5e7", "#b94b48", "#3f806d"], as_cmap=True),
            vmin=0,
            vmax=2,
            cbar=False,
            linewidths=1,
            linecolor="white",
        )
        ax.set_title(f"Architecture Cell Coverage ({primary_bundle})")
        ax.set_xlabel("Protocol")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        _savefig("21_architecture_execution_coverage")
    else:
        _empty_figure(
            "21_architecture_execution_coverage",
            "Architecture Cell Coverage",
            "No completed architecture cells are available.",
        )

    score = "deterministic_composite_score"
    if (
        not architecture.empty
        and "full" in set(architecture["variant"])
        and score in architecture
    ):
        labeled = _label_variants(architecture)
        summary = (
            labeled.groupby(["variant", "variant_label"])[score]
            .agg(["mean", "count"])
            .reset_index()
        )
        full_mean = float(summary.loc[summary["variant"] == "full", "mean"].iloc[0])
        summary["delta_from_full"] = summary["mean"] - full_mean
        summary = summary.sort_values("mean", ascending=True)
        colors = [
            "#2f766d" if variant == "full" else "#66788a"
            for variant in summary["variant"]
        ]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1]})
        axes[0].barh(summary["variant_label"], summary["mean"], color=colors)
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel("Deterministic composite (0-1)")
        axes[0].set_title(f"Matched Architecture Scores ({primary_bundle})")
        for y, (_, row) in enumerate(summary.iterrows()):
            axes[0].text(
                min(row["mean"] + 0.015, 0.94),
                y,
                f'{row["mean"]:.2f} (n={int(row["count"])})',
                va="center",
                fontsize=9,
            )
        ablations = summary[summary["variant"] != "full"].copy()
        delta_colors = [
            "#3f806d" if value <= 0 else "#b94b48"
            for value in ablations["delta_from_full"]
        ]
        axes[1].barh(
            ablations["variant_label"],
            ablations["delta_from_full"],
            color=delta_colors,
        )
        axes[1].axvline(0, color="#222222", linewidth=1)
        axes[1].set_xlabel("Automated composite: ablated minus full")
        axes[1].set_title("Difference from Full Under Automated Metric")
        axes[1].text(
            0.5,
            -0.18,
            "Positive values do not establish greater scientific validity.\n"
            "The composite omits schema validity and council failure handling.",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#4f565a",
        )
        _savefig("22_full_vs_ablation_matched")
    else:
        _empty_figure(
            "22_full_vs_ablation_matched",
            "Full FlowPilot vs Ablations",
            "A matched full-system architecture run is not yet available.",
        )

    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    stages = [
        (0.2, 2.0, 1.8, "Batch\nprotocol"),
        (2.4, 2.0, 1.8, "Structured\nintake"),
        (4.6, 2.0, 1.8, "Upstream\nchemistry"),
        (6.8, 3.0, 1.8, "Literature\nretrieval"),
        (6.8, 1.0, 1.8, "Deterministic\nengineering"),
        (9.2, 2.0, 1.8, "Inventory-\nfeasible candidates"),
        (11.6, 2.0, 1.8, "Specialist\ncouncil"),
        (14.0, 2.0, 1.8, "Final flow\ndesign"),
    ]
    for x, y, width, label in stages:
        color = "#2f766d" if label == "Final flow\ndesign" else "#edf2f1"
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                1.0,
                facecolor=color,
                edgecolor="#2f4f4f",
                linewidth=1.4,
            )
        )
        ax.text(
            x + width / 2,
            y + 0.5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="white" if label == "Final flow\ndesign" else "#172525",
            weight="bold",
        )
    arrows = [
        ((2.0, 2.5), (2.4, 2.5)),
        ((4.2, 2.5), (4.6, 2.5)),
        ((6.4, 2.5), (6.8, 3.5)),
        ((6.4, 2.5), (6.8, 1.5)),
        ((8.6, 3.5), (9.2, 2.7)),
        ((8.6, 1.5), (9.2, 2.3)),
        ((11.0, 2.5), (11.6, 2.5)),
        ((13.4, 2.5), (14.0, 2.5)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#4d5c5c", "lw": 1.5},
        )
    ax.text(
        12.5,
        1.55,
        "Chemistry | Kinetics | Fluidics | Safety\nSkeptic audit | Chief selection",
        ha="center",
        va="top",
        fontsize=9,
        color="#374747",
    )
    ax.set_title(
        "Full FlowPilot System Evaluated in the Ablation Study",
        fontsize=15,
        weight="bold",
        pad=12,
    )
    _savefig("23_full_flowpilot_pipeline")

    if not architecture.empty:
        quality_cost = (
            _label_variants(architecture)
            .groupby(["variant", "variant_label"])
            .agg(
                schema_valid=("schema_valid", "mean"),
                llm_calls=("llm_call_count", "mean"),
            )
            .reset_index()
        )
        quality_cost["order"] = quality_cost["variant"].map(
            {variant: index for index, variant in enumerate(VARIANT_ORDER)}
        )
        quality_cost = quality_cost.sort_values("order", ascending=False)
        colors = [
            "#2f766d" if variant == "full" else "#66788a"
            for variant in quality_cost["variant"]
        ]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(
            quality_cost["variant_label"],
            quality_cost["schema_valid"],
            color=colors,
        )
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel("Fraction schema-valid")
        axes[0].set_title("Formal Output Validity")
        axes[1].barh(
            quality_cost["variant_label"],
            quality_cost["llm_calls"],
            color=colors,
        )
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Mean LLM calls per case (log scale)")
        axes[1].set_title("Architecture Execution Cost")
        fig.suptitle(
            "Validity and Cost of the Matched FlowPilot Architectures",
            fontsize=15,
            weight="bold",
        )
        _savefig("24_schema_validity_and_cost")
    else:
        _empty_figure(
            "24_schema_validity_and_cost",
            "Validity and Architecture Cost",
            "No matched architecture cells are available.",
        )


def _quality_v2_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    completed = frame[frame["status"] == "completed"].copy()
    architecture, primary_bundle = _primary_architecture(completed)
    score_column = "quality_assurance_score_v2"
    readiness_column = "deployment_readiness_score_v2"
    required = {score_column, readiness_column, "variant", "case_id"}
    if architecture.empty or not required.issubset(architecture):
        for name, title in (
            ("25_quality_assurance_score_v2", "Quality and Assurance Score v2"),
            ("26_quality_dimensions_v2", "Quality Score Dimensions"),
            ("27_full_pairwise_advantage_v2", "Full FlowPilot Pairwise Advantage"),
            ("28_weight_sensitivity_v2", "Weight Sensitivity"),
            ("29_deployment_readiness_v2", "Deployment Readiness"),
        ):
            _empty_figure(name, title, "No matched v2 score data are available.")
        return {}

    labeled = _label_variants(architecture)
    order = (
        labeled.groupby("variant_label")[score_column]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    summary = (
        labeled.groupby(["variant", "variant_label"])
        .agg(
            quality_mean=(score_column, "mean"),
            quality_std=(score_column, "std"),
            readiness_mean=(readiness_column, "mean"),
            readiness_std=(readiness_column, "std"),
            deployment_ready_rate=("deployment_ready_v2", "mean"),
            cases=("case_id", "nunique"),
        )
        .reset_index()
        .sort_values("quality_mean", ascending=False)
    )
    summary.to_csv(TABLES / "quality_score_v2_matched.csv", index=False)

    colors = [
        "#2f766d" if label == VARIANT_LABELS["full"] else "#6b7d8c"
        for label in order
    ]
    plt.figure(figsize=(10.5, 5.8))
    ax = plt.gca()
    quality_values = [
        float(
            summary.loc[
                summary["variant_label"] == label,
                "quality_mean",
            ].iloc[0]
        )
        for label in order
    ]
    ax.barh(order, quality_values, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(f"Quality and Assurance Score v2 ({primary_bundle}, matched)")
    ax.set_xlabel("Architecture-blind quality score (0-1)")
    ax.set_ylabel("")
    for y_position, value in enumerate(quality_values):
        ax.text(
            min(value + 0.012, 0.95),
            y_position,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    _savefig("25_quality_assurance_score_v2")

    dimension_columns = {
        dimension: f"quality_assurance_dimensions_v2_{dimension}"
        for dimension in QUALITY_V2_DIMENSIONS
    }
    available_dimensions = {
        dimension: column
        for dimension, column in dimension_columns.items()
        if column in architecture
    }
    dimension_means = (
        labeled.groupby(["variant", "variant_label"])[
            list(available_dimensions.values())
        ]
        .mean()
        .reset_index()
    )
    dimension_means.to_csv(TABLES / "quality_dimensions_v2_matched.csv", index=False)
    heat = (
        dimension_means.set_index("variant_label")[
            list(available_dimensions.values())
        ]
        .rename(
            columns={
                column: QUALITY_V2_DIMENSIONS[dimension]
                for dimension, column in available_dimensions.items()
            }
        )
        .reindex(order)
    )
    plt.figure(figsize=(12.5, 6))
    ax = sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Mean dimension score"},
    )
    ax.set_title("Why Architectures Receive Their Quality Score")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    _savefig("26_quality_dimensions_v2")

    pivot = architecture.pivot(
        index="case_id",
        columns="variant",
        values=score_column,
    )
    rng = np.random.default_rng(240728)
    pairwise_rows = []
    for variant in VARIANT_ORDER:
        if variant == "full" or variant not in pivot:
            continue
        differences = (pivot["full"] - pivot[variant]).dropna().to_numpy()
        bootstrap = rng.choice(
            differences,
            size=(20000, len(differences)),
            replace=True,
        ).mean(axis=1)
        positive = int((differences > 0).sum())
        negative = int((differences < 0).sum())
        non_ties = positive + negative
        minority = min(positive, negative)
        sign_p = (
            min(
                1.0,
                2.0
                * sum(
                    math.comb(non_ties, index)
                    for index in range(minority + 1)
                )
                / (2**non_ties),
            )
            if non_ties
            else 1.0
        )
        pairwise_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "mean_delta_full_minus_variant": float(differences.mean()),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "full_case_wins": positive,
                "full_case_losses": negative,
                "case_ties": int((differences == 0).sum()),
                "two_sided_sign_test_p": sign_p,
            }
        )
    pairwise = pd.DataFrame(pairwise_rows).sort_values(
        "mean_delta_full_minus_variant"
    )
    pairwise.to_csv(TABLES / "quality_score_v2_pairwise.csv", index=False)
    plt.figure(figsize=(10.5, 5.8))
    ax = plt.gca()
    y = np.arange(len(pairwise))
    means = pairwise["mean_delta_full_minus_variant"].to_numpy()
    lower = means - pairwise["bootstrap_ci_low"].to_numpy()
    upper = pairwise["bootstrap_ci_high"].to_numpy() - means
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#2f766d",
        ecolor="#6b7d8c",
        capsize=4,
    )
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y, pairwise["variant_label"])
    ax.set_xlabel("Full FlowPilot minus comparator (paired mean, 95% bootstrap CI)")
    ax.set_title("Full FlowPilot Pairwise Quality Advantage")
    _savefig("27_full_pairwise_advantage_v2")

    dimension_order = list(QUALITY_V2_DIMENSIONS)
    dimension_matrix = (
        architecture.groupby("variant")[
            [dimension_columns[name] for name in dimension_order]
        ]
        .mean()
        .reindex(VARIANT_ORDER)
    )
    range_low = np.array(
        [QUALITY_V2_WEIGHT_RANGES[name][0] for name in dimension_order]
    )
    range_high = np.array(
        [QUALITY_V2_WEIGHT_RANGES[name][1] for name in dimension_order]
    )
    winner_counts = Counter()
    margins = []
    sensitivity_samples = 20000
    for _ in range(sensitivity_samples):
        weights = rng.uniform(range_low, range_high)
        weights /= weights.sum()
        scores = dimension_matrix.to_numpy() @ weights
        winner_counts[dimension_matrix.index[int(np.argmax(scores))]] += 1
        full_index = dimension_matrix.index.get_loc("full")
        margins.append(
            float(
                scores[full_index]
                - np.max(np.delete(scores, full_index))
            )
        )
    sensitivity = pd.DataFrame(
        [
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "winner_count": winner_counts[variant],
                "winner_rate": winner_counts[variant] / sensitivity_samples,
            }
            for variant in VARIANT_ORDER
        ]
    ).sort_values("winner_rate", ascending=False)
    sensitivity.to_csv(TABLES / "quality_score_v2_sensitivity.csv", index=False)
    active_sensitivity = sensitivity[sensitivity["winner_count"] > 0]
    plt.figure(figsize=(9.5, 4.8))
    ax = sns.barplot(
        data=active_sensitivity,
        x="winner_rate",
        y="variant_label",
        color="#617789",
        errorbar=None,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction ranked first across 20,000 plausible weight sets")
    ax.set_ylabel("")
    ax.set_title("Sensitivity of the Architecture Winner to Score Weights")
    for patch, value in zip(ax.patches, active_sensitivity["winner_rate"]):
        ax.text(
            min(float(value) + 0.015, 0.95),
            patch.get_y() + patch.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=9,
        )
    _savefig("28_weight_sensitivity_v2")

    readiness_order = (
        summary.sort_values("readiness_mean", ascending=False)["variant_label"]
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    readiness_colors = [
        "#2f766d" if label == VARIANT_LABELS["full"] else "#6b7d8c"
        for label in readiness_order
    ]
    axes[0].barh(
        readiness_order,
        [
            float(
                summary.loc[
                    summary["variant_label"] == label,
                    "readiness_mean",
                ].iloc[0]
            )
            for label in readiness_order
        ],
        color=readiness_colors,
    )
    axes[0].set_xlim(0, 1)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Capped deployment-readiness score")
    axes[0].set_title("Readiness After Hard Quality Gates")
    axes[1].barh(
        readiness_order,
        [
            float(
                summary.loc[
                    summary["variant_label"] == label,
                    "deployment_ready_rate",
                ].iloc[0]
            )
            for label in readiness_order
        ],
        color=readiness_colors,
    )
    axes[1].set_xlim(0, 1)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Fraction with no deployment gate")
    axes[1].set_title("Immediately Executable Designs")
    fig.suptitle(
        "Quality Assurance Does Not Equal Deployment Readiness",
        fontsize=15,
        weight="bold",
    )
    _savefig("29_deployment_readiness_v2")

    gate_columns = {
        reason: f"deployment_gate_flags_v2_{reason}"
        for reason in DEPLOYMENT_CAPS_V2
    }
    available_gates = {
        reason: column
        for reason, column in gate_columns.items()
        if column in architecture
    }
    gate_frame = labeled[["variant_label", *available_gates.values()]].copy()
    for column in available_gates.values():
        gate_frame[column] = (
            gate_frame[column]
            .map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
            .fillna(pd.to_numeric(gate_frame[column], errors="coerce"))
            .fillna(0.0)
            .astype(float)
        )
    gate_means = (
        gate_frame.groupby("variant_label")[list(available_gates.values())]
        .mean()
        .rename(
            columns={
                column: reason.replace("_", " ")
                for reason, column in available_gates.items()
            }
        )
        .reindex(order)
    )
    gate_means.to_csv(TABLES / "deployment_gate_rates_v2.csv")
    plt.figure(figsize=(13, 6))
    ax = sns.heatmap(
        gate_means,
        annot=True,
        fmt=".0%",
        vmin=0,
        vmax=1,
        cmap=sns.light_palette("#b94b48", as_cmap=True),
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Fraction of cases triggering gate"},
    )
    ax.set_title("Deployment Gates Triggered by Each Architecture")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    _savefig("30_deployment_gate_rates_v2")

    score_specification = {
        "schema_version": "flowpilot_quality_score_v2.0",
        "status": "exploratory_reanalysis_to_freeze_for_future_holdout",
        "primary_score": "quality_assurance_score_v2",
        "primary_weights": QUALITY_ASSURANCE_V2_WEIGHTS,
        "engineering_subweights": ENGINEERING_V2_WEIGHTS,
        "deployment_caps": DEPLOYMENT_CAPS_V2,
        "weight_sensitivity_ranges": QUALITY_V2_WEIGHT_RANGES,
        "sensitivity_samples": sensitivity_samples,
        "bootstrap_samples": 20000,
        "architecture_label_used_in_scoring": False,
        "hidden_reference_used_in_primary_score": False,
    }
    (REPORTS / "quality_score_v2_specification.json").write_text(
        json.dumps(score_specification, indent=2),
        encoding="utf-8",
    )
    methodology_lines = [
        "# FlowPilot Quality and Assurance Score v2",
        "",
        "## Status",
        "",
        "This metric is an exploratory reanalysis designed after the current matched "
        "outputs were available. It is a candidate specification to freeze before "
        "prospective holdout validation; it is not preregistered evidence for the "
        "current dataset.",
        "",
        "## Principles",
        "",
        "- Score observable outputs and run artifacts, never the architecture label.",
        "- Keep direct design quality as the majority of the score.",
        "- Reward evidence provenance and auditable independent review.",
        "- Treat calibrated refusal as safer than confident failure, but not as an executable design.",
        "- Apply hard deployment caps independently of the weighted quality score.",
        "- Exclude the hidden literature reference from the primary score.",
        "",
        "## Primary Weights",
        "",
        "| Dimension | Weight |",
        "|:--|--:|",
        *[
            f"| {QUALITY_V2_DIMENSIONS[name]} | {weight:.0%} |"
            for name, weight in QUALITY_ASSURANCE_V2_WEIGHTS.items()
        ],
        "",
        "Direct design quality is formal validity, engineering integrity, process "
        "completeness, and safety adequacy, totaling 65%.",
        "",
        "## Engineering Integrity",
        "",
        "| Check | Subweight |",
        "|:--|--:|",
        *[
            f"| {name.replace('_', ' ').title()} | {weight:.0%} |"
            for name, weight in ENGINEERING_V2_WEIGHTS.items()
        ],
        "",
        "## Evidence And Assurance",
        "",
        "- Evidence provenance: 70% verified, contamination-free retrieved records and 30% field-level reasoning coverage.",
        "- Decision assurance: calculation trace 25%, deliberation trace 20%, safety review 15%, final audit 15%, real-inventory provenance 15%, documented confidence 10%.",
        "- Actionability/calibration: 60% executable output and 40% confidence calibrated to detected defects.",
        "",
        "## Deployment Gates",
        "",
        "| Gate | Maximum readiness score |",
        "|:--|--:|",
        *[
            f"| {name.replace('_', ' ').title()} | {cap:.2f} |"
            for name, cap in DEPLOYMENT_CAPS_V2.items()
        ],
        "",
        "## Statistical Analysis",
        "",
        "- Architecture comparison uses the same model and exact six-case intersection.",
        "- Pairwise uncertainty uses 20,000 case-level bootstrap resamples.",
        "- Weight sensitivity uses 20,000 uniformly sampled plausible weight sets, normalized to sum to one.",
        "- Case wins and a two-sided sign test are reported without treating the six cases as a large sample.",
        "",
        "## Required Prospective Test",
        "",
        "Freeze this specification, evaluate new blinded protocols, complete blinded "
        "expert review, and report quality, deployment readiness, cost, and wet-lab "
        "outcomes separately. Do not claim overall superiority from the current "
        "post-hoc reanalysis alone.",
    ]
    (REPORTS / "quality_score_v2_methodology.md").write_text(
        "\n".join(methodology_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "primary_bundle": primary_bundle,
        "summary": summary,
        "pairwise": pairwise,
        "sensitivity": sensitivity,
        "full_margin_sensitivity_ci": [
            float(np.quantile(margins, 0.025)),
            float(np.quantile(margins, 0.975)),
        ],
    }


def _write_tables(frame: pd.DataFrame, cases) -> None:
    frame.to_csv(TABLES / "all_runs.csv", index=False)
    completed = frame[frame["status"] == "completed"] if not frame.empty else frame
    metrics = [
        column
        for column in (
            "schema_valid",
            "quality_assurance_score_v2",
            "deployment_readiness_score_v2",
            "deployment_ready_v2",
            "deterministic_composite_score",
            "numeric_completeness",
            "geometry_consistent_10pct",
            "inventory_pump_feasible",
            "inventory_tubing_feasible",
            "topology_coverage",
            "safety_coverage",
            "gas_bookkeeping_complete",
            "reference_accuracy_score",
            "runtime_s",
            "llm_call_count",
            "total_tokens",
        )
        if column in completed
    ]
    architecture, _ = _primary_architecture(completed)
    if not architecture.empty and metrics:
        architecture.groupby("variant")[metrics].agg(["mean", "std", "count"]).to_csv(
            TABLES / "aggregate_architecture.csv"
        )
        full = completed[
            (completed["arm"] == "portability") & (completed["variant"] == "full")
        ]
        if not full.empty:
            full.groupby("bundle_name")[metrics].agg(["mean", "std", "count"]).to_csv(
                TABLES / "aggregate_portability.csv"
            )
    else:
        pd.DataFrame().to_csv(TABLES / "aggregate_architecture.csv", index=False)
        pd.DataFrame().to_csv(TABLES / "aggregate_portability.csv", index=False)

    case_rows = [case.manifest_payload() for case in cases]
    pd.json_normalize(case_rows).to_csv(TABLES / "case_manifest.csv", index=False)

    definitions = [
        (
            "quality_assurance_score_v2",
            "Architecture-blind weighted score of formal validity, engineering integrity, process completeness, safety, evidence provenance, decision assurance, and calibrated actionability.",
        ),
        (
            "deployment_readiness_score_v2",
            "Quality Assurance Score v2 capped by schema, gas, geometry, inventory, critical coverage, and screen-required gates.",
        ),
        (
            "deployment_ready_v2",
            "Whether no deployment gate was triggered; stricter than receiving a high quality-assurance score.",
        ),
        ("schema_valid", "Whether the result satisfies the formal FlowPilot output schema."),
        ("numeric_completeness", "Fraction of seven required numeric design fields present."),
        ("geometry_consistent_10pct", "Whether stated volume agrees with stated flow and residence-time basis within 10%."),
        ("inventory_pump_feasible", "Whether at least one real laboratory pump supports the stated flow and pressure."),
        ("inventory_tubing_feasible", "Whether stated ID, temperature, and pressure match available tubing."),
        ("topology_coverage", "Keyword-audited coverage of required unit operations for the case."),
        ("safety_coverage", "Keyword-audited coverage of case-specific hazards and controls."),
        ("gas_bookkeeping_complete", "Gas designs separately report STP flow, in-channel flow, and equivalents."),
        ("unexpected_gas_stream", "A gas feed appears even though the case does not require a gas reagent."),
        ("reference_accuracy_score", "Exponential score from log-ratio errors against the hidden literature flow reference."),
        ("deterministic_composite_score", "Unweighted mean of completeness, geometry, pump, tubing, topology, safety, and gas bookkeeping."),
    ]
    pd.DataFrame(definitions, columns=["metric", "definition"]).to_csv(
        TABLES / "metric_definitions.csv", index=False
    )

    module_rows = []
    module_presence = {
        "general_one_shot": [0, 0, 0, 0, 0, 0],
        "structured_single_agent": [1, 0, 0, 0, 0, 0],
        "no_retrieval": [1, 1, 0, 1, 1, 1],
        "no_engineering": [1, 1, 1, 0, 0, 0],
        "no_council": [1, 1, 1, 1, 1, 0],
        "no_inventory": [1, 1, 1, 1, 0, 1],
        "full": [1, 1, 1, 1, 1, 1],
    }
    module_names = [
        "structured_schema",
        "chemistry_upstream",
        "retrieval",
        "deterministic_engineering",
        "inventory_enforcement",
        "specialist_council",
    ]
    for variant in VARIANT_ORDER:
        module_rows.append(
            {
                "variant": variant,
                "display_name": VARIANT_LABELS[variant],
                **dict(zip(module_names, module_presence[variant])),
            }
        )
    pd.DataFrame(module_rows).to_csv(
        TABLES / "architecture_module_matrix.csv",
        index=False,
    )

    failures = frame[frame["status"] != "completed"] if not frame.empty else frame
    failures.to_csv(TABLES / "failure_analysis.csv", index=False)


def _write_expert_sheets(frame: pd.DataFrame) -> None:
    completed = frame[frame["status"] == "completed"] if not frame.empty else frame
    review_rows = []
    key_rows = []
    for _, row in completed.iterrows():
        identity = f"{row.get('run_dir', '')}"
        blind_id = "FP-" + hashlib.sha256(identity.encode()).hexdigest()[:10].upper()
        review_rows.append(
            {
                "blind_id": blind_id,
                "case_id": row.get("case_id", ""),
                "chemistry_correctness_1_5": "",
                "engineering_correctness_1_5": "",
                "safety_adequacy_1_5": "",
                "experimental_feasibility_1_5": "",
                "usefulness_as_first_experiment_1_5": "",
                "fatal_flaw_yes_no": "",
                "fatal_flaw_description": "",
                "reviewer_confidence_1_5": "",
                "comments": "",
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "variant": row.get("variant", ""),
                "bundle": row.get("bundle_name", ""),
                "model": row.get("model", ""),
                "run_dir": identity,
            }
        )
    pd.DataFrame(review_rows).to_csv(EXPERT / "expert_scoring_blinded.csv", index=False)
    pd.DataFrame(key_rows).to_csv(EXPERT / "blinding_key_confidential.csv", index=False)


def _classify_agent_component(event: dict[str, Any]) -> str:
    system_header = str(event.get("system_prompt") or "")[:300].upper()
    role_markers = (
        ("CHIEF ENGINEER", "Council: chief"),
        ("REVISION ENGINEER", "Council: revision engineer"),
        ("THE DESIGNER", "Council: design strategy"),
        ("PROBLEM FRAMER", "Council: problem framing"),
        ("DR. SAFETY PERFORMING", "Council: post-selection DFMEA"),
        ("DR. CHEMISTRY", "Council: chemistry"),
        ("DR. KINETICS", "Council: kinetics"),
        ("DR. FLUIDICS", "Council: fluidics"),
        ("DR. SAFETY", "Council: safety"),
    )
    for marker, label in role_markers:
        if marker in system_header:
            return label
    return str(event.get("api_name") or event.get("component") or "unknown")


def _write_agent_call_evidence() -> None:
    rows: list[dict[str, Any]] = []
    trace_lines = [
        "# Agent Call Trace",
        "",
        "This index proves which model-backed components were called. Full prompt and "
        "completion bodies remain in each run's `llm_events.jsonl`.",
        "",
    ]
    for event_path in sorted(RUNS_ROOT.glob("**/llm_events.jsonl")):
        run_dir = event_path.parent
        metadata_path = run_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", payload)
        events = []
        for index, line in enumerate(event_path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            component = _classify_agent_component(event)
            usage = event.get("usage") or {}
            prompt = f"{event.get('system_prompt', '')}\n{event.get('user_prompt', '')}"
            response = str(
                event.get("response_text")
                or event.get("raw_response_text")
                or ""
            )
            total_tokens = (
                usage.get("total_tokens")
                or (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
                or (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)
            )
            row = {
                "run_dir": str(run_dir),
                "case_id": metadata.get("case_id", ""),
                "variant": metadata.get("variant", ""),
                "bundle": metadata.get("bundle_name", ""),
                "event_index": index,
                "component": component,
                "provider": event.get("provider", ""),
                "model": event.get("model", ""),
                "duration_ms": event.get("duration_ms"),
                "total_tokens": total_tokens,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt.strip() else "",
                "response_sha256": hashlib.sha256(response.encode()).hexdigest() if response else "",
                "prompt_captured": bool(prompt.strip()),
                "response_captured": bool(response),
            }
            rows.append(row)
            events.append(row)
        if events and metadata.get("variant") == "full":
            trace_lines.extend(
                [
                    f"## {metadata.get('bundle_name', 'unknown')} / {metadata.get('case_id', 'unknown')}",
                    "",
                    f"- Run: `{run_dir}`",
                    f"- Calls: {len(events)}",
                    f"- Components: {', '.join(dict.fromkeys(row['component'] for row in events))}",
                    f"- Prompt bodies captured: {sum(row['prompt_captured'] for row in events)}/{len(events)}",
                    f"- Response bodies captured: {sum(row['response_captured'] for row in events)}/{len(events)}",
                    "",
                ]
            )

    call_frame = pd.DataFrame(rows)
    call_frame.to_csv(TABLES / "agent_call_events.csv", index=False)
    if not call_frame.empty:
        matrix = (
            call_frame.groupby(["variant", "bundle", "component"])
            .size()
            .rename("call_count")
            .reset_index()
        )
        matrix.to_csv(TABLES / "agent_call_matrix.csv", index=False)
        full_calls = call_frame[call_frame["variant"] == "full"]
        plotted = full_calls if not full_calls.empty else call_frame
        top = plotted["component"].value_counts().head(20).sort_values()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top.values, y=top.index, color="#566b78")
        plt.title(
            "Recorded Full-FlowPilot Calls by Pipeline Component"
            if not full_calls.empty
            else "Recorded Model Calls by Pipeline Component"
        )
        plt.xlabel("Calls")
        plt.ylabel("")
        _savefig("19_agent_component_calls")
    else:
        pd.DataFrame(
            columns=["variant", "bundle", "component", "call_count"]
        ).to_csv(
            TABLES / "agent_call_matrix.csv", index=False
        )
        _empty_figure(
            "19_agent_component_calls",
            "Recorded Agent Calls",
            "No LLM event logs were found.",
        )
    (REPORTS / "agent_trace_summary.md").write_text(
        "\n".join(trace_lines) + "\n",
        encoding="utf-8",
    )


def _write_report(
    frame: pd.DataFrame,
    cases,
    quality_v2: dict[str, Any] | None = None,
) -> None:
    completed = frame[frame["status"] == "completed"] if not frame.empty else frame
    status_counts = frame["status"].value_counts().to_dict() if not frame.empty else {}
    architecture_rows, primary_bundle = _primary_architecture(completed)
    matched_cases = (
        architecture_rows["case_id"].nunique()
        if not architecture_rows.empty
        else 0
    )
    matched_variants = (
        architecture_rows["variant"].nunique()
        if not architecture_rows.empty
        else 0
    )
    lines = [
        "# FlowPilot Ablation Benchmark Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        f"- Frozen literature-derived cases: {sum(bool(case.reference_flow) for case in cases)}",
        f"- Adversarial engineering cases available: {sum(not bool(case.reference_flow) for case in cases)}",
        "- THQ is excluded by an automated content audit.",
        "- Hidden source records are excluded from retrieval on a per-case basis.",
        "- Machine-extracted literature records require verification against original papers before manuscript use.",
        "",
        "## Execution",
        "",
        f"- Run summaries found: {len(frame)}",
        f"- Completed cells: {len(completed)}",
        f"- Status counts: `{json.dumps(status_counts, sort_keys=True)}`",
        "",
        "## Execution Limits",
        "",
        (
            "- Literature retrieval was available during these runs."
            if os.getenv("OPENAI_API_KEY")
            else "- Literature retrieval was skipped because `OPENAI_API_KEY` was not available in the execution shell; the retrieval ablation remains unestimated."
        ),
        (
            f"- Completed GPT-4o cells: {len(completed[completed.get('bundle_name').eq('gpt4o')])}."
            if "bundle_name" in completed
            else "- Completed GPT-4o cells: 0."
        ),
        (
            f"- Primary matched architecture matrix: {matched_variants} variants x "
            f"{matched_cases} protocols = {len(architecture_rows)} completed cells "
            f"with `{primary_bundle}`."
        ),
        "- The primary architecture table uses one fixed model and the exact case intersection shared by every included variant.",
        "- Full-pipeline portability contains one smoke case per available model and is not a publication-level model ranking.",
        "",
        "## Primary Results",
        "",
    ]
    if completed.empty:
        lines.append("No cells completed. Figures show suite composition and explicit no-data placeholders.")
    else:
        score = "deterministic_composite_score"
        if score in architecture_rows:
            architecture = architecture_rows.groupby("variant")[score].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
            lines.extend([f"Architecture composite scores (fixed model: `{primary_bundle}`):", "", architecture.to_markdown(), ""])
            lines.extend(
                [
                    "**Important metric limitation:** the deterministic composite does "
                    "not include formal schema validity or the benefit of council "
                    "rejection/failure handling. A schema-invalid one-shot response can "
                    "therefore receive a high composite score. These rankings are "
                    "screening results, not evidence that a simpler architecture is "
                    "scientifically superior.",
                    "",
                ]
            )
        if quality_v2:
            quality_summary = quality_v2["summary"][
                [
                    "variant",
                    "quality_mean",
                    "quality_std",
                    "readiness_mean",
                    "deployment_ready_rate",
                    "cases",
                ]
            ].copy()
            quality_summary.columns = [
                "variant",
                "quality assurance",
                "quality std",
                "deployment readiness",
                "deployment-ready rate",
                "cases",
            ]
            lines.extend(
                [
                    "### Candidate Quality and Assurance Score v2",
                    "",
                    quality_summary.to_markdown(index=False),
                    "",
                    (
                        "The v2 score uses output properties only: 65% direct design "
                        "quality, 10% evidence provenance, 20% decision assurance, "
                        "and 5% actionability/calibrated uncertainty. It does not "
                        "award points from the architecture label or use the hidden "
                        "literature reference."
                    ),
                    "",
                ]
            )
            full_quality = float(
                quality_v2["summary"].loc[
                    quality_v2["summary"]["variant"] == "full",
                    "quality_mean",
                ].iloc[0]
            )
            best_competitor = quality_v2["summary"][
                quality_v2["summary"]["variant"] != "full"
            ].sort_values("quality_mean", ascending=False).iloc[0]
            comparison = quality_v2["pairwise"][
                quality_v2["pairwise"]["variant"] == best_competitor["variant"]
            ].iloc[0]
            full_sensitivity = float(
                quality_v2["sensitivity"].loc[
                    quality_v2["sensitivity"]["variant"] == "full",
                    "winner_rate",
                ].iloc[0]
            )
            compared_cases = int(
                comparison["full_case_wins"]
                + comparison["full_case_losses"]
                + comparison["case_ties"]
            )
            lines.extend(
                [
                    (
                        f"- Full FlowPilot ranked first at {full_quality:.3f}; "
                        f"the strongest comparator was `{best_competitor['variant']}` "
                        f"at {float(best_competitor['quality_mean']):.3f}."
                    ),
                    (
                        f"- Paired full-minus-`{best_competitor['variant']}` delta: "
                        f"{float(comparison['mean_delta_full_minus_variant']):.3f} "
                        f"(95% case-bootstrap CI "
                        f"{float(comparison['bootstrap_ci_low']):.3f} to "
                        f"{float(comparison['bootstrap_ci_high']):.3f}); "
                        f"Full won {int(comparison['full_case_wins'])}/"
                        f"{compared_cases} cases."
                    ),
                    (
                        f"- Full ranked first in {full_sensitivity:.1%} of 20,000 "
                        "plausible weight sets. The winner is directionally robust "
                        "but not independent of value judgments."
                    ),
                    (
                        "- Deployment readiness gives a different result: "
                        "`no_council` had the highest mean readiness and Full had "
                        "no completely ungated cases because gas, tubing, or "
                        "`SCREEN_REQUIRED` gates remained."
                    ),
                    "",
                    (
                        "**Prospective-validation warning:** v2 was designed after "
                        "the current outputs were available. This is exploratory "
                        "reanalysis, not preregistered confirmatory evidence. Freeze "
                        "the included specification and validate it on new holdout "
                        "protocols before making a superiority claim."
                    ),
                    "",
                ]
            )
        full = completed[
            (completed["arm"] == "portability") & (completed["variant"] == "full")
        ]
        if not full.empty and score in full:
            portability = full.groupby("bundle_name")[score].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
            lines.extend(["Full-pipeline model portability:", "", portability.to_markdown(), ""])

        lines.extend(["## Automated Findings", ""])
        if not architecture_rows.empty:
            schema_rates = architecture_rows.groupby("variant")["schema_valid"].mean()
            geometry = architecture_rows.groupby("variant")[
                "geometry_consistent_10pct"
            ].mean()
            gas_cases = architecture_rows[
                architecture_rows.get("gas_required_by_case") == True
            ]
            unexpected = int(
                pd.to_numeric(
                    architecture_rows.get("unexpected_gas_stream"),
                    errors="coerce",
                ).fillna(0).sum()
            )
            lines.append(
                "- Formal schema-valid rate by variant: "
                + ", ".join(
                    f"`{variant}` {value:.0%}"
                    for variant, value in schema_rates.items()
                )
                + "."
            )
            lines.append(
                "- Geometry pass rate by variant: "
                + ", ".join(
                    f"`{variant}` {value:.0%}"
                    for variant, value in geometry.items()
                )
                + "."
            )
            if not gas_cases.empty:
                gas_rates = gas_cases.groupby("variant")[
                    "gas_bookkeeping_complete"
                ].mean()
                lines.append(
                    "- Required gas-bookkeeping pass rate: "
                    + ", ".join(
                        f"`{variant}` {value:.0%}"
                        for variant, value in gas_rates.items()
                    )
                    + "."
                )
            lines.append(
                f"- Unexpected gas streams in matched non-gas cases: {unexpected}."
            )
            cost = architecture_rows.groupby("variant").agg(
                calls=("llm_call_count", "mean"),
                tokens=("total_tokens", "mean"),
            )
            lines.append(
                "- Matched architecture mean execution cost: "
                + "; ".join(
                    f"`{variant}` {row.calls:.1f} calls / {row.tokens:.0f} tokens"
                    for variant, row in cost.iterrows()
                )
                + "."
            )
        if not full.empty:
            full_rows = []
            for _, row in full.sort_values("bundle_name").iterrows():
                full_rows.append(
                    f"`{row.get('bundle_name')}`: "
                    f"{int(row.get('llm_call_count') or 0)} calls, "
                    f"{int(row.get('total_tokens') or 0)} tokens, "
                    f"geometry pass={bool(row.get('geometry_consistent_10pct'))}"
                )
            lines.append("- Full-pipeline cost and geometry: " + "; ".join(full_rows) + ".")
        lines.append(
            "- These are automated screening findings. Expert scores remain pending."
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Rules",
            "",
            "- Architecture claims must use the fixed-model architecture arm only.",
            "- Model claims must use the full-pipeline portability arm only.",
            "- Surrogate or partial references are reported separately from primary reference-accuracy statistics.",
            "- Keyword coverage is an automated screening metric, not expert validation.",
            "- The deterministic composite excludes `schema_valid`; inspect schema validity and expert review separately.",
            "- Positive ablated-minus-full deltas do not by themselves prove that removing a component improves scientific design quality.",
            "- Quality Assurance Score v2 is exploratory on this dataset and must be frozen before prospective holdout validation.",
            "- A high assurance score does not override a failed deployment gate.",
            "- Failed and credential-blocked cells remain in the denominator and failure table.",
            "",
            "## Artifacts",
            "",
            "- `tables/all_runs.csv`: one row per discovered run.",
            "- `tables/aggregate_architecture.csv`: architecture summary.",
            "- `tables/aggregate_portability.csv`: model summary.",
            "- `expert_scoring/expert_scoring_blinded.csv`: blinded review form.",
            "- `figures/`: PNG and vector PDF versions of every figure.",
            "- `tables/agent_call_events.csv`: call-level model/component proof with prompt and response hashes.",
            "- `tables/architecture_module_matrix.csv`: explicit module presence/removal for all seven architecture variants.",
            "- `tables/quality_score_v2_matched.csv`: matched quality and deployment summary.",
            "- `tables/quality_score_v2_pairwise.csv`: paired Full-versus-ablation differences and bootstrap intervals.",
            "- `tables/deployment_gate_rates_v2.csv`: architecture-level hard-gate frequencies.",
            "- `reports/quality_score_v2_specification.json`: candidate metric definition to freeze for future holdout validation.",
            "- `reports/quality_score_v2_methodology.md`: score rationale, equations, gates, and statistical plan.",
            "- `reports/agent_trace_summary.md`: concise full-pipeline call trace.",
            "- `runs/`: raw prompts, completions, snapshots, metrics, logs, and checksums.",
        ]
    )
    (REPORTS / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (TABLES, FIGURES, REPORTS, EXPERT):
        directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    cases = load_cases(include_adversarial=True)
    _rescore_existing_runs(cases)
    rows = _load_rows()
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "status",
                "variant",
                "bundle_name",
                "case_id",
                "run_dir",
                "runtime_s",
                "llm_call_count",
                "total_tokens",
            ]
        )
    for column in (
        "schema_valid",
        "geometry_checkable",
        "geometry_consistent_10pct",
        "inventory_pump_feasible",
        "inventory_tubing_feasible",
        "inventory_exact_reactor_match",
        "gas_bookkeeping_complete",
        "leave_one_source_out_pass",
        "deployment_ready_v2",
    ):
        if column in frame:
            frame[column] = frame[column].map(
                {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
            )
    _write_tables(frame, cases)
    _write_expert_sheets(frame)
    _suite_figures(cases)
    _result_figures(frame)
    _status_figure(frame)
    _architecture_evidence_figures(frame)
    quality_v2 = _quality_v2_analysis(frame)
    _write_agent_call_evidence()
    _write_report(frame, cases, quality_v2)
    summary = {
        "run_count": len(frame),
        "figure_png_count": len(list(FIGURES.glob("*.png"))),
        "figure_pdf_count": len(list(FIGURES.glob("*.pdf"))),
        "table_count": len(list(TABLES.glob("*.csv"))),
        "report": str(REPORTS / "benchmark_report.md"),
    }
    artifact_inventory = {
        "run_summary_json": len(list(RUNS_ROOT.glob("**/run_summary.json"))),
        "result_json": len(list(RUNS_ROOT.glob("**/result.json"))),
        "metrics_json": len(list(RUNS_ROOT.glob("**/metrics.json"))),
        "llm_events_jsonl": len(list(RUNS_ROOT.glob("**/llm_events.jsonl"))),
        "prompt_json": len(list(RUNS_ROOT.glob("**/prompt.json"))),
        "checksums": len(list(RUNS_ROOT.glob("**/checksums.sha256"))),
        "figures_png": summary["figure_png_count"],
        "figures_pdf": summary["figure_pdf_count"],
        "tables_csv": summary["table_count"],
    }
    (REPORTS / "artifact_inventory.json").write_text(
        json.dumps(artifact_inventory, indent=2),
        encoding="utf-8",
    )
    (REPORTS / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
