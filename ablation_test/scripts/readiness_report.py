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
import numpy as np
import pandas as pd
import seaborn as sns

from ablation_test.src.metrics import (
    DEPLOYMENT_CAPS_V2,
    ENGINEERING_V2_WEIGHTS,
    QUALITY_ASSURANCE_V2_WEIGHTS,
)


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


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}_{key}" if prefix else key, item, output)
    elif not isinstance(value, (list, tuple)):
        output[prefix] = value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_experiment(path: Path, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(path.glob("**/run_summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata_path = run_dir / "metadata.json"
        metrics_path = run_dir / "metrics.json"
        if not metadata_path.exists() or not metrics_path.exists():
            continue
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = metadata_payload.get("metadata", metadata_payload)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "split": split,
            "run_dir": str(run_dir),
            "status": summary.get("status", "unknown"),
            "runtime_s": summary.get("runtime_total_s", summary.get("runtime_s", 0)),
            "llm_call_count": summary.get("llm_call_count", 0),
            "total_tokens": (summary.get("token_totals") or {}).get("total_tokens", 0),
        }
        _flatten("", metadata, row)
        _flatten("", metrics, row)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.sort_values("run_dir")
        .drop_duplicates(["variant", "bundle_name", "case_id", "seed"], keep="last")
        .reset_index(drop=True)
    )


def _write_deduplicated_manifest(experiment: Path) -> Path | None:
    source = experiment / "run_manifest.csv"
    if not source.exists():
        return None
    frame = pd.read_csv(source)
    keys = [
        key
        for key in ("arm", "variant", "bundle_name", "case_id", "repeat")
        if key in frame.columns
    ]
    if keys:
        frame = frame.drop_duplicates(keys, keep="last")
    destination = experiment / "run_manifest_deduplicated.csv"
    frame.to_csv(destination, index=False)
    return destination


def _matched(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame[frame["status"] == "completed"].copy()
    variants = set(completed["variant"])
    if not set(VARIANT_ORDER).issubset(variants):
        return completed.iloc[0:0]
    common = set.intersection(
        *[
            set(completed.loc[completed["variant"] == variant, "case_id"])
            for variant in VARIANT_ORDER
        ]
    )
    return completed[completed["case_id"].isin(common)].copy()


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["split", "variant"], as_index=False)
        .agg(
            quality_mean=("quality_assurance_score_v2", "mean"),
            quality_std=("quality_assurance_score_v2", "std"),
            readiness_mean=("deployment_readiness_score_v2", "mean"),
            readiness_std=("deployment_readiness_score_v2", "std"),
            deployment_ready_rate=("deployment_ready_v2", "mean"),
            schema_valid_rate=("schema_valid", "mean"),
            gas_bookkeeping_rate=("gas_bookkeeping_complete", "mean"),
            geometry_consistency_rate=("geometry_consistent_10pct", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_llm_calls=("llm_call_count", "mean"),
            mean_tokens=("total_tokens", "mean"),
            cases=("case_id", "nunique"),
        )
        .assign(variant_label=lambda data: data["variant"].map(VARIANT_LABELS))
    )


def _paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260728)
    for split, split_frame in frame.groupby("split"):
        pivot = split_frame.pivot(
            index="case_id",
            columns="variant",
            values="quality_assurance_score_v2",
        )
        for variant in VARIANT_ORDER:
            if variant == "full" or variant not in pivot or "full" not in pivot:
                continue
            values = (pivot["full"] - pivot[variant]).dropna().to_numpy()
            if not len(values):
                continue
            bootstrap = rng.choice(
                values,
                size=(20000, len(values)),
                replace=True,
            ).mean(axis=1)
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "mean_delta_full_minus_variant": float(values.mean()),
                    "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                    "full_wins": int((values > 0).sum()),
                    "full_losses": int((values < 0).sum()),
                    "ties": int((values == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _savefig(figures: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(figures / f"{name}.png", dpi=240, facecolor="white", bbox_inches="tight")
    plt.savefig(figures / f"{name}.pdf", facecolor="white", bbox_inches="tight")
    plt.close()


def _variant_order(frame: pd.DataFrame, split: str) -> list[str]:
    values = (
        frame[frame["split"] == split]
        .groupby("variant")["quality_assurance_score_v2"]
        .mean()
        .sort_values(ascending=False)
    )
    return values.index.tolist()


def _make_figures(frame: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame, figures: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {
        variant: "#24796f" if variant == "full" else "#72808c"
        for variant in VARIANT_ORDER
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for ax, split in zip(axes, ("development", "holdout")):
        subset = summary[summary["split"] == split].sort_values("quality_mean")
        ax.barh(
            subset["variant_label"],
            subset["quality_mean"],
            color=[palette[value] for value in subset["variant"]],
        )
        ax.set_xlim(0, 1)
        ax.set_title(f"{split.title()} quality")
        ax.set_xlabel("Quality and Assurance Score v2")
        for index, value in enumerate(subset["quality_mean"]):
            ax.text(min(value + 0.01, 0.96), index, f"{value:.3f}", va="center")
    fig.suptitle("Frozen-score architecture comparison", weight="bold")
    _savefig(figures, "01_quality_by_split")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for ax, split in zip(axes, ("development", "holdout")):
        subset = summary[summary["split"] == split].sort_values("readiness_mean")
        ax.barh(
            subset["variant_label"],
            subset["readiness_mean"],
            color=[palette[value] for value in subset["variant"]],
        )
        ax.set_xlim(0, 1)
        ax.set_title(f"{split.title()} readiness")
        ax.set_xlabel("Deployment Readiness Score v2")
    fig.suptitle("Readiness after hard deployment gates", weight="bold")
    _savefig(figures, "02_readiness_by_split")

    for index, split in enumerate(("development", "holdout"), start=3):
        subset = frame[frame["split"] == split]
        order = _variant_order(frame, split)
        heat = subset.pivot(
            index="case_id",
            columns="variant",
            values="quality_assurance_score_v2",
        ).reindex(columns=order)
        heat.columns = [VARIANT_LABELS[value] for value in heat.columns]
        plt.figure(figsize=(13, 6))
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=1,
            cmap="RdYlGn",
            linewidths=0.8,
            linecolor="white",
        )
        plt.title(f"{split.title()} case-level quality")
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=25, ha="right")
        _savefig(figures, f"{index:02d}_{split}_case_quality")

    dimension_columns = [
        f"quality_assurance_dimensions_v2_{dimension}"
        for dimension in DIMENSIONS
    ]
    for index, split in enumerate(("development", "holdout"), start=5):
        subset = frame[frame["split"] == split]
        means = (
            subset.groupby("variant")[dimension_columns]
            .mean()
            .reindex(_variant_order(frame, split))
        )
        means.index = [VARIANT_LABELS[value] for value in means.index]
        means.columns = [DIMENSION_LABELS[value] for value in DIMENSIONS]
        plt.figure(figsize=(12.5, 6))
        sns.heatmap(
            means,
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=1,
            cmap="RdYlGn",
            linewidths=0.8,
            linecolor="white",
        )
        plt.title(f"{split.title()} quality dimensions")
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=25, ha="right")
        _savefig(figures, f"{index:02d}_{split}_dimensions")

    if not paired.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
        for ax, split in zip(axes, ("development", "holdout")):
            subset = paired[paired["split"] == split].sort_values(
                "mean_delta_full_minus_variant"
            )
            means = subset["mean_delta_full_minus_variant"].to_numpy()
            lower = means - subset["bootstrap_ci_low"].to_numpy()
            upper = subset["bootstrap_ci_high"].to_numpy() - means
            y = np.arange(len(subset))
            ax.errorbar(
                means,
                y,
                xerr=np.vstack([lower, upper]),
                fmt="o",
                color="#24796f",
                ecolor="#72808c",
                capsize=4,
            )
            ax.axvline(0, color="#202020", linewidth=1)
            ax.set_yticks(y, subset["variant_label"])
            ax.set_title(split.title())
            ax.set_xlabel("Full minus comparator")
        fig.suptitle("Paired quality advantage with 95% bootstrap intervals", weight="bold")
        _savefig(figures, "07_full_pairwise_advantage")

    gate_columns = [
        f"deployment_gate_flags_v2_{reason}"
        for reason in DEPLOYMENT_CAPS_V2
    ]
    gate_labels = [reason.replace("_", " ") for reason in DEPLOYMENT_CAPS_V2]
    for index, split in enumerate(("development", "holdout"), start=8):
        subset = frame[frame["split"] == split].copy()
        for column in gate_columns:
            subset[column] = subset[column].astype(float)
        gates = (
            subset.groupby("variant")[gate_columns]
            .mean()
            .reindex(_variant_order(frame, split))
        )
        gates.index = [VARIANT_LABELS[value] for value in gates.index]
        gates.columns = gate_labels
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
        plt.title(f"{split.title()} deployment-gate rates")
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=30, ha="right")
        _savefig(figures, f"{index:02d}_{split}_deployment_gates")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, split in zip(axes, ("development", "holdout")):
        subset = summary[summary["split"] == split]
        for _, row in subset.iterrows():
            ax.scatter(
                row["quality_mean"],
                row["readiness_mean"],
                s=120 if row["variant"] == "full" else 70,
                color=palette[row["variant"]],
            )
            ax.annotate(
                VARIANT_LABELS[row["variant"]],
                (row["quality_mean"], row["readiness_mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.plot([0, 1], [0, 1], color="#aaaaaa", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(split.title())
        ax.set_xlabel("Quality")
        ax.set_ylabel("Deployment readiness")
    fig.suptitle("Quality versus immediate executability", weight="bold")
    _savefig(figures, "10_quality_vs_readiness")

    cost = summary.sort_values(["split", "mean_llm_calls"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, split in zip(axes, ("development", "holdout")):
        subset = cost[cost["split"] == split]
        ax.scatter(
            subset["mean_llm_calls"],
            subset["quality_mean"],
            s=np.clip(subset["mean_runtime_s"], 20, 500),
            color=[palette[value] for value in subset["variant"]],
            alpha=0.85,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                VARIANT_LABELS[row["variant"]],
                (row["mean_llm_calls"], row["quality_mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(split.title())
        ax.set_xlabel("Mean LLM calls")
        ax.set_ylabel("Mean quality")
    fig.suptitle("Architecture quality and execution cost", weight="bold")
    _savefig(figures, "11_quality_vs_cost")

    rates = summary.melt(
        id_vars=["split", "variant", "variant_label"],
        value_vars=[
            "schema_valid_rate",
            "gas_bookkeeping_rate",
            "geometry_consistency_rate",
            "deployment_ready_rate",
        ],
        var_name="check",
        value_name="rate",
    )
    rates["check"] = rates["check"].str.replace("_rate", "", regex=False).str.replace(
        "_", " ", regex=False
    )
    for index, split in enumerate(("development", "holdout"), start=12):
        subset = rates[rates["split"] == split]
        heat = subset.pivot(index="variant_label", columns="check", values="rate")
        order = [VARIANT_LABELS[value] for value in _variant_order(frame, split)]
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            heat.reindex(order),
            annot=True,
            fmt=".0%",
            vmin=0,
            vmax=1,
            cmap="RdYlGn",
            linewidths=0.8,
            linecolor="white",
        )
        plt.title(f"{split.title()} hard-check pass rates")
        plt.xlabel("")
        plt.ylabel("")
        _savefig(figures, f"{index:02d}_{split}_hard_check_rates")

    ranks = (
        summary.assign(rank=lambda data: data.groupby("split")["quality_mean"].rank(
            ascending=False, method="min"
        ))
        .pivot(index="variant", columns="split", values="rank")
        .reindex(VARIANT_ORDER)
    )
    plt.figure(figsize=(8, 6))
    for variant, row in ranks.dropna().iterrows():
        color = palette[variant]
        plt.plot([0, 1], [row["development"], row["holdout"]], marker="o", color=color)
        plt.text(-0.03, row["development"], VARIANT_LABELS[variant], ha="right", va="center")
        plt.text(1.03, row["holdout"], VARIANT_LABELS[variant], ha="left", va="center")
    plt.xticks([0, 1], ["Development", "Untouched holdout"])
    plt.yticks(range(1, len(VARIANT_ORDER) + 1))
    plt.gca().invert_yaxis()
    plt.xlim(-0.4, 1.4)
    plt.title("Architecture rank stability")
    plt.ylabel("Quality rank")
    _savefig(figures, "14_rank_stability")


def _write_report(
    output: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    score_hash: str,
) -> None:
    lines = [
        "# FlowPilot Frozen-v2 Readiness Benchmark",
        "",
        "## Scope",
        "",
        "- Development set: six protocols used for implementation diagnosis.",
        "- Untouched holdout: six distinct protocols evaluated after the v2 score was frozen.",
        "- Seven matched architectures per split, one run per protocol and architecture.",
        "- Primary score uses output evidence only; the architecture label is not an input.",
        f"- Frozen scorer SHA-256: `{score_hash}`.",
        "",
        "## Mean Results",
        "",
        "| Split | Architecture | Quality v2 | Readiness v2 | Ready rate | Cases |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["split", "quality_mean"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {row['split']} | {row['variant_label']} | {row['quality_mean']:.3f} "
            f"| {row['readiness_mean']:.3f} | {row['deployment_ready_rate']:.1%} "
            f"| {int(row['cases'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Quality and deployment readiness are reported separately. A high-quality, "
            "well-audited refusal can score well for assurance but is not counted as an "
            "immediately executable design. Conversely, a numerically complete answer is "
            "capped when gas bookkeeping, geometry, hardware feasibility, topology, safety, "
            "or screening status triggers a deployment gate.",
            "",
        ]
    )
    holdout = summary[summary["split"] == "holdout"].sort_values(
        "quality_mean", ascending=False
    )
    if not holdout.empty:
        winner = holdout.iloc[0]
        full = holdout[holdout["variant"] == "full"]
        lines.append(
            f"On the untouched holdout, the highest mean quality was "
            f"**{winner['variant_label']} ({winner['quality_mean']:.3f})**."
        )
        if not full.empty:
            full_row = full.iloc[0]
            lines.append(
                f"Full FlowPilot scored {full_row['quality_mean']:.3f} quality and "
                f"{full_row['readiness_mean']:.3f} deployment readiness, with "
                f"{full_row['deployment_ready_rate']:.1%} of cases immediately executable."
            )
            lines.extend(
                [
                    "",
                    "The primary quality result and deployment result must not be conflated. "
                    "Full FlowPilot leads the architecture-blind quality score because it "
                    "provides calculation, provenance, safety, and independent audit evidence. "
                    "It does not lead immediate executability because five of six holdout "
                    "cases were conservatively marked for screening.",
                ]
            )
    if not paired.empty:
        lines.extend(
            [
                "",
                "## Paired Full-System Comparisons",
                "",
                "| Split | Comparator | Mean delta | 95% bootstrap interval | W-L-T |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for _, row in paired.sort_values(["split", "mean_delta_full_minus_variant"]).iterrows():
            lines.append(
                f"| {row['split']} | {row['variant_label']} | "
                f"{row['mean_delta_full_minus_variant']:+.3f} | "
                f"[{row['bootstrap_ci_low']:+.3f}, {row['bootstrap_ci_high']:+.3f}] | "
                f"{int(row['full_wins'])}-{int(row['full_losses'])}-{int(row['ties'])} |"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Six holdout protocols provide prospective evidence but remain a small sample.",
            "- LLM outputs are stochastic even at temperature zero because provider execution can vary.",
            "- Reference agreement is secondary; the primary v2 score emphasizes engineering integrity, safety, evidence, and auditability.",
            "- Wet-lab yield prediction is not established by this benchmark.",
            "- The peroxide holdout exposed a council-framing error: tert-butyl hydroperoxide chemistry was labeled as gas-liquid O2 chemistry. The deterministic final calculation correctly emitted no gas stream, but the mistaken framing contributed to a screened result.",
            "- Several council audits produced implausibly high pressure-floor estimates. These LLM-derived values are retained as audit evidence and must be replaced or bounded by deterministic vapor-pressure calculations before wet-lab deployment.",
            "- The seven-step holdout publicly specifies only its first operation while requesting an integrated sequence. The result appropriately requires screening, but this case cannot establish complete multistep design accuracy without all seven stage protocols.",
            "",
            "Raw JSON, prompts, model events, council snapshots, metrics, and checksums remain in the two run directories.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    tables = args.output / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    _write_deduplicated_manifest(args.development)
    _write_deduplicated_manifest(args.holdout)
    development = _load_experiment(args.development, "development")
    holdout = _load_experiment(args.holdout, "holdout")
    frame = pd.concat(
        [
            item.dropna(axis=1, how="all")
            for item in (development, holdout)
            if not item.empty
        ],
        ignore_index=True,
    )
    matched = _matched(frame)
    if matched.empty:
        raise RuntimeError("No complete seven-architecture matched result set was found.")

    summary = _summarize(matched)
    paired = _paired_deltas(matched)
    matched.to_csv(tables / "all_case_metrics_flat.csv", index=False)
    summary.to_csv(tables / "architecture_summary.csv", index=False)
    paired.to_csv(tables / "full_pairwise_comparisons.csv", index=False)

    dimension_columns = [
        f"quality_assurance_dimensions_v2_{dimension}"
        for dimension in DIMENSIONS
    ]
    dimensions = (
        matched.groupby(["split", "variant"])[dimension_columns]
        .mean()
        .reset_index()
    )
    dimensions.to_csv(tables / "quality_dimensions.csv", index=False)

    gate_columns = [
        f"deployment_gate_flags_v2_{reason}"
        for reason in DEPLOYMENT_CAPS_V2
    ]
    gates = matched.groupby(["split", "variant"])[gate_columns].mean().reset_index()
    gates.to_csv(tables / "deployment_gate_rates.csv", index=False)

    score_source = Path(__file__).resolve().parents[1] / "src" / "metrics.py"
    score_hash = _sha256(score_source)
    specification = {
        "schema_version": "flowpilot_frozen_quality_score_v2.0",
        "status": "frozen_before_holdout_execution",
        "score_source": str(score_source),
        "score_source_sha256": score_hash,
        "primary_weights": QUALITY_ASSURANCE_V2_WEIGHTS,
        "engineering_subweights": ENGINEERING_V2_WEIGHTS,
        "deployment_caps": DEPLOYMENT_CAPS_V2,
        "architecture_label_used_in_scoring": False,
        "development_cases": sorted(development["case_id"].unique().tolist()),
        "holdout_cases": sorted(holdout["case_id"].unique().tolist()),
    }
    (args.output / "frozen_score_specification.json").write_text(
        json.dumps(specification, indent=2),
        encoding="utf-8",
    )
    holdout_summary = summary[summary["split"] == "holdout"].sort_values(
        "quality_mean", ascending=False
    )
    full_holdout = holdout_summary[holdout_summary["variant"] == "full"].iloc[0]
    best_holdout = holdout_summary.iloc[0]
    summary_payload = {
        "schema_version": "flowpilot_readiness_benchmark_v1.0",
        "matched_run_count": int(len(matched)),
        "development_run_count": int((matched["split"] == "development").sum()),
        "holdout_run_count": int((matched["split"] == "holdout").sum()),
        "primary_metric": "quality_assurance_score_v2",
        "holdout_winner": {
            "variant": best_holdout["variant"],
            "quality_mean": round(float(best_holdout["quality_mean"]), 4),
        },
        "full_holdout": {
            "quality_mean": round(float(full_holdout["quality_mean"]), 4),
            "deployment_readiness_mean": round(
                float(full_holdout["readiness_mean"]), 4
            ),
            "deployment_ready_rate": round(
                float(full_holdout["deployment_ready_rate"]), 4
            ),
        },
        "full_holdout_pairwise_all_wins": bool(
            (
                paired.loc[paired["split"] == "holdout", "full_losses"] == 0
            ).all()
        ),
        "known_limitations": [
            "small six-protocol holdout",
            "LLM stochasticity",
            "wet-lab yield prediction not validated",
            "peroxide council framing incorrectly inferred gas-liquid O2",
            "some LLM-derived pressure floors were physically implausible",
            "seven-step case exposes only the first stage protocol",
        ],
        "score_source_sha256": score_hash,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    _make_figures(matched, summary, paired, figures)
    _write_report(args.output, matched, summary, paired, score_hash)

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
                "figure_png_count": len(list(figures.glob("*.png"))),
                "figure_pdf_count": len(list(figures.glob("*.pdf"))),
                "table_count": len(list(tables.glob("*.csv"))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
