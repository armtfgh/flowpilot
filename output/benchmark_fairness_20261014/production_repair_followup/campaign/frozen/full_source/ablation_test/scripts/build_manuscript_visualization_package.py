#!/usr/bin/env python3
"""Build a consolidated manuscript figure package from completed NewGen 2.0 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE = ROOT / "deliverables/newgen_2_0_latest_results_20260824"
MODULE_SOURCE = (
    ROOT
    / "deliverables/newgen_2_0_module_attribution_qwen38_screen3_matched_v1_20260821"
)
DEFAULT_OUTPUT = ROOT / "deliverables/manuscript_benchmark_visualizations_20260825"

MODEL_ORDER = [
    "Qwen3.6-27B",
    "Qwen3.8-27B",
    "GPT-4o",
    "Claude Sonnet 4.6",
    "Claude Opus 4.6",
]
CASE_ORDER = ["Hydrogenolysis", "Photochemical oxidation", "CuAAC"]
ARCH_ORDER = ["One-shot", "FlowPilot"]
ARCH_COLORS = {"One-shot": "#C54F45", "FlowPilot": "#147D73"}
CASE_COLORS = {
    "Hydrogenolysis": "#3F6EA7",
    "Photochemical oxidation": "#D69A35",
    "CuAAC": "#7B5AA6",
}
CRITERION_NAMES = {
    "UO-01": "Transformation fidelity",
    "UO-02": "Required materials",
    "UO-03": "Stoichiometry/feed chemistry",
    "UO-04": "Condition/stage mapping",
    "UO-05": "Executable topology",
    "UO-06": "Liquid material balance",
    "UO-07": "Residence-time closure",
    "UO-08": "Gas bookkeeping",
    "UO-09": "Multistage closure",
    "UO-10": "Inventory feasibility",
    "UO-11": "Transport plausibility",
    "UO-12": "Hazards and controls",
    "UO-13": "Procedure and work-up",
    "UO-14": "Evidence calibration",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def read_csv(source: Path, name: str) -> pd.DataFrame:
    path = source / "tables" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".png": {"dpi": 400},
        ".pdf": {},
        ".svg": {},
    }.items():
        fig.savefig(
            figure_dir / f"{stem}{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def save_source(frame: pd.DataFrame, output: Path, stem: str) -> None:
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(source_dir / f"{stem}.csv", index=False)


def model_tables() -> dict[str, pd.DataFrame]:
    scores = read_csv(MODEL_SOURCE, "all_candidate_consensus_scores.csv")
    criteria = read_csv(MODEL_SOURCE, "all_criterion_judgments.csv")
    architecture = read_csv(MODEL_SOURCE, "model_architecture_mean_sd.csv")
    paired = read_csv(MODEL_SOURCE, "paired_delta_mean_sd.csv")
    case_summary = read_csv(MODEL_SOURCE, "case_architecture_mean_sd.csv")

    scores = scores[scores["model"].isin(MODEL_ORDER)].copy()
    criteria = criteria[criteria["model"].isin(MODEL_ORDER)].copy()
    architecture = architecture[architecture["model"].isin(MODEL_ORDER)].copy()
    paired = paired[paired["model"].isin(MODEL_ORDER)].copy()
    case_summary = case_summary[case_summary["model"].isin(MODEL_ORDER)].copy()

    expected = len(MODEL_ORDER) * len(ARCH_ORDER) * len(CASE_ORDER) * 3
    keys = ["model", "architecture", "case", "repeat_id"]
    if len(scores) != expected or scores[keys].duplicated().any():
        raise RuntimeError(
            f"Model benchmark is incomplete: expected {expected} unique outcomes, got {len(scores)}"
        )
    if set(scores["repeat_id"]) != {"repeat_01", "repeat_02", "repeat_03"}:
        raise RuntimeError("Model benchmark does not contain exactly three repeats")
    return {
        "scores": scores,
        "criteria": criteria,
        "architecture": architecture,
        "paired": paired,
        "case_summary": case_summary,
    }


def module_tables() -> dict[str, pd.DataFrame]:
    outcomes = read_csv(MODULE_SOURCE, "module_outcomes.csv")
    summary = read_csv(MODULE_SOURCE, "module_summary.csv")
    effects = read_csv(MODULE_SOURCE, "module_effects.csv")
    criteria = read_csv(MODULE_SOURCE, "criterion_judgments.csv")
    if len(outcomes) != 45 or outcomes[["case", "condition"]].duplicated().any():
        raise RuntimeError("Module benchmark is not the expected 45-cell matched screen")
    return {
        "outcomes": outcomes,
        "summary": summary,
        "effects": effects,
        "criteria": criteria,
    }


def plot_model_architecture(data: dict[str, pd.DataFrame], output: Path) -> None:
    frame = data["architecture"].copy()
    save_source(frame, output, "fig01_model_architecture_mean_sd")
    fig, ax = plt.subplots(figsize=(9.2, 4.7), layout="constrained")
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    for offset, arch in [(-width / 2, "One-shot"), (width / 2, "FlowPilot")]:
        rows = frame[frame["architecture"] == arch].set_index("model").loc[MODEL_ORDER]
        bars = ax.bar(
            x + offset,
            rows["mean"],
            width,
            yerr=rows["sample_sd"],
            capsize=3,
            color=ARCH_COLORS[arch],
            edgecolor="white",
            linewidth=0.6,
            label=arch,
        )
        for bar, value, uncertainty in zip(bars, rows["mean"], rows["sample_sd"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + uncertainty + 0.006,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax.set_ylim(0.55, 1.01)
    ax.set_ylabel("NewGen 2.0 consensus score")
    ax.set_xticks(x, MODEL_ORDER, rotation=18, ha="right")
    ax.set_title("FlowPilot improves matched outcome quality across successful model campaigns")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, output, "fig01_model_architecture_mean_sd")


def plot_model_paired_effect(data: dict[str, pd.DataFrame], output: Path) -> None:
    frame = data["paired"].set_index("model").loc[MODEL_ORDER].reset_index()
    save_source(frame, output, "fig02_paired_architecture_effect")
    fig, ax = plt.subplots(figsize=(8.2, 4.7), layout="constrained")
    y = np.arange(len(frame))
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.errorbar(
        frame["mean_delta"],
        y,
        xerr=frame["sample_sd"],
        fmt="o",
        markersize=7,
        capsize=4,
        color=ARCH_COLORS["FlowPilot"],
        ecolor="#555555",
    )
    for x, yy in zip(frame["mean_delta"], y):
        ax.text(x + 0.006, yy, f"{x:+.3f}", va="center", fontsize=8)
    ax.set_yticks(y, frame["model"])
    ax.invert_yaxis()
    ax.set_xlim(-0.04, 0.27)
    ax.set_xlabel("Paired effect: FlowPilot - one-shot")
    ax.set_title("Matched architecture effect by generator model")
    ax.grid(axis="x", alpha=0.18)
    ax.text(
        0.99,
        0.02,
        "Mean +/- sample SD across three paired repeats",
        transform=ax.transAxes,
        ha="right",
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output, "fig02_paired_architecture_effect")


def plot_flowpilot_case_scores(data: dict[str, pd.DataFrame], output: Path) -> None:
    frame = data["case_summary"]
    frame = frame[frame["architecture"] == "FlowPilot"].copy()
    save_source(frame, output, "fig03_flowpilot_case_scores")
    fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    x = np.arange(len(MODEL_ORDER))
    width = 0.23
    for index, case in enumerate(CASE_ORDER):
        rows = frame[frame["case"] == case].set_index("model").loc[MODEL_ORDER]
        ax.errorbar(
            x + (index - 1) * width,
            rows["mean"],
            yerr=rows["sample_sd"],
            fmt="o",
            markersize=6.5,
            capsize=3,
            color=CASE_COLORS[case],
            label=case,
        )
    ax.set_ylim(0.72, 1.015)
    ax.set_ylabel("FlowPilot consensus score")
    ax.set_xticks(x, MODEL_ORDER, rotation=18, ha="right")
    ax.set_title("FlowPilot performance by chemistry case")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.grid(axis="y", alpha=0.18)
    ax.text(
        0.01,
        0.02,
        "Mean +/- sample SD across three generation repeats",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output, "fig03_flowpilot_case_scores")


def plot_case_paired_effects(data: dict[str, pd.DataFrame], output: Path) -> None:
    scores = data["scores"]
    wide = scores.pivot(
        index=["model", "case", "repeat_id"],
        columns="architecture",
        values="mean_score_0_1",
    ).reset_index()
    wide["paired_delta"] = wide["FlowPilot"] - wide["One-shot"]
    summary = (
        wide.groupby(["model", "case"])["paired_delta"]
        .agg(mean="mean", sample_sd="std", n="count")
        .reset_index()
    )
    save_source(summary, output, "fig04_case_paired_effects")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), sharex=True, sharey=True, layout="constrained")
    for ax, case in zip(axes, CASE_ORDER):
        rows = summary[summary["case"] == case].set_index("model").loc[MODEL_ORDER]
        y = np.arange(len(rows))
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.errorbar(
            rows["mean"],
            y,
            xerr=rows["sample_sd"],
            fmt="o",
            markersize=5.5,
            capsize=3,
            color=CASE_COLORS[case],
        )
        ax.set_title(case)
        ax.set_yticks(y, MODEL_ORDER)
        ax.set_xlim(-0.15, 0.47)
        ax.grid(axis="x", alpha=0.18)
    axes[0].invert_yaxis()
    fig.supxlabel("Paired effect: FlowPilot - one-shot (mean +/- SD, n=3 repeats)")
    fig.suptitle("Architecture effect is chemistry- and model-dependent", fontsize=11)
    save_figure(fig, output, "fig04_case_paired_effects")


def plot_repeat_stability(data: dict[str, pd.DataFrame], output: Path) -> None:
    scores = data["scores"]
    repeat = (
        scores.groupby(["model", "architecture", "repeat_id"], as_index=False)[
            "mean_score_0_1"
        ]
        .mean()
        .rename(columns={"mean_score_0_1": "campaign_mean"})
    )
    save_source(repeat, output, "fig05_repeat_stability")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True, sharey=True, layout="constrained")
    for ax, model in zip(axes.flat, MODEL_ORDER):
        rows = repeat[repeat["model"] == model]
        for arch in ARCH_ORDER:
            part = rows[rows["architecture"] == arch].sort_values("repeat_id")
            ax.plot(
                [1, 2, 3],
                part["campaign_mean"],
                marker="o",
                linewidth=1.6,
                color=ARCH_COLORS[arch],
                label=arch,
            )
        ax.set_title(model)
        ax.set_xticks([1, 2, 3])
        ax.set_ylim(0.62, 0.99)
        ax.grid(alpha=0.16)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.supxlabel("Generation repeat")
    fig.supylabel("Mean score across three cases")
    fig.suptitle("Repeat-level outcome stability", fontsize=11)
    save_figure(fig, output, "fig05_repeat_stability")


def plot_criteria(data: dict[str, pd.DataFrame], output: Path) -> None:
    criteria = data["criteria"]
    scores = data["scores"][["candidate_id", "repeat_id"]]
    applicable = criteria[criteria["applicability"] == "APPLICABLE"].copy()
    applicable["normalized_score"] = applicable["score_0_4"] / 4.0
    candidate = (
        applicable.groupby(
            ["candidate_id", "model", "architecture", "case", "criterion_id"],
            as_index=False,
        )["normalized_score"]
        .mean()
        .merge(scores, on="candidate_id", how="left", validate="many_to_one")
    )
    repeat_units = (
        candidate.groupby(
            ["model", "architecture", "repeat_id", "criterion_id"], as_index=False
        )["normalized_score"]
        .mean()
    )
    profile = (
        repeat_units.groupby(["architecture", "criterion_id"])["normalized_score"]
        .agg(mean="mean", sample_sd="std", n_model_repeats="count")
        .reset_index()
    )
    save_source(profile, output, "fig06_criterion_profile")
    ids = list(CRITERION_NAMES)
    fig, ax = plt.subplots(figsize=(10.8, 5.3), layout="constrained")
    x = np.arange(len(ids))
    for arch in ARCH_ORDER:
        rows = profile[profile["architecture"] == arch].set_index("criterion_id").reindex(ids)
        ax.errorbar(
            x,
            rows["mean"],
            yerr=rows["sample_sd"],
            marker="o",
            markersize=4.5,
            capsize=2.5,
            linewidth=1.5,
            color=ARCH_COLORS[arch],
            label=arch,
        )
    ax.set_xticks(x, ids, rotation=45, ha="right")
    ax.set_ylim(0.42, 1.03)
    ax.set_ylabel("Criterion score")
    ax.set_title("Universal criterion profile across successful model campaigns")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    ax.text(
        0.01,
        0.02,
        "Mean +/- sample SD across 18 model-repeat units (6 models x 3 repeats)",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    ax.text(
        ids.index("UO-09"),
        0.45,
        "N/A",
        ha="center",
        va="center",
        fontsize=7,
        color="#666666",
    )
    save_figure(fig, output, "fig06_criterion_profile")

    wide = repeat_units.pivot_table(
        index=["model", "repeat_id", "criterion_id"],
        columns="architecture",
        values="normalized_score",
    ).reset_index()
    wide["delta"] = wide["FlowPilot"] - wide["One-shot"]
    heat = wide.groupby(["model", "criterion_id"])["delta"].mean().unstack("criterion_id")
    heat = heat.reindex(index=MODEL_ORDER, columns=ids)
    heat_rows = heat.stack().rename("mean_paired_delta").reset_index()
    save_source(heat_rows, output, "fig07_criterion_gain_heatmap")
    fig, ax = plt.subplots(figsize=(11.2, 4.4), layout="constrained")
    image = ax.imshow(heat.values, cmap="RdYlGn", vmin=-0.18, vmax=0.40, aspect="auto")
    ax.set_xticks(np.arange(len(ids)), ids, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_title("Criterion-level architecture gain by model")
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label("FlowPilot - one-shot")
    save_figure(fig, output, "fig07_criterion_gain_heatmap")


def plot_errors_and_judge_dispersion(data: dict[str, pd.DataFrame], output: Path) -> None:
    scores = data["scores"].copy()
    errors = (
        scores.groupby(["model", "architecture", "repeat_id"], as_index=False)[
            "total_judge_critical_flags"
        ]
        .sum()
        .groupby(["model", "architecture"])["total_judge_critical_flags"]
        .agg(mean="mean", sample_sd="std", n_repeats="count")
        .reset_index()
    )
    save_source(errors, output, "fig08_critical_flags")
    fig, ax = plt.subplots(figsize=(9.2, 4.6), layout="constrained")
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    for offset, arch in [(-width / 2, "One-shot"), (width / 2, "FlowPilot")]:
        rows = errors[errors["architecture"] == arch].set_index("model").loc[MODEL_ORDER]
        ax.bar(
            x + offset,
            rows["mean"],
            width,
            yerr=rows["sample_sd"],
            capsize=3,
            color=ARCH_COLORS[arch],
            label=arch,
        )
    ax.set_xticks(x, MODEL_ORDER, rotation=18, ha="right")
    ax.set_ylabel("Critical flags per three-case repeat")
    ax.set_title("Critical-error burden across matched campaigns")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    ax.text(
        0.99,
        0.96,
        "Mean +/- sample SD across three repeats",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output, "fig08_critical_flags")

    dispersion = (
        scores.groupby(["model", "architecture", "repeat_id"], as_index=False)["judge_sd"]
        .mean()
        .groupby(["model", "architecture"])["judge_sd"]
        .agg(mean="mean", sample_sd="std", n_repeats="count")
        .reset_index()
    )
    save_source(dispersion, output, "fig09_judge_dispersion")
    fig, ax = plt.subplots(figsize=(8.6, 4.5), layout="constrained")
    for arch in ARCH_ORDER:
        rows = dispersion[dispersion["architecture"] == arch].set_index("model").loc[MODEL_ORDER]
        ax.errorbar(
            np.arange(len(MODEL_ORDER)),
            rows["mean"],
            yerr=rows["sample_sd"],
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=ARCH_COLORS[arch],
            label=arch,
        )
    ax.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER, rotation=18, ha="right")
    ax.set_ylabel("Mean between-judge SD per outcome")
    ax.set_title("Judge dispersion by model and architecture")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output, "fig09_judge_dispersion")


def plot_model_summary(data: dict[str, pd.DataFrame], output: Path) -> None:
    architecture = data["architecture"]
    paired = data["paired"].set_index("model").loc[MODEL_ORDER]
    scores = data["scores"]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), layout="constrained")

    x = np.arange(len(MODEL_ORDER))
    for arch in ARCH_ORDER:
        rows = architecture[architecture["architecture"] == arch].set_index("model").loc[MODEL_ORDER]
        axes[0].plot(x, rows["mean"], "o-", color=ARCH_COLORS[arch], label=arch)
        axes[0].fill_between(
            x,
            rows["mean"] - rows["sample_sd"],
            rows["mean"] + rows["sample_sd"],
            color=ARCH_COLORS[arch],
            alpha=0.12,
        )
    axes[0].set_ylim(0.62, 0.98)
    axes[0].set_ylabel("Consensus score")
    axes[0].set_title("A  Outcome quality")
    axes[0].legend(frameon=False)

    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].bar(x, paired["mean_delta"], color=ARCH_COLORS["FlowPilot"])
    axes[1].errorbar(x, paired["mean_delta"], yerr=paired["sample_sd"], fmt="none", color="#333333", capsize=3)
    axes[1].set_ylim(-0.04, 0.27)
    axes[1].set_ylabel("FlowPilot - one-shot")
    axes[1].set_title("B  Paired architecture effect")

    critical = scores.groupby(["model", "architecture"])["total_judge_critical_flags"].sum().unstack()
    critical = critical.loc[MODEL_ORDER]
    axes[2].bar(x, critical["One-shot"], color=ARCH_COLORS["One-shot"], label="One-shot")
    axes[2].bar(
        x,
        critical["FlowPilot"],
        bottom=critical["One-shot"],
        color=ARCH_COLORS["FlowPilot"],
        label="FlowPilot",
    )
    axes[2].set_ylabel("Total critical flags (all repeats)")
    axes[2].set_title("C  Critical-error burden")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.set_xticks(x, MODEL_ORDER, rotation=32, ha="right")
        ax.grid(axis="y", alpha=0.16)
    summary_source = architecture.merge(
        data["paired"][["model", "mean_delta", "sample_sd"]].rename(
            columns={"sample_sd": "paired_delta_sample_sd"}
        ),
        on="model",
    )
    save_source(summary_source, output, "fig10_model_benchmark_summary")
    save_figure(fig, output, "fig10_model_benchmark_summary")


def module_order(summary: pd.DataFrame) -> list[str]:
    preferred = [
        "One-shot",
        "No council",
        "Without Chemistry agent",
        "Without Kinetics agent",
        "Without Fluidics agent",
        "Without Safety agent",
        "Without Skeptic audit",
        "Without preselection refinement",
        "Without winner revision",
        "Deterministic selection",
        "Without DFMEA",
        "One-pass council",
        "Candidate budget 1",
        "Full FlowPilot",
        "Candidate budget 4",
    ]
    return [item for item in preferred if item in set(summary["condition"])]


def plot_module_overview(data: dict[str, pd.DataFrame], output: Path) -> None:
    summary = data["summary"].set_index("condition")
    order = [
        "One-shot",
        "No council",
        "Full FlowPilot",
        "Without Chemistry agent",
        "Without Kinetics agent",
        "Without Fluidics agent",
        "Without Safety agent",
    ]
    frame = (
        summary.loc[order]
        .sort_values("mean_score_0_1", ascending=True)
        .reset_index()
    )
    save_source(frame, output, "fig11_module_condition_scores")
    fig, ax = plt.subplots(figsize=(9.2, 6.4), layout="constrained")
    y = np.arange(len(frame))
    colors = [
        ARCH_COLORS["One-shot"]
        if condition in {"One-shot", "No council"}
        else ARCH_COLORS["FlowPilot"]
        if condition == "Full FlowPilot"
        else "#7B858D"
        for condition in frame["condition"]
    ]
    ax.barh(
        y,
        frame["mean_score_0_1"],
        xerr=frame["score_sd"],
        capsize=3,
        color=colors,
        alpha=0.96,
    )
    ax.set_yticks(y, frame["condition"])
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.01)
    ax.set_xlabel("NewGen 2.0 consensus score")
    ax.set_title("Qwen3.8 selected internal-module conditions")
    ax.grid(axis="x", alpha=0.18)
    ax.text(
        0.99,
        0.02,
        "Mean +/- between-case SD (three cases; one generation per condition)",
        transform=ax.transAxes,
        ha="right",
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output, "fig11_module_condition_scores")


def plot_module_effects(data: dict[str, pd.DataFrame], output: Path) -> None:
    effects = data["effects"].copy()
    effects = effects[~effects["condition"].isin(["One-shot"])].sort_values(
        "mean_condition_minus_full"
    )
    save_source(effects, output, "fig12_module_effects_vs_full")
    fig, ax = plt.subplots(figsize=(9.1, 6.0), layout="constrained")
    y = np.arange(len(effects))
    values = effects["mean_condition_minus_full"]
    colors = np.where(values < 0, "#C54F45", "#5C9A72")
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.errorbar(
        values,
        y,
        xerr=effects["sd"],
        fmt="none",
        ecolor="#555555",
        capsize=3,
        zorder=1,
    )
    ax.scatter(values, y, color=colors, s=40, zorder=2)
    ax.set_yticks(y, effects["condition"])
    ax.set_xlim(-0.38, 0.12)
    ax.set_xlabel("Condition - full FlowPilot")
    ax.set_title("Matched effects of internal architecture changes")
    ax.grid(axis="x", alpha=0.18)
    ax.text(
        0.99,
        0.02,
        "Mean +/- SD across three matched chemistry cases",
        transform=ax.transAxes,
        ha="right",
        fontsize=7.5,
        color="#555555",
    )
    save_figure(fig, output, "fig12_module_effects_vs_full")


def plot_module_families(data: dict[str, pd.DataFrame], output: Path) -> None:
    summary = data["summary"].set_index("condition")
    families = {
        "Specialist and audit agents": [
            "Full FlowPilot",
            "Without Chemistry agent",
            "Without Kinetics agent",
            "Without Fluidics agent",
            "Without Safety agent",
            "Without Skeptic audit",
        ],
        "Council and refinement": [
            "Full FlowPilot",
            "No council",
            "One-pass council",
            "Without preselection refinement",
            "Without winner revision",
            "Deterministic selection",
            "Without DFMEA",
        ],
        "Candidate budget": [
            "Candidate budget 1",
            "Full FlowPilot",
            "Candidate budget 4",
        ],
    }
    combined = []
    short_labels = {
        "Full FlowPilot": "Full FlowPilot",
        "Without Chemistry agent": "No Chemistry agent",
        "Without Kinetics agent": "No Kinetics agent",
        "Without Fluidics agent": "No Fluidics agent",
        "Without Safety agent": "No Safety agent",
        "Without Skeptic audit": "No Skeptic audit",
        "No council": "No council",
        "One-pass council": "One-pass council",
        "Without preselection refinement": "No preselection refinement",
        "Without winner revision": "No winner revision",
        "Deterministic selection": "Deterministic selection",
        "Without DFMEA": "No DFMEA",
        "Candidate budget 1": "1 candidate",
        "Candidate budget 4": "4 candidates",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2), layout="constrained")
    for ax, (title, conditions) in zip(axes, families.items()):
        conditions = [c for c in conditions if c in summary.index]
        rows = summary.loc[conditions].copy()
        rows["condition"] = conditions
        rows["family"] = title
        combined.append(rows.reset_index(drop=True))
        y = np.arange(len(rows))
        colors = [ARCH_COLORS["FlowPilot"] if c == "Full FlowPilot" else "#7B858D" for c in conditions]
        labels = [short_labels.get(c, c) for c in conditions]
        if title == "Candidate budget":
            labels = ["2 candidates (full)" if c == "Full FlowPilot" else short_labels.get(c, c) for c in conditions]
        ax.barh(
            y,
            rows["mean_score_0_1"],
            xerr=rows["score_sd"],
            capsize=3,
            color=colors,
        )
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0.50, 1.01)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("Consensus score")
    axes[0].set_ylabel("Condition")
    fig.suptitle("Internal factor groups (mean +/- between-case SD; n=3 cases)", fontsize=11)
    save_source(pd.concat(combined, ignore_index=True), output, "fig13_internal_factor_groups")
    save_figure(fig, output, "fig13_internal_factor_groups")


def plot_module_case_heatmap(data: dict[str, pd.DataFrame], output: Path) -> None:
    outcomes = data["outcomes"].copy()
    order = module_order(data["summary"])
    heat = outcomes.pivot(index="condition", columns="case", values="consensus_score_0_1")
    heat = heat.loc[order, CASE_ORDER]
    save_source(heat.reset_index(), output, "fig14_module_case_heatmap")
    fig, ax = plt.subplots(figsize=(7.8, 6.5), layout="constrained")
    image = ax.imshow(heat.values, cmap="YlGnBu", vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(CASE_ORDER)), CASE_ORDER, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(order)), order)
    for row in range(len(order)):
        for col in range(len(CASE_ORDER)):
            value = heat.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 0.82 else "#222222")
    ax.set_title("Internal-condition performance by chemistry case")
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    bar.set_label("Consensus score")
    save_figure(fig, output, "fig14_module_case_heatmap")


def plot_module_criteria(data: dict[str, pd.DataFrame], output: Path) -> None:
    criteria = data["criteria"]
    applicable = criteria[criteria["applicability"] == "APPLICABLE"].copy()
    applicable["normalized_score"] = applicable["score_0_4"] / 4.0
    candidate = (
        applicable.groupby(["architecture", "case", "criterion_id"], as_index=False)[
            "normalized_score"
        ]
        .mean()
    )
    full = candidate[candidate["architecture"] == "Full FlowPilot"][
        ["case", "criterion_id", "normalized_score"]
    ].rename(columns={"normalized_score": "full_score"})
    paired = candidate.merge(full, on=["case", "criterion_id"], how="left")
    paired["condition_minus_full"] = paired["normalized_score"] - paired["full_score"]
    selected = [
        "No council",
        "Without Chemistry agent",
        "Without Kinetics agent",
        "Without Fluidics agent",
        "Without Safety agent",
        "Without Skeptic audit",
        "One-pass council",
        "Candidate budget 1",
        "Candidate budget 4",
    ]
    summary = (
        paired[paired["architecture"].isin(selected)]
        .groupby(["architecture", "criterion_id"])["condition_minus_full"]
        .agg(mean="mean", between_case_sd="std", n_cases="count")
        .reset_index()
    )
    save_source(summary, output, "fig15_module_criterion_effects")
    heat = summary.pivot(index="architecture", columns="criterion_id", values="mean")
    ids = list(CRITERION_NAMES)
    heat = heat.reindex(selected).dropna(how="all").reindex(columns=ids)
    fig, ax = plt.subplots(figsize=(11.2, 5.2), layout="constrained")
    image = ax.imshow(heat.values, cmap="RdYlGn", vmin=-0.35, vmax=0.35, aspect="auto")
    ax.set_xticks(np.arange(len(ids)), ids, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    ax.set_title("Criterion-level effects of internal architecture changes")
    bar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    bar.set_label("Condition - full FlowPilot")
    save_figure(fig, output, "fig15_module_criterion_effects")


def plot_module_integrity(data: dict[str, pd.DataFrame], output: Path) -> None:
    frame = data["summary"].copy()
    order = module_order(frame)
    frame = frame.set_index("condition").loc[order].reset_index()
    save_source(frame, output, "fig16_module_integrity")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.5), layout="constrained")
    colors = [ARCH_COLORS["FlowPilot"] if c == "Full FlowPilot" else "#7B858D" for c in frame["condition"]]
    axes[0].scatter(
        frame["mean_deterministic_score"],
        frame["mean_score_0_1"],
        s=45,
        c=colors,
    )
    for _, row in frame.iterrows():
        if row["condition"] in {"Full FlowPilot", "No council", "One-shot"}:
            axes[0].annotate(
                row["condition"],
                (row["mean_deterministic_score"], row["mean_score_0_1"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )
    axes[0].set_xlabel("Deterministic score")
    axes[0].set_ylabel("Judge consensus score")
    axes[0].set_title("A  Deterministic vs judged quality")
    axes[0].grid(alpha=0.18)

    y = np.arange(len(frame))
    bars = axes[1].barh(y, frame["critical_flags"], color=colors)
    axes[1].set_yticks(y, frame["condition"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Total critical flags")
    axes[1].set_title("B  Critical-error burden")
    axes[1].grid(axis="x", alpha=0.18)
    for bar, executable in zip(bars, frame["executable_rate"]):
        axes[1].text(
            max(bar.get_width(), 0) + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"{executable:.0%} executable",
            va="center",
            fontsize=7,
        )
    save_figure(fig, output, "fig16_module_integrity")


def write_documentation(output: Path) -> None:
    captions = """# Figure captions

## Model-based benchmark

1. **Figure 1. Model and architecture outcome scores.** Consensus scores for matched one-shot and FlowPilot generations. Error bars are sample standard deviations across three independent generation-repeat campaign means.
2. **Figure 2. Paired architecture effects.** Mean within-model difference between FlowPilot and one-shot outcomes. Error bars are sample standard deviations across three paired repeats.
3. **Figure 3. FlowPilot case-level scores.** Scores for the three frozen chemistry cases, stratified by generator model. Error bars are sample standard deviations across three generations.
4. **Figure 4. Case-specific paired effects.** FlowPilot-minus-one-shot effects for each chemistry and model. Error bars are sample standard deviations across three paired generations.
5. **Figure 5. Repeat stability.** Raw repeat-level campaign means; each point averages the same three frozen cases.
6. **Figure 6. Universal criterion profile.** Mean criterion scores across five models and three repeats. Error bars describe dispersion across 15 model-repeat units.
7. **Figure 7. Criterion gain heatmap.** Mean FlowPilot-minus-one-shot criterion effect for each generator model.
8. **Figure 8. Critical-error burden.** Judge critical flags summed over each three-case repeat. Error bars are sample standard deviations across three repeats.
9. **Figure 9. Judge dispersion.** Mean between-judge standard deviation per outcome, aggregated by repeat. Error bars are sample standard deviations across three repeats.
10. **Figure 10. Model benchmark summary.** Compact multipanel summary of outcome quality, paired effect, and total critical flags.

## Internal-module benchmark

11. **Figure 11. Selected internal-condition scores.** Qwen3.8 consensus scores for full FlowPilot and the selected one-shot, no-council, and specialist-agent removal comparisons. Error bars are between-case standard deviations across three matched chemistries, not generation-repeat uncertainty.
12. **Figure 12. Module effects relative to full FlowPilot.** Paired condition-minus-full effects. Error bars are standard deviations across three matched chemistry cases.
13. **Figure 13. Internal factor groups.** Specialist/audit, council/refinement, and candidate-budget conditions shown separately. Error bars are between-case standard deviations.
14. **Figure 14. Module-by-case heatmap.** Consensus score for every condition and frozen chemistry case.
15. **Figure 15. Criterion-level module effects.** Mean condition-minus-full effects under the universal 14-criterion rubric.
16. **Figure 16. Module integrity.** Relationship between deterministic and judged quality, with critical-error and executability outcomes.
"""
    methods = """# Visualization methods and scope

This package uses the five completed NewGen 2.0 publication campaigns: Qwen3.6-27B, Qwen3.8-27B, GPT-4o, Claude Sonnet 4.6, and Claude Opus 4.6. Every model has the same three frozen cases, two architectures, and three generation repeats. Incomplete or diagnostic campaigns are outside the selected package.

All model-level error bars are based on actual generation repeats. A campaign mean is calculated within each repeat by averaging the three case scores, and sample standard deviation uses the three repeat-level campaign means (n=3, ddof=1). Paired effects compare FlowPilot and one-shot for the same model, case, and repeat before aggregation.

The internal-module benchmark is a matched 45-cell screen with Qwen3.8-27B: 15 conditions across three frozen cases. It contains one generation per condition-case cell. Accordingly, its error bars are explicitly labeled as between-case standard deviations (n=3), not repeatability estimates. This screen supports descriptive module attribution only.

Scores use the fixed 14-criterion NewGen 2.0 rubric and three judge families (Qwen, OpenAI, and Claude). Criterion ratings are normalized from 0-4 to 0-1; non-applicable criteria are excluded. Deterministic evidence is supplied for numerical, inventory, topology, and schema checks.
"""
    readme = """# Manuscript benchmark visualizations

This folder consolidates the publication-ready visualization set for the successful model-based NewGen 2.0 campaigns and the Qwen3.8 internal-module screen.

- `figures/`: every figure in PNG (400 dpi), vector PDF, and editable SVG.
- `source_data/`: one CSV containing the plotted values for every figure.
- `FIGURE_CAPTIONS.md`: manuscript-ready captions and uncertainty definitions.
- `METHODS_AND_SCOPE.md`: benchmark scope, statistical unit, and exclusions.
- `package_manifest.json`: machine-readable sources, counts, and figure inventory.
- `ARTIFACT_MANIFEST.sha256`: checksums for integrity verification.

Important: model error bars are generation-repeat SDs. Internal-module error bars are between-case SDs because that screen has one generation per case-condition cell.
"""
    selection = """# Figure selection guide

## Recommended manuscript or presentation figures

- `fig01_model_architecture_mean_sd`: clearest overall model-by-architecture comparison.
- `fig02_paired_architecture_effect`: statistically appropriate matched effect display.
- `fig03_flowpilot_case_scores`: chemistry-specific FlowPilot performance.
- `fig10_model_benchmark_summary`: compact three-panel presentation summary.
- `fig11_module_condition_scores`: full FlowPilot with selected one-shot, no-council, and specialist-agent removal comparisons.
- `fig12_module_effects_vs_full`: direct module-attribution effect plot.
- `fig13_internal_factor_groups`: presentation-friendly grouped internal factors.

## Recommended supporting-information figures

- Figures 4-9 provide case dependence, repeat traces, criterion behavior, critical flags, and judge dispersion.
- Figures 14-16 provide condition-by-case detail, criterion-level module effects, and deterministic/executability context.

Do not describe the internal-module error bars as repeatability. They are between-case SDs from one generation per condition-case cell. The model benchmark figures use actual three-repeat generation SDs.
"""
    (output / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    (output / "METHODS_AND_SCOPE.md").write_text(methods, encoding="utf-8")
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "FIGURE_SELECTION_GUIDE.md").write_text(selection, encoding="utf-8")


def write_manifest(output: Path, model: dict[str, pd.DataFrame], module: dict[str, pd.DataFrame]) -> None:
    figure_files = sorted((output / "figures").glob("*"))
    manifest = {
        "schema_version": "flowpilot_manuscript_visualization_package_v1.0",
        "model_source": str(MODEL_SOURCE),
        "module_source": str(MODULE_SOURCE),
        "included_models": MODEL_ORDER,
        "scope_note": "Only the five frozen publication campaigns are included.",
        "model_benchmark": {
            "candidate_count": len(model["scores"]),
            "case_count": model["scores"]["case"].nunique(),
            "architectures": sorted(model["scores"]["architecture"].unique()),
            "generation_repeats": sorted(model["scores"]["repeat_id"].unique()),
            "sd_basis": "sample SD across three generation-repeat campaign means (ddof=1)",
        },
        "module_benchmark": {
            "candidate_count": len(module["outcomes"]),
            "condition_count": module["outcomes"]["condition"].nunique(),
            "case_count": module["outcomes"]["case"].nunique(),
            "generation_repeats_per_cell": 1,
            "sd_basis": "between-case sample SD across three matched chemistry cases (ddof=1)",
        },
        "figure_count": len([p for p in figure_files if p.suffix == ".png"]),
        "figure_files": [str(p.relative_to(output)) for p in figure_files],
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    rows = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        if path.name == "ARTIFACT_MANIFEST.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(output)}")
    (output / "ARTIFACT_MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def copy_core_tables(output: Path) -> None:
    destination = output / "core_tables"
    destination.mkdir(parents=True, exist_ok=True)
    model_names = [
        "model_architecture_mean_sd.csv",
        "paired_delta_mean_sd.csv",
        "case_architecture_mean_sd.csv",
        "campaign_repeat_means.csv",
        "all_candidate_consensus_scores.csv",
        "all_criterion_judgments.csv",
        "outcome_contract_summary.csv",
    ]
    module_names = [
        "module_summary.csv",
        "module_effects.csv",
        "module_outcomes.csv",
        "criterion_module_effects.csv",
        "deterministic_outcomes.csv",
    ]
    for name in model_names:
        shutil.copy2(MODEL_SOURCE / "tables" / name, destination / f"model_{name}")
    for name in module_names:
        shutil.copy2(MODULE_SOURCE / "tables" / name, destination / f"module_{name}")


def build(output: Path) -> None:
    configure_style()
    model = model_tables()
    module = module_tables()
    output.mkdir(parents=True, exist_ok=True)

    plot_model_architecture(model, output)
    plot_model_paired_effect(model, output)
    plot_flowpilot_case_scores(model, output)
    plot_case_paired_effects(model, output)
    plot_repeat_stability(model, output)
    plot_criteria(model, output)
    plot_errors_and_judge_dispersion(model, output)
    plot_model_summary(model, output)

    plot_module_overview(module, output)
    plot_module_effects(module, output)
    plot_module_families(module, output)
    plot_module_case_heatmap(module, output)
    plot_module_criteria(module, output)
    plot_module_integrity(module, output)

    copy_core_tables(output)
    write_documentation(output)
    write_manifest(output, model, module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.output.resolve())
    print(args.output.resolve())
