#!/usr/bin/env python3
"""Build the requested revised manuscript figures without touching originals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    TwoSlopeNorm,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
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
ARCH_COLORS = {"One-shot": "#D46459", "FlowPilot": "#167C80"}
MODEL_COLORS = {
    "Qwen3.6-27B": "#376AA0",
    "Qwen3.8-27B": "#4C956C",
    "GPT-4o": "#D58B32",
    "Claude Sonnet 4.6": "#8C61A8",
    "Claude Opus 4.6": "#C55467",
}
CRITERION_NAMES = {
    "UO-01": "Transformation fidelity",
    "UO-02": "Required materials",
    "UO-03": "Stoichiometry/feed chemistry",
    "UO-04": "Condition/stage mapping",
    "UO-05": "Executable topology",
    "UO-06": "Liquid material balance",
    "UO-07": "Residence-time closure",
    "UO-10": "Inventory feasibility",
    "UO-11": "Transport plausibility",
    "UO-12": "Hazards and controls",
    "UO-13": "Procedure and work-up",
    "UO-14": "Evidence calibration",
}
CRITERION_ERROR_PLAIN_LANGUAGE = {
    "UO-01": "The proposed design did not preserve the required chemical transformation.",
    "UO-02": "Required materials or process equipment were missing or assigned incorrectly.",
    "UO-03": "Feed composition, stoichiometry, or reagent equivalents were inconsistent.",
    "UO-04": "Reaction conditions or operations were assigned to the wrong process stage.",
    "UO-05": "The equipment sequence was incomplete, misordered, or not executable.",
    "UO-06": "Liquid flow or material-balance calculations did not close consistently.",
    "UO-07": "Reactor volume, flow rate, and residence time did not agree.",
    "UO-08": "Gas flow, pressure, equivalents, or gas-based residence time did not agree.",
    "UO-09": "The sequence or numerical closure of multiple reaction stages was inconsistent.",
    "UO-10": "The design violated declared inventory or equipment operating limits.",
    "UO-11": "The proposed transport or flow behavior was physically implausible.",
    "UO-12": "Important hazards were not matched with concrete engineering controls.",
    "UO-13": "The operating procedure, sampling plan, or work-up was incomplete or unsafe.",
    "UO-14": "A claim was stronger than the evidence supplied for the design.",
}
ERROR_MAP_CRITERIA = {
    "UO-02": "UO-02\nMaterials",
    "UO-03": "UO-03\nFeeds",
    "UO-04": "UO-04\nStages",
    "UO-05": "UO-05\nTopology",
    "UO-06": "UO-06\nLiquid balance",
    "UO-07": "UO-07\nResidence time",
    "UO-08": "UO-08\nGas",
    "UO-10": "UO-10\nInventory",
    "UO-11": "UO-11\nTransport",
    "UO-12": "UO-12\nSafety",
    "UO-13": "UO-13\nProcedure",
}
PRICES_PER_MILLION = {
    "Qwen3.6-27B": {"input": 0.30, "output": 2.00},
    "Qwen3.8-27B": {"input": 0.35, "output": 2.75},
    "GPT-4o": {"input": 2.50, "output": 10.00},
    "Claude Sonnet 4.6": {"input": 2.50, "output": 10.00},
    "Claude Opus 4.6": {"input": 5.00, "output": 25.00},
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.titlepad": 10,
        }
    )


def color_model_tick_labels(ax: plt.Axes) -> None:
    aliases = {
        "Qwen3.6-27B": ("Qwen3.6-27B", "Qwen3.6"),
        "Qwen3.8-27B": ("Qwen3.8-27B", "Qwen3.8"),
        "GPT-4o": ("GPT-4o",),
        "Claude Sonnet 4.6": ("Claude Sonnet 4.6", "Sonnet 4.6"),
        "Claude Opus 4.6": ("Claude Opus 4.6", "Opus 4.6"),
    }
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        text = label.get_text()
        for model, model_aliases in aliases.items():
            if any(alias in text for alias in model_aliases):
                label.set_color(MODEL_COLORS[model])
                label.set_fontweight("semibold")
                break


def read_table(source: Path, name: str) -> pd.DataFrame:
    path = source / "tables" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    destination = output / "figures_revised"
    destination.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".png": {"dpi": 400},
        ".pdf": {},
        ".svg": {},
    }.items():
        fig.savefig(
            destination / f"{stem}{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def save_source(frame: pd.DataFrame, output: Path, stem: str) -> None:
    destination = output / "source_data_revised"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / f"{stem}.csv", index=False)


def load_model_data() -> dict[str, pd.DataFrame]:
    data = {
        "scores": read_table(MODEL_SOURCE, "all_candidate_consensus_scores.csv"),
        "criteria": read_table(MODEL_SOURCE, "all_criterion_judgments.csv"),
        "architecture": read_table(MODEL_SOURCE, "model_architecture_mean_sd.csv"),
        "paired": read_table(MODEL_SOURCE, "paired_delta_mean_sd.csv"),
        "case_summary": read_table(MODEL_SOURCE, "case_architecture_mean_sd.csv"),
    }
    for key, frame in data.items():
        data[key] = frame[frame["model"].isin(MODEL_ORDER)].copy()
    expected = len(MODEL_ORDER) * len(ARCH_ORDER) * len(CASE_ORDER) * 3
    keys = ["model", "architecture", "case", "repeat_id"]
    if len(data["scores"]) != expected or data["scores"][keys].duplicated().any():
        raise RuntimeError(
            f"Expected {expected} unique retained outcomes, got {len(data['scores'])}"
        )
    return data


def plot_figure_01(data: dict[str, pd.DataFrame], output: Path) -> None:
    frame = data["architecture"].copy()
    save_source(frame, output, "fig01_model_architecture_mean_sd_revised")
    fig, ax = plt.subplots(figsize=(9.8, 5.5), layout="constrained")
    y = np.arange(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        pair = frame[frame["model"] == model].set_index("architecture")
        ax.plot(
            [pair.loc["One-shot", "mean"], pair.loc["FlowPilot", "mean"]],
            [index, index],
            color="#C5CBD0",
            linewidth=2.5,
            zorder=1,
        )
    for arch, offset in [("One-shot", -0.04), ("FlowPilot", 0.04)]:
        rows = frame[frame["architecture"] == arch].set_index("model").loc[MODEL_ORDER]
        ax.errorbar(
            rows["mean"],
            y + offset,
            xerr=rows["sample_sd"],
            fmt="o",
            markersize=7,
            capsize=3,
            color=ARCH_COLORS[arch],
            ecolor=ARCH_COLORS[arch],
            label=arch,
            zorder=3,
        )
        for value, yy in zip(rows["mean"], y + offset):
            ax.text(value + 0.008, yy, f"{value:.2f}", va="center", fontsize=8.8)
    ax.set_yticks(y, MODEL_ORDER)
    color_model_tick_labels(ax)
    ax.invert_yaxis()
    ax.set_xlim(0.62, 0.98)
    ax.set_xlabel("Benchmark score")
    ax.set_title("Matched one-shot and FlowPilot outcome quality")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    ax.text(
        0.01,
        -0.13,
        "Mean +/- SD across three generation-repeat campaign means",
        transform=ax.transAxes,
        fontsize=8.6,
        color="#59636B",
        clip_on=False,
    )
    save_figure(fig, output, "fig01_model_architecture_mean_sd_revised")


def plot_sorted_case_outcomes(data: dict[str, pd.DataFrame], output: Path) -> None:
    frame = data["case_summary"].copy().sort_values("mean", ascending=True)
    frame["display_label"] = (
        frame["model"] + " | " + frame["case"] + " | " + frame["architecture"]
    )
    save_source(frame, output, "fig01b_all_case_architecture_scores_sorted")
    fig, ax = plt.subplots(figsize=(12.0, 13.4), layout="constrained")
    y = np.arange(len(frame))
    for arch in ARCH_ORDER:
        mask = frame["architecture"] == arch
        ax.errorbar(
            frame.loc[mask, "mean"],
            y[mask],
            xerr=frame.loc[mask, "sample_sd"],
            fmt="o",
            markersize=6,
            capsize=2.5,
            color=ARCH_COLORS[arch],
            ecolor=ARCH_COLORS[arch],
            label=arch,
        )
    for value, yy in zip(frame["mean"], y):
        ax.text(value + 0.006, yy, f"{value:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y, frame["display_label"])
    color_model_tick_labels(ax)
    ax.invert_yaxis()
    ax.set_xlim(0.54, 1.0)
    ax.set_xlabel("Benchmark score")
    ax.set_title("All model-case outcomes ranked from lowest to highest")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(
        0.99,
        -0.045,
        "Point = mean; error bar = SD across three generation repeats",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.6,
        color="#59636B",
        clip_on=False,
    )
    save_figure(fig, output, "fig01b_all_case_architecture_scores_sorted")


def plot_sorted_outcomes_by_case(
    data: dict[str, pd.DataFrame], output: Path
) -> None:
    panel_order = ["CuAAC", "Photochemical oxidation", "Hydrogenolysis"]
    frame = data["case_summary"].copy()
    short_models = {
        "Qwen3.6-27B": "Qwen3.6",
        "Qwen3.8-27B": "Qwen3.8",
        "GPT-4o": "GPT-4o",
        "Claude Sonnet 4.6": "Sonnet 4.6",
        "Claude Opus 4.6": "Opus 4.6",
    }
    frame["display_label"] = (
        frame["model"].map(short_models) + " | " + frame["architecture"]
    )
    frame["panel_order"] = frame["case"].map(
        {case: index for index, case in enumerate(panel_order)}
    )
    frame = frame.sort_values(["panel_order", "mean"], ascending=[True, True])
    save_source(frame, output, "fig01c_case_panels_scores_sorted")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(21.5, 8.8),
        sharex=True,
    )
    fig.subplots_adjust(left=0.095, right=0.99, top=0.79, bottom=0.16, wspace=0.54)
    for ax, case in zip(axes, panel_order):
        rows = frame[frame["case"] == case].sort_values("mean").reset_index(drop=True)
        y = np.arange(len(rows))
        for row_index in range(len(rows)):
            if row_index % 2 == 0:
                ax.axhspan(
                    row_index - 0.5,
                    row_index + 0.5,
                    color="#F5F7F8",
                    zorder=0,
                )
        for arch in ARCH_ORDER:
            mask = rows["architecture"] == arch
            ax.errorbar(
                rows.loc[mask, "mean"],
                y[mask],
                xerr=rows.loc[mask, "sample_sd"],
                fmt="o",
                markersize=6.5,
                capsize=2.8,
                linewidth=1.9,
                color=ARCH_COLORS[arch],
                ecolor=ARCH_COLORS[arch],
                zorder=3,
            )
        for value, yy in zip(rows["mean"], y):
            ax.text(
                1.008,
                yy,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=11.0,
                color="#20272C",
            )
        ax.set_yticks(y, rows["display_label"])
        for label, architecture in zip(ax.get_yticklabels(), rows["architecture"]):
            label.set_color(ARCH_COLORS[architecture])
            label.set_fontweight("semibold")
            label.set_fontsize(11.2)
        ax.tick_params(axis="x", labelsize=11.2)
        ax.invert_yaxis()
        ax.set_xlim(0.54, 1.03)
        ax.set_title(case, fontsize=14.5)
        ax.grid(axis="x", alpha=0.18)
        ax.axvline(0.99, color="#D8DDE1", linewidth=0.8)
        ax.text(
            1.008,
            -0.85,
            "Mean",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#59636B",
        )
    fig.suptitle(
        "Ranked model-architecture outcomes within each chemistry case",
        fontsize=17.0,
        y=0.965,
    )
    fig.supxlabel("Benchmark score", y=0.065, fontsize=14.0)
    fig.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=ARCH_COLORS[arch], markeredgecolor="none",
                label=arch, markersize=7,
            )
            for arch in ARCH_ORDER
        ],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        fontsize=12.0,
    )
    fig.text(
        0.995,
        0.01,
        "Point = mean; error bar = SD across three generation repeats",
        ha="right",
        fontsize=10.5,
        color="#59636B",
    )
    save_figure(fig, output, "fig01c_case_panels_scores_sorted")


def plot_figure_04(data: dict[str, pd.DataFrame], output: Path) -> None:
    scores = data["scores"]
    wide = scores.pivot(
        index=["model", "case", "repeat_id"],
        columns="architecture",
        values="mean_score_0_1",
    ).reset_index()
    wide["paired_delta"] = wide["FlowPilot"] - wide["One-shot"]
    summary = (
        wide.groupby(["model", "case"])["paired_delta"]
        .agg(mean="mean", sample_sd="std", n_repeats="count")
        .reset_index()
    )
    save_source(summary, output, "fig04_case_paired_effects_revised")
    fig, axes = plt.subplots(
        1, 3, figsize=(14.6, 5.7), sharex=True, sharey=True, layout="constrained"
    )
    case_colors = ["#456FA3", "#D09137", "#7960A5"]
    for ax, case, color in zip(axes, CASE_ORDER, case_colors):
        rows = summary[summary["case"] == case].set_index("model").loc[MODEL_ORDER]
        y = np.arange(len(rows))
        ax.axvline(0, color="#43494E", linewidth=1)
        ax.errorbar(
            rows["mean"],
            y,
            xerr=rows["sample_sd"],
            fmt="o",
            markersize=6.5,
            capsize=3,
            color=color,
            ecolor=color,
        )
        for value, yy in zip(rows["mean"], y):
            ax.text(value + 0.008, yy, f"{value:+.2f}", va="center", fontsize=8.7)
        ax.set_title(case)
        ax.set_yticks(y, MODEL_ORDER)
        color_model_tick_labels(ax)
        ax.set_xlim(-0.13, 0.42)
        ax.grid(axis="x", alpha=0.18)
    axes[0].invert_yaxis()
    fig.supxlabel("Paired benchmark score difference: FlowPilot - one-shot")
    fig.suptitle("Case-specific architecture effects", fontsize=13)
    save_figure(fig, output, "fig04_case_paired_effects_revised")


def criterion_delta_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    criteria = data["criteria"]
    repeat_lookup = data["scores"][["candidate_id", "repeat_id"]]
    applicable = criteria[criteria["applicability"] == "APPLICABLE"].copy()
    applicable["normalized_score"] = applicable["score_0_4"] / 4.0
    candidate = (
        applicable.groupby(
            ["candidate_id", "model", "architecture", "case", "criterion_id"],
            as_index=False,
        )["normalized_score"]
        .mean()
        .merge(repeat_lookup, on="candidate_id", how="left", validate="many_to_one")
    )
    repeat_units = candidate.groupby(
        ["model", "architecture", "case", "repeat_id", "criterion_id"],
        as_index=False,
    )["normalized_score"].mean()
    wide = repeat_units.pivot_table(
        index=["model", "case", "repeat_id", "criterion_id"],
        columns="architecture",
        values="normalized_score",
    ).reset_index()
    wide["delta"] = wide["FlowPilot"] - wide["One-shot"]
    return (
        wide[wide["criterion_id"].isin(CRITERION_NAMES)]
        .groupby(["model", "case", "criterion_id"])["delta"]
        .agg(mean_paired_delta="mean", sample_sd="std", n_repeats="count")
        .reset_index()
    )


def plot_figure_07(data: dict[str, pd.DataFrame], output: Path) -> None:
    summary = criterion_delta_table(data)
    save_source(summary, output, "fig07_criterion_gain_heatmap_revised")
    ids = list(CRITERION_NAMES)
    panel_order = ["CuAAC", "Photochemical oxidation", "Hydrogenolysis"]
    cmap = LinearSegmentedColormap.from_list(
        "flowpilot_gain",
        ["#B6403A", "#F7F7F5", "#117A65"],
    )
    cmap.set_bad("#E4E7E9")
    norm = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=0.65)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(17.0, 13.6),
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.99,
        top=0.93,
        bottom=0.14,
        hspace=0.38,
    )
    images = []
    for ax, case in zip(axes, panel_order):
        heat = (
            summary[summary["case"] == case]
            .pivot(
                index="model",
                columns="criterion_id",
                values="mean_paired_delta",
            )
            .reindex(index=MODEL_ORDER, columns=ids)
        )
        image = ax.imshow(
            np.ma.masked_invalid(heat.values),
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )
        images.append(image)
        ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
        for label in ax.get_yticklabels():
            label.set_color("#111111")
            label.set_fontweight("semibold")
            label.set_fontsize(11.5)
        ax.set_title(case, loc="left", fontweight="bold", fontsize=14.5)
        ax.set_xticks(np.arange(len(ids)), ids)
        ax.tick_params(axis="x", labelbottom=True, labelsize=11.0)
        for row in range(len(MODEL_ORDER)):
            for col in range(len(ids)):
                value = heat.iloc[row, col]
                if pd.isna(value):
                    ax.text(
                        col,
                        row,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=10.0,
                        color="#59636B",
                    )
                    continue
                rgba = cmap(norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                display_value = 0.0 if abs(value) < 0.005 else value
                ax.text(
                    col,
                    row,
                    f"{display_value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.0,
                    color="white" if luminance < 0.52 else "#20272C",
                )
    fig.suptitle(
        "Criterion-level gain from FlowPilot within each chemistry case",
        fontsize=17.0,
        y=0.98,
    )
    colorbar_ax = fig.add_axes([0.14, 0.072, 0.85, 0.018])
    bar = fig.colorbar(
        images[0],
        cax=colorbar_ax,
        orientation="horizontal",
    )
    bar.set_label(
        "Benchmark score difference: FlowPilot - one-shot",
        labelpad=5,
        fontsize=12.0,
    )
    bar.ax.tick_params(labelsize=10.5)
    fig.text(
        0.14,
        0.012,
        "Red = lower than one-shot; white = no change; green = higher than one-shot. "
        "UO-08 and UO-09 are excluded from this chemistry-comparison figure.",
        fontsize=10.2,
        color="#59636B",
    )
    save_figure(fig, output, "fig07_criterion_gain_heatmap_revised")


def plot_figure_08(data: dict[str, pd.DataFrame], output: Path) -> None:
    scores = data["scores"]
    errors = (
        scores.groupby(["model", "architecture", "repeat_id"], as_index=False)[
            "total_judge_critical_flags"
        ]
        .sum()
        .groupby(["model", "architecture"])["total_judge_critical_flags"]
        .agg(mean="mean", sample_sd="std", n_repeats="count")
        .reset_index()
    )
    save_source(errors, output, "fig08_critical_flags_revised")
    fig, ax = plt.subplots(figsize=(9.8, 5.6), layout="constrained")
    y = np.arange(len(MODEL_ORDER))
    height = 0.33
    for offset, arch in [(-height / 2, "One-shot"), (height / 2, "FlowPilot")]:
        rows = errors[errors["architecture"] == arch].set_index("model").loc[MODEL_ORDER]
        bars = ax.barh(
            y + offset,
            rows["mean"],
            height,
            xerr=rows["sample_sd"],
            capsize=3,
            color=ARCH_COLORS[arch],
            label=arch,
        )
        for bar, value in zip(bars, rows["mean"]):
            ax.text(
                value + 0.35,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                fontsize=8.7,
            )
    ax.set_yticks(y, MODEL_ORDER)
    color_model_tick_labels(ax)
    ax.invert_yaxis()
    ax.set_xlabel("Critical flags per three-case repeat")
    ax.set_title("Critical-error burden across matched campaigns")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.text(
        0.99,
        0.97,
        "Mean +/- SD across three repeats",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        color="#59636B",
    )
    save_figure(fig, output, "fig08_critical_flags_revised")


def _clean_report_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _joined_unique_text(values: pd.Series) -> str:
    unique: list[str] = []
    for value in values:
        cleaned = _clean_report_text(value)
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return " | ".join(unique)


def _representative_error_row(rows: pd.DataFrame) -> pd.Series:
    confidence_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    ranked = rows.copy()
    ranked["_confidence_rank"] = ranked["confidence"].map(confidence_rank).fillna(0)
    ranked["_detail_length"] = ranked["rationale"].fillna("").astype(str).str.len()
    return ranked.sort_values(
        ["_confidence_rank", "_detail_length"], ascending=[False, False]
    ).iloc[0]


def campaign_error_register_tables(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = data["scores"].copy()
    judgments = data["criteria"].copy()
    flagged = judgments[
        judgments["critical_error"].astype(str).str.lower().eq("true")
    ].copy()

    repeat_lookup = scores[["candidate_id", "repeat_id"]]
    flagged = flagged.merge(
        repeat_lookup,
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )
    detail_rows: list[dict[str, object]] = []
    group_columns = [
        "candidate_id",
        "model",
        "architecture",
        "case",
        "repeat_id",
        "criterion_id",
    ]
    for keys, rows in flagged.groupby(group_columns, sort=False, dropna=False):
        representative = _representative_error_row(rows)
        criterion_id = str(keys[-1])
        detail_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "criterion_name": CRITERION_NAMES.get(
                    criterion_id,
                    {
                        "UO-08": "Gas bookkeeping",
                        "UO-09": "Multistage closure",
                    }.get(criterion_id, criterion_id),
                ),
                "plain_english_error": CRITERION_ERROR_PLAIN_LANGUAGE[criterion_id],
                "campaign_specific_finding": _clean_report_text(
                    representative.get("rationale")
                ),
                "observed_evidence": _joined_unique_text(rows["observed_values"]),
                "expected_or_correct": _clean_report_text(
                    representative.get("expected_or_correct")
                ),
                "recommended_fix": _clean_report_text(
                    representative.get("required_correction")
                ),
                "judge_flags": int(len(rows)),
                "judge_families": ", ".join(
                    sorted(rows["judge_family"].dropna().astype(str).unique())
                ),
                "mean_criterion_score_0_4": float(rows["score_0_4"].mean()),
                "minimum_criterion_score_0_4": float(rows["score_0_4"].min()),
            }
        )
    details = pd.DataFrame(detail_rows)

    detail_lookup = {
        candidate_id: rows.sort_values("criterion_id")
        for candidate_id, rows in details.groupby("candidate_id", sort=False)
    }
    summary_rows: list[dict[str, object]] = []
    for _, score_row in scores.iterrows():
        candidate_id = score_row["candidate_id"]
        campaign_errors = detail_lookup.get(candidate_id)
        if campaign_errors is None:
            unique_error_count = 0
            affected_criteria = "None"
            errors_clear_english = "No critical errors reported by any benchmark judge."
            status = "No critical errors"
        else:
            unique_error_count = int(len(campaign_errors))
            affected_criteria = ", ".join(campaign_errors["criterion_id"])
            errors_clear_english = " | ".join(
                f"{row.criterion_id}: {row.plain_english_error}"
                for row in campaign_errors.itertuples()
            )
            status = "Critical error(s) reported"
        raw_flag_count = int(
            flagged[flagged["candidate_id"] == candidate_id].shape[0]
        )
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "model": score_row["model"],
                "architecture": score_row["architecture"],
                "case": score_row["case"],
                "repeat_id": score_row["repeat_id"],
                "benchmark_score": float(score_row["mean_score_0_1"]),
                "status": status,
                "unique_error_count": unique_error_count,
                "judge_flag_count": raw_flag_count,
                "affected_criteria": affected_criteria,
                "errors_clear_english": errors_clear_english,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["_model_order"] = summary["model"].map(
        {model: index for index, model in enumerate(MODEL_ORDER)}
    )
    summary["_architecture_order"] = summary["architecture"].map(
        {architecture: index for index, architecture in enumerate(ARCH_ORDER)}
    )
    summary["_case_order"] = summary["case"].map(
        {case: index for index, case in enumerate(CASE_ORDER)}
    )
    summary = (
        summary.sort_values(
            ["_model_order", "_architecture_order", "_case_order", "repeat_id"]
        )
        .drop(
            columns=["_model_order", "_architecture_order", "_case_order"]
        )
        .reset_index(drop=True)
    )
    details = details.merge(
        scores[["candidate_id", "mean_score_0_1"]],
        on="candidate_id",
        how="left",
        validate="many_to_one",
    ).rename(columns={"mean_score_0_1": "benchmark_score"})
    details = details[
        [
            "candidate_id",
            "model",
            "architecture",
            "case",
            "repeat_id",
            "benchmark_score",
            "criterion_id",
            "criterion_name",
            "plain_english_error",
            "campaign_specific_finding",
            "observed_evidence",
            "expected_or_correct",
            "recommended_fix",
            "judge_flags",
            "judge_families",
            "mean_criterion_score_0_4",
            "minimum_criterion_score_0_4",
        ]
    ].sort_values(["model", "architecture", "case", "repeat_id", "criterion_id"])
    raw_columns = [
        "candidate_id",
        "model",
        "architecture",
        "case",
        "repeat_id",
        "criterion_id",
        "judge_family",
        "severity",
        "confidence",
        "score_0_4",
        "rationale",
        "observed_values",
        "expected_or_correct",
        "required_correction",
        "evidence_paths",
    ]
    raw_flags = flagged[raw_columns].sort_values(
        ["model", "architecture", "case", "repeat_id", "criterion_id", "judge_family"]
    )
    return summary, details.reset_index(drop=True), raw_flags.reset_index(drop=True)


def _write_error_register_markdown(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    output: Path,
) -> None:
    affected = int((summary["unique_error_count"] > 0).sum())
    lines = [
        "# Figure 8 companion: campaign error register",
        "",
        "This register lists benchmark-judge critical errors for every retained campaign. "
        "A judge flag is one judge's report; a unique error consolidates judges that flagged "
        "the same criterion in the same campaign.",
        "",
        f"- Campaigns reviewed: {len(summary)}",
        f"- Campaigns with at least one critical error: {affected}",
        f"- Campaigns with no critical errors: {len(summary) - affected}",
        f"- Consolidated campaign-criterion errors: {len(details)}",
        f"- Raw judge flags: {int(summary['judge_flag_count'].sum())}",
        "",
        "The Excel workbook contains the full campaign-specific findings, evidence, and "
        "recommended corrections. The summary below uses plain-English error categories.",
        "",
    ]
    for model in MODEL_ORDER:
        lines.extend([f"## {model}", ""])
        rows = summary[summary["model"] == model]
        lines.append(
            "| Architecture | Chemistry | Repeat | Score | Unique errors | Errors |"
        )
        lines.append("|---|---|---:|---:|---:|---|")
        for row in rows.itertuples():
            errors = str(row.errors_clear_english).replace("|", ";")
            lines.append(
                f"| {row.architecture} | {row.case} | {row.repeat_id} | "
                f"{row.benchmark_score:.3f} | {row.unique_error_count} | {errors} |"
            )
        lines.append("")
    (output / "FIG08_CAMPAIGN_ERROR_REGISTER.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _format_error_register_workbook(path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="167C80")
    error_fill = PatternFill("solid", fgColor="FCE8E6")
    clear_fill = PatternFill("solid", fgColor="E9F4EF")
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for column_cells in sheet.columns:
            values = [str(cell.value or "") for cell in column_cells[:200]]
            width = min(max(max(map(len, values), default=0) + 2, 10), 58)
            sheet.column_dimensions[column_cells[0].column_letter].width = width
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        if sheet.title == "Campaign Summary":
            status_column = next(
                cell.column for cell in sheet[1] if cell.value == "status"
            )
            for row in sheet.iter_rows(min_row=2):
                fill = (
                    error_fill
                    if row[status_column - 1].value == "Critical error(s) reported"
                    else clear_fill
                )
                for cell in row:
                    cell.fill = fill
    workbook.save(path)


def write_campaign_error_register(
    data: dict[str, pd.DataFrame], output: Path
) -> None:
    summary, details, raw_flags = campaign_error_register_tables(data)
    save_source(summary, output, "fig08_campaign_error_summary")
    save_source(details, output, "fig08_campaign_error_details")
    save_source(raw_flags, output, "fig08_judge_critical_evidence")
    _write_error_register_markdown(summary, details, output)

    table_directory = output / "tables_revised"
    table_directory.mkdir(parents=True, exist_ok=True)
    workbook_path = table_directory / "fig08_campaign_error_register.xlsx"
    readme = pd.DataFrame(
        {
            "Item": [
                "Purpose",
                "Campaign scope",
                "Unique error",
                "Judge flag",
                "Interpretation",
            ],
            "Explanation": [
                "Lists the critical errors reported for every retained benchmark campaign.",
                "Five models, two architectures, three chemistries, and three repeats: 90 campaigns.",
                "One criterion flagged within one campaign after consolidating duplicate judge reports.",
                "One judge's critical-error report. Several judges may flag the same unique error.",
                "No critical errors means no benchmark judge marked a critical failure; it does not prove the design is experimentally optimal.",
            ],
        }
    )
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        summary.to_excel(writer, sheet_name="Campaign Summary", index=False)
        details.to_excel(writer, sheet_name="Error Details", index=False)
        raw_flags.to_excel(writer, sheet_name="Judge Evidence", index=False)
    _format_error_register_workbook(workbook_path)


def plot_campaign_error_map(data: dict[str, pd.DataFrame], output: Path) -> None:
    summary, details, _ = campaign_error_register_tables(data)
    judgments = data["criteria"]
    criteria = list(ERROR_MAP_CRITERIA)
    unrepresented = set(details["criterion_id"]) - set(criteria)
    if unrepresented:
        raise RuntimeError(
            f"Critical-error criteria missing from Figure 08a: {sorted(unrepresented)}"
        )

    detail_flags = details.set_index(["candidate_id", "criterion_id"])["judge_flags"]
    applicability = (
        judgments.groupby(["candidate_id", "criterion_id"])["applicability"]
        .apply(
            lambda values: (
                "APPLICABLE"
                if (values.astype(str) == "APPLICABLE").any()
                else "NOT_APPLICABLE"
            )
        )
        .to_dict()
    )
    long_rows: list[dict[str, object]] = []
    for campaign in summary.itertuples():
        for criterion_id in criteria:
            state = applicability[(campaign.candidate_id, criterion_id)]
            judge_flags = int(
                detail_flags.get((campaign.candidate_id, criterion_id), 0)
            )
            long_rows.append(
                {
                    "candidate_id": campaign.candidate_id,
                    "model": campaign.model,
                    "architecture": campaign.architecture,
                    "case": campaign.case,
                    "repeat_id": campaign.repeat_id,
                    "benchmark_score": campaign.benchmark_score,
                    "criterion_id": criterion_id,
                    "criterion_label": ERROR_MAP_CRITERIA[criterion_id].replace(
                        "\n", " "
                    ),
                    "applicability": state,
                    "judge_flags": judge_flags,
                    "cell_value": -1 if state == "NOT_APPLICABLE" else judge_flags,
                }
            )
    map_data = pd.DataFrame(long_rows)
    save_source(map_data, output, "fig08a_campaign_error_map")

    short_models = {
        "Qwen3.6-27B": "Qwen3.6",
        "Qwen3.8-27B": "Qwen3.8",
        "GPT-4o": "GPT-4o",
        "Claude Sonnet 4.6": "Sonnet 4.6",
        "Claude Opus 4.6": "Opus 4.6",
    }
    short_cases = {
        "Hydrogenolysis": "Hydrogenolysis",
        "Photochemical oxidation": "Photo-oxidation",
        "CuAAC": "CuAAC",
    }
    colors = ["#D9DEE2", "#EEF5F0", "#F5C4BF", "#DF756B", "#A92F2B"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    def add_campaign_hierarchy(ax: plt.Axes, campaigns: pd.DataFrame) -> None:
        transform = ax.get_yaxis_transform()
        for model, group in campaigns.groupby("model", sort=False):
            start = group.index.min() - 0.42
            end = group.index.max() + 0.42
            midpoint = (start + end) / 2
            ax.plot(
                [-0.33, -0.33],
                [start, end],
                transform=transform,
                color="#4F5961",
                linewidth=1.9,
                clip_on=False,
            )
            ax.plot(
                [-0.33, -0.30],
                [start, start],
                transform=transform,
                color="#4F5961",
                linewidth=1.9,
                clip_on=False,
            )
            ax.plot(
                [-0.33, -0.30],
                [end, end],
                transform=transform,
                color="#4F5961",
                linewidth=1.4,
                clip_on=False,
            )
            ax.text(
                -0.37,
                midpoint,
                short_models[model],
                transform=transform,
                ha="right",
                va="center",
                fontsize=14.5,
                fontweight="bold",
                color="#111111",
                bbox={
                    "boxstyle": "square,pad=0.25",
                    "facecolor": "#F2F4F5",
                    "edgecolor": "#AAB2B8",
                    "linewidth": 1.1,
                },
                clip_on=False,
            )
        for (_, case), group in campaigns.groupby(["model", "case"], sort=False):
            start = group.index.min() - 0.38
            end = group.index.max() + 0.38
            midpoint = (start + end) / 2
            ax.plot(
                [-0.105, -0.105],
                [start, end],
                transform=transform,
                color="#7C878F",
                linewidth=1.5,
                clip_on=False,
            )
            ax.plot(
                [-0.105, -0.08],
                [start, start],
                transform=transform,
                color="#7C878F",
                linewidth=1.5,
                clip_on=False,
            )
            ax.plot(
                [-0.105, -0.08],
                [end, end],
                transform=transform,
                color="#7C878F",
                linewidth=1.5,
                clip_on=False,
            )
            ax.text(
                -0.125,
                midpoint,
                short_cases[case],
                transform=transform,
                ha="right",
                va="center",
                fontsize=13.0,
                fontweight="bold",
                color="#111111",
                bbox={
                    "boxstyle": "square,pad=0.16",
                    "facecolor": "#FFFFFF",
                    "edgecolor": "none",
                },
                clip_on=False,
            )
        for x_position, heading, alignment in [
            (-0.50, "Model", "left"),
            (-0.26, "Chemistry", "left"),
            (-0.01, "Repeat", "right"),
        ]:
            ax.text(
                x_position,
                -0.78,
                heading,
                transform=transform,
                ha=alignment,
                va="center",
                fontsize=12.8,
                fontweight="bold",
                color="#111111",
                clip_on=False,
            )

    fig, axes = plt.subplots(1, 2, figsize=(30.0, 20.5))
    fig.subplots_adjust(
        left=0.17,
        right=0.985,
        top=0.89,
        bottom=0.18,
        wspace=0.78,
    )
    image = None
    for panel_index, (ax, architecture) in enumerate(zip(axes, ARCH_ORDER)):
        campaigns = summary[summary["architecture"] == architecture].reset_index(
            drop=True
        )
        values = (
            map_data[map_data["architecture"] == architecture]
            .pivot(
                index="candidate_id",
                columns="criterion_id",
                values="cell_value",
            )
            .reindex(index=campaigns["candidate_id"], columns=criteria)
            .to_numpy()
        )
        image = ax.imshow(
            values,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        repeat_labels = [
            f"R{int(str(row.repeat_id).split('_')[-1])}"
            for row in campaigns.itertuples()
        ]
        ax.set_yticks(
            np.arange(len(campaigns)),
            repeat_labels,
        )
        ax.set_xticks(
            np.arange(len(criteria)),
            [ERROR_MAP_CRITERIA[criterion_id] for criterion_id in criteria],
            rotation=38,
            ha="right",
            rotation_mode="anchor",
        )
        ax.tick_params(axis="x", labelsize=11.2, pad=10)
        ax.tick_params(axis="y", labelsize=11.5, colors="#111111")
        for label in ax.get_yticklabels():
            label.set_color("#111111")
            label.set_fontweight("semibold")
        add_campaign_hierarchy(ax, campaigns)
        flagged_campaigns = int((campaigns["unique_error_count"] > 0).sum())
        no_error_campaigns = len(campaigns) - flagged_campaigns
        ax.set_title(
            f"{'A' if panel_index == 0 else 'B'}  {architecture}",
            loc="left",
            fontweight="bold",
            fontsize=18.0,
            y=1.055,
            pad=18,
        )
        ax.text(
            0.5,
            1.015,
            f"{no_error_campaigns}/{len(campaigns)} campaigns had no critical errors",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13.0,
            fontweight="semibold",
            color="#315D46",
            bbox={
                "boxstyle": "square,pad=0.3",
                "facecolor": "#EEF5F0",
                "edgecolor": "#9DB7A8",
                "linewidth": 0.9,
            },
        )
        ax.set_xticks(np.arange(-0.5, len(criteria), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(campaigns), 1), minor=True)
        ax.grid(which="minor", color="#FFFFFF", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        for boundary in range(3, len(campaigns), 3):
            linewidth = 1.5 if boundary % 9 == 0 else 0.85
            color = "#5E6870" if boundary % 9 == 0 else "#AAB2B8"
            ax.axhline(boundary - 0.5, color=color, linewidth=linewidth)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                if value >= 1:
                    ax.text(
                        column_index,
                        row_index,
                        str(int(value)),
                        ha="center",
                        va="center",
                        fontsize=10.8,
                        fontweight="bold",
                        color="white" if value >= 2 else "#4A2421",
                    )
        ax.set_xlim(-0.5, len(criteria) + 0.85)
        ax.text(
            len(criteria) + 0.20,
            -1.15,
            "Total",
            ha="center",
            va="center",
            fontsize=12.0,
            fontweight="bold",
        )
        for row_index, total in enumerate(campaigns["unique_error_count"]):
            ax.text(
                len(criteria) + 0.20,
                row_index,
                str(int(total)),
                ha="center",
                va="center",
                fontsize=10.8,
                color="#A92F2B" if total else "#7A858D",
                fontweight="bold" if total else "normal",
            )

    fig.suptitle(
        "Campaign-level critical error map",
        fontsize=22.0,
        y=0.97,
    )
    colorbar_ax = fig.add_axes([0.30, 0.058, 0.40, 0.018])
    colorbar = fig.colorbar(
        image,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=[-1, 0, 1, 2, 3],
    )
    colorbar.ax.set_xticklabels(
        ["Not applicable", "No critical error", "1 judge", "2 judges", "3 judges"]
    )
    colorbar.ax.tick_params(labelsize=11.5)
    colorbar.set_label(
        "Independent judge flags for the same campaign-criterion error",
        fontsize=13.0,
    )
    fig.text(
        0.15,
        0.015,
        "Each row is one model x chemistry x repeat campaign. Only criteria with at least "
        "one critical flag are shown. Cell numbers indicate judge agreement; the Total "
        "column counts distinct error criteria, not repeated judge flags.",
        fontsize=11.5,
        color="#59636B",
    )
    save_figure(fig, output, "fig08a_campaign_error_map")


def plot_figure_11(output: Path) -> None:
    summary = read_table(MODULE_SOURCE, "module_summary.csv").set_index("condition")
    selected = [
        "One-shot",
        "No council",
        "Without Chemistry agent",
        "Without Kinetics agent",
        "Without Fluidics agent",
        "Without Safety agent",
        "Full FlowPilot",
    ]
    frame = summary.loc[selected].sort_values("mean_score_0_1").reset_index()
    save_source(frame, output, "fig11_module_condition_scores_revised")
    fig, ax = plt.subplots(figsize=(9.9, 6.2), layout="constrained")
    y = np.arange(len(frame))
    colors = [
        ARCH_COLORS["One-shot"]
        if value in {"One-shot", "No council"}
        else ARCH_COLORS["FlowPilot"]
        if value == "Full FlowPilot"
        else "#7D8991"
        for value in frame["condition"]
    ]
    bars = ax.barh(
        y,
        frame["mean_score_0_1"],
        xerr=frame["score_sd"],
        capsize=3,
        color=colors,
    )
    for bar, value in zip(bars, frame["mean_score_0_1"]):
        ax.text(
            value + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=9.0,
        )
    ax.set_yticks(y, frame["condition"])
    ax.invert_yaxis()
    ax.set_xlim(0.58, 1.01)
    ax.set_xlabel("Benchmark score")
    ax.set_title("Qwen3.8 internal architecture conditions")
    ax.grid(axis="x", alpha=0.18)
    ax.text(
        0.99,
        0.02,
        "Mean +/- between-case SD; three matched cases",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.6,
        color="#59636B",
    )
    save_figure(fig, output, "fig11_module_condition_scores_revised")


def _token_counts(run_directory: str) -> tuple[int, int]:
    path = Path(run_directory) / "run_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    totals = payload.get("token_totals") or {}
    input_tokens = int(totals.get("input_tokens") or totals.get("prompt_tokens") or 0)
    output_tokens = int(totals.get("output_tokens") or totals.get("completion_tokens") or 0)
    if input_tokens <= 0 or output_tokens <= 0:
        raise RuntimeError(f"Missing measured token usage in {path}")
    return input_tokens, output_tokens


def cost_tables(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = data["scores"].copy()
    token_rows = outcomes["run_directory"].map(_token_counts)
    outcomes["input_tokens"] = [value[0] for value in token_rows]
    outcomes["output_tokens"] = [value[1] for value in token_rows]
    outcomes["input_price_per_million_usd"] = outcomes["model"].map(
        lambda model: PRICES_PER_MILLION[model]["input"]
    )
    outcomes["output_price_per_million_usd"] = outcomes["model"].map(
        lambda model: PRICES_PER_MILLION[model]["output"]
    )
    outcomes["generation_cost_usd"] = (
        outcomes["input_tokens"] * outcomes["input_price_per_million_usd"]
        + outcomes["output_tokens"] * outcomes["output_price_per_million_usd"]
    ) / 1_000_000.0
    outcomes["quality_per_usd"] = (
        outcomes["mean_score_0_1"] / outcomes["generation_cost_usd"]
    )
    summary = (
        outcomes.groupby(["model", "architecture"])
        .agg(
            mean_score=("mean_score_0_1", "mean"),
            score_sd=("mean_score_0_1", "std"),
            mean_generation_cost_usd=("generation_cost_usd", "mean"),
            generation_cost_sd_usd=("generation_cost_usd", "std"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            mean_outcome_quality_per_usd=("quality_per_usd", "mean"),
            outcome_quality_per_usd_sd=("quality_per_usd", "std"),
            n_outcomes=("candidate_id", "count"),
        )
        .reset_index()
    )
    summary["aggregate_quality_per_usd"] = (
        summary["mean_score"] / summary["mean_generation_cost_usd"]
    )
    return outcomes, summary


def plot_quality_per_cost(data: dict[str, pd.DataFrame], output: Path) -> None:
    outcomes, summary = cost_tables(data)
    save_source(outcomes, output, "fig17_quality_per_generation_cost_outcomes")
    save_source(summary, output, "fig17_quality_per_generation_cost_summary")
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.6), layout="constrained")
    markers = {"One-shot": "o", "FlowPilot": "s"}
    short_names = {
        "Qwen3.6-27B": "Qwen3.6",
        "Qwen3.8-27B": "Qwen3.8",
        "GPT-4o": "GPT-4o",
        "Claude Sonnet 4.6": "Sonnet 4.6",
        "Claude Opus 4.6": "Opus 4.6",
    }
    for model in MODEL_ORDER:
        pair = summary[summary["model"] == model].set_index("architecture")
        axes[0].plot(
            pair.loc[ARCH_ORDER, "mean_generation_cost_usd"],
            pair.loc[ARCH_ORDER, "mean_score"],
            color=MODEL_COLORS[model],
            linewidth=1.1,
            alpha=0.45,
            zorder=1,
        )
    for _, row in summary.iterrows():
        axes[0].scatter(
            row["mean_generation_cost_usd"],
            row["mean_score"],
            marker=markers[row["architecture"]],
            s=72,
            color=MODEL_COLORS[row["model"]],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.62, 0.98)
    axes[0].set_xlabel("Mean generation cost per outcome (USD, log scale)")
    axes[0].set_ylabel("Benchmark score")
    axes[0].set_title("A  Quality and generation cost")
    axes[0].grid(alpha=0.18)
    model_legend = axes[0].legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="-", linewidth=1.2,
                color=MODEL_COLORS[model], label=short_names[model], markersize=5,
            )
            for model in MODEL_ORDER
        ],
        frameon=False,
        loc="lower right",
        title="Generator model",
    )
    axes[0].add_artist(model_legend)
    axes[0].legend(
        handles=[
            Line2D(
                [0], [0], marker=markers[arch], linestyle="none",
                markerfacecolor="#59636B", markeredgecolor="white",
                label=arch, markersize=7,
            )
            for arch in ARCH_ORDER
        ],
        frameon=False,
        loc="upper left",
        title="Architecture",
    )

    ranked = summary.sort_values("aggregate_quality_per_usd").copy()
    ranked["display_label"] = ranked["model"] + " | " + ranked["architecture"]
    y = np.arange(len(ranked))
    colors = [ARCH_COLORS[value] for value in ranked["architecture"]]
    bars = axes[1].barh(y, ranked["aggregate_quality_per_usd"], color=colors)
    for bar, value in zip(bars, ranked["aggregate_quality_per_usd"]):
        axes[1].text(
            value * 1.06,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=8.4,
        )
    axes[1].set_yticks(y, ranked["display_label"])
    color_model_tick_labels(axes[1])
    axes[1].set_xscale("log")
    axes[1].set_xlabel(
        "Quality-per-cost index (benchmark score / USD, log scale)"
    )
    axes[1].set_title("B  Higher index indicates greater cost efficiency")
    axes[1].grid(axis="x", alpha=0.18)
    axes[1].legend(
        handles=[
            Line2D([0], [0], color=ARCH_COLORS[arch], lw=7, label=arch)
            for arch in ARCH_ORDER
        ],
        frameon=False,
        loc="lower right",
    )
    save_figure(fig, output, "fig17_quality_per_generation_cost")


def plot_flowpilot_quality_per_cost(
    data: dict[str, pd.DataFrame], output: Path
) -> None:
    outcomes, summary = cost_tables(data)
    outcomes = outcomes[outcomes["architecture"] == "FlowPilot"].copy()
    summary = summary[summary["architecture"] == "FlowPilot"].copy()
    save_source(outcomes, output, "fig18_flowpilot_quality_per_generation_cost_outcomes")
    save_source(summary, output, "fig18_flowpilot_quality_per_generation_cost_summary")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), layout="constrained")
    short_names = {
        "Qwen3.6-27B": "Qwen3.6",
        "Qwen3.8-27B": "Qwen3.8",
        "GPT-4o": "GPT-4o",
        "Claude Sonnet 4.6": "Sonnet 4.6",
        "Claude Opus 4.6": "Opus 4.6",
    }

    ordered = summary.set_index("model").loc[MODEL_ORDER].reset_index()
    for _, row in ordered.iterrows():
        axes[0].errorbar(
            row["mean_generation_cost_usd"],
            row["mean_score"],
            xerr=row["generation_cost_sd_usd"],
            yerr=row["score_sd"],
            fmt="o",
            markersize=8,
            capsize=3,
            color=MODEL_COLORS[row["model"]],
            ecolor=MODEL_COLORS[row["model"]],
            alpha=0.95,
        )
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.84, 0.98)
    axes[0].set_xlabel("Mean FlowPilot generation cost per outcome (USD, log scale)")
    axes[0].set_ylabel("Benchmark score")
    axes[0].set_title("A  FlowPilot quality and generation cost")
    axes[0].grid(alpha=0.18)
    axes[0].legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=MODEL_COLORS[model], markeredgecolor="white",
                label=short_names[model], markersize=7,
            )
            for model in MODEL_ORDER
        ],
        frameon=False,
        loc="lower right",
        title="Generator model",
    )

    ranked = summary.sort_values("aggregate_quality_per_usd").copy()
    y = np.arange(len(ranked))
    bars = axes[1].barh(
        y,
        ranked["aggregate_quality_per_usd"],
        color=[MODEL_COLORS[model] for model in ranked["model"]],
    )
    for bar, value in zip(bars, ranked["aggregate_quality_per_usd"]):
        axes[1].text(
            value * 1.04,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=9.2,
        )
    axes[1].set_yticks(y, ranked["model"])
    color_model_tick_labels(axes[1])
    axes[1].set_xscale("log")
    axes[1].set_xlabel(
        "FlowPilot quality-per-cost index (benchmark score / USD, log scale)"
    )
    axes[1].set_title("B  FlowPilot cost-efficiency ranking")
    axes[1].grid(axis="x", alpha=0.18)
    save_figure(fig, output, "fig18_flowpilot_quality_per_generation_cost")


def _runtime_seconds(run_directory: str) -> float:
    path = Path(run_directory) / "run_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("runtime_total_s")
    if value is None:
        value = payload.get("runtime_s")
    if not isinstance(value, (int, float)) or value <= 0:
        raise RuntimeError(f"Missing measured runtime in {path}")
    return float(value)


def flowpilot_campaign_efficiency_tables(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes, _ = cost_tables(data)
    outcomes = outcomes[outcomes["architecture"] == "FlowPilot"].copy()
    outcomes["total_tokens"] = outcomes["input_tokens"] + outcomes["output_tokens"]
    outcomes["runtime_s"] = outcomes["run_directory"].map(_runtime_seconds)
    outcomes["runtime_min"] = outcomes["runtime_s"] / 60.0
    outcomes["campaign_quality_per_usd"] = (
        outcomes["mean_score_0_1"] / outcomes["generation_cost_usd"]
    )

    overall = (
        outcomes.groupby("model")
        .agg(
            mean_total_tokens=("total_tokens", "mean"),
            total_tokens_sd=("total_tokens", "std"),
            mean_generation_cost_usd=("generation_cost_usd", "mean"),
            generation_cost_sd_usd=("generation_cost_usd", "std"),
            mean_runtime_min=("runtime_min", "mean"),
            runtime_sd_min=("runtime_min", "std"),
            mean_benchmark_score=("mean_score_0_1", "mean"),
            n_campaigns=("candidate_id", "count"),
        )
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    by_case = (
        outcomes.groupby(["case", "model"])
        .agg(
            mean_quality_per_usd=("campaign_quality_per_usd", "mean"),
            quality_per_usd_sd=("campaign_quality_per_usd", "std"),
            mean_benchmark_score=("mean_score_0_1", "mean"),
            mean_generation_cost_usd=("generation_cost_usd", "mean"),
            n_repeats=("candidate_id", "count"),
        )
        .reset_index()
    )
    return outcomes, overall, by_case


def plot_flowpilot_campaign_efficiency(
    data: dict[str, pd.DataFrame], output: Path
) -> None:
    outcomes, overall, by_case = flowpilot_campaign_efficiency_tables(data)
    save_source(outcomes, output, "fig18a_flowpilot_campaign_efficiency_outcomes")
    save_source(overall, output, "fig18a_flowpilot_campaign_resource_summary")
    save_source(by_case, output, "fig18a_flowpilot_case_quality_per_cost")

    fig = plt.figure(figsize=(24.0, 13.4))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.105,
        right=0.99,
        top=0.91,
        bottom=0.11,
        wspace=0.72,
        hspace=0.56,
        height_ratios=[0.9, 1.1],
    )
    top_axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    bottom_axes = [fig.add_subplot(grid[1, column]) for column in range(3)]
    y = np.arange(len(MODEL_ORDER))
    model_colors = [MODEL_COLORS[model] for model in MODEL_ORDER]
    resource_model_labels = [
        "Qwen3.6",
        "Qwen3.8",
        "GPT-4o",
        "Sonnet 4.6",
        "Opus 4.6",
    ]

    top_specs = [
        (
            "mean_total_tokens",
            "total_tokens_sd",
            "A  Token use",
            "Total tokens per campaign",
            lambda value: f"{value / 1000:.1f}k",
        ),
        (
            "mean_generation_cost_usd",
            "generation_cost_sd_usd",
            "B  Generation cost",
            "Generation cost per campaign (USD)",
            lambda value: f"${value:.3f}",
        ),
        (
            "mean_runtime_min",
            "runtime_sd_min",
            "C  Observed runtime",
            "Wall-clock time per campaign (min)",
            lambda value: f"{value:.1f}",
        ),
    ]
    for ax, spec in zip(top_axes, top_specs):
        mean_col, sd_col, title, xlabel, formatter = spec
        values = overall[mean_col].to_numpy()
        errors = overall[sd_col].fillna(0).to_numpy()
        bars = ax.barh(
            y,
            values,
            xerr=errors,
            color=model_colors,
            alpha=0.92,
            capsize=3,
            error_kw={"ecolor": "#4B555C", "elinewidth": 1.0},
        )
        upper = values + errors
        x_limit = max(upper) * 1.28
        ax.set_xlim(0, x_limit)
        for bar, value, edge in zip(bars, values, upper):
            ax.text(
                edge + x_limit * 0.018,
                bar.get_y() + bar.get_height() / 2,
                formatter(value),
                va="center",
                fontsize=11.0,
            )
        ax.set_yticks(y, resource_model_labels)
        for label in ax.get_yticklabels():
            label.set_color("#111111")
            label.set_fontweight("semibold")
            label.set_fontsize(12.0)
        ax.tick_params(axis="x", labelsize=11.0)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold", fontsize=14.5)
        ax.set_xlabel(xlabel, fontsize=12.5)
        ax.grid(axis="x", alpha=0.18)
        if ax is top_axes[0]:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value / 1000:.0f}k")
            )

    panel_order = ["CuAAC", "Photochemical oxidation", "Hydrogenolysis"]
    for axis_index, (ax, case) in enumerate(zip(bottom_axes, panel_order)):
        rows = by_case[by_case["case"] == case].set_index("model").loc[MODEL_ORDER]
        values = rows["mean_quality_per_usd"].to_numpy()
        errors = rows["quality_per_usd_sd"].fillna(0).to_numpy()
        for row_index, model in enumerate(MODEL_ORDER):
            ax.errorbar(
                values[row_index],
                row_index,
                xerr=errors[row_index],
                fmt="o",
                markersize=7,
                capsize=3,
                color=MODEL_COLORS[model],
                ecolor=MODEL_COLORS[model],
                linewidth=1.3,
            )
            label_x = (values[row_index] + errors[row_index]) * 1.10
            ax.text(
                label_x,
                row_index,
                f"{values[row_index]:.1f}",
                va="center",
                fontsize=11.0,
            )
        ax.set_yticks(y, resource_model_labels)
        for label in ax.get_yticklabels():
            label.set_color("#111111")
            label.set_fontweight("semibold")
            label.set_fontsize(12.0)
        ax.tick_params(axis="x", labelsize=11.0)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(0.75, 90)
        ax.set_title(
            f"D{axis_index + 1}  {case}",
            loc="left",
            fontweight="bold",
            fontsize=14.5,
        )
        ax.grid(axis="x", which="both", alpha=0.18)

    fig.suptitle(
        "FlowPilot campaign resource use and chemistry-specific cost efficiency",
        fontsize=17.0,
        y=0.975,
    )
    fig.text(
        0.565,
        0.055,
        "FlowPilot quality-per-cost index (benchmark score / USD, log scale)",
        ha="center",
        fontsize=13.0,
    )
    fig.text(
        0.14,
        0.015,
        "Panels A-C: mean +/- SD across nine FlowPilot campaigns per model "
        "(three cases x three repeats). Panels D1-D3: mean +/- SD across three "
        "campaign repeats. Runtime is observed wall-clock time and is environment-dependent.",
        fontsize=10.2,
        color="#59636B",
    )
    save_figure(fig, output, "fig18a_flowpilot_campaign_cost_efficiency")


def write_documentation(output: Path) -> None:
    captions = """# Revised figure captions

**Figure 1. Matched model and architecture outcome quality.** One-shot and FlowPilot benchmark scores are connected within each generator model. Points are means and error bars are sample standard deviations across three generation-repeat campaign means.

**Figure 1b. Ranked model-case-architecture outcomes.** Every retained model, chemistry case, and architecture combination is ordered by mean benchmark score from lowest to highest. Error bars are sample standard deviations across three generation repeats.

**Figure 1c. Ranked outcomes separated by chemistry case.** The same model-architecture outcomes as Figure 1b are separated into CuAAC, photochemical oxidation, and hydrogenolysis panels. Each panel is independently ordered from lowest to highest. Points are means and error bars are sample standard deviations across three generation repeats.

**Figure 4. Case-specific architecture effects.** Mean paired FlowPilot-minus-one-shot score differences for each chemistry and generator model. Error bars are sample standard deviations across three paired generation repeats.

**Figure 7. Chemistry-specific criterion-level architecture gain.** CuAAC, photochemical oxidation, and hydrogenolysis are shown in three equal vertically stacked heatmaps. Each cell reports the mean paired FlowPilot-minus-one-shot benchmark score difference across three generation repeats. Red indicates deterioration, white no change, and green improvement. All panels share one zero-centered color normalization. UO-08 and UO-09 are excluded from this chemistry-comparison figure.

**Figure 8. Critical-error burden.** Judge critical flags summed within each three-case repeat. Bars are repeat means and error bars are sample standard deviations across three repeats.

**Figure 8 companion campaign error register.** The accompanying Excel workbook and Markdown report list all 90 retained campaigns. The campaign summary gives each campaign's benchmark score and plain-English critical-error categories. The error-detail sheet consolidates duplicate judge flags at the campaign-criterion level and preserves campaign-specific findings, observed evidence, and recommended corrections. The judge-evidence sheet retains every raw critical flag for auditability.

**Figure 8a. Campaign-level critical error map.** One-shot and FlowPilot campaigns are shown separately. Each row is one model, chemistry, and generation-repeat campaign; columns are the plain-English error criteria that received at least one critical flag. White indicates no critical error, gray indicates that the criterion was not applicable, and red intensity and the printed number indicate how many independent judges flagged the same campaign-criterion error. The rightmost total counts distinct error criteria and does not count duplicate judge agreement as additional errors.

**Figure 11. Selected internal architecture conditions.** Qwen3.8 outcomes for one-shot, no-council, specialist-agent removals, and full FlowPilot, ordered by mean score. Error bars are between-case standard deviations across three matched cases.

**Figure 17. Outcome quality per generation cost.** Panel A shows mean benchmark score against measured generation cost per outcome. Panel B reports the quality-per-cost index, defined as mean benchmark score divided by mean generation cost in USD. Token prices were supplied by the user; judge/evaluation costs are excluded.

**Figure 18. FlowPilot-only quality per generation cost.** Figure 17 is restricted to full FlowPilot outcomes. Panel A shows mean FlowPilot quality and measured generation cost with standard deviations across nine outcomes. Panel B ranks generator models by the FlowPilot quality-per-cost index. One-shot outcomes are excluded.

**Figure 18a. FlowPilot campaign resource use and chemistry-specific cost efficiency.** Panels A-C show mean total token use, generation cost, and observed wall-clock runtime across nine FlowPilot campaigns per generator model, with sample standard deviations. Panels D1-D3 show the mean campaign-level quality-per-cost index separately for CuAAC, photochemical oxidation, and hydrogenolysis, with sample standard deviations across three repeats. One-shot outcomes are excluded. Runtime reflects the observed benchmark environment and is not a hardware-normalized latency comparison.
"""
    methods = """# Quality-per-cost method

Generation cost is calculated independently for every benchmark outcome:

`cost_USD = (input_tokens * input_price_per_1M + output_tokens * output_price_per_1M) / 1,000,000`

The plotted aggregate index is:

`quality_per_cost = mean_consensus_score / mean_generation_cost_USD`

For Figure 18a, the campaign-level index is calculated before aggregation:

`campaign_quality_per_cost = campaign_benchmark_score / campaign_generation_cost_USD`

Panels D1-D3 report the mean and sample standard deviation of this campaign-level index across three repeats within each chemistry.

Measured token counts come from each generation run's `run_summary.json`. The index includes all LLM calls made to generate the outcome: one call for one-shot or the complete FlowPilot generation pipeline. It excludes the three LLM judges because judging is benchmark evaluation, not design generation. Prices are treated as fixed user-supplied rates and do not include caching discounts, hosting overhead, hardware amortization, or failed attempts outside the completed run summary.

| Model | Input USD / 1M tokens | Output USD / 1M tokens |
|---|---:|---:|
| Qwen3.6-27B | 0.30 | 2.00 |
| Qwen3.8-27B | 0.35 | 2.75 |
| GPT-4o | 2.50 | 10.00 |
| Claude Sonnet 4.6 | 2.50 | 10.00 |
| Claude Opus 4.6 | 5.00 | 25.00 |

Only the five frozen publication models are included in revised model-based figures and tables.
"""
    (output / "FIGURE_CAPTIONS_REVISED.md").write_text(captions, encoding="utf-8")
    (output / "QUALITY_PER_COST_METHOD.md").write_text(methods, encoding="utf-8")


def write_manifest(output: Path) -> None:
    figures = sorted(
        path for path in (output / "figures_revised").glob("*") if path.is_file()
    )
    sources = sorted(
        path for path in (output / "source_data_revised").glob("*") if path.is_file()
    )
    tables = sorted(
        path for path in (output / "tables_revised").glob("*") if path.is_file()
    )
    manifest = {
        "schema_version": "flowpilot_manuscript_visualization_revisions_v1.0",
        "model_source": str(MODEL_SOURCE),
        "module_source": str(MODULE_SOURCE),
        "included_models": MODEL_ORDER,
        "model_scope": "five frozen publication models",
        "retained_outcome_count": 90,
        "prices_per_million_tokens_usd": PRICES_PER_MILLION,
        "quality_per_cost_formula": "mean_consensus_score / mean_generation_cost_usd",
        "figure_files": [str(path.relative_to(output)) for path in figures],
        "source_files": [str(path.relative_to(output)) for path in sources],
        "table_files": [str(path.relative_to(output)) for path in tables],
    }
    (output / "revised_package_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    revised_paths = figures + sources + tables + [
        output / "FIGURE_CAPTIONS_REVISED.md",
        output / "FIG08_CAMPAIGN_ERROR_REGISTER.md",
        output / "QUALITY_PER_COST_METHOD.md",
        output / "revised_package_manifest.json",
    ]
    for path in revised_paths:
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}")
    (output / "ARTIFACT_MANIFEST_REVISED.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def build(output: Path) -> None:
    configure_style()
    data = load_model_data()
    plot_figure_01(data, output)
    plot_sorted_case_outcomes(data, output)
    plot_sorted_outcomes_by_case(data, output)
    plot_figure_04(data, output)
    plot_figure_07(data, output)
    plot_figure_08(data, output)
    write_campaign_error_register(data, output)
    plot_campaign_error_map(data, output)
    plot_figure_11(output)
    plot_quality_per_cost(data, output)
    plot_flowpilot_quality_per_cost(data, output)
    plot_flowpilot_campaign_efficiency(data, output)
    write_documentation(output)
    write_manifest(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.output.resolve())
    print(args.output.resolve() / "figures_revised")
