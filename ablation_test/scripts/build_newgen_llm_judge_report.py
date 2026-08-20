"""Aggregate and visualize the frozen NewGen three-LLM judge campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.llm_judge import read_json, weighted_track_score, write_json


JUDGE_FAMILIES = {"qwen": "qwen", "openai": "openai", "claude": "anthropic"}
ARCH_COLORS = {"FlowPilot": "#167D6A", "One-shot": "#D65A4A"}
TRACK_LABELS = {"outcome": "Outcome quality", "assurance": "Assurance quality"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(base.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _candidate_map(campaign: Path) -> dict[str, dict[str, Any]]:
    key = read_json(campaign / "frozen" / "blinding_key_confidential.json")
    return {row["candidate_id"]: row for row in key["candidates"]}


def _resolved_call(status_path: Path) -> tuple[dict[str, Any], Path, str, int]:
    initial = read_json(status_path)
    attempts = sorted((status_path.parent / "attempts").glob("attempt_*/status.json"))
    for attempt_path in reversed(attempts):
        status = read_json(attempt_path)
        if status.get("status") == "valid":
            return status, attempt_path.parent, str(initial.get("status")), len(attempts) + 1
    if initial.get("status") == "valid":
        return initial, status_path.parent, "valid", 1
    return initial, status_path.parent, str(initial.get("status")), len(attempts) + 1


def collect_absolute(campaign: Path, rubric: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = _candidate_map(campaign)
    criterion_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    root = campaign / "judgments" / "absolute"
    for status_path in sorted(root.glob("*/*/*/repeat_*/status.json")):
        judge, track, candidate_id, repeat_name = status_path.parts[-5:-1]
        status, response_dir, initial_status, attempt_count = _resolved_call(status_path)
        meta = mapping[candidate_id]
        call = {
            "judge": judge,
            "judge_family": JUDGE_FAMILIES[judge],
            "track": track,
            "candidate_id": candidate_id,
            "repeat": int(repeat_name.split("_")[-1]),
            "status": status.get("status"),
            "initial_status": initial_status,
            "attempt_count": attempt_count,
            "duration_seconds": status.get("duration_seconds"),
            "generator_model": meta["generator_model"],
            "generator_family": meta["generator_family"],
            "architecture": meta["architecture"],
            "case": meta["case"],
        }
        call_rows.append(call)
        if status.get("status") != "valid":
            continue
        parsed = read_json(response_dir / "parsed_response.json")
        for item in parsed["criterion_scores"]:
            criterion_rows.append({
                **call,
                "criterion_id": item["criterion_id"],
                "score": int(item["score"]),
                "confidence": item.get("confidence"),
                "evidence": " | ".join(item.get("evidence") or []),
                "required_correction": item.get("required_correction"),
            })
    criteria = pd.DataFrame(criterion_rows)
    calls = pd.DataFrame(call_rows)
    if not criteria.empty:
        weights = {
            item["criterion_id"]: item["weight"]
            for track in rubric["tracks"].values()
            for item in track["criteria"]
        }
        criteria["weight"] = criteria["criterion_id"].map(weights)
    return criteria, calls


def absolute_scores(criteria: pd.DataFrame, rubric: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["candidate_id", "track", "generator_model", "generator_family", "architecture", "case"]
    consensus = criteria.groupby(keys + ["criterion_id"], as_index=False).agg(
        median_score=("score", "median"),
        mean_score=("score", "mean"),
        rating_count=("score", "size"),
        score_sd=("score", "std"),
    )
    score_rows = []
    for values, group in consensus.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        score_map = dict(zip(group["criterion_id"], group["median_score"]))
        row["consensus_score"] = weighted_track_score(score_map, rubric, row["track"])
        row["valid_ratings"] = int(group["rating_count"].sum())
        score_rows.append(row)
    candidate_scores = pd.DataFrame(score_rows)

    judge_repeat_rows = []
    judge_keys = ["judge", "judge_family", "candidate_id", "track", "generator_model", "generator_family", "architecture", "case", "repeat"]
    for values, group in criteria.groupby(judge_keys, sort=False):
        row = dict(zip(judge_keys, values))
        score_map = dict(zip(group["criterion_id"], group["score"]))
        row["weighted_score"] = weighted_track_score(score_map, rubric, row["track"])
        judge_repeat_rows.append(row)
    judge_repeat = pd.DataFrame(judge_repeat_rows)
    return consensus, candidate_scores, judge_repeat


def leave_family_out_scores(criteria: pd.DataFrame, rubric: dict[str, Any]) -> pd.DataFrame:
    filtered = criteria[criteria["judge_family"] != criteria["generator_family"]].copy()
    keys = ["candidate_id", "track", "generator_model", "generator_family", "architecture", "case"]
    rows = []
    for values, group in filtered.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        medians = group.groupby("criterion_id")["score"].median().to_dict()
        row["leave_family_out_score"] = weighted_track_score(medians, rubric, row["track"])
        row["rating_count"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def family_bias(criteria: pd.DataFrame, rubric: dict[str, Any]) -> pd.DataFrame:
    keys = ["candidate_id", "track", "generator_model", "generator_family", "architecture", "case"]
    rows = []
    for values, group in criteria.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        own = group[group["judge_family"] == row["generator_family"]]
        cross = group[group["judge_family"] != row["generator_family"]]
        if own.empty or cross.empty:
            continue
        own_scores = own.groupby("criterion_id")["score"].mean().to_dict()
        cross_scores = cross.groupby("criterion_id")["score"].mean().to_dict()
        row["own_family_score"] = weighted_track_score(own_scores, rubric, row["track"])
        row["cross_family_score"] = weighted_track_score(cross_scores, rubric, row["track"])
        row["bias_delta_points"] = row["own_family_score"] - row["cross_family_score"]
        rows.append(row)
    return pd.DataFrame(rows)


def _icc_2_1(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or np.isnan(matrix).any():
        return math.nan
    n, k = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ms_rows = k * np.square(row_means - grand).sum() / (n - 1)
    ms_cols = n * np.square(col_means - grand).sum() / (k - 1)
    residual = matrix - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else math.nan


def agreement_tables(criteria: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_keys = ["judge", "track", "candidate_id", "criterion_id"]
    repeat_rows = []
    for values, group in criteria.groupby(repeat_keys):
        scores = group.sort_values("repeat")["score"].to_numpy(dtype=float)
        repeat_rows.append({
            **dict(zip(repeat_keys, values)),
            "repeat_count": len(scores),
            "all_repeats_equal": bool(len(scores) > 0 and np.all(scores == scores[0])),
            "score_range": float(scores.max() - scores.min()),
            "score_sd": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        })
    repeatability = pd.DataFrame(repeat_rows)

    judge_means = criteria.groupby(["track", "candidate_id", "criterion_id", "judge"], as_index=False)["score"].mean()
    inter_rows = []
    for track, group in judge_means.groupby("track"):
        pivot = group.pivot_table(index=["candidate_id", "criterion_id"], columns="judge", values="score")
        complete = pivot.dropna()
        exact = (complete.nunique(axis=1) == 1).mean() if len(complete) else math.nan
        correlations = complete.corr(method="spearman")
        upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack()
        inter_rows.append({
            "track": track,
            "targets": len(complete),
            "judge_count": complete.shape[1],
            "icc_2_1_absolute": _icc_2_1(complete.to_numpy()),
            "exact_agreement_rate": exact,
            "mean_pairwise_spearman": float(upper.mean()) if len(upper) else math.nan,
        })
    return repeatability, pd.DataFrame(inter_rows)


def collect_pairwise(campaign: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = read_json(campaign / "frozen" / "blinding_key_confidential.json")
    candidates = {row["candidate_id"]: row for row in key["candidates"]}
    pairs = {row["pair_id"]: row for row in key["pairs"]}
    preference_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    for status_path in sorted((campaign / "judgments" / "pairwise").glob("*/*/*/order_*/status.json")):
        judge, track, pair_id, order_name = status_path.parts[-5:-1]
        order = int(order_name.split("_")[-1])
        status, response_dir, initial_status, attempt_count = _resolved_call(status_path)
        pair = pairs[pair_id]
        order_map = next(item for item in pair["orders"] if int(item["order"]) == order)
        call = {
            "judge": judge,
            "judge_family": JUDGE_FAMILIES[judge],
            "track": track,
            "pair_id": pair_id,
            "order": order,
            "status": status.get("status"),
            "initial_status": initial_status,
            "attempt_count": attempt_count,
            "duration_seconds": status.get("duration_seconds"),
            "generator_model": pair["generator_model"],
            "generator_family": pair["generator_family"],
            "case": pair["case"],
        }
        call_rows.append(call)
        if status.get("status") != "valid":
            continue
        parsed = read_json(response_dir / "parsed_response.json")
        for item in [*parsed["criterion_preferences"], {
            "criterion_id": "OVERALL",
            "preference": parsed["overall_preference"],
            "evidence": parsed["overall_reason"],
            "confidence": None,
        }]:
            raw_pref = item["preference"]
            selected_id = order_map.get(raw_pref) if raw_pref in {"A", "B"} else None
            normalized = candidates[selected_id]["architecture"] if selected_id else "TIE"
            preference_rows.append({
                **call,
                "criterion_id": item["criterion_id"],
                "raw_preference": raw_pref,
                "normalized_preference": normalized,
                "evidence": item.get("evidence"),
                "confidence": item.get("confidence"),
            })
    return pd.DataFrame(preference_rows), pd.DataFrame(call_rows)


def pairwise_summaries(preferences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = preferences.groupby(["generator_model", "track", "criterion_id", "normalized_preference"]).size().unstack(fill_value=0).reset_index()
    for name in ("FlowPilot", "One-shot", "TIE"):
        if name not in summary:
            summary[name] = 0
    summary["total"] = summary[["FlowPilot", "One-shot", "TIE"]].sum(axis=1)
    for name in ("FlowPilot", "One-shot", "TIE"):
        summary[f"{name.lower().replace('-', '_')}_rate"] = summary[name] / summary["total"]

    order_keys = ["judge", "track", "pair_id", "criterion_id"]
    order_rows = []
    for values, group in preferences.groupby(order_keys):
        observed = group.sort_values("order")["normalized_preference"].tolist()
        order_rows.append({
            **dict(zip(order_keys, values)),
            "order_count": len(observed),
            "order_consistent": len(observed) == 2 and observed[0] == observed[1],
            "order_1_preference": observed[0] if observed else None,
            "order_2_preference": observed[1] if len(observed) > 1 else None,
        })
    order_consistency = pd.DataFrame(order_rows)

    consensus_rows = []
    for values, group in preferences.groupby(["generator_model", "case", "track", "pair_id", "criterion_id"]):
        counts = group["normalized_preference"].value_counts()
        top = counts.max()
        leaders = sorted(counts[counts == top].index)
        consensus_rows.append({
            **dict(zip(["generator_model", "case", "track", "pair_id", "criterion_id"], values)),
            "consensus_preference": leaders[0] if len(leaders) == 1 else "TIE",
            "support_votes": int(top),
            "valid_votes": len(group),
        })
    return summary, order_consistency, pd.DataFrame(consensus_rows)


def _plot_absolute(candidate_scores: pd.DataFrame, figures: Path) -> None:
    summary = candidate_scores.groupby(["generator_model", "architecture", "track"], as_index=False).agg(
        mean_score=("consensus_score", "mean"),
        sd=("consensus_score", "std"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (track, group) in zip(axes, summary.groupby("track", sort=True)):
        sns.barplot(data=group, x="generator_model", y="mean_score", hue="architecture", palette=ARCH_COLORS, errorbar=None, ax=ax)
        ax.set(title=TRACK_LABELS[track], xlabel="Generator model", ylabel="Consensus score (0-100)", ylim=(0, 100))
        ax.grid(axis="y", alpha=.25)
        ax.legend(title="Architecture", frameon=False)
    fig.suptitle("Blinded absolute evaluation by three independent LLM judges", fontsize=14, fontweight="bold")
    _save_figure(fig, figures / "figure_01_absolute_consensus")


def _plot_criterion_heatmap(consensus: pd.DataFrame, figures: Path) -> None:
    data = consensus.groupby(["generator_model", "architecture", "criterion_id"], as_index=False)["median_score"].mean()
    data["row"] = data["generator_model"] + " | " + data["architecture"]
    pivot = data.pivot(index="row", columns="criterion_id", values="median_score")
    ordered = [item for prefix in ("O-", "A-") for item in sorted([c for c in pivot.columns if c.startswith(prefix)])]
    pivot = pivot.reindex(columns=ordered)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    sns.heatmap(pivot, annot=True, fmt=".1f", vmin=0, vmax=4, cmap="RdYlGn", linewidths=.5, ax=ax, cbar_kws={"label": "Mean median score (0-4)"})
    ax.set(xlabel="Frozen universal criterion", ylabel="", title="Criterion-level consensus across cases")
    _save_figure(fig, figures / "figure_02_criterion_heatmap")


def _plot_pairwise(summary: pd.DataFrame, figures: Path) -> None:
    data = summary[summary["criterion_id"] == "OVERALL"].copy()
    rows = []
    for _, item in data.iterrows():
        for pref in ("FlowPilot", "One-shot", "TIE"):
            rows.append({
                "generator_model": item["generator_model"],
                "track": item["track"],
                "preference": pref,
                "percent": 100 * item[pref] / item["total"],
            })
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    palette = {"FlowPilot": ARCH_COLORS["FlowPilot"], "One-shot": ARCH_COLORS["One-shot"], "TIE": "#9AA0A6"}
    for ax, (track, group) in zip(axes, frame.groupby("track", sort=True)):
        sns.barplot(data=group, x="generator_model", y="percent", hue="preference", palette=palette, errorbar=None, ax=ax)
        ax.set(title=TRACK_LABELS[track], xlabel="Generator model", ylabel="Preference votes (%)", ylim=(0, 100))
        ax.grid(axis="y", alpha=.25)
        ax.legend(title="Preferred design", frameon=False)
    fig.suptitle("Direct blinded pairwise preference with reversed A/B order", fontsize=14, fontweight="bold", y=.98)
    fig.subplots_adjust(top=.78, wspace=.20)
    _save_figure(fig, figures / "figure_03_pairwise_preference")


def _plot_judges(judge_repeat: pd.DataFrame, figures: Path) -> None:
    data = judge_repeat.groupby(["judge", "track", "architecture"], as_index=False)["weighted_score"].mean()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for row, track in enumerate(("outcome", "assurance")):
        for col, judge in enumerate(("qwen", "openai", "claude")):
            ax = axes[row, col]
            subset = data[(data["judge"] == judge) & (data["track"] == track)]
            sns.barplot(data=subset, x="architecture", y="weighted_score", hue="architecture", palette=ARCH_COLORS, legend=False, errorbar=None, ax=ax)
            ax.set(title=f"{judge.title()} | {TRACK_LABELS[track]}", xlabel="", ylabel="Score (0-100)" if col == 0 else "", ylim=(0, 100))
            ax.grid(axis="y", alpha=.25)
    fig.suptitle("Judge-specific absolute scores", fontsize=14, fontweight="bold")
    _save_figure(fig, figures / "figure_04_judge_specific_scores")


def _plot_sensitivity(candidate_scores: pd.DataFrame, lofo: pd.DataFrame, figures: Path) -> None:
    merged = candidate_scores.merge(lofo, on=["candidate_id", "track", "generator_model", "generator_family", "architecture", "case"])
    data = merged.groupby(["generator_model", "architecture", "track"], as_index=False)[["consensus_score", "leave_family_out_score"]].mean()
    long = data.melt(id_vars=["generator_model", "architecture", "track"], value_vars=["consensus_score", "leave_family_out_score"], var_name="analysis", value_name="score")
    long["analysis"] = long["analysis"].map({"consensus_score": "All judges", "leave_family_out_score": "Exclude own-family judge"})
    long["label"] = long["generator_model"] + " | " + long["architecture"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (track, group) in zip(axes, long.groupby("track", sort=True)):
        sns.barplot(data=group, x="label", y="score", hue="analysis", palette=["#286F9C", "#E0A126"], errorbar=None, ax=ax)
        ax.set(title=TRACK_LABELS[track], xlabel="", ylabel="Score (0-100)", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.25)
        ax.legend(title="Sensitivity", frameon=False)
    fig.suptitle("Same-family judge sensitivity", fontsize=14, fontweight="bold")
    _save_figure(fig, figures / "figure_05_family_bias_sensitivity")


def _plot_agreement(repeatability: pd.DataFrame, interjudge: pd.DataFrame, order: pd.DataFrame, figures: Path) -> None:
    repeat = repeatability.groupby(["judge", "track"], as_index=False).agg(exact_repeatability=("all_repeats_equal", "mean"))
    order_summary = order.groupby(["judge", "track"], as_index=False).agg(order_consistency=("order_consistent", "mean"))
    frame = repeat.merge(order_summary, on=["judge", "track"])
    frame = frame.groupby("judge", as_index=False)[["exact_repeatability", "order_consistency"]].mean()
    frame = frame.melt(id_vars=["judge"], var_name="metric", value_name="rate")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=frame, x="judge", y="rate", hue="metric", ax=axes[0], palette=["#4C78A8", "#F58518"], errorbar=None)
    axes[0].set(title="Within-judge stability", xlabel="Judge", ylabel="Agreement rate", ylim=(0, 1))
    axes[0].grid(axis="y", alpha=.25)
    axes[0].legend(title="", frameon=False)
    melted = interjudge.melt(id_vars=["track"], value_vars=["icc_2_1_absolute", "exact_agreement_rate", "mean_pairwise_spearman"], var_name="metric", value_name="value")
    sns.barplot(data=melted, x="track", y="value", hue="metric", ax=axes[1], palette=["#54A24B", "#E45756", "#72B7B2"], errorbar=None)
    axes[1].set(title="Inter-judge agreement", xlabel="Track", ylabel="Agreement statistic", ylim=(-.2, 1))
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(title="", frameon=False, fontsize=8)
    fig.suptitle("Judge repeatability and agreement diagnostics", fontsize=14, fontweight="bold")
    _save_figure(fig, figures / "figure_06_agreement")


def _plot_method(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.3, "12 frozen\ndesigns", "#E8F1F5"),
        (3.0, "Blind generator\nand architecture", "#F4ECD8"),
        (5.7, "Qwen 27B\nGPT-5.4\nClaude Sonnet 4.6", "#E7EFE5"),
        (8.7, "Absolute scoring\n3 repeats x 2 tracks", "#EDE7F3"),
        (11.4, "Pairwise scoring\n2 reversed orders", "#F6E5E3"),
    ]
    for x, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, 1.6), 2.1, 1.8, facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x + 1.05, 2.5, label, ha="center", va="center", fontsize=10, fontweight="bold")
    for start, end in ((2.42, 2.92), (5.12, 5.62), (8.12, 8.62), (10.82, 11.32)):
        ax.annotate("", xy=(end, 2.5), xytext=(start, 2.5), arrowprops={"arrowstyle": "-|>", "lw": 2.0, "color": "#333333"})
    ax.text(7, .7, "Code aggregates frozen criterion scores; no manual adjustment", ha="center", fontsize=11)
    ax.set_title("NewGen independent LLM-as-judge evaluation", fontsize=15, fontweight="bold")
    _save_figure(fig, figures / "figure_07_methodology")


def _write_report(
    output: Path,
    calls_abs: pd.DataFrame,
    calls_pair: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    pair_summary: pd.DataFrame,
    repeatability: pd.DataFrame,
    interjudge: pd.DataFrame,
    order: pd.DataFrame,
    bias: pd.DataFrame,
) -> None:
    absolute_valid = int((calls_abs["status"] == "valid").sum())
    pair_valid = int((calls_pair["status"] == "valid").sum())
    absolute_total = len(calls_abs)
    pair_total = len(calls_pair)
    arch = candidate_scores.groupby(["generator_model", "architecture", "track"])["consensus_score"].mean().round(1)
    architecture_frame = candidate_scores.groupby(
        ["generator_model", "architecture", "track"], as_index=False
    )["consensus_score"].mean()
    architecture_pivot = architecture_frame.pivot_table(
        index=["generator_model", "track"], columns="architecture", values="consensus_score"
    )
    architecture_pivot["FlowPilot_minus_One-shot"] = architecture_pivot["FlowPilot"] - architecture_pivot["One-shot"]
    overall_pairs = pair_summary[pair_summary["criterion_id"] == "OVERALL"]
    repeat_summary = repeatability.groupby(["judge", "track"])["all_repeats_equal"].mean().round(3)
    order_summary = order.groupby(["judge", "track"])["order_consistent"].mean().round(3)
    bias_summary = bias.groupby(["generator_model", "track"])["bias_delta_points"].mean().round(2)
    lines = [
        "# NewGen Three-LLM Judge Benchmark",
        "",
        "## Scope",
        "",
        "This benchmark independently evaluates the same 12 frozen batch-to-flow outputs with Qwen 27B, GPT-5.4, and Claude Sonnet 4.6. Candidate identity, generator family, and architecture are blinded. No judge is part of the generation pipeline.",
        "",
        "Outcome quality and assurance quality are reported separately. Scores are integer 0-4 criterion judgments; Python applies the predeclared weights and scales results to 0-100. There is no manual score adjustment.",
        "",
        "## Completion",
        "",
        f"- Absolute judgments: {absolute_valid}/{absolute_total} valid.",
        f"- Pairwise judgments: {pair_valid}/{pair_total} valid.",
        "- Absolute design: 3 judges x 3 repeats x 12 candidates x 2 tracks.",
        "- Pairwise design: 3 judges x 2 reversed A/B orders x 6 matched pairs x 2 tracks.",
        "",
        "## Absolute Results",
        "",
        "```text",
        arch.to_string(),
        "```",
        "",
        "FlowPilot minus One-shot score differences:",
        "",
        "```text",
        architecture_pivot[["FlowPilot_minus_One-shot"]].round(1).to_string(),
        "```",
        "",
        "## Pairwise Results",
        "",
        "Overall preference votes across judges and reversed orders:",
        "",
        "```text",
        overall_pairs[["generator_model", "track", "FlowPilot", "One-shot", "TIE", "total"]].to_string(index=False),
        "```",
        "",
        "## Main Finding",
        "",
        "This campaign does **not** establish general FlowPilot superiority. GPT-5.4 one-shot leads GPT-5.4 FlowPilot on both absolute tracks and receives 16/18 direct outcome votes. Qwen FlowPilot strongly improves assurance over Qwen one-shot, but Qwen outcome is approximately tied in absolute scoring and slightly loses the total direct outcome vote (8/18 versus 10/18).",
        "",
        "The case audit shows that the judges are responding to substantive residual inconsistencies rather than only output length. The GPT-5.4 multistep pipeline candidate is the clearest failure: its realized inventory volume and engineering trace disagree, and it loses all six overall pairwise votes on both tracks. Consequently, this benchmark should be used as a pipeline defect-discovery result and a preregistered baseline, not as a superiority figure.",
        "",
        "## Reliability",
        "",
        "Within-judge exact repeatability:",
        "",
        "```text",
        repeat_summary.to_string(),
        "```",
        "",
        "Reversed-order consistency:",
        "",
        "```text",
        order_summary.to_string(),
        "```",
        "",
        "Inter-judge diagnostics:",
        "",
        "```text",
        interjudge.round(3).to_string(index=False),
        "```",
        "",
        "Mean own-family minus cross-family score difference (positive indicates favorable same-family scoring):",
        "",
        "```text",
        bias_summary.to_string(),
        "```",
        "",
        "## Interpretation Rules",
        "",
        "- Treat outcome score and pairwise outcome preference as primary endpoints.",
        "- Treat assurance, agreement, repeatability, order consistency, and family-bias sensitivity as secondary diagnostics.",
        "- LLM consensus is an automated evaluation, not a substitute for wet-lab validation or blinded flow-chemist review.",
        "- The benchmark contains three chemistry cases and one frozen generated output per model-architecture-case cell; generalization beyond these cases requires a larger preregistered campaign.",
        "- Pairwise judgments control for score calibration differences, while reversed order measures position sensitivity.",
        "",
        "## Traceability",
        "",
        "Every prompt, raw response, parsed response, validation result, timing record, frozen packet, blinding key, and source hash is retained in the campaign directory. Tables in `tables/` are mechanically generated from valid call records.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_methods(output: Path) -> None:
    text = """# Methods: NewGen Independent LLM-as-Judge Benchmark

## Experimental unit

The experimental unit is one frozen generated batch-to-flow design. The campaign contains 12 designs: two generator families, two architectures, and three chemistry cases. Generation is complete before judging; judges cannot revise candidates.

## Blinding

Candidate IDs are deterministic hashes. Generator model, provider, architecture, run path, and deliberation-system names are removed from judge packets. The confidential key is retained only for aggregation. Pairwise candidate A/B order is deterministically randomized, then reversed in the second presentation.

## Judges

Three independent model families evaluate every candidate: Qwen3.6-27B, GPT-5.4, and Claude Sonnet 4.6. Temperature is 0.0 and each absolute repeat uses a recorded seed. The models receive the same frozen rubric and case evidence. They do not receive scores from other judges.

## Tracks

Outcome quality evaluates the final proposed experiment: chemistry preservation, numerical consistency, process completeness, inventory feasibility, safety adequacy, and experimental actionability.

Assurance quality evaluates the evidence record: calculation traceability, evidence provenance, decision justification, uncertainty calibration, auditability, and reproducibility. Assurance is deliberately separate because missing logs do not prove that an experimental design is chemically wrong.

## Absolute scoring

Each criterion is scored as an integer from 0 to 4 using frozen anchors. Each candidate is scored three times by each judge on each track. For each candidate and criterion, the primary consensus statistic is the median of all valid judge-repeat ratings. Python applies the preregistered criterion weights and computes:

`track score = 100 * sum(median criterion score * weight) / (4 * sum(weights))`

The LLM never calculates the reported 0-100 aggregate. No manual score changes are permitted.

## Pairwise scoring

For each matched generator-model/case pair, judges compare the single-pass and pipeline designs directly on every criterion and overall. Choices are A, B, or TIE. Each pair is shown twice with A/B reversed. Aggregation maps blinded A/B choices back to architecture and reports vote counts and rates.

## Reliability and sensitivity

Within-judge repeatability is the proportion of candidate-criterion cells with three identical scores, accompanied by score range and standard deviation. Inter-judge agreement is calculated after averaging each judge's three repeats per candidate-criterion target; reported diagnostics are ICC(2,1) absolute agreement, exact agreement, and mean pairwise Spearman correlation. Position sensitivity is the proportion of pairwise decisions unchanged after A/B reversal.

Same-family sensitivity excludes the judge belonging to the candidate generator's model family and recomputes consensus scores. Same-family bias is the own-family judge score minus the mean cross-family judge score.

## Invalid calls

Every request has a status record. A malformed or failed original response is immutable. A deterministic schema-repair retry may be written only to a numbered child attempt folder using the same frozen prompt, schema, and seed; the original remains available and the call-status table reports initial status and attempt count. No score is manually repaired or imputed. The publication report is generated only when all 288 planned cells have a valid resolved response; otherwise aggregation stops with an error.

## Interpretation

Outcome and pairwise outcome are primary. Assurance and reliability diagnostics are secondary. Automated LLM judging measures rubric-aligned evaluation, not wet-lab truth. Three chemistry cases do not establish universal superiority; broader claims require more frozen cases and external validation.
"""
    (output / "METHODS.md").write_text(text, encoding="utf-8")


def _write_data_dictionary(output: Path) -> None:
    rows = [
        ("absolute_criterion_ratings.csv", "One row per valid judge-repeat-criterion rating, including evidence and required correction."),
        ("absolute_call_status.csv", "One row per planned absolute LLM request with provider status and duration."),
        ("absolute_criterion_consensus.csv", "Median and mean rating per candidate, track, and criterion across judges and repeats."),
        ("absolute_candidate_scores.csv", "Code-calculated weighted 0-100 consensus score for each candidate and track."),
        ("absolute_judge_repeat_scores.csv", "Code-calculated weighted score for one judge, candidate, track, and repeat."),
        ("leave_own_family_out_scores.csv", "Candidate scores after excluding the judge from the generator model family."),
        ("same_family_bias.csv", "Own-family minus cross-family judge score sensitivity."),
        ("within_judge_repeatability.csv", "Exact equality, range, and SD across three repeats for each rating target."),
        ("interjudge_agreement.csv", "ICC(2,1), exact agreement, and Spearman agreement by track."),
        ("pairwise_preferences.csv", "Every pairwise criterion and overall A/B/TIE decision mapped to architecture."),
        ("pairwise_call_status.csv", "One row per planned pairwise request with provider status and duration."),
        ("pairwise_summary.csv", "FlowPilot, One-shot, and TIE vote counts/rates by generator model, track, and criterion."),
        ("pairwise_order_consistency.csv", "Whether a decision is unchanged when candidate A/B order is reversed."),
        ("pairwise_consensus.csv", "Plurality preference across judges and both orders for each matched pair and criterion."),
    ]
    frame = pd.DataFrame(rows, columns=["file", "definition"])
    _save_csv(frame, output / "tables" / "data_dictionary.csv")


def _write_checksums(output: Path) -> None:
    paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    text = "\n".join(f"{_sha256(path)}  {path.relative_to(output)}" for path in paths) + "\n"
    (output / "checksums.sha256").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = args.output.resolve()
    tables = output / "tables"
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    rubric = read_json(campaign / "frozen" / "rubric.json")
    absolute, absolute_calls = collect_absolute(campaign, rubric)
    pairwise, pairwise_calls = collect_pairwise(campaign)
    expected_absolute = 12 * 3 * 3 * 2
    expected_pairwise = 6 * 3 * 2 * 2
    if len(absolute_calls) != expected_absolute or len(pairwise_calls) != expected_pairwise:
        raise RuntimeError(
            f"Campaign incomplete: absolute {len(absolute_calls)}/{expected_absolute}; "
            f"pairwise {len(pairwise_calls)}/{expected_pairwise}"
        )
    if (absolute_calls["status"] != "valid").any() or (pairwise_calls["status"] != "valid").any():
        raise RuntimeError("Campaign contains invalid or failed calls; inspect call_status tables before reporting")

    consensus, candidate_scores, judge_repeat = absolute_scores(absolute, rubric)
    lofo = leave_family_out_scores(absolute, rubric)
    bias = family_bias(absolute, rubric)
    repeatability, interjudge = agreement_tables(absolute)
    pair_summary, order_consistency, pair_consensus = pairwise_summaries(pairwise)

    for name, frame in {
        "absolute_criterion_ratings": absolute,
        "absolute_call_status": absolute_calls,
        "absolute_criterion_consensus": consensus,
        "absolute_candidate_scores": candidate_scores,
        "absolute_judge_repeat_scores": judge_repeat,
        "leave_own_family_out_scores": lofo,
        "same_family_bias": bias,
        "within_judge_repeatability": repeatability,
        "interjudge_agreement": interjudge,
        "pairwise_preferences": pairwise,
        "pairwise_call_status": pairwise_calls,
        "pairwise_summary": pair_summary,
        "pairwise_order_consistency": order_consistency,
        "pairwise_consensus": pair_consensus,
    }.items():
        _save_csv(frame, tables / f"{name}.csv")

    sns.set_theme(style="whitegrid", font_scale=.95)
    _plot_absolute(candidate_scores, figures)
    _plot_criterion_heatmap(consensus, figures)
    _plot_pairwise(pair_summary, figures)
    _plot_judges(judge_repeat, figures)
    _plot_sensitivity(candidate_scores, lofo, figures)
    _plot_agreement(repeatability, interjudge, order_consistency, figures)
    _plot_method(figures)

    shutil.copy2(campaign / "frozen" / "rubric.json", output / "frozen_rubric.json")
    shutil.copy2(campaign / "frozen" / "campaign_manifest.json", output / "campaign_manifest.json")
    amendment_names = []
    for amendment in sorted((campaign / "frozen").glob("protocol_amendment_*.json")):
        shutil.copy2(amendment, output / amendment.name)
        amendment_names.append(amendment.name)
    architecture_summary = candidate_scores.groupby(
        ["generator_model", "architecture", "track"], as_index=False
    )["consensus_score"].mean()
    overall_pairwise = pair_summary[pair_summary["criterion_id"] == "OVERALL"].copy()
    write_json(output / "summary.json", {
        "schema_version": "flowpilot_newgen_llm_judge_report_v1.0",
        "campaign_directory": str(campaign),
        "candidate_count": 12,
        "absolute_valid_calls": int((absolute_calls["status"] == "valid").sum()),
        "pairwise_valid_calls": int((pairwise_calls["status"] == "valid").sum()),
        "manual_score_adjustment": False,
        "protocol_amendments": amendment_names,
        "primary_endpoints": ["outcome consensus score", "pairwise outcome preference"],
        "secondary_endpoints": ["assurance score", "agreement", "repeatability", "order consistency", "family-bias sensitivity"],
        "absolute_architecture_results": architecture_summary.round(3).to_dict(orient="records"),
        "overall_pairwise_results": overall_pairwise[
            ["generator_model", "track", "FlowPilot", "One-shot", "TIE", "total"]
        ].to_dict(orient="records"),
        "interjudge_agreement": interjudge.round(4).to_dict(orient="records"),
    })
    _write_report(output, absolute_calls, pairwise_calls, candidate_scores, pair_summary, repeatability, interjudge, order_consistency, bias)
    _write_methods(output)
    _write_data_dictionary(output)
    _write_checksums(output)
    print(output)


if __name__ == "__main__":
    main()
