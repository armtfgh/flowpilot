#!/usr/bin/env python3
"""Build post-hoc audit tables and figures for the frozen 2026-08-20 campaign."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("Claude Sonnet 4.6", "GPT-5.4", "Qwen3.6-27B")
JUDGES = ("claude", "openai", "qwen")
COLORS = {"claude": "#8b6f9f", "openai": "#21818f", "qwen": "#d29343"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build(deliverable: Path) -> None:
    tables = deliverable / "tables"
    figures = deliverable / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    score_rows = _rows(tables / "judge_candidate_scores.csv")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in score_rows:
        grouped[(row["model"], row["judge"], row["architecture"])].append(
            float(row["score_0_1"])
        )

    sensitivity_rows: list[dict[str, object]] = []
    for model in MODELS:
        for judge in JUDGES:
            one_shot = np.mean(grouped[(model, judge, "One-shot")])
            flowpilot = np.mean(grouped[(model, judge, "FlowPilot")])
            sensitivity_rows.append(
                {
                    "model": model,
                    "judge": judge,
                    "one_shot_mean_0_1": round(float(one_shot), 6),
                    "flowpilot_mean_0_1": round(float(flowpilot), 6),
                    "paired_architecture_delta": round(float(flowpilot - one_shot), 6),
                }
            )
    with (tables / "judge_specific_architecture_effect.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    x = np.arange(len(MODELS))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for index, judge in enumerate(JUDGES):
        values = [
            next(
                float(row["paired_architecture_delta"])
                for row in sensitivity_rows
                if row["model"] == model and row["judge"] == judge
            )
            for model in MODELS
        ]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=f"{judge.title()} judge",
            color=COLORS[judge],
        )
        ax.bar_label(bars, labels=[f"{value:+.3f}" for value in values], padding=3, fontsize=9)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("FlowPilot minus one-shot score")
    ax.set_title("Architecture effect depends on judge family")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures / "fig08_judge_family_sensitivity.png", dpi=300, facecolor="white")
    plt.close(fig)

    contract_rows = _rows(tables / "outcome_contract_summary.csv")
    schema = {
        (row["model"], row["architecture"]): int(row["schema_valid"]) / int(row["n_outcomes"])
        for row in contract_rows
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    one_shot = [schema[(model, "One-shot")] for model in MODELS]
    flowpilot = [schema[(model, "FlowPilot")] for model in MODELS]
    bars_a = ax.bar(x - 0.18, one_shot, 0.36, label="One-shot", color="#969da5")
    bars_b = ax.bar(x + 0.18, flowpilot, 0.36, label="FlowPilot", color="#21818f")
    ax.bar_label(bars_a, labels=[f"{round(value * 9):.0f}/9" for value in one_shot], padding=3)
    ax.bar_label(bars_b, labels=[f"{round(value * 9):.0f}/9" for value in flowpilot], padding=3)
    ax.set_xticks(x, MODELS)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Required output schema validity")
    ax.set_title("Deterministic structural validity across repeated outcomes")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures / "fig09_schema_validity.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deliverable", type=Path)
    args = parser.parse_args()
    build(args.deliverable.resolve())


if __name__ == "__main__":
    main()
