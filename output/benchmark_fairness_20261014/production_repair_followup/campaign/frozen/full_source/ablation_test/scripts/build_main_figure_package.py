#!/usr/bin/env python3
"""Build the canonical manuscript figure folder and matching raw-data package."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "deliverables/manuscript_benchmark_visualizations_20260825"
FIGURES = PACKAGE / "figures"
REVISED = PACKAGE / "figures_revised"
SOURCE = PACKAGE / "source_data"
SOURCE_REVISED = PACKAGE / "source_data_revised"
DEFAULT_OUTPUT = REVISED / "main"


CANONICAL_FIGURES = {
    "figs5-1": "fig01c_case_panels_scores_sorted",
    "figs5-2": "fig07_criterion_gain_heatmap_revised",
    "figs5-3": "fig08a_campaign_error_map",
    "figs5-4": "fig18a_flowpilot_campaign_cost_efficiency",
}

RAW_FILES = {
    "figs5-1.csv": SOURCE_REVISED / "fig01c_case_panels_scores_sorted.csv",
    "figs5-2.csv": SOURCE_REVISED / "fig07_criterion_gain_heatmap_revised.csv",
    "figs5-3_error_map.csv": SOURCE_REVISED / "fig08a_campaign_error_map.csv",
    "figs5-3_campaign_summary.csv": SOURCE_REVISED / "fig08_campaign_error_summary.csv",
    "figs5-3_error_details.csv": SOURCE_REVISED / "fig08_campaign_error_details.csv",
    "figs5-3_judge_evidence.csv": SOURCE_REVISED / "fig08_judge_critical_evidence.csv",
    "figs5-4_campaigns.csv": SOURCE_REVISED
    / "fig18a_flowpilot_campaign_efficiency_outcomes.csv",
    "figs5-4_resource_summary.csv": SOURCE_REVISED
    / "fig18a_flowpilot_campaign_resource_summary.csv",
    "figs5-4_case_quality_per_cost.csv": SOURCE_REVISED
    / "fig18a_flowpilot_case_quality_per_cost.csv",
    "figure4_panelA_model_architecture.csv": SOURCE
    / "fig01_model_architecture_mean_sd.csv",
    "figure4_panelB_critical_flags.csv": SOURCE / "fig08_critical_flags.csv",
    "figure4_panelC_module_conditions.csv": SOURCE
    / "fig11_module_condition_scores.csv",
    "figure4_panelD_quality_cost_outcomes.csv": SOURCE_REVISED
    / "fig18_flowpilot_quality_per_generation_cost_outcomes.csv",
    "figure4_panelD_quality_cost_summary.csv": SOURCE_REVISED
    / "fig18_flowpilot_quality_per_generation_cost_summary.csv",
}


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def copy_canonical_figures(output: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for canonical_name, source_stem in CANONICAL_FIGURES.items():
        for suffix in (".png",):
            source = REVISED / f"{source_stem}{suffix}"
            require_file(source)
            destination = output / f"{canonical_name}{suffix}"
            shutil.copy2(source, destination)
            rows.append(
                {
                    "canonical_figure": canonical_name,
                    "role": "figure",
                    "source_file": str(source.relative_to(ROOT)),
                    "packaged_file": str(destination.relative_to(output)),
                }
            )
    return rows


def copy_raw_data(output: Path) -> list[dict[str, str]]:
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for destination_name, source in RAW_FILES.items():
        require_file(source)
        destination = raw / destination_name
        shutil.copy2(source, destination)
        canonical = destination_name.split("_")[0].split(".")[0]
        if destination_name.startswith("figure4"):
            canonical = "figure4"
        rows.append(
            {
                "canonical_figure": canonical,
                "role": "raw_data",
                "source_file": str(source.relative_to(ROOT)),
                "packaged_file": str(destination.relative_to(output)),
            }
        )
    return rows


def trim_white(image: Image.Image, margin: int = 18) -> Image.Image:
    array = np.asarray(image.convert("RGB"))
    mask = np.any(array < 248, axis=2)
    if not mask.any():
        return image
    rows, columns = np.where(mask)
    left = max(int(columns.min()) - margin, 0)
    upper = max(int(rows.min()) - margin, 0)
    right = min(int(columns.max()) + margin + 1, image.width)
    lower = min(int(rows.max()) + margin + 1, image.height)
    return image.crop((left, upper, right, lower))


def figure4_panels() -> list[tuple[str, Image.Image]]:
    panel_paths = [
        FIGURES / "fig01_model_architecture_mean_sd.png",
        FIGURES / "fig08_critical_flags.png",
        FIGURES / "fig11_module_condition_scores.png",
    ]
    for path in panel_paths:
        require_file(path)
    panels = [trim_white(Image.open(path).convert("RGB")) for path in panel_paths]
    return list(zip(("A", "B", "C"), panels))


def draw_figure4_panel_d(axis: plt.Axes) -> None:
    source = SOURCE_REVISED / "fig18_flowpilot_quality_per_generation_cost_summary.csv"
    require_file(source)
    frame = pd.read_csv(source)
    model_order = [
        "Qwen3.6-27B",
        "Qwen3.8-27B",
        "GPT-4o",
        "Claude Sonnet 4.6",
        "Claude Opus 4.6",
    ]
    model_colors = {
        "Qwen3.6-27B": "#376AA0",
        "Qwen3.8-27B": "#4C956C",
        "GPT-4o": "#D58B32",
        "Claude Sonnet 4.6": "#8C61A8",
        "Claude Opus 4.6": "#C55467",
    }
    rows = frame.set_index("model").loc[model_order]
    values = rows["aggregate_quality_per_usd"].to_numpy()
    y = np.arange(len(model_order))
    bars = axis.barh(
        y,
        values,
        color=[model_colors[model] for model in model_order],
    )
    for bar, value in zip(bars, values):
        axis.text(
            value * 1.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=10,
        )
    axis.set_yticks(y, model_order)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlim(0.85, 27)
    axis.set_xlabel(
        "FlowPilot quality-per-cost index (benchmark score / USD, log scale)",
        fontsize=10.5,
    )
    axis.set_title(
        "D  FlowPilot cost-efficiency ranking",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=6,
    )
    axis.grid(axis="x", which="both", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    for label, model in zip(axis.get_yticklabels(), model_order):
        label.set_color(model_colors[model])
        label.set_fontweight("semibold")
        label.set_fontsize(10.5)


def build_figure4(output: Path) -> list[dict[str, str]]:
    panels = figure4_panels()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(17.0, 11.8),
        gridspec_kw={"height_ratios": [0.92, 1.08]},
        layout="constrained",
    )
    for axis, (label, image) in zip(axes.flat[:3], panels):
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(label, loc="left", fontsize=15, fontweight="bold", pad=6)
    draw_figure4_panel_d(axes[1, 1])

    rows: list[dict[str, str]] = []
    for suffix, kwargs in {".png": {"dpi": 350}}.items():
        destination = output / f"figure4{suffix}"
        figure.savefig(destination, bbox_inches="tight", facecolor="white", **kwargs)
        rows.append(
            {
                "canonical_figure": "figure4",
                "role": "figure",
                "source_file": (
                    "figures/fig01 + figures/fig08 + figures/fig11 + "
                    "figures_revised/fig18 panel B"
                ),
                "packaged_file": str(destination.relative_to(output)),
            }
        )
    plt.close(figure)
    return rows


def write_readme(output: Path) -> None:
    text = """# Canonical manuscript figures

This folder defines the manuscript-facing names used from this release onward.
The source figures remain unchanged in their original folders.

| Canonical name | Source |
|---|---|
| `figure4` | Composite: original Figure 01, original Figure 08, original Figure 11, and revised Figure 18 panel B |
| `figs5-1` | `fig01c_case_panels_scores_sorted` |
| `figs5-2` | `fig07_criterion_gain_heatmap_revised` |
| `figs5-3` | `fig08a_campaign_error_map` |
| `figs5-4` | `fig18a_flowpilot_campaign_cost_efficiency` |

The `raw/` directory contains the plotted source tables. Figure 4 panels A and B
are copied exactly from the publication-model `figures/` package.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def write_manifest(output: Path, rows: list[dict[str, str]]) -> None:
    raw = output / "raw"
    manifest = pd.DataFrame(rows).sort_values(
        ["canonical_figure", "role", "packaged_file"]
    )
    manifest.to_csv(raw / "manifest.csv", index=False)

    checksum_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_rows.append(f"{checksum}  {path.relative_to(output)}")
    (output / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_rows) + "\n", encoding="utf-8"
    )


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.pdf", "*.svg"):
        for stale_file in output.glob(pattern):
            stale_file.unlink()
    rows = copy_canonical_figures(output)
    rows.extend(copy_raw_data(output))
    rows.extend(build_figure4(output))
    write_readme(output)
    write_manifest(output, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    destination = arguments.output.resolve()
    build(destination)
    print(destination)
