#!/usr/bin/env python3
"""Render the selected NewGen 2.0 topology atlas with the GUI renderer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder
from flora_translate.schemas import ProcessTopology


BENCHMARK_ROOT = ROOT / "ablation_results" / "manuscript_benchmark"
DEFAULT_OUTPUT = (
    ROOT
    / "deliverables"
    / "manuscript_benchmark_visualizations_20260825"
    / "figures_revised"
    / "topology_atlas"
)

CASES = {
    "cuaac": "CuAAC",
    "photochemical_oxidation": "Photochemical oxidation",
    "hydrogenolysis": "Hydrogenolysis",
}

MODEL_COLORS = {
    "Qwen3.6-27B": "#376AA0",
    "Qwen3.8-27B": "#4C956C",
    "GPT-4o": "#D58B32",
    "Claude Sonnet 4.6": "#8C61A8",
    "Claude Opus 4.6": "#C55467",
}


@dataclass(frozen=True)
class ModelCampaign:
    model: str
    campaign: str
    variant: str

    @property
    def generation_root(self) -> Path:
        return BENCHMARK_ROOT / self.campaign / "generation"


CAMPAIGNS = (
    ModelCampaign(
        "Qwen3.6-27B",
        "manuscript_three_model_three_repeat_postfix_20260820",
        "qwen_flowpilot",
    ),
    ModelCampaign(
        "Qwen3.8-27B",
        "qwen38_three_repeat_20260824",
        "qwen38_flowpilot",
    ),
    ModelCampaign(
        "GPT-4o",
        "alternative_frontier_three_repeat_20260824",
        "gpt4o_flowpilot",
    ),
    ModelCampaign(
        "Claude Sonnet 4.6",
        "manuscript_three_model_three_repeat_postfix_20260820",
        "claude_flowpilot",
    ),
    ModelCampaign(
        "Claude Opus 4.6",
        "alternative_frontier_three_repeat_20260824",
        "claude_opus_flowpilot",
    ),
)


def slug(value: str) -> str:
    return (
        value.lower()
        .replace(".", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def canonical_hash(payload: dict[str, Any]) -> str:
    source = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def connected_operation_count(payload: dict[str, Any]) -> int:
    streams = list(payload.get("streams") or [])
    connected_ids = {
        str(stream[key])
        for stream in streams
        for key in ("from_op", "to_op")
        if stream.get(key)
    }
    return sum(
        str(operation.get("op_id")) in connected_ids
        for operation in payload.get("unit_operations") or []
    )


def render_topology(
    payload: dict[str, Any],
    *,
    model: str,
    case_label: str,
    repeat: str,
    output: Path,
) -> str:
    """Render through the exact icon-based builder used by the GUI."""

    topology = ProcessTopology.model_validate(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    title = f"{case_label} | {model} | FlowPilot | {repeat.replace('_', ' ')}"

    builder = FlowsheetBuilder()
    temporary_svg = output.with_suffix(".svg")
    try:
        rendered_svg, rendered_png = builder.build(
            topology,
            title=title,
            output_svg=str(temporary_svg),
            output_png=str(output),
        )
        del rendered_svg
    finally:
        temporary_svg.unlink(missing_ok=True)

    renderer = str((builder.last_render_info or {}).get("renderer") or "")
    warnings = list((builder.last_render_info or {}).get("warnings") or [])
    if renderer != "graphviz":
        raise RuntimeError(
            "The GUI Graphviz renderer was not used. "
            f"renderer={renderer!r}; warnings={warnings!r}. "
            "Run this script in the project's flent environment."
        )
    if Path(rendered_png).resolve() != output.resolve() or not output.is_file():
        raise RuntimeError(f"GUI renderer did not create the requested PNG: {output}")
    return renderer


def load_font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def stack_images(
    images: list[Path],
    *,
    title: str,
    output: Path,
    accent: str,
    width: int = 3600,
) -> None:
    title_height = 165
    gap = 44
    margin = 70
    prepared: list[Image.Image] = []
    for path in images:
        image = Image.open(path).convert("RGB")
        target_width = width - 2 * margin
        target_height = round(image.height * target_width / image.width)
        prepared.append(
            image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        )

    height = title_height + margin + sum(image.height + gap for image in prepared)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (margin, 46, margin + 24, 112), radius=7, fill=accent
    )
    draw.text(
        (margin + 48, 50),
        title,
        font=load_font(47, bold=True),
        fill="#172033",
    )
    y = title_height
    for image in prepared:
        canvas.paste(image, (margin, y))
        y += image.height + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", dpi=(300, 300), optimize=True)


def source_path(campaign: ModelCampaign, case: str, repeat: str) -> Path:
    return (
        campaign.generation_root
        / case
        / campaign.variant
        / repeat
        / "snapshots"
        / "process_topology.json"
    )


def build(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    image_index: dict[tuple[str, str, str], Path] = {}
    repeats = ("repeat_01", "repeat_02", "repeat_03")

    for campaign in CAMPAIGNS:
        model_slug = slug(campaign.model)
        for case, case_label in CASES.items():
            for repeat in repeats:
                source = source_path(campaign, case, repeat)
                if not source.is_file():
                    raise FileNotFoundError(source)
                payload = json.loads(source.read_text(encoding="utf-8"))
                if not payload.get("unit_operations") or not payload.get("streams"):
                    raise ValueError(f"Topology is not populated: {source}")
                if payload.get("compilation_status") != "inventory_assigned":
                    raise ValueError(f"Topology is not inventory-assigned: {source}")

                relative_image = Path("individual") / case / model_slug / f"{repeat}.png"
                relative_json = Path("source_json") / case / model_slug / f"{repeat}.json"
                destination_image = output / relative_image
                destination_json = output / relative_json
                renderer = render_topology(
                    payload,
                    model=campaign.model,
                    case_label=case_label,
                    repeat=repeat,
                    output=destination_image,
                )
                destination_json.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination_json)
                if canonical_hash(payload) != canonical_hash(
                    json.loads(destination_json.read_text(encoding="utf-8"))
                ):
                    raise RuntimeError(f"Copied topology hash mismatch: {source}")

                image_index[(campaign.model, case, repeat)] = destination_image
                rows.append(
                    {
                        "model": campaign.model,
                        "case": case,
                        "repeat": repeat,
                        "campaign": campaign.campaign,
                        "variant": campaign.variant,
                        "renderer": renderer,
                        "compilation_status": payload.get("compilation_status"),
                        "unit_operation_count": len(payload.get("unit_operations") or []),
                        "connected_operation_count": connected_operation_count(payload),
                        "stream_count": len(payload.get("streams") or []),
                        "topology_sha256": canonical_hash(payload),
                        "source_path": str(source.relative_to(ROOT)),
                        "packaged_json": str(relative_json),
                        "packaged_png": str(relative_image),
                    }
                )

    for case, case_label in CASES.items():
        stack_images(
            [
                image_index[(campaign.model, case, "repeat_01")]
                for campaign in CAMPAIGNS
            ],
            title=f"{case_label}: FlowPilot topology comparison (repeat 1)",
            output=output / "case_comparisons" / f"{case}_models_repeat_01.png",
            accent="#344054",
        )

        for campaign in CAMPAIGNS:
            stack_images(
                [
                    image_index[(campaign.model, case, repeat)]
                    for repeat in repeats
                ],
                title=f"{case_label}: {campaign.model} repeat topologies",
                output=(
                    output
                    / "repeat_comparisons"
                    / case
                    / f"{slug(campaign.model)}_three_repeats.png"
                ),
                accent=MODEL_COLORS[campaign.model],
            )

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    coverage = {
        "schema_version": "flowpilot_topology_atlas_v1.1",
        "scope": "selected successful FlowPilot model campaigns",
        "visual_renderer": "GUI Graphviz FlowsheetBuilder with equipment icons",
        "models": [campaign.model for campaign in CAMPAIGNS],
        "scope_note": "Selected publication models; FlowPilot outputs with canonical topology artifacts.",
        "cases": list(CASES),
        "repeats": list(repeats),
        "expected_topologies": len(CAMPAIGNS) * len(CASES) * len(repeats),
        "rendered_topologies": len(rows),
        "all_inventory_assigned": all(
            row["compilation_status"] == "inventory_assigned" for row in rows
        ),
        "all_rendered_by_gui_graphviz": all(
            row["renderer"] == "graphviz" for row in rows
        ),
    }
    (output / "coverage_summary.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        textwrap.dedent(
            """\
            # FlowPilot model topology atlas

            This PNG-only package renders the frozen, inventory-assigned process
            topologies with the same Graphviz `FlowsheetBuilder` and equipment-icon
            assets used by the FlowPilot GUI. No topology parameters are inferred or
            modified during rendering.

            - `individual/`: all model x case x repeat topology PNGs.
            - `case_comparisons/`: one repeat-01 cross-model sheet per chemistry.
            - `repeat_comparisons/`: one three-repeat sheet per model and chemistry.
            - `source_json/`: exact copies of the corresponding frozen topology JSONs.
            - `manifest.csv`: provenance, renderer, topology hashes, and artifact paths.

            One-shot runs are not included because they did not produce canonical
            process-topology artifacts.

            Rebuild with:

            ```bash
            PATH=/home/amirreza/anaconda3/envs/flent/bin:$PATH \\
            /home/amirreza/anaconda3/envs/flent/bin/python \\
              ablation_test/scripts/build_model_topology_atlas.py
            ```
            """
        ),
        encoding="utf-8",
    )

    checksums = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums.append(f"{digest}  {path.relative_to(output)}")
    (output / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    destination = arguments.output.resolve()
    build(destination)
    print(destination)
