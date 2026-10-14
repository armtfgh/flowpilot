#!/usr/bin/env python3
"""Build matched FlowPilot versus one-shot topology figures from frozen runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
from flora_translate.schemas import ProcessTopology, StreamConnection, UnitOperation


BENCHMARK_ROOT = ROOT / "ablation_results" / "manuscript_benchmark"
FLOWPILOT_ATLAS = (
    ROOT
    / "deliverables"
    / "manuscript_benchmark_visualizations_20260825"
    / "figures_revised"
    / "topology_atlas"
)
DEFAULT_OUTPUT = (
    ROOT
    / "deliverables"
    / "manuscript_benchmark_visualizations_20260825"
    / "figures_revised"
    / "topology_comparison_flowpilot_vs_oneshot"
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
class MatchedCampaign:
    model: str
    campaign: str
    flowpilot_variant: str
    oneshot_variant: str

    @property
    def generation_root(self) -> Path:
        return BENCHMARK_ROOT / self.campaign / "generation"


CAMPAIGNS = (
    MatchedCampaign(
        "Qwen3.6-27B",
        "manuscript_three_model_three_repeat_postfix_20260820",
        "qwen_flowpilot",
        "qwen_one_shot",
    ),
    MatchedCampaign(
        "Qwen3.8-27B",
        "qwen38_three_repeat_20260824",
        "qwen38_flowpilot",
        "qwen38_one_shot",
    ),
    MatchedCampaign(
        "GPT-4o",
        "alternative_frontier_three_repeat_20260824",
        "gpt4o_flowpilot",
        "gpt4o_one_shot",
    ),
    MatchedCampaign(
        "Claude Sonnet 4.6",
        "manuscript_three_model_three_repeat_postfix_20260820",
        "claude_flowpilot",
        "claude_one_shot",
    ),
    MatchedCampaign(
        "Claude Opus 4.6",
        "alternative_frontier_three_repeat_20260824",
        "claude_opus_flowpilot",
        "claude_opus_one_shot",
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


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        preferred = (
            "value",
            "value_mL_min",
            "flow_rate_mL_min",
            "gas_flow_actual_mL_min",
            "gas_flow_sccm",
        )
        for key in preferred:
            if key in value:
                parsed = as_float(value[key])
                if parsed is not None:
                    return parsed
        for item in value.values():
            parsed = as_float(item)
            if parsed is not None:
                return parsed
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(filter(None, (as_text(item) for item in value)))
    if isinstance(value, dict):
        preferred = (
            "name",
            "type",
            "equipment_id",
            "description",
            "step",
            "details",
        )
        pieces = [as_text(value[key]) for key in preferred if key in value]
        if not any(pieces):
            pieces = [as_text(item) for item in value.values()]
        return " ".join(filter(None, pieces))
    return str(value)


def content_names(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    names = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or item.get("compound") or as_text(item)
        else:
            name = str(item)
        if str(name).strip():
            names.append(str(name).strip())
    return names


def is_gas_stream(stream: dict[str, Any]) -> bool:
    phase = as_text(stream.get("phase")).lower()
    role = as_text(stream.get("pump_role")).lower()
    contents = " ".join(content_names(stream.get("contents") or [])).lower()
    gas_words = (
        "gas",
        "mfc",
        "air",
        "hydrogen",
        "h2",
        "nitrogen",
        "n2",
        "oxygen",
        "o2",
    )
    return phase == "gas" or any(word in role or word in contents for word in gas_words)


def declared_inline_deoxygenation(proposal: dict[str, Any]) -> str:
    method = as_text(proposal.get("deoxygenation_method"))
    lowered = method.lower()
    if not method or any(
        token in lowered
        for token in (
            "none",
            "not required",
            "not needed",
            "offline",
            "pre-degassed",
            "predegassed",
            "sparge feed",
        )
    ):
        return ""
    return method


def one_shot_topology(
    proposal: dict[str, Any], *, source_id: str
) -> tuple[ProcessTopology, list[str]]:
    """Project only explicitly declared one-shot fields into a process graph."""

    operations: list[UnitOperation] = []
    connections: list[StreamConnection] = []
    warnings: list[str] = []
    declared_streams = [
        item for item in proposal.get("streams") or [] if isinstance(item, dict)
    ]
    purge_streams = [
        stream
        for stream in declared_streams
        if "purge" in (
            as_text(stream.get("stream_label"))
            + " "
            + as_text(stream.get("pump_role"))
        ).lower()
    ]
    streams = [stream for stream in declared_streams if stream not in purge_streams]
    if purge_streams:
        warnings.append(
            f"Omitted {len(purge_streams)} explicitly purge-only utility stream(s) "
            "from the steady-state reaction topology; declarations remain in the "
            "packaged source proposal."
        )
    if not streams:
        raise ValueError(f"One-shot proposal has no structured streams: {source_id}")

    pump_ids: list[str] = []
    gas_present = False
    liquid_flow_total = 0.0
    for index, stream in enumerate(streams, start=1):
        gas = is_gas_stream(stream)
        gas_present = gas_present or gas
        operation_id = f"feed_{index}"
        label = as_text(stream.get("stream_label")) or f"Feed {index}"
        role = as_text(stream.get("pump_role")) or ("gas feed" if gas else "liquid feed")
        flow = as_float(stream.get("flow_rate_mL_min"))
        gas_sccm = as_float(stream.get("gas_flow_sccm"))
        gas_actual = as_float(stream.get("gas_flow_actual_mL_min"))
        if not gas and flow:
            liquid_flow_total += flow
        operations.append(
            UnitOperation(
                op_id=operation_id,
                op_type="mfc" if gas else "pump",
                label=f"{label} | {role}",
                parameters={
                    "stream": label,
                    "contents": content_names(stream.get("contents") or []),
                    "solvent": as_text(stream.get("solvent")),
                    "concentration_M": as_float(stream.get("concentration_M")),
                    "flow_rate_mL_min": gas_actual if gas else flow,
                    "phase": "gas" if gas else "liquid",
                    "gas_flow_sccm": gas_sccm,
                    "gas_flow_actual_mL_min": gas_actual,
                },
                assignment_status="unassigned",
                rationale="Declared by the one-shot proposal.",
            )
        )
        pump_ids.append(operation_id)

    mixer_text = as_text(proposal.get("mixer_type"))
    mixer_declared = bool(mixer_text) and mixer_text.lower() not in {
        "none",
        "no mixer",
        "n/a",
        "na",
    }
    if mixer_declared:
        operations.append(
            UnitOperation(
                op_id="mixer_1",
                op_type="mixer",
                label=mixer_text,
                parameters={"type": mixer_text},
                assignment_status="unassigned",
                rationale=as_text(proposal.get("mixing_order_reasoning")),
            )
        )
        for pump_id in pump_ids:
            connections.append(
                StreamConnection(
                    stream_id=f"s{len(connections) + 1}",
                    from_op=pump_id,
                    to_op="mixer_1",
                    stream_type=(
                        "gas" if operations[pump_ids.index(pump_id)].op_type == "mfc" else "liquid"
                    ),
                )
            )
        previous_ids = ["mixer_1"]
    else:
        previous_ids = pump_ids
        if len(pump_ids) > 1:
            warnings.append(
                "Multiple feeds were declared without a mixer; feeds are shown entering "
                "the reactor directly."
            )

    deoxygenation = declared_inline_deoxygenation(proposal)
    if deoxygenation:
        operations.append(
            UnitOperation(
                op_id="degas_1",
                op_type="degasser",
                label="Declared inline deoxygenation",
                parameters={"method": deoxygenation},
                assignment_status="unassigned",
                rationale="Declared by the one-shot proposal.",
            )
        )
        for previous in previous_ids:
            connections.append(
                StreamConnection(
                    stream_id=f"s{len(connections) + 1}",
                    from_op=previous,
                    to_op="degas_1",
                    stream_type="gas_liquid" if gas_present else "liquid",
                )
            )
        previous_ids = ["degas_1"]

    reactor_type = as_text(proposal.get("reactor_type")) or "flow reactor"
    reactor_type_lower = reactor_type.lower()
    wavelength = as_float(proposal.get("wavelength_nm"))
    if "packed" in reactor_type_lower or "bed" in reactor_type_lower:
        operation_type = "packed_bed_reactor"
        reactor_label = "One-shot packed-bed reactor"
    elif wavelength and wavelength > 0:
        operation_type = "photoreactor"
        reactor_label = "One-shot photoreactor"
    elif any(token in reactor_type_lower for token in ("micro", "chip")):
        operation_type = "chip_reactor"
        reactor_label = "One-shot microreactor"
    else:
        operation_type = "coil_reactor"
        reactor_label = "One-shot flow reactor"

    residence = as_float(proposal.get("residence_time_min")) or 0.0
    volume = as_float(proposal.get("reactor_volume_mL")) or 0.0
    temperature = as_float(proposal.get("temperature_C")) or 0.0
    operations.append(
        UnitOperation(
            op_id="reactor_1",
            op_type=operation_type,
            label=reactor_label,
            parameters={
                "material": as_text(proposal.get("tubing_material")),
                "ID_mm": as_float(proposal.get("tubing_ID_mm")),
                "volume_mL": volume,
                "Q_inlet_mL_min": liquid_flow_total
                or as_float(proposal.get("flow_rate_mL_min"))
                or 0.0,
                "temperature_C": temperature,
                "wavelength_nm": wavelength if wavelength and wavelength > 0 else None,
                "residence_time_min": residence,
                "reactor_type": reactor_type,
            },
            assignment_status="unassigned",
            rationale="Declared by the one-shot proposal.",
        )
    )
    for previous in previous_ids:
        connections.append(
            StreamConnection(
                stream_id=f"s{len(connections) + 1}",
                from_op=previous,
                to_op="reactor_1",
                stream_type="gas_liquid" if gas_present else "liquid",
            )
        )

    post_text = as_text(proposal.get("post_reactor_steps")).lower()
    post_operations: list[tuple[int, str, UnitOperation]] = []
    pressure = as_float(proposal.get("BPR_bar"))
    if pressure and pressure > 0:
        positions = [
            position
            for token in ("bpr", "back-pressure", "back pressure")
            if (position := post_text.find(token)) >= 0
        ]
        post_operations.append(
            (
                min(positions, default=0),
                "bpr_1",
                UnitOperation(
                    op_id="bpr_1",
                    op_type="bpr",
                    label="One-shot BPR",
                    parameters={"pressure_bar": pressure},
                    assignment_status="unassigned",
                    rationale="Declared by the one-shot proposal.",
                ),
            )
        )

    separator_positions = [
        position
        for token in ("phase separator", "gas-liquid separator", "separator")
        if (position := post_text.find(token)) >= 0
    ]
    if separator_positions:
        post_operations.append(
            (
                min(separator_positions),
                "separator_1",
                UnitOperation(
                    op_id="separator_1",
                    op_type="phase_separator",
                    label="One-shot phase separator",
                    parameters={
                        "phases": ["gas", "liquid"]
                        if gas_present
                        else ["liquid", "liquid"]
                    },
                    assignment_status="unassigned",
                    rationale="Declared by the one-shot post-reactor procedure.",
                ),
            )
        )

    inline_filter_position = post_text.find("inline filter")
    if inline_filter_position >= 0:
        post_operations.append(
            (
                inline_filter_position,
                "filter_1",
                UnitOperation(
                    op_id="filter_1",
                    op_type="inline_filter",
                    label="One-shot inline filter",
                    parameters={},
                    assignment_status="unassigned",
                    rationale="Declared by the one-shot post-reactor procedure.",
                ),
            )
        )

    post_operations.sort(key=lambda item: (item[0], item[1]))
    previous = "reactor_1"
    stream_phase = "gas_liquid" if gas_present else "liquid"
    for _, operation_id, operation in post_operations:
        operations.append(operation)
        connections.append(
            StreamConnection(
                stream_id=f"s{len(connections) + 1}",
                from_op=previous,
                to_op=operation_id,
                stream_type=stream_phase,
            )
        )
        previous = operation_id
        if operation_id == "separator_1":
            stream_phase = "liquid"

    operations.append(
        UnitOperation(
            op_id="collector_1",
            op_type="collector",
            label="One-shot product collection",
            parameters={},
            assignment_status="unassigned",
            rationale="Collection is declared or required to terminate the proposed flow path.",
        )
    )
    connections.append(
        StreamConnection(
            stream_id=f"s{len(connections) + 1}",
            from_op=previous,
            to_op="collector_1",
            stream_type=stream_phase,
        )
    )

    return (
        ProcessTopology(
            topology_id=f"oneshot_projection_{source_id}",
            unit_operations=operations,
            streams=connections,
            total_flow_rate_mL_min=liquid_flow_total
            or as_float(proposal.get("flow_rate_mL_min"))
            or 0.0,
            residence_time_min=residence,
            reactor_volume_mL=volume,
            pid_description=" -> ".join(operation.label for operation in operations),
            topology_confidence=as_text(proposal.get("confidence")) or "NOT_ASSESSED",
            compilation_status="one_shot_proposal_unvalidated",
        ),
        warnings,
    )


def render_gui_topology(topology: ProcessTopology, *, title: str, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_svg = output.with_suffix(".svg")
    builder = FlowsheetBuilder()
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
    if renderer != "graphviz" or not output.is_file():
        raise RuntimeError(
            f"GUI Graphviz rendering failed: renderer={renderer!r}, output={output}"
        )
    if Path(rendered_png).resolve() != output.resolve():
        raise RuntimeError(f"Renderer wrote an unexpected PNG: {rendered_png}")
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


def resized(image_path: Path, target_width: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    target_height = round(image.height * target_width / image.width)
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def compose_pair(
    flowpilot: Path,
    oneshot: Path,
    *,
    title: str,
    output: Path,
    accent: str,
    schema_valid: bool,
) -> None:
    width = 3600
    margin = 70
    image_width = width - 2 * margin
    top = resized(flowpilot, image_width)
    bottom = resized(oneshot, image_width)
    title_height = 150
    band_height = 82
    gap = 44
    height = title_height + 2 * band_height + top.height + bottom.height + 3 * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((margin, 42, margin + 24, 108), radius=7, fill=accent)
    draw.text((margin + 48, 45), title, font=load_font(44, bold=True), fill="#172033")

    y = title_height
    draw.text(
        (margin, y + 16),
        "A  FlowPilot: inventory-assigned executable topology",
        font=load_font(32, bold=True),
        fill="#176B4D",
    )
    y += band_height
    canvas.paste(top, (margin, y))
    y += top.height + gap
    validity = "schema-valid" if schema_valid else "schema-invalid envelope"
    draw.text(
        (margin, y + 16),
        f"B  One-shot: proposed topology, not inventory-validated ({validity})",
        font=load_font(32, bold=True),
        fill="#A13D45" if not schema_valid else "#8A5A16",
    )
    y += band_height
    canvas.paste(bottom, (margin, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", dpi=(300, 300), optimize=True)


def stack_pairs(pair_paths: list[Path], *, title: str, output: Path, accent: str) -> None:
    width = 3800
    margin = 70
    gap = 54
    title_height = 160
    prepared = [resized(path, width - 2 * margin) for path in pair_paths]
    height = title_height + sum(image.height + gap for image in prepared)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((margin, 44, margin + 24, 112), radius=7, fill=accent)
    draw.text((margin + 48, 48), title, font=load_font(46, bold=True), fill="#172033")
    y = title_height
    for image in prepared:
        canvas.paste(image, (margin, y))
        y += image.height + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", dpi=(300, 300), optimize=True)


def result_path(
    campaign: MatchedCampaign, case: str, repeat: str, *, oneshot: bool
) -> Path:
    variant = campaign.oneshot_variant if oneshot else campaign.flowpilot_variant
    return campaign.generation_root / case / variant / repeat / "result.json"


def flowpilot_image(campaign: MatchedCampaign, case: str, repeat: str) -> Path:
    return FLOWPILOT_ATLAS / "individual" / case / slug(campaign.model) / f"{repeat}.png"


def build(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    repeats = ("repeat_01", "repeat_02", "repeat_03")
    rows: list[dict[str, Any]] = []
    pair_index: dict[tuple[str, str, str], Path] = {}

    for campaign in CAMPAIGNS:
        model_slug = slug(campaign.model)
        for case, case_label in CASES.items():
            for repeat in repeats:
                source = result_path(campaign, case, repeat, oneshot=True)
                if not source.is_file():
                    raise FileNotFoundError(source)
                result = json.loads(source.read_text(encoding="utf-8"))
                proposal = result.get("proposal") or {}
                schema_valid = bool(result.get("schema_valid"))
                topology, warnings = one_shot_topology(
                    proposal,
                    source_id=f"{model_slug}_{case}_{repeat}",
                )

                topology_relative = (
                    Path("oneshot_topology_json") / case / model_slug / f"{repeat}.json"
                )
                proposal_relative = (
                    Path("source_proposals") / case / model_slug / f"{repeat}.json"
                )
                image_relative = (
                    Path("oneshot_individual") / case / model_slug / f"{repeat}.png"
                )
                pair_relative = (
                    Path("paired_individual") / case / model_slug / f"{repeat}.png"
                )
                topology_path = output / topology_relative
                proposal_path = output / proposal_relative
                image_path = output / image_relative
                pair_path = output / pair_relative

                topology_path.parent.mkdir(parents=True, exist_ok=True)
                proposal_path.parent.mkdir(parents=True, exist_ok=True)
                topology_payload = topology.model_dump(mode="json")
                topology_path.write_text(
                    json.dumps(topology_payload, indent=2, ensure_ascii=True),
                    encoding="utf-8",
                )
                proposal_path.write_text(
                    json.dumps(
                        {
                            "source_path": str(source.relative_to(ROOT)),
                            "schema_valid": schema_valid,
                            "proposal": proposal,
                        },
                        indent=2,
                        ensure_ascii=True,
                    ),
                    encoding="utf-8",
                )

                validity_label = "schema-valid" if schema_valid else "schema-invalid"
                renderer = render_gui_topology(
                    topology,
                    title=(
                        f"{case_label} | {campaign.model} one-shot | "
                        f"{repeat.replace('_', ' ')} | unvalidated | {validity_label}"
                    ),
                    output=image_path,
                )
                flowpilot_path = flowpilot_image(campaign, case, repeat)
                if not flowpilot_path.is_file():
                    raise FileNotFoundError(flowpilot_path)
                compose_pair(
                    flowpilot_path,
                    image_path,
                    title=(
                        f"{case_label} | {campaign.model} | "
                        f"{repeat.replace('_', ' ')}"
                    ),
                    output=pair_path,
                    accent=MODEL_COLORS[campaign.model],
                    schema_valid=schema_valid,
                )
                pair_index[(campaign.model, case, repeat)] = pair_path
                rows.append(
                    {
                        "model": campaign.model,
                        "case": case,
                        "repeat": repeat,
                        "campaign": campaign.campaign,
                        "oneshot_variant": campaign.oneshot_variant,
                        "schema_valid": schema_valid,
                        "renderer": renderer,
                        "inventory_status": "not_validated",
                        "adapter_warning_count": len(warnings),
                        "adapter_warnings": " | ".join(warnings),
                        "source_result": str(source.relative_to(ROOT)),
                        "source_proposal_sha256": canonical_hash(proposal),
                        "projected_topology_sha256": canonical_hash(topology_payload),
                        "topology_json": str(topology_relative),
                        "proposal_json": str(proposal_relative),
                        "oneshot_png": str(image_relative),
                        "paired_png": str(pair_relative),
                    }
                )

    for campaign in CAMPAIGNS:
        for case, case_label in CASES.items():
            stack_pairs(
                [pair_index[(campaign.model, case, repeat)] for repeat in repeats],
                title=f"{case_label}: {campaign.model}, three matched repeats",
                output=(
                    output
                    / "paired_repeat_comparisons"
                    / case
                    / f"{slug(campaign.model)}_three_repeats.png"
                ),
                accent=MODEL_COLORS[campaign.model],
            )

    for case, case_label in CASES.items():
        stack_pairs(
            [pair_index[(campaign.model, case, "repeat_01")] for campaign in CAMPAIGNS],
            title=f"{case_label}: FlowPilot versus one-shot, repeat 1",
            output=output / "paired_case_comparisons" / f"{case}_models_repeat_01.png",
            accent="#344054",
        )

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    schema_valid_count = sum(bool(row["schema_valid"]) for row in rows)
    coverage = {
        "schema_version": "flowpilot_paired_topology_atlas_v1.0",
        "comparison": "FlowPilot versus matched general one-shot",
        "models": [campaign.model for campaign in CAMPAIGNS],
        "cases": list(CASES),
        "repeats": list(repeats),
        "matched_pairs": len(rows),
        "one_shot_schema_valid": schema_valid_count,
        "one_shot_schema_invalid": len(rows) - schema_valid_count,
        "visual_renderer": "GUI Graphviz FlowsheetBuilder with equipment icons",
            "one_shot_projection_policy": (
                "Deterministic visualization of fields declared in the saved one-shot "
                "proposal; no second LLM call, inventory allocation, numerical repair, "
                "or hidden-reference input. Explicit startup/shutdown purge utilities "
                "are retained in source data but omitted from the steady-state graph."
            ),
    }
    (output / "coverage_summary.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        textwrap.dedent(
            """\
            # FlowPilot versus one-shot topology comparison

            This package compares the frozen FlowPilot and general one-shot outputs for
            the same five models, three chemistry cases, and three generation repeats.
            Both architectures are displayed with the GUI's icon-based Graphviz renderer.

            The one-shot model was not called again. A deterministic visualization adapter
            projects only equipment, streams, ordering, and numerical parameters declared
            in the saved one-shot proposal. It does not allocate inventory, repair numerical
            inconsistencies, add hidden-reference operations, or turn the comparison into a
            second-call workflow. Every one-shot panel is therefore labeled as not
            inventory-validated. Schema-invalid source envelopes remain identified.
            Streams explicitly declared as startup/shutdown purge utilities are retained
            in source data and manifest warnings but omitted from the steady-state graph.

            - `oneshot_individual/`: 45 one-shot topology PNGs.
            - `paired_individual/`: 45 matched FlowPilot/one-shot comparison PNGs.
            - `paired_repeat_comparisons/`: 15 three-repeat comparison sheets.
            - `paired_case_comparisons/`: three repeat-01 cross-model sheets.
            - `oneshot_topology_json/`: deterministic display topology for each one-shot run.
            - `source_proposals/`: exact saved proposal plus source path and schema flag.
            - `manifest.csv`: complete provenance and adapter warnings.

            Rebuild in the Graphviz-enabled environment:

            ```bash
            PATH=/home/amirreza/anaconda3/envs/flent/bin:$PATH \\
            /home/amirreza/anaconda3/envs/flent/bin/python \\
              ablation_test/scripts/build_paired_topology_atlas.py
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
