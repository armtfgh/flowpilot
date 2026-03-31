"""FLORA-Translate — Main entry point.

Pipeline:
  BatchRecord
    → ChemistryAgent          (Layer 1 — pure chemistry analysis)
    → plan-aware Retrieval    (Layer 2 — better semantic search)
    → TranslationLLM          (Layer 2 — proposal grounded in plan + analogies)
    → ENGINE + ChemValidator  (Layer 3 — engineering + chemistry validation)
    → ProcessTopology builder (converts validated proposal → chemistry-aware diagram)
    → FlowsheetBuilder        (SVG/PNG diagram with actual chemical names)
    → OutputFormatter
"""

import json
import logging
import sys
from pathlib import Path

from flora_translate.analogy_selector import AnalogySelector
from flora_translate.chemistry_agent import ChemistryReasoningAgent
from flora_translate.config import LAB_INVENTORY_PATH, RECORDS_DIR
from flora_translate.engine.moderator import Moderator
from flora_translate.input_parser import InputParser
from flora_translate.output_formatter import OutputFormatter
from flora_translate.prompt_builder import TranslationPromptBuilder
from flora_translate.retriever import VectorRetriever
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    LabInventory,
    ProcessStage,
    ProcessTopology,
    StreamConnection,
    UnitOperation,
)
from flora_translate.translation_llm import TranslationLLM
from flora_translate.vector_store import VectorStore

logger = logging.getLogger("flora.translate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Chemistry-aware topology builder
# ---------------------------------------------------------------------------

def _build_translate_topology(
    proposal: FlowProposal,
    chemistry_plan: ChemistryPlan | None,
    batch_record: BatchRecord,
) -> ProcessTopology:
    """Convert a FlowProposal + ChemistryPlan into a ProcessTopology.

    Handles both single-step and multi-step processes:
    - Single-step: pumps → mixer → [degas] → reactor → [BPR] → [quench] → collector
    - Multi-step: each stage gets its own reactor zone with mid-stream
      injection points, inter-stage quench/filter/solvent-switch operations.
    """
    # Dispatch: multi-step if ChemistryPlan has stages
    if chemistry_plan and chemistry_plan.stages and chemistry_plan.n_stages > 1:
        return _build_multistep_topology(proposal, chemistry_plan, batch_record)
    return _build_singlestep_topology(proposal, chemistry_plan, batch_record)


def _connect(ops, streams, counter, from_op, to_op, label=""):
    """Helper: add a stream connection."""
    counter[0] += 1
    streams.append(StreamConnection(
        stream_id=f"s{counter[0]}", from_op=from_op,
        to_op=to_op, stream_type="liquid", label=label
    ))


def _add_pump(ops, label_char, role, contents, solvent, flow_rate, reasoning):
    """Helper: create a pump UnitOperation."""
    op_id = f"pump_{label_char.lower()}"
    ops.append(UnitOperation(
        op_id=op_id, op_type="pump",
        label=f"Pump {label_char} — {role}",
        parameters={
            "stream": label_char,
            "contents": contents,
            "solvent": solvent,
            "flow_rate_mL_min": flow_rate,
        },
        required=True, rationale=reasoning,
    ))
    return op_id


def _build_singlestep_topology(proposal, chemistry_plan, batch_record):
    """Linear single-step topology (original logic, preserved)."""
    import math

    ops: list[UnitOperation] = []
    streams: list[StreamConnection] = []
    sc = [0]

    is_photochem = proposal.wavelength_nm is not None

    # Pumps
    pump_ids = []
    if proposal.streams:
        for s in proposal.streams:
            pid = _add_pump(ops, s.stream_label, s.pump_role,
                            s.contents, s.solvent, s.flow_rate_mL_min, s.reasoning or "")
            pump_ids.append(pid)
    else:
        for lbl in ("A", "B"):
            pid = _add_pump(ops, lbl, "reagent", [], "", None, "")
            pump_ids.append(pid)

    # Mixer
    ops.append(UnitOperation(op_id="mixer_1", op_type="mixer",
        label=proposal.mixer_type or "T-Mixer",
        parameters={"type": proposal.mixer_type or "T-mixer", "material": "PEEK"},
        required=True, rationale=proposal.mixing_order_reasoning or "Combine streams"))
    for pid in pump_ids:
        _connect(ops, streams, sc, pid, "mixer_1")
    prev = "mixer_1"

    # Deoxygenation
    deoxy = proposal.deoxygenation_method
    if not deoxy and chemistry_plan and chemistry_plan.deoxygenation_required:
        deoxy = "N2 sparging"
    if deoxy:
        ops.append(UnitOperation(op_id="deoxy_1", op_type="deoxygenation_unit",
            label="Inline Deoxygenation", parameters={"method": deoxy},
            required=True, rationale=chemistry_plan.deoxygenation_reasoning if chemistry_plan else ""))
        _connect(ops, streams, sc, prev, "deoxy_1")
        prev = "deoxy_1"

    # Reactor
    mat = proposal.tubing_material
    reactor_label = f"{mat} Photoreactor Coil" if is_photochem else f"{mat} Flow Reactor Coil"
    vol = proposal.reactor_volume_mL
    id_mm = proposal.tubing_ID_mm
    length = round((vol * 1e-6) / (math.pi * (id_mm * 5e-4) ** 2), 2) if vol and id_mm else None
    ops.append(UnitOperation(op_id="reactor_1", op_type="coil_reactor",
        label=reactor_label, parameters={
            "material": mat, "ID_mm": id_mm, "volume_mL": vol,
            "temperature_C": proposal.temperature_C, "wavelength_nm": proposal.wavelength_nm,
            "length_m": length,
        }, required=True, rationale="Flow reactor"))
    _connect(ops, streams, sc, prev, "reactor_1")
    prev = "reactor_1"

    # LED
    if is_photochem:
        ops.append(UnitOperation(op_id="led_1", op_type="led_module",
            label=f"LED {proposal.wavelength_nm:.0f} nm",
            parameters={"wavelength_nm": proposal.wavelength_nm},
            required=True, rationale="Photoexcitation"))

    # BPR
    if proposal.BPR_bar and proposal.BPR_bar > 0:
        ops.append(UnitOperation(op_id="bpr_1", op_type="bpr",
            label="BPR", parameters={"pressure_bar": proposal.BPR_bar},
            required=True, rationale="Maintain liquid phase"))
        _connect(ops, streams, sc, prev, "bpr_1")
        prev = "bpr_1"

    # Quench
    if chemistry_plan and chemistry_plan.quench_required:
        ops.append(UnitOperation(op_id="quench_1", op_type="quench_mixer",
            label="Inline Quench",
            parameters={"reagent": chemistry_plan.quench_reagent or "TBD"},
            required=True, rationale=chemistry_plan.quench_reasoning or ""))
        _connect(ops, streams, sc, prev, "quench_1")
        prev = "quench_1"

    # Collector
    ops.append(UnitOperation(op_id="collector_1", op_type="collector",
        label="Product Collection", parameters={}, required=True, rationale="Outlet"))
    _connect(ops, streams, sc, prev, "collector_1")

    pid_parts = [o.label for o in ops if o.op_type != "led_module"]
    return ProcessTopology(
        topology_id="translate", unit_operations=ops, streams=streams,
        total_flow_rate_mL_min=proposal.flow_rate_mL_min,
        residence_time_min=proposal.residence_time_min,
        reactor_volume_mL=proposal.reactor_volume_mL,
        pid_description=" → ".join(pid_parts),
        topology_confidence=proposal.confidence,
    )


def _build_multistep_topology(proposal, chemistry_plan, batch_record):
    """Graph-based multi-step topology.

    Each stage in chemistry_plan.stages becomes:
      [new feed pumps] → [mixer with previous outlet] → [pre-stage ops] →
      [reactor for this stage] → [post-stage ops (quench/filter/solvent switch)]
      → feeds into next stage

    The result is a sequential process graph with mid-stream injection points.
    """
    import math

    ops: list[UnitOperation] = []
    streams: list[StreamConnection] = []
    sc = [0]
    pump_char_counter = [0]

    def next_pump_char():
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        c = chars[pump_char_counter[0] % 26]
        pump_char_counter[0] += 1
        return c

    prev_op = None  # outlet of the previous stage

    for stage in chemistry_plan.stages:
        sn = stage.stage_number
        prefix = f"st{sn}"

        # ── Feed pumps for this stage ──────────────────────────────────────
        stage_pump_ids = []
        for feed in stage.feed_streams:
            char = feed.stream_label or next_pump_char()
            contents = feed.reagents if feed.reagents else []
            pid = _add_pump(ops, char, feed.reasoning or f"Stage {sn} feed",
                            contents, stage.solvent, None,
                            feed.reasoning or f"Feed for {stage.stage_name}")
            stage_pump_ids.append(pid)

        # ── Mixer: combine new feeds + previous stage outlet ───────────────
        mixer_id = f"{prefix}_mixer"
        mixer_inputs = stage_pump_ids[:]
        if prev_op:
            mixer_inputs.append(prev_op)

        if len(mixer_inputs) > 1:
            ops.append(UnitOperation(
                op_id=mixer_id, op_type="mixer",
                label=f"Mixer — Stage {sn}",
                parameters={"type": "T-mixer", "material": "PEEK"},
                required=True,
                rationale=f"Combine feeds for {stage.stage_name}",
            ))
            for mid in mixer_inputs:
                _connect(ops, streams, sc, mid, mixer_id, "")
            prev_op = mixer_id
        elif len(mixer_inputs) == 1:
            prev_op = mixer_inputs[0]

        # ── Pre-stage: deoxygenation if needed ─────────────────────────────
        if stage.deoxygenation_required:
            deoxy_id = f"{prefix}_deoxy"
            ops.append(UnitOperation(
                op_id=deoxy_id, op_type="deoxygenation_unit",
                label=f"Degas — Stage {sn}",
                parameters={"method": "N2 sparging"},
                required=True,
                rationale=f"O2-sensitive: {stage.stage_name}",
            ))
            _connect(ops, streams, sc, prev_op, deoxy_id)
            prev_op = deoxy_id

        # ── Reactor for this stage ─────────────────────────────────────────
        reactor_id = f"{prefix}_reactor"
        rtype = stage.reactor_type or "coil"
        op_type_map = {
            "coil": "coil_reactor", "packed_bed": "inline_filter",
            "chip": "chip_reactor", "CSTR": "coil_reactor",
        }
        is_photo = stage.requires_light
        mat = "FEP" if is_photo else ("SS" if (stage.temperature_C or 25) > 100 else "FEP")
        rlabel = f"{'Photo' if is_photo else ''}{rtype.replace('_', ' ').title()} — {stage.stage_name}"

        ops.append(UnitOperation(
            op_id=reactor_id,
            op_type=op_type_map.get(rtype, "coil_reactor"),
            label=rlabel,
            parameters={
                "material": mat,
                "temperature_C": stage.temperature_C,
                "wavelength_nm": stage.wavelength_nm if is_photo else None,
                "reactor_type": rtype,
            },
            required=True,
            rationale=f"Reactor for {stage.stage_name}",
        ))
        _connect(ops, streams, sc, prev_op, reactor_id)
        prev_op = reactor_id

        # LED if photochemical stage
        if is_photo and stage.wavelength_nm:
            led_id = f"{prefix}_led"
            ops.append(UnitOperation(
                op_id=led_id, op_type="led_module",
                label=f"LED {stage.wavelength_nm:.0f} nm",
                parameters={"wavelength_nm": stage.wavelength_nm},
                required=True, rationale=f"Light for {stage.stage_name}",
            ))

        # ── Post-stage action (quench, filter, solvent switch) ─────────────
        if stage.post_stage_action:
            action = stage.post_stage_action.lower()
            post_id = f"{prefix}_post"

            if "filter" in action:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="inline_filter",
                    label=f"Filter — after Stage {sn}",
                    parameters={"pore_size_um": 10},
                    required=True,
                    rationale=stage.post_stage_reasoning or stage.post_stage_action,
                ))
            elif "quench" in action:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="quench_mixer",
                    label=f"Quench — after Stage {sn}",
                    parameters={"reagent": stage.post_stage_action},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))
            elif "solvent" in action or "switch" in action:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="mixer",
                    label=f"Solvent switch — after Stage {sn}",
                    parameters={"type": "solvent_switch", "details": stage.post_stage_action},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))
            else:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="mixer",
                    label=f"{stage.post_stage_action[:30]}",
                    parameters={"details": stage.post_stage_action},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))
            _connect(ops, streams, sc, prev_op, post_id)
            prev_op = post_id

    # ── Final BPR (if proposal specifies) ──────────────────────────────────
    if proposal.BPR_bar and proposal.BPR_bar > 0:
        ops.append(UnitOperation(op_id="bpr_final", op_type="bpr",
            label="BPR", parameters={"pressure_bar": proposal.BPR_bar},
            required=True, rationale="Back-pressure regulation"))
        _connect(ops, streams, sc, prev_op, "bpr_final")
        prev_op = "bpr_final"

    # ── Collector ──────────────────────────────────────────────────────────
    ops.append(UnitOperation(op_id="collector_1", op_type="collector",
        label="Product Collection", parameters={}, required=True, rationale="Outlet"))
    _connect(ops, streams, sc, prev_op, "collector_1")

    # PID description
    pid_parts = [o.label for o in ops if o.op_type != "led_module"]
    return ProcessTopology(
        topology_id="translate_multistep",
        unit_operations=ops, streams=streams,
        total_flow_rate_mL_min=proposal.flow_rate_mL_min,
        residence_time_min=proposal.residence_time_min,
        reactor_volume_mL=proposal.reactor_volume_mL,
        pid_description=" → ".join(pid_parts),
        topology_confidence=proposal.confidence,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def translate(
    batch_input: str | dict,
    inventory_path: str = str(LAB_INVENTORY_PATH),
) -> dict:
    """Full FLORA-Translate pipeline.

    Returns:
        Dict with proposal, chemistry_plan, explanation, safety report,
        council messages, svg_path, png_path.
    """
    # 1. Parse input
    logger.info("Step 1: Parsing batch input")
    batch_record = InputParser().parse(batch_input)
    logger.info(f"  Parsed: {batch_record.reaction_description[:80]}...")

    # 2. Chemistry Reasoning — Layer 1
    logger.info("Step 2: Chemistry analysis (Layer 1)")
    chemistry_plan = ChemistryReasoningAgent().analyze(batch_record)
    logger.info(f"  Mechanism: {chemistry_plan.mechanism_type}")
    logger.info(f"  Streams: {len(chemistry_plan.stream_logic)}  O2: {chemistry_plan.oxygen_sensitive}")

    # 3. Plan-aware retrieval — Layer 2
    logger.info("Step 3: Retrieving literature analogies (plan-aware)")
    store = VectorStore()
    retriever = VectorRetriever(store=store)
    raw_analogies = retriever.retrieve(batch_record, top_k=3, chemistry_plan=chemistry_plan)
    analogies = AnalogySelector(records_dir=RECORDS_DIR).select(raw_analogies)
    logger.info(f"  Found {len(analogies)} analogies")

    # 4. Generate flow proposal
    logger.info("Step 4: Generating flow proposal via LLM")
    system_prompt, user_prompt = TranslationPromptBuilder().build(
        batch_record, analogies, chemistry_plan=chemistry_plan
    )
    proposal = TranslationLLM().generate(system_prompt, user_prompt)
    logger.info(f"  Proposal: {proposal.residence_time_min}min, {proposal.reactor_type}, "
                f"{len(proposal.streams)} streams")

    # 5. ENGINE validation + Chemistry Validator — Layer 3
    logger.info("Step 5: Engineering + Chemistry validation (ENGINE council)")
    inventory = LabInventory.from_json(inventory_path)
    design_candidate = Moderator().run(
        proposal, batch_record, analogies, inventory, chemistry_plan=chemistry_plan
    )

    # 6. Format output
    logger.info("Step 6: Formatting output")
    result = OutputFormatter().format(design_candidate, analogies)
    result["chemistry_plan"] = chemistry_plan.model_dump(exclude_none=True)

    # 7. Build chemistry-aware topology + generate diagram
    logger.info("Step 7: Generating process flow diagram")
    try:
        from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        svg_path = str(OUTPUT_DIR / "translate_process.svg")
        png_path = str(OUTPUT_DIR / "translate_process.png")

        topology = _build_translate_topology(
            design_candidate.proposal, chemistry_plan, batch_record
        )
        svg, png = FlowsheetBuilder().build(
            topology,
            title=batch_record.reaction_description[:70],
            output_svg=svg_path,
            output_png=png_path,
        )
        result["svg_path"] = svg
        result["png_path"] = png
        result["process_topology"] = topology.model_dump()
        logger.info(f"  Diagram saved: {svg}")
    except Exception as e:
        logger.warning(f"  Diagram generation failed: {e}")
        result["svg_path"] = ""
        result["png_path"] = ""

    logger.info(f"Done — Confidence: {result['confidence']}")
    return result


# ---------------------------------------------------------------------------
# Index helper
# ---------------------------------------------------------------------------

def index_records(records_dir: str = str(RECORDS_DIR)) -> int:
    """Index extraction results into ChromaDB."""
    logger.info(f"Indexing records from {records_dir}")
    store = VectorStore()
    count = store.index_folder(records_dir)
    logger.info(f"Indexed {count} records ({store.pairs_count} with translation pairs)")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n"
              "  python -m flora_translate.main translate '<batch_protocol>'\n"
              "  python -m flora_translate.main index [records_dir]\n")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "index":
        records_dir = sys.argv[2] if len(sys.argv) > 2 else str(RECORDS_DIR)
        index_records(records_dir)
    elif cmd == "translate":
        if len(sys.argv) < 3:
            print("Error: provide batch protocol text or JSON file path")
            sys.exit(1)
        batch_input = sys.argv[2]
        if Path(batch_input).exists():
            batch_input = Path(batch_input).read_text()
        result = translate(batch_input)
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
