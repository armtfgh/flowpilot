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
from flora_translate.engine.council_v4 import CouncilV4
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


_GAS_PURE_ROLES = {
    "n2", "n₂", "nitrogen", "o2", "o₂", "oxygen", "co2", "co₂", "h2", "h₂",
    "hydrogen", "ar", "argon", "helium", "air", "compressed air", "mfc",
    "n2 gas", "o2 gas", "gas injection", "gas stream", "gas feed",
}
_GAS_ROLE_SUBSTRINGS = ("compressed air", "gas injection", " gas feed", "mfc stream")

def _stream_is_gas(stream) -> bool:
    """True only for genuine gas feed streams (not degassed liquid solutions)."""
    role = (getattr(stream, "pump_role", None) or "").lower().strip()
    if role in _GAS_PURE_ROLES:
        return True
    if any(sub in role for sub in _GAS_ROLE_SUBSTRINGS):
        return True
    contents = getattr(stream, "contents", None) or []
    if contents:
        _gas_kw = {"n2", "n₂", "nitrogen", "o2", "o₂", "oxygen", "co2", "co₂",
                   "h2", "h₂", "hydrogen", "ar", "argon", "helium", "air",
                   "compressed air", "mfc"}
        if all(any(kw == str(c).lower().strip() for kw in _gas_kw) for c in contents if c):
            return True
    return False


def _add_pump(ops, label_char, role, contents, solvent, flow_rate, reasoning,
              is_gas: bool = False):
    """Helper: create a pump (or MFC for gas) UnitOperation."""
    op_id = f"pump_{label_char.lower()}"
    ops.append(UnitOperation(
        op_id=op_id, op_type="mfc" if is_gas else "pump",
        label=f"{'MFC' if is_gas else 'Pump'} {label_char} — {role}",
        parameters={
            "stream": label_char,
            "contents": contents,
            "solvent": solvent,
            "flow_rate_mL_min": flow_rate,
        },
        required=True, rationale=reasoning,
    ))
    return op_id


def _stoich_flow_rates(streams_list, total_Q: float) -> list[float]:
    """Compute per-stream flow rates from molar_equiv and concentration.

    weight_i = equiv_i / conc_i  (volume of stream i needed per unit substrate volume)
    Q_i = total_Q × weight_i / Σweights

    Falls back to equal split if all equivs are 1.0 and no concentrations differ.
    Gas streams are excluded from the calculation (they get zero Q from this function).
    """
    liquid = [s for s in streams_list if not _stream_is_gas(s)]
    if not liquid:
        return [0.0] * len(streams_list)

    equivs = [max(getattr(s, "molar_equiv", 1.0) or 1.0, 1e-9) for s in liquid]
    concs  = [max(getattr(s, "concentration_M", None) or 1.0, 1e-9) for s in liquid]
    weights = [e / c for e, c in zip(equivs, concs)]
    total_w = sum(weights)
    qs_liquid = [round(total_Q * w / total_w, 4) for w in weights]

    # Re-expand to full list (gas streams get 0 here — handled separately)
    liq_iter = iter(qs_liquid)
    return [0.0 if _stream_is_gas(s) else next(liq_iter) for s in streams_list]


def _is_quench_stream(s, chemistry_plan) -> bool:
    """Classify a stream as a quench/workup stream.

    Quench streams are injected AFTER the main reactor at a separate mixer.
    They must never feed into the main reactor's T-mixer, or Pump C ends up
    plumbed to two places at once.
    """
    role = (getattr(s, "pump_role", "") or "").lower()
    if any(kw in role for kw in ("quench", "neutraliz", "workup", "post-reactor")):
        return True
    # Also match if the stream contents mention the plan's quench_reagent
    qr = (getattr(chemistry_plan, "quench_reagent", "") or "").lower() if chemistry_plan else ""
    if qr:
        contents = getattr(s, "contents", None) or []
        for c in contents:
            if qr in str(c).lower():
                return True
    return False


def _build_singlestep_topology(proposal, chemistry_plan, batch_record):
    """Linear single-step topology.

    Design contract — ONE SOURCE OF TRUTH:
      Every flow rate shown in the diagram comes from proposal.streams
      (populated by the Chief Engineer). This function does not recompute
      pump rates from stoichiometry when proposal.streams already carries
      Chief-normalised values.

    Stream classification:
      • reactor_feeds → pumps feed the main T-mixer → reactor
      • quench_streams → injected at a post-reactor Quench T-mixer
      A stream is NEVER both; this prevents Pump C appearing twice.

    Q conservation:
      Q_reactor_inlet = Σ Q_i (reactor_feeds)
      Q_quench_inlet  = Q_reactor_inlet + Σ Q_j (quench_streams)
    """
    import math

    ops: list[UnitOperation] = []
    streams: list[StreamConnection] = []
    sc = [0]

    is_photochem = proposal.wavelength_nm is not None

    # ── Classify streams ───────────────────────────────────────────────────
    all_streams = list(proposal.streams or [])
    reactor_feeds = [s for s in all_streams if not _is_quench_stream(s, chemistry_plan)]
    quench_streams = [s for s in all_streams if _is_quench_stream(s, chemistry_plan)]

    # ── Pumps for reactor feeds (only) ─────────────────────────────────────
    pump_ids = []
    if reactor_feeds:
        total_Q = proposal.flow_rate_mL_min or 0.5
        # Only compute stoichiometric split as a fallback when a stream
        # lacks an explicit flow_rate_mL_min.
        qs_fallback = _stoich_flow_rates(reactor_feeds, total_Q)
        for s, fr_fallback in zip(reactor_feeds, qs_fallback):
            fr = s.flow_rate_mL_min if s.flow_rate_mL_min else fr_fallback
            pid = _add_pump(ops, s.stream_label, s.pump_role,
                            s.contents, s.solvent, fr, s.reasoning or "",
                            is_gas=_stream_is_gas(s))
            pump_ids.append(pid)
    else:
        default_Q = round((proposal.flow_rate_mL_min or 0.5) / 2, 4)
        for lbl in ("A", "B"):
            pid = _add_pump(ops, lbl, "reagent", [], "", default_Q, "")
            pump_ids.append(pid)

    # ── Main T-mixer: reactor feeds only ───────────────────────────────────
    ops.append(UnitOperation(op_id="mixer_1", op_type="mixer",
        label=proposal.mixer_type or "T-Mixer",
        parameters={"type": proposal.mixer_type or "T-mixer", "material": "PEEK"},
        required=True, rationale=proposal.mixing_order_reasoning or "Combine reactor feeds"))
    for pid in pump_ids:
        _connect(ops, streams, sc, pid, "mixer_1")
    prev = "mixer_1"

    # ── Deoxygenation ──────────────────────────────────────────────────────
    deoxy = proposal.deoxygenation_method
    if not deoxy and chemistry_plan and chemistry_plan.deoxygenation_required:
        deoxy = "N2 sparging"
    if deoxy:
        ops.append(UnitOperation(op_id="deoxy_1", op_type="deoxygenation_unit",
            label="Inline Deoxygenation", parameters={"method": deoxy},
            required=True, rationale=chemistry_plan.deoxygenation_reasoning if chemistry_plan else ""))
        _connect(ops, streams, sc, prev, "deoxy_1")
        prev = "deoxy_1"

    # ── Main reactor ───────────────────────────────────────────────────────
    mat = proposal.tubing_material
    reactor_label = f"{mat} Photoreactor Coil" if is_photochem else f"{mat} Flow Reactor Coil"
    vol = proposal.reactor_volume_mL
    id_mm = proposal.tubing_ID_mm
    length = round((vol * 1e-6) / (math.pi * (id_mm * 5e-4) ** 2), 2) if vol and id_mm else None
    # Q entering the main reactor = sum of reactor_feed pump rates (gas excluded)
    Q_reactor_inlet = sum(
        (s.flow_rate_mL_min or 0.0)
        for s in reactor_feeds
        if not _stream_is_gas(s)
    ) or proposal.flow_rate_mL_min or 0.0
    ops.append(UnitOperation(op_id="reactor_1", op_type="coil_reactor",
        label=reactor_label, parameters={
            "material": mat, "ID_mm": id_mm, "volume_mL": vol,
            "Q_inlet_mL_min": round(Q_reactor_inlet, 4),
            "temperature_C": proposal.temperature_C, "wavelength_nm": proposal.wavelength_nm,
            "length_m": length, "residence_time_min": proposal.residence_time_min,
        }, required=True, rationale="Flow reactor"))
    _connect(ops, streams, sc, prev, "reactor_1")
    prev = "reactor_1"

    # ── LED ────────────────────────────────────────────────────────────────
    if is_photochem:
        ops.append(UnitOperation(op_id="led_1", op_type="led_module",
            label=f"LED {proposal.wavelength_nm:.0f} nm",
            parameters={"wavelength_nm": proposal.wavelength_nm},
            required=True, rationale="Photoexcitation"))

    # ── BPR ────────────────────────────────────────────────────────────────
    if proposal.BPR_bar and proposal.BPR_bar > 0:
        ops.append(UnitOperation(op_id="bpr_1", op_type="bpr",
            label="BPR", parameters={"pressure_bar": proposal.BPR_bar},
            required=True, rationale="Maintain liquid phase"))
        _connect(ops, streams, sc, prev, "bpr_1")
        prev = "bpr_1"

    # ── Quench: T-mixer + short contact coil ───────────────────────────────
    # Only runs if plan says quench_required OR a quench stream exists.
    needs_quench = bool(quench_streams) or (
        chemistry_plan and getattr(chemistry_plan, "quench_required", False)
    )
    if needs_quench:
        quench_d = proposal.tubing_ID_mm or 1.0
        quench_tau = 1.0  # 1 min contact time (standard for inline neutralisation)

        # Create pumps for each quench stream. If the plan flags quench_required
        # but no quench stream exists in proposal.streams, synthesise a default.
        quench_pump_ids: list[str] = []
        Q_quench_total = 0.0
        if quench_streams:
            for s in quench_streams:
                fr = s.flow_rate_mL_min or 0.0
                Q_quench_total += fr
                pid = _add_pump(
                    ops, s.stream_label or "Q", s.pump_role or "Inline quench",
                    s.contents, s.solvent, fr, s.reasoning or "",
                    is_gas=_stream_is_gas(s),
                )
                quench_pump_ids.append(pid)
        else:
            # Fallback: plan says quench needed but no stream defined. Synthesise.
            fallback_fr = round(Q_reactor_inlet * 0.1, 4) or 0.1
            Q_quench_total = fallback_fr
            pid = _add_pump(
                ops, "Q", "Inline quench",
                [chemistry_plan.quench_reagent or "quench reagent"],
                "H₂O", fallback_fr,
                chemistry_plan.quench_reasoning or
                "Quench excess reagent at reactor outlet",
            )
            quench_pump_ids.append(pid)

        # Q_inlet to the quench coil = reactor outlet + ALL quench pump rates
        Q_quench_inlet = round(Q_reactor_inlet + Q_quench_total, 4)
        quench_vol = round(quench_tau * Q_quench_inlet, 3)
        quench_length = round(
            (quench_vol * 1e-6) / (math.pi * (quench_d * 5e-4) ** 2), 2
        ) if quench_vol and quench_d else None

        # Quench T-mixer: reactor outlet + quench pumps
        ops.append(UnitOperation(op_id="quench_mixer", op_type="mixer",
            label="Quench T-Mixer",
            parameters={"type": "T-mixer", "material": "PEEK"},
            required=True, rationale="Mix reactor outlet with quench stream(s)"))
        _connect(ops, streams, sc, prev, "quench_mixer")
        for qpid in quench_pump_ids:
            _connect(ops, streams, sc, qpid, "quench_mixer")
        prev = "quench_mixer"

        # Quench contact coil (τ = 1 min)
        ops.append(UnitOperation(op_id="quench_coil", op_type="coil_reactor",
            label=f"{quench_d:.1f}mm Quench Coil",
            parameters={
                "material": proposal.tubing_material or "FEP",
                "ID_mm": quench_d,
                "volume_mL": quench_vol,
                "Q_inlet_mL_min": Q_quench_inlet,
                "temperature_C": 25,
                "residence_time_min": quench_tau,
                "length_m": quench_length,
                "reactor_type": "quench_contact",
            },
            required=True,
            rationale=(
                f"τ={quench_tau:.1f} min contact (V_R={quench_vol} mL, "
                f"Q_inlet={Q_quench_inlet} mL/min = reactor outlet "
                f"{Q_reactor_inlet:.3f} + quench {Q_quench_total:.3f})"
            )))
        _connect(ops, streams, sc, prev, "quench_coil")
        prev = "quench_coil"
        final_outlet_Q = Q_quench_inlet
    else:
        final_outlet_Q = Q_reactor_inlet

    # ── Collector ──────────────────────────────────────────────────────────
    ops.append(UnitOperation(op_id="collector_1", op_type="collector",
        label="Product Collection", parameters={}, required=True, rationale="Outlet"))
    _connect(ops, streams, sc, prev, "collector_1")

    pid_parts = [o.label for o in ops if o.op_type != "led_module"]
    return ProcessTopology(
        topology_id="translate", unit_operations=ops, streams=streams,
        total_flow_rate_mL_min=round(final_outlet_Q, 4),
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

    Key correctness rules:
    - Q_inlet_i = Q_from_previous_stage + sum(Q_new_feeds_i)
    - New feed Qs are derived from molar_equiv and concentration (stoichiometric split)
    - V_R_i = τ_i × Q_inlet_i  (not the global Q)
    - τ_i comes from proposal.stage_parameters (council decision) or equal split
    - d_mm per stage also comes from stage_parameters if the council set it
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

    import math as _math

    # Import here to avoid circular; INTENSIFICATION is the IF table
    from flora_translate.design_calculator import INTENSIFICATION, ESTIMATED_EA

    n_stages = len(chemistry_plan.stages)
    total_Q   = proposal.flow_rate_mL_min or 0.5

    # Build a lookup for council-provided per-stage diameter overrides
    stage_params_by_sn: dict[int, dict] = {
        p["stage_number"]: p
        for p in (proposal.stage_parameters or [])
        if isinstance(p, dict) and "stage_number" in p
    }

    prev_op: str | None = None
    Q_prev_outlet: float = 0.0  # cumulative Q leaving the previous reactor

    # Reference Q and C for stoichiometric new-feed sizing in stages 2+
    Q_reference: float = total_Q
    C_reference: float = proposal.concentration_M or 0.1

    # Batch-level data for independent τ computation
    _t_batch_h = getattr(batch_record, "reaction_time_h", None) or 0.0

    # Pre-compute which feed-stream labels first appear in a LATER stage.
    # The chemistry LLM sometimes lists a quench/workup stream in Stage 1's
    # feed_streams AND in Stage 2's — deduplicate by only creating the pump
    # at the LAST stage that declares it. This prevents Pump C from appearing
    # twice (once before the reactor, once at the inter-stage injection point).
    _label_last_stage: dict[str, int] = {}
    for _stg in chemistry_plan.stages:
        for _feed in _stg.feed_streams:
            lbl = (_feed.stream_label or "").upper()
            if lbl:
                _label_last_stage[lbl] = _stg.stage_number

    for stage in chemistry_plan.stages:
        sn = stage.stage_number
        prefix = f"st{sn}"
        sp = stage_params_by_sn.get(sn, {})

        # ── Per-stage τ ─────────────────────────────────────────────────────
        # Priority: (1) council-stored value in stage_parameters, (2) batch/IF,
        # (3) equal split of proposal τ as last resort.
        if sp.get("residence_time_min"):
            stage_rt = round(float(sp["residence_time_min"]), 2)
        else:
            stage_t_batch_h = getattr(stage, "batch_time_h", None) or _t_batch_h
            stage_t_batch_min = (stage_t_batch_h or 0.0) * 60.0

            stage_rc = (getattr(stage, "reaction_class", None) or "").lower()
            if not stage_rc and chemistry_plan.reaction_class:
                stage_rc = chemistry_plan.reaction_class.lower()
            IF_key = next(
                (k for k in INTENSIFICATION if k in stage_rc),
                "default"
            )
            IF_stage = INTENSIFICATION[IF_key]

            stage_T_flow = getattr(stage, "temperature_C", None)
            batch_T = getattr(batch_record, "temperature_C", None) or 25.0
            if stage_T_flow and abs(stage_T_flow - batch_T) > 2:
                Ea = ESTIMATED_EA.get(IF_key, ESTIMATED_EA["default"])
                exponent = -Ea / 8.314 * (1.0 / (stage_T_flow + 273.15) - 1.0 / (batch_T + 273.15))
                IF_stage = IF_stage * math.exp(exponent)

            if stage_t_batch_min > 0 and IF_stage > 0:
                stage_rt = round(stage_t_batch_min / IF_stage, 2)
            else:
                stage_rt = round((proposal.residence_time_min or 5.0) / max(n_stages, 1), 2)

        # ── Per-stage d_mm (council override or proposal default) ──────────
        stage_id_mm = float(sp.get("d_mm") or proposal.tubing_ID_mm or 1.0)

        # ── New feed pump flow rates ────────────────────────────────────────
        # Filter out feeds whose label first belongs to a later stage — the
        # chemistry LLM sometimes lists a quench stream in Stage 1 AND Stage 2.
        # We only create the pump at the LAST declared stage so it appears at
        # the correct injection point (and Q_inlet is computed correctly).
        active_feeds = [
            f for f in stage.feed_streams
            if _label_last_stage.get((f.stream_label or "").upper(), sn) == sn
        ]

        stage_pump_ids: list[str] = []

        if sn == 1:
            # Build a lookup of Chief-applied rates from proposal.streams
            proposal_rate_by_label: dict[str, float] = {
                s.stream_label.upper(): s.flow_rate_mL_min
                for s in (proposal.streams or [])
                if s.stream_label and s.flow_rate_mL_min
            }
            # Only use Chief-derived rates if every active liquid feed is covered.
            active_liquid_labels = {
                (f.stream_label or "").upper()
                for f in active_feeds
                if not _stream_is_gas(f)
            }
            if proposal_rate_by_label and active_liquid_labels.issubset(
                proposal_rate_by_label.keys()
            ):
                new_feed_qs = [
                    0.0 if _stream_is_gas(f)
                    else proposal_rate_by_label[(f.stream_label or "").upper()]
                    for f in active_feeds
                ]
            else:
                new_feed_qs = _stoich_flow_rates(active_feeds, total_Q)
        else:
            # Each new feed Q derived from stoichiometry relative to stage-1 substrate
            new_feed_qs = []
            for feed in active_feeds:
                if _stream_is_gas(feed):
                    new_feed_qs.append(0.0)
                    continue
                equiv = max(getattr(feed, "molar_equiv", 1.0) or 1.0, 1e-9)
                conc  = max(getattr(feed, "concentration_M", None) or C_reference, 1e-9)
                q = round(Q_reference * equiv * C_reference / conc, 4)
                new_feed_qs.append(q)

        for feed, q in zip(active_feeds, new_feed_qs):
            char = feed.stream_label or next_pump_char()
            contents = feed.reagents if feed.reagents else []
            pid = _add_pump(
                ops, char,
                feed.reasoning or f"Stage {sn} feed",
                contents, stage.solvent, q,
                feed.reasoning or f"Feed for {stage.stage_name}",
                is_gas=_stream_is_gas(feed),
            )
            stage_pump_ids.append(pid)

        # Track reference Q/C from stage 1 for use in stages 2+
        if sn == 1 and active_feeds:
            liquid_feeds = [f for f in active_feeds if not _stream_is_gas(f)]
            if liquid_feeds:
                ref_feed = liquid_feeds[0]
                Q_reference = new_feed_qs[active_feeds.index(ref_feed)]
                C_reference = getattr(ref_feed, "concentration_M", None) or C_reference

        # ── Q entering this reactor ─────────────────────────────────────────
        Q_new_feeds = sum(q for q in new_feed_qs if q > 0)
        Q_inlet = round(Q_prev_outlet + Q_new_feeds, 4)

        # ── Reactor volume V_R = τ_i × Q_inlet_i ───────────────────────────
        stage_vol = round(stage_rt * Q_inlet, 4)
        stage_length = (
            round((stage_vol * 1e-6) / (math.pi * (stage_id_mm * 5e-4) ** 2), 2)
            if stage_vol and stage_id_mm else None
        )

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
                "ID_mm": stage_id_mm,
                "volume_mL": stage_vol,
                "Q_inlet_mL_min": Q_inlet,
                "temperature_C": stage.temperature_C,
                "wavelength_nm": stage.wavelength_nm if is_photo else None,
                "residence_time_min": stage_rt,
                "length_m": stage_length,
                "reactor_type": rtype,
            },
            required=True,
            rationale=f"Reactor for {stage.stage_name} — τ={stage_rt} min, Q_inlet={Q_inlet} mL/min, V_R={stage_vol} mL",
        ))
        _connect(ops, streams, sc, prev_op, reactor_id)
        prev_op = reactor_id

        # Q leaving this reactor = Q_inlet (incompressible flow)
        Q_prev_outlet = Q_inlet

        # LED if photochemical stage
        if is_photo and stage.wavelength_nm:
            led_id = f"{prefix}_led"
            ops.append(UnitOperation(
                op_id=led_id, op_type="led_module",
                label=f"LED {stage.wavelength_nm:.0f} nm",
                parameters={"wavelength_nm": stage.wavelength_nm},
                required=True, rationale=f"Light for {stage.stage_name}",
            ))

        # ── Post-stage action (quench, filter, solvent switch, separator, BPR) ─
        if stage.post_stage_action:
            action = stage.post_stage_action.lower()
            post_id = f"{prefix}_post"
            n_ops_before = len(ops)  # track whether a node was actually added

            if "filter" in action:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="inline_filter",
                    label="Filter",
                    parameters={"pore_size_um": 10},
                    required=True,
                    rationale=stage.post_stage_reasoning or stage.post_stage_action,
                ))
            elif "quench" in action:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="quench_mixer",
                    label="Quench",
                    parameters={"reagent": stage.post_stage_action},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))
            elif "solvent" in action or "switch" in action:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="mixer",
                    label="Solvent Switch",
                    parameters={"type": "solvent_switch"},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))
            elif any(k in action for k in ("segment", "gas-liquid", "gas_liquid",
                                            "separator", "l-l", "liquid-liquid",
                                            "extraction", "phase sep")):
                # Only insert a separator when TWO IMMISCIBLE PHASES are genuinely
                # present AND the reasoning is explicit. Prevents hallucinated
                # separators for atmosphere changes or gas switching.
                reasoning = stage.post_stage_reasoning or ""
                if reasoning:
                    ops.append(UnitOperation(
                        op_id=post_id, op_type="liq_liq_extraction",
                        label="L-L Separator",
                        parameters={},
                        required=True,
                        rationale=reasoning,
                    ))
                else:
                    logger.warning(
                        "Skipping L-L separator for stage %d — "
                        "no post_stage_reasoning provided (not chemistry-justified).",
                        stage.stage_number,
                    )
            elif any(k in action for k in ("bpr", "back pressure", "back-pressure",
                                            "backpressure")):
                ops.append(UnitOperation(
                    op_id=post_id, op_type="bpr",
                    label="BPR",
                    parameters={"pressure_bar": proposal.BPR_bar or 5},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))
            else:
                ops.append(UnitOperation(
                    op_id=post_id, op_type="mixer",
                    label=stage.post_stage_action[:25],
                    parameters={"details": stage.post_stage_action},
                    required=True,
                    rationale=stage.post_stage_reasoning or "",
                ))

            # Only wire into the stream graph if a node was actually appended
            if len(ops) > n_ops_before:
                _connect(ops, streams, sc, prev_op, post_id)
                prev_op = post_id

    # ── Final BPR (only if proposal specifies AND no BPR node already exists) ──
    bpr_already_in_ops = any(o.op_type == "bpr" for o in ops)
    if proposal.BPR_bar and proposal.BPR_bar > 0 and not bpr_already_in_ops:
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
    # Q_prev_outlet is the true final outlet Q — Stage 1 Q plus any additions
    # from downstream feeds (e.g. quench). Use it for total_flow_rate so the
    # process diagram and summary agree on the outlet stream.
    final_outlet_Q = Q_prev_outlet or proposal.flow_rate_mL_min
    return ProcessTopology(
        topology_id="translate_multistep",
        unit_operations=ops, streams=streams,
        total_flow_rate_mL_min=final_outlet_Q,
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

    # 3b. Pre-compute engineering calculations (9-step design calculator)
    logger.info("Step 3b: Running 9-step design calculator")
    from flora_translate.design_calculator import DesignCalculator
    calculations = DesignCalculator().run(
        batch_record,
        chemistry_plan=chemistry_plan,
        inventory=LabInventory.from_json(inventory_path),
        analogies=analogies,
    )
    logger.info(
        f"  τ = {calculations.residence_time_min:.1f} min "
        f"(range {calculations.residence_time_range_min}), "
        f"method = {calculations.kinetics_method}, "
        f"Re = {calculations.reynolds_number:.0f}, "
        f"Da = {calculations.damkohler_mass:.2f}, "
        f"ΔP = {calculations.pressure_drop_bar:.4f} bar, "
        f"BPR = {'yes' if calculations.bpr_required else 'no'}"
    )

    # 3c. Design space grid search — enumerate feasible (τ, d, Q) candidates
    logger.info("Step 3c: Design space grid search")
    from flora_translate.engine.design_space import DesignSpaceSearch, candidates_to_dicts, get_council_starting_point
    design_candidates = DesignSpaceSearch().run(
        batch_record=batch_record,
        chemistry_plan=chemistry_plan,
        calculations=calculations,
        inventory=LabInventory.from_json(inventory_path),
        reaction_class=chemistry_plan.reaction_class if chemistry_plan else "default",
    )
    logger.info(f"  Design space: {len(design_candidates)} candidates, "
                f"{sum(1 for c in design_candidates if c.feasible)} feasible")

    # Use top design space candidate as council starting point (replaces LLM guess)
    _top_candidate = get_council_starting_point(design_candidates)
    if _top_candidate:
        logger.info(f"  Top candidate: τ={_top_candidate.tau_min}min, "
                    f"Q={_top_candidate.Q_mL_min}mL/min, d={_top_candidate.d_mm}mm, "
                    f"L={_top_candidate.L_m}m, score={_top_candidate.score:.3f}")

    # 4. Generate flow proposal
    logger.info("Step 4: Generating flow proposal via LLM")
    system_prompt, user_prompt = TranslationPromptBuilder().build(
        batch_record, analogies, chemistry_plan=chemistry_plan, calculations=calculations
    )
    proposal = TranslationLLM().generate(system_prompt, user_prompt)
    logger.info(f"  Proposal: {proposal.residence_time_min}min, {proposal.reactor_type}, "
                f"{len(proposal.streams)} streams")

    # Override proposal geometry with top design space candidate
    if _top_candidate:
        proposal.residence_time_min = _top_candidate.tau_min
        proposal.flow_rate_mL_min = _top_candidate.Q_mL_min
        proposal.tubing_ID_mm = _top_candidate.d_mm
        proposal.reactor_volume_mL = round(_top_candidate.V_R_mL, 3)
        logger.info("  Proposal geometry updated from design space top candidate")

    # 5. ENGINE deliberation council — Layer 3
    logger.info("Step 5: Multi-agent deliberation council (ENGINE)")
    inventory = LabInventory.from_json(inventory_path)
    pre_council_proposal = proposal.model_dump()  # snapshot before council modifies it
    design_candidate, calculations = CouncilV4().run(
        proposal, batch_record, analogies, inventory,
        chemistry_plan=chemistry_plan, calculations=calculations
    )

    # 6. Format output
    logger.info("Step 6: Formatting output")
    result = OutputFormatter().format(design_candidate, analogies)
    result["chemistry_plan"] = chemistry_plan.model_dump(exclude_none=True)

    # Attach 9-step design calculations for Streamlit rendering
    from dataclasses import asdict
    result["design_calculations"] = asdict(calculations)

    # Attach design space grid search results
    result["design_space"] = candidates_to_dicts(design_candidates)

    # ── Single source of truth: synchronise τ across all result sections ──
    # The validated proposal is authoritative for residence_time_min and
    # reactor_volume_mL.  The DesignCalculations may have re-derived a
    # slightly different τ from kinetics — force them to agree.
    _τ_proposal = result["proposal"].get("residence_time_min")
    _Q_proposal  = result["proposal"].get("flow_rate_mL_min")
    if _τ_proposal and _τ_proposal > 0:
        result["design_calculations"]["residence_time_min"] = _τ_proposal
        result["design_calculations"]["residence_time_s"] = round(_τ_proposal * 60, 2)
        if _Q_proposal and _Q_proposal > 0:
            result["design_calculations"]["reactor_volume_mL"] = round(
                _τ_proposal * _Q_proposal, 4
            )

    # Store analogies for the revision agent (confidence + context)
    result["_analogies"] = analogies

    # Attach deliberation log for Streamlit rendering
    if design_candidate.deliberation_log:
        result["deliberation_log"] = design_candidate.deliberation_log.model_dump()

    # Pre-council snapshot for before/after comparison in UI
    result["pre_council_proposal"] = pre_council_proposal

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
