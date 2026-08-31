"""Deterministic post-council realization of one executable flow design.

LLM and council outputs are design candidates.  This module is the final
engineering authority: it binds every feed and stage to inventory, solves
stream rates under pump limits, closes residence time, and emits explicit
validation evidence.  No model call occurs after this pass.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from typing import Any, Iterable

from flora_translate.inventory_constraints import (
    available_pressure_settings,
    select_reactor_for_proposal,
)
from flora_translate.multistage_inventory import reconcile_multistage_inventory
from flora_translate.residence_time_basis import (
    actual_gas_flow_from_stp,
    gas_equiv_from_stp_flow,
    stp_gas_flow_for_equiv,
)
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    FlowProposal,
    LabInventory,
    PumpSpec,
    ReactorSpec,
    StreamAssignment,
    StreamLogic,
)


AVAILABLE = {"available", "ready", "in_service", "in service"}


def realize_executable_design(
    proposal: FlowProposal,
    *,
    batch_record: BatchRecord,
    chemistry_plan: ChemistryPlan | None,
    inventory: LabInventory | None,
    hard_constraints: Any = None,
    operating_limits: dict[str, Any] | None = None,
) -> tuple[FlowProposal, dict[str, Any], dict[str, Any]]:
    """Return a hardware-bound proposal, realization audit, and validation.

    The selected reactor volume and available feed devices are independent
    variables.  Pump rates are solved first; residence times are consequences
    of those rates and the discrete reactor inventory.
    """

    current = proposal.model_copy(deep=True)
    constraint_text = _flatten_text(hard_constraints)
    chemistry_text = "\n".join(
        value
        for value in (
            batch_record.raw_text or "",
            batch_record.reaction_description or "",
        )
        if value
    )
    protocol_text = "\n".join(
        value for value in (chemistry_text, constraint_text) if value
    )
    issues: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    _reconcile_candidate_operations(current, chemistry_plan, decisions)
    _repair_stage_feed_map(chemistry_plan, protocol_text, decisions)
    current.streams = _normalized_streams(
        current.streams,
        chemistry_plan,
        protocol_text,
        authoritative_gas=_protocol_reagent_gas(batch_record),
        decisions=decisions,
    )
    _enforce_stationary_component_placement(
        current,
        chemistry_plan,
        inventory,
        decisions,
    )
    _resolve_pressure(current, inventory, constraint_text, decisions, issues)

    assignments = _assign_feed_devices(
        current,
        inventory,
        decisions,
        issues,
    )
    _solve_liquid_stream_rates(current, assignments, decisions, issues)
    _resolve_component_quantity_assumptions(current, decisions)
    is_multistage = bool(chemistry_plan and len(chemistry_plan.stages or []) > 1)
    if is_multistage:
        # Bind stage temperatures before converting an introduced gas from STP
        # to in-channel flow. The temperature at the gas-introduction stage is
        # authoritative, not the proposal's stage-1 summary temperature.
        _seed_stage_inventory(
            current,
            chemistry_plan,
            inventory,
            decisions,
            issues,
        )
    _solve_gas_stream_rates(
        current,
        assignments,
        inventory,
        decisions,
        issues,
        chemistry_plan=chemistry_plan,
    )
    deoxygenation_text = str(current.deoxygenation_method or "").lower()
    explicitly_not_required = any(
        token in deoxygenation_text
        for token in (
            "no deoxygen",
            "deoxygenation not required",
            "deoxygenation is not required",
            "unnecessary",
            "none required",
        )
    )
    plan_requires_deoxygenation = bool(
        chemistry_plan and chemistry_plan.deoxygenation_required
    )
    deoxygenation_required = plan_requires_deoxygenation or bool(
        deoxygenation_text and not explicitly_not_required
    )
    if explicitly_not_required and not plan_requires_deoxygenation:
        current.deoxygenation_method = ""
        decisions.append(
            {
                "decision": "remove_negated_deoxygenation_operation",
                "basis": (
                    "The candidate explicitly states that deoxygenation is not "
                    "required; negated text cannot create an inline unit operation."
                ),
            }
        )
    if inventory is not None and not inventory.degassers and deoxygenation_required:
        current.deoxygenation_method = (
            "offline nitrogen purge or pre-degassed feed preparation; "
            "no inline degasser"
        )
        current.pre_reactor_steps = [
            step
            for step in current.pre_reactor_steps
            if "inline degas" not in str(step).lower()
        ]
        if inventory and any(_same_gas("N2", item.gas) for item in inventory.gas_hardware):
            current.pre_reactor_steps.append(
                "Purge the assembled system with the declared nitrogen supply before reagent feed startup."
            )
        else:
            current.pre_reactor_steps.append(
                "Prepare the feed offline using the laboratory-approved inerting method; no inline degasser or declared nitrogen line is available."
            )
        decisions.append(
            {
                "decision": "deoxygenation_realization",
                "basis": "No inline degasser is declared; use offline feed preparation/system purge only.",
            }
        )

    multistage_report: dict[str, Any] = {"applied": False}
    if is_multistage:
        current, multistage_report = reconcile_multistage_inventory(
            current,
            chemistry_plan,
            inventory,
            operating_limits=operating_limits,
        )
        if _refresh_gas_at_realized_stage_temperature(current, decisions):
            current, multistage_report = reconcile_multistage_inventory(
                current,
                chemistry_plan,
                inventory,
                operating_limits=operating_limits,
            )
        if multistage_report.get("status") != "complete":
            issues.extend(multistage_report.get("unresolved_requirements") or [])
    else:
        current.stage_parameters = []
        _bind_single_reactor(current, inventory, decisions, issues)
        _close_single_stage_time(current, decisions)

    safety_contract = _derive_safety_contract(
        "\n".join(
            [
                chemistry_text,
                " ".join(
                    content
                    for stream in current.streams
                    for content in stream.contents
                ),
            ]
        ),
        current,
        inventory,
    )
    _compile_canonical_operating_summary(current)
    _replace_stale_reasoning(current, decisions, safety_contract)

    realization = {
        "schema_version": "flowpilot_design_realization_v1.0",
        "status": "complete" if not issues else "incomplete",
        "authority": "deterministic_post_council",
        "solver_order": [
            "normalize chemistry streams and stoichiometry",
            "bind each feed to inventory",
            "solve pump/MFC rates within device ranges",
            "bind discrete reactor and pressure hardware",
            "derive stage and total residence times",
            "validate safety and numerical closure",
        ],
        "decisions": decisions,
        "feed_assignments": [
            {
                "stream_label": stream.stream_label,
                "phase": stream.phase,
                "equipment_id": stream.pump_equipment_id,
                "introduction_stage": stream.introduction_stage,
                "flow_rate_mL_min": stream.flow_rate_mL_min,
                "gas_flow_sccm": stream.gas_flow_sccm,
                "gas_flow_actual_mL_min": stream.gas_flow_actual_mL_min,
                "concentration_M": stream.concentration_M,
                "molar_equiv": stream.molar_equiv,
            }
            for stream in current.streams
        ],
        "multistage": multistage_report,
        "safety_contract": safety_contract,
        "issues": issues,
    }
    constraints = dict(current.inventory_constraints or {})
    constraints["design_realization"] = realization
    constraints["safety_contract"] = safety_contract
    current.inventory_constraints = constraints

    validation = _validation_report(
        current,
        inventory,
        realization,
        multistage_report,
    )
    constraints = dict(current.inventory_constraints or {})
    constraints["final_validation"] = validation
    current.inventory_constraints = constraints
    return current, realization, validation


def _normalized_streams(
    proposal_streams: Iterable[StreamAssignment],
    plan: ChemistryPlan | None,
    protocol_text: str,
    *,
    authoritative_gas: str | None = None,
    decisions: list[dict[str, Any]] | None = None,
) -> list[StreamAssignment]:
    existing = {
        str(stream.stream_label or "").upper(): stream.model_copy(deep=True)
        for stream in proposal_streams or []
        if str(stream.stream_label or "").strip()
    }
    # Global stream_logic is the chemistry authority. Per-stage feeds are used
    # to place those streams in the topology, not to replace better quantified
    # global records with stage summaries that often omit concentration/equiv.
    stage_numbers: dict[str, int] = {}
    if plan:
        for stage in plan.stages or []:
            for feed in stage.feed_streams or []:
                if feed.delivery_mode != "carried_from_previous":
                    stage_numbers[str(feed.stream_label or "").upper()] = int(
                        feed.introduction_stage or stage.stage_number
                    )
    plan_feeds: list[tuple[int, StreamLogic]] = []
    global_labels: set[str] = set()
    if plan:
        for feed in plan.stream_logic or []:
            label = str(feed.stream_label or "").upper()
            global_labels.add(label)
            plan_feeds.append(
                (
                    stage_numbers.get(label, int(feed.introduction_stage or 1)),
                    feed,
                )
            )
        for stage in plan.stages or []:
            for feed in stage.feed_streams or []:
                label = str(feed.stream_label or "").upper()
                if feed.delivery_mode != "carried_from_previous" and label not in global_labels:
                    plan_feeds.append(
                        (int(feed.introduction_stage or stage.stage_number), feed)
                    )

    ordered: list[StreamAssignment] = []
    seen: set[str] = set()
    for stage_number, feed in plan_feeds:
        label = str(feed.stream_label or "").upper()
        if not label or label in seen:
            continue
        seen.add(label)
        stream = existing.pop(label, None) or StreamAssignment(
            stream_label=label,
            pump_role=feed.reasoning or f"Stage {stage_number} feed",
            contents=list(feed.reagents or []),
            concentration_M=feed.concentration_M,
            phase=feed.phase,
            molar_equiv=feed.molar_equiv,
        )
        if plan and plan.canonical_contract is not None:
            previous_contents = list(stream.contents)
            stream.contents = list(feed.reagents or [])
            if previous_contents != stream.contents and decisions is not None:
                decisions.append(
                    {
                        "decision": "restore_canonical_stream_composition",
                        "stream_label": label,
                        "previous_contents": previous_contents,
                        "canonical_contents": list(stream.contents),
                        "basis": (
                            "The downstream candidate cannot add or retain components "
                            "outside the frozen chemistry contract."
                        ),
                    }
                )
        elif not stream.contents:
            stream.contents = list(feed.reagents or [])
        stream.phase = feed.phase or stream.phase
        stream.introduction_stage = stage_number
        if feed.concentration_M and feed.concentration_M > 0:
            stream.concentration_M = feed.concentration_M
            stream.concentration_basis = "chemistry_plan"
        elif stream.concentration_M and not stream.concentration_basis:
            stream.concentration_basis = "model_candidate"
        if feed.molar_equiv and feed.molar_equiv > 0:
            stream.molar_equiv = float(feed.molar_equiv)
            stream.molar_equiv_basis = feed.molar_equiv_basis or "chemistry_plan"
        elif not stream.molar_equiv_basis:
            stream.molar_equiv_basis = "model_candidate"
        stream.requirement_authority = feed.requirement_authority
        stream.source_evidence = list(feed.source_evidence)
        stream.accepted_requirement = feed.accepted_requirement
        stream.separate_feed_required = feed.separate_feed_required
        stream.feed_group = feed.feed_group
        ordered.append(stream)

    if plan and plan.canonical_contract is not None:
        for stream in existing.values():
            if decisions is not None:
                decisions.append(
                    {
                        "decision": "remove_stream_outside_canonical_contract",
                        "stream_label": stream.stream_label,
                        "contents": list(stream.contents),
                        "basis": (
                            "The downstream candidate cannot add a required physical "
                            "feed absent from the frozen chemistry contract."
                        ),
                    }
                )
    else:
        ordered.extend(existing.values())
    for stream in ordered:
        explicit = _explicit_equivalent(protocol_text, stream.contents)
        if explicit is not None:
            stream.molar_equiv = explicit
            stream.molar_equiv_basis = "protocol_fact"
        explicit_concentration = _explicit_concentration(protocol_text, stream.contents)
        if explicit_concentration is not None:
            stream.concentration_M = explicit_concentration
            stream.concentration_basis = "protocol_fact"
        reference = _explicit_stream_reference(stream.contents)
        if reference is not None:
            component, concentration, equivalent = reference
            stream.concentration_M = concentration
            stream.molar_equiv = equivalent
            stream.concentration_basis = f"reference component: {component}"
            stream.molar_equiv_basis = f"reference component: {component}"
        identity = " ".join(stream.contents + [stream.pump_role]).lower()
        if "dinitrat" in protocol_text.lower() and "nitric" in identity:
            stream.molar_equiv = max(float(stream.molar_equiv or 0.0), 2.0)
            if not stream.molar_equiv_basis:
                stream.molar_equiv_basis = "chemistry_rule"
        if stream.concentration_M is None or stream.concentration_M <= 0:
            stream.concentration_M = None
    filtered = [
        stream
        for stream in ordered
        if stream.accepted_requirement
        if not (
            stream.phase == "gas"
            and _gas_identity(stream) == "N2"
            and any(
                token in f"{stream.pump_role} {' '.join(stream.contents)}".lower()
                for token in ("purge", "inert", "blanket", "startup", "shutdown")
            )
        )
    ]
    return _reconcile_physical_gas_feeds(
        filtered,
        authoritative_gas=authoritative_gas,
        decisions=decisions,
    )


def _reconcile_candidate_operations(
    proposal: FlowProposal,
    plan: ChemistryPlan | None,
    decisions: list[dict[str, Any]],
) -> None:
    """Remove candidate-only hardware semantics before deterministic solving."""

    contract = plan.canonical_contract if plan else None
    if contract is None:
        return
    if not contract.light_required_stages and (
        str(proposal.light_setup or "").strip() or proposal.wavelength_nm is not None
    ):
        decisions.append(
            {
                "decision": "remove_light_outside_canonical_contract",
                "previous_light_setup": proposal.light_setup,
                "previous_wavelength_nm": proposal.wavelength_nm,
                "basis": "The frozen batch protocol does not require irradiation.",
            }
        )
        proposal.light_setup = ""
        proposal.wavelength_nm = None


def _protocol_reagent_gas(batch_record: BatchRecord) -> str | None:
    """Return the protocol-authorized reagent gas, not its reactive component.

    For example, oxygen is the reacting species in an aerobic oxidation, but
    the physical feed remains air when the protocol says ``exposed to air``.
    """

    atmosphere = str(batch_record.atmosphere or "").lower().replace("₂", "2")
    raw = " ".join(
        value
        for value in (
            batch_record.raw_text or "",
            batch_record.reaction_description or "",
        )
        if value
    ).lower().replace("₂", "2")
    for text in (atmosphere, raw):
        if re.search(
            r"(?:\bair\b|expos(?:ed|ure)\s+to\s+(?:the\s+)?air|"
            r"under\s+(?:an?\s+)?air\s+atmosphere)",
            text,
        ):
            return "air"
        if re.search(r"\b(?:oxygen|o2)\b", text):
            return "O2"
        if re.search(r"\b(?:hydrogen|h2)\b", text):
            return "H2"
        if re.search(r"\b(?:carbon dioxide|co2)\b", text):
            return "CO2"
    return None


def _reconcile_physical_gas_feeds(
    streams: list[StreamAssignment],
    *,
    authoritative_gas: str | None,
    decisions: list[dict[str, Any]] | None,
) -> list[StreamAssignment]:
    """Collapse alternative descriptions of one physical reagent-gas feed.

    Stream labels are display identifiers and cannot create a second physical
    oxidant feed. Air and pure O2 at the same introduction stage are mutually
    exclusive delivery implementations of one oxygen requirement.
    """

    decisions = decisions if decisions is not None else []
    by_stage: dict[int, list[StreamAssignment]] = {}
    for stream in streams:
        if stream.phase == "gas":
            by_stage.setdefault(int(stream.introduction_stage or 1), []).append(stream)

    removed: set[int] = set()
    oxygen_family = {"air", "o2"}
    target = str(authoritative_gas or "").lower()
    for stage, gas_streams in by_stage.items():
        oxidants = [
            stream
            for stream in gas_streams
            if _gas_identity(stream).lower() in oxygen_family
        ]
        if not oxidants:
            continue
        keep = None
        if target in oxygen_family:
            keep = next(
                (
                    stream
                    for stream in oxidants
                    if _gas_identity(stream).lower() == target
                ),
                None,
            )
            if keep is None:
                keep = oxidants[0]
                previous = _gas_identity(keep)
                keep.contents = ["air" if target == "air" else "O2"]
                keep.pump_role = (
                    "Protocol-authorized air oxidant feed"
                    if target == "air"
                    else "Protocol-authorized O2 oxidant feed"
                )
                decisions.append(
                    {
                        "decision": "restore_protocol_gas_identity",
                        "stream_label": keep.stream_label,
                        "stage": stage,
                        "from_gas": previous,
                        "to_gas": keep.contents[0],
                        "basis": "Frozen batch protocol reagent-gas identity overrides model inference.",
                    }
                )
        else:
            keep = oxidants[0]

        for stream in oxidants:
            if stream is keep:
                continue
            removed.add(id(stream))
            decisions.append(
                {
                    "decision": "remove_duplicate_oxidant_feed",
                    "stream_label": stream.stream_label,
                    "stage": stage,
                    "gas_identity": _gas_identity(stream),
                    "retained_stream_label": keep.stream_label,
                    "retained_gas_identity": _gas_identity(keep),
                    "basis": (
                        "Air and pure O2 are alternative physical implementations "
                        "of one oxygen requirement, not simultaneous feeds."
                    ),
                }
            )
    return [stream for stream in streams if id(stream) not in removed]


def _enforce_stationary_component_placement(
    proposal: FlowProposal,
    plan: ChemistryPlan | None,
    inventory: LabInventory | None,
    decisions: list[dict[str, Any]],
) -> None:
    """Remove fixed-bed catalyst material from pumped stream contents."""

    reactor_identity = " ".join(
        [
            str(proposal.reactor_type or ""),
            *(
                f"{item.name} {item.type} {item.configuration}"
                for item in (inventory.reactors if inventory else [])
            ),
        ]
    ).lower()
    if "packed" not in reactor_identity and "catcart" not in reactor_identity:
        return

    stationary_names = {
        _normalized(item.name)
        for item in (plan.reagents if plan else [])
        if _is_stationary_catalyst(
            role=item.role,
            notes=item.notes,
            source=item.name,
        )
    }
    stationary: list[dict[str, Any]] = []
    for stream in proposal.streams:
        retained: list[str] = []
        for content in stream.contents:
            normalized = _normalized(content)
            explicit_stationary = _is_stationary_catalyst(
                role="catalyst" if "catalyst" in str(content).lower() else "",
                source=content,
            )
            planned_stationary = any(
                name and (name in normalized or normalized in name)
                for name in stationary_names
            )
            if explicit_stationary or planned_stationary:
                stationary.append(
                    {
                        "name": re.sub(r"\s*\([^)]*\)\s*$", "", str(content)).strip(),
                        "source_text": str(content),
                        "placement": "stationary_reactor_phase",
                        "reactor_type": proposal.reactor_type,
                    }
                )
                decisions.append(
                    {
                        "decision": "move_component_to_stationary_reactor_phase",
                        "stream_label": stream.stream_label,
                        "component": str(content),
                        "basis": (
                            "A heterogeneous catalyst assigned to a packed-bed "
                            "reactor cannot also be a pumped feed component."
                        ),
                    }
                )
                continue
            retained.append(content)
        stream.contents = retained

    if stationary:
        constraints = dict(proposal.inventory_constraints or {})
        constraints["stationary_components"] = stationary
        proposal.inventory_constraints = constraints


def _is_stationary_catalyst(
    *,
    role: Any = "",
    notes: Any = "",
    source: Any = "",
) -> bool:
    """Classify an immobilized catalyst without relying on one exact phrase."""

    text = " ".join(str(value or "") for value in (role, notes, source)).lower()
    normalized = " ".join(re.findall(r"[a-z0-9]+", text))
    is_catalyst = any(token in normalized.split() for token in ("catalyst", "catalytic"))
    is_stationary = any(
        marker in normalized
        for marker in (
            "heterogeneous",
            "immobilized",
            "supported catalyst",
            "pre packed",
            "prepacked",
            "packed bed",
            "packed in cartridge",
            "fixed bed",
            "fixed packed bed",
            "stationary phase",
            "not in solution",
            "not in feed",
            "dissolved loading not applicable",
        )
    )
    return is_catalyst and is_stationary


def _explicit_stream_reference(
    contents: list[str],
) -> tuple[str, float, float] | None:
    """Return a component carrying both concentration and equivalents."""

    for content in contents:
        concentration = re.search(r"(?i)(\d+(?:\.\d+)?)\s*M\b", str(content))
        equivalent = re.search(r"(?i)(\d+(?:\.\d+)?)\s*(?:equiv|eq)\b", str(content))
        if concentration and equivalent:
            name = re.sub(r"\s*\([^)]*\)\s*$", "", str(content)).strip()
            return name or str(content), float(concentration.group(1)), float(equivalent.group(1))
    return None


def _repair_stage_feed_map(
    plan: ChemistryPlan | None,
    protocol_text: str,
    decisions: list[dict[str, Any]],
) -> None:
    """Repair incomplete stage/feed labels from the plan's global stream map.

    A lightweight upstream fallback can produce the right global A/B/C streams
    while repeating one placeholder feed in every stage.  Stage topology must
    use the complete global labels, with introduction stage inferred only from
    explicit stage annotations/reasoning/constraints.
    """

    if plan is None or len(plan.stages or []) <= 1 or not plan.stream_logic:
        return
    global_labels = {
        str(feed.stream_label or "").upper()
        for feed in plan.stream_logic
        if str(feed.stream_label or "").strip()
    }
    stage_labels = {
        str(feed.stream_label or "").upper()
        for stage in plan.stages
        for feed in stage.feed_streams or []
        if str(feed.stream_label or "").strip()
    }
    if global_labels <= stage_labels:
        return

    by_stage: dict[int, list[StreamLogic]] = {
        int(stage.stage_number): [] for stage in plan.stages
    }
    for feed in plan.stream_logic:
        stage_number = int(feed.introduction_stage or 0)
        if stage_number <= 0:
            match = re.search(r"stage\s*(\d+)", feed.reasoning or "", re.I)
            if match:
                stage_number = int(match.group(1))
        if stage_number <= 0:
            label = re.escape(str(feed.stream_label or ""))
            match = re.search(
                rf"(?:add|inject|introduce)\s+(?:stream\s+)?{label}[^\n.;]*?"
                rf"(?:after\s+stage\s*(\d+)|before\s+stage\s*(\d+))",
                protocol_text,
                re.I,
            )
            if match:
                stage_number = int(match.group(1) or match.group(2)) + (1 if match.group(1) else 0)
        if stage_number not in by_stage:
            stage_number = 1
        repaired = feed.model_copy(deep=True)
        repaired.introduction_stage = stage_number
        by_stage[stage_number].append(repaired)

    for stage in plan.stages:
        stage.feed_streams = by_stage.get(int(stage.stage_number), [])
    decisions.append(
        {
            "decision": "repair_stage_feed_map",
            "basis": "Global stream labels were more complete than placeholder per-stage feeds.",
            "stage_labels": {
                str(stage_number): [feed.stream_label for feed in feeds]
                for stage_number, feeds in by_stage.items()
            },
        }
    )


def _explicit_equivalent(text: str, contents: list[str]) -> float | None:
    lower = text.lower()
    values: list[float] = []
    for content in contents:
        cleaned = re.sub(r"\([^)]*\)", "", str(content)).strip().lower()
        aliases = [cleaned]
        aliases.extend(
            token for token in re.findall(r"[a-z][a-z0-9]{1,}", cleaned)
            if token not in {"acid", "aqueous", "solution", "water", "feed"}
        )
        for alias in aliases:
            if len(alias) < 2:
                continue
            match = re.search(
                re.escape(alias)
                + r"\s*\([^)]{0,45}?(\d+(?:\.\d+)?)\s*"
                + r"(?:equiv(?:alent)?s?|eq\.?)(?:\b|\))",
                lower,
            )
            if match:
                values.append(float(match.group(1)))
    return max(values) if values else None


def _explicit_concentration(text: str, contents: list[str]) -> float | None:
    """Return a reagent-specific protocol concentration, never a global guess."""

    lower = text.lower()
    values: list[float] = []
    for content in contents:
        cleaned = re.sub(r"\([^)]*\)", "", str(content)).strip().lower()
        aliases = [cleaned]
        aliases.extend(
            token
            for token in re.findall(r"[a-z][a-z0-9-]{2,}", cleaned)
            if token not in {"acid", "aqueous", "solution", "water", "feed", "solvent"}
        )
        for alias in aliases:
            if len(alias) < 3:
                continue
            match = re.search(
                re.escape(alias) + r"[^\n.;]{0,80}?(\d+(?:\.\d+)?)\s*(mM|M)\b",
                lower,
                re.I,
            )
            if match:
                value = float(match.group(1))
                values.append(value / 1000.0 if match.group(2).lower() == "mm" else value)
    return values[0] if values else None


def _assign_feed_devices(
    proposal: FlowProposal,
    inventory: LabInventory | None,
    decisions: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    if inventory is None:
        return assignments

    streams = proposal.streams
    liquid_streams = [stream for stream in streams if stream.phase != "gas"]
    pump_slots = [
        pump
        for pump in inventory.pumps
        if str(pump.service_status).lower() in AVAILABLE
        and (
            pump.max_pressure_bar is None
            or float(pump.max_pressure_bar) + 1e-12 >= float(proposal.BPR_bar or 0.0)
        )
        for _ in range(int(pump.quantity))
    ]
    if liquid_streams and pump_slots:
        best: tuple[float, tuple[PumpSpec, ...]] | None = None
        for candidate in itertools.permutations(pump_slots, min(len(liquid_streams), len(pump_slots))):
            if len(candidate) != len(liquid_streams):
                continue
            score = sum(_pump_match_score(stream, pump) for stream, pump in zip(liquid_streams, candidate))
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None:
            for stream, pump in zip(liquid_streams, best[1]):
                stream.pump_equipment_id = pump.equipment_id
                assignments[stream.stream_label] = pump

    for stream in liquid_streams:
        if stream.stream_label not in assignments:
            issues.append(
                {
                    "category": "pump_assignment",
                    "stream_label": stream.stream_label,
                    "reason": "No quantity-available liquid pump can be assigned.",
                }
            )

    for stream in (item for item in streams if item.phase == "gas"):
        gas = _gas_identity(stream)
        candidates = [
            item
            for item in inventory.gas_hardware
            if str(item.service_status).lower() in AVAILABLE
            and "mfc" in str(item.type).lower()
            and (not item.gas or _same_gas(gas, item.gas))
        ]
        if not candidates and gas.lower() == "air":
            # Air and pure O2 are not numerically interchangeable: the gas
            # solver accounts for the 0.21 O2 fraction. But when the only
            # declared oxidant MFC is O2, an aerobic oxidation can be screened
            # with that source instead of inventing an air MFC. Record the
            # substitution explicitly and recalculate on a pure-O2 basis.
            oxygen_candidates = [
                item
                for item in inventory.gas_hardware
                if str(item.service_status).lower() in AVAILABLE
                and "mfc" in str(item.type).lower()
                and _same_gas("O2", item.gas)
            ]
            role_text = " ".join([stream.pump_role, *stream.contents]).lower()
            if oxygen_candidates and any(
                token in role_text
                for token in ("oxygen", "oxidation", "oxidant", "aerobic", "air")
            ):
                candidates = oxygen_candidates
                stream.contents = ["oxygen"]
                stream.pump_role = "Pure O2 oxidant feed (inventory substitution)"
                decisions.append(
                    {
                        "decision": "oxidant_gas_inventory_substitution",
                        "stream_label": stream.stream_label,
                        "from_gas": "air",
                        "to_gas": "oxygen",
                        "basis": (
                            "No air MFC is declared; use the available O2 MFC "
                            "and recompute flow on a pure-O2 molar basis."
                        ),
                        "confirmation_required": True,
                    }
                )
        if candidates:
            device = candidates[0]
            stream.pump_equipment_id = device.equipment_id
            assignments[stream.stream_label] = device
        else:
            issues.append(
                {
                    "category": "gas_assignment",
                    "stream_label": stream.stream_label,
                    "reason": f"No compatible MFC is available for {gas or 'the reagent gas'}.",
                }
            )
    return assignments


def _pump_match_score(stream: StreamAssignment, pump: PumpSpec) -> float:
    if stream.pump_equipment_id == pump.equipment_id:
        return 100.0
    identity = _normalized(" ".join([*stream.contents, stream.solvent, stream.pump_role]))
    compatible = [_normalized(item) for item in pump.compatible_materials]
    matches = sum(1 for item in compatible if item and (item in identity or identity in item))
    return 10.0 * matches + (1.0 if not compatible else 0.0)


def _solve_liquid_stream_rates(
    proposal: FlowProposal,
    assignments: dict[str, Any],
    decisions: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    reactive = [
        stream
        for stream in proposal.streams
        if stream.phase != "gas" and not _is_quench(stream)
    ]
    if not reactive:
        return
    known = [
        stream
        for stream in reactive
        if stream.concentration_M and stream.concentration_M > 0
    ]
    if not known:
        # A proposal-level concentration describes the limiting liquid feed,
        # not every cofeed. Bind it to one reference stream only.
        reference = min(reactive, key=lambda item: abs(float(item.molar_equiv or 1.0) - 1.0))
        reference.concentration_M = max(float(proposal.concentration_M or 0.1), 1e-9)
        reference.concentration_basis = "screening_assumption: proposal limiting-feed concentration"
        known = [reference]

    # Resolve reagent feeds whose concentration is absent without copying the
    # substrate concentration. Choose a pump-feasible preparation concentration
    # and disclose it as an assumption. This preserves stoichiometry and keeps
    # unknown source strength visibly distinct from a measured protocol fact.
    reference = min(known, key=lambda item: abs(float(item.molar_equiv or 1.0) - 1.0))
    reference_weight = float(reference.molar_equiv or 1.0) / float(reference.concentration_M)
    reference_pump = assignments.get(reference.stream_label)
    nominal_each = max(float(proposal.flow_rate_mL_min or 0.0), 0.0) / max(len(reactive), 1)
    if reference_pump is not None:
        reference_flow = min(
            max(nominal_each, float(reference_pump.min_flow_rate_mL_min)),
            float(reference_pump.max_flow_rate_mL_min),
        )
    else:
        reference_flow = max(nominal_each, 0.01)
    provisional_scale = reference_flow / max(reference_weight, 1e-12)
    for stream in reactive:
        if stream.concentration_M and stream.concentration_M > 0:
            continue
        pump = assignments.get(stream.stream_label)
        if pump is not None:
            target_flow = min(
                max(nominal_each, float(pump.min_flow_rate_mL_min)),
                float(pump.max_flow_rate_mL_min),
            )
        else:
            target_flow = max(nominal_each, 0.01)
        stream.concentration_M = max(
            provisional_scale * float(stream.molar_equiv or 1.0) / target_flow,
            1e-9,
        )
        stream.concentration_basis = "screening_assumption: pump-feasible cofeed preparation"
        decisions.append(
            {
                "decision": "cofeed_concentration_assumption",
                "stream_label": stream.stream_label,
                "selected_concentration_M": round(stream.concentration_M, 8),
                "basis": stream.concentration_basis,
                "confirmation_required": True,
            }
        )

    weights = {
        stream.stream_label: float(stream.molar_equiv or 1.0) / float(stream.concentration_M)
        for stream in reactive
    }
    lower = 0.0
    upper = float("inf")
    for stream in reactive:
        pump = assignments.get(stream.stream_label)
        if pump is None:
            continue
        weight = weights[stream.stream_label]
        lower = max(lower, float(pump.min_flow_rate_mL_min) / weight)
        upper = min(upper, float(pump.max_flow_rate_mL_min) / weight)
    if upper < lower - 1e-12:
        issues.append(
            {
                "category": "pump_ratio_feasibility",
                "reason": "No common stoichiometric scale satisfies every assigned pump range.",
                "scale_interval": [lower, upper],
            }
        )
        scale = lower
    else:
        target = max(float(proposal.flow_rate_mL_min or 0.0), 0.0) / sum(weights.values())
        scale = min(max(target, lower), upper)

    for stream in reactive:
        stream.flow_rate_mL_min = round(scale * weights[stream.stream_label], 6)
        stream.reasoning = (
            "Deterministic stoichiometric flow solution: "
            f"Q_{stream.stream_label}=k*(equiv/C)={stream.flow_rate_mL_min:.6g} mL/min; "
            f"equiv={stream.molar_equiv:g}, C={stream.concentration_M:g} M; "
            f"pump={stream.pump_equipment_id or 'unresolved'}; "
            f"concentration_basis={stream.concentration_basis or 'unresolved'}; "
            f"equiv_basis={stream.molar_equiv_basis or 'unresolved'}."
        )

    for stream in (
        item for item in proposal.streams if item.phase != "gas" and _is_quench(item)
    ):
        pump = assignments.get(stream.stream_label)
        if pump is not None:
            requested = float(stream.flow_rate_mL_min or pump.min_flow_rate_mL_min)
            stream.flow_rate_mL_min = round(
                min(max(requested, pump.min_flow_rate_mL_min), pump.max_flow_rate_mL_min),
                6,
            )

    proposal.flow_rate_mL_min = round(
        sum(float(stream.flow_rate_mL_min or 0.0) for stream in reactive),
        6,
    )
    decisions.append(
        {
            "decision": "liquid_flow_solution",
            "equation": "Q_i = k * (equiv_i / C_i)",
            "feasible_scale_interval": [round(lower, 8), None if upper == float("inf") else round(upper, 8)],
            "selected_scale": round(scale, 8),
            "total_reactive_liquid_flow_mL_min": proposal.flow_rate_mL_min,
        }
    )


def _solve_gas_stream_rates(
    proposal: FlowProposal,
    assignments: dict[str, Any],
    inventory: LabInventory | None,
    decisions: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    chemistry_plan: ChemistryPlan | None = None,
) -> None:
    gas_streams = [stream for stream in proposal.streams if stream.phase == "gas"]
    if not gas_streams:
        proposal.multiphase_metrics = {}
        return
    gas_reports: list[dict[str, Any]] = []
    base_metrics = dict(proposal.multiphase_metrics or {})
    for key in (
        "gas_species",
        "gas_flow_sccm",
        "gas_flow_actual_mL_min",
        "target_gas_equiv_inlet",
        "gas_equiv_supplied",
        "gas_reagent_mole_fraction",
        "gas_streams",
    ):
        base_metrics.pop(key, None)
    proposal.multiphase_metrics = base_metrics
    for stream in gas_streams:
        mfc = assignments.get(stream.stream_label)
        target_equiv = max(float(stream.molar_equiv or 1.0), 1.0)
        gas_identity = _gas_identity(stream)
        if inventory is not None and mfc is None:
            stream.gas_flow_sccm = None
            stream.gas_flow_actual_mL_min = None
            stream.flow_rate_mL_min = None
            decisions.append(
                {
                    "decision": "gas_flow_solution_unresolved",
                    "stream_label": stream.stream_label,
                    "gas_species": gas_identity,
                    "basis": "No compatible physical gas-delivery device is assigned.",
                }
            )
            continue
        # The finalizer must preserve the chemistry-authorized target. Any gas
        # excess is an explicit design decision, not an implicit H2 policy.
        # A declared MFC minimum may still raise the delivered equivalents;
        # that physical excess is calculated and reported below.
        fraction = 0.21 if gas_identity.lower() == "air" else 1.0
        limiting_flow, concentration = _limiting_liquid_basis(proposal)
        sccm = stp_gas_flow_for_equiv(
            limiting_flow,
            concentration,
            target_equiv,
            fraction,
        )
        if mfc is not None:
            if mfc.min_flow_sccm is not None:
                sccm = max(sccm, float(mfc.min_flow_sccm))
            if mfc.max_flow_sccm is not None and sccm > float(mfc.max_flow_sccm):
                factor = float(mfc.max_flow_sccm) / sccm
                liquid_streams = [
                    item
                    for item in proposal.streams
                    if item.phase != "gas" and not _is_quench(item)
                ]
                coupled_feasible = all(
                    assignments.get(item.stream_label) is not None
                    and float(item.flow_rate_mL_min or 0.0) * factor
                    >= float(assignments[item.stream_label].min_flow_rate_mL_min) - 1e-9
                    for item in liquid_streams
                )
                if coupled_feasible:
                    for item in liquid_streams:
                        item.flow_rate_mL_min = round(
                            float(item.flow_rate_mL_min or 0.0) * factor, 6
                        )
                        item.reasoning += (
                            f" Coupled gas-limit scale={factor:.6g} applied to "
                            f"fit MFC {mfc.equipment_id}."
                        )
                    proposal.flow_rate_mL_min = round(
                        sum(float(item.flow_rate_mL_min or 0.0) for item in liquid_streams),
                        6,
                    )
                    limiting_flow, concentration = _limiting_liquid_basis(proposal)
                    sccm = stp_gas_flow_for_equiv(
                        limiting_flow,
                        concentration,
                        target_equiv,
                        fraction,
                    )
                    sccm = min(sccm, float(mfc.max_flow_sccm))
                    decisions.append(
                        {
                            "decision": "coupled_gas_liquid_flow_solution",
                            "stream_label": stream.stream_label,
                            "equipment_id": mfc.equipment_id,
                            "liquid_scale_factor": round(factor, 8),
                            "total_liquid_flow_mL_min": proposal.flow_rate_mL_min,
                            "basis": "Reduce all reactive liquid feeds proportionally to preserve stoichiometry and satisfy the MFC maximum.",
                        }
                    )
                else:
                    issues.append(
                        {
                            "category": "gas_flow_feasibility",
                            "stream_label": stream.stream_label,
                            "reason": (
                                "Required reagent-gas flow exceeds the assigned MFC maximum, "
                                "and proportional liquid-rate reduction would violate a pump minimum."
                            ),
                        }
                    )
                    sccm = float(mfc.max_flow_sccm)
        pressure_absolute = max(float(proposal.BPR_bar or 0.0) + 1.01325, 1.01325)
        proposal.BPR_basis = "gauge"
        proposal.pressure_absolute_bar = round(pressure_absolute, 6)
        gas_temperature_C = _gas_introduction_temperature(
            stream,
            proposal,
            chemistry_plan,
            fallback_C=float(proposal.temperature_C or 25.0),
        )
        actual = actual_gas_flow_from_stp(
            sccm,
            gas_temperature_C,
            float(proposal.BPR_bar or 0.0),
        )
        supplied = gas_equiv_from_stp_flow(
            sccm,
            limiting_flow,
            concentration,
            fraction,
        )
        stream.gas_flow_sccm = round(sccm, 6)
        stream.gas_flow_actual_mL_min = round(actual, 6)
        stream.flow_rate_mL_min = round(actual, 6)
        stream.molar_equiv = round(supplied, 4)
        stream.molar_equiv_basis = "deterministic STP molar-flow calculation"
        stream.contents = [gas_identity]
        stream.reasoning = (
            f"MFC inlet/STP flow={sccm:.6g} sccm at 273.15 K and 1.01325 bar "
            f"(22.414 L/mol); pressure-corrected in-channel "
            f"flow={actual:.6g} mL/min at {gas_temperature_C:g} deg C and "
            f"{pressure_absolute:g} bar absolute; supplied={supplied:.3g} equiv."
        )
        stream_metrics = {
            "stream_label": stream.stream_label,
            "gas_species": gas_identity,
            "gas_flow_sccm": round(sccm, 6),
            "gas_flow_actual_mL_min": round(actual, 6),
            "target_gas_equiv_inlet": round(target_equiv, 4),
            "gas_equiv_supplied": round(supplied, 4),
            "limiting_liquid_flow_mL_min": round(limiting_flow, 6),
            "limiting_reagent_concentration_M": round(concentration, 8),
            "stp_temperature_K": 273.15,
            "stp_pressure_bar": 1.01325,
            "stp_molar_volume_L_mol": 22.414,
            "gas_reagent_mole_fraction": fraction,
            "gas_introduction_temperature_C": gas_temperature_C,
            "pressure_basis": (
                f"BPR setpoint {proposal.BPR_bar:g} bar gauge; "
                f"{pressure_absolute:g} bar absolute used for gas compression"
            ),
        }
        gas_reports.append(stream_metrics)
        if len(gas_reports) == 1:
            proposal.multiphase_metrics.update(stream_metrics)
        decisions.append(
            {
                "decision": "gas_flow_solution",
                "stream_label": stream.stream_label,
                "gas_species": gas_identity,
                "equipment_id": stream.pump_equipment_id,
                "target_equiv": target_equiv,
                "supplied_equiv": round(supplied, 4),
                "inlet_STP_sccm": round(sccm, 6),
                "in_channel_mL_min": round(actual, 6),
                "pressure_absolute_bar": pressure_absolute,
                "gas_introduction_temperature_C": gas_temperature_C,
            }
        )
    proposal.multiphase_metrics["gas_streams"] = gas_reports


def _gas_introduction_temperature(
    stream: StreamAssignment,
    proposal: FlowProposal,
    chemistry_plan: ChemistryPlan | None,
    *,
    fallback_C: float,
) -> float:
    stage_number = int(stream.introduction_stage or 1)
    stage_parameters = next(
        (
            item
            for item in proposal.stage_parameters or []
            if int(item.get("stage_number") or 0) == stage_number
        ),
        None,
    )
    if stage_parameters is not None and stage_parameters.get("temperature_C") is not None:
        return float(stage_parameters["temperature_C"])
    if chemistry_plan is not None:
        stage = next(
            (
                item
                for item in chemistry_plan.stages or []
                if int(item.stage_number or 0) == stage_number
            ),
            None,
        )
        if stage is not None and stage.temperature_C is not None:
            return float(stage.temperature_C)
    return float(fallback_C)


def _refresh_gas_at_realized_stage_temperature(
    proposal: FlowProposal,
    decisions: list[dict[str, Any]],
) -> bool:
    """Recalculate actual gas volume after stage inventory snaps temperature."""

    changed = False
    pressure_gauge = float(proposal.BPR_bar or 0.0)
    for stream in (item for item in proposal.streams if item.phase == "gas"):
        stage_number = int(stream.introduction_stage or 1)
        stage = next(
            (
                item
                for item in proposal.stage_parameters or []
                if int(item.get("stage_number") or 0) == stage_number
            ),
            None,
        )
        if stage is None or stage.get("temperature_C") is None or not stream.gas_flow_sccm:
            continue
        temperature_C = float(stage["temperature_C"])
        actual = actual_gas_flow_from_stp(
            float(stream.gas_flow_sccm),
            temperature_C,
            pressure_gauge,
        )
        if abs(actual - float(stream.gas_flow_actual_mL_min or 0.0)) <= 1e-9:
            continue
        stream.gas_flow_actual_mL_min = round(actual, 6)
        stream.flow_rate_mL_min = round(actual, 6)
        stream.reasoning = re.sub(
            r"pressure-corrected in-channel flow=[^;]+;",
            (
                f"pressure-corrected in-channel flow={actual:.6g} mL/min at "
                f"{temperature_C:g} deg C and "
                f"{pressure_gauge + 1.01325:g} bar absolute;"
            ),
            stream.reasoning,
        )
        metrics = dict(proposal.multiphase_metrics or {})
        metrics["gas_flow_actual_mL_min"] = round(actual, 6)
        metrics["gas_introduction_temperature_C"] = temperature_C
        proposal.multiphase_metrics = metrics
        decisions.append(
            {
                "decision": "gas_stage_temperature_reconciliation",
                "stream_label": stream.stream_label,
                "stage_number": stage_number,
                "temperature_C": temperature_C,
                "inlet_STP_sccm": stream.gas_flow_sccm,
                "in_channel_mL_min": round(actual, 6),
            }
        )
        changed = True
    return changed


def _limiting_liquid_basis(proposal: FlowProposal) -> tuple[float, float]:
    reactive = [
        stream
        for stream in proposal.streams
        if stream.phase != "gas" and not _is_quench(stream)
    ]
    quantified = [
        stream
        for stream in reactive
        if float(stream.flow_rate_mL_min or 0.0) > 0
        and float(stream.concentration_M or 0.0) > 0
    ]
    if quantified:
        limiting = min(
            quantified,
            key=lambda item: (
                abs(float(item.molar_equiv or 1.0) - 1.0),
                float(item.molar_equiv or 1.0),
                str(item.stream_label),
            ),
        )
        return (
            float(limiting.flow_rate_mL_min),
            float(limiting.concentration_M),
        )
    return (
        max(float(proposal.flow_rate_mL_min or 0.0), 1e-9),
        max(float(proposal.concentration_M or 0.1), 1e-9),
    )


def _resolve_component_quantity_assumptions(
    proposal: FlowProposal,
    decisions: list[dict[str, Any]],
) -> None:
    """Make unresolved reactive cofeed quantities explicit screening inputs."""

    solvent_names = {
        "water", "dioxane", "acetonitrile", "methanol", "ethanol", "solvent",
        "carrier", "toluene", "dce", "dcm", "thf", "dmso", "dmf",
    }
    for stream in proposal.streams:
        if stream.phase == "gas" or not stream.concentration_M:
            continue
        revised: list[str] = []
        for content in stream.contents:
            text = str(content)
            lower = text.lower()
            name = re.sub(r"\([^)]*\)", "", lower).strip()
            is_solvent = any(
                re.search(rf"\b{re.escape(token)}\b", name)
                for token in solvent_names
            )
            quantified = bool(
                re.search(
                    r"\d+(?:\.\d+)?\s*(?:mM|M|equiv|eq\.?|mol\s*%|wt\s*%)(?![A-Za-z])",
                    text,
                    re.I,
                )
            )
            if not quantified and not is_solvent:
                clean = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
                if "catalyst" in name:
                    assumption = "1 mol% screening assumption"
                    assumption_value = {"loading_mol_pct": 1.0}
                else:
                    assumption = f"{float(stream.concentration_M):g} M screening assumption"
                    assumption_value = {
                        "selected_concentration_M": float(stream.concentration_M)
                    }
                text = f"{clean} ({assumption}; confirm stock assay before run)"
                decisions.append(
                    {
                        "decision": "component_quantity_screening_assumption",
                        "stream_label": stream.stream_label,
                        "component": clean,
                        **assumption_value,
                        "confirmation_required": True,
                    }
                )
            revised.append(text)
        stream.contents = revised


def _resolve_pressure(
    proposal: FlowProposal,
    inventory: LabInventory | None,
    constraints: str,
    decisions: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    settings = available_pressure_settings(inventory)
    requested = float(proposal.BPR_bar or 0.0)
    explicit = [
        float(value)
        for value in re.findall(
            r"(?:operate\s+at|available|setpoint|bpr)[^\n.;]{0,35}?(\d+(?:\.\d+)?)\s*bar",
            constraints,
            flags=re.I,
        )
    ]
    if not settings:
        proposal.BPR_bar = 0.0
        proposal.BPR_basis = "gauge"
        proposal.pressure_absolute_bar = 1.01325
        decisions.append(
            {
                "decision": "pressure_selection",
                "selected_BPR_bar": 0.0,
                "basis": "No pressure controller or BPR setpoint is declared.",
            }
        )
        return
    target = explicit[0] if explicit else requested
    has_reagent_gas = any(stream.phase == "gas" for stream in proposal.streams)
    if target <= 0 and has_reagent_gas:
        target = settings[0]
    if has_reagent_gas:
        # A gas-liquid screen needs a controlled pressure floor. Never snap a
        # 2.5 bar model suggestion to an unavailable 3 bar value or retain it
        # below the floor; choose the lowest declared setpoint at or above it.
        target = max(target, 3.0)
    if target <= 0 and ("packed" in proposal.reactor_type.lower() or proposal.temperature_C >= 80):
        target = settings[0]
    eligible_settings = (
        [value for value in settings if value >= 3.0]
        if has_reagent_gas
        else settings
    )
    proposal.BPR_bar = (
        min(eligible_settings, key=lambda value: abs(value - target))
        if target > 0 and eligible_settings
        else 0.0
    )
    proposal.BPR_basis = "gauge"
    proposal.pressure_absolute_bar = round(proposal.BPR_bar + 1.01325, 6)
    decisions.append(
        {
            "decision": "pressure_selection",
            "available_setpoints_bar": settings,
            "requested_bar": requested,
            "selected_BPR_bar": proposal.BPR_bar,
        }
    )


def _bind_single_reactor(
    proposal: FlowProposal,
    inventory: LabInventory | None,
    decisions: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    if inventory is None or not inventory.reactors:
        return
    selected_id = str((proposal.inventory_selection or {}).get("equipment_id") or "")
    reactor = next(
        (
            item
            for item in inventory.reactors
            if item.equipment_id == selected_id
            and str(item.service_status).lower() in AVAILABLE
        ),
        None,
    ) or select_reactor_for_proposal(proposal, inventory)
    if reactor is None:
        issues.append({"category": "reactor_assignment", "reason": "No available reactor can be bound."})
        return
    _apply_reactor(proposal, reactor)
    decisions.append(
        {
            "decision": "reactor_selection",
            "equipment_id": reactor.equipment_id,
            "volume_mL": reactor.volume_mL,
            "ID_mm": reactor.ID_mm,
            "material": reactor.material,
        }
    )


def _seed_stage_inventory(
    proposal: FlowProposal,
    plan: ChemistryPlan,
    inventory: LabInventory | None,
    decisions: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    if inventory is None:
        return
    remaining = Counter(
        {
            item.equipment_id: int(item.quantity)
            for item in inventory.reactors
            if str(item.service_status).lower() in AVAILABLE
        }
    )
    stages = list(plan.stages or [])
    batch_weights = [max(float(stage.batch_time_h or 0.0), 0.0) for stage in stages]
    if not batch_weights or sum(batch_weights) <= 0:
        batch_weights = [1.0 for _ in stages]
    target_total_volume = max(float(proposal.reactor_volume_mL or 0.0), 0.0)
    seeded = []
    for index, stage in enumerate(stages, start=1):
        target_volume = (
            target_total_volume * batch_weights[index - 1] / sum(batch_weights)
            if target_total_volume > 0
            else None
        )
        candidates = [item for item in inventory.reactors if remaining[item.equipment_id] > 0]
        candidates.sort(
            key=lambda item: _stage_reactor_score(
                stage,
                item,
                target_volume_mL=target_volume,
            )
        )
        if not candidates:
            issues.append(
                {
                    "category": "stage_reactor_assignment",
                    "stage_number": stage.stage_number,
                    "reason": "No quantity-available reactor remains for this stage.",
                }
            )
            continue
        reactor = candidates[0]
        remaining[reactor.equipment_id] -= 1
        temperature = _supported_temperature(
            float(stage.temperature_C or proposal.temperature_C or 25.0), reactor
        )
        seeded.append(
            {
                "stage_number": int(stage.stage_number or index),
                "reactor_equipment_id": reactor.equipment_id,
                "reactor_name": reactor.name,
                "reactor_volume_mL": reactor.volume_mL,
                "V_R_mL": reactor.volume_mL,
                "d_mm": reactor.ID_mm,
                "material": reactor.material,
                "temperature_C": temperature,
            }
        )
        decisions.append(
            {
                "decision": "stage_reactor_selection",
                "stage_number": int(stage.stage_number or index),
                "equipment_id": reactor.equipment_id,
                "temperature_C": temperature,
            }
        )
    proposal.stage_parameters = seeded


def _stage_reactor_score(
    stage: Any,
    reactor: ReactorSpec,
    *,
    target_volume_mL: float | None = None,
) -> tuple[float, float, float, float, str]:
    desired_temp = float(stage.temperature_C or 25.0)
    supported = _temperature_supported(desired_temp, reactor)
    identity = f"{reactor.name} {reactor.configuration}".lower()
    stage_number_match = f"stage {stage.stage_number}" in identity
    type_match = str(stage.reactor_type or "coil").replace("_", "-") in str(reactor.type).replace("_", "-").lower()
    volume_error = (
        abs(float(reactor.volume_mL) - target_volume_mL)
        / max(float(target_volume_mL), 1e-12)
        if target_volume_mL and target_volume_mL > 0
        else 0.0
    )
    return (
        0.0 if supported else 1000.0,
        0.0 if stage_number_match else 1.0,
        0.0 if type_match else 100.0,
        volume_error,
        reactor.equipment_id,
    )


def _apply_reactor(proposal: FlowProposal, reactor: ReactorSpec) -> None:
    proposal.reactor_type = reactor.type
    proposal.reactor_volume_mL = float(reactor.volume_mL)
    proposal.tubing_ID_mm = float(reactor.ID_mm)
    proposal.tubing_material = reactor.material
    proposal.temperature_C = _supported_temperature(float(proposal.temperature_C), reactor)
    proposal.inventory_selection = {
        **dict(proposal.inventory_selection or {}),
        "equipment_id": reactor.equipment_id,
        "name": reactor.name,
        "system": reactor.system,
        "volume_mL": reactor.volume_mL,
        "ID_mm": reactor.ID_mm,
        "material": reactor.material,
        "configuration": reactor.configuration,
    }


def _close_single_stage_time(
    proposal: FlowProposal,
    decisions: list[dict[str, Any]],
) -> None:
    volume = float(proposal.reactor_volume_mL or 0.0)
    liquid = float(proposal.flow_rate_mL_min or 0.0)
    gas_sccm = sum(float(stream.gas_flow_sccm or 0.0) for stream in proposal.streams if stream.phase == "gas")
    gas_actual = sum(float(stream.gas_flow_actual_mL_min or 0.0) for stream in proposal.streams if stream.phase == "gas")
    if volume <= 0 or liquid <= 0:
        return
    tau_liquid = volume / liquid
    tau_inlet = volume / max(liquid + gas_sccm, 1e-12)
    tau_channel = volume / max(liquid + gas_actual, 1e-12)
    # V/Q_liquid is a reproducible geometric reference. It is not the measured
    # fluid contact time in a packed bed or gas-liquid reactor, which requires
    # a holdup/RTD measurement.
    proposal.residence_time_min = round(tau_liquid, 6)
    packed_bed = "packed" in str(proposal.reactor_type or "").lower()
    if packed_bed:
        proposal.residence_time_basis = (
            "nominal empty-bed liquid space time (geometric reactor volume / liquid flow)"
        )
    elif gas_sccm > 0:
        proposal.residence_time_basis = (
            "nominal liquid-only reactor-volume time (geometric reactor volume / liquid flow)"
        )
    else:
        proposal.residence_time_basis = "liquid-only reactor volume / liquid flow"
    proposal.residence_time_inlet_min = None if packed_bed else round(tau_inlet, 6)
    proposal.residence_time_in_channel_min = None if packed_bed else round(tau_channel, 6)
    metrics = dict(proposal.multiphase_metrics or {})
    if gas_sccm > 0:
        for key in (
            "residence_time_inlet_min",
            "residence_time_in_channel_min",
            "apparent_inlet_STP_time_min",
            "apparent_in_channel_time_min",
        ):
            metrics.pop(key, None)
        metrics.update(
            {
                "nominal_liquid_empty_bed_time_min": round(tau_liquid, 6),
                "measured_contact_time_status": (
                    "not inferred; measure phase holdup or RTD at the final operating point"
                ),
            }
        )
        if not packed_bed:
            metrics.update(
                {
                    "apparent_inlet_STP_time_min": round(tau_inlet, 6),
                    "apparent_in_channel_time_min": round(tau_channel, 6),
                }
            )
        proposal.multiphase_metrics = metrics
    decisions.append(
        {
            "decision": "single_stage_residence_time_closure",
            "equations": {
                "liquid_contact": "tau_liquid = V_R / Q_liquid",
                "inlet_STP_apparent": "tau_inlet = V_R / (Q_liquid + Q_gas_STP)",
                "in_channel_apparent": "tau_channel = V_R / (Q_liquid + Q_gas_actual)",
            },
            "volume_mL": volume,
            "liquid_flow_mL_min": liquid,
            "tau_liquid_min": round(tau_liquid, 6),
            "tau_inlet_min": None if packed_bed else round(tau_inlet, 6),
            "tau_in_channel_min": None if packed_bed else round(tau_channel, 6),
            "authoritative_basis": proposal.residence_time_basis,
        }
    )


def _derive_safety_contract(
    text: str,
    proposal: FlowProposal,
    inventory: LabInventory | None,
) -> dict[str, Any]:
    lower = text.lower()
    hazards: list[str] = []
    controls: list[str] = []
    hardware_checks: dict[str, bool] = {}

    if any(token in lower for token in ("nitrat", "nitric acid", "nitro compound")):
        hazards.extend(["strong oxidizing acid", "highly exothermic nitration", "NOx/decomposition"])
        controls.extend(
            [
                "Use the smallest declared reactive volume and active temperature control.",
                "Verify corrosion-resistant, acid-compatible wetted materials before charging nitric acid.",
                "Collect into the declared cooled quench/collection system and provide NOx-compatible ventilation.",
                "Treat the first condition as a conservative screen after thermal-risk review; do not label it execute-ready.",
            ]
        )
        hardware_checks["active_temperature_control_declared"] = bool(inventory and inventory.temperature_controllers)
        hardware_checks["cooled_or_quench_collection_declared"] = bool(
            inventory and any("quench" in f"{item.name} {item.type}".lower() for item in inventory.collectors)
        )
    if any(token in lower for token in ("h2o2", "hydrogen peroxide", "tbhp", "peroxide")):
        hazards.extend(["organic/inorganic peroxide", "acid-peroxide incompatibility", "runaway/decomposition"])
        controls.extend(
            [
                "Keep peroxide feeds segregated until their assigned mixer and use compatible wetted materials.",
                "Prime with solvent before peroxide, prevent dead-heading, and monitor each reactor temperature.",
                "Stop reagent feeds before solvent flushing and use cooled compatible collection/work-up.",
            ]
        )
        hardware_checks["peroxide_compatible_flow_path"] = bool(
            inventory
            and all(
                item.material.upper() in {"PFA", "PTFE", "FEP", "GLASS", "STAINLESS STEEL"}
                for item in inventory.reactors
            )
        )
    if "azide" in lower:
        hazards.extend(["organic azide", "energetic metal-azide accumulation"])
        controls.extend(
            [
                "Minimize azide holdup and avoid unassessed copper/metal-azide accumulation in stagnant zones.",
                "Verify pressure-rated cartridge connections and flush the catalyst bed before opening.",
            ]
        )
    if _mentions_hydrogen_gas(lower):
        hazards.extend(["flammable hydrogen", "pressurized methanol service"])
        controls.extend(
            [
                "Leak-test and purge with the declared nitrogen line before admitting hydrogen.",
                "Use grounded ventilation and remove ignition sources.",
                "On shutdown stop hydrogen, displace residual gas with nitrogen, then depressurize through the controlled outlet.",
            ]
        )
        hardware_checks["hydrogen_mfc_declared"] = bool(
            inventory
            and any("mfc" in item.type.lower() and _same_gas("H2", item.gas) for item in inventory.gas_hardware)
        )
        hardware_checks["nitrogen_purge_declared"] = bool(
            inventory and any(_same_gas("N2", item.gas) for item in inventory.gas_hardware)
        )

    hazards = list(dict.fromkeys(hazards))
    controls = list(dict.fromkeys(controls))
    complete = bool(controls) and all(hardware_checks.values()) if hazards else True
    proposal.safety_flags = [f"CONTROL: {item}" for item in controls]
    return {
        "hazards": hazards,
        "required_controls": controls,
        "hardware_checks": hardware_checks,
        "complete": complete,
        "disposition_floor": "SCREEN" if hazards else "SCREEN_OR_EXECUTE_AFTER_REVIEW",
    }


def _mentions_hydrogen_gas(text: str) -> bool:
    """Distinguish H2 service from hydrogen-bearing reagent names.

    Formula boundaries keep H2O2 and H2SO4 out.  The word form is accepted
    unless it names hydrogen peroxide, the common false positive in oxidation
    protocols.
    """

    if re.search(r"(?<![a-z0-9])h2(?![a-z0-9])", text, flags=re.IGNORECASE):
        return True
    return bool(re.search(r"\bhydrogen\b(?!\s+peroxide\b)", text, flags=re.IGNORECASE))


def _replace_stale_reasoning(
    proposal: FlowProposal,
    decisions: list[dict[str, Any]],
    safety_contract: dict[str, Any],
) -> None:
    selected = proposal.inventory_selection or {}
    if proposal.stage_parameters:
        reactor_basis = (
            f"Sum of {len(proposal.stage_parameters)} inventory-assigned stages "
            f"supplies V_R,total={proposal.reactor_volume_mL:g} mL."
        )
    else:
        reactor_basis = (
            f"Exact inventory reactor {selected.get('equipment_id') or 'unresolved'} "
            f"supplies V_R={proposal.reactor_volume_mL:g} mL."
        )
    proposal.reasoning_per_field = {
        "finalization_authority": "All numerical run parameters were regenerated by the deterministic post-council realizer; earlier model-written numbers are non-authoritative.",
        "reactor_volume_mL": reactor_basis,
        "flow_rate_mL_min": f"Sum of non-quench liquid feeds is {proposal.flow_rate_mL_min:g} mL/min after per-pump bounds and stoichiometric ratios were solved jointly.",
        "residence_time_min": f"Derived from the selected reactor volume and the declared basis: {proposal.residence_time_min:g} min ({proposal.residence_time_basis}).",
        "BPR_bar": (
            f"Selected from declared pressure hardware only: {proposal.BPR_bar:g} bar gauge; "
            f"absolute pressure for gas calculations is {float(proposal.pressure_absolute_bar or 1.01325):g} bar."
        ),
        "temperature_C": f"Temperature is constrained by the assigned reactor/controller inventory: {proposal.temperature_C:g} deg C.",
        "safety_contract": f"Deterministic hazard controls complete={safety_contract.get('complete')}.",
    }
    proposal.chemistry_notes = (
        "Final numerical values are equipment-bound and deterministic. "
        "See inventory_constraints.design_realization for feed equations, device IDs, "
        "stage closure, gas basis, and safety controls."
    )


def _compile_canonical_operating_summary(proposal: FlowProposal) -> None:
    """Replace mutable model-written run instructions with canonical values."""

    preparation: list[str] = []
    for stream in proposal.streams:
        if stream.phase == "gas":
            continue
        contents = ", ".join(stream.contents) or "declared feed components"
        concentration = (
            f" at {float(stream.concentration_M):g} M"
            if stream.concentration_M is not None
            else ""
        )
        preparation.append(
            f"Prepare Stream {stream.stream_label} containing {contents} in "
            f"{stream.solvent or 'the declared solvent'}{concentration}; "
            f"concentration basis: {stream.concentration_basis or 'unresolved and requiring confirmation'}."
        )
    selected = proposal.inventory_selection or {}
    liquid_devices = [
        f"{stream.stream_label}:{stream.pump_equipment_id or 'UNRESOLVED'}"
        for stream in proposal.streams
        if stream.phase != "gas"
    ]
    gas_devices = [
        f"{stream.stream_label}:{stream.pump_equipment_id or 'UNRESOLVED'}"
        for stream in proposal.streams
        if stream.phase == "gas"
    ]
    if proposal.stage_parameters:
        setup_steps = [
            (
                f"Install Stage {int(stage.get('stage_number') or index)} reactor "
                f"{stage.get('reactor_equipment_id') or 'UNRESOLVED'} "
                f"({float(stage.get('reactor_volume_mL') or stage.get('V_R_mL') or 0):g} mL, "
                f"{float(stage.get('d_mm') or proposal.tubing_ID_mm):g} mm ID, "
                f"{stage.get('material') or proposal.tubing_material}) at "
                f"{float(stage.get('temperature_C') or proposal.temperature_C):g} deg C."
            )
            for index, stage in enumerate(proposal.stage_parameters, 1)
        ]
    else:
        setup_steps = [
            f"Install reactor {selected.get('equipment_id') or 'UNRESOLVED'} "
            f"({proposal.reactor_volume_mL:g} mL, {proposal.tubing_ID_mm:g} mm ID, "
            f"{proposal.tubing_material})."
        ]
    feed_setup = "Install assigned feeds " + ", ".join(liquid_devices + gas_devices) + "."
    startup = (
        f"Set the outlet BPR to {proposal.BPR_bar:g} bar gauge and start liquid "
        f"feeds at final outlet flow {proposal.flow_rate_mL_min:g} mL/min."
    )
    gas_steps = []
    for stream in proposal.streams:
        if stream.phase != "gas":
            continue
        gas_steps.append(
            f"Set gas Stream {stream.stream_label} ({_gas_identity(stream)}) to "
            f"{float(stream.gas_flow_sccm or 0):g} sccm at inlet/STP "
            f"({float(stream.gas_flow_actual_mL_min or 0):g} mL/min in-channel; "
            f"{float(stream.molar_equiv or 0):g} equiv supplied)."
        )
    proposal.pre_reactor_steps = [
        *preparation,
        *setup_steps,
        feed_setup,
        startup,
        *gas_steps,
    ]

    chemistry_actions = [
        str(step)
        for step in proposal.post_reactor_steps
        if any(
            token in str(step).lower()
            for token in ("separator", "quench", "collect", "waste")
        )
        and not any(token in str(step).lower() for token in ("startup", "shutdown"))
    ]
    proposal.post_reactor_steps = [
        *chemistry_actions,
        "Collect only after the canonical residence-time stabilization period; use the inventory-compiled downstream path.",
        "For shutdown, stop reactive cofeeds and reagent gas first, flush with a compatible declared solvent, then depressurize through the controlled outlet.",
    ]


def _validation_report(
    proposal: FlowProposal,
    inventory: LabInventory | None,
    realization: dict[str, Any],
    multistage: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "design_realization_complete": realization.get("status") == "complete",
        "reactor_inventory_match": _reactor_inventory_match(proposal, inventory, multistage),
        "pump_flow_feasible": _all_feed_devices_feasible(proposal, inventory),
        "stream_flow_closure": _stream_flow_closure(proposal),
        "stoichiometric_flow_closure": _stoichiometric_flow_closure(proposal),
        "pressure_inventory_match": _pressure_inventory_match(proposal, inventory),
        "tubing_feasible": _tubing_or_integrated_path_feasible(proposal, inventory),
        "geometry_closure": _geometry_closed(proposal),
        "calculation_matches_serialized_design": _geometry_closed(proposal),
        "gas_bookkeeping_complete": _gas_bookkeeping_complete(proposal),
        "safety_contract_complete": bool(realization["safety_contract"].get("complete")),
    }
    if multistage.get("applied"):
        checks["multistage_stage_inventory_complete"] = multistage.get("status") == "complete"
        checks["stage_geometry_closed"] = bool(multistage.get("checks", {}).get("stage_geometry_closed"))
    unresolved = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "flowpilot_final_validation_v2.0",
        "status": "ready" if not unresolved else "screen_required",
        "checks": checks,
        "unresolved_reasons": unresolved,
        "design_realization": realization,
    }


def _reactor_inventory_match(proposal: FlowProposal, inventory: LabInventory | None, multistage: dict[str, Any]) -> bool:
    if inventory is None or not inventory.reactors:
        return True
    known = {item.equipment_id for item in inventory.reactors}
    if multistage.get("applied"):
        return bool(proposal.stage_parameters) and all(
            str(item.get("reactor_equipment_id") or "") in known
            for item in proposal.stage_parameters
        )
    return str((proposal.inventory_selection or {}).get("equipment_id") or "") in known


def _all_feed_devices_feasible(proposal: FlowProposal, inventory: LabInventory | None) -> bool:
    if inventory is None:
        return True
    pumps = {item.equipment_id: item for item in inventory.pumps}
    mfcs = {item.equipment_id: item for item in inventory.gas_hardware if "mfc" in item.type.lower()}
    for stream in proposal.streams:
        if stream.phase == "gas":
            device = mfcs.get(str(stream.pump_equipment_id or ""))
            flow = float(stream.gas_flow_sccm or 0.0)
            if device is None or flow <= 0:
                return False
            if device.min_flow_sccm is not None and flow < device.min_flow_sccm - 1e-9:
                return False
            if device.max_flow_sccm is not None and flow > device.max_flow_sccm + 1e-9:
                return False
        else:
            device = pumps.get(str(stream.pump_equipment_id or ""))
            flow = float(stream.flow_rate_mL_min or 0.0)
            if device is None or not (device.min_flow_rate_mL_min - 1e-9 <= flow <= device.max_flow_rate_mL_min + 1e-9):
                return False
    return True


def _stream_flow_closure(proposal: FlowProposal) -> bool:
    total = sum(
        float(stream.flow_rate_mL_min or 0.0)
        for stream in proposal.streams
        if stream.phase != "gas" and not _is_quench(stream)
    )
    return total > 0 and abs(total - proposal.flow_rate_mL_min) <= max(1e-6, 0.005 * total)


def _stoichiometric_flow_closure(proposal: FlowProposal) -> bool:
    streams = [stream for stream in proposal.streams if stream.phase != "gas" and not _is_quench(stream)]
    if len(streams) <= 1:
        return True
    scales = []
    for stream in streams:
        concentration = float(stream.concentration_M or proposal.concentration_M or 0.0)
        equivalent = float(stream.molar_equiv or 0.0)
        flow = float(stream.flow_rate_mL_min or 0.0)
        if concentration <= 0 or equivalent <= 0 or flow <= 0:
            return False
        scales.append(flow * concentration / equivalent)
    return max(scales) - min(scales) <= max(1e-8, 0.01 * max(scales))


def _pressure_inventory_match(proposal: FlowProposal, inventory: LabInventory | None) -> bool:
    if proposal.BPR_bar <= 0:
        return not any(stream.phase == "gas" for stream in proposal.streams)
    return any(abs(value - proposal.BPR_bar) <= 1e-9 for value in available_pressure_settings(inventory))


def _tubing_or_integrated_path_feasible(proposal: FlowProposal, inventory: LabInventory | None) -> bool:
    if inventory is None or not inventory.reactors:
        return True
    selected_id = str((proposal.inventory_selection or {}).get("equipment_id") or "")
    selected = next((item for item in inventory.reactors if item.equipment_id == selected_id), None)
    if selected and _reactor_has_integrated_flow_path(selected):
        return True
    if not inventory.tubing:
        return False
    return any(
        item.material.lower() == proposal.tubing_material.lower()
        and abs(item.ID_mm - proposal.tubing_ID_mm) <= 1e-3
        and item.max_pressure_bar >= proposal.BPR_bar
        and item.max_temperature_C >= proposal.temperature_C
        for item in inventory.tubing
    )


def _geometry_closed(proposal: FlowProposal) -> bool:
    if proposal.stage_parameters:
        stages = [item for item in proposal.stage_parameters if item.get("inventory_resolved", True)]
        if not stages:
            return False
        for stage in stages:
            volume = float(stage.get("reactor_volume_mL") or stage.get("V_R_mL") or 0.0)
            flow = float(stage.get("Q_liquid_mL_min") or stage.get("flow_rate_mL_min") or 0.0)
            tau = float(stage.get("residence_time_min") or 0.0)
            if min(volume, flow, tau) <= 0 or abs(volume - flow * tau) > max(0.02, 0.02 * volume):
                return False
        return True
    volume = float(proposal.reactor_volume_mL or 0.0)
    if "inlet" in proposal.residence_time_basis.lower() and any(stream.phase == "gas" for stream in proposal.streams):
        flow = proposal.flow_rate_mL_min + sum(float(stream.gas_flow_sccm or 0.0) for stream in proposal.streams if stream.phase == "gas")
    elif "channel" in proposal.residence_time_basis.lower() and any(stream.phase == "gas" for stream in proposal.streams):
        flow = proposal.flow_rate_mL_min + sum(float(stream.gas_flow_actual_mL_min or 0.0) for stream in proposal.streams if stream.phase == "gas")
    else:
        flow = proposal.flow_rate_mL_min
    return volume > 0 and flow > 0 and abs(volume - flow * proposal.residence_time_min) <= max(0.02, 0.02 * volume)


def _gas_bookkeeping_complete(proposal: FlowProposal) -> bool:
    gases = [stream for stream in proposal.streams if stream.phase == "gas"]
    if not gases:
        return True
    metrics = proposal.multiphase_metrics or {}
    return all(
        float(stream.gas_flow_sccm or 0.0) > 0
        and float(stream.gas_flow_actual_mL_min or 0.0) > 0
        and float(stream.molar_equiv or 0.0) > 0
        for stream in gases
    ) and float(metrics.get("gas_equiv_supplied") or 0.0) > 0


def _supported_temperature(value: float, reactor: ReactorSpec) -> float:
    if reactor.allowed_temperatures_C:
        return float(min(reactor.allowed_temperatures_C, key=lambda item: abs(float(item) - value)))
    if reactor.min_temperature_C is not None:
        value = max(value, float(reactor.min_temperature_C))
    if reactor.max_temperature_C is not None:
        value = min(value, float(reactor.max_temperature_C))
    return value


def _temperature_supported(value: float, reactor: ReactorSpec) -> bool:
    return abs(_supported_temperature(value, reactor) - value) <= 1e-9


def _reactor_has_integrated_flow_path(reactor: ReactorSpec) -> bool:
    identity = f"{reactor.type} {reactor.configuration} {reactor.name}".lower()
    return any(token in identity for token in ("microchannel", "microreactor", "packed-bed", "packed bed", "integrated"))


def _gas_identity(stream: StreamAssignment) -> str:
    text = " ".join([*stream.contents, stream.pump_role]).lower()
    # Air descriptions commonly mention their oxygen fraction. The supplied
    # species remains air and must bind to an air MFC, not an O2 MFC.
    if re.search(r"(?:^|[^a-z0-9])(?:air|dry air|compressed air)(?:[^a-z0-9]|$)", text):
        return "air"
    if "hydrogen" in text or re.search(r"\bh2\b", text):
        return "H2"
    if "oxygen" in text or re.search(r"\bo2\b", text):
        return "O2"
    if "air" in text:
        return "air"
    if "nitrogen" in text or re.search(r"\bn2\b", text):
        return "N2"
    return (stream.contents[0] if stream.contents else stream.pump_role).strip()


def _same_gas(left: str, right: str) -> bool:
    aliases = {
        "h2": "hydrogen", "hydrogen": "hydrogen",
        "o2": "oxygen", "oxygen": "oxygen",
        "n2": "nitrogen", "nitrogen": "nitrogen",
        "air": "air",
    }
    return aliases.get(str(left).strip().lower(), str(left).strip().lower()) == aliases.get(str(right).strip().lower(), str(right).strip().lower())


def _is_quench(stream: StreamAssignment) -> bool:
    text = f"{stream.pump_role} {' '.join(stream.contents)}".lower()
    return any(token in text for token in ("quench", "neutraliz", "neutralis", "workup", "work-up"))


def _normalized(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_flatten_text(item) for item in value)
    return str(value)
