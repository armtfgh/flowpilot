"""Deterministic reconciliation between protocol facts and LLM chemistry plans.

The chemistry model may propose useful implementation ideas, but it is not an
authority for adding mandatory hardware.  This module freezes the operations
that downstream inventory and engineering gates are allowed to require.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Iterable

from flora_translate.schemas import (
    BatchRecord,
    CanonicalOperationRequirement,
    CanonicalReactionContract,
    ChemistryPlan,
    ProcessStage,
    StreamLogic,
    normalized_stream_phase,
)


_INERT_GASES = {"n2", "nitrogen", "ar", "argon"}


def reconcile_chemistry_plan(
    batch_record: BatchRecord,
    chemistry_plan: ChemistryPlan,
    *,
    hard_constraints: Any = None,
) -> tuple[ChemistryPlan, dict[str, Any]]:
    """Return a protocol-anchored plan and an auditable reconciliation report."""

    plan = chemistry_plan.model_copy(deep=True)
    protocol = _protocol_text(batch_record)
    constraint_text = _flatten_text(hard_constraints)
    evidence_text = "\n".join(value for value in (protocol, constraint_text) if value)
    decisions: list[dict[str, Any]] = []

    required_gases = _protocol_reagent_gases(batch_record)
    light_required = _protocol_requires_light(batch_record)
    explicit_separation = _explicit_separate_liquid_feeds(evidence_text)

    stages = _materialize_stages(plan)
    if light_required:
        if len(stages) == 1:
            light_stages = {int(stages[0].stage_number)}
        else:
            light_stages = {
                int(stage.stage_number)
                for stage in stages
                if stage.requires_light or stage.wavelength_nm is not None
            }
            if not light_stages:
                light_stages = {int(stages[0].stage_number)}
                decisions.append(
                    {
                        "decision": "default_unresolved_light_to_first_stage",
                        "stage": int(stages[0].stage_number),
                        "stream": "",
                        "basis": (
                            "The protocol requires irradiation but does not provide "
                            "enough structured evidence to assign another stage."
                        ),
                    }
                )
    else:
        light_stages = set()

    for stage in stages:
        stage_number = int(stage.stage_number or 1)
        feeds = list(stage.feed_streams or [])
        accepted: list[StreamLogic] = []
        liquid: list[StreamLogic] = []
        gas_seen: set[str] = set()

        for feed in feeds:
            original_reagents = list(feed.reagents)
            feed.reagents = [
                reagent
                for reagent in feed.reagents
                if not _is_excluded_component(reagent, evidence_text)
            ]
            removed_reagents = [
                reagent for reagent in original_reagents if reagent not in feed.reagents
            ]
            if removed_reagents:
                decisions.append(
                    _decision(
                        "remove_explicitly_excluded_component",
                        stage_number,
                        feed.stream_label,
                        "Excluded or zero-equivalent component(s): "
                        + ", ".join(removed_reagents),
                    )
                )
            if not feed.reagents:
                continue
            phase = normalized_stream_phase(
                feed.phase,
                feed.reagents,
                gas_flow_sccm=feed.gas_flow_sccm,
                gas_flow_actual_mL_min=feed.gas_flow_actual_mL_min,
            )
            feed.phase = phase
            if feed.delivery_mode == "carried_from_previous":
                feed.accepted_requirement = True
                feed.requirement_authority = "deterministic_derivation"
                feed.source_evidence = ["Material is carried from the preceding accepted reactor stage."]
                accepted.append(feed)
                continue
            if phase == "solid" and _is_stationary_bed_material(feed, evidence_text):
                feed.accepted_requirement = False
                feed.requirement_authority = "protocol_fact"
                feed.source_evidence = [
                    "The catalyst or solid is immobilized in the reactor and is not a pumped feed."
                ]
                feed.feed_group = f"ST{stage_number}-STATIONARY-BED"
                accepted.append(feed)
                decisions.append(
                    _decision(
                        "classify_stationary_bed_material_as_non_pumped",
                        stage_number,
                        feed.stream_label,
                        "Stationary reactor packing cannot consume a liquid-pump slot.",
                    )
                )
                continue
            if phase != "gas":
                liquid.append(feed)
                continue

            gas = _gas_identity(feed.reagents)
            if gas in _INERT_GASES and _is_purge(feed):
                feed.accepted_requirement = False
                feed.requirement_authority = "model_inference"
                decisions.append(
                    _decision(
                        "retain_nonblocking_purge_utility",
                        stage_number,
                        feed.stream_label,
                        f"{gas or 'inert gas'} is a purge utility, not a reaction feed.",
                    )
                )
                accepted.append(feed)
                continue
            matched = _match_authorized_gas(gas, required_gases)
            if matched:
                if matched in gas_seen:
                    decisions.append(
                        _decision(
                            "remove_duplicate_reagent_gas",
                            stage_number,
                            feed.stream_label,
                            f"A single physical {matched} feed satisfies the protocol.",
                        )
                    )
                    continue
                gas_seen.add(matched)
                explicit_equiv = _explicit_gas_equiv(evidence_text, matched)
                feed.reagents = [matched]
                if explicit_equiv is not None:
                    feed.molar_equiv = explicit_equiv
                    feed.molar_equiv_basis = "protocol_fact"
                    quantity_basis = f"{explicit_equiv:g} equiv is explicit in the frozen input."
                else:
                    feed.molar_equiv = 1.0
                    feed.molar_equiv_basis = "deterministic_screening_assumption"
                    quantity_basis = (
                        "1.0 equiv is a provisional stoichiometric screening basis; "
                        "the protocol does not state gas equivalents."
                    )
                feed.reasoning = f"Protocol-required {matched} reagent gas. {quantity_basis}"
                feed.requirement_authority = "protocol_fact"
                feed.source_evidence = [_gas_evidence(matched, protocol)]
                feed.accepted_requirement = True
                feed.separate_feed_required = True
                feed.feed_group = f"ST{stage_number}-GAS-{matched.upper()}"
                accepted.append(feed)
            else:
                decisions.append(
                    _decision(
                        "remove_unsupported_reagent_gas",
                        stage_number,
                        feed.stream_label,
                        f"{gas or 'gas'} is absent from the frozen batch protocol.",
                    )
                )

        if liquid:
            separation_required = explicit_separation or _plan_requires_separation(plan, liquid)
            if separation_required:
                for index, feed in enumerate(liquid, 1):
                    feed.requirement_authority = (
                        "hard_constraint" if explicit_separation else "model_inference"
                    )
                    feed.source_evidence = (
                        [_separation_evidence(evidence_text)] if explicit_separation else []
                    )
                    feed.accepted_requirement = True
                    feed.separate_feed_required = True
                    feed.feed_group = f"ST{stage_number}-LIQ-{index}"
                    accepted.append(feed)
            else:
                merged = _merge_liquid_feeds(liquid, stage_number)
                accepted.append(merged)
                if len(liquid) > 1:
                    decisions.append(
                        _decision(
                            "merge_unsubstantiated_liquid_feeds",
                            stage_number,
                            ", ".join(feed.stream_label for feed in liquid),
                            "The protocol describes one batch charge and no mandatory separate feed.",
                        )
                    )

        # Keep liquid feeds before gas feeds so downstream limiting-stream
        # selection remains stable across provider-specific output ordering.
        accepted.sort(
            key=lambda feed: normalized_stream_phase(feed.phase, feed.reagents) == "gas"
        )
        stage.feed_streams = accepted
        stage.requires_light = stage_number in light_stages
        if not stage.requires_light:
            stage.wavelength_nm = None

    # Add a missing protocol reagent gas exactly once, at the last reaction
    # stage by default. This is deterministic and avoids silently dropping a
    # chemically explicit gas when the upstream model omitted it.
    present_gases = {
        _gas_identity(feed.reagents)
        for stage in stages
        for feed in stage.feed_streams
        if feed.phase == "gas" and feed.accepted_requirement
    }
    target_stage = stages[-1]
    for gas in required_gases:
        if _match_authorized_gas(gas, present_gases):
            continue
        label = _next_stream_label(stages, prefix="G")
        target_stage.feed_streams.append(
            StreamLogic(
                stream_label=label,
                reagents=[gas],
                reasoning=(
                    f"Protocol-required {gas} reagent gas. "
                    + (
                        f"{_explicit_gas_equiv(evidence_text, gas):g} equiv is explicit in the frozen input."
                        if _explicit_gas_equiv(evidence_text, gas) is not None
                        else "1.0 equiv is a provisional stoichiometric screening basis; the protocol does not state gas equivalents."
                    )
                ),
                molar_equiv=_explicit_gas_equiv(evidence_text, gas) or 1.0,
                molar_equiv_basis=(
                    "protocol_fact"
                    if _explicit_gas_equiv(evidence_text, gas) is not None
                    else "deterministic_screening_assumption"
                ),
                phase="gas",
                introduction_stage=target_stage.stage_number,
                requirement_authority="protocol_fact",
                source_evidence=[_gas_evidence(gas, protocol)],
                accepted_requirement=True,
                separate_feed_required=True,
                feed_group=f"ST{target_stage.stage_number}-GAS-{gas.upper()}",
            )
        )
        decisions.append(
            _decision(
                "restore_missing_protocol_reagent_gas",
                target_stage.stage_number,
                label,
                f"The batch protocol explicitly requires {gas}.",
            )
        )

    plan.stages = stages
    plan.n_stages = len(stages)
    plan.stream_logic = _global_streams_from_stages(stages)
    plan.o2_is_reagent = any(gas.lower() in {"o2", "air"} for gas in required_gases)
    if not light_required:
        plan.recommended_wavelength_nm = None
        plan.wavelength_reasoning = ""
        plan.light_sensitive_reagents = []

    contract = _build_contract(batch_record, plan, required_gases)
    plan.canonical_contract = contract
    plan.reconciliation_log = [*list(plan.reconciliation_log or []), *decisions]
    report = {
        "schema_version": "flowpilot_chemistry_reconciliation_v1.0",
        "status": "complete",
        "authority": "protocol_and_hard_constraints",
        "required_reagent_gases": required_gases,
        "light_required": light_required,
        "light_required_stages": sorted(light_stages),
        "explicit_separate_liquid_feeds": explicit_separation,
        "decisions": decisions,
        "canonical_contract": contract.model_dump(exclude_none=True),
    }
    return plan, report


def _build_contract(
    batch_record: BatchRecord,
    plan: ChemistryPlan,
    required_gases: list[str],
) -> CanonicalReactionContract:
    operations: list[CanonicalOperationRequirement] = []
    for stage in plan.stages:
        stage_number = int(stage.stage_number)
        introduced = [
            feed
            for feed in stage.feed_streams
            if feed.delivery_mode != "carried_from_previous"
            and int(feed.introduction_stage or stage_number) == stage_number
            and feed.accepted_requirement
        ]
        for index, feed in enumerate(introduced, 1):
            phase = normalized_stream_phase(feed.phase, feed.reagents)
            op_type = "gas_feed" if phase == "gas" else "liquid_feed"
            operations.append(
                CanonicalOperationRequirement(
                    requirement_id=f"ST{stage_number}-{op_type.upper()}-{index}",
                    operation_type=op_type,
                    stage_number=stage_number,
                    authority=feed.requirement_authority,
                    source_evidence=list(feed.source_evidence),
                    species=list(feed.reagents),
                    feed_group=feed.feed_group,
                    rationale=feed.reasoning,
                )
            )
        inlet_count = len(introduced) + (1 if stage_number > 1 else 0)
        if inlet_count > 1:
            operations.append(
                CanonicalOperationRequirement(
                    requirement_id=f"ST{stage_number}-MIXER-1",
                    operation_type="mixer",
                    stage_number=stage_number,
                    authority="deterministic_derivation",
                    source_evidence=[f"Stage {stage_number} has {inlet_count} physical inlets."],
                    rationale="A passive mixing junction is required by the accepted feed graph.",
                )
            )
        operations.append(
            CanonicalOperationRequirement(
                requirement_id=f"ST{stage_number}-REACTOR-1",
                operation_type="reactor",
                stage_number=stage_number,
                authority="protocol_fact",
                source_evidence=[f"Batch reaction stage {stage_number}."],
                rationale=stage.stage_name or plan.reaction_name,
            )
        )
        if stage.requires_light:
            operations.append(
                CanonicalOperationRequirement(
                    requirement_id=f"ST{stage_number}-LIGHT-1",
                    operation_type="light_source",
                    stage_number=stage_number,
                    authority="protocol_fact",
                    source_evidence=[_light_evidence(_protocol_text(batch_record))],
                    rationale="The batch protocol explicitly uses irradiation.",
                )
            )
    if required_gases:
        operations.append(
            CanonicalOperationRequirement(
                requirement_id="PROCESS-PRESSURE-CONTROL-1",
                operation_type="pressure_control",
                stage_number=max(plan.n_stages, 1),
                authority="deterministic_derivation",
                source_evidence=["An accepted continuously metered reagent-gas feed is present."],
                rationale="Pressure control is required for deterministic gas-flow conversion.",
            )
        )
    protocol = _protocol_text(batch_record)
    return CanonicalReactionContract(
        protocol_sha256=hashlib.sha256(protocol.encode("utf-8")).hexdigest(),
        reaction_name=plan.reaction_name,
        n_stages=plan.n_stages,
        required_reagent_gases=required_gases,
        light_required_stages=[stage.stage_number for stage in plan.stages if stage.requires_light],
        operations=operations,
    )


def _materialize_stages(plan: ChemistryPlan) -> list[ProcessStage]:
    if plan.stages:
        stages = [stage.model_copy(deep=True) for stage in plan.stages]
        if not any(stage.feed_streams for stage in stages):
            stages[0].feed_streams = [feed.model_copy(deep=True) for feed in plan.stream_logic]
    else:
        stages = [
            ProcessStage(
                stage_number=1,
                stage_name=plan.reaction_name or "Reaction stage",
                feed_streams=[feed.model_copy(deep=True) for feed in plan.stream_logic],
            )
        ]
    for index, stage in enumerate(stages, 1):
        stage.stage_number = index
        for feed in stage.feed_streams:
            if feed.delivery_mode != "carried_from_previous":
                feed.introduction_stage = int(feed.introduction_stage or index)
    return stages


def _global_streams_from_stages(stages: Iterable[ProcessStage]) -> list[StreamLogic]:
    output: list[StreamLogic] = []
    seen: set[str] = set()
    for stage in stages:
        for feed in stage.feed_streams:
            label = str(feed.stream_label or "").upper()
            if not label or label in seen or feed.delivery_mode == "carried_from_previous":
                continue
            seen.add(label)
            output.append(feed.model_copy(deep=True))
    return output


def _merge_liquid_feeds(feeds: list[StreamLogic], stage_number: int) -> StreamLogic:
    base = feeds[0].model_copy(deep=True)
    reagents: list[str] = []
    for feed in feeds:
        for reagent in feed.reagents:
            if reagent not in reagents:
                reagents.append(reagent)
    base.stream_label = base.stream_label or f"A{stage_number}"
    base.reagents = reagents
    base.phase = "liquid"
    base.introduction_stage = stage_number
    base.delivery_mode = "new_feed"
    base.requirement_authority = "protocol_fact"
    base.source_evidence = ["All components are charged to the same batch reaction mixture."]
    base.accepted_requirement = True
    base.separate_feed_required = False
    base.feed_group = f"ST{stage_number}-LIQ-1"
    base.reasoning = "Single premixed liquid feed; no mandatory separation is stated."
    return base


def _protocol_text(batch_record: BatchRecord) -> str:
    return "\n".join(
        str(value)
        for value in (
            batch_record.raw_text,
            batch_record.reaction_description,
            batch_record.light_source,
            batch_record.atmosphere,
        )
        if value
    )


def _protocol_requires_light(batch_record: BatchRecord) -> bool:
    light_source = str(batch_record.light_source or "").strip().lower()
    if batch_record.wavelength_nm is not None:
        return True
    microwave_only = "microwave" in light_source and not re.search(
        r"\b(?:leds?|lamps?)\b|\d{3,4}\s*nm\b", light_source
    )
    if light_source and not microwave_only:
        return True
    source_text = "\n".join(
        str(value)
        for value in (batch_record.raw_text, batch_record.reaction_description)
        if value
    ) if microwave_only else _protocol_text(batch_record)
    text = re.sub(
        r"\bmicrowave(?:\s+(?:irradiation|heating|reactor|conditions?))?\b",
        "",
        source_text.lower(),
    )
    return bool(re.search(r"\b(?:irradiat\w*|photoredox|photochem\w*|leds?|lamps?)\b|\d{3,4}\s*nm\b", text))


def _protocol_reagent_gases(batch_record: BatchRecord) -> list[str]:
    atmosphere = str(batch_record.atmosphere or "").lower().replace("₂", "2")
    text = _protocol_text(batch_record).lower().replace("₂", "2")
    combined = f"{atmosphere}\n{text}"
    gases: list[str] = []
    if re.search(r"^\s*air(?:\b|,)", atmosphere) or re.search(r"\b(?:air atmosphere|under air|exposed? to (?:the )?air|aerobic)\b", combined):
        gases.append("air")
    elif re.search(r"^\s*(?:oxygen|o2)(?:\b|,)", atmosphere) or re.search(r"\b(?:oxygen gas|o2 gas|under oxygen|oxygen atmosphere|o2 atmosphere|oxygen-filled|o2-filled|contacted with oxygen)\b", combined):
        gases.append("O2")
    if re.search(r"^\s*(?:hydrogen|h2)(?:\b|,)", atmosphere) or re.search(r"\b(?:hydrogen gas|h2 gas|under hydrogen|hydrogen atmosphere|h2 atmosphere|hydrogenolysis|hydrogenoly[sz]ed|contacted with hydrogen)\b", combined):
        gases.append("H2")
    if re.search(r"\b(?:carbon dioxide gas|co2 gas|under co2|co2 atmosphere)\b", combined):
        gases.append("CO2")
    if re.search(r"\b(?:carbon monoxide gas|co gas|under co|co atmosphere)\b", combined):
        gases.append("CO")
    return list(dict.fromkeys(gases))


def _explicit_separate_liquid_feeds(text: str) -> bool:
    lower = text.lower()
    patterns = (
        r"\b(?:two|2)\s+(?:separate\s+)?(?:liquid\s+)?(?:feeds?|streams?|pumps?)\b",
        r"\b(?:stream|feed)\s+a\b.{0,100}\b(?:stream|feed)\s+b\b",
        r"\bpump(?:ed)?\s+separately\b",
        r"\bseparate(?:d)?\s+(?:liquid\s+)?feeds?\b",
        r"\b(?:t|y)[- ]?mixer\b",
        r"\badded\s+(?:slowly|dropwise|portionwise)\b",
        r"\b(?:slowly|dropwise|portionwise)\s+added\b",
        r"\baddition\s+funnel\b",
    )
    return any(re.search(pattern, lower, re.S) for pattern in patterns)


def _explicit_gas_equiv(text: str, gas: str) -> float | None:
    aliases = {
        "H2": r"(?:H2|H₂|hydrogen)",
        "O2": r"(?:O2|O₂|oxygen)",
        "air": r"air",
        "CO2": r"(?:CO2|CO₂|carbon dioxide)",
        "CO": r"(?:CO|carbon monoxide)",
    }.get(gas, re.escape(gas))
    patterns = (
        rf"{aliases}[^\n.;]{{0,100}}?(\d+(?:\.\d+)?)\s*(?:mol\s*)?(?:equiv|eq\.?\b)",
        rf"(\d+(?:\.\d+)?)\s*(?:mol\s*)?(?:equiv|eq\.?\b)[^\n.;]{{0,100}}?{aliases}",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = float(match.group(1))
            if value > 0:
                return value
    return None


def _plan_requires_separation(plan: ChemistryPlan, feeds: list[StreamLogic]) -> bool:
    # A model-inferred incompatibility remains a design hypothesis, not a hard
    # requirement. Only already authority-labelled non-model facts may force it.
    return any(
        feed.separate_feed_required
        and feed.requirement_authority in {"protocol_fact", "hard_constraint", "measured_evidence"}
        for feed in feeds
    )


def _gas_identity(species: Iterable[str]) -> str:
    text = " ".join(str(item) for item in species).lower().replace("₂", "2")
    if re.search(r"\bair\b", text):
        return "air"
    if re.search(r"\b(?:oxygen|o2)\b", text) and not re.search(r"hydrogen peroxide|h2o2", text):
        return "o2"
    if re.search(r"\b(?:hydrogen|h2)\b", text) and not re.search(r"hydrogen peroxide|h2o2", text):
        return "h2"
    if re.search(r"\b(?:carbon dioxide|co2)\b", text):
        return "co2"
    if re.search(r"\b(?:carbon monoxide|co)\b", text):
        return "co"
    if re.search(r"\b(?:nitrogen|n2)\b", text):
        return "n2"
    if re.search(r"\b(?:argon|ar)\b", text):
        return "ar"
    return text.strip()


def _match_authorized_gas(gas: str, authorized: Iterable[str]) -> str | None:
    normalized = {str(item).lower(): str(item) for item in authorized}
    gas_key = str(gas or "").lower()
    if gas_key in normalized:
        return normalized[gas_key]
    if gas_key in {"air", "o2"}:
        for key in ("air", "o2"):
            if key in normalized:
                return normalized[key]
    return None


def _is_purge(feed: StreamLogic) -> bool:
    text = f"{feed.reasoning} {' '.join(feed.reagents)}".lower()
    return any(token in text for token in ("purge", "inert", "blanket", "startup", "shutdown", "leak check"))


def _is_stationary_bed_material(feed: StreamLogic, context: str) -> bool:
    text = f"{feed.reasoning} {' '.join(feed.reagents)} {context}".lower()
    stationary = any(
        token in text
        for token in (
            "stationary",
            "immobilized",
            "immobilised",
            "packed-bed",
            "packed bed",
            "catalyst cartridge",
            "catcart",
            "not pumped",
            "fixed bed",
        )
    )
    slurry = any(token in text for token in ("slurry feed", "suspension feed", "pump the slurry"))
    return stationary and not slurry


def _is_excluded_component(component: str, context: str) -> bool:
    text = str(component or "").strip().lower()
    if not text:
        return True
    if re.search(r"(?<![\d.])0(?:\.0+)?\s*(?:equiv|eq\.?\b|mol\s*%)", text):
        return True
    if any(
        token in text
        for token in (
            "deliberately excluded",
            "explicitly excluded",
            "not included",
            "not used",
            "omitted",
        )
    ):
        return True
    name = re.sub(r"\([^)]*\)", "", text).strip()
    if len(name) < 3:
        return False
    lower_context = context.lower()
    return bool(
        re.search(rf"\b(?:do not use|without|exclude|omit)\s+{re.escape(name)}\b", lower_context)
        or re.search(
            rf"\b{re.escape(name)}\b[^.;]{{0,100}}?optional[^.;]{{0,100}}?must not be introduced",
            lower_context,
        )
    )


def _next_stream_label(stages: Iterable[ProcessStage], *, prefix: str) -> str:
    labels = {feed.stream_label.upper() for stage in stages for feed in stage.feed_streams}
    if prefix not in labels:
        return prefix
    index = 2
    while f"{prefix}{index}" in labels:
        index += 1
    return f"{prefix}{index}"


def _gas_evidence(gas: str, protocol: str) -> str:
    return f"Batch protocol declares {gas} as a reaction atmosphere or reagent gas."


def _light_evidence(protocol: str) -> str:
    match = re.search(r"[^.\n]{0,80}(?:irradiat\w*|\d{3,4}\s*nm|LED)[^.\n]{0,80}", protocol, re.I)
    return match.group(0).strip() if match else "Batch protocol explicitly requires irradiation."


def _separation_evidence(text: str) -> str:
    return "The frozen protocol or hard constraints explicitly require separate liquid feeds."


def _flatten_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return str(value)


def _decision(action: str, stage: int, stream: str, basis: str) -> dict[str, Any]:
    return {"decision": action, "stage": stage, "stream": stream, "basis": basis}
