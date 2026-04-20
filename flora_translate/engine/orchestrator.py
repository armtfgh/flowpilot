"""
FLORA ENGINE v2 — Phase-based deliberation orchestrator.

Phase 0 — Triage (code, no LLM)
    Calculator runs → TriageReport classifies each domain GREEN / YELLOW / RED.
    Agents receive directed questions, not blank slates.

Phase 1 — Parallel review (LLM)
    Track A: Dr. Chemistry  [+ Dr. Photonics if light_required]
    Track B: Dr. Kinetics + Dr. Fluidics (parallel)
    Track C: Dr. Safety (independent, BLOCKING authority)

    GREEN agents: one-line confirmation, no LLM call.
    Agents with NEEDS_REVISION enter Phase 2.
    If < 2 agents flag issues → consensus immediately, skip Phase 2.

Phase 2 — Priority-ladder resolution (code + 1 LLM call for explanation)
    Conflict on same field → highest-priority agent wins.
    Priority: Safety > Chemistry > Fluidics > Kinetics
    Chief Engineer (LLM): writes explanation only, applies nothing.
    Calculator re-runs after every change.

Phase 3 — Implementation (sequential LLM)
    Dr. Hardware → Dr. Integration (reads Hardware output)

Phase 4 — Design envelope (code)
    Calculator sweep ±30% → feasible operating window.

Phase 5 — Dr. Failure (advisory, one shot)
    Top 3 failure modes. Never blocks. Never proposes changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from flora_translate.design_calculator import DesignCalculator, DesignCalculations
from flora_translate.engine.llm_agents import call_llm, get_max_rounds
from flora_translate.engine.triage import generate_triage, TriageReport
from flora_translate.engine.tools import compute_design_envelope
from flora_translate.engine.agents_v2 import (
    run_kinetics_v2,
    run_fluidics_v2,
    run_safety_v2,
    run_chemistry_v2,
    run_photonics_v2,
    run_hardware_v2,
    run_integration_v2,
    run_failure_v2,
)
from flora_translate.schemas import (
    AgentDeliberation,
    BatchRecord,
    ChemistryPlan,
    CouncilMessage,
    DeliberationLog,
    DesignCandidate,
    FieldProposal,
    FlowProposal,
    LabInventory,
    SanityCheckResult,
)

logger = logging.getLogger("flora.engine.orchestrator")

# Priority ladder for conflict resolution (higher index = lower priority)
_PRIORITY_ORDER = ["SafetyV2", "PhotonicsV2", "ChemistryV2", "FluidicsV2", "KineticsV2"]

# Fields the priority ladder may change
_ALLOWED_CHANGE_FIELDS = {
    "residence_time_min", "flow_rate_mL_min", "temperature_C",
    "concentration_M", "BPR_bar", "tubing_material", "tubing_ID_mm",
    "wavelength_nm", "deoxygenation_method", "mixer_type",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Priority ladder (deterministic, no LLM decision)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_priority_ladder(
    deliberations: list[AgentDeliberation],
) -> dict[str, str]:
    """Resolve field conflicts deterministically by agent priority.

    When multiple agents propose changes to the same field, the
    highest-priority agent (lowest index in _PRIORITY_ORDER) wins.
    Returns a flat dict of {field: value} changes to apply.
    """
    # Collect all proposals with priority
    proposals_by_field: dict[str, list[tuple[int, str, str]]] = {}
    for d in deliberations:
        if d.status != "REVISE":
            continue
        priority = next(
            (i for i, name in enumerate(_PRIORITY_ORDER) if name == d.agent),
            len(_PRIORITY_ORDER),  # lower priority if not in ladder
        )
        for p in d.proposals:
            if hasattr(p, "field") and p.field in _ALLOWED_CHANGE_FIELDS:
                proposals_by_field.setdefault(p.field, []).append(
                    (priority, p.value, d.agent_display_name)
                )

    # Resolve: pick lowest priority index (= highest priority agent)
    resolved: dict[str, str] = {}
    for field, candidates in proposals_by_field.items():
        candidates.sort(key=lambda x: x[0])
        winning_priority, winning_value, winning_agent = candidates[0]
        resolved[field] = winning_value
        if len(candidates) > 1:
            losers = [f"{a}: {v}" for _, v, a in candidates[1:]]
            logger.info(
                "    Conflict on '%s': %s wins over [%s]",
                field, winning_agent, ", ".join(losers),
            )
        else:
            logger.info("    Applying '%s' = %s (from %s)", field, winning_value, winning_agent)

    return resolved


def _enforce_tau_anchor(
    phase1: list[AgentDeliberation],
    current: FlowProposal,
    calc: DesignCalculations,
) -> dict[str, str]:
    """Enforce τ_final = max(all_tau_proposals, τ_lit/2) as a hard code guarantee."""
    tau_proposals = []

    # Collect all τ proposals from any agent
    for d in phase1:
        for p in d.proposals:
            if getattr(p, "field", None) == "residence_time_min":
                try:
                    tau_proposals.append(float(p.value))
                except (ValueError, TypeError):
                    pass

    # τ_lit anchor: τ_proposed ≥ τ_lit / 2
    # Use tau_analogy_min if analogies were found; fall back to tau_class_min otherwise.
    # tau_class_min is now conservative (photoredox class IF = 6×, not 48×), so
    # using it as fallback correctly prevents τ << τ_class from slipping through.
    tau_lit = getattr(calc, "tau_analogy_min", None) or getattr(calc, "tau_class_min", None)
    if tau_lit and tau_lit > 0:
        tau_anchor = tau_lit / 2.0
        tau_proposals.append(tau_anchor)
        logger.info("    τ-anchor: τ_lit = %.1f min → floor = %.1f min", tau_lit, tau_anchor)

    if not tau_proposals:
        return {}

    tau_final = max(tau_proposals)
    if tau_final > current.residence_time_min * 1.02:  # only update if >2% change
        logger.info(
            "    τ-anchor enforcement: %.1f min → %.1f min (max of proposals + lit anchor)",
            current.residence_time_min, tau_final,
        )
        return {"residence_time_min": str(round(tau_final, 1))}
    return {}


_CHIEF_SYSTEM = """\
You are the Chief Engineer on the FLORA ENGINE council.
You receive the resolution decisions that were already made by the priority ladder.
Your job is to write a clear, concise explanation of what changed and why.
Do NOT add new proposals or override any decision.

Output JSON:
{
  "explanation": "2-3 sentence explanation of all changes applied and the reasoning.",
  "conflicts_resolved": ["field: winner beat loser because ..."],
  "advisory_notes": ["any cross-domain notes for the lab chemist"]
}
"""


def _run_chief_explanation(
    deliberations: list[AgentDeliberation],
    changes_applied: dict[str, str],
) -> SanityCheckResult:
    """Chief Engineer writes the explanation — makes no decisions."""
    if not changes_applied:
        return SanityCheckResult(
            round=1,
            consistent=True,
            chain_of_thought="No changes needed — all agents approved or priority ladder had no conflicts.",
        )

    # Build context for explanation
    proposals_summary = []
    for d in deliberations:
        if d.status == "REVISE":
            for p in d.proposals:
                if hasattr(p, "field") and p.field:
                    proposals_summary.append(
                        f"{d.agent_display_name} proposed {p.field} → {p.value}: {p.reason}"
                    )

    context = (
        "## Changes applied by priority ladder:\n"
        + json.dumps(changes_applied, indent=2)
        + "\n\n## All agent proposals received:\n"
        + "\n".join(proposals_summary)
    )

    try:
        raw = call_llm(_CHIEF_SYSTEM, context, max_tokens=600)
        if "```" in raw:
            parts = raw.split("```")
            for i, p in enumerate(parts):
                if i % 2 == 1:
                    try:
                        data = json.loads(p.lstrip("json").strip())
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                data = {}
        else:
            try:
                data = json.loads(raw.strip())
            except json.JSONDecodeError:
                data = {}

        return SanityCheckResult(
            round=1,
            consistent=True,
            chain_of_thought=data.get("explanation", ""),
            conflicts_found=data.get("conflicts_resolved", []),
            resolutions=data.get("advisory_notes", []),
            final_changes=changes_applied,
        )
    except Exception as e:
        logger.warning("Chief Engineer explanation failed: %s", e)
        return SanityCheckResult(
            round=1,
            consistent=True,
            chain_of_thought=f"Priority ladder applied {len(changes_applied)} change(s): {changes_applied}",
            final_changes=changes_applied,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Proposal patcher
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_changes(proposal: FlowProposal, changes: dict[str, str]) -> FlowProposal:
    """Apply priority-ladder changes to the proposal."""
    if not changes:
        return proposal

    data = proposal.model_dump()
    for field, value_str in changes.items():
        if field not in data:
            continue
        try:
            current = data[field]
            if isinstance(current, float):
                data[field] = float(value_str)
            elif isinstance(current, int):
                data[field] = int(float(value_str))
            else:
                data[field] = value_str
        except (ValueError, TypeError):
            data[field] = value_str

    # Enforce V_R = τ × Q
    tau = data.get("residence_time_min", 0)
    Q = data.get("flow_rate_mL_min", 0)
    if tau > 0 and Q > 0:
        data["reactor_volume_mL"] = round(tau * Q, 4)

    return FlowProposal(**data)


# ═══════════════════════════════════════════════════════════════════════════════
#  Legacy compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def _to_legacy_messages(deliberations: list[AgentDeliberation]) -> list[CouncilMessage]:
    messages = []
    for d in deliberations:
        fields = [p.field for p in d.proposals if hasattr(p, "field") and p.field]
        value = " | ".join(f"{p.field}→{p.value}" for p in d.proposals if p.field)
        concern = d.concerns[0] if d.concerns else (d.findings[0] if d.findings else "")
        status_map = {"ACCEPT": "ACCEPT", "WARNING": "WARNING", "REVISE": "REJECT"}
        messages.append(CouncilMessage(
            agent=d.agent,
            status=status_map.get(d.status, d.status),
            field=", ".join(fields[:3]) or "general",
            value=value[:200],
            concern=concern[:200],
            revision_required=(d.status == "REVISE"),
            suggested_revision=f"{fields[0]}={d.proposals[0].value}" if fields else None,
        ))
    return messages


def _build_safety_report(deliberations: list[AgentDeliberation]) -> dict:
    safety = [d for d in deliberations if "Safety" in d.agent]
    return {
        "total_checks": len(safety),
        "accepts": sum(1 for s in safety if s.status == "ACCEPT"),
        "warnings": sum(1 for s in safety if s.status == "WARNING"),
        "rejects": sum(1 for s in safety if s.status == "REVISE"),
        "details": [{"concern": c} for s in safety for c in s.concerns[:3]],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """Phase-based multi-agent deliberation orchestrator (v2).

    Replaces the round-robin committee model with a tiered pipeline:
    Triage → Parallel review → Priority resolution → Implementation → Envelope → Failure.
    """

    def run(
        self,
        proposal: FlowProposal,
        batch_record: BatchRecord,
        analogies: list[dict],
        inventory: LabInventory,
        chemistry_plan: ChemistryPlan | None = None,
        calculations: DesignCalculations | None = None,
    ) -> tuple[DesignCandidate, DesignCalculations]:

        current = proposal.model_copy(deep=True)
        log = DeliberationLog()
        all_deliberations: list[AgentDeliberation] = []

        # ── Ensure initial calculations ───────────────────────────────────────
        if calculations is None:
            calculations = DesignCalculator().run(
                batch_record, chemistry_plan, current, inventory,
                analogies=analogies,
                target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                target_tubing_ID_mm=current.tubing_ID_mm or None,
            )
        calc = calculations

        # ═════════════════════════════════════════════════════════════════════
        #  PHASE 0 — Triage (code, no LLM)
        # ═════════════════════════════════════════════════════════════════════
        logger.info("  Phase 0 — Triage")
        triage = generate_triage(current, chemistry_plan, calc)
        logger.info(
            "    Green: %s | Flagged: %s",
            triage.green_domains, triage.flagged_domains,
        )

        # ═════════════════════════════════════════════════════════════════════
        #  PHASE 1 — Multi-round parallel review
        #  Rounds are configured per-provider in ENGINE_MAX_ROUNDS (config.py).
        #  Local/weaker models use more rounds to compensate for per-call
        #  reasoning limitations. GREEN domains are skipped every round.
        # ═════════════════════════════════════════════════════════════════════
        max_rounds = get_max_rounds()
        logger.info("  Phase 1 — Parallel review (%d round(s) configured)", max_rounds)

        phase1: list[AgentDeliberation] = []
        prior_findings: list[tuple[str, dict]] | None = None  # populated from round 1 for round 2+

        for round_num in range(1, max_rounds + 1):
            logger.info("    Round %d / %d", round_num, max_rounds)
            round_delibs: list[AgentDeliberation] = []

            # Track A: Chemistry (+ Photonics if needed)
            chem = run_chemistry_v2(current, chemistry_plan, calc, triage, prior=prior_findings)
            round_delibs.append(chem)
            logger.info("      Dr. Chemistry: %s", chem.status)

            if triage.light_required:
                phot = run_photonics_v2(current, chemistry_plan, calc, triage, prior=prior_findings)
                round_delibs.append(phot)
                logger.info("      Dr. Photonics: %s", phot.status)

            # Track B: Kinetics first, then Fluidics with kinetics context injected
            kin = run_kinetics_v2(current, chemistry_plan, calc, triage, prior=prior_findings)
            round_delibs.append(kin)
            logger.info("      Dr. Kinetics: %s", kin.status)

            # Build kinetics context for Fluidics
            kin_tau_proposed = next(
                (p.value for p in kin.proposals if getattr(p, "field", None) == "residence_time_min"),
                str(current.residence_time_min)
            )
            kin_context = (prior_findings or []) + [(kin.agent_display_name, {
                "verdict": {"ACCEPT": "APPROVED", "WARNING": "APPROVED_WITH_CONDITIONS",
                            "REVISE": "NEEDS_REVISION"}.get(kin.status, kin.status),
                "summary": kin.chain_of_thought[:300],
                "tau_kinetics_proposed": kin_tau_proposed,
            })]
            flu = run_fluidics_v2(current, chemistry_plan, calc, triage, prior=kin_context)
            round_delibs.append(flu)
            logger.info("      Dr. Fluidics: %s", flu.status)

            # Track C: Safety (independent, blocking authority)
            saf = run_safety_v2(current, chemistry_plan, calc, triage, prior=prior_findings)
            round_delibs.append(saf)
            logger.info("      Dr. Safety: %s", saf.status)

            log.rounds.append(round_delibs)
            phase1 = round_delibs  # keep latest round as phase1

            # Safety BLOCK — apply immediately regardless of round
            if saf.status == "REVISE" and saf.proposals:
                safety_changes = {
                    p.field: p.value for p in saf.proposals
                    if hasattr(p, "field") and p.field in _ALLOWED_CHANGE_FIELDS
                }
                if safety_changes:
                    logger.info("      Safety BLOCK — applying: %s", safety_changes)
                    current = _apply_changes(current, safety_changes)
                    log.all_changes_applied.update(safety_changes)
                    calc = DesignCalculator().run(
                        batch_record, chemistry_plan, current, inventory,
                        analogies=analogies,
                        target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                        target_tubing_ID_mm=current.tubing_ID_mm or None,
                        target_residence_time_min=current.residence_time_min or None,
                    )
                    triage = generate_triage(current, chemistry_plan, calc)

            # Check convergence: stop early if no non-safety REVISE remains.
            # Rule: never exit before round 2 when max_rounds >= 2.
            # Rationale: round 1 may have applied Q/d changes (geometry, mixing)
            # that shift Re, ΔP, Beer-Lambert — agents MUST see updated values in
            # round 2 before we can declare convergence.
            revise_agents = [d for d in round_delibs
                             if d.status == "REVISE" and "Safety" not in d.agent]

            # Detect inter-round changes to apply before deciding convergence
            inter_changes: dict[str, str] = {}
            if round_num < max_rounds:
                inter_changes = _apply_priority_ladder(round_delibs)
                if inter_changes:
                    current = _apply_changes(current, inter_changes)
                    log.all_changes_applied.update(inter_changes)
                    calc = DesignCalculator().run(
                        batch_record, chemistry_plan, current, inventory,
                        analogies=analogies,
                        target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                        target_tubing_ID_mm=current.tubing_ID_mm or None,
                        target_residence_time_min=current.residence_time_min or None,
                    )
                    triage = generate_triage(current, chemistry_plan, calc)
                    logger.info("      Applied %d inter-round change(s)", len(inter_changes))

                # Build prior context for next round (always, for continuity)
                prior_findings = [
                    (d.agent_display_name, {
                        "verdict": {"ACCEPT": "APPROVED", "WARNING": "APPROVED_WITH_CONDITIONS",
                                    "REVISE": "NEEDS_REVISION"}.get(d.status, d.status),
                        "summary": d.chain_of_thought[:200],
                    })
                    for d in round_delibs
                ]

            # Convergence decision:
            # - Never exit on round 1 alone when max_rounds >= 2. Round 1 clears
            #   the obvious flags; round 2 verifies the corrected design is clean.
            # - If inter_changes were applied, we must run at least one more round
            #   regardless (agents need to see the corrected values).
            converged = (not revise_agents) and (not inter_changes)
            must_continue = (round_num == 1 and max_rounds >= 2)

            if converged and not must_continue:
                logger.info("      No REVISE agents, no inter-round changes — converged at round %d",
                            round_num)
                break
            elif not revise_agents and not inter_changes:
                logger.info("      No REVISE agents (round %d of %d minimum — continuing)",
                            round_num, max_rounds)

        all_deliberations.extend(phase1)
        log.total_rounds = min(round_num, max_rounds)

        # ═════════════════════════════════════════════════════════════════════
        #  PHASE 2 — Final priority ladder resolution (after all rounds)
        # ═════════════════════════════════════════════════════════════════════
        revise_agents = [d for d in phase1 if d.status == "REVISE" and "Safety" not in d.agent]
        needs_phase2 = bool(revise_agents)

        if needs_phase2:
            logger.info("  Phase 2 — Priority ladder resolution (%d agents still flagged)",
                        len(revise_agents))
            tau_anchor_changes = _enforce_tau_anchor(phase1, current, calc)
            changes = _apply_priority_ladder(phase1)
            # τ-anchor overrides priority ladder for residence_time_min
            if tau_anchor_changes:
                changes.update(tau_anchor_changes)

            if changes:
                sanity = _run_chief_explanation(phase1, changes)
                log.sanity_checks.append(sanity)
                current = _apply_changes(current, changes)
                log.all_changes_applied.update(changes)
                calc = DesignCalculator().run(
                    batch_record, chemistry_plan, current, inventory,
                    analogies=analogies,
                    target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                    target_tubing_ID_mm=current.tubing_ID_mm or None,
                    target_residence_time_min=current.residence_time_min or None,
                )
                triage = generate_triage(current, chemistry_plan, calc)
                logger.info("    Applied %d final change(s)", len(changes))
            else:
                logger.info("    No remaining conflicts")
        else:
            logger.info("  Phase 2 — skipped (all agents converged)")
            # Still enforce τ-anchor even when no agents flagged REVISE
            tau_anchor_changes = _enforce_tau_anchor(phase1, current, calc)
            if tau_anchor_changes:
                logger.info("  Phase 2 — τ-anchor applied despite convergence: %s", tau_anchor_changes)
                current = _apply_changes(current, tau_anchor_changes)
                log.all_changes_applied.update(tau_anchor_changes)
                calc = DesignCalculator().run(
                    batch_record, chemistry_plan, current, inventory,
                    analogies=analogies,
                    target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                    target_tubing_ID_mm=current.tubing_ID_mm or None,
                    target_residence_time_min=current.residence_time_min or None,
                )
                triage = generate_triage(current, chemistry_plan, calc)

        log.consensus_reached = True

        # ═════════════════════════════════════════════════════════════════════
        #  PHASE 3 — Implementation (Hardware → Integration)
        # ═════════════════════════════════════════════════════════════════════
        logger.info("  Phase 3 — Implementation")
        hw = run_hardware_v2(current, chemistry_plan, calc, triage)
        all_deliberations.append(hw)
        logger.info("    Dr. Hardware: %s", hw.status)

        integ = run_integration_v2(current, chemistry_plan, calc, triage, hardware_findings=hw)
        all_deliberations.append(integ)
        logger.info("    Dr. Integration: %s", integ.status)

        # ═════════════════════════════════════════════════════════════════════
        #  PHASE 4 — Design envelope (code)
        # ═════════════════════════════════════════════════════════════════════
        logger.info("  Phase 4 — Design envelope")
        try:
            envelope = compute_design_envelope(
                tau_center_min=current.residence_time_min,
                Q_center_mL_min=current.flow_rate_mL_min,
                ID_center_mm=current.tubing_ID_mm,
                T_center_C=current.temperature_C,
                BPR_center_bar=current.BPR_bar,
                solvent=(
                    chemistry_plan.stages[0].solvent
                    if chemistry_plan and chemistry_plan.stages
                    else (current.streams[0].solvent if current.streams else "EtOH")
                ),
                is_gas_liquid=triage.is_gas_liquid,
                pump_max_bar=calc.pump_max_bar or 20.0,
            )
        except Exception as e:
            logger.warning("Design envelope computation failed: %s", e)
            envelope = {}

        # ═════════════════════════════════════════════════════════════════════
        #  PHASE 5 — Dr. Failure (advisory)
        # ═════════════════════════════════════════════════════════════════════
        logger.info("  Phase 5 — Dr. Failure (advisory)")
        failure = run_failure_v2(current, chemistry_plan, calc, triage)
        all_deliberations.append(failure)

        # ═════════════════════════════════════════════════════════════════════
        #  Build summary and output
        # ═════════════════════════════════════════════════════════════════════
        log.summary = self._build_summary(log, all_deliberations, envelope, triage)

        legacy_messages = _to_legacy_messages(all_deliberations)
        safety_report = _build_safety_report(all_deliberations)

        current.engine_validated = True
        current.safety_flags = [c for d in all_deliberations
                                 if "Safety" in d.agent for c in d.concerns[:3]]

        # Unit ops from Integration agent
        unit_ops = integ.findings[:10] if integ.findings else []
        pid = integ.chain_of_thought[:500] if integ.chain_of_thought else ""

        # Store envelope in deliberation log summary for downstream use
        if envelope:
            log.summary += (
                "\n\n## Design Envelope\n```json\n"
                + json.dumps(envelope, indent=2, default=str)
                + "\n```"
            )

        return DesignCandidate(
            proposal=current,
            chemistry_plan=chemistry_plan,
            council_messages=legacy_messages,
            council_rounds=log.total_rounds,
            safety_report=safety_report,
            unit_operations=unit_ops,
            pid_description=pid,
            deliberation_log=log,
        ), calc

    def _build_summary(
        self,
        log: DeliberationLog,
        deliberations: list[AgentDeliberation],
        envelope: dict,
        triage: TriageReport,
    ) -> str:
        lines = ["## ENGINE Council v2 — Deliberation Summary"]
        lines.append(f"\n**Triage**: Green: {triage.green_domains} | Flagged: {triage.flagged_domains}")

        lines.append("\n### Phase 1 — Agent Verdicts")
        for d in deliberations:
            icon = {"ACCEPT": "✅", "WARNING": "⚠️", "REVISE": "🔄"}.get(d.status, "•")
            lines.append(f"- {icon} **{d.agent_display_name}**: {d.chain_of_thought[:200]}")

        if log.all_changes_applied:
            lines.append("\n### Changes Applied")
            for field, value in log.all_changes_applied.items():
                original = "★"
                lines.append(f"- {original} `{field}` → `{value}`")

        if envelope:
            lines.append("\n### Design Envelope (feasible operating window)")
            for param, bounds in envelope.items():
                if isinstance(bounds, dict) and "min" in bounds:
                    unit = bounds.get("unit", "")
                    lines.append(
                        f"- **{param}**: {bounds['min']} – {bounds['center']} – {bounds['max']} {unit}"
                    )

        return "\n".join(lines)
