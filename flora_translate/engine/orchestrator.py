"""
ENGINE — Deliberation Orchestrator.

Coordinates the multi-agent engineering council through structured debate:

  Round 1 (Independent Analysis):
    All 5 specialist agents analyze the proposal independently.
    Each produces chain-of-thought reasoning, calculations, and proposals.

  Round 2 (Debate):
    Each agent sees ALL other agents' Round 1 findings.
    Agents can agree, disagree, or refine based on cross-domain insights.
    The orchestrator collects all proposals.

  Sanity Check (after each round):
    A central sanity-checker validates cross-agent consistency:
    - Do the kinetics and fluidics agents agree on τ?
    - Does the safety agent's BPR match the fluidics pressure drop?
    - Are chemistry stream assignments consistent with fluidics mixer choice?
    Resolves conflicts and applies consensus changes to the proposal.

  Round 3 (if needed):
    Only runs if the sanity check found unresolved conflicts.

The DesignCalculator runs after every proposal change to maintain
physics consistency (τ×Q=V, Re, ΔP, BPR).

Full chain-of-thought is logged in a DeliberationLog for transparency.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict

import anthropic

from flora_translate.config import TRANSLATION_MODEL
from flora_translate.design_calculator import DesignCalculator, DesignCalculations
from flora_translate.engine.llm_agents import (
    AGENT_NAMES,
    run_chemistry,
    run_fluidics,
    run_integration,
    run_kinetics,
    run_safety,
)
from flora_translate.schemas import (
    AgentDeliberation,
    BatchRecord,
    ChemistryPlan,
    CouncilMessage,
    DeliberationLog,
    DesignCandidate,
    FlowProposal,
    LabInventory,
    SanityCheckResult,
)

logger = logging.getLogger("flora.engine.orchestrator")

MAX_DELIBERATION_ROUNDS = 3


# ═══════════════════════════════════════════════════════════════════════════════
#  Sanity Check Agent — the central consistency validator
# ═══════════════════════════════════════════════════════════════════════════════

_SANITY_SYSTEM = """\
You are the **Chief Engineer**, the central sanity checker for the FLORA ENGINE \
council. You receive findings and structured proposals from 5 specialist agents.

## Your job
1. **Review all agent proposals**: Each proposal is a structured field change \
   (field name → value). Decide which to ACCEPT, which to REJECT.
2. **Check cross-agent consistency**: Do proposals conflict? \
   (e.g., one agent wants τ=15 min, another wants tubing_ID_mm=0.5 which changes volume)
3. **Resolve conflicts** using priority: safety > chemistry > physics > performance.
4. **Output only the final field changes to apply.**

## CRITICAL CONSTRAINTS on final_changes
- Only include SIMPLE NUMERIC or STRING fields from FlowProposal.
- ALLOWED fields: residence_time_min, flow_rate_mL_min, temperature_C, \
  concentration_M, BPR_bar, tubing_material, tubing_ID_mm, wavelength_nm, \
  deoxygenation_method, mixer_type.
- DO NOT include: streams, pre_reactor_steps, post_reactor_steps, \
  reasoning_per_field, or any list/object field.
- reactor_volume_mL is auto-computed (τ×Q) — do not include it.
- All values must be STRINGS (e.g., "15.0" not 15.0).

## Output — JSON only
{
  "chain_of_thought": "Your analysis — which proposals to accept and why",
  "conflicts_found": ["Dr. Kinetics wants τ=15 but Dr. Fluidics wants τ=12", ...],
  "resolutions": ["Adopt τ=15 per Kinetics — conversion is the priority", ...],
  "final_changes": {"residence_time_min": "15.0", "tubing_ID_mm": "0.75"},
  "consistent": true | false,
  "needs_another_round": true | false
}
"""

# Fields the Chief Engineer is allowed to change
_ALLOWED_CHANGE_FIELDS = {
    "residence_time_min", "flow_rate_mL_min", "temperature_C",
    "concentration_M", "BPR_bar", "tubing_material", "tubing_ID_mm",
    "wavelength_nm", "deoxygenation_method", "mixer_type",
}


def _collect_all_proposals(deliberations: list[AgentDeliberation]) -> list[dict]:
    """Extract all structured proposals from all agents for the Chief Engineer."""
    all_proposals = []
    for d in deliberations:
        if d.status != "REVISE":
            continue
        for p in d.proposals:
            if hasattr(p, 'field') and p.field:
                all_proposals.append({
                    "agent": d.agent_display_name,
                    "field": p.field,
                    "value": p.value,
                    "reason": p.reason,
                })
    return all_proposals


def _run_sanity_check(
    round_num: int,
    deliberations: list[AgentDeliberation],
    proposal: FlowProposal,
    calculations_dict: dict | None,
) -> SanityCheckResult:
    """Run the central sanity check on all agents' findings."""
    logger.info("  Sanity Check — Round %d", round_num)

    # Collect structured proposals from REVISE agents
    all_proposals = _collect_all_proposals(deliberations)

    # If no agent proposed changes, skip the LLM call entirely
    if not all_proposals:
        logger.info("    No REVISE proposals — skipping Chief Engineer")
        return SanityCheckResult(
            round=round_num,
            consistent=True,
            chain_of_thought="All agents accepted the design. No changes needed.",
        )

    # Build context
    sections = [
        "## Current FlowProposal (key fields)\n```json\n"
        + json.dumps({
            k: v for k, v in proposal.model_dump().items()
            if k in _ALLOWED_CHANGE_FIELDS or k in ("reactor_volume_mL", "confidence")
        }, indent=2, default=str)
        + "\n```"
    ]

    sections.append("## Agent Proposals to Review")
    for p in all_proposals:
        sections.append(
            f"- **{p['agent']}** proposes `{p['field']}` = `{p['value']}` — {p['reason']}"
        )

    # Also include agent concerns for context
    sections.append("\n## Agent Concerns")
    for d in deliberations:
        if d.concerns:
            name = AGENT_NAMES.get(d.agent, d.agent)
            for c in d.concerns[:2]:
                sections.append(f"- **{name}**: {c}")

    context = "\n\n".join(sections)

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=TRANSLATION_MODEL,
            max_tokens=1500,
            system=_SANITY_SYSTEM,
            messages=[{"role": "user", "content": context}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        if "```" in raw:
            raw = raw.rsplit("```", 1)[0]
        data = json.loads(raw.strip())

        # Sanitize final_changes — only allow permitted fields with string values
        raw_changes = data.get("final_changes", {})
        clean_changes = {}
        for field, value in raw_changes.items():
            if field in _ALLOWED_CHANGE_FIELDS and isinstance(value, (str, int, float)):
                clean_changes[field] = str(value)
            else:
                logger.warning("    Chief Engineer tried to change '%s' — blocked", field)

        return SanityCheckResult(
            round=round_num,
            consistent=data.get("consistent", True),
            chain_of_thought=data.get("chain_of_thought", ""),
            conflicts_found=data.get("conflicts_found", []),
            resolutions=data.get("resolutions", []),
            final_changes=clean_changes,
        )
    except Exception as e:
        logger.error("Sanity check failed: %s", e)
        # FALLBACK: apply agent proposals directly if Chief Engineer fails
        logger.info("    Falling back to direct agent proposal application")
        fallback_changes = {}
        for p in all_proposals:
            if p["field"] in _ALLOWED_CHANGE_FIELDS:
                fallback_changes[p["field"]] = str(p["value"])
        return SanityCheckResult(
            round=round_num,
            consistent=bool(not fallback_changes),
            chain_of_thought=f"Chief Engineer error ({e}). Applied agent proposals directly.",
            conflicts_found=[],
            resolutions=[f"Direct application of {len(fallback_changes)} agent proposal(s)"],
            final_changes=fallback_changes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Proposal patcher — applies sanity check changes
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_changes(proposal: FlowProposal, changes: dict[str, str]) -> FlowProposal:
    """Apply sanity check's final_changes to the proposal."""
    if not changes:
        return proposal

    data = proposal.model_dump()
    for field, value_str in changes.items():
        if field not in data:
            continue
        # Try numeric conversion
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

    # Enforce V_R = τ × Q consistency
    if data.get("residence_time_min", 0) > 0 and data.get("flow_rate_mL_min", 0) > 0:
        data["reactor_volume_mL"] = round(
            data["residence_time_min"] * data["flow_rate_mL_min"], 4
        )

    return FlowProposal(**data)


# ═══════════════════════════════════════════════════════════════════════════════
#  Convert deliberation to legacy CouncilMessages (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def _deliberation_to_legacy(log: DeliberationLog) -> list[CouncilMessage]:
    """Convert deliberation log to legacy CouncilMessage format for UI compat."""
    messages = []
    for round_deliberations in log.rounds:
        for d in round_deliberations:
            status = d.status
            if status == "REVISE":
                status = "REJECT"
            concern = ""
            if d.concerns:
                concern = d.concerns[0]
            elif d.findings:
                concern = d.findings[0]

            # Extract field info from structured proposals
            value = ""
            fields = []
            suggested = None
            for p in d.proposals:
                if hasattr(p, 'field') and p.field:
                    fields.append(p.field)
                    value = f"{p.field} → {p.value}"
                    suggested = f"{p.field}={p.value}"

            messages.append(CouncilMessage(
                agent=d.agent,
                status=status,
                field=", ".join(fields[:3]) if fields else "general",
                value=value[:200],
                concern=concern[:200],
                revision_required=(d.status == "REVISE"),
                suggested_revision=suggested,
            ))
    return messages


def _build_safety_report(log: DeliberationLog) -> dict:
    """Build safety report from the Safety agent's deliberations."""
    safety_items = []
    for round_deliberations in log.rounds:
        for d in round_deliberations:
            if d.agent == "SafetySpecialist":
                safety_items.append(d)

    return {
        "total_checks": len(safety_items),
        "accepts": sum(1 for s in safety_items if s.status == "ACCEPT"),
        "warnings": sum(1 for s in safety_items if s.status == "WARNING"),
        "rejects": sum(1 for s in safety_items if s.status == "REVISE"),
        "details": [
            {"concern": c, "status": safety_items[0].status if safety_items else "?"}
            for s in safety_items
            for c in s.concerns[:3]
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """Multi-agent deliberation orchestrator.

    Replaces the rule-based Moderator with LLM-powered specialist agents
    that reason, debate, and converge on a consensus design.
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
        calc = calculations
        log = DeliberationLog()

        # Ensure we have initial calculations
        if calc is None:
            calc = DesignCalculator().run(
                batch_record, chemistry_plan, current, inventory,
                analogies=analogies,
                target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                target_tubing_ID_mm=current.tubing_ID_mm or None,
            )

        calc_dict = asdict(calc)
        prev_round_findings: list[AgentDeliberation] | None = None

        for round_num in range(1, MAX_DELIBERATION_ROUNDS + 1):
            logger.info("  ENGINE Deliberation — Round %d / %d",
                        round_num, MAX_DELIBERATION_ROUNDS)

            # ── Run all 5 agents ───────────────────────────────────────────
            round_deliberations: list[AgentDeliberation] = []

            # Build lookup of this agent's prior deliberation (for Round 2+)
            prior_by_agent: dict[str, AgentDeliberation] = {}
            if prev_round_findings:
                prior_by_agent = {d.agent: d for d in prev_round_findings}

            _run_fns = [
                ("KineticsSpecialist", run_kinetics),
                ("FluidicsSpecialist", run_fluidics),
                ("SafetySpecialist",   run_safety),
                ("ChemistrySpecialist", run_chemistry),
                ("IntegrationSpecialist", run_integration),
            ]
            for agent_key, run_fn in _run_fns:
                delib = run_fn(
                    current, chemistry_plan, calc_dict,
                    round_num=round_num,
                    others=prev_round_findings,
                    prior=prior_by_agent.get(agent_key),  # own prior concerns
                )
                round_deliberations.append(delib)
                error_flag = " [ERROR]" if delib.had_error else ""
                logger.info("    %s: %s%s (%d proposals, %d concerns)",
                            delib.agent_display_name, delib.status, error_flag,
                            len(delib.proposals), len(delib.concerns))

            log.rounds.append(round_deliberations)

            # ── Sanity check — cross-agent consistency ─────────────────────
            sanity = _run_sanity_check(round_num, round_deliberations,
                                       current, calc_dict)
            log.sanity_checks.append(sanity)

            logger.info("    Sanity check: consistent=%s, %d changes, %d conflicts",
                        sanity.consistent, len(sanity.final_changes),
                        len(sanity.conflicts_found))

            # ── Apply consensus changes ────────────────────────────────────
            has_revise = any(d.status == "REVISE" for d in round_deliberations)
            has_error  = any(d.had_error for d in round_deliberations)
            changes_applied = False

            if sanity.final_changes:
                old_proposal = current.model_dump()
                current = _apply_changes(current, sanity.final_changes)
                new_proposal = current.model_dump()
                changes_applied = any(
                    str(old_proposal.get(k)) != str(new_proposal.get(k))
                    for k in sanity.final_changes
                )
                if changes_applied:
                    # Accumulate all changes across rounds
                    log.all_changes_applied.update(sanity.final_changes)
                # Re-run calculator for consistency
                # Pass council-approved τ so the calculator uses it directly
                # instead of re-deriving from batch kinetics.
                calc = DesignCalculator().run(
                    batch_record, chemistry_plan, current, inventory,
                    analogies=analogies,
                    target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                    target_tubing_ID_mm=current.tubing_ID_mm or None,
                    target_residence_time_min=current.residence_time_min or None,
                )
                calc_dict = asdict(calc)
                logger.info("    Applied %d changes (%s actually modified)",
                            len(sanity.final_changes),
                            "values" if changes_applied else "no values")

            # ── Check convergence ──────────────────────────────────────────
            # NEVER converge when any agent had an error — their analysis is
            # incomplete and cannot be treated as a pass.
            if has_error:
                logger.info("    %d agent error(s) — cannot converge, continuing",
                            sum(1 for d in round_deliberations if d.had_error))
                prev_round_findings = round_deliberations
                continue

            all_accept = all(d.status == "ACCEPT" for d in round_deliberations)

            if all_accept:
                logger.info("    All agents ACCEPT — converged at round %d", round_num)
                log.consensus_reached = True
                break
            elif round_num >= 2 and changes_applied:
                logger.info("    Changes applied in round %d — converged", round_num)
                log.consensus_reached = True
                break
            elif round_num >= 2 and not has_revise:
                # Only WARNING flags, no REVISE, no errors — acceptable convergence
                logger.info("    Only WARNING flags, no REVISE — converged at round %d",
                            round_num)
                log.consensus_reached = True
                break

            # Prepare for next round — agents see this round's findings
            prev_round_findings = round_deliberations

        log.total_rounds = round_num

        # ── Build summary ──────────────────────────────────────────────────
        log.summary = self._build_summary(log)

        # ── Build legacy structures for backward compat ────────────────────
        legacy_messages = _deliberation_to_legacy(log)
        safety_report = _build_safety_report(log)

        # Process Architect operations (from the Integration agent or fallback)
        from flora_translate.engine.process_architect import ProcessArchitectAgent
        pa = ProcessArchitectAgent()
        unit_ops, pid = pa.build_operations(current, batch_record)

        current.engine_validated = True
        current.safety_flags = []
        for round_delibs in log.rounds:
            for d in round_delibs:
                if d.agent == "SafetySpecialist" and d.concerns:
                    current.safety_flags.extend(d.concerns[:3])

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

    def _build_summary(self, log: DeliberationLog) -> str:
        """Build a human-readable summary of the deliberation."""
        lines = [f"## ENGINE Council Deliberation — {log.total_rounds} Rounds"]

        for i, round_delibs in enumerate(log.rounds, 1):
            lines.append(f"\n### Round {i}")
            for d in round_delibs:
                icon = {"ACCEPT": "✅", "WARNING": "⚠️", "REVISE": "🔄"}.get(d.status, "•")
                lines.append(f"- {icon} **{d.agent_display_name}** ({d.status})")
                if d.findings:
                    lines.append(f"  - {d.findings[0]}")
                if d.proposals:
                    lines.append(f"  - Proposes: {d.proposals[0]}")

            if i <= len(log.sanity_checks):
                sc = log.sanity_checks[i - 1]
                if sc.conflicts_found:
                    lines.append(f"- **Sanity Check:** {len(sc.conflicts_found)} conflict(s)")
                    for c in sc.conflicts_found[:3]:
                        lines.append(f"  - {c}")
                if sc.resolutions:
                    for r in sc.resolutions[:3]:
                        lines.append(f"  - Resolution: {r}")

        if log.consensus_reached:
            lines.append(f"\n**Consensus reached after {log.total_rounds} rounds.**")
        else:
            lines.append(f"\n**Max rounds ({log.total_rounds}) reached.**")

        return "\n".join(lines)
