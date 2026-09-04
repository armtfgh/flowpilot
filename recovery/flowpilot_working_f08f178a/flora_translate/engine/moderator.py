"""ENGINE — Moderator: calculation-driven engineering validation council.

Every revision triggers a full re-calculation through the 9-step
DesignCalculator, guaranteeing internal consistency after each change.

Flow:
  1. Run DesignCalculator on the proposal
  2. Force-sync proposal with physics (τ×Q=V_R, BPR, tubing ID)
  3. Run domain agents — they validate against calculation results
  4. Apply agent revisions → re-run calculator → verify
  5. Iterate until convergence or max rounds
"""

import logging
import re

from flora_translate.config import MAX_COUNCIL_ROUNDS
from flora_translate.design_calculator import DesignCalculator, DesignCalculations
from flora_translate.engine.chemistry_validator import ChemistryValidator
from flora_translate.engine.fluidics_agent import FluidicsAgent
from flora_translate.engine.kinetics_agent import KineticsAgent
from flora_translate.engine.process_architect import ProcessArchitectAgent
from flora_translate.engine.safety_critic import SafetyCriticAgent
from flora_translate.schemas import (
    BatchRecord,
    ChemistryPlan,
    CouncilMessage,
    DesignCandidate,
    FlowProposal,
    LabInventory,
)

logger = logging.getLogger("flora.engine.moderator")


def _extract_number(text: str) -> float | None:
    m = re.search(r"[\d.]+", text or "")
    return float(m.group()) if m else None


def build_safety_report(messages: list[CouncilMessage]) -> dict:
    safety_msgs = [m for m in messages if m.agent == "SafetyCriticAgent"]
    return {
        "total_checks": len(safety_msgs),
        "accepts": len([m for m in safety_msgs if m.status == "ACCEPT"]),
        "warnings": len([m for m in safety_msgs if m.status == "WARNING"]),
        "rejects": len([m for m in safety_msgs if m.status == "REJECT"]),
        "details": [
            {"field": m.field, "status": m.status, "concern": m.concern}
            for m in safety_msgs if m.status != "ACCEPT"
        ],
    }


class Moderator:
    """Calculation-driven engineering validation council.

    The DesignCalculator is the source of truth for physics.
    Domain agents validate feasibility, safety, and chemistry.
    Every revision triggers a full re-calculation to guarantee
    internal consistency (τ=V/Q, L=4V/πd², Re=ρvd/μ, etc.).
    """

    def run(
        self,
        proposal: FlowProposal,
        batch_record: BatchRecord,
        analogies: list[dict],
        inventory: LabInventory,
        chemistry_plan: ChemistryPlan | None = None,
        calculations: DesignCalculations | None = None,
    ) -> DesignCandidate:
        all_messages: list[CouncilMessage] = []
        current = proposal.model_copy(deep=True)
        solvent = batch_record.solvent
        calc = calculations  # may be pre-computed

        for round_num in range(1, MAX_COUNCIL_ROUNDS + 1):
            logger.info(
                "  ENGINE Council — Round %d / %d", round_num, MAX_COUNCIL_ROUNDS
            )

            # ── Phase 1: Run design calculator ──────────────────────────
            calc = DesignCalculator().run(
                batch_record,
                chemistry_plan,
                current,
                inventory,
                analogies=analogies,
                target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                target_tubing_ID_mm=current.tubing_ID_mm or None,
            )

            # ── Phase 2: Force-sync proposal with calculations ──────────
            sync_msgs = self._sync_with_physics(current, calc)
            if sync_msgs:
                all_messages.extend(sync_msgs)
                # Apply sync corrections
                current = self._apply_sync(current, calc, sync_msgs)
                logger.info(
                    "    Physics sync: %d corrections applied", len(sync_msgs)
                )
                # Re-run calculator after sync to update all derived values
                calc = DesignCalculator().run(
                    batch_record, chemistry_plan, current, inventory,
                    analogies=analogies,
                    target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                    target_tubing_ID_mm=current.tubing_ID_mm or None,
                )

            # ── Phase 3: Run domain agents ──────────────────────────────
            round_msgs: list[CouncilMessage] = []

            # Chemistry Validator
            if chemistry_plan:
                chem_msgs = ChemistryValidator().run(current, chemistry_plan)
                round_msgs.extend(chem_msgs)
                for cm in chem_msgs:
                    logger.info("    Chemistry (%s): %s", cm.field, cm.status)

            # Kinetics Agent — uses full calculation context
            kin_msg = KineticsAgent().run(
                batch_record, current, analogies, calculations=calc
            )
            round_msgs.append(kin_msg)
            logger.info(
                "    Kinetics: %s — %s",
                kin_msg.status,
                kin_msg.concern[:80] if kin_msg.concern else "OK",
            )

            # Fluidics Agent — uses full calculation context
            flu_msg = FluidicsAgent().run(
                current, inventory, analogies, solvent=solvent, calculations=calc
            )
            round_msgs.append(flu_msg)
            logger.info("    Fluidics: %s", flu_msg.status)

            # Safety Critic
            safety_msgs = SafetyCriticAgent().run(current, inventory, batch_record)
            # If calculator says BPR required but proposal has none, inject REJECT
            if calc.bpr_required and (not current.BPR_bar or current.BPR_bar < calc.bpr_pressure_bar):
                safety_msgs.append(CouncilMessage(
                    agent="SafetyCriticAgent",
                    status="REJECT",
                    field="BPR_bar",
                    value=str(current.BPR_bar),
                    concern=(
                        f"Calculator: BPR required at {calc.bpr_pressure_bar:.1f} bar "
                        f"(solvent bp = {calc.solvent_boiling_point_C or '?'} °C, "
                        f"T = {calc.temperature_C} °C)"
                    ),
                    revision_required=True,
                    suggested_revision=f"{calc.bpr_pressure_bar:.1f}",
                ))
            round_msgs.extend(safety_msgs)
            for sm in safety_msgs:
                logger.info("    Safety (%s): %s", sm.field, sm.status)

            all_messages.extend(round_msgs)

            # ── Phase 4: Apply agent revisions ──────────────────────────
            revisions = [m for m in round_msgs if m.revision_required]
            if revisions:
                logger.info(
                    "    %d revision(s) — applying and re-calculating",
                    len(revisions),
                )
                current = self._apply_agent_revisions(current, revisions)
            elif round_num >= 2:
                logger.info(
                    "    Council converged after %d rounds", round_num
                )
                break
            else:
                logger.info("    Round %d clean — running refinement round", round_num)

        # ── Final: Process Architect ────────────────────────────────────
        pa = ProcessArchitectAgent()
        pa_msg = pa.run(current, batch_record)
        unit_ops, pid = pa.build_operations(current, batch_record)
        all_messages.append(pa_msg)

        safety_report = build_safety_report(all_messages)

        current.engine_validated = True
        current.safety_flags = [
            m.concern for m in all_messages
            if m.status in ("WARNING", "REJECT") and m.concern
        ]

        return DesignCandidate(
            proposal=current,
            chemistry_plan=chemistry_plan,
            council_messages=all_messages,
            council_rounds=round_num,
            safety_report=safety_report,
            unit_operations=unit_ops,
            pid_description=pid,
        ), calc

    # ── Physics synchronisation ─────────────────────────────────────────

    def _sync_with_physics(
        self, proposal: FlowProposal, calc: DesignCalculations
    ) -> list[CouncilMessage]:
        """Compare proposal with calculator and generate correction messages."""
        msgs: list[CouncilMessage] = []

        # Residence time: calculator is authoritative (from kinetics)
        if calc.residence_time_min > 0 and proposal.residence_time_min > 0:
            ratio = abs(
                proposal.residence_time_min - calc.residence_time_min
            ) / calc.residence_time_min
            if ratio > 0.20:
                msgs.append(CouncilMessage(
                    agent="DesignCalculator",
                    status="REJECT",
                    field="residence_time_min",
                    value=f"{proposal.residence_time_min:.1f} min",
                    concern=(
                        f"Kinetics gives τ = {calc.residence_time_min:.1f} min "
                        f"but proposal has {proposal.residence_time_min:.1f} min "
                        f"(Δ = {ratio*100:.0f}%)"
                    ),
                    revision_required=True,
                    suggested_revision=f"{calc.residence_time_min:.1f}",
                ))

        # Reactor volume: must equal τ × Q
        if proposal.flow_rate_mL_min > 0 and proposal.residence_time_min > 0:
            correct_vol = proposal.residence_time_min * proposal.flow_rate_mL_min
            if abs((proposal.reactor_volume_mL or 0) - correct_vol) > 0.1:
                msgs.append(CouncilMessage(
                    agent="DesignCalculator",
                    status="REJECT",
                    field="reactor_volume_mL",
                    value=f"{proposal.reactor_volume_mL:.2f} mL",
                    concern=(
                        f"V_R must equal τ×Q = {proposal.residence_time_min:.1f}"
                        f" × {proposal.flow_rate_mL_min:.3f}"
                        f" = {correct_vol:.2f} mL"
                    ),
                    revision_required=True,
                    suggested_revision=f"{correct_vol:.2f}",
                ))

        # Tubing ID: if calculator adjusted it (ΔP or Re driven)
        if abs(calc.tubing_ID_mm - (proposal.tubing_ID_mm or 1.0)) > 0.05:
            msgs.append(CouncilMessage(
                agent="DesignCalculator",
                status="REJECT",
                field="tubing_ID_mm",
                value=f"{proposal.tubing_ID_mm:.2f} mm",
                concern=(
                    f"Pressure/flow regime requires d = {calc.tubing_ID_mm:.2f} mm"
                ),
                revision_required=True,
                suggested_revision=f"{calc.tubing_ID_mm:.2f}",
            ))

        return msgs

    def _apply_sync(
        self,
        proposal: FlowProposal,
        calc: DesignCalculations,
        sync_msgs: list[CouncilMessage],
    ) -> FlowProposal:
        """Apply physics-sync corrections to proposal."""
        data = proposal.model_dump()
        for msg in sync_msgs:
            val = _extract_number(msg.suggested_revision)
            if val is None:
                continue
            if msg.field == "residence_time_min":
                data["residence_time_min"] = val
            elif msg.field == "reactor_volume_mL":
                data["reactor_volume_mL"] = val
            elif msg.field == "tubing_ID_mm":
                data["tubing_ID_mm"] = val
        # Enforce V_R = τ × Q after all corrections
        if data["residence_time_min"] > 0 and data["flow_rate_mL_min"] > 0:
            data["reactor_volume_mL"] = round(
                data["residence_time_min"] * data["flow_rate_mL_min"], 4
            )
        return FlowProposal(**data)

    # ── Agent-revision application ──────────────────────────────────────

    def _apply_agent_revisions(
        self, proposal: FlowProposal, revisions: list[CouncilMessage]
    ) -> FlowProposal:
        """Apply domain-agent revisions, maintaining τ×Q=V_R consistency."""
        data = proposal.model_dump()

        for msg in revisions:
            if not msg.suggested_revision:
                continue
            val = _extract_number(msg.suggested_revision)
            if val is None:
                continue

            if "residence_time" in msg.field:
                data["residence_time_min"] = val
            elif "tubing_ID" in msg.field:
                data["tubing_ID_mm"] = val
            elif "BPR" in msg.field:
                data["BPR_bar"] = val
            elif "temperature" in msg.field:
                data["temperature_C"] = val
            elif "tubing_material" in msg.field:
                if "PFA" in msg.suggested_revision:
                    data["tubing_material"] = "PFA"
                elif "SS" in msg.suggested_revision:
                    data["tubing_material"] = "SS"
            elif "reactor_volume" in msg.field:
                data["reactor_volume_mL"] = val

        # Enforce V_R = τ × Q
        if data["residence_time_min"] > 0 and data["flow_rate_mL_min"] > 0:
            data["reactor_volume_mL"] = round(
                data["residence_time_min"] * data["flow_rate_mL_min"], 4
            )

        return FlowProposal(**data)
