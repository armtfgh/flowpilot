"""ENGINE — Moderator: orchestrates the engineering validation council."""

import logging
import re

from flora_translate.config import MAX_COUNCIL_ROUNDS
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
    """Extract the first number from a string."""
    m = re.search(r"[\d.]+", text or "")
    return float(m.group()) if m else None


def apply_revisions(
    proposal: FlowProposal, revisions: list[CouncilMessage]
) -> FlowProposal:
    """Apply suggested revisions from council agents to the proposal."""
    data = proposal.model_dump()

    for msg in revisions:
        if not msg.suggested_revision:
            continue

        suggestion = msg.suggested_revision

        if msg.agent == "FluidicsAgent" and "tubing_ID" in msg.field:
            new_id = _extract_number(suggestion)
            if new_id:
                data["tubing_ID_mm"] = new_id
                # Recompute reactor volume to keep residence time constant
                if data["residence_time_min"] and data["flow_rate_mL_min"]:
                    data["reactor_volume_mL"] = round(
                        data["residence_time_min"] * data["flow_rate_mL_min"], 2
                    )

        elif msg.agent == "SafetyCriticAgent":
            if "tubing_material" in msg.field:
                if "PFA" in suggestion:
                    data["tubing_material"] = "PFA"
                elif "SS" in suggestion:
                    data["tubing_material"] = "SS"

            elif "temperature" in msg.field:
                new_t = _extract_number(suggestion)
                if new_t:
                    data["temperature_C"] = new_t

            elif "BPR" in msg.field:
                new_bpr = _extract_number(suggestion)
                if new_bpr:
                    data["BPR_bar"] = new_bpr

        elif msg.agent == "KineticsAgent" and "residence_time" in msg.field:
            new_rt = _extract_number(suggestion)
            if new_rt:
                data["residence_time_min"] = new_rt
                # Recompute reactor volume
                if data["flow_rate_mL_min"]:
                    data["reactor_volume_mL"] = round(
                        new_rt * data["flow_rate_mL_min"], 2
                    )

    return FlowProposal(**data)


def build_safety_report(messages: list[CouncilMessage]) -> dict:
    """Summarize safety-relevant council messages."""
    safety_msgs = [m for m in messages if m.agent == "SafetyCriticAgent"]
    return {
        "total_checks": len(safety_msgs),
        "accepts": len([m for m in safety_msgs if m.status == "ACCEPT"]),
        "warnings": len([m for m in safety_msgs if m.status == "WARNING"]),
        "rejects": len([m for m in safety_msgs if m.status == "REJECT"]),
        "details": [
            {"field": m.field, "status": m.status, "concern": m.concern}
            for m in safety_msgs
            if m.status != "ACCEPT"
        ],
    }


class Moderator:
    """Orchestrate the ENGINE council: run agents, detect conflicts, iterate."""

    def run(
        self,
        proposal: FlowProposal,
        batch_record: BatchRecord,
        analogies: list[dict],
        inventory: LabInventory,
        chemistry_plan: ChemistryPlan | None = None,
    ) -> DesignCandidate:
        all_messages = []
        current_proposal = proposal
        round_num = 0

        solvent = batch_record.solvent

        while round_num < MAX_COUNCIL_ROUNDS:
            round_num += 1
            logger.info(f"  ENGINE Council — Round {round_num}")

            messages: list[CouncilMessage] = []

            # Chemistry Validator (Layer 3) — only if plan is available
            if chemistry_plan:
                chem_msgs = ChemistryValidator().run(current_proposal, chemistry_plan)
                messages.extend(chem_msgs)
                for cm in chem_msgs:
                    logger.info(f"    Chemistry ({cm.field}): {cm.status}")

            # Kinetics Agent
            kinetics_msg = KineticsAgent().run(
                batch_record, current_proposal, analogies
            )
            messages.append(kinetics_msg)
            logger.info(f"    Kinetics: {kinetics_msg.status}")

            # Fluidics Agent
            fluidics_msg = FluidicsAgent().run(
                current_proposal, inventory, analogies, solvent=solvent
            )
            messages.append(fluidics_msg)
            logger.info(f"    Fluidics: {fluidics_msg.status}")

            # Safety Critic
            safety_msgs = SafetyCriticAgent().run(
                current_proposal, inventory, batch_record
            )
            messages.extend(safety_msgs)
            for sm in safety_msgs:
                logger.info(f"    Safety ({sm.field}): {sm.status}")

            all_messages.extend(messages)

            # Check for required revisions
            revisions = [m for m in messages if m.revision_required]
            if not revisions:
                logger.info(f"    No revisions needed — council done in {round_num} round(s)")
                break

            logger.info(f"    {len(revisions)} revision(s) required — applying")
            current_proposal = apply_revisions(current_proposal, revisions)

        # Final pass: Process Architect builds unit operations
        pa = ProcessArchitectAgent()
        pa_msg = pa.run(current_proposal, batch_record)
        unit_ops, pid = pa.build_operations(current_proposal, batch_record)
        all_messages.append(pa_msg)

        # Build safety report
        safety_report = build_safety_report(all_messages)

        # Mark proposal as validated
        current_proposal.engine_validated = True
        current_proposal.safety_flags = [
            m.concern for m in all_messages
            if m.status in ("WARNING", "REJECT") and m.concern
        ]

        return DesignCandidate(
            proposal=current_proposal,
            chemistry_plan=chemistry_plan,
            council_messages=all_messages,
            council_rounds=round_num,
            safety_report=safety_report,
            unit_operations=unit_ops,
            pid_description=pid,
        )
