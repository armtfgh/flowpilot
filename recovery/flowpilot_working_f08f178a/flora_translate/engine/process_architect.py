"""ENGINE — Process Architect: builds chemistry-aware unit operation sequence and P&ID."""

import logging

from flora_translate.schemas import BatchRecord, CouncilMessage, FlowProposal

logger = logging.getLogger("flora.engine.process_architect")


class ProcessArchitectAgent:
    """Build ordered unit operation sequence from a validated flow proposal.

    Uses the chemistry-aware stream assignments from the LLM proposal
    to produce a specific, named process description — not generic
    "reagent 1 / reagent 2" labels.
    """

    def run(
        self, proposal: FlowProposal, batch_record: BatchRecord
    ) -> CouncilMessage:
        ops, pid = self._build(proposal, batch_record)
        return CouncilMessage(
            agent="ProcessArchitectAgent",
            status="ACCEPT",
            field="unit_operations",
            value=f"{len(ops)} unit operations",
            concern="",
        )

    def build_operations(
        self, proposal: FlowProposal, batch_record: BatchRecord
    ) -> tuple[list[str], str]:
        """Return (unit_operations, pid_description)."""
        return self._build(proposal, batch_record)

    def _build(
        self, proposal: FlowProposal, batch_record: BatchRecord
    ) -> tuple[list[str], str]:
        ops: list[str] = []
        pid_parts: list[str] = []

        # --- Pumps / Streams (chemistry-aware) ---
        if proposal.streams:
            pump_labels = []
            for stream in proposal.streams:
                label = stream.stream_label or "?"
                contents_str = ", ".join(stream.contents) if stream.contents else stream.pump_role
                solvent = stream.solvent or ""
                conc = f" ({stream.concentration_M} M)" if stream.concentration_M else ""
                rate = f" @ {stream.flow_rate_mL_min} mL/min" if stream.flow_rate_mL_min else ""

                op = f"Pump {label}: {contents_str}"
                if solvent:
                    op += f" in {solvent}{conc}"
                op += rate

                if stream.reasoning:
                    op += f"  [{stream.reasoning}]"

                ops.append(op)
                pump_labels.append(f"Pump {label} ({contents_str[:40]})")

            pid_parts.append(" + ".join(pump_labels))
        else:
            # Fallback: generic (shouldn't happen with updated prompts)
            ops.append("Pump A (reagent stream 1)")
            ops.append("Pump B (reagent stream 2)")
            pid_parts.append("Pump A + Pump B")

        # --- Pre-reactor steps (deoxygenation, pre-mixing, etc.) ---
        if proposal.pre_reactor_steps:
            for step in proposal.pre_reactor_steps:
                ops.append(f"Pre-reactor: {step}")
                pid_parts.append(step.split("—")[0].strip()[:30])
        elif proposal.deoxygenation_method:
            ops.append(f"Inline deoxygenation ({proposal.deoxygenation_method})")
            pid_parts.append(f"degas ({proposal.deoxygenation_method})")

        # --- Mixer ---
        mixer = proposal.mixer_type or "T-mixer"
        mixer_desc = f"{mixer} → combined stream"
        if proposal.mixing_order_reasoning:
            mixer_desc += f"  [{proposal.mixing_order_reasoning[:80]}]"
        ops.append(mixer_desc)
        pid_parts.append(mixer)

        # --- Reactor ---
        reactor_desc = (
            f"Photoreactor {proposal.reactor_type} "
            f"({proposal.tubing_material}, {proposal.tubing_ID_mm}mm ID, "
            f"{proposal.reactor_volume_mL}mL"
        )
        if proposal.light_setup:
            reactor_desc += f", {proposal.light_setup}"
        if proposal.wavelength_nm:
            reactor_desc += f", {proposal.wavelength_nm}nm"
        reactor_desc += f", {proposal.temperature_C}°C)"
        ops.append(reactor_desc)

        pid_reactor = (
            f"{proposal.reactor_type} ({proposal.reactor_volume_mL}mL, "
            f"{proposal.temperature_C}°C"
        )
        if proposal.wavelength_nm:
            pid_reactor += f", {proposal.wavelength_nm}nm LED"
        pid_reactor += ")"
        pid_parts.append(pid_reactor)

        # --- BPR ---
        if proposal.BPR_bar and proposal.BPR_bar > 0:
            ops.append(f"Back-pressure regulator ({proposal.BPR_bar} bar)")
            pid_parts.append(f"BPR ({proposal.BPR_bar} bar)")

        # --- Post-reactor steps (quench, workup, etc.) ---
        if proposal.post_reactor_steps:
            for step in proposal.post_reactor_steps:
                ops.append(f"Post-reactor: {step}")
                pid_parts.append(step.split("—")[0].strip()[:30])
        else:
            # Fallback: check batch record for quench mentions
            raw = batch_record.raw_text or batch_record.reaction_description or ""
            if any(kw in raw.lower() for kw in ["quench", "workup", "work-up", "neutralize"]):
                ops.append("Inline quench (specify appropriate quench reagent)")
                pid_parts.append("quench")

        # --- Collection ---
        ops.append("Collection / analysis")
        pid_parts.append("collection")

        # --- Chemistry notes ---
        if proposal.chemistry_notes:
            ops.append(f"NOTE: {proposal.chemistry_notes}")

        pid_description = " → ".join(pid_parts)
        return ops, pid_description
