"""ENGINE — Safety Critic: cross-checks proposal against lab inventory."""

import logging

from flora_translate.config import (
    INCOMPATIBLE_COMBOS,
    SOLVENT_BOILING_POINT_C,
)
from flora_translate.schemas import (
    BatchRecord,
    CouncilMessage,
    FlowProposal,
    LabInventory,
)

logger = logging.getLogger("flora.engine.safety")


class SafetyCriticAgent:
    """Cross-check flow proposal against lab inventory and safety rules."""

    def run(
        self,
        proposal: FlowProposal,
        inventory: LabInventory,
        batch_record: BatchRecord,
    ) -> list[CouncilMessage]:
        messages = []

        solvent = batch_record.solvent or ""

        # Check 1: Tubing material compatibility
        msg = self._check_material_compatibility(
            proposal.tubing_material, solvent, proposal.temperature_C
        )
        if msg:
            messages.append(msg)

        # Check 2: Pump pressure limit
        msg = self._check_pump_pressure(proposal, inventory)
        if msg:
            messages.append(msg)

        # Check 3: Light source availability
        msg = self._check_light_source(proposal, inventory)
        if msg:
            messages.append(msg)

        # Check 4: BPR availability
        msg = self._check_bpr(proposal, inventory)
        if msg:
            messages.append(msg)

        # Check 5: Reactor volume availability
        msg = self._check_reactor_volume(proposal, inventory)
        if msg:
            messages.append(msg)

        # Check 6: Temperature limit
        msg = self._check_temperature_limit(proposal, inventory)
        if msg:
            messages.append(msg)

        # Check 7: BPR required check (solvent bp vs temperature)
        msg = self._check_bpr_required(proposal, solvent)
        if msg:
            messages.append(msg)

        # If no issues, add an explicit ACCEPT
        if not messages:
            messages.append(
                CouncilMessage(
                    agent="SafetyCriticAgent",
                    status="ACCEPT",
                    field="safety",
                    value="All checks passed",
                    concern="",
                )
            )

        return messages

    def _check_material_compatibility(
        self, material: str, solvent: str, temperature: float
    ) -> CouncilMessage | None:
        for (mat, sol), concern in INCOMPATIBLE_COMBOS.items():
            if (
                mat.lower() == material.lower()
                and sol.lower() in solvent.lower()
                and temperature > 40
            ):
                return CouncilMessage(
                    agent="SafetyCriticAgent",
                    status="REJECT",
                    field="tubing_material",
                    value=f"{material} + {solvent} at {temperature}°C",
                    concern=concern,
                    revision_required=True,
                    suggested_revision=f"Use PFA or SS tubing instead of {material}.",
                )
        return None

    def _check_pump_pressure(
        self, proposal: FlowProposal, inventory: LabInventory
    ) -> CouncilMessage | None:
        if not inventory.pumps:
            return None
        max_pressure = max(p.max_pressure_bar for p in inventory.pumps)
        # We check against BPR + safety margin
        required = proposal.BPR_bar + 2  # 2 bar margin above BPR
        if required > max_pressure:
            return CouncilMessage(
                agent="SafetyCriticAgent",
                status="WARNING",
                field="pump_pressure",
                value=f"Required {required:.0f} bar, max available {max_pressure:.0f} bar",
                concern=f"BPR at {proposal.BPR_bar} bar + margin requires pump >{required:.0f} bar.",
                revision_required=False,
            )
        return None

    def _check_light_source(
        self, proposal: FlowProposal, inventory: LabInventory
    ) -> CouncilMessage | None:
        if not proposal.wavelength_nm or not inventory.light_sources:
            return None
        available_wl = [ls.wavelength_nm for ls in inventory.light_sources]
        match = any(
            abs(wl - proposal.wavelength_nm) <= 30 for wl in available_wl
        )
        if not match:
            return CouncilMessage(
                agent="SafetyCriticAgent",
                status="WARNING",
                field="wavelength_nm",
                value=f"{proposal.wavelength_nm} nm",
                concern=(
                    f"No matching light source in inventory for "
                    f"{proposal.wavelength_nm} nm. "
                    f"Available: {available_wl}"
                ),
                revision_required=False,
            )
        return None

    def _check_bpr(
        self, proposal: FlowProposal, inventory: LabInventory
    ) -> CouncilMessage | None:
        if not proposal.BPR_bar or not inventory.BPR_available:
            return None
        match = any(
            abs(bpr - proposal.BPR_bar) <= 1 for bpr in inventory.BPR_available
        )
        if not match:
            return CouncilMessage(
                agent="SafetyCriticAgent",
                status="WARNING",
                field="BPR_bar",
                value=f"{proposal.BPR_bar} bar",
                concern=(
                    f"No BPR at {proposal.BPR_bar} bar (±1) in inventory. "
                    f"Available: {inventory.BPR_available}"
                ),
                revision_required=False,
            )
        return None

    def _check_reactor_volume(
        self, proposal: FlowProposal, inventory: LabInventory
    ) -> CouncilMessage | None:
        if not proposal.reactor_volume_mL or not inventory.reactors:
            return None
        match = any(
            0.5 * r.volume_mL <= proposal.reactor_volume_mL <= 2.0 * r.volume_mL
            for r in inventory.reactors
        )
        if not match:
            available = [r.volume_mL for r in inventory.reactors]
            return CouncilMessage(
                agent="SafetyCriticAgent",
                status="WARNING",
                field="reactor_volume_mL",
                value=f"{proposal.reactor_volume_mL} mL",
                concern=(
                    f"Proposed volume {proposal.reactor_volume_mL} mL not within "
                    f"0.5x-2x of any available reactor. Available: {available} mL"
                ),
                revision_required=False,
            )
        return None

    def _check_temperature_limit(
        self, proposal: FlowProposal, inventory: LabInventory
    ) -> CouncilMessage | None:
        matching_tubing = [
            t
            for t in inventory.tubing
            if t.material.lower() == proposal.tubing_material.lower()
        ]
        if not matching_tubing:
            return None
        max_temp = max(t.max_temperature_C for t in matching_tubing)
        if proposal.temperature_C > max_temp:
            return CouncilMessage(
                agent="SafetyCriticAgent",
                status="REJECT",
                field="temperature_C",
                value=f"{proposal.temperature_C}°C > {max_temp}°C max for {proposal.tubing_material}",
                concern=(
                    f"Proposed temperature ({proposal.temperature_C}°C) exceeds "
                    f"max for {proposal.tubing_material} ({max_temp}°C)."
                ),
                revision_required=True,
                suggested_revision=f"Use SS tubing or reduce temperature to <{max_temp}°C.",
            )
        return None

    def _check_bpr_required(
        self, proposal: FlowProposal, solvent: str
    ) -> CouncilMessage | None:
        """Check if BPR is needed based on solvent boiling point."""
        bp = None
        for name, temp in SOLVENT_BOILING_POINT_C.items():
            if name.lower() == solvent.lower():
                bp = temp
                break
        if bp is None:
            return None
        if proposal.temperature_C > bp - 20 and (not proposal.BPR_bar or proposal.BPR_bar < 2):
            return CouncilMessage(
                agent="SafetyCriticAgent",
                status="WARNING",
                field="BPR_bar",
                value=f"T={proposal.temperature_C}°C, solvent bp={bp}°C, BPR={proposal.BPR_bar}",
                concern=(
                    f"Temperature ({proposal.temperature_C}°C) is within 20°C of "
                    f"solvent boiling point ({bp}°C for {solvent}). "
                    f"BPR is required but set to {proposal.BPR_bar} bar."
                ),
                revision_required=True,
                suggested_revision="Set BPR_bar to at least 5 bar.",
            )
        return None
