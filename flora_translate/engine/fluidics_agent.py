"""ENGINE — Fluidics Agent: validates pressure drop and flow regime from calculations.

Reads DesignCalculator results for authoritative Re and ΔP values.
Only flags issues the calculator identified or inventory mismatches.
"""

import logging
import math

from flora_translate.config import (
    PRESSURE_WARNING_FRACTION,
    RE_TURBULENT,
    RE_VERY_LOW,
    SOLVENT_DENSITY_g_mL,
    SOLVENT_VISCOSITY_cP,
)
from flora_translate.schemas import CouncilMessage, FlowProposal, LabInventory

logger = logging.getLogger("flora.engine.fluidics")


class FluidicsAgent:
    """Validate fluid dynamics and pressure drop using calculator results."""

    def run(
        self,
        proposal: FlowProposal,
        inventory: LabInventory,
        analogies: list[dict],
        solvent: str | None = None,
        calculations=None,
    ) -> CouncilMessage:

        # ── Use calculator results if available ─────────────────────────
        if calculations:
            Re = calculations.reynolds_number
            dP = calculations.pressure_drop_bar
            regime = calculations.flow_regime
            pump_max = calculations.pump_max_bar or 20.0
            L = calculations.tubing_length_m
            d_mm = calculations.tubing_ID_mm
            v = calculations.velocity_m_s

            concerns = []

            # Pressure check
            if dP > pump_max:
                return CouncilMessage(
                    agent="FluidicsAgent",
                    status="REJECT",
                    field="tubing_ID_mm",
                    value=(
                        f"ΔP = {dP:.3f} bar > pump max = {pump_max:.0f} bar"
                    ),
                    concern=(
                        f"Pressure drop ({dP:.3f} bar) exceeds pump maximum "
                        f"({pump_max:.0f} bar). Calculated for "
                        f"{d_mm:.2f} mm ID × {L:.2f} m length."
                    ),
                    revision_required=True,
                    suggested_revision=(
                        f"Increase tubing ID. Calculator suggests "
                        f"{calculations.tubing_ID_mm:.2f} mm."
                    ),
                )
            if pump_max > 0 and dP > PRESSURE_WARNING_FRACTION * pump_max:
                concerns.append(
                    f"ΔP = {dP:.3f} bar is {dP/pump_max*100:.0f}% of "
                    f"pump capacity ({pump_max:.0f} bar)"
                )

            # Reynolds check
            if Re > RE_TURBULENT:
                concerns.append(
                    f"Turbulent flow: Re = {Re:.0f}. Reduce Q or increase d."
                )
            elif Re < RE_VERY_LOW and Re > 0:
                concerns.append(f"Very low Re = {Re:.2f}. Check viscosity.")

            # Mass transfer flag from calculator
            if calculations.mass_transfer_limited:
                concerns.append(
                    f"Da_mass = {calculations.damkohler_mass:.2f} > 1: "
                    "mass-transfer limited — add static mixer or reduce d."
                )

            detail = (
                f"ΔP = {dP:.4f} bar, Re = {Re:.1f} ({regime}), "
                f"v = {v:.4f} m/s, "
                f"d = {d_mm:.2f} mm × {L:.2f} m"
            )

            return CouncilMessage(
                agent="FluidicsAgent",
                status="WARNING" if concerns else "ACCEPT",
                field="fluidics",
                value=detail,
                concern=" | ".join(concerns) if concerns else "",
            )

        # ── Fallback: compute from proposal (legacy path) ──────────────
        return self._compute_from_proposal(proposal, inventory, solvent)

    def _compute_from_proposal(self, proposal, inventory, solvent):
        """Direct calculation when no DesignCalculations available."""
        mu_cP = 1.0
        rho_gmL = 1.0
        if solvent:
            for key in (solvent, solvent.lower()):
                if key in SOLVENT_VISCOSITY_cP:
                    mu_cP = SOLVENT_VISCOSITY_cP[key]
                    break
            for key in (solvent, solvent.lower()):
                if key in SOLVENT_DENSITY_g_mL:
                    rho_gmL = SOLVENT_DENSITY_g_mL[key]
                    break

        d_m = proposal.tubing_ID_mm * 1e-3
        A = math.pi * (d_m / 2) ** 2
        L = (proposal.reactor_volume_mL * 1e-6) / A if A > 0 else 0
        Q = proposal.flow_rate_mL_min * 1e-6 / 60
        v = Q / A if A > 0 else 0
        mu = mu_cP * 1e-3
        rho = rho_gmL * 1000
        Re = rho * v * d_m / mu if mu > 0 else 0
        dP_Pa = 128 * mu * L * Q / (math.pi * d_m ** 4) if d_m > 0 else 0
        dP_bar = dP_Pa * 1e-5

        pump_max = 20.0
        if inventory and inventory.pumps:
            pump_max = max(p.max_pressure_bar for p in inventory.pumps)

        concerns = []
        if dP_bar > pump_max:
            return CouncilMessage(
                agent="FluidicsAgent",
                status="REJECT",
                field="tubing_ID_mm",
                value=f"ΔP = {dP_bar:.2f} bar > pump max = {pump_max:.0f} bar",
                concern=f"Pressure drop exceeds pump capacity.",
                revision_required=True,
                suggested_revision=f"Increase tubing ID to {proposal.tubing_ID_mm * 1.5:.1f} mm",
            )
        if dP_bar > PRESSURE_WARNING_FRACTION * pump_max:
            concerns.append(f"ΔP = {dP_bar:.2f} bar is high")
        if Re > RE_TURBULENT:
            concerns.append(f"Turbulent: Re = {Re:.0f}")

        return CouncilMessage(
            agent="FluidicsAgent",
            status="WARNING" if concerns else "ACCEPT",
            field="fluidics",
            value=f"ΔP = {dP_bar:.3f} bar, Re = {Re:.1f}",
            concern=" | ".join(concerns) if concerns else "",
        )
