"""ENGINE — Fluidics Agent: pressure drop, Reynolds number, hardware compatibility."""

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


def _get_viscosity(solvent: str | None) -> tuple[float, bool]:
    """Return (viscosity_cP, is_default). Tries multiple name variants."""
    if solvent:
        for key in [solvent, solvent.lower(), solvent.upper()]:
            if key in SOLVENT_VISCOSITY_cP:
                return SOLVENT_VISCOSITY_cP[key], False
    return 1.0, True  # default to water-like


def _get_density(solvent: str | None) -> float:
    if solvent:
        for key in [solvent, solvent.lower()]:
            if key in SOLVENT_DENSITY_g_mL:
                return SOLVENT_DENSITY_g_mL[key]
    return 1.0


def calculate_pressure_drop(
    flow_rate_mL_min: float,
    tubing_ID_mm: float,
    tubing_length_m: float,
    viscosity_cP: float = 1.0,
) -> float:
    """Hagen-Poiseuille pressure drop in bar."""
    # Convert units to SI
    Q = flow_rate_mL_min * 1e-6 / 60  # m³/s
    d = tubing_ID_mm * 1e-3           # m
    L = tubing_length_m               # m
    eta = viscosity_cP * 1e-3         # Pa·s

    if d <= 0:
        return float("inf")

    dP_Pa = (128 * eta * L * Q) / (math.pi * d**4)
    return dP_Pa * 1e-5  # Pa → bar


def calculate_reynolds(
    flow_rate_mL_min: float,
    tubing_ID_mm: float,
    density_g_mL: float = 1.0,
    viscosity_cP: float = 1.0,
) -> float:
    """Reynolds number for flow in a tube."""
    d = tubing_ID_mm * 1e-3  # m
    A = math.pi * (d / 2) ** 2  # m²
    Q = flow_rate_mL_min * 1e-6 / 60  # m³/s

    if A <= 0:
        return 0

    v = Q / A  # m/s
    rho = density_g_mL * 1000  # kg/m³
    eta = viscosity_cP * 1e-3  # Pa·s

    if eta <= 0:
        return 0

    return (rho * v * d) / eta


class FluidicsAgent:
    """Compute pressure drop and flow regime; check against hardware limits."""

    def run(
        self,
        proposal: FlowProposal,
        inventory: LabInventory,
        analogies: list[dict],
        solvent: str | None = None,
    ) -> CouncilMessage:
        messages = []

        # Determine solvent from analogies or proposal context
        if not solvent:
            for a in analogies:
                meta = a.get("metadata", {})
                if meta.get("solvent"):
                    solvent = meta["solvent"]
                    break

        viscosity, is_default = _get_viscosity(solvent)
        density = _get_density(solvent)

        # Estimate tubing length from volume and ID
        d_m = proposal.tubing_ID_mm * 1e-3
        A = math.pi * (d_m / 2) ** 2
        tubing_length_m = (proposal.reactor_volume_mL * 1e-6) / A if A > 0 else 0

        # Calculate pressure drop
        dP = calculate_pressure_drop(
            proposal.flow_rate_mL_min,
            proposal.tubing_ID_mm,
            tubing_length_m,
            viscosity,
        )

        # Calculate Reynolds number
        Re = calculate_reynolds(
            proposal.flow_rate_mL_min,
            proposal.tubing_ID_mm,
            density,
            viscosity,
        )

        # Find the best matching pump
        pump_max = 0
        pump_name = "unknown"
        for p in inventory.pumps:
            if p.max_flow_rate_mL_min >= proposal.flow_rate_mL_min:
                if p.max_pressure_bar > pump_max:
                    pump_max = p.max_pressure_bar
                    pump_name = p.name

        if pump_max == 0 and inventory.pumps:
            best = max(inventory.pumps, key=lambda p: p.max_pressure_bar)
            pump_max = best.max_pressure_bar
            pump_name = best.name

        concerns = []

        # Viscosity warning
        if is_default and solvent:
            concerns.append(
                f"Solvent '{solvent}' not in viscosity lookup — using default 1.0 cP."
            )

        # Pressure checks
        if pump_max > 0 and dP > pump_max:
            next_id = self._suggest_larger_id(proposal.tubing_ID_mm, inventory)
            return CouncilMessage(
                agent="FluidicsAgent",
                status="REJECT",
                field="tubing_ID_mm",
                value=f"dP={dP:.2f} bar > pump max={pump_max:.0f} bar ({pump_name})",
                concern=(
                    f"Pressure drop ({dP:.2f} bar) exceeds pump maximum "
                    f"({pump_max:.0f} bar). Flow is not feasible with "
                    f"{proposal.tubing_ID_mm} mm ID tubing "
                    f"({tubing_length_m:.1f} m length)."
                ),
                revision_required=True,
                suggested_revision=(
                    f"Increase tubing ID to {next_id} mm. "
                    f"Recalculate reactor volume to maintain residence time."
                ),
            )

        if pump_max > 0 and dP > PRESSURE_WARNING_FRACTION * pump_max:
            next_id = self._suggest_larger_id(proposal.tubing_ID_mm, inventory)
            concerns.append(
                f"Pressure drop ({dP:.2f} bar) is {dP/pump_max*100:.0f}% "
                f"of pump capacity ({pump_max:.0f} bar). Consider ID={next_id} mm."
            )

        # Reynolds checks
        if Re > RE_TURBULENT:
            concerns.append(
                f"Turbulent flow detected (Re={Re:.0f}). "
                f"Consider reducing flow rate."
            )
        elif Re < RE_VERY_LOW:
            concerns.append(
                f"Very low Re ({Re:.2f}). Check viscosity estimate."
            )

        if concerns:
            status = "WARNING"
        else:
            status = "ACCEPT"

        detail = (
            f"dP={dP:.3f} bar, Re={Re:.1f}, "
            f"tube={proposal.tubing_ID_mm}mm x {tubing_length_m:.2f}m, "
            f"viscosity={viscosity} cP ({solvent or 'default'})"
        )

        return CouncilMessage(
            agent="FluidicsAgent",
            status=status,
            field="fluidics",
            value=detail,
            concern=" | ".join(concerns) if concerns else "",
        )

    def _suggest_larger_id(
        self, current_id: float, inventory: LabInventory
    ) -> float:
        """Suggest the next available larger tubing ID from inventory."""
        available = sorted(set(t.ID_mm for t in inventory.tubing))
        for d in available:
            if d > current_id:
                return d
        return current_id * 1.5  # fallback: 50% increase
