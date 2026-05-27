"""
FLORA-Translate — 9-Step Engineering Design Calculator.

Replaces the simpler flow_calculator.py with a comprehensive,
first-principles design pipeline for batch-to-flow translation.

Each step builds on the previous, creating a fully consistent
and traceable reactor design from batch data.

Steps:
  1. Parse batch conditions → T, C₀, t_batch, X_target
  2. Kinetics → k(T), τ_flow from PFR design equation
  3. Reactor sizing → V_R, d, L from τ and Q
  4. Fluid dynamics → Re, flow regime, velocity
  5. Pressure drop → ΔP via Hagen-Poiseuille, pump check
  6. Mass transfer → t_mix, Da_mass, mixing adequacy
  7. Heat transfer → Q_rxn, Q_removed, Da_thermal
  8. BPR sizing → P_BPR from Antoine + system ΔP
  9. Process metrics → STY, intensification, productivity

Consistency guarantee (verified in step 10):
  τ = V_R / Q              (residence time from hardware)
  L = 4·V_R / (π·d²)       (tube length from volume and diameter)
  v = 4·Q / (π·d²)         (velocity from flow rate and diameter)
  Re = ρ·v·d / μ           (Reynolds from velocity and properties)
  ΔP = 128·μ·L·Q / (π·d⁴) (Hagen-Poiseuille)
"""

from __future__ import annotations

import math
import logging
import re
from dataclasses import dataclass, field

from flora_translate.config import (
    FLOW_MAX_TAU_TO_BATCH_RATIO,
    FLOW_TRANSLATION_POLICY,
    SOLVENT_VISCOSITY_cP,
    SOLVENT_DENSITY_g_mL,
    SOLVENT_BOILING_POINT_C,
    INCOMPATIBLE_COMBOS,
)
from flora_translate.batch_normalization import (
    infer_batch_concentration_M,
    infer_reaction_time_h,
)

logger = logging.getLogger("flora.design_calculator")

PI = math.pi

# ── Diffusion coefficient (m²/s) — small organic molecules in liquid ────────
D_MOLECULAR = 1.0e-9

# ── Overall heat-transfer coefficients (W/m²·K) ────────────────────────────
U_VALUES = {"coil": 300, "chip": 500, "packed_bed": 200, "CSTR": 150}

# ── Solvent specific heat (J/kg·K) ─────────────────────────────────────────
SOLVENT_CP = {
    "MeCN": 2229, "acetonitrile": 2229, "CH3CN": 2229,
    "DMF": 2100, "DMSO": 1960,
    "THF": 1720, "EtOAc": 1920,
    "DCM": 1190, "dichloromethane": 1190, "CH2Cl2": 1190,
    "MeOH": 2530, "methanol": 2530,
    "EtOH": 2440, "ethanol": 2440,
    "toluene": 1700, "water": 4186, "H2O": 4186,
    "DMA": 2010, "NMP": 1670,
    "dioxane": 1740, "acetone": 2170,
    "hexane": 2260, "benzene": 1740,
}

# ── Estimated ΔH_r by reaction class (J/mol, negative = exothermic) ────────
DELTA_H_ESTIMATES = {
    "photoredox": -50_000, "photocatalysis": -50_000,
    "photochem": -50_000, "thermal": -80_000,
    "cross-coupling": -60_000, "hydrogenation": -120_000,
    "electrochemistry": -40_000, "radical": -70_000,
    "oxidation": -100_000, "reduction": -90_000,
    "default": -60_000,
}

# ── Intensification factors (batch time / flow time) from literature ────────
# CLASS-LEVEL defaults — conservative midpoints of the ranges from the
# FLORA Council Calculation Protocol (April 2026).
# Photoredox / photocatalysis: class range 4–8×.  48× was an analogy-derived
# value mistakenly used as the class default; that overshot every τ estimate
# for runs without analogy data.  If strong analogies exist (≥2 datapoints),
# Step 2 will use IF_analogy instead of IF_class automatically.
INTENSIFICATION = {
    "photoredox": 6.0, "photocatalysis": 6.0,   # class range 4–8×
    "photochem": 6.0,                             # same class range
    "thermal": 10.0,                              # class range 8–15×
    "cross-coupling": 15.0,                       # class range 10–20×
    "hydrogenation": 30.0,                        # class range 20–50×
    "electrochemistry": 20.0,
    "biocatalysis": 8.0,
    "radical": 30.0,                              # class range 10–60×
    "default": 10.0,
}

# ── Estimated Ea by reaction class (J/mol) for Arrhenius correction ─────────
# These are order-of-magnitude estimates. Photoredox has weak T-dependence
# because the rate is limited by photon absorption, not thermal activation.
ESTIMATED_EA = {
    "photoredox": 25_000, "photocatalysis": 25_000,
    "photochem": 25_000, "thermal": 80_000,
    "cross-coupling": 65_000, "hydrogenation": 45_000,
    "electrochemistry": 35_000, "biocatalysis": 50_000,
    "radical": 60_000, "default": 60_000,
}

R_GAS = 8.314  # J/(mol·K) — universal gas constant
R_GAS_L_BAR = 0.08314  # L·bar/(mol·K)
T_STP_K = 273.15
P_STP_BAR = 1.01325
GAS_LIQUID_MIN_BPR_BAR = 5.0
GAS_LIQUID_ROUTINE_MAX_BPR_BAR = 10.0
GAS_LIQUID_BPR_MARGIN_BAR = 1.5
GAS_LIQUID_MAX_ROUTINE_DELTA_P_BAR = GAS_LIQUID_ROUTINE_MAX_BPR_BAR - GAS_LIQUID_BPR_MARGIN_BAR

# ── Antoine coefficients: log10(P_mmHg) = A − B/(C + T_°C) ────────────────
ANTOINE = {
    "MeCN": (8.1948, 1747.0, 240.0), "acetonitrile": (8.1948, 1747.0, 240.0),
    "DMF": (6.9623, 1400.87, 196.43), "DMSO": (7.887, 2019.0, 219.0),
    "THF": (6.9952, 1202.29, 226.25), "EtOAc": (7.0986, 1238.71, 217.0),
    "DCM": (7.0803, 1138.91, 231.45), "dichloromethane": (7.0803, 1138.91, 231.45),
    "MeOH": (8.0809, 1582.27, 239.73), "EtOH": (8.1122, 1592.86, 226.18),
    "toluene": (6.9533, 1343.94, 219.38), "water": (8.0713, 1730.63, 233.43),
    "acetone": (7.1171, 1210.6, 229.66), "hexane": (6.876, 1171.17, 224.41),
    "benzene": (6.9057, 1211.03, 220.79), "dioxane": (7.069, 1343.94, 219.38),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """One step of the 9-step calculation, for display and traceability."""
    step: int
    name: str
    status: str                    # PASS | WARNING | FAIL | ADJUSTED | ESTIMATED
    summary: str
    values: dict = field(default_factory=dict)
    equations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


@dataclass
class DesignCalculations:
    """Complete 9-step engineering design.

    Every derived quantity traces back to inputs via the equations
    recorded in each StepResult.  The _verify() pass confirms:
        τ = V_R / Q,   L = 4V_R/(πd²),   Re = ρvd/μ
    """
    steps: list[StepResult] = field(default_factory=list)

    # Step 1 — parsed conditions
    temperature_C: float = 25.0
    temperature_K: float = 298.15
    concentration_M: float = 0.1
    batch_time_s: float = 0.0
    target_conversion: float = 0.95

    # Step 2 — kinetics (multi-method)
    rate_constant: float | None = None
    reaction_order: int = 1
    kinetics_method: str = ""               # "analogy" | "class" | "default"
    residence_time_s: float = 0.0
    residence_time_min: float = 0.0
    residence_time_range_min: tuple[float, float] | None = None  # (τ_low, τ_high)
    intensification_factor: float = 1.0
    tau_analogy_min: float | None = None    # τ from analogy-derived IF
    tau_class_min: float | None = None      # τ from class-level IF
    if_analogy: float | None = None         # IF from analogies
    if_class: float | None = None           # IF from class table
    n_analogy_datapoints: int = 0           # how many analogies had batch+flow data
    arrhenius_correction: float | None = None  # k(T_flow)/k(T_batch), None if no correction

    # Step 3 — geometry
    flow_rate_mL_min: float = 0.0
    flow_rate_m3_s: float = 0.0
    reactor_volume_mL: float = 0.0
    reactor_volume_m3: float = 0.0
    tubing_ID_mm: float = 1.0
    tubing_ID_m: float = 1e-3
    tubing_length_m: float = 0.0

    # Step 4 — fluid dynamics
    velocity_m_s: float = 0.0
    reynolds_number: float = 0.0
    flow_regime: str = "laminar"
    viscosity_Pa_s: float = 1e-3
    density_kg_m3: float = 1000.0

    # Step 5 — pressure
    pressure_drop_bar: float = 0.0
    pump_max_bar: float = 0.0
    pump_adequate: bool = True

    # Multiphase / gas-liquid extension
    is_gas_liquid: bool = False
    gas_species: str = ""
    gas_oxygen_fraction: float = 0.0
    liquid_flow_rate_mL_min: float = 0.0
    gas_flow_sccm: float = 0.0
    gas_sccm_uncapped: float = 0.0
    gas_flow_actual_mL_min: float = 0.0
    gas_actual_uncapped_mL_min: float = 0.0
    gas_flow_capped_by_holdup: bool = False
    gas_pressure_abs_bar: float = 1.01325
    gas_liquid_ratio: float = 0.0
    gas_holdup: float = 0.0
    liquid_holdup_volume_mL: float = 0.0
    two_phase_multiplier: float = 1.0
    two_phase_pressure_drop_bar: float = 0.0
    o2_supply_mmol_min: float = 0.0
    o2_required_mmol_min: float = 0.0
    o2_equiv_supplied: float = 0.0
    dissolved_o2_mM: float = 0.0
    kLa_s: float = 0.0
    o2_transfer_capacity_mmol_min: float = 0.0
    o2_transfer_sufficiency: float = 0.0

    # Step 6 — mass transfer
    mixing_time_s: float = 0.0
    damkohler_mass: float = 0.0
    mass_transfer_limited: bool = False

    # Step 7 — heat transfer
    heat_generation_W: float | None = None
    heat_removal_W: float | None = None
    thermal_damkohler: float | None = None
    thermal_safe: bool = True
    surface_to_volume: float = 0.0
    heat_transfer_area_m2: float = 0.0
    UA_W_K: float = 0.0
    heat_transfer_score: float = 0.0

    # Step 8 — BPR
    bpr_required: bool = False
    bpr_pressure_bar: float = 0.0
    vapor_pressure_bar: float = 0.0

    # Step 9 — metrics
    space_time_yield_mol_L_h: float | None = None
    productivity_mmol_h: float | None = None

    # Step 9 extended — forward productivity (ṅ_limiting path)
    n_molar_flow_mmol_min: float | None = None   # ṅ_limiting = P_batch / Y
    P_batch_mmol_h: float | None = None          # batch productivity baseline
    P_flow_mmol_h: float | None = None           # flow productivity (from ṅ_limiting)
    C_reactor_M: float | None = None             # concentration inside reactor after stream mixing
    startup_waste_mL: float | None = None        # 3×τ × Q_total
    productivity_closure_ok: bool | None = None  # P_flow ≥ P_batch?

    # Step 3 extended — plug flow validity
    Pe: float | None = None                      # Péclet number (plug flow validity)
    Pe_adequate: bool = True                     # Pe ≥ 100 → plug flow valid

    # Consistency
    consistent: bool = True
    consistency_notes: list[str] = field(default_factory=list)
    material_warnings: list[str] = field(default_factory=list)

    # ── Legacy properties (backward compat with FlowCalculations) ───────
    @property
    def estimated_rt_min(self):
        return self.residence_time_min or None

    @property
    def bpr_minimum_bar(self):
        return self.bpr_pressure_bar if self.bpr_required else None

    @property
    def solvent_boiling_point_C(self):
        for s in self.steps:
            bp = s.values.get("solvent_bp_C")
            if bp is not None:
                return bp
        return None

    @property
    def damkohler_number(self):
        return self.damkohler_mass

    @property
    def damkohler_interpretation(self):
        da = self.damkohler_mass
        if da < 1:
            return f"Da = {da:.2f} < 1: mixing may limit conversion — consider active mixer."
        if da < 10:
            return f"Da = {da:.1f}: comparable timescales — adequate for coil/chip reactor."
        return f"Da = {da:.0f} >> 1: kinetically controlled — mixing is not limiting."

    def to_prompt_block(self) -> str:
        """Concise text injected into the translation LLM prompt."""
        lines = ["## Engineering Design Calculations (9-Step)"]
        lines.append(
            "Use these numbers directly — do NOT contradict without justification.\n"
        )
        for s in self.steps:
            icon = {
                "PASS": "✓", "WARNING": "⚠", "FAIL": "✗",
                "ADJUSTED": "↻", "ESTIMATED": "≈",
            }.get(s.status, "?")
            lines.append(f"**Step {s.step}: {s.name}** [{icon}] {s.summary}")
            for w in s.warnings:
                lines.append(f"  ⚠ {w}")
            for a in s.adjustments:
                lines.append(f"  ↻ {a}")

        # Key numbers with kinetics detail
        rt_range = self.residence_time_range_min
        range_str = (f" (range: {rt_range[0]:.1f}–{rt_range[1]:.1f} min)"
                     if rt_range else "")
        lines.append(
            f"\n**Key numbers:** τ = {self.residence_time_min:.1f} min"
            f"{range_str}, "
            f"Q_total = {self.flow_rate_mL_min:.3f} mL/min, "
            f"V_R = {self.reactor_volume_mL:.2f} mL, "
            f"d = {self.tubing_ID_mm:.2f} mm, L = {self.tubing_length_m:.2f} m, "
            f"Re = {self.reynolds_number:.0f}, ΔP = {self.pressure_drop_bar:.3f} bar, "
            f"Pe = {self.Pe or '?'} ({'✓ plug flow' if self.Pe_adequate else '⚠ axial dispersion'})"
        )
        if self.n_molar_flow_mmol_min is not None:
            lines.append(
                f"**Molar flow:** ṅ_limiting = {self.n_molar_flow_mmol_min:.4f} mmol/min | "
                f"P_batch = {self.P_batch_mmol_h or '?'} mmol/h | "
                f"C_reactor = {self.C_reactor_M or '?'} M"
            )
        if self.startup_waste_mL is not None:
            lines.append(f"**Startup waste:** {self.startup_waste_mL} mL (3×τ×Q)")
        # Kinetics provenance
        if self.kinetics_method == "analogy":
            lines.append(
                f"**Kinetics source:** {self.n_analogy_datapoints} literature analogies "
                f"(IF = {self.if_analogy}×, class IF = {self.if_class}×)"
            )
        elif self.kinetics_method == "analogy+class":
            lines.append(
                f"**Kinetics source:** 1 analogy (IF = {self.if_analogy}×) "
                f"averaged with class (IF = {self.if_class}×) → IF = {self.intensification_factor}×"
            )
        else:
            lines.append(
                f"**Kinetics source:** class-level IF = {self.if_class}× "
                "(no analogy data available)"
            )
        if self.arrhenius_correction and self.arrhenius_correction != 1.0:
            lines.append(
                f"**Arrhenius:** k(T_flow)/k(T_batch) = {self.arrhenius_correction:.2f}×"
            )
        if self.bpr_required:
            lines.append(f"**BPR:** {self.bpr_pressure_bar:.1f} bar (required)")
        if self.is_gas_liquid:
            lines.append(
                "**Gas-liquid:** "
                f"{self.gas_species or 'gas'} MFC = {self.gas_flow_sccm:.2f} sccm "
                f"({self.gas_flow_actual_mL_min:.3f} mL/min at reactor), "
                f"ε_g = {self.gas_holdup:.2f}, "
                f"V_total = {self.reactor_volume_mL:.2f} mL, "
                f"O₂ transfer sufficiency = {self.o2_transfer_sufficiency:.2f}×"
            )
        if self.UA_W_K:
            lines.append(
                f"**Heat transfer:** UA = {self.UA_W_K:.4f} W/K, "
                f"A_wall = {self.heat_transfer_area_m2:.5f} m², "
                f"heat-transfer score = {self.heat_transfer_score:.2f}"
            )
        if not self.consistent:
            lines.append("\n⚠ CONSISTENCY ISSUES:")
            for n in self.consistency_notes:
                lines.append(f"  {n}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _lookup(solvent: str | None, table: dict):
    """Look up a solvent property, trying common aliases."""
    if not solvent:
        return None
    s = solvent.strip()
    for key in (s, s.lower(), s.upper()):
        if key in table:
            return table[key]
    sl = s.lower()
    for key, val in table.items():
        if key.lower() in sl or sl in key.lower():
            return val
    return None


def _detect_chemistry(batch_record, chemistry_plan=None) -> str:
    """Infer chemistry type from available data."""
    if chemistry_plan and chemistry_plan.reaction_class:
        rc = chemistry_plan.reaction_class.lower()
        for key in INTENSIFICATION:
            if key in rc:
                return key
    desc = " ".join([
        str(getattr(batch_record, "reaction_description", "") or ""),
        str(getattr(batch_record, "photocatalyst", "") or ""),
    ]).lower()
    if any(k in desc for k in ("photoredox", "ir(ppy", "ru(bpy", "photocatal", "4czipn", "eosin")):
        return "photoredox"
    if any(k in desc for k in ("hν", "light", "uv", "led", "irradiat", "photochem")):
        return "photochem"
    if any(k in desc for k in ("pd", "suzuki", "heck", "cross-coupl", "buchwald")):
        return "cross-coupling"
    if any(k in desc for k in ("h2", "hydrogenat")):
        return "hydrogenation"
    if any(k in desc for k in ("electrochem", "anodic", "cathodic")):
        return "electrochemistry"
    if any(k in desc for k in ("enzyme", "lipase", "biocatal")):
        return "biocatalysis"
    if any(k in desc for k in ("radical", "aibn")):
        return "radical"
    if any(k in desc for k in ("thermal", "heat", "reflux", "cycliz")):
        return "thermal"
    return "default"


def _vapor_pressure_bar(solvent: str | None, T_C: float) -> float | None:
    """Antoine equation → vapor pressure in bar at T_C."""
    coeffs = _lookup(solvent, ANTOINE)
    if not coeffs:
        return None
    A, B, C = coeffs
    try:
        P_mmHg = 10 ** (A - B / (C + T_C))
        return P_mmHg / 750.062
    except (ZeroDivisionError, OverflowError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Calculator
# ═══════════════════════════════════════════════════════════════════════════

class DesignCalculator:
    """9-step first-principles engineering design calculator.

    Parameters
    ----------
    on_step : callable, optional
        Called after each step with ``(step_number, StepResult)``.
        Used for live Streamlit rendering.
    """

    def __init__(self, on_step=None):
        self._on_step = on_step

    def _emit(self, calc: DesignCalculations, result: StepResult):
        calc.steps.append(result)
        if self._on_step:
            self._on_step(result.step, result)

    # ─── Public entry point ─────────────────────────────────────────────

    def run(
        self,
        batch_record,
        chemistry_plan=None,
        proposal=None,
        inventory=None,
        analogies: list[dict] | None = None,
        target_flow_rate_mL_min: float | None = None,
        target_tubing_ID_mm: float | None = None,
        target_residence_time_min: float | None = None,
    ) -> DesignCalculations:
        """Execute all 9 steps and return a consistent DesignCalculations.

        target_residence_time_min: When the ENGINE council has approved a specific
        τ, pass it here.  Step 2 will use it directly instead of re-deriving from
        batch kinetics — making the council's decision the authoritative source.
        All downstream geometry (V_R, L, Re, ΔP) is computed FROM this τ.
        """
        calc = DesignCalculations()

        # Initial geometry from proposal, overrides, or defaults
        Q_init = (
            target_flow_rate_mL_min
            or (proposal.flow_rate_mL_min if proposal and proposal.flow_rate_mL_min else None)
            or 0.5
        )
        d_init = (
            target_tubing_ID_mm
            or (proposal.tubing_ID_mm if proposal and proposal.tubing_ID_mm else None)
            or 1.0
        )
        is_photochem = self._is_photochem(proposal, chemistry_plan)
        if is_photochem and d_init > 1.0:
            d_init = 1.0  # Beer–Lambert constraint

        solvent = getattr(batch_record, "solvent", None)
        is_gas_liquid = self._is_gas_liquid(batch_record, chemistry_plan, proposal)
        calc.is_gas_liquid = is_gas_liquid

        # Flow temperature may differ from batch temperature
        T_flow_C = None
        if proposal and proposal.temperature_C:
            T_batch_C = getattr(batch_record, "temperature_C", None)
            if T_batch_C and abs(proposal.temperature_C - T_batch_C) > 2:
                T_flow_C = proposal.temperature_C

        self._step1(calc, batch_record)
        # Step 2: if the council approved a specific τ, use it directly.
        # Otherwise run the kinetics-based estimation from batch data.
        tau_override = target_residence_time_min or (
            proposal.residence_time_min
            if proposal and proposal.residence_time_min and proposal.residence_time_min > 0
            else None
        )
        if tau_override and tau_override > 0:
            self._step2_override(calc, tau_override, batch_record, chemistry_plan, analogies)
        else:
            self._step2(calc, batch_record, chemistry_plan, analogies, T_flow_C)
        self._steps345(
            calc, Q_init, d_init, solvent, inventory, is_photochem,
            is_gas_liquid=is_gas_liquid, proposal=proposal,
            batch_record=batch_record, chemistry_plan=chemistry_plan,
        )
        self._step6(calc)
        self._step7(calc, batch_record, chemistry_plan, solvent)
        self._step8(calc, solvent, is_gas_liquid)
        self._step9(calc, batch_record)
        self._verify(calc)

        # Material compatibility
        if solvent:
            for (mat, sol_key), concern in INCOMPATIBLE_COMBOS.items():
                if sol_key.lower() in solvent.lower():
                    calc.material_warnings.append(f"{mat} + {sol_key}: {concern}")

        return calc

    @staticmethod
    def annotate_proposal_with_calculations(proposal, calc: DesignCalculations):
        """Attach deterministic gas/heat metadata to the final proposal.

        This keeps topology/GUI rendering synchronized with the calculator
        without asking the LLM to invent gas MFC values.
        """
        if proposal is None:
            return proposal
        # Deterministic post-council corrections must be reflected in the
        # proposal itself; otherwise the GUI can show a council geometry while
        # calculations/topology use a different physical reactor.
        if calc.flow_rate_mL_min > 0:
            proposal.flow_rate_mL_min = round(calc.flow_rate_mL_min, 5)
        if calc.tubing_ID_mm > 0:
            proposal.tubing_ID_mm = round(calc.tubing_ID_mm, 3)
        if calc.reactor_volume_mL > 0:
            proposal.reactor_volume_mL = round(calc.reactor_volume_mL, 4)
        if calc.bpr_required and calc.bpr_pressure_bar > 0:
            proposal.BPR_bar = round(calc.bpr_pressure_bar, 1)
        if getattr(calc, "is_gas_liquid", False):
            proposal.multiphase_metrics = {
                "gas_species": calc.gas_species,
                "gas_oxygen_fraction": calc.gas_oxygen_fraction,
                "liquid_flow_rate_mL_min": calc.liquid_flow_rate_mL_min,
                "gas_flow_sccm": calc.gas_flow_sccm,
                "gas_sccm_uncapped": getattr(calc, "gas_sccm_uncapped", 0.0),
                "gas_flow_actual_mL_min": calc.gas_flow_actual_mL_min,
                "gas_actual_uncapped_mL_min": getattr(calc, "gas_actual_uncapped_mL_min", 0.0),
                "gas_flow_capped_by_holdup": getattr(calc, "gas_flow_capped_by_holdup", False),
                "gas_pressure_abs_bar": calc.gas_pressure_abs_bar,
                "gas_liquid_ratio": calc.gas_liquid_ratio,
                "gas_holdup": calc.gas_holdup,
                "liquid_holdup_volume_mL": calc.liquid_holdup_volume_mL,
                "total_reactor_volume_mL": calc.reactor_volume_mL,
                "two_phase_multiplier": calc.two_phase_multiplier,
                "two_phase_pressure_drop_bar": calc.two_phase_pressure_drop_bar,
                "o2_supply_mmol_min": calc.o2_supply_mmol_min,
                "o2_required_mmol_min": calc.o2_required_mmol_min,
                "o2_equiv_supplied": calc.o2_equiv_supplied,
                "dissolved_o2_mM": calc.dissolved_o2_mM,
                "kLa_s": calc.kLa_s,
                "o2_transfer_capacity_mmol_min": calc.o2_transfer_capacity_mmol_min,
                "o2_transfer_sufficiency": calc.o2_transfer_sufficiency,
            }
            gas_assigned = False
            for stream in proposal.streams or []:
                if DesignCalculator._stream_assignment_is_gas(stream):
                    stream.phase = "gas"
                    stream.gas_flow_sccm = round(calc.gas_flow_sccm, 3)
                    stream.gas_flow_actual_mL_min = round(calc.gas_flow_actual_mL_min, 4)
                    stream.flow_rate_mL_min = round(calc.gas_flow_actual_mL_min, 4)
                    gas_assigned = True
            if not gas_assigned and proposal.streams is not None:
                from flora_translate.schemas import StreamAssignment
                proposal.streams.append(StreamAssignment(
                    stream_label="G",
                    pump_role=f"{calc.gas_species or 'gas'} feed",
                    contents=[calc.gas_species or "gas"],
                    phase="gas",
                    gas_flow_sccm=round(calc.gas_flow_sccm, 3),
                    gas_flow_actual_mL_min=round(calc.gas_flow_actual_mL_min, 4),
                    flow_rate_mL_min=round(calc.gas_flow_actual_mL_min, 4),
                    reasoning="Deterministic gas-liquid calculator added MFC feed for gas-phase reagent.",
                ))
        proposal.heat_transfer_metrics = {
            "surface_to_volume_m_inv": calc.surface_to_volume,
            "heat_transfer_area_m2": calc.heat_transfer_area_m2,
            "UA_W_K": calc.UA_W_K,
            "heat_generation_W": calc.heat_generation_W,
            "heat_removal_W": calc.heat_removal_W,
            "thermal_damkohler": calc.thermal_damkohler,
            "heat_transfer_score": calc.heat_transfer_score,
        }
        return proposal

    # ─── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _is_photochem(proposal, chemistry_plan) -> bool:
        if proposal and proposal.wavelength_nm:
            return True
        if chemistry_plan:
            if chemistry_plan.recommended_wavelength_nm:
                return True
            if any("photo" in (r.role or "").lower() for r in chemistry_plan.reagents):
                return True
        return False

    @staticmethod
    def _is_gas_liquid(batch_record, chemistry_plan, proposal) -> bool:
        """Detect if the process involves a gas-liquid phase regime.

        N2/Ar as atmosphere = inert blanket, NOT gas-liquid.
        O2/H2/CO/CO2 as atmosphere or reagent = gas-liquid.
        """
        # Reagent gases that make a process gas-liquid
        # NOTE: "co" excluded — too short, matches "coupling", "collection", etc.
        _REAGENT_GAS = {"o2", "o₂", "oxygen", "h2", "h₂", "hydrogen",
                        "co2", "co₂", "syngas", "ethylene", "acetylene",
                        "carbon monoxide", "carbonylation", "ozone", "o3", "o₃",
                        "chlorine", "cl2", "cl₂", "ammonia", "nh3", "nh₃",
                        "hydrogen chloride", "hcl gas", "sulfur dioxide", "so2", "so₂"}
        # Phase keywords
        _PHASE_KW = {"gas-liquid", "gas_liquid", "segmented", "slug flow",
                     "mfc", "bubbl", "sparging gas", "gas reagent"}
        # Inert gases — do NOT trigger gas-liquid when used as atmosphere
        _INERT = {"n2", "n₂", "nitrogen", "ar", "argon", "helium"}

        def _has_reagent_gas_context(text: str) -> bool:
            text = text.lower()
            negative = (
                "o2-sensitive", "o₂-sensitive", "oxygen-sensitive",
                "oxygen free", "oxygen-free", "o2-free", "o₂-free",
                "deoxygen", "exclude oxygen", "strictly o2 free",
            )
            if any(term in text for term in negative):
                stripped = text
                for term in negative:
                    stripped = stripped.replace(term, "")
            else:
                stripped = text
            contextual = (
                "molecular oxygen", "oxygen introduced", "o2 introduced", "o₂ introduced",
                "from air", "air introduced", "open to air", "under air",
                "air feed", "oxygen feed", "o2 feed", "o₂ feed",
                "gas feed", "gas injection", "sparge", "sparging", "bubble",
                "bubbling", "segmented", "slug flow", "gas-liquid", "gas_liquid",
                "under h2", "under h₂", "hydrogenation", "carbon monoxide",
                "syngas", "co2 feed", "co₂ feed", "ozone feed", "chlorine feed",
                "ammonia feed", "hcl gas feed", "so2 feed", "so₂ feed",
            )
            return any(term in stripped for term in contextual)

        # Check atmosphere field: only reagent gases count
        atm = str(getattr(batch_record, "atmosphere", "") or "").lower().strip()
        if atm and atm not in _INERT and any(g in atm for g in _REAGENT_GAS):
            return True

        # Check description and additives for reagent gas mentions
        texts = []
        if batch_record:
            texts.append(str(getattr(batch_record, "reaction_description", "") or ""))
            for a in (getattr(batch_record, "additives", None) or []):
                texts.append(str(a))
        if chemistry_plan:
            if "gas" in (chemistry_plan.mechanism_type or "").lower():
                return True
            for stage in chemistry_plan.stages:
                stage_atm = (stage.atmosphere or "").lower()
                if stage_atm and stage_atm not in _INERT:
                    if any(g in stage_atm for g in _REAGENT_GAS):
                        return True
            for r in chemistry_plan.reagents:
                texts.append(str(r.name or ""))
                texts.append(str(r.role or ""))
        if proposal:
            for s in (proposal.streams or []):
                for c in (s.contents or []):
                    texts.append(str(c))

        combined = " ".join(texts).lower()
        if _has_reagent_gas_context(combined):
            return True
        if any(kw in combined for kw in _PHASE_KW):
            return True
        return False

    @staticmethod
    def _stream_assignment_is_gas(stream) -> bool:
        phase = str(getattr(stream, "phase", "") or getattr(stream, "state", "") or "").lower()
        if phase in {"gas", "gaseous", "vapor", "vapour"}:
            return True
        if phase in {"liquid", "solution"}:
            return False
        solvent = str(getattr(stream, "solvent", "") or "")
        if solvent:
            solvent_l = solvent.lower()
            no_real_solvent = solvent_l.strip() in {"none", "no solvent", "n/a", "na", "null", "gas"}
            if not no_real_solvent and "gas" not in solvent_l and "vapor" not in solvent_l and "vapour" not in solvent_l:
                return False
        texts = [
            str(getattr(stream, "pump_role", "") or ""),
            " ".join(str(c) for c in (getattr(stream, "contents", None) or [])),
        ]
        text = " ".join(texts).lower()
        if any(w in text for w in ("quench", "neutralization", "neutralisation", "workup")):
            return False
        gas_words = (
            "air", "oxygen", "o2", "o₂", "hydrogen", "h2", "h₂", "co2", "co₂",
            "carbon monoxide", "syngas", "ethylene", "acetylene", "gas feed",
            "gas injection", "mfc", "ozone", "o3", "o₃", "chlorine", "cl2", "cl₂",
            "ammonia", "nh3", "nh₃", "hydrogen chloride", "hcl gas",
            "sulfur dioxide", "so2", "so₂",
        )
        liquid_words = ("solution", "solvent", "in mecn", "in ethanol", "in etoh", "aqueous")
        return any(w in text for w in gas_words) and not any(w in text for w in liquid_words)

    @staticmethod
    def _detect_gas_species(batch_record, chemistry_plan, proposal) -> tuple[str, float]:
        texts: list[str] = []
        if batch_record:
            texts.append(str(getattr(batch_record, "reaction_description", "") or ""))
            texts.append(str(getattr(batch_record, "atmosphere", "") or ""))
        if chemistry_plan:
            texts.append(str(getattr(chemistry_plan, "mechanism_type", "") or ""))
            for stage in chemistry_plan.stages or []:
                texts.append(str(stage.atmosphere or ""))
                texts.append(str(stage.reaction_type or ""))
                for feed in stage.feed_streams or []:
                    texts.append(str(getattr(feed, "pump_role", "") or ""))
                    texts.append(str(getattr(feed, "reasoning", "") or ""))
                    texts.append(str(getattr(feed, "phase", "") or ""))
                    texts.extend(str(r) for r in feed.reagents or [])
            for reagent in chemistry_plan.reagents or []:
                texts.append(str(reagent.name or ""))
                texts.append(str(reagent.role or ""))
        if proposal:
            for stream in proposal.streams or []:
                texts.append(str(stream.pump_role or ""))
                texts.extend(str(c) for c in stream.contents or [])
        text = " ".join(texts).lower()

        def has_any(*patterns: str) -> bool:
            return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

        # Prioritize explicit reagent gases over incidental sensitivity text
        # such as "oxygen-sensitive". Put O2 after the specific gases.
        if has_any(r"\bair\b"):
            return "air", 0.21
        if has_any(r"\bozone\b", r"(?<![A-Za-z0-9])o3(?![A-Za-z0-9])", r"o₃"):
            return "O3", 0.0
        if has_any(r"\bsyngas\b"):
            return "syngas", 0.0
        if has_any(r"(?<![A-Za-z0-9])co2(?![A-Za-z0-9])", r"co₂", r"\bcarbon dioxide\b"):
            return "CO2", 0.0
        if has_any(r"\bcarbon monoxide\b"):
            return "CO", 0.0
        if has_any(r"\bhydrogen\b", r"(?<![A-Za-z0-9])h2(?![A-Za-z0-9])", r"h₂"):
            return "H2", 0.0
        if has_any(r"\bchlorine\b", r"(?<![A-Za-z0-9])cl2(?![A-Za-z0-9])", r"cl₂"):
            return "Cl2", 0.0
        if has_any(r"\bammonia\b", r"(?<![A-Za-z0-9])nh3(?![A-Za-z0-9])", r"nh₃"):
            return "NH3", 0.0
        if has_any(r"\bhydrogen chloride\b", r"\bhcl gas\b"):
            return "HCl", 0.0
        if has_any(r"\bsulfur dioxide\b", r"(?<![A-Za-z0-9])so2(?![A-Za-z0-9])", r"so₂"):
            return "SO2", 0.0
        o2_negative = (
            "o2-sensitive", "o₂-sensitive", "oxygen-sensitive",
            "oxygen/moisture-sensitive", "oxygen free", "oxygen-free",
            "o2-free", "o₂-free",
        )
        o2_text = text
        for term in o2_negative:
            o2_text = o2_text.replace(term, "")
        if any(k in o2_text for k in ("oxygen", "o2", "o₂")):
            return "O2", 1.0
        return "gas", 0.0

    def _estimate_gas_context(
        self, calc, Q_liquid_mL_min: float, proposal, is_gas_liquid: bool,
        batch_record=None, chemistry_plan=None,
    ) -> dict:
        if not is_gas_liquid or Q_liquid_mL_min <= 0:
            return {}
        species, y_o2 = self._detect_gas_species(batch_record, chemistry_plan, proposal)
        P_gauge_bar = 0.0
        try:
            P_gauge_bar = float(getattr(proposal, "BPR_bar", 0.0) or 0.0)
        except (TypeError, ValueError):
            P_gauge_bar = 0.0
        P_abs_bar = max(P_gauge_bar + P_STP_BAR, 6.0 if is_gas_liquid else P_STP_BAR)
        T_K = calc.temperature_K or 298.15

        explicit_sccm = None
        if proposal:
            for stream in proposal.streams or []:
                if not self._stream_assignment_is_gas(stream):
                    continue
                for attr in ("gas_flow_sccm", "flow_rate_sccm"):
                    value = getattr(stream, attr, None)
                    if value:
                        explicit_sccm = float(value)
                        break
                if explicit_sccm:
                    break

        n_substrate_mmol_min = Q_liquid_mL_min * max(calc.concentration_M or 0.1, 1e-9)
        o2_equiv_required = 1.0 if y_o2 > 0 else 0.0
        o2_required = n_substrate_mmol_min * o2_equiv_required
        supply_factor = 3.0
        if explicit_sccm is not None:
            gas_sccm = explicit_sccm
        elif y_o2 > 0:
            n_gas_mmol_min = o2_required * supply_factor / max(y_o2, 1e-9)
            gas_sccm = (n_gas_mmol_min / 1000.0) * R_GAS_L_BAR * T_STP_K / P_STP_BAR * 1000.0
        else:
            # Non-O2 reagent gas default: one gas mol per substrate mol with 3x excess.
            n_gas_mmol_min = n_substrate_mmol_min * 3.0
            gas_sccm = (n_gas_mmol_min / 1000.0) * R_GAS_L_BAR * T_STP_K / P_STP_BAR * 1000.0

        gas_actual = gas_sccm * (T_K / T_STP_K) * (P_STP_BAR / P_abs_bar)
        gas_sccm_uncapped = gas_sccm
        gas_actual_uncapped = gas_actual
        glr = gas_actual / max(Q_liquid_mL_min, 1e-9)
        # Holdup must be consistent with the displayed actual gas flow. With a
        # no-slip approximation, V_total = tau * (Q_liquid + Q_gas_actual).
        # The upper bound only prevents singular volumes for pathological gas
        # excess; it does not silently alter the MFC setpoint.
        eps = max(0.02, min(0.85, gas_actual / max(gas_actual + Q_liquid_mL_min, 1e-9)))
        multiplier = min(12.0, 1.0 + 12.0 * eps + 25.0 * eps * eps)

        o2_supply = 0.0
        if y_o2 > 0:
            n_gas_mol_min = gas_sccm / 1000.0 * P_STP_BAR / (R_GAS_L_BAR * T_STP_K)
            o2_supply = n_gas_mol_min * 1000.0 * y_o2
        p_o2_abs = P_abs_bar * y_o2
        dissolved_o2_mM = 1.3 * (p_o2_abs / 0.21) if y_o2 > 0 else 0.0
        kLa_s = 0.02 + 0.35 * eps if y_o2 > 0 else 0.0
        liquid_holdup_L = calc.residence_time_min * Q_liquid_mL_min / 1000.0
        transfer_capacity = kLa_s * dissolved_o2_mM * liquid_holdup_L * 60.0
        transfer_suff = transfer_capacity / max(o2_required, 1e-12) if o2_required > 0 else 0.0
        o2_equiv_supplied = o2_supply / max(n_substrate_mmol_min, 1e-12) if n_substrate_mmol_min > 0 else 0.0

        return {
            "species": species,
            "y_o2": y_o2,
            "P_abs_bar": P_abs_bar,
            "gas_sccm": gas_sccm,
            "gas_sccm_uncapped": gas_sccm_uncapped,
            "gas_actual_mL_min": gas_actual,
            "gas_actual_uncapped_mL_min": gas_actual_uncapped,
            "gas_flow_capped_by_holdup": False,
            "gas_liquid_ratio": glr,
            "gas_holdup": eps,
            "two_phase_multiplier": multiplier,
            "o2_supply_mmol_min": o2_supply,
            "o2_required_mmol_min": o2_required,
            "o2_equiv_supplied": o2_equiv_supplied,
            "dissolved_o2_mM": dissolved_o2_mM,
            "kLa_s": kLa_s,
            "o2_transfer_capacity_mmol_min": transfer_capacity,
            "o2_transfer_sufficiency": transfer_suff,
        }

    # ═══════════════════════════════════════════════════════════════════
    #  Step 1 — Parse batch conditions
    # ═══════════════════════════════════════════════════════════════════

    def _step1(self, calc: DesignCalculations, br):
        T_C = getattr(br, "temperature_C", None) or 25.0
        T_K = T_C + 273.15
        raw_text = getattr(br, "raw_text", None) or getattr(br, "reaction_description", "")

        explicit_C = getattr(br, "concentration_M", None)
        inferred_C = infer_batch_concentration_M(
            raw_text,
            explicit_concentration_M=explicit_C,
            explicit_scale_mmol=getattr(br, "scale_mmol", None),
        )
        C_M = explicit_C or inferred_C or 0.1

        explicit_t_h = getattr(br, "reaction_time_h", None)
        inferred_t_h = infer_reaction_time_h(raw_text, explicit_t_h)
        t_h = explicit_t_h or inferred_t_h or 0
        t_s = t_h * 3600.0
        y = getattr(br, "yield_pct", None)
        X = min((y or 95) / 100.0, 0.99)

        calc.temperature_C = T_C
        calc.temperature_K = T_K
        calc.concentration_M = C_M
        calc.batch_time_s = t_s
        calc.target_conversion = X

        assumptions = []
        if not getattr(br, "temperature_C", None):
            assumptions.append("Temperature not specified → assumed 25 °C")
        if not explicit_C:
            if inferred_C is not None:
                assumptions.append(
                    f"Concentration recovered deterministically from protocol text → {inferred_C:.3g} M"
                )
            else:
                assumptions.append("Concentration not specified → assumed 0.1 M")
        if not explicit_t_h and inferred_t_h is not None:
            assumptions.append(
                f"Batch reaction time recovered deterministically from protocol text → {inferred_t_h:.3g} h"
            )
        if not y:
            assumptions.append("Yield not specified → targeting 95 % conversion")

        if FLOW_TRANSLATION_POLICY == "intensify" and t_h > 0:
            assumptions.append(
                f"Intensification policy active downstream → tau_flow must remain ≤ "
                f"{t_h * 60.0 * FLOW_MAX_TAU_TO_BATCH_RATIO:.1f} min"
            )

        self._emit(calc, StepResult(
            step=1, name="Batch Conditions",
            status="PASS",
            summary=(
                f"T = {T_C:.0f} °C, C₀ = {C_M} M, "
                f"t_batch = {t_h:.1f} h ({t_s:.0f} s), X = {X:.2f}"
            ),
            values={"T_C": T_C, "T_K": T_K, "C_M": C_M,
                    "t_batch_h": t_h, "t_batch_s": t_s, "X": X},
            assumptions=assumptions,
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Step 2 — Kinetics & residence time (multi-method)
    #
    #  Method A: Analogy-derived IF — from retrieved literature analogies
    #  Method B: Class-level IF — from hardcoded table
    #  Method C: Arrhenius correction — if T_flow ≠ T_batch
    #
    #  Primary: Method A when ≥ 2 analogy data points, else Method B.
    #  Range:   [min(A,B)/1.5 , max(A,B)×1.5]
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_raw_analogy_IFs(analogies: list[dict] | None) -> list[float]:
        """Extract raw batch-to-flow intensification factors from analogy records."""
        if not analogies:
            return []
        factors = []
        for a in analogies:
            full = a.get("full_record") or {}
            if isinstance(full, list):
                full = full[0] if full else {}

            # Source 1: explicit time_reduction_factor
            tl = full.get("translation_logic") or {}
            if isinstance(tl, list):
                tl = tl[0] if tl else {}
            trf = tl.get("time_reduction_factor")
            if trf and float(trf) > 0:
                factors.append(float(trf))
                continue

            # Source 2: compute from batch_time / flow_time
            bb = full.get("batch_baseline") or {}
            fo = full.get("flow_optimized") or {}
            if isinstance(bb, list):
                bb = bb[0] if bb else {}
            if isinstance(fo, list):
                fo = fo[0] if fo else {}
            bt = bb.get("reaction_time_min")
            ft = fo.get("residence_time_min")
            if bt and ft and float(bt) > 0 and float(ft) > 0:
                factors.append(float(bt) / float(ft))
        return factors

    @staticmethod
    def _floor_analogy_IFs(factors: list[float]) -> list[float]:
        """Prevent literature extraction artifacts from de-intensifying flow designs."""
        return [max(1.0, float(f)) for f in factors if f and float(f) > 0]

    @classmethod
    def _extract_analogy_IFs(cls, analogies: list[dict] | None) -> list[float]:
        """Extract analogy IFs and floor sub-unity values to no intensification."""
        return cls._floor_analogy_IFs(cls._extract_raw_analogy_IFs(analogies))

    def _step2(self, calc: DesignCalculations, br, chem_plan,
               analogies=None, T_flow_C=None):
        X = calc.target_conversion
        t_batch = calc.batch_time_s
        T_batch_K = calc.temperature_K
        chem_type = _detect_chemistry(br, chem_plan)

        equations: list[str] = []
        assumptions: list[str] = []
        warnings: list[str] = []

        # ── Class-level IF (always available) ───────────────────────────
        IF_class = INTENSIFICATION.get(chem_type, INTENSIFICATION["default"])
        calc.if_class = IF_class

        # ── Analogy-derived IF ──────────────────────────────────────────
        raw_analogy_factors = self._extract_raw_analogy_IFs(analogies)
        analogy_factors = self._floor_analogy_IFs(raw_analogy_factors)
        n_analogy = len(analogy_factors)
        calc.n_analogy_datapoints = n_analogy
        if any(float(f) < 1.0 for f in raw_analogy_factors):
            warnings.append(
                "One or more analogy IF values were below 1.0 and were floored to 1.0 "
                "to prevent extracted literature records from de-intensifying the flow design."
            )

        IF_analogy = None
        if n_analogy >= 1:
            import statistics
            IF_analogy = statistics.median(analogy_factors)
            calc.if_analogy = round(IF_analogy, 1)

        # ── Choose primary IF ───────────────────────────────────────────
        # Prefer analogy IF when we have ≥ 2 data points (more reliable).
        # With 1 data point, average analogy and class.
        # With 0, fall back to class.
        if n_analogy >= 2:
            IF_primary = IF_analogy
            method = "analogy"
        elif n_analogy == 1:
            IF_primary = (IF_analogy + IF_class) / 2
            method = "analogy+class"
        else:
            IF_primary = IF_class
            method = "class"

        calc.intensification_factor = round(IF_primary, 1)

        if t_batch <= 0 or X <= 0:
            # No batch data → fall back to defaults
            tau_min = 10.0
            tau_s = 600.0
            calc.residence_time_s = tau_s
            calc.residence_time_min = tau_min
            calc.kinetics_method = "default"
            assumptions.append("No batch reaction time → using 10 min default")
            self._emit(calc, StepResult(
                step=2, name="Kinetics & Residence Time",
                status="ESTIMATED",
                summary=f"τ = {tau_min:.0f} min (default — no batch time provided)",
                values={"tau_min": tau_min, "method": "default"},
                assumptions=assumptions,
            ))
            return

        # ── Back-calculate k_batch (first-order) ───────────────────────
        ln_term = -math.log(1 - X)
        k_batch = ln_term / t_batch  # s⁻¹
        t_batch_min = t_batch / 60.0

        # ── Method B: Class-level τ ─────────────────────────────────────
        tau_class_s = t_batch / IF_class
        tau_class_min = tau_class_s / 60.0
        calc.tau_class_min = round(tau_class_min, 2)

        # ── Method A: Analogy-derived τ ─────────────────────────────────
        tau_analogy_min = None
        if IF_analogy and IF_analogy > 0:
            tau_analogy_s = t_batch / IF_analogy
            tau_analogy_min = tau_analogy_s / 60.0
            calc.tau_analogy_min = round(tau_analogy_min, 2)

        # ── Primary τ (before Arrhenius) ────────────────────────────────
        tau_primary_s = t_batch / IF_primary
        tau_primary_min = tau_primary_s / 60.0

        # ── Method C: Arrhenius temperature correction ──────────────────
        arrhenius_factor = 1.0
        Ea = ESTIMATED_EA.get(chem_type, ESTIMATED_EA["default"])

        if T_flow_C is not None:
            T_flow_K = T_flow_C + 273.15
            # k(T_flow) / k(T_batch) = exp(-Ea/R × (1/T_flow - 1/T_batch))
            exponent = -Ea / R_GAS * (1.0 / T_flow_K - 1.0 / T_batch_K)
            arrhenius_factor = math.exp(exponent)
            calc.arrhenius_correction = round(arrhenius_factor, 3)

            # Higher T → larger k → shorter τ
            tau_primary_s /= arrhenius_factor
            tau_primary_min = tau_primary_s / 60.0

            equations.append(
                rf"\frac{{k(T_{{\mathrm{{flow}}}})}}{{k(T_{{\mathrm{{batch}}}})}} "
                rf"= \exp\!\left(\frac{{-E_a}}{{R}}"
                rf"\left(\frac{{1}}{{{T_flow_K:.1f}}} - \frac{{1}}{{{T_batch_K:.1f}}}\right)\right)"
                rf" = \exp\!\left(\frac{{-{Ea:.0f}}}{{{R_GAS:.3f}}}"
                rf"\times {1/T_flow_K - 1/T_batch_K:.6f}\right)"
                rf" = {arrhenius_factor:.3f}"
            )
            if arrhenius_factor > 1.05:
                equations.append(
                    rf"\tau_{{\mathrm{{corrected}}}} = \frac{{\tau_{{\mathrm{{primary}}}}}}"
                    rf"{{{arrhenius_factor:.3f}}}"
                    rf" = \frac{{{t_batch/IF_primary/60:.1f}}}{{{arrhenius_factor:.3f}}}"
                    rf" = {tau_primary_min:.1f}\;\mathrm{{min}}"
                )

        # ── Store primary result ────────────────────────────────────────
        k_flow = k_batch * IF_primary * arrhenius_factor
        calc.rate_constant = k_flow
        calc.reaction_order = 1
        calc.kinetics_method = method
        calc.residence_time_s = tau_primary_s
        calc.residence_time_min = round(tau_primary_min, 2)

        # ── Compute range from all methods ──────────────────────────────
        all_tau = [tau_class_min]
        if tau_analogy_min is not None:
            all_tau.append(tau_analogy_min)
        # Apply Arrhenius to range bounds too
        if arrhenius_factor != 1.0:
            all_tau = [t / arrhenius_factor for t in all_tau]

        tau_low = round(min(all_tau) / 1.5, 2)
        tau_high = round(max(all_tau) * 1.5, 2)
        calc.residence_time_range_min = (tau_low, tau_high)

        # ── Build equations ─────────────────────────────────────────────
        equations.insert(0,
            (r"k_{\mathrm{batch}} = \frac{-\ln(1-X)}{t_{\mathrm{batch}}} = "
             rf"\frac{{{ln_term:.4f}}}{{{t_batch:.0f}}} = {k_batch:.4e}"
             r"\;\mathrm{s^{-1}}")
        )

        # Method B equation
        equations.append(
            rf"\tau_{{\mathrm{{class}}}} = \frac{{t_{{\mathrm{{batch}}}}}}{{IF_{{\mathrm{{class}}}}}}"
            rf" = \frac{{{t_batch_min:.0f}}}{{{IF_class:.0f}}}"
            rf" = {tau_class_min:.1f}\;\mathrm{{min}}"
            rf"\quad (IF_{{\mathrm{{class}}}} = {IF_class:.0f}\times,"
            rf"\;\text{{{chem_type} median}})"
        )

        # Method A equation
        if tau_analogy_min is not None:
            factors_str = ", ".join(f"{f:.0f}" for f in analogy_factors)
            equations.append(
                rf"\tau_{{\mathrm{{analogy}}}} = \frac{{t_{{\mathrm{{batch}}}}}}{{IF_{{\mathrm{{analogy}}}}}}"
                rf" = \frac{{{t_batch_min:.0f}}}{{{IF_analogy:.0f}}}"
                rf" = {tau_analogy_min:.1f}\;\mathrm{{min}}"
                rf"\quad (IF_{{\mathrm{{analogy}}}} = \mathrm{{median}}({factors_str})"
                rf" = {IF_analogy:.0f}\times,"
                rf"\;n = {n_analogy})"
            )

        # ── Build assumptions ───────────────────────────────────────────
        assumptions.append("First-order kinetics assumed (n = 1)")
        if method == "analogy":
            assumptions.append(
                f"Primary IF from {n_analogy} literature analogies "
                f"(IF = {IF_analogy:.0f}×)"
            )
        elif method == "analogy+class":
            assumptions.append(
                f"IF averaged from 1 analogy ({IF_analogy:.0f}×) "
                f"and class default ({IF_class:.0f}×) → {IF_primary:.0f}×"
            )
        else:
            assumptions.append(
                f"IF = {IF_class:.0f}× from {chem_type} class median "
                "(no analogy data available)"
            )
        if T_flow_C is not None:
            T_batch_C = calc.temperature_C
            assumptions.append(
                f"Arrhenius correction: T_batch = {T_batch_C:.0f}°C → "
                f"T_flow = {T_flow_C:.0f}°C, Ea ≈ {Ea/1000:.0f} kJ/mol "
                f"→ k ratio = {arrhenius_factor:.2f}×"
            )

        # ── Warnings ────────────────────────────────────────────────────
        if tau_primary_min < 0.5:
            warnings.append(
                f"Very short τ ({tau_primary_min:.2f} min) — verify experimentally"
            )
        if tau_primary_min > 120:
            warnings.append(
                f"Long τ ({tau_primary_min:.0f} min) — consider higher T or catalyst loading"
            )
        if tau_analogy_min is not None and tau_class_min > 0:
            ratio = max(tau_analogy_min, tau_class_min) / min(tau_analogy_min, tau_class_min)
            if ratio > 3:
                warnings.append(
                    f"Methods disagree: τ_analogy = {tau_analogy_min:.1f} min vs "
                    f"τ_class = {tau_class_min:.1f} min ({ratio:.1f}× difference). "
                    "Experimental verification strongly recommended."
                )

        status = "WARNING" if warnings else "PASS"

        # ── Summary ─────────────────────────────────────────────────────
        parts = [f"τ = {tau_primary_min:.1f} min"]
        parts.append(f"range [{tau_low:.1f}–{tau_high:.1f}] min")
        parts.append(f"via {method} (IF = {IF_primary:.0f}×)")
        if arrhenius_factor != 1.0:
            parts.append(f"Arrhenius {arrhenius_factor:.2f}×")
        summary = ", ".join(parts)

        # ── Values dict for Streamlit ───────────────────────────────────
        values = {
            "tau_primary_min": tau_primary_min,
            "tau_range_low": tau_low,
            "tau_range_high": tau_high,
            "method": method,
            "IF_primary": IF_primary,
            "IF_class": IF_class,
            "tau_class_min": tau_class_min,
            "chemistry_type": chem_type,
            "k_batch": k_batch,
            "k_flow": k_flow,
        }
        if IF_analogy is not None:
            values["IF_analogy"] = IF_analogy
            values["tau_analogy_min"] = tau_analogy_min
            values["n_analogy_datapoints"] = n_analogy
            values["analogy_IFs"] = analogy_factors
            values["raw_analogy_IFs"] = raw_analogy_factors
            values["IF_floor_applied"] = any(float(f) < 1.0 for f in raw_analogy_factors)
        if T_flow_C is not None:
            values["arrhenius_factor"] = arrhenius_factor
            values["Ea_J_mol"] = Ea
            values["T_flow_C"] = T_flow_C

        self._emit(calc, StepResult(
            step=2, name="Kinetics & Residence Time",
            status=status, summary=summary,
            values=values,
            equations=equations, assumptions=assumptions, warnings=warnings,
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Step 2 (override) — Use council-approved τ directly
    #  Called when a validated proposal already has a specific τ.
    #  All derived quantities (V_R, L, Re, ΔP) will be computed FROM
    #  this τ — it is the authoritative source, not re-derived from kinetics.
    # ═══════════════════════════════════════════════════════════════════

    def _step2_override(self, calc, tau_min: float, batch_record, chemistry_plan,
                        analogies=None):
        """Set τ directly from the approved value, compute kinetics context."""
        tau_s = tau_min * 60.0
        calc.residence_time_min = round(tau_min, 2)
        calc.residence_time_s = tau_s
        calc.kinetics_method = "council-approved"
        calc.residence_time_range_min = (round(tau_min * 0.8, 2), round(tau_min * 1.2, 2))

        # Still compute IF for context / display — but it is NOT used to derive τ
        t_batch = calc.batch_time_s
        chem_type = _detect_chemistry(batch_record, chemistry_plan)
        IF_class = INTENSIFICATION.get(chem_type, INTENSIFICATION["default"])
        calc.if_class = IF_class
        analogy_factors = self._extract_analogy_IFs(analogies)
        n_analogy = len(analogy_factors)
        calc.n_analogy_datapoints = n_analogy
        if n_analogy >= 1:
            import statistics
            calc.if_analogy = round(statistics.median(analogy_factors), 1)

        # Back-calculate what IF this τ implies (for display only)
        if t_batch > 0:
            implied_IF = t_batch / tau_s
            calc.intensification_factor = round(implied_IF, 1)
        else:
            calc.intensification_factor = IF_class

        # ── Compute rate_constant so Step 6 uses k·d²/D (scales as d²) ──
        # Without this, Step 6 falls back to τ/t_mix which scales as 1/d²
        # — the wrong direction when d decreases to fix mixing.
        X = calc.target_conversion
        if t_batch > 0 and X > 0 and tau_s > 0:
            k_batch = -math.log(1.0 - X) / t_batch   # s⁻¹
            # Use implied IF (from approved τ) for k_flow; ensures Da ∝ d²
            k_flow = k_batch * (t_batch / tau_s)
            calc.rate_constant = k_flow
        # else: leave at default 0 — fallback formula used (acceptable for edge cases)

        summary = (
            f"τ = {tau_min:.1f} min (council-approved — not re-derived from kinetics). "
            f"Implied IF = {calc.intensification_factor:.0f}× vs batch."
        )

        self._emit(calc, StepResult(
            step=2, name="Kinetics & Residence Time",
            status="PASS",
            summary=summary,
            values={
                "tau_primary_min": tau_min,
                "method": "council-approved",
                "IF_implied": calc.intensification_factor,
                "IF_class": IF_class,
                "k_flow": calc.rate_constant,
            },
            assumptions=["τ set by ENGINE council — kinetics estimation bypassed"],
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Steps 3-4-5 — Reactor sizing + Fluid dynamics + Pressure drop
    #  (coupled: changing d affects L, Re, and ΔP — iterate until OK)
    # ═══════════════════════════════════════════════════════════════════

    def _steps345(
        self, calc, Q_mL_min, d_mm, solvent, inventory, is_photochem,
        is_gas_liquid=False, proposal=None, batch_record=None, chemistry_plan=None,
    ):
        tau_s = calc.residence_time_s or 600.0
        tau_min = tau_s / 60.0

        # Solvent properties
        mu_cP = _lookup(solvent, SOLVENT_VISCOSITY_cP) or 1.0
        rho_gmL = _lookup(solvent, SOLVENT_DENSITY_g_mL) or 1.0
        mu = mu_cP * 1e-3       # Pa·s
        rho = rho_gmL * 1000.0  # kg/m³
        calc.viscosity_Pa_s = mu
        calc.density_kg_m3 = rho

        # Pump max from inventory
        pump_max = 0.0
        if inventory and inventory.pumps:
            pump_max = max(p.max_pressure_bar for p in inventory.pumps)
        if pump_max == 0:
            pump_max = 20.0  # conservative default
        calc.pump_max_bar = pump_max

        adj_4: list[str] = []   # Re adjustments
        adj_5: list[str] = []   # ΔP adjustments

        gas_ctx = self._estimate_gas_context(
            calc, Q_mL_min, proposal, is_gas_liquid,
            batch_record=batch_record, chemistry_plan=chemistry_plan,
        )

        for _iter in range(5):
            Q_m3s = Q_mL_min * 1e-6 / 60.0
            Q_gas_actual_m3s = (gas_ctx.get("gas_actual_mL_min", 0.0) or 0.0) * 1e-6 / 60.0
            d_m = d_mm * 1e-3
            A = PI * (d_m / 2) ** 2
            V_liquid_m3 = tau_s * Q_m3s
            gas_holdup = gas_ctx.get("gas_holdup", 0.0) or 0.0
            V_m3 = V_liquid_m3 / max(1.0 - gas_holdup, 1e-9)
            V_mL = V_m3 * 1e6
            L = (4.0 * V_m3 / (PI * d_m ** 2)) if d_m > 0 else 0.0
            v = (Q_m3s + Q_gas_actual_m3s) / A if A > 0 else 0.0
            v_liquid = Q_m3s / A if A > 0 else 0.0
            Re = (rho * v_liquid * d_m / mu) if mu > 0 else 0.0
            dP_liquid_Pa = (128.0 * mu * L * Q_m3s / (PI * d_m ** 4)) if d_m > 0 else float("inf")
            dP_Pa = dP_liquid_Pa * (gas_ctx.get("two_phase_multiplier", 1.0) or 1.0)
            dP_bar = dP_Pa * 1e-5

            need_redo = False

            # Re check: must be < 2100 (laminar)
            if Re > 2100:
                d_new_m = 4.0 * rho * Q_m3s / (PI * 2000.0 * mu)
                d_new_mm = math.ceil(d_new_m * 1e3 * 10) / 10  # round up to 0.1 mm
                if is_photochem:
                    d_new_mm = min(d_new_mm, 1.6)
                adj_4.append(
                    f"Re = {Re:.0f} > 2100 → increased d from "
                    f"{d_mm:.2f} to {d_new_mm:.2f} mm"
                )
                d_mm = d_new_mm
                need_redo = True

            # Gas-liquid designs must remain inside routine BPR hardware, not
            # merely below the pump's absolute pressure rating.
            elif is_gas_liquid and dP_bar > GAS_LIQUID_MAX_ROUTINE_DELTA_P_BAR:
                ratio = (dP_bar / max(GAS_LIQUID_MAX_ROUTINE_DELTA_P_BAR * 0.75, 1e-9)) ** 0.25
                d_new_mm = max(math.ceil(d_mm * ratio * 10) / 10, d_mm + 0.1)
                if is_photochem:
                    d_new_mm = min(d_new_mm, 1.0)
                if d_new_mm <= d_mm + 1e-9:
                    adj_5.append(
                        f"Gas-liquid ΔP = {dP_bar:.2f} bar would require "
                        f"BPR > {GAS_LIQUID_ROUTINE_MAX_BPR_BAR:.0f} bar; "
                        "ID already at photochemical upper bound"
                    )
                    break
                adj_5.append(
                    f"Gas-liquid ΔP = {dP_bar:.2f} bar would require "
                    f"BPR > {GAS_LIQUID_ROUTINE_MAX_BPR_BAR:.0f} bar → increased d "
                    f"from {d_mm:.2f} to {d_new_mm:.2f} mm"
                )
                d_mm = d_new_mm
                need_redo = True

            # ΔP check: must be < 90 % of pump max
            elif dP_bar > pump_max * 0.9:
                ratio = (dP_bar / (0.5 * pump_max)) ** 0.25
                d_new_mm = max(round(d_mm * ratio, 1), d_mm + 0.25)
                if is_photochem:
                    d_new_mm = min(d_new_mm, 1.6)
                adj_5.append(
                    f"ΔP = {dP_bar:.2f} bar ≈ {dP_bar/pump_max*100:.0f}% pump "
                    f"capacity → increased d from {d_mm:.2f} to {d_new_mm:.2f} mm"
                )
                d_mm = d_new_mm
                need_redo = True

            if not need_redo:
                break

        # Store final values
        calc.flow_rate_mL_min = Q_mL_min
        calc.flow_rate_m3_s = Q_m3s
        calc.liquid_flow_rate_mL_min = Q_mL_min
        calc.tubing_ID_mm = d_mm
        calc.tubing_ID_m = d_m
        calc.reactor_volume_mL = round(V_mL, 4)
        calc.reactor_volume_m3 = V_m3
        calc.tubing_length_m = round(L, 4)
        calc.velocity_m_s = v
        calc.reynolds_number = round(Re, 2)
        calc.flow_regime = (
            "laminar" if Re < 2100
            else "transitional" if Re < 4000
            else "turbulent"
        )
        calc.pressure_drop_bar = round(dP_bar, 6)
        if gas_ctx:
            calc.is_gas_liquid = True
            calc.gas_species = gas_ctx.get("species", "")
            calc.gas_oxygen_fraction = round(gas_ctx.get("y_o2", 0.0), 4)
            calc.gas_flow_sccm = round(gas_ctx.get("gas_sccm", 0.0), 4)
            calc.gas_sccm_uncapped = round(gas_ctx.get("gas_sccm_uncapped", 0.0), 4)
            calc.gas_flow_actual_mL_min = round(gas_ctx.get("gas_actual_mL_min", 0.0), 4)
            calc.gas_actual_uncapped_mL_min = round(gas_ctx.get("gas_actual_uncapped_mL_min", 0.0), 4)
            calc.gas_flow_capped_by_holdup = bool(gas_ctx.get("gas_flow_capped_by_holdup", False))
            calc.gas_pressure_abs_bar = round(gas_ctx.get("P_abs_bar", 1.01325), 4)
            calc.gas_liquid_ratio = round(gas_ctx.get("gas_liquid_ratio", 0.0), 4)
            calc.gas_holdup = round(gas_ctx.get("gas_holdup", 0.0), 4)
            calc.liquid_holdup_volume_mL = round(V_liquid_m3 * 1e6, 4)
            calc.two_phase_multiplier = round(gas_ctx.get("two_phase_multiplier", 1.0), 4)
            calc.two_phase_pressure_drop_bar = round(dP_bar, 6)
            calc.o2_supply_mmol_min = round(gas_ctx.get("o2_supply_mmol_min", 0.0), 6)
            calc.o2_required_mmol_min = round(gas_ctx.get("o2_required_mmol_min", 0.0), 6)
            calc.o2_equiv_supplied = round(gas_ctx.get("o2_equiv_supplied", 0.0), 4)
            calc.dissolved_o2_mM = round(gas_ctx.get("dissolved_o2_mM", 0.0), 4)
            calc.kLa_s = round(gas_ctx.get("kLa_s", 0.0), 5)
            calc.o2_transfer_capacity_mmol_min = round(gas_ctx.get("o2_transfer_capacity_mmol_min", 0.0), 6)
            calc.o2_transfer_sufficiency = round(gas_ctx.get("o2_transfer_sufficiency", 0.0), 4)
        calc.pump_adequate = dP_bar <= pump_max

        # Péclet number: Pe = 192·τ·D_mol / d²
        # Pe ≥ 100 → plug flow valid; Pe < 100 → significant axial dispersion
        if d_m > 0 and tau_s > 0:
            Pe = 192.0 * tau_s * D_MOLECULAR / (d_m ** 2)
            calc.Pe = round(Pe, 1)
            calc.Pe_adequate = Pe >= 100.0

        # ── Emit Step 3: Reactor Sizing ─────────────────────────────────
        Pe_val = calc.Pe or 0.0
        Pe_warn: list[str] = []
        if calc.Pe is not None and not calc.Pe_adequate:
            Pe_warn.append(
                f"Pe = {Pe_val:.0f} < 100 — significant axial dispersion. "
                f"Reactor behaves closer to CSTR than PFR. "
                f"Reduce d (smaller d → higher Pe) or accept lower conversion efficiency."
            )
        if gas_ctx:
            eqs_3 = [
                (rf"V_{{L}} = \tau_L \times Q_L = {tau_min:.2f}"
                 rf" \times {Q_mL_min:.4f} = {V_liquid_m3 * 1e6:.2f}\;\mathrm{{mL}}"),
                (rf"\varepsilon_g = \frac{{Q_g}}{{Q_g+Q_L}}"
                 rf" = {calc.gas_holdup:.3f}"),
                (rf"V_R = \frac{{V_L}}{{1-\varepsilon_g}}"
                 rf" = \frac{{{V_liquid_m3 * 1e6:.2f}}}{{1-{calc.gas_holdup:.3f}}}"
                 rf" = {V_mL:.2f}\;\mathrm{{mL}}"),
                (rf"L = \frac{{4\,V_R}}{{\pi\,d^2}}"
                 rf" = {L:.2f}\;\mathrm{{m}}"),
            ]
        else:
            eqs_3 = [
                (rf"V_R = \tau \times Q = {tau_s:.1f}"
                 rf" \times {Q_m3s:.3e}"
                 rf" = {V_m3:.3e}\;\mathrm{{m^3}}"
                 rf" = {V_mL:.2f}\;\mathrm{{mL}}"),
                (rf"L = \frac{{4\,V_R}}{{\pi\,d^2}}"
                 rf" = \frac{{4 \times {V_m3:.3e}}}"
                 rf"{{\pi \times ({d_m:.4f})^2}}"
                 rf" = {L:.2f}\;\mathrm{{m}}"),
            ]
        eqs_3 += [
            (rf"Pe = \frac{{192\,\tau\,D_{{\mathrm{{mol}}}}}}{{d^2}}"
             rf" = \frac{{192 \times {tau_s:.1f} \times {D_MOLECULAR:.0e}}}"
             rf"{{{d_m:.4f}^2}}"
             rf" = {Pe_val:.0f}"
             + (r"\;\checkmark" if calc.Pe_adequate else r"\;\mathbf{WARNING}")),
        ]
        self._emit(calc, StepResult(
            step=3, name="Reactor Sizing",
            status="WARNING" if Pe_warn else "PASS",
            summary=(
                f"V_R = {V_mL:.2f} mL, d = {d_mm:.2f} mm, "
                f"L = {L:.2f} m, Pe = {Pe_val:.0f} "
                f"({'✓ plug flow' if calc.Pe_adequate else '⚠ axial dispersion'})"
            ),
            values={"V_R_mL": V_mL, "liquid_holdup_mL": V_liquid_m3 * 1e6,
                    "gas_holdup": calc.gas_holdup, "d_mm": d_mm, "L_m": L,
                    "Q_mL_min": Q_mL_min, "v_m_s": v,
                    "Pe": Pe_val, "Pe_adequate": calc.Pe_adequate},
            equations=eqs_3,
            warnings=Pe_warn,
        ))

        # ── Emit Step 4: Fluid Dynamics ─────────────────────────────────
        eqs_4 = [
            (rf"v = \frac{{4\,Q}}{{\pi\,d^2}}"
             rf" = \frac{{4 \times {Q_m3s:.3e}}}"
             rf"{{\pi \times ({d_m:.4f})^2}}"
             rf" = {v:.4f}\;\mathrm{{m/s}}"),
            (rf"Re = \frac{{\rho\,v\,d}}{{\mu}}"
             rf" = \frac{{{rho:.0f} \times {v:.4f} \times {d_m:.4f}}}"
             rf"{{{mu:.4e}}}"
             rf" = {Re:.1f}"),
        ]
        w4 = [f"Re = {Re:.0f} — turbulent flow"] if Re >= 2100 else []
        self._emit(calc, StepResult(
            step=4, name="Fluid Dynamics",
            status="ADJUSTED" if adj_4 else ("WARNING" if w4 else "PASS"),
            summary=f"Re = {Re:.1f} ({calc.flow_regime}), v = {v:.4f} m/s",
            values={"Re": Re, "regime": calc.flow_regime, "v_m_s": v,
                    "mu_Pa_s": mu, "rho_kg_m3": rho,
                    "mu_cP": mu_cP, "rho_g_mL": rho_gmL},
            equations=eqs_4, adjustments=adj_4, warnings=w4,
        ))

        # ── Emit Step 5: Pressure Drop ──────────────────────────────────
        margin = (1 - dP_bar / pump_max) * 100 if pump_max > 0 else 100
        eqs_5 = [
            (rf"\Delta P = \frac{{128\,\mu\,L\,Q}}{{\pi\,d^4}}"
             rf" = \frac{{128 \times {mu:.4e}"
             rf" \times {L:.2f} \times {Q_m3s:.3e}}}"
             rf"{{\pi \times ({d_m:.4f})^4}}"
             rf" \times \phi^2_{{2p}}"
             rf" = {dP_bar:.4f}\;\mathrm{{bar}}"),
        ]
        if gas_ctx:
            eqs_5.append(
                rf"\phi^2_{{2p}} = 1 + 12\varepsilon_g + 25\varepsilon_g^2"
                rf" = {calc.two_phase_multiplier:.2f}"
            )
        w5: list[str] = []
        if dP_bar > pump_max:
            w5.append(f"ΔP ({dP_bar:.2f} bar) EXCEEDS pump max ({pump_max:.0f} bar)")
        elif margin < 20:
            w5.append(
                f"ΔP is {100-margin:.0f}% of pump capacity — limited headroom"
            )
        s5 = ("FAIL" if dP_bar > pump_max
               else "ADJUSTED" if adj_5
               else "WARNING" if w5
               else "PASS")
        self._emit(calc, StepResult(
            step=5, name="Pressure Drop",
            status=s5,
            summary=(
                f"ΔP = {dP_bar:.4f} bar "
                f"(pump max = {pump_max:.0f} bar, margin = {margin:.0f} %)"
            ),
            values={"dP_bar": dP_bar, "dP_Pa": dP_Pa,
                    "pump_max_bar": pump_max, "margin_pct": margin},
            equations=eqs_5, adjustments=adj_5, warnings=w5,
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Step 6 — Mass transfer
    # ═══════════════════════════════════════════════════════════════════

    def _step6(self, calc: DesignCalculations):
        d = calc.tubing_ID_m
        D = D_MOLECULAR
        k_flow = calc.rate_constant

        # Mixing time by diffusion (Taylor-Aris):  t_mix ≈ d² / (D·π²)
        t_mix = d ** 2 / (D * PI ** 2)
        calc.mixing_time_s = round(t_mix, 2)

        # Damköhler (mass): Da = k·d² / D   (first-order)
        # Da << 1 → kinetically controlled (good)
        # Da >> 1 → mass-transfer limited (bad, need mixer)
        if k_flow and k_flow > 0:
            Da = k_flow * d ** 2 / D
        else:
            Da = calc.residence_time_s / t_mix if t_mix > 0 else 0
        calc.damkohler_mass = round(Da, 4)
        calc.mass_transfer_limited = Da > 1.0

        equations = [
            (rf"t_{{\mathrm{{mix}}}} \approx \frac{{d^2}}{{D \cdot \pi^2}}"
             rf" = \frac{{({d:.4f})^2}}"
             rf"{{{D:.1e} \times \pi^2}}"
             rf" = {t_mix:.1f}\;\mathrm{{s}}"),
        ]
        if k_flow and k_flow > 0:
            equations.append(
                rf"Da = \frac{{k \cdot d^2}}{{D}}"
                rf" = \frac{{{k_flow:.3e} \times ({d:.4f})^2}}"
                rf"{{{D:.1e}}}"
                rf" = {Da:.3f}"
            )

        warnings: list[str] = []
        if Da > 10:
            warnings.append(
                f"Da = {Da:.1f} ≫ 1: severely mass-transfer limited. "
                "Need micromixer, smaller d, or pre-mixed streams."
            )
        elif Da > 1:
            warnings.append(
                f"Da = {Da:.2f} > 1: mixing may limit conversion. "
                "Consider static mixer insert."
            )
        if calc.is_gas_liquid and calc.gas_oxygen_fraction > 0:
            if calc.o2_equiv_supplied < 1.0:
                warnings.append(
                    f"O2 gas feed supplies only {calc.o2_equiv_supplied:.2f} equiv "
                    "relative to substrate; increase MFC setpoint."
                )
            if calc.o2_transfer_sufficiency < 1.0:
                warnings.append(
                    f"Estimated O2 transfer sufficiency = {calc.o2_transfer_sufficiency:.2f}× "
                    "reaction demand; increase gas pressure, gas/liquid interface, or residence time."
                )

        if Da < 1:
            interp = "Kinetically controlled — mixing is not limiting"
        elif Da < 10:
            interp = "Transition zone — mixing becoming significant"
        else:
            interp = "Mass-transfer limited — mixing is the bottleneck"

        values = {"t_mix_s": t_mix, "Da_mass": Da, "D_m2_s": D,
                  "interpretation": interp}
        equations_out = list(equations)
        assumptions = [
            f"D = {D:.0e} m²/s (typical for small organic molecules in liquid)"
        ]
        if calc.is_gas_liquid:
            values.update({
                "gas_species": calc.gas_species,
                "gas_flow_sccm": calc.gas_flow_sccm,
                "gas_flow_actual_mL_min": calc.gas_flow_actual_mL_min,
                "gas_pressure_abs_bar": calc.gas_pressure_abs_bar,
                "gas_liquid_ratio": calc.gas_liquid_ratio,
                "gas_holdup": calc.gas_holdup,
                "o2_supply_mmol_min": calc.o2_supply_mmol_min,
                "o2_required_mmol_min": calc.o2_required_mmol_min,
                "o2_equiv_supplied": calc.o2_equiv_supplied,
                "dissolved_o2_mM": calc.dissolved_o2_mM,
                "kLa_s": calc.kLa_s,
                "o2_transfer_capacity_mmol_min": calc.o2_transfer_capacity_mmol_min,
                "o2_transfer_sufficiency": calc.o2_transfer_sufficiency,
            })
            equations_out.extend([
                rf"Q_{{g,\mathrm{{actual}}}} = Q_{{g,\mathrm{{STP}}}}"
                rf"\frac{{T}}{{T_{{STP}}}}\frac{{P_{{STP}}}}{{P}}"
                rf" = {calc.gas_flow_actual_mL_min:.3f}\;\mathrm{{mL/min}}",
                rf"C^*_{{O2}} \approx 1.3\;\mathrm{{mM}}\times"
                rf"\frac{{P_{{O2}}}}{{0.21\;\mathrm{{bar}}}}"
                rf" = {calc.dissolved_o2_mM:.2f}\;\mathrm{{mM}}",
                rf"\dot n_{{O2,transfer}} = k_La C^* V_L"
                rf" = {calc.o2_transfer_capacity_mmol_min:.4f}\;\mathrm{{mmol/min}}",
            ])
            assumptions.extend([
                "Gas holdup estimated from actual reactor gas/liquid volumetric ratio.",
                "O2 solubility uses a water/ethanol-scale Henry-law approximation; validate experimentally for final scale-up.",
                "kLa is a conservative capillary slug-flow estimate tied to gas holdup.",
            ])

        self._emit(calc, StepResult(
            step=6, name="Mass Transfer",
            status="WARNING" if warnings else "PASS",
            summary=f"t_mix = {t_mix:.1f} s, Da = {Da:.3f} — {interp}",
            values=values,
            equations=equations_out, warnings=warnings,
            assumptions=assumptions,
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Step 7 — Heat transfer
    # ═══════════════════════════════════════════════════════════════════

    def _step7(self, calc, br, chem_plan, solvent):
        d = calc.tubing_ID_m
        L = calc.tubing_length_m
        V_m3 = (
            calc.liquid_holdup_volume_mL * 1e-6
            if calc.is_gas_liquid and calc.liquid_holdup_volume_mL > 0
            else calc.reactor_volume_m3
        )
        C0_mol_m3 = calc.concentration_M * 1000.0  # mol/L → mol/m³
        k = calc.rate_constant or 0

        chem_type = _detect_chemistry(br, chem_plan)
        dH = DELTA_H_ESTIMATES.get(chem_type, DELTA_H_ESTIMATES["default"])

        # Reaction rate at inlet (maximum — conservative estimate)
        if k > 0:
            r = k * C0_mol_m3  # mol/(m³·s)  (first-order)
        elif calc.residence_time_s > 0:
            r = C0_mol_m3 * calc.target_conversion / calc.residence_time_s
        else:
            r = 0

        # Heat generated by reaction: Q_rxn = |ΔH_r| × r × V_R
        Q_gen = abs(dH) * r * V_m3
        calc.heat_generation_W = round(Q_gen, 6)

        # Surface-to-volume ratio: S/V = 4/d
        S_V = 4.0 / d if d > 0 else 0
        calc.surface_to_volume = round(S_V, 1)

        # Wall surface area
        A_wall = PI * d * L
        calc.heat_transfer_area_m2 = round(A_wall, 8)

        # Overall heat-transfer coefficient
        U = U_VALUES.get("coil", 300)

        # Assume 10 °C log-mean ΔT (bath set to desired T)
        dT_lm = 10.0

        # Heat removal capacity: Q_rem = U × A × ΔT_lm
        Q_rem = U * A_wall * dT_lm
        calc.heat_removal_W = round(Q_rem, 6)
        calc.UA_W_K = round(U * A_wall, 6)

        # Thermal Damköhler: Da_th = Q_gen / Q_rem
        if Q_rem > 0:
            Da_th = Q_gen / Q_rem
        else:
            Da_th = float("inf") if Q_gen > 0 else 0
        calc.thermal_damkohler = round(Da_th, 6)
        calc.thermal_safe = Da_th < 1.0
        calc.heat_transfer_score = round(max(0.0, min(1.0, Q_rem / max(2.0 * Q_gen, 1e-12))), 4)

        equations = [
            (rf"\dot{{Q}}_{{\mathrm{{rxn}}}} = |\Delta H_r| \cdot r \cdot V_R"
             rf" = {abs(dH):.0f} \times {r:.3f} \times {V_m3:.3e}"
             rf" = {Q_gen:.4f}\;\mathrm{{W}}"),
            (rf"A_{{\mathrm{{wall}}}} = \pi\,d\,L"
             rf" = \pi \times {d:.4f} \times {L:.2f}"
             rf" = {A_wall:.5f}\;\mathrm{{m^2}}"),
            (rf"\dot{{Q}}_{{\mathrm{{rem}}}} = U \cdot A \cdot \Delta T_{{\mathrm{{lm}}}}"
             rf" = {U} \times {A_wall:.5f} \times {dT_lm:.0f}"
             rf" = {Q_rem:.4f}\;\mathrm{{W}}"),
            (rf"Da_{{\mathrm{{th}}}} = \frac{{\dot{{Q}}_{{\mathrm{{rxn}}}}}}"
             rf"{{\dot{{Q}}_{{\mathrm{{rem}}}}}}"
             rf" = \frac{{{Q_gen:.4f}}}{{{Q_rem:.4f}}}"
             rf" = {Da_th:.4f}"),
            (rf"\frac{{S}}{{V}} = \frac{{4}}{{d}}"
             rf" = \frac{{4}}{{{d:.4f}}}"
             rf" = {S_V:.0f}\;\mathrm{{m^{{-1}}}}"),
        ]

        warnings: list[str] = []
        if Da_th > 1.0:
            warnings.append(
                f"Da_th = {Da_th:.2f} > 1: heat generation exceeds removal! "
                "Risk of thermal runaway. Reduce Q, use smaller d, "
                "or improve cooling."
            )
        elif Da_th > 0.5:
            warnings.append(
                f"Da_th = {Da_th:.2f}: approaching thermal limit — monitor closely"
            )

        interp = (
            "Thermally safe" if Da_th < 0.5
            else "Near thermal limit" if Da_th < 1.0
            else "THERMAL RUNAWAY RISK"
        )

        self._emit(calc, StepResult(
            step=7, name="Heat Transfer",
            status=(
                "FAIL" if Da_th > 1
                else "WARNING" if Da_th > 0.5
                else "PASS"
            ),
            summary=(
                f"Q_gen = {Q_gen:.4f} W, Q_rem = {Q_rem:.4f} W, "
                f"Da_th = {Da_th:.4f} — {interp}"
            ),
            values={
                "Q_gen_W": Q_gen, "Q_rem_W": Q_rem, "Da_th": Da_th,
                "S_V_m_inv": S_V, "U_W_m2K": U,
                "A_wall_m2": A_wall, "UA_W_K": calc.UA_W_K,
                "heat_transfer_score": calc.heat_transfer_score,
                "dH_J_mol": dH,
                "dT_lm_C": dT_lm, "r_mol_m3_s": r,
            },
            equations=equations, warnings=warnings,
            assumptions=[
                f"ΔH_r ≈ {dH / 1000:.0f} kJ/mol (estimated for {chem_type})",
                f"U ≈ {U} W/m²·K (coil in temperature-controlled bath)",
                f"ΔT_lm ≈ {dT_lm} °C (conservative estimate)",
            ],
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Step 8 — BPR sizing
    # ═══════════════════════════════════════════════════════════════════

    def _step8(self, calc, solvent, is_gas_liquid=False):
        T_C = calc.temperature_C
        dP_sys = calc.pressure_drop_bar
        safety = 0.5  # bar margin

        P_vap = _vapor_pressure_bar(solvent, T_C)
        bp_C = _lookup(solvent, SOLVENT_BOILING_POINT_C)

        equations: list[str] = []
        warnings: list[str] = []
        assumptions: list[str] = []
        gas_liquid_reason = ""

        # ── Gas-liquid systems always need BPR ──────────────────────────
        if is_gas_liquid:
            calc.bpr_required = True
            P_BPR_base = (P_vap + dP_sys + safety) if P_vap is not None else (dP_sys + safety + 1.0)
            calc.bpr_pressure_bar = round(max(P_BPR_base, GAS_LIQUID_MIN_BPR_BAR), 1)
            if P_vap is not None:
                calc.vapor_pressure_bar = round(P_vap, 4)
            gas_liquid_reason = (
                "Gas-liquid system detected — BPR mandatory to maintain "
                "gas solubility (Henry's law: C_gas = H·P) and prevent "
                "uncontrolled degassing."
            )
            equations.append(
                rf"C_{{\mathrm{{gas}}}} = H \cdot P_{{\mathrm{{partial}}}}"
                r"\quad\text{(higher BPR} \Rightarrow \text{more dissolved gas)}"
            )
            if P_vap is not None:
                equations.append(
                    rf"P_{{\mathrm{{BPR}}}} \geq \max\left("
                    rf"P_{{\mathrm{{vap}}}} + \Delta P + 0.5,\; {GAS_LIQUID_MIN_BPR_BAR:.0f}\right)"
                    rf" = \max({P_vap:.2f} + {dP_sys:.3f} + {safety},\; {GAS_LIQUID_MIN_BPR_BAR:.0f})"
                    rf" = {calc.bpr_pressure_bar:.1f}\;\mathrm{{bar}}"
                )
        elif P_vap is not None:
            calc.vapor_pressure_bar = round(P_vap, 4)
            P_BPR = P_vap + dP_sys + safety

            if bp_C is not None and T_C > bp_C - 20:
                calc.bpr_required = True
                calc.bpr_pressure_bar = round(max(P_BPR, 2.0), 1)
                equations.append(
                    rf"P_{{\mathrm{{BPR}}}} \geq P_{{\mathrm{{vap}}}}(T)"
                    rf" + \Delta P_{{\mathrm{{sys}}}} + 0.5"
                    rf" = {P_vap:.2f} + {dP_sys:.3f} + {safety}"
                    rf" = {P_BPR:.2f}\;\mathrm{{bar}}"
                )
            else:
                calc.bpr_required = False
                calc.bpr_pressure_bar = 0.0
        else:
            # Fallback: boiling-point heuristic
            if bp_C is not None and T_C > bp_C - 20:
                calc.bpr_required = True
                excess = T_C - (bp_C - 20)
                P_BPR = max(1.0 + excess * 0.05, dP_sys + safety)
                calc.bpr_pressure_bar = round(P_BPR, 1)
                assumptions.append(
                    "Antoine coefficients unavailable — BPR from boiling-point heuristic"
                )
            else:
                calc.bpr_required = False
                calc.bpr_pressure_bar = 0.0

        if calc.bpr_required:
            reason = gas_liquid_reason or f"P_vap = {P_vap or '?'} bar at {T_C} °C"
            summary = (
                f"BPR REQUIRED: {calc.bpr_pressure_bar:.1f} bar "
                f"({reason})"
            )
            status = "PASS"
            if is_gas_liquid and calc.bpr_pressure_bar > GAS_LIQUID_ROUTINE_MAX_BPR_BAR:
                warnings.append(
                    f"BPR {calc.bpr_pressure_bar:.1f} bar exceeds the "
                    f"{GAS_LIQUID_ROUTINE_MAX_BPR_BAR:.0f} bar routine gas-liquid ceiling; "
                    "redesign geometry or lower gas/liquid throughput before validation."
                )
                status = "FAIL"
        else:
            summary = (
                f"BPR not required "
                f"(T = {T_C} °C, bp = {bp_C or '?'} °C — safe margin)"
            )
            status = "PASS"
            if bp_C is not None and T_C > bp_C - 30:
                warnings.append(
                    f"Operating within 30 °C of boiling point — "
                    "consider precautionary BPR"
                )
                status = "WARNING"

        self._emit(calc, StepResult(
            step=8, name="Back-Pressure Regulator",
            status=status, summary=summary,
            values={
                "bpr_required": calc.bpr_required,
                "bpr_bar": calc.bpr_pressure_bar,
                "P_vapor_bar": P_vap or 0,
                "dP_system_bar": dP_sys,
                "solvent_bp_C": bp_C,
                "gas_liquid": is_gas_liquid,
                "gas_species": calc.gas_species,
                "gas_pressure_abs_bar": calc.gas_pressure_abs_bar,
                "gas_flow_sccm": calc.gas_flow_sccm,
                "dissolved_o2_mM": calc.dissolved_o2_mM,
            },
            equations=equations, warnings=warnings, assumptions=assumptions,
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Step 9 — Process metrics
    # ═══════════════════════════════════════════════════════════════════

    def _step9(self, calc, br):
        tau_s = calc.residence_time_s
        tau_min = calc.residence_time_min
        V_mL = calc.reactor_volume_mL
        Q = calc.flow_rate_mL_min
        C0 = calc.concentration_M
        X = calc.target_conversion
        t_batch_s = calc.batch_time_s

        equations: list[str] = []
        warnings: list[str] = []

        # Space-Time Yield:  STY = C₀ · X · 60 / τ_min  [mol/(L·h)]
        STY = None
        if tau_min > 0:
            STY = C0 * X * 60.0 / tau_min
            calc.space_time_yield_mol_L_h = round(STY, 4)
            equations.append(
                rf"STY = \frac{{C_0 \cdot X \cdot 60}}{{\tau}}"
                rf" = \frac{{{C0} \times {X:.2f} \times 60}}{{{tau_min:.1f}}}"
                rf" = {STY:.4f}\;\mathrm{{mol/(L \cdot h)}}"
            )

        # Productivity (backward):  P = Q · C₀ · X · 60  [mmol/h]
        prod = None
        if Q > 0:
            prod = Q * C0 * X * 60.0
            calc.productivity_mmol_h = round(prod, 3)
            equations.append(
                rf"P = Q \cdot C_0 \cdot X \cdot 60"
                rf" = {Q:.3f} \times {C0} \times {X:.2f} \times 60"
                rf" = {prod:.3f}\;\mathrm{{mmol/h}}"
            )

        # ── Forward productivity path (ṅ_limiting method) ───────────────
        # Step 0: Batch baseline
        n_batch_mmol = (
            getattr(br, "batch_scale_mmol", None)
            or getattr(br, "scale_mmol", None)
        )
        Y_pct = getattr(br, "yield_pct", None) or (X * 100)
        t_batch_h_val = getattr(br, "reaction_time_h", None) or (t_batch_s / 3600.0)
        Y_frac = min(max((Y_pct / 100.0) if Y_pct else X, 0.01), 0.9999)

        if n_batch_mmol and n_batch_mmol > 0 and t_batch_h_val and t_batch_h_val > 0:
            # P_batch = (n_batch × Y) / t_batch  [mmol/h]
            P_batch = (n_batch_mmol * Y_frac) / t_batch_h_val
            calc.P_batch_mmol_h = round(P_batch, 3)
            equations.append(
                rf"P_{{\mathrm{{batch}}}} = \frac{{n_{{\mathrm{{batch}}}} \cdot Y}}"
                rf"{{t_{{\mathrm{{batch}}}}}}"
                rf" = \frac{{{n_batch_mmol:.1f} \times {Y_frac:.2f}}}"
                rf"{{{t_batch_h_val:.2f}}}"
                rf" = {P_batch:.2f}\;\mathrm{{mmol/h}}"
            )

            # ṅ_limiting = P_batch / (Y × 60)  [mmol/min]
            n_lim = P_batch / (Y_frac * 60.0)
            calc.n_molar_flow_mmol_min = round(n_lim, 4)
            equations.append(
                rf"\dot{{n}}_{{\mathrm{{lim}}}} = \frac{{P_{{\mathrm{{batch}}}}}}{{Y \cdot 60}}"
                rf" = \frac{{{P_batch:.2f}}}{{{Y_frac:.2f} \times 60}}"
                rf" = {n_lim:.4f}\;\mathrm{{mmol/min}}"
            )

            # C_reactor = ṅ_limiting / Q_total  [mmol/mL = mol/L = M]
            if Q > 0:
                C_rxr = n_lim / Q
                calc.C_reactor_M = round(C_rxr, 4)
                equations.append(
                    rf"C_{{\mathrm{{reactor}}}} = \frac{{\dot{{n}}_{{\mathrm{{lim}}}}}}"
                    rf"{{Q_{{\mathrm{{total}}}}}}"
                    rf" = \frac{{{n_lim:.4f}}}{{{Q:.4f}}}"
                    rf" = {C_rxr:.4f}\;\mathrm{{M}}"
                )

            # P_flow = ṅ_limiting × Y × 60  [mmol/h]
            P_flow = n_lim * Y_frac * 60.0
            calc.P_flow_mmol_h = round(P_flow, 3)
            equations.append(
                rf"P_{{\mathrm{{flow}}}} = \dot{{n}}_{{\mathrm{{lim}}}} \cdot Y \cdot 60"
                rf" = {n_lim:.4f} \times {Y_frac:.2f} \times 60"
                rf" = {P_flow:.2f}\;\mathrm{{mmol/h}}"
            )

            # Productivity closure check
            calc.productivity_closure_ok = P_flow >= P_batch * 0.95
            if not calc.productivity_closure_ok:
                warnings.append(
                    f"Productivity closure FAIL: P_flow = {P_flow:.2f} mmol/h < "
                    f"P_batch = {P_batch:.2f} mmol/h. "
                    "Increase C_feed or check yield assumptions."
                )

        # Startup waste:  V_waste = 3·τ·Q  [mL]
        if tau_min > 0 and Q > 0:
            startup = round(3.0 * tau_min * Q, 1)
            calc.startup_waste_mL = startup
            equations.append(
                rf"V_{{\mathrm{{startup}}}} = 3 \cdot \tau \cdot Q"
                rf" = 3 \times {tau_min:.1f} \times {Q:.3f}"
                rf" = {startup:.1f}\;\mathrm{{mL}}"
            )

        # Intensification factor:  IF = t_batch / τ_flow
        IF = t_batch_s / tau_s if (t_batch_s > 0 and tau_s > 0) else calc.intensification_factor
        equations.append(
            rf"IF = \frac{{t_{{\mathrm{{batch}}}}}}{{\tau_{{\mathrm{{flow}}}}}}"
            rf" = \frac{{{t_batch_s:.0f}}}{{{tau_s:.1f}}}"
            rf" = {IF:.1f}\times"
        )

        summary_parts = []
        if STY is not None:
            summary_parts.append(f"STY = {STY:.4f} mol/(L·h)")
        if prod is not None:
            summary_parts.append(f"P = {prod:.3f} mmol/h")
        if calc.n_molar_flow_mmol_min is not None:
            summary_parts.append(f"ṅ_lim = {calc.n_molar_flow_mmol_min:.4f} mmol/min")
        if calc.startup_waste_mL is not None:
            summary_parts.append(f"startup waste = {calc.startup_waste_mL} mL")
        summary_parts.append(f"IF = {IF:.0f}×")

        self._emit(calc, StepResult(
            step=9, name="Process Metrics",
            status="WARNING" if warnings else "PASS",
            summary=", ".join(summary_parts),
            values={
                "STY_mol_L_h": STY, "productivity_mmol_h": prod, "IF": IF,
                "n_molar_flow_mmol_min": calc.n_molar_flow_mmol_min,
                "P_batch_mmol_h": calc.P_batch_mmol_h,
                "P_flow_mmol_h": calc.P_flow_mmol_h,
                "C_reactor_M": calc.C_reactor_M,
                "startup_waste_mL": calc.startup_waste_mL,
                "productivity_closure_ok": calc.productivity_closure_ok,
                "is_gas_liquid": calc.is_gas_liquid,
                "liquid_flow_rate_mL_min": calc.liquid_flow_rate_mL_min,
                "gas_flow_sccm": calc.gas_flow_sccm,
                "gas_holdup": calc.gas_holdup,
                "liquid_holdup_volume_mL": calc.liquid_holdup_volume_mL,
                "total_reactor_volume_mL": calc.reactor_volume_mL,
            },
            equations=equations,
            warnings=warnings,
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Consistency verification
    # ═══════════════════════════════════════════════════════════════════

    def _verify(self, calc):
        """Cross-check that all derived quantities are internally consistent."""
        notes: list[str] = []

        # τ = V_R / Q
        if calc.flow_rate_m3_s > 0 and calc.residence_time_s > 0:
            if calc.is_gas_liquid and calc.gas_holdup > 0:
                liquid_volume_m3 = calc.reactor_volume_m3 * (1.0 - calc.gas_holdup)
                tau_check = liquid_volume_m3 / calc.flow_rate_m3_s
            else:
                tau_check = calc.reactor_volume_m3 / calc.flow_rate_m3_s
            err = abs(tau_check - calc.residence_time_s) / calc.residence_time_s
            if err > 0.01:
                notes.append(
                    f"τ: V_R/Q = {tau_check:.1f} s vs τ = {calc.residence_time_s:.1f} s "
                    f"(Δ = {err*100:.1f}%)"
                )

        # L = 4V/(πd²)
        d = calc.tubing_ID_m
        if d > 0 and calc.reactor_volume_m3 > 0:
            L_check = 4.0 * calc.reactor_volume_m3 / (PI * d ** 2)
            err = abs(L_check - calc.tubing_length_m) / max(calc.tubing_length_m, 1e-10)
            if err > 0.01:
                notes.append(
                    f"L: 4V/(πd²) = {L_check:.2f} m vs L = {calc.tubing_length_m:.2f} m "
                    f"(Δ = {err*100:.1f}%)"
                )

        # Re = ρvd/μ
        if calc.viscosity_Pa_s > 0 and calc.velocity_m_s > 0:
            if calc.is_gas_liquid:
                area = PI * (calc.tubing_ID_m / 2) ** 2
                v_for_re = calc.flow_rate_m3_s / area if area > 0 else calc.velocity_m_s
            else:
                v_for_re = calc.velocity_m_s
            Re_check = calc.density_kg_m3 * v_for_re * calc.tubing_ID_m / calc.viscosity_Pa_s
            if calc.reynolds_number > 0:
                err = abs(Re_check - calc.reynolds_number) / calc.reynolds_number
                if err > 0.01:
                    notes.append(
                        f"Re: ρvd/μ = {Re_check:.1f} vs Re = {calc.reynolds_number:.1f}"
                    )

        calc.consistency_notes = notes
        calc.consistent = len(notes) == 0

        if notes:
            logger.warning("Consistency issues: %s", notes)
        else:
            logger.info("All consistency checks passed ✓")
