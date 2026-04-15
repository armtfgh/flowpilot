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
from dataclasses import dataclass, field

from flora_translate.config import (
    SOLVENT_VISCOSITY_cP,
    SOLVENT_DENSITY_g_mL,
    SOLVENT_BOILING_POINT_C,
    INCOMPATIBLE_COMBOS,
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
INTENSIFICATION = {
    "photoredox": 48.0, "photocatalysis": 48.0,
    "photochem": 30.0, "thermal": 10.0,
    "cross-coupling": 15.0, "hydrogenation": 50.0,
    "electrochemistry": 20.0, "biocatalysis": 8.0,
    "radical": 30.0, "default": 20.0,
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

    # Step 8 — BPR
    bpr_required: bool = False
    bpr_pressure_bar: float = 0.0
    vapor_pressure_bar: float = 0.0

    # Step 9 — metrics
    space_time_yield_mol_L_h: float | None = None
    productivity_mmol_h: float | None = None

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
            f"Q = {self.flow_rate_mL_min:.3f} mL/min, "
            f"V_R = {self.reactor_volume_mL:.2f} mL, "
            f"d = {self.tubing_ID_mm:.2f} mm, L = {self.tubing_length_m:.2f} m, "
            f"Re = {self.reynolds_number:.0f}, ΔP = {self.pressure_drop_bar:.3f} bar"
        )
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
        self._steps345(calc, Q_init, d_init, solvent, inventory, is_photochem)
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
                        "carbon monoxide", "carbonylation"}
        # Phase keywords
        _PHASE_KW = {"gas-liquid", "gas_liquid", "segmented", "slug flow",
                     "mfc", "bubbl", "sparging gas", "gas reagent"}
        # Inert gases — do NOT trigger gas-liquid when used as atmosphere
        _INERT = {"n2", "n₂", "nitrogen", "ar", "argon", "helium"}

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
        if any(kw in combined for kw in _REAGENT_GAS):
            return True
        if any(kw in combined for kw in _PHASE_KW):
            return True
        return False

    # ═══════════════════════════════════════════════════════════════════
    #  Step 1 — Parse batch conditions
    # ═══════════════════════════════════════════════════════════════════

    def _step1(self, calc: DesignCalculations, br):
        T_C = getattr(br, "temperature_C", None) or 25.0
        T_K = T_C + 273.15
        C_M = getattr(br, "concentration_M", None) or 0.1
        t_h = getattr(br, "reaction_time_h", None) or 0
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
        if not getattr(br, "concentration_M", None):
            assumptions.append("Concentration not specified → assumed 0.1 M")
        if not y:
            assumptions.append("Yield not specified → targeting 95 % conversion")

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
    def _extract_analogy_IFs(analogies: list[dict] | None) -> list[float]:
        """Extract batch-to-flow intensification factors from analogy records."""
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
        analogy_factors = self._extract_analogy_IFs(analogies)
        n_analogy = len(analogy_factors)
        calc.n_analogy_datapoints = n_analogy

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
            },
            assumptions=["τ set by ENGINE council — kinetics estimation bypassed"],
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Steps 3-4-5 — Reactor sizing + Fluid dynamics + Pressure drop
    #  (coupled: changing d affects L, Re, and ΔP — iterate until OK)
    # ═══════════════════════════════════════════════════════════════════

    def _steps345(self, calc, Q_mL_min, d_mm, solvent, inventory, is_photochem):
        tau_s = calc.residence_time_s or 600.0

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

        for _iter in range(5):
            Q_m3s = Q_mL_min * 1e-6 / 60.0
            d_m = d_mm * 1e-3
            A = PI * (d_m / 2) ** 2
            V_m3 = tau_s * Q_m3s
            V_mL = V_m3 * 1e6
            L = (4.0 * V_m3 / (PI * d_m ** 2)) if d_m > 0 else 0.0
            v = Q_m3s / A if A > 0 else 0.0
            Re = (rho * v * d_m / mu) if mu > 0 else 0.0
            dP_Pa = (128.0 * mu * L * Q_m3s / (PI * d_m ** 4)) if d_m > 0 else float("inf")
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
        calc.pump_adequate = dP_bar <= pump_max

        # ── Emit Step 3: Reactor Sizing ─────────────────────────────────
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
        self._emit(calc, StepResult(
            step=3, name="Reactor Sizing",
            status="PASS",
            summary=(
                f"V_R = {V_mL:.2f} mL, d = {d_mm:.2f} mm, "
                f"L = {L:.2f} m (Q = {Q_mL_min:.3f} mL/min)"
            ),
            values={"V_R_mL": V_mL, "d_mm": d_mm, "L_m": L,
                    "Q_mL_min": Q_mL_min, "v_m_s": v},
            equations=eqs_3,
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
             rf" = {dP_bar:.4f}\;\mathrm{{bar}}"),
        ]
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

        if Da < 1:
            interp = "Kinetically controlled — mixing is not limiting"
        elif Da < 10:
            interp = "Transition zone — mixing becoming significant"
        else:
            interp = "Mass-transfer limited — mixing is the bottleneck"

        self._emit(calc, StepResult(
            step=6, name="Mass Transfer",
            status="WARNING" if warnings else "PASS",
            summary=f"t_mix = {t_mix:.1f} s, Da = {Da:.3f} — {interp}",
            values={"t_mix_s": t_mix, "Da_mass": Da, "D_m2_s": D,
                    "interpretation": interp},
            equations=equations, warnings=warnings,
            assumptions=[
                f"D = {D:.0e} m²/s (typical for small organic molecules in liquid)"
            ],
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Step 7 — Heat transfer
    # ═══════════════════════════════════════════════════════════════════

    def _step7(self, calc, br, chem_plan, solvent):
        d = calc.tubing_ID_m
        L = calc.tubing_length_m
        V_m3 = calc.reactor_volume_m3
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

        # Overall heat-transfer coefficient
        U = U_VALUES.get("coil", 300)

        # Assume 10 °C log-mean ΔT (bath set to desired T)
        dT_lm = 10.0

        # Heat removal capacity: Q_rem = U × A × ΔT_lm
        Q_rem = U * A_wall * dT_lm
        calc.heat_removal_W = round(Q_rem, 6)

        # Thermal Damköhler: Da_th = Q_gen / Q_rem
        if Q_rem > 0:
            Da_th = Q_gen / Q_rem
        else:
            Da_th = float("inf") if Q_gen > 0 else 0
        calc.thermal_damkohler = round(Da_th, 6)
        calc.thermal_safe = Da_th < 1.0

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
                "A_wall_m2": A_wall, "dH_J_mol": dH,
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
            calc.bpr_pressure_bar = round(max(P_BPR_base, 5.0), 1)  # min 5 bar for gas solubility
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
                    rf"P_{{\mathrm{{vap}}}} + \Delta P + 0.5,\; 5\right)"
                    rf" = \max({P_vap:.2f} + {dP_sys:.3f} + {safety},\; 5)"
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

        # Productivity:  P = Q · C₀ · X · 60  [mmol/h]
        prod = None
        if Q > 0:
            prod = Q * C0 * X * 60.0
            calc.productivity_mmol_h = round(prod, 3)
            equations.append(
                rf"P = Q \cdot C_0 \cdot X \cdot 60"
                rf" = {Q:.3f} \times {C0} \times {X:.2f} \times 60"
                rf" = {prod:.3f}\;\mathrm{{mmol/h}}"
            )

        # Intensification factor:  IF = t_batch / τ_flow
        IF = t_batch_s / tau_s if (t_batch_s > 0 and tau_s > 0) else calc.intensification_factor
        equations.append(
            rf"IF = \frac{{t_{{\mathrm{{batch}}}}}}{{\tau_{{\mathrm{{flow}}}}}}"
            rf" = \frac{{{t_batch_s:.0f}}}{{{tau_s:.1f}}}"
            rf" = {IF:.1f}\times"
        )

        self._emit(calc, StepResult(
            step=9, name="Process Metrics",
            status="PASS",
            summary=(
                f"STY = {STY:.4f} mol/(L·h), "
                f"Productivity = {prod:.3f} mmol/h, "
                f"Intensification = {IF:.0f}×"
            ),
            values={"STY_mol_L_h": STY, "productivity_mmol_h": prod, "IF": IF},
            equations=equations,
        ))

    # ═══════════════════════════════════════════════════════════════════
    #  Consistency verification
    # ═══════════════════════════════════════════════════════════════════

    def _verify(self, calc):
        """Cross-check that all derived quantities are internally consistent."""
        notes: list[str] = []

        # τ = V_R / Q
        if calc.flow_rate_m3_s > 0 and calc.residence_time_s > 0:
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
            Re_check = (
                calc.density_kg_m3 * calc.velocity_m_s * calc.tubing_ID_m
                / calc.viscosity_Pa_s
            )
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
