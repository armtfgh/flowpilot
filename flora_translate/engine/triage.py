"""
FLORA ENGINE — Triage Report Generator.

Runs domain tools against calculator results BEFORE any LLM call.
Produces a structured TriageReport that tells each agent:
  - Whether their domain needs attention (GREEN / YELLOW / RED)
  - Exactly what question to answer (not a blank slate)
  - Pre-computed tool values they can cite as authoritative

GREEN domains: agent files a one-line confirmation and exits.
YELLOW / RED domains: agent enters deliberation with a directed question.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

from flora_translate.schemas import FlowProposal, ChemistryPlan
from flora_translate.design_calculator import DesignCalculations
from flora_translate.engine.tools import (
    calculate_mixing_ratio,
    calculate_bpr_required,
    check_material_compatibility,
    calculate_reynolds,
    calculate_pressure_drop,
)

# ── Physical constants ────────────────────────────────────────────────────────
R_COIL_M = 0.030          # 30 mm typical coil radius (m)
D_MOLECULAR = 1.0e-9      # molecular diffusivity (m²/s)
PI = math.pi

# ── Commercial FEP/PFA tubing inner diameters (mm, 1/16" OD standard) ────────
# d_fix should always round UP to the nearest size so mixing stays adequate.
STANDARD_TUBING_IDS_MM = [0.25, 0.50, 0.75, 1.00, 1.60, 2.00]

# ── Bench-scale reactor geometry limits ──────────────────────────────────────
# Above L_MAX_BENCH_M the assembly is physically impractical on a lab bench.
# A single FEP coil >20 m is unwieldy; two 10 m coils in series is the
# normal workaround, but the council should flag this and ask the chemist.
L_MAX_BENCH_M = 20.0           # hard practical limit (m)
V_MAX_SINGLE_REACTOR_ML = 25.0 # soft limit — >25 mL reactor is unusual
Q_MIN_PRACTICAL_ML_MIN = 0.01  # syringe pump floor (mL/min)


def _round_up_to_standard_id(d_mm: float) -> float:
    """Round d UP to the nearest commercially available FEP/PFA tubing ID.

    Rounding UP (not down) is mandatory: rounding down would make the tube
    narrower than d_fix, increasing t_mix above the accepted threshold.
    """
    for std in STANDARD_TUBING_IDS_MM:
        if std >= d_mm - 1e-6:   # allow tiny floating-point tolerance
            return std
    return d_mm  # larger than any standard — keep exact value


def _compute_geometry_sweet_spots(
    tau_min: float,
    solvent: str,
    T_C: float,
    pump_max_bar: float,
    is_photochem: bool,
    C_M: float = 0.1,
) -> list[dict]:
    """Return feasible (d, Q) operating points that keep L ≤ L_MAX_BENCH_M.

    For each standard d, computes the maximum Q such that:
      L = 4·τ·Q / (π·d²)  ≤  L_MAX_BENCH_M
    and then verifies Re < 2300 and ΔP < 0.8·pump_max.

    For photochemical reactions, excludes d > 1.0 mm if Beer-Lambert
    risk would be HIGH (qualitative: C > 0.1 M at d > 1.0 mm).

    Returns a list of dicts, one per feasible (d, Q) point, sorted by
    Q descending (highest throughput first).
    """
    from flora_translate.engine.tools import calculate_reynolds, calculate_pressure_drop

    spots = []
    for d_mm in STANDARD_TUBING_IDS_MM:
        d_m = d_mm * 1e-3
        V_max_at_Lmax_mL = L_MAX_BENCH_M * PI * d_m ** 2 / 4.0 * 1e6  # mL

        # Skip if even at L_max the single-reactor volume is too large
        if V_max_at_Lmax_mL > V_MAX_SINGLE_REACTOR_ML:
            continue

        # Beer-Lambert pre-filter for photochem:
        # d >= 1.0 mm AND C >= 0.1 M risks inner filter with typical Ir catalysts
        # (ε ~ 100–500 M⁻¹cm⁻¹; A = ε × C × d×0.1 can reach 0.5–5 at d=1mm).
        if is_photochem and d_mm >= 1.0 and C_M >= 0.1:
            # Still include but mark as photochem_marginal — let Dr. Photonics decide
            photochem_marginal = True
        else:
            photochem_marginal = False

        Q_max_geo = V_max_at_Lmax_mL / tau_min if tau_min > 0 else 0.0  # mL/min

        if Q_max_geo < Q_MIN_PRACTICAL_ML_MIN:
            continue  # below syringe pump floor at this d

        # Use 90% of geometric limit for comfortable headroom
        Q_target = round(Q_max_geo * 0.90, 3)

        re = calculate_reynolds(Q_target, d_mm, solvent, T_C)
        dp = calculate_pressure_drop(Q_target, d_mm, L_MAX_BENCH_M, solvent)

        if re["Re"] > 2300:
            continue  # turbulent
        if dp["delta_P_bar"] > 0.8 * pump_max_bar:
            continue  # pressure headroom violated

        V_R_mL = round(tau_min * Q_target, 2)
        L_m = round(4 * V_R_mL * 1e-6 / (PI * d_m ** 2), 1) if d_m > 0 else 0

        spots.append({
            "d_mm": d_mm,
            "Q_mL_min": Q_target,
            "Q_max_geo_mL_min": round(Q_max_geo, 3),
            "V_R_mL": V_R_mL,
            "L_m": L_m,
            "Re": round(re["Re"], 1),
            "delta_P_bar": round(dp["delta_P_bar"], 4),
            "photochem_check_needed": photochem_marginal,
            "note": ("Beer-Lambert check needed at this d" if photochem_marginal else ""),
        })

    # Sort by Q descending — highest throughput first
    spots.sort(key=lambda s: s["Q_mL_min"], reverse=True)
    return spots


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TriageEntry:
    domain: str             # "KINETICS" | "FLUIDICS" | "SAFETY" | "CHEMISTRY" | "PHOTONICS"
    status: str             # "GREEN" | "YELLOW" | "RED"
    message: str            # one-line summary for display
    directed_question: str  # specific question the agent must answer
    tool_results: dict = field(default_factory=dict)  # pre-computed values


@dataclass
class TriageReport:
    """Structured pre-seeding for the agent council."""
    design_center: str = ""
    entries: list[TriageEntry] = field(default_factory=list)
    green_domains: list[str] = field(default_factory=list)
    flagged_domains: list[str] = field(default_factory=list)

    # Context flags
    is_gas_liquid: bool = False
    is_multistage: bool = False
    light_required: bool = False
    n_stages: int = 1

    def agent_block(self, domain: str) -> str:
        """Return the triage section relevant to a specific agent domain."""
        entry = next((e for e in self.entries if e.domain == domain), None)
        if not entry:
            return ""
        lines = [
            f"## TRIAGE — YOUR DOMAIN: {domain}",
            f"Status: {entry.status}",
            f"Summary: {entry.message}",
            f"Your question: {entry.directed_question}",
        ]
        if entry.tool_results:
            import json
            lines.append(
                "Pre-computed tool results (AUTHORITATIVE — do not re-derive):\n"
                + json.dumps(entry.tool_results, indent=2, default=str)
            )
        return "\n".join(lines)

    def full_block(self) -> str:
        """Complete triage block injected into every agent prompt."""
        lines = [
            "## CALCULATOR TRIAGE REPORT",
            f"Design centre: {self.design_center}",
        ]
        if self.is_multistage:
            lines.append(f"⚠️  MULTI-STAGE DESIGN ({self.n_stages} stages) — each stage has its own atmosphere, feeds, and conditions.")
        if self.is_gas_liquid:
            lines.append("⚠️  GAS-LIQUID PHASE — BPR mandatory (≥5 bar), O₂ mass transfer is a design constraint.")

        flagged = [e for e in self.entries if e.status in ("RED", "YELLOW")]
        green = [e for e in self.entries if e.status == "GREEN"]

        if flagged:
            lines.append("\n### FLAGGED (need agent attention):")
            for e in flagged:
                icon = "🔴" if e.status == "RED" else "⚠️ "
                lines.append(f"  {icon} [{e.domain}] {e.message}")
                lines.append(f"       → {e.directed_question}")

        if green:
            lines.append("\n### CONFIRMED GREEN (no debate needed):")
            for e in green:
                lines.append(f"  ✓  [{e.domain}] {e.message}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Triage generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_triage(
    proposal: FlowProposal,
    chemistry_plan: ChemistryPlan | None,
    calc: DesignCalculations,
) -> TriageReport:
    """Run domain tools against the proposal/calculator and classify each domain."""

    entries: list[TriageEntry] = []

    # ── Detect context flags ──────────────────────────────────────────────────
    stages = chemistry_plan.stages if chemistry_plan else []
    is_multistage = len(stages) > 1
    n_stages = len(stages) if stages else 1

    is_gas_liquid = any(
        s.atmosphere.lower() in ("air", "o2", "o₂", "h2", "co2") for s in stages
    ) if stages else False

    light_required = (
        any(s.requires_light for s in stages) if stages else bool(proposal.wavelength_nm)
    )

    # Solvent for property lookups (prefer stage 1 or first stream)
    solvent = "EtOH"
    if stages and stages[0].solvent:
        solvent = stages[0].solvent
    elif proposal.streams:
        solvent = proposal.streams[0].solvent or solvent

    T = proposal.temperature_C
    tau = proposal.residence_time_min
    Q = proposal.flow_rate_mL_min
    ID = proposal.tubing_ID_mm
    L = calc.tubing_length_m

    # ── KINETICS ──────────────────────────────────────────────────────────────
    IF = calc.intensification_factor
    IF_ok = 1.5 <= IF <= 300.0
    method = calc.kinetics_method
    kinetics_status = "GREEN" if IF_ok else "YELLOW"

    # τ_lit anchor: literature τ from analogy-derived IF (preferred),
    # falling back to tau_class_min when no analogies were retrieved.
    # tau_class_min is already conservative (e.g. 100 min for a 10h photoredox
    # batch at class IF = 6×), so τ << tau_class/2 is a genuine warning.
    tau_analogy = getattr(calc, "tau_analogy_min", None)
    tau_class = getattr(calc, "tau_class_min", None)
    tau_lit = tau_analogy if (tau_analogy and tau_analogy > 0) else tau_class
    if_analogy = getattr(calc, "if_analogy", None)
    if_class = getattr(calc, "if_class", None)
    res_time_range = getattr(calc, "residence_time_range_min", None)

    tau_lit_anchor_check = bool(tau_lit and tau_lit > 0 and tau < tau_lit / 2.0)

    # Override to YELLOW even if IF looks fine when τ < τ_lit / 2
    if tau_lit_anchor_check and kinetics_status == "GREEN":
        kinetics_status = "YELLOW"

    rc = chemistry_plan.reaction_class if chemistry_plan else "unknown"

    if not IF_ok:
        kin_question = (
            f"Is IF = {IF:.1f}× reasonable for '{rc}'? "
            f"Is τ = {tau:.1f} min sufficient for target conversion?"
        )
    elif tau_lit_anchor_check:
        kin_question = (
            f"τ_lit anchor FLAGGED: τ_current = {tau:.1f} min < τ_lit/2 = {tau_lit/2.0:.1f} min "
            f"(τ_lit = {tau_lit:.1f} min from literature analogy). "
            f"You MUST provide a strong physical justification for why this design can run 2× faster "
            f"than the literature value, OR propose residence_time_min = {tau_lit/2.0:.1f} min."
        )
    else:
        kin_question = (
            f"Confirm τ = {tau:.1f} min and IF = {IF:.1f}× are within acceptable range for '{rc}'."
        )

    if res_time_range is not None:
        tau_range = list(res_time_range)
    else:
        tau_range = [round(tau * 0.67, 2), round(tau * 1.5, 2)]

    entries.append(TriageEntry(
        domain="KINETICS",
        status=kinetics_status,
        message=f"τ = {tau:.1f} min, IF = {IF:.1f}× ({method})",
        directed_question=kin_question,
        tool_results={
            "residence_time_min": tau,
            "intensification_factor": IF,
            "kinetics_method": method,
            "Da_mass": round(calc.damkohler_mass, 3),
            "IF_in_range": IF_ok,
            "tau_analogy_min": tau_analogy,     # Method A (None if no analogies)
            "tau_class_min": tau_class,          # Method B (always available)
            "tau_lit": tau_lit,                  # anchor used: analogy if available, else class
            "IF_analogy": if_analogy,
            "IF_class": if_class,
            "tau_range_min": tau_range,
            "tau_lit_anchor_check": tau_lit_anchor_check,
        },
    ))

    # ── FLUIDICS ──────────────────────────────────────────────────────────────
    mix = calculate_mixing_ratio(ID, tau)
    re_res = calculate_reynolds(Q, ID, solvent, T)
    dP_res = calculate_pressure_drop(Q, ID, max(L, 0.01), solvent)
    pump_max = calc.pump_max_bar or 20.0

    da_mass = calc.damkohler_mass
    mixing_concern = da_mass > 1.0 and mix["actionable"]
    pressure_concern = dP_res["delta_P_bar"] > 0.80 * pump_max

    # ── Geometry check: reactor length and volume ─────────────────────────────
    # V_R = τ × Q (the design identity). L = 4·V_R / (π·d²).
    # A 63 m FEP coil is not physically assembleable on a bench — flag RED
    # and compute the Q reduction needed to bring L within L_MAX_BENCH_M.
    V_R_mL = round(tau * Q, 2)          # mL
    L_actual_m = calc.tubing_length_m   # from calculator (already consistent with τ,Q,d)
    geometry_ok = (L_actual_m <= L_MAX_BENCH_M and V_R_mL <= V_MAX_SINGLE_REACTOR_ML)

    # Q_sweet: flow rate that gives L = L_MAX_BENCH_M at current τ and d
    V_max_at_Lmax_mL = L_MAX_BENCH_M * PI * (ID * 1e-3) ** 2 / 4.0 * 1e6
    Q_sweet_ml_min = round(V_max_at_Lmax_mL / tau, 4) if tau > 0 else Q

    # Pre-compute sweet spots table (all feasible d/Q combos at τ = current)
    pump_max = calc.pump_max_bar or 20.0
    is_photochem_geom = light_required
    C_M_val = proposal.concentration_M or 0.1
    sweet_spots = _compute_geometry_sweet_spots(
        tau_min=tau,
        solvent=solvent,
        T_C=T,
        pump_max_bar=pump_max,
        is_photochem=is_photochem_geom,
        C_M=C_M_val,
    )

    if mixing_concern or pressure_concern or not geometry_ok:
        fluid_status = "RED"
    elif da_mass > 1.0:
        fluid_status = "YELLOW"
    else:
        fluid_status = "GREEN"

    # ── Dean number ────────────────────────────────────────────────────────
    ID_m = ID * 1e-3  # mm → m
    Re_val = re_res["Re"]
    De = round(Re_val * math.sqrt(ID_m / (2.0 * R_COIL_M)), 2)

    # ── d_fix and τ_mixing_required (only when BOTH thresholds fail) ──────
    # d_fix is rounded UP to the nearest standard commercial tubing size.
    # Rounding UP keeps r_mix ≤ 0.15 (tighter than the threshold), so the
    # fix stays safe even at a slightly larger diameter than the math demands.
    d_fix_mm = None
    d_fix_mm_exact = None
    tau_mixing_required_min = None
    if mixing_concern and da_mass > 1.0:
        try:
            d_fix_m_exact = ID_m * math.sqrt(0.15 / mix["mixing_ratio"])
            d_fix_mm_exact = round(d_fix_m_exact * 1e3, 3)         # exact math result
            d_fix_mm = _round_up_to_standard_id(d_fix_mm_exact)    # → commercial stock
            d_fix_m = d_fix_mm * 1e-3
            t_mix_new_s = (d_fix_m ** 2) / (4.0 * D_MOLECULAR)
            tau_mixing_required_min = round(t_mix_new_s / 0.15 / 60.0, 2)
            # Verify the rounded size still satisfies the mixing criterion
            r_mix_at_std = t_mix_new_s / (tau * 60.0) if tau > 0 else float("inf")
        except (ZeroDivisionError, ValueError):
            d_fix_mm = None
            tau_mixing_required_min = None
            r_mix_at_std = None

    # Build directed question — geometry concern takes priority in the message
    if not geometry_ok:
        spots_summary = ""
        if sweet_spots:
            top = sweet_spots[:3]
            spots_summary = (
                " Pre-computed feasible operating points at L ≤ 20 m:\n"
                + "\n".join(
                    f"  Option {i+1}: d={s['d_mm']}mm, Q={s['Q_mL_min']}mL/min, "
                    f"L={s['L_m']}m, V_R={s['V_R_mL']}mL, Re={s['Re']}"
                    for i, s in enumerate(top)
                )
            )
        else:
            spots_summary = " No standard-size option satisfies all constraints — flag to user."

        if mixing_concern:
            fq = (
                f"DUAL PROBLEM: (1) GEOMETRY — L = {L_actual_m:.1f} m > {L_MAX_BENCH_M:.0f} m limit. "
                f"V_R = {V_R_mL:.1f} mL. Reduce Q to Q_sweet = {Q_sweet_ml_min} mL/min "
                f"(keeps L = {L_MAX_BENCH_M:.0f} m at d = {ID:.2f} mm). "
                f"(2) MIXING — Da_mass = {da_mass:.2f} > 1 AND r_mix = {mix['mixing_ratio']:.3f} > 0.20. "
                f"d_fix = {d_fix_mm} mm, τ_mixing = {tau_mixing_required_min} min."
                + spots_summary
            )
        else:
            fq = (
                f"GEOMETRY PROBLEM: L = {L_actual_m:.1f} m > {L_MAX_BENCH_M:.0f} m limit "
                f"(V_R = {V_R_mL:.1f} mL — impractical for a bench coil reactor). "
                f"Root cause: τ = {tau:.1f} min × Q = {Q:.4f} mL/min = {V_R_mL:.1f} mL. "
                f"Fix: reduce Q to Q_sweet = {Q_sweet_ml_min} mL/min to keep L ≤ {L_MAX_BENCH_M:.0f} m. "
                f"Propose flow_rate_mL_min = {Q_sweet_ml_min}."
                + spots_summary
            )
    elif mixing_concern:
        if d_fix_mm is not None and tau_mixing_required_min is not None:
            fq = (
                f"Da_mass = {da_mass:.2f} > 1 AND mixing_ratio = {mix['mixing_ratio']:.3f} > 0.20 "
                f"— both thresholds exceeded. "
                f"Pre-computed fix: d_fix = {d_fix_mm_exact:.3f} mm (math) → "
                f"rounded UP to nearest standard stock: {d_fix_mm:.2f} mm. "
                f"τ_mixing_required = {tau_mixing_required_min:.1f} min. "
                f"To improve mixing, DECREASE d (t_mix ∝ d²). "
                f"Propose tubing_ID_mm={d_fix_mm} AND residence_time_min={tau_mixing_required_min}. "
                f"Standard tubing IDs available: {STANDARD_TUBING_IDS_MM}."
            )
        else:
            fq = (
                f"Da_mass = {da_mass:.2f} > 1 AND mixing_ratio = {mix['mixing_ratio']:.3f} > 0.20 "
                f"— both thresholds exceeded. Propose ID change or active mixer."
            )
    elif pressure_concern:
        fq = (f"ΔP = {dP_res['delta_P_bar']:.4f} bar > 80% pump max ({0.8 * pump_max:.1f} bar). "
              f"Propose larger ID or reduced flow rate.")
    elif da_mass > 1.0:
        fq = (f"Da_mass = {da_mass:.2f} > 1 but mixing_ratio = {mix['mixing_ratio']:.3f} < 0.20 "
              f"— is this acceptable? (Diffusion completes within τ despite Da > 1.)")
    else:
        fq = (f"Confirm Re = {re_res['Re']:.1f} ({re_res['flow_regime']}), "
              f"ΔP = {dP_res['delta_P_bar']:.4f} bar, L = {L_actual_m:.1f} m, "
              f"V_R = {V_R_mL:.1f} mL are all within safe limits.")

    fluid_message = (
        f"Re = {re_res['Re']:.1f} ({re_res['flow_regime']}), "
        f"ΔP = {dP_res['delta_P_bar']:.4f} bar, "
        f"L = {L_actual_m:.1f} m, V_R = {V_R_mL:.1f} mL, "
        f"mixing_ratio = {mix['mixing_ratio']:.3f}"
    )
    if not geometry_ok:
        fluid_message = f"⚠ GEOMETRY: L = {L_actual_m:.1f} m > {L_MAX_BENCH_M:.0f} m | " + fluid_message

    entries.append(TriageEntry(
        domain="FLUIDICS",
        status=fluid_status,
        message=fluid_message,
        directed_question=fq,
        tool_results={
            "Re": re_res["Re"],
            "flow_regime": re_res["flow_regime"],
            "delta_P_bar": dP_res["delta_P_bar"],
            "pump_max_bar": pump_max,
            "mixing_time_s": mix["mixing_time_s"],
            "mixing_ratio": mix["mixing_ratio"],
            "mixing_actionable": mix["actionable"],
            "Da_mass": round(da_mass, 3),
            "De": De,
            # Geometry
            "L_m": round(L_actual_m, 2),
            "V_R_mL": V_R_mL,
            "L_max_bench_m": L_MAX_BENCH_M,
            "V_max_single_reactor_mL": V_MAX_SINGLE_REACTOR_ML,
            "geometry_ok": geometry_ok,
            "Q_sweet_mL_min": Q_sweet_ml_min,
            "sweet_spots": sweet_spots[:4],   # top-4 feasible points
            # Mixing fix
            "d_fix_mm_exact": d_fix_mm_exact,
            "standard_tubing_IDs_mm": STANDARD_TUBING_IDS_MM,
            "d_fix_mm": d_fix_mm,
            "tau_mixing_required_min": tau_mixing_required_min,
        },
    ))

    # ── SAFETY ────────────────────────────────────────────────────────────────
    bpr = calculate_bpr_required(T, solvent, dP_res["delta_P_bar"], is_gas_liquid)
    mat = check_material_compatibility(proposal.tubing_material, solvent, T)
    thermal_ok = calc.thermal_safe

    safety_issues = []
    if bpr["required"] and proposal.BPR_bar < bpr["P_min_bar"]:
        safety_issues.append(
            f"BPR = {proposal.BPR_bar:.1f} bar < required minimum {bpr['P_min_bar']:.2f} bar"
        )
    if not mat["compatible"]:
        safety_issues.append(f"Material concern: {mat['concern']}")
    if not thermal_ok:
        safety_issues.append(
            f"Thermal runaway risk — Da_thermal = {calc.thermal_damkohler:.2f} ≥ 1"
        )
    # Multi-stage: check atmosphere transition risk
    if is_multistage and len(stages) >= 2:
        s1 = stages[0]
        if s1.oxygen_sensitive:
            safety_issues.append(
                "Multi-stage: O₂ introduction at Stage 2 must be physically isolated "
                "from Stage 1 (e.g., tube-in-tube contactor or separate gas stream after Stage 1 outlet)"
            )

    safety_status = "RED" if safety_issues else "GREEN"
    sq = " | ".join(safety_issues) if safety_issues else (
        f"Confirm: BPR = {proposal.BPR_bar:.1f} bar ≥ {bpr['P_min_bar']:.2f} bar, "
        f"{proposal.tubing_material} in {solvent} at {T}°C is compatible, "
        f"thermal Da = {calc.thermal_damkohler:.3f} (safe)."
    )

    entries.append(TriageEntry(
        domain="SAFETY",
        status=safety_status,
        message=(
            f"BPR {'OK' if not safety_issues else 'ISSUE'}, "
            f"material {'OK' if mat['compatible'] else 'CONCERN'}, "
            f"thermal {'safe' if thermal_ok else 'RISK'}"
        ),
        directed_question=sq,
        tool_results={
            "BPR_current_bar": proposal.BPR_bar,
            "BPR_min_required_bar": bpr["P_min_bar"],
            "BPR_required": bpr["required"],
            "BPR_reason": bpr["reason"],
            "material_compatible": mat["compatible"],
            "material_concern": mat["concern"],
            "thermal_safe": thermal_ok,
            "thermal_Da": round(calc.thermal_damkohler or 0, 4),
            "is_gas_liquid": is_gas_liquid,
            "is_multistage": is_multistage,
        },
    ))

    # ── CHEMISTRY ─────────────────────────────────────────────────────────────
    chem_issues = []
    if chemistry_plan:
        if chemistry_plan.oxygen_sensitive and not proposal.deoxygenation_method:
            chem_issues.append("O₂-sensitive but no deoxygenation_method specified")
        if chemistry_plan.quench_required and not proposal.post_reactor_steps:
            chem_issues.append("Quench required but post_reactor_steps is empty")
        if is_multistage:
            for i, stage in enumerate(stages[1:], 2):
                if stage.atmosphere != stages[i - 2].atmosphere:
                    chem_issues.append(
                        f"Stage {i - 1} → Stage {i} atmosphere change "
                        f"({stages[i - 2].atmosphere} → {stage.atmosphere}) — "
                        "verify stream routing and physical separation"
                    )

    chem_status = "YELLOW" if chem_issues else "GREEN"
    cq = " | ".join(chem_issues) if chem_issues else (
        "Confirm stream assignments, atmosphere per stage, and mechanism fidelity."
    )

    entries.append(TriageEntry(
        domain="CHEMISTRY",
        status=chem_status,
        message=(
            f"Mechanism: {chemistry_plan.mechanism_type if chemistry_plan else 'unknown'}, "
            f"{n_stages} stage(s)"
        ),
        directed_question=cq,
        tool_results={
            "mechanism": chemistry_plan.mechanism_type if chemistry_plan else "",
            "n_stages": n_stages,
            "O2_sensitive": chemistry_plan.oxygen_sensitive if chemistry_plan else False,
            "deoxygenation_specified": bool(proposal.deoxygenation_method),
            "quench_required": chemistry_plan.quench_required if chemistry_plan else False,
            "quench_specified": bool(proposal.post_reactor_steps),
            "chem_issues": chem_issues,
        },
    ))

    # ── PHOTONICS (conditional) ────────────────────────────────────────────────
    if light_required:
        wl = proposal.wavelength_nm
        conc = proposal.concentration_M
        # Without ε we estimate risk from concentration + ID heuristics
        if conc and conc > 0.15 and ID > 1.0:
            phot_status = "YELLOW"
            pm = f"C = {conc:.2f} M at ID = {ID:.1f} mm — moderate inner filter risk"
            pq = (f"At C = {conc:.2f} M and tubing ID = {ID:.1f} mm, assess inner filter effect. "
                  f"Is ID reduction (0.75 mm?) or concentration adjustment required?")
        elif conc and conc > 0.3:
            phot_status = "YELLOW"
            pm = f"C = {conc:.2f} M — high concentration, inner filter likely regardless of ID"
            pq = (f"Concentration {conc:.2f} M is high for photocatalysis. "
                  f"Assess photon penetration and suggest adjustment if needed.")
        else:
            phot_status = "GREEN"
            pm = f"λ = {wl} nm, C = {conc:.3f} M, ID = {ID:.1f} mm — low inner filter risk"
            pq = f"Confirm λ = {wl} nm matches photocatalyst and LED. Confirm photon delivery is adequate."

        entries.append(TriageEntry(
            domain="PHOTONICS",
            status=phot_status,
            message=pm,
            directed_question=pq,
            tool_results={
                "wavelength_nm": wl,
                "concentration_M": conc,
                "tubing_ID_mm": ID,
                "light_required": True,
            },
        ))

    # ── Classify domains ───────────────────────────────────────────────────────
    green_domains = [e.domain for e in entries if e.status == "GREEN"]
    flagged_domains = [e.domain for e in entries if e.status in ("YELLOW", "RED")]

    design_center = (
        f"τ = {tau:.1f} min | Q = {Q:.4f} mL/min | ID = {ID:.2f} mm | "
        f"T = {T}°C | BPR = {proposal.BPR_bar:.1f} bar"
    )

    return TriageReport(
        design_center=design_center,
        entries=entries,
        green_domains=green_domains,
        flagged_domains=flagged_domains,
        is_gas_liquid=is_gas_liquid,
        is_multistage=is_multistage,
        light_required=light_required,
        n_stages=n_stages,
    )
