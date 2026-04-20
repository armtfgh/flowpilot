"""
FLORA ENGINE — Design Space Grid Search.

Runs BEFORE the council. Enumerates (τ, d, Q) combinations:
  τ values : [τ_lit/2, τ_center×0.75, τ_center, τ_center×1.25, τ_center×1.5]
             plus τ_lit itself if τ_lit > τ_center
  d values : commercial FEP/PFA sizes filtered by photochem constraint
  Q per (τ,d): at L fractions [0.3, 0.5, 0.7, 0.9] × L_MAX_BENCH_M

Hard filters (violators get score=0, excluded from table):
  - L ≤ L_MAX_BENCH_M = 20 m
  - V_R ≤ V_MAX_SINGLE_REACTOR_ML = 25 mL
  - Re < 2300
  - ΔP < 0.8 × pump_max
  - Q ≥ 0.01 mL/min (syringe pump floor)
  - For photochem: d ≤ 1.0 mm (Beer-Lambert constraint at typical ε)

Soft scoring (0–1, weighted sum):
  productivity_score = Productivity_mg_h / max(Productivity_mg_h) — normalized
  L_score = 1 − L_m / L_MAX
  mixing_score = 1 − min(r_mix / 0.20, 1.0)
  re_score = 1 − Re / 2300
  conversion_score = expected_conversion (X = 1 − exp(−τ/τ_kinetics))

Weights by reaction class:
  photoredox: productivity=0.25, L=0.30, mixing=0.25, re=0.05, conversion=0.15
  thermal:    productivity=0.40, L=0.25, mixing=0.20, re=0.05, conversion=0.10
  default:    productivity=0.30, L=0.30, mixing=0.20, re=0.05, conversion=0.15
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

PI = math.pi
L_MAX_BENCH_M = 20.0
V_MAX_SINGLE_REACTOR_ML = 25.0
# Practical syringe pump floor: 0.05 mL/min avoids runs of >200 min per 10 mL syringe
Q_MIN_ML_MIN = 0.05
D_MOLECULAR = 1.0e-9  # m²/s
STANDARD_D_PHOTOCHEM = [0.50, 0.75, 1.00]          # mm — Beer-Lambert limited
STANDARD_D_NOPHOTO   = [0.75, 1.00, 1.60]          # mm
# L fractions: stay in the 10–20 m range — avoids both extremes
# 0.5 → 10 m (compact, easy to handle), 0.9 → 18 m (near limit, more volume)
L_FRACTIONS          = [0.50, 0.65, 0.80, 0.90]    # fraction of L_MAX

# Score weights: productivity is the primary objective — L is a constraint
# that's already satisfied by the hard filter. Rewarding shorter L beyond
# what's needed just selects impractically low Q.
# L_score uses a soft step: 1.0 for L ≤ 15 m, degrades linearly toward 0
# only for L ∈ [15, 20] m. This prevents the optimizer from choosing
# Q = 0.01 mL/min just because it gives L = 4 m.
SCORE_WEIGHTS = {
    "photoredox":    {"productivity": 0.50, "L": 0.15, "mixing": 0.25, "re": 0.05, "conversion": 0.05},
    "photocatalysis":{"productivity": 0.50, "L": 0.15, "mixing": 0.25, "re": 0.05, "conversion": 0.05},
    "photochem":     {"productivity": 0.50, "L": 0.15, "mixing": 0.25, "re": 0.05, "conversion": 0.05},
    "thermal":       {"productivity": 0.55, "L": 0.15, "mixing": 0.15, "re": 0.05, "conversion": 0.10},
    "default":       {"productivity": 0.50, "L": 0.15, "mixing": 0.20, "re": 0.05, "conversion": 0.10},
}


def _l_score(L_m: float) -> float:
    """Non-linear L score: full marks up to 15 m, then linear penalty to 20 m.
    L is already guaranteed ≤ L_MAX by the hard filter, so this only matters
    in the 15–20 m zone where lab assembly becomes more difficult.
    """
    if L_m <= 15.0:
        return 1.0
    return max(0.0, 1.0 - (L_m - 15.0) / 5.0)


@dataclass
class DesignPoint:
    # Inputs
    tau_min: float          # residence time (min)
    d_mm: float             # tubing ID (mm)
    Q_mL_min: float         # flow rate (mL/min)
    L_fraction: float       # fraction of L_max used (0.3–0.9)

    # Derived geometry
    V_R_mL: float = 0.0
    L_m: float = 0.0

    # Fluid dynamics
    Re: float = 0.0
    delta_P_bar: float = 0.0

    # Mass transfer
    t_mix_s: float = 0.0
    r_mix: float = 0.0
    Da_mass: float = 0.0

    # Kinetics
    expected_conversion: float = 0.0
    tau_kinetics_min: float = 0.0   # τ that achieves 90% conversion
    IF_used: float = 1.0

    # Process metrics
    STY_mol_L_h: float = 0.0
    productivity_mg_h: float = 0.0
    assumed_MW: float = 250.0  # g/mol default

    # Score
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)

    # Status
    feasible: bool = True
    violations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # Metadata
    tau_source: str = ""  # "center", "center×0.75", "τ_lit/2", etc.
    is_council_candidate: bool = False   # True for the top candidate passed to council


class DesignSpaceSearch:
    """Grid search over (τ, d, Q) combinations before the council."""

    def run(
        self,
        batch_record,
        chemistry_plan=None,
        calculations=None,    # DesignCalculations from initial calc run
        inventory=None,
        reaction_class: str = "default",
    ) -> list[DesignPoint]:
        """Enumerate and score all (τ, d, Q) design points.

        Returns the full list (feasible + infeasible), sorted by score desc.
        """
        # ── 1. Get τ_center and τ_lit ─────────────────────────────────────
        tau_center = 30.0  # fallback
        if calculations is not None:
            try:
                tau_center = float(calculations.residence_time_min or 30.0)
            except (TypeError, AttributeError):
                tau_center = 30.0

        tau_lit = None
        if calculations is not None:
            try:
                tau_lit = calculations.tau_analogy_min
            except AttributeError:
                tau_lit = None
            if not tau_lit:
                try:
                    tau_lit = calculations.tau_class_min
                except AttributeError:
                    tau_lit = None

        # ── 2. Build τ_values ─────────────────────────────────────────────
        tau_candidates = [
            tau_center * 0.75,
            tau_center,
            tau_center * 1.25,
            tau_center * 1.50,
        ]
        if tau_lit and tau_lit > 0:
            tau_candidates.append(tau_lit / 2.0)
            if tau_lit > tau_center:
                tau_candidates.append(tau_lit)

        # Deduplicate, round to 1 decimal, filter minimum 5 min
        seen = set()
        tau_values = []
        tau_sources = {}
        for t in tau_candidates:
            t_r = round(t, 1)
            if t_r < 5.0:
                t_r = 5.0
            if t_r not in seen:
                seen.add(t_r)
                tau_values.append(t_r)
                # Assign source label
                if abs(t - tau_center) < 0.01:
                    tau_sources[t_r] = "center"
                elif abs(t - tau_center * 0.75) < 0.01:
                    tau_sources[t_r] = "center×0.75"
                elif abs(t - tau_center * 1.25) < 0.01:
                    tau_sources[t_r] = "center×1.25"
                elif abs(t - tau_center * 1.50) < 0.01:
                    tau_sources[t_r] = "center×1.50"
                elif tau_lit and abs(t - tau_lit / 2.0) < 0.01:
                    tau_sources[t_r] = "τ_lit/2"
                elif tau_lit and abs(t - tau_lit) < 0.01:
                    tau_sources[t_r] = "τ_lit"
                else:
                    tau_sources[t_r] = "derived"

        tau_values.sort()

        # ── 3. Detect photochem ───────────────────────────────────────────
        is_photochem = False
        if chemistry_plan is not None:
            try:
                rc = (chemistry_plan.reaction_class or "").lower()
                if any(x in rc for x in ("photo", "redox", "photocatalysis")):
                    is_photochem = True
            except AttributeError:
                pass
            try:
                wl = chemistry_plan.wavelength_nm
                if wl and wl > 0:
                    is_photochem = True
            except AttributeError:
                pass
        if batch_record is not None and not is_photochem:
            try:
                wl_br = batch_record.wavelength_nm
                if wl_br and wl_br > 0:
                    is_photochem = True
            except AttributeError:
                pass

        # ── 4. Choose d_values ────────────────────────────────────────────
        d_values = STANDARD_D_PHOTOCHEM if is_photochem else STANDARD_D_NOPHOTO

        # ── 5. Get pump_max from inventory ────────────────────────────────
        pump_max = 20.0  # default bar
        if inventory is not None:
            try:
                pump_max = float(inventory.pump_max_bar or 20.0)
            except AttributeError:
                pass

        # ── 6. Get solvent ────────────────────────────────────────────────
        solvent = "MeCN"  # safe default
        if batch_record is not None:
            try:
                solvent = batch_record.solvent or solvent
            except AttributeError:
                pass
        if chemistry_plan is not None:
            try:
                sp = chemistry_plan.solvent
                if sp:
                    solvent = sp
            except AttributeError:
                pass

        # ── 7. Get τ_kinetics and rate_constant ───────────────────────────
        k_flow = None
        tau_kinetics_min = tau_center  # fallback: τ_center ≈ 90% conversion τ
        if calculations is not None:
            try:
                k_flow = calculations.rate_constant
            except AttributeError:
                k_flow = None
            # τ_kinetics from k: τ = -ln(0.10)/k
            if k_flow and k_flow > 0:
                tau_kinetics_s = -math.log(0.10) / k_flow
                tau_kinetics_min = tau_kinetics_s / 60.0

        # ── 8. Get MW from batch_record ───────────────────────────────────
        assumed_MW = 250.0
        if batch_record is not None:
            try:
                mw = batch_record.product_MW
                if mw and mw > 0:
                    assumed_MW = float(mw)
            except AttributeError:
                pass

        # ── 9. Get concentration from calculations ────────────────────────
        concentration_M = 0.1
        if calculations is not None:
            try:
                concentration_M = float(calculations.concentration_M or 0.1)
            except AttributeError:
                pass
        if batch_record is not None:
            try:
                c = batch_record.concentration_M
                if c and c > 0:
                    concentration_M = float(c)
            except AttributeError:
                pass

        # ── 10. Get IF ────────────────────────────────────────────────────
        IF_used = 1.0
        if calculations is not None:
            try:
                IF_used = float(calculations.intensification_factor or 1.0)
            except AttributeError:
                pass

        # ── 11. Import tools ──────────────────────────────────────────────
        from flora_translate.engine.tools import calculate_reynolds, calculate_pressure_drop

        # ── 12. Enumerate ─────────────────────────────────────────────────
        candidates: list[DesignPoint] = []

        for tau_min in tau_values:
            tau_source = tau_sources.get(tau_min, "derived")
            for d_mm in d_values:
                d_m = d_mm * 1e-3
                for L_frac in L_FRACTIONS:
                    # Q from L_fraction
                    L_target_m = L_frac * L_MAX_BENCH_M
                    # V_R = π/4 * d² * L  → Q = V_R / τ
                    V_R_from_L = PI * (d_m ** 2) / 4.0 * L_target_m * 1e6  # mL
                    Q_mL_min = V_R_from_L / tau_min  # mL/min

                    if Q_mL_min < Q_MIN_ML_MIN:
                        continue

                    # V_R from τ and Q
                    V_R_mL = tau_min * Q_mL_min  # mL
                    if V_R_mL > V_MAX_SINGLE_REACTOR_ML:
                        continue

                    # Actual L
                    V_R_m3 = V_R_mL * 1e-6
                    L_m = 4.0 * V_R_m3 / (PI * d_m ** 2)

                    # Reynolds number
                    re_result = calculate_reynolds(Q_mL_min, d_mm, solvent)
                    Re = re_result["Re"]

                    # Pressure drop
                    dP_result = calculate_pressure_drop(Q_mL_min, d_mm, L_m, solvent)
                    delta_P_bar = dP_result["delta_P_bar"]

                    # Mixing
                    t_mix_s = d_m ** 2 / (4.0 * D_MOLECULAR)
                    tau_s = tau_min * 60.0
                    r_mix = t_mix_s / tau_s if tau_s > 0 else float("inf")

                    # Da_mass
                    if k_flow and k_flow > 0:
                        Da_mass = k_flow * (d_m ** 2) / D_MOLECULAR
                    else:
                        tau_kinetics_s_est = tau_kinetics_min * 60.0
                        k_estimated = 2.303 / tau_kinetics_s_est if tau_kinetics_s_est > 0 else 1e-6
                        Da_mass = k_estimated * (d_m ** 2) / D_MOLECULAR

                    # Expected conversion
                    tau_k_s = tau_kinetics_min * 60.0
                    if tau_k_s > 0:
                        X = 1.0 - math.exp(-tau_s / tau_k_s)
                    else:
                        X = 1.0 - math.exp(-tau_s / (tau_s + 1e-6))
                    X = min(X, 1.0)

                    # Productivity: C [mol/L] × Q [mL/min] × X × MW [g/mol] × 60 [min/h] × 1000 [mg/g] / 1000 [mL/L]
                    # = C [mol/L] × Q [L/h] × X × MW [g/mol] × 1000 [mg/g]
                    Q_L_h = Q_mL_min * 60.0 / 1000.0
                    productivity_mg_h = concentration_M * Q_L_h * X * assumed_MW * 1000.0

                    # STY (mol/L/h): C × X × Q / V_R
                    STY_mol_L_h = (concentration_M * X * Q_L_h) / (V_R_mL / 1000.0) if V_R_mL > 0 else 0.0

                    # ── Hard filters ──────────────────────────────────────
                    violations = []
                    if L_m > L_MAX_BENCH_M:
                        violations.append(f"L={L_m:.1f}m > {L_MAX_BENCH_M}m max")
                    if V_R_mL > V_MAX_SINGLE_REACTOR_ML:
                        violations.append(f"V_R={V_R_mL:.1f}mL > {V_MAX_SINGLE_REACTOR_ML}mL max")
                    if Re >= 2300:
                        violations.append(f"Re={Re:.0f} ≥ 2300 (turbulent)")
                    if delta_P_bar >= 0.8 * pump_max:
                        violations.append(f"ΔP={delta_P_bar:.3f}bar ≥ 0.8×{pump_max}={0.8*pump_max:.1f}bar")
                    if Q_mL_min < Q_MIN_ML_MIN:
                        violations.append(f"Q={Q_mL_min:.4f}mL/min < {Q_MIN_ML_MIN}mL/min floor")
                    if is_photochem and d_mm > 1.0:
                        violations.append(f"d={d_mm}mm > 1.0mm (Beer-Lambert photochem constraint)")

                    feasible = len(violations) == 0

                    # ── Warnings (soft) ───────────────────────────────────
                    point_warnings = []
                    if r_mix > 0.20:
                        point_warnings.append(f"Mixing ratio {r_mix:.3f} > 0.20 — consider active mixer")
                    if Re > 1000:
                        point_warnings.append(f"Re={Re:.0f} > 1000 — transitional regime")
                    if delta_P_bar > 0.5 * pump_max:
                        point_warnings.append(f"ΔP={delta_P_bar:.3f}bar > 50% pump max")

                    pt = DesignPoint(
                        tau_min=tau_min,
                        d_mm=d_mm,
                        Q_mL_min=round(Q_mL_min, 5),
                        L_fraction=L_frac,
                        V_R_mL=round(V_R_mL, 4),
                        L_m=round(L_m, 3),
                        Re=round(Re, 2),
                        delta_P_bar=round(delta_P_bar, 5),
                        t_mix_s=round(t_mix_s, 3),
                        r_mix=round(r_mix, 5),
                        Da_mass=round(Da_mass, 4),
                        expected_conversion=round(X, 4),
                        tau_kinetics_min=round(tau_kinetics_min, 2),
                        IF_used=round(IF_used, 2),
                        STY_mol_L_h=round(STY_mol_L_h, 4),
                        productivity_mg_h=round(productivity_mg_h, 4),
                        assumed_MW=assumed_MW,
                        feasible=feasible,
                        violations=violations,
                        warnings=point_warnings,
                        tau_source=tau_source,
                    )
                    candidates.append(pt)

        # ── 13. Normalize productivity and score ──────────────────────────
        feasible_candidates = [c for c in candidates if c.feasible]
        max_prod = max((c.productivity_mg_h for c in feasible_candidates), default=1.0)
        if max_prod <= 0:
            max_prod = 1.0

        # Get score weights
        weights = SCORE_WEIGHTS.get(
            (reaction_class or "default").lower(),
            SCORE_WEIGHTS["default"]
        )
        # Try to match substring
        if (reaction_class or "").lower() not in SCORE_WEIGHTS:
            for key in SCORE_WEIGHTS:
                if key in (reaction_class or "").lower():
                    weights = SCORE_WEIGHTS[key]
                    break

        for c in feasible_candidates:
            prod_score = c.productivity_mg_h / max_prod
            L_score = _l_score(c.L_m)     # non-linear: full credit ≤15m, penalty 15-20m
            mixing_score = 1.0 - min(c.r_mix / 0.20, 1.0)
            re_score = 1.0 - (c.Re / 2300.0)
            conv_score = c.expected_conversion

            score = (
                weights["productivity"] * prod_score
                + weights["L"] * L_score
                + weights["mixing"] * mixing_score
                + weights["re"] * re_score
                + weights["conversion"] * conv_score
            )
            c.score = round(score, 5)
            c.score_breakdown = {
                "productivity": round(prod_score, 4),
                "L": round(L_score, 4),
                "mixing": round(mixing_score, 4),
                "re": round(re_score, 4),
                "conversion": round(conv_score, 4),
            }

        # ── 14. Sort by score descending ──────────────────────────────────
        feasible_candidates.sort(key=lambda c: c.score, reverse=True)
        infeasible = [c for c in candidates if not c.feasible]

        # Rebuild combined list: feasible first (sorted), then infeasible
        candidates = feasible_candidates + infeasible

        # ── 15. Mark top candidate ────────────────────────────────────────
        if feasible_candidates:
            feasible_candidates[0].is_council_candidate = True

        return candidates


def get_council_starting_point(candidates: list[DesignPoint]) -> DesignPoint | None:
    """Return the top feasible candidate to use as the council starting proposal."""
    feasible = [c for c in candidates if c.feasible]
    return feasible[0] if feasible else None


def candidates_to_dicts(candidates: list[DesignPoint]) -> list[dict]:
    """Convert to JSON-serializable list of dicts for storage in result dict."""
    import dataclasses
    return [dataclasses.asdict(c) for c in candidates]
