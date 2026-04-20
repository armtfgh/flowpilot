"""
FLORA ENGINE v3 — Deterministic design-space sampling + metric computation.

These are the TOOLS the Designer agent calls. The LLM in the Designer picks
the sampling *strategy* (what ranges, what bias); the functions here generate
the actual numbers and compute the full metric set for each candidate. No
LLM ever invents a τ, d, Q, Re, ΔP, or productivity value.

Pipeline for one candidate:
    (τ, d, Q)  →  compute_metrics  →  hard_filter  →  accept/reject

A candidate dict has this schema (all floats rounded for display):
    {
      # inputs
      "tau_min": 45.0, "d_mm": 0.75, "Q_mL_min": 0.55, "tau_source": "center",
      # geometry
      "V_R_mL": 24.75, "L_m": 5.6,
      # fluidics
      "Re": 21.3, "flow_regime": "laminar",
      "delta_P_bar": 0.048, "delta_P_headroom_pct": 99.7,
      "velocity_m_s": 0.021,
      # mixing / mass transfer
      "t_mix_s": 140.6, "r_mix": 0.052, "Da_mass": 0.39,
      # kinetics / yield
      "expected_conversion": 0.94, "tau_kinetics_min": 15.0, "IF_used": 6.0,
      # throughput
      "STY_mol_L_h": 0.235, "productivity_mg_h": 117.5, "assumed_MW": 250,
      # photochem
      "is_photochem": True, "absorbance": 0.27, "inner_filter_risk": "LOW",
      # feasibility
      "feasible": True, "violations": [], "warnings": [...],
    }
"""

from __future__ import annotations
import math
from typing import Optional

from flora_translate.engine.tools import (
    calculate_reynolds,
    calculate_pressure_drop,
    beer_lambert,
)

PI = math.pi

# Bench constraints — hard physical limits, not weights
L_MAX_BENCH_M = 20.0
V_MAX_SINGLE_REACTOR_ML = 25.0
Q_MIN_ML_MIN = 0.05                # syringe pump floor (practical)
RE_TURBULENT = 2300.0
D_MOLECULAR = 1.0e-9               # m²/s, small organics in liquid
DELTA_P_SAFETY_FACTOR = 0.8        # ΔP < 0.8 × pump_max

# Commercially available FEP/PFA tubing IDs for lab-scale flow chem
STANDARD_D_PHOTOCHEM = [0.50, 0.75, 1.00]          # mm — Beer-Lambert limited
STANDARD_D_GAS_LIQUID = [0.75, 1.00, 1.60]         # mm — needs slightly larger for slug flow
STANDARD_D_LIQUID = [0.75, 1.00, 1.60, 2.00]       # mm — general purpose

# BPR floor for gas-liquid (O₂, H₂, CO₂) — protocol rule
BPR_MIN_GAS_LIQUID_BAR = 5.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Sampling strategy → (τ, d, Q) triplets
# ═══════════════════════════════════════════════════════════════════════════════

def _log_spaced(low: float, high: float, n: int) -> list[float]:
    """Return n values log-spaced in [low, high] — good for τ ranges spanning >3×."""
    if n < 2 or low <= 0 or high <= 0:
        return [low]
    ratio = (high / low) ** (1.0 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]


def _linear_spaced(low: float, high: float, n: int) -> list[float]:
    if n < 2:
        return [(low + high) / 2.0]
    step = (high - low) / (n - 1)
    return [low + step * i for i in range(n)]


def generate_tau_samples(
    tau_center_min: float,
    tau_lit_min: Optional[float],
    low_factor: float = 0.3,
    high_factor: float = 2.0,
    n_samples: int = 5,
    log_spaced: bool = True,
    min_tau_min: float = 2.0,
) -> list[tuple[float, str]]:
    """Generate τ candidate values with provenance labels.

    Always includes:
      - τ_center (calculator anchor)
      - τ_lit/2 (literature anchor floor; Dr. Kinetics rule)
      - τ_lit itself if provided
    Plus log-/linear-spaced samples between low_factor·τ_center and high_factor·τ_center.
    """
    low = max(min_tau_min, tau_center_min * low_factor)
    high = max(low + 1.0, tau_center_min * high_factor)

    grid = _log_spaced(low, high, n_samples) if log_spaced \
        else _linear_spaced(low, high, n_samples)

    out: list[tuple[float, str]] = [(round(tau_center_min, 1), "center")]
    for t in grid:
        out.append((round(t, 1), "sampled"))
    if tau_lit_min and tau_lit_min > 0:
        out.append((round(tau_lit_min / 2.0, 1), "τ_lit/2"))
        if tau_lit_min > tau_center_min * 1.1:
            out.append((round(tau_lit_min, 1), "τ_lit"))

    # Deduplicate (keep first label for each value)
    seen: dict[float, str] = {}
    for t, src in out:
        if t >= min_tau_min and t not in seen:
            seen[t] = src
    return sorted(seen.items())


def choose_d_set(
    is_photochem: bool,
    is_gas_liquid: bool,
    exclude_above_mm: Optional[float] = None,
) -> list[float]:
    """Commercial tubing IDs appropriate for the chemistry."""
    if is_photochem:
        base = STANDARD_D_PHOTOCHEM
    elif is_gas_liquid:
        base = STANDARD_D_GAS_LIQUID
    else:
        base = STANDARD_D_LIQUID
    if exclude_above_mm is not None:
        base = [d for d in base if d <= exclude_above_mm]
    return base


def sample_design_space(
    tau_center_min: float,
    tau_lit_min: Optional[float],
    is_photochem: bool,
    is_gas_liquid: bool,
    tau_low_factor: float = 0.3,
    tau_high_factor: float = 2.0,
    n_tau: int = 5,
    tau_log_spaced: bool = True,
    d_exclude_above_mm: Optional[float] = None,
    L_fractions: Optional[list[float]] = None,
) -> list[tuple[float, float, float, str]]:
    """Full (τ, d, Q, τ_source) enumeration.

    Q is derived from τ and d via target L fractions: V_R = τ·Q = π/4·d²·L
    → Q = (π/4·d²·L) / τ. Trying several L fractions of L_MAX_BENCH_M gives
    a spread of Q values at fixed (τ, d) — some will violate ΔP/Re, others won't.

    Returns a raw candidate list (not yet filtered, not yet metriced).
    """
    L_fractions = L_fractions or [0.40, 0.60, 0.80, 0.95]

    tau_samples = generate_tau_samples(
        tau_center_min, tau_lit_min,
        low_factor=tau_low_factor, high_factor=tau_high_factor,
        n_samples=n_tau, log_spaced=tau_log_spaced,
    )
    d_values = choose_d_set(is_photochem, is_gas_liquid,
                             exclude_above_mm=d_exclude_above_mm)

    triplets: list[tuple[float, float, float, str]] = []
    for tau_min, tau_source in tau_samples:
        for d_mm in d_values:
            d_m = d_mm * 1e-3
            for L_frac in L_fractions:
                L_target_m = L_frac * L_MAX_BENCH_M
                V_R_target_mL = (PI * d_m ** 2 / 4.0) * L_target_m * 1e6
                Q_mL_min = V_R_target_mL / tau_min
                if Q_mL_min < Q_MIN_ML_MIN:
                    continue
                triplets.append((tau_min, d_mm, round(Q_mL_min, 5), tau_source))
    return triplets


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric computation — one candidate, all numbers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    tau_min: float,
    d_mm: float,
    Q_mL_min: float,
    solvent: str,
    temperature_C: float,
    concentration_M: float,
    assumed_MW: float,
    IF_used: float,
    tau_kinetics_min: float,
    pump_max_bar: float,
    is_photochem: bool,
    extinction_coeff_M_cm: Optional[float] = None,
    tau_source: str = "",
) -> dict:
    """Compute the full metric set for a single (τ, d, Q) candidate."""
    d_m = d_mm * 1e-3

    # Geometry: V_R = τ · Q ; L = 4 V_R / (π d²)
    V_R_mL = tau_min * Q_mL_min
    V_R_m3 = V_R_mL * 1e-6
    L_m = 4.0 * V_R_m3 / (PI * d_m ** 2) if d_m > 0 else 0.0

    # Fluidics
    re = calculate_reynolds(Q_mL_min, d_mm, solvent, temperature_C)
    dP = calculate_pressure_drop(Q_mL_min, d_mm, L_m, solvent)
    dP_bar = dP["delta_P_bar"]
    dP_headroom_pct = max(0.0, 100.0 * (1.0 - dP_bar / (DELTA_P_SAFETY_FACTOR * pump_max_bar))) \
        if pump_max_bar > 0 else 0.0

    # Mixing / mass transfer
    t_mix_s = d_m ** 2 / (4.0 * D_MOLECULAR)
    tau_s = tau_min * 60.0
    r_mix = t_mix_s / tau_s if tau_s > 0 else float("inf")

    # Damköhler (mass-transfer): k·d²/D where k inferred from τ_kinetics (90% conv)
    # τ_k = -ln(0.10)/k  →  k = 2.303/τ_k_s
    tau_k_s = max(tau_kinetics_min * 60.0, 1.0)
    k_s_inv = 2.303 / tau_k_s
    Da_mass = k_s_inv * d_m ** 2 / D_MOLECULAR

    # Kinetics — first-order expected conversion
    X = 1.0 - math.exp(-tau_s / tau_k_s)
    X = min(X, 1.0)

    # Throughput
    Q_L_h = Q_mL_min * 60.0 / 1000.0
    productivity_mg_h = concentration_M * Q_L_h * X * assumed_MW * 1000.0
    STY_mol_L_h = (concentration_M * X * Q_L_h) / (V_R_mL / 1000.0) if V_R_mL > 0 else 0.0

    # Photochem (optional)
    photochem_block: dict = {"is_photochem": is_photochem}
    if is_photochem and extinction_coeff_M_cm and extinction_coeff_M_cm > 0:
        bl = beer_lambert(concentration_M, extinction_coeff_M_cm, d_mm)
        photochem_block.update({
            "absorbance": bl["absorbance"],
            "transmission": bl["transmission"],
            "inner_filter_risk": bl["inner_filter_risk"],
        })
    elif is_photochem:
        photochem_block.update({
            "absorbance": None,
            "inner_filter_risk": "UNKNOWN (ε not provided)",
        })

    return {
        # inputs
        "tau_min": round(tau_min, 2),
        "d_mm": round(d_mm, 3),
        "Q_mL_min": round(Q_mL_min, 5),
        "tau_source": tau_source,
        # geometry
        "V_R_mL": round(V_R_mL, 4),
        "L_m": round(L_m, 3),
        # fluidics
        "Re": re["Re"],
        "flow_regime": re["flow_regime"],
        "velocity_m_s": re["velocity_m_s"],
        "delta_P_bar": round(dP_bar, 5),
        "delta_P_headroom_pct": round(dP_headroom_pct, 1),
        # mixing
        "t_mix_s": round(t_mix_s, 3),
        "r_mix": round(r_mix, 5),
        "Da_mass": round(Da_mass, 4),
        # kinetics
        "expected_conversion": round(X, 4),
        "tau_kinetics_min": round(tau_kinetics_min, 2),
        "IF_used": round(IF_used, 2),
        # throughput
        "STY_mol_L_h": round(STY_mol_L_h, 4),
        "productivity_mg_h": round(productivity_mg_h, 2),
        "assumed_MW": assumed_MW,
        # photochem
        **photochem_block,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Hard filter — physical/bench/safety constraints
# ═══════════════════════════════════════════════════════════════════════════════

def hard_filter(
    m: dict,
    is_photochem: bool,
    is_gas_liquid: bool,
    pump_max_bar: float,
    BPR_bar: float = 0.0,
) -> tuple[bool, list[str], list[str]]:
    """Apply hard bench/safety constraints.

    Returns (feasible, violations, warnings). Violations = candidate rejected.
    Warnings = kept but flagged for council attention.
    """
    violations: list[str] = []
    warnings: list[str] = []

    # Bench geometry
    if m["L_m"] > L_MAX_BENCH_M:
        violations.append(f"L={m['L_m']:.1f} m > {L_MAX_BENCH_M} m (bench limit)")
    elif m["L_m"] > 15.0:
        warnings.append(f"L={m['L_m']:.1f} m — upper practical range; plan 2 × 10 m coils")

    if m["V_R_mL"] > V_MAX_SINGLE_REACTOR_ML:
        violations.append(f"V_R={m['V_R_mL']:.1f} mL > {V_MAX_SINGLE_REACTOR_ML} mL (single coil)")

    # Fluidics
    if m["Re"] >= RE_TURBULENT:
        violations.append(f"Re={m['Re']:.0f} ≥ {RE_TURBULENT:.0f} (turbulent — out of regime)")
    elif m["Re"] > 1000:
        warnings.append(f"Re={m['Re']:.0f} > 1000 (transitional)")

    if m["delta_P_bar"] >= DELTA_P_SAFETY_FACTOR * pump_max_bar:
        violations.append(
            f"ΔP={m['delta_P_bar']:.3f} bar ≥ {DELTA_P_SAFETY_FACTOR:.0%} of pump_max "
            f"({pump_max_bar} bar)"
        )

    # Flow floor
    if m["Q_mL_min"] < Q_MIN_ML_MIN:
        violations.append(f"Q={m['Q_mL_min']:.4f} mL/min < {Q_MIN_ML_MIN} (syringe pump floor)")

    # Photochem: Beer-Lambert constraint (bench-physics rule — hard)
    if is_photochem and m["d_mm"] > 1.0:
        violations.append(f"d={m['d_mm']} mm > 1.0 mm — Beer-Lambert inner-filter risk")
    # Inner-filter HIGH is a chemistry-level concern (can be mitigated by dilution
    # or ε at LED wavelength). The Photonics advocate + Skeptic evaluate it.
    # Keep as a warning, not a hard filter.
    if is_photochem and m.get("inner_filter_risk") == "HIGH":
        A = m.get("absorbance")
        warnings.append(
            f"Inner-filter HIGH (A={A}) at d={m['d_mm']} mm — consider dilution "
            f"or verify ε at LED wavelength"
        )

    # Gas-liquid: BPR floor
    if is_gas_liquid and BPR_bar < BPR_MIN_GAS_LIQUID_BAR:
        warnings.append(
            f"BPR={BPR_bar} bar < {BPR_MIN_GAS_LIQUID_BAR} bar (gas-liquid hard rule — "
            f"Safety will enforce)"
        )

    # Mixing co-failure (Da AND r_mix)
    if m["Da_mass"] > 1.0 and m["r_mix"] > 0.20:
        warnings.append(
            f"Mixing-limited: Da_mass={m['Da_mass']:.2f} > 1 AND r_mix={m['r_mix']:.2f} > 0.20"
        )

    # Conversion
    if m["expected_conversion"] < 0.85:
        warnings.append(f"X={m['expected_conversion']:.2f} < 0.85 — undersized τ for kinetics")

    return (len(violations) == 0, violations, warnings)


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: end-to-end generate + evaluate
# ═══════════════════════════════════════════════════════════════════════════════

def generate_candidates(
    tau_center_min: float,
    tau_lit_min: Optional[float],
    solvent: str,
    temperature_C: float,
    concentration_M: float,
    assumed_MW: float,
    IF_used: float,
    tau_kinetics_min: float,
    pump_max_bar: float,
    is_photochem: bool,
    is_gas_liquid: bool,
    BPR_bar: float = 0.0,
    extinction_coeff_M_cm: Optional[float] = None,
    # strategy controls (Designer's LLM output flows in here)
    tau_low_factor: float = 0.3,
    tau_high_factor: float = 2.0,
    n_tau: int = 5,
    tau_log_spaced: bool = True,
    d_exclude_above_mm: Optional[float] = None,
    L_fractions: Optional[list[float]] = None,
    N_target: int = 12,
) -> tuple[list[dict], list[dict]]:
    """Generate → metrics → hard filter. Returns (feasible, infeasible).

    Feasible candidates are sorted by an objective-free composite:
    we DO NOT score here (that's the Expert's job). We return feasible
    candidates in a deterministic order based on Pareto dominance on
    (productivity↑, L↓, r_mix↓) — so the Designer's "shortlist" is
    non-dominated points first, then the rest.

    Limits the returned feasible list to ~N_target candidates using
    farthest-point sampling in the normalized objective space for diversity.
    """
    triplets = sample_design_space(
        tau_center_min=tau_center_min, tau_lit_min=tau_lit_min,
        is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
        tau_low_factor=tau_low_factor, tau_high_factor=tau_high_factor,
        n_tau=n_tau, tau_log_spaced=tau_log_spaced,
        d_exclude_above_mm=d_exclude_above_mm, L_fractions=L_fractions,
    )

    feasible: list[dict] = []
    infeasible: list[dict] = []

    for tau_min, d_mm, Q_mL_min, tau_source in triplets:
        m = compute_metrics(
            tau_min=tau_min, d_mm=d_mm, Q_mL_min=Q_mL_min,
            solvent=solvent, temperature_C=temperature_C,
            concentration_M=concentration_M, assumed_MW=assumed_MW,
            IF_used=IF_used, tau_kinetics_min=tau_kinetics_min,
            pump_max_bar=pump_max_bar, is_photochem=is_photochem,
            extinction_coeff_M_cm=extinction_coeff_M_cm,
            tau_source=tau_source,
        )
        ok, viol, warns = hard_filter(
            m, is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
            pump_max_bar=pump_max_bar, BPR_bar=BPR_bar,
        )
        m["feasible"] = ok
        m["violations"] = viol
        m["warnings"] = warns
        (feasible if ok else infeasible).append(m)

    # Non-dominated ordering on (productivity↑, L↓, r_mix↓)
    pareto = _pareto_front(feasible, [
        ("productivity_mg_h", "max"),
        ("L_m", "min"),
        ("r_mix", "min"),
    ])
    pareto_ids = {id(p) for p in pareto}
    dominated = [c for c in feasible if id(c) not in pareto_ids]
    # Tag non-dominated points explicitly
    for p in pareto:
        p["pareto_front"] = True
    for d in dominated:
        d["pareto_front"] = False
    # Diversity-preserving reduction: round-robin across distinct d values from
    # the Pareto front, so d=0.75 points aren't drowned out by high-productivity
    # d=1.0 points. Then fill the remaining slots from the front, then from
    # dominated points if needed.
    ordered: list[dict] = []
    if len(pareto) + len(dominated) > N_target:
        # Group Pareto points by d
        by_d: dict[float, list[dict]] = {}
        for p in pareto:
            by_d.setdefault(p["d_mm"], []).append(p)
        # Sort each d-group by productivity desc (top ones first)
        for d in by_d:
            by_d[d].sort(key=lambda c: c.get("productivity_mg_h", 0), reverse=True)
        # Round-robin pick: one point per d per sweep until we run out or hit target
        while len(ordered) < N_target and any(by_d.values()):
            for d in sorted(by_d.keys()):
                if by_d[d]:
                    ordered.append(by_d[d].pop(0))
                    if len(ordered) >= N_target:
                        break
        # If we still have room, fill from dominated points (highest-prod first)
        if len(ordered) < N_target:
            for c in dominated[: N_target - len(ordered)]:
                ordered.append(c)
    else:
        ordered = pareto + dominated

    return ordered, infeasible


def _pareto_front(items: list[dict], objectives: list[tuple[str, str]]) -> list[dict]:
    """Return non-dominated subset. objectives = [(key, 'max'|'min'), ...]."""
    front: list[dict] = []
    for a in items:
        dominated = False
        for b in items:
            if a is b:
                continue
            if _dominates(b, a, objectives):
                dominated = True
                break
        if not dominated:
            front.append(a)
    # Stable sort: highest productivity first on the front
    front.sort(key=lambda c: c.get("productivity_mg_h", 0), reverse=True)
    return front


def _dominates(a: dict, b: dict, objectives: list[tuple[str, str]]) -> bool:
    strictly_better = False
    for key, direction in objectives:
        av, bv = a.get(key, 0.0), b.get(key, 0.0)
        if direction == "max":
            if av < bv:
                return False
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                return False
            if av < bv:
                strictly_better = True
    return strictly_better


def format_candidate_table(candidates: list[dict], max_rows: int = 15) -> str:
    """Render candidates as a compact markdown table for LLM consumption."""
    if not candidates:
        return "_no candidates_"
    header = (
        "| id | τ min | d mm | Q mL/min | V_R mL | L m | Re | ΔP bar | "
        "r_mix | X | prod mg/h | pareto | flags |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for i, c in enumerate(candidates[:max_rows]):
        flags = []
        if c.get("pareto_front"):
            flags.append("★")
        if c.get("warnings"):
            flags.append(f"⚠{len(c['warnings'])}")
        if not c.get("feasible", True):
            flags.append("✗")
        rows.append(
            f"| {i+1} | {c['tau_min']} | {c['d_mm']} | {c['Q_mL_min']} | "
            f"{c['V_R_mL']} | {c['L_m']} | {c['Re']:.0f} | {c['delta_P_bar']:.3f} | "
            f"{c['r_mix']:.3f} | {c['expected_conversion']:.2f} | "
            f"{c['productivity_mg_h']:.1f} | "
            f"{'✓' if c.get('pareto_front') else ' '} | {' '.join(flags)} |"
        )
    return header + "\n" + "\n".join(rows)
