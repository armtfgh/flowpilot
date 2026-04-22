"""
FLORA ENGINE v4 — Designer agent.

Stage 0: Problem framing — confirms reaction class, batch conditions, flags, objectives.
Stage 1: Candidate matrix — deterministic (tau x d x Q) grid with hard-gate filtering.

All hard-gate thresholds are v4 values (L<=25m, V_R<=50mL, X_min=0.50).
No LLM invents tau, d, Q, Re, dP, or any derived metric — all come from sampling tools.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Optional

from flora_translate.engine.llm_agents import call_llm
from flora_translate.engine.sampling import (
    generate_candidates,
    format_candidate_table,
)

logger = logging.getLogger("flora.engine.council_v4.designer")

PI = math.pi

# ═══════════════════════════════════════════════════════════════════════════════
#  v4 hard-gate thresholds (differ from v3)
# ═══════════════════════════════════════════════════════════════════════════════

V4_L_MAX_M = 25.0
V4_V_R_MAX_ML = 50.0
V4_X_MIN = 0.50
V4_RE_MAX = 2300.0
V4_DP_FRACTION = 0.80        # dP > 0.80 * pump_max → hard disqualify
V4_A_MAX = 1.5               # Beer-Lambert absorbance hard ceiling
V4_RMIX_THRESHOLD = 0.20     # r_mix dual-criterion threshold
V4_DA_THRESHOLD = 1.0        # Da_mass dual-criterion threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 0 — Problem framing
# ═══════════════════════════════════════════════════════════════════════════════

_FRAMING_SYSTEM = """\
You are the FLORA ENGINE problem framer. Parse and confirm the chemistry context
into a structured problem statement for the council.

Return JSON only:
{
  "reaction_class": "photoredox | thermal | hydrogenation | gas-liquid | other",
  "special_flags": ["O2_sensitive", "moisture_sensitive", "exothermic", "gas_liquid",
                    "photochemical", "multi_stage"],
  "flow_justified": true,
  "flow_justification_note": "brief note if flow is not obviously justified",
  "ambiguities": ["list anything unclear that the council should assume or ask about"]
}
"""


def run_problem_framing(
    reaction_class: str,
    is_photochem: bool,
    is_gas_liquid: bool,
    is_O2_sensitive: bool,
    tau_center_min: float,
    tau_lit_min: Optional[float],
    solvent: str,
    temperature_C: float,
    concentration_M: float,
    objectives: str,
) -> dict:
    """Stage 0: Confirm problem framing. Returns structured problem statement."""
    flags = []
    if is_photochem:
        flags.append("photochemical")
    if is_gas_liquid:
        flags.append("gas_liquid")
    if is_O2_sensitive:
        flags.append("O2_sensitive")

    user_msg = (
        f"reaction_class={reaction_class} | solvent={solvent} | T={temperature_C}°C\n"
        f"C={concentration_M} M | tau_batch_equiv={tau_center_min:.1f} min | "
        f"tau_lit={'%.1f' % tau_lit_min if tau_lit_min else 'unknown'} min\n"
        f"flags={flags} | objectives={objectives}\n\n"
        "Confirm and output JSON."
    )

    framing: dict = {
        "reaction_class": reaction_class,
        "special_flags": flags,
        "flow_justified": True,
        "flow_justification_note": "",
        "ambiguities": [],
    }
    try:
        raw = call_llm(_FRAMING_SYSTEM, user_msg, max_tokens=400)
        s = raw.strip()
        if "```" in s:
            for part in s.split("```")[1::2]:
                try:
                    framing.update(json.loads(part.lstrip("json").strip()))
                    break
                except json.JSONDecodeError:
                    continue
        else:
            try:
                framing.update(json.loads(s))
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.warning("Stage 0 framing LLM call failed: %s — using defaults", e)

    return framing


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1 sampling strategy prompt (same role as v3 Designer)
# ═══════════════════════════════════════════════════════════════════════════════

_DESIGNER_SYSTEM_V4 = """\
You are the DESIGNER in the FLORA ENGINE v4 council. Your job is to select a
SAMPLING STRATEGY — not specific numbers. The actual tau, d, Q values and all
derived metrics are computed deterministically by tools; you only decide the
*coverage* of the design space.

## Your expertise
• Hagen-Poiseuille: dP ∝ L·Q/d⁴. Laminar (Re < 2300). Coils add Dean mixing.
• Mixing: t_mix ≈ d²/(4·D). Halving d gives 4× faster mixing.
• Photochem: Beer-Lambert A = ε·C·(d[mm]·0.1). Inner-filter risk if A > 1.5.
  FEP/PFA mandatory; PTFE is opaque. Prefer d ≤ 1.0 mm for photoredox.
• Gas-liquid: slug flow needs d ≥ 0.75 mm. BPR ≥ 5 bar mandatory.
• IF class ranges: photoredox 4–8×, thermal 8–15×, hydrogenation 20–50×,
  radical 10–60×, cross-coupling 5–20×.
• v4 bench envelope: L ≤ 25 m, V_R ≤ 50 mL.
• Commercial d values: 0.5, 0.75, 1.0, 1.5, 2.0 mm.

## Sampling strategy guidance
- Short-tau aggressive (high photon-flux photoredox, fast thermal): sample down to tau_center/3.
- Long-tau (slow thermal, cross-coupling): sample up to 2×–3×.
- O2-sensitive photoredox: bias toward d ≤ 1.0 mm; exclude d > 1.0 mm.
- Gas-liquid: exclude d < 0.75 mm.
- Always include tau_lit as an anchor point when known.
- Log-spaced when range > 3×; linear when narrow.

## REQUIRED OUTPUT — JSON only
```json
{
  "reasoning": "2-3 sentences on WHY this strategy fits this chemistry.",
  "tau_low_factor": 0.3,
  "tau_high_factor": 2.0,
  "n_tau": 5,
  "tau_log_spaced": true,
  "d_exclude_above_mm": 1.0,
  "include_long_L_fraction": true
}
```

Rules: tau_low_factor ∈ [0.15, 0.8], tau_high_factor ∈ [1.2, 3.0], n_tau ∈ [3, 8].
d_exclude_above_mm: 1.0 for photochem, 2.0 for general liquid, null otherwise.
"""


_DEFAULT_STRATEGY_V4 = {
    "reasoning": "Default balanced sampling — calculator center-point is trusted.",
    "tau_low_factor": 0.4,
    "tau_high_factor": 2.0,
    "n_tau": 5,
    "tau_log_spaced": True,
    "d_exclude_above_mm": None,
    "include_long_L_fraction": True,
}


def _parse_strategy(raw: str) -> dict:
    s = raw.strip()
    if "```" in s:
        for part in s.split("```")[1::2]:
            cleaned = part.lstrip("json").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(s[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


def _sanitize_strategy(data: dict) -> dict:
    s = dict(_DEFAULT_STRATEGY_V4)
    s.update({k: v for k, v in data.items() if k in _DEFAULT_STRATEGY_V4})
    try:
        s["tau_low_factor"] = max(0.15, min(0.8, float(s["tau_low_factor"])))
    except (TypeError, ValueError):
        s["tau_low_factor"] = 0.4
    try:
        s["tau_high_factor"] = max(1.2, min(3.0, float(s["tau_high_factor"])))
    except (TypeError, ValueError):
        s["tau_high_factor"] = 2.0
    try:
        s["n_tau"] = int(max(3, min(8, int(s["n_tau"]))))
    except (TypeError, ValueError):
        s["n_tau"] = 5
    s["tau_log_spaced"] = bool(s.get("tau_log_spaced", True))
    try:
        v = s.get("d_exclude_above_mm")
        s["d_exclude_above_mm"] = float(v) if v is not None else None
    except (TypeError, ValueError):
        s["d_exclude_above_mm"] = None
    s["include_long_L_fraction"] = bool(s.get("include_long_L_fraction", True))
    return s


# ═══════════════════════════════════════════════════════════════════════════════
#  v4 hard-gate filter (applied AFTER sampling, BEFORE agents see candidates)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_v4_hard_gates(
    candidates: list[dict],
    pump_max_bar: float,
    is_photochem: bool,
    is_gas_liquid: bool,
    BPR_bar: float,
    X_minimum: float = V4_X_MIN,
    tubing_material: str = "FEP",
) -> tuple[list[dict], list[dict]]:
    """Flag candidates that violate hard-gate thresholds.

    All candidates are returned in survivors so scoring agents can still
    evaluate them. Flagged candidates carry hard_gate_flags and hard_gate_status
    so every downstream agent can see exactly why they were flagged.

    Returns (all_candidates, flagged).
    Each flagged entry has {'candidate': dict, 'reason': str}.
    """
    flagged: list[dict] = []

    _OPAQUE = {"PTFE", "PEEK", "SS", "SS316", "HASTELLOY"}

    for c in candidates:
        reasons: list[str] = []

        def _f(key: str, default: float = 0.0) -> float:
            v = c.get(key)
            return float(v) if v is not None else default

        # Dual-criterion mixing: BOTH conditions must fail
        Da_mass = _f("Da_mass")
        r_mix = _f("r_mix")
        if Da_mass > V4_DA_THRESHOLD and r_mix > V4_RMIX_THRESHOLD:
            reasons.append(
                f"dual-criterion mixing fail: Da_mass={Da_mass:.2f} > 1.0 "
                f"AND r_mix={r_mix:.3f} > 0.20"
            )

        # Pressure
        dP = _f("delta_P_bar")
        if dP > V4_DP_FRACTION * pump_max_bar:
            reasons.append(
                f"dP={dP:.3f} bar > {V4_DP_FRACTION:.0%} of pump_max={pump_max_bar} bar"
            )

        # Length
        L = _f("L_m")
        if L > V4_L_MAX_M:
            reasons.append(f"L={L:.1f} m > {V4_L_MAX_M} m bench limit")

        # Reynolds
        Re = _f("Re")
        if Re > V4_RE_MAX:
            reasons.append(f"Re={Re:.0f} > 2300 (turbulent)")

        # Beer-Lambert inner filter (hard only at A > 1.5)
        A = _f("absorbance")
        if A > V4_A_MAX:
            reasons.append(f"A={A:.3f} > 1.5 (severe inner filter; no photons reach core)")

        # BPR (if known)
        BPR_min = _f("BPR_min_bar")
        if BPR_min > 0 and BPR_bar < BPR_min:
            reasons.append(
                f"BPR_current={BPR_bar} bar < BPR_min={BPR_min:.2f} bar"
            )
        # Gas-liquid floor
        if is_gas_liquid and BPR_bar < 5.0:
            reasons.append(
                f"gas-liquid system: BPR={BPR_bar} bar < 5.0 bar mandatory floor"
            )

        # Reactor volume
        V_R = _f("V_R_mL")
        if V_R > V4_V_R_MAX_ML:
            reasons.append(f"V_R={V_R:.1f} mL > {V4_V_R_MAX_ML} mL hard ceiling")

        # Conversion
        X = _f("expected_conversion", 1.0)
        if X < X_minimum:
            reasons.append(
                f"X={X:.2f} < X_minimum={X_minimum:.2f} (insufficient conversion)"
            )

        # Opaque material for photochemical reactor
        mat = tubing_material.upper()
        if is_photochem and mat in _OPAQUE:
            reasons.append(
                f"material={mat} is opaque — photochemical reactor requires FEP/PFA"
            )

        if reasons:
            c["hard_gate_status"] = "FLAGGED: " + "; ".join(reasons)
            c["hard_gate_flags"] = reasons
            flagged.append({"candidate": c, "reason": "; ".join(reasons)})
        else:
            c["hard_gate_status"] = "PASS"
            c["hard_gate_flags"] = []

    # All candidates proceed — flagged ones carry their flags for agents to see
    return candidates, flagged


# ═══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_designer_v4(
    *,
    reaction_class: str,
    is_photochem: bool,
    is_gas_liquid: bool,
    is_O2_sensitive: bool,
    tau_center_min: float,
    tau_lit_min: Optional[float],
    tau_kinetics_min: float,
    d_center_mm: float,
    Q_center_mL_min: float,
    solvent: str,
    temperature_C: float,
    concentration_M: float,
    assumed_MW: float,
    IF_used: float,
    pump_max_bar: float,
    BPR_bar: float = 0.0,
    extinction_coeff_M_cm: Optional[float] = None,
    tubing_material: str = "FEP",
    X_minimum: float = V4_X_MIN,
    N_target: int = 12,
    problem_statement: Optional[dict] = None,
) -> dict:
    """Run Stage 1: Designer candidate matrix for v4.

    Returns:
        {
          "problem_statement": {...},      # from Stage 0
          "strategy": {...},
          "strategy_reasoning": "...",
          "survivors": [ ... candidate dicts ... ],  # passed hard gates
          "disqualified": [{"candidate": {...}, "reason": "..."}],
          "all_candidates": [...],
          "table_markdown": "...",         # survivors only
          "design_envelope_preliminary": {...},
        }
    """
    chem_brief = (
        f"reaction_class={reaction_class} | photochem={is_photochem} | "
        f"gas_liquid={is_gas_liquid} | O2_sensitive={is_O2_sensitive}\n"
        f"tau_center={tau_center_min:.1f} min | tau_lit="
        f"{'%.1f' % tau_lit_min if tau_lit_min else 'none'} min | "
        f"tau_kinetics(90% X)={tau_kinetics_min:.1f} min | IF={IF_used:.1f}\n"
        f"d_center={d_center_mm} mm | Q_center={Q_center_mL_min:.3f} mL/min\n"
        f"solvent={solvent} | T={temperature_C}°C | C={concentration_M} M | "
        f"MW={assumed_MW:.0f} g/mol | pump_max={pump_max_bar} bar"
    )
    if is_photochem:
        chem_brief += (
            f"\nε(photocatalyst)={extinction_coeff_M_cm or 'not provided'} M⁻¹cm⁻¹"
        )

    user_msg = (
        "## Chemistry & calculator center-point\n" + chem_brief +
        "\n\nChoose the v4 sampling strategy. Output JSON only."
    )

    strategy = dict(_DEFAULT_STRATEGY_V4)
    strategy_reasoning = strategy["reasoning"]
    try:
        raw = call_llm(_DESIGNER_SYSTEM_V4, user_msg, max_tokens=500)
        parsed = _parse_strategy(raw)
        if parsed:
            strategy = _sanitize_strategy(parsed)
            strategy_reasoning = strategy.get("reasoning", strategy_reasoning)
    except Exception as e:
        logger.warning("Designer v4 LLM call failed — using default strategy: %s", e)

    logger.info(
        "    Designer v4 strategy: tau∈[%.2f×, %.2f×], n_tau=%d, %s, d≤%s mm",
        strategy["tau_low_factor"], strategy["tau_high_factor"],
        strategy["n_tau"], "log" if strategy["tau_log_spaced"] else "lin",
        strategy["d_exclude_above_mm"],
    )

    L_fractions = (
        [0.40, 0.60, 0.80, 0.95] if strategy["include_long_L_fraction"]
        else [0.40, 0.60, 0.80]
    )

    # Generate candidates (v3 sampling engine — still correct)
    all_feasible, all_infeasible = generate_candidates(
        tau_center_min=tau_center_min,
        tau_lit_min=tau_lit_min,
        solvent=solvent, temperature_C=temperature_C,
        concentration_M=concentration_M, assumed_MW=assumed_MW,
        IF_used=IF_used, tau_kinetics_min=tau_kinetics_min,
        pump_max_bar=pump_max_bar, is_photochem=is_photochem,
        is_gas_liquid=is_gas_liquid, BPR_bar=BPR_bar,
        extinction_coeff_M_cm=extinction_coeff_M_cm,
        tau_low_factor=strategy["tau_low_factor"],
        tau_high_factor=strategy["tau_high_factor"],
        n_tau=strategy["n_tau"],
        tau_log_spaced=strategy["tau_log_spaced"],
        d_exclude_above_mm=strategy["d_exclude_above_mm"],
        L_fractions=L_fractions,
        N_target=N_target,
    )

    # Re-assign sequential IDs to all_feasible
    for i, c in enumerate(all_feasible, 1):
        c["id"] = i

    # Apply v4 hard-gate flags (all candidates proceed; flagged ones carry reasons)
    survivors, flagged = _apply_v4_hard_gates(
        all_feasible, pump_max_bar=pump_max_bar,
        is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
        BPR_bar=BPR_bar, X_minimum=X_minimum,
        tubing_material=tubing_material,
    )

    table = format_candidate_table(survivors, max_rows=N_target)

    logger.info(
        "    Designer v4: %d total feasible → %d to council (%d flagged, %d sampling-infeasible)",
        len(all_feasible), len(survivors), len(flagged), len(all_infeasible),
    )

    tau_range = [c["tau_min"] for c in survivors] if survivors else [tau_center_min]
    d_range = list(sorted({c["d_mm"] for c in survivors})) if survivors else [d_center_mm]
    Q_range = list(sorted({c["Q_mL_min"] for c in survivors})) if survivors else [Q_center_mL_min]

    return {
        "problem_statement": problem_statement or {},
        "strategy": strategy,
        "strategy_reasoning": strategy_reasoning,
        "survivors": survivors,
        "disqualified": flagged,   # kept for UI backward compat; these are flags, not removals
        "all_candidates": all_feasible,
        "table_markdown": table,
        "design_envelope_preliminary": {
            "tau_range": [min(tau_range), max(tau_range)],
            "d_range": d_range,
            "Q_range": [min(Q_range), max(Q_range)],
        },
    }
