"""
FLORA ENGINE v4 — Stage 3: Skeptic audit.

The v4 Skeptic is primarily a code-layer auditor, not an LLM assumption attacker.
Its job is to verify that all prior agents computed and applied values correctly.

Checks performed:
  1. Arithmetic verification (Beer-Lambert units, V_R=tau*Q, L formula)
  2. Mixing direction rule (decreasing d improves mixing — never increasing)
  3. Threshold correctness (Re turbulent = 2300, r_mix = 0.20, dual-criterion)
  4. Scope violation detection (agents operating outside their domain)
  5. Recalculation chain enforcement (parameter changes → dependent recalcs)

Returns council_may_proceed = False if any CRITICAL or HIGH errors are found.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import flora_translate.config as cfg
from flora_translate.config import FLOW_MAX_TAU_TO_BATCH_RATIO, FLOW_TRANSLATION_POLICY
from flora_translate.engine.flow_value import compute_pvs_for_candidates, mandate_dict

logger = logging.getLogger("flora.engine.council_v4.skeptic")

PI = math.pi
D_MOLECULAR = 1.0e-9   # m²/s, small organics

# ═══════════════════════════════════════════════════════════════════════════════
#  Arithmetic verification helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_beer_lambert(
    chemistry_scores: list[dict],
    candidates: list[dict],
    concentration_M: float,
) -> list[dict]:
    """Verify Beer-Lambert calculations: d must be in cm, not mm."""
    errors: list[dict] = []
    for entry in chemistry_scores:
        cid = entry.get("candidate_id")
        A_claimed = entry.get("beer_lambert_A")
        eps_used = entry.get("epsilon_used")
        if A_claimed is None or eps_used is None:
            continue
        # Find candidate
        cand = next((c for c in candidates if c.get("id") == cid), None)
        if cand is None:
            continue
        d_mm = cand.get("d_mm", 1.0)
        d_cm = d_mm * 0.1  # correct conversion
        A_expected = eps_used * concentration_M * d_cm
        A_diff = abs(float(A_claimed) - A_expected)
        if A_diff > 0.05 * max(A_expected, 0.01):
            # More than 5% discrepancy — likely used d in mm instead of cm
            A_wrong_units = eps_used * concentration_M * d_mm
            if abs(float(A_claimed) - A_wrong_units) < 0.05 * max(A_wrong_units, 0.01):
                errors.append({
                    "agent": "DR. CHEMISTRY",
                    "candidate_id": cid,
                    "error_type": "BEER_LAMBERT_UNITS",
                    "description": (
                        f"A={A_claimed:.4f} matches ε×C×d_mm={A_wrong_units:.4f} — "
                        f"path length used in mm not cm. Correct: A=ε×C×(d_mm×0.1)="
                        f"{A_expected:.4f}"
                    ),
                    "severity": "CRITICAL",
                })
            else:
                errors.append({
                    "agent": "DR. CHEMISTRY",
                    "candidate_id": cid,
                    "error_type": "BEER_LAMBERT_DISCREPANCY",
                    "description": (
                        f"Claimed A={A_claimed:.4f}, expected {A_expected:.4f} "
                        f"(ε={eps_used}, C={concentration_M} M, d={d_mm} mm → "
                        f"{d_cm} cm)"
                    ),
                    "severity": "MEDIUM",
                })
        # Sanity-check epsilon: standard Ir/Ru photocatalysts have ε < 5000 at LED wavelength
        if float(eps_used) > 5000:
            errors.append({
                "agent": "DR. CHEMISTRY",
                "candidate_id": cid,
                "error_type": "EPSILON_IMPLAUSIBLE",
                "description": (
                    f"ε={eps_used} M⁻¹cm⁻¹ > 5000 — unusually high for a standard "
                    "transition-metal photocatalyst at LED wavelength. Verify source."
                ),
                "severity": "HIGH",
            })
    return errors


def _verify_v_r_equals_tau_q(candidates: list[dict]) -> list[dict]:
    """Verify V_R = tau * Q for all candidates."""
    errors: list[dict] = []
    for c in candidates:
        cid = c.get("id", "?")
        tau = c.get("tau_min", 0.0)
        Q = c.get("Q_mL_min", 0.0)
        V_R = c.get("V_R_mL", 0.0)
        expected = tau * Q
        if abs(V_R - expected) > 0.05 * max(expected, 0.01):
            errors.append({
                "agent": "DESIGNER",
                "candidate_id": cid,
                "error_type": "V_R_CONSTRAINT_VIOLATED",
                "description": (
                    f"V_R={V_R:.3f} mL ≠ tau×Q={tau:.2f}×{Q:.4f}={expected:.3f} mL "
                    f"(discrepancy {abs(V_R-expected):.3f} mL)"
                ),
                "severity": "HIGH",
            })
    return errors


def _verify_length_formula(candidates: list[dict]) -> list[dict]:
    """Verify L = 4*V_R / (pi * d^2)."""
    errors: list[dict] = []
    for c in candidates:
        cid = c.get("id", "?")
        V_R_mL = c.get("V_R_mL", 0.0)
        d_mm = c.get("d_mm", 1.0)
        L_claimed = c.get("L_m", 0.0)
        V_R_m3 = V_R_mL * 1e-6
        d_m = d_mm * 1e-3
        L_expected = 4 * V_R_m3 / (PI * d_m ** 2)
        if abs(L_claimed - L_expected) > 0.05 * max(L_expected, 0.01):
            errors.append({
                "agent": "DESIGNER",
                "candidate_id": cid,
                "error_type": "LENGTH_FORMULA_ERROR",
                "description": (
                    f"L={L_claimed:.2f} m ≠ 4×V_R/(π×d²) = 4×{V_R_mL}mL/(π×({d_mm}mm)²)"
                    f" = {L_expected:.2f} m"
                ),
                "severity": "HIGH",
            })
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
#  Mixing direction check
# ═══════════════════════════════════════════════════════════════════════════════

def _check_mixing_direction(fluidics_scores: list[dict]) -> list[dict]:
    """Verify no agent proposed INCREASING d to fix r_mix > 0.20."""
    errors: list[dict] = []
    for entry in fluidics_scores:
        direction = str(entry.get("d_change_direction", "none")).lower()
        r_mix = entry.get("r_mix", 0.0)
        if direction == "increase" and float(r_mix) > 0.20:
            errors.append({
                "agent": "DR. FLUIDICS",
                "candidate_id": entry.get("candidate_id"),
                "error_type": "MIXING_DIRECTION_ERROR",
                "description": (
                    f"Candidate {entry.get('candidate_id')}: r_mix={r_mix:.3f} > 0.20 "
                    "but d_change_direction='increase'. "
                    "WRONG — t_mix ∝ d², increasing d makes mixing WORSE. "
                    "Fix: decrease d. d_fix = d × sqrt(0.15 / r_mix)."
                ),
                "severity": "CRITICAL",
            })
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
#  Threshold correctness checks
# ═══════════════════════════════════════════════════════════════════════════════

def _check_thresholds(
    fluidics_scores: list[dict],
    chemistry_scores: list[dict],
    candidates: list[dict],
) -> list[dict]:
    """Check that agents applied correct threshold values."""
    errors: list[dict] = []

    for entry in fluidics_scores:
        cid = entry.get("candidate_id")
        cand = next((c for c in candidates if c.get("id") == cid), None)
        if cand is None:
            continue

        # Re check: turbulent threshold is 2300 (not 1400 or 2000)
        Re = float(cand.get("Re", 0.0))
        flow_regime = str(entry.get("flow_regime", "")).lower()
        if Re < 2300 and "turbulent" in flow_regime:
            errors.append({
                "agent": "DR. FLUIDICS",
                "candidate_id": cid,
                "error_type": "RE_THRESHOLD_ERROR",
                "description": (
                    f"Candidate {cid}: Re={Re:.0f} < 2300 but flow_regime='{flow_regime}'. "
                    "Laminar-turbulent transition is Re=2300. "
                    "Re 1500–2300 is still laminar (mark as WARNING, not turbulent)."
                ),
                "severity": "HIGH",
            })

        # Dual-criterion mixing: BOTH r_mix > 0.20 AND Da_mass > 1.0 required
        r_mix = float(cand.get("r_mix", 0.0))
        Da_mass = float(cand.get("Da_mass", 0.0))
        dual_fail = entry.get("dual_criterion_mixing_fail", False)
        if dual_fail and not (r_mix > 0.20 and Da_mass > 1.0):
            errors.append({
                "agent": "DR. FLUIDICS",
                "candidate_id": cid,
                "error_type": "DUAL_CRITERION_ERROR",
                "description": (
                    f"Candidate {cid}: dual_criterion_mixing_fail=True but "
                    f"r_mix={r_mix:.3f}, Da_mass={Da_mass:.3f}. "
                    "Both conditions (r_mix > 0.20 AND Da_mass > 1.0) are required."
                ),
                "severity": "HIGH",
            })

    # Check A threshold for chemistry scores
    for entry in chemistry_scores:
        cid = entry.get("candidate_id")
        A = entry.get("beer_lambert_A")
        if A is None:
            continue
        verdict = str(entry.get("verdict", "")).upper()
        hard_gate = entry.get("hard_gate_failed", False)
        if float(A) > 1.5 and not hard_gate and verdict != "BLOCK":
            errors.append({
                "agent": "DR. CHEMISTRY",
                "candidate_id": cid,
                "error_type": "INNER_FILTER_THRESHOLD_ERROR",
                "description": (
                    f"Candidate {cid}: A={A:.3f} > 1.5 (hard disqualify threshold) "
                    f"but hard_gate_failed=False and verdict='{verdict}'. "
                    "A > 1.5 is a BLOCK condition."
                ),
                "severity": "CRITICAL",
            })

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
#  Scope violation detection
# ═══════════════════════════════════════════════════════════════════════════════

_SCOPE_KEYWORDS: dict[str, list[str]] = {
    "DR. CHEMISTRY": ["flow_rate", "tubing_length", "L_m", "pump_type", "pressure_drop"],
    "DR. KINETICS": ["pump_type", "tubing_ID", "tubing_material", "reactor_topology"],
    "DR. SAFETY": ["reaction_rate", "kinetics", "mechanism"],
}

def _check_scope_violations(
    chemistry_scores: list[dict],
    kinetics_scores: list[dict],
    safety_scores: list[dict],
) -> list[dict]:
    """Detect agents proposing changes outside their domain."""
    violations: list[dict] = []
    checks = [
        ("DR. CHEMISTRY", chemistry_scores),
        ("DR. KINETICS", kinetics_scores),
        ("DR. SAFETY", safety_scores),
    ]
    for agent_name, score_list in checks:
        keywords = _SCOPE_KEYWORDS.get(agent_name, [])
        for entry in score_list:
            for concern in (entry.get("concerns") or []):
                concern_lower = str(concern).lower()
                for kw in keywords:
                    if kw.lower() in concern_lower:
                        violations.append({
                            "agent": agent_name,
                            "candidate_id": entry.get("candidate_id"),
                            "violation_type": "SCOPE_VIOLATION",
                            "description": (
                                f"{agent_name} references '{kw}' in concerns — "
                                f"this is outside {agent_name}'s domain: {concern[:200]}"
                            ),
                            "severity": "LOW",
                        })
                        break
    return violations


# ═══════════════════════════════════════════════════════════════════════════════
#  BPR gas-liquid floor check
# ═══════════════════════════════════════════════════════════════════════════════

def _check_bpr_gas_liquid(
    safety_scores: list[dict],
    is_gas_liquid: bool,
) -> list[dict]:
    """Verify BPR >= 5.0 bar for all gas-liquid candidates."""
    errors: list[dict] = []
    if not is_gas_liquid:
        return errors
    for entry in safety_scores:
        bpr_current = entry.get("BPR_current_bar", 0.0)
        if float(bpr_current) < 5.0 and str(entry.get("verdict", "")).upper() != "BLOCK":
            errors.append({
                "agent": "DR. SAFETY",
                "candidate_id": entry.get("candidate_id"),
                "error_type": "GAS_LIQUID_BPR_FLOOR",
                "description": (
                    f"Gas-liquid system: BPR_current={bpr_current} bar < 5.0 bar "
                    "mandatory floor but verdict is not BLOCK. "
                    "Gas-liquid designs must have BPR >= 5.0 bar (spec: never reduce below 5 bar)."
                ),
                "severity": "CRITICAL",
            })
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
#  Disqualification recommendations from error list
# ═══════════════════════════════════════════════════════════════════════════════

def _build_disqualify_recommendations(errors: list[dict]) -> list[dict]:
    """Recommend disqualification for candidates with CRITICAL errors."""
    critical_ids: dict[int, list[str]] = {}
    for e in errors:
        if e.get("severity") in ("CRITICAL", "HIGH"):
            cid = e.get("candidate_id")
            if cid is not None:
                critical_ids.setdefault(cid, []).append(e["description"])
    return [
        {"candidate_id": cid, "reason": " | ".join(reasons[:3])}
        for cid, reasons in critical_ids.items()
    ]


def _check_flow_translation_sanity(
    candidates: list[dict],
    *,
    batch_time_min: Optional[float],
    batch_concentration_M: Optional[float],
    solvent_name: Optional[str],
    translation_policy: str,
    max_tau_to_batch_ratio: float,
) -> list[dict]:
    """Check intensification and basic flow-practicality issues."""
    errors: list[dict] = []
    solvent_lower = (solvent_name or "").lower()
    viscous_media = any(token in solvent_lower for token in ("des", "tbab", "glycol", "eg"))
    intensify_mode = (translation_policy or "").lower() == "intensify"

    for c in candidates:
        cid = c.get("id", "?")
        tau_min = float(c.get("tau_min", 0.0) or 0.0)
        Q_mL_min = float(c.get("Q_mL_min", 0.0) or 0.0)
        L_m = float(c.get("L_m", 0.0) or 0.0)
        candidate_conc = c.get("concentration_M")
        try:
            candidate_conc = float(candidate_conc) if candidate_conc is not None else None
        except (TypeError, ValueError):
            candidate_conc = None

        if intensify_mode and batch_time_min and batch_time_min > 0:
            tau_ceiling = batch_time_min * max_tau_to_batch_ratio
            if tau_min > tau_ceiling:
                errors.append({
                    "agent": "SKEPTIC",
                    "candidate_id": cid,
                    "error_type": "DEINTENSIFIED_FLOW",
                    "description": (
                        f"Candidate {cid}: tau={tau_min:.1f} min exceeds batch time ceiling "
                        f"{tau_ceiling:.1f} min under intensification policy."
                    ),
                    "severity": "HIGH",
                })

        if (
            batch_concentration_M is not None
            and batch_concentration_M > 0
            and candidate_conc is not None
            and candidate_conc < 0.5 * batch_concentration_M
        ):
            errors.append({
                "agent": "SKEPTIC",
                "candidate_id": cid,
                "error_type": "SUSPICIOUS_DILUTION",
                "description": (
                    f"Candidate {cid}: reactor concentration {candidate_conc:.3g} M is <50% of "
                    f"batch concentration {batch_concentration_M:.3g} M without explicit justification."
                ),
                "severity": "HIGH",
            })

        if viscous_media and Q_mL_min < 0.1 and L_m > 10.0:
            errors.append({
                "agent": "SKEPTIC",
                "candidate_id": cid,
                "error_type": "ULTRA_LOW_FLOW_VISCOUS_MEDIA",
                "description": (
                    f"Candidate {cid}: Q={Q_mL_min:.4f} mL/min through L={L_m:.1f} m in viscous "
                    f"medium '{solvent_name}' is likely impractical (dispersion / clogging risk)."
                ),
                "severity": "HIGH",
            })

        hard_gate_flags = [str(flag) for flag in (c.get("hard_gate_flags") or [])]
        if any("X=" in flag and "X_minimum" in flag for flag in hard_gate_flags):
            errors.append({
                "agent": "SKEPTIC",
                "candidate_id": cid,
                "error_type": "INSUFFICIENT_CONVERSION_HARD_GATE",
                "description": (
                    f"Candidate {cid}: Designer hard-gate flagged insufficient conversion "
                    f"({'; '.join(hard_gate_flags[:2])}). This cannot be rescued by "
                    "batch-like tau inflation under the intensification policy."
                ),
                "severity": "HIGH",
            })

    return errors


def _build_weak_pool_report(
    *,
    candidates: list[dict],
    pvs_by_id: dict[int, float],
    batch_time_min: Optional[float],
    intensification_mandate: dict,
    pool_metadata: dict,
) -> Optional[dict]:
    if not candidates:
        return None

    pool_size = len(candidates)
    best_pvs = max(pvs_by_id.values(), default=0.0)
    triggered: list[str] = []
    failures: list[str] = []

    if best_pvs < cfg.PVS_THRESHOLD:
        triggered.append("A")
        failures.append(
            f"Best PVS in pool is {best_pvs:.2f}, below target {cfg.PVS_THRESHOLD:.2f}."
        )

    batch_proximate_fraction = 0.0
    if batch_time_min and batch_time_min > 0:
        near_batch = [
            c for c in candidates
            if float(c.get("tau_min", 0.0) or 0.0) >= batch_time_min * cfg.BATCH_PROXIMITY_THRESHOLD
        ]
        batch_proximate_fraction = len(near_batch) / pool_size
        if batch_proximate_fraction >= 0.5:
            triggered.append("B")
            failures.append(
                f"{len(near_batch)}/{pool_size} candidates have tau >= "
                f"{cfg.BATCH_PROXIMITY_THRESHOLD:.0%} of batch time."
            )

    if (
        str(pool_metadata.get("pool_quality", "")).upper() == "DEGRADED"
        and best_pvs < cfg.PVS_THRESHOLD * 1.2
    ):
        triggered.append("C")
        failures.append("Designer marked pool_quality=DEGRADED and PVS remains weak.")

    advantage = str(intensification_mandate.get("minimum_flow_advantage") or "productivity")
    advantage_scores = []
    for c in candidates:
        report = c.get("flow_sense_report") or {}
        advantage_scores.append(float(report.get("primary_advantage_proxy_score") or 0.0))
    if advantage_scores and max(advantage_scores) <= 0.5:
        triggered.append("D")
        failures.append(
            f"No candidate strongly exploits declared primary flow advantage '{advantage}'."
        )

    if not triggered:
        return None

    target = max(float(intensification_mandate.get("tau_reduction_target") or 2.0), 1.0)
    tau_ceiling = (batch_time_min / target) if batch_time_min and batch_time_min > 0 else None
    tau_floor = (
        batch_time_min / (target * 3.0)
        if batch_time_min and batch_time_min > 0 else None
    )
    return {
        "verdict": "WEAK_POOL",
        "triggered_by": sorted(set(triggered)),
        "best_pvs_in_pool": round(best_pvs, 4),
        "target_pvs": cfg.PVS_THRESHOLD,
        "batch_proximate_fraction": round(batch_proximate_fraction, 3),
        "specific_failures": failures,
        "regeneration_instructions": {
            "tau_ceiling": round(tau_ceiling, 3) if tau_ceiling else None,
            "tau_floor": round(tau_floor, 3) if tau_floor else None,
            "required_advantage": advantage,
            "note": (
                "Regenerate with sub-batch tau values, avoid batch-boundary hugging, "
                "preserve practical flow rate, and explicitly exploit the primary flow advantage."
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_skeptic_audit(
    *,
    candidates: list[dict],
    chemistry_scores: list[dict],
    kinetics_scores: list[dict],
    fluidics_scores: list[dict],
    safety_scores: list[dict],
    is_gas_liquid: bool,
    concentration_M: float,
    pump_max_bar: float,
    batch_time_min: Optional[float] = None,
    batch_concentration_M: Optional[float] = None,
    solvent_name: Optional[str] = None,
    translation_policy: str = FLOW_TRANSLATION_POLICY,
    max_tau_to_batch_ratio: float = FLOW_MAX_TAU_TO_BATCH_RATIO,
    intensification_mandate: Optional[dict] = None,
    pool_metadata: Optional[dict] = None,
    process_value_scores: Optional[list[dict]] = None,
) -> dict:
    """Run the Stage 3 Skeptic audit.

    Returns:
      {
        "calculation_errors": [...],
        "threshold_errors": [...],
        "scope_violations": [...],
        "bpr_errors": [...],
        "all_errors": [...],           # merged list
        "disqualify_recommendations": [{"candidate_id": int, "reason": str}],
        "audit_summary": str,
        "council_may_proceed": bool,   # False if any CRITICAL errors found
      }
    """
    calc_errors = (
        _verify_beer_lambert(chemistry_scores, candidates, concentration_M) +
        _verify_v_r_equals_tau_q(candidates) +
        _verify_length_formula(candidates)
    )
    mixing_errors = _check_mixing_direction(fluidics_scores)
    threshold_errors = _check_thresholds(fluidics_scores, chemistry_scores, candidates)
    scope_violations = _check_scope_violations(
        chemistry_scores, kinetics_scores, safety_scores
    )
    bpr_errors = _check_bpr_gas_liquid(safety_scores, is_gas_liquid)
    flow_sense_errors = _check_flow_translation_sanity(
        candidates,
        batch_time_min=batch_time_min,
        batch_concentration_M=batch_concentration_M,
        solvent_name=solvent_name,
        translation_policy=translation_policy,
        max_tau_to_batch_ratio=max_tau_to_batch_ratio,
    )

    all_errors = (
        calc_errors + mixing_errors + threshold_errors + scope_violations + bpr_errors + flow_sense_errors
    )

    mandate = mandate_dict(intensification_mandate)
    pvs_rows = process_value_scores or compute_pvs_for_candidates(
        candidates,
        {
            "chemistry_scores": chemistry_scores,
            "kinetics_scores": kinetics_scores,
            "fluidics_scores": fluidics_scores,
            "safety_scores": safety_scores,
        },
    )
    pvs_by_id = {int(row.get("candidate_id", 0)): float(row.get("PVS", 0.0) or 0.0) for row in pvs_rows}
    weak_pool_report = _build_weak_pool_report(
        candidates=candidates,
        pvs_by_id=pvs_by_id,
        batch_time_min=batch_time_min,
        intensification_mandate=mandate,
        pool_metadata=pool_metadata or {},
    )
    if weak_pool_report:
        all_errors.append({
            "agent": "SKEPTIC",
            "candidate_id": None,
            "error_type": "WEAK_POOL",
            "description": "; ".join(weak_pool_report.get("specific_failures", []))[:500],
            "severity": "HIGH",
        })

    critical_count = sum(1 for e in all_errors if e.get("severity") == "CRITICAL")
    high_count = sum(1 for e in all_errors if e.get("severity") == "HIGH")

    disqualify_recs = _build_disqualify_recommendations(all_errors)
    disqualify_ids = {r["candidate_id"] for r in disqualify_recs}

    # council_may_proceed = False if ANY CRITICAL error found
    council_may_proceed = (critical_count == 0)

    # Build summary
    verdict = "WEAK_POOL" if weak_pool_report else ("BLOCK" if critical_count else ("EDIT" if high_count else "PASS"))

    if not all_errors:
        summary = (
            f"AUDIT PASSED — all {len(candidates)} candidates passed arithmetic and "
            "threshold verification. No scope violations. Council may proceed."
        )
    else:
        summary = (
            f"AUDIT: {len(all_errors)} issues found "
            f"({critical_count} CRITICAL, {high_count} HIGH, "
            f"{len(scope_violations)} scope violations). "
            f"{len(disqualify_recs)} candidates flagged for disqualification. "
        )
        if not council_may_proceed:
            summary += "CRITICAL errors found — council MUST correct before proceeding."
        elif weak_pool_report:
            summary += "WEAK_POOL — candidate pool lacks sufficient process-intensification value."
        else:
            summary += "No CRITICAL errors — council may proceed with caution."

    logger.info("    Skeptic audit: %s", summary[:200])
    for e in all_errors:
        if e.get("severity") in ("CRITICAL", "HIGH"):
            logger.info(
                "      [%s] id=%s %s: %s",
                e["severity"], e.get("candidate_id"), e.get("error_type"),
                e["description"][:100],
            )

    return {
        "calculation_errors": calc_errors,
        "threshold_errors": threshold_errors,
        "scope_violations": scope_violations,
        "bpr_errors": bpr_errors,
        "flow_sense_errors": flow_sense_errors,
        "process_value_scores": pvs_rows,
        "weak_pool_report": weak_pool_report,
        "verdict": verdict,
        "all_errors": all_errors,
        "disqualify_recommendations": disqualify_recs,
        "disqualify_ids": sorted(disqualify_ids),
        "audit_summary": summary,
        "council_may_proceed": council_may_proceed,
    }
