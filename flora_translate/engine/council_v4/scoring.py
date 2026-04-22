"""
FLORA ENGINE v4 — Stage 2: Domain scoring agents.

Four agents (Dr. Chemistry, Dr. Kinetics, Dr. Fluidics, Dr. Safety) each score
ALL surviving candidates from 0.0 to 1.0 in their domain.

Each agent must provide:
  - overall_analysis: a multi-sentence paragraph describing the full picture
  - per candidate: reasoning (detailed), score, verdict, specific fields

The rich text is used to populate the Streamlit deliberation tab.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from flora_translate.engine.llm_agents import call_llm_with_tools
from flora_translate.engine.tool_definitions import (
    CHEMISTRY_TOOLS, KINETICS_TOOLS, SAFETY_TOOLS,
    TOOL_CALCULATE_BPR_REQUIRED, TOOL_CALCULATE_MIXING_RATIO,
    TOOL_CALCULATE_PRESSURE_DROP, TOOL_CALCULATE_REYNOLDS,
    execute_tool,
)

logger = logging.getLogger("flora.engine.council_v4.scoring")

# Fluidics v4 tools include BPR (merged with hardware)
_FLUIDICS_V4_TOOLS = [
    TOOL_CALCULATE_PRESSURE_DROP,
    TOOL_CALCULATE_REYNOLDS,
    TOOL_CALCULATE_MIXING_RATIO,
    TOOL_CALCULATE_BPR_REQUIRED,
]


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_score_response(raw: str) -> tuple[str, list[dict]]:
    """Extract (overall_analysis, per_candidate_scores) from LLM output.

    Returns (overall_analysis_str, list_of_candidate_dicts).
    Handles both {"overall_analysis":..., "scores":[...]} and bare list formats.
    """
    s = raw.strip()

    def _extract_obj(text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        start = text.find("[")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        return None

    candidates_text = s
    if "```" in s:
        for part in s.split("```")[1::2]:
            obj = _extract_obj(part.lstrip("json").strip())
            if obj is not None:
                candidates_text = part.lstrip("json").strip()
                break

    obj = _extract_obj(candidates_text)
    if obj is None:
        return "", []

    if isinstance(obj, list):
        return "", obj

    if isinstance(obj, dict):
        overall = str(obj.get("overall_analysis", ""))
        scores = obj.get("scores", [])
        if not scores and "candidate_id" in obj:
            scores = [obj]
        return overall, list(scores) if isinstance(scores, list) else []

    return "", []


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clamp_score(v) -> float:
    return max(0.0, min(1.0, _safe_float(v, 0.5)))


# ═══════════════════════════════════════════════════════════════════════════════
#  DR. CHEMISTRY — scoring prompt
# ═══════════════════════════════════════════════════════════════════════════════

_CHEMISTRY_SYSTEM = """\
You are DR. CHEMISTRY in the FLORA ENGINE v4 council. You are a senior organic
and photochemistry expert. Your task is to score every surviving candidate and
write detailed, specific reasoning that would convince a PhD chemist or reviewer.

## Your scoring domains

**Chemistry domain (score 0–1):**
  - Mechanism validity: is τ long enough for the SET/EnT/HAT step? Quote the
    relevant electron-transfer kinetics or literature precedent.
  - Stream compatibility: identify any incompatible co-feeds (oxidant+reductant,
    base+acid-sensitive electrophile, catalyst+quencher in same stream).
  - Atmosphere logic: is the atmosphere correct for each stage? O₂-sensitive
    radicals must NEVER contact O₂ at any point in the reactive stage.
  - Redox feasibility: E*(catalyst) ≥ E_ox(substrate)? Quantify the margin.
  - Concentration: photoredox sweet spot 0.05–0.2 M; above 0.3 M triggers
    inner-filter competition; below 0.02 M photon dilution.
  - Byproduct risk: are there known side reactions at this τ, T, or [substrate]?

**Photonics domain (score 0–1, for photochemical reactions):**
  - Wavelength match: compare λ_LED to λ_max of the photocatalyst (±20 nm).
  - Beer-Lambert: A = ε × C × (d_mm × 0.1). ALWAYS convert d to cm.
    A < 0.3 → excellent | 0.3–0.8 → moderate | 0.8–1.5 → concerning | > 1.5 → BLOCK.
    Call the `beer_lambert` tool to compute A. Explain what fraction of photons
    reach the tube core vs. are absorbed at the wall.
  - Material: FEP/PFA/glass transparent 200–800 nm. PTFE/PEEK/SS opaque → BLOCK.
  - d selection: for high-ε catalysts (ε > 500), d ≤ 0.75 mm preferred to avoid
    inner filter. Explain the trade-off between photon penetration and pressure.

**Combined:** 0.6 × chemistry_score + 0.4 × photonics_score

## CRITICAL UNIT RULE
A = ε [M⁻¹cm⁻¹] × C [M] × (d_mm × 0.1). The path length is d in cm = d_mm × 0.1.
Never use d in mm as the path length. Always show your arithmetic.

## Verdicts
  ACCEPT:  combined ≥ 0.70, no hard blocks
  WARNING: combined 0.50–0.70 or resolvable concerns
  REVISE:  combined 0.30–0.50 or fixable chemistry problem
  BLOCK:   hard_gate_failed — opaque material, A > 1.5, redox thermodynamically
           impossible, incompatible stream logic

## Tools
Call `beer_lambert` for each photochem candidate. Show: ε used, A computed,
what this means physically. Call `check_material_compatibility` to verify
tubing/solvent/temperature compatibility. Use `check_redox_feasibility` when
redox potentials are available.

## REQUIRED OUTPUT — JSON only
```json
{
  "overall_analysis": "3–5 sentences describing the chemistry landscape across all candidates: what drives the differences in chemistry scores, what are the key photonics risks or advantages, which candidates stand out and why from a chemistry/photonics perspective.",
  "scores": [
    {
      "candidate_id": 1,
      "reasoning": "3–5 sentences of chemistry + photonics reasoning for THIS specific candidate. Cite the actual numbers: τ, d, C, ε, A, λ, material. Explain what you verified with tools. State what specific mechanism requirement is met or missed. Make the reasoning self-contained so a reviewer can follow it without looking at the table.",
      "chemistry_score": 0.85,
      "photonics_score": 0.90,
      "combined_score": 0.87,
      "verdict": "ACCEPT",
      "beer_lambert_A": 0.18,
      "epsilon_used": 20.0,
      "wavelength_match": true,
      "material_transparent": true,
      "stream_logic_valid": true,
      "atmosphere_valid": true,
      "hard_gate_failed": false,
      "blocking_issues": [],
      "concerns": ["If ε is actually 60 M⁻¹cm⁻¹ rather than 20, A rises to 0.54 — borderline moderate; verify spectral data."]
    }
  ]
}
```
Score ALL candidates. One entry per candidate_id. Reasoning must be specific to each candidate's numbers.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  DR. KINETICS — scoring prompt
# ═══════════════════════════════════════════════════════════════════════════════

_KINETICS_SYSTEM = """\
You are DR. KINETICS in the FLORA ENGINE v4 council. You are a process kinetics
expert who anchors every design decision to measured rate data and literature
residence times. Your scoring and reasoning must be quantitative throughout.

## Your scoring domain (0–1)

**Intensification factor (IF = τ_batch / τ_flow):**
  Validate the IF against reaction class ranges:
    photoredox:    4–8×  — cite whether the improvement is from photon flux,
                           mixing, or temperature. >10× needs cited analogy.
    thermal:       8–15× — improvement from heat/mass transfer enhancement.
    hydrogenation: 20–50× — gas-liquid interface area.
  IF > 20× without analogy → WARNING. IF > class upper bound → explain why.
  State the calculated IF and whether it is justified.

**τ vs. literature anchor:**
  Compare τ_proposed to τ_lit (the literature analogy residence time):
  τ ∈ [0.5×τ_lit, 1.5×τ_lit] → excellent alignment (score 1.0)
  τ < 0.5×τ_lit → explain what physical justification exists for the shorter time
  τ > 2×τ_lit → explain why longer time is needed (slow mechanism, RTD margin)

**Conversion adequacy:**
  Compute X = 1 − exp(−τ / τ_kinetics) and compare to target (default 0.85):
  X ≥ 0.85 → 1.0 | X 0.70–0.85 → 0.7 | X 0.50–0.70 → 0.4 | X < 0.50 → 0.1
  Always state X explicitly and whether it meets the target.
  RTD effect: for laminar flow at τ/τ_k < 2×, fast-moving core sees τ/2 —
  mention this when the safety margin is thin.

**Mixing-kinetics coupling:**
  τ_mixing_required = t_mix / 0.15. If τ_proposed < τ_mixing_required, kinetics
  prediction is optimistic (reaction-rate-limited assumption breaks down).
  State: tau_mixing_required_min and whether τ_proposed exceeds it.

**Integration checks:**
  Steady-state wait time: t_steady = 3 × τ (how long before reactor reaches SS).
  Productivity: mg/h at design conditions. Space-time yield (mol/L/h).

**Final τ decision rule:**
  τ_final = max(τ_kinetics, τ_mixing_required, τ_lit / 2)
  State this explicitly for each candidate.

## Verdicts
  ACCEPT:  score ≥ 0.70, X ≥ 0.85, IF valid, τ ≥ τ_mixing_required
  WARNING: score 0.50–0.70, X 0.70–0.85, or thin τ/τ_k margin (< 1.5×)
  REVISE:  X < 0.70, or IF clearly out of class range without justification
  BLOCK:   X < 0.50 (unless user approved)

## Tools
Use `estimate_residence_time` to independently verify τ_kinetics. Call it and
show the result. Use `check_redox_feasibility` if redox potential data is available.

## REQUIRED OUTPUT — JSON only
```json
{
  "overall_analysis": "3–5 sentences: what is the kinetics landscape across all candidates? Which τ values are well-supported by literature? Where is conversion at risk? What is the range of IF values and are they all justified? What should the Chief know about the kinetics dimension?",
  "scores": [
    {
      "candidate_id": 1,
      "reasoning": "3–5 sentences specific to this candidate. State τ, τ_kinetics, IF, X, t_steady. Explain the τ/τ_k safety margin and RTD risk explicitly. State τ_mixing_required and whether τ_proposed exceeds it. Explain the final τ decision rule result. Make it self-contained.",
      "kinetics_score": 0.82,
      "IF_used": 6.0,
      "IF_valid": true,
      "tau_vs_literature": "τ=45 min within [0.5×, 1.5×] τ_lit=50 min — well-anchored",
      "X_estimated": 0.94,
      "X_adequate": true,
      "tau_mixing_required_min": 2.1,
      "tau_proposed_final_min": 45.0,
      "STY": 0.235,
      "productivity_mg_h": 117.5,
      "t_steady_min": 135.0,
      "verdict": "ACCEPT",
      "concerns": ["Laminar RTD: τ/τ_k = 3.0×. Centerline fluid sees 22.5 min — still above τ_kinetics (15 min). Acceptable margin."]
    }
  ]
}
```
Score ALL candidates. Reasoning must cite specific numbers for each candidate.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  DR. FLUIDICS — scoring prompt
# ═══════════════════════════════════════════════════════════════════════════════

_FLUIDICS_SYSTEM = """\
You are DR. FLUIDICS in the FLORA ENGINE v4 council. You are a transport phenomena
and process hardware expert. Every scoring decision must be traceable to specific
numbers: Re, ΔP, r_mix, L, De, pump headroom. You also own hardware specification.

## Your scoring domain (0–1)

**Flow regime (most important):**
  Re < 100 → score 1.0. Re 100–2000 → linear scale. Re > 2300 → BLOCK.
  For coiled reactors, compute Dean number De = Re × √(d_outer / R_coil).
  De > 10 → secondary flow bonus (improves mixing, tightens RTD). Note this.
  State Re and flow regime explicitly.

**Pressure headroom:**
  ΔP / P_pump_max: < 0.20 → 1.0, 0.20–0.50 → good, 0.50–0.80 → caution,
  > 0.80 → 0.0. Compute pump_headroom_pct = (1 − ΔP/P_max) × 100.
  If headroom < 20%, explain what happens under partial blockage (Q drift +10%,
  filter fouling +20% ΔP) — does the pump stall?

**Mixing:**
  t_mix = d_m² / (4 × 10⁻⁹) seconds, where d_m is d in metres.
  r_mix = t_mix / τ_s (τ in seconds). Target: r_mix < 0.10 (safe), tolerate to 0.20.
  Dual criterion: if r_mix > 0.20 AND Da_mass > 1.0, mixing is reaction-limiting.
  MIXING DIRECTION RULE (ABSOLUTE): to fix r_mix → DECREASE d (never increase).
  Formula: d_fix = d_current × √(0.15 / r_mix_current). Round to commercial size.
  Show the calculation if a d change is recommended.

**Geometry:**
  L < 15 m → single coil, 1.0. L 15–25 m → two-coil build, penalise slightly.
  L > 25 m → BLOCK.

**Hardware specification (merged):**
  Pump: Q < 2 mL/min → syringe pump (Harvard, New Era); Q 2–10 → HPLC piston.
  Gas streams → MFC required.
  Tubing: FEP/PFA for photoreactor section. SS316 for non-photochem high-P.
  Dead volume: V_dead / Q gives τ_error. Flag if > 5% of τ.
  Coil winding: R_coil ≥ 5 × d_outer to prevent kinking.
  Blockage risk: d ≤ 0.5 mm at Q < 0.08 mL/min → flag particulate/crystal risk.

## Verdicts
  ACCEPT:  fluidics_score ≥ 0.70, no hard blocks, hardware feasible
  WARNING: score 0.50–0.70, elevated Re or ΔP, small d blockage risk
  REVISE:  dual mixing fail (need d decrease), ΔP headroom < 20%
  BLOCK:   Re > 2300, L > 25 m

## Tools
Use `calculate_pressure_drop` for what-if scenarios (Q +20%, partial blockage).
Use `calculate_reynolds` to verify Re at design and stress conditions.
Use `calculate_mixing_ratio` to verify r_mix and flag if d change needed.
Use `calculate_bpr_required` for hardware BPR sizing at operating T.
Show tool call results in your reasoning.

## REQUIRED OUTPUT — JSON only
```json
{
  "overall_analysis": "3–5 sentences: what is the fluidics landscape? Which candidates have healthy pressure and mixing margins? Where are the risks? Which pump type is needed? Are there any geometry concerns across the set? What hardware trade-offs does the Chief need to know?",
  "scores": [
    {
      "candidate_id": 1,
      "reasoning": "3–5 sentences for THIS candidate. State Re, ΔP, pump_headroom_pct, r_mix, Da_mass, L explicitly. Explain mixing adequacy and whether dual criterion is near or far. State pump type and tubing material. Note any blockage risk. If d change needed, show the d_fix formula. Make it self-contained.",
      "fluidics_score": 0.88,
      "Re": 21.3,
      "flow_regime": "laminar",
      "De": 4.2,
      "dP_bar": 0.048,
      "pump_headroom_pct": 99.7,
      "t_mix_s": 14.1,
      "r_mix": 0.052,
      "dual_criterion_mixing_fail": false,
      "d_change_direction": "none",
      "d_fix_mm": null,
      "tau_mixing_required_min": null,
      "L_m": 5.6,
      "geometry_feasible": true,
      "pump_type": "syringe pump (Q=0.55 mL/min < 2 mL/min)",
      "tubing_material": "FEP",
      "dead_volume_mL": 0.1,
      "dead_volume_impact": "V_dead/Q = 0.18 min = 0.4% of τ=45 min — negligible",
      "verdict": "ACCEPT",
      "concerns": []
    }
  ]
}
```
Score ALL candidates. Reasoning must cite actual numbers for each candidate.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  DR. SAFETY — scoring prompt
# ═══════════════════════════════════════════════════════════════════════════════

_SAFETY_SYSTEM = """\
You are DR. SAFETY in the FLORA ENGINE v4 council. Safety is Priority 2 in FLORA
(below physical feasibility only). Your job is to score every candidate, identify
every safety-relevant risk, and justify why it can or cannot be managed.
Write as if you are signing off a risk assessment — be specific and thorough.

## Your scoring domain (0–1)

**Thermal safety:**
  Da_thermal = k_ΔH × τ × (−ΔH) / (ρ × Cp × T_ref). Interpret:
  Da_thermal << 0.1 → isothermal, safe. > 0.3 → REVISE (verify heat removal).
  > 1.0 → BLOCK (runaway risk). For most photoredox, thermal hazard is low —
  say so explicitly rather than leaving it unaddressed.

**BPR adequacy:**
  Liquid-only: BPR_set = P_vap(T) + ΔP + 0.5 bar minimum safety margin.
  Gas-liquid:  BPR_set = max(5.0, P_vap + ΔP) + 2.0 bar mandatory margin.
               NEVER recommend BPR < 5.0 bar for gas-liquid. Ever.
  Call `calculate_bpr_required` and show the calculated P_min and P_recommended.
  Compare to current BPR setting. State whether BPR is adequate with margin.

**Material compatibility:**
  Call `check_material_compatibility` for the specific solvent + material + T.
  RULES:
  - FEP/PFA mandatory for photochemical reactor section. PTFE = opaque = BLOCK.
  - FEP at ≤ 25°C, pH ≤ 12: fully compatible — do not create false concerns.
  - High pressure (> 20 bar) or Alkaline (pH > 10) at T > 60°C: PFA preferred.
  - SS316: check for halide content before approving.
  - material_rationale MUST state pressure, temperature, pH, and solvent — never generic.

**Atmosphere isolation:**
  Multi-stage processes with different atmospheres: verify physical separation exists.
  State what isolation method is needed (gas-permeable membrane, T-piece, degasser).
  Single-stage processes: note this is not applicable.

**Hazard identification:**
  List every toxic, flammable, reactive, or carcinogenic species explicitly.
  Include: hazard class (CMR, flammable, irritant), handling requirements (extraction,
  inert atmosphere, spill containment), and any downstream inline quench needed.
  If a reaction produces a hazardous intermediate (diazonium, peroxide, organolithium),
  state the downstream quench requirement.

**Pressure drop safety:**
  ΔP headroom < 20% → safety risk (overpressure on restriction). Prefer > 40%.

## Verdicts
  ACCEPT:              safety_score ≥ 0.80, all gates passed
  APPROVED_WITH_CONDITIONS: score 0.60–0.80, manageable conditions explicitly stated
  REVISE:              BPR inadequate, material concern, thermal caution
  BLOCK:               Da_thermal > 1.0, gas-liquid BPR < 5 bar, opaque photoreactor,
                       incompatible material without a safe alternative

## Tools
Call `check_material_compatibility` for each candidate. Show the result.
Call `calculate_bpr_required` to compute BPR need. Show P_vapor, P_min, P_recommended.

## REQUIRED OUTPUT — JSON only
```json
{
  "overall_analysis": "3–5 sentences: what is the safety landscape? Are thermal risks present? What are the key hazard species and how are they managed? Is BPR adequate across all candidates? Any material concerns? What conditions does the lab chemist MUST follow?",
  "scores": [
    {
      "candidate_id": 1,
      "reasoning": "3–5 sentences for THIS candidate. State Da_thermal, BPR_required, BPR_current, material check result, hazard flags, atmosphere requirements. Cite tool results explicitly. Explain why the candidate is safe or what specific conditions must be met. Make it self-contained.",
      "safety_score": 0.92,
      "Da_thermal": 0.03,
      "thermal_safe": true,
      "BPR_required_bar": 1.8,
      "BPR_current_bar": 3.0,
      "BPR_adequate": true,
      "system_type": "liquid",
      "material_compatible": true,
      "material_recommendation": "FEP",
      "material_rationale": "DMF solvent at 25°C, ambient pressure, pH ~7 — FEP is fully compatible per IDEX/Swagelok compatibility data. No halides, no elevated temperature concern.",
      "atmosphere_isolation_required": false,
      "isolation_method": "N/A — single stage, inert N₂ atmosphere throughout",
      "hazard_flags": [
        "DMF — SVHC reproductive toxin (REACH); use fume hood, closed system, PPE (nitrile gloves + goggles)",
        "Ir(ppy)₃ — potential skin sensitiser; handle as potent pharmaceutical"
      ],
      "verdict": "ACCEPT",
      "blocking_issues": [],
      "conditions": ["Ensure N₂ blanket on all feed reservoirs before pressurising"]
    }
  ]
}
```
Score ALL candidates. Reasoning must cite tool results and specific numbers.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Common context builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_scoring_context(
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
) -> str:
    n = len(candidates)
    return (
        f"## Chemistry brief\n{chemistry_brief}\n\n"
        f"## User objectives\n{objectives or 'balanced'}\n\n"
        f"## Photochemical reaction: {is_photochem}\n\n"
        f"## Pump max pressure: {pump_max_bar} bar\n\n"
        f"## Surviving candidate table ({n} candidates — score ALL)\n"
        f"{table_markdown}\n\n"
        "Score every candidate. Provide detailed reasoning per candidate. "
        "Return JSON with 'overall_analysis' and 'scores' array."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-agent run functions
# ═══════════════════════════════════════════════════════════════════════════════

def _run_scoring_agent(
    agent_name: str,
    system_prompt: str,
    context: str,
    tools: list[dict],
    valid_ids: set[int],
    max_tokens: int = 4000,
) -> tuple[str, list[dict], list[dict]]:
    """Run one scoring agent with one automatic retry if zero scores are parsed.

    Returns (overall_analysis, score_list, tool_calls_log).
    """
    def _attempt(ctx: str, mt: int) -> tuple[str, list[dict], list[dict]]:
        raw, tc_log = call_llm_with_tools(
            system_prompt, ctx,
            tools=tools,
            tool_executor=execute_tool,
            max_tokens=mt,
            max_tool_turns=3,
        )
        overall_analysis, scores = _parse_score_response(raw)
        cleaned: list[dict] = []
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("candidate_id")
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            if cid not in valid_ids:
                logger.debug("%s: candidate_id=%s not in valid_ids %s — skipped",
                             agent_name, cid, valid_ids)
                continue
            entry["candidate_id"] = cid
            cleaned.append(entry)
        return overall_analysis, cleaned, tc_log

    try:
        oa, cleaned, tc_log = _attempt(context, max_tokens)

        # Retry if no candidates were scored — likely a truncated JSON response
        if not cleaned and valid_ids:
            missing_ids = sorted(valid_ids)
            logger.warning(
                "%s returned 0 scores for %d candidates %s — retrying with explicit "
                "reminder and higher token budget",
                agent_name, len(valid_ids), missing_ids,
            )
            retry_context = (
                context
                + f"\n\n⚠️ RETRY: Your previous response contained no per-candidate scores. "
                f"You MUST score ALL {len(valid_ids)} candidates with IDs {missing_ids}. "
                f"Return ONLY valid JSON with 'overall_analysis' and 'scores' array. "
                f"No prose before the JSON. Start your response with '{{'"
            )
            oa, cleaned, tc_log2 = _attempt(retry_context, max_tokens + 1500)
            tc_log = tc_log + tc_log2
            if not cleaned:
                logger.error(
                    "%s still returned 0 scores after retry — domain scores will "
                    "default to 0.500. Chief selection will be unreliable.",
                    agent_name,
                )

        return oa, cleaned, tc_log

    except Exception as e:
        logger.warning("%s scoring LLM call failed: %s", agent_name, e)
        return "", [], []


def run_chemistry_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    max_tokens: int = 3000,
) -> tuple[str, list[dict], list[dict]]:
    """Run Dr. Chemistry. Returns (overall_analysis, per_candidate_scores, tool_calls)."""
    context = _build_scoring_context(
        candidates, table_markdown, chemistry_brief, objectives,
        is_photochem, pump_max_bar,
    )
    valid_ids = {c.get("id", i + 1) for i, c in enumerate(candidates)}
    oa, scores, tc = _run_scoring_agent(
        "Dr. Chemistry", _CHEMISTRY_SYSTEM, context,
        CHEMISTRY_TOOLS, valid_ids, max_tokens,
    )
    logger.info("    Dr. Chemistry scored %d/%d candidates", len(scores), len(candidates))
    return oa, scores, tc


def run_kinetics_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    max_tokens: int = 3000,
) -> tuple[str, list[dict], list[dict]]:
    """Run Dr. Kinetics. Returns (overall_analysis, per_candidate_scores, tool_calls)."""
    context = _build_scoring_context(
        candidates, table_markdown, chemistry_brief, objectives,
        is_photochem, pump_max_bar,
    )
    valid_ids = {c.get("id", i + 1) for i, c in enumerate(candidates)}
    oa, scores, tc = _run_scoring_agent(
        "Dr. Kinetics", _KINETICS_SYSTEM, context,
        KINETICS_TOOLS, valid_ids, max_tokens,
    )
    logger.info("    Dr. Kinetics scored %d/%d candidates", len(scores), len(candidates))
    return oa, scores, tc


def run_fluidics_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    max_tokens: int = 3000,
) -> tuple[str, list[dict], list[dict]]:
    """Run Dr. Fluidics. Returns (overall_analysis, per_candidate_scores, tool_calls)."""
    context = _build_scoring_context(
        candidates, table_markdown, chemistry_brief, objectives,
        is_photochem, pump_max_bar,
    )
    valid_ids = {c.get("id", i + 1) for i, c in enumerate(candidates)}
    oa, scores, tc = _run_scoring_agent(
        "Dr. Fluidics", _FLUIDICS_SYSTEM, context,
        _FLUIDICS_V4_TOOLS, valid_ids, max_tokens,
    )
    logger.info("    Dr. Fluidics scored %d/%d candidates", len(scores), len(candidates))
    return oa, scores, tc


def run_safety_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
    max_tokens: int = 3000,
) -> tuple[str, list[dict], list[dict]]:
    """Run Dr. Safety. Returns (overall_analysis, per_candidate_scores, tool_calls)."""
    context = _build_scoring_context(
        candidates, table_markdown, chemistry_brief, objectives,
        is_photochem, pump_max_bar,
    )
    valid_ids = {c.get("id", i + 1) for i, c in enumerate(candidates)}
    oa, scores, tc = _run_scoring_agent(
        "Dr. Safety", _SAFETY_SYSTEM, context,
        SAFETY_TOOLS, valid_ids, max_tokens,
    )
    logger.info("    Dr. Safety scored %d/%d candidates", len(scores), len(candidates))
    return oa, scores, tc


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator: run all 4 agents
# ═══════════════════════════════════════════════════════════════════════════════

def run_domain_scoring(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    pump_max_bar: float,
) -> dict:
    """Run all four domain scoring agents (Stage 2).

    Returns:
        {
          "chemistry_overall": str,
          "chemistry_scores":   [per-candidate dicts],
          "kinetics_overall":   str,
          "kinetics_scores":    [per-candidate dicts],
          "fluidics_overall":   str,
          "fluidics_scores":    [per-candidate dicts],
          "safety_overall":     str,
          "safety_scores":      [per-candidate dicts],
          "tool_calls": {...},
          "blocked_by_scoring": [candidate_ids with a BLOCK verdict]
        }
    """
    chem_oa, chemistry_scores, chem_tc = run_chemistry_scoring(
        candidates=candidates, table_markdown=table_markdown,
        chemistry_brief=chemistry_brief, objectives=objectives,
        is_photochem=is_photochem, pump_max_bar=pump_max_bar,
    )
    kin_oa, kinetics_scores, kin_tc = run_kinetics_scoring(
        candidates=candidates, table_markdown=table_markdown,
        chemistry_brief=chemistry_brief, objectives=objectives,
        is_photochem=is_photochem, pump_max_bar=pump_max_bar,
    )
    flu_oa, fluidics_scores, flu_tc = run_fluidics_scoring(
        candidates=candidates, table_markdown=table_markdown,
        chemistry_brief=chemistry_brief, objectives=objectives,
        is_photochem=is_photochem, pump_max_bar=pump_max_bar,
    )
    saf_oa, safety_scores, saf_tc = run_safety_scoring(
        candidates=candidates, table_markdown=table_markdown,
        chemistry_brief=chemistry_brief, objectives=objectives,
        is_photochem=is_photochem, pump_max_bar=pump_max_bar,
    )

    blocked: set[int] = set()
    for domain_scores in (chemistry_scores, kinetics_scores, fluidics_scores, safety_scores):
        for entry in domain_scores:
            if str(entry.get("verdict", "")).upper() == "BLOCK":
                blocked.add(entry.get("candidate_id", -1))

    return {
        "chemistry_overall": chem_oa,
        "chemistry_scores": chemistry_scores,
        "kinetics_overall": kin_oa,
        "kinetics_scores": kinetics_scores,
        "fluidics_overall": flu_oa,
        "fluidics_scores": fluidics_scores,
        "safety_overall": saf_oa,
        "safety_scores": safety_scores,
        "tool_calls": {
            "chemistry": chem_tc,
            "kinetics": kin_tc,
            "fluidics": flu_tc,
            "safety": saf_tc,
        },
        "blocked_by_scoring": sorted(blocked),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Score lookup helpers (used by Chief)
# ═══════════════════════════════════════════════════════════════════════════════

def get_domain_score(score_list: list[dict], candidate_id: int, score_key: str,
                     default: float = 0.5) -> float:
    for entry in score_list:
        if entry.get("candidate_id") == candidate_id:
            v = entry.get(score_key, default)
            try:
                return max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                return default
    return default


def get_chemistry_combined(scores: list[dict], cid: int) -> float:
    return get_domain_score(scores, cid, "combined_score")


def get_kinetics_score(scores: list[dict], cid: int) -> float:
    return get_domain_score(scores, cid, "kinetics_score")


def get_fluidics_score(scores: list[dict], cid: int) -> float:
    return get_domain_score(scores, cid, "fluidics_score")


def get_safety_score(scores: list[dict], cid: int) -> float:
    return get_domain_score(scores, cid, "safety_score")


def geometry_practicality_score(candidate: dict) -> float:
    L = float(candidate.get("L_m") or 0.0)
    V_R = float(candidate.get("V_R_mL") or 0.0)
    L_score = max(0.0, min(1.0, 1.0 - (L - 10.0) / 15.0)) if L > 10.0 else 1.0
    V_score = max(0.0, min(1.0, 1.0 - (V_R - 10.0) / 40.0)) if V_R > 10.0 else 1.0
    return 0.5 * L_score + 0.5 * V_score
