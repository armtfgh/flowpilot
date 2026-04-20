"""
FLORA ENGINE v2 — Agent prompts and runners.

Design principles for model-agnosticism (GPT-4o, Claude, local models):
  1. Short, narrow system prompts — one domain per agent
  2. Triage pre-seeding — agents answer SPECIFIC questions, not blank slates
  3. Simple output format — 4 fields only; local models can handle this
  4. Authoritative values injected — agents interpret, never recompute
  5. GREEN-exit logic — agents with no issues file one-line confirmation

Output JSON (all agents, all models):
{
  "verdict":    "APPROVED" | "APPROVED_WITH_CONDITIONS" | "NEEDS_REVISION",
  "proposals":  [{"field": "...", "value": "...", "reason": "..."}],
  "conditions": ["..."],
  "summary":    "one sentence"
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from flora_translate.config import ENGINE_PROVIDER
from flora_translate.engine.llm_agents import call_llm
from flora_translate.engine.triage import TriageReport, TriageEntry
from flora_translate.schemas import (
    AgentDeliberation, FieldProposal, FlowProposal, ChemistryPlan,
)
from flora_translate.design_calculator import DesignCalculations

logger = logging.getLogger("flora.engine.agents_v2")

# ═══════════════════════════════════════════════════════════════════════════════
#  Output format — same for every agent, every model
# ═══════════════════════════════════════════════════════════════════════════════

_OUTPUT_FORMAT = """
## Required output — JSON ONLY, no text outside the JSON block

```json
{
  "verdict": "APPROVED | APPROVED_WITH_CONDITIONS | NEEDS_REVISION",
  "proposals": [
    {"field": "residence_time_min", "value": "15.0", "reason": "..."}
  ],
  "conditions": ["condition 1", "condition 2"],
  "summary": "One sentence describing your conclusion."
}
```

Rules:
- APPROVED: domain is fine, no changes needed. proposals must be [].
- APPROVED_WITH_CONDITIONS: design works but requires specific lab precautions.
  proposals must be []. Put precautions in conditions[].
- NEEDS_REVISION: a parameter MUST change. proposals must have ≥1 entry.
  Allowed fields: residence_time_min, flow_rate_mL_min, temperature_C,
  concentration_M, BPR_bar, tubing_material, tubing_ID_mm, wavelength_nm,
  deoxygenation_method, mixer_type.
  Note: flow_rate_mL_min is the ONLY way to fix a geometry (L > 20 m) problem.
  Do NOT change residence_time_min to fix geometry — that is Dr. Kinetics' domain.
- All values in proposals must be strings (e.g. "15.0" not 15.0).
- summary must be exactly one sentence.
"""

# Compact output format for local models — same structure, shorter description
_OUTPUT_FORMAT_COMPACT = """
Output JSON ONLY:
```json
{"verdict":"APPROVED|APPROVED_WITH_CONDITIONS|NEEDS_REVISION","proposals":[{"field":"field_name","value":"value","reason":"why"}],"conditions":["note"],"summary":"one sentence"}
```
APPROVED=no change, APPROVED_WITH_CONDITIONS=precautions only (proposals=[]), NEEDS_REVISION=must have proposals.
"""


def _fmt() -> str:
    """Return the appropriate output format block for the active ENGINE_PROVIDER."""
    return _OUTPUT_FORMAT_COMPACT if ENGINE_PROVIDER == "ollama" else _OUTPUT_FORMAT

# ═══════════════════════════════════════════════════════════════════════════════
#  Shared preamble — injected into every agent
# ═══════════════════════════════════════════════════════════════════════════════

_PREAMBLE = """\
You are a specialist on the FLORA ENGINE council for flow chemistry process design.

## CRITICAL RULES
1. The triage report contains PRE-COMPUTED tool values. They are authoritative.
   Do NOT re-derive Re, ΔP, Da, τ, BPR, or any value already in the triage.
2. Do NOT invent numbers. If a value is not provided, say "not available".
3. Answer ONLY the directed question in your domain. Other domains are not your concern.
4. If your domain triage status is GREEN, output APPROVED with a one-sentence confirmation.
   Do not add warnings, conditions, or proposals for a GREEN domain.
5. Proposals must reference the exact FlowProposal field name and a specific numeric value.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  Agent system prompts — narrow domain, short
# ═══════════════════════════════════════════════════════════════════════════════

def _kinetics_system(triage: TriageReport) -> str:
    entry = next((e for e in triage.entries if e.domain == "KINETICS"), None)
    status_note = f"Your domain triage status: {entry.status}" if entry else ""
    return _PREAMBLE + f"""
You are **Dr. Kinetics** — τ decision-maker.

{status_note}

## Your role: Dr. Kinetics — τ decision-maker

## MANDATORY τ Decision Tree (execute in this order)
All values below come from your triage tool_results — do NOT re-derive them.

Step 1 — IF validation
  Validate IF_used against class range:
  - Photoredox: 4–8× typical, up to 10× with citation; IF > 20× = HIGH_IF_WARNING
  - Thermal: 8–15×; Hydrogenation: 20–50×; Radical: 10–60×; Cross-coupling: 5–20×
  If IF outside acceptable range → propose τ correction to τ_batch/IF_class_center.

Step 2 — τ_lit anchor (THE KEY CHECK — do not skip)
  Read tau_lit from tool_results. This is the literature anchor:
    - If tau_analogy_min is present: τ_lit = tau_analogy_min (real paper used this τ in flow)
    - If no analogies: τ_lit = tau_class_min (class default; photoredox class IF = 6×)
  tau_lit_anchor_check = True means τ_current is already < τ_lit / 2.

  RULE: If tau_lit_anchor_check is True (or τ_current < τ_lit / 2.0) → MUST flag.
  Provide an explicit physical justification for running 2× faster than the literature value
  (e.g., higher photon flux with known quantum yield data, better mixing measured at this d,
  higher catalyst loading confirmed to scale rate). Qualitative claims do not count.
  If no quantitative justification exists → propose residence_time_min = τ_lit / 2.0.

  Photoredox class IF check: IF > 10× requires explicit analogy citation with matching mechanism.
  IF > 20× = HIGH_IF_WARNING regardless of class. Class IF for photoredox = 6× (4–8× range).

Step 3 — Conversion check
  For first-order approximation: X = 1 − exp(−τ_current / τ_kinetics)
  where τ_kinetics = τ_batch / IF_used (already computed in tool_results)
  If X < 0.90 → propose τ increase to achieve X ≥ 0.90

Step 4 — Final τ_proposed
  τ_proposed = max(τ_kinetics, τ_lit / 2.0)
  Only output residence_time_min proposal if τ_proposed > τ_current.

## Notes on Da_mass
  Da_mass > 1 AND r_mix > 0.20 simultaneously: flag, but let Dr. Fluidics handle mixing.
  Dr. Fluidics will pass τ_mixing_required — take the max in Phase 2.
  Da_mass > 1 alone with r_mix < 0.20: mixing completes within τ, no action needed.
""" + _fmt()


def _fluidics_system(triage: TriageReport) -> str:
    entry = next((e for e in triage.entries if e.domain == "FLUIDICS"), None)
    status_note = f"Your domain triage status: {entry.status}" if entry else ""
    return _PREAMBLE + f"""
You are **Dr. Fluidics** — pressure drop, flow regime, mixing specialist.

{status_note}

## Your role: Dr. Fluidics — pressure drop, flow regime, mixing

## Step 0 — Geometry check (CHECK FIRST — most common failure mode)
  Read L_m, V_R_mL, geometry_ok from tool_results.
  Bench limit: L ≤ 20 m, V_R ≤ 25 mL for a single coil reactor.

  If geometry_ok = False:
    Root cause: L = 4·τ·Q / (π·d²). L is too long because τ × Q is too large.
    Fix: reduce Q to Q_sweet_mL_min (pre-computed in tool_results).
    Q_sweet is the flow rate that gives L = 20 m at current τ and d.
    This is the FIRST proposal to make. After Q is corrected, all other
    constraints (Re, ΔP, mixing) must be rechecked in the next round.

    Also report the sweet_spots table from tool_results — give the top-3
    feasible (d, Q) options so the chemist can choose throughput vs. d size.

    Propose: flow_rate_mL_min = Q_sweet_mL_min

  If geometry_ok = True AND L_m > 15 m: APPROVED_WITH_CONDITIONS with a note
    "Reactor is 15–20 m — at the upper end of practical range. Two 10 m coils
     in series will be needed."

## Step 1 — Flow regime check
  Re from tool_results:
  - Re < 100: deep laminar (normal for microreactors)
  - Re 100–2300: laminar (monitor)
  - Re > 2300: TURBULENT — propose increase d OR decrease Q

## Step 2 — Pressure headroom
  Check ΔP < 0.8 × pump_max_bar.
  If exceeded → propose larger d (ΔP ∝ d⁻⁴) OR shorter L via lower τ.

## Step 3 — Dual-criterion mixing test
  BOTH of these must fail before action is required:
  - Da_mass > 1.0
  - mixing_ratio > 0.20
  If only ONE fails → APPROVED_WITH_CONDITIONS with a note. NO d change.

## Step 4 — d direction rules (UNAMBIGUOUS — this is physics, not opinion)

| Problem                          | Correct direction          | Reason           |
|----------------------------------|----------------------------|------------------|
| Da_mass > 1 AND r_mix > 0.20    | DECREASE d                 | t_mix ∝ d²       |
| Re > 2300                        | INCREASE d OR decrease Q   | Re ∝ v × d       |
| ΔP > 0.8 × pump_max             | INCREASE d OR decrease L   | ΔP ∝ d⁻⁴         |
| Inner filter (from Photonics)    | DECREASE d                 | optical path     |

CRITICAL: NEVER increase d to fix mixing (it makes mixing worse).
CRITICAL: NEVER decrease d to fix pressure (it makes ΔP 16× worse per halving).

## Step 5 — d_fix calculation (when mixing fails)
The pre-computed d_fix_mm and tau_mixing_required_min are in your tool_results.
d_fix_mm is ALREADY rounded UP to the nearest standard commercial tubing size
(standard_tubing_IDs_mm list is in tool_results). Do NOT change it — rounding down
would put r_mix above the threshold. d_fix_mm_exact shows the raw math result.

When mixing fails (Da_mass > 1 AND mixing_ratio > 0.20), output TWO proposals:
  1. tubing_ID_mm = d_fix_mm (standard commercial size from tool_results)
  2. residence_time_min = tau_mixing_required_min (from tool_results)
     (This τ_mixing_required is passed to Dr. Kinetics; the orchestrator takes the max.)

Never propose a non-standard tubing ID — the chemist cannot purchase 0.671 mm tubing.

## Step 6 — Dean number (mitigating factor)
  De from tool_results. If De > 10: report as secondary flow that helps mixing.
  This may mean mixing is better than pure diffusion model predicts.

## ΔP recheck after d change
  If proposing d decrease: ΔP ∝ d⁻⁴ — a 20% d decrease → 2× ΔP increase.
  Always verify ΔP at proposed d_fix still within pump_max.
""" + _fmt()


def _safety_system(triage: TriageReport) -> str:
    entry = next((e for e in triage.entries if e.domain == "SAFETY"), None)
    status_note = f"Your domain triage status: {entry.status}" if entry else ""
    is_gl = "GAS-LIQUID SYSTEM: BPR >= 5 bar minimum (hard rule). O2 introduction requires physical isolation from Stage 1." if triage.is_gas_liquid else ""
    is_ms = "MULTI-STAGE: verify atmosphere isolation between stages." if triage.is_multistage else ""
    return _PREAMBLE + f"""
You are **Dr. Safety** — the council's safety gatekeeper with BLOCKING authority.

{status_note}
{is_gl}
{is_ms}

## Your role: Dr. Safety — BLOCKING authority

## Calculation sequence
1. Thermal check: Da_thermal from tool_results
   - Da_thermal < 0.1: SAFE
   - Da_thermal 0.1–0.3: WARNING
   - Da_thermal > 0.3: REVISE
   - Da_thermal > 1.0: BLOCK (runaway risk)

2. BPR check: compare BPR_current_bar vs BPR_min_required_bar (from tool_results)
   - If BPR_current < BPR_min_required → REVISE, propose BPR_bar = BPR_recommended (add 1.5 bar margin)
   - Gas-liquid systems: BPR >= 5 bar minimum (hard rule)

3. Material check (in this order for photochem):
   a. Photochemical reactor section → FEP or PFA ONLY (PTFE is opaque)
   b. Aqueous base? → FEP compatible up to pH 13 at < 60°C
   c. Strong oxidant? → check compatibility
   d. T > 150°C? → PFA preferred over FEP

4. Atmosphere integrity: O2-sensitive stage without deoxygenation_method → BLOCK

5. Gas-liquid rule: any design with O2, H2, CO2 → BPR mandatory >= 5 bar

## Blocking rule
  You are the ONLY agent with BLOCK authority. A BLOCK = NEEDS_REVISION with
  "BLOCKING: [reason]" as the first item in conditions[].
  The priority ladder cannot override your BLOCK.
""" + _fmt()


def _chemistry_system(triage: TriageReport) -> str:
    entry = next((e for e in triage.entries if e.domain == "CHEMISTRY"), None)
    status_note = f"Your domain triage status: {entry.status}" if entry else ""
    is_ms = (
        f"MULTI-STAGE ({triage.n_stages} stages): verify that each stage's atmosphere, "
        f"feeds, and stream routing are correctly specified in the proposal."
        if triage.is_multistage else ""
    )
    return _PREAMBLE + f"""
You are **Dr. Chemistry** — mechanism fidelity, stream logic, and photochemistry specialist.

{status_note}
{is_ms}

## Calculation sequence (execute in order)
1. List every species: name, role, equivalents
2. Identify incompatible pairs:
   - Radical + O2 → quenching (must separate)
   - Oxidant + reductant in same stream → side reactions
   - Photocatalyst excited state + quencher → separate streams
3. Stream assignment: photocatalyst and substrate may share if short-lived excited state
4. Atmosphere per stage: O2-sensitive → inert + degassing; O2-required stages AFTER O2-sensitive
5. Redox matching: verify E*[catalyst] >= E_ox[substrate] for oxidative quenching
   (qualitative only — state if insufficient data)
6. Post-reactor: if quench_required is True, verify post_reactor_steps is populated

## Decision rules
  Any incompatible pair sharing a stream → NEEDS_REVISION (mandatory separation)
  Atmosphere wrong for mechanism → BLOCK (output NEEDS_REVISION with "BLOCKING:" prefix)
  Redox mismatch → BLOCK
  Missing quench → NEEDS_REVISION
""" + _fmt()


def _photonics_system(triage: TriageReport) -> str:
    entry = next((e for e in triage.entries if e.domain == "PHOTONICS"), None)
    status_note = f"Your domain triage status: {entry.status}" if entry else ""
    return _PREAMBLE + f"""
You are **Dr. Photonics** — light delivery specialist (active only for photochemical reactions).

{status_note}

## Beer-Lambert MANDATORY
  A = ε × C × (d × 0.1)   (d in mm → cm for path length)
  T = 10^(-A)

  If ε not available: state "ε not available" and classify risk qualitatively from C and d alone.

  Risk thresholds:
  - A < 0.5 (T > 32%): LOW — no action needed
  - A 0.5–1.0: MODERATE — monitor, consider d reduction
  - A > 1.0: HIGH — REVISE: reduce d or reduce C

  d direction for inner filter: DECREASE d (reduces optical path length).
  d_max = 2.303 / (ε × C × 10) cm (if ε known) — maximum d for A < 1.

## Recalculation mandate
  If d or C changed in a prior round: you MUST re-run Beer-Lambert with new d/C.
  A cached APPROVED from a prior round is NEVER valid after d or C changes.

## Wavelength–catalyst match
  Ir(III) complexes: 380–460 nm (blue LED 450 nm is standard).
  Ru(II) complexes: 420–480 nm. Organic dyes vary — state qualitative match only.
  Do NOT invent extinction coefficients. State "ε not available — assess qualitatively."

## Material transparency
  FEP: transparent 200–8000 nm — APPROVED for photoreactor section
  PFA: transparent — APPROVED for photoreactor section
  PTFE: OPAQUE (white) — NEVER for photoreactor section
  SS316, PEEK: OPAQUE — NEVER for photoreactor section

## Reactor geometry
  FEP/PFA coil around LED strip is standard for photoredox.
  Chip reactors for short τ and high photon flux.
  Falling film for gas-liquid + light.

- Propose tubing_ID_mm reduction only if inner filter risk is MODERATE or HIGH.
""" + _fmt()


def _hardware_system(triage: TriageReport) -> str:
    return _PREAMBLE + """
You are **Dr. Hardware** — physical equipment specification specialist.
You run AFTER the design is confirmed. You translate parameters into lab equipment.

## Your domain
- Pump type per stream:
    syringe pump: 0.001–5 mL/min, high precision, no pulsation (preferred for photoredox)
    HPLC piston: 0.01–50 mL/min, pulsation at low end
    peristaltic: for corrosive, 0.5–50 mL/min, tube wear issue
    MFC (mass flow controller): for gas streams
- Tubing: FEP standard for photochemistry. PFA for higher T/P. SS for corrosive.
  Standard OD: 1/16" (1.59 mm) — ID varies 0.25–2.0 mm.
- Mixer: T-mixer for Re < 100. Y-mixer for slightly higher Re.
  Static mixer (e.g., Kenics) for viscous fluids. Avoid dead volume.
- Reactor: FEP coil around LED strip for photoredox. Chip for fast reactions.
- BPR: specify commercially available setting (e.g., Equilibar, IDEX 0–250 psi adjustable).
- Dead volume: connector dead volume ~ 5–15 µL per fitting. Flag if > 5% of V_R.
- LED: specify wavelength (nm), power (W), geometry (coil wrap or plate).
- For multi-stage: specify which tube carries effluent from Stage 1 into Stage 2 mixer.

Output your equipment list in the "conditions" field as a structured list.
APPROVED_WITH_CONDITIONS is your expected verdict — you are specifying, not blocking.
""" + _fmt()


def _integration_system(triage: TriageReport) -> str:
    return _PREAMBLE + """
You are **Dr. Integration** — unit operation sequence and process procedure specialist.
You run AFTER Hardware. You write the complete ordered procedure.

## Your domain
- Unit operation sequence: pumps → [degasser] → mixer → reactor → [BPR] → [quench] → collector
  For multi-stage: Stage 1 effluent → [gas introduction] → Stage 2 mixer → Stage 2 reactor → ...
- Steady-state wait: t_ss = 3 × τ before sample collection begins.
- STY and productivity: calculate from C × Q × X × MW / V_R.
  Estimate MW ≈ 200–400 g/mol if not given.
- Scale-out (NOT scale-up): how many parallel trains for 10× throughput.
  State as advisory note — never propose design parameter changes.
- Procedure steps: priming, atmosphere purging, steady-state wait, collection, shutdown.
- Inline analytics: UV-Vis monitoring is feasible for photoredox; FTIR for gas-generating.

Write the unit op sequence in the "conditions" field as an ordered list.
APPROVED_WITH_CONDITIONS is your expected verdict.
""" + _fmt()


def _failure_system() -> str:
    return """
You are **Dr. Failure** — a devil's advocate who stress-tests the final design.
You are ADVISORY ONLY. You never propose design parameter changes.

## Your task
Identify the top 3 most likely practical failure modes for this specific design in the lab.
For each failure mode:
  - What breaks, and why
  - How probable (high/medium/low)
  - One specific mitigation (hardware fix, procedure step, or monitoring point)

## Special attention for:
- Tube blockage: at small ID, partial blockage → exponential ΔP rise
- Gas-liquid: what happens if O₂ supply fails mid-run?
- Degasser: is rated capacity sufficient at the operating flow rate?
- Photocatalyst degradation: does extended irradiation bleach the catalyst?
- Multi-stage atmosphere isolation: what happens if Stage 2 O₂ back-diffuses to Stage 1?

Output format:
```json
{
  "verdict": "APPROVED_WITH_CONDITIONS",
  "proposals": [],
  "conditions": [
    "FAILURE 1: [description] | Probability: [H/M/L] | Mitigation: [action]",
    "FAILURE 2: ...",
    "FAILURE 3: ..."
  ],
  "summary": "Top 3 failure modes identified — all operational, none requiring design change."
}
```
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON parser — robust against markdown fences
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_agent_json(raw: str, agent_name: str) -> dict:
    """Extract JSON from agent response with 4 fallback strategies."""
    text = raw.strip()
    if not text:
        return {"verdict": "APPROVED", "proposals": [], "conditions": [],
                "summary": f"{agent_name} returned empty response — treating as APPROVED."}

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for i, p in enumerate(parts):
            if i % 2 == 1:  # inside fences
                candidate = p.lstrip("json").strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace matching
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

    # Fallback: extract verdict from text
    verdict = "APPROVED"
    for word in ("NEEDS_REVISION", "APPROVED_WITH_CONDITIONS", "APPROVED"):
        if word in text.upper():
            verdict = word
            break

    logger.warning("JSON parse failed for %s — using fallback verdict '%s'", agent_name, verdict)
    return {
        "verdict": verdict,
        "proposals": [],
        "conditions": ["Could not parse full JSON — see raw response"],
        "summary": text[:300],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Context builder — what each agent sees
# ═══════════════════════════════════════════════════════════════════════════════

def _build_context(
    domain: str,
    proposal: FlowProposal,
    chemistry_plan: ChemistryPlan | None,
    calc: DesignCalculations,
    triage: TriageReport,
    prior_findings: list[tuple[str, dict]] | None = None,
) -> str:
    """Build the user-message context for an agent call.

    When ENGINE_PROVIDER == "ollama", produces a compact context (~700–900 tokens)
    so the full system+user prompt fits within Ollama's default num_ctx of 2048.
    Cloud providers get the full context.
    """
    is_local = (ENGINE_PROVIDER == "ollama")
    sections = []

    if is_local:
        # ── Compact path for local models (fits within 2048-token context) ──
        entry = next((e for e in triage.entries if e.domain == domain), None)
        if entry:
            # Keep only the most critical tool_results (drop sweet_spots, verbose lists)
            _COMPACT_KEYS = {
                "KINETICS": ["residence_time_min", "intensification_factor", "kinetics_method",
                             "Da_mass", "tau_lit", "tau_lit_anchor_check", "IF_in_range"],
                "FLUIDICS": ["Re", "flow_regime", "delta_P_bar", "pump_max_bar",
                             "mixing_ratio", "Da_mass", "geometry_ok",
                             "L_m", "V_R_mL", "Q_sweet_mL_min", "d_fix_mm", "tau_mixing_required_min"],
                "SAFETY":   ["BPR_current_bar", "BPR_min_required_bar", "BPR_required",
                             "material_compatible", "material_concern", "thermal_safe", "is_gas_liquid"],
                "CHEMISTRY": ["mechanism", "n_stages", "O2_sensitive",
                              "deoxygenation_specified", "quench_required", "quench_specified"],
                "PHOTONICS": ["wavelength_nm", "concentration_M", "tubing_ID_mm", "light_required"],
            }
            keep = _COMPACT_KEYS.get(domain, list(entry.tool_results.keys()))
            compact_results = {k: v for k, v in entry.tool_results.items() if k in keep}
            lines = [
                f"## TRIAGE — {domain}: {entry.status}",
                f"Summary: {entry.message}",
                f"Question: {entry.directed_question}",
                "Tool values: " + json.dumps(compact_results, default=str),
            ]
            sections.append("\n".join(lines))

        # One-liner triage summary
        flagged = ", ".join(f"{e.domain}({e.status})" for e in triage.entries
                            if e.status in ("RED", "YELLOW")) or "none"
        sections.append(
            f"## Design: {triage.design_center}\n"
            f"Flagged: {flagged} | "
            f"Gas-liquid: {triage.is_gas_liquid} | Multi-stage: {triage.is_multistage}"
        )

        # Minimal FlowProposal (core numeric fields only)
        core_fields = ["residence_time_min", "flow_rate_mL_min", "temperature_C",
                       "concentration_M", "BPR_bar", "tubing_material", "tubing_ID_mm",
                       "reactor_volume_mL", "wavelength_nm", "deoxygenation_method"]
        prop_dict = {k: v for k, v in proposal.model_dump().items()
                     if k in core_fields and v is not None}
        sections.append("## Proposal: " + json.dumps(prop_dict, default=str))

        # Chemistry one-liner
        if chemistry_plan:
            atm = ""
            if chemistry_plan.stages:
                atm = " | atm: " + "/".join(s.atmosphere for s in chemistry_plan.stages)
            sections.append(
                f"## Chemistry: {chemistry_plan.reaction_class} ({chemistry_plan.mechanism_type})"
                f" | O2_sensitive={chemistry_plan.oxygen_sensitive}"
                f" | quench={chemistry_plan.quench_required}{atm}"
            )

    else:
        # ── Full context path for cloud models ──
        sections.append(triage.agent_block(domain))
        sections.append(triage.full_block())

        key_fields = [
            "residence_time_min", "flow_rate_mL_min", "temperature_C",
            "concentration_M", "BPR_bar", "reactor_type", "tubing_material",
            "tubing_ID_mm", "reactor_volume_mL", "wavelength_nm",
            "deoxygenation_method", "mixer_type",
        ]
        prop_dict = {k: v for k, v in proposal.model_dump().items()
                     if k in key_fields and v is not None}
        if proposal.streams:
            prop_dict["streams"] = [
                {"label": s.stream_label, "contents": s.contents, "flow_rate": s.flow_rate_mL_min}
                for s in proposal.streams
            ]
        if proposal.pre_reactor_steps:
            prop_dict["pre_reactor_steps"] = proposal.pre_reactor_steps
        if proposal.post_reactor_steps:
            prop_dict["post_reactor_steps"] = proposal.post_reactor_steps

        sections.append(
            "## Current FlowProposal\n```json\n"
            + json.dumps(prop_dict, indent=2, default=str)
            + "\n```"
        )

        if chemistry_plan:
            cp_dict = {
                "reaction_class": chemistry_plan.reaction_class,
                "mechanism_type": chemistry_plan.mechanism_type,
                "n_stages": chemistry_plan.n_stages,
                "oxygen_sensitive": chemistry_plan.oxygen_sensitive,
                "quench_required": chemistry_plan.quench_required,
            }
            if chemistry_plan.stages:
                cp_dict["stages"] = [
                    {
                        "stage": s.stage_number,
                        "name": s.stage_name,
                        "atmosphere": s.atmosphere,
                        "requires_light": s.requires_light,
                        "wavelength_nm": s.wavelength_nm,
                        "oxygen_sensitive": s.oxygen_sensitive,
                        "post_stage_action": s.post_stage_action,
                    }
                    for s in chemistry_plan.stages
                ]
            sections.append(
                "## ChemistryPlan\n```json\n"
                + json.dumps(cp_dict, indent=2, default=str)
                + "\n```"
            )

    # Prior findings (both paths — but truncated for local)
    if prior_findings:
        if is_local:
            sections.append("## Prior round: " + " | ".join(
                f"{name}:{f.get('verdict','?')}" for name, f in prior_findings
            ))
        else:
            sections.append("## Other Agents' Findings (for your information)")
            for agent_name, finding in prior_findings:
                sections.append(
                    f"**{agent_name}**: {finding.get('verdict', '?')} — {finding.get('summary', '')}"
                )

    # For local models: reinforce JSON output format as the final thing the model reads.
    # This is the last line before generation — placement matters for thinking models.
    if is_local:
        sections.append(
            'NOW OUTPUT JSON ONLY (no thinking, no explanation):\n'
            '{"verdict":"APPROVED","proposals":[],"conditions":[],"summary":"one sentence"}'
        )

    return "\n\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════════
#  Generic agent runner
# ═══════════════════════════════════════════════════════════════════════════════

def _run_agent(
    agent_name: str,
    display_name: str,
    system_prompt: str,
    domain: str,
    proposal: FlowProposal,
    chemistry_plan: ChemistryPlan | None,
    calc: DesignCalculations,
    triage: TriageReport,
    max_tokens: int = 1024,
    prior_findings: list[tuple[str, dict]] | None = None,
) -> AgentDeliberation:
    """Run a single agent call and return a structured AgentDeliberation."""
    # Thinking models (gemma4) consume tokens on internal reasoning before writing content.
    # Complex agent system prompts can still trigger partial thinking despite think=False —
    # give 3× budget so reasoning + JSON output both fit within num_ctx.
    if ENGINE_PROVIDER == "ollama":
        max_tokens = max_tokens * 3

    # GREEN-exit: skip the LLM call entirely
    entry = next((e for e in triage.entries if e.domain == domain), None)
    if entry and entry.status == "GREEN" and not prior_findings:
        logger.info("    %s: GREEN — skipping LLM call (domain confirmed)", display_name)
        return AgentDeliberation(
            agent=agent_name,
            agent_display_name=display_name,
            round=1,
            chain_of_thought=f"Domain {domain} is in the green zone per triage. {entry.message}",
            findings=[entry.message],
            proposals=[],
            concerns=[],
            status="ACCEPT",
            had_error=False,
        )

    context = _build_context(domain, proposal, chemistry_plan, calc, triage, prior_findings)
    logger.info("    Running %s...", display_name)

    raw = ""
    try:
        raw = call_llm(system_prompt, context, max_tokens=max_tokens)
        data = _parse_agent_json(raw, display_name)

        verdict = data.get("verdict", "APPROVED")
        # Map verdict to legacy status
        status = {
            "APPROVED": "ACCEPT",
            "APPROVED_WITH_CONDITIONS": "WARNING",
            "NEEDS_REVISION": "REVISE",
        }.get(verdict, "ACCEPT")

        raw_proposals = data.get("proposals", [])
        parsed_proposals = [
            FieldProposal(
                field=str(p.get("field", "")),
                value=str(p.get("value", "")),
                reason=str(p.get("reason", "")),
            )
            for p in raw_proposals
            if isinstance(p, dict) and p.get("field")
        ]

        conditions = data.get("conditions", [])
        summary = data.get("summary", "")

        return AgentDeliberation(
            agent=agent_name,
            agent_display_name=display_name,
            round=1,
            chain_of_thought=summary,
            findings=conditions[:5] if conditions else [summary],
            proposals=parsed_proposals,
            concerns=[c for c in conditions if "concern" in c.lower() or "risk" in c.lower()][:3],
            status=status,
            had_error=False,
        )

    except Exception as e:
        logger.error("Agent %s failed: %s", display_name, e)
        if raw:
            logger.debug("Raw: %s", raw[:300])
        return AgentDeliberation(
            agent=agent_name,
            agent_display_name=display_name,
            round=1,
            chain_of_thought=f"Agent error: {e}",
            status="WARNING",
            had_error=True,
            concerns=[f"Agent error — analysis incomplete: {e}"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Public agent runner functions
# ═══════════════════════════════════════════════════════════════════════════════

def run_kinetics_v2(proposal, chemistry_plan, calc, triage, prior=None) -> AgentDeliberation:
    return _run_agent("KineticsV2", "Dr. Kinetics", _kinetics_system(triage),
                      "KINETICS", proposal, chemistry_plan, calc, triage,
                      max_tokens=800, prior_findings=prior)


def run_fluidics_v2(proposal, chemistry_plan, calc, triage, prior=None) -> AgentDeliberation:
    return _run_agent("FluidicsV2", "Dr. Fluidics", _fluidics_system(triage),
                      "FLUIDICS", proposal, chemistry_plan, calc, triage,
                      max_tokens=800, prior_findings=prior)


def run_safety_v2(proposal, chemistry_plan, calc, triage, prior=None) -> AgentDeliberation:
    return _run_agent("SafetyV2", "Dr. Safety", _safety_system(triage),
                      "SAFETY", proposal, chemistry_plan, calc, triage,
                      max_tokens=900, prior_findings=prior)


def run_chemistry_v2(proposal, chemistry_plan, calc, triage, prior=None) -> AgentDeliberation:
    return _run_agent("ChemistryV2", "Dr. Chemistry", _chemistry_system(triage),
                      "CHEMISTRY", proposal, chemistry_plan, calc, triage,
                      max_tokens=900, prior_findings=prior)


def run_photonics_v2(proposal, chemistry_plan, calc, triage, prior=None) -> AgentDeliberation:
    return _run_agent("PhotonicsV2", "Dr. Photonics", _photonics_system(triage),
                      "PHOTONICS", proposal, chemistry_plan, calc, triage,
                      max_tokens=700, prior_findings=prior)


def run_hardware_v2(proposal, chemistry_plan, calc, triage) -> AgentDeliberation:
    return _run_agent("HardwareV2", "Dr. Hardware", _hardware_system(triage),
                      "HARDWARE", proposal, chemistry_plan, calc, triage,
                      max_tokens=1200)


def run_integration_v2(proposal, chemistry_plan, calc, triage,
                        hardware_findings: AgentDeliberation | None = None) -> AgentDeliberation:
    prior = [("Dr. Hardware", {
        "verdict": "APPROVED_WITH_CONDITIONS",
        "summary": hardware_findings.chain_of_thought if hardware_findings else "",
    })] if hardware_findings else None
    return _run_agent("IntegrationV2", "Dr. Integration", _integration_system(triage),
                      "INTEGRATION", proposal, chemistry_plan, calc, triage,
                      max_tokens=1200, prior_findings=prior)


def run_failure_v2(proposal, chemistry_plan, calc, triage) -> AgentDeliberation:
    return _run_agent("FailureV2", "Dr. Failure", _failure_system(),
                      "FAILURE", proposal, chemistry_plan, calc, triage,
                      max_tokens=800)
