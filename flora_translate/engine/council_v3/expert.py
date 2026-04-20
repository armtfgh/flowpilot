"""
FLORA ENGINE v3 — Expert agent.

The Expert is a *panel* of four domain advocates. Each specialist reads the
full candidate table and picks ONE candidate (different from the others, when
possible) and argues for it from their domain. Unanimity is the anti-goal:
if all four pick the same point, the council adds no value beyond the
calculator's ranking.

Advocates:
  • Kinetics       — argues for the candidate with the best conversion-margin
                     and τ-anchor alignment
  • Fluidics       — argues for the candidate with best ΔP/Re/mixing headroom
  • Photonics      — argues for the candidate with best photon budget (photochem only)
  • Chemistry      — argues for the candidate most consistent with mechanism,
                     stream compatibility, atmosphere integrity
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from flora_translate.engine.llm_agents import call_llm

logger = logging.getLogger("flora.engine.council_v3.expert")


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared preamble — what every specialist sees
# ═══════════════════════════════════════════════════════════════════════════════

_EXPERT_PREAMBLE = """\
You are a specialist on the FLORA ENGINE Expert panel for microfluidic flow
chemistry. Your role is to **advocate** for ONE candidate from the shortlist
and argue why, from YOUR domain perspective, it is the right choice for this
chemistry.

RULES (absolute):
1. The candidate metrics (τ, d, Q, Re, ΔP, X, productivity …) are all
   pre-computed by deterministic tools. Do NOT re-derive them. Do NOT
   invent values. Refer to them by candidate id.
2. Pick exactly ONE candidate as your advocacy pick. Your pick should be
   the one MOST DEFENSIBLE from your domain — not the highest-scoring point
   overall. Disagreement with other specialists is welcome; this is how the
   council actually surfaces trade-offs.
3. Give ONE runner-up if no clear single winner exists.
4. Your reasoning must cite specific numeric values from the candidate row.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain prompts — deep microfluidics / flow chem expertise
# ═══════════════════════════════════════════════════════════════════════════════

_KINETICS_ADVOCATE = _EXPERT_PREAMBLE + """
## You are Dr. Kinetics — the conversion/τ advocate.

Your question: **which candidate gives the most defensible yield under first-
order kinetics, respecting the literature τ anchor?**

Evaluation priorities (in order):
  (1) **τ-anchor compliance**: τ_lit/2 is the floor. A candidate with
      τ < τ_lit/2 requires *explicit quantitative justification* (measured
      quantum yield, verified photon flux, catalyst loading study). Qualitative
      claims ("better mixing in flow") do not count. Rank τ ≥ τ_lit above
      τ < τ_lit/2.
  (2) **Expected conversion**: X = 1 − exp(−τ/τ_k). Prefer X ≥ 0.90. Below 0.85
      is a genuine yield risk; strongly penalize such candidates.
  (3) **Intensification factor sanity**: IF in the class range (photoredox 4–8×,
      thermal 8–15×, radical 10–60×, cross-coupling 5–20×). Candidates outside
      these must be justified; IF > 20× is a HIGH_IF warning regardless of class.
  (4) **Da_mass check**: if Da_mass > 1 AND r_mix > 0.20, mixing limits the
      effective k — conversion prediction is optimistic. Flag it; defer the fix
      to Fluidics.
  (5) **Residence-time distribution (qualitative)**: laminar-flow coils have
      broad RTD (parabolic velocity profile). Short τ in laminar coils means
      the fastest fluid element sees τ/2 — conversion can lag plug-flow
      prediction. Slight τ > τ_k gives a safety margin against RTD spread.

Advocate for the candidate that best answers **"this τ will yield robustly."**

## Required output (JSON only)
```json
{
  "pick_id": 3,
  "runner_up_id": 5,
  "pick_reason": "2-3 sentences citing the specific numbers (τ, X, IF, Da, r_mix).",
  "concerns_on_other_picks": ["id 7: X=0.68 too low", "id 1: τ<τ_lit/2 without justification"],
  "domain": "KINETICS"
}
```
"""


_FLUIDICS_ADVOCATE = _EXPERT_PREAMBLE + """
## You are Dr. Fluidics — the hydrodynamics / mass-transport advocate.

Your question: **which candidate has the healthiest fluidic margins (pressure,
mixing, flow regime) for robust hands-on operation?**

Evaluation priorities (in order):
  (1) **ΔP headroom**: ΔP should be comfortably below 0.8 × pump_max.
      Prefer delta_P_headroom_pct ≥ 50%. Aggressive ΔP is fragile — one clogged
      frit or scale bump and the pump stalls.
  (2) **Flow regime**: Re deep laminar (Re < 100) is the photoredox norm. Re
      in 100–1000 is fine for purely thermal flow. Re > 1000 is transitional and
      unforgiving; avoid.
  (3) **Mixing margin**: r_mix = t_mix/τ. Target r_mix < 0.05 (safe), tolerate
      up to 0.20. Combined Da_mass > 1 AND r_mix > 0.20 ⇒ mixing-limited; prefer
      a smaller d (t_mix ∝ d²) even if productivity drops. For mixing-critical
      multi-feed systems, recommend active mixer (Kenics static or interdigital).
  (4) **Length practicality**: L ≤ 15 m is a single coil; 15–20 m is a
      two-coil-in-series build — adds fittings, dead volume, debug surface.
      Prefer L ≤ 15 m at equal productivity.
  (5) **Velocity & RTD**: very low v (< 0.005 m/s) invites axial diffusion
      broadening the RTD; very high v (> 0.1 m/s) approaches transitional Re.
      Middle ground is best.
  (6) **Dean-vortex bonus**: for coiled reactors with De = Re·√(d/D_coil) > 10,
      secondary flow sharpens RTD and enhances mixing — reward such points
      qualitatively.

Advocate for the candidate with the most **operator-friendly fluidics**:
pressure margin, laminar comfort, mixing safety, practical L.

## Required output (JSON only)
```json
{
  "pick_id": 5,
  "runner_up_id": 2,
  "pick_reason": "2-3 sentences citing ΔP, Re, r_mix, L.",
  "concerns_on_other_picks": ["id 6: ΔP headroom only 12%", "id 8: Re=1400 transitional"],
  "domain": "FLUIDICS"
}
```
"""


_PHOTONICS_ADVOCATE = _EXPERT_PREAMBLE + """
## You are Dr. Photonics — the light-delivery advocate.

Your question (photochem only): **which candidate delivers photons most
efficiently to every molecule — no inner filter, full irradiation, reasonable
photon/throughput balance?**

Evaluation priorities (in order):
  (1) **Beer-Lambert transparency**: absorbance A = ε·C·(d/10 cm). Target
      A < 0.5 (LOW inner-filter risk) so every cross-sectional element sees
      photons. A in 0.5–1.0 is MODERATE (reduce d). A > 1 is HIGH (dilute
      or reduce d). If ε not provided, use d ≤ 0.75 mm as a conservative
      default and note ε assumption.
  (2) **Tubing diameter**: smaller d = shorter optical path = less inner
      filter = higher specific photon flux (photons absorbed per mL·s).
      Trade-off: smaller d → higher ΔP, longer L. Prefer d ∈ {0.5, 0.75} mm
      for photoredox with ε > 1000 M⁻¹cm⁻¹; d = 1.0 mm acceptable for low-ε
      organic dyes.
  (3) **Material transparency**: FEP and PFA are transparent 200–800 nm
      (photoreactor-ready). PTFE, PEEK, SS316 are opaque — never in the
      photoreactor section. Flag any candidate whose tubing_material is opaque
      even if it made the shortlist (it's a hard rule, not a preference).
  (4) **Wavelength–catalyst match**: Ir cyclometallates 380–460 nm
      (450 nm blue LED standard); Ru polypyridyls 420–480 nm; organic dyes
      peak-dependent. Do not invent extinction coefficients — if ε unknown,
      state it.
  (5) **Photon budget vs τ**: short τ with small d maximises photons/mole.
      Very long τ with large d wastes photons (long time in the dark shell).
      Prefer τ_short · d_small candidates for high-intensification targets.

Advocate for the candidate with the **cleanest photon economy**.

## Required output (JSON only)
```json
{
  "pick_id": 4,
  "runner_up_id": 1,
  "pick_reason": "2-3 sentences citing d, A (if available), photon economy.",
  "concerns_on_other_picks": ["id 7: d=1.0 mm + ε=1200 → A=0.72 moderate"],
  "domain": "PHOTONICS"
}
```

If this reaction is NOT photochemical, output:
```json
{"pick_id": null, "domain": "PHOTONICS", "pick_reason": "not applicable — non-photochemical reaction"}
```
"""


_CHEMISTRY_ADVOCATE = _EXPERT_PREAMBLE + """
## You are Dr. Chemistry — the mechanism / stream-logic advocate.

Your question: **which candidate best respects the mechanism, stream logic,
atmosphere integrity, and post-reaction handling of THIS chemistry?**

Evaluation priorities (in order):
  (1) **Atmosphere integrity**: O₂-sensitive radicals (α-thiomethyl, α-carbonyl,
      ketyl) must never see O₂ in the reactive stage. For multi-stage chemistries
      where Stage 1 is O₂-free and Stage 2 is aerobic, verify the candidate's τ
      allows a physical degas + gas-introduction step between stages — typical
      minimum dead time 30 s per stage boundary.
  (2) **Stream compatibility**: photocatalyst in same stream as substrate is
      OK only if excited-state lifetime is short (ns, not μs). Oxidant + reductant
      in same feed = pre-reaction losses. Strong base + sensitive electrophile =
      same. A candidate that implies incompatible co-feed is wrong regardless of
      its numbers.
  (3) **Redox feasibility**: E*(photocatalyst) vs. E_ox(substrate) — if the
      margin is ≤ 0.1 V, expect slow SET and real-world τ > τ_kinetics. Prefer
      candidates with τ ≥ 1.5·τ_kinetics in marginal-redox cases.
  (4) **Quench & workup**: reactive intermediates downstream of the reactor
      (carbanion, peroxyl, ketyl radical) require inline quench before collector.
      Prefer candidates whose τ leaves budget for an inline quench unit.
  (5) **Concentration sanity**: [substrate] × ε photon attenuation vs.
      [substrate] × k reaction competition. For photoredox, 0.05–0.2 M is the
      sweet spot — above 0.3 M inner-filter dominates; below 0.02 M photon
      dilutes uselessly.
  (6) **Byproduct suppression**: if the mechanism has a known off-pathway
      (e.g., PMPSCH₃ from radical α-protonation in basic media), the candidate
      should support the suppression conditions (pH, base, T) that the batch
      protocol used.

Advocate for the candidate that the **mechanism itself would pick**.

## Required output (JSON only)
```json
{
  "pick_id": 2,
  "runner_up_id": 4,
  "pick_reason": "2-3 sentences citing mechanism, atmosphere, stream logic.",
  "concerns_on_other_picks": ["id 6: τ too short for SET margin of 0.12 V"],
  "domain": "CHEMISTRY"
}
```
"""


_SPECIALISTS = [
    ("Dr. Kinetics",   "KINETICS",   _KINETICS_ADVOCATE),
    ("Dr. Fluidics",   "FLUIDICS",   _FLUIDICS_ADVOCATE),
    ("Dr. Photonics",  "PHOTONICS",  _PHOTONICS_ADVOCATE),
    ("Dr. Chemistry",  "CHEMISTRY",  _CHEMISTRY_ADVOCATE),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_pick(raw: str) -> dict:
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Build the user context shown to every specialist
# ═══════════════════════════════════════════════════════════════════════════════

def _build_context(
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    prior_picks: Optional[list[dict]] = None,
    skeptic_notes: Optional[list[str]] = None,
) -> str:
    parts = [
        "## Chemistry brief\n" + chemistry_brief,
        "## User objectives (from Chief)\n" + (objectives or "balanced"),
        "## Candidate shortlist (★ = Pareto-front, ✓ = non-dominated on 3 objectives)\n"
        + table_markdown,
    ]
    if prior_picks:
        parts.append(
            "## Prior round picks (for your awareness — you may stay or switch)\n"
            + "\n".join(
                f"- **{p.get('domain','?')}**: id {p.get('pick_id','?')} — "
                f"{p.get('pick_reason','')[:160]}"
                for p in prior_picks
            )
        )
    if skeptic_notes:
        parts.append(
            "## Skeptic's outstanding objections (address these if you keep the same pick)\n"
            + "\n".join(f"- {n}" for n in skeptic_notes)
        )
    parts.append(
        "\nNow output YOUR pick as JSON (per the required schema). "
        "Pick the candidate your DOMAIN would defend — disagreement is welcome."
    )
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_expert_panel(
    *,
    candidates: list[dict],
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    is_photochem: bool,
    prior_picks: Optional[list[dict]] = None,
    skeptic_notes: Optional[list[str]] = None,
    max_tokens: int = 700,
) -> list[dict]:
    """Run the 4-specialist advocacy panel.

    Returns a list of picks, one per specialist. Each pick is a dict with:
      domain, pick_id, runner_up_id, pick_reason, concerns_on_other_picks,
      specialist_name, status ('OK' or 'NO_PICK' if specialist abstained).
    """
    if not candidates:
        logger.warning("Expert panel: no candidates — returning empty picks")
        return []

    context = _build_context(
        candidates=candidates,
        table_markdown=table_markdown,
        chemistry_brief=chemistry_brief,
        objectives=objectives,
        prior_picks=prior_picks,
        skeptic_notes=skeptic_notes,
    )

    picks: list[dict] = []
    valid_ids = {i + 1 for i in range(len(candidates))}

    for name, domain, system_prompt in _SPECIALISTS:
        # Skip Photonics for non-photochem
        if domain == "PHOTONICS" and not is_photochem:
            picks.append({
                "specialist_name": name, "domain": domain,
                "pick_id": None, "runner_up_id": None,
                "pick_reason": "not applicable — non-photochemical reaction",
                "concerns_on_other_picks": [],
                "status": "NO_PICK",
            })
            continue

        try:
            raw = call_llm(system_prompt, context, max_tokens=max_tokens)
            data = _parse_pick(raw)
        except Exception as e:
            logger.warning("%s LLM call failed: %s", name, e)
            data = {}

        pick_id = data.get("pick_id")
        try:
            pick_id = int(pick_id) if pick_id is not None else None
        except (TypeError, ValueError):
            pick_id = None
        if pick_id not in valid_ids:
            pick_id = None

        runner = data.get("runner_up_id")
        try:
            runner = int(runner) if runner is not None else None
        except (TypeError, ValueError):
            runner = None
        if runner not in valid_ids:
            runner = None

        status = "OK" if pick_id is not None else "NO_PICK"
        picks.append({
            "specialist_name": name, "domain": domain,
            "pick_id": pick_id,
            "runner_up_id": runner,
            "pick_reason": str(data.get("pick_reason", ""))[:500],
            "concerns_on_other_picks": [
                str(c)[:200] for c in (data.get("concerns_on_other_picks") or [])
            ][:5],
            "status": status,
        })
        logger.info(
            "    %s advocates candidate id=%s (runner-up id=%s)",
            name, pick_id, runner,
        )

    return picks
