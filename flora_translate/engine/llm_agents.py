"""
ENGINE — LLM-powered specialist agents for multi-agent deliberation.

Each agent is a domain expert in flow chemistry, backed by:
  1. A detailed system prompt with deep domain knowledge
  2. Relevant fundamentals rules from the handbook knowledge base
  3. The full DesignCalculator results + proposal + chemistry plan
  4. Ability to PROPOSE concrete alternatives, not just accept/reject
  5. Awareness of other agents' findings (in Round 2+)

Agents:
  - KineticsSpecialist     — reaction kinetics, residence time, conversion
  - FluidicsSpecialist     — pressure drop, mixing, flow regime, mass transfer
  - SafetySpecialist       — thermal safety, material compatibility, runaway risk
  - ChemistrySpecialist    — mechanism fidelity, stream logic, wavelength, quench
  - IntegrationSpecialist  — overall process architecture, unit ops sequence, scale-up

All agents return AgentDeliberation objects with full chain-of-thought.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

import anthropic

from flora_translate.config import TRANSLATION_MODEL
from flora_translate.schemas import AgentDeliberation, FieldProposal, FlowProposal, ChemistryPlan

logger = logging.getLogger("flora.engine.llm_agents")

_CLIENT = None

def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED PREAMBLE — injected into every agent's system prompt
# ═══════════════════════════════════════════════════════════════════════════════

_SHARED_PREAMBLE = """\
## CRITICAL RULES — READ BEFORE ANYTHING ELSE

1. **The DesignCalculator has already computed all physics.** The values in
   "Engineering Calculations" are AUTHORITATIVE. DO NOT re-derive Re, ΔP, Da,
   τ, V_R, or any other value the calculator already provides. Instead, READ
   the provided values and INTERPRET what they mean for your domain.

2. **DO NOT invent numbers.** If a value is not in the context (rate constants,
   absorption coefficients, activation energies), say "not available — needs
   experimental measurement" instead of fabricating a number. Hallucinated
   precision is worse than acknowledged uncertainty.

3. **Proposals must be concrete field changes.** When you propose a change,
   specify the exact FlowProposal field name and target value. No vague
   suggestions. Use only these modifiable fields:
   residence_time_min, flow_rate_mL_min, temperature_C, concentration_M,
   BPR_bar, tubing_material, tubing_ID_mm, reactor_volume_mL, wavelength_nm,
   deoxygenation_method, mixer_type.
   (reactor_volume_mL is auto-computed as τ×Q — propose τ or Q instead.)

4. **Status rules:**
   - ACCEPT = design is adequate in your domain, no changes needed
   - WARNING = minor concerns but workable, no field changes proposed
   - REVISE = you are proposing specific field changes that MUST be applied

5. **If the batch protocol specifies a value** (e.g., wavelength, temperature,
   solvent), that is experimental evidence. Do not recommend changing it unless
   there is a clear engineering reason.
"""

_PROPOSAL_FORMAT = """
## Output — JSON only, no prose outside the JSON block
{
  "chain_of_thought": "Your reasoning — reference calculator values by name, don't re-derive them",
  "values_referenced": ["Re = 7.75 (from calculator)", "Da_mass = 3.99 (from calculator)", ...],
  "findings": ["Laminar flow confirmed", "Pressure drop is within limits", ...],
  "proposals": [
    {"field": "tubing_ID_mm", "value": "0.75", "reason": "Reduce diffusion path for mixing-limited regime"},
    {"field": "residence_time_min", "value": "15.0", "reason": "Allow complete conversion at Da = 3"}
  ],
  "concerns": ["Mass transfer limited — Da_mass > 1", ...],
  "status": "ACCEPT | WARNING | REVISE",
  "rules_cited": ["rule_id_1", ...]
}

IMPORTANT: "proposals" must be an array of objects with "field", "value", "reason".
If status is ACCEPT or WARNING, proposals should be an empty array [].
If status is REVISE, proposals must contain at least one field change.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  System prompts — each agent is a master of flow chemistry in their domain
# ═══════════════════════════════════════════════════════════════════════════════

KINETICS_SYSTEM = _SHARED_PREAMBLE + """\
You are **Dr. Kinetics**, a reaction engineering specialist on the FLORA ENGINE
council. Your domain: residence time adequacy, conversion feasibility, and
kinetic regime assessment.

## What you assess (using CALCULATOR VALUES, not your own re-derivation)
1. **Residence time**: Read τ (residence_time_min), kinetics_method, and
   intensification_factor from the calculator. Is the IF reasonable for this
   reaction class (10-100× for photochem, 5-20× for thermal)?
2. **CRITICAL — NO CIRCULAR VALIDATION**: The calculator derived τ from the
   intensification factor. DO NOT back-calculate k = -ln(1-X)/τ and then
   use that k to "confirm" τ — this is circular and adds zero information.
   If you have no independent k from the literature, say so. Evaluate the IF
   directly: is 48× reasonable for photoredox? Yes — literature supports 20-100×.
   That IS the validation. No k needed.
3. **Damköhler analysis**: Read damkohler_mass from the calculator.
   - Da >> 1 means kinetically controlled (reaction faster than diffusion)
   - Da < 1 means mixing-limited (diffusion controls)
   - ALSO check mixing_time_s / residence_time_s — if this ratio is small
     (< 0.2), diffusion completes well within the residence time even if Da > 1.
     In that case, mixing limitation is NOT a practical problem.
4. **Multi-step processes**: If chemistry_plan has multiple stages, assess
   whether a single τ is appropriate. Only propose splitting if there is a
   mechanistic reason (e.g., one stage requires O₂, one requires anaerobics).
5. **Temperature effects**: If T_flow ≠ T_batch, the calculator already applied
   Arrhenius correction. Read arrhenius_correction — do not re-derive it.
""" + _PROPOSAL_FORMAT

FLUIDICS_SYSTEM = _SHARED_PREAMBLE + """\
You are **Dr. Fluidics**, a fluid dynamics and transport specialist on the
FLORA ENGINE council. Your domain: pressure drop, flow regime, mixing quality,
and mass transfer assessment.

## What you assess (using CALCULATOR VALUES, not your own re-derivation)
1. **Flow regime**: Read reynolds_number and flow_regime from the calculator.
   - Re < 2300 = laminar (expected for microfluidics)
   - Comment on whether this Re provides adequate convective transport.
2. **Pressure drop**: Read pressure_drop_bar and pump_max_bar.
   - Is ΔP < 80% of pump capacity? If not, propose larger tubing ID.
3. **Mixing**: Read damkohler_mass, mixing_time_s from the calculator.
   - If Da_mass > 1 AND mixing_time_s > τ/3, mixing is genuinely limiting.
   - If mixing_time_s << τ, mixing completes in time — not a real limitation.
   - For Da_mass > 1 with long τ: note it but don't overstate — diffusion
     completes within residence time in most microfluidic systems.
4. **Tubing geometry**: Read tubing_ID_mm and tubing_length_m.
   - Is length reasonable? (< 15 m for coil, < 3 m for chip)
   - For photochem: would smaller ID improve light penetration?
""" + _PROPOSAL_FORMAT

SAFETY_SYSTEM = _SHARED_PREAMBLE + """\
You are **Dr. Safety**, a process safety specialist on the FLORA ENGINE council.
Your domain: thermal safety, material compatibility, pressure integrity, and
reagent hazards.

## What you assess (using CALCULATOR VALUES, not your own re-derivation)
1. **Thermal safety**: Read heat_generation_W, heat_removal_W, thermal_damkohler,
   and thermal_safe from the calculator.
   - Da_thermal < 1 = safe. Da_thermal > 1 = thermal runaway risk.
   - If thermal_safe is True, acknowledge it — don't re-derive.
2. **Material compatibility**: Check tubing_material vs solvent and temperature.
   - FEP: max 200°C, swells in THF/toluene above 60°C
   - SS: corrodes in HCl, good for high T/P
   - PFA: similar to FEP but better chemical resistance
3. **Pressure**: Read bpr_required, bpr_pressure_bar, pressure_drop_bar.
   - Is BPR_bar ≥ bpr_pressure_bar from calculator?
   - Is there adequate margin between operating P and tubing burst P?
4. **Reagent hazards**: Identify any hazardous reagents (azides, peroxides,
   pyrophorics, toxic gases) and note handling requirements.
   DO NOT invent hazard classifications — only flag what is clearly hazardous.
""" + _PROPOSAL_FORMAT

CHEMISTRY_SYSTEM = _SHARED_PREAMBLE + """\
You are **Dr. Chemistry**, a synthetic organic chemistry and photochemistry
specialist on the FLORA ENGINE council. Your domain: mechanism fidelity,
stream separation logic, and photocatalyst compatibility.

## What you assess
1. **Stream separation**: Are incompatible reagents in separate streams?
   Check the ChemistryPlan's incompatible_pairs against the proposal's streams.
2. **Wavelength**: If the BATCH PROTOCOL specifies a wavelength, that is
   experimental evidence that it works — do NOT recommend changing it unless
   there is a clear incompatibility with the photocatalyst. Common catalyst/LED
   pairings (Ir complexes + 450 nm, Ru complexes + 450 nm, organic dyes +
   green/blue) are well-established — do not second-guess them.
3. **Deoxygenation**: If chemistry_plan says oxygen_sensitive=True, verify
   that deoxygenation_method is specified in the proposal.
4. **Quench**: If chemistry_plan says quench_required=True, verify a quench
   step exists in post_reactor_steps.
5. **Concentration**: Is the proposed concentration reasonable for the mechanism?
   For photochemistry, high concentrations (> 0.3M) can cause inner filter
   effects in small ID tubing. But DO NOT invent extinction coefficients —
   just flag the risk qualitatively if concentration is high.

## IMPORTANT: Do not hallucinate spectroscopic data
You do NOT have access to absorption spectra, extinction coefficients, or
quantum yields. Do not invent these numbers. If you need to comment on light
absorption, state it qualitatively ("high concentration in narrow tubing may
reduce light penetration") without fabricating ε values.
""" + _PROPOSAL_FORMAT

INTEGRATION_SYSTEM = _SHARED_PREAMBLE + """\
You are **Dr. Process**, a process engineering and integration specialist on
the FLORA ENGINE council. Your domain: unit operation coherence, throughput
assessment, and scale-up considerations.

## What you assess (using CALCULATOR VALUES, not your own re-derivation)
1. **Consistency check**: Read from calculator — verify τ×Q=V_R from the
   proposal values. If inconsistent, flag it (but the calculator already
   enforces this, so it should be fine).
2. **Throughput**: Calculate productivity = C × Q × X × MW.
   This is one of the few calculations you SHOULD do, since it depends on
   the specific substrate molecular weight (estimate MW ≈ 200-400 for typical
   organic substrates if not specified).
3. **Unit operation sequence**: Is the order logical?
   pumps → [degas] → mixer → reactor → [BPR] → [quench] → collector
4. **Scale-up**: At this Q, how many parallel reactors for 10× throughput?
   Is the reactor length manageable for coiling?
5. **Process analytical technology**: Would inline monitoring (FTIR, UV-Vis)
   add value for this reaction? This is a suggestion, not a REVISE.
""" + _PROPOSAL_FORMAT

# Agent display names
AGENT_NAMES = {
    "KineticsSpecialist": "Dr. Kinetics",
    "FluidicsSpecialist": "Dr. Fluidics",
    "SafetySpecialist": "Dr. Safety",
    "ChemistrySpecialist": "Dr. Chemistry",
    "IntegrationSpecialist": "Dr. Process",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Fundamentals rule loader — category-filtered for each agent
# ═══════════════════════════════════════════════════════════════════════════════

_AGENT_RULE_CATEGORIES = {
    "KineticsSpecialist": ["residence_time", "reactor_design", "catalyst", "scale_up"],
    "FluidicsSpecialist": ["mixing", "pressure", "mass_transfer", "reactor_design"],
    "SafetySpecialist": ["safety", "materials", "heat_transfer", "temperature", "pressure"],
    "ChemistrySpecialist": ["photochemistry", "catalyst", "solvent", "concentration",
                            "reactor_design"],
    "IntegrationSpecialist": ["reactor_design", "scale_up", "general"],
}

_MAX_RULES_PER_AGENT = 25  # keep prompt size reasonable


def _load_rules_for_agent(agent_name: str) -> str:
    """Load top fundamentals rules relevant to this agent's domain."""
    try:
        from flora_fundamentals.knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        if store.n_rules == 0:
            return ""

        categories = _AGENT_RULE_CATEGORIES.get(agent_name, [])
        rules = []
        for cat in categories:
            cat_rules = [r for r in store.rules if r.category == cat]
            # Prefer hard_rules
            cat_rules.sort(key=lambda r: (r.severity != "hard_rule", -(r.confidence or 0)))
            rules.extend(cat_rules[:8])

        rules = rules[:_MAX_RULES_PER_AGENT]
        if not rules:
            return ""

        lines = ["## HANDBOOK RULES (use these as authoritative knowledge)"]
        for r in rules:
            lines.append(
                f"- [{r.category}] {r.condition}: "
                f"{r.recommendation} "
                f"({(r.reasoning or '')[:100]})"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.debug("Could not load fundamentals for %s: %s", agent_name, e)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Context builder — what each agent sees
# ═══════════════════════════════════════════════════════════════════════════════

def _build_agent_context(
    proposal: FlowProposal,
    chemistry_plan: ChemistryPlan | None,
    calculations_dict: dict | None,
    other_agents_findings: list[AgentDeliberation] | None = None,
) -> str:
    """Build the user message containing all context for an agent."""
    sections = []

    # Proposal
    sections.append(
        "## Current FlowProposal\n```json\n"
        + json.dumps(proposal.model_dump(), indent=2, default=str)
        + "\n```"
    )

    # Chemistry plan
    if chemistry_plan:
        sections.append(
            "## ChemistryPlan\n```json\n"
            + json.dumps(chemistry_plan.model_dump(exclude_none=True), indent=2, default=str)
            + "\n```"
        )

    # Design calculations — AUTHORITATIVE, agents must not re-derive
    if calculations_dict:
        key_calc = {
            k: v for k, v in calculations_dict.items()
            if k not in ("steps",) and v is not None
        }
        sections.append(
            "## AUTHORITATIVE Engineering Calculations (from DesignCalculator)\n"
            "**These values are computed from first principles. DO NOT re-derive them.**\n"
            "**Reference them by name in your reasoning.**\n"
            "```json\n"
            + json.dumps(key_calc, indent=2, default=str)
            + "\n```"
        )

    # Other agents' findings (for Round 2+)
    if other_agents_findings:
        sections.append("## Other Agents' Findings from Previous Round")
        for delib in other_agents_findings:
            agent_name = AGENT_NAMES.get(delib.agent, delib.agent)
            sections.append(f"### {agent_name} ({delib.status})")
            sections.append(delib.chain_of_thought[:500])
            if delib.proposals:
                prop_strs = []
                for p in delib.proposals:
                    if hasattr(p, 'field') and p.field:
                        prop_strs.append(f"{p.field} → {p.value} ({p.reason})")
                    elif hasattr(p, 'reason'):
                        prop_strs.append(p.reason)
                if prop_strs:
                    sections.append("**Proposals:** " + " | ".join(prop_strs))
            if delib.concerns:
                sections.append("**Concerns:** " + " | ".join(delib.concerns))

    return "\n\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON parser — robust against markdown fences, truncation, partial output
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_agent_json(raw: str) -> dict:
    """Extract JSON from agent response, with multiple fallback strategies."""
    text = raw.strip()
    if not text:
        raise ValueError("Empty response from agent")

    # Strategy 1: strip markdown fences
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    if "```" in text:
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    # Strategy 2: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find the outermost JSON object via brace matching
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

    # Strategy 4: minimal status extraction from plain text
    status = "WARNING"
    for word in ("REVISE", "ACCEPT", "WARNING"):
        if word in text.upper():
            status = word
            break
    logger.warning("JSON parse failed — extracting status '%s' from text", status)
    return {
        "chain_of_thought": text[:800],
        "values_referenced": [],
        "findings": [],
        "proposals": [],
        "concerns": ["Could not parse full JSON — see chain_of_thought"],
        "status": status,
        "rules_cited": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Generic agent runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent(
    agent_name: str,
    system_prompt: str,
    proposal: FlowProposal,
    chemistry_plan: ChemistryPlan | None,
    calculations_dict: dict | None,
    round_num: int = 1,
    other_agents_findings: list[AgentDeliberation] | None = None,
    own_prior_deliberation: AgentDeliberation | None = None,
) -> AgentDeliberation:
    """Run a single LLM-powered agent and return its deliberation."""

    # Build system prompt with fundamentals rules
    rules_block = _load_rules_for_agent(agent_name)
    full_system = system_prompt
    if rules_block:
        full_system += "\n\n" + rules_block

    # Build context (user message)
    context = _build_agent_context(
        proposal, chemistry_plan, calculations_dict, other_agents_findings
    )

    if other_agents_findings:
        # Inject the agent's own prior concerns — forces genuine re-evaluation
        own_concern_block = ""
        if own_prior_deliberation and own_prior_deliberation.concerns:
            concern_lines = "\n".join(
                f"  - {c}" for c in own_prior_deliberation.concerns
            )
            own_concern_block = (
                f"\n\n## Your Round 1 Concerns — you MUST address each one\n"
                f"In Round 1 you raised these concerns:\n{concern_lines}\n\n"
                "For EACH concern above, you must now:\n"
                "  (a) RESOLVE it — 'This concern is resolved because [specific reason]'\n"
                "  (b) DEFER it — 'This is an acknowledged risk; it does not require a design change'\n"
                "  (c) ESCALATE it — 'This still requires a design change: [structured proposal]'\n"
                "You CANNOT copy your Round 1 concerns verbatim. "
                "Doing so means you are not genuinely deliberating."
            )
        context += (
            "\n\n## Your task for Round " + str(round_num) + "\n"
            "Review the other agents' findings above and your own Round 1 analysis. "
            "Where you agree with other agents, say so briefly. "
            "Where you disagree, explain why — referencing CALCULATOR values. "
            "Update your proposals based on cross-agent findings. "
            "Remember: proposals must be structured as {\"field\": \"...\", \"value\": \"...\", \"reason\": \"...\"}."
            + own_concern_block
        )
    else:
        context += (
            "\n\n## Your task\n"
            "Analyze this flow design from your specialist perspective. "
            "Reference specific values from the Engineering Calculations — do NOT re-derive them. "
            "If you propose changes, use the structured format: {\"field\": \"...\", \"value\": \"...\", \"reason\": \"...\"}."
        )

    display_name = AGENT_NAMES.get(agent_name, agent_name)
    logger.info("    Running %s (Round %d)...", display_name, round_num)

    raw = ""
    try:
        resp = _get_client().messages.create(
            model=TRANSLATION_MODEL,
            max_tokens=2048,
            system=full_system,
            messages=[{"role": "user", "content": context}],
        )
        raw = resp.content[0].text.strip()
        data = _parse_agent_json(raw)

        # Parse structured proposals — handle both new format and fallback
        raw_proposals = data.get("proposals", [])
        parsed_proposals: list[FieldProposal] = []
        for p in raw_proposals:
            if isinstance(p, dict) and "field" in p and p.get("field"):
                parsed_proposals.append(FieldProposal(
                    field=str(p.get("field", "")),
                    value=str(p.get("value", "")),
                    reason=str(p.get("reason", "")),
                ))
            elif isinstance(p, str) and p.strip():
                parsed_proposals.append(FieldProposal(field="", value="", reason=p))

        return AgentDeliberation(
            agent=agent_name,
            agent_display_name=display_name,
            round=round_num,
            chain_of_thought=data.get("chain_of_thought", ""),
            values_referenced=data.get("values_referenced", []),
            findings=data.get("findings", []),
            proposals=parsed_proposals,
            concerns=data.get("concerns", []),
            status=data.get("status", "ACCEPT"),
            had_error=False,
            references_to_agents=data.get("references_to_agents", []),
            rules_cited=data.get("rules_cited", []),
        )
    except Exception as e:
        logger.error("Agent %s failed (Round %d): %s", agent_name, round_num, e)
        if raw:
            logger.debug("Raw response was: %s", raw[:400])
        return AgentDeliberation(
            agent=agent_name,
            agent_display_name=display_name,
            round=round_num,
            chain_of_thought=f"⚠️ Agent encountered an error and could not complete its analysis: {e}",
            status="WARNING",
            had_error=True,         # ← blocks convergence
            concerns=[f"Agent error — analysis incomplete: {e}"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience functions for each specialist
# ═══════════════════════════════════════════════════════════════════════════════

def run_kinetics(proposal, chemistry_plan, calculations_dict,
                 round_num=1, others=None, prior=None) -> AgentDeliberation:
    return run_agent("KineticsSpecialist", KINETICS_SYSTEM,
                     proposal, chemistry_plan, calculations_dict, round_num, others,
                     own_prior_deliberation=prior)

def run_fluidics(proposal, chemistry_plan, calculations_dict,
                 round_num=1, others=None, prior=None) -> AgentDeliberation:
    return run_agent("FluidicsSpecialist", FLUIDICS_SYSTEM,
                     proposal, chemistry_plan, calculations_dict, round_num, others,
                     own_prior_deliberation=prior)

def run_safety(proposal, chemistry_plan, calculations_dict,
               round_num=1, others=None, prior=None) -> AgentDeliberation:
    return run_agent("SafetySpecialist", SAFETY_SYSTEM,
                     proposal, chemistry_plan, calculations_dict, round_num, others,
                     own_prior_deliberation=prior)

def run_chemistry(proposal, chemistry_plan, calculations_dict,
                  round_num=1, others=None, prior=None) -> AgentDeliberation:
    return run_agent("ChemistrySpecialist", CHEMISTRY_SYSTEM,
                     proposal, chemistry_plan, calculations_dict, round_num, others,
                     own_prior_deliberation=prior)

def run_integration(proposal, chemistry_plan, calculations_dict,
                    round_num=1, others=None, prior=None) -> AgentDeliberation:
    return run_agent("IntegrationSpecialist", INTEGRATION_SYSTEM,
                     proposal, chemistry_plan, calculations_dict, round_num, others,
                     own_prior_deliberation=prior)
