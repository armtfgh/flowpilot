"""FLORA-Translate — Chemistry Reasoning Agent (Layer 1).

Runs BEFORE retrieval and hardware translation.
Analyzes the batch protocol purely from a chemistry perspective:
  - Identifies all species and their roles
  - Proposes a mechanism
  - Determines stream separation logic
  - Flags sensitivities (O2, moisture, light)
  - Generates retrieval hints for plan-aware search

Outputs a ChemistryPlan — no hardware decisions.
"""

import json
import logging
import re

import anthropic

from flora_translate.config import TRANSLATION_MODEL, PROMPTS_DIR
from flora_translate.schemas import BatchRecord, ChemistryPlan

logger = logging.getLogger("flora.chemistry_agent")


def _get_client():
    return anthropic.Anthropic()


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


CHEMISTRY_SYSTEM = """\
You are an expert organic chemist with broad expertise across photocatalysis,
thermal catalysis, electrochemistry, biocatalysis, and organocatalysis. Your
task is to analyze a batch chemistry protocol and produce a CHEMISTRY PLAN —
a pure chemical analysis with no hardware or engineering decisions.

You must:

1. IDENTIFY every chemical species and its role:
   - substrate(s), catalyst (any type), co-catalyst, base, oxidant, reductant,
     sensitizer, additives, solvent, electrode material (if electrochemistry)
   - For each: name, role, equivalents/loading, any special notes
     (light-sensitive, moisture-sensitive, air-sensitive, etc.)
   - For photocatalytic reactions: identify the photocatalyst specifically,
     its excited state type (singlet/triplet), and absorption maximum.
   - For thermal reactions: identify the metal catalyst/ligand system.
   - For enzymatic reactions: identify the enzyme, cofactors, and pH range.

2. PROPOSE the reaction mechanism step-by-step:
   - For photochem: photoexcitation, SET/EnT/HAT, radical/ionic steps, catalyst
     regeneration. Classify: oxidative vs reductive quenching vs energy transfer.
   - For thermal: coordination, oxidative addition, reductive elimination, etc.
   - For electrochemistry: electrode half-reactions, charge balance.
   - For biocatalysis: substrate binding, catalytic cycle, cofactor regeneration.
   - Identify the rate-limiting step and key intermediate(s).
   - Flag which steps (if any) require photons (is_photon_dependent).

3. DETERMINE stream separation logic:
   - Which reagents MUST be in separate streams and WHY
   - Which reagents MUST be co-dissolved and WHY
   - List any incompatible pairs (must NOT be in the same syringe)
   - Explain the mixing order and why the order matters chemically

4. FLAG sensitivities:
   - Oxygen-sensitive? (most photoredox and organometallic reactions are)
   - Moisture-sensitive? (Grignard, organolithium, many metal catalysts)
   - Light-sensitive reagents? (decompose under irradiation)
   - Temperature-sensitive? (exothermic, decomposition risk)

5. SPECIFY pre/post reactor chemistry:
   - Deoxygenation: required? method? which streams?
   - Quench: required? with what? why?

6. GENERATE retrieval hints — keywords for searching flow chemistry papers:
   - Mechanism keywords, catalyst class, reaction type
   - Similar reaction classes that would provide useful analogies

7. IF photocatalytic: RECOMMEND wavelength (match to photocatalyst absorption).
   IF NOT photocatalytic: set recommended_wavelength_nm to null.

Return ONLY valid JSON matching the ChemistryPlan schema. No prose outside JSON.
"""

CHEMISTRY_USER_TEMPLATE = """\
Analyze this batch protocol and produce a ChemistryPlan.

BATCH PROTOCOL:
{batch_json}

CRITICAL — MULTI-STEP REACTIONS:
If the protocol describes more than one synthetic step (even if it says
"one-pot" or "telescoped"), you MUST populate the "stages" array.
Each stage is a SEPARATE reactor zone in the flow process, with its own
feeds, temperature, reactor type, and inter-stage actions.

For EACH stage, specify:
- What feeds IN to this stage (new pump streams)
- What comes FROM the previous stage (the outlet stream)
- The reactor type for this stage (coil, packed_bed, chip, CSTR)
- Temperature, solvent, atmosphere for THIS stage
- What happens BETWEEN this stage and the next (quench, solvent switch,
  inline filter, heat exchange)

If it is a single-step reaction, set n_stages=1 and leave stages=[].

Return JSON:
{{
  "reaction_name": "",
  "reaction_class": "",
  "mechanism_type": "",
  "bond_formed": "",
  "bond_broken": "",

  "n_stages": 1,
  "stages": [
    {{
      "stage_number": 1,
      "stage_name": "e.g. Grignard formation",
      "reaction_type": "e.g. organometallic",
      "reactor_type": "coil | packed_bed | chip | CSTR",
      "temperature_C": null,
      "requires_light": false,
      "wavelength_nm": null,
      "feed_streams": [
        {{"stream_label": "A", "reagents": ["ArBr (1.0 equiv)"], "reasoning": "substrate feed"}}
      ],
      "inlet_from_previous": "",
      "solvent": "",
      "atmosphere": "N2",
      "oxygen_sensitive": false,
      "moisture_sensitive": false,
      "deoxygenation_required": false,
      "post_stage_action": "e.g. inline filter to remove Mg fines",
      "post_stage_reasoning": "e.g. Mg particles must not enter next reactor"
    }}
  ],

  "reagents": [
    {{"name": "", "role": "", "equiv_or_loading": "", "smiles": null, "notes": ""}}
  ],
  "mechanism_steps": [
    {{"step_number": 1, "description": "", "species_involved": [], \
"is_photon_dependent": false, "is_rate_limiting": false}}
  ],
  "key_intermediate": "",
  "excited_state_type": "",
  "energy_transfer_or_redox": "",
  "oxygen_sensitive": false,
  "moisture_sensitive": false,
  "temperature_sensitive": false,
  "light_sensitive_reagents": [],
  "stream_logic": [
    {{"stream_label": "A", "reagents": [], "reasoning": ""}}
  ],
  "mixing_order_reasoning": "",
  "incompatible_pairs": [],
  "deoxygenation_required": false,
  "deoxygenation_reasoning": "",
  "quench_required": false,
  "quench_reagent": "",
  "quench_reasoning": "",
  "retrieval_keywords": [],
  "similar_reaction_classes": [],
  "recommended_wavelength_nm": null,
  "wavelength_reasoning": "",
  "confidence_notes": ""
}}

Think carefully about the mechanism. Name every species explicitly.
For multi-step: the STAGES array is the most important part — get the
inter-stage connections right (what flows from where into what).
"""


def _normalize_plan_data(data: dict) -> dict:
    """Fix common LLM output deviations before Pydantic validation.

    The LLM sometimes returns fields in slightly different shapes
    than the schema expects. This normalizes them.
    """
    # incompatible_pairs: expected [[A, B], ...] but LLM may return
    # [{"species_1": A, "species_2": B, ...}, ...]
    pairs = data.get("incompatible_pairs", [])
    if pairs and isinstance(pairs[0], dict):
        normalized = []
        for p in pairs:
            if isinstance(p, dict):
                # Extract the two species from whatever keys the LLM used
                vals = [v for k, v in p.items() if isinstance(v, str) and k != "reason" and k != "reasoning"]
                if len(vals) >= 2:
                    normalized.append(vals[:2])
                elif vals:
                    normalized.append(vals)
            else:
                normalized.append(p)
        data["incompatible_pairs"] = normalized

    # stream_logic: expected list of dicts with stream_label/reagents/reasoning
    # but LLM may nest differently
    streams = data.get("stream_logic", [])
    if streams and isinstance(streams[0], dict):
        for s in streams:
            # reagents should be a list of strings
            r = s.get("reagents", [])
            if r and isinstance(r[0], dict):
                s["reagents"] = [
                    item.get("name", str(item)) for item in r
                ]

    # mechanism_steps: species_involved should be list[str]
    steps = data.get("mechanism_steps", [])
    for step in steps:
        if isinstance(step, dict):
            sp = step.get("species_involved", [])
            if sp and isinstance(sp[0], dict):
                step["species_involved"] = [
                    item.get("name", str(item)) for item in sp
                ]

    # stages: feed_streams may come as dicts with wrong shape
    stages = data.get("stages", [])
    for stage in stages:
        if isinstance(stage, dict):
            feeds = stage.get("feed_streams", [])
            for f in feeds:
                if isinstance(f, dict):
                    r = f.get("reagents", [])
                    if r and isinstance(r[0], dict):
                        f["reagents"] = [item.get("name", str(item)) for item in r]

    return data


class ChemistryReasoningAgent:
    """Layer 1: Pure chemistry analysis before any hardware decisions.

    If FLORA-Fundamentals rules are available, they are injected into
    the system prompt so the LLM has access to handbook-level domain
    knowledge alongside its own training data.
    """

    def analyze(self, batch_record: BatchRecord) -> ChemistryPlan:
        """Analyze a batch protocol and return a ChemistryPlan."""
        logger.info("  Chemistry Agent: Analyzing reaction mechanism and species")

        # Load fundamentals rules if available
        fundamentals_block = self._load_fundamentals(batch_record)
        system = CHEMISTRY_SYSTEM
        if fundamentals_block:
            system = system + "\n\n" + fundamentals_block
            logger.info(f"    Injected fundamentals knowledge into prompt")

        batch_json = json.dumps(
            batch_record.model_dump(exclude_none=True), indent=2
        )
        user_prompt = CHEMISTRY_USER_TEMPLATE.format(batch_json=batch_json)

        resp = _get_client().messages.create(
            model=TRANSLATION_MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )

        data = _parse_json(resp.content[0].text)
        data = _normalize_plan_data(data)
        plan = ChemistryPlan(**data)

        # Log key findings
        logger.info(f"    Reaction: {plan.reaction_name} ({plan.mechanism_type})")
        logger.info(f"    Key intermediate: {plan.key_intermediate}")
        logger.info(f"    O2-sensitive: {plan.oxygen_sensitive}")
        logger.info(f"    Streams: {len(plan.stream_logic)}")
        for sl in plan.stream_logic:
            logger.info(f"      Stream {sl.stream_label}: {sl.reagents}")
        logger.info(f"    Retrieval hints: {plan.retrieval_keywords[:5]}")

        return plan

    def _load_fundamentals(self, batch_record: BatchRecord) -> str:
        """Load relevant fundamentals rules for injection into the prompt."""
        try:
            from flora_fundamentals.knowledge_store import KnowledgeStore

            store = KnowledgeStore()
            if store.n_rules == 0:
                return ""

            rules = store.query_for_reaction(
                photocatalyst=batch_record.photocatalyst or "",
                solvent=batch_record.solvent or "",
                temperature_C=batch_record.temperature_C,
                oxygen_sensitive=batch_record.atmosphere in ("N2", "Ar") if batch_record.atmosphere else False,
            )
            return store.format_for_prompt(rules)
        except Exception as e:
            logger.debug(f"    Fundamentals not available: {e}")
            return ""
