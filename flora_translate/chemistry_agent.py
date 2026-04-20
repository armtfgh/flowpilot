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

from flora_translate.config import MODEL_CHEMISTRY_AGENT as CHEMISTRY_MODEL, CHEMISTRY_MAX_TOKENS, PROMPTS_DIR
from flora_translate.schemas import BatchRecord, ChemistryPlan

logger = logging.getLogger("flora.chemistry_agent")


def _get_client():
    return anthropic.Anthropic()


def _parse_json_from_tagged(text: str) -> dict:
    """Extract JSON from between <JSON> ... </JSON> tags, or fall back to raw JSON."""
    if "<JSON>" in text:
        start = text.index("<JSON>") + len("<JSON>")
        end   = text.index("</JSON>") if "</JSON>" in text else len(text)
        text  = text[start:end].strip()
    return _parse_json(text)


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
You are an expert organic chemist and flow chemistry engineer with deep expertise
across photocatalysis, thermal catalysis, electrochemistry, biocatalysis, and
organocatalysis. Your task: analyze a batch chemistry protocol and produce a
rigorous CHEMISTRY PLAN grounded in first principles.

## Output structure

First, write a BRIEF NOTES block (5-8 bullet points maximum, concise):
<NOTES>
• Chemistry type: [what type, what drives it]
• Rate-limiting step in batch: [what, why]
• Flow advantage: [specific quantitative reason, e.g. "100× better photon delivery in 1mm ID tube"]
• Key incompatible pairs: [species A + species B → why incompatible]
• Critical sensitivity: [O2/moisture/light/temperature + reason]
• Retrieval strategy: [2-3 most distinctive keywords for search]
</NOTES>

Then output the complete JSON (no markdown fences):
<JSON>
{ ... ChemistryPlan fields ... }
</JSON>

Rules:
- The NOTES must be bullet points only — no paragraphs, no lengthy explanations.
- The JSON must be complete, valid, and consistent with NOTES.
- Name every species explicitly. Never use "reagent 1" or "catalyst".
- For photochem: recommended_wavelength_nm must match photocatalyst λ_abs.
- For non-photochem: set recommended_wavelength_nm to null.
- confidence_notes: note anything you are uncertain about.
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

    # ── Coerce None → "" for str fields with empty-string defaults ───────────
    # Pydantic v2 rejects None for plain `str` fields even with a "" default.
    _str_fields = [
        "excited_state_type", "energy_transfer_or_redox",
        "key_intermediate", "reaction_name", "reaction_class",
        "mechanism_type", "bond_formed", "phase_regime",
        "mixing_order_reasoning", "photocatalyst_loading", "scale_note",
    ]
    for field in _str_fields:
        if data.get(field) is None:
            data[field] = ""

    # ── Coerce "" / non-numeric strings → None for Optional[float/int] fields ─
    # The LLM sometimes returns "" or "N/A" for numeric fields that have no
    # value (e.g. recommended_wavelength_nm for a thermal reaction).
    _numeric_fields = [
        "recommended_wavelength_nm", "n_stages",
    ]
    for field in _numeric_fields:
        val = data.get(field)
        if val is None:
            continue
        if isinstance(val, str):
            stripped = val.strip()
            try:
                data[field] = float(stripped) if stripped else None
            except ValueError:
                data[field] = None

    return data


class ChemistryReasoningAgent:
    """Layer 1: Pure chemistry analysis before any hardware decisions.

    If FLORA-Fundamentals rules are available, they are injected into
    the system prompt so the LLM has access to handbook-level domain
    knowledge alongside its own training data.
    """

    def analyze(self, batch_record: BatchRecord) -> ChemistryPlan:
        """Analyze a batch protocol and return a ChemistryPlan."""
        logger.info(f"  Chemistry Agent: Analyzing with {CHEMISTRY_MODEL}")

        # Load fundamentals rules if available
        fundamentals_block = self._load_fundamentals(batch_record)
        system = CHEMISTRY_SYSTEM
        if fundamentals_block:
            system = system + "\n\n## FLOW CHEMISTRY HANDBOOK RULES\n" + fundamentals_block
            logger.info("    Injected fundamentals knowledge into prompt")

        batch_json = json.dumps(
            batch_record.model_dump(exclude_none=True), indent=2
        )
        user_prompt = CHEMISTRY_USER_TEMPLATE.format(batch_json=batch_json)

        raw_text = self._call_with_retry(system, user_prompt)

        # Extract reasoning block for logging
        reasoning = self._extract_reasoning(raw_text)
        if reasoning:
            logger.info(f"    Reasoning summary: {reasoning[:300]}...")

        data = _parse_json_from_tagged(raw_text)
        data = _normalize_plan_data(data)
        plan = ChemistryPlan(**data)
        plan._reasoning = reasoning  # attach for downstream use (not in schema)

        logger.info(f"    Reaction: {plan.reaction_name} ({plan.mechanism_type})")
        logger.info(f"    Key intermediate: {plan.key_intermediate}")
        logger.info(f"    O2-sensitive: {plan.oxygen_sensitive}")
        logger.info(f"    Streams: {len(plan.stream_logic)}")
        for sl in plan.stream_logic:
            logger.info(f"      Stream {sl.stream_label}: {sl.reagents}")
        logger.info(f"    Retrieval hints: {plan.retrieval_keywords[:5]}")

        return plan

    def _call_with_retry(self, system: str, user_prompt: str) -> str:
        """Call the model. Warn if truncated but do not retry — 8192 is the hard cap."""
        resp = _get_client().messages.create(
            model=CHEMISTRY_MODEL,
            max_tokens=CHEMISTRY_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if resp.stop_reason == "max_tokens":
            logger.warning("    Chemistry Agent output hit token limit — JSON may be incomplete")
        return resp.content[0].text

    def _extract_reasoning(self, text: str) -> str:
        """Extract the NOTES section (between <NOTES> tags)."""
        if "<NOTES>" in text and "</NOTES>" in text:
            start = text.index("<NOTES>") + len("<NOTES>")
            end   = text.index("</NOTES>")
            return text[start:end].strip()
        if "<JSON>" in text:
            return text[:text.index("<JSON>")].strip()
        return ""

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
