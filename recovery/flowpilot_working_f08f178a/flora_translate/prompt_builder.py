"""FLORA-Translate — Prompt builder: assembles the translation LLM prompt."""

import json
from pathlib import Path

from flora_translate.config import PROMPTS_DIR
from flora_translate.schemas import BatchRecord, ChemistryPlan


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def _format_analogy(idx: int, result: dict) -> str:
    """Format a single analogy block for the prompt."""
    meta = result.get("metadata", {})
    full = result.get("full_record", {})
    score = result.get("final_score", 0)

    doi = meta.get("doi", "unknown")
    title = meta.get("title", "")
    year = meta.get("year", "")

    # Extract batch baseline and flow optimized from full record
    batch_baseline = full.get("batch_baseline", {}) if full else {}
    flow_optimized = full.get("flow_optimized", {}) if full else {}
    process_design = {
        "reactor": full.get("reactor", {}),
        "light_source": full.get("light_source", {}),
        "pump": full.get("pump", {}),
    } if full else {}
    translation = full.get("translation_logic", {}) if full else {}

    summary = result.get("summary", "")

    return f"""Analogy {idx} (similarity score: {score:.3f})
Paper: {doi} — {title} ({year})
Summary: {summary}

Batch baseline:
{json.dumps(batch_baseline, indent=2, default=str)}

Flow optimized:
{json.dumps(flow_optimized, indent=2, default=str)}

Process design:
{json.dumps(process_design, indent=2, default=str)}

Engineering reasoning: {json.dumps(translation, indent=2, default=str)}
"""


class TranslationPromptBuilder:
    """Build the complete prompt for the translation LLM."""

    def build(
        self,
        batch_record: BatchRecord,
        analogies: list[dict],
        chemistry_plan: ChemistryPlan | None = None,
        calculations=None,
    ) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt).

        Parameters
        ----------
        calculations : FlowCalculations | None
            Pre-computed engineering numbers to inject into the prompt.
        """
        system = _load_prompt("translate_system.txt")

        # Format analogies with explicit comparison template
        analogy_blocks = []
        for i, a in enumerate(analogies, 1):
            analogy_blocks.append(_format_analogy(i, a))

        if analogy_blocks:
            analogy_header = (
                "## LITERATURE ANALOGIES\n"
                "For EACH analogy below you MUST explicitly state in reasoning_per_field:\n"
                "  (a) What is chemically similar to our reaction\n"
                "  (b) What is the key difference\n"
                "  (c) How you adjusted parameters because of that difference\n\n"
            )
            analogies_text = analogy_header + "\n---\n".join(analogy_blocks)
        else:
            analogies_text = (
                "No close literature analogies found. "
                "Reason from the pre-computed engineering calculations and first principles."
            )

        # Format chemistry plan
        if chemistry_plan:
            plan_json = json.dumps(
                chemistry_plan.model_dump(exclude_none=True), indent=2
            )
            # Include reasoning if Opus produced it
            reasoning = getattr(chemistry_plan, "_reasoning", "")
        else:
            plan_json = '{"note": "No chemistry plan available."}'
            reasoning = ""

        # Format calculations block
        calc_block = calculations.to_prompt_block() if calculations else (
            "## Pre-computed Engineering Calculations\n"
            "No calculations available — use analogy data and first principles."
        )

        user_template = _load_prompt("translate_user.txt")

        # Use simple replacement instead of .format() to avoid KeyError on
        # literal { } braces inside the JSON example in the template.
        user = (user_template
                .replace("{batch_record_json}",
                         json.dumps(batch_record.model_dump(exclude_none=True), indent=2))
                .replace("{chemistry_plan_json}", plan_json)
                .replace("{chemistry_reasoning}", reasoning or "(not available)")
                .replace("{analogies_text}", analogies_text)
                .replace("{calculations_block}", calc_block))

        return system, user
