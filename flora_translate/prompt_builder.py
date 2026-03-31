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
    ) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt).

        If a ChemistryPlan is provided, it is included in the user prompt
        as the authoritative chemistry reference for the translation LLM.
        """
        system = _load_prompt("translate_system.txt")

        # Format analogies
        analogy_blocks = []
        for i, a in enumerate(analogies, 1):
            analogy_blocks.append(_format_analogy(i, a))

        analogies_text = "\n---\n".join(analogy_blocks) if analogy_blocks else (
            "No close literature analogies found. Use general chemistry "
            "knowledge for photochemical flow reactions."
        )

        # Format chemistry plan
        if chemistry_plan:
            plan_json = json.dumps(
                chemistry_plan.model_dump(exclude_none=True), indent=2
            )
        else:
            plan_json = '{"note": "No chemistry plan available — use your own chemistry knowledge."}'

        user_template = _load_prompt("translate_user.txt")
        user = user_template.format(
            batch_record_json=json.dumps(
                batch_record.model_dump(exclude_none=True), indent=2
            ),
            chemistry_plan_json=plan_json,
            analogies_text=analogies_text,
        )

        return system, user
