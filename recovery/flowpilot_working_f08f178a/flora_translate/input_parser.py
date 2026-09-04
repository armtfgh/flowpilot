"""FLORA-Translate — Input parser: normalizes batch protocol to BatchRecord."""

import json
import logging

import anthropic

from flora_translate.config import TRANSLATION_MODEL
from flora_translate.schemas import BatchRecord

logger = logging.getLogger("flora.input_parser")


def _get_client():
    return anthropic.Anthropic()

PARSE_SYSTEM = (
    "You are a chemistry data extraction assistant. "
    "Given a batch chemistry protocol (free text or structured), extract "
    "the fields into the JSON schema below. Use null for anything not stated. "
    "Return ONLY valid JSON, no markdown fences."
)

PARSE_PROMPT = """\
Extract the batch protocol fields from this input.

JSON schema:
{{
  "reaction_description": "",
  "photocatalyst": null,
  "catalyst_loading_mol_pct": null,
  "base": null,
  "solvent": null,
  "temperature_C": null,
  "reaction_time_h": null,
  "concentration_M": null,
  "scale_mmol": null,
  "yield_pct": null,
  "light_source": null,
  "wavelength_nm": null,
  "additives": [],
  "atmosphere": null
}}

Input:
{input_text}
"""


class InputParser:
    """Normalize user input (free text or JSON) into a BatchRecord."""

    def parse(self, batch_input: str | dict) -> BatchRecord:
        # If already a dict or JSON string representing a dict, try direct parse
        if isinstance(batch_input, dict):
            return BatchRecord(**batch_input)

        # Try parsing as JSON first
        try:
            data = json.loads(batch_input)
            if isinstance(data, dict):
                return BatchRecord(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Free text → use LLM to extract structured fields
        logger.info("Parsing free-text batch protocol via LLM")
        return self._llm_parse(batch_input)

    def _llm_parse(self, text: str) -> BatchRecord:
        resp = _get_client().messages.create(
            model=TRANSLATION_MODEL,
            max_tokens=1024,
            system=PARSE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": PARSE_PROMPT.format(input_text=text),
                }
            ],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)
        data["raw_text"] = text
        return BatchRecord(**data)
