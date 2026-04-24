"""FLORA-Translate — Translation LLM: generates flow proposal from analogies."""

import json
import logging
import re
import time

import anthropic

from flora_translate.config import MODEL_TRANSLATION as TRANSLATION_MODEL
from flora_translate.engine.llm_agents import emit_component_llm_event, get_llm_runtime_overrides
from flora_translate.schemas import FlowProposal

logger = logging.getLogger("flora.translation_llm")

def _get_client():
    return anthropic.Anthropic()


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, tolerating markdown fences."""
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


class TranslationLLM:
    """Call Claude to generate a flow proposal from batch + analogies."""

    def generate(self, system_prompt: str, user_prompt: str) -> FlowProposal:
        """Generate a FlowProposal from the translation prompt."""
        logger.info("Generating flow proposal via LLM")
        kwargs = {}
        if "temperature" in get_llm_runtime_overrides():
            kwargs["temperature"] = get_llm_runtime_overrides()["temperature"]
        started = time.perf_counter()
        resp = _get_client().messages.create(
            model=TRANSLATION_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **kwargs,
        )
        emit_component_llm_event(
            component="translation_llm",
            provider="anthropic",
            model=TRANSLATION_MODEL,
            max_tokens=4096,
            system=system_prompt,
            user_content=user_prompt,
            resp=resp,
            started=started,
            extra={"response_chars": len(resp.content[0].text)},
        )
        data = _parse_json(resp.content[0].text)

        # Validate residence_time consistency
        vol = data.get("reactor_volume_mL", 0)
        flow = data.get("flow_rate_mL_min", 0)
        rt = data.get("residence_time_min", 0)
        if vol > 0 and flow > 0 and rt > 0:
            computed_rt = vol / flow
            if abs(computed_rt - rt) / max(rt, 0.01) > 0.1:
                logger.warning(
                    f"Residence time inconsistency: stated={rt:.2f}, "
                    f"computed={computed_rt:.2f} (vol/flow). Using computed."
                )
                data["residence_time_min"] = round(computed_rt, 2)

        return FlowProposal(**data)
