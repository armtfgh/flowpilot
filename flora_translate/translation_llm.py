"""FLORA-Translate — Translation LLM: generates flow proposal from analogies."""

import json
import logging
import re
import time

import flora_translate.config as cfg
from flora_translate.engine.llm_agents import call_model_text
from flora_translate.schemas import FlowProposal

logger = logging.getLogger("flora.translation_llm")


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

    @staticmethod
    def _normalize_proposal_data(data: dict) -> dict:
        """Keep basic reactor-volume/flow/residence-time consistency."""
        def _as_float(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        vol = _as_float(data.get("reactor_volume_mL"))
        flow = _as_float(data.get("flow_rate_mL_min"))
        rt = _as_float(data.get("residence_time_min"))
        if vol > 0 and flow > 0 and rt > 0:
            computed_rt = vol / flow
            if abs(computed_rt - rt) / max(rt, 0.01) > 0.1:
                logger.warning(
                    f"Residence time inconsistency: stated={rt:.2f}, "
                    f"computed={computed_rt:.2f} (vol/flow). Using computed."
                )
                data["residence_time_min"] = round(computed_rt, 2)
        return data

    def generate(self, system_prompt: str, user_prompt: str) -> FlowProposal:
        """Generate a FlowProposal from the translation prompt."""
        logger.info("Generating flow proposal via LLM")
        compact_system = (
            system_prompt
            + "\n\nCRITICAL OUTPUT FORMAT: Return only one compact valid JSON object. "
            "No markdown. No prose outside JSON. Keep text fields concise; do not "
            "use multiline paragraphs."
        )
        compact_user = (
            user_prompt
            + "\n\nReturn only valid JSON matching the FlowProposal schema. Keep "
            "reasoning_per_field values short and omit any non-required long "
            "explanation."
        )
        retry_system = (
            "You are the FLORA translation agent. Output only compact valid JSON "
            "for the FlowProposal schema. No markdown, no commentary, no trailing text."
        )
        retry_user = (
            user_prompt
            + "\n\nGenerate the FlowProposal as compact JSON only. Required top-level "
            "fields include: residence_time_min, flow_rate_mL_min, temperature_C, "
            "concentration_M, BPR_bar, reactor_type, tubing_material, tubing_ID_mm, "
            "reactor_volume_mL, light_setup, wavelength_nm, streams, mixer_type, "
            "mixing_order_reasoning, pre_reactor_steps, post_reactor_steps, "
            "chemistry_notes, reasoning_per_field, literature_analogies, "
            "engine_validated, safety_flags, confidence."
        )

        attempts = [
            ("translation_llm", compact_system, compact_user),
            ("translation_llm_retry", retry_system, retry_user),
        ]
        last_text = ""
        last_error: Exception | None = None
        for api_name, system, user in attempts:
            started = time.perf_counter()
            result = call_model_text(
                model=cfg.MODEL_TRANSLATION,
                api_name=api_name,
                max_tokens=8192,
                system=system,
                user_content=user,
            )
            logger.debug(
                "%s completed in %.2f ms with %d chars",
                api_name,
                (time.perf_counter() - started) * 1000,
                len(result.text or ""),
            )
            last_text = result.text or ""
            try:
                data = _parse_json(last_text)
                return FlowProposal(**self._normalize_proposal_data(data))
            except Exception as exc:
                last_error = exc
                logger.warning("%s returned non-parseable JSON: %s", api_name, exc)

        repair_system = "You repair malformed JSON. Output only one valid compact JSON object and nothing else."
        repair_user = (
            "Repair the following invalid FlowProposal JSON into valid JSON. Preserve "
            "numeric values where present. If a field is missing, use a conservative "
            f"empty/default value.\n\nINVALID OUTPUT:\n{last_text[:20000]}"
        )
        result = call_model_text(
            model=cfg.MODEL_TRANSLATION,
            api_name="translation_llm_repair",
            max_tokens=8192,
            system=repair_system,
            user_content=repair_user,
        )
        try:
            data = _parse_json(result.text or "")
            return FlowProposal(**self._normalize_proposal_data(data))
        except Exception:
            if last_error is not None:
                raise last_error
            raise
