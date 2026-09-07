"""FLORA-Translate — Translation LLM: generates flow proposal from analogies."""

import json
import logging
import re
import time

import flora_translate.config as cfg
from flora_translate.engine.llm_agents import call_model_text
from flora_translate.residence_time_basis import (
    INLET_STP_BASIS,
    UNKNOWN_BASIS,
    normalize_residence_time_basis,
    stp_gas_flow_from_actual,
)
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
        def _unwrap(value, default=None):
            if not isinstance(value, dict):
                return value
            for key in (
                "value",
                "selected",
                "recommended",
                "estimate",
                "nominal",
                "result",
                "text",
                "name",
            ):
                if key in value and not isinstance(value[key], (dict, list)):
                    return value[key]
            return default

        scalar_defaults = {
            "residence_time_min": 0.0,
            "flow_rate_mL_min": 0.0,
            "temperature_C": 25.0,
            "concentration_M": 0.1,
            "BPR_bar": 0.0,
            "tubing_ID_mm": 1.0,
            "reactor_volume_mL": 0.0,
        }
        string_defaults = {
            "reactor_type": "coil",
            "tubing_material": "FEP",
            "residence_time_basis": "",
            "light_setup": "",
            "mixer_type": "T-mixer",
            "mixing_order_reasoning": "",
            "chemistry_notes": "",
            "confidence": "LOW",
        }
        list_defaults = (
            "streams",
            "pre_reactor_steps",
            "post_reactor_steps",
            "stage_parameters",
            "literature_analogies",
            "safety_flags",
        )
        dict_defaults = (
            "multiphase_metrics",
            "heat_transfer_metrics",
            "inventory_selection",
            "inventory_constraints",
            "evidence_calibration",
            "reasoning_per_field",
        )
        for key, default in scalar_defaults.items():
            data[key] = _unwrap(data.get(key), default)
            if data.get(key) is None:
                data[key] = default
        for key, default in string_defaults.items():
            value = data.get(key)
            if isinstance(value, dict):
                unwrapped = _unwrap(value)
                data[key] = (
                    str(unwrapped)
                    if unwrapped is not None
                    else json.dumps(value, sort_keys=True, ensure_ascii=True)
                )
            elif value is None:
                data[key] = default
        for key in list_defaults:
            if data.get(key) is None:
                data[key] = []
            elif isinstance(data.get(key), dict):
                value = data[key]
                data[key] = value.get("items") if isinstance(value.get("items"), list) else [value]
        for key in dict_defaults:
            if data.get(key) is None:
                data[key] = {}
        reasoning = data.get("reasoning_per_field") or {}
        if isinstance(reasoning, dict):
            data["reasoning_per_field"] = {
                str(key): (
                    value
                    if isinstance(value, str)
                    else json.dumps(value, sort_keys=True, ensure_ascii=True)
                )
                for key, value in reasoning.items()
            }
        engine_validated = data.get("engine_validated")
        if isinstance(engine_validated, dict):
            unwrapped = _unwrap(engine_validated)
            if unwrapped is None:
                status = str(engine_validated.get("status") or "").lower()
                unwrapped = status in {"valid", "validated", "pass", "passed", "ready"}
            engine_validated = unwrapped
        if isinstance(engine_validated, str):
            engine_validated = engine_validated.strip().lower() in {
                "true", "yes", "1", "valid", "validated", "pass", "passed", "ready"
            }
        data["engine_validated"] = bool(engine_validated)
        for stream in data.get("streams") or []:
            if not isinstance(stream, dict):
                continue
            if stream.get("molar_equiv") is None:
                stream["molar_equiv"] = 1.0
            for key in (
                "concentration_M",
                "flow_rate_mL_min",
                "gas_flow_sccm",
                "gas_flow_actual_mL_min",
                "molar_equiv",
                "introduction_stage",
            ):
                if isinstance(stream.get(key), dict):
                    stream[key] = _unwrap(stream[key])

        def _as_float(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        vol = _as_float(data.get("reactor_volume_mL"))
        flow = _as_float(data.get("flow_rate_mL_min"))
        rt = _as_float(data.get("residence_time_min"))
        gas_actual = 0.0
        gas_stp = 0.0
        for stream in data.get("streams") or []:
            if str(stream.get("phase", "")).lower() == "gas":
                gas_actual = _as_float(stream.get("gas_flow_actual_mL_min") or stream.get("flow_rate_mL_min"))
                gas_stp = _as_float(stream.get("gas_flow_sccm"))
                break
        basis = normalize_residence_time_basis(data.get("residence_time_basis"))
        if gas_actual > 0 or gas_stp > 0:
            basis = INLET_STP_BASIS
            data["residence_time_basis"] = "inlet/STP apparent residence time"
            if gas_stp <= 0 and gas_actual > 0:
                gas_stp = stp_gas_flow_from_actual(
                    gas_actual,
                    _as_float(data.get("temperature_C")) or 25.0,
                    _as_float(data.get("BPR_bar")),
                )
                for stream in data.get("streams") or []:
                    if str(stream.get("phase", "")).lower() == "gas":
                        stream["gas_flow_sccm"] = round(gas_stp, 6)
                        break
        if vol > 0 and flow > 0 and rt > 0:
            if basis == INLET_STP_BASIS and gas_stp > 0:
                computed_rt = vol / (flow + gas_stp)
                data["residence_time_inlet_min"] = round(computed_rt, 2)
                if gas_actual > 0:
                    data["residence_time_in_channel_min"] = round(vol / (flow + gas_actual), 2)
            else:
                computed_rt = vol / flow
            if abs(computed_rt - rt) / max(rt, 0.01) > 0.1:
                logger.warning(
                    f"Residence time inconsistency: stated={rt:.2f}, "
                    f"computed={computed_rt:.2f} on {basis}. Using computed."
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
            "reactor_volume_mL, residence_time_basis, residence_time_inlet_min, "
            "residence_time_in_channel_min, light_setup, wavelength_nm, streams, mixer_type, "
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
