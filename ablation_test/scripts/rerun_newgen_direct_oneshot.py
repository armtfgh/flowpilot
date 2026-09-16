"""Rerun the NewGen pilot one-shot baseline with a direct JSON object contract.

The first pilot used a JSON string nested inside a provider-validated envelope.
That representation was syntactically fragile for the local Qwen endpoint. This
runner preserves that attempt and requests the same design as a direct object so
the benchmark measures chemistry and engineering rather than string escaping.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.cases import load_cases_from_path
from ablation_test.src.metrics import score_run
from ablation_test.src.providers import activate_bundle
from ablation_test.src.runner import _parse_json_object, write_checksums
from benchmark.recorder import BenchmarkRecorder
from flora_translate.engine.llm_agents import (
    call_model_text,
    clear_llm_observer,
    clear_llm_runtime_overrides,
    set_llm_observer,
    set_llm_runtime_overrides,
)
from flora_translate.schemas import FlowProposal
from flora_translate.translation_llm import TranslationLLM


BENCHMARK = ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_cuaac"
OUTPUT = ROOT / "ablation_results" / "newgen_benchmark" / "newgen_benchmark_v1_pilot_cuaac_20260813"
BUNDLE = {
    "provider": "ollama",
    "model": "/models/Qwen3.6-27B",
    "base_url": "http://10.13.24.169:8000/v1",
    "upstream_mode": "always",
}


def _prepare_for_validation(data: dict) -> dict:
    normalized = dict(data)
    for field in ("pre_reactor_steps", "post_reactor_steps"):
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = [value]
    stages = normalized.get("stage_parameters")
    if isinstance(stages, dict):
        normalized["stage_parameters"] = [stages]
    confidence = normalized.get("confidence")
    if isinstance(confidence, (int, float)):
        normalized["confidence"] = (
            "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"
        )
    return TranslationLLM._normalize_proposal_data(normalized)


def main() -> None:
    case = load_cases_from_path(BENCHMARK / "case.json")[0]
    runs = OUTPUT / "runs"
    current = runs / "qwen27b_one_shot"
    archived = runs / "qwen27b_one_shot_attempt1_malformed"
    if current.is_dir() and (current / "schema_error.json").is_file() and not archived.exists():
        current.rename(archived)
    if current.exists():
        shutil.rmtree(current)

    recorder = BenchmarkRecorder(
        current,
        {
            "case_id": case.case_id,
            "variant": "general_one_shot",
            "contract": "direct_json_object_v1",
            "model": BUNDLE["model"],
            "provider": BUNDLE["provider"],
            "base_url": BUNDLE["base_url"],
            "temperature": 0.0,
            "seed": 20260813,
            "design_input_sha256": case.design_input_sha256,
            "excluded_record_ids": sorted(case.excluded_record_ids),
        },
    )
    system = (
        "You are a general chemistry assistant. Translate the supplied batch "
        "protocol into one practical continuous-flow screening experiment. "
        "Use only the supplied inventory. Return one valid JSON object and no markdown."
    )
    user = f"""FROZEN DESIGN INPUT:
{case.design_input_text}

Return exactly one JSON object with these top-level fields:
- recommended_disposition: EXECUTE, SCREEN, or BLOCK
- disposition_rationale: string
- proposal: an object containing residence_time_min, flow_rate_mL_min,
  temperature_C, concentration_M, BPR_bar, reactor_type, tubing_material,
  tubing_ID_mm, reactor_volume_mL, residence_time_basis,
  residence_time_inlet_min, residence_time_in_channel_min, light_setup,
  wavelength_nm, streams, mixer_type, mixing_order_reasoning,
  pre_reactor_steps, post_reactor_steps, chemistry_notes, stage_parameters,
  reasoning_per_field, literature_analogies, engine_validated, safety_flags,
  and confidence.

Use arrays for streams, steps, stage_parameters, literature_analogies, and
safety_flags. Use HIGH, MEDIUM, or LOW for confidence. Each liquid stream must
report its own flow and feed composition. Ensure total liquid flow equals the
sum of liquid-stream flows, and reactor volume = total liquid flow x catalyst
contact residence time. Use null for non-applicable photochemical or gas fields.
Do not cite or infer any held-out source answer.
"""
    recorder.write_json("prompt.json", {"system": system, "user": user})
    recorder.write_json("input_public.json", case.public_payload())
    recorder.write_json("input_inventory.json", case.inventory)

    set_llm_observer(recorder.observe_llm)
    set_llm_runtime_overrides(
        temperature=0.0,
        seed=20260813,
        capture_content=True,
        json_mode=True,
    )
    try:
        with activate_bundle(BUNDLE):
            response = call_model_text(
                model=BUNDLE["model"],
                provider=BUNDLE["provider"],
                api_name="newgen_general_one_shot_direct",
                max_tokens=16384,
                system=system,
                user_content=user,
            )
        recorder.write_json("raw_response.json", asdict(response))
        envelope = _parse_json_object(response.text)
        raw_proposal = envelope.get("proposal")
        if not isinstance(raw_proposal, dict):
            raise ValueError("Direct one-shot response did not contain a proposal object")
        prepared = _prepare_for_validation(raw_proposal)
        proposal = FlowProposal(**prepared).model_dump()
        result = {
            "variant": "general_one_shot",
            "proposal": proposal,
            "raw_proposal": raw_proposal,
            "response_envelope": envelope,
            "reported_disposition": envelope.get("recommended_disposition"),
            "disposition_rationale": envelope.get("disposition_rationale"),
            "schema_valid": True,
            "output_contract": "direct_json_object_v1",
        }
        recorder.write_json("result.json", result)
        metrics = score_run(case, result, current)
        recorder.write_json("metrics.json", metrics)
        summary = recorder.finalize(status="completed", extra={"external_metrics": metrics})
        print(json.dumps(summary, indent=2))
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()
        write_checksums(current)


if __name__ == "__main__":
    main()
