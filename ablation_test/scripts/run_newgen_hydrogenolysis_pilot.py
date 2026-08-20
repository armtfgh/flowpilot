"""Run the second NewGen pilot: three-phase packed-bed hydrogenolysis."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.cases import load_cases_from_path
from ablation_test.src.metrics import score_run
from ablation_test.src.providers import activate_bundle, endpoint_health
from ablation_test.src.runner import _parse_json_object, execute_cell, write_checksums
from benchmark.recorder import BenchmarkRecorder
from flora_translate.engine.llm_agents import (
    call_model_text,
    clear_llm_observer,
    clear_llm_runtime_overrides,
    set_llm_observer,
    set_llm_runtime_overrides,
)
from flora_translate.schemas import FlowProposal, LabInventory
from flora_translate.translation_llm import TranslationLLM


BENCHMARK = ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_hydrogenolysis"
OUTPUT = ROOT / "ablation_results" / "newgen_benchmark" / "newgen_benchmark_v1_pilot_hydrogenolysis_20260813"
SOURCE_PDF = Path("/home/amirreza/SharedFolder/win2fed/mined_papers_2/acs.oprd.9b00416.pdf")
BUNDLE = {
    "provider": "ollama",
    "model": "/models/Qwen3.6-27B",
    "base_url": "http://10.13.24.169:8000/v1",
    "upstream_mode": "always",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_direct_proposal(data: dict) -> dict:
    data = dict(data)
    for field in ("pre_reactor_steps", "post_reactor_steps", "literature_analogies", "safety_flags"):
        if isinstance(data.get(field), str):
            data[field] = [data[field]]
    if isinstance(data.get("stage_parameters"), dict):
        data["stage_parameters"] = [data["stage_parameters"]]
    confidence = data.get("confidence")
    if isinstance(confidence, (int, float)):
        data["confidence"] = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"
    return TranslationLLM._normalize_proposal_data(data)


def run_direct_one_shot(case, run_dir: Path) -> dict:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    recorder = BenchmarkRecorder(
        run_dir,
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
        "You are a general chemistry assistant. Translate the supplied batch protocol "
        "into one practical continuous-flow screening experiment. Use only the supplied "
        "inventory. Return one valid JSON object and no markdown."
    )
    user = f"""FROZEN DESIGN INPUT:
{case.design_input_text}

Return exactly one JSON object with recommended_disposition, disposition_rationale,
and proposal. proposal must contain residence_time_min, flow_rate_mL_min,
temperature_C, concentration_M, BPR_bar, reactor_type, tubing_material,
tubing_ID_mm, reactor_volume_mL, residence_time_basis,
residence_time_inlet_min, residence_time_in_channel_min, streams, mixer_type,
mixing_order_reasoning, pre_reactor_steps, post_reactor_steps, chemistry_notes,
stage_parameters, multiphase_metrics, reasoning_per_field, literature_analogies,
engine_validated, safety_flags, and confidence.

Each stream must use exactly these keys: stream_label, pump_role, contents (array),
solvent, concentration_M, flow_rate_mL_min, phase, gas_flow_sccm,
gas_flow_actual_mL_min, molar_equiv, and reasoning. Use separate liquid and H2 gas
streams. Gas flow_rate_mL_min must not be used as an ambiguous substitute for both
STP and in-channel flow. Include formulas and pressure basis in multiphase_metrics.
Use arrays for all step and safety fields, and HIGH/MEDIUM/LOW for confidence.
Do not cite or infer any held-out source answer.
"""
    recorder.write_json("prompt.json", {"system": system, "user": user})
    recorder.write_json("input_public.json", case.public_payload())
    recorder.write_json("input_inventory.json", case.inventory)
    set_llm_observer(recorder.observe_llm)
    set_llm_runtime_overrides(temperature=0.0, seed=20260813, capture_content=True, json_mode=True)
    try:
        with activate_bundle(BUNDLE):
            response = call_model_text(
                model=BUNDLE["model"],
                provider=BUNDLE["provider"],
                api_name="newgen_hydrogenolysis_one_shot_direct",
                max_tokens=16384,
                system=system,
                user_content=user,
            )
        recorder.write_json("raw_response.json", asdict(response))
        envelope = _parse_json_object(response.text)
        raw = envelope.get("proposal")
        if not isinstance(raw, dict):
            raise ValueError("Direct one-shot response did not contain a proposal object")
        proposal = FlowProposal(**normalize_direct_proposal(raw)).model_dump()
        result = {
            "variant": "general_one_shot",
            "proposal": proposal,
            "raw_proposal": raw,
            "response_envelope": envelope,
            "reported_disposition": envelope.get("recommended_disposition"),
            "disposition_rationale": envelope.get("disposition_rationale"),
            "schema_valid": True,
            "output_contract": "direct_json_object_v1",
        }
        recorder.write_json("result.json", result)
        metrics = score_run(case, result, run_dir)
        recorder.write_json("metrics.json", metrics)
        return recorder.finalize(status="completed", extra={"external_metrics": metrics})
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()
        write_checksums(run_dir)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT / "frozen"
    runs = OUTPUT / "runs"
    frozen.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    case_path = BENCHMARK / "case.json"
    rubric_path = BENCHMARK / "universal_rubric.json"
    oracle_path = BENCHMARK / "hidden_oracle.json"
    case = load_cases_from_path(case_path)[0]
    LabInventory(**case.inventory)

    health = endpoint_health(BUNDLE)
    if not health.get("reachable") or not health.get("model_advertised"):
        raise RuntimeError(f"Qwen endpoint is not ready: {health}")
    for source in (case_path, rubric_path, oracle_path, SOURCE_PDF):
        shutil.copy2(source, frozen / source.name)
    record = ROOT / "flora_translate" / "data" / "records" / "acs.oprd.9b00416.json"
    write_json(
        frozen / "execution_manifest.json",
        {
            "schema_version": "flowpilot_newgen_execution_manifest_v1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "case_id": case.case_id,
            "model_bundle": BUNDLE,
            "conditions": [
                {"condition_id": "qwen27b_one_shot", "variant": "general_one_shot"},
                {"condition_id": "qwen27b_full_flowpilot", "variant": "full"}
            ],
            "temperature": 0.0,
            "seed": 20260813,
            "candidate_budget": 3,
            "design_input_sha256": case.design_input_sha256,
            "excluded_record_ids": sorted(case.excluded_record_ids),
            "source_hashes": {str(p): sha256(p) for p in (case_path, rubric_path, oracle_path, SOURCE_PDF, record)},
            "endpoint_health": health,
        },
    )

    one_summary = run_direct_one_shot(case, runs / "qwen27b_one_shot")
    full_summary = execute_cell(
        case=case,
        variant="full",
        bundle_name="qwen36_27b",
        bundle=BUNDLE,
        run_dir=runs / "qwen27b_full_flowpilot",
        candidate_budget=3,
        temperature=0.0,
        seed=20260813,
    )
    write_json(
        OUTPUT / "execution_summary.json",
        {
            "runs": [
                {"condition_id": "qwen27b_one_shot", "variant": "general_one_shot", **one_summary},
                {"condition_id": "qwen27b_full_flowpilot", "variant": "full", **full_summary},
            ]
        },
    )
    write_checksums(OUTPUT)
    print(OUTPUT)
    print(json.dumps(read_json(OUTPUT / "execution_summary.json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
