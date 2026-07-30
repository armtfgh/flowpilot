from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark.cases import BenchmarkCase
from benchmark.pipeline import prepare_case_context, run_council_from_context
from benchmark.recorder import BenchmarkRecorder, _safe
from flora_translate.analogy_selector import AnalogySelector
from flora_translate.config import LAB_INVENTORY_PATH, RECORDS_DIR
from flora_translate.engine.llm_agents import (
    call_model_text,
    clear_llm_observer,
    clear_llm_runtime_overrides,
    set_llm_observer,
    set_llm_runtime_overrides,
)
from flora_translate.final_design_validator import finalize_design
from flora_translate.design_disposition import apply_design_disposition_gate
from flora_translate.lightweight_upstream import analyze_batch_chemistry, parse_batch_input
from flora_translate.prompt_builder import TranslationPromptBuilder
from flora_translate.retriever import VectorRetriever
from flora_translate.schemas import FlowProposal, LabInventory
from flora_translate.translation_llm import TranslationLLM
from flora_translate.vector_store import VectorStore

from .cases import AblationCase, ROOT
from .metrics import score_run
from .providers import activate_bundle, credential_status


logger = logging.getLogger("flowpilot.ablation")

CORE_VARIANTS = {"no_retrieval", "no_council", "no_inventory", "full"}


def _benchmark_output_schema() -> dict[str, Any]:
    """Strict provider-native envelope shared by all one-shot conditions."""
    return {
        "type": "object",
        "properties": {
            "recommended_disposition": {
                "type": "string",
                "enum": ["EXECUTE", "SCREEN", "BLOCK"],
            },
            "disposition_rationale": {"type": "string"},
            "proposal_json": {
                "type": "string",
                "description": (
                    "A serialized JSON object containing the complete flow proposal."
                ),
            },
        },
        "required": [
            "recommended_disposition",
            "disposition_rationale",
            "proposal_json",
        ],
        "additionalProperties": False,
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _flow_schema_instruction() -> str:
    return """
Return one provider-validated JSON envelope with exactly three top-level fields:
recommended_disposition, disposition_rationale, and proposal_json.

proposal_json must be a JSON-encoded string whose decoded object uses these fields:
residence_time_min, flow_rate_mL_min, temperature_C, concentration_M,
BPR_bar, reactor_type, tubing_material, tubing_ID_mm, reactor_volume_mL,
residence_time_basis, residence_time_inlet_min,
residence_time_in_channel_min, light_setup, wavelength_nm, streams,
mixer_type, mixing_order_reasoning, pre_reactor_steps,
post_reactor_steps, chemistry_notes, stage_parameters,
reasoning_per_field, literature_analogies, engine_validated,
safety_flags, confidence.

Each stream is an object with stream_label, pump_role, contents, solvent,
phase, flow_rate_mL_min, concentration_M, molar_equiv, gas_flow_sccm, and
gas_flow_actual_mL_min where applicable. Distinguish inlet/STP gas flow from
pressure-corrected in-channel gas flow. Ensure reactor_volume_mL is consistent
with flow and residence time. State assumptions rather than inventing missing
facts.

recommended_disposition must be exactly EXECUTE, SCREEN, or BLOCK. Use BLOCK
when the hard constraints make a safe design physically or operationally
impossible. Use SCREEN for a feasible first experiment that still requires
experimental validation. Use EXECUTE only for a directly executable design
supported by sufficient evidence.
""".strip()


def _general_one_shot(
    case: AblationCase,
    model: str,
    provider: str,
    recorder: BenchmarkRecorder,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    system = (
        "You are a general chemistry assistant. Translate a batch chemistry "
        "protocol into a practical continuous-flow experiment."
    )
    user = (
        f"FROZEN DESIGN INPUT:\n{case.design_input_text}\n\n"
        "Propose one complete flow design with numeric operating conditions, "
        "stream assignments, safety controls, and a brief rationale.\n\n"
        + _flow_schema_instruction()
    )
    recorder.write_json("prompt.json", {"system": system, "user": user})
    set_llm_observer(recorder.observe_llm)
    set_llm_runtime_overrides(
        temperature=temperature,
        seed=seed,
        capture_content=True,
        json_mode=True,
        json_schema=_benchmark_output_schema(),
    )
    try:
        result = call_model_text(
            model=model,
            provider=provider,
            api_name="ablation_general_one_shot",
            max_tokens=16384,
            system=system,
            user_content=user,
        )
        recorder.write_json("raw_response.json", asdict(result))
        try:
            envelope = _parse_json_object(result.text)
            parsed = _parse_json_object(str(envelope.get("proposal_json") or ""))
        except json.JSONDecodeError as exc:
            recorder.write_json(
                "schema_error.json",
                {
                    "error": str(exc),
                    "classification": "malformed_json",
                    "finish_reason": result.finish_reason or result.stop_reason,
                },
            )
            return {
                "variant": "general_one_shot",
                "proposal": {},
                "schema_valid": False,
                "output_validity": "malformed_json",
                "parse_error": str(exc),
            }
        try:
            proposal = FlowProposal(**TranslationLLM._normalize_proposal_data(parsed))
            normalized = proposal.model_dump()
            schema_valid = True
        except Exception as exc:
            normalized = parsed
            schema_valid = False
            recorder.write_json("schema_error.json", {"error": str(exc)})
        return {
            "variant": "general_one_shot",
            "proposal": normalized,
            "raw_proposal": parsed,
            "response_envelope": envelope,
            "reported_disposition": envelope.get("recommended_disposition"),
            "disposition_rationale": envelope.get("disposition_rationale"),
            "schema_valid": schema_valid,
        }
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()


def _structured_single_agent(
    case: AblationCase,
    model: str,
    provider: str,
    recorder: BenchmarkRecorder,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    system = (
        "You are the FlowPilot single-agent baseline. Produce a conservative "
        "batch-to-flow translation. You have no retrieval, external calculator, "
        "inventory enforcement, or specialist council."
    )
    user = (
        "AUTHORITY ORDER:\n"
        "1. Protocol facts\n2. Explicit safety constraints\n3. Model inference\n\n"
        f"FROZEN DESIGN INPUT:\n{case.design_input_text}\n\n"
        + _flow_schema_instruction()
    )
    recorder.write_json("prompt.json", {"system": system, "user": user})
    set_llm_observer(recorder.observe_llm)
    set_llm_runtime_overrides(
        temperature=temperature,
        seed=seed,
        capture_content=True,
        json_mode=True,
        json_schema=_benchmark_output_schema(),
    )
    try:
        result = call_model_text(
            model=model,
            provider=provider,
            api_name="ablation_structured_single_agent",
            max_tokens=16384,
            system=system,
            user_content=user,
        )
        recorder.write_json("raw_response.json", asdict(result))
        try:
            envelope = _parse_json_object(result.text)
            parsed = _parse_json_object(str(envelope.get("proposal_json") or ""))
        except json.JSONDecodeError as exc:
            recorder.write_json(
                "schema_error.json",
                {
                    "error": str(exc),
                    "classification": "malformed_json",
                    "finish_reason": result.finish_reason or result.stop_reason,
                },
            )
            return {
                "variant": "structured_single_agent",
                "proposal": {},
                "schema_valid": False,
                "output_validity": "malformed_json",
                "parse_error": str(exc),
            }
        try:
            proposal = FlowProposal(**TranslationLLM._normalize_proposal_data(parsed))
            proposal_data = proposal.model_dump()
            schema_valid = True
        except Exception as exc:
            proposal_data = parsed
            schema_valid = False
            recorder.write_json("schema_error.json", {"error": str(exc)})
        return {
            "variant": "structured_single_agent",
            "proposal": proposal_data,
            "raw_proposal": parsed,
            "response_envelope": envelope,
            "reported_disposition": envelope.get("recommended_disposition"),
            "disposition_rationale": envelope.get("disposition_rationale"),
            "schema_valid": schema_valid,
        }
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()


def _no_engineering(
    case: AblationCase,
    recorder: BenchmarkRecorder,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    """Use FlowPilot chemistry/retrieval prompts but omit deterministic modules."""
    set_llm_observer(recorder.observe_llm)
    set_llm_runtime_overrides(
        temperature=temperature,
        seed=seed,
        capture_content=True,
    )
    try:
        batch = parse_batch_input(case.design_input_text)
        chemistry = analyze_batch_chemistry(batch)
        analogies: list[dict[str, Any]] = []
        retrieval_note = "OPENAI_API_KEY unavailable"
        import os

        if os.getenv("OPENAI_API_KEY"):
            raw = VectorRetriever(VectorStore()).retrieve(
                batch,
                top_k=3,
                chemistry_plan=chemistry,
                exclude_record_ids=case.excluded_record_ids,
            )
            analogies = AnalogySelector(records_dir=RECORDS_DIR).select(raw)
            retrieval_note = "completed with leave-one-source-out exclusion"
        system, user = TranslationPromptBuilder().build(
            batch,
            analogies,
            chemistry_plan=chemistry,
            calculations=None,
        )
        recorder.write_json("prompt.json", {"system": system, "user": user})
        proposal = TranslationLLM().generate(system, user)
        return {
            "variant": "no_engineering",
            "proposal": proposal.model_dump(),
            "schema_valid": True,
            "batch_record": batch.model_dump(),
            "chemistry_plan": chemistry.model_dump(),
            "analogies": analogies,
            "retrieval_note": retrieval_note,
        }
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()


def _benchmark_case(case: AblationCase) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case.case_id,
        title=case.title,
        protocol=case.design_input_text,
        precedent_level="withheld_source",
        difficulty=case.difficulty,
        notes="FlowPilot ablation case; source flow answer withheld.",
        tags=case.tags,
        objective=case.objective,
        hard_constraints=case.hard_constraints,
    )


def _core_variant(
    case: AblationCase,
    variant: str,
    recorder: BenchmarkRecorder,
    candidate_budget: int,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    enable_retrieval = variant not in {"no_retrieval"}
    inventory_path = (
        str(ROOT / "configs" / "unconstrained_inventory.json")
        if variant == "no_inventory"
        else (
            str(recorder.run_dir / "input_inventory.json")
            if case.inventory
            else str(LAB_INVENTORY_PATH)
        )
    )
    context = prepare_case_context(
        _benchmark_case(case),
        recorder,
        inventory_path=inventory_path,
        temperature=temperature,
        enable_retrieval=enable_retrieval,
        exclude_record_ids=case.excluded_record_ids,
        capture_llm_content=True,
    )
    if variant == "no_council":
        proposal, calculations, final_validation = finalize_design(
            context.proposal,
            batch_record=context.batch_record,
            chemistry_plan=context.chemistry_plan,
            analogies=context.analogies,
            inventory=LabInventory.from_json(context.inventory_path),
        )
        result = {
            "variant": variant,
            "proposal": proposal.model_dump(),
            "schema_valid": True,
            "frozen_context": context.snapshot(),
            "final_calculations": asdict(calculations),
            "final_validation": final_validation,
        }
        decision = apply_design_disposition_gate(
            result,
            proposal=proposal,
            final_validation=final_validation,
            inventory=LabInventory.from_json(context.inventory_path),
            batch_record=context.batch_record,
            chemistry_plan=context.chemistry_plan,
            objective=context.case.objective,
            hard_constraints=context.case.hard_constraints,
        )
        recorder.save_snapshot("design_disposition", decision.to_dict())
        recorder.save_snapshot("final_validation", final_validation)
        recorder.write_json("result.json", result)
        recorder.finalize(
            status="completed",
            extra={"final_metrics": _proposal_summary(result["proposal"])},
        )
        return result

    result = run_council_from_context(
        context,
        recorder,
        candidate_budget=candidate_budget,
        temperature=temperature,
        seed=seed,
        benchmark_strict_scoring=True,
        benchmark_scoring_batch_size=2,
        benchmark_claude_compact_mode=False,
        benchmark_strong_revision_mode=True,
        benchmark_branching_revision_mode=False,
        benchmark_max_descendants_per_candidate=1,
        benchmark_max_total_revised_candidates=candidate_budget,
        capture_llm_content=True,
    )
    proposal = (
        result.get("final_design_candidate", {}).get("proposal", {})
        if isinstance(result.get("final_design_candidate"), dict)
        else {}
    )
    result["variant"] = variant
    result["proposal"] = proposal
    recorder.write_json("result.json", result)
    return result


def _proposal_summary(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        key: proposal.get(key)
        for key in (
            "residence_time_min",
            "flow_rate_mL_min",
            "temperature_C",
            "concentration_M",
            "BPR_bar",
            "reactor_volume_mL",
            "tubing_ID_mm",
        )
    }


def write_checksums(run_dir: Path) -> None:
    rows: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(run_dir)}")
    (run_dir / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def execute_cell(
    *,
    case: AblationCase,
    variant: str,
    bundle_name: str,
    bundle: dict[str, Any],
    run_dir: Path,
    candidate_budget: int,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    public_input = case.public_payload()
    hidden_reference = {
        "source_record_id": case.source_record_id,
        "source_record_aliases": list(case.source_record_aliases),
        "reference_quality": case.reference_quality,
        "expected_features": case.expected_features,
        "reference_flow": case.reference_flow,
    }
    (run_dir / "input_public.json").write_text(
        json.dumps(public_input, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "hidden_reference.json").write_text(
        json.dumps(hidden_reference, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if case.inventory:
        (run_dir / "input_inventory.json").write_text(
            json.dumps(case.inventory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    metadata = {
        "case_id": case.case_id,
        "case_title": case.title,
        "protocol_sha256": case.protocol_sha256,
        "design_input_sha256": case.design_input_sha256,
        "scenario_id": case.scenario_id or case.case_id,
        "pair_id": case.pair_id,
        "scenario_kind": case.scenario_kind,
        "variant": variant,
        "bundle_name": bundle_name,
        "provider": bundle["provider"],
        "model": bundle["model"],
        "base_url": bundle.get("base_url"),
        "candidate_budget": candidate_budget,
        "temperature": temperature,
        "seed": seed,
        "excluded_record_ids": sorted(case.excluded_record_ids),
        "capture_llm_content": True,
    }
    recorder = BenchmarkRecorder(run_dir, metadata)
    ready, reason = credential_status(bundle)
    if not ready:
        summary = recorder.finalize(
            status="blocked_missing_credential",
            extra={"error": reason},
        )
        write_checksums(run_dir)
        return summary

    started = time.perf_counter()
    try:
        with activate_bundle(bundle):
            if variant == "general_one_shot":
                result = _general_one_shot(
                    case,
                    bundle["model"],
                    bundle["provider"],
                    recorder,
                    temperature,
                    seed,
                )
            elif variant == "structured_single_agent":
                result = _structured_single_agent(
                    case,
                    bundle["model"],
                    bundle["provider"],
                    recorder,
                    temperature,
                    seed,
                )
            elif variant == "no_engineering":
                result = _no_engineering(case, recorder, temperature, seed)
            elif variant in CORE_VARIANTS:
                result = _core_variant(
                    case,
                    variant,
                    recorder,
                    candidate_budget,
                    temperature,
                    seed,
                )
            else:
                raise ValueError(f"Unknown architecture variant: {variant}")

        recorder.write_json("result.json", result)
        metrics = score_run(case, result, run_dir)
        recorder.write_json("metrics.json", metrics)
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["status"] = "completed"
            summary.pop("error", None)
            summary.pop("error_type", None)
            summary["external_metrics"] = metrics
            recorder.write_json("run_summary.json", summary)
        else:
            summary = recorder.finalize(
                status="completed",
                extra={"external_metrics": metrics},
            )
        summary["runtime_total_s"] = round(time.perf_counter() - started, 3)
        recorder.write_json("run_summary.json", summary)
    except Exception as exc:
        logger.exception("Ablation cell failed: %s/%s/%s", variant, bundle_name, case.case_id)
        recorder.write_json(
            "error.json",
            {"type": type(exc).__name__, "error": str(exc)},
        )
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update({"status": "failed", "error": str(exc)})
            recorder.write_json("run_summary.json", summary)
        else:
            summary = recorder.finalize(
                status="failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
            )
    write_checksums(run_dir)
    return summary
