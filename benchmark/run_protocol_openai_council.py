from __future__ import annotations

import json
import os
import sys
import traceback
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.recorder import BenchmarkRecorder, _safe
from flora_translate.chemistry_agent import ChemistryReasoningAgent
from flora_translate.config import LAB_INVENTORY_PATH
from flora_translate.design_calculator import DesignCalculator
from flora_translate.engine.council_v4 import CouncilV4
from flora_translate.engine.design_space import (
    DesignSpaceSearch,
    candidates_to_dicts,
    get_council_starting_point,
)
from flora_translate.engine.llm_agents import (
    clear_llm_observer,
    clear_llm_runtime_overrides,
    set_llm_observer,
    set_llm_runtime_overrides,
)
from flora_translate.input_parser import InputParser
from flora_translate.output_formatter import OutputFormatter
from flora_translate.prompt_builder import TranslationPromptBuilder
from flora_translate.schemas import LabInventory
from flora_translate.translation_llm import TranslationLLM


PROTOCOL = """Batch Synthesis of 3,5-Disubstituted Isoxazole in TBAB/EG (1:5) DES
- Phenylacetylene (1a, 0.5 mmol, 1.0 equiv) and ethyl nitroacetate (2, 1.0 mmol, 2.0 equiv) were added directly to TBAB/EG (1:5) DES (1 mL) in an oven-dried 30 mL vial equipped with a magnetic stirring bar; no additional solvent, catalyst, base, or additive was used. The reaction mixture was stirred at 120 °C in an oil bath for 15 min, at which point full conversion (99%) to ethyl 5-phenylisoxazole-3-carboxylate (3a) was achieved via 1,3-dipolar cycloaddition of the in-situ-generated nitrile oxide intermediate with phenylacetylene. After the reaction, the mixture was quenched with water, extracted with dichloromethane, dried over MgSO₄, and the solvent removed under reduced pressure; the isolated NMR yield was 83% (quantified against 1,3,5-trimethoxybenzene as internal standard)."""


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _switch_council_to_openai(model_name: str = "gpt-4o") -> dict:
    import flora_translate.config as cfg
    import flora_translate.engine.llm_agents as llm_agents

    state = {
        "cfg_engine_provider": cfg.ENGINE_PROVIDER,
        "cfg_engine_model_openai": cfg.ENGINE_MODEL_OPENAI,
        "llm_engine_provider": llm_agents.ENGINE_PROVIDER,
        "llm_engine_model_openai": llm_agents.ENGINE_MODEL_OPENAI,
        "openai_client": llm_agents._OPENAI_CLIENT,
    }
    cfg.ENGINE_PROVIDER = "openai"
    cfg.ENGINE_MODEL_OPENAI = model_name
    llm_agents.ENGINE_PROVIDER = "openai"
    llm_agents.ENGINE_MODEL_OPENAI = model_name
    llm_agents._OPENAI_CLIENT = None
    return state


def _restore_council_provider(state: dict) -> None:
    import flora_translate.config as cfg
    import flora_translate.engine.llm_agents as llm_agents

    cfg.ENGINE_PROVIDER = state["cfg_engine_provider"]
    cfg.ENGINE_MODEL_OPENAI = state["cfg_engine_model_openai"]
    llm_agents.ENGINE_PROVIDER = state["llm_engine_provider"]
    llm_agents.ENGINE_MODEL_OPENAI = state["llm_engine_model_openai"]
    llm_agents._OPENAI_CLIENT = state["openai_client"]


def main() -> int:
    run_dir = Path("benchmark/data") / f"protocol_openai_council_{_timestamp()}"
    recorder = BenchmarkRecorder(
        run_dir,
        metadata={
            "case_id": "user_protocol_isoxazole_des",
            "title": "Isoxazole in TBAB/EG DES with OpenAI council",
            "protocol": PROTOCOL,
            "council_provider": "openai",
            "council_model": "gpt-4o",
            "candidate_budget": 12,
            "allow_warning_refinement": True,
            "retrieval_mode": "full_openai_retrieval",
        },
    )

    inventory = LabInventory.from_json(str(LAB_INVENTORY_PATH))
    result_payload: dict = {
        "protocol": PROTOCOL,
        "run_dir": str(run_dir),
        "council_provider": "openai",
        "council_model": "gpt-4o",
        "candidate_budget": 12,
        "allow_warning_refinement": True,
    }

    set_llm_observer(recorder.observe_llm)
    set_llm_runtime_overrides(temperature=0.0)
    council_state = None

    try:
        recorder.start_stage("input_parse", {"protocol_chars": len(PROTOCOL)})
        batch_record = InputParser().parse(PROTOCOL)
        recorder.save_snapshot("batch_record", batch_record)
        recorder.end_stage(
            "input_parse",
            {"reaction_description": getattr(batch_record, "reaction_description", "")[:160]},
        )

        recorder.start_stage("chemistry_analysis")
        chemistry_plan = ChemistryReasoningAgent().analyze(batch_record)
        recorder.save_snapshot("chemistry_plan", chemistry_plan)
        recorder.end_stage(
            "chemistry_analysis",
            {
                "reaction_class": getattr(chemistry_plan, "reaction_class", None),
                "mechanism_type": getattr(chemistry_plan, "mechanism_type", None),
                "stream_count": len(getattr(chemistry_plan, "stream_logic", []) or []),
            },
        )

        recorder.start_stage("retrieval")
        analogies: list[dict] = []
        if os.getenv("OPENAI_API_KEY"):
            from flora_translate.analogy_selector import AnalogySelector
            from flora_translate.config import RECORDS_DIR
            from flora_translate.retriever import VectorRetriever
            from flora_translate.vector_store import VectorStore

            store = VectorStore()
            retriever = VectorRetriever(store=store)
            raw_analogies = retriever.retrieve(batch_record, top_k=3, chemistry_plan=chemistry_plan)
            analogies = AnalogySelector(records_dir=RECORDS_DIR).select(raw_analogies)
            retrieval_status = "completed"
            retrieval_note = f"selected {len(analogies)} analogies"
        else:
            retrieval_status = "skipped"
            retrieval_note = "OPENAI_API_KEY missing in current shell; proceeding with analogies=[]"
        recorder.save_snapshot("analogies", analogies)
        recorder.end_stage(
            "retrieval",
            {
                "status": retrieval_status,
                "analogy_count": len(analogies),
                "note": retrieval_note,
            },
            status=retrieval_status,
        )

        recorder.start_stage("design_calculator")
        calculations = DesignCalculator().run(
            batch_record,
            chemistry_plan=chemistry_plan,
            inventory=inventory,
            analogies=analogies,
        )
        recorder.save_snapshot("design_calculations", asdict(calculations))
        recorder.end_stage(
            "design_calculator",
            {
                "tau_min": calculations.residence_time_min,
                "reynolds_number": calculations.reynolds_number,
                "pressure_drop_bar": calculations.pressure_drop_bar,
            },
        )

        recorder.start_stage("design_space")
        design_candidates = DesignSpaceSearch().run(
            batch_record=batch_record,
            chemistry_plan=chemistry_plan,
            calculations=calculations,
            inventory=inventory,
            reaction_class=getattr(chemistry_plan, "reaction_class", "default") or "default",
        )
        design_space = candidates_to_dicts(design_candidates)
        top_candidate = get_council_starting_point(design_candidates)
        recorder.save_snapshot("design_space", design_space)
        recorder.end_stage(
            "design_space",
            {
                "candidate_count": len(design_space),
                "feasible_count": sum(1 for c in design_space if c.get("feasible")),
                "top_candidate_id": getattr(top_candidate, "id", None) if top_candidate else None,
            },
        )

        recorder.start_stage("translation_llm")
        system_prompt, user_prompt = TranslationPromptBuilder().build(
            batch_record,
            analogies,
            chemistry_plan=chemistry_plan,
            calculations=calculations,
        )
        proposal = TranslationLLM().generate(system_prompt, user_prompt)
        if top_candidate:
            proposal.residence_time_min = top_candidate.tau_min
            proposal.flow_rate_mL_min = top_candidate.Q_mL_min
            proposal.tubing_ID_mm = top_candidate.d_mm
            proposal.reactor_volume_mL = round(top_candidate.V_R_mL, 3)
        pre_council_proposal = proposal.model_dump()
        recorder.save_snapshot("proposal_pre_council", pre_council_proposal)
        recorder.end_stage(
            "translation_llm",
            {
                "proposal_tau_min": proposal.residence_time_min,
                "proposal_flow_rate_mL_min": proposal.flow_rate_mL_min,
                "proposal_tubing_ID_mm": proposal.tubing_ID_mm,
            },
        )

        recorder.start_stage("council_run", {"provider": "openai", "model": "gpt-4o"})
        council_state = _switch_council_to_openai("gpt-4o")
        design_candidate, final_calc = CouncilV4().run(
            proposal=proposal,
            batch_record=deepcopy(batch_record),
            analogies=deepcopy(analogies),
            inventory=inventory,
            chemistry_plan=deepcopy(chemistry_plan),
            calculations=deepcopy(calculations),
            objectives="balanced",
            candidate_budget=12,
            allow_warning_refinement=True,
            benchmark_recorder=recorder,
        )
        recorder.end_stage(
            "council_run",
            {
                "final_tau_min": final_calc.residence_time_min,
                "final_flow_rate_mL_min": final_calc.flow_rate_mL_min,
                "final_tubing_ID_mm": final_calc.tubing_ID_mm,
            },
        )

        recorder.start_stage("format_output")
        formatted = OutputFormatter().format(design_candidate, analogies)
        recorder.end_stage("format_output", {"formatted_chars": len(json.dumps(_safe(formatted)))})

        result_payload.update(
            {
                "status": "completed",
                "batch_record": _safe(batch_record),
                "chemistry_plan": _safe(chemistry_plan),
                "analogies": _safe(analogies),
                "design_calculations_pre_council": _safe(asdict(calculations)),
                "design_space": _safe(design_space),
                "pre_council_proposal": _safe(pre_council_proposal),
                "final_design_candidate": _safe(design_candidate),
                "final_calculations": _safe(asdict(final_calc)),
                "deliberation_log": _safe(
                    design_candidate.deliberation_log.model_dump()
                    if design_candidate.deliberation_log else None
                ),
                "formatted_result": _safe(formatted),
            }
        )
        recorder.write_json("result.json", result_payload)
        recorder.finalize(
            "completed",
            extra={
                "final_tau_min": final_calc.residence_time_min,
                "final_flow_rate_mL_min": final_calc.flow_rate_mL_min,
                "final_tubing_ID_mm": final_calc.tubing_ID_mm,
                "final_BPR_bar": final_calc.bpr_pressure_bar if final_calc.bpr_required else proposal.BPR_bar,
            },
        )
        return 0
    except Exception as exc:
        result_payload.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        recorder.write_json("result.json", result_payload)
        recorder.finalize("error", extra=result_payload)
        return 1
    finally:
        if council_state is not None:
            _restore_council_provider(council_state)
        clear_llm_runtime_overrides()
        clear_llm_observer()


if __name__ == "__main__":
    raise SystemExit(main())
