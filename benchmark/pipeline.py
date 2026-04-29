from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark.cases import BenchmarkCase
from benchmark.recorder import BenchmarkRecorder, _safe
from flora_translate.analogy_selector import AnalogySelector
from flora_translate.chemistry_agent import ChemistryReasoningAgent
from flora_translate.config import LAB_INVENTORY_PATH, RECORDS_DIR
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
from flora_translate.retriever import VectorRetriever
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
from flora_translate.translation_llm import TranslationLLM
from flora_translate.vector_store import VectorStore
from flora_translate.design_calculator import DesignCalculations


@dataclass
class PreparedContext:
    case: BenchmarkCase
    inventory_path: str
    batch_record: object
    chemistry_plan: object
    analogies: list[dict]
    calculations: object
    design_space: list[dict]
    proposal: object
    pre_council_proposal: dict

    def snapshot(self) -> dict:
        return {
            "case": _safe(self.case),
            "inventory_path": self.inventory_path,
            "batch_record": _safe(self.batch_record),
            "chemistry_plan": _safe(self.chemistry_plan),
            "analogies": _safe(self.analogies),
            "calculations": _safe(self.calculations),
            "design_space": _safe(self.design_space),
            "proposal": _safe(self.proposal),
            "pre_council_proposal": _safe(self.pre_council_proposal),
        }


def prepare_case_context(
    case: BenchmarkCase,
    recorder: BenchmarkRecorder,
    *,
    inventory_path: str = str(LAB_INVENTORY_PATH),
    temperature: float | None = None,
) -> PreparedContext:
    set_llm_observer(recorder.observe_llm)
    set_llm_runtime_overrides(temperature=temperature)
    inventory = LabInventory.from_json(inventory_path)
    try:
        if case.cached_result_path and Path(case.cached_result_path).exists():
            recorder.start_stage("prepare_cached_context", {"cached_result_path": case.cached_result_path})
            with Path(case.cached_result_path).open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            batch_record = BatchRecord(**(case.batch_record_data or {"reaction_description": case.protocol}))
            chemistry_plan = ChemistryPlan(**cached["chemistry_plan"])
            calculations = DesignCalculations(**cached["design_calculations"])
            proposal = FlowProposal(**cached["pre_council_proposal"])
            context = PreparedContext(
                case=case,
                inventory_path=inventory_path,
                batch_record=batch_record,
                chemistry_plan=chemistry_plan,
                analogies=copy.deepcopy(cached.get("_analogies", [])),
                calculations=calculations,
                design_space=copy.deepcopy(cached.get("design_space", [])),
                proposal=proposal,
                pre_council_proposal=copy.deepcopy(cached["pre_council_proposal"]),
            )
            recorder.write_json("prepared_context.json", context.snapshot())
            recorder.end_stage(
                "prepare_cached_context",
                {
                    "analogy_count": len(context.analogies),
                    "design_space_count": len(context.design_space),
                },
            )
            return context

        recorder.start_stage("prepare_input_parse", {"case_id": case.case_id})
        batch_record = InputParser().parse(case.protocol)
        recorder.save_snapshot("batch_record", batch_record)
        recorder.end_stage(
            "prepare_input_parse",
            {"reaction_description": batch_record.reaction_description[:120]},
        )

        recorder.start_stage("prepare_chemistry_analysis")
        chemistry_plan = ChemistryReasoningAgent().analyze(batch_record)
        recorder.save_snapshot("chemistry_plan", chemistry_plan)
        recorder.end_stage(
            "prepare_chemistry_analysis",
            {
                "reaction_class": getattr(chemistry_plan, "reaction_class", None),
                "mechanism_type": getattr(chemistry_plan, "mechanism_type", None),
                "stream_count": len(getattr(chemistry_plan, "stream_logic", []) or []),
            },
        )

        recorder.start_stage("prepare_retrieval")
        store = VectorStore()
        retriever = VectorRetriever(store=store)
        raw_analogies = retriever.retrieve(batch_record, top_k=3, chemistry_plan=chemistry_plan)
        analogies = AnalogySelector(records_dir=RECORDS_DIR).select(raw_analogies)
        recorder.save_snapshot("analogies", analogies)
        recorder.end_stage("prepare_retrieval", {"analogy_count": len(analogies)})

        recorder.start_stage("prepare_design_calculator")
        calculations = DesignCalculator().run(
            batch_record,
            chemistry_plan=chemistry_plan,
            inventory=inventory,
            analogies=analogies,
        )
        recorder.save_snapshot("design_calculations", asdict(calculations))
        recorder.end_stage(
            "prepare_design_calculator",
            {
                "tau_min": calculations.residence_time_min,
                "reynolds_number": calculations.reynolds_number,
                "pressure_drop_bar": calculations.pressure_drop_bar,
            },
        )

        recorder.start_stage("prepare_design_space")
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
            "prepare_design_space",
            {
                "candidate_count": len(design_space),
                "feasible_count": sum(1 for c in design_space if c.get("feasible")),
                "top_candidate_id": getattr(top_candidate, "id", None) if top_candidate else None,
            },
        )

        recorder.start_stage("prepare_translation_llm")
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
            "prepare_translation_llm",
            {
                "proposal_tau_min": proposal.residence_time_min,
                "proposal_flow_rate_mL_min": proposal.flow_rate_mL_min,
                "proposal_tubing_ID_mm": proposal.tubing_ID_mm,
            },
        )

        context = PreparedContext(
            case=case,
            inventory_path=inventory_path,
            batch_record=batch_record,
            chemistry_plan=chemistry_plan,
            analogies=analogies,
            calculations=calculations,
            design_space=design_space,
            proposal=proposal,
            pre_council_proposal=pre_council_proposal,
        )
        recorder.write_json("prepared_context.json", context.snapshot())
        return context
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()


def run_council_from_context(
    context: PreparedContext,
    recorder: BenchmarkRecorder,
    *,
    candidate_budget: int,
    objectives: str = "balanced",
    allow_warning_refinement: bool = True,
    temperature: float = 0.2,
    seed: int | None = None,
    benchmark_strict_scoring: bool = True,
    benchmark_scoring_batch_size: int | None = None,
    benchmark_claude_compact_mode: bool = False,
    benchmark_strong_revision_mode: bool = False,
    benchmark_branching_revision_mode: bool = False,
    benchmark_max_descendants_per_candidate: int = 2,
    benchmark_max_total_revised_candidates: int | None = None,
) -> dict:
    set_llm_observer(recorder.observe_llm)
    set_llm_runtime_overrides(temperature=temperature, seed=seed)

    try:
        recorder.start_stage(
            "run_council",
            {
                "case_id": context.case.case_id,
                "candidate_budget": candidate_budget,
                "temperature": temperature,
                "seed": seed,
                "allow_warning_refinement": allow_warning_refinement,
                "benchmark_claude_compact_mode": benchmark_claude_compact_mode,
                "benchmark_strong_revision_mode": benchmark_strong_revision_mode,
                "benchmark_branching_revision_mode": benchmark_branching_revision_mode,
                "benchmark_max_descendants_per_candidate": benchmark_max_descendants_per_candidate,
                "benchmark_max_total_revised_candidates": benchmark_max_total_revised_candidates,
            },
        )

        design_candidate, final_calc = CouncilV4().run(
            proposal=context.proposal.model_copy(deep=True),
            batch_record=copy.deepcopy(context.batch_record),
            analogies=copy.deepcopy(context.analogies),
            inventory=LabInventory.from_json(context.inventory_path),
            chemistry_plan=copy.deepcopy(context.chemistry_plan),
            calculations=copy.deepcopy(context.calculations),
            objectives=objectives,
            candidate_budget=candidate_budget,
            allow_warning_refinement=allow_warning_refinement,
            benchmark_recorder=recorder,
            benchmark_strict_scoring=benchmark_strict_scoring,
            benchmark_scoring_batch_size=benchmark_scoring_batch_size,
            benchmark_claude_compact_mode=benchmark_claude_compact_mode,
            benchmark_strong_revision_mode=benchmark_strong_revision_mode,
            benchmark_branching_revision_mode=benchmark_branching_revision_mode,
            benchmark_max_descendants_per_candidate=benchmark_max_descendants_per_candidate,
            benchmark_max_total_revised_candidates=benchmark_max_total_revised_candidates,
        )

        formatted = OutputFormatter().format(design_candidate, context.analogies)
        result = {
            "case": _safe(context.case),
            "candidate_budget": candidate_budget,
            "objectives": objectives,
            "temperature": temperature,
            "seed": seed,
            "allow_warning_refinement": allow_warning_refinement,
            "benchmark_strong_revision_mode": benchmark_strong_revision_mode,
            "benchmark_branching_revision_mode": benchmark_branching_revision_mode,
            "benchmark_max_descendants_per_candidate": benchmark_max_descendants_per_candidate,
            "pre_council_proposal": _safe(context.pre_council_proposal),
            "final_design_candidate": _safe(design_candidate),
            "final_calculations": _safe(asdict(final_calc)),
            "formatted_result": _safe(formatted),
            "frozen_context": context.snapshot(),
        }
        recorder.write_json("result.json", result)
        recorder.end_stage(
            "run_council",
            {
                "final_tau_min": design_candidate.proposal.residence_time_min,
                "final_flow_rate_mL_min": design_candidate.proposal.flow_rate_mL_min,
                "final_tubing_ID_mm": design_candidate.proposal.tubing_ID_mm,
            },
        )
        recorder.finalize(
            status="completed",
            extra={
                "final_metrics": {
                    "residence_time_min": design_candidate.proposal.residence_time_min,
                    "flow_rate_mL_min": design_candidate.proposal.flow_rate_mL_min,
                    "tubing_ID_mm": design_candidate.proposal.tubing_ID_mm,
                    "BPR_bar": design_candidate.proposal.BPR_bar,
                    "reactor_volume_mL": design_candidate.proposal.reactor_volume_mL,
                }
            },
        )
        return result
    except Exception as exc:
        recorder.write_json("error.json", {"error": str(exc)})
        recorder.finalize(status="failed", extra={"error": str(exc)})
        raise
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()
