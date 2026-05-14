from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.pipeline import PreparedContext, prepare_case_context, run_council_from_context
from benchmark.recorder import BenchmarkRecorder
from benchmark.run_model_matrix_benchmark import (
    COUNCIL_BUNDLES,
    GEMMA_BASE_URL,
    PROTOCOL,
    UPSTREAM_BUNDLES,
    _case,
    council_bundle,
    upstream_bundle,
)
from benchmark.summarize import summarize_experiment
from flora_translate.config import LAB_INVENTORY_PATH
from flora_translate.design_calculator import DesignCalculator
from flora_translate.engine.design_space import DesignSpaceSearch, candidates_to_dicts, get_council_starting_point
from flora_translate.schemas import FlowProposal, IntensificationMandate, LabInventory


RESCUED_UPSTREAMS = ("gpt4omini", "gemma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weak-upstream benchmark with deterministic upstream rescue enabled."
    )
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objectives", default="balanced")
    parser.add_argument("--output-root", default="benchmark/data")
    parser.add_argument("--upstream-bundles", nargs="+", choices=RESCUED_UPSTREAMS)
    parser.add_argument("--council-bundles", nargs="+", choices=sorted(COUNCIL_BUNDLES.keys()))
    parser.add_argument("--allow-warning-refinement", action="store_true", default=True)
    parser.add_argument("--strong-revision-mode", action="store_true", default=True)
    parser.add_argument("--branching-revision-mode", action="store_true", default=True)
    parser.add_argument("--max-descendants-per-candidate", type=int, default=3)
    parser.add_argument("--max-total-revised-candidates", type=int, default=16)
    return parser.parse_args()


def _rescue_context(
    context: PreparedContext,
    recorder: BenchmarkRecorder,
    *,
    upstream_bundle_name: str,
    rescue_tau_min: float = 5.0,
    rescue_target_reduction: float = 3.0,
) -> PreparedContext:
    """Convert a weak raw upstream context into an explicitly rescued context.

    This does not pretend the raw weak upstream was sufficient. It preserves the
    parsed protocol, then replaces only the kinetic/intensification anchor with a
    deterministic thermal-flow anchor so the downstream council can be evaluated.
    """
    rescued = copy.deepcopy(context)
    batch = rescued.batch_record
    plan = rescued.chemistry_plan

    batch.reaction_time_h = 0.25
    batch.concentration_M = 0.5
    batch.temperature_C = 120.0
    batch.solvent = batch.solvent or "TBAB/EG (1:5) DES"

    plan.reaction_class = "thermal"
    plan.mechanism_type = "thermal 1,3-dipolar cycloaddition"
    plan.key_intermediate = plan.key_intermediate or "nitrile oxide"
    plan.confidence_notes = (
        (plan.confidence_notes or "").rstrip()
        + "\n[RESCUED_UPSTREAM] Weak upstream kinetic anchor replaced with deterministic "
        "thermal-flow rescue anchor for benchmark evaluation; raw upstream remains recorded separately."
    ).strip()
    plan.intensification_mandate = IntensificationMandate(
        tau_reduction_target=rescue_target_reduction,
        minimum_flow_advantage="productivity",
        required_mixing_regime="laminar_acceptable",
        flow_justification_basis=(
            "Rescued benchmark anchor: evaluate weak upstream with a conservative "
            f"{rescue_target_reduction:.1f}x thermal-flow residence-time reduction target "
            "rather than accepting the raw weak-model kinetic anchor."
        ),
    )

    proposal = rescued.proposal.model_copy(deep=True)
    proposal.temperature_C = 120.0
    proposal.concentration_M = 0.5
    proposal.BPR_bar = 5.0
    proposal.tubing_material = proposal.tubing_material or "PFA"
    proposal.residence_time_min = rescue_tau_min
    proposal.tubing_ID_mm = 1.0
    proposal.flow_rate_mL_min = 1.0
    proposal.reactor_volume_mL = rescue_tau_min
    proposal.chemistry_notes = (
        (proposal.chemistry_notes or "").rstrip()
        + "\n[RESCUED_UPSTREAM] Council input uses deterministic rescued kinetics for weak-upstream benchmark."
    ).strip()

    inventory = LabInventory.from_json(rescued.inventory_path or str(LAB_INVENTORY_PATH))
    calculations = DesignCalculator().run(
        batch,
        chemistry_plan=plan,
        inventory=inventory,
        analogies=rescued.analogies,
        proposal=proposal,
        target_residence_time_min=rescue_tau_min,
        target_tubing_ID_mm=proposal.tubing_ID_mm,
        target_flow_rate_mL_min=proposal.flow_rate_mL_min,
    )
    design_points = DesignSpaceSearch().run(
        batch_record=batch,
        chemistry_plan=plan,
        calculations=calculations,
        inventory=inventory,
        reaction_class=plan.reaction_class or "thermal",
    )
    design_space = candidates_to_dicts(design_points)
    top_candidate = get_council_starting_point(design_points)
    if top_candidate:
        proposal.residence_time_min = top_candidate.tau_min
        proposal.flow_rate_mL_min = top_candidate.Q_mL_min
        proposal.tubing_ID_mm = top_candidate.d_mm
        proposal.reactor_volume_mL = round(top_candidate.V_R_mL, 3)

    rescued.calculations = calculations
    rescued.design_space = design_space
    rescued.proposal = proposal
    rescued.pre_council_proposal = proposal.model_dump()

    rescue_report = {
        "upstream_bundle": upstream_bundle_name,
        "rescue_mode": "deterministic_thermal_anchor",
        "raw_context_preserved": True,
        "rescued_tau_min": rescue_tau_min,
        "rescued_tau_reduction_target": rescue_target_reduction,
        "batch_time_min": (batch.reaction_time_h or 0.0) * 60.0,
        "batch_concentration_M": batch.concentration_M,
        "proposal_after_rescue": rescued.pre_council_proposal,
        "design_space_count": len(design_space),
        "note": (
            "This benchmark should be compared against raw weak-upstream runs. "
            "It tests whether a weak upstream can be made council-evaluable after deterministic rescue."
        ),
    }
    recorder.save_snapshot("upstream_rescue_report", rescue_report)
    recorder.write_json("prepared_context_rescued.json", rescued.snapshot())
    return rescued


def _write_post_run_audit(experiment_dir: Path) -> None:
    rows: list[dict] = []
    for result_path in sorted(experiment_dir.glob("U_*/C_*/runs/protocol_isoxazole_des_full/budget_*/repeat_01/result.json")):
        parts = result_path.parts
        upstream = next(part[2:] for part in parts if part.startswith("U_"))
        council = next(part[2:] for part in parts if part.startswith("C_"))
        data = json.loads(result_path.read_text(encoding="utf-8"))
        design = data.get("final_design_candidate") or {}
        proposal = design.get("proposal") or {}
        safety = design.get("safety_report") or {}
        summary = ((design.get("deliberation_log") or {}).get("summary") or "")
        concerns = []
        if proposal.get("engine_validated") and safety.get("critical", 0):
            concerns.append("validated_with_global_critical_audit")
        if proposal.get("BPR_bar") in (0, 0.0, None):
            concerns.append("missing_or_zero_BPR")
        if "batch=0" in summary or "batch=0.00" in summary:
            concerns.append("bad_batch_reference")
        if "Warning: no concrete flow advantage" in summary:
            concerns.append("weak_chief_justification")
        rows.append({
            "upstream": upstream,
            "council": council,
            "engine_validated": proposal.get("engine_validated"),
            "screen_required": bool(safety.get("screen_required")),
            "fallback_reason": safety.get("fallback_reason") or "",
            "tau_min": proposal.get("residence_time_min"),
            "Q_mL_min": proposal.get("flow_rate_mL_min"),
            "ID_mm": proposal.get("tubing_ID_mm"),
            "BPR_bar": proposal.get("BPR_bar"),
            "V_mL": proposal.get("reactor_volume_mL"),
            "critical": safety.get("critical"),
            "total_checks": safety.get("total_checks"),
            "screen_count": len(safety.get("screen_candidates") or []),
            "concerns": ";".join(concerns),
            "result_path": str(result_path),
        })
    if not rows:
        return
    with (experiment_dir / "post_run_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    case = _case()
    selected_upstreams = args.upstream_bundles or list(RESCUED_UPSTREAMS)
    selected_councils = args.council_bundles or list(COUNCIL_BUNDLES)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_root) / f"weak_upstream_rescue_benchmark_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with (experiment_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "case": asdict(case),
            "budget": args.budget,
            "temperature": args.temperature,
            "seed": args.seed,
            "objectives": args.objectives,
            "selected_upstream_bundles": selected_upstreams,
            "selected_council_bundles": selected_councils,
            "rescue_mode": "deterministic_thermal_anchor",
            "raw_upstream_bundles": UPSTREAM_BUNDLES,
            "council_bundles": COUNCIL_BUNDLES,
            "gemma_model": "google/gemma-4-31B-it",
            "gemma_base_url": GEMMA_BASE_URL,
        }, handle, indent=2, ensure_ascii=False)

    matrix_rows: list[dict] = []
    for upstream_name in selected_upstreams:
        with upstream_bundle(upstream_name) as upstream_models:
            group_dir = experiment_dir / f"U_{upstream_name}_rescued"
            context_dir = group_dir / "contexts" / case.case_id
            prep_recorder = BenchmarkRecorder(
                context_dir,
                {
                    "phase": "prepare_rescued_context",
                    "case_id": case.case_id,
                    "case_title": case.title,
                    "temperature": args.temperature,
                    "protocol": case.protocol,
                    "upstream_bundle": f"{upstream_name}_rescued",
                    "raw_upstream_bundle": upstream_name,
                    "upstream_models": upstream_models,
                    "rescue_mode": "deterministic_thermal_anchor",
                    "gemma_base_url": GEMMA_BASE_URL if upstream_name == "gemma" else None,
                },
            )
            try:
                context = prepare_case_context(case, prep_recorder, temperature=args.temperature)
                context = _rescue_context(
                    context,
                    prep_recorder,
                    upstream_bundle_name=upstream_name,
                )
                prep_recorder.finalize(status="completed")
            except Exception as exc:
                prep_recorder.finalize(status="failed", extra={"error": str(exc)})
                print(json.dumps({
                    "upstream_bundle": upstream_name,
                    "phase": "prepare_rescued_context",
                    "status": "failed",
                    "error": str(exc),
                }, ensure_ascii=False))
                continue

            for council_name in selected_councils:
                cell_dir = group_dir / f"C_{council_name}"
                run_dir = cell_dir / "runs" / case.case_id / f"budget_{args.budget}" / "repeat_01"
                benchmark_claude_compact_mode = council_name == "claude"
                with council_bundle(council_name) as council_cfg:
                    recorder = BenchmarkRecorder(
                        run_dir,
                        {
                            "phase": "rescued_council_run",
                            "case_id": case.case_id,
                            "case_title": case.title,
                            "candidate_budget": args.budget,
                            "repeat_index": 1,
                            "temperature": args.temperature,
                            "seed": args.seed,
                            "objectives": args.objectives,
                            "protocol": case.protocol,
                            "upstream_bundle": f"{upstream_name}_rescued",
                            "raw_upstream_bundle": upstream_name,
                            "council_bundle": council_name,
                            "council_provider": council_cfg["provider"],
                            "upstream_models": upstream_models,
                            "council_model": council_cfg["model"],
                            "rescue_mode": "deterministic_thermal_anchor",
                            "gemma_base_url": GEMMA_BASE_URL if (upstream_name == "gemma" or council_name == "gemma") else None,
                        },
                    )
                    try:
                        run_council_from_context(
                            context,
                            recorder,
                            candidate_budget=args.budget,
                            objectives=args.objectives,
                            allow_warning_refinement=args.allow_warning_refinement,
                            temperature=args.temperature,
                            seed=args.seed,
                            benchmark_strict_scoring=True,
                            benchmark_scoring_batch_size=3,
                            benchmark_claude_compact_mode=benchmark_claude_compact_mode,
                            benchmark_strong_revision_mode=args.strong_revision_mode,
                            benchmark_branching_revision_mode=args.branching_revision_mode,
                            benchmark_max_descendants_per_candidate=args.max_descendants_per_candidate,
                            benchmark_max_total_revised_candidates=args.max_total_revised_candidates,
                        )
                    except Exception as exc:
                        print(json.dumps({
                            "upstream_bundle": f"{upstream_name}_rescued",
                            "council_bundle": council_name,
                            "status": "failed",
                            "error": str(exc),
                            "run_dir": str(run_dir),
                        }, ensure_ascii=False))

                cell_summary = summarize_experiment(cell_dir)
                run_manifest = cell_dir / "run_manifest.csv"
                if run_manifest.exists():
                    rows = list(csv.DictReader(run_manifest.open("r", encoding="utf-8")))
                    if rows:
                        row = rows[0]
                        matrix_rows.append({
                            "upstream_bundle": f"{upstream_name}_rescued",
                            "raw_upstream_bundle": upstream_name,
                            "council_bundle": council_name,
                            "status": row.get("status"),
                            "runtime_s": row.get("runtime_s"),
                            "llm_call_count": row.get("llm_call_count"),
                            "total_tokens": row.get("total_tokens"),
                            "final_tau_min": row.get("final_tau_min"),
                            "final_flow_rate_mL_min": row.get("final_flow_rate_mL_min"),
                            "final_tubing_ID_mm": row.get("final_tubing_ID_mm"),
                            "final_BPR_bar": row.get("final_BPR_bar"),
                            "final_reactor_volume_mL": row.get("final_reactor_volume_mL"),
                            "cell_dir": str(cell_dir),
                            "run_summary_path": row.get("summary_path", ""),
                            "result_path": row.get("result_path", ""),
                            "llm_event_count": cell_summary.get("llm_event_count", 0),
                        })
                    else:
                        matrix_rows.append({
                            "upstream_bundle": f"{upstream_name}_rescued",
                            "raw_upstream_bundle": upstream_name,
                            "council_bundle": council_name,
                            "status": "no_run_manifest_rows",
                            "cell_dir": str(cell_dir),
                        })
                else:
                    matrix_rows.append({
                        "upstream_bundle": f"{upstream_name}_rescued",
                        "raw_upstream_bundle": upstream_name,
                        "council_bundle": council_name,
                        "status": "no_run_manifest",
                        "cell_dir": str(cell_dir),
                    })

    manifest_path = experiment_dir / "matrix_manifest.csv"
    if matrix_rows:
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
            writer.writeheader()
            writer.writerows(matrix_rows)
    _write_post_run_audit(experiment_dir)
    summary = {
        "experiment_dir": str(experiment_dir),
        "cell_count": len(matrix_rows),
        "matrix_manifest_csv": str(manifest_path),
        "post_run_audit_csv": str(experiment_dir / "post_run_audit.csv"),
    }
    with (experiment_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
