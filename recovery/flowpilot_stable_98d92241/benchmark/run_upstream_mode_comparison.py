from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flora_translate.config as cfg
from benchmark.cases import BenchmarkCase
from benchmark.pipeline import prepare_case_context, run_council_from_context
from benchmark.recorder import BenchmarkRecorder
from benchmark.summarize import summarize_experiment
from benchmark.run_model_matrix_benchmark import COUNCIL_BUNDLES, PROTOCOL, UPSTREAM_BUNDLES
from flora_translate.engine import llm_agents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare main vs lightweight upstream modes on a fixed model pair.")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objectives", default="balanced")
    parser.add_argument("--allow-warning-refinement", action="store_true", default=True)
    parser.add_argument("--strong-revision-mode", action="store_true", default=True)
    parser.add_argument("--branching-revision-mode", action="store_true", default=True)
    parser.add_argument("--max-descendants-per-candidate", type=int, default=3)
    parser.add_argument("--max-total-revised-candidates", type=int, default=16)
    parser.add_argument("--upstream-bundle", choices=sorted(UPSTREAM_BUNDLES.keys()), default="gpt4omini")
    parser.add_argument("--council-bundle", choices=sorted(COUNCIL_BUNDLES.keys()), default="claude")
    parser.add_argument("--modes", nargs="+", choices=("never", "always"), default=["never", "always"])
    parser.add_argument("--output-root", default="benchmark/data")
    return parser.parse_args()


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="protocol_isoxazole_des_full",
        title="Batch Synthesis of 3,5-Disubstituted Isoxazole in TBAB/EG (1:5) DES",
        protocol=PROTOCOL,
        precedent_level="weak_precedent",
        difficulty="high",
        notes="Compare main vs lightweight upstream mode on one fixed model pair.",
        tags=("thermal", "cycloaddition", "des", "user_protocol", "upstream_mode_compare"),
    )


@contextmanager
def upstream_bundle(name: str):
    overrides = UPSTREAM_BUNDLES[name]
    original = {key: getattr(cfg, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(cfg, key, value)
        yield overrides
    finally:
        for key, value in original.items():
            setattr(cfg, key, value)


@contextmanager
def council_bundle(name: str):
    bundle = COUNCIL_BUNDLES[name]
    original = {
        "cfg_engine_provider": cfg.ENGINE_PROVIDER,
        "cfg_engine_model_anthropic": cfg.ENGINE_MODEL_ANTHROPIC,
        "cfg_engine_model_openai": cfg.ENGINE_MODEL_OPENAI,
        "cfg_engine_model_ollama": cfg.ENGINE_MODEL_OLLAMA,
        "llm_engine_provider": llm_agents.ENGINE_PROVIDER,
        "llm_engine_model_anthropic": llm_agents.ENGINE_MODEL_ANTHROPIC,
        "llm_engine_model_openai": llm_agents.ENGINE_MODEL_OPENAI,
        "llm_engine_model_ollama": llm_agents.ENGINE_MODEL_OLLAMA,
    }
    try:
        cfg.ENGINE_PROVIDER = bundle["provider"]
        llm_agents.ENGINE_PROVIDER = bundle["provider"]
        if bundle["provider"] == "anthropic":
            cfg.ENGINE_MODEL_ANTHROPIC = bundle["model"]
            llm_agents.ENGINE_MODEL_ANTHROPIC = bundle["model"]
        elif bundle["provider"] == "openai":
            cfg.ENGINE_MODEL_OPENAI = bundle["model"]
            llm_agents.ENGINE_MODEL_OPENAI = bundle["model"]
        else:
            cfg.ENGINE_MODEL_OLLAMA = bundle["model"]
            llm_agents.ENGINE_MODEL_OLLAMA = bundle["model"]
        yield bundle
    finally:
        cfg.ENGINE_PROVIDER = original["cfg_engine_provider"]
        cfg.ENGINE_MODEL_ANTHROPIC = original["cfg_engine_model_anthropic"]
        cfg.ENGINE_MODEL_OPENAI = original["cfg_engine_model_openai"]
        cfg.ENGINE_MODEL_OLLAMA = original["cfg_engine_model_ollama"]
        llm_agents.ENGINE_PROVIDER = original["llm_engine_provider"]
        llm_agents.ENGINE_MODEL_ANTHROPIC = original["llm_engine_model_anthropic"]
        llm_agents.ENGINE_MODEL_OPENAI = original["llm_engine_model_openai"]
        llm_agents.ENGINE_MODEL_OLLAMA = original["llm_engine_model_ollama"]


@contextmanager
def lightweight_mode(mode: str):
    original = cfg.LIGHTWEIGHT_UPSTREAM_MODE
    try:
        cfg.LIGHTWEIGHT_UPSTREAM_MODE = mode
        yield mode
    finally:
        cfg.LIGHTWEIGHT_UPSTREAM_MODE = original


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_root) / f"upstream_mode_comparison_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    case = _case()
    config = {
        "case": asdict(case),
        "budget": args.budget,
        "temperature": args.temperature,
        "seed": args.seed,
        "objectives": args.objectives,
        "allow_warning_refinement": args.allow_warning_refinement,
        "strong_revision_mode": args.strong_revision_mode,
        "branching_revision_mode": args.branching_revision_mode,
        "max_descendants_per_candidate": args.max_descendants_per_candidate,
        "max_total_revised_candidates": args.max_total_revised_candidates,
        "upstream_bundle": args.upstream_bundle,
        "council_bundle": args.council_bundle,
        "modes": args.modes,
    }
    with (experiment_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    case_rows: list[dict] = []

    with upstream_bundle(args.upstream_bundle) as upstream_models, council_bundle(args.council_bundle) as council_cfg:
        for mode in args.modes:
            with lightweight_mode(mode):
                group_dir = experiment_dir / f"mode_{mode}"
                context_dir = group_dir / "contexts" / case.case_id
                prep_recorder = BenchmarkRecorder(
                    context_dir,
                    {
                        "phase": "prepare_context",
                        "case_id": case.case_id,
                        "case_title": case.title,
                        "temperature": args.temperature,
                        "protocol": case.protocol,
                        "upstream_bundle": args.upstream_bundle,
                        "upstream_models": upstream_models,
                        "council_bundle": args.council_bundle,
                        "council_config": council_cfg,
                        "lightweight_upstream_mode": mode,
                    },
                )

                try:
                    context = prepare_case_context(case, prep_recorder, temperature=args.temperature)
                    prep_recorder.finalize(status="completed")
                except Exception as exc:
                    prep_recorder.finalize(status="failed", extra={"error": str(exc)})
                    case_rows.append({
                        "mode": mode,
                        "phase": "prepare_context",
                        "status": "failed",
                        "run_summary_path": str(context_dir / "run_summary.json"),
                        "result_path": "",
                        "error": str(exc),
                    })
                    continue

                run_dir = group_dir / "runs" / case.case_id / f"budget_{args.budget}" / "repeat_01"
                run_recorder = BenchmarkRecorder(
                    run_dir,
                    {
                        "phase": "run_council",
                        "case_id": case.case_id,
                        "case_title": case.title,
                        "protocol": case.protocol,
                        "upstream_bundle": args.upstream_bundle,
                        "upstream_models": upstream_models,
                        "council_bundle": args.council_bundle,
                        "council_config": council_cfg,
                        "candidate_budget": args.budget,
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "lightweight_upstream_mode": mode,
                    },
                )

                try:
                    result = run_council_from_context(
                        context,
                        run_recorder,
                        candidate_budget=args.budget,
                        objectives=args.objectives,
                        allow_warning_refinement=args.allow_warning_refinement,
                        temperature=args.temperature,
                        seed=args.seed,
                        benchmark_strict_scoring=True,
                        benchmark_scoring_batch_size=3,
                        benchmark_claude_compact_mode=args.council_bundle == "claude",
                        benchmark_strong_revision_mode=args.strong_revision_mode,
                        benchmark_branching_revision_mode=args.branching_revision_mode,
                        benchmark_max_descendants_per_candidate=args.max_descendants_per_candidate,
                        benchmark_max_total_revised_candidates=args.max_total_revised_candidates,
                    )
                    final = result["final_design_candidate"]["proposal"]
                    case_rows.append({
                        "mode": mode,
                        "phase": "run_council",
                        "status": "completed",
                        "run_summary_path": str(run_dir / "run_summary.json"),
                        "result_path": str(run_dir / "result.json"),
                        "final_tau_min": final.get("residence_time_min"),
                        "final_flow_rate_mL_min": final.get("flow_rate_mL_min"),
                        "final_tubing_ID_mm": final.get("tubing_ID_mm"),
                        "final_BPR_bar": final.get("BPR_bar"),
                        "final_reactor_volume_mL": final.get("reactor_volume_mL"),
                    })
                except Exception as exc:
                    case_rows.append({
                        "mode": mode,
                        "phase": "run_council",
                        "status": "failed",
                        "run_summary_path": str(run_dir / "run_summary.json"),
                        "result_path": "",
                        "error": str(exc),
                    })

    with (experiment_dir / "comparison_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in case_rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(case_rows)

    summarize_experiment(experiment_dir)
    print(json.dumps({"experiment_dir": str(experiment_dir), "rows": case_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
