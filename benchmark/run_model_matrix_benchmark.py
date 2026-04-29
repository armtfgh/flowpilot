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
from flora_translate.engine import llm_agents


PROTOCOL = """Phenylacetylene (1a, 0.5 mmol, 1.0 equiv) and ethyl nitroacetate (2, 1.0 mmol, 2.0 equiv) were added directly to TBAB/EG (1:5) DES (1 mL) in an oven-dried 30 mL vial equipped with a magnetic stirring bar; no additional solvent, catalyst, base, or additive was used. The reaction mixture was stirred at 120 °C in an oil bath for 15 min, at which point full conversion (99%) to ethyl 5-phenylisoxazole-3-carboxylate (3a) was achieved via 1,3-dipolar cycloaddition of the in-situ-generated nitrile oxide intermediate with phenylacetylene. After the reaction, the mixture was quenched with water, extracted with dichloromethane, dried over MgSO₄, and the solvent removed under reduced pressure; the isolated NMR yield was 83% (quantified against 1,3,5-trimethoxybenzene as internal standard)."""


UPSTREAM_BUNDLES = {
    "claude": {
        "MODEL_INPUT_PARSER": "claude-sonnet-4-20250514",
        "MODEL_CHEMISTRY_AGENT": "claude-opus-4-6",
        "MODEL_TRANSLATION": "claude-sonnet-4-20250514",
        "MODEL_OUTPUT_FORMATTER": "claude-sonnet-4-20250514",
        "MODEL_REVISION_AGENT": "claude-sonnet-4-20250514",
        "MODEL_CONVERSATION_AGENT": "claude-sonnet-4-20250514",
        "MODEL_EMBEDDING_SUMMARY": "claude-sonnet-4-20250514",
        "MODEL_TOPOLOGY_POLISHER": "claude-haiku-4-5-20251001",
    },
    "gpt4o": {
        "MODEL_INPUT_PARSER": "gpt-4o",
        "MODEL_CHEMISTRY_AGENT": "gpt-4o",
        "MODEL_TRANSLATION": "gpt-4o",
        "MODEL_OUTPUT_FORMATTER": "gpt-4o",
        "MODEL_REVISION_AGENT": "gpt-4o",
        "MODEL_CONVERSATION_AGENT": "gpt-4o",
        "MODEL_EMBEDDING_SUMMARY": "gpt-4o",
        "MODEL_TOPOLOGY_POLISHER": "gpt-4o",
    },
    "gpt4omini": {
        "MODEL_INPUT_PARSER": "gpt-4o-mini",
        "MODEL_CHEMISTRY_AGENT": "gpt-4o-mini",
        "MODEL_TRANSLATION": "gpt-4o-mini",
        "MODEL_OUTPUT_FORMATTER": "gpt-4o-mini",
        "MODEL_REVISION_AGENT": "gpt-4o-mini",
        "MODEL_CONVERSATION_AGENT": "gpt-4o-mini",
        "MODEL_EMBEDDING_SUMMARY": "gpt-4o-mini",
        "MODEL_TOPOLOGY_POLISHER": "gpt-4o-mini",
    },
}

COUNCIL_BUNDLES = {
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "gpt4o": {"provider": "openai", "model": "gpt-4o"},
    "gpt4omini": {"provider": "openai", "model": "gpt-4o-mini"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run upstream x council model-matrix benchmark for the DES isoxazole protocol.")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objectives", default="balanced")
    parser.add_argument("--allow-warning-refinement", action="store_true", default=True)
    parser.add_argument("--strong-revision-mode", action="store_true", default=True)
    parser.add_argument("--branching-revision-mode", action="store_true", default=True)
    parser.add_argument("--max-descendants-per-candidate", type=int, default=3)
    parser.add_argument("--max-total-revised-candidates", type=int, default=16)
    parser.add_argument("--output-root", default="benchmark/data")
    parser.add_argument("--upstream-bundles", nargs="+", choices=sorted(UPSTREAM_BUNDLES.keys()))
    parser.add_argument("--council-bundles", nargs="+", choices=sorted(COUNCIL_BUNDLES.keys()))
    return parser.parse_args()


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="protocol_isoxazole_des_full",
        title="Batch Synthesis of 3,5-Disubstituted Isoxazole in TBAB/EG (1:5) DES",
        protocol=PROTOCOL,
        precedent_level="weak_precedent",
        difficulty="high",
        notes="Model-matrix benchmark for upstream/council combinations on the DES isoxazole protocol.",
        tags=("thermal", "cycloaddition", "des", "user_protocol", "model_matrix"),
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


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_root) / f"model_matrix_benchmark_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    case = _case()
    config = {
        "case": asdict(case),
        "budget": args.budget,
        "repeats": 1,
        "temperature": args.temperature,
        "seed": args.seed,
        "objectives": args.objectives,
        "allow_warning_refinement": args.allow_warning_refinement,
        "strong_revision_mode": args.strong_revision_mode,
        "branching_revision_mode": args.branching_revision_mode,
        "max_descendants_per_candidate": args.max_descendants_per_candidate,
        "max_total_revised_candidates": args.max_total_revised_candidates,
        "upstream_bundles": UPSTREAM_BUNDLES,
        "council_bundles": COUNCIL_BUNDLES,
        "selected_upstream_bundles": args.upstream_bundles,
        "selected_council_bundles": args.council_bundles,
    }
    with (experiment_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    matrix_rows: list[dict] = []

    selected_upstream_bundles = args.upstream_bundles or list(UPSTREAM_BUNDLES.keys())
    selected_council_bundles = args.council_bundles or list(COUNCIL_BUNDLES.keys())

    for upstream_name in selected_upstream_bundles:
        with upstream_bundle(upstream_name) as upstream_models:
            group_dir = experiment_dir / f"U_{upstream_name}"
            context_dir = group_dir / "contexts" / case.case_id
            prep_recorder = BenchmarkRecorder(
                context_dir,
                {
                    "phase": "prepare_context",
                    "case_id": case.case_id,
                    "case_title": case.title,
                    "temperature": args.temperature,
                    "protocol": case.protocol,
                    "upstream_bundle": upstream_name,
                    "upstream_models": upstream_models,
                },
            )
            try:
                context = prepare_case_context(case, prep_recorder, temperature=args.temperature)
                prep_recorder.finalize(status="completed")
            except Exception as exc:
                prep_recorder.finalize(status="failed", extra={"error": str(exc)})
                for council_name in selected_council_bundles:
                    matrix_rows.append({
                        "upstream_bundle": upstream_name,
                        "council_bundle": council_name,
                        "status": "upstream_prepare_failed",
                        "cell_dir": str((experiment_dir / f"U_{upstream_name}" / f"C_{council_name}")),
                        "run_summary_path": "",
                        "result_path": "",
                    })
                print(json.dumps({
                    "upstream_bundle": upstream_name,
                    "phase": "prepare_context",
                    "status": "failed",
                    "error": str(exc),
                    "context_dir": str(context_dir),
                }, ensure_ascii=False))
                continue

            for council_name in selected_council_bundles:
                cell_dir = group_dir / f"C_{council_name}"
                run_dir = cell_dir / "runs" / case.case_id / f"budget_{args.budget}" / "repeat_01"
                benchmark_claude_compact_mode = council_name == "claude"
                with council_bundle(council_name) as council_cfg:
                    recorder = BenchmarkRecorder(
                        run_dir,
                        {
                            "phase": "council_run",
                            "case_id": case.case_id,
                            "case_title": case.title,
                            "candidate_budget": args.budget,
                            "repeat_index": 1,
                            "temperature": args.temperature,
                            "seed": args.seed,
                            "allow_warning_refinement": args.allow_warning_refinement,
                            "benchmark_strong_revision_mode": args.strong_revision_mode,
                            "benchmark_branching_revision_mode": args.branching_revision_mode,
                            "benchmark_max_descendants_per_candidate": args.max_descendants_per_candidate,
                            "benchmark_max_total_revised_candidates": args.max_total_revised_candidates,
                            "benchmark_claude_compact_mode": benchmark_claude_compact_mode,
                            "objectives": args.objectives,
                            "protocol": case.protocol,
                            "upstream_bundle": upstream_name,
                            "council_bundle": council_name,
                            "council_provider": council_cfg["provider"],
                            "upstream_models": upstream_models,
                            "council_model": council_cfg["model"],
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
                            benchmark_claude_compact_mode=benchmark_claude_compact_mode,
                            benchmark_strong_revision_mode=args.strong_revision_mode,
                            benchmark_branching_revision_mode=args.branching_revision_mode,
                            benchmark_max_descendants_per_candidate=args.max_descendants_per_candidate,
                            benchmark_max_total_revised_candidates=args.max_total_revised_candidates,
                        )
                    except Exception as exc:
                        print(json.dumps({
                            "upstream_bundle": upstream_name,
                            "council_bundle": council_name,
                            "status": "failed",
                            "error": str(exc),
                            "run_dir": str(run_dir),
                        }, ensure_ascii=False))

                cell_summary = summarize_experiment(cell_dir)
                run_manifest = cell_dir / "run_manifest.csv"
                if run_manifest.exists():
                    with run_manifest.open("r", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    if rows:
                        row = rows[0]
                        matrix_rows.append({
                            "upstream_bundle": upstream_name,
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
                            "upstream_bundle": upstream_name,
                            "council_bundle": council_name,
                            "status": "no_run_manifest_rows",
                            "cell_dir": str(cell_dir),
                            "run_summary_path": "",
                            "result_path": "",
                        })
                else:
                    matrix_rows.append({
                        "upstream_bundle": upstream_name,
                        "council_bundle": council_name,
                        "status": "no_run_manifest",
                        "cell_dir": str(cell_dir),
                        "run_summary_path": "",
                        "result_path": "",
                    })

    manifest_path = experiment_dir / "matrix_manifest.csv"
    if matrix_rows:
        fieldnames = list(matrix_rows[0].keys())
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matrix_rows)
    summary = {
        "experiment_dir": str(experiment_dir),
        "cell_count": len(matrix_rows),
        "matrix_manifest_csv": str(manifest_path),
    }
    with (experiment_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
