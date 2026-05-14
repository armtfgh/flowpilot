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

GEMMA_MODEL = "google/gemma-4-31B-it"
GEMMA_BASE_URL = "http://10.13.24.45:8000/v1"

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
        "LIGHTWEIGHT_UPSTREAM_MODE": "never",
    },
    "gemma": {
        "MODEL_INPUT_PARSER": GEMMA_MODEL,
        "MODEL_CHEMISTRY_AGENT": GEMMA_MODEL,
        "MODEL_TRANSLATION": GEMMA_MODEL,
        "MODEL_OUTPUT_FORMATTER": GEMMA_MODEL,
        "MODEL_REVISION_AGENT": GEMMA_MODEL,
        "MODEL_CONVERSATION_AGENT": GEMMA_MODEL,
        "MODEL_EMBEDDING_SUMMARY": GEMMA_MODEL,
        "MODEL_TOPOLOGY_POLISHER": GEMMA_MODEL,
        "LIGHTWEIGHT_UPSTREAM_MODE": "always",
    },
}

COUNCIL_BUNDLES = {
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "gemma": {"provider": "ollama", "model": GEMMA_MODEL},
}

DEFAULT_CELLS = [
    ("gemma", "gemma"),
    ("gemma", "claude"),
    ("claude", "gemma"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local benchmark mixing vLLM Gemma and Claude across upstream/council.")
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
    return parser.parse_args()


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="protocol_isoxazole_des_full",
        title="Batch Synthesis of 3,5-Disubstituted Isoxazole in TBAB/EG (1:5) DES",
        protocol=PROTOCOL,
        precedent_level="weak_precedent",
        difficulty="high",
        notes="Local benchmark for vLLM Gemma upstream/council combinations against Claude.",
        tags=("thermal", "cycloaddition", "des", "user_protocol", "local_benchmark", "vllm", "gemma"),
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def upstream_bundle(name: str):
    overrides = UPSTREAM_BUNDLES[name]
    model_keys = [
        "MODEL_INPUT_PARSER",
        "MODEL_CHEMISTRY_AGENT",
        "MODEL_TRANSLATION",
        "MODEL_OUTPUT_FORMATTER",
        "MODEL_REVISION_AGENT",
        "MODEL_CONVERSATION_AGENT",
        "MODEL_EMBEDDING_SUMMARY",
        "MODEL_TOPOLOGY_POLISHER",
    ]
    original = {key: getattr(cfg, key) for key in model_keys}
    original["LIGHTWEIGHT_UPSTREAM_MODE"] = cfg.LIGHTWEIGHT_UPSTREAM_MODE
    original["OLLAMA_BASE_URL_cfg"] = cfg.OLLAMA_BASE_URL
    original["OLLAMA_BASE_URL_llm"] = llm_agents.OLLAMA_BASE_URL
    original["ollama_client"] = llm_agents._OLLAMA_CLIENT
    try:
        for key in model_keys:
            setattr(cfg, key, overrides[key])
        cfg.LIGHTWEIGHT_UPSTREAM_MODE = overrides["LIGHTWEIGHT_UPSTREAM_MODE"]
        if name == "gemma":
            cfg.OLLAMA_BASE_URL = GEMMA_BASE_URL
            llm_agents.OLLAMA_BASE_URL = GEMMA_BASE_URL
            llm_agents._OLLAMA_CLIENT = None
        yield overrides
    finally:
        for key in model_keys:
            setattr(cfg, key, original[key])
        cfg.LIGHTWEIGHT_UPSTREAM_MODE = original["LIGHTWEIGHT_UPSTREAM_MODE"]
        cfg.OLLAMA_BASE_URL = original["OLLAMA_BASE_URL_cfg"]
        llm_agents.OLLAMA_BASE_URL = original["OLLAMA_BASE_URL_llm"]
        llm_agents._OLLAMA_CLIENT = original["ollama_client"]


@contextmanager
def council_bundle(name: str):
    bundle = COUNCIL_BUNDLES[name]
    original = {
        "cfg_engine_provider": cfg.ENGINE_PROVIDER,
        "cfg_engine_model_anthropic": cfg.ENGINE_MODEL_ANTHROPIC,
        "cfg_engine_model_openai": cfg.ENGINE_MODEL_OPENAI,
        "cfg_engine_model_ollama": cfg.ENGINE_MODEL_OLLAMA,
        "cfg_ollama_base_url": cfg.OLLAMA_BASE_URL,
        "llm_engine_provider": llm_agents.ENGINE_PROVIDER,
        "llm_engine_model_anthropic": llm_agents.ENGINE_MODEL_ANTHROPIC,
        "llm_engine_model_openai": llm_agents.ENGINE_MODEL_OPENAI,
        "llm_engine_model_ollama": llm_agents.ENGINE_MODEL_OLLAMA,
        "llm_ollama_base_url": llm_agents.OLLAMA_BASE_URL,
        "ollama_client": llm_agents._OLLAMA_CLIENT,
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
            cfg.OLLAMA_BASE_URL = GEMMA_BASE_URL
            llm_agents.OLLAMA_BASE_URL = GEMMA_BASE_URL
            llm_agents._OLLAMA_CLIENT = None
        yield bundle
    finally:
        cfg.ENGINE_PROVIDER = original["cfg_engine_provider"]
        cfg.ENGINE_MODEL_ANTHROPIC = original["cfg_engine_model_anthropic"]
        cfg.ENGINE_MODEL_OPENAI = original["cfg_engine_model_openai"]
        cfg.ENGINE_MODEL_OLLAMA = original["cfg_engine_model_ollama"]
        cfg.OLLAMA_BASE_URL = original["cfg_ollama_base_url"]
        llm_agents.ENGINE_PROVIDER = original["llm_engine_provider"]
        llm_agents.ENGINE_MODEL_ANTHROPIC = original["llm_engine_model_anthropic"]
        llm_agents.ENGINE_MODEL_OPENAI = original["llm_engine_model_openai"]
        llm_agents.ENGINE_MODEL_OLLAMA = original["llm_engine_model_ollama"]
        llm_agents.OLLAMA_BASE_URL = original["llm_ollama_base_url"]
        llm_agents._OLLAMA_CLIENT = original["ollama_client"]


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_root) / f"local_model_benchmark_{stamp}"
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
        "selected_cells": DEFAULT_CELLS,
        "gemma_base_url": GEMMA_BASE_URL,
        "gemma_model": GEMMA_MODEL,
    }
    with (experiment_dir / "benchmark_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    matrix_rows: list[dict] = []
    prepared_contexts: dict[str, object] = {}

    for upstream_name, council_name in DEFAULT_CELLS:
        if upstream_name not in prepared_contexts:
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
                        "gemma_base_url": GEMMA_BASE_URL if upstream_name == "gemma" else None,
                    },
                )
                try:
                    prepared_contexts[upstream_name] = prepare_case_context(case, prep_recorder, temperature=args.temperature)
                    prep_recorder.finalize(status="completed")
                except Exception as exc:
                    prep_recorder.finalize(status="failed", extra={"error": str(exc)})
                    prepared_contexts[upstream_name] = exc

        context_or_exc = prepared_contexts[upstream_name]
        cell_dir = experiment_dir / f"U_{upstream_name}" / f"C_{council_name}"
        if isinstance(context_or_exc, Exception):
            matrix_rows.append(
                {
                    "upstream_bundle": upstream_name,
                    "council_bundle": council_name,
                    "status": "upstream_prepare_failed",
                    "runtime_s": "",
                    "llm_call_count": "",
                    "total_tokens": "",
                    "final_tau_min": "",
                    "final_flow_rate_mL_min": "",
                    "final_tubing_ID_mm": "",
                    "final_BPR_bar": "",
                    "final_reactor_volume_mL": "",
                    "cell_dir": str(cell_dir),
                    "run_summary_path": "",
                    "result_path": "",
                    "llm_event_count": "",
                    "error": str(context_or_exc),
                }
            )
            continue

        with upstream_bundle(upstream_name) as upstream_models, council_bundle(council_name) as council_cfg:
            run_dir = cell_dir / "runs" / case.case_id / f"budget_{args.budget}" / "repeat_01"
            benchmark_claude_compact_mode = council_name == "claude"
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
                    "gemma_base_url": GEMMA_BASE_URL if (upstream_name == "gemma" or council_name == "gemma") else None,
                },
            )
            try:
                result = run_council_from_context(
                    context_or_exc,
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
                summary = _load_json(run_dir / "run_summary.json")
                final = result["final_design_candidate"]["proposal"]
                llm_events = run_dir / "llm_events.jsonl"
                llm_event_count = sum(1 for _ in llm_events.open()) if llm_events.exists() else ""
                matrix_rows.append(
                    {
                        "upstream_bundle": upstream_name,
                        "council_bundle": council_name,
                        "status": "completed",
                        "runtime_s": summary.get("runtime_s", ""),
                        "llm_call_count": summary.get("llm_call_count", ""),
                        "total_tokens": summary.get("token_totals", {}).get("total_tokens", ""),
                        "final_tau_min": final.get("residence_time_min"),
                        "final_flow_rate_mL_min": final.get("flow_rate_mL_min"),
                        "final_tubing_ID_mm": final.get("tubing_ID_mm"),
                        "final_BPR_bar": final.get("BPR_bar"),
                        "final_reactor_volume_mL": final.get("reactor_volume_mL"),
                        "cell_dir": str(cell_dir),
                        "run_summary_path": str(run_dir / "run_summary.json"),
                        "result_path": str(run_dir / "result.json"),
                        "llm_event_count": llm_event_count,
                        "error": "",
                    }
                )
            except Exception as exc:
                matrix_rows.append(
                    {
                        "upstream_bundle": upstream_name,
                        "council_bundle": council_name,
                        "status": "failed",
                        "runtime_s": "",
                        "llm_call_count": "",
                        "total_tokens": "",
                        "final_tau_min": "",
                        "final_flow_rate_mL_min": "",
                        "final_tubing_ID_mm": "",
                        "final_BPR_bar": "",
                        "final_reactor_volume_mL": "",
                        "cell_dir": str(cell_dir),
                        "run_summary_path": str(run_dir / "run_summary.json"),
                        "result_path": "",
                        "llm_event_count": "",
                        "error": str(exc),
                    }
                )

    manifest_path = experiment_dir / "matrix_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)

    summary = {
        "experiment_dir": str(experiment_dir),
        "cell_count": len(matrix_rows),
        "matrix_manifest_csv": str(manifest_path),
    }
    with (experiment_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    summarize_experiment(experiment_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
