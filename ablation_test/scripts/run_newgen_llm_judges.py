"""Run blinded Qwen/OpenAI/Claude judging for the NewGen candidate set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.src.llm_judge import (
    absolute_prompt,
    absolute_response_schema,
    blind_candidate_id,
    blind_value,
    build_assurance_view,
    build_outcome_view,
    pairwise_prompt,
    pairwise_response_schema,
    parse_json_object,
    read_json,
    validate_absolute_response,
    validate_pairwise_response,
    write_json,
)
from ablation_test.src.providers import activate_bundle, endpoint_health
from flora_translate.engine.llm_agents import (
    call_model_text,
    clear_llm_observer,
    clear_llm_runtime_overrides,
    set_llm_observer,
    set_llm_runtime_overrides,
)


RUBRIC_PATH = ROOT / "ablation_test" / "benchmarks" / "newgen_llm_judge_rubric_v1.json"
SOURCE_MANIFEST = (
    ROOT
    / "deliverables"
    / "flowpilot_newgen_qwen_openai_20260813"
    / "source_data"
    / "run_metrics.csv"
)
DEFAULT_PARENT = ROOT / "ablation_results" / "newgen_benchmark"

JUDGES = {
    "qwen": {
        "provider": "ollama",
        "model": "/models/Qwen3.6-27B",
        "base_url": "http://10.13.24.169:8000/v1",
        "upstream_mode": "always",
        "family": "qwen",
        "label": "Qwen 27B judge",
    },
    "openai": {
        "provider": "openai",
        "model": "gpt-5.4-2026-03-05",
        "upstream_mode": "never",
        "family": "openai",
        "label": "GPT-5.4 judge",
    },
    "claude": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "upstream_mode": "never",
        "family": "anthropic",
        "label": "Claude Sonnet 4.6 judge",
    },
}

CASE_ORACLES = {
    "CuAAC": ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_cuaac" / "hidden_oracle.json",
    "Hydrogenolysis": ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_hydrogenolysis" / "hidden_oracle.json",
    "Two-stage amidation": ROOT / "ablation_test" / "benchmarks" / "newgen_benchmark_v1_pilot_multistep" / "hidden_oracle.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_id(value: str) -> str:
    return re_sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def re_sub(pattern: str, replacement: str, value: str) -> str:
    import re

    return re.sub(pattern, replacement, value)


def _case_packet(row: dict[str, str]) -> dict[str, Any]:
    run_dir = Path(row["run_directory"])
    public = read_json(run_dir / "input_public.json")
    oracle = read_json(CASE_ORACLES[row["case"]])
    return blind_value({
        "case_label": row["case"],
        "protocol": public.get("protocol"),
        "objective": public.get("objective"),
        "hard_constraints": public.get("hard_constraints") or [],
        "authoritative_inventory": public.get("inventory") or read_json(run_dir / "input_inventory.json"),
        "held_out_source_fact_check": {
            "batch_reference": oracle.get("batch_reference") or {},
            "published_flow_reference": oracle.get("published_flow_reference") or {},
            "interpretation": oracle.get("interpretation"),
        },
        "source_use_rule": "The published flow reference is a fact-check anchor, not the only permissible valid design. Score physical and protocol validity rather than exact copying.",
    }, max_string=2400, max_list=50)


def build_packets(output: Path) -> None:
    frozen = output / "frozen"
    packets = output / "packets"
    frozen.mkdir(parents=True, exist_ok=True)
    packets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUBRIC_PATH, frozen / "rubric.json")
    shutil.copy2(SOURCE_MANIFEST, frozen / "source_candidate_manifest.csv")
    rubric = read_json(RUBRIC_PATH)
    rows = list(csv.DictReader(SOURCE_MANIFEST.open(encoding="utf-8")))

    case_packets: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    blinding: list[dict[str, Any]] = []
    for row in rows:
        if row["case"] not in case_packets:
            case_packets[row["case"]] = _case_packet(row)
            write_json(packets / "cases" / f"{_safe_id(row['case'])}.json", case_packets[row["case"]])
        run_dir = Path(row["run_directory"])
        result = read_json(run_dir / "result.json")
        candidate_id = blind_candidate_id(str(run_dir))
        packet = {
            "schema_version": "blinded_judge_candidate_v1.0",
            "candidate_id": candidate_id,
            "case_label": row["case"],
            "outcome_view": build_outcome_view(result),
            "assurance_view": build_assurance_view(result, run_dir),
        }
        write_json(packets / "candidates" / f"{candidate_id}.json", packet)
        candidates.append({"candidate_id": candidate_id, "case_label": row["case"]})
        blinding.append({
            "candidate_id": candidate_id,
            "case": row["case"],
            "case_id": row.get("case_id") or "",
            "generator_model": row["model"],
            "generator_family": "qwen" if "qwen" in row["model"].lower() else "openai",
            "architecture": row["architecture"],
            "run_directory": str(run_dir),
            "result_sha256": _sha256(run_dir / "result.json"),
        })

    randomizer = random.Random(20260814)
    pairs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in blinding:
        grouped.setdefault((item["generator_model"], item["case"]), []).append(item)
    for index, ((model, case), items) in enumerate(sorted(grouped.items()), start=1):
        by_arch = {item["architecture"]: item for item in items}
        one_shot = by_arch["One-shot"]
        flowpilot = by_arch["FlowPilot"]
        first = [one_shot, flowpilot]
        randomizer.shuffle(first)
        pair_id = f"P-{index:02d}-{hashlib.sha256(f'{model}:{case}'.encode()).hexdigest()[:5].upper()}"
        pairs.append({
            "pair_id": pair_id,
            "generator_model": model,
            "generator_family": one_shot["generator_family"],
            "case": case,
            "orders": [
                {"order": 1, "A": first[0]["candidate_id"], "B": first[1]["candidate_id"]},
                {"order": 2, "A": first[1]["candidate_id"], "B": first[0]["candidate_id"]},
            ],
        })

    write_json(frozen / "blinding_key_confidential.json", {"candidates": blinding, "pairs": pairs})
    write_json(frozen / "campaign_manifest.json", {
        "schema_version": "flowpilot_newgen_llm_judge_campaign_v1.0",
        "created_at": datetime.now().isoformat(),
        "rubric_sha256": _sha256(frozen / "rubric.json"),
        "source_manifest_sha256": _sha256(frozen / "source_candidate_manifest.csv"),
        "candidate_count": len(candidates),
        "pair_count": len(pairs),
        "judges": {
            key: {name: value for name, value in bundle.items() if name != "base_url"}
            for key, bundle in JUDGES.items()
        },
        "absolute_repeats": rubric["absolute_judging"]["repeats_per_judge"],
        "pairwise_orders": rubric["pairwise_judging"]["orders_per_pair_judge"],
        "tracks": list(rubric["tracks"]),
        "manual_score_adjustment": False,
    })


def check_health(output: Path, selected_judges: list[str]) -> dict[str, Any]:
    health = {judge: endpoint_health(JUDGES[judge]) for judge in selected_judges}
    write_json(output / "endpoint_health.json", health)
    failures = [judge for judge, value in health.items() if not value.get("reachable") or not value.get("model_advertised")]
    if failures:
        raise RuntimeError(f"Judge endpoints unavailable: {failures}; see endpoint_health.json")
    return health


def _call_judge(
    *,
    bundle: dict[str, Any],
    schema: dict[str, Any],
    system: str,
    user: str,
    seed: int,
    api_name: str,
    call_dir: Path,
    validator,
    force_attempt: bool = False,
) -> dict[str, Any]:
    if (call_dir / "status.json").is_file():
        # Preserve every original response. A malformed/failed cell may be retried
        # only in a numbered child folder so the complete attempt history remains.
        original_dir = call_dir
        original = read_json(original_dir / "status.json")
        if original.get("status") == "valid" and not force_attempt:
            return original
        prior_attempts = sorted((original_dir / "attempts").glob("attempt_*"))
        if not force_attempt:
            for attempt_dir in reversed(prior_attempts):
                attempt_status = attempt_dir / "status.json"
                if attempt_status.is_file() and read_json(attempt_status).get("status") == "valid":
                    return read_json(attempt_status)
        call_dir = original_dir / "attempts" / f"attempt_{len(prior_attempts) + 2}"
        if (call_dir / "status.json").is_file():
            return read_json(call_dir / "status.json")
    call_dir.mkdir(parents=True, exist_ok=True)
    write_json(call_dir / "prompt.json", {"system": system, "user": user, "json_schema": schema, "seed": seed})
    events: list[dict[str, Any]] = []
    set_llm_observer(events.append)
    set_llm_runtime_overrides(
        temperature=0.0,
        seed=seed,
        json_mode=True,
        json_schema=schema,
        capture_content=False,
    )
    started = time.perf_counter()
    try:
        with activate_bundle(bundle):
            response = call_model_text(
                model=bundle["model"],
                provider=bundle["provider"],
                api_name=api_name,
                max_tokens=5000,
                system=system,
                user_content=user,
            )
        raw = {
            "provider": response.provider,
            "model": response.model,
            "usage": response.usage,
            "stop_reason": response.stop_reason,
            "finish_reason": response.finish_reason,
            "text": response.text,
        }
        write_json(call_dir / "raw_response.json", raw)
        write_json(call_dir / "telemetry.json", events)
        try:
            parsed = parse_json_object(response.text)
            errors = validator(parsed)
        except Exception as exc:
            parsed = None
            errors = [f"{type(exc).__name__}: {exc}"]
        if parsed is not None:
            write_json(call_dir / "parsed_response.json", parsed)
        status = {
            "status": "valid" if not errors else "invalid",
            "validation_errors": errors,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "provider": bundle["provider"],
            "model": bundle["model"],
            "seed": seed,
        }
        write_json(call_dir / "status.json", status)
        return status
    except Exception as exc:
        status = {
            "status": "request_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "provider": bundle["provider"],
            "model": bundle["model"],
            "seed": seed,
        }
        write_json(call_dir / "status.json", status)
        return status
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()


def run_absolute(
    output: Path,
    selected_judges: list[str],
    repeats: int,
    *,
    tracks: list[str] | None = None,
    candidate_offset: int = 0,
    candidate_limit: int | None = None,
    force_attempt: bool = False,
) -> dict[str, int]:
    rubric = read_json(output / "frozen" / "rubric.json")
    candidates = [read_json(path) for path in sorted((output / "packets" / "candidates").glob("*.json"))]
    candidates = candidates[candidate_offset:]
    if candidate_limit is not None:
        candidates = candidates[:candidate_limit]
    selected_tracks = tracks or list(rubric["tracks"])
    case_packets = {
        read_json(path)["case_label"]: read_json(path)
        for path in sorted((output / "packets" / "cases").glob("*.json"))
    }
    counts = {"valid": 0, "invalid": 0, "request_failed": 0}
    total = len(candidates) * len(selected_judges) * repeats * len(selected_tracks)
    current = 0
    for judge_index, judge in enumerate(selected_judges, start=1):
        bundle = JUDGES[judge]
        for track_index, track in enumerate(selected_tracks, start=1):
            schema = absolute_response_schema(rubric, track)
            for candidate_index, candidate in enumerate(candidates, start=1):
                system, user = absolute_prompt(rubric, track, case_packets[candidate["case_label"]], candidate)
                for repeat in range(1, repeats + 1):
                    current += 1
                    seed = 2026081400 + judge_index * 100000 + track_index * 10000 + candidate_index * 100 + repeat
                    call_dir = output / "judgments" / "absolute" / judge / track / candidate["candidate_id"] / f"repeat_{repeat}"
                    status = _call_judge(
                        bundle=bundle,
                        schema=schema,
                        system=system,
                        user=user,
                        seed=seed,
                        api_name=f"newgen_judge_absolute_{track}",
                        call_dir=call_dir,
                        validator=lambda value, t=track, c=candidate: validate_absolute_response(value, rubric, t, c["candidate_id"]),
                        force_attempt=force_attempt,
                    )
                    counts[status["status"]] = counts.get(status["status"], 0) + 1
                    print(f"absolute {current}/{total} {judge}/{track}/{candidate['candidate_id']}/r{repeat}: {status['status']}", flush=True)
    judge_suffix = "_".join(selected_judges)
    shard_suffix = f"offset{candidate_offset}_limit{candidate_limit if candidate_limit is not None else 'all'}"
    write_json(output / f"absolute_execution_summary_{judge_suffix}_{shard_suffix}.json", counts)
    return counts


def run_pairwise(
    output: Path,
    selected_judges: list[str],
    *,
    tracks: list[str] | None = None,
    pair_offset: int = 0,
    pair_limit: int | None = None,
    force_attempt: bool = False,
) -> dict[str, int]:
    rubric = read_json(output / "frozen" / "rubric.json")
    key = read_json(output / "frozen" / "blinding_key_confidential.json")
    pairs = key["pairs"][pair_offset:]
    if pair_limit is not None:
        pairs = pairs[:pair_limit]
    selected_tracks = tracks or list(rubric["tracks"])
    candidates = {
        path.stem: read_json(path)
        for path in sorted((output / "packets" / "candidates").glob("*.json"))
    }
    case_packets = {
        read_json(path)["case_label"]: read_json(path)
        for path in sorted((output / "packets" / "cases").glob("*.json"))
    }
    counts = {"valid": 0, "invalid": 0, "request_failed": 0}
    total = len(pairs) * len(selected_judges) * 2 * len(selected_tracks)
    current = 0
    for judge_index, judge in enumerate(selected_judges, start=1):
        bundle = JUDGES[judge]
        for track_index, track in enumerate(selected_tracks, start=1):
            schema = pairwise_response_schema(rubric, track)
            for pair_index, pair in enumerate(pairs, start=1):
                for order_data in pair["orders"]:
                    current += 1
                    order = int(order_data["order"])
                    system, user = pairwise_prompt(
                        rubric,
                        track,
                        case_packets[pair["case"]],
                        pair["pair_id"],
                        candidates[order_data["A"]],
                        candidates[order_data["B"]],
                    )
                    seed = 2026081450 + judge_index * 100000 + track_index * 10000 + pair_index * 100 + order
                    call_dir = output / "judgments" / "pairwise" / judge / track / pair["pair_id"] / f"order_{order}"
                    status = _call_judge(
                        bundle=bundle,
                        schema=schema,
                        system=system,
                        user=user,
                        seed=seed,
                        api_name=f"newgen_judge_pairwise_{track}",
                        call_dir=call_dir,
                        validator=lambda value, t=track, p=pair: validate_pairwise_response(value, rubric, t, p["pair_id"]),
                        force_attempt=force_attempt,
                    )
                    counts[status["status"]] = counts.get(status["status"], 0) + 1
                    print(f"pairwise {current}/{total} {judge}/{track}/{pair['pair_id']}/o{order}: {status['status']}", flush=True)
    judge_suffix = "_".join(selected_judges)
    shard_suffix = f"offset{pair_offset}_limit{pair_limit if pair_limit is not None else 'all'}"
    write_json(output / f"pairwise_execution_summary_{judge_suffix}_{shard_suffix}.json", counts)
    return counts


def _latest_or_new(path: Path | None) -> Path:
    if path:
        return path.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_PARENT / f"newgen_llm_judge_v1_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("packets", "health", "absolute", "pairwise", "all"), default="all")
    parser.add_argument("--judges", nargs="+", choices=tuple(JUDGES), default=list(JUDGES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tracks", nargs="+", choices=("outcome", "assurance"), default=None)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--pair-offset", type=int, default=0)
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--force-attempt", action="store_true")
    args = parser.parse_args()
    output = _latest_or_new(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"packets", "all"} or not (output / "frozen" / "campaign_manifest.json").is_file():
        build_packets(output)
    if args.phase in {"health", "absolute", "pairwise", "all"} and not args.skip_health:
        check_health(output, args.judges)
    if args.phase in {"absolute", "all"}:
        run_absolute(
            output,
            args.judges,
            args.repeats,
            tracks=args.tracks,
            candidate_offset=args.candidate_offset,
            candidate_limit=args.candidate_limit,
            force_attempt=args.force_attempt,
        )
    if args.phase in {"pairwise", "all"}:
        run_pairwise(
            output,
            args.judges,
            tracks=args.tracks,
            pair_offset=args.pair_offset,
            pair_limit=args.pair_limit,
            force_attempt=args.force_attempt,
        )
    print(output.resolve())


if __name__ == "__main__":
    main()
