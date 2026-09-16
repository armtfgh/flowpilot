#!/usr/bin/env python3
"""Export a verbatim one-shot versus FlowPilot prompt audit package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


FLOWPILOT_STAGE_LABELS = [
    "batch_protocol_extraction",
    "upstream_chemistry_analysis",
    "flow_translation_proposal",
    "council_problem_framing",
    "council_sampling_designer",
    "final_chemist_report",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _event_record(architecture: str, call_index: int, stage: str, event: dict) -> dict:
    usage = event.get("usage") or {}
    system = str(event.get("system_prompt") or "")
    user = str(event.get("user_prompt") or "")
    return {
        "architecture": architecture,
        "call_index": call_index,
        "stage": stage,
        "provider": event.get("provider"),
        "model": event.get("model"),
        "temperature": event.get("temperature"),
        "seed": event.get("seed"),
        "system_chars": len(system),
        "user_chars": len(user),
        "total_prompt_chars": len(system) + len(user),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "duration_ms": event.get("duration_ms"),
        "timestamp": event.get("timestamp"),
    }


def export(one_shot_dir: Path, flowpilot_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    verbatim_dir = output_dir / "verbatim_prompts"
    verbatim_dir.mkdir(exist_ok=True)
    response_dir = output_dir / "verbatim_responses"
    response_dir.mkdir(exist_ok=True)

    one_shot_prompt = json.loads((one_shot_dir / "prompt.json").read_text(encoding="utf-8"))
    one_shot_events = _read_jsonl(one_shot_dir / "llm_events.jsonl")
    flowpilot_events = _read_jsonl(flowpilot_dir / "llm_events.jsonl")
    if len(one_shot_events) != 1:
        raise ValueError(f"Expected one one-shot LLM event, found {len(one_shot_events)}")
    if len(flowpilot_events) != len(FLOWPILOT_STAGE_LABELS):
        raise ValueError(
            "FlowPilot event count does not match the named stage map: "
            f"{len(flowpilot_events)} versus {len(FLOWPILOT_STAGE_LABELS)}"
        )

    public_hashes = {
        "one_shot": _sha256(one_shot_dir / "input_public.json"),
        "flowpilot": _sha256(flowpilot_dir / "input_public.json"),
    }
    inventory_hashes = {
        "one_shot": _sha256(one_shot_dir / "input_inventory.json"),
        "flowpilot": _sha256(flowpilot_dir / "input_inventory.json"),
    }
    if len(set(public_hashes.values())) != 1 or len(set(inventory_hashes.values())) != 1:
        raise ValueError("The paired runs do not have identical public input and inventory")

    one_system = str(one_shot_prompt.get("system") or "")
    one_user = str(one_shot_prompt.get("user") or "")
    _write_text(verbatim_dir / "oneshot_system_prompt.txt", one_system)
    _write_text(verbatim_dir / "oneshot_user_prompt.txt", one_user)

    prompt_payload = {
        "one_shot": {"system_prompt": one_system, "user_prompt": one_user},
        "flowpilot": [],
    }
    one_response = str(one_shot_events[0].get("response_text") or "")
    _write_text(response_dir / "oneshot_model_response.txt", one_response)
    response_payload = {
        "one_shot": {"response_text": one_response},
        "flowpilot": [],
    }
    response_sections = [
        "# Complete Response Comparison: Claude Opus 4.6\n",
        "This document contains the exact model response text recorded for the matched "
        "photochemical oxidation, repeat 01 campaign.\n",
        "## One-Shot: Single Model Response\n",
        "````text\n" + one_response + "\n````\n",
        "## FlowPilot: Model Response Sequence\n",
    ]
    rows = [_event_record("one_shot", 1, "general_one_shot", one_shot_events[0])]
    sections = [
        "# Complete Prompt Comparison: Claude Opus 4.6\n",
        "This document contains the exact system and user prompts recorded for the matched "
        "photochemical oxidation, repeat 01 campaign. Responses are intentionally excluded.\n",
        "## One-Shot: Single LLM Call\n",
        "### System Prompt\n\n````text\n" + one_system + "\n````\n",
        "### User Prompt\n\n````text\n" + one_user + "\n````\n",
        "## FlowPilot: Prompt Sequence\n",
    ]

    for index, (stage, event) in enumerate(zip(FLOWPILOT_STAGE_LABELS, flowpilot_events), 1):
        system = str(event.get("system_prompt") or "")
        user = str(event.get("user_prompt") or "")
        stem = f"flowpilot_{index:02d}_{stage}"
        _write_text(verbatim_dir / f"{stem}_system_prompt.txt", system)
        _write_text(verbatim_dir / f"{stem}_user_prompt.txt", user)
        response = str(event.get("response_text") or "")
        _write_text(response_dir / f"{stem}_response.txt", response)
        prompt_payload["flowpilot"].append(
            {
                "call_index": index,
                "stage": stage,
                "system_prompt": system,
                "user_prompt": user,
            }
        )
        response_payload["flowpilot"].append(
            {"call_index": index, "stage": stage, "response_text": response}
        )
        rows.append(_event_record("flowpilot", index, stage, event))
        sections.extend(
            [
                f"### Call {index}: {stage.replace('_', ' ').title()}\n",
                "#### System Prompt\n\n````text\n" + system + "\n````\n",
                "#### User Prompt\n\n````text\n" + user + "\n````\n",
            ]
        )
        response_sections.extend(
            [
                f"### Call {index}: {stage.replace('_', ' ').title()}\n",
                "````text\n" + response + "\n````\n",
            ]
        )

    _write_text(output_dir / "FULL_PROMPT_COMPARISON.md", "\n".join(sections))
    _write_text(output_dir / "FULL_RESPONSE_COMPARISON.md", "\n".join(response_sections))
    (output_dir / "prompts_exact.json").write_text(
        json.dumps(prompt_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "responses_exact.json").write_text(
        json.dumps(response_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    shutil.copy2(one_shot_dir / "raw_response.json", output_dir / "oneshot_raw_response.json")
    shutil.copy2(one_shot_dir / "result.json", output_dir / "oneshot_parsed_result.json")
    shutil.copy2(flowpilot_dir / "result.json", output_dir / "flowpilot_final_result.json")

    with (output_dir / "prompt_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    one_tokens = sum(row["input_tokens"] for row in rows if row["architecture"] == "one_shot")
    flow_tokens = sum(row["input_tokens"] for row in rows if row["architecture"] == "flowpilot")
    one_output_tokens = sum(
        row["output_tokens"] for row in rows if row["architecture"] == "one_shot"
    )
    flow_output_tokens = sum(
        row["output_tokens"] for row in rows if row["architecture"] == "flowpilot"
    )
    one_chars = len(one_system) + len(one_user)
    flow_chars = sum(
        len(str(event.get("system_prompt") or "")) + len(str(event.get("user_prompt") or ""))
        for event in flowpilot_events
    )
    summary = {
        "schema_version": "flowpilot_prompt_comparison_v1.0",
        "case": "photochemical_oxidation",
        "repeat": "repeat_01",
        "model": one_shot_events[0].get("model"),
        "provider": one_shot_events[0].get("provider"),
        "matched_settings": {
            "temperature": one_shot_events[0].get("temperature"),
            "seed": one_shot_events[0].get("seed"),
            "public_input_identical": True,
            "inventory_identical": True,
            "public_input_sha256": public_hashes["one_shot"],
            "inventory_sha256": inventory_hashes["one_shot"],
        },
        "one_shot": {
            "llm_calls": 1,
            "prompt_characters": one_chars,
            "recorded_input_tokens": one_tokens,
            "recorded_output_tokens": one_output_tokens,
        },
        "flowpilot": {
            "llm_calls": len(flowpilot_events),
            "prompt_characters": flow_chars,
            "recorded_input_tokens": flow_tokens,
            "recorded_output_tokens": flow_output_tokens,
            "stages": FLOWPILOT_STAGE_LABELS,
        },
        "ratios_flowpilot_over_one_shot": {
            "llm_calls": len(flowpilot_events),
            "prompt_characters": round(flow_chars / one_chars, 4),
            "recorded_input_tokens": round(flow_tokens / one_tokens, 4),
        },
        "interpretation": (
            "The matched comparison controls external protocol and inventory input, model, "
            "temperature, and seed. It compares one inference against a staged architecture; "
            "it is not a matched-token comparison."
        ),
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    readme = f"""# Claude Opus 4.6 Prompt Audit

Matched case: photochemical oxidation, repeat 01.

The public protocol/objective input and inventory are byte-identical between architectures.
Both runs used `{summary['provider']}/{summary['model']}`, temperature
`{summary['matched_settings']['temperature']}`, and seed `{summary['matched_settings']['seed']}`.

| Architecture | LLM calls | Prompt characters | Recorded input tokens |
|---|---:|---:|---:|
| One-shot | 1 | {one_chars:,} | {one_tokens:,} |
| FlowPilot | {len(flowpilot_events)} | {flow_chars:,} | {flow_tokens:,} |

FlowPilot used {summary['ratios_flowpilot_over_one_shot']['recorded_input_tokens']:.2f} times the
recorded input tokens. This benchmark is matched by external information and model settings,
not by inference count or token budget.

Files:

- `FULL_PROMPT_COMPARISON.md`: all prompts in reading order.
- `FULL_RESPONSE_COMPARISON.md`: all corresponding model responses in reading order.
- `prompts_exact.json`: machine-readable exact prompt text.
- `responses_exact.json`: machine-readable exact response text.
- `verbatim_prompts/`: one plain-text file per system/user prompt.
- `verbatim_responses/`: one plain-text response per model call.
- `oneshot_raw_response.json`: original one-shot provider response envelope.
- `oneshot_parsed_result.json`: parsed one-shot benchmark result.
- `flowpilot_final_result.json`: complete compiled FlowPilot result.
- `prompt_inventory.csv`: per-call size, token, and runtime metadata.
- `comparison_summary.json`: controls, totals, ratios, and hashes.
"""
    _write_text(output_dir / "README.md", readme)

    checksum_files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
        and not path.name.startswith(".")
        and not path.name.endswith("#")
    )
    checksum_lines = [
        f"{_sha256(path)}  {path.relative_to(output_dir)}" for path in checksum_files
    ]
    _write_text(output_dir / "SHA256SUMS.txt", "\n".join(checksum_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-shot-dir", type=Path, required=True)
    parser.add_argument("--flowpilot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.one_shot_dir, args.flowpilot_dir, args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
