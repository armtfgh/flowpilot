#!/usr/bin/env python3
"""Isolated, resumable Astra extension of the existing architecture benchmark."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_test.scripts import run_manuscript_three_model_repeated_benchmark as campaign
from ablation_test.scripts.run_newgen_2_0_benchmark import JUDGES
from ablation_test.src.providers import activate_bundle, endpoint_health
from flora_translate.engine import llm_agents as llm

DEFAULT_OUTPUT = ROOT / "ablation_results/manuscript_benchmark/astra_matched_streaming_20261001"


def configure():
    campaign.MODELS = {"astra": {
        "provider": "openai", "model": "gpt-6-astra", "upstream_mode": "never",
        "family": "openai", "display": "GPT-6 Astra",
    }}
    campaign.REPEAT_IDS = ("repeat_01", "repeat_02", "repeat_03")
    campaign.JUDGES = ("qwen", "openai", "claude")
    campaign.TEMPERATURE = None
    campaign.CANDIDATE_BUDGET = 2
    campaign.SEED_NAMESPACE = "astra-matched-20261001-v1"
    campaign.MATCHED_SEED_ACROSS_ARCHITECTURES = False


def freeze(output, cases):
    campaign.initialize(output, cases)
    frozen = output / "frozen"
    manifest_path = frozen / "campaign_manifest.json"
    manifest = campaign.read_json(manifest_path)
    manifest.update({
        "benchmark_name": "GPT-6 Astra matched-model architecture comparison",
        "independent_call_policy": "Three fresh calls per case and architecture; no seed or temperature sent. Recorded seed values are bookkeeping only.",
        "api": "responses", "streaming": True, "reasoning_effort": "medium",
        "minimum_max_output_tokens_per_call": 16384,
        "compute_matched": False,
        "judge_models": JUDGES,
        "historical_comparison_caveat": "Current production code and current judge panel, not a frozen historical FlowPilot binary. Do not pool with older campaigns without qualifying these differences.",
    })
    campaign.write_json(manifest_path, manifest)
    (frozen / "PREREGISTRATION.md").write_text(
        "# GPT-6 Astra: Matched Model, Different Architectures\n\n"
        "Three previously fixed cases (hydrogenolysis, photochemical oxidation, CuAAC), "
        "two architectures, three fresh repeats: 18 outcomes and 54 judgments. "
        "Cases are carried forward from the earlier pilot; they are not a random sample "
        "of flow chemistry and are not guaranteed absent from model training.\n\n"
        "Both architectures receive the same authoritative protocol, inventory, objective, "
        "and constraints. FlowPilot additionally has its production retrieval, rules, "
        "engineering tools, and council; this is not an equal-compute comparison. "
        "The held-out source records are excluded from FlowPilot retrieval.\n\n"
        "Astra uses Responses API, medium reasoning, and an output budget of at least "
        "16,384 tokens per call including reasoning. No temperature or seed is applied. "
        "FlowPilot's candidate budget is 2, matching this existing benchmark configuration, "
        "not the unrestricted GUI's candidate count. All generation roles use Astra.\n\n"
        "Use the unchanged 14-criterion, integer 0-4 anchored rubric. Each judge's mean "
        "over applicable criteria is divided by four, then the three judge scores are "
        "averaged. Critical flags remain separately reported. Judges are Qwen3.6-27B, "
        "GPT-4o, and Claude Sonnet 4.6 in separate calls, with model/architecture labels "
        "withheld (perfect blinding is not claimed). Also report the score excluding "
        "the OpenAI-family judge. These are evaluations, not wet-lab validation.\n\n"
        "Retain blocked designs and every failed attempt. Existing runner permits up "
        "to three attempts only on generation failure, never on low score. Do not "
        "modify pipeline or rubric after inspecting outcomes. All attempts contribute "
        "to usage totals. SD within each case is across three repeats, not across judges. "
        "Pooled outcome SD mixes chemistry and repeat variation. Pairwise confidence "
        "intervals from the legacy report are descriptive, not evidence of population-wide superiority.\n",
        encoding="utf-8",
    )
    for source in (Path(__file__), ROOT / "flora_translate/tests/test_astra_provider.py",
                   ROOT / "ablation_test/src/providers.py"):
        target = frozen / "source_code" / source.name
        if target.exists() and campaign.sha256_file(target) != campaign.sha256_file(source):
            raise RuntimeError(f"Frozen source changed: {source.name}")
        if not target.exists():
            shutil.copy2(source, target)
    campaign.write_json(frozen / "astra_source_checksums.json", {
        p.name: campaign.sha256_file(p) for p in sorted((frozen / "source_code").glob("*.py"))
    })


def preflight(output):
    path = output / "preflight.json"
    if path.exists() and campaign.read_json(path).get("status") == "passed":
        print("PREFLIGHT already passed", flush=True)
        return
    health = {"astra": endpoint_health(campaign.MODELS["astra"]), **{
        f"judge_{name}": endpoint_health(bundle) for name, bundle in JUDGES.items()
    }}
    campaign.write_json(output / "endpoint_health.json", health)
    if any(not row.get("reachable") or not row.get("model_advertised") for row in health.values()):
        raise RuntimeError("Endpoint check failed; see endpoint_health.json")
    events = []
    llm.set_llm_observer(events.append)
    llm.set_llm_runtime_overrides(capture_content=True)
    try:
        with activate_bundle(campaign.MODELS["astra"]):
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
            structured = llm.call_model_text(
                model="gpt-6-astra", provider="openai", system="Return JSON.",
                user_content='Return {"ok":true}.', max_tokens=256,
                json_schema=schema, api_name="astra_schema_preflight",
            )
            if json.loads(structured.text) != {"ok": True}:
                raise RuntimeError("Structured output probe failed")
            tools = [{"name": "multiply", "description": "Multiply x by two.",
                      "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"], "additionalProperties": False}}]
            text, calls = llm.call_llm_with_tools(
                "Use the multiply tool once, then report the answer.",
                "Call multiply with x=2. Do not calculate it without the tool.", tools,
                lambda name, args: {"answer": args["x"] * 2}, max_tokens=256,
            )
            if not calls or calls[0]["result"] != {"answer": 4}:
                raise RuntimeError("Live council tool round trip failed")
            campaign.write_json(path, {"status": "passed", "model": "gpt-6-astra",
                                      "structured_output": structured.text,
                                      "tool_answer": text, "tool_calls": calls})
    finally:
        llm.clear_llm_observer()
        llm.clear_llm_runtime_overrides()
        campaign.write_json(output / "preflight_llm_events.json", events)
    print("PREFLIGHT passed: structured output and council tool round trip", flush=True)


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.terminal.write(text)

    def flush(self):
        self.log.flush()
        self.terminal.flush()


def execute(args):
    configure()
    output = args.output.resolve()
    cases = campaign.selected_cases()
    freeze(output, cases)
    print(f"Campaign: {output}", flush=True)
    if args.phase == "init":
        return
    if args.phase in {"all", "preflight", "generate"}:
        preflight(output)
    if args.phase in {"all", "generate"}:
        try:
            campaign.run_generation(output, cases, ("astra",), 1)
        except RuntimeError:
            if not (output / "generation_failures.json").exists():
                raise
            print("Generation failures retained for evaluation, not excluded", flush=True)
    if args.phase in {"all", "packets"}:
        campaign.build_packets(output, cases)
    if args.phase in {"all", "judge"}:
        campaign.run_judges(output, campaign.JUDGES, 1)
    if args.phase in {"all", "report"}:
        campaign.build_repeated_report(output, output / "report")
    print(f"Phase {args.phase} finished", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("init", "preflight", "generate", "packets", "judge", "report", "all"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
            try:
                execute(args)
            except Exception:
                logging.exception("Astra campaign phase failed; all artifacts retained")
                raise


if __name__ == "__main__":
    main()
