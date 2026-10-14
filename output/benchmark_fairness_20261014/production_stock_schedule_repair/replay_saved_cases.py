"""Recompile only executable procedures from frozen designs; no LLM or scoring."""

from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from flora_translate.executable_artifacts import compile_executable_artifacts
from flora_translate.schemas import ExecutableProcessGraph

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "output/benchmark_fairness_20261014/schema_routing_followup/campaign/generation"
summary = {"analysis": "offline procedure replay only; unscored; zero model calls", "cases": {}}
for case in ("hydrogenolysis", "photochemical_oxidation", "cuaac"):
    source = SOURCE / case / "claude_flowpilot/repeat_01/result.json"
    source_hash = sha256(source.read_bytes()).hexdigest()
    result = json.loads(source.read_text())
    before_result = json.dumps(result, sort_keys=True)
    final = result["final_design"]
    graph = ExecutableProcessGraph.model_validate(final["process_graph"])
    bundle = compile_executable_artifacts(
        result, process_graph=graph, parameters=final["parameters"], streams=final["streams"]
    )
    new_steps = [step.model_dump() for step in bundle.operating_procedure]
    old_prep = {step["step_id"]: step for step in final["operating_procedure"] if step["step_id"].startswith("PREP-")}
    new_prep = {step["step_id"]: step for step in new_steps if step["step_id"].startswith("PREP-")}
    (OUT / f"{case}_procedure.json").write_text(json.dumps(new_steps, indent=2) + "\n")
    comparison = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": source_hash,
        "source_unchanged": sha256(source.read_bytes()).hexdigest() == source_hash,
        "input_result_unchanged": json.dumps(result, sort_keys=True) == before_result,
        "before_preparation": old_prep,
        "after_preparation": new_prep,
        "ordered_procedure_ids": [step["step_id"] for step in new_steps],
    }
    for step in new_prep.values():
        values = step["parameter_bindings"]
        if "minimum_feed_stock_volume_mL" in values:
            required = values["flow_rate_mL_min"] * values["minimum_scheduled_feed_duration_min"]
            assert abs(values["recipe_basis_volume_mL"] - required) < 1e-9
            assert values["additional_feed_volume_mL"] is None
            assert values["stock_availability_confirmed"] is False
    summary["cases"][case] = comparison

summary["compiler_sha256"] = sha256((ROOT / "flora_translate/executable_artifacts.py").read_bytes()).hexdigest()
(OUT / "replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({case: {
    step_id: step["parameter_bindings"].get("minimum_feed_stock_volume_mL")
    for step_id, step in data["after_preparation"].items()
} for case, data in summary["cases"].items()}, indent=2))
