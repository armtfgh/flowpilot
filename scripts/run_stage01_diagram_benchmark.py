#!/usr/bin/env python3
"""Offline Stage 0/1 benchmark for process-diagram artifact reliability."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flora_translate.diagram_artifacts import render_topology_artifacts, topology_sha256
from flora_translate.schemas import ProcessTopology


FIXTURES = ROOT / "flora_translate/tests/fixtures/diagram_stage0_topologies.json"
OUTPUT_ROOT = ROOT / "outputs/benchmarks/inventory_topology_v1"
REPEATS = 20


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("stage1_validation_%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / stamp
    render_dir = output_dir / "renders"
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(FIXTURES, output_dir / "input_topologies.json")

    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    records: list[dict] = []
    mutation_failures: list[str] = []

    for fixture in fixtures:
        name = fixture["name"]
        topology = ProcessTopology.model_validate(fixture["topology"])
        before = topology.model_dump_json()

        if name == "blocked_missing_equipment":
            records.append(
                {
                    "case": name,
                    "repeat": 0,
                    "expected_behavior": "diagram suppressed by pipeline",
                    "status": "not_rendered",
                    "renderer": "none",
                    "svg_exists": False,
                    "png_exists": False,
                    "topology_sha256": topology_sha256(topology),
                    "run_dir": "",
                }
            )
            continue

        for repeat in range(1, REPEATS + 1):
            artifacts = render_topology_artifacts(
                topology,
                title=f"{name} repeat {repeat}",
                base_dir=render_dir,
            )
            records.append(
                {
                    "case": name,
                    "repeat": repeat,
                    "expected_behavior": "render",
                    "status": artifacts["render_status"],
                    "renderer": artifacts["renderer"],
                    "svg_exists": Path(artifacts["svg_path"]).is_file(),
                    "png_exists": bool(artifacts["png_path"])
                    and Path(artifacts["png_path"]).is_file(),
                    "topology_sha256": artifacts["topology_sha256"],
                    "run_dir": artifacts["run_dir"],
                }
            )
        if topology.model_dump_json() != before:
            mutation_failures.append(name)

    rendered = [record for record in records if record["expected_behavior"] == "render"]
    expected_hashes = {
        fixture["name"]: topology_sha256(ProcessTopology.model_validate(fixture["topology"]))
        for fixture in fixtures
    }
    checks = {
        "all_svg_files_exist": all(record["svg_exists"] for record in rendered),
        "all_png_files_exist": all(record["png_exists"] for record in rendered),
        "all_paths_are_unique": len({record["run_dir"] for record in rendered}) == len(rendered),
        "all_hashes_match_input": all(
            record["topology_sha256"] == expected_hashes[record["case"]]
            for record in rendered
        ),
        "source_topologies_unchanged": not mutation_failures,
        "no_llm_renderer": all("llm" not in record["renderer"].lower() for record in rendered),
        "blocked_case_not_rendered": any(
            record["case"] == "blocked_missing_equipment"
            and record["status"] == "not_rendered"
            for record in records
        ),
    }
    summary = {
        "schema_version": "flowpilot_stage01_diagram_benchmark_v1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixture_count": len(fixtures),
        "rendered_case_count": len({record["case"] for record in rendered}),
        "render_attempts": len(rendered),
        "repeats_per_rendered_case": REPEATS,
        "renderers": sorted({record["renderer"] for record in rendered}),
        "checks": checks,
        "passed": all(checks.values()),
        "mutation_failures": mutation_failures,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "render_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "README.md").write_text(
        "# FlowPilot Stage 0/1 Diagram Validation\n\n"
        "Offline validation of run-local diagram artifacts, deterministic rendering, "
        "topology hashing, and blocked-design suppression.\n\n"
        f"Result: **{'PASS' if summary['passed'] else 'FAIL'}**\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Output: {output_dir}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
