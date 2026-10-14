#!/usr/bin/env python3
"""Reproducible Astra benchmark entry point with unmodified-outcome evaluation."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ablation_test.scripts.run_astra_matched_benchmark import configure, campaign, freeze, preflight
from ablation_test.scripts.run_astra_workers import workers
from ablation_test.scripts.evaluate_astra_delivered_outputs import prepare


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory, or resume the same frozen campaign")
    args = parser.parse_args()
    output = args.output.resolve()
    configure()
    cases = campaign.selected_cases()
    plan_path = output / "frozen/execution_plan.json"
    if plan_path.exists():
        prior = {row["case_id"]: row["design_input_sha256"] for row in campaign.read_json(plan_path)["cells"]}
        if prior != {case.case_id: case.design_input_sha256 for _, case in cases}:
            raise RuntimeError("Frozen case inputs changed; use a new campaign directory")
    freeze(output, cases)
    for name in (Path(__file__).name, "evaluate_astra_delivered_outputs.py", "summarize_astra_benchmark.py", "run_astra_workers.py"):
        source = Path(__file__).with_name(name)
        target = output / "frozen/source_code" / name
        if target.exists() and campaign.sha256_file(target) != campaign.sha256_file(source):
            raise RuntimeError(f"Frozen benchmark source differs: {name}")
        if not target.exists():
            shutil.copy2(source, target)
    preflight(output)
    codes = workers(output, "generation", range(3))
    campaign.write_json(output / "generation_worker_exit_codes.json", codes)
    amendment = output / "frozen/amendments/delivered_output_evaluation.json"
    if not amendment.exists():
        prepare(output)
    script = str(Path(__file__).with_name("evaluate_astra_delivered_outputs.py"))
    for phase in ("judge", "report"):
        subprocess.run([sys.executable, "-u", script, str(output), "--phase", phase], check=True)


if __name__ == "__main__":
    main()
