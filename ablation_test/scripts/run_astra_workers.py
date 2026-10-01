#!/usr/bin/env python3
"""Run the frozen Astra cells in isolated processes, with periodic checkpoints."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ablation_test.scripts.run_astra_matched_benchmark import (
    DEFAULT_OUTPUT, campaign, configure, freeze, preflight,
)


def progress(output, phase):
    runs = list((output / "generation").glob("*/*/repeat_*/run_summary.json"))
    completed = sum(json.loads(p.read_text()).get("status") == "completed" for p in runs)
    judgments = list((output / "judgments").glob("*/*/status.json"))
    valid = sum(json.loads(p.read_text()).get("status") == "valid" for p in judgments)
    value = {"phase": phase, "completed_generation": completed, "planned_generation": 18,
             "valid_judgments": valid, "planned_judgments": 54,
             "updated_at": datetime.now().astimezone().isoformat()}
    campaign.write_json(output / "supervisor_progress.json", value)
    print(json.dumps(value), flush=True)


def workers(output, mode, items):
    processes, handles = [], []
    try:
        for item in items:
            handle = (output / f"worker_{mode}_{item}.log").open("a", buffering=1)
            handles.append(handle)
            processes.append(subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(output),
                 "--worker", mode, "--item", str(item)],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            ))
        while any(p.poll() is None for p in processes):
            progress(output, mode)
            time.sleep(60)
        progress(output, mode)
        return [p.returncode for p in processes]
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", choices=("generation", "judging"))
    parser.add_argument("--item")
    args = parser.parse_args()
    output = args.output.resolve()
    configure()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cases = campaign.selected_cases()
    if args.worker == "generation":
        campaign.run_generation(output, [cases[int(args.item)]], ("astra",), 1)
        return
    if args.worker == "judging":
        campaign.run_judges(output, (args.item,), 1)
        return

    output.mkdir(parents=True, exist_ok=True)
    freeze(output, cases)
    source = Path(__file__).resolve()
    target = output / "frozen/source_code" / source.name
    if target.exists() and campaign.sha256_file(source) != campaign.sha256_file(target):
        raise RuntimeError("Supervisor source differs from its frozen copy")
    if not target.exists():
        shutil.copy2(source, target)
    campaign.write_json(output / "frozen/execution_schedule.json", {
        "generation_workers": 3, "partition": "one process per chemistry",
        "judge_workers": 3, "judge_partition": "one process per judge model",
        "sampling_or_scoring_changes": False,
        "supervisor_sha256": campaign.sha256_file(source),
    })
    preflight(output)
    codes = workers(output, "generation", range(3))
    campaign.write_json(output / "generation_worker_exit_codes.json", codes)
    # Failed generation cells remain scoreable missing-design outcomes.
    campaign.build_packets(output, cases)
    codes = workers(output, "judging", campaign.JUDGES)
    campaign.write_json(output / "judge_worker_exit_codes.json", codes)
    campaign.build_repeated_report(output, output / "report")
    from ablation_test.scripts.summarize_astra_benchmark import summarize
    summarize(output)
    progress(output, "complete")


if __name__ == "__main__":
    main()
