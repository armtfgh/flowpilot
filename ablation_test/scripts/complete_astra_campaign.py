#!/usr/bin/env python3
"""Finish an in-flight generation-only campaign before corrected evaluation."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def active(pid):
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return False
    return path.read_text().split(")", 1)[1].strip().split()[0] not in {"Z", "X"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--supervisor", type=int, required=True)
    parser.add_argument("--workers", type=int, nargs="+", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        while any(active(pid) for pid in args.workers):
            rows = [json.loads(p.read_text()) for p in (output / "generation").glob("*/*/repeat_*/run_summary.json")]
            value = {"phase": "generation", "completed_generation": sum(r.get("status") == "completed" for r in rows),
                     "planned_generation": 18, "valid_judgments": 0, "planned_judgments": 54,
                     "updated_at": datetime.now().astimezone().isoformat()}
            (output / "supervisor_progress.json").write_text(json.dumps(value, indent=2))
            print(json.dumps(value), flush=True)
            time.sleep(60)
    finally:
        # The old coordinator is stopped so it cannot construct legacy packets.
        # Its existing cleanup reaps the finished workers on this interrupt.
        if active(args.supervisor):
            os.kill(args.supervisor, signal.SIGINT)
            os.kill(args.supervisor, signal.SIGCONT)
    with (output / "evaluation.log").open("a") as log:
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).with_name("evaluate_astra_delivered_outputs.py")),
            str(output), "--phase", "all",
        ], stdout=log, stderr=subprocess.STDOUT, check=True)
    (output / "supervisor_progress.json").write_text(json.dumps({"phase": "complete", "outcomes": 18, "judgments": 54}, indent=2))
    print(f"Complete: {output / 'report/ASTRA_RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
