"""Submit a fresh model run with a saved intake and its exact inventory snapshot."""

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8512")
    parser.add_argument("--upstream", default="qwen3.6-27b")
    parser.add_argument("--downstream", default="qwen3.6-27b")
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    package = source["intake_package"]
    request = {
        "batch_input": package["raw_protocol"], "intake_package": package,
        "inventory_profile": package["inventory_profile_snapshot"],
        "upstream_model_id": args.upstream, "downstream_model_id": args.downstream,
        "runtime_options": {"candidate_budget": args.candidates},
    }
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")

    def log(message):
        print(message, flush=True)
        with (args.output / "monitor.log").open("a") as handle:
            handle.write(message + "\n")

    save("request.json", request)
    url = args.base_url.rstrip("/") + "/api/design/jobs"
    with urlopen(Request(url, data=json.dumps(request).encode(), headers={"Content-Type": "application/json"}), timeout=120) as response:
        job = json.load(response)
    log(f"New model job: {job['job_id']}")
    start = time.monotonic()
    while True:
        save("job.json", job)
        log(f"{time.monotonic() - start:.0f}s: {job['status']} / {job['phase']}")
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(30)
        with urlopen(url + "/" + job["job_id"], timeout=120) as response:
            job = json.load(response)
    result = job.get("result") or {}
    save("result.json", result)
    summary = {
        "job_id": job["job_id"], "autosave_dir": job.get("autosave_dir"),
        "status": result.get("final_design", {}).get("status", job["status"]),
        "issues": result.get("final_design", {}).get("consistency", {}).get("issues", []),
        "error": job.get("error"), "new_generation": True,
        "upstream": args.upstream, "downstream": args.downstream,
    }
    save("summary.json", summary)
    log(json.dumps(summary))
    return 0 if summary["status"] == "executable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
