"""Replay the archived KHU inventory gap, optionally run a real local-model recovery."""

import argparse
import json
from pathlib import Path
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8512")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", default="outputs/validation/20260907_inventory_resolution")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    def save(name, data):
        (output / name).write_text(json.dumps(data, indent=2, default=str) + "\n")

    def log(message):
        print(message, flush=True)
        with (output / "validation.log").open("a") as handle:
            handle.write(message + "\n")

    with httpx.Client(base_url=args.base_url, timeout=120, trust_env=False) as client:
        def post(path, payload):
            response = client.post(path, json=payload)
            response.raise_for_status()
            return response.json()

        archived = json.loads(Path("outputs/gui_runs/20260907_094109_webapp/result.json").read_text())
        package = archived["intake_package"]
        before = post("/api/inventory/review", {"intake_package": package, "chemistry_plan": archived["chemistry_plan"]})
        save("old_profile_review.json", before)
        assert not before["package"]["ready_for_design"]
        assert [q["question_id"] for q in before["package"]["inventory_review"]["questions"]] == ["INV-PRESSURE-001"]
        log("Old KHU profile: reproduced missing pressure-controller confirmation.")

        response = client.get("/api/inventory/profiles/khu_laboratory_inventory_updated_20260807")
        response.raise_for_status()
        profile = response.json()
        after = post("/api/inventory/review", {"intake_package": package, "inventory_profile": profile, "chemistry_plan": archived["chemistry_plan"]})
        save("updated_profile_review.json", after)
        assert after["package"]["ready_for_design"]
        assert after["package"]["raw_protocol"] == package["raw_protocol"]
        log("Updated KHU profile: equipment precheck passed; protocol unchanged.")
        if not args.live:
            return
        request = {
            "batch_input": package["raw_protocol"], "intake_package": after["package"],
            "inventory_profile": profile, "upstream_model_id": "qwen3.6-27b",
            "downstream_model_id": "qwen3.6-27b", "runtime_options": {"candidate_budget": 2},
        }
        save("live_request.json", request)
        job = post("/api/design/jobs", request)
        save("job.json", job)
        log(f"Real Qwen upstream/downstream job: {job['job_id']} (2 candidates, council enabled).")
        start = time.monotonic()
        while job["status"] not in {"completed", "failed"}:
            time.sleep(30)
            response = client.get(f"/api/design/jobs/{job['job_id']}")
            response.raise_for_status()
            job = response.json()
            save("job.json", job)
            log(f"{int(time.monotonic()-start)} s: {job['status']} / {job['phase']}")
        if job["status"] == "failed":
            raise RuntimeError(job.get("error"))
        result = job["result"]
        save("result.json", result)
        summary = {"job_id": job["job_id"], "autosave_dir": job.get("autosave_dir"),
                   "final_status": result.get("final_design", {}).get("status"),
                   "parameters": result.get("final_design", {}).get("parameters"),
                   "consistency": result.get("final_design", {}).get("consistency"),
                   "inventory_allocation": result.get("inventory_allocation", {}).get("status")}
        save("summary.json", summary)
        log(f"Completed: {summary['final_status']}; autosaved to {summary['autosave_dir']}")
        for kind, suffix in [("process-svg", "svg"), ("process-png", "png")]:
            artifact = client.get(f"/api/design/jobs/{job['job_id']}/artifacts/{kind}")
            if artifact.is_success:
                (output / f"process.{suffix}").write_bytes(artifact.content)
        assert summary["final_status"] == "executable", "Live run did not produce a closed design; inspect result.json"
        assert summary["consistency"]["passed"], "Final consistency did not pass"


if __name__ == "__main__":
    main()
