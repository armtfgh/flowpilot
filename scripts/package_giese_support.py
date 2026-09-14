"""Package the audited Giese repair campaign without overwriting older runs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    root = Path("outputs/flowpilot2/20260907_giese_support")
    run = root / "resumed_08"
    out = root / "deliverables"
    out.mkdir(exist_ok=False)
    result = json.loads((run / "result.json").read_text())
    checks = json.loads((run / "independent_checks.json").read_text())
    if not checks["all_passed"]:
        raise ValueError("Do not package an unchecked result as the final screen")
    for name in ("result.json", "summary.json", "request.json", "intake_package.json", "prompt.txt", "stages.csv", "streams.csv", "independent_checks.json", "run.log"):
        shutil.copy2(run / name, out / name)
    shutil.copy2(root / "REPORT.md", out / "REPORT.md")
    shutil.copytree(run / "gui_export", out / "original_gui_export")
    (out / "council_audit.json").write_text(json.dumps(result["scientific_assessment"], indent=2))
    from flora_translate.diagram_artifacts import render_topology_artifacts
    from flora_translate.schemas import ProcessTopology
    topology = ProcessTopology.model_validate_json((run / "gui_export/topology.json").read_text())
    rendering = render_topology_artifacts(topology, title="", base_dir=out / "topology_render")
    for key, filename in (("svg_path", "process.svg"), ("png_path", "process.png")):
        shutil.copy2(rendering[key], out / filename)
    usage = []
    for folder in sorted(root.iterdir()):
        path = folder / "llm_calls.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            call = json.loads(line)
            usage.append({"attempt": folder.name, "api_name": call["api_name"], "model": call["model"],
                          "duration_ms": call["duration_ms"], "usage": call.get("usage", {})})
    (out / "model_usage.json").write_text(json.dumps({"actual_recorded_calls": usage,
        "replayed_generation_not_counted_twice": True}, indent=2))
    archive = root.with_suffix(".zip")
    if archive.exists():
        raise FileExistsError(archive)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path = root / "sha256_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in files + [manifest_path]:
            z.write(path, str(Path(root.name) / path.relative_to(root)))
    print(json.dumps({"folder": str(out), "archive": str(archive), "files": len(files),
                      "archive_MiB": round(archive.stat().st_size / 1024**2, 1)}, indent=2))


if __name__ == "__main__":
    main()
