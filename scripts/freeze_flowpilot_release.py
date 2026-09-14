"""Create a local immutable-by-convention source/evidence release, without secrets."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    archive = args.output.resolve() / "source.tar.gz"
    subprocess.run(["git", "archive", "--format=tar.gz", f"--output={archive}", commit], cwd=root, check=True)
    shutil.copytree(args.baseline, args.output / "baseline_run")
    versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output / "python_packages.txt").write_text(versions)
    tracked_changes = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    (args.output / "working_tree.patch").write_bytes(tracked_changes)
    manifest = {"release": "FlowPilot legacy before private 2.0", "git_commit": commit,
                "baseline": str(args.baseline), "python": sys.version,
                "scope": "Tracked source, exact baseline artifacts, installed Python package versions; no environment secrets. External model services and ignored retrieval indexes are not frozen.",
                "sha256": {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(args.output.rglob("*")) if p.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"folder": str(args.output), "commit": commit, "archive_sha256": manifest["sha256"]["source.tar.gz"]}))


if __name__ == "__main__":
    main()
