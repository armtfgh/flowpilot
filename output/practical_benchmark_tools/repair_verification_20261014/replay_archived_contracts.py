"""Offline contract replay of archived Claude designs; never a new generation.

Run with the flent Python environment from any working directory. All writes
stay beside this helper; the campaign and its raw generations remain read-only.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

from flora_translate.final_design_contract import build_final_design_contract, publish_final_design_artifacts

ARCHIVE = ROOT / "ablation_results/practical_input/claude_practical_pilot_20261014"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def get_path(value, path):
    for key in path.split("."):
        value = value[key]
    return value


def set_path(value, path, replacement):
    keys = path.split(".")
    for key in keys[:-1]:
        value = value[key]
    value[keys[-1]] = deepcopy(replacement)


def restore(archived):
    restored = deepcopy(archived)
    changes = []
    # Every supplied replacement comes from an exact archived diagnostic
    # snapshot. No false check is changed to true and no proposal is edited.
    paths = {
        "process_topology": "diagnostic_topology",
        "svg_path": "diagnostic_svg_path",
        "png_path": "diagnostic_png_path",
        "diagram_render_manifest": "diagnostic_diagram_render_manifest",
        "final_validation.checks": "final_design.consistency.validation_checks",
        "design_disposition": "final_validation.design_disposition",
        "recommended_disposition": "final_validation.design_disposition.recommended_disposition",
        "reported_disposition": "final_validation.design_disposition.recommended_disposition",
        "disposition_rationale": "final_validation.design_disposition.rationale",
        "safety_report": "council_safety_report_diagnostic",
    }
    for target, source in paths.items():
        previous = get_path(restored, target)
        replacement = get_path(archived, source)
        changes.append({"target": target, "source_in_archived_result": source,
                        "changed": previous != replacement,
                        "before_sha256": object_sha(previous), "restored_sha256": object_sha(replacement),
                        "before": previous, "restored": replacement})
        set_path(restored, target, replacement)
    assert restored["proposal"] == archived["proposal"]
    assert restored["chemistry_plan"] == archived["chemistry_plan"]
    assert restored["batch_record"] == archived["batch_record"]
    assert restored["inventory_snapshot"] == archived["inventory_snapshot"]
    assert restored["design_calculations"] == archived["design_calculations"]
    # The validation status/reasons overwritten at publication have no saved
    # independent pre-contract snapshot. Leave them unchanged in the replay
    # input; build_final_design_contract consumes the exact restored checks.
    # Publishing the replay contract deterministically recomputes their values.
    return restored, changes


def forbid_network(*args, **kwargs):
    raise AssertionError("Network calls are forbidden during offline contract replay")


def main():
    before = {str(p.relative_to(ARCHIVE)): sha(p) for p in sorted(ARCHIVE.rglob("*")) if p.is_file()}
    code_before = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "flora_translate").rglob("*.py"))}
    rows = []
    paths = sorted((ARCHIVE / "generation").glob("*/claude_flowpilot/repeat_01/result.json"))
    assert len(paths) == 3
    with patch("socket.socket.connect", forbid_network), patch("socket.create_connection", forbid_network):
        for path in paths:
            case = path.parts[-4]
            original = json.loads(path.read_text())
            restored, changes = restore(original)
            folder = OUT / case
            write(folder / "restored_contract_input.json", restored)
            write(folder / "restoration_log.json", changes)
            contract = build_final_design_contract(restored)
            write(folder / "replayed_final_design.json", contract)
            published = deepcopy(restored)
            publish_final_design_artifacts(published, contract)
            write(folder / "replayed_published_result.json", published)
            negative_input = deepcopy(restored)
            negative_input["final_validation"]["checks"]["pump_flow_feasible"] = False
            negative_contract = build_final_design_contract(negative_input)
            assert negative_contract["status"] == "blocked"
            assert any(issue["code"] == "FINAL-PUMP_FLOW_FEASIBLE" for issue in negative_contract["consistency"]["issues"])
            write(folder / "synthetic_negative_control.json", {
                "kind": "synthetic negative test, not an observed design defect",
                "change": "Set pump_flow_feasible=False in a separate copy of the restored input",
                "status": negative_contract["status"],
                "issues": negative_contract["consistency"]["issues"],
            })
            row = {
                "case": case, "source_result": str(path.relative_to(ROOT)), "source_sha256": sha(path),
                "kind": "offline deterministic full-contract replay; not fresh generation or judge evaluation",
                "archived_status": original["final_design"]["status"],
                "replayed_status": contract["status"],
                "archived_disposition": original["recommended_disposition"],
                "restored_pre_contract_disposition": restored["recommended_disposition"],
                "archived_contract_issues": original["final_design"]["consistency"]["issues"],
                "replayed_contract_issues": contract["consistency"]["issues"],
                "archived_published_validation_checks": original["final_validation"]["checks"],
                "restored_pre_contract_checks": restored["final_validation"]["checks"],
                "replayed_published_validation_checks": published["final_validation"]["checks"],
                "archived_semantic_checks": original["final_design"]["consistency"]["semantic_checks"],
                "replayed_semantic_checks": contract["consistency"]["semantic_checks"],
                "chemistry_identity": contract.get("chemistry_identity"),
                "restored_topology_operation_count": len(restored["process_topology"].get("unit_operations", [])),
                "replayed_procedure_step_count": len(contract.get("operating_procedure") or []),
                "replayed_canonical_sha256": contract.get("canonical_sha256"),
                "proposal_unchanged": restored["proposal"] == original["proposal"],
                "batch_record_unchanged": restored["batch_record"] == original["batch_record"],
                "inventory_unchanged": restored["inventory_snapshot"] == original["inventory_snapshot"],
                "synthetic_failed_pump_check_still_blocks": negative_contract["status"] == "blocked",
                "restoration_log": str((folder / "restoration_log.json").relative_to(ROOT)),
            }
            write(folder / "verification.json", row)
            rows.append(row)
    after = {str(p.relative_to(ARCHIVE)): sha(p) for p in sorted(ARCHIVE.rglob("*")) if p.is_file()}
    code_after = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "flora_translate").rglob("*.py"))}
    assert before == after, "Archived campaign changed during replay"
    assert code_before == code_after, "Production sources changed during replay; repeat against a stable version"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "offline deterministic full-contract replay, not fresh generation",
        "purpose": "Check whether repaired validation releases the exact archived proposals after restoring their recorded pre-contract context.",
        "api_calls": 0, "network_blocked": True,
        "scope_limit": "Does not regenerate or rescore designs, prove laboratory readiness, or repair underlying generated chemistry/numerical content.",
        "archive_unchanged": True, "archive_file_count": len(before),
        "archive_hashes_before_and_after": before, "production_code_sha256": code_before,
        "replay_script_sha256": sha(Path(__file__)),
        "cases": rows,
    }
    write(OUT / "replay_summary.json", report)
    lines = ["OFFLINE FULL-CONTRACT REPLAY — NOT FRESH GENERATION", "",
             "No model or judge calls were made. Original proposals, chemistry plans, inventories, batch records and engineering calculations were not edited.",
             "Only exact archived diagnostic snapshots were restored before calling the repaired production contract builder.",
             "Publication-derived validation status and unresolved-reason fields were not guessed; the contract consumes restored checks and publication recomputes those fields.", ""]
    for row in rows:
        lines.append(f"{row['case']}: {row['archived_status']} -> {row['replayed_status']}; original pre-contract disposition {row['restored_pre_contract_disposition']}; remaining issues {len(row['replayed_contract_issues'])}; procedure steps {row['replayed_procedure_step_count']}.")
        for issue in row["replayed_contract_issues"]:
            lines.append(f"  {issue['code']}: {issue['message']}")
    lines += ["", f"All {len(before)} archived file hashes matched before and after replay.",
              "Per-case restoration logs record source field, before/restored value and hashes for every restored field.",
              "Separate synthetic negative controls confirmed that a failed pump-feasibility check still blocks every case; these controls are not observed design defects.",
              "An executable software contract remains a proposed experimental screen; this replay is not evidence of laboratory success."]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n")
    hashes = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}
    write(OUT / "artifact_hashes.json", hashes)
    print(json.dumps([{k: row[k] for k in ("case", "archived_status", "replayed_status", "replayed_contract_issues")} for row in rows], indent=2))


if __name__ == "__main__":
    main()
