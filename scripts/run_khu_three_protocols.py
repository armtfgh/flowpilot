"""Run three inventory-constrained FlowPilot cases against the KHU profile.

The two DPDTC cases are reconstructed from ``2 protocols.pdf``. Published
flow conditions and yields are stored as reference metadata but are not passed
to FlowPilot, preserving a batch-to-flow evaluation boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flora_translate.config as cfg
from benchmark.run_model_matrix_benchmark import council_bundle, upstream_bundle
from flora_translate.engine.llm_agents import (
    clear_llm_observer,
    clear_llm_runtime_overrides,
    set_llm_observer,
    set_llm_runtime_overrides,
)
from flora_translate.gui_autosave import autosave_gui_result
from flora_translate.intake_agent import get_question
from flora_translate.inventory_profiles import inventory_profile_from_payload
from flora_translate.main import translate
from flora_translate.schemas import DesignInputPackage, IntakeAnswer


QUESTION_IDS = (
    "Q-BATCH-001",
    "Q-OBJ-001",
    "Q-HIST-001",
    "Q-INV-001",
    "Q-CONSTR-001",
    "Q-HYP-001",
    "Q-PREF-001",
)


PHOTOREDOX_PROTOCOL = """Reaction: One-pot photoredox Giese addition followed by aerobic oxidation.

Substrate/radical precursor: PMPSCH2TMS (1a), 0.2 mmol, 1.0 equiv
Michael acceptor: Acrylonitrile (2a), 2.0 equiv
Photocatalyst: Ir(dF(CF3)ppy)2(dtbpy)PF6, 0.5 mol%
Solvent: EtOH/pH 9 aqueous buffer, 5:1 v/v
Concentration: 0.1 M
Scale: 0.2 mmol

Step 1 - Giese radical addition:
The reaction mixture was maintained under argon under strictly oxygen-free
conditions. It was irradiated using a 10 W, 452 nm blue LED at 25 deg C for
4 hours in a sealed pressure tube. This produced sulfide intermediate 3a
in 92% yield.

Step 2 - Aerobic oxidation:
After Step 1, the reaction vessel was opened to air. Irradiation was
continued using the same 10 W, 452 nm blue LED at 25 deg C for another
6 hours. The sulfide intermediate was oxidized to sulfoxide 4a.

Total batch reaction time: 10 hours
Final isolated yield of sulfoxide 4a: 95%

Critical requirements:
- Oxygen must be excluded during Step 1.
- Air/O2 is introduced only after Step 1 is complete.
- Light and the photocatalyst are essential.
- The pH 9 buffer suppresses formation of the PMPSCH3 byproduct.
"""


DPDTC_COMPOUND_1_PROTOCOL = """Two-step, one-pot DPDTC amide coupling in batch (General Procedure A).

Target: N-benzyl-3-methyl-4-nitrobenzamide (candidate compound 1).
Carboxylic acid: 3-methyl-4-nitrobenzoic acid, 0.50 mmol, 1.0 equiv.
Amine: benzylamine, 0.525 mmol, 1.05 equiv.

Stage 1 - thioester formation:
To a 1-dram vial equipped with a PTFE stir bar were added the carboxylic
acid, N,N-dimethylpyridin-4-amine (DMAP; 6.1 mg, 0.050 mmol, 0.10 equiv),
and 2,2'-dipyridyldithiocarbonate (DPDTC; 130 mg, 0.525 mmol, 1.05 equiv).
2-MeTHF (1.0 mL, substrate concentration 0.50 M) was added. The vial was
capped, sealed with PTFE tape, placed in a heating block preheated to
95 deg C, and stirred vigorously for 30 min. The thioester intermediate
was not isolated.

Stage 2 - aminolysis:
The vial was removed from the heating block and allowed to cool briefly to
room temperature. Benzylamine and 2-MeTHF (0.25 mL; stated Stage 2
concentration 0.40 M) were quickly added. The vial was capped and returned
to 95 deg C with vigorous stirring for another 30 min before workup.

Batch reference result: 98% yield. Preserve the two-stage order, reagent
stoichiometry, and 2-MeTHF solvent when translating to continuous flow.
"""


DPDTC_COMPOUND_2_PROTOCOL = """Two-step, one-pot DPDTC amide coupling in batch (General Procedure B for water-soluble amines).

Target: N-benzoylmorpholine (candidate compound 2).
Carboxylic acid: benzoic acid, 0.50 mmol, 1.0 equiv.
Amine: morpholine, 0.525 mmol, 1.05 equiv.

Stage 1 - thioester formation:
To a 1-dram vial equipped with a PTFE stir bar were added benzoic acid,
N,N-dimethylpyridin-4-amine (DMAP; 6.1 mg, 0.050 mmol, 0.10 equiv), and
2,2'-dipyridyldithiocarbonate (DPDTC; 130 mg, 0.525 mmol, 1.05 equiv).
2-MeTHF (1.0 mL, substrate concentration 0.50 M) was added. The vial was
capped, sealed with PTFE tape, placed in a heating block preheated to
95 deg C, and stirred vigorously for 30 min. The thioester intermediate
was not isolated.

Stage 2 - aqueous aminolysis:
The vial was removed from the heating block and allowed to cool briefly to
room temperature. Morpholine and water (1.0 mL; stated Stage 2
concentration 0.25 M) were quickly added. The vial was capped and returned
to 95 deg C with vigorous stirring for another 30 min before workup.

Batch reference result: 95% yield. Preserve the two-stage order, reagent
stoichiometry, 2-MeTHF Stage 1 medium, and aqueous morpholine addition when
translating to continuous flow.
"""


CASES = (
    {
        "case_id": "case_01_photoredox_giese_aerobic_oxidation",
        "title": "Photoredox Giese addition followed by aerobic oxidation",
        "protocol": PHOTOREDOX_PROTOCOL,
        "objective": (
            "Produce one executable two-stage continuous-flow design using only "
            "the frozen KHU inventory, preserving oxygen exclusion in Stage 1 and "
            "introducing air or oxygen only between Stage 1 and Stage 2."
        ),
        "reference": {"batch_yield_pct": 95, "source": "prior case-study input"},
    },
    {
        "case_id": "case_02_dpdtc_compound_1_procedure_a",
        "title": "DPDTC coupling: 3-methyl-4-nitrobenzoic acid and benzylamine",
        "protocol": DPDTC_COMPOUND_1_PROTOCOL,
        "objective": (
            "Produce one executable two-stage continuous-flow design for the "
            "DPDTC amide coupling using only the frozen KHU inventory, while "
            "preserving yield, stoichiometry, sequence, and safe 2-MeTHF operation."
        ),
        "reference": {
            "batch_yield_pct": 98,
            "published_flow_yield_pct": 97,
            "published_flow_conditions_withheld_from_prompt": True,
            "source": "2 protocols.pdf; ACS Sustainable Chem. Eng. 2025, 13, 6646-6655",
        },
    },
    {
        "case_id": "case_03_dpdtc_compound_2_procedure_b",
        "title": "DPDTC coupling: benzoic acid and morpholine",
        "protocol": DPDTC_COMPOUND_2_PROTOCOL,
        "objective": (
            "Produce one executable two-stage continuous-flow design for the "
            "DPDTC amide coupling using only the frozen KHU inventory, while "
            "preserving yield, stoichiometry, sequence, aqueous Stage 2 addition, "
            "and safe 2-MeTHF operation."
        ),
        "reference": {
            "batch_yield_pct": 95,
            "published_flow_yield_pct": 100,
            "published_flow_conditions_withheld_from_prompt": True,
            "source": "2 protocols.pdf; ACS Sustainable Chem. Eng. 2025, 13, 6646-6655",
        },
    },
)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_package(case: dict[str, Any], profile: Any) -> DesignInputPackage:
    inventory = profile.lab_inventory.model_dump()
    limits = profile.design_operating_limits()
    preferences = (
        "Return one best executable design with stage-specific feed identities, "
        "pump and reactor equipment IDs, liquid flow rates, residence times, "
        "temperature, pressure, and a fully inventory-bound process topology."
    )
    return DesignInputPackage(
        raw_protocol=case["protocol"],
        objective=case["objective"],
        historical_data=None,
        inventory_constraints=inventory,
        hypotheses=[],
        operating_limits=limits,
        inventory_profile_snapshot=profile.model_dump(),
        output_preferences=preferences,
        question_log=[get_question(question_id) for question_id in QUESTION_IDS],
        answers=[
            IntakeAnswer(question_id="Q-BATCH-001", answer=case["protocol"]),
            IntakeAnswer(question_id="Q-OBJ-001", answer=case["objective"]),
            IntakeAnswer(question_id="Q-HIST-001", status="unavailable"),
            IntakeAnswer(question_id="Q-INV-001", answer=inventory),
            IntakeAnswer(question_id="Q-CONSTR-001", answer=limits),
            IntakeAnswer(question_id="Q-HYP-001", status="unavailable"),
            IntakeAnswer(question_id="Q-PREF-001", answer=preferences),
        ],
        missing_question_ids=[],
        ready_for_design=True,
    )


def _summary(case: dict[str, Any], result: dict[str, Any], events: int) -> dict[str, Any]:
    final = result.get("final_design") or {}
    parameters = final.get("parameters") or {}
    allocation = result.get("inventory_allocation") or {}
    topology = result.get("process_topology") or {}
    return {
        "case_id": case["case_id"],
        "title": case["title"],
        "completed_at_utc": _now(),
        "final_design_status": final.get("status"),
        "recommended_disposition": result.get("recommended_disposition"),
        "hard_failures": (result.get("design_disposition") or {}).get("hard_failures") or [],
        "consistency_issues": (final.get("consistency") or {}).get("issues") or [],
        "inventory_allocation_status": allocation.get("status"),
        "unresolved_inventory": allocation.get("unresolved") or [],
        "residence_time_min": parameters.get("residence_time_min"),
        "residence_time_inlet_min": parameters.get("residence_time_inlet_min"),
        "residence_time_in_channel_min": parameters.get("residence_time_in_channel_min"),
        "flow_rate_mL_min": parameters.get("flow_rate_mL_min"),
        "reactor_volume_mL": parameters.get("reactor_volume_mL"),
        "temperature_C": parameters.get("temperature_C"),
        "BPR_bar": parameters.get("BPR_bar"),
        "stage_parameters": final.get("stages") or [],
        "topology_operation_count": len(topology.get("unit_operations") or []),
        "diagram_png_available": bool(result.get("png_path")),
        "diagram_svg_available": bool(result.get("svg_path")),
        "llm_event_count": events,
        "models": {
            "input_parser": cfg.MODEL_INPUT_PARSER,
            "chemistry_agent": cfg.MODEL_CHEMISTRY_AGENT,
            "translation": cfg.MODEL_TRANSLATION,
            "council_provider": cfg.ENGINE_PROVIDER,
            "council_model": cfg.ENGINE_MODEL_OPENAI,
        },
    }


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(root)}")
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run_case(
    case: dict[str, Any],
    profile: Any,
    root: Path,
    *,
    temperature: float,
    upstream: str | None = None,
) -> dict[str, Any]:
    case_dir = root / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=False)
    package = _build_package(case, profile)
    (case_dir / "protocol.txt").write_text(case["protocol"], encoding="utf-8")
    (case_dir / "reference_metadata.json").write_text(
        json.dumps(case["reference"], indent=2), encoding="utf-8"
    )
    (case_dir / "intake_package.json").write_text(
        json.dumps(package.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger = logging.getLogger()
    log_handler = logging.FileHandler(case_dir / "run.log", encoding="utf-8")
    log_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(log_handler)
    event_count = 0

    def observe(event: dict[str, Any]) -> None:
        nonlocal event_count
        event_count += 1
        with (case_dir / "llm_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(event), ensure_ascii=False) + "\n")

    set_llm_observer(observe)
    set_llm_runtime_overrides(temperature=temperature, capture_content=True)
    try:
        upstream_context = upstream_bundle(upstream) if upstream else nullcontext()
        with upstream_context, council_bundle("gpt4o"):
            result = translate(case["protocol"], intake_package=package)
        result = _json_safe(result)
        artifact_dir = autosave_gui_result(
            result,
            intake_package=package,
            source="flowpilot_result",
            user_input=case["protocol"],
            base_dir=case_dir / "artifacts",
        )
        summary = _summary(case, result, event_count)
        summary["artifact_dir"] = str(artifact_dir.resolve())
        (case_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary
    except Exception as exc:
        error = {
            "case_id": case["case_id"],
            "failed_at_utc": _now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "llm_event_count": event_count,
        }
        (case_dir / "error.json").write_text(
            json.dumps(error, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return error
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()
        logger.removeHandler(log_handler)
        log_handler.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="khu_inventory_updated_flowpilot.json",
        help="InventoryProfile or LabInventory JSON",
    )
    parser.add_argument("--source-pdf", default="2 protocols.pdf")
    parser.add_argument("--output-root", default="outputs/benchmarks")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--upstream",
        choices=["gpt4o", "gpt4omini"],
        help="Optionally route every upstream stage through one OpenAI bundle.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=[case["case_id"] for case in CASES],
    )
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    profile = inventory_profile_from_payload(json.loads(inventory_path.read_text()))
    if not profile.validation.valid:
        raise SystemExit(f"Inventory profile is invalid: {profile.validation.errors}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_root) / f"khu_three_protocols_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(inventory_path, root / "source_inventory_profile.json")
    source_pdf = Path(args.source_pdf)
    if source_pdf.is_file():
        shutil.copy2(source_pdf, root / "source_2_protocols.pdf")

    selected = [case for case in CASES if not args.cases or case["case_id"] in args.cases]
    manifest = {
        "schema_version": "flowpilot_three_case_benchmark_v1.0",
        "created_at_utc": _now(),
        "inventory_profile_id": profile.profile_id,
        "inventory_profile_version": profile.version,
        "evaluation_boundary": (
            "Published flow conditions are excluded from model prompts and retained "
            "only in reference_metadata.json."
        ),
        "cases": [
            {"case_id": case["case_id"], "title": case["title"]} for case in selected
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summaries = []
    for case in selected:
        print(f"\n=== Running {case['case_id']} ===", flush=True)
        summary = _run_case(
            case,
            profile,
            root,
            temperature=args.temperature,
            upstream=args.upstream,
        )
        summaries.append(summary)
        (root / "aggregate_summary.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    _write_checksums(root)
    print(json.dumps({"run_dir": str(root.resolve()), "cases": summaries}, indent=2))
    if any("error" in summary for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
