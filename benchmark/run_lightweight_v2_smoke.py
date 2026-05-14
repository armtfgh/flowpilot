from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flora_translate.config as cfg
from benchmark.run_model_matrix_benchmark import GEMMA_BASE_URL, GEMMA_MODEL, PROTOCOL
from flora_translate.design_calculator import DesignCalculator
from flora_translate.engine import llm_agents
from flora_translate.engine.design_space import DesignSpaceSearch, candidates_to_dicts, get_council_starting_point
from flora_translate.engine.llm_agents import clear_llm_observer, clear_llm_runtime_overrides, set_llm_observer, set_llm_runtime_overrides
from flora_translate.lightweight_upstream import analyze_batch_chemistry, parse_batch_input
from flora_translate.schemas import LabInventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test fair lightweight upstream v2 without council rescue.")
    parser.add_argument("--model", default=GEMMA_MODEL)
    parser.add_argument("--base-url", default=GEMMA_BASE_URL)
    parser.add_argument("--output-root", default="benchmark/data")
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


@contextmanager
def lightweight_v2_model(model: str, base_url: str):
    original = {
        "MODEL_INPUT_PARSER": cfg.MODEL_INPUT_PARSER,
        "MODEL_CHEMISTRY_AGENT": cfg.MODEL_CHEMISTRY_AGENT,
        "LIGHTWEIGHT_UPSTREAM_MODE": cfg.LIGHTWEIGHT_UPSTREAM_MODE,
        "OLLAMA_BASE_URL_cfg": cfg.OLLAMA_BASE_URL,
        "OLLAMA_BASE_URL_llm": llm_agents.OLLAMA_BASE_URL,
        "ollama_client": llm_agents._OLLAMA_CLIENT,
    }
    try:
        cfg.MODEL_INPUT_PARSER = model
        cfg.MODEL_CHEMISTRY_AGENT = model
        cfg.LIGHTWEIGHT_UPSTREAM_MODE = "v2"
        if llm_agents.infer_provider_for_model(model) == "ollama":
            cfg.OLLAMA_BASE_URL = base_url
            llm_agents.OLLAMA_BASE_URL = base_url
            llm_agents._OLLAMA_CLIENT = None
        yield
    finally:
        cfg.MODEL_INPUT_PARSER = original["MODEL_INPUT_PARSER"]
        cfg.MODEL_CHEMISTRY_AGENT = original["MODEL_CHEMISTRY_AGENT"]
        cfg.LIGHTWEIGHT_UPSTREAM_MODE = original["LIGHTWEIGHT_UPSTREAM_MODE"]
        cfg.OLLAMA_BASE_URL = original["OLLAMA_BASE_URL_cfg"]
        llm_agents.OLLAMA_BASE_URL = original["OLLAMA_BASE_URL_llm"]
        llm_agents._OLLAMA_CLIENT = original["ollama_client"]


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"lightweight_v2_smoke_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_events: list[dict] = []

    def observe(event: dict) -> None:
        llm_events.append(event)
        with (out_dir / "llm_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    set_llm_observer(observe)
    set_llm_runtime_overrides(temperature=args.temperature)
    try:
        with lightweight_v2_model(args.model, args.base_url):
            batch_record = parse_batch_input(PROTOCOL)
            chemistry_plan = analyze_batch_chemistry(batch_record)
            inventory = LabInventory.from_json(str(cfg.LAB_INVENTORY_PATH))
            calculations = DesignCalculator().run(
                batch_record,
                chemistry_plan=chemistry_plan,
                inventory=inventory,
                analogies=[],
            )
            design_points = DesignSpaceSearch().run(
                batch_record=batch_record,
                chemistry_plan=chemistry_plan,
                calculations=calculations,
                inventory=inventory,
                reaction_class=chemistry_plan.reaction_class or "default",
            )
            design_space = candidates_to_dicts(design_points)
            top = get_council_starting_point(design_points)

        evidence = getattr(batch_record, "_evidence_report", {})
        result = {
            "status": "completed",
            "model": args.model,
            "base_url": args.base_url if llm_agents.infer_provider_for_model(args.model) == "ollama" else None,
            "lightweight_upstream_mode": "v2",
            "batch_record": batch_record.model_dump(),
            "batch_evidence_report": evidence,
            "chemistry_plan": chemistry_plan.model_dump(),
            "upstream_mode": getattr(chemistry_plan, "_upstream_mode", None),
            "design_calculations": asdict(calculations),
            "design_space_count": len(design_space),
            "top_design_candidate": top.__dict__ if top else None,
            "llm_event_count": len(llm_events),
            "notes": [
                "No rescued kinetic anchor was applied.",
                "No forced tau, concentration, temperature, ID, flow rate, or BPR was inserted.",
                "Concentration/time corrections come only from protocol-text arithmetic.",
            ],
        }
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps({
            "status": "completed",
            "output_dir": str(out_dir),
            "batch_time_min": (batch_record.reaction_time_h or 0.0) * 60.0,
            "concentration_M": batch_record.concentration_M,
            "reaction_class": chemistry_plan.reaction_class,
            "mechanism_type": chemistry_plan.mechanism_type,
            "upstream_mode": getattr(chemistry_plan, "_upstream_mode", None),
            "design_space_count": len(design_space),
            "top_tau_min": getattr(top, "tau_min", None) if top else None,
            "top_Q_mL_min": getattr(top, "Q_mL_min", None) if top else None,
            "top_ID_mm": getattr(top, "d_mm", None) if top else None,
            "llm_event_count": len(llm_events),
        }, indent=2))
    finally:
        clear_llm_observer()
        clear_llm_runtime_overrides()


if __name__ == "__main__":
    main()
