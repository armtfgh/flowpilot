"""Single THQ benchmark run with Claude upstream + Gemma 4 31B (vLLM) council.

- Upstream (input parser, chemistry agent, translation, formatter, revision): Claude defaults
- Council backend: vLLM Gemma 4 31B served at http://10.13.24.45:8000/v1
  (the codebase routes this through the "ollama" provider name; the OpenAI-
  compatible client just talks to that base URL.)
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flora_translate.config as cfg
from flora_translate.engine import llm_agents
from flora_translate.main import translate

GEMMA_MODEL = "google/gemma-4-31B-it"
GEMMA_BASE_URL = "http://10.13.24.45:8000/v1"

PROTOCOL = (
    "A flame-dried screw-cap reaction tube (13 x 100 mm, 8 mL) equipped with a "
    "magnetic stir bar was charged with 6-methyl-1,2,3,4-tetrahydroquinoline "
    "(1a, 0.20 mmol, 1.0 equiv, 29.4 mg, MW = 147.22 g/mol) and dimethyl "
    "sulfoxide (DMSO, 0.40 mL, 0.50 M) as solvent. No photocatalyst, transition "
    "metal, or any additive was used. Oxygen gas (O2, 1 atm, 0.4 mmol, 2.0 equiv, "
    "12.8 mg, MW = 32 g/mol) was bubbled through the reaction mixture for 10 "
    "minutes using an oxygen-filled balloon to saturate the solution prior to "
    "irradiation. The sealed reaction tube was placed between two MR16 blue LED "
    "lamps (lambda = 450 +/- 15 nm, 5 W x 2) and irradiated from both sides "
    "simultaneously with no external cooling applied, allowing the reaction "
    "temperature to rise naturally to 40 C driven by the heat output of the LEDs. "
    "The reaction mixture was stirred continuously at 40 C under blue LED "
    "irradiation for 15 hours. This protocol afforded 6-methylquinoline "
    "(2a, MW = 143.19 g/mol) in 75% isolated yield (21.5 mg)."
)


OUT_DIR = Path("outputs/thq_claude_to_gemma")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _switch_council_to_gemma() -> None:
    """Set ENGINE_PROVIDER=ollama with the vLLM Gemma model + base URL."""
    cfg.ENGINE_PROVIDER = "ollama"
    cfg.ENGINE_MODEL_OLLAMA = GEMMA_MODEL
    cfg.OLLAMA_BASE_URL = GEMMA_BASE_URL
    llm_agents.ENGINE_PROVIDER = "ollama"
    llm_agents.ENGINE_MODEL_OLLAMA = GEMMA_MODEL
    llm_agents.OLLAMA_BASE_URL = GEMMA_BASE_URL
    llm_agents._OLLAMA_CLIENT = None


def _summarize(result: dict) -> dict:
    p = result.get("proposal") or {}
    c = result.get("design_calculations") or {}
    d = result.get("deliberation_log") or {}
    streams = p.get("streams") or []
    liq, gas = None, None
    for s in streams:
        ph = (s.get("phase") or "").lower()
        if ph == "gas" or s.get("gas_flow_sccm") is not None:
            gas = s
        elif liq is None:
            liq = s
    V_R = p.get("reactor_volume_mL")
    d_mm = p.get("tubing_ID_mm")
    L_m = round(V_R / (math.pi * (d_mm / 2) ** 2), 2) if (V_R and d_mm) else None

    tau = p.get("residence_time_min") or 0.0
    tk = c.get("tau_kinetics_min") or 0.0
    X_proj = round(1 - math.exp(-2.303 * tau / tk), 3) if tk > 0 else None

    rounds = d.get("rounds") or []
    agents = set()
    for rnd in rounds:
        for entry in rnd:
            agents.add(str(entry.get("agent") or entry.get("agent_display_name") or "?"))

    return {
        "upstream": "claude (defaults)",
        "council_provider": cfg.ENGINE_PROVIDER,
        "council_model": cfg.ENGINE_MODEL_OLLAMA,
        "council_base_url": cfg.OLLAMA_BASE_URL,
        "confidence": result.get("confidence"),
        "engine_validated": p.get("engine_validated"),
        "safety_flags": p.get("safety_flags") or [],
        "residence_time_min": p.get("residence_time_min"),
        "flow_rate_mL_min": p.get("flow_rate_mL_min"),
        "reactor_volume_mL": V_R,
        "tubing_ID_mm": d_mm,
        "tube_length_m_derived": L_m,
        "tubing_material": p.get("tubing_material"),
        "temperature_C": p.get("temperature_C"),
        "BPR_bar": p.get("BPR_bar"),
        "Re": c.get("reynolds_number"),
        "dP_bar": c.get("pressure_drop_bar"),
        "dP_bar_two_phase": c.get("two_phase_pressure_drop_bar"),
        "gas_holdup": c.get("gas_holdup"),
        "tau_kinetics_min": tk,
        "X_projected": X_proj,
        "streamA_solvent": (liq or {}).get("solvent"),
        "streamA_concentration_M": (liq or {}).get("concentration_M"),
        "streamA_flow_rate_mL_min": (liq or {}).get("flow_rate_mL_min"),
        "streamA_contents": (liq or {}).get("contents"),
        "gas_contents": (gas or {}).get("contents") if gas else None,
        "gas_SCCM_at_STP": (gas or {}).get("gas_flow_sccm") if gas else None,
        "gas_actual_mL_min_at_reactor": (gas or {}).get("gas_flow_actual_mL_min") if gas else None,
        "council_rounds": len(rounds),
        "consensus_reached": d.get("consensus_reached"),
        "agents_spoke": sorted(agents),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "run.log"
    log_path.write_text("")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    print("=" * 90)
    print("FLORA-Translate run: Claude upstream + Gemma 4 31B (vLLM) council")
    print("=" * 90)

    _switch_council_to_gemma()
    print(f"council provider : {cfg.ENGINE_PROVIDER}")
    print(f"council model    : {cfg.ENGINE_MODEL_OLLAMA}")
    print(f"council base_url : {cfg.OLLAMA_BASE_URL}")
    print(f"upstream parser  : {cfg.MODEL_INPUT_PARSER}")
    print(f"upstream chem    : {cfg.MODEL_CHEMISTRY_AGENT}")
    print(f"upstream trans   : {cfg.MODEL_TRANSLATION}")
    print()

    result = translate(PROTOCOL)

    # Persist full result
    safe = json.loads(json.dumps(result, default=str))
    (OUT_DIR / "result.json").write_text(json.dumps(safe, indent=2))
    summary = _summarize(result)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Also write a single-row CSV
    import csv
    flat = {k: v if not isinstance(v, (list, dict)) else json.dumps(v, default=str) for k, v in summary.items()}
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader()
        w.writerow(flat)

    print("\n" + "=" * 90)
    print("RESULT SUMMARY")
    print("=" * 90)
    for k, v in summary.items():
        print(f"  {k:32s}: {v}")
    print(f"\nFiles: {OUT_DIR}/result.json, summary.json, summary.csv, run.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
