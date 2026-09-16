"""Smoke run for the THQ -> 6-methylquinoline aerobic photo-oxidation protocol.

Validates the post-fix pipeline:
  - council should NOT be skipped
  - if Designer sampling is empty, Design Space fallback should seed it
  - domain agents must run (chemistry_scores / fluidics_scores / ...
    non-empty in the deliberation log)
  - Skeptic should run its arithmetic re-derivations
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Repo-relative import root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from flora_translate.main import translate  # noqa: E402

PROTOCOL = (
    "A flame-dried screw-cap reaction tube (13 × 100 mm, 8 mL) equipped with a "
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


def _summarize(result: dict) -> None:
    print()
    print("=" * 78)
    print("POST-RUN SUMMARY")
    print("=" * 78)
    prop = result.get("proposal", {}) or {}
    calc = result.get("design_calculations", {}) or {}
    print(f"confidence           : {result.get('confidence')!r}")
    print(f"engine_validated     : {prop.get('engine_validated')}")
    print(f"safety_flags         : {prop.get('safety_flags', [])}")
    print(f"residence_time_min   : {prop.get('residence_time_min')}")
    print(f"flow_rate_mL_min     : {prop.get('flow_rate_mL_min')}")
    print(f"tubing_ID_mm         : {prop.get('tubing_ID_mm')}")
    print(f"reactor_volume_mL    : {prop.get('reactor_volume_mL')}")
    print(f"BPR_bar              : {prop.get('BPR_bar')}")
    print(f"calc.is_gas_liquid   : {calc.get('is_gas_liquid')}")
    print(f"calc.delta_P_bar     : {calc.get('pressure_drop_bar')}")
    print(f"calc.gas_holdup      : {calc.get('gas_holdup')}")

    delib = result.get("deliberation_log") or {}
    rounds = delib.get("rounds") or []
    print()
    print(f"deliberation rounds  : {len(rounds)}")
    print(f"consensus_reached    : {delib.get('consensus_reached')}")
    if delib.get("summary"):
        print(f"deliberation summary : {delib['summary'][:200]}")

    # Check that domain agents actually ran (the original regression: council skipped)
    n_council_msgs = 0
    n_agents_seen: set[str] = set()
    for rnd in rounds:
        for entry in rnd:
            n_council_msgs += 1
            agent = entry.get("agent") or entry.get("agent_display_name") or "?"
            n_agents_seen.add(str(agent))
    print(f"agents that spoke    : {sorted(n_agents_seen) or '∅ (COUNCIL WAS SKIPPED)'}")
    print(f"total agent messages : {n_council_msgs}")

    # Surface any pre-council "council was skipped" markers in safety_flags
    flagged_skip = [f for f in (prop.get("safety_flags") or []) if "SCREEN_REQUIRED" in str(f)]
    if flagged_skip:
        print()
        print("SCREEN_REQUIRED flags:")
        for f in flagged_skip:
            print(f"  - {f}")

    # Dump full result for inspection
    out_path = Path("outputs/thq_smoke_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip non-serializable bits
    safe = json.loads(json.dumps(result, default=str))
    out_path.write_text(json.dumps(safe, indent=2))
    print()
    print(f"full result dumped to: {out_path}")


def main() -> int:
    print("=" * 78)
    print("FLORA-Translate smoke run — THQ → 6-methylquinoline aerobic photo-ox")
    print("=" * 78)
    result = translate(PROTOCOL)
    _summarize(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
