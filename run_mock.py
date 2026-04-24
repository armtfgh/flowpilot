"""Run a single challenging mock protocol through FLORA-Translate and save
the full council output to outputs/mock_council_run.json.

Run once to generate data; visualization/council_panels.py reads the JSON.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MOCK_PROTOCOL = """
Ru(bpy)3Cl2-catalyzed visible-light photoredox dehalogenation of
alpha-bromoacetophenone in MeCN/H2O (9:1, v/v), 0.1 M substrate,
460 nm blue LED irradiation, strict N2 atmosphere, 85 degC,
24 h batch time, 78% isolated yield.
Conditions: 1 mol% Ru(bpy)3Cl2 photocatalyst, 2.0 equiv DIPEA
as sacrificial reductant, reaction quenched with sat. NH4Cl (aq).
"""

OUTPUT_PATH = Path("outputs/mock_council_run.json")


def _safe_serialise(obj):
    """Fallback JSON encoder for Path, bytes, and other non-serialisable types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def main():
    from flora_translate.main import translate

    print("Running FLORA-Translate on mock protocol …")
    result = translate(MOCK_PROTOCOL)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2, default=_safe_serialise)

    print(f"\nSaved → {OUTPUT_PATH}")

    # Quick summary
    proposal = result.get("proposal", {})
    dlog = result.get("deliberation_log", {})
    print(f"  τ = {proposal.get('residence_time_min')} min")
    print(f"  d = {proposal.get('tubing_ID_mm')} mm")
    print(f"  Q = {proposal.get('flow_rate_mL_min')} mL/min")
    print(f"  V_R = {proposal.get('reactor_volume_mL')} mL")
    print(f"  Confidence: {result.get('confidence')}")
    if dlog:
        rounds = dlog.get("rounds", [])
        n_agents = sum(len(r) for r in rounds)
        print(f"  Deliberation rounds: {len(rounds)} ({n_agents} agent turns)")
        print(f"  Consensus: {dlog.get('consensus_reached')}")


if __name__ == "__main__":
    main()
