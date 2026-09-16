"""Run 5 design-priority variations of the THQ -> 6-methylquinoline protocol.

Same chemistry, different one-sentence design directive appended. For each
run we dump the full result JSON and a per-variation log, then print a
side-by-side comparison table of the topology fields the user requested.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flora_translate.main import translate  # noqa: E402


BASE_PROTOCOL = (
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


VARIATIONS = [
    ("v1_baseline", "Baseline (no extra directive)", ""),
    (
        "v2_productivity",
        "Productivity-first",
        " Design priority: maximize space-time yield (productivity). "
        "Tolerate moderately lower per-pass conversion if it boosts throughput.",
    ),
    (
        "v3_yield",
        "Yield-first",
        " Design priority: maximize conversion and isolated yield "
        "(target single-pass conversion >= 90%). Use longer residence time if needed.",
    ),
    (
        "v4_intensify",
        "Aggressive intensification",
        " Design priority: minimize residence time and reactor footprint; "
        "push intensification hard (smallest practical tau, smallest reactor volume).",
    ),
    (
        "v5_balanced",
        "Balanced bench-practical",
        " Design priority: balance productivity and conversion. Favor practical "
        "bench operation: BPR <= 8 bar, single-pass conversion >= 75%, "
        "FEP tubing 0.75-1.6 mm ID.",
    ),
]


OUT_DIR = Path("outputs/thq_variations")


def _setup_per_variation_logging(slug: str) -> logging.FileHandler:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"{slug}.log"
    # Truncate any previous log for this slug
    log_path.write_text("")
    handler = logging.FileHandler(log_path, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _extract_summary(slug: str, label: str, result: dict) -> dict:
    prop = result.get("proposal") or {}
    calc = result.get("design_calculations") or {}
    streams = prop.get("streams") or []

    liquid = None
    gas = None
    for s in streams:
        phase = (s.get("phase") or "").lower()
        if phase == "gas" or s.get("gas_flow_sccm") is not None:
            gas = s
        elif liquid is None:
            liquid = s

    def _g(d, *keys):
        if not d:
            return None
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
        return None

    return {
        "slug": slug,
        "label": label,
        "confidence": result.get("confidence"),
        "engine_validated": prop.get("engine_validated"),
        "safety_flags": prop.get("safety_flags") or [],
        "residence_time_min": prop.get("residence_time_min"),
        "flow_rate_mL_min": prop.get("flow_rate_mL_min"),
        "tubing_ID_mm": prop.get("tubing_ID_mm"),
        "tubing_length_m": prop.get("tubing_length_m"),
        "reactor_volume_mL": prop.get("reactor_volume_mL"),
        "temperature_C": prop.get("temperature_C"),
        "BPR_bar": prop.get("BPR_bar"),
        "reynolds_number": calc.get("reynolds_number"),
        "pressure_drop_bar": calc.get("pressure_drop_bar"),
        "two_phase_pressure_drop_bar": calc.get("two_phase_pressure_drop_bar"),
        "gas_holdup": calc.get("gas_holdup"),
        "is_gas_liquid": calc.get("is_gas_liquid"),
        "stream_A": {
            "name": _g(liquid, "name", "label"),
            "contents": _g(liquid, "contents"),
            "solvent": _g(liquid, "solvent"),
            "concentration_M": _g(liquid, "concentration_M"),
            "flow_rate_mL_min": _g(liquid, "flow_rate_mL_min", "flowrate_mL_min"),
            "phase": _g(liquid, "phase"),
        },
        "stream_gas": {
            "name": _g(gas, "name", "label"),
            "contents": _g(gas, "contents"),
            "gas_flow_sccm": _g(gas, "gas_flow_sccm"),
            "gas_flow_actual_mL_min": _g(gas, "gas_flow_actual_mL_min"),
            "phase": _g(gas, "phase"),
        }
        if gas
        else None,
    }


def _print_table(summaries: list[dict]) -> None:
    line = "=" * 100
    print()
    print(line)
    print("THQ -> 6-methylquinoline — variation comparison")
    print(line)

    def _fmt(v, digits=3):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{digits}g}"
        return str(v)

    rows = [
        ("Label", lambda s: s["label"]),
        ("confidence", lambda s: s["confidence"]),
        ("engine_validated", lambda s: s["engine_validated"]),
        ("tau_min", lambda s: _fmt(s["residence_time_min"])),
        ("Q_total_mL_min", lambda s: _fmt(s["flow_rate_mL_min"])),
        ("reactor_volume_mL", lambda s: _fmt(s["reactor_volume_mL"])),
        ("tubing_ID_mm", lambda s: _fmt(s["tubing_ID_mm"])),
        ("tubing_length_m", lambda s: _fmt(s["tubing_length_m"])),
        ("temperature_C", lambda s: _fmt(s["temperature_C"])),
        ("BPR_bar", lambda s: _fmt(s["BPR_bar"])),
        ("Re", lambda s: _fmt(s["reynolds_number"], 4)),
        ("dP_bar (single)", lambda s: _fmt(s["pressure_drop_bar"], 4)),
        ("dP_bar (2-phase)", lambda s: _fmt(s["two_phase_pressure_drop_bar"], 4)),
        ("gas_holdup", lambda s: _fmt(s["gas_holdup"], 3)),
        ("StreamA name", lambda s: _fmt(s["stream_A"]["name"])),
        ("StreamA solvent", lambda s: _fmt(s["stream_A"]["solvent"])),
        ("StreamA conc_M", lambda s: _fmt(s["stream_A"]["concentration_M"])),
        ("StreamA Q_mL_min", lambda s: _fmt(s["stream_A"]["flow_rate_mL_min"])),
        ("Gas name", lambda s: _fmt((s["stream_gas"] or {}).get("name"))),
        ("Gas SCCM", lambda s: _fmt((s["stream_gas"] or {}).get("gas_flow_sccm"))),
        ("Gas actual mL_min", lambda s: _fmt((s["stream_gas"] or {}).get("gas_flow_actual_mL_min"))),
    ]

    # Header
    header = f"{'field':22s} | " + " | ".join(f"{s['slug']:>16s}" for s in summaries)
    print(header)
    print("-" * len(header))
    for name, getter in rows:
        cells = []
        for s in summaries:
            try:
                cells.append(f"{str(getter(s)):>16s}")
            except Exception:  # noqa: BLE001
                cells.append(f"{'ERR':>16s}")
        print(f"{name:22s} | " + " | ".join(cells))
    print()
    # Safety flags per variation
    for s in summaries:
        flags = s.get("safety_flags") or []
        if flags:
            print(f"[{s['slug']}] safety_flags:")
            for f in flags:
                print(f"   - {f}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Root logger config; per-run we add a FileHandler with the variation slug.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    summaries: list[dict] = []

    for slug, label, directive in VARIATIONS:
        print("=" * 100)
        print(f"RUNNING {slug}: {label}")
        print("=" * 100)
        handler = _setup_per_variation_logging(slug)
        prompt = BASE_PROTOCOL + directive
        try:
            result = translate(prompt)
            # Persist full result
            res_path = OUT_DIR / f"{slug}_result.json"
            safe = json.loads(json.dumps(result, default=str))
            res_path.write_text(json.dumps(safe, indent=2))
            summary = _extract_summary(slug, label, result)
            summaries.append(summary)
            print(f"  -> tau={summary['residence_time_min']} min, "
                  f"V={summary['reactor_volume_mL']} mL, "
                  f"BPR={summary['BPR_bar']} bar, "
                  f"engine_validated={summary['engine_validated']}")
        except Exception as exc:  # noqa: BLE001
            logging.getLogger().exception("Variation %s failed: %s", slug, exc)
            summaries.append({
                "slug": slug,
                "label": label,
                "error": f"{exc!r}",
                "traceback": traceback.format_exc(),
            })
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()

    # Aggregate report
    summary_path = OUT_DIR / "all_summaries.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    _print_table([s for s in summaries if "error" not in s])

    failed = [s for s in summaries if "error" in s]
    if failed:
        print()
        print("=" * 100)
        print("FAILED variations")
        print("=" * 100)
        for s in failed:
            print(f"[{s['slug']}] {s['label']}: {s['error']}")

    print()
    print(f"Per-variation logs: {OUT_DIR}/*.log")
    print(f"Per-variation results: {OUT_DIR}/*_result.json")
    print(f"Aggregate summary: {summary_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
