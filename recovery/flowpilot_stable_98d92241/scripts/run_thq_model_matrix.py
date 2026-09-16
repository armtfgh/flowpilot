"""THQ benchmark across 3 design-priority variations x 2 model configurations.

Model configurations:
  A) claude_to_4omini : upstream Claude (default Sonnet/Opus), downstream GPT-4o-mini council
  B) gpt4o_to_gpt4o   : upstream GPT-4o (parser/chemistry/translation/formatter), downstream GPT-4o council

Output: per-run JSON, per-config CSV, combined CSV, per-run log.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flora_translate import config as cfg  # noqa: E402
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
]


# Capture defaults so config A can restore them
DEFAULTS = {
    "MODEL_INPUT_PARSER": cfg.MODEL_INPUT_PARSER,
    "MODEL_CHEMISTRY_AGENT": cfg.MODEL_CHEMISTRY_AGENT,
    "MODEL_TRANSLATION": cfg.MODEL_TRANSLATION,
    "MODEL_OUTPUT_FORMATTER": cfg.MODEL_OUTPUT_FORMATTER,
    "MODEL_REVISION_AGENT": cfg.MODEL_REVISION_AGENT,
    "ENGINE_PROVIDER": cfg.ENGINE_PROVIDER,
    "ENGINE_MODEL_OPENAI": cfg.ENGINE_MODEL_OPENAI,
}

MODEL_CONFIGS = [
    (
        "claude_to_4omini",
        "Upstream Claude (default) -> downstream GPT-4o-mini council",
        {
            # Upstream stays on defaults (Claude Sonnet/Opus)
            "MODEL_INPUT_PARSER": DEFAULTS["MODEL_INPUT_PARSER"],
            "MODEL_CHEMISTRY_AGENT": DEFAULTS["MODEL_CHEMISTRY_AGENT"],
            "MODEL_TRANSLATION": DEFAULTS["MODEL_TRANSLATION"],
            "MODEL_OUTPUT_FORMATTER": DEFAULTS["MODEL_OUTPUT_FORMATTER"],
            "MODEL_REVISION_AGENT": DEFAULTS["MODEL_REVISION_AGENT"],
            "ENGINE_PROVIDER": "openai",
            "ENGINE_MODEL_OPENAI": "gpt-4o-mini",
        },
    ),
    (
        "gpt4o_to_gpt4o",
        "Upstream GPT-4o -> downstream GPT-4o council",
        {
            "MODEL_INPUT_PARSER": "gpt-4o",
            "MODEL_CHEMISTRY_AGENT": "gpt-4o",
            "MODEL_TRANSLATION": "gpt-4o",
            "MODEL_OUTPUT_FORMATTER": "gpt-4o",
            "MODEL_REVISION_AGENT": "gpt-4o",
            "ENGINE_PROVIDER": "openai",
            "ENGINE_MODEL_OPENAI": "gpt-4o",
        },
    ),
]


OUT_DIR = Path("outputs/thq_model_matrix")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _apply_cfg(overrides: dict[str, str]) -> None:
    for k, v in overrides.items():
        setattr(cfg, k, v)


def _setup_log(slug: str) -> logging.FileHandler:
    p = OUT_DIR / f"{slug}.log"
    p.write_text("")
    h = logging.FileHandler(p, mode="a")
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(h)
    return h


def _summarize(slug: str, label: str, model_slug: str, model_desc: str, result: dict) -> dict:
    p = result.get("proposal") or {}
    c = result.get("design_calculations") or {}
    delib = result.get("deliberation_log") or {}
    streams = p.get("streams") or []
    liq, gas = None, None
    for s in streams:
        phase = (s.get("phase") or "").lower()
        if phase == "gas" or s.get("gas_flow_sccm") is not None:
            gas = s
        elif liq is None:
            liq = s
    V_R = p.get("reactor_volume_mL")
    d = p.get("tubing_ID_mm")
    L_m = round(V_R / (math.pi * (d / 2) ** 2), 2) if (V_R and d) else None
    return {
        "model_config": model_slug,
        "model_description": model_desc,
        "variation": slug,
        "label": label,
        "tau_min": p.get("residence_time_min"),
        "Q_total_mL_min": p.get("flow_rate_mL_min"),
        "reactor_volume_mL": V_R,
        "tubing_ID_mm": d,
        "tube_length_m_derived": L_m,
        "tubing_material": p.get("tubing_material"),
        "temperature_C": p.get("temperature_C"),
        "BPR_bar": p.get("BPR_bar"),
        "Re": c.get("reynolds_number"),
        "dP_bar_single_phase": c.get("pressure_drop_bar"),
        "dP_bar_two_phase": c.get("two_phase_pressure_drop_bar"),
        "gas_holdup": c.get("gas_holdup"),
        "streamA_solvent": (liq or {}).get("solvent"),
        "streamA_concentration_M": (liq or {}).get("concentration_M"),
        "streamA_flow_rate_mL_min": (liq or {}).get("flow_rate_mL_min"),
        "streamA_contents": " + ".join((liq or {}).get("contents") or []),
        "gas_contents": " + ".join((gas or {}).get("contents") or []) if gas else "",
        "gas_SCCM_at_STP": (gas or {}).get("gas_flow_sccm") if gas else None,
        "gas_actual_mL_min_at_reactor": (gas or {}).get("gas_flow_actual_mL_min") if gas else None,
        "confidence": result.get("confidence"),
        "engine_validated": p.get("engine_validated"),
        "council_rounds": len(delib.get("rounds") or []),
        "consensus_reached": delib.get("consensus_reached"),
        "safety_flags": " | ".join(p.get("safety_flags") or []),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    all_rows: list[dict] = []

    for model_slug, model_desc, overrides in MODEL_CONFIGS:
        _apply_cfg(overrides)
        print("\n" + "=" * 100)
        print(f"MODEL CONFIG: {model_slug} — {model_desc}")
        print(f"  upstream  parser     = {cfg.MODEL_INPUT_PARSER}")
        print(f"  upstream  chemistry  = {cfg.MODEL_CHEMISTRY_AGENT}")
        print(f"  upstream  translation= {cfg.MODEL_TRANSLATION}")
        print(f"  upstream  formatter  = {cfg.MODEL_OUTPUT_FORMATTER}")
        print(f"  council provider    = {cfg.ENGINE_PROVIDER}")
        print(f"  council model       = {cfg.ENGINE_MODEL_OPENAI}")
        print("=" * 100)

        rows_for_cfg: list[dict] = []
        for slug, label, directive in VARIATIONS:
            run_slug = f"{model_slug}__{slug}"
            print("-" * 100)
            print(f"RUNNING {run_slug}: {label}")
            print("-" * 100)
            handler = _setup_log(run_slug)
            try:
                result = translate(BASE_PROTOCOL + directive)
                (OUT_DIR / f"{run_slug}_result.json").write_text(
                    json.dumps(json.loads(json.dumps(result, default=str)), indent=2)
                )
                row = _summarize(slug, label, model_slug, model_desc, result)
                rows_for_cfg.append(row)
                all_rows.append(row)
                print(
                    f"  -> tau={row['tau_min']} min  V={row['reactor_volume_mL']} mL  "
                    f"d={row['tubing_ID_mm']} mm  BPR={row['BPR_bar']} bar  "
                    f"engine_validated={row['engine_validated']}"
                )
            except Exception as exc:  # noqa: BLE001
                logging.getLogger().exception("Run %s failed: %s", run_slug, exc)
                err_row = {
                    "model_config": model_slug,
                    "variation": slug,
                    "error": f"{exc!r}",
                }
                rows_for_cfg.append(err_row)
                all_rows.append(err_row)
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()

        # Per-config CSV
        ok_rows = [r for r in rows_for_cfg if "error" not in r]
        if ok_rows:
            _write_csv(ok_rows, OUT_DIR / f"{model_slug}_summary.csv")

    # Combined CSV
    ok_all = [r for r in all_rows if "error" not in r]
    if ok_all:
        _write_csv(ok_all, OUT_DIR / "all_runs_summary.csv")

    (OUT_DIR / "all_runs_summary.json").write_text(json.dumps(all_rows, indent=2, default=str))

    # Final console table
    print("\n" + "=" * 100)
    print("COMBINED RESULTS")
    print("=" * 100)
    headers = ["model_config", "variation", "tau_min", "V_R_mL", "d_mm", "Q_mL_min", "BPR_bar", "Re", "dP_bar"]
    print(" | ".join(f"{h:>18s}" for h in headers))
    for r in ok_all:
        cells = [
            r["model_config"],
            r["variation"],
            r.get("tau_min"),
            r.get("reactor_volume_mL"),
            r.get("tubing_ID_mm"),
            r.get("Q_total_mL_min"),
            r.get("BPR_bar"),
            r.get("Re"),
            r.get("dP_bar_single_phase"),
        ]
        print(" | ".join(f"{str(c):>18s}" for c in cells))

    failed = [r for r in all_rows if "error" in r]
    if failed:
        print("\nFAILED runs:")
        for r in failed:
            print(f"  {r.get('model_config')} / {r.get('variation')}: {r.get('error')}")

    print(f"\nOutputs in: {OUT_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
