"""
FLORA ENGINE v4 — Chief Engineer and CouncilV4 orchestrator.

Selection logic:
  1. Hard-gate violations are FLAGS, not removals — all candidates reach scoring.
     Flagged candidates carry hard_gate_flags so every agent sees the reasons.
     Only scoring-blocked and Skeptic CRITICAL disqualifications remove candidates.
  2. Compute weighted combined score:
       combined = 0.25*chemistry + 0.20*kinetics + 0.20*fluidics
                + 0.20*safety + 0.15*geometry_practicality
  3. Apply objective modifier (multiply domain score by 1.5 for user objective).
  4. Tiebreaker: safety → uncertainty → tau_literature proximity.
  5. If no clear winner (all within 10%), return TOP-3 DESIGN ENVELOPE.

After selection: run DFMEA on winning candidate (Dr. Safety).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Optional

from flora_translate.config import ENGINE_PROVIDER
from flora_translate.design_calculator import DesignCalculator, DesignCalculations
from flora_translate.engine.llm_agents import call_llm, call_llm_with_tools
from flora_translate.engine.tool_definitions import CHIEF_TOOLS, execute_tool
from flora_translate.engine.council_v4.designer import run_designer_v4, run_problem_framing
from flora_translate.engine.council_v4.scoring import (
    run_domain_scoring,
    run_revision_stage,
    get_chemistry_combined, get_kinetics_score,
    get_fluidics_score, get_safety_score,
    geometry_practicality_score,
)
from flora_translate.engine.council_v4.skeptic import run_skeptic_audit
from flora_translate.schemas import (
    BatchRecord, ChemistryPlan, FlowProposal, LabInventory,
    DesignCandidate, CouncilMessage, DeliberationLog,
    AgentDeliberation, FieldProposal,
)

logger = logging.getLogger("flora.engine.council_v4.chief")


# ═══════════════════════════════════════════════════════════════════════════════
#  Weighted scoring
# ═══════════════════════════════════════════════════════════════════════════════

_WEIGHTS = {
    "chemistry": 0.25,
    "kinetics":  0.20,
    "fluidics":  0.20,
    "safety":    0.20,
    "geometry":  0.15,
}

_OBJECTIVE_MODIFIER: dict[str, tuple[str, float]] = {
    "conversion":   ("kinetics",   1.5),
    "yield":        ("kinetics",   1.5),
    "throughput":   ("geometry",   1.5),  # productivity is embedded in geometry/fluidics
    "footprint":    ("geometry",   1.5),
    "de-risk":      ("safety",     1.5),
    "safe":         ("safety",     1.5),
    "balanced":     ("chemistry",  1.0),  # no boost
}


def _objective_key(objectives: str) -> tuple[str, float]:
    """Map user objectives string to (domain_to_boost, boost_factor)."""
    obj_lower = objectives.lower()
    for kw, (domain, factor) in _OBJECTIVE_MODIFIER.items():
        if kw in obj_lower:
            return domain, factor
    return "chemistry", 1.0  # balanced default — no boost


def compute_weighted_scores(
    candidates: list[dict],
    scoring: dict,
    objectives: str,
    disqualify_ids: set[int],
) -> list[dict]:
    """Compute combined weighted scores for all surviving candidates.

    Returns list of {candidate_id, combined, chemistry, kinetics, fluidics,
                     safety, geometry, objective_boosted, disqualified}.
    """
    boost_domain, boost_factor = _objective_key(objectives)

    results: list[dict] = []
    for c in candidates:
        cid = c.get("id", 0)
        disq = cid in disqualify_ids

        chem = get_chemistry_combined(scoring["chemistry_scores"], cid)
        kin  = get_kinetics_score(scoring["kinetics_scores"], cid)
        flu  = get_fluidics_score(scoring["fluidics_scores"], cid)
        saf  = get_safety_score(scoring["safety_scores"], cid)
        geo  = geometry_practicality_score(c)

        # Apply objective modifier
        scores = {"chemistry": chem, "kinetics": kin, "fluidics": flu,
                  "safety": saf, "geometry": geo}
        scores[boost_domain] = min(1.0, scores[boost_domain] * boost_factor)

        combined = (
            _WEIGHTS["chemistry"] * scores["chemistry"] +
            _WEIGHTS["kinetics"]  * scores["kinetics"]  +
            _WEIGHTS["fluidics"]  * scores["fluidics"]  +
            _WEIGHTS["safety"]    * scores["safety"]    +
            _WEIGHTS["geometry"]  * scores["geometry"]
        )

        results.append({
            "candidate_id": cid,
            "combined": round(combined, 4),
            "chemistry": round(scores["chemistry"], 4),
            "kinetics":  round(scores["kinetics"], 4),
            "fluidics":  round(scores["fluidics"], 4),
            "safety":    round(scores["safety"], 4),
            "geometry":  round(scores["geometry"], 4),
            "objective_boosted": boost_domain,
            "disqualified": disq,
        })

    # Sort: disqualified last, then by combined score descending
    results.sort(key=lambda r: (r["disqualified"], -r["combined"]))

    # Scoring health check: warn if ALL domain scores for a candidate are at the
    # 0.5 default — this indicates a silent scoring failure, not a real assessment.
    all_default_domains = {"chemistry", "kinetics", "fluidics", "safety"}
    for r in results:
        if not r["disqualified"] and all(
            abs(r[d] - 0.5) < 0.01 for d in all_default_domains
        ):
            logger.error(
                "SCORING FAILURE: candidate %d has all domain scores at 0.500 default — "
                "domain agents returned no scores. Chief selection will be unreliable. "
                "Check scoring agent logs for truncated JSON or API errors.",
                r["candidate_id"],
            )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  DFMEA prompt (Dr. Safety, post-selection)
# ═══════════════════════════════════════════════════════════════════════════════

_DFMEA_SYSTEM = """\
You are DR. SAFETY performing a post-selection DFMEA for the winning flow chemistry
design candidate. Identify the top 3 failure modes, their severity and likelihood,
and recommend 1-3 targeted validation experiments.

## REQUIRED OUTPUT — JSON only
```json
{
  "selected_candidate_id": 3,
  "failure_modes": [
    {
      "rank": 1,
      "mode": "Pump cavitation at low Q",
      "cause": "Vapour pressure of solvent exceeds pump inlet pressure at T > 40°C",
      "severity": 8,
      "likelihood": 4,
      "RPN": 32,
      "mitigation": "Pressurise feed reservoir to 0.5 bar above vapour pressure"
    }
  ],
  "single_points_of_failure": ["BPR valve — blockage stops all flow"],
  "validation_experiments": [
    "Run blank solvent at Q=0.5 mL/min for 60 min; monitor pressure for drift > 5%"
  ]
}
```
"""


def run_dfmea(
    winner_id: int,
    winner: dict,
    chemistry_brief: str,
    safety_score_entry: Optional[dict] = None,
) -> dict:
    """Run DFMEA on the selected candidate. Returns DFMEA dict."""
    context = (
        f"## Selected candidate\n"
        f"id={winner_id} | tau={winner.get('tau_min')} min | "
        f"d={winner.get('d_mm')} mm | Q={winner.get('Q_mL_min')} mL/min | "
        f"V_R={winner.get('V_R_mL')} mL | L={winner.get('L_m')} m | "
        f"dP={winner.get('delta_P_bar', 0):.3f} bar | "
        f"Re={winner.get('Re', 0):.0f} | X={winner.get('expected_conversion', 0):.2f}\n\n"
        f"## Chemistry brief\n{chemistry_brief}\n\n"
        f"## Safety assessment from scoring phase\n"
        f"{json.dumps(safety_score_entry or {}, indent=2)[:600]}\n\n"
        "Identify top 3 failure modes. Output JSON only."
    )
    dfmea: dict = {
        "selected_candidate_id": winner_id,
        "failure_modes": [],
        "single_points_of_failure": [],
        "validation_experiments": [],
    }
    try:
        raw = call_llm(_DFMEA_SYSTEM, context, max_tokens=800)
        s = raw.strip()
        if "```" in s:
            for part in s.split("```")[1::2]:
                try:
                    dfmea.update(json.loads(part.lstrip("json").strip()))
                    break
                except json.JSONDecodeError:
                    continue
        else:
            try:
                dfmea.update(json.loads(s))
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.warning("DFMEA LLM call failed: %s", e)
    return dfmea


# ═══════════════════════════════════════════════════════════════════════════════
#  Chief LLM call
# ═══════════════════════════════════════════════════════════════════════════════

_CHIEF_SYSTEM_V4 = """\
You are the CHIEF ENGINEER of the FLORA ENGINE v4 council. You are the senior
process engineer who closes the stage-gated evaluation and selects the final design.
You do not argue domain science and do not invent new parameter values. You select
from audited, scored, surviving candidates — and you explain your reasoning fully.

## Your selection logic
1. Disqualified candidates (hard gates + Skeptic CRITICAL errors) are off the table.
2. You receive pre-computed weighted scores (chemistry 25%, kinetics 20%,
   fluidics 20%, safety 20%, geometry 15%, objective-boosted).
3. Walk through the top 3 candidates explicitly: what each scored, where they
   differ, and which domain drove the difference.
4. If top two are within 5% combined: tiebreak by (a) safety score, (b) design
   envelope width (call compute_design_envelope), (c) proximity to tau_lit.
5. If ALL surviving within 10%: return TOP-3 DESIGN ENVELOPE rather than a pick.
6. If flow chemistry is not justified for this reaction, state explicitly:
   "Batch is currently the better engineering choice for this reaction."
7. If you need clarification on a specific domain issue before deciding, set
   `ask_expert` to the domain name ("kinetics" or "chemistry") and state your
   question in `ask_expert_question`. The council will answer and you will be
   called again with the response. Use this AT MOST ONCE.

## Conflict resolution (when domain scores conflict)
  Safety > Chemistry > Kinetics > Fluidics > Hardware (advisory only)

## Pump flowrate derivation (REQUIRED)
After selecting the winner, derive each pump's flowrate using:
  Q_i = (ṅ_limiting × eq_i) / C_feed_i
where ṅ_limiting = P_batch / (Y × 60) in mmol/min (provided in chemistry brief).
If ṅ_limiting is not provided, derive it from: Q_total_winner × C_reactor × Y
Show the derivation explicitly for each stream in `pump_flowrates`.

## What to write in selection_rationale
Write 5–8 sentences that:
  - Name the winner and runner-ups with their combined scores
  - Explain what drove the winner's highest domain score
  - Explain what held back the runner-up(s)
  - State how the user objective influenced the tie-breaking
  - Name the 1–2 most important remaining uncertainties a lab chemist should test
  - State what the FIRST experiment on the bench should verify

## REQUIRED OUTPUT — JSON only
```json
{
  "selected_candidate_id": 3,
  "runner_up_ids": [5, 1],
  "selection_rationale": "5-8 sentences.",
  "weighted_scores": {},
  "resolved_tradeoffs": [
    "id=3 vs id=5: id=3 has higher safety score (0.93 vs 0.78) because BPR margin is 2.1 bar vs 0.4 bar."
  ],
  "overridden_concerns": [
    {"agent": "DR. FLUIDICS", "concern": "r_mix=0.12 slightly elevated", "override_reason": "Da_mass=0.4 < 1.0 so dual criterion not triggered"}
  ],
  "remaining_uncertainties": [
    "k estimated from batch; flow enhancement factor of 10× is class-typical but not experimentally verified"
  ],
  "requires_new_experiment": false,
  "experiment_recommendation": "Run blank solvent at design Q for 30 min, then switch to reaction mixture. Sample at t=3τ and t=5τ.",
  "design_envelope": {
    "tau_range_min": [2, 5], "Q_range_mL_min": [0.8, 1.2],
    "d_range_mm": [0.75, 1.0], "T_range_C": [20, 30], "BPR_range_bar": [0, 2]
  },
  "final_consensus": {
    "tau_min": 3.0, "Q_mL_min": 1.0, "d_mm": 1.0,
    "V_R_mL": 3.0, "L_m": 3.82, "BPR_bar": 0.0,
    "material": "FEP", "X_estimated": 0.94, "productivity_mg_h": 250.0
  },
  "pump_flowrates": [
    {
      "stream_label": "A",
      "role": "substrate",
      "Q_mL_min": 0.500,
      "derivation": "ṅ_lim=0.333 mmol/min × eq=1.0 / C_feed=0.667 M = 0.500 mL/min",
      "C_feed_M": 0.667,
      "molar_equiv": 1.0
    },
    {
      "stream_label": "B",
      "role": "reagent",
      "Q_mL_min": 0.500,
      "derivation": "ṅ_lim=0.333 mmol/min × eq=1.0 / C_feed=0.667 M = 0.500 mL/min",
      "C_feed_M": 0.667,
      "molar_equiv": 1.0
    }
  ],
  "n_limiting_mmol_min": 0.333,
  "Q_total_mL_min": 1.000,
  "ask_expert": null,
  "ask_expert_question": null
}
```

## Multi-stage guidance (only if n_stages > 1)
For multi-stage processes, each stage has its OWN independently derived τ from its
own chemistry class and batch time. τ per stage is NEVER a fraction of a shared total.
Provide `stage_d_mm` only — one diameter per stage.
Do NOT provide `stage_tau_fractions` — residence times come from the design calculator.

Tool available: `compute_design_envelope` — call on 2–3 top candidates before deciding.
Show your envelope comparison in resolved_tradeoffs.
"""


def _parse_chief(raw: str) -> dict:
    s = raw.strip()
    if "```" in s:
        for part in s.split("```")[1::2]:
            cleaned = part.lstrip("json").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(s[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


def _run_chief_llm(
    candidates: list[dict],
    weighted_scores: list[dict],
    scoring: dict,
    audit: dict,
    table_markdown: str,
    chemistry_brief: str,
    objectives: str,
    disqualify_ids: set[int],
    n_limiting_mmol_min: Optional[float] = None,
    P_batch_mmol_h: Optional[float] = None,
    expert_answer: Optional[str] = None,
) -> tuple[dict, list[dict]]:
    surviving = [r for r in weighted_scores if not r["disqualified"]]

    score_table = (
        "| id | combined | chemistry | kinetics | fluidics | safety | geometry |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for r in surviving:
        score_table += (
            f"| {r['candidate_id']} | {r['combined']:.3f} | "
            f"{r['chemistry']:.3f} | {r['kinetics']:.3f} | "
            f"{r['fluidics']:.3f} | {r['safety']:.3f} | {r['geometry']:.3f} |\n"
        )

    molar_flow_block = ""
    if n_limiting_mmol_min is not None:
        molar_flow_block = (
            f"\n## Molar flow (for pump flowrate derivation)\n"
            f"ṅ_limiting = {n_limiting_mmol_min:.4f} mmol/min\n"
            f"P_batch = {P_batch_mmol_h:.2f} mmol/h\n"
            f"Use Q_i = (ṅ_limiting × eq_i) / C_feed_i for each pump.\n"
        )

    expert_block = ""
    if expert_answer:
        expert_block = f"\n## Expert answer to your question\n{expert_answer}\n"

    context = (
        f"## User objectives\n{objectives or 'balanced'}\n\n"
        f"## Chemistry brief\n{chemistry_brief}\n\n"
        f"## Candidate table\n{table_markdown}\n\n"
        f"## Weighted scores (objective-adjusted)\n{score_table}\n"
        f"## Skeptic audit summary\n{audit.get('audit_summary', '')}\n\n"
        f"## Skeptic disqualify recommendations\n"
        + "\n".join(
            f"  - id {r['candidate_id']}: {r['reason']}"
            for r in audit.get("disqualify_recommendations", [])
        )
        + molar_flow_block
        + expert_block
        + "\n\nSelect the winner and derive pump flowrates. Output JSON only."
    )

    try:
        raw, tc = call_llm_with_tools(
            _CHIEF_SYSTEM_V4, context,
            tools=CHIEF_TOOLS,
            tool_executor=execute_tool,
            max_tokens=1600,
            max_tool_turns=4,
        )
        return _parse_chief(raw), tc
    except Exception as e:
        logger.warning("Chief v4 LLM call failed: %s", e)
        return {}, []


def _run_expert_askback(
    domain: str,
    question: str,
    chemistry_brief: str,
    candidates: list[dict],
    scoring: dict,
) -> str:
    """Run a targeted single-question query to a domain expert for the Chief."""
    _EXPERT_SYSTEM = (
        f"You are DR. {domain.upper()} in the FLORA council. "
        f"The Chief Engineer has a focused question for you. "
        f"Answer concisely (3–5 sentences) with specific numbers where relevant. "
        f"Do not repeat the question. Answer directly."
    )
    # Build a brief candidate summary
    top_ids = [c.get('id') for c in candidates[:3]]
    scores_subset = [
        e for e in scoring.get(f"{domain.lower()}_scores", [])
        if e.get("candidate_id") in top_ids
    ]
    context = (
        f"## Chemistry brief\n{chemistry_brief}\n\n"
        f"## Chief Engineer's question\n{question}\n\n"
        f"## Relevant domain scores (top candidates)\n"
        + "\n".join(
            f"- id={e.get('candidate_id')}: score={e.get(f'{domain.lower()}_score', '?')}, "
            f"reasoning={str(e.get('reasoning', ''))[:200]}"
            for e in scores_subset
        )
        + "\n\nAnswer the question directly."
    )
    try:
        answer = call_llm(_EXPERT_SYSTEM, context, max_tokens=400)
        logger.info("  Chief ask-back (%s) answered: %s", domain, answer[:100])
        return answer.strip()
    except Exception as e:
        logger.warning("Expert ask-back failed: %s", e)
        return f"Expert answer unavailable ({e})"


def _deterministic_resolve(weighted_scores: list[dict]) -> tuple[Optional[int], str]:
    """Deterministic fallback: pick highest combined non-disqualified score."""
    survivors = [r for r in weighted_scores if not r["disqualified"]]
    if not survivors:
        return None, "No surviving candidates after disqualification."
    best = survivors[0]  # already sorted descending by combined
    return best["candidate_id"], (
        f"Deterministic selection: id={best['candidate_id']} "
        f"(combined={best['combined']:.3f}, safety={best['safety']:.3f})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Apply winner to FlowProposal
# ═══════════════════════════════════════════════════════════════════════════════

_ALLOWED_PATCH_FIELDS = {
    "residence_time_min", "flow_rate_mL_min", "tubing_ID_mm",
    "BPR_bar", "temperature_C", "concentration_M",
    "tubing_material", "wavelength_nm", "deoxygenation_method",
    "mixer_type", "reactor_volume_mL",
}


def _apply_winner(
    proposal: FlowProposal,
    winner: dict,
    extra_changes: dict,
    chief_data: Optional[dict] = None,
) -> tuple[FlowProposal, dict]:
    changes: dict = {
        "residence_time_min": str(round(winner["tau_min"], 2)),
        "flow_rate_mL_min":   str(round(winner["Q_mL_min"], 4)),
        "tubing_ID_mm":       str(round(winner["d_mm"], 3)),
        "reactor_volume_mL":  str(round(winner["V_R_mL"], 3)),
    }
    for k, v in (extra_changes or {}).items():
        if k in _ALLOWED_PATCH_FIELDS and k not in changes:
            try:
                current = proposal.model_dump().get(k)
                if isinstance(current, (int, float)):
                    float(v)
                changes[k] = str(v)
            except (TypeError, ValueError):
                pass

    data = proposal.model_dump()
    for field, value_str in changes.items():
        if field not in data:
            continue
        try:
            current = data[field]
            if isinstance(current, float):
                data[field] = float(value_str)
            elif isinstance(current, int):
                data[field] = int(float(value_str))
            else:
                data[field] = value_str
        except (ValueError, TypeError):
            data[field] = value_str

    # ── Per-stage diameter decisions (multi-step, from Chief LLM) ──────────
    if chief_data:
        d_per_stage = chief_data.get("stage_d_mm") or []
        if d_per_stage:
            existing_params = data.get("stage_parameters") or []
            existing_by_sn = {
                p.get("stage_number"): p
                for p in existing_params
                if isinstance(p, dict) and "stage_number" in p
            }
            stage_params = []
            for i, d_val in enumerate(d_per_stage, start=1):
                sp: dict = dict(existing_by_sn.get(i, {"stage_number": i}))
                sp["stage_number"] = i
                sp.pop("tau_fraction", None)
                if d_val:
                    try:
                        sp["d_mm"] = float(d_val)
                    except (TypeError, ValueError):
                        pass
                stage_params.append(sp)
            data["stage_parameters"] = stage_params
            logger.info(
                "  Chief: per-stage d_mm applied — %s",
                [sp.get("d_mm") for sp in stage_params],
            )

        # ── Apply pump flowrates from Chief derivation ──────────────────────
        pump_flowrates = chief_data.get("pump_flowrates") or []
        if pump_flowrates and data.get("streams"):
            pf_by_label = {
                pf["stream_label"].upper(): pf
                for pf in pump_flowrates
                if isinstance(pf, dict) and pf.get("stream_label")
            }
            for stream in data.get("streams", []):
                label = (stream.get("stream_label") or "").upper()
                if label in pf_by_label:
                    pf = pf_by_label[label]
                    try:
                        stream["flow_rate_mL_min"] = float(pf["Q_mL_min"])
                        if pf.get("derivation"):
                            stream["reasoning"] = pf["derivation"]
                    except (TypeError, ValueError):
                        pass
            logger.info(
                "  Chief: pump flowrates applied — %s",
                {pf.get("stream_label"): pf.get("Q_mL_min") for pf in pump_flowrates},
            )

            # Normalise so REACTOR-FEED pump rates (excluding quench streams)
            # sum exactly to Q_total_winner.
            #
            # Why exclude quench streams from the sum?  The council winner's Q
            # is the design flow through the MAIN reactor. Quench streams are
            # injected post-reactor and must NOT be counted in the reactor
            # inlet — otherwise we scale the main feeds down incorrectly.
            q_winner = winner.get("Q_mL_min") or 0.0
            _GAS_ROLES_SET = {
                "n2", "n₂", "nitrogen", "o2", "o₂", "oxygen", "co2", "co₂",
                "h2", "h₂", "hydrogen", "ar", "argon", "helium", "air",
                "compressed air", "mfc", "n2 gas", "o2 gas", "gas injection",
                "gas stream", "gas feed",
            }
            _QUENCH_ROLE_KEYWORDS = ("quench", "neutraliz", "workup", "post-reactor")

            def _is_quench_role(role: str) -> bool:
                rl = (role or "").lower()
                return any(kw in rl for kw in _QUENCH_ROLE_KEYWORDS)

            reactor_feed_streams = [
                s for s in data.get("streams", [])
                if s.get("flow_rate_mL_min")
                and (s.get("stream_label") or "").upper() in pf_by_label
                and (s.get("pump_role") or "").lower() not in _GAS_ROLES_SET
                and not _is_quench_role(s.get("pump_role") or "")
            ]
            q_sum = sum(s["flow_rate_mL_min"] for s in reactor_feed_streams)
            if q_winner > 0 and q_sum > 0 and abs(q_sum - q_winner) / q_winner > 0.02:
                scale = q_winner / q_sum
                for s in reactor_feed_streams:
                    old_rate = s["flow_rate_mL_min"]
                    new_rate = round(old_rate * scale, 4)
                    s["flow_rate_mL_min"] = new_rate
                    # Keep the derivation text consistent with the displayed rate.
                    orig = s.get("reasoning") or ""
                    scale_note = (
                        f" → scaled ×{scale:.3f} to {new_rate:.4f} mL/min "
                        f"so Σ reactor-feed pumps = Q_reactor_inlet = "
                        f"{q_winner:.4f} mL/min"
                    )
                    if orig and "scaled ×" not in orig:
                        s["reasoning"] = orig.rstrip() + scale_note
                    elif not orig:
                        s["reasoning"] = (
                            f"Chief-derived Q_i = {old_rate:.4f} mL/min"
                            + scale_note
                        )
                logger.info(
                    "  Chief: reactor-feed flowrates renormalised ×%.4f "
                    "(sum %.4f → %.4f mL/min; quench streams untouched)",
                    scale, q_sum, q_winner,
                )

    # ── Propagate council-selected τ into stage_parameters ─────────────────
    # Done unconditionally (not gated on chief_data) so the multi-step topology
    # builder always sees the council τ for Stage 1, even when the Chief LLM
    # returned no pump flowrates.
    winner_tau = winner.get("tau_min")
    if winner_tau:
        stage_params = data.get("stage_parameters") or []
        existing_by_sn = {
            p["stage_number"]: p for p in stage_params
            if isinstance(p, dict) and "stage_number" in p
        }
        if 1 not in existing_by_sn:
            new_entry: dict = {"stage_number": 1}
            existing_by_sn[1] = new_entry
            stage_params = [new_entry] + list(stage_params)
        existing_by_sn[1].setdefault("residence_time_min", round(winner_tau, 3))
        data["stage_parameters"] = [
            p for p in stage_params if isinstance(p, dict) and "stage_number" in p
        ]

    return FlowProposal(**data), changes


# ═══════════════════════════════════════════════════════════════════════════════
#  Build AgentDeliberation records for UI compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def _verdict_icon(verdict: str) -> str:
    v = str(verdict).upper()
    return {"ACCEPT": "✅", "WARNING": "⚠️", "REVISE": "🔄", "BLOCK": "🚫",
            "APPROVED_WITH_CONDITIONS": "✅⚠️"}.get(v, "❓")


def _score_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled) + f" {score:.2f}"


def _build_chemistry_cot(scoring: dict) -> str:
    overall = scoring.get("chemistry_overall", "")
    scores = scoring.get("chemistry_scores", [])
    lines = []
    if overall:
        lines.append(f"### Overall Chemistry Assessment\n\n{overall}\n")
    lines.append("---\n### Per-Candidate Analysis\n")
    for e in scores:
        cid = e.get("candidate_id", "?")
        verdict = e.get("verdict", "?")
        icon = _verdict_icon(verdict)
        chem_s = e.get("chemistry_score", 0.0) or 0.0
        phot_s = e.get("photonics_score", 0.0) or 0.0
        comb_s = e.get("combined_score", 0.0) or 0.0
        reasoning = e.get("reasoning", "")
        A = e.get("beer_lambert_A")
        eps = e.get("epsilon_used")
        wl = e.get("wavelength_match")
        mat = e.get("material_transparent")
        blocking = e.get("blocking_issues") or []
        concerns = e.get("concerns") or []

        lines.append(f"#### {icon} Candidate {cid} — {verdict}")
        lines.append(
            f"| Domain | Score |\n|---|---|\n"
            f"| Chemistry | {_score_bar(chem_s)} |\n"
            f"| Photonics | {_score_bar(phot_s)} |\n"
            f"| **Combined** | **{_score_bar(comb_s)}** |\n"
        )
        if reasoning:
            lines.append(f"{reasoning}\n")
        details = []
        if A is not None:
            details.append(f"- Beer-Lambert A = **{A:.4f}** (ε = {eps} M⁻¹cm⁻¹)")
        if wl is not None:
            details.append(f"- Wavelength match: {'✓' if wl else '✗'}")
        if mat is not None:
            details.append(f"- Material transparent: {'✓' if mat else '✗ (BLOCK)'}")
        if details:
            lines.append("\n".join(details) + "\n")
        if blocking:
            lines.append("**⛔ Blocking issues:**\n" + "\n".join(f"- {b}" for b in blocking) + "\n")
        if concerns:
            lines.append("**⚠️ Concerns:**\n" + "\n".join(f"- {c}" for c in concerns) + "\n")
        lines.append("---\n")
    return "\n".join(lines)


def _build_kinetics_cot(scoring: dict) -> str:
    overall = scoring.get("kinetics_overall", "")
    scores = scoring.get("kinetics_scores", [])
    lines = []
    if overall:
        lines.append(f"### Overall Kinetics Assessment\n\n{overall}\n")
    lines.append("---\n### Per-Candidate Analysis\n")
    for e in scores:
        cid = e.get("candidate_id", "?")
        verdict = e.get("verdict", "?")
        icon = _verdict_icon(verdict)
        kin_s = e.get("kinetics_score", 0.0) or 0.0
        reasoning = e.get("reasoning", "")
        X = e.get("X_estimated")
        IF = e.get("IF_used")
        tau_vs_lit = e.get("tau_vs_literature", "")
        tau_mix = e.get("tau_mixing_required_min")
        tau_final = e.get("tau_proposed_final_min")
        t_steady = e.get("t_steady_min")
        prod = e.get("productivity_mg_h")
        concerns = e.get("concerns") or []

        lines.append(f"#### {icon} Candidate {cid} — {verdict}")
        lines.append(f"**Kinetics score:** {_score_bar(kin_s)}\n")
        if reasoning:
            lines.append(f"{reasoning}\n")
        details = []
        if X is not None:
            x_ok = "✓" if float(X) >= 0.85 else ("⚠️" if float(X) >= 0.70 else "✗")
            details.append(f"- Conversion X = **{X:.3f}** {x_ok}")
        if IF is not None:
            details.append(f"- Intensification factor IF = **{IF}×**  {'✓' if e.get('IF_valid') else '⚠️'}")
        if tau_vs_lit:
            details.append(f"- τ vs literature: {tau_vs_lit}")
        if tau_mix is not None:
            details.append(f"- τ_mixing_required = {tau_mix:.1f} min")
        if tau_final is not None:
            details.append(f"- τ_final (decision rule) = **{tau_final:.1f} min**")
        if t_steady is not None:
            details.append(f"- Steady-state wait = {t_steady:.0f} min (3×τ)")
        if prod is not None:
            details.append(f"- Productivity = {prod:.1f} mg/h")
        if details:
            lines.append("\n".join(details) + "\n")
        if concerns:
            lines.append("**⚠️ Concerns:**\n" + "\n".join(f"- {c}" for c in concerns) + "\n")
        lines.append("---\n")
    return "\n".join(lines)


def _build_fluidics_cot(scoring: dict) -> str:
    overall = scoring.get("fluidics_overall", "")
    scores = scoring.get("fluidics_scores", [])
    lines = []
    if overall:
        lines.append(f"### Overall Fluidics & Hardware Assessment\n\n{overall}\n")
    lines.append("---\n### Per-Candidate Analysis\n")
    for e in scores:
        cid = e.get("candidate_id", "?")
        verdict = e.get("verdict", "?")
        icon = _verdict_icon(verdict)
        flu_s = e.get("fluidics_score", 0.0) or 0.0
        reasoning = e.get("reasoning", "")
        Re = e.get("Re")
        regime = e.get("flow_regime", "")
        dP = e.get("dP_bar")
        headroom = e.get("pump_headroom_pct")
        r_mix = e.get("r_mix")
        dual_fail = e.get("dual_criterion_mixing_fail", False)
        d_dir = e.get("d_change_direction", "none")
        d_fix = e.get("d_fix_mm")
        L = e.get("L_m")
        pump = e.get("pump_type", "")
        mat = e.get("tubing_material", "")
        dv_impact = e.get("dead_volume_impact", "")
        concerns = e.get("concerns") or []

        lines.append(f"#### {icon} Candidate {cid} — {verdict}")
        lines.append(f"**Fluidics score:** {_score_bar(flu_s)}\n")
        if reasoning:
            lines.append(f"{reasoning}\n")
        details = []
        if Re is not None:
            details.append(f"- Re = **{Re:.0f}** ({regime})")
        if dP is not None and headroom is not None:
            hp_icon = "✓" if float(headroom) > 40 else ("⚠️" if float(headroom) > 20 else "✗")
            details.append(f"- ΔP = {dP:.3f} bar | Pump headroom = **{headroom:.1f}%** {hp_icon}")
        if r_mix is not None:
            mix_icon = "✓" if float(r_mix) < 0.10 else ("⚠️" if float(r_mix) < 0.20 else "✗")
            details.append(f"- r_mix = **{r_mix:.3f}** {mix_icon}{' — dual-criterion FAIL ⛔' if dual_fail else ''}")
        if d_dir and d_dir != "none":
            details.append(f"- d change needed: **{d_dir}**" + (f" → d_fix = {d_fix} mm" if d_fix else ""))
        if L is not None:
            details.append(f"- L = {L:.1f} m")
        if pump:
            details.append(f"- Pump: {pump}")
        if mat:
            details.append(f"- Tubing material: {mat}")
        if dv_impact:
            details.append(f"- Dead volume: {dv_impact}")
        if details:
            lines.append("\n".join(details) + "\n")
        if concerns:
            lines.append("**⚠️ Concerns:**\n" + "\n".join(f"- {c}" for c in concerns) + "\n")
        lines.append("---\n")
    return "\n".join(lines)


def _build_safety_cot(scoring: dict) -> str:
    overall = scoring.get("safety_overall", "")
    scores = scoring.get("safety_scores", [])
    lines = []
    if overall:
        lines.append(f"### Overall Safety Assessment\n\n{overall}\n")
    lines.append("---\n### Per-Candidate Analysis\n")
    for e in scores:
        cid = e.get("candidate_id", "?")
        verdict = e.get("verdict", "?")
        icon = _verdict_icon(verdict)
        saf_s = e.get("safety_score", 0.0) or 0.0
        reasoning = e.get("reasoning", "")
        Da_th = e.get("Da_thermal")
        BPR_req = e.get("BPR_required_bar")
        BPR_cur = e.get("BPR_current_bar")
        BPR_ok = e.get("BPR_adequate")
        mat_rec = e.get("material_recommendation", "")
        mat_rat = e.get("material_rationale", "")
        atm_req = e.get("atmosphere_isolation_required", False)
        iso_method = e.get("isolation_method", "")
        hazards = e.get("hazard_flags") or []
        blocking = e.get("blocking_issues") or []
        conditions = e.get("conditions") or []

        lines.append(f"#### {icon} Candidate {cid} — {verdict}")
        lines.append(f"**Safety score:** {_score_bar(saf_s)}\n")
        if reasoning:
            lines.append(f"{reasoning}\n")
        details = []
        if Da_th is not None:
            th_icon = "✓" if float(Da_th) < 0.1 else ("⚠️" if float(Da_th) < 1.0 else "✗")
            details.append(f"- Da_thermal = **{Da_th:.4f}** {th_icon}")
        if BPR_req is not None and BPR_cur is not None:
            bpr_icon = "✓" if BPR_ok else "✗ REVISE"
            details.append(
                f"- BPR required = {BPR_req:.2f} bar | current = {BPR_cur:.2f} bar {bpr_icon}"
            )
        if mat_rec:
            details.append(f"- Material: **{mat_rec}**" + (f" — {mat_rat}" if mat_rat else ""))
        if atm_req:
            details.append(f"- Atmosphere isolation required: {iso_method or 'see conditions'}")
        if details:
            lines.append("\n".join(details) + "\n")
        if hazards:
            lines.append("**🔶 Hazard flags:**\n" + "\n".join(f"- {h}" for h in hazards) + "\n")
        if blocking:
            lines.append("**⛔ Blocking issues:**\n" + "\n".join(f"- {b}" for b in blocking) + "\n")
        if conditions:
            lines.append("**📋 Required conditions:**\n" + "\n".join(f"- {c}" for c in conditions) + "\n")
        lines.append("---\n")
    return "\n".join(lines)


def _build_skeptic_cot(audit: dict) -> str:
    lines = []
    summary = audit.get("audit_summary", "")
    if summary:
        lines.append(f"### Arithmetic & Consistency Audit\n\n{summary}\n")

    all_errors = audit.get("all_errors", [])
    if not all_errors:
        lines.append("✅ **All arithmetic checks passed.** No calculation errors, threshold "
                     "violations, or scope issues detected.")
        return "\n".join(lines)

    for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        errs = [e for e in all_errors if e.get("severity") == severity]
        if not errs:
            continue
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(severity, "⚪")
        lines.append(f"\n#### {icon} {severity} severity ({len(errs)} issues)\n")
        for e in errs:
            lines.append(
                f"**[{e.get('error_type', '?')}]** candidate {e.get('candidate_id', '?')} "
                f"({e.get('agent', '?')})\n\n> {e.get('description', '')}\n"
            )

    disq = audit.get("disqualify_recommendations", [])
    if disq:
        lines.append(f"\n#### 🚫 Disqualification Recommendations\n")
        for r in disq:
            lines.append(f"- Candidate {r['candidate_id']}: {r['reason']}\n")

    return "\n".join(lines)


def _build_chief_cot(
    chief_data: dict,
    weighted_scores: list[dict],
    winner_id: Optional[int],
    dfmea: Optional[dict],
) -> str:
    lines = []

    rationale = chief_data.get("selection_rationale", "")
    if rationale:
        lines.append(f"### Selection Rationale\n\n{rationale}\n")

    # Scoring health warning — shown prominently if domain agents failed
    survivors = [r for r in weighted_scores if not r["disqualified"]]
    all_default_domains = {"chemistry", "kinetics", "fluidics", "safety"}
    failed_scoring = [
        r["candidate_id"] for r in survivors
        if all(abs(r[d] - 0.5) < 0.01 for d in all_default_domains)
    ]
    if failed_scoring:
        lines.append(
            f"\n> ⚠️ **SCORING FAILURE** — domain agents returned no per-candidate scores "
            f"for candidates {failed_scoring}. All domain scores are at the 0.500 default. "
            f"The selection below is based on geometry only and has no engineering basis. "
            f"Re-run or check the scoring agent logs.\n"
        )

    # Score comparison table
    if survivors:
        lines.append("### Weighted Score Comparison\n")
        lines.append("| id | combined | chem | kin | fluid | safety | geo | boosted |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in survivors:
            winner_marker = " ★" if r["candidate_id"] == winner_id else ""
            lines.append(
                f"| **{r['candidate_id']}{winner_marker}** | **{r['combined']:.3f}** | "
                f"{r['chemistry']:.3f} | {r['kinetics']:.3f} | {r['fluidics']:.3f} | "
                f"{r['safety']:.3f} | {r['geometry']:.3f} | {r.get('objective_boosted', '')} |"
            )
        lines.append("")

    # Resolved trade-offs
    tradeoffs = chief_data.get("resolved_tradeoffs") or []
    if tradeoffs:
        lines.append("### Resolved Trade-offs\n")
        for t in tradeoffs:
            lines.append(f"- {t}")
        lines.append("")

    # Overridden concerns
    overrides = chief_data.get("overridden_concerns") or []
    if overrides:
        lines.append("### Overridden Concerns\n")
        for o in (overrides if isinstance(overrides[0], dict) else
                  [{"agent": "?", "concern": str(x), "override_reason": ""} for x in overrides]):
            lines.append(
                f"- **{o.get('agent', '?')}**: {o.get('concern', '')} "
                f"→ *{o.get('override_reason', '')}*"
            )
        lines.append("")

    # Remaining uncertainties
    uncerts = chief_data.get("remaining_uncertainties") or []
    if uncerts:
        lines.append("### Remaining Uncertainties\n")
        for u in uncerts:
            lines.append(f"- {u}")
        lines.append("")

    # Pump flowrates derived by chief
    pump_frs = chief_data.get("pump_flowrates") or []
    if pump_frs:
        lines.append("### Pump Flowrate Derivation\n")
        lines.append("| Stream | Role | Q (mL/min) | Derivation |")
        lines.append("|---|---|---|---|")
        for pf in pump_frs:
            if isinstance(pf, dict):
                lines.append(
                    f"| {pf.get('stream_label', '?')} | {pf.get('role', '?')} | "
                    f"**{pf.get('Q_mL_min', '?')}** | {pf.get('derivation', '')} |"
                )
        n_lim_shown = chief_data.get("n_limiting_mmol_min")
        if n_lim_shown:
            lines.append(f"\nṅ_limiting = **{n_lim_shown} mmol/min** | Q_total = **{chief_data.get('Q_total_mL_min', '?')} mL/min**\n")

    # Ask-back
    ask_d = chief_data.get("ask_expert")
    ask_q = chief_data.get("ask_expert_question")
    if ask_d and ask_q:
        lines.append(f"\n### Chief Ask-Back\n\nDomain consulted: **{ask_d}**\n\nQuestion: {ask_q}\n")

    # Validation experiment
    exp = chief_data.get("experiment_recommendation")
    if exp:
        lines.append(f"### Recommended First Experiment\n\n{exp}\n")

    # DFMEA
    if dfmea and dfmea.get("failure_modes"):
        lines.append("### DFMEA — Failure Mode Analysis\n")
        lines.append("| Rank | Mode | Cause | Severity | Likelihood | RPN | Mitigation |")
        lines.append("|---|---|---|---|---|---|---|")
        for fm in dfmea["failure_modes"]:
            lines.append(
                f"| {fm.get('rank', '?')} | {fm.get('mode', '')} | {fm.get('cause', '')} | "
                f"{fm.get('severity', '?')} | {fm.get('likelihood', '?')} | "
                f"**{fm.get('RPN', '?')}** | {fm.get('mitigation', '')} |"
            )
        lines.append("")
        spof = dfmea.get("single_points_of_failure") or []
        if spof:
            lines.append("**Single points of failure:**\n" + "\n".join(f"- {s}" for s in spof) + "\n")
        val_exp = dfmea.get("validation_experiments") or []
        if val_exp:
            lines.append("**Validation experiments:**\n" + "\n".join(f"1. {v}" for v in val_exp) + "\n")

    return "\n".join(lines)


def _to_deliberations_v4(
    designer_result: dict,
    scoring: dict,
    audit: dict,
    weighted_scores: list[dict],
    winner_id: Optional[int],
    candidates: list[dict],
    dfmea: Optional[dict],
    chief_data: dict,
    tool_calls: dict,
    revision_result: Optional[dict] = None,
) -> list[list[AgentDeliberation]]:
    """Convert v4 council output to UI-compatible AgentDeliberation lists with rich markdown."""
    round1: list[AgentDeliberation] = []

    # ── Designer ─────────────────────────────────────────────────────────────
    n_surv = len(designer_result.get("survivors", []))
    n_disq = len(designer_result.get("disqualified", []))
    ps = designer_result.get("problem_statement", {})
    disq_list = designer_result.get("disqualified", [])

    designer_cot_lines = [
        f"### Stage 0 — Problem Framing\n",
        f"**Reaction class:** {ps.get('reaction_class', 'unknown')}  \n"
        f"**Flags:** {', '.join(ps.get('special_flags', [])) or 'none'}  \n"
        f"**Flow justified:** {'Yes' if ps.get('flow_justified', True) else '⚠️ Questionable — ' + ps.get('flow_justification_note', '')}",
        f"\n### Stage 1 — Candidate Matrix\n",
        f"**Strategy:** {designer_result.get('strategy_reasoning', '')}  \n"
        f"**Candidates to council:** {n_surv}  \n"
        f"**Hard-gate flagged (proceed with warnings):** {n_disq}",
    ]
    if disq_list:
        designer_cot_lines.append("\n**Hard-gate flag details** *(candidates still evaluated — flags inform scoring agents)*:")
        for d in disq_list[:8]:
            cand = d.get("candidate", {})
            designer_cot_lines.append(
                f"- ⛔ Candidate {cand.get('id', '?')} (τ={cand.get('tau_min', '?')} min, "
                f"d={cand.get('d_mm', '?')} mm): {d.get('reason', '')}"
            )

    envelope = designer_result.get("design_envelope_preliminary", {})
    if envelope:
        designer_cot_lines.append(
            f"\n**Preliminary design envelope:** τ ∈ {envelope.get('tau_range', '?')} min | "
            f"d ∈ {envelope.get('d_range', '?')} mm | Q ∈ {envelope.get('Q_range', '?')} mL/min"
        )

    round1.append(AgentDeliberation(
        agent="DesignerV4", agent_display_name="Designer",
        round=1,
        chain_of_thought="\n".join(designer_cot_lines),
        findings=[
            f"Candidates to council: {n_surv} (all proceed to scoring)",
            f"Hard-gate flagged: {n_disq} carry warning flags visible to all agents",
        ],
        status="ACCEPT",
    ))

    # ── Dr. Chemistry ─────────────────────────────────────────────────────────
    chem_blocked = [e for e in scoring.get("chemistry_scores", [])
                    if str(e.get("verdict", "")).upper() == "BLOCK"]
    n_chem = len(scoring.get("chemistry_scores", []))

    round1.append(AgentDeliberation(
        agent="DrChemistryV4", agent_display_name="Dr. Chemistry",
        round=1,
        chain_of_thought=_build_chemistry_cot(scoring),
        findings=[
            f"Scored {n_chem} candidates across chemistry and photonics domains",
            f"Blocked: {len(chem_blocked)} | Accepted: {n_chem - len(chem_blocked)}",
        ],
        concerns=[
            f"**Candidate {e['candidate_id']} BLOCKED:** {', '.join(e.get('blocking_issues') or ['unspecified'])}"
            for e in chem_blocked
        ],
        status="ACCEPT" if not chem_blocked else "WARNING",
        tool_calls=scoring.get("tool_calls", {}).get("chemistry", []),
    ))

    # ── Dr. Kinetics ──────────────────────────────────────────────────────────
    kin_blocked = [e for e in scoring.get("kinetics_scores", [])
                   if str(e.get("verdict", "")).upper() == "BLOCK"]
    n_kin = len(scoring.get("kinetics_scores", []))

    round1.append(AgentDeliberation(
        agent="DrKineticsV4", agent_display_name="Dr. Kinetics",
        round=1,
        chain_of_thought=_build_kinetics_cot(scoring),
        findings=[
            f"Scored {n_kin} candidates across kinetics, conversion, and integration domains",
            f"Blocked: {len(kin_blocked)} | Accepted: {n_kin - len(kin_blocked)}",
        ],
        concerns=[
            f"**Candidate {e['candidate_id']} BLOCKED:** {', '.join(e.get('concerns') or ['X below minimum'])}"
            for e in kin_blocked
        ],
        status="ACCEPT" if not kin_blocked else "WARNING",
        tool_calls=scoring.get("tool_calls", {}).get("kinetics", []),
    ))

    # ── Dr. Fluidics ──────────────────────────────────────────────────────────
    flu_blocked = [e for e in scoring.get("fluidics_scores", [])
                   if str(e.get("verdict", "")).upper() == "BLOCK"]
    n_flu = len(scoring.get("fluidics_scores", []))

    round1.append(AgentDeliberation(
        agent="DrFluidicsV4", agent_display_name="Dr. Fluidics",
        round=1,
        chain_of_thought=_build_fluidics_cot(scoring),
        findings=[
            f"Scored {n_flu} candidates across fluidics, transport phenomena, and hardware",
            f"Blocked: {len(flu_blocked)} | Accepted: {n_flu - len(flu_blocked)}",
        ],
        concerns=[
            f"**Candidate {e['candidate_id']} BLOCKED:** {', '.join(e.get('concerns') or ['Re or geometry limit'])}"
            for e in flu_blocked
        ],
        status="ACCEPT" if not flu_blocked else "WARNING",
        tool_calls=scoring.get("tool_calls", {}).get("fluidics", []),
    ))

    # ── Dr. Safety ────────────────────────────────────────────────────────────
    saf_blocked = [e for e in scoring.get("safety_scores", [])
                   if str(e.get("verdict", "")).upper() == "BLOCK"]
    n_saf = len(scoring.get("safety_scores", []))

    round1.append(AgentDeliberation(
        agent="DrSafetyV4", agent_display_name="Dr. Safety",
        round=1,
        chain_of_thought=_build_safety_cot(scoring),
        findings=[
            f"Scored {n_saf} candidates across thermal safety, BPR, materials, and hazards",
            f"Blocked: {len(saf_blocked)} | Accepted/Conditional: {n_saf - len(saf_blocked)}",
        ],
        concerns=[
            f"**Candidate {e['candidate_id']} BLOCKED:** {', '.join(e.get('blocking_issues') or ['safety gate failed'])}"
            for e in saf_blocked
        ],
        status="ACCEPT" if not saf_blocked else "WARNING",
        tool_calls=scoring.get("tool_calls", {}).get("safety", []),
    ))

    # ── Skeptic ───────────────────────────────────────────────────────────────
    errors = audit.get("all_errors", [])
    critical_count = sum(1 for e in errors if e.get("severity") == "CRITICAL")
    high_count = sum(1 for e in errors if e.get("severity") == "HIGH")

    round1.append(AgentDeliberation(
        agent="SkepticV4", agent_display_name="Skeptic",
        round=1,
        chain_of_thought=_build_skeptic_cot(audit),
        findings=[
            f"Audit complete: {len(errors)} issues found "
            f"({critical_count} CRITICAL, {high_count} HIGH)",
            f"Disqualification recommendations: {len(audit.get('disqualify_recommendations', []))} candidates",
            f"Council may proceed: {'✅ Yes' if audit.get('council_may_proceed') else '🔴 No — CRITICAL errors must be resolved'}",
        ],
        concerns=[
            f"[{e['severity']}] Candidate {e.get('candidate_id')} — "
            f"{e.get('error_type', '?')}: {e['description'][:200]}"
            for e in errors if e.get("severity") in ("CRITICAL", "HIGH")
        ],
        status=("REVISE" if not audit.get("council_may_proceed") else
                ("WARNING" if errors else "ACCEPT")),
    ))

    # ── Revision Agent (Stage 3.5) ────────────────────────────────────────────
    if revision_result is not None:
        rev_rationale = revision_result.get("revision_rationale", {})
        rev_domains   = revision_result.get("revision_domains", [])
        rev_lines = [
            "### Stage 3.5 — Parameter Revision\n",
            f"**Domains that triggered revision:** {', '.join(rev_domains)}\n",
            "**Changes applied to winner:**\n",
        ]
        if rev_rationale:
            for param, reason in rev_rationale.items():
                rev_lines.append(f"- **{param}**: {reason}")
        else:
            rev_lines.append("(No detailed rationale provided)")

        round1.append(AgentDeliberation(
            agent="RevisionAgent", agent_display_name="Revision Engineer",
            round=1,
            chain_of_thought="\n".join(rev_lines),
            findings=[
                f"Revised {len(rev_rationale)} parameter(s) on winner id={winner_id}: "
                + ", ".join(f"{k}={v}" for k, v in {
                    "tau_min": revision_result.get("tau_min"),
                    "d_mm": revision_result.get("d_mm"),
                    "BPR_bar": revision_result.get("BPR_bar"),
                    "tubing_material": revision_result.get("tubing_material"),
                }.items() if k in rev_rationale)
            ],
            status="REVISE",
        ))

    # ── Chief Engineer + DFMEA ────────────────────────────────────────────────
    round2: list[AgentDeliberation] = []
    if winner_id is not None:
        w_entry = next((r for r in weighted_scores if r["candidate_id"] == winner_id), {})

        round2.append(AgentDeliberation(
            agent="ChiefV4", agent_display_name="Chief Engineer",
            round=2,
            chain_of_thought=_build_chief_cot(chief_data, weighted_scores, winner_id, dfmea),
            findings=[
                f"**Winner: Candidate {winner_id}** | Combined score: {w_entry.get('combined', '?'):.3f}",
                f"Runner-ups: {chief_data.get('runner_up_ids', [])}",
                f"Objective boosted: {w_entry.get('objective_boosted', 'none')}",
            ],
            concerns=[str(u) for u in (chief_data.get("remaining_uncertainties") or [])[:5]],
            status="ACCEPT",
            tool_calls=tool_calls.get("chief", []),
        ))

    return [round1, round2] if round2 else [round1]


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_summary_v4(
    designer_result: dict,
    scoring: dict,
    audit: dict,
    weighted_scores: list[dict],
    winner_id: Optional[int],
    candidates: list[dict],
    chief_data: dict,
    dfmea: Optional[dict],
    objectives: str,
) -> str:
    lines = ["## Council v4 — Stage-gated Engineering Design Summary"]
    lines.append(f"\n**Objectives**: {objectives}")
    lines.append(f"**Designer**: {len(designer_result.get('survivors', []))} survivors, "
                 f"{len(designer_result.get('disqualified', []))} hard-gate disqualified")
    lines.append(f"\n### Candidate Shortlist\n{designer_result.get('table_markdown', '')}")

    lines.append("\n### Weighted Scores (all surviving candidates)")
    lines.append("| id | combined | chemistry | kinetics | fluidics | safety | geometry | disq |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in weighted_scores:
        lines.append(
            f"| {r['candidate_id']} | {r['combined']:.3f} | "
            f"{r['chemistry']:.3f} | {r['kinetics']:.3f} | {r['fluidics']:.3f} | "
            f"{r['safety']:.3f} | {r['geometry']:.3f} | "
            f"{'✗' if r['disqualified'] else ''} |"
        )

    lines.append(f"\n### Skeptic Audit\n{audit.get('audit_summary', '')}")
    if audit.get("all_errors"):
        for e in audit["all_errors"]:
            if e.get("severity") in ("CRITICAL", "HIGH"):
                lines.append(f"- [{e['severity']}] id {e.get('candidate_id')}: {e['description'][:200]}")

    if winner_id is not None:
        w = next((c for c in candidates if c.get("id") == winner_id), {})
        lines.append(f"\n### Winner — candidate id={winner_id}")
        lines.append(
            f"- τ={w.get('tau_min')} min, d={w.get('d_mm')} mm, "
            f"Q={w.get('Q_mL_min')} mL/min, V_R={w.get('V_R_mL')} mL, "
            f"L={w.get('L_m')} m"
        )
        lines.append(
            f"- Re={w.get('Re', 0):.0f}, ΔP={w.get('delta_P_bar', 0):.3f} bar, "
            f"r_mix={w.get('r_mix', 0):.3f}, X={w.get('expected_conversion', 0):.2f}, "
            f"prod={w.get('productivity_mg_h', 0):.1f} mg/h"
        )
        lines.append(f"- **Rationale**: {chief_data.get('selection_rationale', '')}")

        if chief_data.get("remaining_uncertainties"):
            lines.append("- **Remaining uncertainties**:")
            for u in chief_data["remaining_uncertainties"]:
                lines.append(f"  - {u}")

        if dfmea and dfmea.get("failure_modes"):
            lines.append("\n### DFMEA — Top Failure Modes")
            for fm in dfmea["failure_modes"]:
                lines.append(
                    f"- [{fm.get('rank')}] {fm.get('mode')}: "
                    f"severity={fm.get('severity')}, likelihood={fm.get('likelihood')}, "
                    f"RPN={fm.get('RPN')} | mitigation: {fm.get('mitigation', '')}"
                )
            if dfmea.get("validation_experiments"):
                lines.append("\n### Validation Experiments")
                for exp in dfmea["validation_experiments"]:
                    lines.append(f"- {exp}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_chemistry_brief(
    batch: BatchRecord,
    plan: Optional[ChemistryPlan],
    proposal: FlowProposal,
) -> str:
    rc = (plan.reaction_class if plan else "unknown") or "unknown"
    mech = (plan.mechanism_type if plan else "") or "—"
    n_stages = plan.n_stages if plan else 1
    o2 = bool(plan and plan.oxygen_sensitive)
    quench = bool(plan and plan.quench_required)
    atm = ""
    if plan and plan.stages:
        atm = " | atm: " + "/".join(s.atmosphere for s in plan.stages)
    return (
        f"reaction_class: {rc} | mechanism: {mech} | n_stages: {n_stages} | "
        f"O₂-sensitive: {o2} | quench_required: {quench}{atm}\n"
        f"solvent: {proposal.streams[0].solvent if proposal.streams else 'EtOH'} | "
        f"T: {proposal.temperature_C}°C | C: {proposal.concentration_M} M"
    )


def _extract_extinction_coeff(
    batch: BatchRecord,
    plan: Optional[ChemistryPlan],
) -> Optional[float]:
    for attr in ("extinction_coeff_M_cm", "epsilon_M_cm", "photocatalyst_epsilon"):
        for obj in (batch, plan):
            if obj is None:
                continue
            v = getattr(obj, attr, None)
            try:
                if v is not None and float(v) > 0:
                    return float(v)
            except (TypeError, ValueError):
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  CouncilV4 — main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class CouncilV4:
    """Stage-gated multi-agent scoring council.

    Usage (same API as CouncilV3):
        result, calc = CouncilV4().run(proposal, batch, analogies, inventory,
                                        chemistry_plan=..., calculations=...,
                                        objectives="de-risk first-run")
    """

    def run(
        self,
        proposal: FlowProposal,
        batch_record: BatchRecord,
        analogies: list[dict],
        inventory: LabInventory,
        chemistry_plan: Optional[ChemistryPlan] = None,
        calculations: Optional[DesignCalculations] = None,
        objectives: str = "balanced",
        max_loop_rounds: int = 3,
        fixed_candidates: Optional[list[dict]] = None,
    ) -> tuple[DesignCandidate, DesignCalculations]:
        """Run the full council pipeline.

        If *fixed_candidates* is provided, Stage 1 (Designer) is skipped and
        those candidates are passed directly to Stage 2 (domain scoring).
        Use this for ablation studies comparing 1-candidate vs N-candidate modes.
        """

        current = proposal.model_copy(deep=True)
        log = DeliberationLog()

        # ── Ensure calculator center-point ──────────────────────────────────
        if calculations is None:
            calculations = DesignCalculator().run(
                batch_record, chemistry_plan, current, inventory,
                analogies=analogies,
                target_flow_rate_mL_min=current.flow_rate_mL_min or None,
                target_tubing_ID_mm=current.tubing_ID_mm or None,
            )
        calc = calculations

        # ── Extract context ─────────────────────────────────────────────────
        is_photochem = bool(
            (chemistry_plan and "photo" in (chemistry_plan.reaction_class or "").lower()) or
            (current.wavelength_nm and current.wavelength_nm > 0)
        )
        is_gas_liquid = bool(getattr(calc, "is_gas_liquid", False))
        is_O2_sensitive = bool(chemistry_plan and chemistry_plan.oxygen_sensitive)

        solvent = (
            chemistry_plan.stages[0].solvent
            if chemistry_plan and chemistry_plan.stages
            else (current.streams[0].solvent if current.streams else "EtOH")
        )

        tau_center = calc.residence_time_min or current.residence_time_min or 30.0
        tau_lit = getattr(calc, "tau_analogy_min", None) or getattr(calc, "tau_class_min", None)
        tau_kinetics = getattr(calc, "tau_kinetics_min", None) or tau_center
        IF_used = calc.intensification_factor or 6.0
        assumed_MW = getattr(batch_record, "product_MW", None) or 250.0
        pump_max = calc.pump_max_bar or 20.0
        ext_coeff = _extract_extinction_coeff(batch_record, chemistry_plan)
        chem_brief = _build_chemistry_brief(batch_record, chemistry_plan, current)
        reaction_class = (chemistry_plan.reaction_class if chemistry_plan else "unknown") or "unknown"

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 0 — Problem Framing
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Stage 0: Problem framing")
        problem_statement = run_problem_framing(
            reaction_class=reaction_class,
            is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
            is_O2_sensitive=is_O2_sensitive,
            tau_center_min=tau_center, tau_lit_min=tau_lit,
            solvent=solvent, temperature_C=current.temperature_C,
            concentration_M=current.concentration_M,
            objectives=objectives,
        )
        logger.info(
            "    Problem framing: reaction_class=%s | flags=%s | flow_justified=%s",
            problem_statement.get("reaction_class"),
            problem_statement.get("special_flags"),
            problem_statement.get("flow_justified"),
        )

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 1 — Designer: Candidate Matrix  (or bypass with fixed_candidates)
        # ═══════════════════════════════════════════════════════════════════
        if fixed_candidates is not None:
            logger.info(
                "  Council v4 — Stage 1: BYPASSED (fixed_candidates=%d provided)",
                len(fixed_candidates),
            )
            from flora_translate.engine.sampling import format_candidate_table
            survivors = fixed_candidates
            table_markdown = format_candidate_table(survivors, max_rows=len(survivors))
            designer_result = {
                "survivors":                  survivors,
                "disqualified":               [],
                "table_markdown":             table_markdown,
                "strategy_reasoning":         f"Stage 1 bypassed — {len(survivors)} fixed candidate(s) provided",
                "design_envelope_preliminary":{},
                "problem_statement":          problem_statement,
            }
        else:
            logger.info("  Council v4 — Stage 1: Designer candidate matrix")
            designer_result = run_designer_v4(
                reaction_class=reaction_class,
                is_photochem=is_photochem, is_gas_liquid=is_gas_liquid,
                is_O2_sensitive=is_O2_sensitive,
                tau_center_min=tau_center, tau_lit_min=tau_lit,
                tau_kinetics_min=tau_kinetics,
                d_center_mm=current.tubing_ID_mm or 1.0,
                Q_center_mL_min=current.flow_rate_mL_min or 1.0,
                solvent=solvent, temperature_C=current.temperature_C,
                concentration_M=current.concentration_M,
                assumed_MW=assumed_MW, IF_used=IF_used,
                pump_max_bar=pump_max,
                BPR_bar=current.BPR_bar or 0.0,
                extinction_coeff_M_cm=ext_coeff,
                tubing_material=current.tubing_material or "FEP",
                N_target=12,
                problem_statement=problem_statement,
            )

        survivors = designer_result["survivors"]
        if not survivors:
            logger.warning("Council v4: no survivors after hard-gate filter — falling back")
            return self._fallback(current, chemistry_plan, calc, log, designer_result)

        table_markdown = designer_result["table_markdown"]

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 2 — Domain Scoring (max max_loop_rounds iterations if needed)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Stage 2: Domain scoring (%d survivors)", len(survivors))
        scoring = run_domain_scoring(
            candidates=survivors,
            table_markdown=table_markdown,
            chemistry_brief=chem_brief,
            objectives=objectives,
            is_photochem=is_photochem,
            pump_max_bar=pump_max,
        )

        # Remove scoring-blocked candidates from survivors for audit
        scoring_blocked_ids = set(scoring.get("blocked_by_scoring", []))
        survivors_for_audit = [c for c in survivors if c.get("id") not in scoring_blocked_ids]

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 3 — Skeptic Audit
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Stage 3: Skeptic audit")
        audit = run_skeptic_audit(
            candidates=survivors,
            chemistry_scores=scoring["chemistry_scores"],
            kinetics_scores=scoring["kinetics_scores"],
            fluidics_scores=scoring["fluidics_scores"],
            safety_scores=scoring["safety_scores"],
            is_gas_liquid=is_gas_liquid,
            concentration_M=current.concentration_M,
            pump_max_bar=pump_max,
        )

        # If CRITICAL errors found, log and continue (v4 selects best available)
        if not audit["council_may_proceed"]:
            logger.warning(
                "  Council v4 — Skeptic found CRITICAL errors: %s. "
                "Proceeding with disqualification of affected candidates.",
                [e["description"][:80] for e in audit["all_errors"]
                 if e.get("severity") == "CRITICAL"]
            )

        # Collect disqualified IDs — hard-gate flags are NOT removals; only scoring
        # blocks and Skeptic CRITICAL disqualifications actually remove a candidate.
        disqualify_ids = (
            scoring_blocked_ids
            | set(audit.get("disqualify_ids", []))
        )

        # ═══════════════════════════════════════════════════════════════════
        #  CHIEF ENGINEER — Weighted Selection (with optional ask-back)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Chief Engineer: weighted selection")
        weighted_scores = compute_weighted_scores(
            candidates=survivors,
            scoring=scoring,
            objectives=objectives,
            disqualify_ids=disqualify_ids,
        )

        # Pass ṅ_limiting from calculator so chief can derive per-pump Q
        n_lim = getattr(calc, "n_molar_flow_mmol_min", None)
        P_batch_val = getattr(calc, "P_batch_mmol_h", None)

        chief_data, chief_tc = _run_chief_llm(
            candidates=survivors,
            weighted_scores=weighted_scores,
            scoring=scoring,
            audit=audit,
            table_markdown=table_markdown,
            chemistry_brief=chem_brief,
            objectives=objectives,
            disqualify_ids=disqualify_ids,
            n_limiting_mmol_min=n_lim,
            P_batch_mmol_h=P_batch_val,
        )

        # Ask-back round: chief may request one expert clarification (max once)
        ask_domain = chief_data.get("ask_expert")
        ask_question = chief_data.get("ask_expert_question")
        if ask_domain and ask_question and ask_domain.lower() in ("kinetics", "chemistry"):
            logger.info(
                "  Chief ask-back: requesting %s expert for: %s",
                ask_domain, str(ask_question)[:100],
            )
            expert_answer = _run_expert_askback(
                domain=ask_domain,
                question=ask_question,
                chemistry_brief=chem_brief,
                candidates=survivors,
                scoring=scoring,
            )
            chief_data2, chief_tc2 = _run_chief_llm(
                candidates=survivors,
                weighted_scores=weighted_scores,
                scoring=scoring,
                audit=audit,
                table_markdown=table_markdown,
                chemistry_brief=chem_brief,
                objectives=objectives,
                disqualify_ids=disqualify_ids,
                n_limiting_mmol_min=n_lim,
                P_batch_mmol_h=P_batch_val,
                expert_answer=f"[{ask_domain.upper()} EXPERT] {expert_answer}",
            )
            if chief_data2.get("selected_candidate_id") is not None:
                chief_data = chief_data2
                chief_tc = chief_tc + chief_tc2
                logger.info("  Chief re-ran after ask-back — updated decision applied")

        winner_id = chief_data.get("selected_candidate_id")
        try:
            winner_id = int(winner_id) if winner_id is not None else None
        except (TypeError, ValueError):
            winner_id = None

        # Validate winner_id is in survivors and not disqualified
        valid_survivor_ids = {c.get("id") for c in survivors if c.get("id") not in disqualify_ids}
        if winner_id not in valid_survivor_ids:
            winner_id, fallback_reason = _deterministic_resolve(weighted_scores)
            logger.info("  Chief LLM invalid/failed — deterministic: %s", fallback_reason)
            chief_data.setdefault("selection_rationale", fallback_reason)

        if winner_id is None:
            return self._fallback(current, chemistry_plan, calc, log, designer_result)

        winner = next(c for c in survivors if c.get("id") == winner_id)
        logger.info(
            "  Council v4 — Winner: id=%d τ=%s min, d=%s mm, Q=%s mL/min, X=%s",
            winner_id, winner.get("tau_min"), winner.get("d_mm"),
            winner.get("Q_mL_min"), winner.get("expected_conversion"),
        )

        # ═══════════════════════════════════════════════════════════════════
        #  STAGE 3.5 — Revision Agent: fix REVISE/BLOCK issues on winner
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — Stage 3.5: Revision Agent on winner id=%d", winner_id)
        revision_result = run_revision_stage(
            winner           = winner,
            scoring          = scoring,
            chemistry_brief  = chem_brief,
            is_photochem     = is_photochem,
            pump_max_bar     = pump_max,
            solvent          = solvent,
            temperature_C    = current.temperature_C,
            concentration_M  = current.concentration_M,
            extinction_coeff_M_cm = ext_coeff,
        )
        if revision_result is not None:
            winner = revision_result
            logger.info(
                "  Stage 3.5 applied revisions to winner id=%d: τ→%.1f min, d→%.2f mm, BPR→%.2f bar",
                winner_id, winner.get("tau_min", 0), winner.get("d_mm", 0), winner.get("BPR_bar", 0),
            )

        # ═══════════════════════════════════════════════════════════════════
        #  DFMEA on winner
        # ═══════════════════════════════════════════════════════════════════
        logger.info("  Council v4 — DFMEA on winning candidate id=%d", winner_id)
        safety_entry = next(
            (e for e in scoring["safety_scores"] if e.get("candidate_id") == winner_id),
            None,
        )
        dfmea = run_dfmea(
            winner_id=winner_id,
            winner=winner,
            chemistry_brief=chem_brief,
            safety_score_entry=safety_entry,
        )

        # ═══════════════════════════════════════════════════════════════════
        #  Apply winner + re-run calculator
        # ═══════════════════════════════════════════════════════════════════
        final_consensus = chief_data.get("final_consensus", {})
        extra_changes: dict = {}
        for field_map in [
            ("BPR_bar", "BPR_bar"),
            ("material", "tubing_material"),
        ]:
            fc_key, patch_key = field_map
            if fc_key in final_consensus:
                extra_changes[patch_key] = str(final_consensus[fc_key])

        # Revision Stage 3.5: also apply BPR and material from revised winner
        # (these are not in chief final_consensus, but were set by revision agent)
        if revision_result is not None:
            if "BPR_bar" not in extra_changes and revision_result.get("BPR_bar") is not None:
                extra_changes["BPR_bar"] = str(revision_result["BPR_bar"])
            if "tubing_material" not in extra_changes and revision_result.get("tubing_material"):
                extra_changes["tubing_material"] = revision_result["tubing_material"]

        current, changes = _apply_winner(
            current, winner, extra_changes, chief_data=chief_data,
        )
        log.all_changes_applied.update(changes)

        calc = DesignCalculator().run(
            batch_record, chemistry_plan, current, inventory,
            analogies=analogies,
            target_flow_rate_mL_min=current.flow_rate_mL_min or None,
            target_tubing_ID_mm=current.tubing_ID_mm or None,
            target_residence_time_min=current.residence_time_min or None,
        )

        # ── Build output ────────────────────────────────────────────────────
        tool_calls = {
            "chief": chief_tc,
            **scoring.get("tool_calls", {}),
        }
        all_round_delibs = _to_deliberations_v4(
            designer_result=designer_result,
            scoring=scoring,
            audit=audit,
            weighted_scores=weighted_scores,
            winner_id=winner_id,
            candidates=survivors,
            dfmea=dfmea,
            chief_data=chief_data,
            tool_calls=tool_calls,
            revision_result=revision_result,
        )
        log.rounds = all_round_delibs

        # Skeptic trade-off summary
        log.trade_off_summary = audit.get("audit_summary", "")
        log.trade_off_matrix = (
            "| id | combined | chemistry | kinetics | fluidics | safety | geometry |\n"
            "|---|---|---|---|---|---|---|\n" +
            "\n".join(
                f"| {r['candidate_id']} | {r['combined']:.3f} | "
                f"{r['chemistry']:.3f} | {r['kinetics']:.3f} | "
                f"{r['fluidics']:.3f} | {r['safety']:.3f} | {r['geometry']:.3f} |"
                for r in weighted_scores
            )
        )
        log.total_rounds = 2
        log.consensus_reached = audit["council_may_proceed"]

        log.summary = _build_summary_v4(
            designer_result=designer_result,
            scoring=scoring,
            audit=audit,
            weighted_scores=weighted_scores,
            winner_id=winner_id,
            candidates=survivors,
            chief_data=chief_data,
            dfmea=dfmea,
            objectives=objectives,
        )

        legacy = [
            CouncilMessage(
                agent=d.agent, status=d.status, field="design",
                value=d.chain_of_thought[:200],
                concern=(d.concerns[0] if d.concerns else ""),
                revision_required=(d.status == "REVISE"),
            )
            for rnd in all_round_delibs
            for d in rnd
        ]

        current.engine_validated = True
        current.confidence = "HIGH" if winner.get("pareto_front") else "MEDIUM"

        return DesignCandidate(
            proposal=current,
            chemistry_plan=chemistry_plan,
            council_messages=legacy,
            council_rounds=log.total_rounds,
            safety_report={
                "total_checks": len(audit.get("all_errors", [])),
                "critical": len([e for e in audit.get("all_errors", [])
                                 if e.get("severity") == "CRITICAL"]),
                "dfmea_modes": len(dfmea.get("failure_modes", [])) if dfmea else 0,
                "validation_experiments": dfmea.get("validation_experiments", []) if dfmea else [],
            },
            unit_operations=[],
            pid_description="",
            deliberation_log=log,
        ), calc

    def _fallback(
        self,
        current: FlowProposal,
        chemistry_plan: Optional[ChemistryPlan],
        calc: DesignCalculations,
        log: DeliberationLog,
        designer_result: dict,
    ) -> tuple[DesignCandidate, DesignCalculations]:
        n_surv = len(designer_result.get("survivors", []))
        n_disq = len(designer_result.get("disqualified", []))
        log.summary = (
            f"Council v4 fallback: no usable candidates after filtering. "
            f"Designer: {n_surv} survivors, {n_disq} hard-gate disqualified. "
            "Keeping calculator center-point."
        )
        current.engine_validated = False
        return DesignCandidate(
            proposal=current,
            chemistry_plan=chemistry_plan,
            council_messages=[],
            council_rounds=0,
            safety_report={},
            unit_operations=[],
            pid_description="",
            deliberation_log=log,
        ), calc
