#!/usr/bin/env python3
"""Audit actual routing and produce case-level Astra comparisons without rescoring."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    report = root / "report"
    tables, figures = report / "tables", report / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    key = read(root / "frozen/candidate_key_confidential.json")["candidates"]
    call_rows, usage_rows, issues, artifacts, delivered_parameters = [], [], [], [], []
    blocked_rows, graph_rows = [], []
    roles = Counter()
    response_ids = []
    for cell in key:
        run = Path(cell["run_directory"])
        costs, inputs, outputs, cached, reasoning, calls = 0., 0, 0, 0, 0, 0
        for path in sorted(run.rglob("llm_events.jsonl")):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event.get("model") != "gpt-6-astra":
                    issues.append({"run": str(run), "issue": "non-Astra generator call", "model": event.get("model")})
                if event.get("returned_model") != "gpt-6-astra":
                    issues.append({"run": str(run), "issue": "unexpected returned model", "model": event.get("returned_model")})
                response_ids.append(event.get("response_id"))
                if event.get("reasoning_effort") != "medium":
                    issues.append({"run": str(run), "issue": "reasoning setting mismatch"})
                u = event.get("usage") or {}
                inp, out = u.get("input_tokens", 0), u.get("output_tokens", 0)
                details = u.get("input_tokens_details") or {}
                cache = details.get("cached_tokens", 0)
                cache_write = details.get("cache_write_tokens", 0)
                reason = (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
                # List-price estimate, not an invoice. Cache categories partition
                # total input; reasoning is already included in output_tokens.
                input_factor, output_factor = (2., 1.5) if inp > 272000 else (1., 1.)
                cost = (input_factor * (max(0, inp-cache-cache_write)*10 + cache*1 + cache_write*12.5)
                        + output_factor*out*50) / 1e6
                row = {"case": cell["case"], "architecture": cell["architecture"],
                       "repeat_id": cell["repeat_id"], "candidate_id": cell["candidate_id"],
                       "api_name": event.get("api_name"), "model": event.get("model"),
                       "response_id": event.get("response_id"),
                       "returned_model": event.get("returned_model"), "input_tokens": inp,
                       "output_tokens": out, "cached_tokens": cache, "cache_write_tokens": cache_write,
                       "reasoning_tokens": reason, "estimated_usd": cost,
                       "duration_ms": event.get("duration_ms"), "source": str(path.relative_to(root))}
                call_rows.append(row)
                roles[event.get("api_name", "unknown")] += 1
                costs += cost; inputs += inp; outputs += out; cached += cache; reasoning += reason; calls += 1
        summary = read(run / "run_summary.json")
        result_path = run / "result.json"
        result = read(result_path) if result_path.exists() else {}
        final = result.get("final_design") or {}
        for failure in (final.get("consistency") or {}).get("issues", []):
            blocked_rows.append({"case": cell["case"], "repeat_id": cell["repeat_id"],
                                 "candidate_id": cell["candidate_id"], "code": failure.get("code"),
                                 "message": failure.get("message"), "source": str(result_path.relative_to(root))})
        diagram_dir = (result.get("diagram_artifacts") or {}).get("run_dir")
        if diagram_dir and Path(diagram_dir).is_dir():
            category = "executable_candidates" if final.get("status") == "executable" else "rejected_diagnostics"
            destination = root / "topologies" / category / cell["case_id"] / cell["repeat_id"]
            shutil.copytree(diagram_dir, destination, dirs_exist_ok=True)
            artifacts.append({"candidate_id": cell["candidate_id"], "final_status": final.get("status"),
                              "source": diagram_dir, "copy": str(destination.relative_to(root))})
            topology_file = Path(diagram_dir) / "topology.json"
            if final.get("status") != "executable" and topology_file.exists():
                for op in read(topology_file).get("unit_operations", []):
                    if "reactor" not in op.get("op_type", ""):
                        continue
                    params = op.get("parameters") or {}
                    q = params.get("Q_liquid_mL_min") or params.get("Q_inlet_mL_min")
                    volume = params.get("volume_mL")
                    graph_rows.append({"case": cell["case"], "repeat_id": cell["repeat_id"],
                                       "operation": op.get("op_id"), "volume_mL": volume,
                                       "liquid_flow_mL_min": q, "stored_time_min": params.get("residence_time_min"),
                                       "stored_basis": params.get("residence_time_basis"),
                                       "compiler_default_basis_if_absent": "liquid-only",
                                       "liquid_only_V_over_Q_min": volume/q if isinstance(volume, (int, float)) and isinstance(q, (int, float)) and q else None})
        packet = read(root / "packets" / f"{cell['candidate_id']}.json")
        values = packet["delivered_final_outcome"].get("parameters") or {}
        for field in ("residence_time_min", "residence_time_basis", "flow_rate_mL_min", "reactor_volume_mL", "temperature_C", "concentration_M", "BPR_bar"):
            delivered_parameters.append({"candidate_id": cell["candidate_id"], "case": cell["case"],
                                         "architecture": cell["architecture"], "repeat_id": cell["repeat_id"],
                                         "field": field, "value": values.get(field), "source": "delivered evaluation packet"})
        usage_rows.append({"candidate_id": cell["candidate_id"], "case": cell["case"],
                           "architecture": cell["architecture"], "repeat_id": cell["repeat_id"],
                           "generation_status": summary.get("status"), "final_status": final.get("status"),
                           "input_tokens": inputs, "output_tokens": outputs, "cached_tokens": cached,
                           "reasoning_tokens": reasoning, "llm_calls": calls, "estimated_usd": costs,
                           "runtime_s": summary.get("runtime_total_s", summary.get("runtime_s"))})
    for case in {row["case"] for row in key}:
        hashes = {row["design_input_sha256"] for row in key if row["case"] == case}
        if len(hashes) != 1:
            issues.append({"case": case, "issue": "authoritative input hash mismatch"})
    if not all(response_ids) or len(set(response_ids)) != len(response_ids):
        issues.append({"issue": "missing or duplicate generation response IDs"})
    audit = {"status": "passed" if not issues else "failed", "issues": issues,
             "generation_llm_calls": len(call_rows), "roles": dict(roles),
             "unique_generation_response_ids": len(set(response_ids)),
             "estimated_generation_cost_usd": sum(row["estimated_usd"] for row in usage_rows),
             "pricing_url": "https://developers.openai.com/api/docs/models/gpt-6-astra",
             "pricing_date": "2026-10-01", "cost_scope": "Successful recorded generation API responses across all retained attempts. Judge calls and preflight excluded. Unreturned/billed failures cannot be inferred.",
             "case_count": len({row["case"] for row in key}), "outcomes": len(key)}
    (report / "routing_and_usage_audit.json").write_text(json.dumps(audit, indent=2))
    write_csv(tables / "generation_api_calls.csv", call_rows)
    write_csv(tables / "generation_usage_by_outcome.csv", usage_rows)
    write_csv(tables / "delivered_parameters.csv", delivered_parameters)
    write_csv(tables / "flowpilot_final_contract_failures.csv", blocked_rows)
    write_csv(tables / "rejected_graph_time_basis_diagnostics.csv", graph_rows)
    (report / "topology_manifest.json").write_text(json.dumps(artifacts, indent=2))
    diagnostics = report / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for name in ("parameter_values_by_repeat.csv", "parameter_repeatability.csv"):
        path = tables / name
        if path.exists():
            shutil.move(str(path), str(diagnostics / f"schema_normalized_{name}"))
    generic_report = report / "REPEATED_BENCHMARK_REPORT.md"
    if generic_report.exists():
        text = generic_report.read_text().replace("UO-09 (solids/slurry handling)", "UO-09 (multistage closure)")
        text = text.replace("- Temperature is None; each call records a distinct seed. Anthropic does not guarantee seed control.",
                            "- Astra used medium reasoning. Temperature and seed were not sent; recorded seeds are bookkeeping only.")
        text = text.replace("- Blinded judges evaluate every outcome:", "- Judges with generator/architecture labels withheld evaluate every outcome (perfect blinding is not claimed):")
        note = "See ASTRA_RESULTS.md and frozen/CONFIGURATION_SCOPE.md for the legacy-policy scope and delivered-output evaluation correction. Schema-normalized parameter summaries are diagnostic only, under diagnostics/."
        if note not in text:
            text += "\n" + note + "\n"
        generic_report.write_text(text)
    shutil.copy2(root / "frozen/CONFIGURATION_SCOPE.md", report / "frozen/CONFIGURATION_SCOPE.md")
    if issues:
        raise RuntimeError("Routing audit failed; inspect report/routing_and_usage_audit.json")

    scores = pd.read_csv(tables / "candidate_scores_with_repeats.csv")
    criteria = pd.read_csv(tables / "criterion_judgments.csv")
    criteria[criteria.critical_error == True].to_csv(tables / "judge_flagged_issues.csv", index=False)
    criteria[criteria.score_0_4 <= 2].to_csv(tables / "judge_material_concerns.csv", index=False)
    cases = ["CuAAC", "Photochemical oxidation", "Hydrogenolysis"]
    colors = {"One-shot": "#B84B4B", "FlowPilot": "#167D8D"}
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), sharey=True, layout="constrained")
    for i, (ax, case) in enumerate(zip(axes, cases)):
        for x, architecture in enumerate(colors):
            vals = scores[(scores.case == case) & (scores.architecture == architecture)].mean_score_0_1
            mean, sd = vals.mean(), vals.std(ddof=1)
            ax.errorbar(x, mean, yerr=sd, fmt="o", markersize=9, capsize=6, lw=2.2, color=colors[architecture])
            ax.scatter(x + np.linspace(-.09, .09, len(vals)), vals, s=22, color=colors[architecture], alpha=.6)
            ax.text(x, max(.06, mean - sd - .065), f"{mean:.3f}\nSD {sd:.3f}", ha="center", va="top", fontsize=11)
        ax.set(title=f"({chr(97+i)}) {case}", xlim=(-.5, 1.5), ylim=(0, 1.05))
        ax.set_xticks([0, 1], list(colors))
        for tick, color in zip(ax.get_xticklabels(), colors.values()):
            tick.set_color(color)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Benchmark score")
    fig.suptitle("GPT-6 Astra in both architectures | three repeats per case", fontsize=15)
    save(fig, "astra_01_case_scores_mean_sd")

    pairs = pd.read_csv(tables / "repeat_level_paired_comparisons.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for y, case in enumerate(cases):
        vals = pairs[pairs.case == case].paired_delta
        ax.scatter(vals, y + np.linspace(-.1, .1, len(vals)), s=55, color="#167D8D")
        ax.errorbar(vals.mean(), y, xerr=vals.std(ddof=1), fmt="s", color="#222222", capsize=5)
    ax.axvline(0, color="#777777", linestyle="--")
    ax.set_yticks(range(3), cases)
    ax.set_xlabel("FlowPilot score minus one-shot score")
    ax.set_title("Each dot is one case-repeat pair; square is mean +/- SD")
    ax.grid(axis="x", alpha=.2)
    save(fig, "astra_02_paired_differences")

    applicable = criteria[criteria.applicability == "APPLICABLE"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), layout="constrained")
    ids = sorted(criteria.criterion_id.unique())
    for ax, case in zip(axes, cases):
        values = applicable[applicable.case == case].groupby(["architecture", "criterion_id"]).score_0_4.mean().unstack()
        values = values.reindex(index=list(colors), columns=ids) / 4
        im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for y in range(2):
            for x in range(len(ids)):
                v = values.iloc[y, x]
                ax.text(x, y, "N/A" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if pd.notna(v) and v > .6 else "black")
        ax.set_xticks(range(len(ids)), ids, rotation=45, ha="right")
        ax.set_yticks([0, 1], list(colors)); ax.set_title(case)
    fig.colorbar(im, ax=axes, label="Mean normalized criterion score", shrink=.8)
    save(fig, "astra_03_criterion_profiles")

    usage = pd.DataFrame(usage_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, (field, label, divisor) in zip(axes, [
        ("estimated_usd", "Estimated generation cost (USD)", 1),
        ("runtime_s", "Generation wall time (min)", 60),
        ("llm_calls", "Recorded API calls", 1),
    ]):
        for x, architecture in enumerate(colors):
            values = usage[usage.architecture == architecture][field] / divisor
            ax.bar(x, values.mean(), color=colors[architecture], width=.55,
                   yerr=values.std(ddof=1), capsize=4)
        ax.set_xticks([0, 1], list(colors)); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Generation resources | mean +/- SD across nine outcomes per architecture")
    save(fig, "astra_04_generation_resources")

    summary = read(report / "summary.json")
    m = summary["model_architecture_summary"][0]
    case_lines = ["## Case results", "", "Mean +/- sample SD across three repeats; scores range from 0 to 1.", "",
                  "| Chemistry | One-shot | FlowPilot | FlowPilot minus one-shot |",
                  "| --- | --- | --- | --- |"]
    for case in cases:
        row = []
        means = []
        for architecture in colors:
            vals = scores[(scores.case == case) & (scores.architecture == architecture)].mean_score_0_1
            row.append(f"{vals.mean():.3f} +/- {vals.std(ddof=1):.3f}")
            means.append(vals.mean())
        case_lines.append(f"| {case} | {row[0]} | {row[1]} | {means[1]-means[0]:+.3f} |")
    case_lines.extend(["", "All nine one-shot outputs were assessable. FlowPilot published executable contracts for 3/9 outcomes (all CuAAC); 6/9 were blocked (all gas-process cases). One-shot does not use FlowPilot's executable-contract gate, so its zero in that internal field must not be interpreted as nine failed designs. A structurally valid JSON result also does not establish an executable or chemically validated design.", "",
                       "No judge marked a critical-error flag, but judges did assign material-defect scores. An empty critical-flag table does not mean the designs are error-free; see judge_material_concerns.csv and FAILURE_ANALYSIS.md.", ""])
    lines = ["# GPT-6 Astra Architecture Benchmark", "",
             "Same generator in both architectures; fixed three cases and three repeats per case.", "",
             f"- One-shot mean: {m['one_shot_mean_0_1']:.4f}",
             f"- FlowPilot mean: {m['flowpilot_mean_0_1']:.4f}",
             f"- Mean paired difference: {m['mean_paired_delta']:+.4f}",
             f"- Pairwise wins / ties / losses: {m['wins']} / {m['ties']} / {m['losses']}",
             f"- Difference excluding OpenAI-family judge: {m['generator_family_excluded_delta']:+.4f}",
             f"- Generation usage estimate: ${audit['estimated_generation_cost_usd']:.2f} (judge and preflight costs excluded).", "",
             *case_lines,
             "## Scope and interpretation", "",
             "The scores are LLM assessments against the unchanged anchored rubric, accompanied by deterministic verification. "
             "They do not establish experimental yield or laboratory safety. Judge flags are opinions requiring inspection, not independently proven errors. "
             "Current code and the Qwen3.6 / GPT-4o / Sonnet4.6 judge panel were frozen before this campaign. "
             "The three cases were inherited from a selected earlier pilot; they do not represent all chemistry. "
             "Three stochastic repeats estimate within-case variability only. FlowPilot has more computation and retrieved context, so this is not an equal-budget experiment.", "",
             "This is the existing manuscript benchmark configuration: legacy design policy, candidate budget 2, "
             "and lexical held-out-source retrieval. It is not the GUI's scientific_v2 twelve-candidate configuration. "
             "Before judging, one-shot packet construction was corrected to preserve raw delivered values. "
             "A gas-process time was not automatically assessed using the liquid-only volume equation. "
             "See the evaluation amendment and normalization audit. Do not silently pool these scores with older normalized-output assessments.", "",
             "## Files", "",
             "- `figures/astra_01_case_scores_mean_sd.*`: separate chemistry panels with raw repeats and SD.",
             "- `figures/astra_02_paired_differences.*`: case-repeat differences.",
             "- `figures/astra_03_criterion_profiles.*`: all 14 criteria; N/A remains N/A.",
             "- `figures/astra_04_generation_resources.*`: cost, time, and API call counts.",
             "- `tables/criterion_judgments.csv`: every rationale and requested correction.",
             "- `tables/delivered_parameters.csv`: exact parameters from the evaluation packets, before benchmark schema rewriting.",
             "- `tables/flowpilot_final_contract_failures.csv` and `rejected_graph_time_basis_diagnostics.csv`: software blockers and the corresponding saved graph quantities, not one-shot error counts.",
             "- Parent `topologies/`: original generated diagrams, separated into executable candidates and rejected diagnostics. Rejected diagrams are not executable instructions; none are invented by this report.",
             "- `tables/judge_flagged_issues.csv`: critical flags, not a count of distinct verified errors.",
             "- `tables/judge_material_concerns.csv`: individual criterion assessments at 2/4 or below; repeated observations are not deduplicated errors.",
             "- `FAILURE_ANALYSIS.md`: verified defects separated from judge interpretations and follow-up priorities.",
             "- `routing_and_usage_audit.json`: actual generation model routing and usage scope.",
             "- Parent `generation/`, `judgments/`, `packets/`, `frozen/`, and worker logs: complete provenance.", "",
             "## Fresh campaign", "", "Activate `flent`, then run from the project directory:", "",
             "```bash", ".venv-flowpilot/bin/python ablation_test/scripts/run_astra_outcome_benchmark.py --output ablation_results/manuscript_benchmark/astra_new_campaign", "```", "",
             "Use a new output directory for a changed configuration. Resume the same directory only with unchanged sources and inputs."]
    (report / "ASTRA_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"audit": audit["status"], "results": m, "report": str(report)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root.resolve())
