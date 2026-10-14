#!/usr/bin/env python3
"""Build ESI Section 1 from frozen NewGen 2.0 benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "deliverables/manuscript_benchmark_visualizations_20260825"
DEFAULT_OUTPUT = PACKAGE / "ESI_SECTION_1_ABLATION_TEST_AND_ARCHITECTURE_COMPARISON.md"
CASE_MANIFEST = (
    ROOT
    / "deliverables/manuscript_three_model_three_repeat_postfix_20260820"
    / "frozen/case_manifest.json"
)
RUBRIC = ROOT / "ablation_test/benchmarks/newgen_2_0_outcome_rubric.json"
SCORES = (
    ROOT
    / "deliverables/newgen_2_0_latest_results_20260824"
    / "tables/all_candidate_consensus_scores.csv"
)
MODULE_SUMMARY = (
    ROOT
    / "deliverables/newgen_2_0_module_attribution_qwen38_screen3_matched_v1_20260821"
    / "tables/module_summary.csv"
)
REPRESENTATIVE_ONE_SHOT = (
    ROOT
    / "ablation_results/manuscript_benchmark"
    / "manuscript_three_model_three_repeat_postfix_20260820"
    / "generation/cuaac/qwen_one_shot/repeat_01/prompt.json"
)

CASE_ORDER = ["Hydrogenolysis", "Photochemical oxidation", "CuAAC"]
MODEL_ORDER = [
    "Qwen3.6-27B",
    "Qwen3.8-27B",
    "GPT-4o",
    "Claude Sonnet 4.6",
    "Claude Opus 4.6",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def inventory_summary(inventory: dict) -> str:
    parts: list[str] = []
    category_labels = [
        ("pumps", "Pump(s)"),
        ("gas_hardware", "gas hardware"),
        ("mixers", "mixer(s)"),
        ("reactors", "reactor(s)"),
        ("pressure_controllers", "pressure controller(s)"),
        ("light_sources", "light source(s)"),
        ("separators", "separator(s)"),
        ("temperature_controllers", "temperature controller(s)"),
        ("collectors", "collector(s)"),
        ("safety_accessories", "safety accessory/accessories"),
    ]
    for key, label in category_labels:
        items = inventory.get(key) or []
        if items:
            names = ", ".join(
                str(item.get("equipment_id") or item.get("name")) for item in items
            )
            parts.append(f"{label}: {names}")
    if inventory.get("BPR_available"):
        values = ", ".join(str(value) for value in inventory["BPR_available"])
        parts.append(f"BPR setpoint(s): {values} bar")
    return "; ".join(parts)


def architecture_wrapper() -> tuple[str, str]:
    prompt = read_json(REPRESENTATIVE_ONE_SHOT)
    system = str(prompt["system"])
    user = str(prompt["user"])
    marker = "\n\nPropose one complete flow design"
    if marker not in user:
        raise RuntimeError("Could not isolate the frozen one-shot output contract")
    suffix = "Propose one complete flow design" + user.split(marker, 1)[1]
    return system, suffix


def result_tables() -> tuple[str, str]:
    scores = pd.read_csv(SCORES)
    scores = scores[scores["model"].isin(MODEL_ORDER)].copy()
    repeat = (
        scores.groupby(["model", "architecture", "repeat_id"], as_index=False)[
            "mean_score_0_1"
        ]
        .mean()
        .rename(columns={"mean_score_0_1": "repeat_mean"})
    )
    summary = (
        repeat.groupby(["model", "architecture"])["repeat_mean"]
        .agg(["mean", "std"])
        .reset_index()
    )
    wide = summary.pivot(index="model", columns="architecture", values=["mean", "std"])
    rows: list[list[str]] = []
    for model in MODEL_ORDER:
        one_mean = float(wide.loc[model, ("mean", "One-shot")])
        one_sd = float(wide.loc[model, ("std", "One-shot")])
        flow_mean = float(wide.loc[model, ("mean", "FlowPilot")])
        flow_sd = float(wide.loc[model, ("std", "FlowPilot")])
        rows.append(
            [
                model,
                f"{one_mean:.3f} +/- {one_sd:.3f}",
                f"{flow_mean:.3f} +/- {flow_sd:.3f}",
                f"{flow_mean - one_mean:+.3f}",
            ]
        )
    model_table = md_table(
        ["Generator model", "One-shot", "FlowPilot", "Mean difference"], rows
    )

    architecture = scores.groupby("architecture").agg(
        mean_score=("mean_score_0_1", "mean"),
        outcome_sd=("mean_score_0_1", "std"),
        raw_critical_flags=("total_judge_critical_flags", "sum"),
        outcomes=("candidate_id", "count"),
    )
    no_error = (
        scores.assign(no_error=scores["total_judge_critical_flags"].eq(0))
        .groupby("architecture")["no_error"]
        .sum()
    )
    architecture_rows = []
    for name in ("One-shot", "FlowPilot"):
        row = architecture.loc[name]
        architecture_rows.append(
            [
                name,
                f"{float(row['mean_score']):.3f}",
                f"{int(no_error.loc[name])}/{int(row['outcomes'])}",
                str(int(row["raw_critical_flags"])),
            ]
        )
    architecture_table = md_table(
        [
            "Architecture",
            "Mean candidate score",
            "Campaigns without critical flags",
            "Raw judge critical flags",
        ],
        architecture_rows,
    )
    return model_table, architecture_table


def module_table() -> str:
    frame = pd.read_csv(MODULE_SUMMARY)
    rows = []
    for row in frame.itertuples():
        rows.append(
            [
                str(row.condition),
                f"{float(row.mean_score_0_1):.3f}",
                f"{float(row.score_sd):.3f}",
                f"{100 * float(row.executable_rate):.0f}%",
                f"{float(row.mean_llm_calls):.1f}",
                str(int(row.critical_flags)),
            ]
        )
    return md_table(
        ["Condition", "Mean score", "Between-case SD", "Executable", "LLM calls", "Critical flags"],
        rows,
    )


def case_section(case: dict, index: int) -> str:
    constraints = "\n".join(f"- {item}" for item in case["hard_constraints"])
    expected = case.get("expected_features") or {}
    topology = " -> ".join(expected.get("topology") or [])
    hazard_values = expected.get("hazards") or {
        "cuaac_adsc_200900726": [
            "energetic azide",
            "flammable acetone",
            "high temperature",
            "pressurized operation",
        ]
    }.get(case["case_id"], ["case-specific chemical and process hazards"])
    hazards = ", ".join(str(value).replace("_", " ") for value in hazard_values)
    source = str(case.get("source_record_id") or "Not recorded")
    exact_prompt = str(case["design_input_text"])
    return f"""### 1.{index + 4} Case study {index}: {case['case_label']}

**Benchmark title:** {case['title']}  
**Case identifier:** `{case['case_id']}`  
**Reaction category:** `{case['category']}`  
**Held-out source record:** `{source}`  
**Design-input SHA-256:** `{case['design_input_sha256']}`

#### Batch protocol and design task

{case['protocol']}

**Objective.** {case['objective']}

**Frozen constraints.**

{constraints}

**Inventory summary.** {inventory_summary(case['inventory'])}.

**Expected process features used for deterministic verification.** The expected phase regime was `{expected.get('phase_regime', 'not specified')}`. The required process sequence was `{topology}`. Principal hazards were {hazards}.

<details>
<summary><strong>Exact frozen design input supplied at the architecture boundary</strong></summary>

```text
{exact_prompt}
```

</details>
"""


def build() -> str:
    manifest = read_json(CASE_MANIFEST)
    cases_by_label = {case["case_label"]: case for case in manifest["cases"]}
    cases = [cases_by_label[label] for label in CASE_ORDER]
    rubric = read_json(RUBRIC)
    system_prompt, response_contract = architecture_wrapper()
    model_results, architecture_results = result_tables()

    criterion_rows = []
    for criterion in rubric["criteria"]:
        criterion_rows.append(
            [
                criterion["criterion_id"],
                str(criterion["domain"]).title(),
                criterion["name"],
                criterion["applicability"].replace("_", " ").title(),
            ]
        )
    criteria_table = md_table(
        ["ID", "Domain", "Criterion", "Applicability"], criterion_rows
    )
    anchor_rows = [
        [score, description]
        for score, description in sorted(
            rubric["score_anchors"].items(), key=lambda item: int(item[0])
        )
    ]
    anchors_table = md_table(["Score", "Anchor"], anchor_rows)

    case_text = "\n".join(case_section(case, index + 1) for index, case in enumerate(cases))
    return f"""# 1 Ablation test and architecture comparison

## 1.1 Purpose and scope

The NewGen 2.0 benchmark was designed to compare the quality of final batch-to-flow designs produced by a direct one-shot large language model and by the complete FlowPilot architecture under matched inputs. The comparison was architecture-neutral at the evaluation stage: both methods received the same frozen chemistry, objective, laboratory inventory, and operating constraints, and both were required to return the same normalized final-design contract. The assessed object was the delivered design, not the internal reasoning trace or the number of intermediate agent calls.

Two related experiments are reported in this section. The **architecture comparison** tested one-shot generation against full FlowPilot using five retained generator models, three chemistry cases, and three independent generation repeats. The **internal module-ablation screen** fixed the generator to Qwen3.8-27B and selectively removed or altered FlowPilot components to examine their contribution to outcome quality and executability.

The benchmark measures outcome quality and repeatability for the frozen cases and inventory. It does not establish universal superiority across all reaction classes and does not replace prospective wet-lab validation.

## 1.2 Paired architecture-comparison design

The retained manuscript comparison included the following generator models:

- Qwen3.6-27B
- Qwen3.8-27B
- GPT-4o
- Claude Sonnet 4.6
- Claude Opus 4.6

For every model, each of the three frozen chemistry cases was generated using both architectures in three independent calls. This produced:

`5 models x 2 architectures x 3 cases x 3 repeats = 90 candidate outcomes`

The paired unit was the same generator model, chemistry case, and repeat under the two architectures. The temperature was 0.2, calls were independently recorded, and the stopping rule was fixed before evaluation. The held-out source record for each case was excluded from FlowPilot retrieval. This exclusion prevents direct RAG leakage but does not prove that the underlying publication was absent from a model's pretraining data.

### Compared architectures

**One-shot.** A single model call received the complete frozen design input and a fixed output contract. No FlowPilot specialist agents, council deliberation, deterministic candidate selection, or post-generation correction operated inside this architecture.

**Full FlowPilot.** The same frozen design input entered the complete FlowPilot workflow, including chemistry interpretation, engineering calculations, specialist-agent assessment, council selection and revision, safety analysis, deterministic closure checks, and final schema enforcement. The internal workflow could use several LLM calls, but only the final normalized design was scored.

### Common one-shot wrapper

The one-shot system instruction was:

```text
{system_prompt}
```

For each chemistry, the corresponding frozen design input reported in Sections 1.5-1.7 was inserted after the header `FROZEN DESIGN INPUT:`. The following common response contract was then appended:

<details>
<summary><strong>Exact one-shot response contract</strong></summary>

```text
{response_contract}
```

</details>

FlowPilot did not use one monolithic system prompt. It accepted the same frozen design input at its architecture boundary and distributed the information across its intake, chemistry, engineering, council, and validation stages. All internal prompts and stage events were retained in the archived run directories.

## 1.3 Identity-masked evaluation

Each final outcome was converted to a standardized candidate packet. Generator identity and architecture labels were removed before judging. Complete architectural blinding is not claimed because formatting or output style could reveal provenance. Every judge received the same protocol, objective, inventory, normalized final outcome, held-out reference facts, deterministic verification sheet, and fixed rubric.

Three independent judge families evaluated every candidate:

- Qwen
- OpenAI
- Claude

The deterministic sheet exposed machine-checkable evidence for schema validity, numerical closure, inventory compliance, reactor and residence-time consistency, gas bookkeeping where applicable, topology coverage, and safety requirements. Deterministic evidence informed the judges but did not itself add an architecture-specific score bonus.

## 1.4 NewGen 2.0 scoring method

The same 14 criteria were applied to every candidate. Each applicable criterion received an integer rating from 0 to 4. All criteria and all judges had equal weight; no architecture bonus or hand-tuned criterion weighting was used.

{anchors_table}

{criteria_table}

`NOT_APPLICABLE` was permitted only when the criterion's applicability rule was false. Missing required information was rated 0 or 1 rather than excluded. In the frozen rubric, gas bookkeeping (`UO-08`) and multistage closure (`UO-09`) were the only criteria that could routinely become non-applicable. Critical-error flags were recorded separately and did not impose an arbitrary cap on the numerical score.

For candidate *i*, criterion *k*, and judge *j*, the criterion consensus was:

`C(i,k) = mean_j[s(i,k,j)]`

where `s(i,k,j)` is the judge's integer rating from 0 to 4. If `A(i)` is the set of applicable criteria, the candidate score was:

`S(i) = mean_k in A(i)[C(i,k)] / 4`

The resulting candidate score ranged from 0 to 1. For each model, architecture, and repeat, the three chemistry candidate scores were first averaged to obtain a repeat-level campaign score:

`R(model, architecture, repeat) = mean_case[S(i)]`

The reported model-architecture value was the mean of the three repeat-level scores, and the error bar was their sample standard deviation (`n = 3`, `ddof = 1`). It was not a confidence interval. The matched architecture effect was calculated before aggregation:

`Delta(i) = S(FlowPilot, model, case, repeat) - S(one-shot, model, case, repeat)`

A positive value therefore favored FlowPilot for that matched case-repeat pair. Judge disagreement was retained separately and was not used as a substitute for generation-repeat variability.

{case_text}

## 1.8 Internal FlowPilot module-ablation screen

The internal screen used Qwen3.8-27B as the fixed generator and the same three chemistry cases. Fifteen architecture conditions were tested, giving 45 condition-case outcomes. Each condition-case cell was generated once; therefore, the reported standard deviations describe variation between the three chemistry cases and must not be interpreted as generation-repeat uncertainty.

Disabled specialist agents received no LLM call and no default-score penalty. Active council weights were renormalized. Deterministic inventory and safety gates remained active for executable FlowPilot conditions. The full condition table is provided below for transparency; the manuscript figure may display a selected subset for readability.

{module_table()}

The screen supports descriptive module attribution. It should not be interpreted as a statistically powered ranking of every internal configuration. Notably, removal of an individual specialist did not necessarily reduce the aggregate score for this small case set, whereas eliminating the complete council caused a substantial loss of executable design quality.

## 1.9 Architecture-comparison results

{model_results}

Values are mean +/- sample SD across three repeat-level campaign means. The difference column is FlowPilot minus one-shot.

Across the 45 retained outcomes per architecture:

{architecture_results}

Raw judge critical flags count separate judge reports and are not the same as the number of unique error types. The campaign-level error register consolidates judges that identified the same criterion-level error.

## 1.10 Figures and source data

### Figure 4. Architecture and module-ablation overview

![Figure 4](figures_revised/main/figure4.png)

**Caption.** Architecture-level outcome quality, critical-error burden, selected Qwen3.8 internal conditions, and FlowPilot quality per generation cost. The source tables for each panel are provided in `figures_revised/main/raw/`.

### Figure S5-1. Case-specific architecture comparison

![Figure S5-1](figures_revised/main/figs5-1.png)

**Caption.** Ranked one-shot and FlowPilot benchmark scores for CuAAC, photochemical oxidation, and hydrogenolysis. Points are means and error bars are sample standard deviations across three generation repeats.

### Figure S5-2. Criterion-level architecture effect

![Figure S5-2](figures_revised/main/figs5-2.png)

**Caption.** Mean paired FlowPilot-minus-one-shot score difference for each applicable universal criterion, separated by chemistry. Positive values favor FlowPilot. `UO-08` and `UO-09` are excluded from this cross-chemistry heatmap because their applicability is not uniform.

### Figure S5-3. Campaign-level critical-error map

![Figure S5-3](figures_revised/main/figs5-3.png)

**Caption.** Critical-error criteria identified by the independent judge panel for each model, chemistry, architecture, and repeat. Cell numbers report independent judge agreement for the same campaign-criterion error. The total column counts distinct affected criteria rather than duplicate judge flags.

### Figure S5-4. FlowPilot resource use and cost efficiency

![Figure S5-4](figures_revised/main/figs5-4.png)

**Caption.** FlowPilot-only token use, measured generation cost, observed runtime, and chemistry-specific quality-per-cost index. Resource panels report mean +/- sample SD across nine campaigns per model. Runtime is environment-dependent, and judge-evaluation costs are excluded.

The plotted source tables are stored under `figures_revised/main/raw/`. Candidate-level consensus scores, criterion judgments, paired case-repeat differences, and repeat-level campaign means are retained under `deliverables/newgen_2_0_latest_results_20260824/tables/`.

## 1.11 Interpretation limits

This benchmark uses only three reaction classes and three generation repeats. The source publications were excluded from FlowPilot retrieval, but the cases are not guaranteed to be absent from model pretraining. LLM judges may share model-family biases and cannot substitute for evaluation by flow chemists or wet-lab execution. The generator-family-excluded sensitivity records and deterministic evidence should therefore accompany the consensus scores. Claims should be restricted to the frozen benchmark conditions and should not be phrased as universal superiority across all batch-to-flow translation problems.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build(), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
