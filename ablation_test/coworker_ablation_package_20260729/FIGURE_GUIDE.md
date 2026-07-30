# Figure Reading Guide

Use this document as a caption and interpretation reference. Each figure is available as PNG for presentations and PDF for manuscripts.

| Figure | Purpose | What it shows | Main takeaway |
|---|---|---|---|
| `01_study_design` | Benchmark structure | Shows the 12 protocols, five matched conditions, saved evidence, frozen scorer, and two output families. | This is a matched output-based evaluation; hidden references and architecture labels do not enter scoring. |
| `02_quality_clustered_ci` | Primary quality result | Mean Quality and Assurance Score v2 with protocol-clustered bootstrap intervals. | Both full pipelines exceed all one-shot conditions; Qwen Full has the highest mean quality. |
| `03_direct_quality_vs_assurance` | Where the advantage comes from | Separates normalized direct design quality from evidence and assurance. | Full FlowPilot leads on both, so its advantage is not only additional logging. |
| `04_quality_dimensions` | Score decomposition | Displays all seven measured dimensions for each condition. | One-shot models receive real engineering, process, and safety credit but lack formal validity and audit evidence. |
| `05_paired_quality_effects` | Matched architecture effects | Shows protocol-level paired differences and 95% bootstrap intervals. | Qwen Full beats each one-shot comparator for all 12 protocols. |
| `06_schema_neutral_sensitivity` | Fairness check | Removes the formal-validity component and schema readiness cap. | The full-pipeline advantage remains, although the one-shot gap becomes smaller. |
| `07_readiness_and_ready_rate` | Quality versus executability | Compares mean capped readiness with the fraction of immediately executable runs. | High assurance does not imply that every design is ready for unsupervised wet-lab execution. |
| `08_hard_check_rates` | Engineering pass rates | Shows schema, geometry, gas, pump, tubing, reactor, and deployment checks. | Full pipelines consistently satisfy machine and inventory checks. |
| `09_deployment_gate_rates` | Reasons for screening | Shows how often each hard deployment gate activates. | One-shot readiness is dominated by schema invalidity; full-pipeline readiness is mainly limited by explicit screening. |
| `10_case_quality_heatmap` | Protocol-level performance | Shows mean quality for each protocol and condition. | The architecture effect is broad rather than driven by one chemistry. |
| `11_quality_repeatability` | Run-to-run score stability | Mean within-protocol quality standard deviation over three runs. | Qwen Full has the lowest quality variability; lower is better. |
| `12_parameter_repeatability` | Parameter stability | Fraction of protocols whose repeated numerical values remain within predefined tolerances. | One-shot Qwen is highly repetitive numerically, but gas-flow stability is weaker for the full systems; gas columns use only gas-required protocols. |
| `13_functional_reproducibility` | Useful reproducibility | Adds engineering feasibility, exact inventory, valid structure, and minimum quality to raw parameter stability. | Both full pipelines retain 58.3% reliable translation; repetitive one-shot values often fail functional requirements. |
| `14_quality_repeatability_tradeoff` | Joint performance | Plots average quality against within-protocol variability. | The desirable region is high and left; Qwen Full occupies the strongest position. |
| `15_execution_cost` | Resource tradeoff | Compares runtime, generative calls, and token volume. | Full FlowPilot is substantially more expensive; local Qwen reduces frontier generation dependence but was slower on the tested server. |
| `16_qwen_gain_by_protocol` | Matched Qwen architecture gain | Subtracts Qwen one-shot quality from Qwen Full for each protocol. | All values are positive, but the magnitude varies by chemistry. |
| `17_score_weights` | Scoring transparency | Displays the fixed weights used by Quality and Assurance Score v2. | Direct design contributes 65%; evidence, review, and calibration contribute 35%. |

## Reading Rules

- Quality, readiness, and repeatability are different outcomes.
- A lower variability value is favorable only when the repeated design is also valid and feasible.
- A deployment-ready rate of zero does not mean that every chemical suggestion is useless; it means every run triggered at least one hard gate.
- The one-shot schema penalty is shown transparently and tested with a schema-neutral sensitivity analysis.
- Three repeats provide preliminary repeatability evidence, not cross-server reproducibility.
