# NewGen 2.0 Qwen3.8 Module Attribution

## Frozen design

- Fixed generator: Qwen3.8-27B.
- Universal 14-criterion NewGen 2.0 rubric.
- Same protocol, objective, strict case inventory, source exclusion, temperature, and seed schedule across matched conditions.
- Three independent judge families: Qwen, OpenAI, and Claude.
- Disabled specialists receive no call and no default-score penalty; active council weights are renormalized.
- Deterministic inventory and safety gates remain active in executable conditions.

## Results

| Condition | Consensus | Deterministic | Executable | LLM calls | Critical flags |
|---|---:|---:|---:|---:|---:|
| Without preselection refinement | 0.935 | 0.988 | 100.0% | 11.0 | 0 |
| One-pass council | 0.932 | 0.988 | 100.0% | 9.7 | 0 |
| Full FlowPilot | 0.928 | 0.988 | 100.0% | 13.3 | 0 |
| Without winner revision | 0.926 | 0.988 | 100.0% | 13.0 | 0 |
| Without DFMEA | 0.926 | 0.988 | 100.0% | 12.7 | 0 |
| Without Kinetics agent | 0.924 | 0.988 | 100.0% | 12.0 | 0 |
| Deterministic selection | 0.920 | 0.988 | 100.0% | 12.7 | 0 |
| Candidate budget 4 | 0.915 | 0.988 | 100.0% | 18.7 | 0 |
| Without Safety agent | 0.915 | 0.988 | 100.0% | 12.3 | 0 |
| Without Fluidics agent | 0.913 | 0.988 | 100.0% | 10.7 | 0 |
| Candidate budget 1 | 0.907 | 0.988 | 100.0% | 12.3 | 0 |
| Without Skeptic audit | 0.904 | 0.988 | 100.0% | 13.3 | 0 |
| Without Chemistry agent | 0.902 | 0.988 | 100.0% | 12.0 | 0 |
| One-shot | 0.743 | 0.946 | 0.0% | 1.0 | 14 |
| No council | 0.670 | 0.993 | 0.0% | 3.0 | 19 |

The execution-contract audit passed 45/45 saved runs. This three-case screen reports descriptive paired effects and does not make inferential or universal-superiority claims. It does not claim complete architecture blinding because output structure can reveal provenance. Raw prompts, responses, retries, stage snapshots, and model metadata remain in `/home/amirreza/Documents/codes/Flow Agent/ablation_results/newgen_2_0_module_attribution/qwen38_screen3_matched_v1_20260821`.
