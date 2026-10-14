"""Describe the frozen comparison without modifying any grade or candidate."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'ablation_results/practical_input/claude_fair_bridge_20261014'

def read(path):
    return json.loads(path.read_text())

def number(value):
    return 'incomplete' if value is None else f'{value:.6f}'

def main():
    history = read(HERE / 'historical/historical_audit.json')
    strict = read(CAMPAIGN / 'report/summary_all.json')
    current = read(HERE / 'format_recovery/summary_all.json')
    if sum(r['stage']=='A' for r in current['results']) != 45 or not (CAMPAIGN/'worker_completion.json').exists():
        raise RuntimeError('Complete saved-output rescoring and settle the original finite run first')
    groups = current['groups']
    lines = [
        'FLOWPILOT BENCHMARK FAIRNESS AUDIT AND FIXED FOLLOW-UP',
        '14 October 2026', '',
        'The recent practical-input scores were not directly comparable with the manuscript benchmark. '
        'The generator cohort, judge panel, packet representation, numerical worksheets and software version changed together. '
        'This investigation restores the original criteria and judge panel and separates saved-output rescoring from fresh generation. '
        'The first fresh run exposed an output-contract routing error introduced by the matched-format harness; it is retained as a diagnostic, not a valid architecture-effect estimate.', '',
        'Historical manuscript results, reproduced from the archived accepted grades',
        'Generator                         One-shot   FlowPilot   Designs per arm',
    ]
    model_order = history['exact_manuscript_models']
    for model in model_order:
        values = {r['architecture']: r for r in history['model_architecture'] if r['model'] == model}
        lines.append(f"{model:33} {values['One-shot']['mean']:.6f}   {values['FlowPilot']['mean']:.6f}   9")
    lines += ['',
        'The pooled manuscript means were 0.793697 and 0.916916, from 45 designs per architecture. '
        'Historical Sonnet one-shot alone scored 0.890076. Comparing a new Sonnet-only result with the five-model mean confounds generator choice with architecture.', '',
        'Historical Sonnet means separated by judge',
        'Judge                             One-shot   FlowPilot',
    ]
    for judge, label in [('openai','GPT-5.4'),('claude','Claude Sonnet 4.6'),('qwen','Qwen3.6-27B')]:
        values = {r['architecture']: r['mean'] for r in history['historical_sonnet_per_judge'] if r['judge'] == judge}
        lines.append(f"{label:33} {values['One-shot']:.6f}   {values['FlowPilot']:.6f}")
    lines += ['',
        'The recent practical study used GPT-4o as primary judge and Qwen as secondary, with no Claude evaluator. '
        'The manuscript used GPT-5.4, Claude and Qwen with equal weighting. Qwen already assigned almost perfect historical Sonnet one-shot scores. '
        'The historical requested model IDs are recorded; the old client did not independently archive returned IDs or endpoint routing. The new campaign does.', '',
        'Historical Sonnet one-shot inputs were 2,132, 2,095 and 1,488 tokens for hydrogenolysis, oxidation and CuAAC. '
        'The corresponding practical inputs were 1,293, 1,311 and 861. This is about 1.6–1.7 times the input, not an extraordinary or humanly impossible brief. '
        'Internal multi-agent tokens are a separate quantity. None of these cross-campaign comparisons isolates a causal effect of prompt length.', '',
        'The new structured one-shot inputs contain 2,073, 2,091 and 1,641 tokens, respectively. '
        'The common JSON contract adds 780 tokens to each preceding practical prose input, making the new total lengths approximately equal to the historical engineered inputs. '
        'The new experiment changes the expert chemical guidance available to the generator, not simply the length of its token input. '
        'The corrected follow-up reuses these exact one-shot outputs and grades.', '',
        'Why the recent grades were unreliable',
        'For the same saved practical outputs, the recent GPT-4o judge assigned means of 0.987179 to one-shot and 0.935897 to FlowPilot. '
        'The restored GPT-5.4 evaluator assigned 0.643162 and 0.785791 to those unchanged packets. '
        'This reversal is attributable to the evaluation configuration rather than a change in the designs or their generation prompt length. '
        'Both judge model and historical judge instructions changed, so it is not a model-only causal estimate.',
        'The manual audit found missed one-shot gas-unit and material-balance errors, unsupported deductions against FlowPilot gas calculations and confidence labels, '
        'and a real FlowPilot duplicate-charge defect that judges also missed. In one accepted Qwen assessment, the rationale said Score 2 while the stored integer was 4. '
        'The earlier blocked FlowPilot designs also contained genuine software input-boundary/identity problems. Those repairs and the later feed-alias repair were frozen before fresh generation.', '',
        'What was preserved and what was corrected',
        'The same three chemistry cases, original fourteen criteria, 0–4 anchors, applicability rules and equal criterion/judge weights were retained. '
        'The historical judge prompt and validation were restored. No applicable criterion score, critical flag or rationale was manually changed. '
        'The fresh experiment uses the same practical protocol, inventory and factual constraints in both arms, with a shared historical structured-output contract. '
        'The normal one-shot wrapper repeats its schema; this is documented and is not a pure token-length experiment. '
        'One-shot uses one call; FlowPilot keeps two initial candidates and its native review. Computation is not matched.', '',
        'Fifteen saved packets were copied byte-for-byte: six historical Sonnet repeat-01 outcomes and all nine fresh outcomes from the preceding practical campaign. '
        'They were rescored as a diagnostic under the restored historical evaluator package. Their original scores and output archives are unchanged. '
        'The copied historical packets retain their known phase-insensitive worksheet limitation and must not be pooled with the new primary comparison.', '',
        'Six fresh designs use an adapter that preserves raw one-shot values and the final delivered FlowPilot artifact. '
        'It does not substitute Pydantic-normalized one-shot numbers. Phase-aware time checks use the declared basis; gas reference calculations are explicitly conditional, '
        'and total carrier-gas equivalents are not automatically treated as active-reagent equivalents. Internal council traces do not earn credit. '
        'The original rubric and acceptance rules remain unchanged; a stricter current validator runs as a separate audit only.', '',
        'Fixed follow-up results',
        'All means below use the three historical judges with equal weights. Each arm contains the same three cases.',
        f"The strict archive contains {current['strict_valid_judgments']} valid judgments. "
        f"A separately declared formatting supplement resolves {len(current['recovered_cells'])} otherwise complete responses "
        'whose excluded NOT_APPLICABLE criteria incorrectly contained integer scores rather than null. '
        'It selects the earliest qualifying response, preserves every applicable score and flag, and leaves the original status and raw response untouched. '
        'The strict report and exact field-level changes remain available separately. '
        f"Fresh-comparison recovered cells: {sum(r['stage']=='B' for r in current['recovered_cells'])}.",
        f"Unresolved diagnostic grades after the fixed attempt budget: {len(current['missing'])}. "
        'Missing grades are not imputed or replaced; a pooled arm mean requiring one is reported as incomplete.',
    ]
    labels = {'historical_manuscript':'Saved historical Sonnet repeat 01, rescored',
              'current_practical':'Saved recent practical outputs, rescored',
              'fresh_matched_format':'Initial fresh run — output-contract routing confounded (diagnostic only)'}
    for cohort in ('historical_manuscript','current_practical','fresh_matched_format'):
        lines += ['', labels[cohort], 'Architecture                       Mean       Judge-level critical flags']
        for arm, result in sorted(groups[cohort].items()):
            lines.append(f"{arm:34} {number(result['mean'])}   {result['critical_flags']} (recorded)")
    fresh = groups['fresh_matched_format']
    delta = fresh['FlowPilot']['mean'] - fresh['One-shot']['mean'] if all(fresh[a]['mean'] is not None for a in ('FlowPilot','One-shot')) else None
    lines += ['', f'Confounded fresh-run FlowPilot minus one-shot = {number(delta)}; not interpretable as an architecture effect.', '', 'Initial fresh diagnostic separated by judge',
              'Judge                             One-shot   FlowPilot']
    for judge,label in [('openai','GPT-5.4'),('claude','Claude Sonnet 4.6'),('qwen','Qwen3.6-27B')]:
        lines.append(f"{label:33} {number(fresh['One-shot']['by_judge'][judge]['mean'])}   {number(fresh['FlowPilot']['by_judge'][judge]['mean'])}")
    lines += ['', 'Initial fresh diagnostic by chemistry', 'Case                              One-shot   FlowPilot']
    case_rows = []
    for case in ('Hydrogenolysis','Photochemical oxidation','CuAAC'):
        cells={arm:[r['score'] for r in current['results'] if r['cohort']=='fresh_matched_format' and r['case']==case and r['architecture']==arm] for arm in ('One-shot','FlowPilot')}
        values={arm:statistics.fmean(scores) if len(scores)==3 else None for arm,scores in cells.items()}
        case_rows.append({'case':case,**values})
        lines.append(f"{case:33} {number(values['One-shot'])}   {number(values['FlowPilot'])}")
    lines += ['',
        f"Excluding the generator-family Claude judge, diagnostic fresh means are one-shot {number(fresh['One-shot']['generator_family_excluded_mean'])} and FlowPilot {number(fresh['FlowPilot']['generator_family_excluded_mean'])}. This sensitivity was specified before grading.", '',
        'Why the first fresh run cannot answer the architecture question',
        'The common final-JSON output instruction was appended to the design objective, which propagated it into intermediate specialist prompts. '
        'All three translation calls returned a nested proposal envelope. The parser expected a flat proposal and silently accepted defaults from the outer envelope, losing supplied fields before review. '
        'All three preceding practical-repair translation responses had the expected flat structure. This is a harness-induced confound exposing a pre-existing parser robustness defect. '
        'It is not evidence that the same chemistry information became insufficient or that the one-shot architecture is intrinsically better. '
        'A separate solvent-loss inventory-mapping defect also blocked valid pump assignment; both original blocked deliveries remain preserved. '
        'An independently reviewed follow-up reuses all three fixed one-shot outputs and reruns all three FlowPilot tasks after routing only final serialization requirements out of internal specialist objectives, retaining scientific requirements and frozen core code. '
        'It changes neither rubric nor judging instructions and must be reported separately from this diagnostic.', '',
        'Interpretation limits',
        'This is a fixed development comparison with one generation per case and arm, not a replacement for the manuscript three-repeat benchmark or an independent holdout. '
        'The cases have been inspected during software development. Scores are model judgments and critical flags are judge findings, not independently confirmed failures. '
        'The result must be read with the per-judge breakdown and the independent content audit. '
        'It does not establish that shorter prompts improve quality, that any architecture always wins, or that laboratory performance follows directly from assessed design quality.', '',
        f"Returned-usage API cost: ${current['usage']['estimated_paid_usd']:.6f}; local inference hardware costs are not estimated. The frozen shared spending guard was $10.",
        'Rate sources: https://developers.openai.com/api/docs/models/gpt-5.4 and https://platform.claude.com/docs/en/models/sonnet-4-6/overview. '
        'Usage estimates cannot account for unreported billable work after a transport failure.', '',
        'Evidence',
        str(CAMPAIGN / 'frozen/PREREGISTRATION.txt'),
        str(CAMPAIGN / 'report/summary_all.json'),
        str(HERE / 'format_recovery/summary_all.json'),
        str(HERE / 'historical/historical_audit.txt'),
        str(HERE / 'packet_audit/analysis_report.txt'),
        str(HERE / 'independent_verification.json'),
        str(HERE / 'fresh_content_audit.txt'),
    ]
    followup_path = HERE/'schema_routing_followup/campaign/report/summary_B.json'
    followup = read(followup_path) if followup_path.exists() else None
    secondary_path=HERE/'schema_routing_followup/common_complete_judge_sensitivity.json'
    secondary=read(secondary_path) if secondary_path.exists() else None
    if followup and secondary:
        arms=secondary['groups']
        corrected=['Corrected routing comparison — all three cases, fixed one-shot baseline',
            'Only final serialization instructions were moved out of intermediate scientific objectives. '
            'All task facts, semantic guidance, public judge context, core source, candidate budgets and scoring rules were preserved. '
            'All three FlowPilot cases were regenerated once; all three one-shot outputs and their nine original accepted grades were reused byte-for-byte. '
            'These are development results following a diagnosed setup error, not an independent confirmatory replication.',
            f"The original three-judge plan has {followup['accepted_judgments']}/18 strictly accepted judgments. Its missing judgments are retained and its full FlowPilot mean remains unavailable. "
            'The Qwen photochemical assessment repeatedly exhausted its 7,500-token output allowance. The unchanged evidence occupied 22,980 tokens within a served 32,768-token context. '
            'No input was truncated, no incomplete judgment was scored, and no token limit or judge instruction was changed to complete that cell.',
            'The table below is a separately labeled, post-hoc sensitivity using only judges with accepted grades for every one of the six paired outcomes. '
            'GPT-5.4 and Claude qualify; Qwen is excluded from both arms symmetrically. Eligibility depends on completeness, not score direction. '
            'No extra model calls or score edits were made for this arithmetic summary.',
            'Architecture                       Mean       Judge-level critical flags']
        for arm in ('One-shot','FlowPilot'):
            corrected.append(f"{arm:34} {arms[arm]['mean']:.6f}   {arms[arm]['critical_flags']}")
        corrected += ['', 'Case                              One-shot   FlowPilot']
        for case in ('Hydrogenolysis','Photochemical oxidation','CuAAC'):
            values={arm:arms[arm]['by_case'][case] for arm in ('One-shot','FlowPilot')}
            corrected.append(f"{case:33} {values['One-shot']:.6f}   {values['FlowPilot']:.6f}")
        corrected += ['', 'Judge                             One-shot   FlowPilot']
        for judge in secondary['included_judges']:
            corrected.append(f"{judge:33} {arms['One-shot']['by_judge'][judge]:.6f}   {arms['FlowPilot']['by_judge'][judge]:.6f}")
        corrected += ['',f"Two-judge mean difference = {secondary['difference_FP_minus_OS']:+.6f}. The judges disagree in direction, and this small difference is not convincing evidence of superiority.",
            'All twelve grades used in this sensitivity passed the original validator. No formatting recovery was needed for them.',
            'Source corrections prepared during diagnosis were not applied before this run; the intervention was limited to benchmark prompt routing.',
            'Evidence: '+str(followup_path),'Sensitivity evidence: '+str(secondary_path),'']
        corrected += ['Checking the hydrogenolysis findings against the delivered evidence',
            'The delivered 1.569036 min inlet/STP index, 5.178932 min pressure-corrected channel index and 6 min liquid-only index each close with their explicitly stated denominators. '
            'Treating their difference as a geometric contradiction is unsupported. However, FlowPilot retained erroneous earlier gas/time rationale, explicitly marked as requiring review after parameter changes. '
            'That remains a real presentation defect. Its illustrative 10 mL stock basis also needs scaling to cover the minimum 12 mL commissioning-plus-collection schedule. '
            'Conversely, all three judges missed the one-shot instruction to pressure-test the assembled system at 25 bar despite a supplied BPR limit of 21 bar and its own instruction never to exceed that value. '
            'This audit qualifies the reliability of the judgments; it does not replace their integers, erase critical flags or recalculate an adjusted winning score.',
            'Evidence: '+str(HERE/'audit_judge_findings_adjudication.txt'),'']
        index=lines.index('Interpretation limits')
        lines[index:index]=corrected
        lines=[f"Total returned-usage API cost for this investigation and follow-up: ${followup['usage']['estimated_paid_usd']:.6f}; local inference hardware costs are not estimated. The shared spending guard was $10."
               if line.startswith('Returned-usage API cost:') else line for line in lines]
    repair_path=HERE/'post_benchmark_repairs/manifest.json'
    if repair_path.exists():
        repairs=read(repair_path)
        if repairs.get('after'):
            lines += ['', 'Repairs applied after benchmark verification',
                'The reusable benchmark driver now scopes the final JSON contract away from intermediate agents while retaining scientific guidance. '
                'The native translation parser rejects unexpected nested or unrecognized responses before assigning default scientific values, using its existing bounded retry/repair path. '
                'Topology reconstruction preserves explicitly declared same-feed solvent components, and pump compatibility checks every declared medium component without borrowing another feed’s solvent. '
                'The targeted regression suite passed 214 tests. The isolated all-three pump replay and incompatible-pump negative control are archived separately.',
                'These post-benchmark production edits were not used to generate or grade the scored outcomes. The frozen source copies and before/after hashes identify both versions. '
                'The stale-rationale and stock/schedule issues identified above remain development targets, and have not been relabeled as fixed.',
                'Repair manifest: '+str(repair_path)]
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    (HERE/'comparison_summary.json').write_text(json.dumps({'initial_confounded_diagnostic':fresh,'diagnostic_difference':delta,'diagnostic_case_results':case_rows,
        'rescored_groups':groups,'schema_routing_followup':followup,'common_complete_judge_sensitivity':secondary,'usage':followup['usage'] if followup else current['usage']},indent=2)+'\n')
    print(HERE/'REPORT.txt')

if __name__ == '__main__':
    main()
