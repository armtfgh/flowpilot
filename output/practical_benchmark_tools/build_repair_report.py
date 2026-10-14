#!/usr/bin/env python3
"""Render a completed repair benchmark without changing frozen evidence.

Usage: python output/practical_benchmark_tools/build_repair_report.py [campaign]
Requires matplotlib/numpy; does not import production code or call model APIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'ablation_results/practical_input/claude_practical_repair_20261014'
CASES = ('Hydrogenolysis', 'Photochemical oxidation', 'CuAAC')
ARMS = ('Practical one-shot', 'Guided one-shot', 'FlowPilot')
JUDGES = ('openai', 'qwen')
JUDGE_NAMES = {'openai': 'GPT-4o', 'qwen': 'Qwen3.6-27B'}
COLORS = ('#8391A2', '#B78338', '#087E83')
AUDIT_PATHS = (
    ROOT / 'output/practical_benchmark_tools/fresh_content_audit_20261014.txt',
    ROOT / 'output/practical_benchmark_tools/fresh_flowpilot_audit.txt',
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(root, summary, accepted):
    """Check immutable packets/results plus the accepted-rating provenance."""
    assert summary['generation_outcomes'] == 9
    assert summary['archived_outcomes_reevaluated'] == 6
    assert summary['accepted_judgments'] == 30
    assert len(summary['results']) == len(accepted) == 30
    assert summary['evaluation_version'] == 'practical_content_verification_v2'
    for name in ('frozen/input_manifest.json', 'prepared_manifest.json'):
        for relative, digest in read(root / name).items():
            assert sha(root / relative) == digest, f'Immutable evidence changed: {relative}'
    seen = set()
    for item in accepted:
        pair = item['candidate_id'], item['judge']
        assert pair not in seen, f'Duplicate accepted assessment: {pair}'
        seen.add(pair)
        status = root / item['status_path']
        assert read(status)['status'] == 'valid'
        assert sha(status.with_name('parsed_response.json')) == item['response_sha256']
        response = read(status.with_name('parsed_response.json'))
        scores = [r['score'] for r in response['criterion_scores'] if r['applicability'] == 'APPLICABLE']
        row = next(r for r in summary['results'] if (r['candidate_id'], r['judge']) == pair)
        assert abs(statistics.fmean(scores) / 4 - row['score']) < 1e-12
    for judge in JUDGES:
        for arm in ARMS:
            rows = [r for r in summary['results'] if r['judge'] == judge and r['architecture'] == arm and r['cohort'] == 'fresh']
            assert len(rows) == 3 and {r['case'] for r in rows} == set(CASES)
            assert abs(statistics.fmean(r['score'] for r in rows) - summary['groups'][judge]['fresh'][arm]['mean']) < 1e-12


def case_score(summary, judge, cohort, arm, case):
    return next(row['score'] for row in summary['results'] if row['judge'] == judge and row['cohort'] == cohort and row['architecture'] == arm and row['case'] == case)


def mean(summary, judge, cohort, arm):
    return summary['groups'][judge][cohort][arm]['mean']


def collect_delivery(root, key):
    deliveries = []
    resources = []
    for row in key:
        run = Path(row['run_directory'])
        result = read(run / 'result.json') if (run / 'result.json').exists() else {}
        recorded = read(run / 'run_summary.json') if (run / 'run_summary.json').exists() else {}
        final = result.get('final_design') or {}
        deliveries.append({
            'candidate_id': row['candidate_id'], 'cohort': row['cohort'], 'case': row['case'],
            'architecture': row['architecture'], 'generation_status': row['generation_status'],
            'delivery_status': final.get('status') or ('prose_delivered' if result.get('raw_delivered_text') else 'unavailable'),
            'blocking_issues': (final.get('consistency') or {}).get('issues') or final.get('issues') or [],
        })
        if row['cohort'] != 'fresh':
            continue
        events_path = run / 'llm_events.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()] if events_path.exists() else []
        usages = [e.get('usage') or {} for e in events]
        resources.append({
            'candidate_id': row['candidate_id'], 'case': row['case'], 'architecture': row['architecture'],
            'model_ids': sorted({e.get('model') for e in events if e.get('model')}),
            'runtime_seconds': recorded.get('runtime_s'), 'recorded_model_events': len(events),
            'input_tokens': sum((u.get('input_tokens') or 0) + (u.get('cache_read_input_tokens') or 0) + (u.get('cache_creation_input_tokens') or 0) for u in usages),
            'output_tokens': sum(u.get('output_tokens') or 0 for u in usages),
        })
    return deliveries, resources


def figure(root, summary, tokens, deliveries):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(15.4, 10.8), facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, .88), left=.066, right=.984, top=.79, bottom=.15, hspace=.58, wspace=.19)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    fig.text(.066, .957, 'Practical-input repair benchmark', fontsize=22, weight='bold')
    fig.text(.066, .92, 'Claude Sonnet 4.6 in all three arms · identical batch facts and inventory · one generation per case and arm', fontsize=11, color='#465365')
    fig.text(.066, .889, 'Nine fresh designs + six archived designs re-evaluated under the same revised evaluation protocol', fontsize=10, color='#465365')
    for axis, judge, heading in zip(axes, JUDGES, ('Primary evaluator · GPT-4o', 'Secondary evaluator · Qwen3.6-27B')):
        x = np.arange(4)
        width = .245
        for i, arm in enumerate(ARMS):
            values = [case_score(summary, judge, 'fresh', arm, case) for case in CASES] + [mean(summary, judge, 'fresh', arm)]
            bars = axis.bar(x + (i - 1) * width, values, width=.23, color=COLORS[i], label=arm, zorder=3)
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, value + .015, f'{value:.3f}', ha='center', va='bottom', fontsize=8.3)
        axis.set_title(heading, loc='left', pad=15, fontsize=13, weight='bold')
        axis.set_xticks(x, ['Hydrogenolysis', 'Photochemical\noxidation', 'CuAAC', 'Mean'])
        axis.tick_params(axis='x', labelsize=9, pad=8, length=0)
        axis.set_ylim(0, 1.085)
        axis.set_yticks(np.arange(0, 1.001, .2))
        axis.grid(axis='y', alpha=.18, zorder=0)
        fp_delta = mean(summary, judge, 'fresh', 'FlowPilot') - mean(summary, judge, 'fresh', 'Practical one-shot')
        guide_delta = mean(summary, judge, 'fresh', 'Guided one-shot') - mean(summary, judge, 'fresh', 'Practical one-shot')
        axis.text(0, -.205, f'FlowPilot − practical  {fp_delta:+.3f}     Guided − practical  {guide_delta:+.3f}', transform=axis.transAxes, fontsize=10)
    axes[0].set_ylabel('Mean applicable-criterion score / 4')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(.52, .875), fontsize=10)

    table_axis = fig.add_subplot(grid[1, 0])
    table_axis.axis('off')
    table_axis.set_title('Archived and fresh designs · same revised evaluation', loc='left', fontsize=12, weight='bold', pad=14)
    data = []
    for judge in JUDGES:
        old_fp = mean(summary, judge, 'archived', 'FlowPilot')
        new_fp = mean(summary, judge, 'fresh', 'FlowPilot')
        data.append([JUDGE_NAMES[judge], f"{mean(summary, judge, 'archived', 'Practical one-shot'):.3f}", f"{mean(summary, judge, 'fresh', 'Practical one-shot'):.3f}", f'{old_fp:.3f}', f'{new_fp:.3f}', f'{new_fp-old_fp:+.3f}'])
    table = table_axis.table(cellText=data, colLabels=['Evaluator', 'Old\none-shot', 'Fresh\none-shot', 'Old\nFlowPilot', 'Fresh\nFlowPilot', 'FlowPilot\nchange'],
        colWidths=[.235, .145, .155, .145, .165, .155], cellLoc='center', bbox=[0, .48, 1, .43])
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#D9E1E8')
        cell.set_linewidth(.5)
        if r == 0:
            cell.set_facecolor('#EDF2F5')
            cell.set_text_props(weight='bold')
    old_blocked = sum(d['cohort'] == 'archived' and d['architecture'] == 'FlowPilot' and d['delivery_status'] == 'blocked' for d in deliveries)
    new_blocked = sum(d['cohort'] == 'fresh' and d['architecture'] == 'FlowPilot' and d['delivery_status'] == 'blocked' for d in deliveries)
    new_executable = sum(d['cohort'] == 'fresh' and d['architecture'] == 'FlowPilot' and d['delivery_status'] == 'executable' for d in deliveries)
    table_axis.text(0, .33, f'FlowPilot blocked final deliveries  {old_blocked}/3 archived → {new_blocked}/3 fresh', fontsize=10, transform=table_axis.transAxes)
    table_axis.text(0, .21, f'Fresh executable contracts  {new_executable}/3  ·  software status, not laboratory validation', fontsize=9, color='#465365', transform=table_axis.transAxes)
    table_axis.text(0, .06, 'Original pilot ratings are not pooled with these re-evaluations.', fontsize=9, color='#465365', transform=table_axis.transAxes)

    token_axis = fig.add_subplot(grid[1, 1])
    token_axis.set_title('One-shot input length · full system + user input', loc='left', fontsize=12, weight='bold', pad=14)
    x = np.arange(3)
    for i, (variant, label) in enumerate((('practical', 'Practical'), ('guided', 'Guided'))):
        values = [next(r['input_tokens'] for r in tokens if r['case'] == case and r['input_variant'] == variant) for case in CASES]
        bars = token_axis.bar(x + (i - .5) * .32, values, width=.30, color=COLORS[i], zorder=3)
        for bar, value in zip(bars, values):
            token_axis.text(bar.get_x()+bar.get_width()/2, value + 20, f'{value:,}', ha='center', fontsize=9)
    token_axis.set_xticks(x, ['Hydrogenolysis', 'Photochemical\noxidation', 'CuAAC'])
    token_axis.tick_params(axis='x', labelsize=9, pad=8, length=0)
    token_axis.set_ylabel('Input tokens')
    token_axis.set_ylim(0, max(r['input_tokens'] for r in tokens) * 1.17)
    token_axis.grid(axis='y', alpha=.18, zorder=0)
    fig.text(.066, .098, 'One generation per case–arm; no error bars. Guidance content and token count change together. Development cases informed the repairs.', fontsize=9.7, color='#465365')
    fig.text(.066, .073, 'Independent audit found defects in all arms, including duplicate substrate charging in a FlowPilot recipe.', fontsize=10, color='#8B3B2D')
    fig.text(.066, .048, 'Judges still missed explicit arithmetic errors. Scores and software contract passes do not establish verified design quality.', fontsize=10, color='#8B3B2D')
    fig.text(.066, .023, f"Fixed 14-criterion rubric · revised judging instructions · estimated paid API use ${summary['usage']['estimated_paid_usd']:.2f} · local hardware cost excluded", fontsize=9.7, color='#465365')
    for extension in ('png', 'svg'):
        fig.savefig(root / 'report' / f'practical_repair_comparison.{extension}', dpi=220, facecolor='white')
    plt.close(fig)


def report_text(root, summary, tokens, deliveries, resources):
    lines = ['PRACTICAL-INPUT REPAIR BENCHMARK — CLAUDE SONNET 4.6', f"Completed {summary['completed_at']}", '',
        'Nine fresh designs compare practical one-shot, guidance-enriched one-shot and repaired FlowPilot over hydrogenolysis, photochemical oxidation and CuAAC. Six previous designs are re-evaluated unchanged under the same revised judging protocol. Each case–arm has one generation. GPT-4o is the primary evaluator; local Qwen3.6-27B is secondary.']
    for judge in JUDGES:
        lines += ['', f'FRESH RESULTS — {JUDGE_NAMES[judge]}', 'Case | Practical one-shot | Guided one-shot | FlowPilot']
        for case in CASES:
            lines.append(case + ' | ' + ' | '.join(f"{case_score(summary, judge, 'fresh', arm, case):.6f}" for arm in ARMS))
        lines.append('Mean | ' + ' | '.join(f"{mean(summary, judge, 'fresh', arm):.6f}" for arm in ARMS))
        fp_delta = mean(summary, judge, 'fresh', 'FlowPilot') - mean(summary, judge, 'fresh', 'Practical one-shot')
        guided_delta = mean(summary, judge, 'fresh', 'Guided one-shot') - mean(summary, judge, 'fresh', 'Practical one-shot')
        lines.append(f'FlowPilot minus practical one-shot: {fp_delta:+.6f}; guided minus practical one-shot: {guided_delta:+.6f}.')
        lines.append('Judge-reported critical flags: ' + '; '.join(f"{arm} {summary['groups'][judge]['fresh'][arm]['critical_flags']}" for arm in ARMS) + '.')
    lines += ['', 'ARCHIVED VERSUS FRESH — ALL SCORED WITH THE SAME REVISED EVALUATION', 'Evaluator | Archived one-shot | Fresh one-shot | Archived FlowPilot | Fresh FlowPilot | FlowPilot change']
    for judge in JUDGES:
        values = [mean(summary, judge, cohort, arm) for cohort, arm in [('archived','Practical one-shot'),('fresh','Practical one-shot'),('archived','FlowPilot'),('fresh','FlowPilot')]]
        lines.append(JUDGE_NAMES[judge] + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + f' | {values[3]-values[2]:+.6f}')
    lines += ['', 'INDEPENDENT CONTENT AUDIT — KEPT SEPARATE FROM JUDGE SCORES',
        'Read-only spot checks covered all nine fresh outputs. Findings were not supplied to the judges and do not change any grade. These checks are not exhaustive chemical, engineering or safety validation. The full audit is retained as report/manual_content_audit.txt.',
        'Hydrogenolysis: practical one-shot places a vented separator before its only pressure-defining BPR and gives incompatible bed-volume, collection and gas calculations. Guided one-shot has a correct BPR-before-separator diagram and consistent gas arithmetic, but a written assembly sequence contradicts its own diagram. FlowPilot has the correct canonical pressure boundary and closes calculations at the nominal concentration, but its final recipe separately charges 1.26 mmol DMAOL and 1.26 mmol N-diphenylmethylazetidin-3-ol (DMAOL) into 10 mL. They are the same substrate. Literal execution doubles concentration to 0.252 M while the calculations assume 0.126 M. A stale explanatory sentence also reverses BPR/separator order. This is a real delivered-recipe defect despite an executable software contract.',
        'Photochemical oxidation: practical one-shot equates 0.60 standard mL/min with 10 sccm and 5 sccm with about 0.30 mL/min, and gives conflicting liquid settings. Guided one-shot correctly compresses its gas flow, but understates substrate delivery by a factor of 16.67; its reported 4.3 oxygen equivalents are approximately 0.258 when its stated concentration and flow are used. FlowPilot’s spot-checked concentrations, oxygen-equivalent calculation, pressure conversion and graph close, with nominal time indices distinguished from unmeasured contact time. Its generic operating sequence still lacks a distinct LED-on/LED-off sequence.',
        'CuAAC: practical one-shot gives conflicting equilibration-volume statements and conflates substrate throughput with product throughput at an assumed yield. Guided one-shot miscalculates shutdown flushing volumes, supplies mutually inconsistent solvent-property statements and suggests subsequent flows below the pump minimum without resolving that limit. FlowPilot’s checked molar ratios, pressure boundary and empty-bed calculation close; duplicated stationary-catalyst descriptions, repeated collection instructions and generic heating-sequence instructions remain.',
        f"The revised judge instructions did not solve evaluator arithmetic failures. GPT-4o assigned the fresh practical photochemical design {case_score(summary, 'openai', 'fresh', 'Practical one-shot', 'Photochemical oxidation'):.3f} despite its directly observable gas-unit contradictions. The measured score ranking must therefore be reported as an LLM assessment, separately from the independently observed defects and delivery status.",
        'Any alias-normalization correction prepared or applied after this campaign is an additional, unscored change. These frozen outputs and grades contain the double-charge defect; they do not measure the effect of a later correction. A saved-output replay or unit test of that correction is not a fresh scored benchmark.',
        '', 'WHY THE PREVIOUS SCORES FELL',
        'The previous practical adapter removed the explicit chemistry confirmation and passed a combined protocol/inventory brief into parsing. The software treated the inventory phrase “hydrogenation skid” as reaction identity in hydrogenolysis and demanded confirmation when oxidation/CuAAC were described chemically without reaction-name keywords. All three Claude FlowPilot final contracts were consequently blocked. Frozen source hashes showed the same gate code was already present during the earlier Astra campaign; engineered confirmation had masked this input-path defect.',
        'The accepted Astra dev04 pilot and this campaign both use the legacy two-candidate workflow. Their generator models, task instructions, chemistry confirmation, evaluator aggregation and packet adapters differ, so the historical Astra means are not a matched control for the current Claude experiment.',
        'The previous one-shot scores also overstated numerical quality. Its oxidation response equated 0.30 standard mL/min with 5 sccm; both judges missed that contradiction. One secondary assessment accepted 1.0 = 0.10 × 5. The current evaluator instructions require explicit arithmetic verification and checking software claims against source facts. Original pilot judgments remain unchanged and are not pooled here.',
        '', 'CHANGES AND VERIFICATION',
        '1. Original batch protocol is separated from objectives, constraints and equipment in every parser and in the shared production entry point. Structured fields and supported aliases are preserved; model-extracted raw text cannot overwrite the original protocol.',
        '2. Chemistry identity can use conservative, traceable reactant-to-product evidence for the supported transformations. Named hardware is excluded from reaction-label authority. Source-declared acronyms are retained, while changed substrates, products and explicit stereochemistry remain detectable. No researcher confirmation is fabricated. These checks do not establish complete mechanistic or molecular-structure correctness.',
        '3. The matched benchmark uses three fresh arms with the same factual inputs. A generic checklist is the only additional input to guided one-shot. Both one-shot arms use ordinary prose and the same 8192-token output allowance. FlowPilot retains two initial candidates and its existing calculations, retrieval and review, so computation is not matched.',
        '4. The 14 criteria, applicability and weights are unchanged. Revised symmetric judging instructions distinguish provisional fallback from a released design, examine software warnings rather than adopting them as chemical truth, require numeric operands, and do not penalize appropriate uncertainty by itself. GPT-4o uses strict JSON from the start. Valid judgments are not repeated; invalid format/transport attempts are retained.',
        'The pre-run combined regression check passed 157 tests. Cases, source, prompts, execution plan, archived inputs, generated records and prepared packets are frozen and hash-checked. No score-dependent regeneration or selection is part of this experiment.',
        '', 'INPUT TOKENS', 'Case | Practical one-shot | Guided one-shot']
    for case in CASES:
        values = [next(t['input_tokens'] for t in tokens if t['case'] == case and t['input_variant'] == v) for v in ('practical','guided')]
        lines.append(f'{case} | {values[0]} | {values[1]}')
    lines += ['Counts use the Anthropic token-count API for the complete system-plus-user input. The checklist changes content and length simultaneously; a score difference cannot be attributed to token count alone. Input length also does not measure researcher effort.', '', 'DELIVERED OUTPUTS']
    for cohort in ('archived','fresh'):
        for case in CASES:
            for arm in ARMS:
                matching = [d for d in deliveries if d['cohort'] == cohort and d['case'] == case and d['architecture'] == arm]
                if matching:
                    d = matching[0]
                    codes = ', '.join(x.get('code','') for x in d['blocking_issues'])
                    lines.append(f"{cohort} | {case} | {arm} | {d['delivery_status']}" + (f' | {codes}' if codes else ''))
    lines += ['', 'RESOURCE USE — FRESH GENERATION ONLY']
    for r in sorted(resources, key=lambda r:(CASES.index(r['case']), ARMS.index(r['architecture']))):
        minutes = f"{r['runtime_seconds']/60:.2f} min" if r['runtime_seconds'] is not None else 'runtime not recorded'
        lines.append(f"{r['case']} | {r['architecture']} | {r['recorded_model_events']} recorded calls | {r['input_tokens']} input + {r['output_tokens']} output tokens | {minutes}")
    lines += [f"Total new campaign API estimate, including both cohorts' new evaluations: ${summary['usage']['estimated_paid_usd']:.4f}.",
        'This is calculated from returned usage, not an invoice; local hardware costs and unreported work on failed calls are not estimated.',
        'Price sources retained by the campaign: https://platform.claude.com/docs/en/models/sonnet-4-6/overview and https://developers.openai.com/api/docs/models/gpt-4o.',
        '', 'INTERPRETATION',
        'The fresh comparison measures this repaired workflow against two forms of direct generation. The archived comparison evaluates the old and new delivered outcomes with the same revised assessment; it is not a replay with identical model outputs and cannot isolate every code effect from generation variability. An executable software contract means the delivery gate passed, not that a laboratory experiment has been validated.',
        'These are previously inspected development chemistries, with one generation per cell. No error bars, statistical superiority, independent generalization or experimental-success probabilities are established. All arm results and both evaluators are reported, including unfavorable differences. The revised instructions require stronger checks, but the continuing arithmetic misses show that reliable automatic verification has not been established.',
        '', 'EVIDENCE',
        'report/summary.json and accepted_judgments.json identify the results and accepted assessment files. frozen/ contains the protocol, cases, original/frozen-source hashes, input tokens and judge addendum. archived_inputs/ contains copied original designs; generation/ contains fresh attempts and telemetry; packets/ and judgments/ contain all evidence and assessments. prepared_manifest.json freezes generated outcomes and packets. Report-generation provenance is in report/report_artifact_manifest.json.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', nargs='?', type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.campaign.resolve()
    if not (root / 'report/summary.json').exists():
        raise SystemExit('Campaign summary is not ready; no report was generated.')
    summary = read(root / 'report/summary.json')
    accepted = read(root / 'accepted_judgments.json')
    tokens = read(root / 'frozen/input_tokens.json')
    key = read(root / 'frozen/candidate_key_confidential.json')['candidates']
    verify_inputs(root, summary, accepted)
    deliveries, resources = collect_delivery(root, key)
    available_audits = [path for path in AUDIT_PATHS if path.exists()]
    if not available_audits:
        raise SystemExit('Independent content audit is missing; refusing to generate a report that cites it.')
    (root / 'report/manual_content_audit.txt').write_text('\n\n'.join(path.read_text() for path in available_audits))
    write(root / 'report/delivery_status.json', deliveries)
    write(root / 'report/resources.json', resources)
    figure(root, summary, tokens, deliveries)
    (root / 'report/benchmark_report.txt').write_text(report_text(root, summary, tokens, deliveries, resources))
    paths = [root / 'report' / name for name in ('summary.json','delivery_status.json','resources.json','practical_repair_comparison.png','practical_repair_comparison.svg','benchmark_report.txt','manual_content_audit.txt')]
    write(root / 'report/report_artifact_manifest.json', {'script':str(Path(__file__).resolve()),'script_sha256':sha(Path(__file__)),
        'artifacts':{str(p.relative_to(root)):sha(p) for p in paths}, 'manual_audit_sources':{str(p):sha(p) for p in available_audits}, 'summary_completed_at':summary['completed_at']})
    print(json.dumps({'report':str(root/'report/benchmark_report.txt'),'figure':str(root/'report/practical_repair_comparison.png'),
        'primary_fresh':summary['groups']['openai']['fresh'],'paid_usd':summary['usage']['estimated_paid_usd']}, indent=2))


if __name__ == '__main__':
    main()
