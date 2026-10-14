"""Reproduce the latest completed Astra pilot figure from accepted judgments."""
from pathlib import Path
import hashlib
import json
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from ablation_test.scripts.build_newgen_2_0_report import resolved_response

CAMPAIGN = ROOT / 'ablation_results/manuscript_benchmark/astra_verified_repairs_dev04_20261012'
sources = {}

def read(path):
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

manifest = read(CAMPAIGN / 'frozen/campaign_manifest.json')
rubric = read(CAMPAIGN / 'frozen/outcome_rubric.json')
key = read(CAMPAIGN / 'frozen/candidate_key_confidential.json')['candidates']
summary = read(CAMPAIGN / 'report/summary.json')
pilot = read(CAMPAIGN / 'report/pilot_summary.json')
integrity = read(CAMPAIGN / 'generation_integrity_audit.json')
inputs = read(CAMPAIGN / 'matched_input_audit.json')
usage = read(CAMPAIGN / 'report/api_usage.json')
assert inputs['frozen_source_unchanged'] and all(c['inputs_identical'] for c in inputs['cases'])
assert integrity['response_ids_unique_and_not_reused']
for relative, expected in integrity['generation_files_sha256'].items():
    assert hashlib.sha256((CAMPAIGN / relative).read_bytes()).hexdigest() == expected, relative

rows = []
for candidate in key:
    cid = candidate['candidate_id']
    packet = read(CAMPAIGN / 'packets' / f'{cid}.json')
    scores, flags = {}, 0
    for judge in manifest['selected_judges']:
        call = CAMPAIGN / 'judgments' / judge / cid
        # Retain hashes of accepted and rejected attempts for provenance.
        for path in call.rglob('*.json'):
            if path.name in ('status.json', 'parsed_response.json'):
                read(path)
        response = resolved_response(CAMPAIGN, judge, cid, rubric, packet)
        assert response is not None
        applicable = [r for r in response['criterion_scores'] if r['applicability'] == 'APPLICABLE']
        scores[judge] = round(statistics.fmean(r['score'] for r in applicable) / 4, 6)
        flags += sum(bool(r['critical_error']) for r in applicable)
    rows.append(dict(case=candidate['case'], architecture=candidate['architecture'],
                     scores_by_judge=scores, score=round(statistics.fmean(scores.values()), 6),
                     excluding_openai=round(statistics.fmean(scores[j] for j in ('qwen', 'claude')), 6),
                     critical_flags=flags))

means = {}
for arch in ('One-shot', 'FlowPilot'):
    selected = [r for r in rows if r['architecture'] == arch]
    means[arch] = dict(score=round(statistics.fmean(r['score'] for r in selected), 6),
                       excluding_openai=round(statistics.fmean(r['excluding_openai'] for r in selected), 6),
                       critical_flags=sum(r['critical_flags'] for r in selected))
    stored = next(r for r in summary['architecture_summary'] if r['architecture'] == arch)
    assert means[arch]['score'] == stored['mean_score_0_1']
    assert means[arch]['excluding_openai'] == stored['generator_family_excluded_mean_0_1']
    assert means[arch]['critical_flags'] == stored['critical_flags']
assert len(rows) == 6 and len(manifest['selected_judges']) == 3

data = dict(campaign=str(CAMPAIGN.relative_to(ROOT)), completed_at=pilot['completed_at'],
            prepared_date='2026-10-13', candidates=rows, means=means,
            delta=round(means['FlowPilot']['score'] - means['One-shot']['score'], 6),
            generation_usd=usage['generation_usd'], judging_usd=usage['judging_usd'],
            estimated_total_usd=usage['total_usd'], fresh_astra_calls=integrity['fresh_astra_calls'],
            verification=dict(accepted_judgments=18, generation_files_checked=len(integrity['generation_files_sha256']),
                              matched_inputs=True, archived_scores_reproduced=True),
            source_sha256=sources)
(OUT / 'verified_results.json').write_text(json.dumps(data, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.edgecolor': '#CDD4DD',
                     'xtick.color': '#475569', 'ytick.color': '#25364A',
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
teal, gray, ink = '#087F8C', '#8594A6', '#182B42'
fig = plt.figure(figsize=(10, 5.3), facecolor='white')
ax = fig.add_axes([.20, .29, .39, .46])
delta_ax = fig.add_axes([.72, .29, .24, .46], sharey=ax)
cases = ['CuAAC', 'Photochemical oxidation', 'Hydrogenolysis', 'Overall mean']
ys = [3, 2, 1, 0]
for case, y in zip(cases, ys):
    values = {arch: means[arch]['score'] if case == 'Overall mean' else
              next(r['score'] for r in rows if r['case'] == case and r['architecture'] == arch)
              for arch in ('One-shot', 'FlowPilot')}
    for arch, offset, color in [('One-shot', .15, gray), ('FlowPilot', -.15, teal)]:
        v = values[arch]
        ax.barh(y + offset, v, height=.23, color=color, zorder=3)
        ax.text(v + .012, y + offset, f'{v:.4f}', va='center', fontsize=9, color=ink)
    d = (values['FlowPilot'] - values['One-shot']) * 100
    color = teal if d > 0 else ('#925939' if d < 0 else gray)
    delta_ax.plot([0, d], [y, y], color=color, lw=2.7, zorder=3)
    delta_ax.scatter([d], [y], s=60 if case == 'Overall mean' else 43,
                     color=color, marker='D' if case == 'Overall mean' else 'o', zorder=4)
    delta_ax.annotate(f'{d:+.2f}' if d else '0.00', (d, y),
                      xytext=(0, 12), textcoords='offset points', ha='center',
                      fontsize=10, weight='bold' if case == 'Overall mean' else 'normal', color=ink)
for a in [ax, delta_ax]:
    a.set_ylim(-.52, 3.6)
    a.axhspan(-.5, .5, color='#F0F6F7', zorder=0)
    a.grid(axis='x', color='#E7ECF0', linewidth=.7, zorder=0)
    a.tick_params(axis='both', length=0)
ax.set_xlim(0, 1.12)
ax.set_xticks([0, .25, .5, .75, 1], ['0', '0.25', '0.50', '0.75', '1.00'])
ax.set_yticks(ys, ['CuAAC', 'Photochemical\noxidation', 'Hydrogenolysis', 'Overall mean'])
ax.tick_params(axis='y', pad=10)
ax.set_xlabel('Assessed score (0-1)', labelpad=9)
delta_ax.set_xlim(-1.10, 1.85)
delta_ax.set_xticks([-1, 0, 1], ['-1', '0', '+1'])
delta_ax.tick_params(axis='y', labelleft=False)
delta_ax.axvline(0, color='#AAB5C1', lw=1)
delta_ax.set_xlabel('Difference (percentage points)', labelpad=9, fontsize=9)
delta_ax.set_title('FlowPilot minus one-shot', fontsize=10, color=ink, pad=14)
fig.text(.055, .945, 'Latest Astra benchmark', size=19, weight='bold', color=ink)
fig.text(.055, .893, 'Completed 12 October 2026 | GPT-6 Astra in both architectures | Development pilot',
         size=10, color='#536478')
fig.legend(handles=[Patch(color=gray, label='One-shot Astra'), Patch(color=teal, label='FlowPilot (Astra)')],
           loc='upper left', bbox_to_anchor=(.195, .862), frameon=False, ncol=2, fontsize=10)
fig.text(.055, .155, '3 chemistries | 1 generation per chemistry and architecture | 3 judges per design | No repeat error bars',
         size=9, color='#536478')
fig.text(.055, .105, 'Overall difference: +0.25 percentage points. Zero judge-reported critical flags for either architecture.',
         size=9, color=ink)
fig.text(.055, .058, 'Excluding the GPT-4o judge reverses the mean ranking (FlowPilot 0.9607; one-shot 0.9768).',
         size=9, color='#536478')
fig.savefig(OUT / 'astra_latest_comparison.png', dpi=300)
fig.savefig(OUT / 'astra_latest_comparison.svg')
plt.close(fig)
print(json.dumps({'verified_means': means, 'delta': data['delta'], 'output': str(OUT)}, indent=2))
