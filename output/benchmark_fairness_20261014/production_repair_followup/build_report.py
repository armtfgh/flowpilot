"""Render the fixed, unedited development comparison from accepted integer grades."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
r=json.loads((P/'campaign/report/summary.json').read_text())
s=r['predeclared_secondary']; assert s['complete']
judges=set(s['judges'])
rows=[x for x in r['results'] if x['judge'] in judges]
cases=['Hydrogenolysis','Photochemical oxidation','CuAAC']
summary=[]
for case in cases:
    by={arm:[x for x in rows if x['case']==case and x['architecture']==arm] for arm in ('One-shot','FlowPilot')}
    assert all(len(v)==2 for v in by.values())
    values={arm:statistics.mean(x['score'] for x in rr) for arm,rr in by.items()}
    summary.append({'case':case,**values,'difference':values['FlowPilot']-values['One-shot']})
previous=json.loads((P.parent/'schema_routing_followup/common_complete_judge_sensitivity.json').read_text())
lines=[
'FLOWPILOT PRODUCTION REPAIR BENCHMARK — 14 OCTOBER 2026',
'',
'Claude Sonnet 4.6 generator; fixed one-shot outputs and grades; one fresh FlowPilot generation per chemistry.',
'This is a development comparison on previously inspected cases, not independent validation or proof of historical repeatability.',
'',
'PRESPECIFIED TWO-JUDGE COMPARISON',
'GPT-5.4 and Claude Sonnet 4.6; original fourteen criteria and applicability rules; equal judge/criterion/case weights.',
'A criterion receives an integer from 0 to 4. Applicable criteria are averaged and divided by four; the two judge scores and three cases then receive equal weight. Critical findings remain separate.',
'',
'Case                         One-shot     FlowPilot    Difference',
]
for v in summary:lines.append(f"{v['case']:28s} {v['One-shot']:.6f}     {v['FlowPilot']:.6f}     {v['difference']:+.6f}")
a,b=s['groups']['One-shot'],s['groups']['FlowPilot']
lines.extend([
f"{'Overall':28s} {a['mean']:.6f}     {b['mean']:.6f}     {s['flowpilot_minus_one_shot']:+.6f}",
f"Judge-level critical flags: one-shot {a['critical_flags']}; FlowPilot {b['critical_flags']}.",
'',
f"Previous corrected-routing FlowPilot mean with the same two judges: {previous['groups']['FlowPilot']['mean']:.6f}, with {previous['groups']['FlowPilot']['critical_flags']} flags. The new run uses fresh generations, so score movement includes generation/judging variability and does not isolate a causal effect of any individual fix.",
'',
'By judge (one-shot / FlowPilot):',
])
for j in s['judges']:lines.append(f"{j}: {a['by_judge'][j]['mean']:.6f} / {b['by_judge'][j]['mean']:.6f}")
lines.extend([
'',
'THREE-JUDGE PRIMARY',
f"Strict accepted judgments: {r['accepted_judgments']}/{r['expected_judgments']}. The primary FlowPilot mean is unavailable because the original local Qwen judge could not fit two complete input packets with the unchanged 7,500-token output allowance into its configured 32,768-token context. No evidence was truncated, no missing grade imputed and no valid response retried. The fixed two-judge comparison above was declared before generation and is reported as secondary.",
'',
'WHAT CHANGED',
'Current numerical rationale is computed from final settings; all original numerical explanations remain delivered as history and qualitative concerns remain explicitly unverified. Stream/stage fingerprints detect dependencies missed by the old rationale tracker.',
'Prepared feed quantities now cover the same commissioning and collection schedule used in the procedure. Every feed uses its own rate; multistage time uses local stage flow. Additional dead volume, extended commissioning and available stock remain to be confirmed.',
'Specific shutdown/emergency instructions no longer lose to earlier generic fallback text. Irradiation start/stop sequencing is explicit. Unresolved procedure compilation blocks release with a diagnostic.',
'Previously applied parser-envelope, same-feed solvent and final-output routing fixes are included. No judge instructions, rubric, packet adapter, one-shot artifact or one-shot grade changed.',
'',
'VALIDATION',
'779 available-suite tests passed; one unrelated FastAPI test module could not collect because that dependency is absent. Final targeted tests after the exception guard passed. Offline replay covered all three saved designs with unchanged physical settings and three failed-pump negative controls. See test logs and offline_replay/summary.json.',
'',
'REMAINING CONTENT FINDINGS',
'The independent content audit, completed without reading new judge grades, retains stale numbers in explicitly unverified chemistry prose; an unspecified phase/reference basis for the photochemistry separator throughput limit; and incomplete CuAAC hot-effluent cooling/flash-handling instructions. Numerical closure and high judge scores do not resolve these engineering questions. See content_audit/FINDINGS.txt for exact evidence and qualifications.',
'',
f"Estimated API cost: ${r['usage']['estimated_paid_usd']:.4f}, under the separate $5 campaign guard. Local hardware cost is not estimated.",
'All three attempted FlowPilot generations, every raw judge response, failed context attempts, settings, source snapshot, original one-shot copies and provenance are retained in campaign/.',
'Frozen plan: campaign/frozen/campaign_manifest.json and campaign/frozen/PROTOCOL.txt.',
'Raw accepted-grade summary: campaign/report/summary.json.',
'Independent integrity verification: independent_audit/.',
])
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
(P/'paired_comparison.json').write_text(json.dumps({'evaluation':'predeclared secondary','judges':s['judges'],'cases':summary,'overall':s,'missing_primary':r['missing']},indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8,4.6))
labels=['Hydrogenolysis','Photochemical\noxidation','CuAAC','Overall']
os=[v['One-shot'] for v in summary]+[a['mean']]
fp=[v['FlowPilot'] for v in summary]+[b['mean']]
x=list(range(4));width=.34
ax.bar([v-width/2 for v in x],os,width,color='#94A3B8',label='Frozen one-shot')
ax.bar([v+width/2 for v in x],fp,width,color='#167D8D',label='Repaired FlowPilot')
for vals,shift in ((os,-width/2),(fp,width/2)):
 for xx,yy in zip(x,vals):ax.text(xx+shift,yy+.014,f'{yy:.3f}',ha='center',fontsize=9)
ax.set_xticks(x,labels);ax.set_ylim(0,1.08);ax.set_ylabel('Assessed design score (0–1)')
ax.set_title('FlowPilot repair check · Claude Sonnet 4.6',loc='left',fontweight='bold',pad=16)
ax.legend(loc='upper left',bbox_to_anchor=(0,1.01),frameon=False,ncol=2)
ax.yaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
fig.text(.08,.025,'Prespecified GPT-5.4 + Claude comparison; one design per case and arm.\nThree-judge primary incomplete (two Qwen context failures). Development cases, not independent validation.',fontsize=8,color='#475569')
fig.subplots_adjust(left=.08,right=.98,top=.86,bottom=.20)
fig.savefig(P/'benchmark_comparison.png',dpi=220)
fig.savefig(P/'benchmark_comparison.svg')
print(json.dumps({'cases':summary,'overall':s['flowpilot_minus_one_shot'],'report':str(P/'REPORT.txt'),'figure':str(P/'benchmark_comparison.png')},indent=2))
