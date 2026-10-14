"""Generate a readable report and static figure from a completed practical pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);args=p.parse_args()
root=args.campaign.resolve(); report=root/'report'; report.mkdir(exist_ok=True)
def read(path): return json.loads(path.read_text())
def write(path, data): path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
summary=read(report/'summary.json'); key=read(root/'frozen/candidate_key_confidential.json')['candidates']
tokens=read(root/'frozen/input_token_comparison.json')
resources=[]; deliveries=[]; topologies=[]
for row in key:
    run=Path(row['run_directory']); rs=read(run/'run_summary.json')
    result=read(run/'result.json') if (run/'result.json').exists() else {}
    events=[json.loads(line) for line in (run/'llm_events.jsonl').read_text().splitlines()] if (run/'llm_events.jsonl').exists() else []
    models=sorted({e.get('model') for e in events if e.get('model')})
    assert models==['claude-sonnet-4-6'], models
    usage=[e.get('usage') or {} for e in events]
    resources.append({'case':row['case'],'architecture':row['architecture'],'status':rs['status'],
        'runtime_seconds':rs['runtime_s'],'model_ids':models,'calls':len(events),
        'input_tokens':sum(u.get('input_tokens',0)+u.get('cache_read_input_tokens',0)+u.get('cache_creation_input_tokens',0) for u in usage),
        'output_tokens':sum(u.get('output_tokens',0) for u in usage),
        'estimated_generation_usd':sum((u.get('input_tokens',0)*3+u.get('output_tokens',0)*15+u.get('cache_read_input_tokens',0)*.3+u.get('cache_creation_input_tokens',0)*3.75)/1e6 for u in usage)})
    final=result.get('final_design') or {}
    deliveries.append({'case':row['case'],'architecture':row['architecture'],
        'software_delivery_status':final.get('status','prose_delivered' if row['architecture']=='One-shot' else 'unavailable'),
        'blocking_issues':(final.get('consistency') or {}).get('issues',[]), 'gate':final.get('consistency',{}),
        'recommended_disposition':result.get('recommended_disposition'),
        'run_directory':str(run)})
    export=root/'readable_outputs'/row['case'].lower().replace(' ','_')
    export.mkdir(parents=True,exist_ok=True)
    (export/(row['architecture'].lower().replace('-','_')+'.txt')).write_text(
        result.get('raw_delivered_text') or json.dumps(final or result,ensure_ascii=False,indent=2))
    diagram=result.get('diagram_artifacts') or {}
    if diagram.get('run_dir'):
        source=Path(diagram['run_dir']).resolve()
        if source.is_dir():
            target=root/'topologies'/row['case'].lower().replace(' ','_')
            shutil.copytree(source,target,dirs_exist_ok=True)
            topologies.append({'case':row['case'],'source':str(source),'copy':str(target.relative_to(root)),
                'software_delivery_status':final.get('status'),'experimental_validation':False})
write(report/'resources.json',resources);write(report/'delivery_status.json',deliveries)
write(root/'topology_manifest.json',topologies)

cases=['Hydrogenolysis','Photochemical oxidation','CuAAC']
colors=['#8C9AAB','#007B83']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12,5.9),sharey=True)
fig.suptitle('Practical-input benchmark · Claude Sonnet 4.6',x=.075,y=.97,ha='left',weight='bold',fontsize=19)
fig.text(.075,.915,'Identical batch facts, short request and complete inventory in both architectures',fontsize=11,color='#4b5563')
for ax,judge,title in zip(axes,['openai','qwen'],['Primary evaluator · GPT-4o','Secondary evaluator · Qwen3.6-27B']):
    x=np.arange(4);w=.34
    for k,arch in enumerate(['One-shot','FlowPilot']):
        vals=[next(r['score'] for r in summary['judge_results'] if r['judge']==judge and r['case']==c and r['architecture']==arch) for c in cases]
        vals.append(summary['by_judge'][judge][arch]['mean'])
        bars=ax.bar(x+(k-.5)*w,vals,w,label=arch,color=colors[k],zorder=3)
        for bar,v in zip(bars,vals): ax.text(bar.get_x()+w/2,v+.014,f'{v:.3f}',ha='center',fontsize=9)
    ax.set_title(title,loc='left',pad=17,fontsize=12,weight='bold')
    ax.set_xticks(x,['Hydrogenolysis','Photochemical\noxidation','CuAAC','Mean'])
    ax.tick_params(axis='x',labelsize=9,length=0,pad=8)
    ax.set_ylim(0,1.085);ax.set_yticks(np.arange(0,1.01,.2));ax.grid(axis='y',alpha=.18,zorder=0)
    delta=summary['by_judge'][judge]['difference']
    a=summary['by_judge'][judge]['One-shot']['critical_flags'];b=summary['by_judge'][judge]['FlowPilot']['critical_flags']
    ax.text(0,-.24,f'FlowPilot − one-shot  {delta:+.3f}   |   Critical flags  {a} → {b}',transform=ax.transAxes,fontsize=10)
axes[0].set_ylabel('Mean applicable-criterion score / 4')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.167),frameon=False,ncol=2)
fig.text(.075,.075,'All three FlowPilot final designs were blocked by chemistry-identity validation.',fontsize=9,color='#4b5563')
fig.text(.075,.043,'Three development cases · one generation per case and architecture · two initial FlowPilot candidates',fontsize=9,color='#4b5563')
fig.text(.075,.016,'No repeatability estimate. Flags are evaluator findings; scores do not establish laboratory performance.',fontsize=9,color='#4b5563')
fig.subplots_adjust(left=.075,right=.985,top=.815,bottom=.27,wspace=.17)
for ext in ['png','svg']: fig.savefig(report/f'practical_input_comparison.{ext}',dpi=220,facecolor='white')
plt.close(fig)

lines=['PRACTICAL-INPUT BENCHMARK — CLAUDE SONNET 4.6','Completed '+summary['completed_at'],'',
'Both architectures used claude-sonnet-4-6 through the direct Anthropic API. The three development chemistries were hydrogenolysis, photochemical oxidation and CuAAC. Each received one fresh generation in each architecture. GPT-4o was the prespecified primary evaluator; local Qwen3.6-27B provided a secondary assessment.','',
'RESULTS','Case | GPT-4o one-shot | GPT-4o FlowPilot | Difference']
for case in cases:
    rows=[r for r in summary['judge_results'] if r['judge']=='openai' and r['case']==case]
    a=next(r['score'] for r in rows if r['architecture']=='One-shot');b=next(r['score'] for r in rows if r['architecture']=='FlowPilot')
    lines.append(f'{case} | {a:.6f} | {b:.6f} | {b-a:+.6f}')
for judge in ['openai','qwen']:
    rr=summary['by_judge'][judge];a=rr['One-shot'];b=rr['FlowPilot']
    lines.append(f"{judge} mean: one-shot {a['mean']:.6f}; FlowPilot {b['mean']:.6f}; difference {rr['difference']:+.6f}. Critical flags {a['critical_flags']} versus {b['critical_flags']}.")
lines+=['','INPUT AND COMPARISON',
'A 26-word ordinary request accompanies the factual batch procedure and the complete inventory. Both architectures receive the same information. The one-shot response is ordinary prose, without a JSON requirement, with an 8192-token output allowance. No researcher clarification is simulated. The former engineered prompt is token-counted for context only; it is not a regenerated control in this campaign.',
'FlowPilot uses the existing production path with two initial candidates, calculations, source-excluded lexical literature retrieval, council review and native revisions. It has more computation. This is not the 12-candidate GUI campaign and not an equal-token test. Production code and final-delivery gates remain unchanged.',
'Anthropic token-count API estimates for full system-plus-user inputs:']
for t in tokens: lines.append(f"  {t['case']}: practical {t['practical_total_input_tokens']}; engineered {t['engineered_total_input_tokens']}; inventory identical.")
lines+=['','SCORING',
'The fixed 14-criterion rubric scores applicable criteria from 0 to 4. Scores are averaged within each design and divided by four, then averaged equally across the three cases. GPT-4o and Qwen results remain separate. Criteria marked inapplicable are excluded. Full prose is preserved, and format-neutral instructions apply to both arms. Automatic numerical worksheets are omitted for both; judges assess arithmetic directly. Generator and architecture labels are withheld from packets, although output style can reveal the architecture.',
'Critical flags are judge findings, not independently confirmed physical errors. All six generation attempts are retained. Only judge formatting or transport failures may be retried. The original limit was three attempts; a documented amendment enabled strict JSON output and up to three additional attempts for an exhausted format failure. Valid judgments were not rerun. Every attempt is archived in full. No score-dependent regeneration, production repair or stopping occurred.',
'','DELIVERY STATUS']
for d in deliveries:lines.append(f"  {d['case']} / {d['architecture']}: {d['software_delivery_status']}")
lines+=['','RESOURCE USE']
for r in resources: lines.append(f"  {r['case']} / {r['architecture']}: {r['calls']} calls; {r['input_tokens']} input + {r['output_tokens']} output tokens; {r['runtime_seconds']/60:.2f} minutes; ${r['estimated_generation_usd']:.4f} generation estimate.")
lines+=['',f"Paid API estimate from returned usage: ${summary['usage']['estimated_paid_usd']:.4f}. Local inference hardware cost is not estimated. This is a token-based estimate, not an invoice.",
'Pricing checked 14 October 2026: Claude input/output $3/$15 per million tokens; GPT-4o $2.50/$10. Cache rates are applied from full API usage records.',
'  https://platform.claude.com/docs/en/models/sonnet-4-6/overview',
'  https://developers.openai.com/api/docs/models/gpt-4o',
'','INTERPRETATION AND SCOPE',
'This is a six-outcome development pilot with one generation per cell. It describes this frozen practical-input configuration; it cannot establish repeatability, statistical superiority or generalization to unseen chemistry. It does not isolate the effect of prompt shortening because an engineered-input control was not regenerated. Token counts do not measure researcher preparation time. A higher-capability generator cannot necessarily resolve a deterministic delivery gate that requires researcher confirmation.',
'','ARCHIVE',
'frozen/ contains the preregistration, cases, full source snapshot, source hashes, rubric, prompt-token counts, input changes and execution plan. generation/ retains prompts, raw responses, events, stage snapshots and final designs. packets/ and judgments/ retain normalized evidence and all judge responses. paid_usage.jsonl retains full provider usage, resolved model IDs and estimates. readable_outputs/ exports the delivered prose or final design for inspection. topologies/ contains generated diagrams with delivery status in topology_manifest.json. No diagram is a validated laboratory apparatus.']
if (root/'evaluation_format_amendment.json').exists():
    lines+=['','EVALUATION FORMAT AMENDMENT',
        'GPT-4o returned three invalid assessments for one design under strict:false, omitting required fields or returning incompatible critical-flag and score fields. Before further calls, evaluation_format_amendment.json authorized strict:true on the existing JSON schema and up to three additional format-only attempts for the unresolved design. Remaining unattempted GPT-4o judgments also used strict output. The original rubric and validator remained unchanged; all earlier attempts were preserved, and valid judgments were not regenerated. This is a documented departure from the original three-attempt stopping rule, not a fully unchanged evaluation protocol. See strict_transport_events.jsonl and resume_strict_judging.py.']
if (report/'qualitative_audit.txt').exists(): lines+=['','QUALITATIVE AUDIT',(report/'qualitative_audit.txt').read_text()]
(report/'benchmark_report.txt').write_text('\n'.join(lines)+'\n')
manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file() and f.name not in ['artifact_manifest.json','paid_usage.lock']}
write(root/'artifact_manifest.json',manifest)
print(json.dumps({'report':str(report),'archived_files':len(manifest),'paid_usd':summary['usage']['estimated_paid_usd']}))
