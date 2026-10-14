"""Read-only archival analysis. Writes only its own diagnostic directory; no APIs."""
from pathlib import Path
import csv, json, hashlib, statistics, collections
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
OLD=ROOT/'ablation_results/manuscript_benchmark/manuscript_three_model_three_repeat_postfix_20260820'
NEW=ROOT/'ablation_results/practical_input/claude_practical_repair_20261014'
COMBINED=ROOT/'deliverables/newgen_2_0_all_models_three_repeat_20260824'
MODELS=['Qwen3.6-27B','Qwen3.8-27B','GPT-4o','Claude Sonnet 4.6','Claude Opus 4.6']
SOURCES={}
def read(p):
 p=Path(p); b=p.read_bytes(); SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest(); return b.decode()
def js(p):return json.loads(read(p))
def rows(p):return list(csv.DictReader(read(p).splitlines()))
def mean(v):return statistics.mean(v)
def grouping(items,keys,score='score'):
 groups=collections.defaultdict(list)
 for r in items:groups[tuple(r[k] for k in keys)].append(r)
 ans=[]
 for key,rs in sorted(groups.items()):
  vs=[r[score] for r in rs]
  ans.append(dict(zip(keys,key),n=len(rs),mean=mean(vs),minimum=min(vs),maximum=max(vs),perfect_count=sum(v==1 for v in vs),critical_flags=sum(r.get('critical_flags',0) for r in rs)))
 return ans
combined=rows(COMBINED/'tables/all_candidate_consensus_scores.csv')
selected=[r for r in combined if r['model'] in MODELS]
assert len(selected)==90
crit=[r for r in rows(COMBINED/'tables/all_criterion_judgments.csv') if r['model'] in MODELS]
by_judge=collections.defaultdict(list)
for r in crit:by_judge[(r['model'],r['architecture'],r['case'],r['candidate_id'],r['judge'])].append(r)
old_judge=[]
for (model,arm,case,cid,judge),rr in by_judge.items():
 ar=[r for r in rr if r['applicability']=='APPLICABLE'];assert len(rr)==14
 old_judge.append(dict(model=model,architecture=arm,case=case,candidate_id=cid,judge=judge,score=mean(float(r['score_0_4']) for r in ar)/4,critical_flags=sum(r['critical_error'].lower()=='true' for r in rr),applicable_count=len(ar)))
assert len(old_judge)==270
by_candidate=collections.defaultdict(list)
for r in old_judge:by_candidate[(r['model'],r['candidate_id'])].append(r)
old_candidates=[]
for r in selected:
 j=by_candidate[(r['model'],r['candidate_id'])];assert len(j)==3
 v=mean(x['score'] for x in j);assert abs(v-float(r['mean_score_0_1']))<0.000001
 old_candidates.append(dict(model=r['model'],architecture=r['architecture'],case=r['case'],candidate_id=r['candidate_id'],repeat_id=r['repeat_id'],score=v,critical_flags=sum(x['critical_flags'] for x in j),run_directory=r['run_directory']))
sonnet=[r for r in old_candidates if r['model']=='Claude Sonnet 4.6']
old_manifest=js(OLD/'frozen/campaign_manifest.json');new_manifest=js(NEW/'frozen/campaign_manifest.json')
old_key=js(OLD/'frozen/candidate_key_confidential.json')['candidates']
new_key=js(NEW/'frozen/candidate_key_confidential.json')['candidates']
old_settings=[];raw_old_sonnet=[]
for k in old_key:
 if k['generator_model']!='Claude Sonnet 4.6':continue
 for judge in ['qwen','openai','claude']:
  p=OLD/'judgments'/judge/k['candidate_id']
  candidates=[p/'status.json', *sorted((p/'attempts').glob('attempt_*/status.json'))]
  valid=[(z.stat().st_mtime_ns,z,js(z)) for z in candidates if js(z).get('status')=='valid']
  assert valid
  _,accepted_path,st=max(valid,key=lambda t:t[0]);p=accepted_path.parent
  ev=js(p/'telemetry.json');pr=js(p/'prompt.json');jud=js(p/'parsed_response.json')
  scored=[x for x in jud['criterion_scores'] if x['applicability']=='APPLICABLE']
  sr=dict(model=k['generator_model'],architecture=k['architecture'],case=k['case'],repeat_id=k['repeat_id'],candidate_id=k['candidate_id'],judge=judge,score=mean(x['score'] for x in scored)/4,critical_flags=sum(bool(x['critical_error']) for x in jud['criterion_scores']))
  assert abs(sr['score']-next(x['score'] for x in old_judge if x['model']==sr['model'] and x['candidate_id']==sr['candidate_id'] and x['judge']==judge))<1e-12
  raw_old_sonnet.append(sr)
  for e in ev:
   old_settings.append(dict(candidate_id=k['candidate_id'],judge=judge,provider=st['provider'],model=st['model'],temperature=e.get('temperature'),max_tokens=e.get('max_tokens'),seed=e.get('seed'),system_chars=len(pr['system']),user_chars=len(pr['user']),usage=e.get('usage'),response_chars=e.get('response_chars'),status_path=str((p/'status.json').relative_to(ROOT))))
accepted=js(NEW/'accepted_judgments.json');summary=js(NEW/'report/summary.json')
new_scores=[];new_settings=[];new_criteria=[]
for a in accepted:
 k=next(k for k in new_key if k['candidate_id']==a['candidate_id'])
 if k.get('cohort')!='fresh':continue
 p=NEW/a['status_path'];st=js(p);q=p.parent;pr=js(q/'prompt.json');jud=js(q/'parsed_response.json');ev=js(q/'telemetry.json')
 assert st['status']=='valid'; scored=[x for x in jud['criterion_scores'] if x['applicability']=='APPLICABLE']
 new_scores.append(dict(model=k['generator_model'],architecture=k['architecture'],case=k['case'],repeat_id=k['repeat_id'],candidate_id=k['candidate_id'],judge=a['judge'],score=mean(x['score'] for x in scored)/4,critical_flags=sum(bool(x['critical_error']) for x in jud['criterion_scores'])))
 for x in jud['criterion_scores']:new_criteria.append(dict(candidate_id=k['candidate_id'],architecture=k['architecture'],case=k['case'],judge=a['judge'],**x))
 for e in ev:new_settings.append(dict(candidate_id=k['candidate_id'],judge=a['judge'],provider=st['provider'],model=st['model'],temperature=e.get('temperature'),max_tokens=e.get('max_tokens'),seed=e.get('seed'),system_chars=len(pr['system']),user_chars=len(pr['user']),usage=e.get('usage'),response_chars=e.get('response_chars'),status_path=str(p.relative_to(ROOT))))
# Detailed input, output and packet comparison; actual token usage, not word-count estimates.
generations=[];contexts=[]
for label,base,key in [('historical',OLD,old_key),('current',NEW,new_key)]:
 for k in key:
  if k['generator_model']!='Claude Sonnet 4.6' or (label=='current' and k.get('cohort')!='fresh'):continue
  rd=Path(k['run_directory']);events=[json.loads(x) for x in read(rd/'llm_events.jsonl').splitlines() if x.strip()]
  packet=js(base/'packets'/f"{k['candidate_id']}.json")
  if k['architecture']!='FlowPilot':
   prompt=js(rd/'prompt.json');raw=js(rd/'raw_response.json');res=js(rd/'result.json')
   generations.append(dict(campaign=label,case=k['case'],architecture=k['architecture'],repeat_id=k['repeat_id'],candidate_id=k['candidate_id'],prompt_chars={t:len(prompt[t]) for t in ['system','user']},model_ids=sorted(set(e.get('model','') for e in events)),max_tokens=sorted(set(e.get('max_tokens',0) for e in events)),temperature=sorted(set(e.get('temperature',0) for e in events)),usage=[e.get('usage') for e in events],raw_response_chars=len(raw.get('text','')),finish_reason=raw.get('finish_reason'),stop_reason=raw.get('stop_reason'),prompt_path=str((rd/'prompt.json').relative_to(ROOT)),json_output_required='Return one JSON object' in prompt['user'],chemist_confirmed_input='CHEMIST-CONFIRMED' in prompt['user']))
  if k['repeat_id']=='repeat_01' and k['architecture'] in ['One-shot','Practical one-shot']:
   contexts.append(dict(campaign=label,case=k['case'],candidate_id=k['candidate_id'],packet=packet))
comparisons=[]
for case in sorted(set(c['case'] for c in contexts)):
 o=next(c['packet'] for c in contexts if c['campaign']=='historical' and c['case']==case);n=next(c['packet'] for c in contexts if c['campaign']=='current' and c['case']==case)
 comparisons.append(dict(case=case,inventory_identical=o['case_context']['authoritative_inventory']==n['case_context']['authoritative_inventory'],held_out_facts_identical=o['case_context']['held_out_source_fact_check']==n['case_context']['held_out_source_fact_check'],old_context=o['case_context'],new_context=n['case_context'],old_verification_sheet=o['deterministic_verification_sheet'],new_verification_sheet=n['deterministic_verification_sheet'],old_evaluation_boundary=o['evaluation_boundary'],new_evaluation_boundary=n['evaluation_boundary']))
rubric_old=js(OLD/'frozen/outcome_rubric.json');rubric_new=js(NEW/'frozen/outcome_rubric.json');assert rubric_old==rubric_new
for p in [OLD/'frozen/source_code/newgen_2_outcome.py',NEW/'frozen/source_code/newgen_2_outcome.py',OLD/'frozen/source_code/runner.py',NEW/'frozen/source_code/runner.py',ROOT/'ablation_test/scripts/build_all_models_three_repeat_summary.py',COMBINED/'source_provenance.json']:read(p)
cohort=grouping(old_candidates,['architecture']);sonnet_stats=grouping(raw_old_sonnet,['architecture','judge'])
v=lambda arm,j:next(r['mean'] for r in sonnet_stats if r['architecture']==arm and r['judge']==j)
cohort_os=next(x['mean'] for x in cohort if x['architecture']=='One-shot');historical_sonnet=mean(r['score'] for r in sonnet if r['architecture']=='One-shot')
new_gpt=mean(r['score'] for r in new_scores if r['architecture']=='Practical one-shot' and r['judge']=='openai')
data=dict(scope='Read-only audit; no network/API/model generation or archive edits.',exact_manuscript_models=MODELS,excluded_models_in_extended_combined_table=sorted(set(r['model'] for r in combined)-set(MODELS)),cohort=cohort,model_architecture=grouping(old_candidates,['model','architecture']),historical_sonnet_candidates=sonnet,historical_sonnet_judgments=raw_old_sonnet,historical_sonnet_per_judge=sonnet_stats,historical_sonnet_per_case_per_judge=grouping(raw_old_sonnet,['architecture','case','judge']),historical_sonnet_per_case=grouping(sonnet,['architecture','case']),historical_sonnet_per_repeat=grouping(sonnet,['architecture','repeat_id']),current_judgments=new_scores,current_per_judge=grouping(new_scores,['architecture','judge']),current_criteria=new_criteria,old_settings=old_settings,new_settings=new_settings,generation_inputs=generations,context_comparisons=comparisons,old_manifest=old_manifest,new_manifest=new_manifest,unchanged_rubric=rubric_old,new_input_tokens=js(NEW/'frozen/input_tokens.json'),decomposition=dict(manuscript_cohort_one_shot=cohort_os,historical_sonnet_three_judge=historical_sonnet,model_composition_shift=historical_sonnet-cohort_os,current_practical_primary_gpt4o=new_gpt,remaining_unidentified_cross_campaign_shift=new_gpt-historical_sonnet,same_named_qwen_historical_sonnet=v('One-shot','qwen'),same_named_qwen_current_practical=1.0,same_named_qwen_gap=1.0-v('One-shot','qwen'),historical_sonnet_qwen_claude_mean=(v('One-shot','qwen')+v('One-shot','claude'))/2,historical_sonnet_judge_architecture_deltas={j:v('FlowPilot',j)-v('One-shot',j) for j in ['qwen','openai','claude']}))
all_judge_models=collections.defaultdict(set)
for r in selected:
 campaign=Path(r['run_directory']).parents[3]
 for judge in ['qwen','openai','claude']:
  call=campaign/'judgments'/judge/r['candidate_id']
  statuses=[call/'status.json',*sorted((call/'attempts').glob('attempt_*/status.json'))]
  valid=[(p.stat().st_mtime_ns,js(p)) for p in statuses if p.exists() and js(p).get('status')=='valid']
  assert valid
  st=max(valid,key=lambda x:x[0])[1]
  all_judge_models[judge].add((st['provider'],st['model']))
data['historical_full_cohort_judge_models']={j:[{'provider':p,'model':m} for p,m in sorted(v)] for j,v in all_judge_models.items()}
data['historical_generator_ids']={}
for model in MODELS:
 r=next(r for r in selected if r['model']==model and r['architecture']=='One-shot')
 raw=js(Path(r['run_directory'])/'raw_response.json')
 data['historical_generator_ids'][model]={'provider':raw.get('provider'),'model':raw.get('model')}
for f in ['llm_agents.py','run_newgen_2_0_benchmark.py','run_manuscript_three_model_repeated_benchmark.py']:read(OLD/'frozen/source_code'/f)
data['historical_transport_provenance']={
 'requested_openai_model':'gpt-5.4-2026-03-05',
 'actual_request_api':'OpenAI SDK Chat Completions',
 'actual_token_limit':{'max_completion_tokens':7500},
 'temperature':'Runtime override 0.0 was recorded but frozen _runtime_kwargs omits temperature for GPT-5*; not sent for GPT-5.4.',
 'seed':'Recorded stable seed passed for OpenAI.',
 'reasoning_effort':'Not set in call_model_text/call_model_messages; none appears only in the separate tool path.',
 'response_format':'json_schema, strict false',
 'routing':'Configured provider openai, upstream_mode never; default SDK client. Historical environment-derived base URL not frozen.',
 'returned_model':'Not independently captured. TextGenerationResult.model and raw_response.model repeat requested model; SDK resp.model was discarded.'}
(OUT/'historical_audit.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
(OUT/'source_hashes.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
print('Cohort:',cohort);print('Sonnet:',sonnet_stats);print('Decomposition:',data['decomposition']);print('Context',[(x['case'],x['inventory_identical'],x['held_out_facts_identical']) for x in comparisons]);print('Saved',OUT)
