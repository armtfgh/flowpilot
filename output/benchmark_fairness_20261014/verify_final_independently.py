"""Read-only raw-judgment and immutable-file audit. No APIs; no benchmark edits."""
from pathlib import Path
import collections, hashlib, importlib.util, json, statistics
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'independent_verification.json'
P=ROOT/'ablation_results/practical_input/claude_fair_bridge_20261014'
JUDGES=('qwen','openai','claude')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 report=js(OUT) if OUT.exists() else {}
 checks={}
 for name in ('input_manifest.json','full_source_manifest.json','archive_manifest.json'):
  h=js(P/'frozen'/name)
  for relative,digest in h.items():
   q=P/relative if name=='input_manifest.json' else ROOT/relative
   assert sha(q)==digest, str(q)
   if name=='full_source_manifest.json':assert sha(P/'frozen/full_source'/relative)==digest
  checks[name]={'files_checked':len(h),'all_match':True}
 for p in P.glob('prepared_manifest_*.json'):
  h=js(p)
  for rel,digest in h.items():assert sha(P/rel)==digest,rel
  checks[p.name]={'files_checked':len(h),'all_match':True}
 report['final_hash_checks']=checks
 path=P/'frozen/historical_source/newgen_2_outcome.py'
 spec=importlib.util.spec_from_file_location('independent_frozen_validator',path)
 legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 rubric=js(P/'frozen/outcome_rubric.json')
 expected_ids={r['criterion_id'] for r in rubric['criteria']}
 keys=[]
 for stage in ('A','B'):
  path=P/f'candidate_key_{stage}.json'
  if path.exists():keys+=js(path)['candidates']
 assert len({r['candidate_id'] for r in keys})==len(keys)
 scores=[];missing=[];accepted=[];attempts=[]
 for row in keys:
  cid=row['candidate_id'];packet=js(P/'packets'/f'{cid}.json')
  if row['stage']=='A':
   assert (ROOT/row['source_packet']).read_bytes()==(P/'packets'/f'{cid}.json').read_bytes()
  for judge in JUDGES:
   candidates=[]
   folder=P/'judgments'/judge/cid
   for statuspath in sorted(folder.rglob('status.json')):
    status=js(statuspath);attempts.append(dict(candidate_id=cid,judge=judge,status=status['status'],validation_errors=status.get('validation_errors',[]),path=str(statuspath.relative_to(P))))
    if status['status']!='valid':continue
    response=js(statuspath.with_name('parsed_response.json'))
    errors=legacy.validate_response(response,rubric,packet);assert not errors,(cid,judge,errors)
    candidates.append((statuspath,response))
   assert len(candidates)<=1,(cid,judge,'Multiple valid grades')
   if not candidates:
    cell_attempts=[a for a in attempts if a['candidate_id']==cid and a['judge']==judge]
    missing.append(dict(candidate_id=cid,judge=judge,attempt_count=len(cell_attempts),attempt_budget_exhausted=len(cell_attempts)>=3));continue
   statuspath,response=candidates[0];criteria=response['criterion_scores']
   assert len(criteria)==14 and {r['criterion_id'] for r in criteria}==expected_ids
   applicable=[r for r in criteria if r['applicability']=='APPLICABLE']
   assert all(isinstance(r['score'],int) and not isinstance(r['score'],bool) and 0<=r['score']<=4 for r in applicable)
   expected_count=12+int(packet['case_context']['applicability']['gas_process'])+int(packet['case_context']['applicability']['multistage_process'])
   assert len(applicable)==expected_count
   scores.append(dict(candidate_id=cid,stage=row['stage'],cohort=row['cohort'],case=row['case'],architecture=row['architecture'],judge=judge,score=sum(r['score'] for r in applicable)/(4*len(applicable)),critical_flags=sum(r['critical_error'] is True for r in criteria),applicable_count=len(applicable)))
   accepted.append(dict(candidate_id=cid,judge=judge,status_path=str(statuspath.relative_to(P)),parsed_response_sha256=sha(statuspath.with_name('parsed_response.json'))))
 groups=collections.defaultdict(list)
 for r in scores:groups[(r['cohort'],r['architecture'])].append(r)
 summary=[]
 for (cohort,arm),rr in sorted(groups.items()):
  complete=len(rr)==9 and len({(x['case'],x['judge']) for x in rr})==9
  summary.append(dict(cohort=cohort,architecture=arm,complete=complete,n_judgments=len(rr),mean=statistics.mean(r['score'] for r in rr) if complete else None,critical_flags=sum(r['critical_flags'] for r in rr),by_judge={j:statistics.mean(r['score'] for r in rr if r['judge']==j) if sum(r['judge']==j for r in rr)==3 else None for j in JUDGES},by_case={c:statistics.mean(r['score'] for r in rr if r['case']==c) if sum(r['case']==c for r in rr)==3 else None for c in sorted({r['case'] for r in rr})}))
 official=P/'report/summary_all.json';compared=False
 if official.exists():
  d=js(official);assert d['accepted_judgments']==len(scores)
  for r in scores:
   s=next(x for x in d['results'] if x['candidate_id']==r['candidate_id'] and x['judge']==r['judge'])
   assert abs(r['score']-s['score'])<1e-12 and r['critical_flags']==s['critical_flags']
  for g in summary:
   v=d['groups'][g['cohort']][g['architecture']]
   assert v['complete']==g['complete'] and v['critical_flags']==g['critical_flags']
   if g['complete']:assert abs(v['mean']-g['mean'])<1e-12
  compared=True
 complete=len(keys)==21 and len(scores)==63 and not missing
 report.update(status='complete_raw_judgment_and_integrity_verification_passed' if complete and compared else 'partial_read_only_verification',candidate_count=len(keys),accepted_judgments=len(scores),expected_final_judgments=63,missing=missing,score_groups=summary,raw_scores=scores,accepted_response_hashes=accepted,attempts=attempts,official_summary_matches=compared,complete=complete,api_calls=0,read_only=True)
 report['strict_fresh_stage_B_complete']=sum(r['stage']=='B' for r in scores)==18
 report['all_planned_cells_resolved_or_exhausted']=len(keys)==21 and all(r['attempt_budget_exhausted'] for r in missing)
 report['pending']=None if complete and compared else 'Wait for all21 fixed outcomes,63 accepted grades and official summary_all.json; missing rows are not omitted from target.'
 OUT.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ('status','candidate_count','accepted_judgments','complete','official_summary_matches','score_groups')},indent=2))
if __name__=='__main__':verify()
