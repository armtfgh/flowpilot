"""Recompute supplemental scores from ORIGINAL applicable rows, not patched values."""
from pathlib import Path
from copy import deepcopy
import hashlib, importlib.util, json, statistics, collections
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'ablation_results/practical_input/claude_fair_bridge_20261014'
O=ROOT/'output/benchmark_fairness_20261014'
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 data=js(O/'independent_verification.json')
 overlay=js(O/'format_recovery/summary_all.json')
 accepted=js(O/'format_recovery/accepted_judgments_overlay.json')
 spec=importlib.util.spec_from_file_location('overlay_independent_validator',P/'frozen/historical_source/newgen_2_outcome.py')
 legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 rubric=js(P/'frozen/outcome_rubric.json');records=[]
 for a in accepted:
  packet=js(P/'packets'/f"{a['candidate_id']}.json")
  derived_path=Path(a['response_path']);derived=js(derived_path)
  assert sha(derived_path)==a['response_sha256']
  assert not legacy.validate_response(derived,rubric,packet)
  source_path=derived_path
  if a['kind']=='recovered_na_serialization':
   proof=js(Path(a['status_path']));source=Path(proof['source_attempt']);source_path=source/'parsed_response.json'
   assert sha(source_path)==proof['source_parsed_sha256']
   assert sha(source/'raw_response.json')==proof['raw_response_sha256']
   assert sha(source/'status.json')==proof['source_status_sha256']
   assert sha(P/'packets'/f"{a['candidate_id']}.json")==proof['packet_sha256']
   assert js(source/'status.json')['status']=='invalid'
   original=js(source_path);patched=deepcopy(original);diff=[]
   allowed=set()
   flags=packet['case_context']['applicability']
   if not flags['gas_process']:allowed.add('UO-08')
   if not flags['multistage_process']:allowed.add('UO-09')
   for i,r in enumerate(patched['criterion_scores']):
    if r['criterion_id'] in allowed:
     assert r['applicability']=='NOT_APPLICABLE' and r['critical_error'] is False and r['severity']=='NONE'
     if r['score'] is not None:
      assert type(r['score']) is int and 0<=r['score']<=4
      diff.append({'path':f'$.criterion_scores[{i}].score','criterion_id':r['criterion_id'],'before':r['score'],'after':None});r['score']=None
   assert diff==proof['diff'] and patched==derived
   assert [r for r in original['criterion_scores'] if r['applicability']=='APPLICABLE']==[r for r in derived['criterion_scores'] if r['applicability']=='APPLICABLE']
  else:
   assert a['kind']=='original_valid';original=derived
   assert js(Path(a['status_path']))['status']=='valid'
  applicable=[r for r in original['criterion_scores'] if r['applicability']=='APPLICABLE']
  score=sum(r['score'] for r in applicable)/(4*len(applicable));flags=sum(r['critical_error'] is True for r in original['criterion_scores'])
  official=next(r for r in overlay['results'] if r['candidate_id']==a['candidate_id'] and r['judge']==a['judge'])
  assert abs(score-official['score'])<1e-12 and flags==official['critical_flags']
  records.append(dict(candidate_id=a['candidate_id'],judge=a['judge'],stage=a['stage'],case=a['case'],architecture=a['architecture'],cohort=official['cohort'],acceptance_kind=a['kind'],score_from_original_applicable_rows=score,critical_flags=flags,original_response_path=str(source_path),original_response_sha256=sha(source_path)))
 assert len({(r['candidate_id'],r['judge']) for r in records})==len(records)
 groups=collections.defaultdict(list)
 for r in records:groups[(r['cohort'],r['architecture'])].append(r)
 summaries=[]
 for (cohort,arm),rr in sorted(groups.items()):
  complete=len(rr)==9 and len({(r['case'],r['judge']) for r in rr})==9
  score=statistics.mean(r['score_from_original_applicable_rows'] for r in rr) if complete else None
  official=overlay['groups'][cohort][arm]
  assert official['complete']==complete
  if complete:assert abs(score-official['mean'])<1e-12
  assert sum(r['critical_flags'] for r in rr)==official['critical_flags']
  summaries.append(dict(cohort=cohort,architecture=arm,complete=complete,mean=score,critical_flags=official['critical_flags']))
 strict=sum(r['acceptance_kind']=='original_valid' for r in records);rec=len(records)-strict
 assert strict==overlay['strict_valid_judgments'] and rec==len(overlay['recovered_cells'])
 primary=[r for r in records if r['stage']=='B']
 primary_strict=len(primary)==18 and all(r['acceptance_kind']=='original_valid' for r in primary)
 data['serialization_supplement_audit']={'records':records,'groups':summaries,'total_resolved':len(records),'strict_valid':strict,'serialization_recovered':rec,'scores_recomputed_from_original_applicable_rows':True,'complete':len(records)==63 and overlay['complete'],'strict_fresh_stage_B_complete':primary_strict,'source_records_unmodified':True,'api_calls':0}
 (O/'independent_verification.json').write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps({k:v for k,v in data['serialization_supplement_audit'].items() if k!='records'},indent=2))
if __name__=='__main__':main()
