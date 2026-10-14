#!/usr/bin/env python3
"""Declared schema-routing control. No production edits; primary artifacts immutable."""
from __future__ import annotations
import argparse
from dataclasses import replace
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ablation_test.scripts import run_fair_practical_benchmark as primary
from ablation_test.src.fair_practical_packet import build_fair_packet
from ablation_test.src.holistic_audit import build_case_context
from ablation_test.src.newgen_2_outcome import candidate_id

PRIMARY = primary.DEFAULT
DEFAULT = Path(__file__).resolve().parent / 'campaign'
read, write, sha, now = primary.read, primary.write, primary.sha, primary.now
NATIVE_PRODUCTION = primary.runner._production_full
# Exact semantic suffix retained from the shared contract; only the JSON-envelope
# and field-enumeration instructions are routed away from intermediate agents.
SEMANTIC_START = 'Distinguish inlet/STP gas flow from\n'
assert primary.SHARED_OUTPUT.count(SEMANTIC_START) == 1
RETAINED_GUIDANCE = primary.SHARED_OUTPUT.split(SEMANTIC_START, 1)[1]
RETAINED_GUIDANCE = SEMANTIC_START + RETAINED_GUIDANCE
REMOVED_SERIALIZATION = primary.SHARED_OUTPUT.split(SEMANTIC_START, 1)[0]
INTERNAL_OBJECTIVE = primary.base.OBJECTIVE + '\n\n' + RETAINED_GUIDANCE


def sha_text(s): return hashlib.sha256(s.encode()).hexdigest()


def plan_cells():
    return [r for r in primary.campaign.campaign_cells(primary.fair_cases()) if r['architecture_key'] == 'flowpilot']


def routing_record(case):
    expected = primary.base.OBJECTIVE + '\n\n' + primary.SHARED_OUTPUT
    assert case.objective == expected, 'Only the exact preregistered harness suffix may be routed'
    return {
        'public_objective': case.objective, 'internal_objective': INTERNAL_OBJECTIVE,
        'removed_serialization': REMOVED_SERIALIZATION,
        'retained_guidance_exact': RETAINED_GUIDANCE,
        'public_objective_sha256': sha_text(case.objective),
        'internal_objective_sha256': sha_text(INTERNAL_OBJECTIVE),
        'change': 'Final-output envelope and field list remain in public context, excluded only from internal scientific objective. All semantic guidance retained verbatim.',
        'protocol_unchanged': True, 'inventory_unchanged': True, 'constraints_unchanged': True,
    }


def routed_production(case, recorder, candidate_budget, temperature, seed, execution_config=None):
    record = routing_record(case)
    recorder.write_json('schema_routing.json', record)
    internal_case = replace(case, objective=INTERNAL_OBJECTIVE)
    # execute_cell retains the unchanged public case, metadata and external scoring.
    # Only the internal native pipeline receives its scientific objective.
    return NATIVE_PRODUCTION(internal_case, recorder, candidate_budget, temperature, seed, execution_config)


def configure():
    primary.configure(PRIMARY)
    primary.runner._production_full = routed_production


def local_source_hashes():
    paths = [Path(__file__).resolve(), Path(__file__).with_name('test_followup.py')]
    return {str(p): sha(p) for p in paths}


def verify(out):
    primary.verify(PRIMARY)
    assert read(out/'frozen/source_manifest.json') == primary.source_hashes()
    assert read(out/'frozen/helper_manifest.json') == local_source_hashes()
    assert read(out/'frozen/plan.json')['cells'] == plan_cells()
    for path,digest in read(out/'frozen/input_manifest.json').items():
        assert sha(Path(path)) == digest, 'Frozen follow-up input changed: '+path
    if (out/'prepared_manifest_B.json').exists():
        for path,digest in read(out/'prepared_manifest_B.json').items():
            assert sha(Path(path)) == digest, 'Prepared artifact changed: '+path


def initialize(out):
    primary.verify(PRIMARY)
    assert not out.exists(), 'Choose a new campaign directory; never replace a run'
    frozen = out/'frozen'
    frozen.mkdir(parents=True)
    source_rows = primary.keys(PRIMARY,'B')
    assert len(source_rows) == 6
    one_shots = [r for r in source_rows if r['architecture_key']=='one_shot']
    assert len(one_shots) == 3
    write(frozen/'source_manifest.json',primary.source_hashes())
    write(frozen/'helper_manifest.json',local_source_hashes())
    shutil.copytree(PRIMARY/'frozen/historical_source',frozen/'historical_source')
    shutil.copy2(PRIMARY/'frozen/outcome_rubric.json',frozen/'outcome_rubric.json')
    shutil.copytree(PRIMARY/'frozen/held_out_oracles',frozen/'held_out_oracles')
    cases = primary.fair_cases()
    write(frozen/'case_manifest.json',{'cases':[{'case_label':label,**case.manifest_payload()} for label,case in cases]})
    write(frozen/'reused_one_shots.json',{'candidates':one_shots})
    write(frozen/'routing.json',routing_record(cases[0][1]))
    (frozen/'objective_diff.txt').write_text(''.join(difflib.unified_diff(cases[0][1].objective.splitlines(True),INTERNAL_OBJECTIVE.splitlines(True),fromfile='public_objective',tofile='internal_scientific_objective')))
    write(frozen/'plan.json',{
        'declared_at':now(),'cells':plan_cells(),'primary_campaign':str(PRIMARY),
        'design':'Fixed all-three-case schema-routing control; one new FlowPilot run per case, all three original one-shot outcomes and grades reused exactly.',
        'motivation':'All three primary translator responses used a final envelope inconsistent with the intermediate flat FlowProposal schema; original responses and failures remain in the primary campaign.',
        'intervention':'Remove only shared final serialization envelope/field enumeration from FlowPilot intermediate objectives, retaining exact gas/time/assumptions/disposition guidance. Public task/context/shared output contract unchanged.',
        'unchanged':['production source','case facts','inventory','candidate budget 2','temperature 0.2','original per-case seeds','native specialist prompts/schemas','packet adapter','original fourteen criteria','equal three historical judges'],
        'stopping':'Exactly three fresh FlowPilot generation attempts. No outcome-dependent regeneration, case removal, production repair, or grade selection. Incompleteness retained if shared cap stops execution.',
        'grading':'Reuse original one-shot judgments. At most three format attempts for each new FlowPilot judge cell, frozen historical validator and seed rule. Never retry a valid grade.',
        'format_recovery':'The already-declared score-to-null-only N/A serialization supplement may be applied separately, preserving strict results and original raw replies.',
        'judgments_new_max_valid':9,'generation_new':3,'primary_paid_ledger':str(PRIMARY/'paid_usage.jsonl'),
        'aggregate_paid_limit_usd':10,'price_and_guard':'Exact primary guard shared across simultaneous primary and follow-up calls; provider billing without returned usage remains unknown.',
        'interpretation':'Diagnostic control, not an independent holdout or a pure prompt-length experiment. No claim that corrected FlowPilot must win.',
        'pump_material_patch_applied':False,
    })
    paths = [p for p in frozen.rglob('*') if p.is_file()]
    paths += [PRIMARY/'prepared_manifest_B.json',PRIMARY/'candidate_key_B.json']
    for r in one_shots:
        paths += [PRIMARY/'packets'/f"{r['candidate_id']}.json",Path(r['run_directory'])/'result.json',Path(r['run_directory'])/'raw_response.json']
    write(frozen/'input_manifest.json',{str(p.resolve()):sha(p) for p in paths})
    verify(out)
    print(json.dumps({'initialized':str(out),'new_runs':3,'new_judgments':9,'shared_usage':primary.cost(PRIMARY)}),flush=True)


def generate(out,index):
    verify(out)
    row=plan_cells()[index]
    run=primary.campaign.run_directory(out,row)
    marker=run/'routing_attempt_started.json'
    if marker.exists() or (run/'run_summary.json').exists():
        print('Retaining existing generation attempt',index,flush=True);return
    case=next(c for _,c in primary.fair_cases() if c.case_id==row['case_id'])
    write(marker,{'at':now(),'cell':row,'routing':routing_record(case)})
    status=primary.runner.execute_cell(case=case,variant=row['variant'],bundle_name=row['condition_id'],bundle=primary.GENERATOR,run_dir=run,candidate_budget=row['candidate_budget'],temperature=.2,seed=row['seed'])
    print('GENERATION',index,row['case_label'],status.get('status'),flush=True)
    verify(out)


def prepare(out):
    verify(out)
    if (out/'prepared_manifest_B.json').exists():print('Retaining prepared packets');return
    rows=[]
    (out/'packets').mkdir(exist_ok=True)
    for original in read(out/'frozen/reused_one_shots.json')['candidates']:
        row={**original,'cohort':'schema_routing_control','reused_from_primary':True}
        shutil.copy2(PRIMARY/'packets'/f"{row['candidate_id']}.json",out/'packets'/f"{row['candidate_id']}.json")
        rows.append(row)
    for row in plan_cells():
        run=primary.campaign.run_directory(out,row)
        assert (run/'run_summary.json').exists(), 'Wait for every fixed generation attempt to finish before preparing'
        label,case=next((label,c) for label,c in primary.fair_cases() if c.case_id==row['case_id'])
        result,present,complete=primary.generation_artifact(run,row['variant'])
        context=build_case_context(case_label=label,public_input=case.public_payload(),inventory=case.inventory,oracle=read(out/'frozen/held_out_oracles'/f'{primary.campaign.safe_name(label)}.json'))
        cid=candidate_id('schema_routing_control/'+str(run.relative_to(out)))
        packet=build_fair_packet(candidate=cid,case_context=context,result=result,deterministic_verification={},has_gas=bool(case.expected_features.get('gas_required')),is_multistage='multistage' in case.category or 'multistep' in case.category)
        assert not primary.historical_module(out).packet_gate(packet)
        write(out/'packets'/f'{cid}.json',packet)
        rows.append({**row,'candidate_id':cid,'case':label,'stage':'B','cohort':'schema_routing_control','run_directory':str(run),'generation_status':'completed' if complete else 'failed_or_interrupted','delivered_result_present':present,'postprocessing_complete':complete,'reused_from_primary':False})
    write(out/'candidate_key_B.json',{'candidates':rows})
    paths=[out/'candidate_key_B.json']+list((out/'packets').glob('*.json'))+[p for p in (out/'generation').rglob('*') if p.is_file()]
    write(out/'prepared_manifest_B.json',{str(p.resolve()):sha(p) for p in paths})
    verify(out)


def judge(out,name,shard=0,shards=1):
    verify(out)
    assert (out/'prepared_manifest_B.json').exists()
    if name=='qwen':
        primary.llm._bounded_local_messages=lambda system,messages,max_tokens:primary.base.exact_local_context(system,messages,max_tokens=max_tokens,audit_path=out/f'local_context_{shard}.jsonl')
    rows=[r for r in primary.keys(out,'B') if not r['reused_from_primary']]
    for row in rows[shard::shards]:
        if primary.accepted(out,name,row):continue
        for _ in range(3):
            status=primary.call_judge(out,name,row)
            print('JUDGE',name,row['case'],status['status'],flush=True)
            if status['status']=='valid':break
            if status.get('error') and 'reservation' in status['error'].lower():break
        if status['status']!='valid':raise RuntimeError('Incomplete fixed judge cell '+row['candidate_id'])
    verify(out)


def report(out):
    verify(out)
    results=[];missing=[];accepted=[]
    rows=primary.keys(out,'B')
    assert len(rows)==6, 'Prepare all fixed outcomes first'
    for row in rows:
        source=PRIMARY if row['reused_from_primary'] else out
        for name in primary.JUDGES:
            item=primary.accepted(source,name,row)
            if not item:missing.append({'candidate_id':row['candidate_id'],'judge':name});continue
            path,response=item
            scores=[r['score'] for r in response['criterion_scores'] if r['applicability']=='APPLICABLE']
            results.append({'candidate_id':row['candidate_id'],'stage':'B','cohort':'schema_routing_control','case':row['case'],'architecture':row['architecture'],'judge':name,'score':statistics.fmean(scores)/4,'critical_flags':sum(bool(r['critical_error']) for r in response['criterion_scores']),'criteria':response['criterion_scores'],'reused_from_primary':row['reused_from_primary']})
            accepted.append({'candidate_id':row['candidate_id'],'judge':name,'status_path':str(path),'response_path':str(path.with_name('parsed_response.json')),'response_sha256':sha(path.with_name('parsed_response.json'))})
    groups={}
    for arm in ('One-shot','FlowPilot'):
        rr=[r for r in results if r['architecture']==arm]
        complete=len(rr)==9
        groups[arm]={'complete':complete,'n_judgments':len(rr),'mean':statistics.fmean(r['score'] for r in rr) if complete else None,'critical_flags':sum(r['critical_flags'] for r in rr),'by_judge':{j:{'mean':statistics.fmean(r['score'] for r in rr if r['judge']==j) if sum(r['judge']==j for r in rr)==3 else None,'n':sum(r['judge']==j for r in rr)} for j in primary.JUDGES}}
    summary={'created_at':now(),'complete':not missing,'expected_judgments':18,'accepted_judgments':len(results),'missing':missing,'groups':{'schema_routing_control':groups},'results':results,'usage':primary.cost(PRIMARY),'primary_unchanged':True,'reused_one_shot_judgments':sum(r['reused_from_primary'] for r in results)}
    write(out/'report/summary_B.json',summary);write(out/'accepted_judgments_B.json',accepted)
    print(json.dumps({k:summary[k] for k in ('complete','accepted_judgments','missing','groups','usage')},indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['init','generate','prepare','judge','report']);parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--item',default='0');parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();out=args.output.resolve();configure()
    if args.phase in ('generate','judge'):primary.install_spending_guard(PRIMARY)
    if args.phase=='init':initialize(out)
    elif args.phase=='generate':generate(out,int(args.item))
    elif args.phase=='prepare':prepare(out)
    elif args.phase=='judge':judge(out,args.item,args.shard,args.shards)
    else:report(out)

if __name__=='__main__':main()
