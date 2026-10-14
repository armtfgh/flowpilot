#!/usr/bin/env python3
"""Unscored offline baseline/candidate artifact replay; never modifies live code."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
CAMPAIGN = ROOT/'ablation_results/practical_input/claude_fair_bridge_20261014'
CANDIDATE = OUT.parent/'pump_material_repair'
CASES = ('hydrogenolysis','photochemical_oxidation','cuaac')

from flora_translate import main as production_main
from flora_translate.inventory_allocator import InventoryAllocator
from flora_translate.design_realizer import _validation_report
from flora_translate.design_disposition import apply_design_disposition_gate
from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.final_design_contract import build_final_design_contract, publish_final_design_artifacts
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory
from flora_translate.topology_semantics import normalize_topology_semantics


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def object_sha(value): return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def hashes(directory): return {str(p.relative_to(ROOT)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name not in ('paid_usage.jsonl','paid_usage.lock')}
def source_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for folder in ('flora_translate','ablation_test/src','ablation_test/scripts','ablation_test/tests','benchmark')
            for p in sorted((ROOT/folder).rglob('*.py'))}


def load_candidate(path, name):
    module=ModuleType(name)
    module.__file__=str(path)
    sys.modules[name]=module
    exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
    return module


def diff_values(old,new,path='$'):
    differences=[]
    if isinstance(old,dict) and isinstance(new,dict):
        for key in sorted(set(old)|set(new)):
            if key not in old or key not in new:
                differences.append({'path':path+'.'+key,'before_present':key in old,'after_present':key in new,'before':old.get(key),'after':new.get(key)})
            else:differences.extend(diff_values(old[key],new[key],path+'.'+key))
    elif old!=new:differences.append({'path':path,'before':old,'after':new})
    return differences


def inspect_unmodified_physics(result):
    """Independent descriptive audit; does not silently repair or override gates."""
    p=result['proposal'];liquids=[s for s in p.get('streams',[]) if s.get('phase')!='gas']
    q=sum(float(s.get('flow_rate_mL_min') or 0) for s in liquids)
    rows={'liquid_flow_sum_mL_min':q,'reported_total_liquid_flow_mL_min':p.get('flow_rate_mL_min'),
          'liquid_flow_closure':math.isclose(q,float(p['flow_rate_mL_min']),rel_tol=.005,abs_tol=1e-6),
          'top_level_concentration_M':p.get('concentration_M'),'feed_concentrations_M':[{k:s.get(k) for k in ('stream_label','concentration_M','concentration_basis')} for s in liquids],
          'remaining_findings':[]}
    if len(liquids)==1 and isinstance(liquids[0].get('concentration_M'),(int,float)) and isinstance(p.get('concentration_M'),(int,float)):
        a,b=float(p['concentration_M']),float(liquids[0]['concentration_M'])
        rows['single_feed_concentration_agrees_with_top_level']=math.isclose(a,b,rel_tol=.01,abs_tol=1e-8)
        if not rows['single_feed_concentration_agrees_with_top_level']:
            rows['remaining_findings'].append({'code':'TOPLEVEL-FEED-CONCENTRATION-DISAGREEMENT','top_level_M':a,'single_liquid_feed_M':b,
                'interpretation':'Both values are retained. The isolated pump/material patch does not reconcile this existing discrepancy.'})
    calc=result.get('design_calculations') or {}
    needed=('flow_rate_mL_min','concentration_M','target_conversion','reactor_volume_mL','space_time_yield_mol_L_h')
    if all(isinstance(calc.get(k),(int,float)) for k in needed) and calc['reactor_volume_mL']>0:
        expected=calc['flow_rate_mL_min']*calc['concentration_M']*calc['target_conversion']*60/calc['reactor_volume_mL']
        rows['conditional_STY']={'recorded':calc['space_time_yield_mol_L_h'],'expected_from_declared_calculator_basis':expected,
            'passed':math.isclose(calc['space_time_yield_mol_L_h'],expected,abs_tol=.000051),
            'conversion_basis':'unchanged assumed target conversion; not measured flow yield'}
    return rows


def replay_one(source, folder, label, topology_builder, allocator_class):
    original=read(source);public=read(source.with_name('input_public.json'))
    result=deepcopy(original)
    batch=BatchRecord.model_validate(original['batch_record'])
    plan=ChemistryPlan.model_validate(original['chemistry_plan'])
    proposal=FlowProposal.model_validate(original['proposal'])
    inventory=LabInventory.model_validate(original['inventory_snapshot'])
    realization=deepcopy(original['design_realization'])
    multistage=deepcopy(original.get('multistage_inventory_plan') or realization.get('multistage') or {})
    # Recompute normal realization predicates on the reviewed physical proposal.
    # Do not re-run realization, which may change the reviewed candidate itself.
    validation=_validation_report(proposal,inventory,realization,multistage)
    abstract=topology_builder(proposal,plan,batch,inventory)
    compiled,allocation=allocator_class(inventory,proposal).compile(abstract)
    compiled=normalize_topology_semantics(compiled)
    before_sync=proposal.model_dump()
    production_main._sync_stage_hardware_from_compiled_topology(proposal,compiled)
    assert proposal.model_dump()==before_sync, 'Replay attempted to change reviewed hardware'
    validation['checks']['topology_matches_serialized_design']=production_main._topology_matches_serialized_proposal(abstract,proposal)
    validation['checks']['inventory_topology_assignment_complete']=bool(allocation.get('checks',{}).get('all_required_operations_assigned'))
    validation['unresolved_reasons']=[key for key,value in validation['checks'].items() if not value]
    validation['status']='ready' if not validation['unresolved_reasons'] else 'screen_required'
    result['final_validation']=validation
    result['process_requirements_topology']=abstract.model_dump()
    result['inventory_allocation']=allocation
    result['instrument_manifest']=allocation.get('instrument_manifest',[])
    safety=deepcopy(original['council_safety_report_diagnostic'])
    result['safety_report']=safety
    result.pop('final_design',None)
    # This gate legitimately recomputes release status; physical choices stay fixed.
    apply_design_disposition_gate(result,proposal=proposal,final_validation=validation,inventory=inventory,
        batch_record=batch,chemistry_plan=plan,objective=public['objective'],hard_constraints=public['hard_constraints'],council_safety_report=safety)
    result['process_topology']=compiled.model_dump()
    artifacts=render_topology_artifacts(compiled,title=f'UNSCORED isolated pump/material replay — {label}',base_dir=folder/'diagrams')
    result['diagram_artifacts']=artifacts
    result['diagram_render_manifest']=read(Path(artifacts['render_manifest_path']))
    result['svg_path']=artifacts['svg_path'];result['png_path']=artifacts['png_path']
    # A failed physical check must still block the otherwise identical candidate.
    negative=deepcopy(result);negative['final_validation']['checks']['pump_flow_feasible']=False
    negative_contract=build_final_design_contract(negative)
    assert negative_contract['status']=='blocked'
    assert any(i['code']=='FINAL-PUMP_FLOW_FEASIBLE' for i in negative_contract['consistency']['issues'])
    result['final_design']=build_final_design_contract(result)
    publish_final_design_artifacts(result,result['final_design'])
    allowed_derived_fields={'recommended_disposition','disposition_rationale','engine_validated'}
    physical_before={k:v for k,v in original['proposal'].items() if k not in allowed_derived_fields}
    physical_after={k:v for k,v in result['proposal'].items() if k not in allowed_derived_fields}
    assert physical_before==physical_after, 'Physical proposal was changed'
    assert result['batch_record']==original['batch_record']
    assert result['inventory_snapshot']==original['inventory_snapshot']
    assert result['chemistry_plan']==original['chemistry_plan']
    assert result['design_realization']==original['design_realization']
    assert result['design_calculations']==original['design_calculations']
    for key in ('council_rounds','council_messages','pre_council_proposal'):
        assert result.get(key)==original.get(key), key
    result['secondary_validation_provenance']={'kind':'UNSCORED isolated deterministic artifact recompilation',
        'model_and_judge_calls':0,'source_result':str(source.relative_to(ROOT)),'source_sha256':sha(source),
        'generated_proposal_physical_values_unchanged':True,'council_records_unchanged':True,
        'source_proposal_preserved_in_archive':True,'derived_release_metadata_changes':diff_values(
            {k:original['proposal'].get(k) for k in allowed_derived_fields},{k:result['proposal'].get(k) for k in allowed_derived_fields}),
        'restored_source_fields':{'safety_report':'council_safety_report_diagnostic'},
        'recomputed_checks':'Normal _validation_report, topology-assignment checks, disposition gate, final contract and publication',
        'not_recomputed':'Generated choices, council review, physical realization assignments and engineering metrics remain archived values.',
        'benchmark_scores_changed':False}
    verification={'variant':label,'source_status':original['final_design']['status'],'replayed_status':result['final_design']['status'],
        'source_issues':original['final_design']['consistency']['issues'],'replayed_issues':result['final_design']['consistency']['issues'],
        'disposition':result['recommended_disposition'],'validation_checks':result['final_validation']['checks'],
        'normal_allocation_checks':allocation.get('checks'),'unresolved_requirements':allocation.get('unresolved_requirements'),
        'physical_proposal_unchanged':True,'physical_proposal_sha256':object_sha(physical_after),
        'chemistry_plan_unchanged':True,'inventory_unchanged':True,'council_unchanged':True,'engineering_metrics_unchanged':True,
        'negative_failed_pump_still_blocks':True,'independent_physical_audit':inspect_unmodified_physics(result),
        'recipe':result['final_design'].get('operating_procedure'),
        'topology_sha256':artifacts['topology_sha256'],'diagram_status':artifacts['render_status'],
        'pump_operations':[op.model_dump() for op in compiled.unit_operations if op.op_type=='pump']}
    write(folder/'replayed_result.json',result)
    write(folder/'verification.json',verification)
    write(folder/'changes_from_original.json',diff_values(original,result))
    return verification


def forbidden(*args,**kwargs):raise AssertionError('Network access is forbidden in offline secondary replay')


def require_complete_primary():
    """Require settled fixed attempts, retaining exhausted invalid cells as missing."""
    summary_path=OUT.parent/'format_recovery/summary_all.json'
    summary=read(summary_path)
    assert summary.get('expected_judgments')==63
    for path,digest in summary['source_hashes_before_and_after'].items():
        assert sha(Path(path))==digest, 'Primary provenance changed'
    from ablation_test.scripts import run_fair_practical_benchmark as driver
    driver.configure(CAMPAIGN)
    driver.verify(CAMPAIGN)
    recovered={(r['candidate_id'],r['judge']) for r in summary.get('recovered_cells',[])}
    settled=[]
    rows=driver.keys(CAMPAIGN,'all')
    assert len(rows)==21
    for row in rows:
        for judge in driver.JUDGES:
            accepted=driver.accepted(CAMPAIGN,judge,row)
            key=(row['candidate_id'],judge)
            if accepted:state='strict_valid'
            elif key in recovered:state='declared_na_serialization_recovery'
            else:
                folder=CAMPAIGN/'judgments'/judge/row['candidate_id']
                attempts=list(folder.rglob('status.json'))
                assert len(attempts)==3, 'Primary cell still pending/unattempted: '+str(key)
                assert all(read(p).get('status') in ('invalid','request_failed') for p in attempts)
                state='exhausted_invalid_or_failed_attempts'
            settled.append({'candidate_id':key[0],'judge':judge,'state':state})
    assert len(settled)==63
    return {'summary':str(summary_path.relative_to(ROOT)),'sha256':sha(summary_path),
            'accepted_judgments':summary['accepted_judgments'],'valid_score_panel_complete':summary['complete'],
            'settled_cells':settled,'gate_amendment_sha256':sha(OUT/'gate_amendment_20261014.json')}


def run(diagnostic_case=None):
    registration=read(OUT/'preregistration.json')
    primary=None if diagnostic_case else require_complete_primary()
    chosen=(diagnostic_case,) if diagnostic_case else CASES
    paths=[CAMPAIGN/'generation'/case/'claude_flowpilot/repeat_01/result.json' for case in chosen]
    assert all(p.exists() for p in paths), 'Not all fixed outputs are available'
    before_archives=hashes(CAMPAIGN) if not diagnostic_case else {str(p.relative_to(ROOT)):sha(p) for source in paths for p in sorted(source.parent.rglob('*')) if p.is_file()}
    ledger_before={'sha256':sha(CAMPAIGN/'paid_usage.jsonl'),'bytes':(CAMPAIGN/'paid_usage.jsonl').stat().st_size}
    code_before=source_hashes()
    candidate_paths=[CANDIDATE/name for name in ('candidate_main.py','candidate_inventory_allocator.py','preserve_feed_solvent.patch','production_main_sha256.txt','production_inventory_allocator_sha256.txt')]
    candidate_hashes={str(p.relative_to(ROOT)):sha(p) for p in candidate_paths}
    for name in ('main','inventory_allocator'):
        assert sha(ROOT/f'flora_translate/{name}.py')==(CANDIDATE/f'production_{name}_sha256.txt').read_text().strip()
    run_id=('diagnostic_' if diagnostic_case else 'all_three_')+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder=OUT/'runs'/run_id
    with patch('socket.socket.connect',forbidden),patch('socket.create_connection',forbidden):
        candidate_main=load_candidate(CANDIDATE/'candidate_main.py','isolated_secondary_main')
        candidate_allocator=load_candidate(CANDIDATE/'candidate_inventory_allocator.py','isolated_secondary_allocator')
        cases=[]
        for source in paths:
            case=source.parts[-4]
            base=replay_one(source,folder/case/'baseline','baseline',production_main._build_translate_topology,InventoryAllocator)
            patched=replay_one(source,folder/case/'candidate','isolated_candidate',candidate_main._build_translate_topology,candidate_allocator.InventoryAllocator)
            cases.append({'case':case,'baseline':base,'candidate':patched,
                'all_physical_choices_same_between_variants':base['physical_proposal_sha256']==patched['physical_proposal_sha256']})
    assert code_before==source_hashes(), 'Live source changed during replay'
    assert candidate_hashes=={str(p.relative_to(ROOT)):sha(p) for p in candidate_paths}, 'Candidate patch changed during replay'
    assert all(sha(ROOT/path)==digest for path,digest in before_archives.items()), 'Source archive file changed'
    if not diagnostic_case:assert before_archives==hashes(CAMPAIGN), 'Campaign archive membership changed'
    summary={'created_at':datetime.now(timezone.utc).isoformat(),'kind':'secondary engineering validation, UNSCORED',
        'diagnostic_only':bool(diagnostic_case),'primary_completion_evidence':primary,
        'gate_amendment_sha256':sha(OUT/'gate_amendment_20261014.json'),
        'shared_ledger_excluded_from_immutable_archives':{'reason':'Concurrent separately preregistered follow-up uses primary ledger','before':ledger_before,'after':{'sha256':sha(CAMPAIGN/'paid_usage.jsonl'),'bytes':(CAMPAIGN/'paid_usage.jsonl').stat().st_size}},'preregistration_sha256':sha(OUT/'preregistration.json'),
        'fixed_cases':list(CASES),'cases_replayed':list(chosen),'case_results':cases,'api_calls':0,'live_source_unchanged':True,
        'source_archive_files_unchanged':len(before_archives),'archive_sha256_before_and_after':before_archives,
        'live_source_sha256_before_and_after':code_before,'isolated_candidate_sha256_before_and_after':candidate_hashes,
        'helper_sha256':sha(Path(__file__)),'generated_designs_regenerated':False,'benchmark_scores_changed':False,
        'interpretation':'Normal software gate outcomes plus independent audit findings, not new scored evidence or proof of experimental success.'}
    write(folder/'summary.json',summary)
    write(folder/'artifact_manifest.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})
    write(OUT/('latest_diagnostic.json' if diagnostic_case else 'latest_all_three.json'),{'path':str(folder.relative_to(ROOT)),'summary_sha256':sha(folder/'summary.json')})
    print(json.dumps({'run':str(folder.relative_to(ROOT)),'cases':[{'case':r['case'],'baseline':r['baseline']['replayed_status'],
        'candidate':r['candidate']['replayed_status'],'findings':r['candidate']['independent_physical_audit']['remaining_findings']} for r in cases]},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--diagnostic-case',choices=CASES)
    args=p.parse_args()
    run(args.diagnostic_case)
