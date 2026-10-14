#!/usr/bin/env python3
"""Frozen three-arm practical-input repair experiment, with archived-output re-evaluation."""
from __future__ import annotations
import argparse, copy, hashlib, json, random, shutil, statistics, subprocess, sys
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ablation_test.scripts import run_practical_input_pilot as base
from ablation_test.src.newgen_2_outcome import candidate_id, packet_gate, _scrub, validate_response
from ablation_test.src.providers import endpoint_health
campaign=base.campaign;judging=base.judging;runner=base.runner;llm=base.llm
read=base.read;write=base.write;sha=base.sha;now=base.now
DEFAULT=ROOT/'ablation_results/practical_input/claude_practical_repair_20261014'
PREVIOUS=ROOT/'ablation_results/practical_input/claude_practical_pilot_20261014'
GUIDANCE_PATH=ROOT/'output/practical_benchmark_tools/audit_20261014/guidance_enrichment.txt'
JUDGE_ADDENDUM_PATH=ROOT/'output/practical_benchmark_tools/audit_20261014/evaluator_instruction_suggestion.txt'
ORIGINAL_JUDGE_PROMPT=judging.judge_prompt
ARMS=('Practical one-shot','Guided one-shot','FlowPilot')
GUIDANCE='';JUDGE_ADDENDUM=''
FALLBACK_POLICY=('If delivery_status is blocked, chemistry plans, proposal parameters/streams and pre/post-reactor steps retained in this packet are provisional supporting artifacts. They do not establish a released executable topology or operating procedure. Judge their content where applicable and score withheld final delivery explicitly; do not substitute an internal apparatus. A software issue label is a claim to verify against the raw chemistry, not independent chemical evidence.')

def one_shot(case,model,provider,recorder,temperature,seed):
    guided='guided_one_shot' in recorder.metadata['bundle_name']
    user=case.design_input_text+('\n\nDESIGN CHECKLIST\n'+GUIDANCE if guided else '')
    recorder.write_json('prompt.json',{'system':base.SYSTEM,'user':user,'format':'ordinary prose; no required JSON schema',
        'guidance_enriched':guided,'prompt_sha256':hashlib.sha256(user.encode()).hexdigest()})
    llm.set_llm_observer(recorder.observe_llm)
    llm.set_llm_runtime_overrides(temperature=temperature,seed=seed,capture_content=True)
    try:
        r=llm.call_model_text(model=model,provider=provider,system=base.SYSTEM,user_content=user,max_tokens=8192,api_name='practical_input_one_shot')
        recorder.write_json('raw_response.json',asdict(r))
        if not r.text.strip(): raise RuntimeError('Empty one-shot response')
        return {'variant':'general_one_shot','proposal':{'chemistry_notes':r.text},'raw_proposal':{'chemistry_notes':r.text},
            'schema_valid':True,'output_format':'prose','output_validity':'prose_delivered','raw_delivered_text':r.text,'finish_reason':r.finish_reason}
    finally: llm.clear_llm_observer();llm.clear_llm_runtime_overrides()

def judge_prompt(rubric,packet):
    system,user=ORIGINAL_JUDGE_PROMPT(rubric,packet)
    return system,user+'\n\nCONTENT AND NUMERICAL VERIFICATION\n'+JUDGE_ADDENDUM

def packet(**kwargs):
    p=base.practical_packet(**kwargs)
    p['evaluation_boundary']+=' '+FALLBACK_POLICY
    return p

def configure(output):
    global GUIDANCE,JUDGE_ADDENDUM
    base.configure()
    frozen=output/'frozen'
    GUIDANCE=(frozen/'guidance.txt' if (frozen/'guidance.txt').exists() else GUIDANCE_PATH).read_text().strip()
    JUDGE_ADDENDUM=(frozen/'judge_addendum.txt' if (frozen/'judge_addendum.txt').exists() else JUDGE_ADDENDUM_PATH).read_text().strip()
    campaign.ARCHITECTURES={'practical_one_shot':('general_one_shot',ARMS[0]),'guided_one_shot':('general_one_shot',ARMS[1]),'repaired_flowpilot':('full',ARMS[2])}
    campaign.CANDIDATE_BUDGETS={'practical_one_shot':1,'guided_one_shot':1,'repaired_flowpilot':2}
    campaign.SEED_NAMESPACE='practical-repair-claude-20261014-v1'
    campaign.build_outcome_packet=packet
    runner._general_one_shot=one_shot
    judging.judge_prompt=judge_prompt
    if (frozen/'held_out_oracles').exists():
        campaign.ORACLES={label:frozen/'held_out_oracles'/f'{campaign.safe_name(label)}.json' for label in campaign.SELECTED_CASES}

def verify(output):
    base.verify_source(output)
    cases=base.practical_cases()
    assert read(output/'frozen/case_manifest.json')['cases']==json.loads(json.dumps([
        {'case_label':label,**case.manifest_payload()} for label,case in cases])), 'Live case facts differ from frozen cases'
    assert read(output/'frozen/execution_plan.json')['cells']==campaign.campaign_cells(cases), 'Execution plan changed'
    for relative,digest in read(output/'frozen/previous_source_hashes.json').items():
        assert sha(PREVIOUS/relative)==digest,relative
    prepared=output/'prepared_manifest.json'
    if prepared.exists():
        for relative,digest in read(prepared).items():
            assert sha(output/relative)==digest,relative

def initialize(output):
    if output.exists():raise RuntimeError('Use a new campaign directory')
    cases=base.practical_cases();campaign.initialize(output,cases);frozen=output/'frozen'
    write(frozen/'previous_source_hashes.json',{})
    (frozen/'guidance.txt').write_text(GUIDANCE+'\n');(frozen/'judge_addendum.txt').write_text(JUDGE_ADDENDUM+'\n')
    source_hashes={}
    for path in base.source_files():
        relative=path.relative_to(ROOT);dest=frozen/'full_source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);source_hashes[str(relative)]=sha(path)
    write(frozen/'full_source_manifest.json',source_hashes)
    token_rows=[]
    for label,case in cases:
        folder=frozen/'practical_inputs'/campaign.safe_name(label);write(folder/'public_input.json',case.public_payload())
        for arm,text in [('practical',case.design_input_text),('guided',case.design_input_text+'\n\nDESIGN CHECKLIST\n'+GUIDANCE)]:
            (folder/f'{arm}_request.txt').write_text(text)
            count=llm._get_anthropic_client().messages.count_tokens(model='claude-sonnet-4-6',system=base.SYSTEM,messages=[{'role':'user','content':text}]).input_tokens
            token_rows.append({'case':label,'input_variant':arm,'input_tokens':count,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'count_method':'Anthropic messages.count_tokens; complete system plus user input'})
    write(frozen/'input_tokens.json',token_rows)
    previous_hashes={name:sha(PREVIOUS/name) for name in ('frozen/candidate_key_confidential.json','frozen/campaign_manifest.json')};prior=[]
    for c in read(PREVIOUS/'frozen/candidate_key_confidential.json')['candidates']:
        cid=candidate_id('archived_practical_v1/'+c['candidate_id'])
        folder=output/'archived_inputs'/cid;folder.mkdir(parents=True)
        source=Path(c['run_directory'])
        for name in ['result.json','run_summary.json','input_public.json','metadata.json','llm_events.jsonl','prompt.json']:
            path=source/name
            if path.exists():shutil.copy2(path,folder/name);previous_hashes[str(path.relative_to(PREVIOUS))]=sha(path)
        oldpacket=PREVIOUS/'packets'/f"{c['candidate_id']}.json";previous_hashes[str(oldpacket.relative_to(PREVIOUS))]=sha(oldpacket)
        old=read(oldpacket);old['candidate_id']=cid;old['evaluation_boundary']+=' '+FALLBACK_POLICY
        write(folder/'packet.json',old)
        entry=copy.deepcopy(c);entry.update(candidate_id=cid,original_candidate_id=c['candidate_id'],original_run_directory=c['run_directory'],
            run_directory=str(folder),architecture=ARMS[0] if c['architecture']=='One-shot' else ARMS[2],cohort='archived',result_sha256=sha(folder/'result.json'))
        prior.append(entry)
    write(frozen/'previous_source_hashes.json',previous_hashes);write(frozen/'archived_candidate_key.json',{'candidates':prior})
    manifest=read(frozen/'campaign_manifest.json')
    manifest.update(benchmark_name='Practical-input repair and guidance-control development experiment',
        generation_outcomes=9,archived_outcomes_reevaluated=6,planned_judgments=30,primary_evaluator='openai',secondary_evaluator='qwen',
        primary_comparison='Fresh FlowPilot versus fresh practical one-shot; equal-case mean applicable criterion score / 4.',
        secondary_comparisons=['Fresh guided versus fresh practical one-shot with identical factual inputs and prose format.','Archived versus fresh FlowPilot under the same revised evaluation protocol.'],
        evaluation_version='practical_content_verification_v2',historical_scores_not_pooled=True,
        compute_matched=False,development_cases=True,independent_holdout=False,interactive_clarification=False,
        guidance_adds_facts=False,chemistry_confirmation_invented=False,
        seed_note='Claude does not accept a seed; seeds are bookkeeping only.',
        retry_policy='One generation per cell. Native pipeline revisions and transport retries are retained. Up to five judge attempts ONLY for invalid format/consistency or transport failure; all attempts retained. Never regenerate a valid assessment.',
        stopping_rule='Nine fresh generation attempts, six immutable archived outcomes, thirty accepted assessments if infrastructure permits. No score-based retry, exclusion or extension.',
        spending_guard_usd=base.SPEND_LIMIT,pricing=read(PREVIOUS/'frozen/campaign_manifest.json')['pricing'],
        configuration_scope='Existing two-initial-candidate legacy production path, source-excluded lexical retrieval. Not the twelve-candidate GUI or Astra scientific council configuration.',
        packet_policy='Full prose retained, no automated numerical worksheet for either architecture, blocked fallback provenance explicit. Fixed14criterion rubric unchanged; revised symmetric verification instructions and strict GPT JSON formatting.')
    write(frozen/'campaign_manifest.json',manifest)
    (frozen/'PREREGISTRATION.txt').write_text(
        'Practical-input repair experiment\nFrozen before generation '+now()+'\n\n'
        'Nine fresh Claude Sonnet4.6 outputs: three cases by practical one-shot, guidance-enriched one-shot and practical FlowPilot. Same factual protocol, objective, inventory; same prose output allowance in both one-shot arms. Guidance adds general checks, no new chemical facts or answers. FlowPilot receives no fabricated chemist confirmation. Two initial candidates, unchanged native repair budgets.\n\n'
        'Six prior outputs are copied unchanged and re-evaluated using the same revised evaluation as all new outputs. Original judgments and results remain untouched. The primary endpoint is fresh FlowPilot minus fresh practical one-shot. Guidance effect and archived-versus-fresh differences are secondary exploratory comparisons. One generation per cell cannot establish repeatability or isolate token count from guidance content. No independent new chemistry holdout.\n\n'
        'Fourteen fixed criteria/applicability/weights. GPT-4o primary and local Qwen secondary, reported separately. The revised judge instruction requires numerical verification, source-checking software flags and content-neutral handling of prose/uncertainty. No scoring criteria change. Strict JSON is enabled from the start for GPT-4o. Up to five attempts only for invalid format or transport; never repeat a valid grade. Preserve all raw outputs and attempts, even unfavorable outcomes. Mixed archived/fresh packets are evaluated in a deterministic shuffled order without cohort labels in their content.\n\n'
        'No generation retry, score-dependent code repair, adaptive extra repeats or favorable-outcome selection. Existing source records excluded from retrieval. Source and prompts are hashed. Shared paid-use guard $10, returned-usage cost estimates; local hardware cost not estimated. Stop if infrastructure or usage prevents completion and report incompleteness honestly.\n')
    (frozen/'PREREGISTRATION.md').write_text((frozen/'PREREGISTRATION.txt').read_text())
    paths=[p for p in frozen.rglob('*') if p.is_file() and 'full_source' not in p.parts and p.name!='input_manifest.json']
    paths+=list((output/'archived_inputs').rglob('*'))
    write(frozen/'input_manifest.json',{str(p.relative_to(output)):sha(p) for p in paths if p.is_file()})
    write(output/'endpoint_health.json',{name:endpoint_health(judging.JUDGES[name]) for name in ['claude','openai','qwen']})
    write(output/'source_git_state.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)})
    print(json.dumps({'initialized':str(output),'generation_outcomes':9,'total_evaluated_outcomes':15,'tokens':token_rows}),flush=True)

def generate(output,index):
    verify(output);cases=base.practical_cases();row=campaign.campaign_cells(cases)[index];run=campaign.run_directory(output,row)
    if (run/'run_summary.json').exists():print('Retained existing attempt',index,flush=True);return
    case=next(c for _,c in cases if c.case_id==row['case_id'])
    campaign.append_jsonl(output/'campaign_events.jsonl',{'event':'generation_started','at':now(),'cell':row})
    status=runner.execute_cell(case=case,variant=row['variant'],bundle_name=row['condition_id'],bundle=campaign.MODELS['claude'],
        run_dir=run,candidate_budget=row['candidate_budget'],temperature=.2,seed=row['seed'])
    campaign.append_jsonl(output/'campaign_events.jsonl',{'event':'generation_finished','at':now(),'cell':row,'status':status.get('status')})
    print('GENERATION',index,row['case_label'],row['architecture'],status.get('status'),flush=True);verify(output)

def prepare(output):
    verify(output)
    if (output/'prepared_manifest.json').exists():raise RuntimeError('Packets already frozen')
    if any((output/'judgments').rglob('status.json')):raise RuntimeError('Already judged')
    for row in campaign.campaign_cells(base.practical_cases()):
        assert (campaign.run_directory(output,row)/'run_summary.json').exists()
    campaign.build_packets(output,base.practical_cases())
    key=read(output/'frozen/candidate_key_confidential.json')['candidates']
    for c in key:c['cohort']='fresh'
    for c in read(output/'frozen/archived_candidate_key.json')['candidates']:
        key.append(c);write(output/'packets'/f"{c['candidate_id']}.json",read(Path(c['run_directory'])/'packet.json'))
    random.Random(2026101401).shuffle(key)
    write(output/'frozen/candidate_key_confidential.json',{'candidates':key})
    audit=[]
    for c in key:
        p=output/'packets'/f"{c['candidate_id']}.json";d=read(p);assert not packet_gate(d)
        if 'one-shot' in c['architecture'].lower() and c['generation_status']=='completed':
            original=read(Path(c['run_directory'])/'result.json')['raw_delivered_text']
            assert d['delivered_final_outcome']['chemistry_and_operations']['chemistry_notes']==_scrub(original)
        audit.append({'candidate_id':c['candidate_id'],'sha256':sha(p),'bytes':p.stat().st_size})
    write(output/'packet_integrity.json',audit)
    immutable=[output/'frozen/candidate_key_confidential.json',output/'frozen/packet_metrics.json',output/'packet_integrity.json']
    for folder in ('generation','packets'):
        immutable.extend(p for p in (output/folder).rglob('*') if p.is_file())
    write(output/'prepared_manifest.json',{str(p.relative_to(output)):sha(p) for p in immutable})
    verify(output);print('Prepared 15 packets',flush=True)

def install_strict_output(output):
    resource=llm._get_openai_client().chat.completions;original=resource.create
    def strict(**kwargs):
        kwargs=copy.deepcopy(kwargs)
        if (kwargs.get('response_format') or {}).get('type')=='json_schema':
            kwargs['response_format']['json_schema']['strict']=True
            campaign.append_jsonl(output/'strict_transport_events.jsonl',{'at':now(),'model':kwargs['model'],'response_format':kwargs['response_format']})
        return original(**kwargs)
    resource.create=strict

def judge(output,name,shard=0,shards=1):
    verify(output)
    if name=='qwen':llm._bounded_local_messages=lambda system,messages,max_tokens:base.exact_local_context(system,messages,max_tokens=max_tokens,audit_path=output/f'judge_context_checks_{shard}.jsonl')
    rubric=read(output/'frozen/outcome_rubric.json')
    key=read(output/'frozen/candidate_key_confidential.json')['candidates'][shard::shards]
    for c in key:
        if campaign.valid_judgment(output,name,c['candidate_id']):continue
        for attempt in range(5):
            status=judging.call_judge(output,name,c,rubric)
            print('JUDGE',name,c['cohort'],c['case'],c['architecture'],status['status'],flush=True)
            if status['status']=='valid':break
        if status['status']!='valid':raise RuntimeError('Incomplete assessment '+c['candidate_id'])
    verify(output)

def report(output):
    verify(output);rubric=read(output/'frozen/outcome_rubric.json');rows=[];accepted=[]
    key=read(output/'frozen/candidate_key_confidential.json')['candidates']
    for c in key:
        p=read(output/'packets'/f"{c['candidate_id']}.json")
        for name in campaign.JUDGES:
            r=base.resolved_response(output,name,c['candidate_id'],rubric,p)
            if r is None:raise RuntimeError('Missing accepted judgment')
            assert not validate_response(r,rubric,p)
            scores=[x['score'] for x in r['criterion_scores'] if x['applicability']=='APPLICABLE']
            rows.append({'candidate_id':c['candidate_id'],'cohort':c['cohort'],'case':c['case'],'architecture':c['architecture'],
                'judge':name,'score':statistics.fmean(scores)/4,'critical_flags':sum(bool(x['critical_error']) for x in r['criterion_scores'])})
            for sp in (output/'judgments'/name/c['candidate_id']).rglob('status.json'):
                if read(sp)['status']=='valid':accepted.append({'candidate_id':c['candidate_id'],'judge':name,'status_path':str(sp.relative_to(output)),'response_sha256':sha(sp.with_name('parsed_response.json'))})
    assert len(rows)==30 and len(accepted)==30
    groups={}
    for name in campaign.JUDGES:
        groups[name]={}
        for cohort in ['fresh','archived']:
            groups[name][cohort]={}
            for arm in ARMS:
                rr=[r for r in rows if r['judge']==name and r['cohort']==cohort and r['architecture']==arm]
                if rr:groups[name][cohort][arm]={'mean':statistics.fmean(r['score'] for r in rr),'critical_flags':sum(r['critical_flags'] for r in rr),'n':len(rr)}
    summary={'completed_at':now(),'generation_outcomes':9,'archived_outcomes_reevaluated':6,'accepted_judgments':30,'primary_judge':'openai',
        'evaluation_version':'practical_content_verification_v2','results':rows,'groups':groups,'usage':base.cost(output)}
    write(output/'report/summary.json',summary);write(output/'accepted_judgments.json',accepted)
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--phase',choices=['init','generate','prepare','judge','report'],required=True)
    p.add_argument('--item',default='0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();out=a.output.resolve();configure(out)
    if a.phase in ['generate','judge']:base.install_spending_guard(out)
    if a.phase=='init':initialize(out)
    elif a.phase=='generate':generate(out,int(a.item))
    elif a.phase=='prepare':prepare(out)
    elif a.phase=='judge':install_strict_output(out);judge(out,a.item,a.shard,a.shards)
    else:report(out)
if __name__=='__main__':main()
