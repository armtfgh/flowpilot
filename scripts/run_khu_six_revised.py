"""Six exact revised KHU inputs; append-only attempts and complete LLM logs."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from contextlib import ExitStack

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def save(path, data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def cases():
    src=json.loads((ROOT/'inventory_khu/source_review_20260914/source_extraction.json').read_text())
    slides={s['slide']:s for s in src['slides']}
    out=[]
    for fig,protocol_slide,sets in [(5,1,[5,8,11]),(6,12,[16,19,22])]:
        protocol=next(t for t in slides[protocol_slide]['text'] if 'stir bar' in t)
        if fig==5:
            # Both steps are in the same text shape in the supplied PPT.
            assert 'Step 2' in protocol
        for number,slide in enumerate(sets,1):
            answers={}
            for text in slides[slide]['text']:
                m=re.search(r'Respones to (Q-[A-Z]+-\d+): Please enter here\s*(?:\u2014>\s*)?(.*)',text,re.S)
                if m: answers[m[1]]=m[2].strip()
            assert 'Q-OBJ-001' in answers and 'Q-HYP-001' in answers
            out.append(dict(id=f'figure{fig}_set{number}',figure=fig,set=number,
                            protocol=protocol,answers=answers,source_slide=slide))
    return out


def package_for(case,profile):
    from flora_translate.intake_agent import IntakeAgent
    from flora_translate.inventory_resolution import bind_inventory
    answers=[dict(question_id=k,answer=v,source=f'KHU revised PPT slide {case["source_slide"]}') for k,v in case['answers'].items()]
    answers += [dict(question_id='Q-HIST-001',status='unavailable',source='No measured historical yields supplied in revised sets'),
                dict(question_id='Q-PREF-001',answer='Report stage conditions, each stream component concentration and molar flow, assigned equipment and inlet/STP gas flow. Do not predict measured yield.',source='User requested reporting')]
    p=IntakeAgent().analyze(case['protocol'],answers=answers,use_llm=False)
    return bind_inventory(p,profile)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/khu_revised_six_20260914')
    parser.add_argument('--case')
    parser.add_argument('--attempt',default='attempt_01')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--reuse-generation-from',type=Path)
    parser.add_argument('--reuse-council-from',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    from flora_translate.inventory_profiles import load_inventory_profile
    from flora_translate.intake_agent import intake_context_block
    from flora_translate.engine.llm_agents import set_llm_observer,set_llm_runtime_overrides
    from flora_translate.main import translate
    from flora_translate.gui_autosave import autosave_gui_result
    profile_path=ROOT/'inventory_khu/KHU_inventory_20260914_v5.json'
    profile=load_inventory_profile(profile_path)
    runtime=dict(design_policy='scientific_v2',candidate_budget=12,
                 upstream_model='claude-opus-4-6',downstream_model='claude-sonnet-4-6')
    manifest=dict(runtime=runtime,inventory_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                  history='Previous FlowPilot outputs are NOT measured evidence.',
                  rerun_policy='Keep every attempt. Diagnose technical failures before rerunning; no selection by favorable predicted yield.')
    manifest_path=args.output/'manifest.json'
    if not manifest_path.exists(): save(manifest_path,manifest)
    logging.basicConfig(level=logging.INFO)
    set_llm_runtime_overrides(capture_content=True)
    for case in cases():
        if args.case and args.case!=case['id']: continue
        p=package_for(case,profile)
        print(case['id'],'READY',p.ready_for_design,'MISSING',p.missing_question_ids,flush=True)
        if args.prepare_only: continue
        folder=args.output/case['id']/args.attempt
        folder.mkdir(parents=True,exist_ok=False)
        save(folder/'provided_input.json',case)
        save(folder/'inventory_profile.json',profile.model_dump())
        save(folder/'intake_package.json',p.model_dump())
        save(folder/'request.json',dict(intake_package=p.model_dump(),runtime_options=runtime))
        (folder/'prompt.txt').write_text(intake_context_block(p),encoding='utf-8')
        snapshot=folder/'source_snapshot'
        snapshot.mkdir()
        hashes={}
        for path in [*ROOT.glob('flora_translate/*.py'),ROOT/'flora_translate/engine/council_v4/scientific.py',Path(__file__)]:
            target=snapshot/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        save(folder/'source_hashes.json',hashes)
        handler=logging.FileHandler(folder/'run.log')
        logging.getLogger().addHandler(handler)
        def observe(event):
            with (folder/'llm_calls.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(event,default=str,ensure_ascii=False)+'\n')
        set_llm_observer(observe)
        started=time.monotonic()
        row=dict(case=case['id'],status='running')
        save(folder/'summary.json',row)
        before={x for base in ['scientific_pipeline','scientific_council'] for x in (ROOT/'outputs'/base).glob('*')}
        try:
            if not p.ready_for_design: raise ValueError(f'Intake incomplete: {p.missing_question_ids}')
            with ExitStack() as stack:
                if args.reuse_generation_from:
                    from unittest.mock import patch
                    from flora_translate.schemas import ChemistryPlan,FlowProposal,LabInventory
                    old=json.loads((args.reuse_generation_from/'result.json').read_text())
                    old_input=json.loads((args.reuse_generation_from/'provided_input.json').read_text())
                    assert old_input==case, 'Cannot reuse generation for changed chemist inputs'
                    old_inventory=LabInventory.model_validate(old['inventory_snapshot'])
                    assert old_inventory.model_dump(exclude={'schema_version'})==profile.lab_inventory.model_dump(exclude={'schema_version'}), 'Physical inventory changed'
                    upstream=next(args.reuse_generation_from.glob('model_artifacts/scientific_pipeline/*/upstream.json'))
                    up=json.loads(upstream.read_text())
                    stack.enter_context(patch('flora_translate.main.analyze_batch_chemistry',return_value=ChemistryPlan.model_validate(up['chemistry_plan'])))
                    stack.enter_context(patch('flora_translate.main.VectorRetriever.retrieve',return_value=old['_analogies']))
                    stack.enter_context(patch('flora_translate.main.AnalogySelector.select',return_value=old['_analogies']))
                    stack.enter_context(patch('flora_translate.main.TranslationLLM.generate',return_value=FlowProposal.model_validate(old['engineering_history']['before_council']['proposal'])))
                    save(folder/'reused_generation_provenance.json',dict(source=str(args.reuse_generation_from),
                        reused='Upstream chemistry, retrieval and initial downstream proposal; same chemist inputs and physical inventory.',
                        fresh='Deterministic realization, 12 candidates, all six council calls, final gates and rendering.'))
                if args.reuse_council_from:
                    from unittest.mock import patch
                    from flora_translate.engine import llm_agents
                    from flora_translate.engine.council_v4.scientific import _json_response
                    assert args.reuse_generation_from, 'Council reuse requires a verified generation source'
                    old_audit=next(args.reuse_council_from.glob('model_artifacts/scientific_council/*/audit.json'))
                    cached={}
                    for entry in json.loads(old_audit.read_text())['calls']:
                        try:_json_response(entry['raw_response'])
                        except ValueError:continue
                        if entry['role'].startswith('Dr'):cached[entry['role']]=entry
                    original_call=llm_agents.call_llm
                    replayed=[]
                    def reuse_review(system,user,max_tokens=4500,**kwargs):
                        request=json.loads(user)
                        entry=cached.get(request.get('role'))
                        if entry:
                            assert system==entry['system'] and request==entry['request'], 'Cannot replay a council review with changed inputs'
                            replayed.append(entry['role'])
                            save(folder/'reused_council_provenance.json',dict(source=str(old_audit),replayed_roles=replayed,
                                policy='Exact complete system and request equality; valid responses only. Skeptic and Chief are fresh calls.'))
                            return entry['raw_response']
                        return original_call(system,user,max_tokens,**kwargs)
                    stack.enter_context(patch('flora_translate.engine.llm_agents.call_llm',side_effect=reuse_review))
                    reuse=json.loads((folder/'reused_generation_provenance.json').read_text())
                    reuse['fresh']='Deterministic realization, 12 candidates, Skeptic and Chief, final gates and rendering; four exact-matched specialist reviews reused.'
                    save(folder/'reused_generation_provenance.json',reuse)
                result=translate(case['protocol'],intake_package=p.model_dump(),runtime_options=runtime)
                if args.reuse_generation_from:
                    result['reused_generation_provenance']=json.loads((folder/'reused_generation_provenance.json').read_text())
                if args.reuse_council_from:
                    result['reused_council_provenance']=json.loads((folder/'reused_council_provenance.json').read_text())
            save(folder/'result.json',result)
            archive=Path(autosave_gui_result(result,intake_package=p.model_dump(),source='khu_six_revised',user_input=case['protocol']))
            shutil.copytree(archive,folder/'gui_export')
            row.update(status=result.get('final_design',{}).get('status','unknown'),archive=str(archive))
        except Exception as exc:
            row.update(status='failed',error=str(exc))
            (folder/'failure.txt').write_text(traceback.format_exc())
            logging.exception('Case failed; preserving all evidence')
        finally:
            after={x for base in ['scientific_pipeline','scientific_council'] for x in (ROOT/'outputs'/base).glob('*')}
            for path in sorted(after-before):
                shutil.copytree(path,folder/'model_artifacts'/path.parent.name/path.name)
            row['elapsed_seconds']=time.monotonic()-started
            save(folder/'summary.json',row)
            logging.getLogger().removeHandler(handler)
            handler.close()
            print('CASE_COMPLETE',json.dumps(row),flush=True)


if __name__=='__main__': main()
