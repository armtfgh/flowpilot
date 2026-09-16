"""Recompile display fields from unchanged final designs; no model calls or new selection."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_khu_six_revised import check
from flora_translate.schemas import FlowProposal,ChemistryPlan,BatchRecord,LabInventory
from flora_translate.main import _build_translate_topology,_store_process_topology
from flora_translate.topology_compiler import compile_inventory_topology
from flora_translate.diagram_artifacts import render_topology_artifacts
from flora_translate.final_design_contract import build_final_design_contract,publish_final_design_artifacts
from flora_translate.gui_autosave import autosave_gui_result


def main():
    root=ROOT/'outputs/khu_revised_six_20260914'
    for source in sorted(root.glob('figure*/attempt_01/result.json')):
        reviewed=check(source)
        if set(reviewed['failed']) != {'diagram_liquid_feed_concentrations'}:
            continue
        folder=source.parent.parent/'attempt_01_display_verified'
        if folder.exists(): continue
        folder.mkdir()
        for name in ['provided_input.json','intake_package.json','request.json','inventory_profile.json','prompt.txt','llm_calls.jsonl','run.log','summary.json']:
            if (source.parent/name).exists():shutil.copy2(source.parent/name,folder/name)
        result=json.loads(source.read_text())
        old_proposal=json.dumps(result['proposal'],sort_keys=True)
        p=FlowProposal.model_validate(result['proposal'])
        inventory=LabInventory.model_validate(json.loads((folder/'inventory_profile.json').read_text())['lab_inventory'])
        topology=_build_translate_topology(p,ChemistryPlan.model_validate(result['chemistry_plan']),BatchRecord.model_validate(result['batch_record']),inventory)
        result['process_requirements_topology']=topology.model_dump()
        topology,allocation=compile_inventory_topology(topology,proposal=p,inventory=inventory)
        assert allocation['checks']['all_required_operations_assigned']
        result['inventory_allocation']=allocation
        result['instrument_manifest']=allocation['instrument_manifest']
        _store_process_topology(result,topology)
        artifacts=render_topology_artifacts(topology,title=result['chemistry_plan'].get('reaction_name','FlowPilot'))
        result.update(svg_path=artifacts['svg_path'],png_path=artifacts['png_path'],diagram_render_manifest=artifacts['manifest'],
                      diagram_artifacts={k:v for k,v in artifacts.items() if k!='manifest'})
        assert json.dumps(result['proposal'],sort_keys=True)==old_proposal
        contract=build_final_design_contract(result)
        if contract['status']!='executable':
            raise ValueError(contract.get('consistency'))
        publish_final_design_artifacts(result,contract)
        result['display_revision_provenance']={
            'source_result':str(source),'fresh_model_generation':False,'selected_numerical_design_unchanged':True,
            'change':'Topology feed labels now include final concentrations and resolved component quantities; no new candidate or model review.',
            'model_logs_and_runtime':'Copied from original run; display regeneration adds no model tokens.'}
        archive=autosave_gui_result(result,intake_package=result['intake_package'],source='khu_display_verified')
        shutil.copytree(archive,folder/'gui_export')
        (folder/'result.json').write_text(json.dumps(result,indent=2))
        (folder/'display_revision_provenance.json').write_text(json.dumps(result['display_revision_provenance'],indent=2))
        print(folder,check(folder/'result.json')['failed'])


if __name__=='__main__': main()
