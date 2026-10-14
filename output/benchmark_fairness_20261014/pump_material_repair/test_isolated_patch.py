"""Offline candidate-only regressions. Imports no paid API call; edits no production."""
import hashlib, importlib.util, json
from pathlib import Path
from copy import deepcopy
import pytest
from flora_translate.schemas import FlowProposal, ChemistryPlan, BatchRecord, LabInventory, ReagentRole

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,BASE/filename)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
main=load('candidate_main','candidate_main.py')
allocator=load('candidate_allocator','candidate_inventory_allocator.py')
SOURCE=ROOT/'ablation_results/practical_input/claude_fair_bridge_20261014/generation/hydrogenolysis/claude_flowpilot/repeat_01/result.json'
DATA=json.loads(SOURCE.read_text())
def objects():
    return (FlowProposal.model_validate(DATA['proposal']),ChemistryPlan.model_validate(DATA['chemistry_plan']),BatchRecord.model_validate(DATA['batch_record']),LabInventory.model_validate(DATA['inventory_snapshot']))
def pumpops(t): return {x.parameters['stream']:x for x in t.unit_operations if x.op_type=='pump'}

def test_actual_blocked_example_rebuild_does_not_change_original_proposal():
    p,c,b,i=objects();before=p.model_dump();source_before=SOURCE.read_bytes()
    t=main._build_translate_topology(p,c,b,i)
    a=pumpops(t)['A'];assert a.parameters['solvent']=='Methanol'
    assert 'Methanol' in a.parameters['contents']
    assert a.parameters['solvent_provenance']['source_text']==['Methanol']
    assert a.parameters['solvent_provenance']['authority']=='explicit_same_feed_component'
    compiled,report=allocator.InventoryAllocator(i,p).compile(t)
    assert report['status']=='complete' and report['unresolved_requirements']==[]
    assert p.model_dump()==before and SOURCE.read_bytes()==source_before

def test_explicit_conventional_solvent_synonym_works():
    _,_,_,i=objects()
    assert allocator._pump_material_compatible(i.pumps[0],'Different solute MeOH',solvent='MeOH')
    assert allocator._pump_material_compatible(i.pumps[0],'Different solute methanol',solvent='methanol')
    assert not allocator._pump_material_compatible(i.pumps[0],'Different solute DCM',solvent='DCM')

def test_full_substrate_identity_does_not_erase_explicit_medium():
    p,c,b,i=objects();p.streams[0].contents=['Different substrate (0.126 M, 1 equiv)','Methanol'];c.reagents=[ReagentRole(name='Different substrate',role='substrate',equiv_or_loading='1 equiv'),ReagentRole(name='Methanol',role='solvent')]
    t=main._build_translate_topology(p,c,b,i);a=pumpops(t)['A']
    assert a.parameters['solvent']=='Methanol'
    assert allocator._pump_material_compatible(i.pumps[0],' '.join(a.parameters['contents']),solvent=a.parameters['solvent'])

def test_unknown_feed_does_not_inherit_batch_solvent():
    p,c,b,i=objects();p.streams[0].contents=['Unassigned solute (0.126 M, 1 equiv)'];p.streams[0].solvent='';c.reagents=[ReagentRole(name='Unassigned solute',role='substrate',equiv_or_loading='1 equiv')]
    t=main._build_translate_topology(p,c,b,i);a=pumpops(t)['A']
    assert b.solvent=='methanol' and a.parameters['solvent']==''
    assert 'solvent_provenance' not in a.parameters
    assert not allocator._pump_material_compatible(i.pumps[0],' '.join(a.parameters['contents']))

def test_other_feed_solvent_does_not_leak_or_get_replaced():
    p,c,b,i=objects();first=p.streams[0];second=first.model_copy(deep=True);second.stream_label='C';second.contents=['Other solute (0.1 M, 1 equiv)','DCM (solvent)'];second.solvent=''
    p.streams.append(second);c.reagents.append(ReagentRole(name='Other solute',role='substrate',equiv_or_loading='1 equiv'))
    t=main._build_translate_topology(p,c,b,i);ops=pumpops(t)
    assert ops['A'].parameters['solvent']=='Methanol'
    assert ops['C'].parameters['solvent']=='DCM'
    assert 'Methanol' not in ops['C'].parameters['contents']
    assert not allocator._pump_material_compatible(i.pumps[0],' '.join(ops['C'].parameters['contents']),solvent=ops['C'].parameters['solvent'])

def test_explicit_feed_solvent_never_overridden_and_gas_not_filled():
    p,c,b,i=objects();p.streams[0].solvent='acetonitrile';t=main._build_translate_topology(p,c,b,i)
    assert pumpops(t)['A'].parameters['solvent']=='acetonitrile'
    assert next(x for x in t.unit_operations if x.op_type=='mfc').parameters['solvent']==''

def test_production_sources_unchanged():
    for filename in ['main','inventory_allocator']:
        assert hashlib.sha256((ROOT/f'flora_translate/{filename}.py').read_bytes()).hexdigest()==(BASE/f'production_{filename}_sha256.txt').read_text().strip()

def test_conflicting_explicit_medium_cannot_pass_via_restored_methanol_or_DMAOL():
    p,c,b,i=objects();p.streams[0].solvent='acetonitrile';t=main._build_translate_topology(p,c,b,i)
    a=pumpops(t)['A'];assert 'Methanol' in a.parameters['contents']
    _,report=allocator.InventoryAllocator(i,p).compile(t)
    assert any(x['operation_id']=='pump_a' for x in report['unresolved_requirements'])
    assert not allocator._pump_material_compatible(i.pumps[0],'DMAOL Methanol acetonitrile',solvent='acetonitrile',solvent_components=['Methanol'])

@pytest.mark.parametrize('mixture',['methanol/DCM','MeOH:DCM (1:1)','methanol and dichloromethane 1:1 v/v'])
def test_mixture_requires_every_explicit_medium_member(mixture):
    _,_,_,i=objects();pump=i.pumps[0]
    assert not allocator._pump_material_compatible(pump,'DMAOL '+mixture,solvent=mixture)
    pump.compatible_materials.append('dichloromethane')
    assert allocator._pump_material_compatible(pump,'DMAOL '+mixture,solvent=mixture)

def test_canonical_component_only_mixture_and_unknown_remain_unresolved():
    _,_,_,i=objects();pump=i.pumps[0]
    assert not allocator._pump_material_compatible(pump,'DMAOL Methanol DCM',solvent_components=['Methanol','DCM'])
    assert not allocator._pump_material_compatible(pump,'DMAOL unknown liquid',solvent='unidentified solvent')
    assert allocator._pump_material_compatible(pump,'Other substrate MeOH',solvent='MeOH',solvent_components=['methanol'])
