"""Feed-medium provenance and conservative pump compatibility regressions.

Self-contained fixtures use the physical shape of a hydrogenolysis feed. These
unit tests require no archived benchmark, API, model, or candidate-source copy.
"""
from copy import deepcopy

import pytest

from flora_translate import main
from flora_translate import inventory_allocator as allocator
from flora_translate.schemas import BatchRecord, ChemistryPlan, FlowProposal, LabInventory, ReagentRole


@pytest.fixture
def objects():
    substrate = 'N-Diphenylmethylazetidin-3-ol'
    batch = BatchRecord(
        raw_text=f'{substrate} (DMAOL) was dissolved in methanol at 0.126 M and hydrogenated at 60 C and 21 bar.',
        reaction_description='Hydrogenolysis of N-benzhydryl azetidinol',
        solvent='methanol', temperature_C=60, reaction_time_h=5,
    )
    plan = ChemistryPlan(
        reaction_class='Heterogeneous catalytic hydrogenolysis',
        reagents=[ReagentRole(name=substrate,role='substrate',equiv_or_loading='1 equiv'),
                  ReagentRole(name='Methanol',role='solvent',equiv_or_loading='bulk solvent')],
    )
    proposal = FlowProposal(
        flow_rate_mL_min=.1,concentration_M=.126,temperature_C=60,BPR_bar=21,
        reactor_type='packed-bed',tubing_material='stainless steel',tubing_ID_mm=4.35,
        reactor_volume_mL=3,residence_time_min=2.727273,
        residence_time_basis='inlet/STP apparent residence time',
        streams=[{'stream_label':'A','contents':[f'{substrate} (0.126 M, 1 equiv)','Methanol'],
                  'solvent':'','concentration_M':.126,'flow_rate_mL_min':.1,'phase':'liquid',
                  'pump_equipment_id':'pump_h2','molar_equiv':1},
                 {'stream_label':'B','contents':['H2'],'solvent':'','phase':'gas',
                  'flow_rate_mL_min':.05614,'gas_flow_sccm':1,'pump_equipment_id':'mfc_h2'}],
    )
    inventory = LabInventory(schema_version='flowpilot_lab_inventory_v3.0',strict_assignment=True,pumps=[{
        'equipment_id':'pump_h2','name':'Hydrogenolysis liquid-feed pump','type':'plunger',
        'min_flow_rate_mL_min':.1,'max_flow_rate_mL_min':2,'max_pressure_bar':40,
        'compatible_materials':['methanol','DMAOL'],
    }])
    return proposal,plan,batch,inventory


def pump_operations(topology):
    return {op.parameters['stream']:op for op in topology.unit_operations if op.op_type=='pump'}


def build(objects):
    p,c,b,i=objects
    return main._build_translate_topology(p,c,b,i)


def test_explicit_feed_methanol_survives_canonical_substrate_label(objects):
    p,c,b,i=objects
    before=deepcopy(p.model_dump())
    topology=build(objects)
    op=pump_operations(topology)['A']
    assert op.parameters['solvent']=='Methanol'
    assert 'Methanol' in op.parameters['contents']
    assert op.parameters['solvent_provenance']['source_text']==['Methanol']
    assert op.parameters['solvent_provenance']['authority']=='explicit_same_feed_component'
    # Other equipment is deliberately absent from this minimal fixture. The
    # regression concerns successful assignment of the declared liquid pump.
    _,report=allocator.InventoryAllocator(i,p).compile(topology)
    assert not any(r['operation_id']==op.op_id for r in report['unresolved_requirements'])
    assert p.model_dump()==before


@pytest.mark.parametrize('solvent',['MeOH','methanol'])
def test_conventional_solvent_alias_is_compatible(objects,solvent):
    pump=objects[3].pumps[0]
    assert allocator._pump_material_compatible(pump,'Different substrate '+solvent,solvent=solvent)


def test_arbitrary_substrate_name_does_not_erase_declared_medium(objects):
    p,c,_,i=objects
    p.streams[0].contents=['Different substrate (0.126 M, 1 equiv)','Methanol']
    c.reagents=[ReagentRole(name='Different substrate',role='substrate',equiv_or_loading='1 equiv'),ReagentRole(name='Methanol',role='solvent')]
    op=pump_operations(build(objects))['A']
    assert op.parameters['solvent']=='Methanol'
    assert allocator._pump_material_compatible(i.pumps[0],' '.join(op.parameters['contents']),solvent=op.parameters['solvent'])


def test_missing_medium_does_not_inherit_batch_solvent(objects):
    p,c,b,i=objects
    p.streams[0].contents=['Unassigned solute (0.126 M, 1 equiv)']
    p.streams[0].solvent=''
    c.reagents=[ReagentRole(name='Unassigned solute',role='substrate',equiv_or_loading='1 equiv')]
    op=pump_operations(build(objects))['A']
    assert b.solvent=='methanol'
    assert op.parameters['solvent']==''
    assert 'solvent_provenance' not in op.parameters
    assert not allocator._pump_material_compatible(i.pumps[0],' '.join(op.parameters['contents']))


def test_other_feed_medium_is_not_inherited(objects):
    p,c,_,i=objects
    second=p.streams[0].model_copy(deep=True)
    second.stream_label='C';second.contents=['Other solute (0.1 M, 1 equiv)','DCM (solvent)'];second.solvent=''
    p.streams.append(second)
    c.reagents.append(ReagentRole(name='Other solute',role='substrate',equiv_or_loading='1 equiv'))
    ops=pump_operations(build(objects))
    assert ops['A'].parameters['solvent']=='Methanol'
    assert ops['C'].parameters['solvent']=='DCM'
    assert 'Methanol' not in ops['C'].parameters['contents']
    assert not allocator._pump_material_compatible(i.pumps[0],' '.join(ops['C'].parameters['contents']),solvent=ops['C'].parameters['solvent'])


def test_explicit_medium_is_not_overridden_and_gas_gets_no_liquid_medium(objects):
    p,_,_,_=objects
    p.streams[0].solvent='acetonitrile'
    topology=build(objects)
    assert pump_operations(topology)['A'].parameters['solvent']=='acetonitrile'
    assert next(op for op in topology.unit_operations if op.op_type=='mfc').parameters['solvent']==''


def test_incompatible_explicit_medium_cannot_pass_via_methanol_or_substrate(objects):
    p,_,_,i=objects
    p.streams[0].solvent='acetonitrile'
    topology=build(objects);op=pump_operations(topology)['A']
    assert 'Methanol' in op.parameters['contents']
    _,report=allocator.InventoryAllocator(i,p).compile(topology)
    assert any(r['operation_id']==op.op_id for r in report['unresolved_requirements'])
    assert not allocator._pump_material_compatible(i.pumps[0],'DMAOL Methanol acetonitrile',solvent='acetonitrile',solvent_components=['Methanol'])


@pytest.mark.parametrize('mixture',['methanol/DCM','MeOH:DCM (1:1)','methanol and dichloromethane 1:1 v/v'])
def test_mixture_requires_compatibility_for_every_explicit_medium(objects,mixture):
    pump=objects[3].pumps[0]
    assert not allocator._pump_material_compatible(pump,'DMAOL '+mixture,solvent=mixture)
    pump.compatible_materials.append('dichloromethane')
    assert allocator._pump_material_compatible(pump,'DMAOL '+mixture,solvent=mixture)


def test_explicit_component_mixture_and_unknown_medium_do_not_pass(objects):
    pump=objects[3].pumps[0]
    assert not allocator._pump_material_compatible(pump,'DMAOL Methanol DCM',solvent_components=['Methanol','DCM'])
    assert not allocator._pump_material_compatible(pump,'DMAOL unknown liquid',solvent='unidentified solvent')
    assert allocator._pump_material_compatible(pump,'Other substrate MeOH',solvent='MeOH',solvent_components=['methanol'])
