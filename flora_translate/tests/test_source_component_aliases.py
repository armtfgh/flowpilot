from copy import deepcopy

from flora_translate.source_component_aliases import (
    source_aliases, collapse_source_alias_components, source_alias_conflicts,
    normalize_plan_component_aliases,
)
from flora_translate.schemas import CanonicalStreamComponent, BatchRecord, ChemistryPlan, StreamLogic, ReagentRole
from flora_translate.chemistry_contract import reconcile_chemistry_plan
from flora_translate.executable_artifacts import _stream_components, _stream_preparation_instruction

PROTOCOL = 'N-diphenylmethylazetidin-3-ol (DMAOL; MW239.32) was dissolved in methanol.'
FULL = 'N-diphenylmethylazetidin-3-ol'


def component(name, concentration=.126, text=None, stream='A'):
    return CanonicalStreamComponent(stream_label=stream, name=name, source_text=text or name,
        concentration_M=concentration, molar_equiv=1, role='substrate', quantified=True)


def test_explicit_source_acronym_collapses_identical_dose_only():
    items = [component('DMAOL'), component(FULL+' (DMAOL)')]
    before = deepcopy(items)
    out = collapse_source_alias_components(items, PROTOCOL)
    assert items == before
    assert len(out) == 1 and out[0].name == FULL and out[0].concentration_M == .126
    assert any(PROTOCOL[:-len(' was dissolved in methanol.')] in note for note in out[0].provenance)


def test_conflicting_quantities_remain_visible_under_same_canonical_name():
    out = collapse_source_alias_components([component('DMAOL'),component(FULL+' (DMAOL)',.252)],PROTOCOL)
    assert len(out)==2 and out[0].name==out[1].name==FULL
    assert [x.concentration_M for x in out]==[.126,.252]


def test_legitimate_portions_are_never_silently_deleted():
    protocol=PROTOCOL+' The solution was added in two portions.'
    out=collapse_source_alias_components([component('DMAOL',text='DMAOL (first portion)'),component(FULL+' (DMAOL)',text=FULL+' (second portion)')],protocol)
    assert len(out)==2


def test_portion_marker_on_earlier_description_prevents_merging():
    out=collapse_source_alias_components([component('DMAOL',text='DMAOL (first portion)'),component(FULL+' (DMAOL)')],PROTOCOL)
    assert len(out)==2


def test_aliquots_and_equal_parts_are_preserved_as_possible_separate_charges():
    for phrase in ['two aliquots', 'two equal parts', 'two separate charges']:
        out=collapse_source_alias_components([component('DMAOL'),component(FULL+' (DMAOL)')],PROTOCOL+' It was added in '+phrase+'.')
        assert len(out)==2


def test_separate_feeds_retain_their_independent_amounts():
    first=collapse_source_alias_components([component('DMAOL')],PROTOCOL)
    second=collapse_source_alias_components([component(FULL+' (DMAOL)',.252,stream='B')],PROTOCOL)
    assert len(first)==len(second)==1 and first[0].concentration_M==.126 and second[0].concentration_M==.252


def test_mixed_stream_input_never_collapses_separate_feeds():
    out=collapse_source_alias_components([component('DMAOL',stream='A'),component(FULL+' (DMAOL)',stream='B')],PROTOCOL)
    assert len(out)==2 and [x.stream_label for x in out]==['A','B']


def test_conflicting_roles_preserved_and_flagged():
    other=component(FULL+' (DMAOL)')
    other.role='catalyst'
    out=collapse_source_alias_components([component('DMAOL'),other],PROTOCOL)
    assert len(out)==2 and [x.role for x in out]==['substrate','catalyst']
    assert source_alias_conflicts(out)


def test_upstream_normalization_does_not_mask_role_or_loading_conflicts():
    for role, loading in [('catalyst','1 equiv'), ('substrate','2 equiv')]:
        plan=ChemistryPlan(reagents=[ReagentRole(name='DMAOL',role='substrate',equiv_or_loading='1 equiv'),
            ReagentRole(name=FULL+' (DMAOL)',role=role,equiv_or_loading=loading)],
            stream_logic=[StreamLogic(stream_label='A',reagents=['DMAOL',FULL+' (DMAOL)'],phase='liquid')])
        before=plan.model_copy(deep=True)
        normalize_plan_component_aliases(plan,PROTOCOL)
        assert plan==before


def test_model_only_alias_does_not_authorize_merging():
    out=collapse_source_alias_components([component('DMAOL'),component(FULL+' (DMAOL)')],'An unnamed substrate was dissolved in methanol.')
    assert len(out)==2
    assert source_alias_conflicts(out)


def test_one_acronym_for_two_source_compounds_is_ambiguous():
    protocol='Substrate A (XYZ) was charged. Substrate B (XYZ) was also charged.'
    out=collapse_source_alias_components([component('XYZ'),component('Substrate A (XYZ)'),component('Substrate B (XYZ)')],protocol)
    assert len(out)==3 and source_alias_conflicts(out)


def test_omitted_second_source_compound_cannot_hide_acronym_ambiguity():
    protocol='Substrate A (XYZ) was charged. Substrate B (XYZ) was also charged.'
    out=collapse_source_alias_components([component('XYZ'),component('Substrate A (XYZ)')],protocol)
    assert len(out)==2 and source_alias_conflicts(out)
    assert not source_aliases(protocol,['XYZ','Substrate A (XYZ)']).canonical_by_key


def test_repeated_same_original_alias_declaration_is_not_ambiguous():
    protocol=PROTOCOL+' More '+FULL+' (DMAOL) was reserved for analysis.'
    reg=source_aliases(protocol,['DMAOL',FULL+' (DMAOL)'])
    assert reg.canonical_name('DMAOL')==FULL and not reg.conflicts


def test_molecular_parentheses_are_not_aliases():
    reg=source_aliases('Pd(OH)2 catalyst and H2O2 were used.', ['Pd(OH)2','H2O2','H2'])
    assert not reg.canonical_by_key and not reg.conflicts


def test_inventory_cannot_supply_alias_authority():
    protocol='BATCH PROCEDURE\nA compound was heated.\n\nAVAILABLE EQUIPMENT\n'+PROTOCOL
    out=collapse_source_alias_components([component('DMAOL'),component(FULL+' (DMAOL)')],protocol)
    assert len(out)==2 and source_alias_conflicts(out)


def test_unrelated_substituent_names_are_not_merged():
    protocol='Benzyl azide (BA) was used.'
    reg=source_aliases(protocol,['BA','4-nitrobenzyl azide (BA)'])
    assert not reg.canonical_by_key
    assert reg.conflicts


def test_reconciliation_does_not_restore_fullname_as_an_additional_charge():
    batch=BatchRecord(raw_text=PROTOCOL,solvent='methanol',concentration_M=.126)
    plan=ChemistryPlan(reaction_name='Hydrogenolysis',reagents=[ReagentRole(name=FULL+' (DMAOL)',role='substrate',equiv_or_loading='1 equiv')],
        stream_logic=[StreamLogic(stream_label='A',reagents=['DMAOL (1 equiv, 0.126 M)','methanol'],phase='liquid')])
    before=plan.model_copy(deep=True)
    out,_=reconcile_chemistry_plan(batch,plan)
    assert plan==before
    names=out.stream_logic[0].reagents
    assert len([x for x in names if FULL in x])==1


def test_stock_recipe_deduplication_does_not_mutate_raw_proposal():
    stream={'stream_label':'A','phase':'liquid','solvent':'methanol','concentration_M':.126,'molar_equiv':1,
            'flow_rate_mL_min':.5,'contents':['DMAOL (1 equiv, 0.126 M)',FULL+' (DMAOL)','methanol']}
    result={'batch_record':{'raw_text':PROTOCOL},'chemistry_plan':{'reagents':[{'name':FULL+' (DMAOL)','role':'substrate','equiv_or_loading':'1 equiv'},{'name':'methanol','role':'solvent'}]}}
    before=deepcopy((stream,result))
    comps=_stream_components(result,[stream])
    recipe=_stream_preparation_instruction(stream,comps,parameters={'residence_time_min':1.569036})
    assert recipe.count('1.26 mmol')==1
    assert (stream,result)==before


def test_conflicting_alias_quantities_block_full_release():
    from flora_translate.final_design_contract import build_final_design_contract
    from flora_translate.tests.test_executable_artifact_contract import _hydrogen_result, _issue_codes
    result=_hydrogen_result()
    result['batch_record']['raw_text']='Hydrogenolysis. '+PROTOCOL
    result['chemistry_plan']['reagents'][0]['name']=FULL+' (DMAOL)'
    result['proposal']['streams'][0]['contents']=['DMAOL (0.126 M, 1 equiv)',FULL+' (DMAOL) (0.252 M, 1 equiv)','Methanol']
    contract=build_final_design_contract(result)
    assert contract['status']=='blocked'
    assert 'FINAL-COMPONENT-DESCRIPTIONS-CONFLICT' in _issue_codes(contract)


def test_unverified_alias_pair_blocks_full_release():
    from flora_translate.final_design_contract import build_final_design_contract
    from flora_translate.tests.test_executable_artifact_contract import _hydrogen_result, _issue_codes
    result=_hydrogen_result()
    result['batch_record']['raw_text']='Hydrogenolysis of the supplied substrate under H2.'
    result['chemistry_plan']['reagents'][0]['name']=FULL+' (DMAOL)'
    result['proposal']['streams'][0]['contents']=['DMAOL (0.126 M, 1 equiv)',FULL+' (DMAOL)','Methanol']
    contract=build_final_design_contract(result)
    assert contract['status']=='blocked'
    assert 'FINAL-COMPONENT-DESCRIPTIONS-CONFLICT' in _issue_codes(contract)
