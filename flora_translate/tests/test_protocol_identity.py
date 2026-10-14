"""Identity authority must come from the supplied chemistry, not its equipment."""
from copy import deepcopy
import hashlib

import pytest

from flora_translate.executable_artifacts import _chemistry_identity
from flora_translate.final_design_contract import build_final_design_contract
from flora_translate.protocol_identity import infer_protocol_identity, plan_identity_issues
from flora_translate.tests.test_executable_artifact_contract import _hydrogen_result, _issue_codes


H2 = (
    'N-Diphenylmethylazetidin-3-ol was dissolved in methanol. '
    'The solution was contacted with hydrogen over Pd(OH)2/Al2O3 at 60 C. '
    'Conversion to 3-azetidinol was 38.1%.'
)
PHOTO = (
    'Fmoc-L-methionine and a photocatalyst were dissolved in acetonitrile. '
    'The solution was exposed to air and irradiated. '
    'Fmoc-methionine sulfoxide was formed in 99% yield.'
)
CLICK = (
    'Benzyl azide and phenylacetylene were combined with Cu/C catalyst in acetone. '
    'The corresponding 1,4-disubstituted 1,2,3-triazole was isolated in 94% yield.'
)


def _plan(family, source, product):
    return {'reaction_name': f'{family} of {source} to {product}', 'reaction_class': family,
            'reagents': [{'name': source, 'role': 'substrate'}, {'name': product, 'role': 'target product'}]}


@pytest.mark.parametrize('text,family', [(H2, 'hydrogenolysis'), (PHOTO, 'oxidation'), (CLICK, 'cuaac')])
def test_unnamed_observed_transformations_have_traceable_evidence(text, family):
    evidence = infer_protocol_identity(text)
    assert evidence.family == family
    assert evidence.rule.endswith('_v1')
    assert evidence.substrates and evidence.product
    assert all(span in text for span in evidence.source_evidence)


@pytest.mark.parametrize('equipment', ['hydrogenation skid', 'oxidation apparatus', 'nitration pump', 'click reactor'])
def test_equipment_cannot_change_authority_or_protocol_hash(equipment):
    result = {'batch_record': {'raw_text': f'BATCH PROCEDURE\n{H2}\n\nREQUEST\nDesign flow.\n\nAVAILABLE EQUIPMENT\n{{"name":"{equipment}"}}'},
              'chemistry_plan': _plan('hydrogenolysis', 'N-diphenylmethylazetidin-3-ol', '3-azetidinol')}
    identity = _chemistry_identity(result)
    assert identity.protocol_transformation_family == 'hydrogenolysis'
    assert identity.authority_source == 'protocol_evidence'
    assert identity.confirmed and not identity.validation_issues
    assert identity.protocol_sha256 == hashlib.sha256(H2.encode()).hexdigest()
    assert equipment not in ' '.join(identity.source_evidence)


@pytest.mark.parametrize('protocol,family', [
    ('N-benzylpiperidin-4-ol was stirred over Pd/C under H2. The reaction afforded 4-piperidinol in 85% yield.', 'hydrogenolysis'),
    ('Methyl phenyl sulfide was exposed to hydrogen peroxide. Methyl phenyl sulfoxide was isolated in 90% yield.', 'oxidation'),
    ('Ethyl azide and hex-1-yne were heated with CuI. The reaction afforded 1,4-disubstituted 1,2,3-triazole in 85% yield.', 'cuaac'),
])
def test_rules_generalize_to_other_substrates_and_outcome_phrasings(protocol, family):
    assert infer_protocol_identity(protocol).family == family


@pytest.mark.parametrize('protocol', [
    H2.replace('3-azetidinol', '3-piperidinol'),
    H2.replace('was 38.1%', 'was not observed'),
    H2.replace('Conversion to 3-azetidinol was 38.1%.', '3-azetidinol might be produced.'),
    H2.replace('N-Diphenylmethylazetidin-3-ol', 'styrene'),
    PHOTO.replace('was formed', 'was not formed'),
    PHOTO.replace('was formed', 'might be formed'),
    PHOTO.replace('Fmoc-L-methionine and', 'Fmoc-L-lysine and'),
    PHOTO.replace('Fmoc-L-methionine and', 'Fmoc-L-methionine sulfoxide and'),
    CLICK.replace('was isolated', 'could be isolated'),
    CLICK.replace('phenylacetylene', 'benzonitrile'),
    CLICK.replace('Cu/C', 'ruthenium'),
    'A substrate and catalyst were heated, affording a product in 90% yield.',
])
def test_insufficient_or_contradictory_source_evidence_is_not_confirmed(protocol):
    assert infer_protocol_identity(protocol) is None


@pytest.mark.parametrize('which', ['substrate', 'product'])
def test_correct_family_name_does_not_hide_changed_species(which):
    evidence = infer_protocol_identity(H2)
    plan = _plan('hydrogenolysis', 'N-diphenylmethylazetidin-3-ol', '3-azetidinol')
    plan['reagents'][0 if which == 'substrate' else 1]['name'] = 'unrelated compound'
    assert plan_identity_issues(evidence, plan)


def test_sulfoxide_to_sulfone_change_is_identity_drift_even_with_same_family():
    evidence = infer_protocol_identity(PHOTO)
    plan = _plan('oxidation', 'Fmoc-L-methionine', 'Fmoc-L-methionine sulfone')
    assert plan_identity_issues(evidence, plan)


def test_stereochemical_substrate_change_is_not_a_spelling_alias():
    evidence = infer_protocol_identity(PHOTO)
    plan = _plan('oxidation', 'Fmoc-D-methionine', 'Fmoc-D-methionine sulfoxide')
    assert plan_identity_issues(evidence, plan)


def test_cuaac_product_regiochemistry_is_not_a_family_synonym():
    evidence = infer_protocol_identity(CLICK)
    plan = {'reaction_name': 'CuAAC', 'reagents': [
        {'name': 'benzyl azide', 'role': 'substrate'},
        {'name': 'phenylacetylene', 'role': 'substrate'},
        {'name': '1-benzyl-5-phenyl-1,2,3-triazole', 'role': 'target product'},
    ]}
    assert plan_identity_issues(evidence, plan)


def test_ambiguous_protocol_not_authorized_by_model_description_or_inventory():
    identity = _chemistry_identity({'batch_record': {
        'raw_text': 'A solution was heated for 5 minutes.',
        'reaction_description': 'Hydrogenolysis of benzyl-protected amine'},
        'chemistry_plan': {'reaction_name': 'Hydrogenolysis'},
        'inventory_snapshot': {'name': 'hydrogenation skid'}})
    assert not identity.confirmed
    assert identity.authority_source == 'model_inference'


def test_named_hydrogenation_is_not_equivalent_to_cleavage_evidence():
    result = _hydrogen_result()
    result['batch_record']['raw_text'] = H2
    result['chemistry_plan'] = _plan('hydrogenation', 'N-diphenylmethylazetidin-3-ol', '3-azetidinol')
    contract = build_final_design_contract(result)
    assert contract['status'] == 'blocked'
    assert 'FINAL-CHEMISTRY-IDENTITY-DRIFT' in _issue_codes(contract)


def test_true_product_drift_blocks_complete_artifact_release():
    result = _hydrogen_result()
    result['batch_record']['raw_text'] = H2
    result['chemistry_plan']['reaction_name'] = 'Hydrogenolysis of N-diphenylmethylazetidin-3-ol to piperidine'
    result['chemistry_plan']['reagents'][0]['name'] = 'N-diphenylmethylazetidin-3-ol'
    contract = build_final_design_contract(result)
    assert contract['status'] == 'blocked'
    assert 'FINAL-CHEMISTRY-IDENTITY-DRIFT' in _issue_codes(contract)


def test_compilation_preserves_provenance_and_does_not_invent_human_confirmation():
    result = _hydrogen_result()
    result['batch_record']['raw_text'] = H2
    result['chemistry_plan']['reaction_name'] = 'Hydrogenolysis of N-diphenylmethylazetidin-3-ol to 3-azetidinol'
    result['chemistry_plan']['reagents'][0]['name'] = 'N-diphenylmethylazetidin-3-ol'
    before = deepcopy(result)
    identity = _chemistry_identity(result)
    assert result == before
    assert identity.authority_source == 'protocol_evidence'
    assert identity.evidence_rule
    assert not identity.validation_issues


def test_substituent_addition_is_not_allowed_by_substring_matching():
    evidence = infer_protocol_identity(CLICK)
    plan = {'reaction_name': 'CuAAC', 'reagents': [
        {'name': '4-nitrobenzyl azide', 'role': 'substrate'},
        {'name': 'phenylacetylene', 'role': 'substrate'},
    ]}
    assert plan_identity_issues(evidence, plan)


def test_unspecified_product_stereochemistry_does_not_authorize_inversion():
    evidence = infer_protocol_identity(PHOTO)
    plan = _plan('oxidation', 'Fmoc-L-methionine', 'Fmoc-D-methionine sulfoxide')
    assert plan_identity_issues(evidence, plan)


def test_unicode_recognition_keeps_literal_source_quotes():
    protocol = H2.replace('N-Diphenyl', 'N–Diphenyl')
    evidence = infer_protocol_identity(protocol)
    assert evidence.family == 'hydrogenolysis'
    assert all(quote in protocol for quote in evidence.source_evidence)


@pytest.mark.parametrize('hardware', ['hydrogenation vessel', 'nitration skid', 'reduction glass reactor'])
def test_equipment_words_inside_procedure_are_not_reaction_labels(hardware):
    protocol = PHOTO.replace('exposed to air', f'exposed to air in a {hardware}')
    identity = _chemistry_identity({'batch_record': {'raw_text': protocol},
        'chemistry_plan': _plan('oxidation', 'Fmoc-L-methionine', 'Fmoc-L-methionine sulfoxide')})
    assert identity.protocol_transformation_family == 'oxidation'
    assert identity.authority_source == 'protocol_evidence'
    assert not identity.validation_issues


def test_equipment_mask_does_not_hide_an_actual_reaction_claim():
    protocol = H2.replace('The solution was contacted', 'Hydrogenation in a reactor was performed. The solution was contacted')
    identity = _chemistry_identity({'batch_record': {'raw_text': protocol},
        'chemistry_plan': _plan('hydrogenolysis', 'N-diphenylmethylazetidin-3-ol', '3-azetidinol')})
    assert identity.validation_issues


def test_original_protocol_acronym_can_preserve_substrate_identity():
    protocol = H2.replace('N-Diphenylmethylazetidin-3-ol', 'N-Diphenylmethylazetidin-3-ol (DMAOL; MW 239.32 g/mol)')
    evidence = infer_protocol_identity(protocol)
    plan = _plan('hydrogenolysis', 'DMAOL', '3-azetidinol')
    assert evidence.substrate_aliases == (('DMAOL',),)
    assert not plan_identity_issues(evidence, plan)


def test_model_invented_acronym_is_not_original_source_evidence():
    evidence = infer_protocol_identity(H2)
    plan = _plan('hydrogenolysis', 'DMAOL', '3-azetidinol')
    assert plan_identity_issues(evidence, plan)
