"""Independent arithmetic/artifact audit. Reads FlowPilot outputs, never judges."""
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
GEN = OUT.parent / 'campaign/generation'
STP_MOLAR_VOLUME_ML_PER_MMOL = 0.08314462618 * 273.15 / 1.01325


def match(a, b):
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-4, abs_tol=2e-6)


def equipment_ids(value):
    if isinstance(value, dict):
        found = {str(value['equipment_id'])} if value.get('equipment_id') else set()
        for child in value.values():
            found.update(equipment_ids(child))
        return found
    if isinstance(value, list):
        return set().union(*(equipment_ids(child) for child in value)) if value else set()
    return set()


def audit(case):
    source = GEN / case / 'claude_flowpilot/repeat_01/result.json'
    if not source.exists():
        return {'available': False}
    raw = source.read_bytes()
    result = json.loads(raw)
    final = result.get('final_design') or {}
    parameters = final.get('parameters') or {}
    graph = final.get('process_graph') or {}
    streams = final.get('streams') or []
    procedure = {row['step_id']: row for row in final.get('operating_procedure') or []}
    order = {key: index for index, key in enumerate(procedure)}
    known_ids = equipment_ids(result.get('inventory_snapshot') or {})
    topology = graph.get('topology') or {}
    operations = topology.get('unit_operations') or []
    assigned_ids = {str(row['inventory_item_id']) for row in operations if row.get('inventory_item_id')}
    graph_feeds = graph.get('feeds') or []
    liquid_feeds = [s for s in graph_feeds if s.get('phase') != 'gas']
    qliquid = sum(float(s.get('flow_rate_mL_min') or 0) for s in liquid_feeds)
    timing = procedure.get('STEADY-STATE', {}).get('parameter_bindings') or {}
    collection = procedure.get('COLLECT-PRODUCT', {}).get('parameter_bindings') or {}
    duration = (float(timing.get('minimum_initial_stabilization_time_min') or 0)
                + float(collection.get('minimum_initial_collection_time_min') or 0))
    stock = []
    for feed in liquid_feeds:
        prep = next((row for row in procedure.values()
                     if row.get('section') == 'preparation'
                     and row.get('parameter_bindings', {}).get('stream_label') == feed['stream_label']), {})
        binding = prep.get('parameter_bindings') or {}
        expected = float(feed['flow_rate_mL_min']) * duration
        stock.append({'stream_label': feed['stream_label'], 'expected_minimum_mL': expected,
                      'declared_minimum_mL': binding.get('minimum_feed_stock_volume_mL'),
                      'recipe_basis_mL': binding.get('recipe_basis_volume_mL'),
                      'minimum_closes': match(binding.get('minimum_feed_stock_volume_mL'), expected),
                      'recipe_closes': match(binding.get('recipe_basis_volume_mL'), expected),
                      'extra_volume_explicitly_unresolved': binding.get('additional_feed_volume_mL') is None,
                      'availability_not_invented': binding.get('stock_availability_confirmed') is False})
    pressure = parameters.get('pressure_absolute_bar')
    expected_pressure = (float(parameters.get('BPR_bar') or 0)
                         + (1.01325 if parameters.get('BPR_basis') == 'gauge' else 0))
    gas = []
    substrate_feeds = [s for s in streams if s.get('phase') != 'gas' and s.get('concentration_M')]
    # These cases each have one premixed substrate-bearing liquid feed.
    substrate_flux = sum(float(s.get('flow_rate_mL_min') or 0) * float(s['concentration_M'])
                         / float(s.get('molar_equiv') or 1) for s in substrate_feeds)
    for stream in streams:
        if stream.get('phase') != 'gas':
            continue
        qstandard = float(stream['gas_flow_sccm'])
        expected_actual = qstandard * (float(parameters['temperature_C']) + 273.15) / 273.15 * 1.01325 / float(pressure)
        active_flux = qstandard / STP_MOLAR_VOLUME_ML_PER_MMOL * float(stream.get('gas_reagent_mole_fraction') or 1)
        equivalent = active_flux / substrate_flux if substrate_flux else None
        gas.append({'stream_label': stream['stream_label'], 'standard_mL_min': qstandard,
                    'actual_mL_min': stream.get('gas_flow_actual_mL_min'), 'expected_actual_mL_min': expected_actual,
                    'actual_flow_closes': match(stream.get('gas_flow_actual_mL_min'), expected_actual),
                    'active_molar_flow_mmol_min': active_flux, 'substrate_molar_flow_mmol_min': substrate_flux,
                    'calculated_equivalents': equivalent, 'declared_equivalents': stream.get('molar_equiv'),
                    'equivalents_close': match(stream.get('molar_equiv'), equivalent)})
    stage_times = []
    for stage in graph.get('stages') or []:
        expected = float(stage['reactor_volume_mL']) / float(stage['residence_flow_mL_min'])
        stage_times.append({'stage': stage['stage_number'], 'basis': stage['residence_time_basis'],
                            'declared_min': stage['residence_time_min'], 'calculated_min': expected,
                            'closes': match(stage['residence_time_min'], expected)})
    topology_rows = [{k: op.get(k) for k in ('op_id', 'op_type', 'inventory_item_id')} for op in operations]
    record = {'available': True, 'source': str(source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'status': final.get('status'), 'parameters': parameters,
              'stock_schedule': stock, 'scheduled_minimum_duration_min': duration,
              'pressure_check': {'declared_absolute_bar': pressure, 'expected_absolute_bar': expected_pressure,
                                 'closes': match(pressure, expected_pressure)} if gas else None,
              'gas_closure': gas, 'stage_time_closure': stage_times,
              'total_liquid_flow_closes': match(graph.get('total_liquid_flow_mL_min'), qliquid),
              'unknown_assigned_inventory_ids': sorted(assigned_ids - known_ids),
              'procedure_order': order, 'topology_operations': topology_rows,
              'topology_edges': topology.get('streams'),
              'final_rationale_status': final.get('decision_rationale', {}).get('final_quantitative_basis', {}).get('status'),
              'rationale_history_keys': list(final.get('rationale_history') or {}),
              'explanation_status': result.get('explanation_status'),
              'source_unchanged': source.read_bytes() == raw}
    (OUT / f'{case}_arithmetic.json').write_text(json.dumps(record, indent=2) + '\n')
    # Solely final delivered design content, for manual review, no judge inputs/grades.
    (OUT / f'{case}_final_design.txt').write_text(json.dumps({
        'parameters': parameters, 'streams': streams,
        'stationary_components': final.get('stationary_components'),
        'stream_components': final.get('stream_components'),
        'operating_procedure': list(procedure.values()),
        'decision_rationale': final.get('decision_rationale'),
        'rationale_history': final.get('rationale_history'),
        'canonical_explanation': result.get('canonical_explanation'),
    }, indent=2) + '\n')
    return record


report = {'scope': 'Independent read-only audit of FlowPilot outputs only. No judge responses opened.',
          'cases': {case: audit(case) for case in ('hydrogenolysis', 'photochemical_oxidation', 'cuaac')}}
(OUT / 'arithmetic_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({case: {'available': value['available'], 'status': value.get('status')}
                  for case, value in report['cases'].items()}, indent=2))
