"""Independent source, equipment and arithmetic checks on every saved attempt."""
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flora_translate.equipment_resources import resources_fit, light_fits_stage
from flora_translate.schemas import LabInventory, FlowProposal
from flora_translate.executable_artifacts import _stream_components


def close(a,b):
    return a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=3e-5,abs_tol=1e-8)


def check(path):
    result=json.loads(path.read_text())
    folder=path.parent
    source=json.loads((folder/'provided_input.json').read_text())
    inventory=LabInventory.model_validate(json.loads((folder/'inventory_profile.json').read_text())['lab_inventory'])
    p=FlowProposal.model_validate(result.get('proposal') or {})
    stages=p.stage_parameters
    streams=[s.model_dump() for s in p.streams]
    reactors={r.equipment_id:r for r in inventory.reactors}
    lights={r.equipment_id:r for r in inventory.light_sources}
    pumps={r.equipment_id:r for r in inventory.pumps}
    checks={'closed_final':result.get('final_design',{}).get('status')=='executable',
            'two_stages':len(stages)==2,
            'original_protocol_preserved':result.get('intake_package',{}).get('raw_protocol')==source['protocol'],
            'twelve_candidates':result.get('scientific_assessment',{}).get('candidate_count')==12}
    stage_rows=[]
    for stage in stages:
        n=stage['stage_number']
        reactor=reactors.get(stage.get('reactor_equipment_id'))
        liquid=sum(s['flow_rate_mL_min'] or 0 for s in streams if s['phase']!='gas' and s['introduction_stage']<=n)
        gas=sum(s['gas_flow_sccm'] or 0 for s in streams if s['phase']=='gas' and s['introduction_stage']<=n)
        checks[f'stage{n}_cumulative_liquid']=close(liquid,stage.get('Q_liquid_mL_min'))
        checks[f'stage{n}_inlet_time']=close(stage.get('residence_time_min'),stage.get('reactor_volume_mL',0)/(liquid+gas)) if liquid+gas>0 else False
        checks[f'stage{n}_stock_geometry']=bool(reactor and close(reactor.volume_mL,stage.get('reactor_volume_mL')) and close(reactor.ID_mm,stage.get('d_mm')))
        checks[f'stage{n}_tubing_temperature']=bool(reactor and stage['temperature_C']<=reactor.max_temperature_C)
        light=lights.get(stage.get('light_equipment_id'))
        if source['figure']==5:
            checks[f'stage{n}_module_compatible']=bool(light and reactor and light_fits_stage(light,reactor,p,inventory))
            checks[f'stage{n}_light_temperature']=bool(light and (not light.allowed_temperatures_C or stage['temperature_C'] in light.allowed_temperatures_C)
                and light.min_temperature_C<=stage['temperature_C']<=light.max_temperature_C)
        stage_rows.append(dict(case=source['id'],stage=n,volume_mL=stage.get('reactor_volume_mL'),ID_mm=stage.get('d_mm'),
            material=stage.get('material'),temperature_C=stage.get('temperature_C'),liquid_flow_mL_min=liquid,
            gas_inlet_STP_mL_min=gas,nominal_inlet_residence_min=stage.get('residence_time_min'),BPR_bar=p.BPR_bar,
            reactor=reactor.name if reactor else 'UNRESOLVED',module=light.module_name if light else 'Thermal, no light',
            light=light.name if light else 'None',wavelength_nm=light.wavelength_nm if light else None))
    used=[pumps[s.pump_equipment_id] for s in p.streams if s.phase!='gas' and s.pump_equipment_id in pumps]
    used += [lights[s['light_equipment_id']] for s in stages if s.get('light_equipment_id') in lights]
    checks['shared_resources']=resources_fit(used,inventory.resource_capacities)
    components=[c.model_dump() for c in _stream_components(result,streams)]
    component_rows=[]
    for s in streams:
        for c in [c for c in components if c['stream_label']==s['stream_label'] and c['role']!='solvent']:
            conc=c.get('concentration_M')
            gas=s['phase']=='gas'
            flow=s['gas_flow_sccm'] if gas else s['flow_rate_mL_min']
            molar=(flow/(.08314*273.15/1.01325)*(s.get('gas_reagent_mole_fraction') or 1)) if gas and flow else conc*flow if conc and flow else None
            component = c['name'] + ' (O2 component, 21 mol% of air)' if gas and c['name'].lower()=='air' else c['name']
            equiv=c.get('molar_equiv')
            if equiv is None and c.get('loading_mol_pct') is not None:
                equiv=c['loading_mol_pct']/100
            component_rows.append(dict(case=source['id'],stream=s['stream_label'],stage=s['introduction_stage'],component=component,
                phase=s['phase'],solvent=s['solvent'],concentration_M=None if gas else conc,
                flow_mL_min=flow,molar_flow_mmol_min=molar,equivalents=equiv,
                concentration_basis='; '.join(c.get('provenance',[])),pump=s['pump_equipment_id']))
    checks['all_liquid_components_quantified']=bool(component_rows) and all(x['concentration_M'] is not None and x['molar_flow_mmol_min'] is not None for x in component_rows if x['phase']!='gas')
    checks['no_invented_buffer_molarity']=not any('buffer' in text.lower() and 'M screening assumption' in text for s in streams for text in s['contents'])
    if source['figure']==5:
        gases=[s for s in streams if s['phase']=='gas']
        checks['oxygen_stage2_only']=bool(gases) and all(s['introduction_stage']==2 for s in gases)
        checks['oxygen_at_least2equiv']=bool(gases) and all(s['molar_equiv']>=2-1e-8 for s in gases)
        precursor=[x for x in component_rows if 'trimethylsilane' in x['component'].lower()]
        oxygen=[x for x in component_rows if x['phase']=='gas']
        checks['oxygen_equiv_from_actual_stp_setting']=bool(precursor and oxygen) and sum(x['molar_flow_mmol_min'] or 0 for x in oxygen)>=1.9999*sum(x['molar_flow_mmol_min'] or 0 for x in precursor)
        ops=result.get('process_topology',{}).get('unit_operations',[])
        checks['check_valve_assigned']=any(o['op_type']=='check_valve' and o.get('inventory_item_id')=='cv-3301' for o in ops)
    if source['figure']==6:
        acid=[x for x in component_rows if 'nitrobenzoic acid' in x['component'].lower()]
        amine=[x for x in component_rows if 'benzylamine' in x['component'].lower()]
        checks['amine_equiv_1_05']=bool(acid and amine) and close(sum(x['molar_flow_mmol_min'] or 0 for x in amine),1.05*sum(x['molar_flow_mmol_min'] or 0 for x in acid))
        checks['amine_stage2_only']=bool(amine) and all(x['stage']==2 for x in amine)
    svg=folder/'gui_export/process.svg'
    checks['icon_topology_exists']=svg.exists() and 'data:image/png;base64,' in svg.read_text()
    ops=result.get('process_topology',{}).get('unit_operations',[])
    checks['diagram_liquid_feed_concentrations']=all(any(o['op_type']=='pump' and o.get('parameters',{}).get('stream')==s['stream_label']
        and close(o['parameters'].get('concentration_M'),s.get('concentration_M')) for o in ops) for s in streams if s['phase']!='gas')
    for stage in stages:
        matches=[o for o in ops if o['op_type'] in {'coil_reactor','photoreactor','reactor','heated_coil'} and o.get('parameters',{}).get('inventory_equipment_id')==stage['reactor_equipment_id']]
        checks[f'stage{stage["stage_number"]}_diagram_time']=any(close(o['parameters'].get('residence_time_inlet_min',o['parameters'].get('residence_time_min')),stage['residence_time_min']) for o in matches)
    report=dict(checks=checks,all_passed=all(checks.values()),failed=[k for k,v in checks.items() if not v],
        status='screening proposal; not wet-lab validated',stage_rows=stage_rows,component_rows=component_rows)
    (folder/'independent_checks.json').write_text(json.dumps(report,indent=2))
    for name,rows in [('stage_summary',stage_rows),('component_feeds',component_rows)]:
        if rows:
            with (folder/f'{name}.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'outputs/khu_revised_six_20260914'
    for p in sorted(root.glob('figure*/attempt*/result.json')):
        r=check(p)
        print(p.parent.relative_to(root),r['all_passed'],r['failed'])
