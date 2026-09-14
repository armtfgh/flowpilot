"""Reproducible workbook conversion with explicit unresolved source fields."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flora_translate.inventory_profiles import InventoryProfile
from flora_translate.schemas import LabInventory

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "inventory_khu"


def build():
    source = json.loads((FOLDER / "source_review_20260914/source_extraction.json").read_text())
    sheets = {s['sheet']: {r['row']: {c['cell'].rstrip('0123456789'): c['value'] for c in r['cells']}
                           for r in s['rows']} for s in source['sheets']}
    inventory = dict(strict_assignment=True, resource_capacities={}, pumps=[], reactors=[], tubing=[],
                     light_sources=[], mixers=[], gas_hardware=[], pressure_controllers=[], safety_accessories=[],
                     capability_status={'inline_degasser': 'unavailable'}, standard_reactor_connectors_available=True)
    resources = inventory['resource_capacities']
    for row, platform, channels in [(3,'e_series',3),(4,'sf10',1),(5,'r_series',2),(6,'r2c',2)]:
        v = sheets['Pump'][row]
        resources[platform] = channels
        for colour in (['blue','red'] if row < 5 else ['']):
            exclusions = ['n-BuLi','LDA','LAH','slurry','suspension'] if row >= 5 else []
            if colour == 'red':
                exclusions += ['2-MeTHF','2-methyltetrahydrofuran','acrylonitrile']
            inventory['pumps'].append(dict(equipment_id=f'pump_{platform}_{colour or "hplc"}',
                name=v['B'].strip() + (f' / {colour.upper()} pump tubing' if colour else ''),
                type=v['C'], quantity=channels, platform_id=platform,
                min_flow_rate_mL_min=v['E'], max_flow_rate_mL_min=v['F'], max_pressure_bar=v['J'],
                resource_requirements={platform:1}, excluded_chemicals=exclusions,
                notes=f'Pump!row {row}. One system; shared channels across tubing configurations. Full mixture compatibility requires confirmation.'))
    resources.update(chemyx=3, psd4=2)
    for row in range(3,12):
        v=sheets['Syringe'][row]
        platform = 'psd4' if row>=10 else 'chemyx'
        resources[f'syringe_row{row}'] = int(v['J'])
        inventory['pumps'].append(dict(equipment_id=f'pump_{platform}_syringe_row{row}',
            name=f'{"Hamilton PSD/4" if platform == "psd4" else "Chemyx Fusion 100"} / {v["B"]} {v["C"]} mL',
            type='syringe', platform_id=platform, quantity=min(resources[platform],int(v['J'])),
            min_flow_rate_mL_min=v['F'], max_flow_rate_mL_min=v['G'],
            max_pressure_bar=float(str(v['I']).replace('<','').strip()) - 1e-6,
            resource_requirements={platform:1,f'syringe_row{row}':1},
            notes=f'Syringe!row {row}. Strict pressure bound {v["I"]} bar. One independently controlled stream per pump unit; equal-flow paired Chemyx use is not enumerated.'))
    for row in range(3,14):
        v=sheets['Reactor (tubing)'][row]
        if isinstance(v['D'],str):
            continue  # Fabrication stock is retained below, not fabricated into installed coils.
        entry=dict(equipment_id=f'coil_row{row}',name=f'{v["C"]} coil {v["D"]} mL / ID {v["E"]} mm',
            type='coil', material=v['C'],volume_mL=v['D'],ID_mm=v['E'],quantity=v['G'],
            max_temperature_C=v['I'],max_pressure_bar=v['J'],notes=f'Reactor (tubing)!row {row}; tubing independent of illumination module.')
        inventory['reactors'].append(entry)
        inventory['tubing'].append(dict(equipment_id=f'tubing_row{row}',name=entry['name']+' flow path',
            material=v['C'], ID_mm=v['E'],quantity=v['G'],max_temperature_C=v['I'],max_pressure_bar=v['J'],
            notes='Same installed coil flow path; not additional loose tubing stock.'))
    for row in [3,*range(6,13),13]:
        v=sheets['Photoreactor module'][row]
        module='manual1' if row==3 else 'manual2' if row==13 else 'uv150'
        resources[module]=1
        wave=v['F'] if isinstance(v['F'],(int,float)) else float(v['E'].split()[0])
        inventory['light_sources'].append(dict(equipment_id=f'light_{module}_row{row}',name=f'{v["B"]} / {v["E"]}',
            wavelength_nm=wave,power_W=float(v['H'].split()[0]) if isinstance(v['H'],str) and 'W' in v['H'] else None,
            compatible_reactor='coil',quantity=v['N'],module_id=module,module_name=v['B'],
            resource_requirements={module:1},max_reactor_volume_mL=v['J'],
            min_temperature_C=v['K'],max_temperature_C=v['L'],allowed_temperatures_C=[40,50] if module=='manual1' else [],
            intensity_mW_cm2=v['I'] if isinstance(v['I'],(int,float)) else None,
            compatible_pump_platforms=['e_series','r_series'] if module=='uv150' else [],
            notes=f'Photoreactor module!row {row}. One source installed per module; wavelength {"measured" if isinstance(v["F"],(int,float)) else "nominal"}.'))
    for row in range(3,9):
        v=sheets['BPR'][row]
        inventory['pressure_controllers'].append(dict(equipment_id=f'bpr_row{row}',name=f'BPR {v["D"]} bar',
            quantity=v['E'],type='adjustable' if row==3 else 'fixed',
            max_pressure_bar=8 if row==3 else v['D'],setpoints_bar=[] if row==3 else [v['D']]))
    inventory['gas_hardware'].append(dict(equipment_id='mfc_ffc00',name='Bronkhorst FLEXI-FLOW Compact FF-C00',
        type='MFC',gas='O2, N2, Air, CO2',min_flow_sccm=.01,max_flow_sccm=10,max_pressure_bar=9,
        required_outlet_accessory_type='check_valve',
        notes='MFC!row3. One listed instrument assumed; quantity and pressure reference basis need confirmation. Inlet 10 bar, outlet 9 bar as supplied.'))
    for row in range(3,10):
        v=sheets['Accessories'][row]
        if 'mixer' in v['C'].lower():
            inventory['mixers'].append(dict(equipment_id=v['B'].lower(),name=v['B']+' '+v['C'],type=v['C'],
                material=v['D'],quantity=v['E'],max_inputs=3 if row==9 else 2,max_pressure_bar=34))
        else:
            inventory['safety_accessories'].append(dict(equipment_id=v['B'].lower(),name=v['B'],
                type='check_valve' if row==3 else 'switching_valve',quantity=v['E'],
                cracking_pressure_bar=1 if row==3 else None,max_pressure_bar=None if row==3 else 34,
                notes=f'{v["D"]}; {v["F"]}; {v["G"]}'))
    profile=InventoryProfile(profile_id='khu_laboratory_inventory_20260914',name='KHU revised inventory - September 2026',
        laboratory='KHU',version=5,lab_inventory=LabInventory(**inventory),
        equipment_capabilities={'inline_degasser':{'available':False,'notes':'Earlier explicit KHU constraint, retained separately from workbook.'}},
        operating_constraints={
            'inline_degasser_available':False,
            'equipment_assembly_policy':'Manual modules use only their own strip LEDs. UV-150 uses its own lights and E/R-series platform. One module cannot illuminate two separate coils simultaneously.',
            'pump_platform_policy':'Prefer channels of one compatible pumping system. Report justification if crossing platforms.',
            'reporting':'Report each feed component concentration, molar flow and equivalent; distinguish theoretical intermediate concentration. Report gas volume at inlet STP in mL/min.',
            'catalog_projection_limits':['Fabricatable tubing remains stock, not installed reactor candidates; strict < versus up-to volume needs confirmation.',
                'Red/green strip wavelengths unspecified: retained in source catalog, not wavelength-selectable candidates.',
                'Integrated H-Cube retained in source catalog, not decomposed into external pump/MFC/BPR.',
                'Unknown check-valve pressure rating and MFC quantity require laboratory verification.',
                'Compatibility PDF is not a certification for unlisted solutes or mixtures.']},
        provenance=[dict(source_id=f'source_{i}',**s) for i,s in enumerate(source['sources'],1)],
        extraction_metadata={'source_review':'inventory_khu/source_review_20260914','no_llm_extraction':True,'source_catalog':sheets})
    profile.validation.warnings=profile.operating_constraints['catalog_projection_limits']
    profile.validation.valid=True
    destination=FOLDER/'KHU_inventory_20260914_v5.json'
    destination.write_text(profile.model_dump_json(indent=2),encoding='utf-8')
    print(destination)
    print('Schema-valid draft; unresolved laboratory confirmations retained.')
    return profile


if __name__=='__main__':
    build()
