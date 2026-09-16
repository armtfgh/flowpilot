"""Bind manual/UV-150 assemblies using an explicitly confirmed installed reactor."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flora_translate.inventory_profiles import load_inventory_profile
from flora_translate.schemas import ReactorSpec


def corrected_profile(previous, definition):
    required={'material','volume_mL','ID_mm','max_pressure_bar','quantity','source',
              'separate_from_manual_stock'}
    if required-definition.keys():
        raise ValueError('Missing confirmed installed-reactor fields: '+', '.join(sorted(required-definition.keys())))
    if not definition['source'] or type(definition['separate_from_manual_stock']) is not bool:
        raise ValueError('A source and explicit physical stock ownership are required')
    if definition['volume_mL']>10 or definition['volume_mL']<=0 or definition['ID_mm']<=0:
        raise ValueError('Installed reactor geometry must be positive and fit the 10 mL UV-150 capacity')
    profile=previous.model_copy(deep=True)
    profile.version=6
    profile.name='KHU revised inventory - assembly-bound v6'
    profile.status='draft'
    for reactor in profile.lab_inventory.reactors:
        reactor.system='manual'
        reactor.photoreactor_module_ids=['manual1','manual2']
        reactor.name='Manual '+reactor.name
        reactor.notes='Workbook reactor stock; manual illumination only per user assembly clarification.'
    shared_id=definition.get('shared_with_manual_equipment_id')
    if not definition['separate_from_manual_stock']:
        shared=next((r for r in profile.lab_inventory.reactors if r.equipment_id==shared_id),None)
        if shared is None or any(getattr(shared,k)!=definition[k] for k in ('material','volume_mL','ID_mm','quantity')):
            raise ValueError('Shared installed coil must exactly match a declared manual-stock entry')
        shared.service_status='integrated_component'
        shared.notes+=' Assigned exclusively to the UV-150 assembly, not additional manual stock.'
    elif shared_id:
        raise ValueError('A separate reactor cannot also claim a shared manual-stock item')
    profile.lab_inventory.reactors.append(ReactorSpec(
        equipment_id='reactor_uv150_integrated',name='Vapourtec UV-150 integrated reactor',
        type='coil',system='uv150',photoreactor_module_ids=['uv150'],
        material=definition['material'],volume_mL=definition['volume_mL'],ID_mm=definition['ID_mm'],
        quantity=definition['quantity'],min_temperature_C=25,max_temperature_C=80,
        max_pressure_bar=definition['max_pressure_bar'],
        notes='Dedicated reactor and associated UV-150 illumination; not a loose manual coil. '+definition['source']))
    profile.operating_constraints['equipment_assembly_policy']=(
        'UV-150 is an integrated reactor/light assembly. Only reactor_uv150_integrated may use uv150 lights. '
        'Manual coils may use only their explicitly permitted manual1/manual2 modules. '
        'Pump selection never grants reactor/light compatibility. Each module is available once.')
    profile.extraction_metadata['assembly_correction']={'reason':'User clarified integrated UV-150 versus manual reactor ownership.',
        'installed_uv150_definition':definition,'supersedes':'KHU_inventory_20260914_v5.json',
        'previous_error':'Manual coil system was empty and UV-150 had only light entries.'}
    profile.validation.valid=True
    return profile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--confirmed-uv150',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'inventory_khu/KHU_inventory_20260914_v6.json')
    args=parser.parse_args()
    old=load_inventory_profile(ROOT/'inventory_khu/KHU_inventory_20260914_v5.json')
    profile=corrected_profile(old,json.loads(args.confirmed_uv150.read_text()))
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(profile.model_dump_json(indent=2))
    print(args.output)


if __name__=='__main__':main()
