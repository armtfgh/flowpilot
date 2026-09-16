"""Updated workbook plus KHU's explicit module and pump-setting clarifications."""
import hashlib
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_khu_inventory_20260914 import build as base_build
from flora_translate.inventory_profiles import InventoryProfile, InventorySource

FOLDER = ROOT / 'inventory_khu'
REVIEW = FOLDER / 'source_review_20260915'
DESTINATION = FOLDER / 'KHU_inventory_20260915_v6.json'


def build(write=True):
    profile = base_build(REVIEW / 'source_extraction.json', destination=False)
    catalog = profile.extraction_metadata['source_catalog']
    profile.profile_id = 'khu_laboratory_inventory_20260915'
    profile.name = 'KHU Laboratory Inventory - confirmed modules and pump increments'
    profile.version = 6
    profile.extraction_metadata['source_review'] = str(REVIEW.relative_to(ROOT))
    for reactor in profile.lab_inventory.reactors:
        row = int(reactor.equipment_id.removeprefix('coil_row'))
        value = catalog['Reactor (tubing)'][row]['K']
        tokens = {t.strip().lower() for t in value.split(',')}
        assert tokens <= {'uv-150', 'manual photoreactor'}, value
        reactor.photoreactor_module_ids = (['uv150'] if 'uv-150' in tokens else []) + (['manual1', 'manual2'] if 'manual photoreactor' in tokens else [])
        reactor.notes = f'Reactor (tubing)!K{row}: {value}. Coil stock counted once, regardless of possible mounting module. Module capacity and temperature limits still apply.'
    platform_rows = {'e_series': 3, 'sf10': 4, 'r_series': 5, 'r2c': 6}
    for pump in profile.lab_inventory.pumps:
        if 'syringe_row' in pump.equipment_id:
            row = int(re.search(r'row(\d+)', pump.equipment_id)[1])
            step = catalog['Syringe'][row]['H']
            cell = f'Syringe!H{row}'
        else:
            row = platform_rows[pump.platform_id]
            step = catalog['Pump'][row]['G']
            cell = f'Pump!G{row}'
        pump.flow_rate_increment_mL_min = float(step)
        pump.notes += f' Setting increment {step:g} mL/min from {cell}, confirmed in KHU email; distinct from minimum usable flow.'
    profile.operating_constraints.update({
        'equipment_assembly_policy': 'Mount tubing only in its declared compatible module, respecting module capacity, temperature and shared quantity. UV-150 lights are not detachable: use UV-150 with an E-series or R-series platform; external syringe cofeeds are permitted when necessary. Manual modules use their own strip lights.',
        'pump_platform_policy': 'Prefer compatible channels within one system. Mixed platforms are permitted but disfavored; justify necessity, especially reagent incompatibility requiring an external syringe pump. Do not force every UV-150 feed onto system pumps.',
        'pump_setting_policy': 'Minimum flow is the lower operating limit. Adjustment flow is the setting increment, represented as multiples of that increment; jointly solve all feeds without changing stated stoichiometry.',
    })
    clarification = REVIEW / 'khu_clarification.md'
    profile.provenance.append(InventorySource(source_id='khu_email_20260915', filename=clarification.name,
        media_type='text/markdown', sha256=hashlib.sha256(clarification.read_bytes()).hexdigest(),
        notes='User-supplied KHU clarification; source text archived alongside workbook extraction.'))
    profile = InventoryProfile.model_validate(profile.model_dump())
    if write:
        DESTINATION.write_text(profile.model_dump_json(indent=2), encoding='utf-8')
        print(DESTINATION)
    return profile


if __name__ == '__main__':
    build()
