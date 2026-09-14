"""Shared physical resources and stage-local photoreactor compatibility."""

from collections import Counter


def resources_fit(items, capacities):
    used = Counter()
    for item in items:
        used.update(item.resource_requirements)
    return all(key in capacities and count <= capacities[key] for key, count in used.items())


def light_fits_stage(light, reactor, proposal, inventory):
    if light.max_reactor_volume_mL is not None and reactor.volume_mL > light.max_reactor_volume_mL + 1e-9:
        return False
    # Bare tubing is platform-independent; a named assembly is not.
    system = str(reactor.system or "").lower()
    if light.module_id and system and system not in {"independent", "platform-independent", "tubing"}:
        if system not in {light.module_id.lower(), light.module_name.lower()}:
            return False
    if light.compatible_pump_platforms:
        pumps = {p.equipment_id: p for p in inventory.pumps}
        platforms = {pumps[s.pump_equipment_id].platform_id for s in proposal.streams
                     if s.phase != "gas" and s.pump_equipment_id in pumps}
        if not platforms.intersection(light.compatible_pump_platforms):
            return False
    return True


def pump_accepts_feed(pump, stream):
    text = " ".join([*stream.contents, stream.solvent, stream.pump_role]).lower()
    return not any(chemical.lower() in text for chemical in pump.excluded_chemicals)
