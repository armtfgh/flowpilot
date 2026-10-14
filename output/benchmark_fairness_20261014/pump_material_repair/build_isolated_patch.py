"""Build an unapplied candidate patch; do not mutate production or frozen records."""
from pathlib import Path
import difflib, hashlib
base=Path(__file__).resolve().parent;root=base.parents[2]
patches=[]
def write_candidate(name, original, patched):
    (base/f'candidate_{name}.py').write_text(patched)
    (base/f'production_{name}_sha256.txt').write_text(hashlib.sha256((root/f'flora_translate/{name}.py').read_bytes()).hexdigest()+'\n')
    patches.append(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile=f'a/flora_translate/{name}.py',tofile=f'b/flora_translate/{name}.py')))
path=root/'flora_translate/main.py';original=path.read_text()
old='''        operation.parameters["contents"] = labels or list(feed.contents)
    return normalize_topology_semantics(topology)
'''
new='''        # Preserve explicit same-feed solvent evidence during solute rendering.
        # Do not inherit another feed's or the batch solvent.
        solvent_components = [component for component in components
                              if component.stream_label == feed.stream_label
                              and component.role == "solvent"]
        solvent_labels = list(dict.fromkeys(component.name for component in solvent_components))
        operation.parameters["declared_solvent_components"] = [
            {"name": component.name, "source_text": component.source_text,
             "authority": "explicit_same_feed_component"}
            for component in solvent_components
        ]
        operation.parameters["contents"] = (labels + solvent_labels) or list(feed.contents)
        if not feed.solvent and len(solvent_labels) == 1:
            operation.parameters["solvent"] = solvent_labels[0]
            operation.parameters["solvent_provenance"] = {
                "authority": "explicit_same_feed_component",
                "stream_label": feed.stream_label,
                "source_text": [component.source_text for component in solvent_components],
            }
    return normalize_topology_semantics(topology)
'''
assert original.count(old)==1;write_candidate('main',original,original.replace(old,new))
path=root/'flora_translate/inventory_allocator.py';original=path.read_text()
old='and _pump_material_compatible(item, stream_identity)'
new='''and _pump_material_compatible(
                    item, stream_identity,
                    solvent=str(operation.parameters.get("solvent") or ""),
                    solvent_components=[str(entry.get("name") or "") for entry in
                                        operation.parameters.get("declared_solvent_components") or []],
                )'''
assert original.count(old)==1;patched=original.replace(old,new)
old='''def _pump_material_compatible(pump, stream_identity: str) -> bool:
    """Require a declared wetted-material match when a pump lists one."""

    if not pump.compatible_materials:
        return True
'''
new='''def _explicit_medium_members(value: str) -> list[str]:
    """Split explicit medium lists; ratios are amounts, not extra identities.

    Unrecognized descriptions remain literal and cannot gain compatibility from
    an incidental known-solvent substring.
    """
    text = re.sub(r"\\(\\s*\\d+(?:\\.\\d+)?(?:\\s*:\\s*\\d+(?:\\.\\d+)?)+\\s*(?:v/v)?\\s*\\)", "", value, flags=re.I)
    text = re.sub(r"\\s+\\d+(?:\\.\\d+)?(?:\\s*:\\s*\\d+(?:\\.\\d+)?)+\\s*(?:v/v)?\\s*$", "", text, flags=re.I)
    return [part.strip() for part in re.split(r"\\s*(?:/|:|,|\\+|\\band\\b)\\s*", text) if part.strip()]


def _pump_material_compatible(pump, stream_identity: str, *, solvent: str = "", solvent_components: list[str] | None = None) -> bool:
    """Require every explicit medium identity when a pump lists compatibility."""

    if not pump.compatible_materials:
        return True
    from flora_translate.component_identity import declared_solvent_member
    explicit_members = [member for value in [solvent, *(solvent_components or [])]
                        for member in _explicit_medium_members(value)]
    if explicit_members:
        # A matching solute cannot authorize an incompatible or unknown medium.
        # Conflicting solvent declarations and mixtures are checked in full.
        return all(any(declared_solvent_member(member, str(material))
                       for material in pump.compatible_materials)
                   for member in explicit_members)
'''
assert original.count(old)==1;patched=patched.replace(old,new)
write_candidate('inventory_allocator',original,patched)
(base/'preserve_feed_solvent.patch').write_text(''.join(patches))
print('Unapplied candidates written; production sources untouched.')
