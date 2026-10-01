"""Re-layout archived SVG node groups without changing values or connectivity."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import cairosvg
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flora_design.visualizer.flowsheet_builder import FlowsheetBuilder, _main_process_path
from flora_translate.diagram_artifacts import _inline_svg_images
from flora_translate.schemas import ProcessTopology

OUT = ROOT / "manuscript/layout_revision_20260928"
FIG = OUT / "figures"
S = "{http://www.w3.org/2000/svg}"
NS = {"s": S[1:-1]}


def number(value):
    return float(re.sub(r"[^0-9.eE+-]", "", value))


def node_geometry(group):
    images = group.findall(S + "image")
    assert len(images) == 1, "Each selected archived node must have one equipment icon"
    im = images[0]
    x, y, width, height = [number(im.get(k)) for k in ["x", "y", "width", "height"]]
    cx, cy = x + width/2, y + height/2
    left, top, right, bottom = x, y, x+width, y+height
    for t in group.findall(S + "text"):
        size = number(t.get("font-size", "12"))
        tx, ty = number(t.get("x")), number(t.get("y"))
        # Conservative text bounds avoid adjacent labels colliding.
        tw = max(1, len("".join(t.itertext()))) * size * 0.70
        anchor = t.get("text-anchor", "start")
        tx -= tw/2 if anchor == "middle" else tw if anchor == "end" else 0
        left, right = min(left, tx), max(right, tx+tw)
        top, bottom = min(top, ty-size), max(bottom, ty+size*0.25)
    return {"cx": cx, "cy": cy, "iw": width, "ih": height, "left": left-cx, "right": right-cx, "top": top-cy, "bottom": bottom-cy}


def signature(groups):
    return {k: {"texts": ["".join(t.itertext()) for t in g.findall(S + "text")], "icons": [sha256((n.get("{http://www.w3.org/1999/xlink}href") or n.get("href") or "").encode()).hexdigest() for n in g.findall(S + "image")]} for k, g in groups.items()}


def reflow(svg_source, json_source, name):
    payload = json.loads(json_source.read_text())
    old = E.parse(str(svg_source)).getroot()
    nodes = {g.find(S + "title").text: g for g in old.findall(".//" + S + "g") if g.get("class") == "node"}
    old_edges = [g for g in old.findall(".//" + S + "g") if g.get("class") == "edge"]
    edges = []
    for g in old_edges:
        title = g.find(S + "title").text
        source, target = title.split("->")
        edges.append((source.split(":")[0], target.split(":")[0], g))
    ops = [SimpleNamespace(**o) for o in payload["unit_operations"]]
    streams = [SimpleNamespace(**s) for s in payload["streams"]]
    sanitize = lambda s: s.replace("-", "_").replace(" ", "_")
    main = [sanitize(i) for i in _main_process_path(ops, streams)]
    main = [i for i in main if i in nodes]
    assert len(main) > 2
    main_edges = set(zip(main, main[1:]))
    assert main_edges.issubset({(s,t) for s,t,_ in edges})
    geometry = {key: node_geometry(g) for key, g in nodes.items()}
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    for source, target, _ in edges:
        successors[source].append(target)
        predecessors[target].append(source)
    columns = {key: i for i, key in enumerate(main)}
    branch_chains = []
    remaining = set(nodes) - set(main)
    for join in main:
        for last in predecessors[join]:
            if last not in remaining:
                continue
            chain = [last]
            while predecessors[chain[0]]:
                previous = [p for p in predecessors[chain[0]] if p in remaining]
                assert len(previous) == 1, "Unexpected branching; requires explicit layout"
                chain.insert(0, previous[0])
            assert columns[join] >= len(chain), "Insufficient preceding columns"
            for i, key in enumerate(chain):
                columns[key] = columns[join] - len(chain) + i
                remaining.remove(key)
            branch_chains.append((chain, join))
    assert not remaining, remaining
    # These archived examples each have exactly one secondary-feed branch.
    assert len(branch_chains) == 1
    half_left = [max(-geometry[k]["left"] for k in nodes if columns[k] == i) for i in range(len(main))]
    half_right = [max(geometry[k]["right"] for k in nodes if columns[k] == i) for i in range(len(main))]
    xs, cursor = [], 18.0
    for left, right in zip(half_left, half_right):
        xs.append(cursor + left)
        cursor += left + right + 28
    branch = branch_chains[0][0]
    branch_y = 18 + max(-geometry[k]["top"] for k in branch)
    main_y = branch_y + max(geometry[k]["bottom"] for k in branch) + 42 + max(-geometry[k]["top"] for k in main)
    positions = {k: (xs[columns[k]], main_y if k in main else branch_y) for k in nodes}
    width = cursor - 28 + 18
    height = main_y + max(geometry[k]["bottom"] for k in main) + 18
    root = E.Element(S + "svg", nsmap={None: S[1:-1], "xlink": "http://www.w3.org/1999/xlink"}, width=f"{width:.3f}pt", height=f"{height:.3f}pt", viewBox=f"0 0 {width:.3f} {height:.3f}")
    E.SubElement(root, S + "title").text = name + ": archived topology, straightened layout"
    E.SubElement(root, S + "rect", x="0", y="0", width=str(width), height=str(height), fill="white")
    new_nodes, line_paths = {}, []
    for source, target, original in edges:
        sx, sy = positions[source]
        tx, ty = positions[target]
        a = (sx + geometry[source]["iw"]/2 + 4, sy)
        vertical = (source, target) not in main_edges and target in main
        if vertical:
            b = (tx, ty-geometry[target]["ih"]/2-5)
            points = [a, (tx, sy), b]
            arrow = [(b[0], b[1]), (b[0]-3, b[1]-7), (b[0]+3, b[1]-7)]
            points[-1] = (b[0], b[1]-7)
        else:
            assert abs(sy-ty) < 1e-8
            b = (tx-geometry[target]["iw"]/2-5, ty)
            points = [a, (b[0]-7, b[1])]
            arrow = [(b[0], b[1]), (b[0]-7, b[1]-3), (b[0]-7, b[1]+3)]
        assert all(abs(a[0]-b[0]) < 1e-8 or abs(a[1]-b[1]) < 1e-8 for a,b in zip(points,points[1:]))
        group = E.SubElement(root, S + "g", {"class": "edge", "data-source": source, "data-target": target})
        E.SubElement(group, S + "title").text = original.find(S + "title").text
        path = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x,y in points)
        E.SubElement(group, S + "path", d=path, fill="none", stroke="#27408b", **{"stroke-width": "1.6", "stroke-linejoin": "round"})
        E.SubElement(group, S + "polygon", points=" ".join(f"{x:.3f},{y:.3f}" for x,y in arrow), fill="#27408b")
        line_paths.append({"source": source, "target": target, "points": points})
    for key, original in nodes.items():
        g = deepcopy(original)
        cx, cy = positions[key]
        g.set("transform", f"translate({cx-geometry[key]['cx']:.6f} {cy-geometry[key]['cy']:.6f})")
        root.append(g)
        new_nodes[key] = g
    assert signature(nodes) == signature(new_nodes)
    overlaps = []
    boxes = {k: (positions[k][0]+g["left"], positions[k][1]+g["top"], positions[k][0]+g["right"], positions[k][1]+g["bottom"]) for k,g in geometry.items()}
    for i, key in enumerate(nodes):
        a = boxes[key]
        for other in list(nodes)[i+1:]:
            b = boxes[other]
            if min(a[2],b[2]) > max(a[0],b[0]) and min(a[3],b[3]) > max(a[1],b[1]):
                overlaps.append([key, other])
    assert not overlaps
    target = FIG / (name + ".svg")
    target.write_bytes(E.tostring(root, xml_declaration=True, encoding="UTF-8"))
    cairosvg.svg2png(url=str(target), write_to=str(target.with_suffix(".png")), output_width=3900)
    cairosvg.svg2pdf(url=str(target), write_to=str(target.with_suffix(".pdf")))
    return {"figure": name, "source_svg": str(svg_source.relative_to(ROOT)), "source_json": str(json_source.relative_to(ROOT)), "source_json_sha256": sha256(json_source.read_bytes()).hexdigest(), "source_svg_sha256": sha256(svg_source.read_bytes()).hexdigest(), "svg": str(target.relative_to(ROOT)), "png": str(target.with_suffix('.png').relative_to(ROOT)), "main_path": main, "main_icon_center_y": main_y, "node_positions": positions, "source_node_text_and_icon_preserved": True, "edges": line_paths, "label_box_overlaps": overlaps}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    source13 = ROOT / "deliverables/manuscript_benchmark_visualizations_20260825/figures_revised/topology_atlas/source_json/photochemical_oxidation/claude_opus_46/repeat_01.json"
    svg13 = FIG / "S13_gui_renderer_source.svg"
    payload = json.loads(source13.read_text())
    topology = ProcessTopology.model_validate(payload)
    original = topology.model_dump(mode="json")
    renderer = FlowsheetBuilder()
    renderer.build(topology, output_svg=str(svg13), output_png=str(svg13.with_suffix('.png')))
    assert renderer.last_render_info["renderer"] == "graphviz"
    assert original == topology.model_dump(mode="json")
    _inline_svg_images(svg13)
    # The old record stores oxygen equivalents in its rationale, not a modern field.
    gas = next(o for o in payload["unit_operations"] if o["op_id"] == "pump_b")
    equivalent = re.search(r"supplied=([0-9.]+) equiv", gas["rationale"])
    assert equivalent
    source13_svg = E.parse(str(svg13))
    gas_node = source13_svg.xpath('//s:g[@class="node"][s:title="pump_b"]', namespaces=NS)[0]
    last = gas_node.findall(S + "text")[-1]
    label = deepcopy(last)
    label.set("y", str(number(last.get("y"))+14))
    label.text = equivalent[1] + " equiv O2 (inlet/STP)"
    gas_node.append(label)
    svg13.write_bytes(E.tostring(source13_svg, xml_declaration=True, encoding="UTF-8"))
    rows = [reflow(svg13, source13, "Figure_S13_straight")]
    p = ROOT / "outputs/gui_runs/20260907_175308_three_protocol_scientific"
    # Keep archived labels intact, but avoid a syringe glyph misidentifying HPLC pumps.
    s17 = E.parse(str(p / "process.svg"))
    source_icons = E.parse(str(svg13))
    generic = source_icons.xpath('//s:g[@class="node"][s:title="pump_a"]/s:image', namespaces=NS)[0]
    href = "{http://www.w3.org/1999/xlink}href"
    changed_icons = []
    for node in s17.xpath('//s:g[@class="node"]', namespaces=NS):
        key = node.find(S + "title").text
        if key in {"pump_a", "pump_b"}:
            node.find(S + "image").set(href, generic.get(href))
            changed_icons.append(key)
    assert len(changed_icons) == 2
    svg17 = FIG / "S17a_display_source.svg"
    svg17.write_bytes(E.tostring(s17, xml_declaration=True, encoding="UTF-8"))
    row17 = reflow(svg17, p / "topology.json", "Figure_S17a_straight")
    row17.update(archived_svg=str((p / "process.svg").relative_to(ROOT)), display_updates={"generic_pump_icons": changed_icons, "archived_labels_unchanged": True})
    rows.append(row17)
    for fig, folder, case in [
        (20, ROOT / "outputs/figure5_pure_oxygen_physics_20260918/slides", "figure5"),
        (21, ROOT / "outputs/khu_revised_six_20260915/collaborator_slides_20260915_163845", "figure6")]:
        for i in range(1,4):
            p = folder / f"{case}_set{i}"
            js = p / ("display_topology.json" if (p / "display_topology.json").exists() else "process_topology.json")
            rows.append(reflow(p / "topology.svg", js, f"Figure_S{fig}{chr(96+i)}_straight"))
    (OUT / "figure_layout_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Rendered {len(rows)} straightened panels; archived source files unchanged.")


if __name__ == "__main__":
    main()
