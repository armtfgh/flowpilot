"""
FLORA — Graphviz-based Flowsheet Builder.

Replaces the old hand-drawn SVG approach with a professional Graphviz diagram
using real equipment icons (syringe pump, coil reactor, BPR, vial, etc.).

Icon files live in:  flora_design/visualizer/icons/

Features:
  - Pump nodes: syringe icon + reagent list + solvent + flow rate (fully readable)
  - Reactor nodes: coil image + temp / residence time / wavelength / ID / volume
  - Mixer nodes: clean grey box with correct in/out ports
  - BPR / collector: icon + label
  - LED module: SKIPPED — wavelength shown under reactor
  - 3+ streams into one mixer: auto-chained into two mixers in series
  - Unknown op types: plain text box

Falls back silently to the legacy SVG builder if graphviz executable
is not found on the system.

Public API (unchanged from legacy):
    FlowsheetBuilder().build(topology, title, output_svg, output_png)
    → (svg_path, png_path)
"""

from __future__ import annotations

import logging
import math
import shutil
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Optional

logger = logging.getLogger("flora.flowsheet")

# ── Paths ─────────────────────────────────────────────────────────────────────
ICONS_DIR = Path(__file__).parent / "icons"

ASSETS = {
    "pump":         ICONS_DIR / "syringe_pump.png",
    "reactor":      ICONS_DIR / "coil_reactor.png",
    "photoreactor": ICONS_DIR / "photoreactor.png",
    "microchannel": ICONS_DIR / "microchannel.png",
    "bpr":          ICONS_DIR / "bpr.png",
    "vial":         ICONS_DIR / "vial.png",
    "mixer2":       ICONS_DIR / "mixer2.png",
    "mixer3":       ICONS_DIR / "mixer3.png",
}

# ── Graph-level style ─────────────────────────────────────────────────────────
GRAPH_ATTR = {
    "rankdir":     "LR",
    "splines":     "polyline",    # respects headport corners — clean nw/sw routing
    "nodesep":     "0.35",        # compact: less vertical gap between nodes
    "ranksep":     "0.6",         # compact: less horizontal gap between columns
    "pad":         "0.2",
    "dpi":         "180",
    "bgcolor":     "white",
    "forcelabels": "true",
    "fontname":    "Arial",
}

EDGE_ATTR = {
    "color":     "royalblue4",
    "penwidth":  "2.0",
    "arrowsize": "0.55",          # compact: slightly smaller arrow heads
    "fontname":  "Arial",
    "fontsize":  "9",
    "fontcolor": "royalblue4",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _esc(s) -> str:
    return escape(str(s)) if s is not None else ""


def _html_lines(*lines: str, sizes: list[int] | None = None,
                colors: list[str] | None = None) -> str:
    """Build multi-line HTML label content as <BR/>-joined FONT tags."""
    parts = []
    for i, line in enumerate(lines):
        if not line:
            continue
        sz    = sizes[i] if sizes and i < len(sizes) else 10
        color = colors[i] if colors and i < len(colors) else "#111827"
        parts.append(
            f'<FONT POINT-SIZE="{sz}" COLOR="{color}">{_esc(line)}</FONT>'
        )
    return "<BR/>".join(parts)


def _trunc(s: str, n: int = 35) -> str:
    s = str(s).strip()
    return s if len(s) <= n else s[:n - 1] + "…"


# ── Gas stream detection ──────────────────────────────────────────────────────

_GAS_KEYWORDS = {
    "n2", "n₂", "nitrogen", "o2", "o₂", "oxygen", "co2", "co₂",
    "h2", "h₂", "hydrogen", "ar", "argon", "helium", "air", "gas", "mfc",
}

def _is_gas_pump(op) -> bool:
    """Return True if this pump carries a gas stream (use MFC node instead)."""
    p   = op.parameters or {}
    txt = " ".join([
        str(op.label or ""),
        " ".join(str(c) for c in (p.get("contents") or [])),
    ]).lower()
    return any(kw in txt for kw in _GAS_KEYWORDS)


# ── Label generators ──────────────────────────────────────────────────────────

def _pump_label(op) -> str:
    """
    Clean pump label: Pump letter, materials, solvent/conc,
    then flow rate in blue on its own line for visibility.
    """
    p      = op.parameters or {}
    stream = p.get("stream", "?")
    title  = f"Pump {stream}"

    # Materials line
    contents = p.get("contents") or []
    if isinstance(contents, str):
        contents = [contents]
    material_parts = []
    for item in contents[:3]:
        name = str(item).split("(")[0].strip()
        material_parts.append(_trunc(name, 22))
    if len(contents) > 3:
        material_parts.append(f"+{len(contents)-3}")
    materials_line = ",  ".join(material_parts) if material_parts else ""

    # Solvent · concentration line
    cond_parts = []
    if p.get("solvent"):
        cond_parts.append(str(p["solvent"]))
    if p.get("concentration_M") is not None:
        cond_parts.append(f"{float(p['concentration_M']):.2f} M")
    cond_line = "  ·  ".join(cond_parts)

    rows = [
        f'<FONT POINT-SIZE="10" COLOR="#111827"><B>{_esc(title)}</B></FONT>',
    ]
    if materials_line:
        rows.append(f'<FONT POINT-SIZE="8.5" COLOR="#1E40AF">{_esc(materials_line)}</FONT>')
    if cond_line:
        rows.append(f'<FONT POINT-SIZE="8" COLOR="#374151">{_esc(cond_line)}</FONT>')
    # Flow rate: separate blue line for visibility
    if p.get("flow_rate_mL_min") is not None:
        rows.append(
            f'<FONT POINT-SIZE="9" COLOR="#2563EB"><B>'
            f'{float(p["flow_rate_mL_min"]):.2f} mL/min</B></FONT>'
        )
    return "<BR/>".join(rows)


def _mfc_label(op) -> str:
    """Label for gas MFC node."""
    p      = op.parameters or {}
    stream = p.get("stream", "?")
    contents = p.get("contents") or []
    gas_name = str(contents[0]).split("(")[0].strip() if contents else "Gas"
    fr   = p.get("flow_rate_mL_min")
    rows = [
        f'<FONT POINT-SIZE="10" COLOR="#111827"><B>MFC {stream}</B></FONT>',
        f'<FONT POINT-SIZE="8.5" COLOR="#DC2626">{_esc(gas_name)}</FONT>',
    ]
    if fr is not None:
        rows.append(f'<FONT POINT-SIZE="8" COLOR="#374151">{float(fr):.2f} mL/min</FONT>')
    return "<BR/>".join(rows)


def _reactor_label(op) -> str:
    """Compact reactor label: temperature · wavelength · residence time only.
    No verbose description — the icon already communicates 'reactor'.
    """
    p = op.parameters or {}
    parts = []

    if p.get("temperature_C") is not None:
        parts.append(f"{p['temperature_C']}°C")
    if p.get("wavelength_nm") is not None:
        parts.append(f"λ={p['wavelength_nm']:.0f} nm")
    if p.get("residence_time_min") is not None:
        rt = float(p["residence_time_min"])
        parts.append(f"τ={rt:.1f} min" if rt >= 1 else f"τ={rt*60:.0f} s")

    return "  ·  ".join(parts) if parts else ""


def _bpr_label(op) -> str:
    p = op.parameters or {}
    bar = p.get("pressure_bar") or p.get("BPR_bar")
    return f"BPR\n  {bar:.0f} bar" if bar else "BPR"


_SHORT_LABELS = {
    "liq_liq_extraction": "L-L Separator",
    "liquid_liquid_extraction": "L-L Separator",
    "lle": "L-L Separator",
    "separator": "Separator",
    "phase_separator": "Phase Sep.",
    "deoxygenation_unit": "Degasser",
    "degas": "Degasser",
    "inline_filter": "Filter",
    "filter": "Filter",
    "quench_mixer": "Quench",
}

def _textbox_label(op) -> str:
    p   = op.parameters or {}
    ot  = op.op_type.lower().replace(" ", "_")
    lbl = _SHORT_LABELS.get(ot) or _SHORT_LABELS.get(op.op_type.lower()) or \
          op.label or op.op_type.replace("_", " ").title()

    # Add one short detail if meaningful
    detail = p.get("method") or p.get("reagent") or ""
    if detail and len(str(detail)) < 20:
        return lbl + "\n" + str(detail)
    return lbl


# ── Node builders ─────────────────────────────────────────────────────────────

NODE_ICON_SIZE = 110   # uniform icon size for all components

def _add_pump(dot, node_id: str, op, pump_img: Path):
    """Syringe pump node with needle port at right-center and clean label."""
    image_w   = NODE_ICON_SIZE
    image_h   = NODE_ICON_SIZE
    half_h    = image_h // 2
    port_h    = 2      # tiny port cell (graphviz attaches the edge here)

    label_html = _pump_label(op)
    label_rows = "".join(
        f'<TR><TD ALIGN="CENTER" WIDTH="{image_w}" CELLPADDING="1">{part}</TD></TR>'
        for part in label_html.split("<BR/>")
    )

    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        # Row 1: image (spans 3 rows) + space above needle
        "<TR>"
        f'<TD ROWSPAN="3" PORT="img" FIXEDSIZE="TRUE" WIDTH="{image_w}" HEIGHT="{image_h}">'
        f'<IMG SRC="{escape(pump_img.name)}" SCALE="TRUE"/></TD>'
        f'<TD WIDTH="0" HEIGHT="{half_h - port_h // 2}"></TD>'
        "</TR>"
        # Row 2: needle port (right side, vertically centred)
        "<TR>"
        f'<TD PORT="needle" WIDTH="0" HEIGHT="{port_h}"></TD>'
        "</TR>"
        # Row 3: space below needle
        "<TR>"
        f'<TD WIDTH="0" HEIGHT="{half_h - port_h // 2}"></TD>'
        "</TR>"
        # Label rows below the image
        f"{label_rows}"
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_mixer(dot, node_id: str, label: str,
               w_inch: float = 0.9, h_inch: float = 0.85):
    """T-mixer box with in_top / in_bottom / out ports."""
    w_pt = int(w_inch * 72)
    h_pt = int(h_inch * 72)
    half = h_pt // 2
    port_w = 10

    html = (
        '<<TABLE BORDER="1" COLOR="grey40" CELLBORDER="0" CELLSPACING="0"'
        ' CELLPADDING="3" BGCOLOR="#F1F5F9">'
        "<TR>"
        f'<TD PORT="in_top" WIDTH="{port_w}" HEIGHT="{half}"></TD>'
        f'<TD ROWSPAN="2" FIXEDSIZE="TRUE" WIDTH="{w_pt}" HEIGHT="{h_pt}">'
        f'<FONT POINT-SIZE="10" COLOR="#1E293B"><B>{_esc(label)}</B></FONT></TD>'
        f'<TD PORT="out" ROWSPAN="2" WIDTH="{port_w}" HEIGHT="{h_pt}"></TD>'
        "</TR><TR>"
        f'<TD PORT="in_bottom" WIDTH="{port_w}" HEIGHT="{half}"></TD>'
        "</TR></TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_image_node(dot, node_id: str, label: str, img_path: Path,
                    img_w: int = 125, img_h: int = 125):
    """Image node (reactor / BPR / vial / mixer) with caption centered below."""
    # Each label line gets its own row, centered, with explicit width = image width
    label_rows = "".join(
        f'<TR><TD ALIGN="CENTER" WIDTH="{img_w}"><FONT POINT-SIZE="9" COLOR="#111827">'
        f'{_esc(ln)}</FONT></TD></TR>'
        for ln in label.split("\n") if ln.strip()
    )
    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">'
        f'<TR><TD PORT="img" FIXEDSIZE="TRUE" WIDTH="{img_w}" HEIGHT="{img_h}">'
        f'<IMG SRC="{escape(img_path.name)}" SCALE="TRUE"/></TD></TR>'
        f'{label_rows}'
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_mixer_image(dot, node_id: str, n_inputs: int):
    """Mixer image node: mixer2 (T-mixer) for ≤2 inputs, mixer3 (cross) for 3+."""
    asset = ASSETS["mixer3"] if n_inputs >= 3 else ASSETS["mixer2"]
    sz = NODE_ICON_SIZE
    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD PORT="img" FIXEDSIZE="TRUE" WIDTH="{sz}" HEIGHT="{sz}">'
        f'<IMG SRC="{escape(asset.name)}" SCALE="TRUE"/></TD></TR>'
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_textbox(dot, node_id: str, op):
    """Styled text box with explicit left/right ports for clean arrow routing."""
    label = _textbox_label(op)
    content_rows = "".join(
        f'<TD ALIGN="CENTER"><FONT POINT-SIZE="9" COLOR="#1E293B">{_esc(ln)}</FONT></TD>'
        for ln in label.split("\n")
    )
    # Outer table: [left-port cell | content cell | right-port cell]
    # Named ports ensure arrows attach exactly to the left/right border edge.
    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        "<TR>"
        '<TD PORT="inp" WIDTH="1" HEIGHT="1"></TD>'
        '<TD>'
        '<TABLE BORDER="1" COLOR="#94A3B8" CELLBORDER="0" CELLSPACING="0"'
        ' CELLPADDING="8" BGCOLOR="#F8FAFC">'
        f'<TR>{content_rows}</TR>'
        '</TABLE>'
        '</TD>'
        '<TD PORT="out" WIDTH="1" HEIGHT="1"></TD>'
        "</TR>"
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


def _add_mfc_node(dot, node_id: str, op):
    """Mass Flow Controller node for gas streams.

    Rendered as a red-bordered rounded box with the gas name in red to
    visually distinguish it from syringe pumps. Exposes a `needle` port
    (same name as the syringe pump) so the edge router can stay generic.
    """
    label_html = _mfc_label(op)
    label_rows = "".join(
        f'<TR><TD ALIGN="CENTER">{part}</TD></TR>'
        for part in label_html.split("<BR/>")
    )
    # Outer table has two cells in the header row: the inner MFC box,
    # plus a tiny right-edge port cell named "needle" so edges attach
    # at the right side. Keeps the edge-routing code provider-agnostic.
    html = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        '<TR>'
        '<TD>'
        '<TABLE BORDER="2" COLOR="#DC2626" CELLBORDER="0" CELLSPACING="0"'
        ' CELLPADDING="8" BGCOLOR="#FFF5F5" STYLE="ROUNDED">'
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="9" COLOR="#7F1D1D"><B>MFC</B></FONT></TD></TR>'
        f'{label_rows}'
        '</TABLE>'
        '</TD>'
        '<TD PORT="needle" WIDTH="1" HEIGHT="1"></TD>'
        '</TR>'
        "</TABLE>>"
    )
    dot.node(node_id, label=html, shape="plain")


# ── Pump image prep ───────────────────────────────────────────────────────────

def _prepare_pump_img() -> Path:
    """Return a tight-cropped syringe pump image for cleaner rendering."""
    src = ASSETS["pump"]
    out = ICONS_DIR / "_pump_trimmed.png"
    if out.exists():
        return out
    try:
        from PIL import Image
        img = Image.open(src).convert("RGBA")
        coords = [
            (x, y)
            for y in range(img.height) for x in range(img.width)
            if img.getpixel((x, y))[3] > 0
            and not all(c > 245 for c in img.getpixel((x, y))[:3])
        ]
        if not coords:
            return src
        margin = 4
        min_x = max(min(x for x, _ in coords) - margin, 0)
        max_x = min(max(x for x, _ in coords) + margin + 1, img.width)
        min_y = max(min(y for _, y in coords) - margin, 0)
        max_y = min(max(y for _, y in coords) + margin + 1, img.height)
        img.crop((min_x, min_y, max_x, max_y)).save(out)
        return out
    except Exception as e:
        logger.warning(f"Pump trim failed: {e} — using original")
        return src


# ── Main builder ──────────────────────────────────────────────────────────────

class FlowsheetBuilder:
    """
    Builds a publication-quality process flow diagram from a ProcessTopology.

    Usage (identical API to legacy builder):
        svg_path, png_path = FlowsheetBuilder().build(topology, title=...,
                                                       output_svg=..., output_png=...)
    """

    def build(
        self,
        topology,
        title: str = "",
        output_svg: str = "flora_process.svg",
        output_png: str = "flora_process.png",
    ) -> tuple[str, str]:
        """Build and save the diagram. Returns (svg_path, png_path)."""
        # Graceful fallback to legacy builder if graphviz not available
        if not shutil.which("dot"):
            logger.warning("Graphviz 'dot' not found — falling back to legacy SVG builder")
            return self._legacy_fallback(topology, title, output_svg, output_png)

        try:
            return self._build_graphviz(topology, title, output_svg, output_png)
        except Exception as e:
            logger.error(f"Graphviz build failed: {e} — falling back to legacy builder", exc_info=True)
            return self._legacy_fallback(topology, title, output_svg, output_png)

    def _build_graphviz(self, topology, title, output_svg, output_png):
        import graphviz

        ops    = self._clean_ops(topology)
        if not ops:
            return "", ""

        pump_img = _prepare_pump_img()

        # Polish topology: remove logically redundant nodes (LLM + deterministic)
        try:
            from flora_translate.topology_polisher import polish
            polished = polish(topology, use_llm=True)
            ops = polished.unit_operations
            # Use polished streams for adjacency
            topology = polished
        except Exception as e:
            logger.warning(f"Topology polisher failed ({e}) — rendering as-is")

        active_ids = {o.op_id for o in ops}

        # ── Adjacency (filter streams to active ops only) ─────────────────────
        out_edges: dict[str, list[str]] = defaultdict(list)
        in_edges:  dict[str, list[str]] = defaultdict(list)
        for s in topology.streams:
            if s.from_op in active_ids and s.to_op in active_ids:
                out_edges[s.from_op].append(s.to_op)
                in_edges[s.to_op].append(s.from_op)

        op_map   = {o.op_id: o for o in ops}
        pump_ids = {o.op_id for o in ops if o.op_type in ("pump", "mfc")}
        led_ids  = {o.op_id for o in ops if o.op_type == "led_module"}
        skip_ids = pump_ids | led_ids

        # ── Graph init ────────────────────────────────────────────────────────
        stem = str(Path(output_png).with_suffix(""))
        dot  = graphviz.Digraph(comment=title or "FLORA Process", format="png")
        dot.attr(**GRAPH_ATTR)
        dot.attr(imagepath=str(ICONS_DIR))
        dot.attr("node",  fontname="Arial")
        dot.attr("edge", **EDGE_ATTR)

        if title:
            dot.attr(label=f'<<FONT POINT-SIZE="13" COLOR="#111827"><B>{_esc(title)}</B></FONT>>',
                     labelloc="t", labeljust="c")

        # ── Build graphviz nodes ──────────────────────────────────────────────
        # node_id → graphviz node name (same as op_id, sanitised)
        gv_id = {op.op_id: op.op_id.replace("-", "_").replace(" ", "_")
                 for op in ops}

        # Track synthetic mixer IDs created for 3+ input chaining
        # synth_pre[original_mixer_gv_id] → list of (synth_id, inputs)
        synth_pre: dict[str, list[tuple[str, list[str]]]] = {}

        # Pre-compute input counts for all mixers (needed to pick mixer2 vs mixer3)
        MIXER_TYPES_SET = {"mixer", "t_mixer", "y_mixer", "quench_mixer"}
        mixer_input_counts: dict[str, int] = {}
        for op in ops:
            if op.op_type in MIXER_TYPES_SET:
                all_inp = in_edges.get(op.op_id, [])
                # Exclude led_module from count
                real_inp = [i for i in all_inp if op_map.get(i) and
                            op_map[i].op_type != "led_module"]
                mixer_input_counts[op.op_id] = len(real_inp)

        for op in ops:
            if op.op_type == "led_module":
                continue  # skip — wavelength shown under reactor label

            vid = gv_id[op.op_id]

            if op.op_type == "mfc":
                _add_mfc_node(dot, vid, op)

            elif op.op_type == "pump":
                _add_pump(dot, vid, op, pump_img)

            elif op.op_type in MIXER_TYPES_SET:
                n_inp = mixer_input_counts.get(op.op_id, 2)
                _add_mixer_image(dot, vid, n_inp)

            elif op.op_type in ("coil_reactor", "reactor", "heated_coil"):
                is_photo = bool((op.parameters or {}).get("wavelength_nm"))
                img = ASSETS["photoreactor"] if is_photo else ASSETS["reactor"]
                _add_image_node(dot, vid, _reactor_label(op), img, NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type == "photoreactor":
                _add_image_node(dot, vid, _reactor_label(op),
                                ASSETS["photoreactor"], NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type in ("packed_bed", "packed_bed_reactor"):
                _add_image_node(dot, vid, _reactor_label(op),
                                ASSETS["reactor"], NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type in ("microchannel", "microreactor", "chip",
                                 "microfluidic"):
                _add_image_node(dot, vid, _reactor_label(op),
                                ASSETS["microchannel"], NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type == "bpr":
                _add_image_node(dot, vid, _bpr_label(op), ASSETS["bpr"],
                                NODE_ICON_SIZE, NODE_ICON_SIZE)

            elif op.op_type == "collector":
                _add_image_node(dot, vid, op.label or "Product",
                                ASSETS["vial"], NODE_ICON_SIZE, NODE_ICON_SIZE)

            else:
                _add_textbox(dot, vid, op)

        # ── Edge routing ──────────────────────────────────────────────────────
        # All op types that expose an :img port (image-based nodes)
        IMAGE_TYPES = {
            "coil_reactor", "reactor", "heated_coil", "photoreactor",
            "packed_bed", "packed_bed_reactor", "bpr", "collector",
            "microchannel", "microreactor", "chip",
            # Mixers are now image-based too
            "mixer", "t_mixer", "y_mixer", "quench_mixer",
        }

        def output_port(op_id: str) -> str:
            """Graphviz port string for the OUTPUT (right side) of a node."""
            op = op_map.get(op_id)
            if op is None:
                return gv_id.get(op_id, op_id)
            vid = gv_id[op_id]
            ot  = op.op_type
            if ot in ("pump", "mfc"):
                return f"{vid}:needle:e"
            if ot == "led_module":
                return vid
            if ot in IMAGE_TYPES:
                return f"{vid}:img:e"
            # textbox: use named right port
            return f"{vid}:out:e"

        def input_target(op_id: str) -> str:
            """Base port string for connecting INTO a node."""
            op = op_map.get(op_id)
            if op is None:
                return gv_id.get(op_id, op_id)
            vid = gv_id[op_id]
            ot  = op.op_type
            if ot in IMAGE_TYPES:
                return f"{vid}:img"
            # textbox: use named left port
            return f"{vid}:inp:w"

        for op in ops:
            if op.op_id in skip_ids:
                continue

            vid    = gv_id[op.op_id]
            ot     = op.op_type
            inputs = in_edges[op.op_id]

            pump_inp = [i for i in inputs if i in pump_ids]
            main_inp = [i for i in inputs if i not in skip_ids]

            is_mixer = ot in MIXER_TYPES_SET

            if is_mixer:
                # all_inp: main-flow inputs first, then pump inputs
                all_inp  = main_inp + pump_inp
                n_total  = len(all_inp)
                tgt_base = input_target(op.op_id)  # e.g. "Mixer1:img"

                if n_total <= 2:
                    # ── T-mixer: two inputs from pumps/flow ──────────────────
                    # nw = top-left corner, sw = bottom-left corner.
                    # With ortho routing this creates clean right-angle bends.
                    headports = ["nw", "sw"]
                    for i, src_id in enumerate(all_inp):
                        dot.edge(output_port(src_id),
                                 tgt_base, headport=headports[i])

                elif n_total == 3:
                    # ── Cross-mixer: main flow → w, pumps → nw / sw ──────────
                    if main_inp:
                        # Main flow enters from the left (west) — direct horizontal
                        dot.edge(output_port(main_inp[0]),
                                 tgt_base, headport="w")
                        for i, src_id in enumerate(pump_inp[:2]):
                            dot.edge(output_port(src_id),
                                     tgt_base, headport=["nw", "sw"][i])
                    else:
                        # Three pumps, no main flow → pre-mix first two
                        synth_id = f"_pre_{vid}"
                        _add_mixer_image(dot, synth_id, 2)
                        op_map[synth_id] = type("SynthOp", (), {
                            "op_id": synth_id, "op_type": "mixer", "parameters": {}
                        })()
                        gv_id[synth_id] = synth_id
                        dot.edge(output_port(all_inp[0]),
                                 f"{synth_id}:img", headport="nw")
                        dot.edge(output_port(all_inp[1]),
                                 f"{synth_id}:img", headport="sw")
                        dot.edge(f"{synth_id}:img:e", tgt_base, headport="w")
                        dot.edge(output_port(all_inp[2]),
                                 tgt_base, headport="nw")

                else:
                    # ── 4+ inputs: chain two mixers ───────────────────────────
                    synth_id = f"_pre_{vid}"
                    _add_mixer_image(dot, synth_id, 2)
                    op_map[synth_id] = type("SynthOp", (), {
                        "op_id": synth_id, "op_type": "mixer", "parameters": {}
                    })()
                    gv_id[synth_id] = synth_id
                    dot.edge(output_port(all_inp[0]),
                             f"{synth_id}:img", headport="nw")
                    dot.edge(output_port(all_inp[1]),
                             f"{synth_id}:img", headport="sw")
                    dot.edge(f"{synth_id}:img:e", tgt_base, headport="w")
                    for i, src_id in enumerate(all_inp[2:4]):
                        dot.edge(output_port(src_id),
                                 tgt_base, headport=["nw", "sw"][i])

            else:
                # Non-mixer: connect all non-led inputs to west side of node
                for src_id in inputs:
                    if src_id in led_ids:
                        continue
                    tgt = input_target(op.op_id)
                    # Use w (west/left) for clean horizontal connections
                    dot.edge(output_port(src_id), tgt, headport="w")

        # ── Render ────────────────────────────────────────────────────────────
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)

        # PNG
        dot.format = "png"
        rendered_png = dot.render(stem, cleanup=True)
        if not Path(rendered_png).exists() and Path(stem + ".png").exists():
            rendered_png = stem + ".png"
        if Path(rendered_png).exists() and rendered_png != output_png:
            Path(rendered_png).rename(output_png)
        logger.info(f"PNG saved: {output_png}")

        # SVG
        dot.format = "svg"
        rendered_svg = dot.render(stem, cleanup=True)
        if not Path(rendered_svg).exists() and Path(stem + ".svg").exists():
            rendered_svg = stem + ".svg"
        if Path(rendered_svg).exists() and rendered_svg != output_svg:
            Path(rendered_svg).rename(output_svg)
        logger.info(f"SVG saved: {output_svg}")

        return output_svg, output_png

    def _clean_ops(self, topology) -> list:
        """
        Remove logically redundant unit operations before rendering:
          - LED modules (shown as reactor icon + wavelength label instead)
          - Consecutive duplicate op types (e.g. two BPRs back to back)
        """
        ops     = list(topology.unit_operations)
        streams = topology.streams

        # Build sequential order from streams (main lane only)
        out_edges: dict[str, list[str]] = defaultdict(list)
        in_edges:  dict[str, list[str]] = defaultdict(list)
        for s in streams:
            out_edges[s.from_op].append(s.to_op)
            in_edges[s.to_op].append(s.from_op)

        op_map   = {o.op_id: o for o in ops}
        pump_ids = {o.op_id for o in ops if o.op_type in ("pump", "mfc")}
        led_ids  = {o.op_id for o in ops if o.op_type == "led_module"}

        # Walk main lane (non-pump, non-led) and find consecutive duplicates
        remove_ids = set(led_ids)  # always remove LED

        DEDUP_TYPES = {"bpr", "inline_filter", "filter"}

        # Check every non-pump op: if its only non-pump predecessor has the same type → remove
        for op in ops:
            if op.op_id in remove_ids or op.op_type == "pump":
                continue
            preds = [
                op_map[pid] for pid in in_edges.get(op.op_id, [])
                if pid in op_map and pid not in pump_ids and pid not in remove_ids
            ]
            for pred in preds:
                if pred.op_type == op.op_type and op.op_type in DEDUP_TYPES:
                    remove_ids.add(op.op_id)
                    break

        if not remove_ids:
            return ops

        # Return ops with removed ones filtered out
        return [o for o in ops if o.op_id not in remove_ids]

    def _legacy_fallback(self, topology, title, output_svg, output_png):
        """Use the legacy hand-drawn SVG builder as fallback."""
        try:
            from flora_design.visualizer.flowsheet_builder_legacy import (
                FlowsheetBuilder as LegacyBuilder,
            )
            return LegacyBuilder().build(topology, title, output_svg, output_png)
        except Exception as e:
            logger.error(f"Legacy builder also failed: {e}")
            return "", ""
