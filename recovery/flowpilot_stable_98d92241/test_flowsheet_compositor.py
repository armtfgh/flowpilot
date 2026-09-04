# %% ─────────────────────────────────────────────────────────────────────────
# FLORA Flowsheet Compositor — Test Bench
#
# Tests SVG icon composition with orthogonal pipe routing.
# Run interactively with ipykernel (VS Code Python Interactive / Jupyter).
#
# Layout being tested:
#
#   [Pump A] ──────────────────────────────┐
#                                          │ (vertical drop)
#                                          ↓ (top input)
#   [Pump B] ──────────────────────[T-Mixer]──────────→
#                              (left input)  (right output)
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from IPython.display import SVG, display

# ── Where your icon SVG files live ────────────────────────────────────────────
ICON_DIR = Path(__file__).parent / "assets" / "equipment_icons"
ICON_DIR.mkdir(parents=True, exist_ok=True)

# ── Visual style ──────────────────────────────────────────────────────────────
PIPE_WIDTH       = 3.0      # stroke-width for pipes
PIPE_COLOR_A     = "#2563EB"  # stream A — blue
PIPE_COLOR_B     = "#DC2626"  # stream B — red
PIPE_COLOR_OUT   = "#16A34A"  # output stream — green
ARROW_SIZE       = 8          # arrowhead marker size
LABEL_FONT       = "Arial, Helvetica, sans-serif"
LABEL_SIZE       = 10
CANVAS_BG        = "#FFFFFF"
CANVAS_PADDING   = 40         # whitespace around content


# %% ─────────────────────────────────────────────────────────────────────────
# SECTION 1 — Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Port:
    """A connection point on an icon, relative to icon's own origin."""
    x: float          # x relative to icon top-left
    y: float          # y relative to icon top-left
    direction: str    # which direction the pipe exits/enters: 'east' 'west' 'north' 'south'


@dataclass
class IconSpec:
    """One equipment icon placed on the canvas."""
    name:       str                     # human label, e.g. "Pump A"
    icon_type:  str                     # key into EQUIPMENT_REGISTRY
    canvas_x:   float                   # top-left x on canvas
    canvas_y:   float                   # top-left y on canvas
    svg_content: str   = ""             # raw SVG inner content (loaded from file)
    view_w:      float = 0.0            # icon width  (from viewBox)
    view_h:      float = 0.0            # icon height (from viewBox)
    label:       str   = ""             # optional label override (uses name if empty)

    def port_canvas(self, port_name: str) -> tuple[float, float]:
        """Convert a port's icon-relative position to absolute canvas coords."""
        port = EQUIPMENT_REGISTRY[self.icon_type]["ports"][port_name]
        return (self.canvas_x + port.x, self.canvas_y + port.y)


@dataclass
class PipeConnection:
    """A pipe from one icon's output port to another icon's input port."""
    from_icon:  str           # name of source IconSpec
    from_port:  str           # port name on source
    to_icon:    str           # name of target IconSpec
    to_port:    str           # port name on target
    color:      str = PIPE_COLOR_A
    label:      str = ""      # e.g. "0.05 mL/min, 0.1 M"
    label_color: str = "#374151"


# %% ─────────────────────────────────────────────────────────────────────────
# SECTION 2 — Equipment registry
# Port positions must exactly match what is drawn in the SVG file.
# ─────────────────────────────────────────────────────────────────────────────

EQUIPMENT_REGISTRY: dict[str, dict] = {
    "syringe_pump": {
        "view_w": 120, "view_h": 60,
        "ports": {
            "output": Port(120, 30, "east"),
        },
    },
    "t_mixer": {
        "view_w": 80, "view_h": 80,
        "ports": {
            "input_left":  Port(0,  40, "west"),
            "input_top":   Port(40,  0, "north"),
            "output":      Port(80, 40, "east"),
        },
    },
}


# %% ─────────────────────────────────────────────────────────────────────────
# SECTION 3 — SVG icon loader
# Falls back to a clean placeholder if the real SVG file is not found yet.
# ─────────────────────────────────────────────────────────────────────────────

def _make_placeholder_svg(icon_type: str, w: float, h: float) -> str:
    """Generate a clean placeholder box with port markers when icon file missing."""
    reg = EQUIPMENT_REGISTRY[icon_type]
    lines = [f'<rect x="2" y="2" width="{w-4}" height="{h-4}" '
             f'rx="6" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1.5"/>',
             f'<text x="{w/2}" y="{h/2+4}" text-anchor="middle" '
             f'font-family="Arial" font-size="10" fill="#64748B">'
             f'{icon_type.replace("_"," ").title()}</text>']
    # Draw small red circles at each port position
    for port_name, port in reg["ports"].items():
        lines.append(
            f'<circle cx="{port.x}" cy="{port.y}" r="4" '
            f'fill="#EF4444" stroke="white" stroke-width="1"/>'
        )
    return "\n".join(lines)


def load_icon(icon_type: str) -> tuple[str, float, float]:
    """
    Load SVG inner content + dimensions for an equipment type.
    Returns (svg_inner_content, view_w, view_h).
    Falls back to placeholder if file not found.
    """
    reg  = EQUIPMENT_REGISTRY[icon_type]
    path = ICON_DIR / f"{icon_type}.svg"

    if path.exists():
        tree = ET.parse(path)
        root = tree.getroot()
        ns   = {"svg": "http://www.w3.org/2000/svg"}

        # Parse viewBox
        vb = root.get("viewBox", "")
        if vb:
            parts = vb.replace(",", " ").split()
            view_w, view_h = float(parts[2]), float(parts[3])
        else:
            view_w = float(root.get("width", reg["view_w"]))
            view_h = float(root.get("height", reg["view_h"]))

        # Strip namespace from all tags for clean embedding
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        inner_parts = []
        for child in root:
            inner_parts.append(ET.tostring(child, encoding="unicode"))
        content = "\n".join(inner_parts)

        print(f"  Loaded {icon_type}.svg ({view_w}×{view_h})")
        return content, view_w, view_h

    else:
        print(f"  [placeholder] {icon_type}.svg not found — using placeholder")
        w, h = reg["view_w"], reg["view_h"]
        return _make_placeholder_svg(icon_type, w, h), w, h


# %% ─────────────────────────────────────────────────────────────────────────
# SECTION 4 — Orthogonal pipe router
# ─────────────────────────────────────────────────────────────────────────────

def _route_orthogonal(
    x1: float, y1: float, exit_dir: str,
    x2: float, y2: float, entry_dir: str,
) -> str:
    """
    Compute an orthogonal SVG path (right-angle elbows) from (x1,y1) to (x2,y2).

    exit_dir  : direction the pipe LEAVES the source port  ('east','south',...)
    entry_dir : direction the pipe ENTERS the target port  ('west','north',...)

    Returns an SVG path d-string.
    """
    EPS = 2.0  # treat as "same level" if within this many pixels

    # ── East → West (same horizontal level) ──────────────────────────────────
    if exit_dir == "east" and entry_dir == "west":
        if abs(y1 - y2) < EPS:
            # Perfectly aligned — straight horizontal
            return f"M {x1:.1f} {y1:.1f} H {x2:.1f}"
        else:
            # Mid-point elbow: go right halfway, adjust height, continue right
            mid = (x1 + x2) / 2
            return f"M {x1:.1f} {y1:.1f} H {mid:.1f} V {y2:.1f} H {x2:.1f}"

    # ── East → North (pump on left, entering mixer from above) ───────────────
    if exit_dir == "east" and entry_dir == "north":
        # Go right to target x, then drop straight down
        return f"M {x1:.1f} {y1:.1f} H {x2:.1f} V {y2:.1f}"

    # ── East → South (pump on left, entering from below) ─────────────────────
    if exit_dir == "east" and entry_dir == "south":
        return f"M {x1:.1f} {y1:.1f} H {x2:.1f} V {y2:.1f}"

    # ── South → West (pump above, exiting downward, entering left) ───────────
    if exit_dir == "south" and entry_dir == "west":
        return f"M {x1:.1f} {y1:.1f} V {y2:.1f} H {x2:.1f}"

    # ── Generic fallback: two-segment elbow ──────────────────────────────────
    mid = (x1 + x2) / 2
    return f"M {x1:.1f} {y1:.1f} H {mid:.1f} V {y2:.1f} H {x2:.1f}"


def _port_direction_pair(from_port: Port, to_port: Port) -> tuple[str, str]:
    """Return (exit_direction, entry_direction) for a connection."""
    return from_port.direction, to_port.direction


# %% ─────────────────────────────────────────────────────────────────────────
# SECTION 5 — SVG Compositor
# ─────────────────────────────────────────────────────────────────────────────

class FlowsheetCompositor:
    """
    Composes multiple equipment SVG icons with orthogonal pipe connections
    into a single publication-quality SVG.

    Usage:
        comp = FlowsheetCompositor()
        comp.add_icon("Pump A", "syringe_pump", canvas_x=50, canvas_y=50)
        comp.add_icon("T-Mixer", "t_mixer", canvas_x=350, canvas_y=210)
        comp.add_pipe("Pump A", "output", "T-Mixer", "input_top",
                      color=PIPE_COLOR_A, label="Stream A: 0.05 mL/min")
        svg = comp.render()
    """

    def __init__(self):
        self._icons:  dict[str, IconSpec] = {}
        self._pipes:  list[PipeConnection] = []
        self._markers_added: set[str] = set()

    def add_icon(
        self,
        name:      str,
        icon_type: str,
        canvas_x:  float,
        canvas_y:  float,
        label:     str = "",
    ) -> "FlowsheetCompositor":
        content, view_w, view_h = load_icon(icon_type)
        spec = IconSpec(
            name=name, icon_type=icon_type,
            canvas_x=canvas_x, canvas_y=canvas_y,
            svg_content=content, view_w=view_w, view_h=view_h,
            label=label or name,
        )
        self._icons[name] = spec
        return self   # fluent API

    def add_pipe(
        self,
        from_icon: str, from_port: str,
        to_icon:   str, to_port:   str,
        color:     str = PIPE_COLOR_A,
        label:     str = "",
    ) -> "FlowsheetCompositor":
        self._pipes.append(PipeConnection(
            from_icon, from_port, to_icon, to_port, color, label
        ))
        return self

    def render(self, dpi: int = 96) -> str:
        """Build and return the complete SVG string."""
        if not self._icons:
            return "<svg/>"

        # ── Compute canvas bounding box ────────────────────────────────────
        max_x = max(s.canvas_x + s.view_w for s in self._icons.values())
        max_y = max(s.canvas_y + s.view_h for s in self._icons.values())
        W = max_x + CANVAS_PADDING
        H = max_y + CANVAS_PADDING

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{W:.0f}" height="{H:.0f}" '
            f'viewBox="0 0 {W:.0f} {H:.0f}" '
            f'style="background:{CANVAS_BG}; font-family:{LABEL_FONT};">',
        ]

        # ── Arrow marker definitions ───────────────────────────────────────
        lines.append("<defs>")
        for color_hex in {PIPE_COLOR_A, PIPE_COLOR_B, PIPE_COLOR_OUT}:
            marker_id = f"arrow_{color_hex.lstrip('#')}"
            lines.append(
                f'<marker id="{marker_id}" markerWidth="{ARROW_SIZE}" '
                f'markerHeight="{ARROW_SIZE}" refX="{ARROW_SIZE-1}" refY="{ARROW_SIZE/2}" '
                f'orient="auto" markerUnits="userSpaceOnUse">'
                f'<path d="M0,0 L0,{ARROW_SIZE} L{ARROW_SIZE},{ARROW_SIZE/2} Z" '
                f'fill="{color_hex}"/></marker>'
            )
        lines.append("</defs>")

        # ── Pipe connections (drawn BEHIND icons) ──────────────────────────
        lines.append('<g id="pipes">')
        for pipe in self._pipes:
            src  = self._icons[pipe.from_icon]
            dst  = self._icons[pipe.to_icon]
            fp   = EQUIPMENT_REGISTRY[src.icon_type]["ports"][pipe.from_port]
            tp   = EQUIPMENT_REGISTRY[dst.icon_type]["ports"][pipe.to_port]

            x1, y1 = src.port_canvas(pipe.from_port)
            x2, y2 = dst.port_canvas(pipe.to_port)

            path_d = _route_orthogonal(x1, y1, fp.direction, x2, y2, tp.direction)
            marker_id = f"arrow_{pipe.color.lstrip('#')}"

            lines.append(
                f'<path d="{path_d}" '
                f'fill="none" stroke="{pipe.color}" stroke-width="{PIPE_WIDTH}" '
                f'stroke-linecap="round" stroke-linejoin="round" '
                f'marker-end="url(#{marker_id})"/>'
            )

            # Pipe label at midpoint
            if pipe.label:
                segs = path_d.replace("M", "").replace("H", " ").replace("V", " ").split()
                coords = []
                for s in segs:
                    try: coords.append(float(s))
                    except: pass
                if len(coords) >= 2:
                    lx = (x1 + coords[-2]) / 2 if len(coords) >= 4 else (x1+x2)/2
                    ly = (y1 + coords[-1]) / 2 if len(coords) >= 4 else (y1+y2)/2
                    lines.append(
                        f'<rect x="{lx-2:.1f}" y="{ly-8:.1f}" '
                        f'width="{len(pipe.label)*5.5:.0f}" height="13" '
                        f'rx="3" fill="white" fill-opacity="0.85"/>'
                        f'<text x="{lx:.1f}" y="{ly+2:.1f}" '
                        f'text-anchor="middle" font-size="{LABEL_SIZE-1}" '
                        f'fill="{pipe.label_color}">{pipe.label}</text>'
                    )
        lines.append("</g>")

        # ── Icon stamps (drawn ON TOP of pipes) ───────────────────────────
        lines.append('<g id="icons">')
        for spec in self._icons.values():
            lines.append(
                f'<g id="icon_{spec.name.replace(" ","_")}" '
                f'transform="translate({spec.canvas_x:.1f},{spec.canvas_y:.1f})">'
            )
            # Embed the icon SVG content inside a nested svg element
            lines.append(
                f'<svg width="{spec.view_w}" height="{spec.view_h}" '
                f'viewBox="0 0 {spec.view_w} {spec.view_h}">'
            )
            lines.append(spec.svg_content)
            lines.append("</svg>")
            lines.append("</g>")
        lines.append("</g>")

        # ── Icon labels (drawn on top of everything) ───────────────────────
        lines.append('<g id="labels">')
        for spec in self._icons.values():
            lx = spec.canvas_x + spec.view_w / 2
            ly = spec.canvas_y + spec.view_h + 14
            lines.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" '
                f'text-anchor="middle" font-size="{LABEL_SIZE}" '
                f'font-weight="600" fill="#111827">{spec.label}</text>'
            )
        lines.append("</g>")

        lines.append("</svg>")
        return "\n".join(lines)

    def display(self):
        """Display SVG inline in Jupyter / VS Code Python Interactive."""
        svg = self.render()
        display(SVG(svg))
        return svg

    def save(self, path: str | Path, also_png: bool = True):
        """Save SVG (and optionally PNG via cairosvg) to disk."""
        svg = self.render()
        path = Path(path)
        path.write_text(svg, encoding="utf-8")
        print(f"Saved SVG → {path}")

        if also_png:
            try:
                import cairosvg
                png_path = path.with_suffix(".png")
                cairosvg.svg2png(
                    bytestring=svg.encode(),
                    write_to=str(png_path),
                    dpi=300,
                )
                print(f"Saved PNG → {png_path}")
            except ImportError:
                print("(cairosvg not installed — skipping PNG export)")


# %% ─────────────────────────────────────────────────────────────────────────
# SECTION 6 — Test: Pump A + Pump B → T-Mixer
#
# Layout:
#   [Pump A]  at (50, 50)   → feeds T-Mixer top input
#   [Pump B]  at (50, 220)  → feeds T-Mixer left input (same height)
#   [T-Mixer] at (350, 180) → left input at (350,220), top at (390,180)
# ─────────────────────────────────────────────────────────────────────────────

# Build the diagram
comp = FlowsheetCompositor()

(comp
    .add_icon("Pump A",  "syringe_pump", canvas_x=50,  canvas_y=50,  label="Pump A\n(Stream A)")
    .add_icon("Pump B",  "syringe_pump", canvas_x=50,  canvas_y=220, label="Pump B\n(Stream B)")
    .add_icon("T-Mixer", "t_mixer",      canvas_x=350, canvas_y=180, label="T-Mixer")
)

# Pump A output (170, 80) → T-Mixer top input (390, 180)
# Router: east → north  →  go RIGHT to x=390, then DOWN to y=180
comp.add_pipe(
    "Pump A", "output",
    "T-Mixer", "input_top",
    color=PIPE_COLOR_A,
    label="Stream A",
)

# Pump B output (170, 250) → T-Mixer left input (350, 220)
# Same height (y=250 ≠ y=220 by 30px) → mid-elbow routing
comp.add_pipe(
    "Pump B", "output",
    "T-Mixer", "input_left",
    color=PIPE_COLOR_B,
    label="Stream B",
)

# Display
svg = comp.display()

# %% ─────────────────────────────────────────────────────────────────────────
# SECTION 7 — Save to disk
# ─────────────────────────────────────────────────────────────────────────────

comp.save("test_flowsheet_output.svg", also_png=True)
print("\nDone. Open test_flowsheet_output.svg in a browser to inspect.")
