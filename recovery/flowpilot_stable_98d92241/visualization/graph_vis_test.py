#%%
from html import escape
from pathlib import Path

import graphviz
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "custom_flow_setup"

# Main customization controls.
GRAPH_STYLE = {
    "rankdir": "LR",
    "splines": "polyline",
    "nodesep": "0.55",
    "ranksep": "0.8",
    "pad": "0.25",
    "dpi": "180",
    "bgcolor": "white",
    "forcelabels": "true",
}

IMAGE_NODE_STYLE = {
    "shape": "plain",
}

UV_NODE_STYLE = {
    **IMAGE_NODE_STYLE,
}

MIXER_STYLE = {
    "shape": "box",
    "style": "filled",
    "fillcolor": "lightgrey",
    "color": "grey35",
    "fontname": "Arial",
    "fontsize": "11",
    "width": "0.9",
    "height": "0.9",
    "fixedsize": "true",
    "penwidth": "1.2",
}

EDGE_STYLE = {
    "color": "royalblue4",
    "penwidth": "2.0",
    "arrowsize": "0.7",
}

ASSETS = {
    "pump": BASE_DIR / "syringe_pump.png",
    "reactor": BASE_DIR / "coil_reactor.png",
    "uv": BASE_DIR / "uv.png",
    "photoreactor": BASE_DIR / "coil_photoreactor.png",
    "bpr": BASE_DIR / "bpr.png",
    "vial": BASE_DIR / "vial.png",
    "liq_liq_sep: BASE_DIR/ "liq_sep.png",
    
        
}


def get_trimmed_pump_asset():
    src = ASSETS["pump"]
    out = BASE_DIR / "_generated_syringe_pump_trimmed.png"
    img = Image.open(src).convert("RGBA")
    coords = []
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = img.getpixel((x, y))
            if a == 0:
                continue
            # Ignore near-white background around the syringe so the crop ends at the actual tip.
            if r > 245 and g > 245 and b > 245:
                continue
            coords.append((x, y))
    if not coords:
        return src
    min_x = min(x for x, _ in coords)
    max_x = max(x for x, _ in coords)
    min_y = min(y for _, y in coords)
    max_y = max(y for _, y in coords)
    margin = 4
    cropped = img.crop(
        (
            max(min_x - margin, 0),
            max(min_y - margin, 0),
            min(max_x + margin + 1, img.width),
            min(max_y + margin + 1, img.height),
        )
    )
    cropped.save(out)
    return out


def format_html_caption(text, font_size):
    lines = "<BR/>".join(escape(part) for part in text.splitlines())
    return f'<FONT POINT-SIZE="{font_size}">{lines}</FONT>'


def add_image_node(dot, name, label, image_path, **overrides):
    image_width = overrides.pop("image_width", 110)
    image_height = overrides.pop("image_height", 110)
    font_size = overrides.pop("font_size", 11)
    cell_padding = overrides.pop("cell_padding", 8)
    image_row = (
        f'<TD PORT="img" FIXEDSIZE="TRUE" WIDTH="{image_width}" HEIGHT="{image_height}">'
        f'<IMG SRC="{escape(Path(image_path).name)}" SCALE="TRUE"/></TD>'
    )
    html_label = (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="{cell_padding}">'
        f"<TR>{image_row}</TR>"
        f'<TR><TD ALIGN="CENTER">{format_html_caption(label, font_size)}</TD></TR>'
        "</TABLE>>"
    )
    node_style = IMAGE_NODE_STYLE | {
        "label": html_label,
    }
    node_style.update(overrides)
    dot.node(name, **node_style)


def add_captioned_image_node(dot, name, caption, image_path, **overrides):
    add_image_node(dot, name, caption, image_path, **overrides)


def add_pump_node(dot, name, label, image_path, **overrides):
    image_width = overrides.pop("image_width", 110)
    image_height = overrides.pop("image_height", 24)
    font_size = overrides.pop("font_size", 11)
    needle_top_height = overrides.pop("needle_top_height", 10)
    needle_port_height = overrides.pop("needle_port_height", 2)
    needle_bottom_height = overrides.pop(
        "needle_bottom_height",
        max(image_height - needle_top_height - needle_port_height, 0),
    )
    html_label = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        "<TR>"
        f'<TD ROWSPAN="3" PORT="img" FIXEDSIZE="TRUE" WIDTH="{image_width}" HEIGHT="{image_height}">'
        f'<IMG SRC="{escape(Path(image_path).name)}" SCALE="TRUE"/></TD>'
        f'<TD WIDTH="0" HEIGHT="{needle_top_height}"></TD>'
        "</TR>"
        "<TR>"
        f'<TD PORT="needle" WIDTH="0" HEIGHT="{needle_port_height}"></TD>'
        "</TR>"
        "<TR>"
        f'<TD WIDTH="0" HEIGHT="{needle_bottom_height}"></TD>'
        "</TR>"
        f'<TR><TD COLSPAN="2" ALIGN="CENTER">{format_html_caption(label, font_size)}</TD></TR>'
        "</TABLE>>"
    )
    node_style = IMAGE_NODE_STYLE | {
        "label": html_label,
    }
    node_style.update(overrides)
    dot.node(name, **node_style)


def add_mixer_node(dot, name, label, **overrides):
    width = overrides.pop("width", 0.95)
    height = overrides.pop("height", 0.95)
    font_size = overrides.pop("font_size", 11)
    mixer_width = int(width * 72)
    mixer_height = int(height * 72)
    inlet_height = mixer_height // 2
    port_width = 10
    html_label = (
        '<<TABLE BORDER="1" COLOR="grey35" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" BGCOLOR="lightgrey">'
        "<TR>"
        f'<TD PORT="in_top" WIDTH="{port_width}" HEIGHT="{inlet_height}"></TD>'
        f'<TD ROWSPAN="2" FIXEDSIZE="TRUE" WIDTH="{mixer_width}" HEIGHT="{mixer_height}">'
        f'{format_html_caption(label, font_size)}</TD>'
        f'<TD PORT="out" ROWSPAN="2" WIDTH="{port_width}" HEIGHT="{mixer_height}"></TD>'
        "</TR>"
        "<TR>"
        f'<TD PORT="in_bottom" WIDTH="{port_width}" HEIGHT="{inlet_height}"></TD>'
        "</TR>"
        "</TABLE>>"
    )
    node_style = IMAGE_NODE_STYLE | {"label": html_label}
    node_style.update(overrides)
    dot.node(name, **node_style)


def create_custom_flow_diagram():
    dot = graphviz.Digraph(comment="Flow Chemistry", format="png")
    dot.attr(**GRAPH_STYLE)
    dot.attr(imagepath=str(BASE_DIR))
    dot.attr("node", fontname="Arial")
    dot.attr("edge", **EDGE_STYLE)
    pump_asset = get_trimmed_pump_asset()

    add_pump_node(
        dot,
        "Pump_A",
        "Syringe Pump A\n(0.5 mL/min)",
        pump_asset,
        image_width=110,
        image_height=24,
    )
    add_pump_node(
        dot,
        "Pump_B",
        "Syringe Pump B\n(0.5 mL/min)",
        pump_asset,
        image_width=110,
        image_height=24,
    )

   add_pump_node(
        dot,
        "Separator",
        "Liquid Liquid Separation",
        ASSETS["liq_liq_separator.png"],
        image_width=110,
        image_height=24,
    )



    dot.node("Mixer1", label="Mixer1", group="main_flow", **MIXER_STYLE)
    dot.node("Mixer2", label="Mixer2", group="main_flow", **MIXER_STYLE)
    dot.node("Mixer3", label="Mixer3", group="main_flow", **MIXER_STYLE)


    add_captioned_image_node(
        dot,
        "Reactor",
        "Heated Coil Reactor\n(60 °C, 10 min, 420 nm)",
        ASSETS["photoreactor"],
        image_width=125,
        image_height=125,
        group="main_flow",
    )

    add_pump_node(
        dot,
        "Pump_C",
        "Syringe Pump C\n(0.1 mL/min)",
        pump_asset,
        image_width=110,
        image_height=24,
    )

    add_pump_node(
        dot,
        "Pump_D",
        "Syringe Pump D\n(0.1 mL/min)",
        pump_asset,
        group="main_flow",
        image_width=110,
        image_height=24,
    )

    add_pump_node(
        dot,
        "bpr1",
        "10 psi",
        ASSETS["bpr"],
        group="main_flow",
        image_width=110,
        image_height=110,
    )

    add_pump_node(
        dot,
        "vial",
        "Product",
        ASSETS["vial"],
        group="main_flow",
        image_width=110,
        image_height=110,
    )

    add_captioned_image_node(
        dot,
        "Reactor_2",
        "Heated Coil Reactor 2\n(60 °C, 10 min res.)",
        ASSETS["reactor"],
        image_width=125,
        image_height=125,
        group="main_flow",
    )

    with dot.subgraph() as reactor_stack:
        reactor_stack.attr(rank="same")
        reactor_stack.node("Reactor")

    dot.edge("Pump_A:needle:e", "Mixer1", headport="nw")
    dot.edge("Pump_B:needle:e", "Mixer1", headport="sw")
    dot.edge("Mixer1:e", "Mixer2", headport="nw")
    dot.edge("Pump_C:needle:e", "Mixer2", headport="sw")
    dot.edge("Mixer2:e", "Reactor:img:w")
    dot.edge("Reactor:img:e", "Mixer3:w", headport="sw")
    dot.edge("Pump_D:needle:e", "Mixer3:w", headport="nw")
    dot.edge("Mixer3:e", "Reactor_2:img:w")
    dot.edge("Reactor_2:img:e", "bpr1")
    dot.edge("bpr1:e", "vial:w")

    return dot

if __name__ == "__main__":
    diagram = create_custom_flow_diagram()
    diagram.render(str(OUTPUT_FILE), cleanup=False)
    print(f"Diagram generated successfully at {OUTPUT_FILE}.png")
# %%
