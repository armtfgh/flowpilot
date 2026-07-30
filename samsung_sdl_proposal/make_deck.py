"""Generate the Samsung SDL high-k proposal deck (8 slides, 16:9)."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

NAVY = RGBColor(0x1B, 0x2A, 0x4A)
TEAL = RGBColor(0x2A, 0x9D, 0x8F)
AMBER = RGBColor(0xE0, 0x9A, 0x2B)
GRAY = RGBColor(0x5A, 0x64, 0x72)
LIGHT = RGBColor(0xF4, 0xF6, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xB4, 0x3E, 0x3E)

FONT = "Calibri"
SW, SH = Inches(13.333), Inches(7.5)

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tb.text_frame.word_wrap = True
    return tb


def set_text(tf, runs_per_para, size=14, color=GRAY, bold=False,
             align=PP_ALIGN.LEFT, space_after=6, line_spacing=1.0):
    """runs_per_para: list of paragraphs; each is str or list of (text, dict) runs."""
    tf.word_wrap = True
    for i, para in enumerate(runs_per_para):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        if isinstance(para, str):
            para = [(para, {})]
        for text, opts in para:
            r = p.add_run()
            r.text = text
            r.font.name = FONT
            r.font.size = Pt(opts.get("size", size))
            r.font.bold = opts.get("bold", bold)
            r.font.italic = opts.get("italic", False)
            r.font.color.rgb = opts.get("color", color)


def shape(slide, kind, x, y, w, h, fill=NAVY, line=None, line_w=None):
    sp = slide.shapes.add_shape(kind, x, y, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(line_w or 1)
    sp.shadow.inherit = False
    return sp


def card(slide, x, y, w, h, title, body, fill=WHITE, title_color=NAVY,
         body_color=GRAY, title_size=13, body_size=11, accent=None):
    sp = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill,
               line=RGBColor(0xD9, 0xDE, 0xE4), line_w=0.75)
    sp.adjustments[0] = 0.08
    if accent is not None:
        bar = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(0.09), h, fill=accent)
        bar.adjustments[0] = 0.5
    tf = sp.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.22)
    tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.12)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    paras = [[(title, {"size": title_size, "bold": True, "color": title_color})]]
    for b in body:
        paras.append([(b, {"size": body_size, "color": body_color})])
    set_text(tf, paras, space_after=4, line_spacing=1.05)
    return sp


def header(slide, kicker, title, n):
    tb = box(slide, Inches(0.55), Inches(0.28), Inches(11), Inches(0.35))
    set_text(tb.text_frame, [[(kicker.upper(), {"size": 12, "bold": True, "color": TEAL})]])
    tb = box(slide, Inches(0.55), Inches(0.60), Inches(12.2), Inches(0.75))
    set_text(tb.text_frame, [[(title, {"size": 27, "bold": True, "color": NAVY})]])
    shape(slide, MSO_SHAPE.RECTANGLE, Inches(0.60), Inches(1.32), Inches(1.6), Pt(2.6), fill=AMBER)
    ftr = box(slide, Inches(11.9), Inches(7.05), Inches(1.0), Inches(0.3))
    set_text(ftr.text_frame, [[(str(n), {"size": 10, "color": GRAY})]], align=PP_ALIGN.RIGHT)
    ftr = box(slide, Inches(0.55), Inches(7.05), Inches(6.0), Inches(0.3))
    set_text(ftr.text_frame, [[("Self-Driving Lab for High-k Thin Films — proposal draft",
                                {"size": 10, "color": RGBColor(0xAA, 0xB2, 0xBC)})]])


def bullets(slide, x, y, w, h, items, size=14, gap=8):
    tb = box(slide, x, y, w, h)
    paras = []
    for it in items:
        if isinstance(it, tuple):
            lead, rest = it
            paras.append([("▸  ", {"size": size, "color": TEAL, "bold": True}),
                          (lead, {"size": size, "bold": True, "color": NAVY}),
                          (rest, {"size": size, "color": GRAY})])
        else:
            paras.append([("▸  ", {"size": size, "color": TEAL, "bold": True}),
                          (it, {"size": size, "color": GRAY})])
    set_text(tb.text_frame, paras, space_after=gap, line_spacing=1.12)
    return tb


def takeaway(slide, text, y=Inches(6.25), x=Inches(0.55), w=Inches(12.23), h=Inches(0.62)):
    sp = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=NAVY)
    sp.adjustments[0] = 0.5
    tf = sp.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.3)
    set_text(tf, [[("Key message   ", {"size": 12, "bold": True, "color": AMBER}),
                   (text, {"size": 13.5, "color": WHITE})]])


def arrow_between(slide, x1, y1, x2, y2, color=NAVY, w=2.2):
    conn = slide.shapes.add_connector(2, x1, y1, x2, y2)  # straight
    conn.line.color.rgb = color
    conn.line.width = Pt(w)
    lineEl = conn.line._get_or_add_ln()
    tail = lineEl.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'})
    lineEl.append(tail)
    conn.shadow.inherit = False
    return conn


# ---------------------------------------------------------------- slide 1: title
s = add_slide()
shape(s, MSO_SHAPE.RECTANGLE, 0, 0, SW, SH, fill=NAVY)
shape(s, MSO_SHAPE.RECTANGLE, 0, Inches(5.05), SW, Pt(3), fill=AMBER)
# subtle film-stack motif
stack_x, stack_w = Inches(10.1), Inches(2.3)
layers = [(Inches(5.7), Inches(0.5), RGBColor(0x2E, 0x3F, 0x63)),   # substrate
          (Inches(5.42), Inches(0.28), RGBColor(0x3D, 0x52, 0x7A)), # bottom electrode
          (Inches(5.18), Inches(0.24), AMBER),                       # high-k
          (Inches(4.90), Inches(0.28), RGBColor(0x3D, 0x52, 0x7A))]  # top electrode
for ly, lh, col in layers:
    shape(s, MSO_SHAPE.RECTANGLE, stack_x, ly, stack_w, lh, fill=col)
lbl = box(s, stack_x - Inches(0.1), Inches(6.28), stack_w + Inches(0.2), Inches(0.3))
set_text(lbl.text_frame, [[("the high-k layer, engineered by data",
                            {"size": 10.5, "italic": True, "color": RGBColor(0x9B, 0xA7, 0xBB)})]],
         align=PP_ALIGN.CENTER)

tb = box(s, Inches(0.8), Inches(1.5), Inches(10.5), Inches(0.5))
set_text(tb.text_frame, [[("WORKFLOW PROPOSAL", {"size": 14, "bold": True, "color": TEAL})]])
tb = box(s, Inches(0.8), Inches(2.0), Inches(10.8), Inches(1.8))
set_text(tb.text_frame, [[("A Self-Driving Laboratory for High-k Thin Films",
                           {"size": 40, "bold": True, "color": WHITE})]])
tb = box(s, Inches(0.8), Inches(3.55), Inches(9.6), Inches(1.2))
set_text(tb.text_frame, [[("Learning structure–property maps of ALD-grown dielectrics "
                           "with the fewest possible experiments",
                           {"size": 19, "color": RGBColor(0xC9, 0xD2, 0xDE)})]], line_spacing=1.15)
tb = box(s, Inches(0.8), Inches(5.5), Inches(8.5), Inches(1.2))
set_text(tb.text_frame, [
    [("[Your name]  ·  [Affiliation]", {"size": 15, "bold": True, "color": WHITE})],
    [("Prepared for Samsung  ·  July 2026  ·  Confidential draft",
      {"size": 12.5, "color": RGBColor(0x9B, 0xA7, 0xBB)})],
], space_after=4)

# ---------------------------------------------------------------- slide 2: bottleneck
s = add_slide()
header(s, "Motivation", "High-k development is rate-limited by measurement, not deposition", 2)
bullets(s, Inches(0.6), Inches(1.65), Inches(6.7), Inches(4.3), [
    ("Vast design space. ", "HfO₂/ZrO₂-family stacks are tuned by composition, doping, "
     "cycle ratios, deposition temperature, plasma conditions, and anneal — a combinatorial space "
     "no grid study can cover."),
    ("The properties that matter are slow to measure. ", "Dielectric constant (k), leakage, and "
     "breakdown require electrodes, annealing, and probing: hours to days per sample."),
    ("Fast measurements exist — but measure the wrong thing. ", "Ellipsometry, XRR, and XRD take "
     "minutes yet report structure, not electrical performance."),
    ("Blind sampling wastes the expensive tier. ", "Most electrical tests land on samples that "
     "teach us nothing new about the process window."),
], size=14.5, gap=12)
# right panel: cost asymmetry graphic
panel = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.7), Inches(1.65), Inches(5.05), Inches(4.35), fill=LIGHT)
panel.adjustments[0] = 0.04
tb = box(s, Inches(7.95), Inches(1.85), Inches(4.6), Inches(0.4))
set_text(tb.text_frame, [[("The two-tier cost asymmetry", {"size": 14, "bold": True, "color": NAVY})]])
card(s, Inches(7.95), Inches(2.35), Inches(4.55), Inches(1.5),
     "TIER 1 — Structural fingerprint  (minutes)",
     ["Spectroscopic ellipsometry · XRR · GIXRD",
      "Non-destructive, automatable, every sample"],
     accent=TEAL, title_size=12.5, body_size=11.5)
card(s, Inches(7.95), Inches(4.05), Inches(4.55), Inches(1.75),
     "TIER 2 — Electrical truth  (hours–days)",
     ["Anneal → electrodes → C–V and I–V probing",
      "Gives k, leakage, breakdown — the real objective",
      "Expensive, delicate, human skill required"],
     accent=AMBER, title_size=12.5, body_size=11.5)
takeaway(s, "The scarce resource is the electrical measurement — so that is what an "
            "intelligent laboratory must spend wisely.")

# ---------------------------------------------------------------- slide 3: the idea
s = add_slide()
header(s, "Proposal", "A different kind of SDL: an efficient data engine, not a hero-film hunter", 3)
card(s, Inches(0.6), Inches(1.65), Inches(5.95), Inches(2.1),
     "Conventional SDL framing",
     ["Objective: autonomously find the single best film.",
      "Optimizer converges, then the dataset is a by-product.",
      "Result transfers poorly when the target spec changes."],
     accent=GRAY, title_size=14, body_size=12.5)
card(s, Inches(6.85), Inches(1.65), Inches(5.95), Inches(2.1),
     "This proposal",
     ["Objective: learn the map (process, anneal, structure) → (k, leakage) — "
      "and the boundaries of where that map is valid.",
      "Every expensive electrical test is chosen by an algorithm to be maximally informative.",
      "The model itself is the product: reusable for any future spec."],
     accent=AMBER, title_size=14, body_size=12.5)
bullets(s, Inches(0.6), Inches(4.15), Inches(12.1), Inches(1.9), [
    ("Virtual metrology. ", "Once trained, a 2-minute optical scan predicts k and leakage — "
     "electrical probing becomes the exception, not the routine."),
    ("Validity mapping. ", "The model reports where fast measurements reliably predict electrical "
     "behavior, and flags regions where hidden variables (phase mixture, interface layers) take over "
     "— exactly where the interesting physics and reliability risks live."),
], size=14, gap=10)
takeaway(s, "We treat expensive characterization as a scarce resource allocated by an "
            "information-gain policy — value that survives any change of material target.")

# ---------------------------------------------------------------- slide 4: why ALD
s = add_slide()
header(s, "Platform", "ALD is the most automation-ready synthesis tool in the fab", 4)
cw, ch = Inches(3.95), Inches(2.05)
card(s, Inches(0.6), Inches(1.7), cw, ch, "The run is already digital",
     ["An ALD recipe is a parameter file: pulse/purge times, temperature, plasma power, cycle count.",
      "Optimizer → recipe file → tool: no new robotics needed for synthesis."],
     accent=TEAL, body_size=12)
card(s, Inches(4.72), Inches(1.7), cw, ch, "Composition is software",
     ["Hf:Zr ratio and dopant level set by supercycle cycle ratios — an integer in the recipe.",
      "No new solutions, targets, or precursor swaps to explore composition."],
     accent=TEAL, body_size=12)
card(s, Inches(8.84), Inches(1.7), cw, ch, "In-situ metrology for free",
     ["In-chamber ellipsometry / QCM give growth-per-cycle and film data every run, without unloading.",
      "A fast autonomous inner loop with zero sample handling."],
     accent=TEAL, body_size=12)
card(s, Inches(0.6), Inches(3.95), Inches(12.19), Inches(1.15), "Proven starting point",
     ["Autonomous ALD process tuning with in-situ feedback has been demonstrated (e.g., Argonne National "
      "Laboratory). This proposal extends autonomy from process parameters to film properties — "
      "k and leakage — which no published SDL has closed the loop on."],
     accent=AMBER, body_size=12.5)
bullets(s, Inches(0.6), Inches(5.25), Inches(12.1), Inches(0.9), [
    ("Honest constraint: ", "one run = one composition (uniformity is ALD's virtue). We compensate "
     "with sample-efficient algorithms and rich in-situ data — not brute-force throughput."),
], size=13.5)
takeaway(s, "Spin-coating SDLs had to build the automation; an ALD SDL only has to connect "
            "what already exists — and add the intelligence.")

# ---------------------------------------------------------------- slide 5: architecture diagram
s = add_slide()
header(s, "Architecture", "Two nested loops around one decision engine", 5)

def node(x, y, w, h, title, sub, fill, tcolor=WHITE, scolor=None, tsize=13, ssize=10.5):
    sp = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill)
    sp.adjustments[0] = 0.12
    tf = sp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    scolor = scolor or RGBColor(0xD3, 0xDA, 0xE4)
    set_text(tf, [[(title, {"size": tsize, "bold": True, "color": tcolor})],
                  [(sub, {"size": ssize, "color": scolor})]],
             align=PP_ALIGN.CENTER, space_after=1, line_spacing=1.0)
    return sp

ny = Inches(2.0)
nh = Inches(1.15)
nw = Inches(2.55)
# inner loop nodes
plan = node(Inches(0.7), ny, nw, nh, "PLAN", "active-learning engine picks next recipe", AMBER,
            tcolor=NAVY, scolor=RGBColor(0x5c, 0x4a, 0x1e))
dep = node(Inches(4.05), ny, nw, nh, "DEPOSIT", "ALD · recipe-as-code supercycles", NAVY)
fast = node(Inches(7.4), ny, nw, nh, "FAST METROLOGY", "SE · XRR · GIXRD — minutes, every sample", TEAL,
            scolor=RGBColor(0xE2, 0xF2, 0xEF))
arrow_between(s, Inches(3.25), ny + nh/2, Inches(4.05), ny + nh/2)
arrow_between(s, Inches(6.60), ny + nh/2, Inches(7.40), ny + nh/2)
# return arrow (inner loop) via top
arrow_between(s, Inches(8.67), ny, Inches(8.67), Inches(1.62), color=TEAL)
conn = s.shapes.add_connector(2, Inches(8.67), Inches(1.62), Inches(1.97), Inches(1.62))
conn.line.color.rgb = TEAL
conn.line.width = Pt(2.2)
conn.shadow.inherit = False
arrow_between(s, Inches(1.97), Inches(1.62), Inches(1.97), ny, color=TEAL)
lb = box(s, Inches(4.4), Inches(1.28), Inches(3.6), Inches(0.3))
set_text(lb.text_frame, [[("INNER LOOP — fully autonomous, ~minutes",
                           {"size": 11, "bold": True, "color": TEAL})]], align=PP_ALIGN.CENTER)

# escalation gate
gate = shape(s, MSO_SHAPE.DIAMOND, Inches(10.55), Inches(1.83), Inches(1.9), Inches(1.5), fill=WHITE,
             line=AMBER, line_w=2)
tf = gate.text_frame
tf.word_wrap = True
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
set_text(tf, [[("ESCALATE?", {"size": 11.5, "bold": True, "color": NAVY})],
              [("info gain vs cost", {"size": 9.5, "color": GRAY})]],
         align=PP_ALIGN.CENTER, space_after=0)
arrow_between(s, Inches(9.95), ny + nh/2, Inches(10.55), ny + nh/2)

# outer loop nodes
oy = Inches(4.35)
anneal = node(Inches(9.6), oy, nw, nh, "ANNEAL", "RTP · thermal budget as model input", NAVY)
elec = node(Inches(5.55), oy, Inches(3.3), nh, "ELECTRICAL TEST", "C–V · I–V on pre-patterned substrates — human-executed", NAVY)
mdl = node(Inches(1.2), oy, Inches(3.6), nh, "MODEL UPDATE", "structure→property map + uncertainty", TEAL,
           scolor=RGBColor(0xE2, 0xF2, 0xEF))
arrow_between(s, Inches(11.5), Inches(3.33), Inches(11.5), oy, color=AMBER)
arrow_between(s, Inches(9.6), oy + nh/2, Inches(8.85), oy + nh/2)
arrow_between(s, Inches(5.55), oy + nh/2, Inches(4.8), oy + nh/2)
arrow_between(s, Inches(1.9), oy, Inches(1.9), Inches(3.15), color=AMBER)
lb = box(s, Inches(3.4), oy + nh + Inches(0.08), Inches(6.8), Inches(0.32))
set_text(lb.text_frame, [[("OUTER LOOP — algorithm-selected samples only, human-in-the-loop",
                           {"size": 11, "bold": True, "color": AMBER})]], align=PP_ALIGN.CENTER)
hum = box(s, Inches(5.4), oy - Inches(0.36), Inches(3.6), Inches(0.3))
set_text(hum.text_frame, [[("expert judgment where robots fail", {"size": 10.5, "italic": True,
                                                                  "color": GRAY})]], align=PP_ALIGN.CENTER)

# data backbone
bb = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(6.15), Inches(11.8), Inches(0.55),
           fill=LIGHT, line=RGBColor(0xD9, 0xDE, 0xE4), line_w=0.75)
bb.adjustments[0] = 0.5
tf = bb.text_frame
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
set_text(tf, [[("DATA BACKBONE   ", {"size": 12, "bold": True, "color": NAVY}),
               ("every sample carries full provenance: recipe → tool logs → structural fingerprint → "
                "anneal → electrical result  ·  drift-control reference runs scheduled automatically",
                {"size": 11.5, "color": GRAY})]], align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------- slide 6: the brain
s = add_slide()
header(s, "Decision engine", "Multi-fidelity active learning that knows what it doesn't know", 6)
bullets(s, Inches(0.6), Inches(1.65), Inches(6.9), Inches(4.4), [
    ("Model. ", "Probabilistic map from structural fingerprint + anneal parameters to k and leakage, "
     "with calibrated uncertainty (multi-fidelity Gaussian-process family)."),
    ("Acquisition. ", "Chooses the next recipe AND whether a sample earns escalation to electrical "
     "testing — maximizing information gain per unit cost, with the human's time in the cost model."),
    ("Two kinds of uncertainty, two actions. ", "\"Model needs more data\" → sample nearby. "
     "\"Fast measurements cannot resolve this region\" → flag for deep analysis (TEM/XPS): "
     "hidden-variable physics found automatically."),
    ("Drift-aware by design. ", "Scheduled reference depositions and replicate measurements separate "
     "real materials physics from tool drift and metrology noise."),
], size=13.5, gap=11)
panel = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.85), Inches(1.65), Inches(4.9), Inches(4.4), fill=LIGHT)
panel.adjustments[0] = 0.04
tb = box(s, Inches(8.1), Inches(1.85), Inches(4.4), Inches(0.4))
set_text(tb.text_frame, [[("Validated before hardware", {"size": 14, "bold": True, "color": NAVY})]])
card(s, Inches(8.1), Inches(2.35), Inches(4.4), Inches(1.6),
     "In-silico benchmark (month 0–6)",
     ["Synthetic HfO₂/HZO landscape built from literature (thickness–anneal–phase–leakage behavior).",
      "Target: reach a fixed map accuracy with ≥3× fewer electrical tests than grid or random sampling."],
     title_size=12.5, body_size=11, accent=TEAL)
card(s, Inches(8.1), Inches(4.15), Inches(4.4), Inches(1.65),
     "Why this matters",
     ["The algorithm is proven on a known ground truth before a single wafer is spent.",
      "The same benchmark becomes the acceptance test for the physical system."],
     title_size=12.5, body_size=11, accent=AMBER)
takeaway(s, "Sampling is steered toward the boundaries where predictability changes — "
            "where blind grids waste 90% of their budget.")

# ---------------------------------------------------------------- slide 7: in-silico demo
s = add_slide()
header(s, "Demonstration", "Proof of mechanism: the decision engine already works in silico", 7)
import os
_fig = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "benchmark", "fig_slide_summary.png")
s.shapes.add_picture(_fig, Inches(0.55), Inches(1.5), width=Inches(12.23))
bullets(s, Inches(0.6), Inches(5.35), Inches(12.1), Inches(1.0), [
    ("Result (6 campaigns, identical 150-test budget): ", "our policy reaches the baselines' "
     "final map accuracy with ~half the electrical tests, wastes 16% vs 30\u201335% of budget in the "
     "unlearnable zone, and detects that zone best (AUC 0.74)."),
    ("What this does NOT claim: ", "the landscape is synthetic (literature-inspired). It validates "
     "the algorithm, not the material physics \u2014 and it surfaced two deployment musts: replicate "
     "measurements enable validity mapping; an exploration safeguard prevents noise-map lock-in."),
], size=12.5, gap=6)
takeaway(s, "Physics aside \u2014 the engine demonstrably learns what is learnable, flags what is not, "
            "and spends human time where it counts.", y=Inches(6.55))

# ---------------------------------------------------------------- slide 7: roadmap
s = add_slide()
header(s, "Execution", "A phased roadmap where every stage delivers standalone value", 8)
ph = [
    ("PHASE 0", "Months 0–6", "Foundations", RGBColor(0x8A, 0x93, 0xA2),
     ["Data infrastructure & provenance schema",
      "Retrospective modeling on existing ALD data",
      "In-silico algorithm benchmark",
      "Deliverable: validated decision engine + data backbone"]),
    ("PHASE 1", "Months 6–18", "Human-in-the-loop SDL", TEAL,
     ["Automated recipes + fast structural loop",
      "Algorithm-selected electrical tests, human-executed",
      "First structure→property map for one material system",
      "Deliverable: virtual-metrology model v1 + mapped process window"]),
    ("PHASE 2", "Months 18+", "Closing the loop", NAVY,
     ["Automated sample handling & annealing integration",
      "Expanded palette: dopants, laminates, electrodes",
      "Transfer validation on device-representative stacks",
      "Deliverable: autonomous platform + reusable materials dataset"]),
]
x0 = Inches(0.6)
for i, (tag, when, name, col, items) in enumerate(ph):
    x = x0 + i * Inches(4.12)
    chev = shape(s, MSO_SHAPE.CHEVRON, x, Inches(1.7), Inches(3.95), Inches(0.85), fill=col)
    tf = chev.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    set_text(tf, [[(f"{tag}  ·  {when}", {"size": 12.5, "bold": True, "color": WHITE})],
                  [(name, {"size": 11.5, "color": RGBColor(0xE8, 0xEC, 0xF1)})]],
             align=PP_ALIGN.CENTER, space_after=0)
    card(s, x, Inches(2.75), Inches(3.95), Inches(2.9), "", items, body_size=11.5, accent=col)
bullets(s, Inches(0.6), Inches(5.85), Inches(12.1), Inches(0.7), [
    ("De-risked by design: ", "if Phase 2 automation slips, Phases 0–1 have already delivered the "
     "dataset, the trained models, and the process windows — the scientific value does not depend "
     "on full autonomy."),
], size=13)

# ---------------------------------------------------------------- slide 8: value & risks
s = add_slide()
header(s, "Outcome", "What Samsung gets — and how we handle what can go wrong", 9)
cw = Inches(3.95)
card(s, Inches(0.6), Inches(1.65), cw, Inches(1.95), "Virtual metrology",
     ["Predict k and leakage of new high-k films from a minutes-long optical scan — with quantified "
      "confidence and known validity limits."], accent=AMBER, body_size=12)
card(s, Inches(4.72), Inches(1.65), cw, Inches(1.95), "Mapped process windows",
     ["Safe operating boundaries (thickness, composition, thermal budget) with the failure "
      "mechanisms at each edge identified."], accent=AMBER, body_size=12)
card(s, Inches(8.84), Inches(1.65), cw, Inches(1.95), "A compounding dataset",
     ["FAIR, provenance-complete recipe→structure→electrical data — reusable for every future "
      "dielectric program and model."], accent=AMBER, body_size=12)

tb = box(s, Inches(0.6), Inches(3.8), Inches(6, ), Inches(0.4))
set_text(tb.text_frame, [[("Risks we have designed for", {"size": 15, "bold": True, "color": NAVY})]])
risks = [
    ("Tool drift & noise mimic physics", "scheduled reference runs + replicates; noise terms explicit in the model"),
    ("Probe automation is fragile", "human-in-the-loop tier + pre-patterned test substrates from day one"),
    ("Lab optimum ≠ device stack", "Phase 2 transfer validation on device-representative structures"),
    ("Temperature changes are slow", "cost-aware scheduling batches experiments by thermal budget"),
]
y = Inches(4.25)
for risk, mit in risks:
    tb = box(s, Inches(0.6), y, Inches(12.2), Inches(0.42))
    set_text(tb.text_frame, [[("⚠ ", {"size": 12, "color": RED}),
                              (risk + "  —  ", {"size": 12.5, "bold": True, "color": NAVY}),
                              (mit, {"size": 12.5, "color": GRAY})]])
    y += Inches(0.44)
takeaway(s, "A laboratory that learns where measurements can be trusted — and spends human "
            "expertise only where it is irreplaceable.", y=Inches(6.35))

out = "/home/amirreza/Documents/codes/Flow Agent/samsung_sdl_proposal/SDL_HighK_Proposal.pptx"
prs.save(out)
print("saved", out)
