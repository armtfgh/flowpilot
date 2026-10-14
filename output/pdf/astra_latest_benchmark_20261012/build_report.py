"""Build a one-page report from the independently verified figure data."""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
data = json.loads((OUT / 'verified_results.json').read_text())
pdf = OUT / 'astra_latest_benchmark_report.pdf'
ink = colors.HexColor('#182B42')
muted = colors.HexColor('#536478')
styles = {
    'eyebrow': ParagraphStyle('eyebrow', fontName='Helvetica-Bold', fontSize=8, leading=11, textColor=muted, spaceAfter=9),
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=ink, spaceAfter=8),
    'subtitle': ParagraphStyle('subtitle', fontName='Helvetica', fontSize=10, leading=14, textColor=muted, spaceAfter=13),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10.1, leading=14, textColor=ink, spaceAfter=8),
    'label': ParagraphStyle('label', fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=ink, spaceAfter=5),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8, leading=11, textColor=muted, spaceAfter=4),
}
p = lambda text, kind='body': Paragraph(text, styles[kind])
doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=37, leftMargin=37, topMargin=34, bottomMargin=32,
                        title='Latest Astra benchmark - 12 October 2026', author='FlowPilot project',
                        subject='Verified summary of the latest completed development pilot')
story = [p('FLOWPILOT / BENCHMARK NOTE', 'eyebrow'),
         p('Latest Astra benchmark', 'title'),
         p('Completed 12 October 2026 | Prepared 13 October 2026 | Development pilot', 'subtitle'),
         p('<b>FlowPilot averaged 0.9631, compared with 0.9606 for one-shot Astra.</b> '
           'The difference was +0.002493 on the 0-1 scale, or <b>+0.25 percentage points</b>. '
           'Both architectures received zero judge-reported critical flags.'),
         Image(str(OUT / 'astra_latest_comparison.png'), width=505, height=267.65),
         Spacer(1, 8), p('How the comparison was made', 'label'),
         p('GPT-6 Astra with medium reasoning generated one design per architecture for each of three chemistries '
           '(six designs). Both architectures received matched task and inventory information. FlowPilot used the '
           'cheaper legacy configuration with two initial candidates and source-excluded lexical retrieval; computation '
           'was not matched. Qwen3.6-27B, GPT-4o and Claude Sonnet 4.6 evaluated every design, giving 18 accepted judgments.'),
         p('Each judge rated applicable criteria from a fixed 14-criterion rubric on a 0-4 scale. The applicable ratings '
           'were averaged and divided by four, then averaged equally across judges and chemistries. Critical flags were '
           'counted separately. Generation labels were withheld, although complete blinding was not claimed.'),
         p('What the result supports', 'label'),
         p('The CuAAC gain (+1.39 percentage points) outweighed the oxidation decrease (-0.64); hydrogenolysis tied. '
           'The aggregate lead is small and depends on the judge panel. Excluding GPT-4o gives <b>0.9607 for FlowPilot '
           'and 0.9768 for one-shot</b>. These cases informed development, and the two preceding completed comparisons '
           'favored one-shot. With one generation per condition, this pilot establishes a higher observed mean, '
           'but does not establish repeatability or general superiority.'),
         p('Verification and recorded cost', 'label'),
         p('Scores were reproduced from the accepted judge responses; all 95 recorded generation-file hashes and the '
           'matched-input audit were checked. The saved API estimate for this round is <b>$11.34</b> '
           '($9.91 generation; $1.43 judging), excluding local hardware and unreturned billable failures. '
           'No new model calls were made to prepare this report.'),
         Spacer(1, 3),
         p('Source campaign: ablation_results/manuscript_benchmark/<br/>'
           'astra_verified_repairs_dev04_20261012<br/>'
           'Source hashes, accepted-score reconstruction and exact values accompany the figure in verified_results.json.', 'small')]

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#DFE6EC'))
    canvas.line(37, 27, A4[0] - 37, 27)
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(muted)
    canvas.drawString(37, 16, 'Development evidence | Computational assessment; no laboratory validation in this pilot')
    canvas.drawRightString(A4[0] - 37, 16, str(doc.page))

doc.build(story, onFirstPage=footer, onLaterPages=footer)
reader = PdfReader(pdf)
assert len(reader.pages) == 1, f'Expected one page; got {len(reader.pages)}'
text = reader.pages[0].extract_text()
for required in ['0.9631', '0.9606', '18 accepted judgments', '0.9768', '$11.34']:
    assert required in text, required
print(f'Created and text-checked one-page report: {pdf}')
