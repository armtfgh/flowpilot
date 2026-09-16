"""Render the final slide PDF and record page-bound checks for manual inspection."""
import json
import sys
from pathlib import Path
import fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'outputs/khu_revised_six_20260914/presentation'


def main():
    review=OUT/'visual_review';review.mkdir(exist_ok=True)
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    filename=sys.argv[2] if len(sys.argv)>2 else 'KHU_revised_six_designs.pdf'
    doc=fitz.open(OUT/filename)
    pages=[];outside=[]
    for i,page in enumerate(doc):
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans'] if s['text'].strip()]
        for s in spans:
            if not (page.rect+(-1,-1,1,1)).contains(fitz.Rect(s['bbox'])):
                outside.append({'page':i+1,'text':s['text'],'bbox':s['bbox']})
        pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5))
        path=review/f'slide_{i+1:02d}.png';pix.save(path)
        pages.append({'page':i+1,'title':page.get_text().splitlines()[0],'text_spans':len(spans),'render':path.name})
    for start in range(0,len(pages),12):
        selected=pages[start:start+12]
        sheet=Image.new('RGB',(1600,4*250),'#dfe4e6');draw=ImageDraw.Draw(sheet)
        for j,item in enumerate(selected):
            with Image.open(review/item['render']) as im:
                im.thumbnail((520,220));x=(j%3)*533;y=(j//3)*250
                sheet.paste(im,(x,y+23));draw.text((x+5,y+4),f'Slide {item["page"]}',fill='black')
        sheet.save(review/f'contact_{start+1:02d}_{start+len(selected):02d}.png')
    audit={'slides':len(pages),'out_of_page_text':outside,'pages':pages,
           'scope':'PDF page bounds checked automatically; contact sheets and selected full renders inspected visually. This is not a GUI browser test.'}
    (review/'render_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'slides':len(pages),'out_of_page_text_count':len(outside)}))
    assert not outside,outside


if __name__=='__main__':main()
