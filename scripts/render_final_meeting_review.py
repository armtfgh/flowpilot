"""Render review sheets and locate figures in the Word-to-PDF exports."""
from pathlib import Path
import json
import re
import fitz

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/october2_final_review'

def main():
    shots=OUT/'page_previews';shots.mkdir(exist_ok=True)
    locations={}
    for label in ['manuscript','ESI']:
        pdf=fitz.open(OUT/'rendered'/f'FlowPilot_{label}_october2_final_real1.pdf')
        locations[label]={}
        for i,p in enumerate(pdf):
            for line in p.get_text().splitlines():
                match=re.match(r'^(Figure|Table) (S?\d+)\.',line)
                if match:locations[label][match[0][:-1]]=i+1
        for start in range(0,len(pdf),12):
            contact=fitz.open();page=contact.new_page(width=1800,height=2460)
            for j,i in enumerate(range(start,min(start+12,len(pdf)))):
                x=(j%3)*600;y=(j//3)*615
                page.insert_text((x+20,y+20),f'{label} page {i+1}',fontsize=17)
                rect=fitz.Rect(x+8,y+30,x+592,y+605)
                page.show_pdf_page(rect,pdf,i,keep_proportion=True)
            page.get_pixmap().save(shots/f'{label}_contact_{start+1:03d}.png')
            contact.close()
        for name,i in locations[label].items():
            pdf[i-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(shots/f'{label}_{name.replace(" ","_")}.png')
    (OUT/'figure_table_pages.json').write_text(json.dumps(locations,indent=2))
    print(json.dumps(locations,indent=2))

if __name__=='__main__':main()
