"""Submission figures from archived numbers and supplied KHU artwork, no model calls."""
from pathlib import Path
import csv
import json
import io
import struct
import hashlib
import numpy as np
from PIL import Image
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/Submission/revision_20260929'
FIG=OUT/'figures'
RAW=OUT/'source_data'
SRC=OUT/'sources'
BENCH=ROOT/'deliverables/manuscript_benchmark_visualizations_20260825'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
INK='#24333c'; BLUE='#5d88b5'; TEAL='#2c867a'; RULE='#d6dcdf'

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def save(fig,name):
    for suffix in ('png','pdf','svg'):fig.savefig(FIG/f'{name}.{suffix}',dpi=600,facecolor='white')
    fig.savefig(FIG/f'{name}_preview.png',dpi=150,facecolor='white')
    plt.close(fig)

def emf_assets():
    gallery=fitz.open(OUT/'source_emf_gallery.pdf')
    names=json.loads((OUT/'emf_gallery_pages.json').read_text())
    result={}
    for i,name in enumerate(names):
        page=gallery[i]
        # Inline EMF is below the filename; union the actual drawing/text bounds.
        boxes=[fitz.Rect(b) for kind,b in page.get_bboxlog() if b[1]>48 and kind!='ignore-text']
        rect=boxes[0]
        for b in boxes[1:]:rect |= b
        rect += (-2,-2,2,2)
        rect &= page.rect
        dest=fitz.open();p=dest.new_page(width=rect.width,height=rect.height)
        p.show_pdf_page(p.rect,gallery,i,clip=rect)
        stem=name.split('.')[0]
        dest.save(FIG/f'khu_{stem}.pdf')
        p.get_pixmap(dpi=400).save(FIG/f'khu_{stem}.png')
        (FIG/f'khu_{stem}.svg').write_text(p.get_svg_image())
        result[stem]={'page':i+1,'clip':list(rect),'source':name}
    (OUT/'khu_artwork_extraction.json').write_text(json.dumps(result,indent=2)+'\n')
    # Chemical sequences are extracted, not redrawn or inferred.
    for stem,band in [('image10',(.086,.412,.94,.645)),('image12',(.17,.373,.835,.68))]:
        doc=fitz.open(FIG/f'khu_{stem}.pdf');p=doc[0]
        r=fitz.Rect(p.rect.width*band[0],p.rect.height*band[1],p.rect.width*band[2],p.rect.height*band[3])
        # Fill generic condition annotations only in the main-text derivative.
        # The full original source figure above remains unchanged in the ESI.
        old,new,size=('EtOH/pH 9.00 buffer (5:1, v/v, M)','EtOH/pH 9 buffer (5:1); 1a 0.10 M',7.3) if stem=='image10' else ('DPDTC (equiv)','DPDTC (1.05 equiv)',8.3)
        for box in p.search_for(old):
            if not r.contains(box):continue
            p.draw_rect(box+(-1,-.3,1,.3),color=None,fill=(1,1,1))
            width=fitz.get_text_length(new,fontname='helv',fontsize=size)
            p.insert_text((box.x0+box.width/2-width/2,box.y1-1.7),new,fontname='helv',fontsize=size)
        p.get_pixmap(dpi=500,clip=r).save(FIG/f'chemistry_{stem}.png')

def figure1():
    image=SRC/'main/image1.png'
    im=Image.open(image);w,h=im.size
    doc=fitz.open();p=doc.new_page(width=w,height=h)
    p.insert_image(p.rect,filename=str(image))
    # Only the incorrect unit in the illustrative recipe is replaced. The value
    # and every other element are retained; this is a schematic, not run data.
    r=fitz.Rect(3970,3294,4090,3348)
    color=tuple(v/255 for v in im.getpixel((4110,3340))[:3])
    p.draw_rect(r,color=None,fill=color)
    p.insert_text((3978,3335),'bar',fontname='hebo',fontsize=47,color=(0,0,0))
    doc.save(FIG/'figure1_unit_corrected.pdf')
    p.get_pixmap().save(FIG/'figure1_unit_corrected.png')
    (OUT/'figure1_patch.json').write_text(json.dumps({'source_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),'region_pixels':list(r),'change':'BPR schematic unit min -> bar; value 12 retained'},indent=2)+'\n')

def figure4():
    paths={'scores':BENCH/'source_data_revised/fig01_model_architecture_mean_sd_revised.csv',
           'flags':BENCH/'source_data_revised/fig08_critical_flags_revised.csv',
           'cost':BENCH/'source_data_revised/fig18_flowpilot_quality_per_generation_cost_summary.csv',
           'modules':BENCH/'source_data/fig11_module_condition_scores.csv'}
    data={k:rows(p) for k,p in paths.items()}
    for k,p in paths.items():(RAW/f'figure4_{k}.csv').write_bytes(p.read_bytes())
    models=['Qwen3.6-27B','Qwen3.8-27B','GPT-4o','Claude Sonnet 4.6','Claude Opus 4.6']
    labels=['Qwen\n3.6-27B','Qwen\n3.8-27B','GPT-4o','Claude\nSonnet\n4.6','Claude\nOpus\n4.6']
    fig=plt.figure(figsize=(6.5,7.05))
    ax=fig.add_axes([.015,.755,.97,.24]);ax.axis('off')
    # Retain the supplied architecture panel; only quantitative plots are rebuilt.
    im=Image.open(SRC/'main/image4.png');crop=im.crop((0,0,im.width,round(im.height*.288)))
    ax.imshow(crop,aspect='equal')
    for box,key,title in [([.095,.425,.38,.27],'scores','b  Outcome quality'),([.59,.425,.38,.27],'flags','c  Critical-flag burden')]:
        a=fig.add_axes(box);x=np.arange(5)
        for j,(architecture,color) in enumerate([('One-shot',BLUE),('FlowPilot',TEAL)]):
            rs=[next(r for r in data[key] if r['model']==m and r['architecture']==architecture) for m in models]
            vals=[float(r['mean']) for r in rs];sd=[float(r['sample_sd']) for r in rs]
            a.bar(x+(j-.5)*.36,vals,width=.33,yerr=sd,color=color,capsize=2,error_kw={'lw':.7},label=architecture)
        a.set_xticks(x,labels,fontsize=7.8);a.tick_params(axis='x',length=0,pad=6)
        a.set_title(title,loc='left',fontweight='bold',fontsize=10,pad=22)
        a.set_ylim(0,1.04 if key=='scores' else 19)
        a.set_ylabel('Benchmark score' if key=='scores' else 'Flags per three-case repeat',fontsize=8.5)
        a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
        a.legend(fontsize=7.5,loc='lower left',bbox_to_anchor=(-.04,1.01),ncol=2,frameon=False,handlelength=1,columnspacing=1)
    a=fig.add_axes([.25,.074,.33,.24])
    ms=data['modules']
    names=[r['condition'].replace('Without ','No ').replace(' agent','') for r in ms]
    vals=[float(r['mean_score_0_1']) for r in ms];sd=[float(r['score_sd']) for r in ms]
    a.barh(range(len(ms)),vals,xerr=sd,color=[TEAL if 'Full' in n else BLUE for n in names],height=.57,error_kw={'lw':.7},capsize=2)
    a.set_yticks(range(len(ms)),names,fontsize=8);a.invert_yaxis();a.set_xlim(.4,1.04)
    a.set_xlabel('Benchmark score',fontsize=9);a.grid(axis='x',alpha=.18);a.set_axisbelow(True)
    fig.text(.025,.338,'d  Internal module ablation',fontsize=10,fontweight='bold')
    a=fig.add_axes([.71,.074,.265,.24])
    ds=sorted(data['cost'],key=lambda r:float(r['aggregate_quality_per_usd']),reverse=True)
    ys=range(5);v=[float(r['aggregate_quality_per_usd']) for r in ds]
    a.barh(ys,v,height=.57,color=TEAL);a.set_xscale('log');a.set_xlim(.9,32);a.invert_yaxis()
    a.set_yticks(ys,[r['model'].replace('Claude ','').replace('Qwen','Q') for r in ds],fontsize=7.7)
    for y,value in zip(ys,v):a.text(value*1.07,y,f'{value:.1f}',va='center',fontsize=8)
    a.set_xlabel('Score / USD (log scale)',fontsize=8.3);a.grid(axis='x',alpha=.18);a.set_axisbelow(True)
    fig.text(.64,.338,'e  Cost-efficiency ranking',fontsize=10,fontweight='bold')
    save(fig,'figure4_five_model_cohort')

class Page:
    def __init__(self,h=7.05):
        self.h=h;self.fig=plt.figure(figsize=(6.5,h));self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set(xlim=(0,6.5),ylim=(h,0));self.ax.axis('off');self.labels=[]
    def text(self,x,y,s,size=9,weight='normal',color=INK,ha='left'):
        a=self.ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,ha=ha,va='top',linespacing=1.25);self.labels.append(a);return a
    def heading(self,letter,y,s):self.text(.12,y,letter,11,'bold');self.text(.36,y,s,10,'bold')
    def image(self,path,x,y,w,h):
        im=Image.open(path);scale=min(w/im.width,h/im.height);iw,ih=im.width*scale,im.height*scale
        a=self.fig.add_axes([(x+(w-iw)/2)/6.5,1-(y+ih)/self.h,iw/6.5,ih/self.h]);a.imshow(im);a.axis('off')
    def table(self,y,headers,values,widths,rowh=.4,size=9):
        x=.14
        for i,row in enumerate([headers,*values]):
            self.ax.add_patch(Rectangle((x,y+i*rowh),sum(widths),rowh,facecolor='#e6e6e6' if i==0 else ('#f5f7f7' if i%2 else 'white'),edgecolor='none'))
            xx=x
            for s,w in zip(row,widths):self.text(xx+.07,y+i*rowh+.075,s,size,'bold' if i==0 else 'normal');xx+=w
            self.ax.plot([x,x+sum(widths)],[y+(i+1)*rowh]*2,color=RULE,lw=.5)
    def save(self,name):
        self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer();issues=[]
        boxes=[(t.get_text(),t.get_window_extent(renderer)) for t in self.labels]
        for i,(txt,b) in enumerate(boxes):
            if b.x0<0 or b.y0<0 or b.x1>self.fig.bbox.width or b.y1>self.fig.bbox.height:issues.append('Outside: '+txt)
            for other,c in boxes[i+1:]:
                if min(b.x1,c.x1)-max(b.x0,c.x0)>1 and min(b.y1,c.y1)-max(b.y0,c.y0)>1:issues.append('Overlap: '+txt+' / '+other)
        assert not issues,issues
        save(self.fig,name)

def process(page,figure):
    # Reader-facing summary of the documented connected train. Complete archived
    # icons and equipment IDs remain in Figures S20-S21.
    y=2.27
    stages=[(.28,1.10,'Feed', '#edf2f7'),(1.68,1.10,'R1','#e4f0ec'),(3.12,.85,'T-mixer','#e8edf2'),(4.28,1.12,'R2','#e4f0ec'),(5.72,.60,'BPR','#f2ece4')]
    for x,w,label,color in stages:
        page.ax.add_patch(Rectangle((x,y),w,.44,facecolor=color,edgecolor='#8698a4',linewidth=.8))
        page.text(x+w/2,y+.13,label,10,'bold',ha='center')
    for x0,x1 in [(1.38,1.68),(2.78,3.12),(3.97,4.28),(5.40,5.72)]:
        page.ax.add_patch(FancyArrowPatch((x0,y+.22),(x1,y+.22),arrowstyle='-|>',mutation_scale=9,color=INK,lw=.9))
    page.ax.add_patch(FancyArrowPatch((3.545,2.08),(3.545,y),arrowstyle='-|>',mutation_scale=9,color=INK,lw=.9))
    if figure==5:
        page.text(3.545,1.79,'Pure O$_2$: 0.090 mL/min (STP)\nMFC + check valve',8.7,ha='center')
        desc=[(.83,'0.10 M 1a\n0.020 mL/min'),(2.23,'PFA, 2 or 5 mL\nUV-150, 450 nm'),(4.84,'FEP, 20 mL\nStrip LED, ~448 nm'),(6.02,'7 bar\ncartridge')]
        for x,s in desc:page.text(x,2.78,s,8.5,ha='center')
        page.text(.25,3.21,'Ar-prepared feed; O$_2$ enters only after R1. Both stages: 25 °C.',9)
    else:
        page.text(3.545,1.79,'Benzylamine: 2.10 M\n0.040 or 0.070 mL/min',8.7,ha='center')
        desc=[(.83,'Acid: 0.50 M\n0.160 or 0.280'),(2.23,'ETFE\n5 or 10 mL'),(4.84,'ETFE\n5 mL'),(6.02,'7 bar\ncartridge')]
        for x,s in desc:page.text(x,2.78,s,8.5,ha='center')
        page.text(.25,3.21,'Flows: mL/min. Both stages: 95 °C; no intermediate isolation.',9)

def case(figure):
    page=Page(7.15)
    page.heading('a',.06,'Connected chemical sequence')
    page.image(FIG/f'chemistry_image{10 if figure==5 else 12}.png',.10,.27,6.3,1.23)
    page.heading('b',1.57,'FlowPilot proposal: connected reactor train')
    process(page,figure)
    page.heading('c',3.46,'Reported operation and nominal stage times')
    if figure==5:
        page.table(3.73,['Response\nset(s)','R1 / R2\n(mL)','R1 time\n(min)','R2 index*\n(min)','Recorded\npressure (bar)'],[
            ['1 and 3','2 / 20','100','181.8','5.1–5.4'],['2','5 / 20','250','181.8','5.2–5.6']], [1.00,1.08,1.04,1.20,1.90],rowh=.48,size=9)
        page.heading('d',5.39,'Measured sulfoxide 4a yields')
        page.table(5.67,['Response set(s)','NMR yield (%)','Isolated yield (%)'],[['1 and 3','97','94'],['2','98','Not reported']],[2.07,2.08,2.07],rowh=.33,size=9)
        page.text(.17,6.77,'*R2 index = 20/(0.020 + 0.090), not operating-pressure residence time.',8)
    else:
        page.table(3.73,['Response\nset(s)','Q$_A$ / Q$_B$\n(mL/min)','R1 / R2\n(mL)','R1 / R2\ntime (min)','Recorded\npressure (bar)'],[
            ['1','0.160 / 0.040','5 / 5','31.25 / 25.00','5.2–5.3'],['2 and 3','0.280 / 0.070','10 / 5','35.71 / 14.29','5.5–5.8']], [1.00,1.47,1.00,1.43,1.32],rowh=.48,size=8.5)
        page.heading('d',5.39,'Measured amide 3b yields')
        page.table(5.67,['Response set(s)','NMR yield (%)','Isolated yield (%)'],[['1','86','84'],['2 and 3','68','Not reported']],[2.07,2.08,2.07],rowh=.33,size=9)
        page.text(.17,6.77,'R1 time = V$_1$/Q$_A$; R2 time = V$_2$/(Q$_A$ + Q$_B$).',8.6)
    page.text(.17,6.98,'Shared-set entries are reported as supplied; experimental repeat counts were not provided.',7.7)
    page.save(f'figure{figure}_experimental_revision')

def led_figures():
    for letter,paths,title in [('a',['image1.png','image2.png'],'Batch: two MR16 5 W lamps'),('b',['image6.png'],'Flow R1: Vapourtec 450 nm LED'),('c',['image8.png'],'Flow R2: blue LED strip')]:
        p=Page(3.8);p.heading(letter,.08,title)
        if len(paths)==2:
            p.image(SRC/'khu'/paths[0],.2,.5,1.45,2.7);p.image(SRC/'khu'/paths[1],1.82,.5,4.48,2.7)
        else:p.image(SRC/'khu'/paths[0],.12,.5,6.25,2.7)
        p.text(.16,3.44,'Original KHU lamp photograph and emission-spectrum image; specifications in Table S37.',8)
        p.save(f'figureS23{letter}_irradiation')

def main():
    FIG.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    emf_assets();figure1();figure4();case(5);case(6);led_figures()
    print('Built Figure 1 unit patch, five-model Figure 4, experimental Figures 5-6, and KHU artwork.')

if __name__=='__main__':main()
