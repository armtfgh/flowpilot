"""Archive a read-only, structured inspection of the three supplied Word files."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
import json
import io
from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript/Submission'
OUT=BASE/'revision_20260929'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
NS={'w':W[1:-1],'a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def main():
    names={'main':'manuscript_revised_20260928.docx','esi':'esi_revised_20260928.docx','khu':'Supplementary Information (KRICT)_final (1).docx'}
    manifest={}
    for tag,name in names.items():
        path=BASE/name
        folder=OUT/'sources'/tag
        folder.mkdir(parents=True,exist_ok=True)
        with ZipFile(path) as z:
            doc=E.fromstring(z.read('word/document.xml'))
            rels={r.get('Id'):r.get('Target') for r in E.fromstring(z.read('word/_rels/document.xml.rels'))}
            body=doc.find(W+'body')
            rows=[]; thumbs=[]
            for i,n in enumerate(body):
                value=''.join(n.xpath('.//w:t/text()',namespaces=NS))
                row={'index':i,'type':E.QName(n).localname,'text':value,'images':[]}
                for rid in n.xpath('.//a:blip/@r:embed',namespaces=NS):
                    target=rels[rid]; data=z.read('word/'+target)
                    dest=folder/Path(target).name
                    dest.write_bytes(data)
                    item={'rid':rid,'target':target,'bytes':len(data),'sha256':sha256(data).hexdigest()}
                    try:
                        im=Image.open(io.BytesIO(data)).convert('RGBA')
                        item['pixels']=list(im.size)
                        im.thumbnail((950,530))
                        canvas=Image.new('RGB',(980,570),'white')
                        canvas.paste(im,((980-im.width)//2,35+(530-im.height)//2),im)
                        ImageDraw.Draw(canvas).text((10,8),f'{tag} paragraph {i}: {Path(target).name}',font=ImageFont.load_default(size=18),fill='black')
                        thumbs.append(canvas)
                    except Exception as exc:item['preview_error']=str(exc)
                    row['images'].append(item)
                if n.tag==W+'tbl':
                    row['rows']=[[''.join(c.xpath('.//w:t/text()',namespaces=NS)) for c in tr.findall(W+'tc')] for tr in n.findall(W+'tr')]
                rows.append(row)
            (folder/'document.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
            (folder/'document.txt').write_text('\n\n'.join(f"[{r['index']}] {r['type']} {r['text']}" for r in rows)+'\n')
            if 'word/comments.xml' in z.namelist():
                (folder/'comments.xml').write_bytes(z.read('word/comments.xml'))
            for offset in range(0,len(thumbs),6):
                chunk=thumbs[offset:offset+6]
                sheet=Image.new('RGB',(1960,570*((len(chunk)+1)//2)),'#ddd')
                for j,im in enumerate(chunk):sheet.paste(im,(980*(j%2),570*(j//2)))
                sheet.save(folder/f'contact_{offset//6+1:02d}.png')
            manifest[tag]={'path':str(path.relative_to(ROOT)),'sha256':sha256(path.read_bytes()).hexdigest(),'body_items':len(rows),'images':sum(len(r['images']) for r in rows)}
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
