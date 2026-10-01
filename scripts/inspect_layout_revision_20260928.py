"""Extract existing ESI graphics and inspection contact sheets, without editing sources."""
from pathlib import Path
from zipfile import ZipFile
import io
import json

from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript/layout_revision_20260928"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "a": "http://schemas.openxmlformats.org/drawingml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def main():
    folder = OUT / "original_figures"
    folder.mkdir(parents=True, exist_ok=True)
    manifest, thumbs = [], []
    font = ImageFont.load_default(size=18)
    with ZipFile(ROOT / "manuscript/esi_revised_20260928.docx") as archive:
        doc = E.fromstring(archive.read("word/document.xml"))
        rels = {r.get("Id"): r.get("Target") for r in E.fromstring(archive.read("word/_rels/document.xml.rels"))}
        for i, blip in enumerate(doc.xpath(".//a:blip", namespaces=NS), 1):
            target = rels[blip.get("{" + NS["r"] + "}embed")]
            data = archive.read("word/" + target)
            path = folder / f"{i:02d}_{Path(target).name}"
            path.write_bytes(data)
            row = {"index": i, "target": target, "extracted": str(path.relative_to(ROOT))}
            try:
                im = Image.open(io.BytesIO(data)).convert("RGBA")
                row["size"] = list(im.size)
                im.thumbnail((880, 425))
                canvas = Image.new("RGB", (920, 480), "white")
                canvas.paste(im, ((920-im.width)//2, 43+(425-im.height)//2), im)
                ImageDraw.Draw(canvas).text((16, 12), f"{i:02d}  {Path(target).name}", font=font, fill="black")
                thumbs.append(canvas)
            except Exception as exc:
                row["preview_note"] = str(exc)
            manifest.append(row)
    for offset in range(0, len(thumbs), 6):
        group = thumbs[offset:offset+6]
        sheet = Image.new("RGB", (1840, 480*((len(group)+1)//2)), "#dddddd")
        for i, im in enumerate(group):
            sheet.paste(im, (920*(i%2), 480*(i//2)))
        sheet.save(OUT / f"figures_contact_{offset//6+1:02d}.png")
    (OUT / "original_image_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Inspected {len(manifest)} image placements; contact sheets in {OUT}")


if __name__ == "__main__":
    main()
