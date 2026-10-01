"""Refresh the new ESI contents without round-tripping its embedded figures."""
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import fitz
from lxml import etree as E

import revise_prior_work_20260928 as revision
from refresh_revision_toc_20260922 import refresh


def main():
    audit = {}
    for suffix in ["", "_marked"]:
        path = revision.BASE / f"esi_revised_20260928{suffix}.docx"
        with fitz.open(revision.OUT / "rendered" / path.with_suffix(".pdf").name) as pdf:
            rows = refresh(path, pdf)
        with ZipFile(path) as archive:
            parts = {name: archive.read(name) for name in archive.namelist()}
        doc = E.fromstring(parts["word/document.xml"])
        labels = revision.bold_references(doc)
        revision.intro_tools.write_package(parts, doc, path)
        audit[suffix or "clean"] = rows
        if not suffix:
            with (revision.OUT / "esi_bold_references.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["paragraph", "label", "paragraph_text"])
                writer.writeheader()
                writer.writerows(labels)
    (revision.OUT / "contents_page_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Refreshed {len(rows)} contents entries in both ESI copies.")


if __name__ == "__main__":
    main()
