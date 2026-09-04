#!/usr/bin/env python3
"""Refresh Writer indexes and fields in an existing DOCX through LibreOffice UNO."""

from __future__ import annotations

import sys
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


def prop(name: str, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: update_docx_indexes_uno.py DOCUMENT.docx")
    path = Path(sys.argv[1]).resolve()
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    context = resolver.resolve("uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext")
    desktop = context.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", context)
    document = desktop.loadComponentFromURL(
        uno.systemPathToFileUrl(str(path)),
        "_blank",
        0,
        (prop("Hidden", True), prop("ReadOnly", False), prop("UpdateDocMode", 3)),
    )
    if document is None:
        raise RuntimeError(f"Could not open {path}")
    indexes = document.getDocumentIndexes()
    for index in range(indexes.getCount()):
        indexes.getByIndex(index).update()
    document.getTextFields().refresh()
    document.store()
    document.close(True)


if __name__ == "__main__":
    main()
