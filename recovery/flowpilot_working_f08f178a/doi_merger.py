import csv
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


DOIS_DIR = Path("dois")
OUTPUT_CSV = DOIS_DIR / "doi_merged.csv"
MAIN_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def cell_value(cell, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        text_nodes = cell.findall(".//main:t", MAIN_NS)
        return "".join(node.text or "" for node in text_nodes).strip()

    value_node = cell.find("main:v", MAIN_NS)
    if value_node is None or value_node.text is None:
        return ""

    if cell_type == "s":
        index = int(value_node.text)
        return shared_strings[index].strip() if 0 <= index < len(shared_strings) else ""

    return value_node.text.strip()


def read_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    values = []

    for item in root.findall("main:si", MAIN_NS):
        text_nodes = item.findall(".//main:t", MAIN_NS)
        values.append("".join(node.text or "" for node in text_nodes))

    return values


def worksheet_rows(xlsx_path: Path) -> list[list[str]]:
    with zipfile.ZipFile(xlsx_path, "r") as workbook:
        if "xl/worksheets/sheet1.xml" not in workbook.namelist():
            return []

        shared_strings = read_shared_strings(workbook)
        root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = root.findall(".//main:sheetData/main:row", MAIN_NS)
        parsed_rows = []

        for row in rows:
            values = []
            for cell in row.findall("main:c", MAIN_NS):
                values.append(cell_value(cell, shared_strings))
            if any(value.strip() for value in values):
                parsed_rows.append(values)

        return parsed_rows


def collect_workbooks(dois_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    headers: list[str] = []
    merged_rows: list[dict[str, str]] = []

    for xlsx_path in sorted(dois_dir.glob("*.xlsx")):
        rows = worksheet_rows(xlsx_path)
        if not rows:
            continue

        file_headers = [header.strip() for header in rows[0]]
        for header in file_headers:
            if header and header not in headers:
                headers.append(header)

        if "source_file" not in headers:
            headers.append("source_file")

        for row in rows[1:]:
            row_dict = {header: "" for header in headers}
            for index, header in enumerate(file_headers):
                if index < len(row):
                    row_dict[header] = row[index]
            row_dict["source_file"] = xlsx_path.name
            merged_rows.append(row_dict)

    return headers, merged_rows


def write_merged_csv(output_csv: Path, headers: list[str], rows: list[dict[str, str]]):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    headers, rows = collect_workbooks(DOIS_DIR)
    if not rows:
        print(f"No workbook rows found in {DOIS_DIR}")
        return

    write_merged_csv(OUTPUT_CSV, headers, rows)
    print(f"Merged {len(rows)} rows into {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
