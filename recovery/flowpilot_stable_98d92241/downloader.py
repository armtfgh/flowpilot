#%%
import json
import random
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

import requests


DOWNLOAD_DIR = Path("papers")
FAILED_LOG_FILE = Path("failed_downloads.json")
BASE_PAUSE_SECONDS = (1.5, 3.0)
LONG_PAUSE_EVERY = 10
LONG_PAUSE_SECONDS = (8.0, 12.0)
REQUEST_HEADERS = {
    "User-Agent": "Flow-Agent-Paper-Downloader/1.0 (mailto:armtfgh@postech.ac.kr)"
}


def normalize_doi(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""

    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
    )
    lower_value = value.lower()
    for prefix in prefixes:
        if lower_value.startswith(prefix):
            return value[len(prefix):].strip()
    return value


def cell_value(cell, namespace, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        text_nodes = cell.findall(".//main:t", namespace)
        return "".join(node.text or "" for node in text_nodes)

    value_node = cell.find("main:v", namespace)
    if value_node is None or value_node.text is None:
        return ""

    if cell_type == "s":
        index = int(value_node.text)
        return shared_strings[index] if 0 <= index < len(shared_strings) else ""

    return value_node.text


def read_shared_strings(workbook: zipfile.ZipFile, namespace) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    values = []

    for item in root.findall("main:si", namespace):
        text_nodes = item.findall(".//main:t", namespace)
        values.append("".join(node.text or "" for node in text_nodes))

    return values


def extract_dois_from_xlsx_file(xlsx_path: Path) -> list[str]:
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(xlsx_path, "r") as workbook:
        shared_strings = read_shared_strings(workbook, namespace)

        if "xl/worksheets/sheet1.xml" not in workbook.namelist():
            return []

        root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = root.findall(".//main:sheetData/main:row", namespace)
        parsed_rows = []

        for row in rows:
            values = []
            for cell in row.findall("main:c", namespace):
                values.append(cell_value(cell, namespace, shared_strings).strip())
            if values:
                parsed_rows.append(values)

    if not parsed_rows:
        return []

    header = [value.strip().lower() for value in parsed_rows[0]]
    if "doi" not in header:
        return []

    doi_index = header.index("doi")
    dois = []

    for row in parsed_rows[1:]:
        if doi_index >= len(row):
            continue
        doi = normalize_doi(row[doi_index])
        if doi:
            dois.append(doi)

    return dois


def discover_project_dois(project_dir: Path) -> list[dict]:
    records = []
    seen = set()

    for xlsx_path in sorted(project_dir.glob("*.xlsx")):
        try:
            dois = extract_dois_from_xlsx_file(xlsx_path)
        except Exception as e:
            print(f"[warn] Could not read {xlsx_path.name}: {e}")
            continue

        for doi in dois:
            if doi in seen:
                continue
            seen.add(doi)
            records.append({"doi": doi, "source_file": xlsx_path.name})

    return records


def load_failed_log(log_path: Path = FAILED_LOG_FILE) -> list[dict]:
    if not log_path.exists():
        return []

    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_failed_log(entries: list[dict], log_path: Path = FAILED_LOG_FILE):
    log_path.write_text(json.dumps(entries, indent=2, ensure_ascii=True), encoding="utf-8")


def record_failed_download(doi: str, source_file: str, reason: str):
    entries = load_failed_log()
    entries.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "doi": doi,
        "source_file": source_file,
        "reason": reason,
    })
    save_failed_log(entries)


def strategic_pause(attempt_index: int):
    sleep_seconds = random.uniform(*BASE_PAUSE_SECONDS)
    print(f"[pause] Sleeping {sleep_seconds:.1f}s before next DOI...")
    time.sleep(sleep_seconds)

    if attempt_index % LONG_PAUSE_EVERY == 0:
        long_sleep = random.uniform(*LONG_PAUSE_SECONDS)
        print(f"[pause] Long pause {long_sleep:.1f}s after {attempt_index} attempts...")
        time.sleep(long_sleep)


def download_pdf_from_doi(doi: str, session: requests.Session, output_dir: Path) -> tuple[bool, str]:
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"{doi.replace('/', '_')}.pdf"

    if filename.exists():
        return True, f"Already exists: {filename.name}"

    api_url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    resp = session.get(api_url, timeout=30)
    resp.raise_for_status()
    data = resp.json()["message"]

    pdf_url = None
    for link in data.get("link", []):
        if "application/pdf" in (link.get("content-type") or "").lower():
            pdf_url = link.get("URL")
            break

    if not pdf_url:
        return False, "No Crossref PDF link found"

    pdf_resp = session.get(pdf_url, stream=True, timeout=60, allow_redirects=True)
    if pdf_resp.status_code != 200:
        return False, f"PDF request returned {pdf_resp.status_code}"

    content_type = (pdf_resp.headers.get("content-type") or "").lower()
    if "pdf" not in content_type and "octet-stream" not in content_type:
        return False, f"Unexpected content-type: {content_type or 'unknown'}"

    with open(filename, "wb") as f:
        for chunk in pdf_resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return True, f"Saved {filename.name}"


def download_all_project_dois(project_dir: Path = Path("."), output_dir: Path = DOWNLOAD_DIR):
    doi_records = discover_project_dois(project_dir)
    print(f"Found {len(doi_records)} unique DOI(s) in project Excel files.")

    if not doi_records:
        return

    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    successes = 0
    failures = 0

    for index, record in enumerate(doi_records, start=1):
        doi = record["doi"]
        source_file = record["source_file"]
        print(f"\n[{index}/{len(doi_records)}] {doi}  (from {source_file})")

        try:
            ok, message = download_pdf_from_doi(doi, session=session, output_dir=output_dir)
            if ok:
                successes += 1
                print(f"[ok] {message}")
            else:
                failures += 1
                print(f"[fail] {message}")
                record_failed_download(doi, source_file, message)
        except Exception as e:
            failures += 1
            reason = str(e)
            print(f"[fail] {reason}")
            record_failed_download(doi, source_file, reason)

        if index < len(doi_records):
            strategic_pause(index)

    print(
        f"\nDone. Success: {successes}  |  Failed: {failures}  |  "
        f"Failure log: {FAILED_LOG_FILE}"
    )

#%%
if __name__ == "__main__":
    download_all_project_dois()

# %%
