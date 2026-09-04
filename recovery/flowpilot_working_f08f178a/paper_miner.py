# ─────────────────────────────────────────────────────────────
#  Literature Agent — Top-K Relevant Paper Finder
#  Requirements: pip install pyalex openai
# ─────────────────────────────────────────────────────────────
#%%
import json
import os
import re
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape

import openai
import pyalex
from pyalex import Works


# ── Configuration ─────────────────────────────────────────────
openai.api_key = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
HISTORY_FILE = "paper_miner_history.json"

pyalex.config.email = "armtfgh@postech.ac.kr"
pyalex.config.max_retries = 3
pyalex.config.retry_backoff_factor = 0.2
pyalex.config.retry_http_codes = [429, 500, 503]


# ── Helpers ───────────────────────────────────────────────────
def extract_abstract(work: dict) -> str:
    """
    PyAlex reconstructs abstracts through work["abstract"], but dict.get("abstract")
    returns None because the key is stored as abstract_inverted_index.
    """
    try:
        abstract = work["abstract"]
    except Exception:
        abstract = ""

    if not abstract:
        abstract = work.get("abstract", "") or ""

    return abstract


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


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


def get_open_access_metadata(work: dict) -> tuple[bool, str]:
    open_access = work.get("open_access") or {}
    primary_location = work.get("primary_location") or {}

    is_oa = normalize_bool(open_access.get("is_oa", False))
    oa_url = open_access.get("oa_url") or primary_location.get("landing_page_url") or ""

    return is_oa, oa_url


def topic_requires_flow(topic: str) -> bool:
    topic_lower = (topic or "").lower()
    flow_markers = (
        "flow chemistry",
        "continuous flow",
        "flow reaction",
        "flow reactor",
        "microreactor",
        "micro-reactor",
        "microfluidic",
        "in flow",
    )
    return any(marker in topic_lower for marker in flow_markers)


def text_has_flow_context(text: str) -> bool:
    text_lower = (text or "").lower()
    flow_markers = (
        "flow chemistry",
        "continuous flow",
        "flow reactor",
        "flow reaction",
        "microreactor",
        "micro-reactor",
        "microfluidic",
        "capillary reactor",
        "plug flow",
        "tubular reactor",
        "segmented flow",
        "mesofluidic",
        "in flow",
    )
    return any(marker in text_lower for marker in flow_markers)


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    text = text.strip("._")
    return text or "papers"


def excel_column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def make_inline_string_cell(cell_ref: str, value: str) -> str:
    return (
        f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
    )


def export_results_to_excel(papers: list[dict], topic: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"paper_results_{safe_filename(topic)}_{timestamp}.xlsx"

    rows = [["Title", "DOI", "Open Access"]]
    for paper in papers:
        rows.append([
            paper.get("title", "") or "",
            normalize_doi(paper.get("doi", "")),
            "Yes" if paper.get("is_open_access") else "No",
        ])

    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            cell_ref = f"{excel_column_name(col_idx)}{row_idx}"
            cells.append(make_inline_string_cell(cell_ref, str(value)))
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        f'{"".join(sheet_rows)}'
        "</sheetData>"
        "</worksheet>"
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        '<sheet name="Papers" sheetId="1" r:id="rId1"/>'
        "</sheets>"
        "</workbook>"
    )

    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    return filename


def load_history(history_file: str = HISTORY_FILE) -> dict:
    if not os.path.exists(history_file):
        return {"seen_oa_ids": [], "seen_dois": [], "records": []}

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"seen_oa_ids": [], "seen_dois": [], "records": []}

    if not isinstance(data, dict):
        return {"seen_oa_ids": [], "seen_dois": [], "records": []}

    data.setdefault("seen_oa_ids", [])
    data.setdefault("seen_dois", [])
    data.setdefault("records", [])
    return data


def save_history(history: dict, history_file: str = HISTORY_FILE):
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=True)


def build_seen_sets(history: dict) -> tuple[set[str], set[str]]:
    seen_oa_ids = {
        (oa_id or "").strip()
        for oa_id in history.get("seen_oa_ids", [])
        if (oa_id or "").strip()
    }
    seen_dois = {
        normalize_doi(doi)
        for doi in history.get("seen_dois", [])
        if normalize_doi(doi)
    }
    return seen_oa_ids, seen_dois


def is_previously_seen(paper: dict, seen_oa_ids: set[str], seen_dois: set[str]) -> bool:
    oa_id = (paper.get("oa_id", "") or "").strip()
    doi = normalize_doi(paper.get("doi", ""))
    return (oa_id and oa_id in seen_oa_ids) or (doi and doi in seen_dois)


def append_results_to_history(
    history: dict,
    papers: list[dict],
    topic: str,
    history_file: str = HISTORY_FILE,
):
    seen_oa_ids, seen_dois = build_seen_sets(history)

    for paper in papers:
        oa_id = (paper.get("oa_id", "") or "").strip()
        doi = normalize_doi(paper.get("doi", ""))

        if oa_id and oa_id not in seen_oa_ids:
            history["seen_oa_ids"].append(oa_id)
            seen_oa_ids.add(oa_id)

        if doi and doi not in seen_dois:
            history["seen_dois"].append(doi)
            seen_dois.add(doi)

        history["records"].append({
            "topic": topic,
            "oa_id": oa_id,
            "doi": doi,
            "title": paper.get("title", ""),
            "year": paper.get("year"),
        })

    save_history(history, history_file=history_file)


def rebuild_history_indexes(history: dict):
    seen_oa_ids = []
    seen_dois = []
    seen_oa_id_set = set()
    seen_doi_set = set()

    for record in history.get("records", []):
        oa_id = (record.get("oa_id", "") or "").strip()
        doi = normalize_doi(record.get("doi", ""))

        if oa_id and oa_id not in seen_oa_id_set:
            seen_oa_ids.append(oa_id)
            seen_oa_id_set.add(oa_id)

        if doi and doi not in seen_doi_set:
            seen_dois.append(doi)
            seen_doi_set.add(doi)

    history["seen_oa_ids"] = seen_oa_ids
    history["seen_dois"] = seen_dois


def rollback_history(k: int, history_file: str = HISTORY_FILE) -> list[dict]:
    history = load_history(history_file=history_file)
    records = history.get("records", [])

    if k <= 0 or not records:
        return []

    removed = records[-k:]
    history["records"] = records[:-k]
    rebuild_history_indexes(history)
    save_history(history, history_file=history_file)
    return removed


# ── Step 1: LLM query generation ──────────────────────────────
def generate_search_queries(topic: str) -> list[str]:
    prompt = f"""
You are an expert literature search assistant.

Topic to search: "{topic}"

Return a JSON object with this exact shape:
{{
  "search_queries": ["...", "...", "...", "...", "..."]
}}

Rules:
- Produce 5 to 8 OpenAlex-friendly search queries.
- Keep them specific to the topic.
- Include the exact user topic once if it is already a useful query.
- Add close synonyms, common technical names, abbreviations, or phrasing variants.
- Do not add batch/flow terms unless the topic itself asks for them.
- Do not add very broad generic chemistry phrases.
"""
    response = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = json.loads(response.choices[0].message.content)
    queries = raw.get("search_queries", [])

    if not isinstance(queries, list):
        raise ValueError("LLM query generation did not return a 'search_queries' list")

    cleaned = []
    seen = set()

    for query in [topic] + queries:
        query = (query or "").strip()
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(query)

    return cleaned


# ── Step 2: OpenAlex retrieval ────────────────────────────────
def fetch_candidates(
    search_queries: list[str],
    per_query: int = 20,
    seen_oa_ids: set[str] | None = None,
    seen_dois: set[str] | None = None,
    require_flow_context: bool = False,
) -> list[dict]:
    seen, papers = set(), []
    seen_oa_ids = seen_oa_ids or set()
    seen_dois = seen_dois or set()

    for query in search_queries:
        try:
            results = (
                Works()
                .search(query)
                .filter(type="article")
                .sort(relevance_score="desc")
                .get(per_page=per_query)
            )

            for work in results:
                oa_id = work.get("id", "")
                if not oa_id or oa_id in seen:
                    continue

                abstract = extract_abstract(work)
                if len(abstract) < 80:
                    continue

                if require_flow_context:
                    combined_text = (
                        f"{work.get('display_name', '')}\n{abstract}"
                    )
                    if not text_has_flow_context(combined_text):
                        continue

                is_oa, oa_url = get_open_access_metadata(work)
                seen.add(oa_id)
                papers.append({
                    "oa_id": oa_id.replace("https://openalex.org/", ""),
                    "doi": work.get("doi", ""),
                    "title": work.get("display_name", "No title"),
                    "abstract": abstract,
                    "year": work.get("publication_year"),
                    "cited_by": work.get("cited_by_count", 0),
                    "journal": (
                        (work.get("primary_location") or {})
                        .get("source", {})
                        .get("display_name", "")
                    ),
                    "is_open_access": is_oa,
                    "open_access_url": oa_url,
                    "matched_query": query,
                })

                if is_previously_seen(papers[-1], seen_oa_ids, seen_dois):
                    papers.pop()
        except Exception as e:
            print(f"  [warn] Search failed for '{query}': {e}")

    return papers


# ── Step 3: LLM relevance scoring ─────────────────────────────
def score_papers(papers: list[dict], topic: str, batch_size: int = 15) -> list[dict]:
    """
    LLM scores each abstract for relevance to the topic.
    """
    scored = []
    require_flow_context = topic_requires_flow(topic)

    for i in range(0, len(papers), batch_size):
        chunk = papers[i : i + batch_size]
        block = "\n\n".join(
            f"[{j}] TITLE: {p['title']}\nABSTRACT: {p['abstract'][:700]}"
            for j, p in enumerate(chunk)
        )
        prompt = f"""
Topic of interest: "{topic}"

Score each paper for relevance to the topic. Return a JSON object with this exact shape:
{{
  "papers": [
    {{
      "index": <int>,
      "relevance": <0|1|2|3>,
      "reason": "<one sentence>"
    }}
  ]
}}

Scoring:
  3 → directly focused on the topic
  2 → clearly relevant and useful
  1 → somewhat related but not central
  0 → irrelevant

Critical rule:
  If the topic explicitly requires flow chemistry / continuous flow, then any paper
  that does not explicitly describe a flow setup, flow reactor, continuous-flow
  method, microreactor, or in-flow experiment must be scored 0.
  Do not give relevance 1 or 2 to non-flow batch papers just because the chemistry matches.

Papers:
{block}
"""
        try:
            response = openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = json.loads(response.choices[0].message.content)
            items = raw.get("papers", [])

            if not isinstance(items, list):
                raise ValueError("LLM scoring response did not include a 'papers' list")

            for item in items:
                idx = item.get("index", -1)
                if 0 <= idx < len(chunk):
                    chunk[idx].update({
                        "relevance": int(item.get("relevance", 0)),
                        "reason": item.get("reason", ""),
                    })
                    scored.append(chunk[idx])
        except Exception as e:
            print(f"  [warn] Scoring failed for chunk starting at {i}: {e}")
            for paper in chunk:
                paper.setdefault("relevance", 0)
                paper.setdefault("reason", "Scoring failed.")
                scored.append(paper)

    return scored


# ── Selection & display ───────────────────────────────────────
def select_top(scored: list[dict], k: int, min_relevance: int = 2) -> list[dict]:
    eligible = [paper for paper in scored if paper.get("relevance", 0) >= min_relevance]
    eligible.sort(
        key=lambda x: (x.get("relevance", 0), x.get("cited_by", 0)),
        reverse=True,
    )
    return eligible[:k]


def display(papers: list[dict], topic: str, k: int):
    print(f"\n{'═' * 70}")
    print(f"  TOP {min(k, len(papers))} PAPERS FOR: {topic}")
    print(f"{'═' * 70}")

    for i, paper in enumerate(papers, 1):
        print(f"\n  [{i:02d}] {paper['title']} ({paper['year']})")
        print(f"       Journal   : {paper['journal']}")
        print(f"       DOI       : {paper['doi'] or 'N/A'}")
        print(f"       Cited by  : {paper['cited_by']}")
        print(f"       Open Acc. : {'yes' if paper.get('is_open_access') else 'no'}")
        if paper.get("open_access_url"):
            print(f"       OA URL    : {paper['open_access_url']}")
        print(f"       Relevance : {paper.get('relevance', '?')}/3")
        print(f"       Note      : {paper.get('reason', '')}")


# ── Main orchestrator ─────────────────────────────────────────
def run(topic: str, k: int = 10) -> list[dict]:
    print(f"\n{'─' * 70}")
    print(f"  Literature Agent  |  Topic: {topic}  |  Top K: {k}")
    print(f"{'─' * 70}")

    history = load_history()
    seen_oa_ids, seen_dois = build_seen_sets(history)
    print(
        f"\n[0/3] Loaded history: {len(seen_oa_ids)} seen OpenAlex IDs, "
        f"{len(seen_dois)} seen DOIs"
    )

    print("\n[1/3] Generating search queries...")
    search_queries = generate_search_queries(topic)
    for query in search_queries:
        print(f"  - {query}")

    per_query = max(10, min(30, k * 3))
    require_flow_context = topic_requires_flow(topic)
    print(
        f"\n[2/3] Searching OpenAlex with {len(search_queries)} queries "
        f"(up to {per_query} papers each)..."
    )
    if require_flow_context:
        print("  Flow gate enabled: candidates must explicitly mention flow context.")
    papers = fetch_candidates(
        search_queries,
        per_query=per_query,
        seen_oa_ids=seen_oa_ids,
        seen_dois=seen_dois,
        require_flow_context=require_flow_context,
    )
    print(f"  → {len(papers)} unseen candidate papers with abstracts")

    if not papers:
        print("\n  No unseen candidate papers were found.\n")
        return []

    print(f"\n[3/3] Scoring {len(papers)} candidate abstracts with LLM...")
    scored = score_papers(papers, topic)
    top_papers = select_top(scored, k, min_relevance=2)

    append_results_to_history(history, top_papers, topic)
    excel_file = export_results_to_excel(top_papers, topic)
    display(top_papers, topic, k)
    print(
        f"\n  Done. Returned {len(top_papers)} new paper(s) and updated "
        f"'{HISTORY_FILE}'."
    )
    print(f"  Excel file written: {excel_file}\n")
    return top_papers


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    mode = input("Mode [search/rollback]: ").strip().lower() or "search"

    if mode == "rollback":
        k_raw = input("Remove how many recent history items? ").strip()
        try:
            k = int(k_raw)
        except ValueError:
            k = 0

        removed = rollback_history(k)
        print(f"\nRemoved {len(removed)} item(s) from '{HISTORY_FILE}'.")
        for item in removed:
            print(f"  - {item.get('title', '')} | {item.get('doi', '')}")
        print()
    else:
        topic = input("Enter keyword/topic to search: ").strip()
        k_raw = input("How many papers do you want (K)? ").strip()
        try:
            k = int(k_raw)
        except ValueError:
            k = 10

        run(topic, k=max(1, k))
#%%
