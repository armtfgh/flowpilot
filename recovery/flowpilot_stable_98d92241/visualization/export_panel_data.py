"""
Export numeric data behind visualization figure panels to CSV.

This script mirrors the current logic in the figure generator modules and writes
panel-level CSV files to visualization/panel_data_exports/.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from data_loader import (  # noqa: E402
    load_records,
    get_reaction_classes,
    get_reactor_types,
    get_reactor_materials,
    get_bond_types,
    get_light_sources,
    get_pump_inlets,
    get_yields,
    get_classified_counts,
)
from fig2a_rule_landscape import CATEGORY_RENAME  # noqa: E402
from fig3b_query_enrichment import (  # noqa: E402
    _build_naive_query,
    _build_flora_query,
    _count_info_terms,
    _flora_fields,
    _context_summary,
    _load_rich_records,
)
from fig3c_score_decomposition import run_comparison  # noqa: E402
from fig3d_rag_quality import (  # noqa: E402
    _compute_family_match_rates,
    _field_score as rag_quality_field_score,
    _find_demo_query,
    _load_chroma,
    _pc_family as rag_quality_pc_family,
    _retrieve,
    _row_match,
)
from llm_classifier import load_classifications, _get_record_id  # noqa: E402
from rule_classifier import (  # noqa: E402
    CHEMISTRY_CLASSES,
    load_rule_classifications,
    load_rules,
)

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover
    raise ImportError("networkx is required for network-panel exports") from exc


OUT_DIR = ROOT / "panel_data_exports"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _manifest_row(file_name: str, figure: str, panel: str, description: str) -> dict:
    return {
        "file_name": file_name,
        "figure": figure,
        "panel": panel,
        "description": description,
    }


def _counts_with_cache(records: list[dict], field: str, fallback_fn) -> Counter:
    cached = get_classified_counts(field, records)
    return cached if cached else fallback_fn(records)


def export_fig1(records: list[dict], manifest: list[dict]) -> None:
    counts = _counts_with_cache(records, "chemistry_class", get_reaction_classes)
    total = sum(counts.values())
    rows = []
    for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        rows.append({
            "category": label,
            "count": count,
            "percent": round(100.0 * count / total, 4) if total else 0.0,
            "rank_desc": len(rows) + 1,
        })
    _write_csv(OUT_DIR / "fig1a_reaction_classes.csv", rows)
    manifest.append(_manifest_row("fig1a_reaction_classes.csv", "Fig1A", "A", "Reaction class counts and percentages"))

    counts = _counts_with_cache(records, "reactor_type", get_reactor_types)
    total = sum(counts.values())
    rows = []
    for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        rows.append({
            "category": label,
            "count": count,
            "percent": round(100.0 * count / total, 4) if total else 0.0,
            "rank_desc": len(rows) + 1,
        })
    _write_csv(OUT_DIR / "fig1b_reactor_types.csv", rows)
    manifest.append(_manifest_row("fig1b_reactor_types.csv", "Fig1B", "B", "Reactor type counts and percentages"))

    counts = _counts_with_cache(records, "reactor_material", get_reactor_materials)
    total = sum(counts.values())
    rows = []
    for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        rows.append({
            "category": label,
            "count": count,
            "percent": round(100.0 * count / total, 4) if total else 0.0,
            "rank_desc": len(rows) + 1,
        })
    _write_csv(OUT_DIR / "fig1c_reactor_materials.csv", rows)
    manifest.append(_manifest_row("fig1c_reactor_materials.csv", "Fig1C", "C", "Reactor material counts and donut/bar percentages"))

    counts = _counts_with_cache(records, "bond_type", get_bond_types)
    total = sum(counts.values())
    rows = []
    for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        rows.append({
            "category": label,
            "count": count,
            "percent": round(100.0 * count / total, 4) if total else 0.0,
            "rank_desc": len(rows) + 1,
        })
    _write_csv(OUT_DIR / "fig1d_bond_types.csv", rows)
    manifest.append(_manifest_row("fig1d_bond_types.csv", "Fig1D", "D", "Bond-type counts and percentages"))

    counts = _counts_with_cache(records, "light_source", get_light_sources)
    total = sum(counts.values())
    dark = counts.get("None / Dark", 0)
    photo_counts = {k: v for k, v in counts.items() if k != "None / Dark"}
    photo_total = sum(photo_counts.values())

    rows = []
    for label, count in sorted(photo_counts.items(), key=lambda kv: kv[1], reverse=True):
        rows.append({
            "subset": "photochem_only",
            "category": label,
            "count": count,
            "percent_of_photo": round(100.0 * count / photo_total, 4) if photo_total else 0.0,
            "percent_of_total": round(100.0 * count / total, 4) if total else 0.0,
        })
    rows.extend([
        {
            "subset": "corpus_split",
            "category": "Photochem",
            "count": photo_total,
            "percent_of_photo": 100.0 if photo_total else 0.0,
            "percent_of_total": round(100.0 * photo_total / total, 4) if total else 0.0,
        },
        {
            "subset": "corpus_split",
            "category": "Thermal/Dark",
            "count": dark,
            "percent_of_photo": 0.0,
            "percent_of_total": round(100.0 * dark / total, 4) if total else 0.0,
        },
    ])
    _write_csv(OUT_DIR / "fig1e_light_sources.csv", rows)
    manifest.append(_manifest_row("fig1e_light_sources.csv", "Fig1E", "E", "Photochemical light-source breakdown and photochem-vs-dark corpus split"))

    inlet_counts = get_pump_inlets(records)
    total = sum(inlet_counts.values())
    rows = []
    for inlet_n in sorted(inlet_counts):
        count = inlet_counts[inlet_n]
        rows.append({
            "inlet_streams": inlet_n,
            "count": count,
            "percent": round(100.0 * count / total, 4) if total else 0.0,
        })
    _write_csv(OUT_DIR / "fig1f_inlet_streams.csv", rows)
    manifest.append(_manifest_row("fig1f_inlet_streams.csv", "Fig1F", "F", "Counts per number of inlet streams"))

    flow_yields = get_yields(records)
    batch_yields = []
    for record in records:
        value = record.get("batch_baseline", {}).get("yield_percent")
        try:
            value = float(value)
            if 0 <= value <= 100:
                batch_yields.append(value)
        except (TypeError, ValueError):
            pass

    _write_csv(
        OUT_DIR / "fig1g_flow_yields_raw.csv",
        [{"series": "flow", "yield_percent": value} for value in flow_yields],
        ["series", "yield_percent"],
    )
    _write_csv(
        OUT_DIR / "fig1g_batch_yields_raw.csv",
        [{"series": "batch", "yield_percent": value} for value in batch_yields],
        ["series", "yield_percent"],
    )
    flow_hist, flow_edges = np.histogram(flow_yields, bins=20)
    rows = []
    for idx, count in enumerate(flow_hist):
        rows.append({
            "series": "flow",
            "bin_index": idx + 1,
            "bin_left": round(float(flow_edges[idx]), 6),
            "bin_right": round(float(flow_edges[idx + 1]), 6),
            "count": int(count),
        })
    if batch_yields:
        batch_hist, batch_edges = np.histogram(batch_yields, bins=20)
        for idx, count in enumerate(batch_hist):
            rows.append({
                "series": "batch",
                "bin_index": idx + 1,
                "bin_left": round(float(batch_edges[idx]), 6),
                "bin_right": round(float(batch_edges[idx + 1]), 6),
                "count": int(count),
            })
    _write_csv(OUT_DIR / "fig1g_yield_histograms.csv", rows)
    _write_csv(
        OUT_DIR / "fig1g_yield_summary.csv",
        [
            {
                "series": "flow",
                "n": len(flow_yields),
                "mean": round(float(np.mean(flow_yields)), 6) if flow_yields else "",
                "median": round(float(np.median(flow_yields)), 6) if flow_yields else "",
                "min": round(float(np.min(flow_yields)), 6) if flow_yields else "",
                "max": round(float(np.max(flow_yields)), 6) if flow_yields else "",
            },
            {
                "series": "batch",
                "n": len(batch_yields),
                "mean": round(float(np.mean(batch_yields)), 6) if batch_yields else "",
                "median": round(float(np.median(batch_yields)), 6) if batch_yields else "",
                "min": round(float(np.min(batch_yields)), 6) if batch_yields else "",
                "max": round(float(np.max(batch_yields)), 6) if batch_yields else "",
            },
        ],
    )
    manifest.extend([
        _manifest_row("fig1g_flow_yields_raw.csv", "Fig1G", "G", "Raw flow-yield values"),
        _manifest_row("fig1g_batch_yields_raw.csv", "Fig1G", "G", "Raw batch-yield values"),
        _manifest_row("fig1g_yield_histograms.csv", "Fig1G", "G", "Histogram bin counts for flow and batch yield overlays"),
        _manifest_row("fig1g_yield_summary.csv", "Fig1G", "G", "Summary statistics used for mean/median lines"),
    ])

    classifications = load_classifications()
    papers = []
    for record in records:
        rec_id = _get_record_id(record)
        clf = classifications.get(rec_id)
        if clf:
            papers.append((rec_id, clf))

    node_counts: dict[tuple[str, str], int] = defaultdict(int)
    edge_weights: dict[tuple[tuple[str, str], tuple[str, str]], int] = defaultdict(int)
    categories = [
        "chemistry_class",
        "reactor_type",
        "reactor_material",
        "bond_type",
        "light_source",
    ]

    for _, paper in papers:
        for category in categories:
            value = paper.get(category, "")
            if value:
                node_counts[(category, value)] += 1
        for i, cat_a in enumerate(categories):
            for cat_b in categories[i + 1:]:
                val_a = paper.get(cat_a, "")
                val_b = paper.get(cat_b, "")
                if val_a and val_b:
                    key = tuple(sorted(((cat_a, val_a), (cat_b, val_b))))
                    edge_weights[key] += 1

    def is_other(value: str) -> bool:
        return value.lower().startswith("other") if value else True

    graph = nx.Graph()
    for (category, value), count in node_counts.items():
        if is_other(value):
            continue
        graph.add_node((category, value), category=category, label=value, count=count)
    min_edge_weight = 3
    for ((cat_a, val_a), (cat_b, val_b)), weight in edge_weights.items():
        if weight < min_edge_weight:
            continue
        node_a = (cat_a, val_a)
        node_b = (cat_b, val_b)
        if node_a in graph.nodes and node_b in graph.nodes:
            graph.add_edge(node_a, node_b, weight=weight)

    pos = nx.spring_layout(graph, k=2.8, iterations=200, seed=42, weight="weight")
    node_rows = []
    for (category, value), data in graph.nodes(data=True):
        x, y = pos[(category, value)]
        node_rows.append({
            "node_id": f"{category}::{value}",
            "category": category,
            "label": value,
            "count": data["count"],
            "x": round(float(x), 8),
            "y": round(float(y), 8),
        })
    edge_rows = []
    for node_a, node_b, data in graph.edges(data=True):
        edge_rows.append({
            "source_node_id": f"{node_a[0]}::{node_a[1]}",
            "target_node_id": f"{node_b[0]}::{node_b[1]}",
            "source_category": node_a[0],
            "target_category": node_b[0],
            "weight": data["weight"],
        })
    _write_csv(OUT_DIR / "fig1h_knowledge_graph_nodes.csv", node_rows)
    _write_csv(OUT_DIR / "fig1h_knowledge_graph_edges.csv", edge_rows)
    manifest.extend([
        _manifest_row("fig1h_knowledge_graph_nodes.csv", "Fig1H", "H", "Knowledge-graph nodes with frequencies and deterministic layout positions"),
        _manifest_row("fig1h_knowledge_graph_edges.csv", "Fig1H", "H", "Knowledge-graph edge co-occurrence weights"),
    ])


def export_fig2(manifest: list[dict]) -> None:
    rules = load_rules()
    rule_cache = load_rule_classifications()

    counts = defaultdict(lambda: defaultdict(int))
    for rule in rules:
        counts[rule["category"]][rule["severity"]] += 1
    categories = sorted(counts.keys(), key=lambda cat: sum(counts[cat].values()), reverse=True)
    rows = []
    severities = ["hard_rule", "guideline", "tip", "safety"]
    for category in categories:
        total = sum(counts[category].values())
        for severity in severities:
            count = counts[category].get(severity, 0)
            rows.append({
                "category_key": category,
                "category_label": CATEGORY_RENAME.get(category, category.replace("_", " ").title()),
                "severity": severity,
                "count": count,
                "category_total": total,
            })
    _write_csv(OUT_DIR / "fig2a_rule_landscape.csv", rows)
    manifest.append(_manifest_row("fig2a_rule_landscape.csv", "Fig2A", "A", "Stacked-bar counts by rule category and severity"))

    total_per_cat = defaultdict(int)
    formula_per_cat = defaultdict(int)
    for rule in rules:
        category = rule["category"]
        total_per_cat[category] += 1
        if rule.get("quantitative", "").strip():
            formula_per_cat[category] += 1
    categories = sorted(total_per_cat.keys(), key=lambda cat: formula_per_cat[cat] / total_per_cat[cat], reverse=True)
    rows = []
    for category in categories:
        total = total_per_cat[category]
        formula_count = formula_per_cat[category]
        rows.append({
            "category_key": category,
            "category_label": CATEGORY_RENAME.get(category, category.replace("_", " ").title()),
            "total_rules": total,
            "rules_with_formula": formula_count,
            "percent_with_formula": round(100.0 * formula_count / total, 6) if total else 0.0,
        })
    _write_csv(OUT_DIR / "fig2b_formula_coverage.csv", rows)
    manifest.append(_manifest_row("fig2b_formula_coverage.csv", "Fig2B", "B", "Per-category quantitative-formula coverage"))

    cat_totals = Counter(rule["category"] for rule in rules)
    cats_ordered = [cat for cat, _ in cat_totals.most_common() if cat_totals[cat] >= 3]
    matrix = np.zeros((len(cats_ordered), len(CHEMISTRY_CLASSES)), dtype=int)
    for rule in rules:
        category = rule["category"]
        if category not in cats_ordered:
            continue
        clf = rule_cache.get(rule["rule_id"], {})
        applicable = clf.get("applicable_chemistry_classes", [])
        row_idx = cats_ordered.index(category)
        for chemistry_class in applicable:
            if chemistry_class in CHEMISTRY_CLASSES:
                col_idx = CHEMISTRY_CLASSES.index(chemistry_class)
                matrix[row_idx, col_idx] += 1
    long_rows = []
    wide_rows = []
    for row_idx, category in enumerate(cats_ordered):
        row = {
            "category_key": category,
            "category_label": CATEGORY_RENAME.get(category, category.replace("_", " ").title()),
        }
        total = 0
        for col_idx, chemistry_class in enumerate(CHEMISTRY_CLASSES):
            value = int(matrix[row_idx, col_idx])
            total += value
            row[chemistry_class] = value
            long_rows.append({
                "category_key": category,
                "category_label": CATEGORY_RENAME.get(category, category.replace("_", " ").title()),
                "chemistry_class": chemistry_class,
                "count": value,
            })
        row["row_total"] = total
        wide_rows.append(row)
    _write_csv(OUT_DIR / "fig2c_coverage_heatmap_long.csv", long_rows)
    _write_csv(OUT_DIR / "fig2c_coverage_heatmap_matrix.csv", wide_rows)
    manifest.extend([
        _manifest_row("fig2c_coverage_heatmap_long.csv", "Fig2C", "C", "Long-format heatmap counts by category and chemistry class"),
        _manifest_row("fig2c_coverage_heatmap_matrix.csv", "Fig2C", "C", "Wide-format heatmap matrix with row totals"),
    ])

    concept_freq = Counter()
    concept_to_cat = defaultdict(Counter)
    edge_weights = defaultdict(int)
    for rule in rules:
        clf = rule_cache.get(rule["rule_id"], {})
        concepts = [concept.strip() for concept in clf.get("key_concepts", []) if concept.strip()]
        category = rule.get("category", "general")
        for concept in concepts:
            concept_freq[concept] += 1
            concept_to_cat[concept][category] += 1
        for idx, concept_a in enumerate(concepts):
            for concept_b in concepts[idx + 1:]:
                edge_weights[tuple(sorted([concept_a, concept_b]))] += 1

    valid_nodes = {concept for concept, freq in concept_freq.items() if freq >= 4}
    graph = nx.Graph()
    for concept in valid_nodes:
        dominant_category = concept_to_cat[concept].most_common(1)[0][0]
        graph.add_node(concept, freq=concept_freq[concept], category=dominant_category)
    for (concept_a, concept_b), weight in edge_weights.items():
        if concept_a in valid_nodes and concept_b in valid_nodes and weight >= 3:
            graph.add_edge(concept_a, concept_b, weight=weight)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    pos = nx.spring_layout(graph, k=2.2, iterations=250, seed=7, weight="weight")
    node_rows = []
    for concept, data in graph.nodes(data=True):
        x, y = pos[concept]
        node_rows.append({
            "concept": concept,
            "frequency": data["freq"],
            "dominant_category_key": data["category"],
            "dominant_category_label": CATEGORY_RENAME.get(data["category"], data["category"].replace("_", " ").title()),
            "x": round(float(x), 8),
            "y": round(float(y), 8),
        })
    edge_rows = []
    for concept_a, concept_b, data in graph.edges(data=True):
        edge_rows.append({
            "source_concept": concept_a,
            "target_concept": concept_b,
            "weight": data["weight"],
        })
    _write_csv(OUT_DIR / "fig2d_concept_network_nodes.csv", node_rows)
    _write_csv(OUT_DIR / "fig2d_concept_network_edges.csv", edge_rows)
    manifest.extend([
        _manifest_row("fig2d_concept_network_nodes.csv", "Fig2D", "D", "Concept-network nodes with frequencies and deterministic layout positions"),
        _manifest_row("fig2d_concept_network_edges.csv", "Fig2D", "D", "Concept-network edge co-occurrence weights"),
    ])


def export_fig3(manifest: list[dict]) -> None:
    fig3a_boxes = [
        {"panel": "standard_rag", "label": "Batch Protocol", "x": 0.0, "y": 0.88, "w": 0.42, "h": 0.075},
        {"panel": "standard_rag", "label": "LLM Query Summary", "x": 0.0, "y": 0.73, "w": 0.42, "h": 0.075},
        {"panel": "standard_rag", "label": "Text Embedding", "x": 0.0, "y": 0.57, "w": 0.42, "h": 0.075},
        {"panel": "standard_rag", "label": "Cosine Similarity", "x": 0.0, "y": 0.41, "w": 0.42, "h": 0.075},
        {"panel": "standard_rag", "label": "Sort by Semantic Score Only", "x": 0.0, "y": 0.25, "w": 0.42, "h": 0.075},
        {"panel": "standard_rag", "label": "Return Top-K Analogies", "x": 0.0, "y": 0.10, "w": 0.42, "h": 0.075},
        {"panel": "flowpilot", "label": "Batch Protocol", "x": 0.0, "y": 0.93, "w": 0.44, "h": 0.065},
        {"panel": "flowpilot", "label": "Chemistry Reasoning Agent", "x": 0.0, "y": 0.845, "w": 0.44, "h": 0.065},
        {"panel": "flowpilot", "label": "Plan-Aware Rich Query", "x": 0.0, "y": 0.755, "w": 0.44, "h": 0.065},
        {"panel": "flowpilot", "label": "Text Embedding", "x": 0.0, "y": 0.695, "w": 0.44, "h": 0.052},
        {"panel": "flowpilot", "label": "Step 1: Hard Metadata Filter", "x": 0.0, "y": 0.620, "w": 0.44, "h": 0.062},
        {"panel": "flowpilot", "label": "Step 2: Relax Filters", "x": 0.0, "y": 0.535, "w": 0.44, "h": 0.062},
        {"panel": "flowpilot", "label": "Step 3: No Filters", "x": 0.0, "y": 0.455, "w": 0.44, "h": 0.062},
        {"panel": "flowpilot", "label": "Field Similarity Scoring", "x": 0.0, "y": 0.360, "w": 0.44, "h": 0.065},
        {"panel": "flowpilot", "label": "Weighted Final Score", "x": 0.0, "y": 0.275, "w": 0.44, "h": 0.062},
        {"panel": "flowpilot", "label": "Return Top-K Analogies", "x": 0.0, "y": 0.185, "w": 0.44, "h": 0.062},
    ]
    _write_csv(OUT_DIR / "fig3a_architecture_boxes.csv", fig3a_boxes)
    _write_csv(
        OUT_DIR / "fig3a_architecture_constants.csv",
        [
            {"name": "semantic_weight", "value": 0.6},
            {"name": "field_weight", "value": 0.4},
            {"name": "standard_box_width", "value": 0.42},
            {"name": "flowpilot_box_width", "value": 0.44},
        ],
    )
    manifest.extend([
        _manifest_row("fig3a_architecture_boxes.csv", "Fig3A", "A", "Conceptual architecture layout box coordinates and dimensions"),
        _manifest_row("fig3a_architecture_constants.csv", "Fig3A", "A", "Architecture numeric constants shown in the diagram"),
    ])

    rich_records = _load_rich_records(n=2)
    rows = []
    for index, (record_id, meta, doc) in enumerate(rich_records, start=1):
        naive_text = _build_naive_query(meta, doc)
        flora_text = " ".join(fragment for fragment, _ in _build_flora_query(meta, doc))
        naive_count = _count_info_terms(naive_text)
        flora_count = _count_info_terms(flora_text)
        rows.append({
            "example_index": index,
            "record_id": record_id,
            "chemistry_class": meta.get("chemistry_class", ""),
            "naive_term_count": naive_count,
            "flowpilot_term_count": flora_count,
            "richness_multiplier": round(flora_count / max(naive_count, 1), 6),
            "flowpilot_field_count": len(_flora_fields(meta)),
            "context_sentence_count": len([s for s in doc.split(".") if len(s.strip()) > 15][:2]),
        })
    _write_csv(OUT_DIR / "fig3b_query_richness_examples.csv", rows)
    manifest.append(_manifest_row("fig3b_query_richness_examples.csv", "Fig3B", "B", "Example-level query term counts and richness multipliers"))

    comparison_data = run_comparison(n_queries=80)
    _write_csv(OUT_DIR / "fig3c_retrieval_pairs_raw.csv", comparison_data)
    sem = np.array([row["sem_score"] for row in comparison_data])
    final = np.array([row["final_score"] for row in comparison_data])
    field = np.array([row["field_score"] for row in comparison_data])
    deltas = np.array([row["rank_delta"] for row in comparison_data])
    pc_vals = [row["fs_pc"] for row in comparison_data if row["q_has_pc"]]
    sol_vals = [row["fs_sol"] for row in comparison_data if row["q_has_sol"]]
    wl_vals = [row["fs_wl"] for row in comparison_data if row["q_has_wl"]]
    component_rows = []
    for label, values in [
        ("Photocatalyst Class Match", pc_vals),
        ("Solvent Match", sol_vals),
        ("Wavelength Proximity", wl_vals),
    ]:
        arr = np.array(values) if values else np.array([])
        component_rows.append({
            "component": label,
            "mean_score": round(float(arr.mean()), 8) if len(arr) else 0.0,
            "percent_nonzero": round(float(100.0 * (arr > 0).mean()), 8) if len(arr) else 0.0,
            "n_values": int(len(arr)),
        })
    summary_rows = [
        {
            "metric": "n_pairs",
            "value": len(comparison_data),
        },
        {
            "metric": "pct_reranked",
            "value": round(float(100.0 * (1 - np.mean(deltas == 0))), 8),
        },
        {
            "metric": "mean_sem_score",
            "value": round(float(sem.mean()), 8),
        },
        {
            "metric": "mean_final_score",
            "value": round(float(final.mean()), 8),
        },
        {
            "metric": "mean_field_score",
            "value": round(float(field.mean()), 8),
        },
    ]
    _write_csv(OUT_DIR / "fig3c_component_summary.csv", component_rows)
    _write_csv(OUT_DIR / "fig3c_summary_metrics.csv", summary_rows)
    manifest.extend([
        _manifest_row("fig3c_retrieval_pairs_raw.csv", "Fig3C", "C", "Raw retrieval-pair scores and rank changes"),
        _manifest_row("fig3c_component_summary.csv", "Fig3C", "C", "Field-component means and nonzero rates"),
        _manifest_row("fig3c_summary_metrics.csv", "Fig3C", "C", "Top-line summary metrics for the reranking comparison"),
    ])

    embs, metas, ids, docs = _load_chroma()
    match_rates = _compute_family_match_rates(embs, metas)
    match_rows = []
    for family, stats in sorted(match_rates.items(), key=lambda kv: kv[1]["flora"], reverse=True):
        match_rows.append({
            "family": family,
            "semantic_match_rate_pct": round(float(stats["semantic"]), 8),
            "flowpilot_match_rate_pct": round(float(stats["flora"]), 8),
            "n_queries": int(stats["n"]),
            "absolute_improvement_pct": round(float(stats["flora"] - stats["semantic"]), 8),
        })
    _write_csv(OUT_DIR / "fig3d_family_match_rates.csv", match_rows)

    query_index = _find_demo_query(embs, metas, ids)
    query_meta = metas[query_index]
    top_sem = _retrieve(query_index, embs, metas, 5, False)
    top_flora = _retrieve(query_index, embs, metas, 5, True)
    example_rows = []
    for method, top_indices in [("semantic_only", top_sem), ("flowpilot", top_flora)]:
        for rank, result_index in enumerate(top_indices, start=1):
            result_meta = metas[result_index]
            example_rows.append({
                "method": method,
                "rank": rank,
                "query_id": ids[query_index],
                "query_photocatalyst_family": rag_quality_pc_family(query_meta.get("photocatalyst")) or "",
                "query_wavelength_nm": query_meta.get("wavelength_nm") or "",
                "result_id": ids[result_index],
                "result_photocatalyst": result_meta.get("photocatalyst") or "",
                "result_photocatalyst_family": rag_quality_pc_family(result_meta.get("photocatalyst")) or "",
                "result_wavelength_nm": result_meta.get("wavelength_nm") or "",
                "result_class": result_meta.get("chemistry_class") or result_meta.get("mechanism_type") or "",
                "match_level": _row_match(query_meta, result_meta),
                "field_score": round(float(rag_quality_field_score(query_meta, result_meta)), 8),
            })
    _write_csv(OUT_DIR / "fig3d_demo_retrieval_table.csv", example_rows)
    manifest.extend([
        _manifest_row("fig3d_family_match_rates.csv", "Fig3D", "D", "Photocatalyst-family match rates by family"),
        _manifest_row("fig3d_demo_retrieval_table.csv", "Fig3D", "D", "Concrete top-5 retrieval example for semantic-only vs FlowPilot"),
    ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    records = load_records()

    export_fig1(records, manifest)
    export_fig2(manifest)
    export_fig3(manifest)

    _write_csv(OUT_DIR / "manifest.csv", manifest)
    print(f"Exported {len(manifest)} CSV files to {OUT_DIR}")


if __name__ == "__main__":
    main()
