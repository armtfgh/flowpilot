"""Read-only audit of frozen packet/evaluator evidence; no model calls or scoring."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ablation_test.src.newgen_2_outcome import _scrub

CAMPAIGN = ROOT / 'ablation_results/practical_input/claude_practical_repair_20261014'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
key = read(CAMPAIGN / 'frozen/candidate_key_confidential.json')['candidates']
key_by_id = {row['candidate_id']: row for row in key}
accepted = read(CAMPAIGN / 'accepted_judgments.json')
packets = {row['candidate_id']: read(CAMPAIGN / 'packets' / (row['candidate_id']+'.json')) for row in key}
judgments = {}
for row in accepted:
    path = (CAMPAIGN / row['status_path']).with_name('parsed_response.json')
    judgments[(row['candidate_id'],row['judge'])] = (path, read(path))

fresh_rows = []
all_ratings = []
for row in key:
    if row['cohort'] != 'fresh': continue
    cid = row['candidate_id']
    run = Path(row['run_directory'])
    result = read(run / 'result.json')
    packet_path = CAMPAIGN / 'packets' / (cid+'.json')
    packet = packets[cid]
    metrics = read(run/'metrics.json')
    one_shot = row['architecture'] != 'FlowPilot'
    prose_equal = None
    finish = None
    if one_shot:
        raw = read(run/'raw_response.json')
        prose_equal = _scrub(raw['text']) == packet['delivered_final_outcome']['chemistry_and_operations']['chemistry_notes']
        finish = raw['stop_reason']
    scores = {}
    for judge in ['openai','qwen']:
        path,response = judgments[(cid,judge)]
        scores[judge] = {r['criterion_id']:r['score'] for r in response['criterion_scores']}
        all_ratings += [{**r,'judge':judge,'candidate_id':cid} for r in response['criterion_scores'] if r['score'] is not None]
    fresh_rows.append({'candidate_id':cid,'case':row['case'],'architecture':row['architecture'],
        'packet_path':str(packet_path.relative_to(ROOT)),'packet_sha256':sha(packet_path),
        'packet_file_bytes':packet_path.stat().st_size,
        'full_prose_preserved_after_identity_scrub':prose_equal,'generation_stop_reason':finish,
        'software_confidence':packet['delivered_final_outcome']['evidence_and_reasoning'].get('confidence'),
        'deterministic_verification_sheet':packet['deterministic_verification_sheet'],
        'source_exclusion_metrics':{k:metrics.get(k) for k in ['prompt_source_id_leaks','retrieval_source_leaks','leave_one_source_out_pass']},
        'criterion_scores':scores})

historical = list(csv.DictReader((ROOT/'deliverables/newgen_2_0_latest_results_20260824/tables/all_candidate_consensus_scores.csv').open()))
models = {'Qwen3.6-27B','Qwen3.8-27B','GPT-4o','Claude Sonnet 4.6','Claude Opus 4.6'}
campaigns = sorted({str(Path(row['run_directory']).parents[3]) for row in historical if row['model'] in models})
old_sources = []
for directory in campaigns:
    p=Path(directory)
    old_sources.append({'campaign':str(p.relative_to(ROOT)),
        'rubric_sha256':sha(p/'frozen/outcome_rubric.json'),
        'formatter_sha256':sha(p/'frozen/source_code/newgen_2_outcome.py'),
        'manifest':read(p/'frozen/campaign_manifest.json')})
normalization_changes=[]
for row in historical:
    if row['model'] not in models or row['architecture']!='One-shot':continue
    path=Path(row['run_directory'])/'result.json'
    d=read(path);raw=d.get('raw_proposal',{});norm=d.get('proposal',{})
    fields=['flow_rate_mL_min','concentration_M','reactor_volume_mL','residence_time_min','residence_time_inlet_min','residence_time_in_channel_min','residence_time_basis']
    diff={k:{'raw':raw.get(k),'normalized':norm.get(k)} for k in fields if k in raw and raw.get(k)!=norm.get(k)}
    if diff:normalization_changes.append({'model':row['model'],'case':row['case'],'repeat':row['repeat_id'],'path':str(path.relative_to(ROOT)),'changed_explicit_fields':diff})

context_logs=[read_value for name in ['judge_context_checks_0.jsonl','judge_context_checks_1.jsonl'] for read_value in [json.loads(line) for line in (CAMPAIGN/name).read_text().splitlines()]]
explicit_score_contradictions=[]
for rating in all_ratings:
    explicit=re.findall(r'\bScore\s+([0-4])\b',rating['rationale'],re.I)
    if explicit and int(explicit[-1])!=rating['score']:
        explicit_score_contradictions.append(rating)

audit={
    'scope':'Read-only forensic audit; no new judgments, no adjusted benchmark scores, no source/archive changes.',
    'historical_sources':old_sources,
    'current_rubric_sha256':sha(CAMPAIGN/'frozen/outcome_rubric.json'),
    'identical_case_context_across_arms':{case:len({json.dumps(packets[r['candidate_id']]['case_context'],sort_keys=True) for r in key if r['case']==case})==1 for case in {r['case'] for r in key}},
    'fresh_outcomes':fresh_rows,
    'applicable_rating_count':len(all_ratings),
    'rating_counts':dict(Counter(str(r['score']) for r in all_ratings)),
    'fresh_critical_flag_count':sum(bool(r['critical_error']) for r in all_ratings),
    'qwen_complete_context_check_count':len(context_logs),
    'qwen_all_full_prompts_fit_without_truncation':all(r['full_prompt_fits'] and not r['truncated'] for r in context_logs),
    'qwen_input_tokens_range':[min(r['input_tokens'] for r in context_logs),max(r['input_tokens'] for r in context_logs)],
    'explicit_score_rationale_contradictions':explicit_score_contradictions,
    'historical_normalization_changed_explicit_fields':normalization_changes,
}
(OUT/'machine_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')

findings=[
    ('H2-P-pressure-boundary','N2-CFB2CBF4',['UO-05','UO-12'],'Delivered diagram places vented separator before only BPR; its gas vent bypasses the pressure boundary.',
     'PBR → vented gas–liquid separator → liquid BPR; separate safe H2 vent leaves separator directly.',
     'A necessary pressurized boundary is not supplied as drawn. The original UO-05 critical condition covers a topology unable to execute the intended process.'),
    ('H2-P-startup-arithmetic','N2-CFB2CBF4',['UO-07','UO-13'],'Startup and collection time assertions contradict delivered volumes and flow.',
     '3–5 × 3 mL / 0.4 mL/min = 22.5–37.5 min, not 7–12 min. 250 mL / 0.4 = 625 min, not approximately 10 min.',
     'These discrepancies affect execution; a score of 4 requires no material defect. Distinguish exact fill time from the valid separate 60 min collection plan.'),
    ('PHOTO-P-gas-units','N2-2AC84057',['UO-08'],'The same output equates 0.60 mL/min with 10 sccm and 5 sccm with 0.30 mL/min at NTP.',
     'At one reference state 1 sccm = 1 standard mL/min. Both written conversions differ by 16.67-fold.',
     'Original UO-08 explicitly tests standard gas flow/reference state and molar bookkeeping. These are contradictory delivered claims, not missing JSON fields.'),
    ('PHOTO-G-molar-flow','N2-D5F033BB',['UO-03','UO-06','UO-08','UO-13'],'Substrate molar flow is understated and falsely supports oxygen excess and the collected amount.',
     '0.10 M × 0.20 mL/min = 0.020 mmol/min = 20 µmol/min, not 1.2. Air 0.55 sccm × 0.21 / 22.4 mL/mmol = 5.156 µmol/min O2; O2/substrate = 0.258, not 4.3. A 25 min collection receives 500 µmol, not 30.',
     'A material stoichiometric defect conflicts with score 4. Qwen explicitly concludes Score 2 in UO-06 rationale but serializes score 4; this alone differs by 2/52 = 0.03846 in the case score.'),
    ('H2-FP-duplicate-dose','N2-A55DB624',['UO-03','UO-06','UO-13'],'Final preparation charges source-identical DMAOL twice.',
     '1.26 mmol + 1.26 mmol in 10 mL = 0.252 M; canonical design and calculations specify 0.126 M.',
     'This is a genuine FlowPilot defect missed by both judges. It was fixed only after the frozen benchmark and must not be excused in this audit.'),
    ('Cu-G-flush-count','N2-857B124A',['UO-13'],'Shutdown flush volume is misreported.',
     '10 min × 0.10 mL/min / 0.30 mL = 3.33 reactor volumes, not at least 10.',
     'Original UO-13 evaluates reproducible shutdown instructions; distinguish the executed flush duration from an unsupported volume assertion.'),
    ('Cu-G-internal-phase-claim','N2-857B124A',['UO-11'],'Its own estimated boiling point at 20 bar is below the proposed temperature while it concludes liquid retention.',
     'The same table states a 130–135 °C boiling point at 20 bar and claims that 150 °C at 20 bar maintains liquid.',
     'This finding is internal inconsistency only; it does not endorse the estimated boiling point or add externally researched physical-property data.'),
    ('Cu-P-throughput','N2-8F6A5220',['UO-06'],'Predicted product rate uses substrate feed rate without the claimed 94% conversion/yield factor.',
     '0.25 M × 0.10 mL/min × 60 = 1.50 mmol/h substrate feed; at 94% the corresponding product rate is 1.41 mmol/h.',
     'This is a smaller interpretive arithmetic error and should not be conflated with a critical failure.'),
]
records=[]
for fid,cid,criteria,description,check,anchor in findings:
    observed={}
    for judge in ['openai','qwen']:
        path,response=judgments[(cid,judge)]
        observed[judge]={'accepted_response_path':str(path.relative_to(ROOT)),
            'criterion_rows':[r for r in response['criterion_scores'] if r['criterion_id'] in criteria]}
    records.append({'finding_id':fid,'candidate_id':cid,'architecture':key_by_id[cid]['architecture'],
        'packet_path':str((CAMPAIGN/'packets'/(cid+'.json')).relative_to(ROOT)),
        'description':description,'independent_check':check,'relation_to_original_anchor':anchor,'accepted_judgments':observed})
(OUT/'checked_findings.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'fresh_outcomes':len(fresh_rows),'applicable_ratings':len(all_ratings),'rating_counts':audit['rating_counts'],
    'explicit_score_contradictions':len(explicit_score_contradictions),'historical_normalization_changes':len(normalization_changes),
    'output':str(OUT.relative_to(ROOT))}))
