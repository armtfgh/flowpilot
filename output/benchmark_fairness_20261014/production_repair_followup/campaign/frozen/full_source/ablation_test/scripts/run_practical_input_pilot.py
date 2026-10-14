#!/usr/bin/env python3
"""Frozen practical-input pilot; all inference and delivered responses are archived."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import fcntl
import uuid
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ablation_test.scripts import run_manuscript_three_model_repeated_benchmark as campaign
from ablation_test.scripts import run_newgen_2_0_benchmark as judging
from ablation_test.scripts.run_manuscript_five_case_benchmark import load_frozen_cases
from ablation_test.scripts.evaluate_astra_delivered_outputs import _local_context_tokens, exact_local_context
from ablation_test.scripts.build_newgen_2_0_report import resolved_response
from ablation_test.src import runner
from ablation_test.src.cases import AblationCase
from ablation_test.src.newgen_2_outcome import build_outcome_packet, packet_gate
from flora_translate.engine import llm_agents as llm

DEFAULT = ROOT / 'ablation_results/practical_input/claude_practical_pilot_20261014'
OBJECTIVE = ('Suggest one practical first flow experiment for this batch reaction using only the available equipment, '
             'with operating conditions, setup, procedure, and any assumptions or missing information.')
SYSTEM = 'You are a chemistry assistant helping a researcher design a continuous-flow experiment.'
PRICING_URL = 'https://platform.claude.com/docs/en/models/sonnet-4-6/overview'
OPENAI_PRICING_URL = 'https://developers.openai.com/api/docs/models/gpt-4o'
SPEND_LIMIT = 10.0
GENERATOR = 'claude'
PRIMARY_JUDGE = 'openai'

def read(p): return json.loads(Path(p).read_text())
def write(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

@dataclass(frozen=True)
class PracticalCase(AblationCase):
    @property
    def design_input_text(self):
        return ('BATCH PROCEDURE\n' + self.protocol.strip() + '\n\nREQUEST\n' + self.objective +
                '\n\nAVAILABLE EQUIPMENT\n' + json.dumps(self.inventory, ensure_ascii=False, sort_keys=True, separators=(',', ':')))

def practical_cases():
    by_label = dict(load_frozen_cases())
    cases = []
    for label in campaign.SELECTED_CASES:
        original = by_label[label]
        protocol = original.protocol.split('Translate this', 1)[0].strip()
        protocol = protocol.replace(
            'For molarity conversion, explicitly use and label an assumed solution density of 0.792 g/mL, giving approximately 0.126 M DMAOL.',
            'The solution density is assumed to be 0.792 g/mL (approximately 0.126 M DMAOL).')
        fields = asdict(original)
        fields.update(protocol=protocol, objective=OBJECTIVE, hard_constraints=('Use only the supplied equipment inventory.',),
                      chemistry_identity_confirmation={}, suite_id='practical_input_pilot_v1')
        cases.append((label, PracticalCase(**fields)))
    return cases

def practical_one_shot(case, model, provider, recorder, temperature, seed):
    user = case.design_input_text
    recorder.write_json('prompt.json', {'system': SYSTEM, 'user': user, 'format': 'ordinary prose; no required JSON schema'})
    llm.set_llm_observer(recorder.observe_llm)
    llm.set_llm_runtime_overrides(temperature=temperature, seed=seed, capture_content=True)
    try:
        response = llm.call_model_text(model=model, provider=provider, system=SYSTEM, user_content=user,
                                      max_tokens=8192, api_name='practical_one_shot')
        recorder.write_json('raw_response.json', asdict(response))
        if not response.text.strip(): raise RuntimeError('Empty one-shot response')
        # The complete delivered prose is evidence. Do not infer or repair numbers.
        return {'variant': 'general_one_shot', 'proposal': {'chemistry_notes': response.text},
                'raw_proposal': {'chemistry_notes': response.text}, 'schema_valid': True,
                'output_format': 'prose', 'output_validity': 'prose_delivered',
                'raw_delivered_text': response.text, 'finish_reason': response.finish_reason}
    finally:
        llm.clear_llm_observer(); llm.clear_llm_runtime_overrides()

FORMAT_POLICY = ('Assess information anywhere in the delivered outcome, including complete prose in '
    'chemistry_and_operations.chemistry_notes. Empty structured fields do not mean a fact is absent if it is '
    'stated in that prose. Neither prose nor JSON formatting earns or loses credit. Judge chemical and '
    'operational content and recalculate reported arithmetic. Do not impose instructions removed from the '
    'practical brief. Do not infer missing content from the equipment inventory.')

def practical_packet(**kwargs):
    # Apply the same absence of automated numeric extraction to both arms. The
    # fixed rubric still requires judges to assess numerical correctness.
    kwargs['deterministic_verification'] = {'scope': 'No automated numerical pass/fail is supplied in this pilot.',
                                           'format_policy': FORMAT_POLICY}
    packet = build_outcome_packet(**kwargs)
    packet['deterministic_verification_sheet'] = kwargs['deterministic_verification']
    packet['evaluation_boundary'] += ' ' + FORMAT_POLICY
    return packet

def configure():
    campaign.MODELS = {'claude': {'provider': 'anthropic', 'model': 'claude-sonnet-4-6',
        'upstream_mode': 'never', 'family': 'anthropic', 'display': 'Claude Sonnet 4.6'}}
    campaign.REPEAT_IDS = ('repeat_01',)
    campaign.JUDGES = ('openai', 'qwen')
    campaign.TEMPERATURE = .2
    campaign.CANDIDATE_BUDGET = 2
    campaign.SEED_NAMESPACE = 'practical-input-claude-20261014-v1'
    campaign.MATCHED_SEED_ACROSS_ARCHITECTURES = False
    campaign.selected_cases = practical_cases
    campaign.build_outcome_packet = practical_packet
    campaign.deterministic_sheet = lambda result, metrics, case: {}
    runner._general_one_shot = practical_one_shot

def source_files():
    for directory in ('flora_translate', 'benchmark', 'ablation_test/src', 'ablation_test/scripts', 'ablation_test/tests'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.txt', '.json', '.yaml', '.yml') and '__pycache__' not in path.parts:
                yield path

def verify_source(output):
    expected = read(output / 'frozen/full_source_manifest.json')
    assert {str(p.relative_to(ROOT)): sha(p) for p in source_files()} == expected, 'Source changed after freeze'
    for relative, digest in read(output / 'frozen/input_manifest.json').items():
        assert sha(output / relative) == digest, relative

def initialize(output):
    if output.exists(): raise RuntimeError('Use a new output directory; existing campaigns are immutable')
    cases = practical_cases()
    campaign.initialize(output, cases)
    frozen = output / 'frozen'
    hashes = {}
    for source in source_files():
        relative = source.relative_to(ROOT); target = frozen / 'full_source' / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        hashes[str(relative)] = sha(source)
    write(frozen / 'full_source_manifest.json', hashes)
    token_rows, edits = [], []
    originals = dict(load_frozen_cases())
    for label, case in cases:
        folder = frozen / 'practical_inputs' / campaign.safe_name(label)
        write(folder / 'public_input.json', case.public_payload())
        (folder / 'request.txt').write_text(case.design_input_text)
        previous = originals[label]
        engineered_fields = runner._flow_schema_instruction().split('proposal_json must be a JSON-encoded string whose decoded object uses these fields:', 1)[1]
        engineered_user = (f'FROZEN DESIGN INPUT:\n{previous.design_input_text}\n\n'
            'Propose one complete flow design with numeric operating conditions, '
            'stream assignments, safety controls, and a brief rationale. Return one '
            'JSON object and no markdown with exactly these top-level fields: '
            'recommended_disposition, disposition_rationale, and proposal. The proposal '
            'value must be a JSON object, not a JSON-encoded string.\n\n'
            'The proposal object uses these fields:\n' + engineered_fields)
        engineered_system = ('You are a general chemistry assistant. Translate a batch chemistry '
                             'protocol into a practical continuous-flow experiment.')
        def tokens(system, text):
            return llm._get_anthropic_client().messages.count_tokens(model='claude-sonnet-4-6',
                system=system, messages=[{'role':'user','content':text}]).input_tokens
        token_rows.append({'case': label, 'practical_total_input_tokens': tokens(SYSTEM, case.design_input_text),
                          'engineered_total_input_tokens': tokens(engineered_system, engineered_user),
                          'practical_request_words': len(OBJECTIVE.split()),
                          'practical_protocol_words': len(case.protocol.split()),
                          'inventory_json_characters': len(json.dumps(case.inventory, sort_keys=True, separators=(',', ':'))),
                          'count_method': 'Anthropic messages.count_tokens; claude-sonnet-4-6; system and user messages',
                          'shared_inventory_identical': previous.inventory == case.inventory})
        edits.append({'case': label, 'original_protocol': previous.protocol, 'practical_protocol': case.protocol,
                      'removed_constraints': list(previous.hard_constraints),
                      'retained_constraint': list(case.hard_constraints),
                      'removed_chemistry_identity_confirmation': previous.chemistry_identity_confirmation,
                      'reason': 'Remove flow-design instructions and detailed reporting checklist; preserve batch chemistry, declared density assumption and the full inventory.'})
    write(frozen / 'input_token_comparison.json', token_rows)
    write(frozen / 'input_changes.json', edits)
    manifest = read(frozen / 'campaign_manifest.json')
    manifest.update(benchmark_name='Practical input: one-shot versus FlowPilot', created_at=now(),
        primary_evaluator=PRIMARY_JUDGE, secondary_evaluator='qwen',
        primary_endpoint='Equal-case mean of GPT-4o criterion scores, divided by four',
        secondary_endpoint='Local Qwen judge reported separately as a sensitivity check',
        independent_call_policy='One fresh generation per case and architecture; all outcomes retained.',
        retry_policy='No campaign-level generation retries. Provider transport retries and native pipeline repairs remain recorded. At most three attempts per judge only for invalid formatting or transport failure.',
        development_cases=True, pilot=True, compute_matched=False, interactive_clarification=False,
        configuration_scope='Legacy two-initial-candidate production path with source-excluded lexical retrieval; not the twelve-candidate GUI configuration.',
        input_policy='Same concise factual protocol, ordinary request and complete inventory in both arms; no specialist checklist or confirmed mechanism label.',
        output_policy='One-shot may answer in prose; full prose retained with no numerical repair. Both outcomes are judged for content rather than serialization.',
        evaluation_adapter='practical_prose_v1; same 14-criterion rubric; neutral format instructions; no automatic numeric worksheets for either arm',
        spending_stop_threshold_usd=SPEND_LIMIT,
        spending_guard='Reserve an upper estimate based on UTF-8 request bytes plus maximum output before each paid request, with cross-process locking; settle from returned usage. Not a provider billing hard cap.',
        pricing={'checked_date':'2026-10-14',
            'anthropic':{'source':PRICING_URL,'input_per_million':3,'output_per_million':15,'cache_read_per_million':.3,'cache_write_5m_per_million':3.75,'cache_write_1h_per_million':6},
            'openai':{'source':OPENAI_PRICING_URL,'input_per_million':2.5,'cached_input_per_million':1.25,'output_per_million':10},
            'local_hardware_cost':'not estimated'},
        stopping_rule='Exactly six generation attempts and twelve accepted judge assessments; do not extend or repeat based on score. Stop on infrastructure or spending failure.')
    write(frozen / 'campaign_manifest.json', manifest)
    (frozen / 'PREREGISTRATION.md').write_text('''# Practical-input development pilot

Three previously studied chemistries, two architectures, one generation per cell. Both use Claude Sonnet 4.6 (claude-sonnet-4-6, direct Anthropic API) at temperature 0.2. Claude does not accept a seed; recorded seeds are bookkeeping only. The batch facts, declared density assumption and complete inventory are retained. A short ordinary request replaces the hand-written flow-engineering checklist. One-shot answers in ordinary prose with an 8192-token output allowance. FlowPilot uses its existing two-candidate production pipeline, calculations, source-excluded lexical retrieval and review. No researcher clarification is simulated and no new laboratory information is invented.

GPT-4o is the prespecified primary judge. Qwen3.6-27B is a secondary cross-family sensitivity check. The unchanged 14-criterion rubric uses integer 0-4 scores and fixed applicability. Average applicable criteria within a judgment and divide by four, then average equally across cases. Report critical flags separately. Do not average the two judges for the primary endpoint. Keep both full input/output forms; formatting alone earns no credit. Remove automatic numeric diagnostic sheets from both arms; judges assess arithmetic in the delivered content. Their architecture and generator labels are withheld, but formatting may be recognizable.

Exactly six generation attempts. No low-score retry, rejection or code repair during the campaign. A failed output remains a failure. Provider transport retries and native workflow repairs remain archived. Up to three judge attempts only for format/transport failure. Preserve every attempt, source version, source exclusion, token count, timing and cost. Before each paid API call, reserve its UTF-8 request-byte input estimate and maximum output cost against a $10 shared usage guard, then settle using returned usage. This is not a provider billing-system hard cap. Claude Sonnet 4.6 generates both arms; GPT-4o evaluates them. Local Qwen is a secondary judge only.

This is an initial practical-input pilot, not the full two-by-two input-preparation experiment. The engineered prompt is counted but not regenerated. These development cases do not provide independent confirmation. One generation cannot estimate repeatability, and input tokens do not measure researcher preparation time. A null or reversed advantage is a valid result.
''')
    paths = [p for p in frozen.rglob('*') if p.is_file() and 'full_source' not in p.parts and p.name != 'input_manifest.json']
    write(frozen / 'input_manifest.json', {str(p.relative_to(output)): sha(p) for p in paths})
    write(output / 'source_git_state.json', {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)})
    print(json.dumps({'phase':'initialized','input_tokens':token_rows,'outcomes':6,'judgments':12}), flush=True)

def generate_cell(output, index):
    verify_source(output)
    cases = practical_cases(); cells = campaign.campaign_cells(cases); row = cells[index]
    run = campaign.run_directory(output, row)
    if (run / 'run_summary.json').exists():
        print('Retained existing generation attempt',run,flush=True); return
    case = next(c for _, c in cases if c.case_id == row['case_id'])
    campaign.append_jsonl(output / 'campaign_events.jsonl', {'event':'generation_started','at':now(),'cell':row})
    status = runner.execute_cell(case=case,variant=row['variant'],bundle_name=row['condition_id'],
         bundle=campaign.MODELS[GENERATOR],run_dir=run,candidate_budget=2,temperature=.2,seed=row['seed'])
    campaign.append_jsonl(output / 'campaign_events.jsonl', {'event':'generation_finished','at':now(),'cell':row,'status':status.get('status')})
    print('GENERATION',index,row['case_label'],row['architecture'],status.get('status'),flush=True)
    verify_source(output)

def prepare(output):
    verify_source(output)
    if any((output/'judgments').glob('*/*/status.json')): raise RuntimeError('Packets already judged')
    cases = practical_cases()
    for row in campaign.campaign_cells(cases):
        if not (campaign.run_directory(output,row)/'run_summary.json').exists():raise RuntimeError('Generation attempt missing')
    campaign.build_packets(output,cases)
    audit=[]
    for candidate in read(output/'frozen/candidate_key_confidential.json')['candidates']:
        path=output/'packets'/f"{candidate['candidate_id']}.json"; packet=read(path)
        assert not packet_gate(packet)
        if candidate['architecture']=='One-shot' and candidate['generation_status']=='completed':
            source=read(Path(candidate['run_directory'])/'result.json')['raw_delivered_text']
            from ablation_test.src.newgen_2_outcome import _scrub
            assert packet['delivered_final_outcome']['chemistry_and_operations']['chemistry_notes']==_scrub(source)
        audit.append({'candidate_id':candidate['candidate_id'],'sha256':sha(path),'bytes':path.stat().st_size})
    write(output/'packet_integrity.json',audit)
    print('Prepared six complete, format-neutral packets',flush=True)

def usage_cost(provider, usage):
    if provider == 'anthropic':
        creation = usage.get('cache_creation') or {}
        one_hour = creation.get('ephemeral_1h_input_tokens', 0) or 0
        five_min = max(0, (usage.get('cache_creation_input_tokens', 0) or 0) - one_hour)
        return ((usage.get('input_tokens',0) or 0)*3 + (usage.get('output_tokens',0) or 0)*15 +
                (usage.get('cache_read_input_tokens',0) or 0)*.3 + five_min*3.75 + one_hour*6)/1e6
    if provider == 'openai':
        inputs = usage.get('prompt_tokens', usage.get('input_tokens',0)) or 0
        outputs = usage.get('completion_tokens', usage.get('output_tokens',0)) or 0
        details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
        cached = details.get('cached_tokens',0) or 0
        return ((inputs-cached)*2.5 + cached*1.25 + outputs*10)/1e6
    return 0.

def ledger_rows(output):
    path=output/'paid_usage.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def cost(output):
    rows=ledger_rows(output); settled=[r for r in rows if r['event']=='completed']
    done={r['request_id'] for r in rows if r['event'] in ('completed','failed')}
    pending=[r for r in rows if r['event']=='reserved' and r['request_id'] not in done]
    return {'estimated_paid_usd':sum(r['usd'] for r in settled), 'recorded_paid_calls':len(settled),
            'by_provider':{p:sum(r['usd'] for r in settled if r['provider']==p) for p in ('anthropic','openai')},
            'pending_reserved_usd':sum(r['reserved_usd'] for r in pending),
            'pricing_sources':[PRICING_URL,OPENAI_PRICING_URL],
            'local_hardware_cost':'not estimated',
            'billing_limit':'Returned token usage only; unreported server work after failed requests is unknown.'}

@contextmanager
def ledger_lock(output):
    with (output/'paid_usage.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def guarded_create(output, provider, original, kwargs):
    allowed={'anthropic':'claude-sonnet-4-6','openai':'gpt-4o'}
    if kwargs.get('model') != allowed[provider]: raise RuntimeError('Unexpected paid model')
    # UTF-8 byte length deliberately overestimates normal text token counts.
    input_bytes=len(json.dumps(kwargs,ensure_ascii=False,default=str).encode('utf-8'))+4096
    reservation=(input_bytes*(3 if provider=='anthropic' else 2.5) +
                 kwargs.get('max_tokens',8192)*(15 if provider=='anthropic' else 10))/1e6
    rid=uuid.uuid4().hex
    with ledger_lock(output):
        used=cost(output)
        if used['estimated_paid_usd']+used['pending_reserved_usd']+reservation>SPEND_LIMIT:
            raise RuntimeError('Paid usage reservation would exceed $10 guard')
        campaign.append_jsonl(output/'paid_usage.jsonl',{'event':'reserved','at':now(),'request_id':rid,
            'provider':provider,'model':kwargs['model'],'reserved_usd':reservation})
    try: response=original(**kwargs)
    except Exception as exc:
        with ledger_lock(output):
            campaign.append_jsonl(output/'paid_usage.jsonl',{'event':'failed','at':now(),'request_id':rid,
                'provider':provider,'error_type':type(exc).__name__})
        raise
    usage=response.usage.model_dump()
    with ledger_lock(output):
        campaign.append_jsonl(output/'paid_usage.jsonl',{'event':'completed','at':now(),'request_id':rid,
            'provider':provider,'requested_model':kwargs['model'],'returned_model':response.model,
            'usage':usage,'usd':usage_cost(provider,usage)})
    return response

def install_spending_guard(output):
    # Wrap transport methods without changing prompts or the production algorithm.
    for provider, resource in [('anthropic',llm._get_anthropic_client().messages),
                               ('openai',llm._get_openai_client().chat.completions)]:
        original=resource.create
        resource.create=lambda _provider=provider,_original=original,**kwargs: guarded_create(output,_provider,_original,kwargs)

def judge(output, name):
    verify_source(output)
    if name=='qwen':
        llm._bounded_local_messages=lambda system,messages,max_tokens:exact_local_context(system,messages,max_tokens=max_tokens,audit_path=output/'judge_context_checks.jsonl')
    rubric=read(output/'frozen/outcome_rubric.json')
    for candidate in read(output/'frozen/candidate_key_confidential.json')['candidates']:
        cid=candidate['candidate_id']
        if campaign.valid_judgment(output,name,cid):continue
        for attempt in range(3):
            status=judging.call_judge(output,name,candidate,rubric)
            write(output/'usage_so_far.json',cost(output))
            print('JUDGE',name,candidate['case'],candidate['architecture'],status['status'],flush=True)
            if status['status']=='valid':break
        if status['status']!='valid':raise RuntimeError(f'Incomplete judgment {name}/{cid}')
    verify_source(output)

def report(output):
    verify_source(output)
    rubric=read(output/'frozen/outcome_rubric.json');rows=[]
    for c in read(output/'frozen/candidate_key_confidential.json')['candidates']:
        packet=read(output/'packets'/f"{c['candidate_id']}.json")
        for name in campaign.JUDGES:
            response=resolved_response(output,name,c['candidate_id'],rubric,packet)
            if response is None:raise RuntimeError('A required judgment is missing')
            scores=[r for r in response['criterion_scores'] if r['applicability']=='APPLICABLE']
            rows.append({'case':c['case'],'architecture':c['architecture'],'judge':name,
                         'score':statistics.fmean(r['score'] for r in scores)/4,
                         'critical_flags':sum(bool(r['critical_error']) for r in scores),
                         'candidate_id':c['candidate_id']})
    summary={'completed_at':now(),'scope':'Single-generation development pilot; practical inputs; Claude Sonnet 4.6 generator in both arms',
             'primary_judge':PRIMARY_JUDGE,'outcomes':6,'judgments':12,'judge_results':rows,'usage':cost(output),'by_judge':{}}
    for name in campaign.JUDGES:
        summary['by_judge'][name]={}
        for arch in ('One-shot','FlowPilot'):
            rr=[r for r in rows if r['judge']==name and r['architecture']==arch]
            summary['by_judge'][name][arch]={'mean':statistics.fmean(r['score'] for r in rr),'critical_flags':sum(r['critical_flags'] for r in rr)}
        summary['by_judge'][name]['difference']=summary['by_judge'][name]['FlowPilot']['mean']-summary['by_judge'][name]['One-shot']['mean']
    write(output/'report/summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--phase',choices=['init','generate','prepare','judge','report'],required=True)
    p.add_argument('--item',default='0');args=p.parse_args();configure();output=args.output.resolve()
    if args.phase in ('generate','judge'): install_spending_guard(output)
    if args.phase=='init':initialize(output)
    elif args.phase=='generate':generate_cell(output,int(args.item))
    elif args.phase=='prepare':prepare(output)
    elif args.phase=='judge':judge(output,args.item)
    else:report(output)

if __name__=='__main__':main()
