"""Read-only experiment planning. No network/model call; writes only beside this file."""
import ast
from datetime import datetime, timezone
import hashlib
from types import ModuleType
import json
import os
from pathlib import Path
import sys
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT))
import flora_translate.config

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
OLD=ROOT/'ablation_results/manuscript_benchmark/manuscript_three_model_three_repeat_postfix_20260820'
NEW=ROOT/'ablation_results/practical_input/claude_practical_repair_20261014'
oldsource=OLD/'frozen/source_code/newgen_2_outcome.py'
legacy=ModuleType('frozen_historical_outcome')
exec(compile(oldsource.read_text(),str(oldsource),'exec'),legacy.__dict__)
rubric=read(OLD/'frozen/outcome_rubric.json')
assert sha(OLD/'frozen/outcome_rubric.json')==sha(NEW/'frozen/outcome_rubric.json')
oldkey=read(OLD/'frozen/candidate_key_confidential.json')['candidates']
newkey=read(NEW/'frozen/candidate_key_confidential.json')['candidates']
selected=[('historical_manuscript',OLD,r) for r in oldkey if r['generator_model']=='Claude Sonnet 4.6' and r['repeat_id']=='repeat_01']
selected += [('current_practical',NEW,r) for r in newkey if r['cohort']=='fresh']
assert len(selected)==15
try:
 import tiktoken
 enc=tiktoken.get_encoding('o200k_base')
 def ntokens(s):return len(enc.encode(s))
 method='offline o200k_base token estimate, exact tokenizer for GPT-4o text; proxy only for Claude'
except ImportError:
 def ntokens(s):return (len(s)+2)//3
 method='ceil(character_count/3) planning estimate, not provider count'
rows=[]
for cohort,folder,r in selected:
 path=folder/'packets'/f"{r['candidate_id']}.json";packet=read(path)
 system,user=legacy.judge_prompt(rubric,packet)
 tok=ntokens(system)+ntokens(user)
 rows.append({'cohort':cohort,'candidate_id':r['candidate_id'],'case':r['case'],'architecture':r['architecture'],
  'repeat_id':r['repeat_id'],'result_path':str(Path(r['run_directory'])/'result.json'),'result_sha256':sha(Path(r['run_directory'])/'result.json'),
  'packet_path':str(path.relative_to(ROOT)),'packet_sha256':sha(path),'historical_prompt_input_tokens_estimate':tok,
  'prompt_bytes':len((system+user).encode()),'historical_system_sha256':hashlib.sha256(system.encode()).hexdigest()})
write('saved_outcome_selection.json',{'rule':'All six Claude Sonnet 4.6 repeat_01 original-manuscript outcomes, then all nine fresh outcomes of the latest completed practical campaign, including guided one-shot. No score-dependent selection.', 'rows':rows})
avail={'checked_at':datetime.now(timezone.utc).isoformat(),'method':'Credential-presence booleans and completed archived calls only; no endpoint or model API requested.',
 'credentials_present':{k:bool(os.getenv(k)) for k in ['ANTHROPIC_API_KEY','OPENAI_API_KEY','OPENROUTER_API_KEY']},'models':[]}
for judge in ('qwen','openai'):
 paths=list((NEW/'judgments'/judge).glob('*/raw_response.json'))
 latest=max(paths,key=lambda p:p.stat().st_mtime_ns);raw=read(latest)
 avail['models'].append({'role':'judge','provider':raw['provider'],'requested_configuration':{'qwen':'/models/Qwen3.6-27B','openai':'gpt-4o'}[judge],
  'returned_model':raw['model'],'latest_evidence':str(latest.relative_to(ROOT)),'evidence_sha256':sha(latest),'evidence_mtime_utc':datetime.fromtimestamp(latest.stat().st_mtime,timezone.utc).isoformat()})
ledger=[json.loads(s) for s in (NEW/'paid_usage.jsonl').read_text().splitlines() if s]
claude=[x for x in ledger if x.get('event')=='completed' and x['provider']=='anthropic'][-1]
avail['models'].append({'role':'generator_recently_used_and_historical_judge_configured','provider':'anthropic','requested_model':claude['requested_model'],'returned_model':claude['returned_model'],'completed_at':claude['at'],'evidence':str((NEW/'paid_usage.jsonl').relative_to(ROOT))})
write('provider_availability.json',avail)
# Rates are those already recorded in the completed local usage guard, not newly verified prices.
rates={'gpt54_input_per_million':2.5,'gpt54_cached_input_per_million':.25,'gpt54_output_per_million':15,'sonnet_input_per_million':3,'sonnet_output_per_million':15}
Ain=sum(r['historical_prompt_input_tokens_estimate'] for r in rows)
Aexpected=Ain*5.5/1e6 + len(rows)*2500*30/1e6
Aoutputceil=Ain*5.5/1e6 + len(rows)*7500*30/1e6
# Fresh packet input planning average uses every current practical packet (all 3 arms).
Bin=sum(r['historical_prompt_input_tokens_estimate'] for r in rows if r['cohort']=='current_practical')/9*6
Bjudges=Bin*5.5/1e6 + 6*2500*30/1e6
write('budget.json',{'currency':'USD','token_estimation':method,'rates_verified_by_root_for_actual_historical_judges':rates,
 'rate_source_code':'ablation_test/scripts/run_fair_practical_benchmark.py:usage_cost','rate_source_urls':['https://platform.claude.com/docs/en/models/sonnet-4-6/overview','https://developers.openai.com/api/docs/models/gpt-5.4'],
 'rates_not_refetched_this_read_only_task':True,'paid_limit':10,
 'stage_A_saved_outputs':15,'stage_A_judgments':45,'stage_A_total_input_per_paid_judge_estimate':Ain,
 'stage_A_paid_estimate_2500_output_tokens':Aexpected,'stage_A_paid_if_every_output_hits_7500':Aoutputceil,
 'stage_B_fresh_outcomes':6,'stage_B_judgments':18,'stage_B_judging_estimate':Bjudges,
 'stage_B_generation_allowance':3.0,'retry_allowance':1.5,
 'planning_total':Aexpected+Bjudges+3+1.5,
 'policy':'Freeze both stages before scores. Run saved-output rescoring first. Stop before any request whose conservative input/output reservation would exceed $10; report all completed and missing cells. No score-dependent generation/retry/extension. Local inference has unestimated hardware cost. Returned-usage guard is not a provider billing guarantee.'})
files=[OLD/'frozen/outcome_rubric.json',oldsource,OLD/'frozen/source_code/run_newgen_2_0_benchmark.py',OLD/'frozen/source_code/run_manuscript_five_case_benchmark.py',OLD/'frozen/source_code/runner.py',NEW/'frozen/outcome_rubric.json',NEW/'frozen/judge_addendum.txt',ROOT/'ablation_test/src/newgen_2_outcome.py',ROOT/'ablation_test/scripts/run_practical_repair_benchmark.py',ROOT/'ablation_test/scripts/run_practical_input_pilot.py',ROOT/'flora_translate/main.py']
write('source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
# Store exact old instructions for review; the user packet is inserted only at execution time.
empty={};system,user=legacy.judge_prompt(rubric,empty)
write('historical_judge_prompt_example.json',{'system':system,'user':user,'placeholder_packet':{},'source_sha256':sha(oldsource)})
runner=(OLD/'frozen/source_code/runner.py').read_text(); tree=ast.parse(runner)
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ('_flow_schema_instruction','_general_one_shot'):
  (OUT/f'historical{node.name}.txt').write_text(ast.get_source_segment(runner,node)+'\n')
write('criterion_summary.json',{'rubric_sha256':sha(OLD/'frozen/outcome_rubric.json'),'weights':'unchanged equal judge and applicable criterion weights','judges':['/models/Qwen3.6-27B','gpt-5.4-2026-03-05','claude-sonnet-4-6'],'score':'mean_judge(mean_applicable_criterion(score)/4), then equal mean across three cases','criteria':[{'id':r['criterion_id'],'name':r['name'],'applicability':r['applicability']} for r in rubric['criteria']]})
print(json.dumps({'selected':len(rows),'stage_A_cost':Aexpected,'planned_total':Aexpected+Bjudges+3+1.5,'provider_check':'configured/recently evidenced only; no APIs'},indent=2))
