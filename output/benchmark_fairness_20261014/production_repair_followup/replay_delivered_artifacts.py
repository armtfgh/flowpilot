"""Offline production replay of every saved corrected design; no model calls or grades."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from flora_translate.final_design_contract import build_final_design_contract
SOURCE = ROOT/'output/benchmark_fairness_20261014/schema_routing_followup/campaign/generation'
OUT = Path(__file__).resolve().parent/'offline_replay'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def main():
    assert not OUT.exists(), 'Never overwrite an offline replay'
    rows=[]
    for case in ('hydrogenolysis','photochemical_oxidation','cuaac'):
        source=SOURCE/case/'claude_flowpilot/repeat_01/result.json'
        digest=sha(source); original=json.loads(source.read_text()); saved=deepcopy(original)
        after=build_final_design_contract(deepcopy(original)); before=original['final_design']
        assert original==saved
        assert sha(source)==digest
        assert after['parameters']==before['parameters'], 'Physical settings must not change in artifact-only replay'
        assert after['streams']==before['streams'], 'Delivered feed settings changed'
        assert after['status']==before['status']=='executable'
        negative=deepcopy(original)
        negative['final_validation']['checks']['pump_flow_feasible']=False
        blocked=build_final_design_contract(negative)
        assert blocked['status']=='blocked', 'Publication fixes must not bypass release gates'
        write(OUT/case/'final_design.json',after)
        rows.append({'case':case,'source':str(source),'source_sha256':digest,'before_status':before['status'],
            'after_status':after['status'],'parameters_and_streams_unchanged':True,
            'failed_pump_negative_control_blocks':True,
            'before_procedure':before['operating_procedure'],'after_procedure':after['operating_procedure'],
            'before_rationale_keys':list(before.get('decision_rationale',{})),
            'after_rationale_keys':list(after.get('decision_rationale',{}))})
    write(OUT/'summary.json',{'at':datetime.now(timezone.utc).isoformat(),'model_calls':0,'scores_assigned':False,'cases':rows})
    print(json.dumps({'replayed':len(rows),'physical_settings_unchanged':True,'negative_controls_passed':len(rows),'output':str(OUT)},indent=2))
if __name__=='__main__':main()
