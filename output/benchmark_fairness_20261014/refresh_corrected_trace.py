"""Read saved generation events; no models, no judgments, no source mutations."""
from pathlib import Path
import json,copy,hashlib,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flora_translate.translation_llm import TranslationLLM,_parse_json
from flora_translate.schemas import FlowProposal
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CAM=OUT/'schema_routing_followup/campaign'
fields=['flow_rate_mL_min','temperature_C','concentration_M','BPR_bar','reactor_volume_mL','wavelength_nm','light_setup']
rows=[]
for path in sorted((CAM/'generation').glob('*/claude_flowpilot/repeat_01/llm_events.jsonl')):
    for index,line in enumerate(path.read_text().splitlines()):
        e=json.loads(line)
        if not str(e.get('api_name','')).startswith('translation_llm'):continue
        row={'case':path.parents[2].name,'source_path':str(path.relative_to(ROOT)),'event_index':index,'api_name':e['api_name'],'timestamp':e['timestamp'],'response_sha256':hashlib.sha256(e.get('response_text','').encode()).hexdigest(),'envelope_directive_count':e.get('user_prompt','').count('Required final output: one JSON object with recommended_disposition')}
        try:
            raw=_parse_json(e['response_text']);p=FlowProposal(**TranslationLLM._normalize_proposal_data(copy.deepcopy(raw)))
            source=raw.get('proposal',raw)
            row.update(raw_top_level_keys=list(raw),has_nested_proposal=isinstance(raw.get('proposal'),dict),raw_selected_fields={k:source.get(k) for k in fields},production_parser_fields={k:p.model_dump().get(k) for k in fields},raw_stream_count=len(source.get('streams') or []),parsed_stream_count=len(p.streams))
            row['selected_fields_preserved']=row['raw_selected_fields']==row['production_parser_fields']
            time_fields=['residence_time_min','residence_time_inlet_min','residence_time_in_channel_min']
            row['existing_numeric_normalizer_time_changes']={k:{'raw':source.get(k),'normalized':p.model_dump().get(k)} for k in time_fields if source.get(k)!=p.model_dump().get(k)}
        except Exception as exc:row['parse_error']=str(exc)
        result=path.parent/'result.json'
        if result.exists():
            d=json.loads(result.read_text());f=d.get('final_design') or {};row['delivery_status']=f.get('status');row['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest();row['final_parameters']=f.get('parameters');row['final_instrument_ids']=[x.get('equipment_id') for x in f.get('instrument_manifest') or []]
        rows.append(row)
(OUT/'corrected_translation_trace.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{'case':r['case'],'event':r['api_name'],'nested':r.get('has_nested_proposal'),'raw_preserved':r.get('selected_fields_preserved'),'envelope_directive_count':r['envelope_directive_count'],'delivery':r.get('delivery_status'),'parse_error':r.get('parse_error')} for r in rows],indent=2))
