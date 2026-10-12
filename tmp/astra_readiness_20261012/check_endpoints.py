import sys,json,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/home/amirreza/Documents/codes/Flow Agent')
sys.path.insert(0,str(ROOT))
from ablation_test.scripts.run_astra_matched_benchmark import configure,campaign
from ablation_test.scripts.run_newgen_2_0_benchmark import JUDGES
from ablation_test.src.providers import endpoint_health
from flora_translate.engine import llm_agents
configure()
bundles={'astra':campaign.MODELS['astra'],**{'judge_'+k:v for k,v in JUDGES.items()}}
def check(item):
 name,b=item
 result={'role':name,'model':b['model']}
 try:
  if b['provider']=='ollama':
   h=endpoint_health(b)
   result.update(reachable=h.get('reachable',False),model_available=h.get('model_advertised',False))
   if not result['reachable']:result['error']='Local model endpoint is unreachable'
   elif not result['model_available']:result['advertised_models']=h.get('advertised_models',[])
  else:
   key='OPENAI_API_KEY' if b['provider']=='openai' else 'ANTHROPIC_API_KEY'
   if not os.getenv(key):
    result.update(reachable=False,model_available=False,error=key+' is not configured')
    return result
   import openai,anthropic
   cls=openai.OpenAI if b['provider']=='openai' else anthropic.Anthropic
   with cls(http_client=llm_agents.build_verified_httpx_client(),timeout=15,max_retries=0) as c:
    data=c.models.list(limit=100).data if b['provider']=='anthropic' else c.models.list().data
    result.update(reachable=True,model_available=b['model'] in {m.id for m in data})
 except Exception as e:
  # Do not expose credentials or raw error bodies in the console or audit.
  result.update(reachable=False,model_available=False,error_type=type(e).__name__,http_status=getattr(e,'status_code',None))
 return result
with ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(check,bundles.items()))
report={'environment':os.getenv('CONDA_DEFAULT_ENV'),'checks':rows,'all_available':all(r.get('reachable') and r.get('model_available') for r in rows)}
(ROOT/'tmp/astra_readiness_20261012/endpoints.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
