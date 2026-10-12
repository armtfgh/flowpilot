import sys,json
from pathlib import Path
ROOT=Path('/home/amirreza/Documents/codes/Flow Agent')
sys.path.insert(0,str(ROOT))
from ablation_test.scripts.run_astra_matched_benchmark import configure,preflight
out=ROOT/'tmp/astra_readiness_20261012'
configure()
try:
 preflight(out)
except Exception as exc:
 message=str(exc).lower()
 result={'status':'failed','error_type':type(exc).__name__,'billing_exhausted':any(s in message for s in ['no credits remaining','insufficient_quota','exceeded your current quota']),'http_status':getattr(exc,'status_code',None)}
 (out/'preflight_failure.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
 raise SystemExit(1)
