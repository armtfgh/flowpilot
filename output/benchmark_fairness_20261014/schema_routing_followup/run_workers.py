"""Run the fixed routing follow-up; no score-dependent scheduling."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / 'campaign'
DRIVER = HERE / 'run_followup.py'

def run(label, *args):
    command = [sys.executable, '-u', str(DRIVER), *args]
    print(json.dumps({'event':'start','label':label,'at':time.time()}), flush=True)
    with (OUT / f'worker_{label}.log').open('a') as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({'event':'finish','label':label,'code':code,'at':time.time()}), flush=True)
    return code

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {str(i):pool.submit(run, f'generation_{i}', '--phase','generate','--item',str(i)) for i in range(3)}
        generated = {key:future.result() for key,future in futures.items()}
    prepared = run('prepare','--phase','prepare')
    judged = {}
    if prepared == 0:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {name:pool.submit(run,f'judge_{name}','--phase','judge','--item',name) for name in ('qwen','openai','claude')}
            judged = {key:future.result() for key,future in futures.items()}
    reported = run('report','--phase','report') if prepared == 0 else None
    result={'generation':generated,'prepare':prepared,'judges':judged,'report':reported}
    (OUT/'worker_completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
