"""Execute the already frozen finite campaign; no result-dependent branching."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'ablation_results/practical_input/claude_fair_bridge_20261014'
DRIVER = ROOT / 'ablation_test/scripts/run_fair_practical_benchmark.py'
READY = threading.Event()
generation_errors = []
log_lock = threading.Lock()

def event(data):
    with log_lock:
        with (OUT / 'worker_events.jsonl').open('a') as handle:
            handle.write(json.dumps({'unix_time': time.time(), **data}) + '\n')
    print(json.dumps(data), flush=True)

def command(label, *args):
    cmd = [sys.executable, '-u', str(DRIVER), '--output', str(OUT), *args]
    event({'event': 'start', 'label': label, 'command': cmd})
    with (OUT / f'worker_{label}.log').open('a') as handle:
        code = subprocess.call(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    event({'event': 'finish', 'label': label, 'returncode': code})
    return code

def generate():
    try:
        for index in range(6):
            code = command(f'generation_{index}', '--phase', 'generate', '--item', str(index))
            if code:
                generation_errors.append({'index': index, 'returncode': code})
        # Every generation process has exited before any artifacts are frozen.
        return command('prepare_B', '--phase', 'prepare', '--stage', 'B')
    finally:
        READY.set()

def judge(name):
    first = command(f'judge_A_{name}', '--phase', 'judge', '--stage', 'A', '--item', name)
    READY.wait()
    if not (OUT / 'prepared_manifest_B.json').exists():
        return {'A': first, 'B': 'not_prepared'}
    second = command(f'judge_B_{name}', '--phase', 'judge', '--stage', 'B', '--item', name)
    return {'A': first, 'B': second}

if __name__ == '__main__':
    if not (OUT / 'prepared_manifest_A.json').exists():
        raise RuntimeError('Initialize and freeze the campaign first')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        work = {'generation': pool.submit(generate)}
        work.update({name: pool.submit(judge, name) for name in ('qwen', 'openai', 'claude')})
        outcomes = {name: job.result() for name, job in work.items()}
    report_code = command('report_all', '--phase', 'report', '--stage', 'all')
    (OUT / 'worker_completion.json').write_text(json.dumps({
        'workers': outcomes, 'generation_process_errors': generation_errors,
        'report_returncode': report_code}, indent=2) + '\n')
    event({'event': 'complete', 'workers': outcomes, 'report_returncode': report_code})
