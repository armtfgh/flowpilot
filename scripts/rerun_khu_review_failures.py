"""One fresh retry after a diagnosed buffer-identity correction; never loop for scores."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_khu_six_revised import check


def main():
    root=ROOT/'outputs/khu_revised_six_20260914'
    for path in sorted(root.glob('figure*/attempt_01/result.json')):
        report=check(path)
        if 'no_invented_buffer_molarity' not in report['failed'] or set(report['failed'])-{'no_invented_buffer_molarity','diagram_liquid_feed_concentrations'}:
            print(path.parent.parent.name,'no automatic retry:',report['failed'],flush=True)
            continue
        reason={'source_attempt':str(path.parent),'diagnosis':'Unnamed pH buffer medium was assigned an invented substrate-level molarity.',
                'correction':'Recognize the explicitly declared solvent-mixture member; preserve pH without inventing buffer molarity.',
                'fresh_model_run':True,'inventory_hardware_unchanged':True}
        (path.parent.parent/'retry_reason.json').write_text(json.dumps(reason,indent=2))
        subprocess.run([sys.executable,str(ROOT/'scripts/run_khu_six_revised.py'),'--output',str(root),
                        '--case',path.parent.parent.name,'--attempt','attempt_02'],check=True,cwd=ROOT)
        result=path.parent.parent/'attempt_02/result.json'
        print('RECHECK',path.parent.parent.name,check(result)['failed'] if result.exists() else 'run failed',flush=True)


if __name__=='__main__': main()
