"""Fresh council reviews of two corrected candidates using unchanged generation inputs."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_khu_six_revised import check


def main():
    root=ROOT/'outputs/khu_revised_six_20260914'
    for case in ['figure5_set1','figure5_set2']:
        previous=root/case/'attempt_02'
        result=json.loads((previous/'result.json').read_text())
        codes={x['code'] for x in result['final_design']['consistency']['issues']}
        assert codes=={'FINAL-GAS-COMPOSITION-INCONSISTENT'},codes
        number=3
        while (root/case/f'attempt_{number:02d}').exists():number+=1
        attempt=f'attempt_{number:02d}'
        subprocess.run([sys.executable,str(ROOT/'scripts/run_khu_six_revised.py'),'--output',str(root),
                        '--case',case,'--attempt',attempt,'--reuse-generation-from',str(previous)],check=True,cwd=ROOT)
        result_path=root/case/attempt/'result.json'
        report=check(result_path) if result_path.exists() else {'all_passed':False,'failed':['No result produced']}
        print('FINAL_RECHECK',case,report['failed'],flush=True)
        if not report['all_passed']:raise RuntimeError(report['failed'])


if __name__=='__main__':main()
