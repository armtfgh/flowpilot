"""Run production-facing tests against an unapplied candidate in this subprocess."""
from pathlib import Path
from types import ModuleType
import hashlib
import sys
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'ablation_test/scripts/run_fair_practical_benchmark.py'
BEFORE=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
from ablation_test.scripts import run_fair_practical_benchmark as original
candidate=ModuleType('isolated_fair_routing_driver');candidate.__file__=str(SOURCE)
exec(compile((HERE/'candidate_run_fair_practical_benchmark.py').read_text(),str(SOURCE),'exec'),candidate.__dict__)

class UseCandidate:
    def pytest_collection_modifyitems(self,items):
        for item in items:
            if Path(item.module.__file__).name in ('test_fair_objective_routing.py','test_fair_practical_benchmark.py'):
                item.module.fair=candidate

if __name__=='__main__':
    status=pytest.main(['-q',str(HERE/'test_fair_objective_routing.py'),str(ROOT/'ablation_test/tests/test_fair_practical_benchmark.py')],plugins=[UseCandidate()])
    candidate.configure()
    reloaded=ModuleType('isolated_fair_routing_reload');reloaded.__file__=str(SOURCE)
    exec(compile((HERE/'candidate_run_fair_practical_benchmark.py').read_text(),str(SOURCE),'exec'),reloaded.__dict__)
    assert reloaded.NORMAL_PRODUCTION is candidate.NORMAL_PRODUCTION
    assert reloaded.NORMAL_PRODUCTION is not candidate._production_with_scoped_output
    reloaded.configure();reloaded.configure()
    assert reloaded.runner._production_full is reloaded._production_with_scoped_output
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==BEFORE
    print('Reload/unwrapping proof passed; production source unchanged.')
    raise SystemExit(status)
