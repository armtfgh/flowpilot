"""Exercise standalone production-facing tests against isolated candidate only."""
from pathlib import Path
from types import ModuleType
import sys
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))

def load(filename,name):
    path=HERE/filename
    module=ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
    return module

class IsolatedTestImports:
    def pytest_collection_modifyitems(self,items):
        main=load('candidate_main.py','isolated_standalone_main')
        allocator=load('candidate_inventory_allocator.py','isolated_standalone_allocator')
        for item in items:
            if Path(item.module.__file__).name=='test_feed_solvent_preservation.py':
                item.module.main=main;item.module.allocator=allocator

if __name__=='__main__':
    raise SystemExit(pytest.main(['-q',str(HERE/'test_feed_solvent_preservation.py')],plugins=[IsolatedTestImports()]))
