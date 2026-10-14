from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3]
stage=Path(__file__).resolve().parent/'staging'
sys.path.insert(0,str(root))
sys.path.insert(0,str(stage))
import pytest
raise SystemExit(pytest.main([
    '-q', str(stage/'flora_translate/tests/test_source_component_aliases.py'),
    str(root/'flora_translate/tests/test_component_identity.py'),
    str(root/'flora_translate/tests/test_chemistry_contract.py'),
    str(root/'flora_translate/tests/test_protocol_identity.py'),
    str(root/'flora_translate/tests/test_executable_artifact_contract.py'),
    str(root/'flora_translate/tests/test_executable_artifact_normalization.py'),
    str(root/'flora_translate/tests/test_final_design_contract.py'),
    str(root/'flora_translate/tests/test_final_design_validator.py'),
]))
