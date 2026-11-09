import importlib.util
from pathlib import Path


def load_preparer():
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_kbs_external_study.py'
    spec=importlib.util.spec_from_file_location('external_preparer',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text_payload_reaches_prompt_without_second_string_escaping():
    module=load_preparer()
    left={'observed_record':'First observation.\nSecond observation.'}
    right={'observed_record':'Another observation.\nLast observation.'}
    prompt=module.prompt_for('Prefer reliable retrieval.',left,right)
    assert 'First observation.\nSecond observation.' in prompt
    assert 'First observation.\\nSecond observation.' not in prompt
    assert 'POLICY A EVIDENCE:\n'+left['observed_record'] in prompt
    assert 'POLICY B EVIDENCE:\n'+right['observed_record'] in prompt
    assert 'EVIDENCE ARM: BehavioralEvidence' in prompt
    assert 'gaps do not imply that no action or event occurred' in prompt
