"""Unapplied boundary proposal. No model calls or production monkey-patching."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from flora_translate.schemas import FlowProposal

@dataclass(frozen=True)
class ExtractedProposal:
    payload: dict[str, Any]
    envelope: dict[str, Any]


def extract_proposal_payload(data: dict[str, Any]) -> ExtractedProposal:
    """Separate one explicit wrapper without guessing, repairing or defaulting data.

    The caller must retain and process envelope disposition separately. This
    function does not convert a model's SCREEN/BLOCK decision into execution.
    """
    if not isinstance(data, dict):
        raise ValueError('Flow proposal response must be a JSON object.')
    envelope = {}
    payload = data
    if 'proposal' in data:
        if not isinstance(data['proposal'], dict):
            raise ValueError('Nested proposal must be an object; no implicit JSON decoding.')
        duplicate = set(data).intersection(FlowProposal.model_fields)
        if duplicate:
            raise ValueError('Ambiguous flat and nested proposal fields: '+', '.join(sorted(duplicate)))
        payload = data['proposal']
        envelope = {k:deepcopy(v) for k,v in data.items() if k!='proposal'}
    if not payload or not any(payload.get(key) not in (None, '', [], {}, 0, 0.0)
                              for key in ('flow_rate_mL_min', 'streams', 'stage_parameters')):
        raise ValueError('No explicit flow-design payload; do not synthesize a design from defaults.')
    return ExtractedProposal(deepcopy(payload), envelope)
