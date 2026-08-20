from types import SimpleNamespace

import httpx

from flora_translate.batch_normalization import enrich_batch_record_dict
from flora_translate.engine import llm_agents
from flora_translate.schemas import FlowProposal
from flora_translate.translation_llm import TranslationLLM


def test_structured_additive_is_normalized_to_chemical_name():
    normalized = enrich_batch_record_dict(
        {
            "reaction_description": "Hydrogenolysis",
            "additives": [
                {
                    "name": "acetic acid",
                    "role": "promoter",
                }
            ],
        },
        "Hydrogenolysis with acetic acid.",
    )

    assert normalized["additives"] == ["acetic acid"]


def test_null_infeasible_proposal_values_reach_schema_as_safe_defaults():
    normalized = TranslationLLM._normalize_proposal_data(
        {
            "residence_time_min": None,
            "flow_rate_mL_min": None,
            "reactor_volume_mL": None,
            "temperature_C": None,
            "streams": [{"phase": "gas", "molar_equiv": None}],
            "chemistry_notes": "BLOCK: required wavelength is unavailable.",
            "reasoning_per_field": {
                "analogy_comparison": {
                    "Analogy 1": "Similar mixing regime.",
                }
            },
        }
    )
    proposal = FlowProposal(**normalized)

    assert proposal.residence_time_min == 0
    assert proposal.flow_rate_mL_min == 0
    assert proposal.reactor_volume_mL == 0
    assert proposal.temperature_C == 25
    assert proposal.streams[0].molar_equiv == 1
    assert proposal.chemistry_notes.startswith("BLOCK:")
    assert proposal.reasoning_per_field["analogy_comparison"] == (
        '{"Analogy 1": "Similar mixing regime."}'
    )


def test_verified_http_client_uses_system_trust_store(monkeypatch):
    calls = []
    context = SimpleNamespace(verify_flags=0)

    def fake_create_default_context(*args, **kwargs):
        calls.append((args, kwargs))
        return context

    monkeypatch.setattr(llm_agents.ssl, "create_default_context", fake_create_default_context)
    monkeypatch.setattr(httpx, "Client", lambda *, verify: ("client", verify))

    client = llm_agents.build_verified_httpx_client()

    assert calls == [((), {})]
    assert client == ("client", context)
