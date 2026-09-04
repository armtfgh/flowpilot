"""Verify the /no_think + think:false fix for gemma4 thinking model."""

import requests, json

BASE_URL = "http://10.13.24.45:11434"
MODEL    = "gemma4-flora"

SYSTEM = (
    "You are Dr. Kinetics on the FLORA ENGINE council for flow chemistry.\n"
    "Your domain triage status: GREEN\n\n"
    "RULES:\n"
    "1. Use only pre-computed triage values — do not re-derive anything.\n"
    "2. If domain is GREEN, output APPROVED with one-sentence summary.\n\n"
    "Output JSON ONLY:\n"
    '{"verdict":"APPROVED","proposals":[],"conditions":[],"summary":"one sentence"}'
)
USER = json.dumps({
    "triage": {"domain": "KINETICS", "status": "GREEN",
               "tau_min": 50.0, "IF": 6.0, "tau_lit": 100.0},
    "proposal": {"residence_time_min": 50.0, "flow_rate_mL_min": 0.25, "BPR_bar": 5.0}
})


def raw_call(messages, label, max_tokens=800, think=None):
    payload = {"model": MODEL, "messages": messages,
                "max_tokens": max_tokens, "stream": False}
    if think is not None:
        payload["think"] = think
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=120)
    body   = r.json()
    choice = body.get("choices", [{}])[0]
    msg    = choice.get("message", {})
    content   = msg.get("content", "")
    reasoning = msg.get("reasoning", "")
    print(f"\n{'='*60}")
    print(f"TEST  : {label}")
    print(f"  finish_reason : {choice.get('finish_reason')}")
    print(f"  usage         : {body.get('usage')}")
    print(f"  content       : {repr(content[:200])}")
    if reasoning:
        print(f"  reasoning     : {repr(reasoning[:100])}...")


# WITHOUT fix (baseline — should still be empty)
raw_call([
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": USER},
], label="1 - baseline (no fix) — expect empty content")

# WITH think=False API flag only
raw_call([
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": USER},
], label="2 - think=False API flag", think=False)

# WITH /no_think prefix only
raw_call([
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": f"/no_think\n{USER}"},
], label="3 - /no_think prefix in user message")

# WITH both (what FLORA now does)
raw_call([
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": f"/no_think\n{USER}"},
], label="4 - both think=False + /no_think (FLORA fix)", think=False)
