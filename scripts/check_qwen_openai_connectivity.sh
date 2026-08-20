#!/usr/bin/env bash

set -u

QWEN_BASE_URL="${QWEN_BASE_URL:-http://10.13.24.169:8000/v1}"
QWEN_MODEL="${QWEN_MODEL:-/models/Qwen3.6-27B}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.4-2026-03-05}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-20}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

qwen_ok=0
openai_ok=0

request() {
  local name="$1"
  local output="$2"
  shift 2

  local code
  code="$(curl --silent --show-error \
    --connect-timeout 5 \
    --max-time "$TIMEOUT_SECONDS" \
    --output "$output" \
    --write-out '%{http_code}' \
    "$@")"
  local curl_status=$?

  if [[ $curl_status -ne 0 ]]; then
    printf 'FAIL  %s: curl exit code %s\n' "$name" "$curl_status"
    return 1
  fi

  if [[ ! "$code" =~ ^2 ]]; then
    printf 'FAIL  %s: HTTP %s\n' "$name" "$code"
    sed -n '1,20p' "$output"
    return 1
  fi

  printf 'PASS  %s: HTTP %s\n' "$name" "$code"
  return 0
}

printf 'Connectivity check started: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
printf 'Qwen endpoint: %s\n' "$QWEN_BASE_URL"
printf 'Qwen model:    %s\n' "$QWEN_MODEL"
printf 'OpenAI endpoint: %s\n' "$OPENAI_BASE_URL"
printf 'OpenAI model:    %s\n\n' "$OPENAI_MODEL"

printf '%s\n' '--- Qwen model-list check ---'
if request "Qwen model list" "$workdir/qwen_models.json" \
  "$QWEN_BASE_URL/models"; then
  if grep -Fq "$QWEN_MODEL" "$workdir/qwen_models.json"; then
    printf 'PASS  Qwen configured model is advertised\n'
  else
    printf 'WARN  Qwen endpoint responded, but %s was not found in /models\n' "$QWEN_MODEL"
  fi

  printf '%s\n' '--- Qwen inference check ---'
  if request "Qwen chat completion" "$workdir/qwen_chat.json" \
    --header 'Content-Type: application/json' \
    --data "{\"model\":\"$QWEN_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly CONNECTED\"}],\"max_tokens\":8,\"temperature\":0}" \
    "$QWEN_BASE_URL/chat/completions"; then
    qwen_ok=1
    printf 'Qwen response preview: '
    tr '\n' ' ' < "$workdir/qwen_chat.json" | cut -c1-300
    printf '\n'
  fi
fi

printf '\n%s\n' '--- OpenAI credential check ---'
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  printf 'FAIL  OPENAI_API_KEY is not set in this shell\n'
  printf '%s\n' "Set it with: export OPENAI_API_KEY='your-key'"
else
  printf 'PASS  OPENAI_API_KEY is set (value hidden)\n'

  printf '%s\n' '--- OpenAI model-list check ---'
  if request "OpenAI model list" "$workdir/openai_models.json" \
    --header "Authorization: Bearer $OPENAI_API_KEY" \
    "$OPENAI_BASE_URL/models"; then
    if grep -Fq "$OPENAI_MODEL" "$workdir/openai_models.json"; then
      printf 'PASS  OpenAI configured model is advertised\n'
    else
      printf 'WARN  OpenAI endpoint responded, but %s was not found in /models\n' "$OPENAI_MODEL"
    fi

    printf '%s\n' '--- OpenAI inference check ---'
    if request "OpenAI Responses API" "$workdir/openai_response.json" \
      --header "Authorization: Bearer $OPENAI_API_KEY" \
      --header 'Content-Type: application/json' \
      --data "{\"model\":\"$OPENAI_MODEL\",\"input\":\"Reply with exactly CONNECTED\",\"max_output_tokens\":16}" \
      "$OPENAI_BASE_URL/responses"; then
      openai_ok=1
      printf 'OpenAI response preview: '
      tr '\n' ' ' < "$workdir/openai_response.json" | cut -c1-300
      printf '\n'
    fi
  fi
fi

printf '\n%s\n' '--- Final result ---'
if [[ $qwen_ok -eq 1 && $openai_ok -eq 1 ]]; then
  printf 'PASS  Qwen and OpenAI are both reachable and generated a response.\n'
  exit 0
fi

[[ $qwen_ok -eq 1 ]] && printf 'PASS  Qwen\n' || printf 'FAIL  Qwen\n'
[[ $openai_ok -eq 1 ]] && printf 'PASS  OpenAI\n' || printf 'FAIL  OpenAI\n'
exit 1
