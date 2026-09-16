#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR"
PORT="${FLOWPILOT_STREAMLIT_PORT:-8504}"

export PATH="/home/amirreza/anaconda3/envs/flent/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/flowpilot-font-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-flowpilot-stable}"

mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$APP_DIR"

exec "$ROOT_DIR/.venv-flowpilot/bin/python" -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.headless true
