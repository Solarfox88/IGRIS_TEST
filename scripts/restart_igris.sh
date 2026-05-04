#!/usr/bin/env bash
# restart_igris.sh — Restart the IGRIS_GPT server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/stop_igris.sh"
sleep 1
bash "$SCRIPT_DIR/start_igris.sh"
