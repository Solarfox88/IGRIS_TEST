#!/usr/bin/env bash
set -euo pipefail

if command -v ollama >/dev/null 2>&1; then
  echo "[IGRIS] Ollama already installed."
else
  echo "[IGRIS] Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "[IGRIS] Starting Ollama service..."
ollama serve >/dev/null 2>&1 &
sleep 2

echo "[IGRIS] Pulling phi4-mini model..."
ollama pull phi4-mini || true

echo "[IGRIS] Verifying model tags..."
curl -s http://127.0.0.1:11434/api/tags || true