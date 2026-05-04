#!/usr/bin/env bash
# setup_ollama.sh — Install and configure Ollama for IGRIS_GPT
# Idempotent: safe to run multiple times.
set -euo pipefail

MODEL="${1:-phi4-mini}"

echo "=== Ollama Setup for IGRIS_GPT ==="

# Check if Ollama is installed
if command -v ollama &>/dev/null; then
  echo ">> Ollama is already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
  echo ">> Ollama not found. Installing..."
  if curl -fsSL https://ollama.com/install.sh | sh; then
    echo ">> Ollama installed successfully."
  else
    echo ""
    echo "!! Ollama installation failed."
    echo "!! You can install manually from: https://ollama.com/download"
    echo "!! IGRIS_GPT will use deterministic fallback responses without Ollama."
    echo ""
    exit 0
  fi
fi

# Check if Ollama service is running
echo ">> Checking Ollama service..."
if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo ">> Ollama is running."
else
  echo ">> Starting Ollama service..."
  if command -v systemctl &>/dev/null && systemctl is-active ollama &>/dev/null 2>&1; then
    echo ">> Ollama systemd service is active."
  else
    echo ">> Starting ollama serve in background..."
    nohup ollama serve >/dev/null 2>&1 &
    sleep 3
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      echo ">> Ollama service started."
    else
      echo ""
      echo "!! Could not start Ollama service."
      echo "!! Try running: ollama serve"
      echo "!! IGRIS_GPT will use deterministic fallback without Ollama."
      exit 0
    fi
  fi
fi

# Pull model
echo ">> Pulling model: $MODEL"
if ollama pull "$MODEL" 2>/dev/null; then
  echo ">> Model $MODEL ready."
else
  echo ""
  echo "!! Model $MODEL could not be pulled."
  echo "!! Available models:"
  ollama list 2>/dev/null || echo "  (none)"
  echo ""
  echo "!! You can try a different model:"
  echo "!!   ollama pull llama3.2"
  echo "!! Then set LOCAL_LLM_MODEL=llama3.2 in .env"
  echo ""
  echo "!! IGRIS_GPT will use deterministic fallback without a working model."
  exit 0
fi

# Verify
echo ""
echo ">> Verifying Ollama..."
TAGS=$(curl -sf http://127.0.0.1:11434/api/tags 2>/dev/null || echo '{}')
echo ">> Available models: $TAGS"
echo ""
echo "=== Ollama setup complete ==="
echo "Model: $MODEL"
echo "Base URL: http://127.0.0.1:11434"
