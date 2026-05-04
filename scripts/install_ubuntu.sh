#!/usr/bin/env bash
# install_ubuntu.sh — Install IGRIS_GPT on Ubuntu
# Idempotent: safe to run multiple times.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== IGRIS_GPT Ubuntu Installer ==="
echo "Repository: $REPO_DIR"

# System packages
echo ">> Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git curl >/dev/null

# Python virtual environment
VENV="$REPO_DIR/.venv"
if [ ! -d "$VENV" ]; then
  echo ">> Creating virtual environment at $VENV..."
  python3 -m venv "$VENV"
else
  echo ">> Virtual environment already exists."
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Upgrade pip
echo ">> Upgrading pip..."
python -m pip install -U pip -q

# Install IGRIS_GPT
echo ">> Installing IGRIS_GPT..."
python -m pip install -e "$REPO_DIR[dev]" -q

# Create runtime directories
echo ">> Creating runtime directories..."
mkdir -p "$REPO_DIR/logs"
mkdir -p "$REPO_DIR/.igris/tasks"
mkdir -p "$REPO_DIR/.igris/reports"
mkdir -p "$REPO_DIR/.igris/timeline"
mkdir -p "$REPO_DIR/.igris/memory"

# Copy .env.example if .env doesn't exist
if [ ! -f "$REPO_DIR/.env" ]; then
  echo ">> Copying .env.example to .env — edit it with your settings."
  cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
else
  echo ">> .env already exists, keeping it."
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (optional)"
echo "  2. bash scripts/setup_ollama.sh   (optional, for local LLM)"
echo "  3. bash scripts/start_igris.sh    (start server)"
echo "  4. Open http://localhost:7778"
echo ""
echo "Run tests with:"
echo "  source .venv/bin/activate && python -m pytest -q"
