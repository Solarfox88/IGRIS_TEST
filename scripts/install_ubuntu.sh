#!/usr/bin/env bash
set -euo pipefail

echo "[IGRIS] Updating apt cache and installing dependencies..."
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip git curl

echo "[IGRIS] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[IGRIS] Installing Python dependencies..."
pip install --upgrade pip
pip install -e . || pip install fastapi uvicorn jinja2 pydantic python-dotenv aiofiles

echo "[IGRIS] Creating runtime directories..."
mkdir -p logs workspace project

echo "[IGRIS] Installation complete."