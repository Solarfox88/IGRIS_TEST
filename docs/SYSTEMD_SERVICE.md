# Systemd Service Setup

Run IGRIS_GPT as a systemd service for production deployments.

## Prerequisites

- Ubuntu 20.04+ or similar
- IGRIS_GPT installed in `/opt/igris-gpt` (or adjust paths)
- A dedicated `igris` user (recommended)

## Setup

```bash
# Create user
sudo useradd -r -s /bin/bash -m -d /opt/igris-gpt igris

# Clone and install
sudo -u igris git clone https://github.com/Solarfox88/IGRIS_GPT.git /opt/igris-gpt
cd /opt/igris-gpt
sudo -u igris bash scripts/install_ubuntu.sh
sudo -u igris cp .env.example .env
# Edit .env with your settings

# Install service
sudo cp config/igris-gpt.service.example /etc/systemd/system/igris-gpt.service
sudo systemctl daemon-reload
sudo systemctl enable igris-gpt
sudo systemctl start igris-gpt

# Check status
sudo systemctl status igris-gpt
journalctl -u igris-gpt -f
```

## Commands

```bash
sudo systemctl start igris-gpt
sudo systemctl stop igris-gpt
sudo systemctl restart igris-gpt
sudo systemctl status igris-gpt
journalctl -u igris-gpt --since "1 hour ago"
```

## Updating

```bash
cd /opt/igris-gpt
sudo -u igris git pull origin main
sudo -u igris .venv/bin/pip install -e ".[dev]"
sudo systemctl restart igris-gpt
```
