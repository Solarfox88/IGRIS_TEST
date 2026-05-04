# Installing IGRIS_GPT on Ubuntu

These instructions assume a fresh Ubuntu 22.04 or later system with internet access.  Commands should be run as a user with sudo privileges.

1. **Clone the repository**

   ```bash
   git clone https://github.com/Solarfox88/IGRIS_GPT.git
   cd IGRIS_GPT
   ```

2. **Run the installation script**

   The `install_ubuntu.sh` script installs system packages, sets up a virtual environment and installs Python dependencies.

   ```bash
   bash scripts/install_ubuntu.sh
   ```

3. **Configure environment**

   Copy the example environment file and edit it to provide API keys or override defaults:

   ```bash
   cp .env.example .env
   # Edit .env to set OPENAI_API_KEY or VASTAI_API_KEY if using remote models
   ```

4. **Install and configure Ollama**

   To use the local default model you need an Ollama daemon running.  The provided `setup_ollama.sh` script will install Ollama, start the service and pull the `phi4-mini` model.

   ```bash
   bash scripts/setup_ollama.sh
   ```

5. **Start the server**

   Use `start_igris.sh` to launch the FastAPI server via Uvicorn.  By default it listens on port `7778` on all interfaces.

   ```bash
   bash scripts/start_igris.sh
   ```

   Once running, point your browser to `http://<server-ip>:7778` to access the web interface.

6. **Running tests**

   With the virtual environment activated, run the test suite to verify the installation:

   ```bash
   python -m pytest -q
   ```

7. **Service management**

   The `scripts` directory also contains helper scripts to stop, restart and check the status of the server (`stop_igris.sh`, `restart_igris.sh`, `status_igris.sh`).

---

For more details about the architecture and roadmap see the other documents in the `docs/` folder.