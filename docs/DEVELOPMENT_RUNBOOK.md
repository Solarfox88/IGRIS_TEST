# Development Runbook

This runbook outlines common tasks for developing and maintaining IGRIS_GPT.

## Setting up the environment

Run the installation script and activate the virtual environment:

```bash
bash scripts/install_ubuntu.sh
source .venv/bin/activate
```

Install the package in editable mode to enable local changes:

```bash
pip install -e .
```

## Running the server

Start the FastAPI server with:

```bash
source .venv/bin/activate
python -m igris.web.server
```

or use the provided script:

```bash
bash scripts/start_igris.sh
```

## Running tests

All tests live in the `tests/` directory.  Use `pytest` to run them:

```bash
python -m pytest -q
```

For asynchronous endpoints the project uses `pytest-asyncio` fixtures.  Many unit tests rely on a mock LLM to avoid network calls.

## Contributing

* Follow the commit conventions described in the project README (e.g. `feat:`, `chore:`, `ui:`).  Small, focused commits make review easier.
* Always run the test suite before pushing changes.
* Do not commit secrets or runtime artifacts.
* Update documentation in `docs/` when adding new features.