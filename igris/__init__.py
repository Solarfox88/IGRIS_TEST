"""
Top level package for the IGRIS_GPT project.

This package exposes a factory to create the FastAPI application and imports
other submodules to make them discoverable when the package is installed.

Usage:

    from igris.web.server import create_app
    app = create_app()

"""

from .web.server import create_app  # noqa: F401