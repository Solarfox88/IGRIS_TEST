"""
Entry point for the IGRIS command line interface.

The CLI currently exposes a single command to run the web server.  Over time
additional subcommands (e.g. for running tests or managing sessions) can be
added here using `argparse` or a more advanced library like `typer`.
"""

import argparse
import asyncio
from typing import Optional

from igris.web.server import create_app, run_app


def main(argv: Optional[list[str]] = None) -> None:
    """Parse command line arguments and execute the chosen action."""
    parser = argparse.ArgumentParser(prog="igris")
    sub = parser.add_subparsers(dest="command", required=False)
    run_server = sub.add_parser("serve", help="Start the web server")
    run_server.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind (default: 0.0.0.0)",
    )
    run_server.add_argument(
        "--port",
        type=int,
        default=7778,
        help="Port to bind (default: 7778)",
    )
    args = parser.parse_args(argv)

    if args.command == "serve" or args.command is None:
        app = create_app()
        run_app(app=app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()