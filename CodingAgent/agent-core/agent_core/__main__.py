"""Main entry point for running agent-core as a module (`python -m agent_core`)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import signal
import sys

from .config import load_server_config
from .server import AgentCoreServer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="agent-core",
        description="Personal Desktop Coding Agent - Core Python Sidecar",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the WebSocket server (or set AGENT_CORE_PORT env var)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to models.yaml configuration file (defaults to config/models.yaml)",
    )
    return parser.parse_args()


async def async_main() -> None:
    """Async main routine."""
    args = parse_args()

    try:
        config = load_server_config(port=args.port, models_path=args.config)
    except Exception as exc:
        print(f"Error loading configuration: {exc}", file=sys.stderr)
        sys.exit(1)

    server = AgentCoreServer(config)

    # Setup signal handlers for graceful shutdown where supported
    loop = asyncio.get_running_loop()

    def handle_signal():
        server.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except (NotImplementedError, AttributeError):
            # Windows asyncio loop may not support add_signal_handler for all signals
            pass

    try:
        await server.run()
    except asyncio.CancelledError:
        server.stop()
    except KeyboardInterrupt:
        server.stop()


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
