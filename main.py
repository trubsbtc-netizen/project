from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from core.config import load_config
from core.runtime.bot import run_bot


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Poly Institutional Bot")
    parser.add_argument("--no-tui", action="store_true", help="disable terminal UI")
    parser.add_argument("--log-file", help="log to file instead of stdout")
    args, _ = parser.parse_known_args()

    log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    if args.log_file:
        handler: logging.Handler = RotatingFileHandler(
            args.log_file, maxBytes=100 * 1024 * 1024, backupCount=3,
        )
    elif args.no_tui:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.WARNING)

    handler.setFormatter(logging.Formatter(log_fmt))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(log_level)

    config = load_config(".env")
    asyncio.run(run_bot(config, enable_tui=not args.no_tui))


if __name__ == "__main__":
    main()
