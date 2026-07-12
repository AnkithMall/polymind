from __future__ import annotations

import logging
import sys

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)-35s | %(message)s"
)

_configured: bool = False


def setup_logging(level: int = logging.DEBUG) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True
