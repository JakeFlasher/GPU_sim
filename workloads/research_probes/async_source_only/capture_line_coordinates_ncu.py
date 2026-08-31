#!/usr/bin/env python3
"""Capture NCU evidence for the coordinate-corrected ownership generation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from line_ownership_campaign_contract import (
    COORDINATE_CONTRACT,
    configure_once,
)

# Configuration precedes producer import so no predecessor default can be
# observed, cached, or used to construct a path or command.
configure_once(COORDINATE_CONTRACT)

import capture_line_ownership_ncu as producer  # noqa: E402


producer.configure_contract(COORDINATE_CONTRACT)


def main(argv: Sequence[str] | None = None) -> int:
    return producer.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
