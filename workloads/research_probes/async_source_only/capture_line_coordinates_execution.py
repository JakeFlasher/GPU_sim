#!/usr/bin/env python3
"""Capture the fresh coordinate-corrected source-line ownership generation."""

from __future__ import annotations

from typing import Sequence

from line_ownership_campaign_contract import (
    COORDINATE_CONTRACT,
    configure_once,
)


# Configuration happens before importing the producer, so all import-time
# defaults and its shared static helper see one closed generation.
configure_once(COORDINATE_CONTRACT)

import capture_line_ownership_execution as producer  # noqa: E402


producer.configure_campaign(COORDINATE_CONTRACT)


def main(argv: Sequence[str] | None = None) -> int:
    return producer.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
