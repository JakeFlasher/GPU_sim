#!/usr/bin/env python3
"""Fail-closed owner boundary for the not-yet-authorized M5 collector."""

from __future__ import annotations

# Direct execution refuses before argument parsing, repository discovery, optional
# imports, hardware access, or filesystem mutation.
if __name__ == "__main__":
    import sys as _bootstrap_sys

    _bootstrap_sys.stderr.write("M5_COLLECT_DISABLED\n")
    raise SystemExit(2)


M5_COLLECT_DISABLED = "M5_COLLECT_DISABLED"


class M5CollectorDisabled(RuntimeError):
    """M5 collection was requested before its external gates were satisfied."""


def collect(*_args: object, **_kwargs: object) -> None:
    """Refuse collection without reading inputs or creating outputs."""

    raise M5CollectorDisabled(M5_COLLECT_DISABLED)
