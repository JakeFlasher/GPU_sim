"""SimBench-BW row-to-arch resolver (single source of truth).

`run_suite.py` uses this for artifact routing and validators use it for
cross-checks, so the record side and the verify side cannot drift apart.

Design rule: the row's declared `arch_gate` is authoritative. The current device's CC
family narrows the row's declared list to the archs that physically apply
on this GPU; the row's declared order wins when multiple are applicable.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parent


# Each CC tuple maps to the set of archs a row may declare for that device
# family. sm_86 lands on CC 8.6 hardware (Ampere consumer; smoke-test only —
# no sm_120 / sm_120a / sm_120f / sm_90a workloads can run on it). sm_90 /
# sm_90a both land on CC 9.0 hardware; sm_120 / sm_120a / sm_120f all land on
# CC 12.0 hardware (the `a` and `f` variants are architecture-conditional PTX
# targets on the same chip, not distinct SKUs).
CC_TO_ARCH_FAMILY: dict[tuple[int, int], frozenset[str]] = {
    (8, 6): frozenset({"sm_86"}),
    (9, 0): frozenset({"sm_90", "sm_90a"}),
    (12, 0): frozenset({"sm_120", "sm_120a", "sm_120f"}),
}


# Canonical platform to CC-family mapping for active run/evidence paths.
PLATFORM_TO_CC_FAMILY: dict[str, tuple[int, int]] = {
    "h100": (9, 0),
    "rtx_5090_blackwell": (12, 0),
    "rtx_5090": (12, 0),
    "rtx_3090_ti": (8, 6),
}


def resolve_build_arch_for_row(row: dict[str, Any], cc: tuple[int, int]) -> str | None:
    """Return the row's declared build arch that applies on the given CC.

    Returns `None` when the row declares no `arch_gate`, or when no declared
    arch belongs to the CC family. Callers decide the fallback policy; for
    example, `run_suite.py` falls back to a CC-derived tag for unknown-CC
    hosts."""
    gates = [g for g in row.get("arch_gate", []) if isinstance(g, str)]
    family = CC_TO_ARCH_FAMILY.get(cc)
    if not family or not gates:
        return None
    for g in gates:
        if g in family:
            return g
    return None


def resolve_build_arch_for_platform(row: dict[str, Any], platform: str) -> str | None:
    """Convenience: resolve by canonical platform name."""
    cc = PLATFORM_TO_CC_FAMILY.get(platform)
    if cc is None:
        return None
    return resolve_build_arch_for_row(row, cc)


def load_roster_row(slug: str, roster_path: Path | None = None) -> dict[str, Any] | None:
    """Return the roster row for `slug`, or None if the slug is absent.

    Useful for tests and validators that need to cross-check a caller-supplied
    platform against the row's declared `arch_gate`."""
    roster_path = roster_path or (SUITE_ROOT / "roster.toml")
    with open(roster_path, "rb") as f:
        data = tomllib.load(f)
    for r in data.get("rows", []):
        if r.get("slug") == slug:
            return r
    return None
