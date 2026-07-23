"""AC-6.2 split-leakage scanner for ``calibration_head_v4.json``.

Reads the frozen roster + the v4 head sidecar and asserts that
NONE of the 8 effective-validation cells nor the 7 effective-holdout
cells appears in v4's ``training_cells`` list. The 2 admitted
replacement cells inherit their typed-excluded primary's split role
via the ``EFFECTIVE_SPLIT_FOR_ADMITTED_REPLACEMENT`` map exported
from ``fit_calibration_head_v4_sm120``.

The scanner is a pure function suitable for unit tests; the CLI
wrapper invokes it against the committed v4 head.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO_ROOT / "data" / "local_labels"
_DEFAULT_HEAD_PATH = _DATA_ROOT / "calibration_head_v4.json"
_DEFAULT_ROSTER_PATH = _DATA_ROOT / "sm120_cohort_expansion_roster.toml"


sys.path.insert(0, str(_REPO_ROOT / "tools" / "local_labels"))
from fit_calibration_head_v4_sm120 import (  # noqa: E402
    EFFECTIVE_SPLIT_FOR_ADMITTED_REPLACEMENT,
)


class SplitLeak(NamedTuple):
    """Records one leaked cell: a v4 training-cells entry whose
    EFFECTIVE split is validation or holdout (i.e., NOT
    calibration)."""

    slug: str
    shape_point: str
    effective_split: str


class SplitLeakageScanResult(NamedTuple):
    """Result of the v4 split-leakage scan. ``ok`` is True iff
    ``leaks`` is empty. ``leaks`` is a tuple of ``SplitLeak``
    entries (one per leaked cell); each entry names the slug,
    shape_point, and the effective split that caused the leak."""

    ok: bool
    leaks: tuple[SplitLeak, ...]


def _build_effective_split_index(
    roster_path: Path,
    replacement_map: dict[tuple[str, str], str],
) -> dict[tuple[str, str], str]:
    """Build a ``(slug, shape_point) -> effective_split`` index
    from the frozen roster + replacement map. Mirrors
    ``fit_calibration_head_v4_sm120.resolve_effective_splits``
    but operates over the entire roster (not just admitted cells)
    so the scanner can flag training-cell entries pointing at
    cells that are not admitted but ARE on the roster's
    validation/holdout split."""
    with roster_path.open("rb") as fh:
        data = tomllib.load(fh)
    index: dict[tuple[str, str], str] = {}
    for row in data.get("rows", []) or []:
        key = (str(row.get("slug", "")), str(row.get("shape_point", "")))
        split = str(row.get("split", ""))
        if split in ("calibration", "validation", "holdout"):
            index[key] = split
        elif split == "replacement":
            effective = replacement_map.get(key)
            if effective is not None:
                index[key] = effective
            # Unmapped replacement rows have no effective split;
            # a training-cell entry that points at one falls
            # through (the scanner only flags explicit
            # validation/holdout leaks).
    return index


def scan_v4_for_split_leakage(
    *,
    head_payload: dict,
    roster_path: Path = _DEFAULT_ROSTER_PATH,
    replacement_map: dict[tuple[str, str], str] | None = None,
) -> SplitLeakageScanResult:
    """Run the AC-6.2 split-leakage scan over a v4 head payload.

    Returns ``ok == True`` iff none of the head's ``training_cells``
    entries map to an effective split of ``validation`` or
    ``holdout``. Cells whose effective split is ``calibration``
    are fine. Cells absent from the effective-split index (e.g.,
    a synthetic training-cell entry for a slug not on the
    roster) do NOT count as leaks — the scanner's contract is
    "no validation/holdout cell appears in training", not "every
    training cell must be on the roster"; off-roster cell
    inclusion is a separate provenance concern handled by the
    bundle-set validator (AC-1)."""
    if replacement_map is None:
        replacement_map = EFFECTIVE_SPLIT_FOR_ADMITTED_REPLACEMENT
    eff_index = _build_effective_split_index(
        roster_path=roster_path,
        replacement_map=replacement_map,
    )
    leaks: list[SplitLeak] = []
    for cell in head_payload.get("training_cells", []) or []:
        slug = str(cell.get("slug", ""))
        shape = str(cell.get("shape_point", ""))
        eff = eff_index.get((slug, shape))
        if eff in ("validation", "holdout"):
            leaks.append(SplitLeak(
                slug=slug, shape_point=shape, effective_split=eff,
            ))
    return SplitLeakageScanResult(ok=not leaks, leaks=tuple(leaks))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-path", type=Path, default=_DEFAULT_HEAD_PATH,
        help="Path to calibration_head_v4.json.",
    )
    parser.add_argument(
        "--roster-path", type=Path, default=_DEFAULT_ROSTER_PATH,
        help="Path to sm120_cohort_expansion_roster.toml.",
    )
    args = parser.parse_args(argv)
    with args.head_path.open("r", encoding="utf-8") as fh:
        head = json.load(fh)
    result = scan_v4_for_split_leakage(
        head_payload=head, roster_path=args.roster_path,
    )
    if result.ok:
        print(
            f"OK: split-leakage scan clean over "
            f"{args.head_path.name} ({len(head.get('training_cells', []))} "
            f"training cells; 0 validation/holdout leaks)",
        )
        return 0
    print(
        f"FAIL: split-leakage scan found {len(result.leaks)} "
        f"validation/holdout cell(s) in {args.head_path.name}'s "
        f"training_cells:",
        file=sys.stderr,
    )
    for leak in result.leaks:
        print(
            f"  {leak.slug}/{leak.shape_point} "
            f"(effective_split={leak.effective_split!r})",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
