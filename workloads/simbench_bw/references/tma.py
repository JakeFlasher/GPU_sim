"""SimBench-BW TMA / cp.async reference family.

cp.async + cp.async.bulk.tensor tile-copy microbenches. Reference is a pure
identity copy; the output must match the source tensor under the family's
`none`-reduction tolerance band.
"""

from __future__ import annotations

from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "tma"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "cp_async_shared_tile_copy",
    "tma_tensor_copy_2d",
)


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    rows = int(axes.get("rows", axes.get("height", 256)))
    cols = int(axes.get("cols", axes.get("width", 256)))
    return deterministic_seeded_tensor((rows, cols), seed)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
