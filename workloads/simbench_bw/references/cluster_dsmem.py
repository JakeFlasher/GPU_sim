"""SimBench-BW Cluster / DSMEM reference family.

Cluster + DSMEM microbenches (pairwise latency, ring bandwidth, remote
reduction, cluster RMSNorm). The reference simulates the cooperative-launch
semantics on CPU: pairs of CTAs write seeded tiles into shared memory, and
the expected post-reduction output is computed in FP32.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "cluster_dsmem"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "dsmem_pairwise_latency",
    "dsmem_ring_bandwidth",
    "cluster_remote_reduction",
    "cluster_rmsnorm_dsmem_bf16",
)


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]

    if slug == "dsmem_pairwise_latency":
        # Pairwise: peer CTA 1 reads CTA 0's seeded tile. Reference = identity
        # of the seeded tile (the kernel should report the same bytes).
        tiles = int(axes.get("tiles", 4))
        elements = int(axes.get("elements", 64))
        return deterministic_seeded_tensor((tiles, elements), seed)

    if slug == "dsmem_ring_bandwidth":
        # Ring: each CTA writes to its right neighbor. Reference = shifted
        # copy of the seeded tiles.
        cluster_size = int(axes.get("cluster_size", 4))
        elements = int(axes.get("elements", 64))
        base = deterministic_seeded_tensor((cluster_size, elements), seed)
        shifted = [base[(i - 1) % cluster_size] for i in range(cluster_size)]
        return shifted

    if slug == "cluster_remote_reduction":
        # Remote reduction: every CTA atomically adds its seeded value to a
        # single accumulator per element. Reference = sum across CTAs.
        cluster_size = int(axes.get("cluster_size", 4))
        tiles = int(axes.get("tiles", 4))
        elements = int(axes.get("elements", 64))
        out = [0.0] * elements
        for c in range(cluster_size):
            tile = deterministic_seeded_tensor((tiles, elements), seed + c)
            for t in range(tiles):
                for e in range(elements):
                    out[e] += tile[t][e]
        return out

    if slug == "cluster_rmsnorm_dsmem_bf16":
        # Cluster-wide RMSNorm: mean-square across cluster members, then
        # normalize each member's contribution.
        cluster_size = int(axes.get("cluster_size", 4))
        rows = int(axes.get("rows", 8))
        hidden = int(axes.get("hidden", 32))
        eps = 1e-5
        segments = [deterministic_seeded_tensor((rows, hidden // cluster_size), seed + c) for c in range(cluster_size)]
        rebuilt = [
            [v for c in range(cluster_size) for v in segments[c][r]]
            for r in range(rows)
        ]
        out = [[0.0] * hidden for _ in range(rows)]
        for r in range(rows):
            ms = sum(v * v for v in rebuilt[r]) / hidden
            scale = 1.0 / math.sqrt(ms + eps)
            for c in range(hidden):
                out[r][c] = rebuilt[r][c] * scale
        return out

    raise NotImplementedError(slug)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
