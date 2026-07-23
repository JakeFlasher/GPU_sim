"""SimBench-BW RoPE reference family.

Standard / YaRN / 2D RoPE applied to fused QKV projections. Reference builds
cos / sin tables directly and rotates pairs of head-dim components in FP32.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "rope"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "qkv_rope_standard_bf16",
    "qkv_rope_yarn_bf16",
    "rope2d_mm_qkv_bf16",
)


def _cos_sin_table(max_pos: int, head_dim: int, base: float = 10000.0, yarn_scale: float = 1.0) -> tuple[list[list[float]], list[list[float]]]:
    half = head_dim // 2
    inv_freq = [1.0 / (base ** (2 * k / head_dim)) for k in range(half)]
    cos_tbl = [[math.cos((p * f) / yarn_scale) for f in inv_freq] for p in range(max_pos)]
    sin_tbl = [[math.sin((p * f) / yarn_scale) for f in inv_freq] for p in range(max_pos)]
    return cos_tbl, sin_tbl


def _apply_rope(row: list[float], cos_row: list[float], sin_row: list[float]) -> list[float]:
    half = len(row) // 2
    out = row[:]
    for k in range(half):
        r_even = row[2 * k]
        r_odd = row[2 * k + 1]
        c = cos_row[k]
        s = sin_row[k]
        out[2 * k] = r_even * c - r_odd * s
        out[2 * k + 1] = r_even * s + r_odd * c
    return out


def _apply_rope_2d(row: list[float], cos_h: list[float], sin_h: list[float],
                    cos_w: list[float], sin_w: list[float]) -> list[float]:
    """2D RoPE: split head_dim into four quarters. The first half
    (indices [0, head_dim/2)) is rotated under the h-coordinate
    theta schedule; the second half is rotated under the w-coordinate
    schedule. Each half follows the same adjacent-pair rotation
    structure as 1D RoPE.

    R39 replaces the R38 alias of `rope2d_mm_qkv_bf16`
    with a real 2D positional schedule so token outputs depend on
    BOTH h and w coordinates."""
    head_dim = len(row)
    half = head_dim // 2
    out = row[:]
    # h-rotation on [0, half).
    quarter_h = half // 2
    for k in range(quarter_h):
        r_even = row[2 * k]
        r_odd  = row[2 * k + 1]
        c = cos_h[k]
        s = sin_h[k]
        out[2 * k]     = r_even * c - r_odd * s
        out[2 * k + 1] = r_even * s + r_odd * c
    # w-rotation on [half, head_dim).
    quarter_w = half // 2
    for k in range(quarter_w):
        base = half + 2 * k
        r_even = row[base]
        r_odd  = row[base + 1]
        c = cos_w[k]
        s = sin_w[k]
        out[base]     = r_even * c - r_odd * s
        out[base + 1] = r_even * s + r_odd * c
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    tokens = int(axes["tokens"])
    head_dim = int(axes["head_dim"])
    heads = int(axes["heads"])

    if slug == "rope2d_mm_qkv_bf16":
        # R39: 2D positional schedule. Token p is placed on a
        # spatial grid of shape [grid_h, grid_w] in row-major order:
        # p_h = p // grid_w, p_w = p % grid_w. `grid_h` / `grid_w`
        # default to `axes.grid_h` / `axes.grid_w` if provided;
        # otherwise tokens must be a perfect square and we infer a
        # square grid. Each token's h-half and w-half are rotated
        # under their respective coordinate schedule (half the head
        # dim per axis).
        grid_h = int(axes.get("grid_h", 0))
        grid_w = int(axes.get("grid_w", 0))
        if grid_h == 0 or grid_w == 0:
            import math as _m
            side = int(_m.isqrt(tokens))
            assert side * side == tokens, (
                "rope2d_mm_qkv_bf16 requires either (grid_h, grid_w) axes or a perfect-square `tokens`"
            )
            grid_h = grid_w = side
        assert grid_h * grid_w == tokens, (
            f"rope2d_mm_qkv_bf16: grid_h * grid_w ({grid_h} * {grid_w}) must equal tokens ({tokens})"
        )
        # Quarter frequencies for each half of the head dim.
        quarter = head_dim // 4
        cos_h_tbl = [[math.cos(p * (1.0 / (10000.0 ** (2 * k / (head_dim // 2))))) for k in range(quarter)] for p in range(grid_h)]
        sin_h_tbl = [[math.sin(p * (1.0 / (10000.0 ** (2 * k / (head_dim // 2))))) for k in range(quarter)] for p in range(grid_h)]
        cos_w_tbl = [[math.cos(p * (1.0 / (10000.0 ** (2 * k / (head_dim // 2))))) for k in range(quarter)] for p in range(grid_w)]
        sin_w_tbl = [[math.sin(p * (1.0 / (10000.0 ** (2 * k / (head_dim // 2))))) for k in range(quarter)] for p in range(grid_w)]
        qkv = deterministic_seeded_tensor((tokens, heads, head_dim), seed)
        out = []
        for p in range(tokens):
            p_h = p // grid_w
            p_w = p %  grid_w
            head_out = []
            for h in range(heads):
                head_out.append(_apply_rope_2d(
                    qkv[p][h], cos_h_tbl[p_h], sin_h_tbl[p_h], cos_w_tbl[p_w], sin_w_tbl[p_w],
                ))
            out.append(head_out)
        return out

    yarn_scale = 1.0
    if slug == "qkv_rope_yarn_bf16":
        yarn_scale = float(axes.get("yarn_scale", 4.0))
    cos_tbl, sin_tbl = _cos_sin_table(tokens, head_dim, yarn_scale=yarn_scale)

    qkv = deterministic_seeded_tensor((tokens, heads, head_dim), seed)
    out = []
    for p in range(tokens):
        head_out = []
        for h in range(heads):
            head_out.append(_apply_rope(qkv[p][h], cos_tbl[p], sin_tbl[p]))
        out.append(head_out)
    return out


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
