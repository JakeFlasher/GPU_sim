"""SimBench-BW Activation reference family.

Fused FFN activations (SwiGLU / GeGLU). Reference computes gate + up
projections, applies the elementwise activation, multiplies, then projects
down. All in FP32.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "activation"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "swiglu_ffn_fused_bf16",
    "geglu_ffn_fused_bf16",
)


def _silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, k = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k == k2
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            aip = a[i][p]
            for j in range(n):
                out[i][j] += aip * b[p][j]
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    tokens = int(axes["tokens"])
    hidden = int(axes["hidden"])
    intermediate = int(axes["intermediate"])

    x = deterministic_seeded_tensor((tokens, hidden), seed)
    w_gate = deterministic_seeded_tensor((hidden, intermediate), seed + 1, low=-0.1, high=0.1)
    w_up = deterministic_seeded_tensor((hidden, intermediate), seed + 2, low=-0.1, high=0.1)
    w_down = deterministic_seeded_tensor((intermediate, hidden), seed + 3, low=-0.1, high=0.1)

    gate = _matmul(x, w_gate)
    up = _matmul(x, w_up)
    act = _silu if slug == "swiglu_ffn_fused_bf16" else _gelu
    fused = [[act(gate[r][c]) * up[r][c] for c in range(intermediate)] for r in range(tokens)]
    return _matmul(fused, w_down)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
