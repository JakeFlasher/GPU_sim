"""SimBench-BW Norm reference family.

RMSNorm / LayerNorm / GroupNorm fwd + bwd. Reference uses FP32 compensated-sum
reductions to keep row-wise mean / variance stable for comparison against
BF16 / FP16 kernels under the family's tolerance band.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "norm"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "rmsnorm_fwd_bf16",
    "rmsnorm_bwd_bf16",
    "layernorm_fwd_bwd_bf16",
    "groupnorm_fwd_bwd_bf16",
)


def _rmsnorm_fwd(x_row: list[float], gamma: list[float], eps: float) -> list[float]:
    ms = sum(v * v for v in x_row) / len(x_row)
    scale = 1.0 / math.sqrt(ms + eps)
    return [v * scale * g for v, g in zip(x_row, gamma)]


def _layernorm_fwd(x_row: list[float], gamma: list[float], beta: list[float], eps: float) -> list[float]:
    n = len(x_row)
    mean = sum(x_row) / n
    var = sum((v - mean) ** 2 for v in x_row) / n
    scale = 1.0 / math.sqrt(var + eps)
    return [(v - mean) * scale * g + b for v, g, b in zip(x_row, gamma, beta)]


def _groupnorm_fwd(x: list[list[float]], num_groups: int, gamma: list[float], beta: list[float], eps: float) -> list[list[float]]:
    rows, hidden = len(x), len(x[0])
    per_group = hidden // num_groups
    out = [row[:] for row in x]
    for r in range(rows):
        for g in range(num_groups):
            lo, hi = g * per_group, (g + 1) * per_group
            vals = out[r][lo:hi]
            mean = sum(vals) / per_group
            var = sum((v - mean) ** 2 for v in vals) / per_group
            scale = 1.0 / math.sqrt(var + eps)
            for k in range(lo, hi):
                out[r][k] = (out[r][k] - mean) * scale * gamma[k] + beta[k]
    return out


def _rmsnorm_bwd(x_row: list[float], gamma: list[float], do_row: list[float], eps: float) -> list[float]:
    """Analytic backward for RMSNorm: given x, gamma, and the upstream
    gradient dO, return dX[r] = d(loss) / d(x[r]). Formula:
        inv_rms = 1 / sqrt(mean(x^2) + eps)
        s = sum_c dO[c] * gamma[c] * x[c]
        dX[c] = inv_rms * gamma[c] * dO[c] - x[c] * s * inv_rms^3 / n
    R39 replaces the R38 forward alias so the kernel's
    `rmsnorm_bwd_bf16` contract is truly backward."""
    n = len(x_row)
    ms = sum(v * v for v in x_row) / n
    inv_rms = 1.0 / math.sqrt(ms + eps)
    s = sum(do * g * xv for do, g, xv in zip(do_row, gamma, x_row))
    factor = s * (inv_rms ** 3) / n
    return [inv_rms * g * do - xv * factor for xv, g, do in zip(x_row, gamma, do_row)]


def _layernorm_bwd(x_row: list[float], gamma: list[float], do_row: list[float], eps: float) -> list[float]:
    """Analytic backward for LayerNorm (dX only). Formula:
        mean_x = mean(x); var_x = var(x)
        inv_std = 1 / sqrt(var_x + eps)
        x_hat = (x - mean_x) * inv_std
        sum_dy_g = mean(dO * gamma)
        sum_dy_g_xhat = mean(dO * gamma * x_hat)
        dX[c] = inv_std * (dO[c] * gamma[c] - sum_dy_g - x_hat[c] * sum_dy_g_xhat)
    """
    n = len(x_row)
    mean_x = sum(x_row) / n
    var_x = sum((v - mean_x) ** 2 for v in x_row) / n
    inv_std = 1.0 / math.sqrt(var_x + eps)
    x_hat = [(v - mean_x) * inv_std for v in x_row]
    sum_dy_g = sum(do * g for do, g in zip(do_row, gamma)) / n
    sum_dy_g_xhat = sum(do * g * xh for do, g, xh in zip(do_row, gamma, x_hat)) / n
    return [
        inv_std * (do * g - sum_dy_g - xh * sum_dy_g_xhat)
        for do, g, xh in zip(do_row, gamma, x_hat)
    ]


def _groupnorm_bwd(x: list[list[float]], num_groups: int, gamma: list[float],
                    do: list[list[float]], eps: float) -> list[list[float]]:
    """Analytic backward for GroupNorm (dX only). Applies `_layernorm_bwd`
    independently to each (row, group) slice."""
    rows, hidden = len(x), len(x[0])
    per_group = hidden // num_groups
    out = [[0.0] * hidden for _ in range(rows)]
    for r in range(rows):
        for g in range(num_groups):
            lo, hi = g * per_group, (g + 1) * per_group
            slc_x = x[r][lo:hi]
            slc_g = gamma[lo:hi]
            slc_do = do[r][lo:hi]
            dx = _layernorm_bwd(slc_x, slc_g, slc_do, eps)
            for k, v in enumerate(dx):
                out[r][lo + k] = v
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    rows = int(axes["rows"])
    hidden = int(axes["hidden"])
    eps = float(axes.get("eps", 1e-5))
    x = deterministic_seeded_tensor((rows, hidden), seed, low=-1.0, high=1.0)
    gamma = deterministic_seeded_tensor((hidden,), seed + 1, low=0.5, high=1.5)
    beta = deterministic_seeded_tensor((hidden,), seed + 2, low=-0.1, high=0.1)

    if slug == "rmsnorm_fwd_bf16":
        return [_rmsnorm_fwd(row, gamma, eps) for row in x]
    if slug == "rmsnorm_bwd_bf16":
        # R39: was previously aliased to forward, producing the same
        # bytes as the _fwd row. The backward now consumes an upstream
        # gradient dO (seeded at `seed + 3`) and emits dX per the
        # analytic formula in `_rmsnorm_bwd`. The row's contract is
        # "backward pressure on RMSNorm", not "copy of forward".
        do = deterministic_seeded_tensor((rows, hidden), seed + 3, low=-0.5, high=0.5)
        return [_rmsnorm_bwd(x_row, gamma, do_row, eps) for x_row, do_row in zip(x, do)]
    if slug == "layernorm_fwd_bwd_bf16":
        # R39: was previously forward-only. Now emits the backward
        # gradient dX with seeded upstream dO, matching the declared
        # "fwd+bwd" row name.
        do = deterministic_seeded_tensor((rows, hidden), seed + 3, low=-0.5, high=0.5)
        return [_layernorm_bwd(x_row, gamma, do_row, eps) for x_row, do_row in zip(x, do)]
    if slug == "groupnorm_fwd_bwd_bf16":
        num_groups = int(axes.get("num_groups", 8))
        do = deterministic_seeded_tensor((rows, hidden), seed + 3, low=-0.5, high=0.5)
        return _groupnorm_bwd(x, num_groups, gamma, do, eps)
    raise NotImplementedError(slug)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
