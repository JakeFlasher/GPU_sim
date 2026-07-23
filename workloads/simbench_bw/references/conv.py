"""SimBench-BW Conv reference family.

VAE-style fused operator: 3x3 same-padding convolution -> GroupNorm ->
SiLU -> residual-add. Reference is computed in FP32 against BF16-cast
inputs, matching the kernel's FP32-accumulate / BF16-I/O contract.

Seeded inputs (shared MWC PRNG from `_common.deterministic_seeded_tensor`):
    X     = seed        : (batch, channels, height, width)
    W     = seed + 1    : (channels, channels, 3, 3)   # conv weights, cout x cin x ky x kx
    b     = seed + 2    : (channels,)                    # conv bias
    gamma = seed + 3    : (channels,)                    # GN per-channel gain
    beta  = seed + 4    : (channels,)                    # GN per-channel shift
all drawn from [-0.5, 0.5).

Group count: `channels // 32` (standard VAE convention); falls back to 1
group when `channels < 32`. Epsilon for GN is 1e-5.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "conv"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "vae_conv_gn_silu_residual_bf16",
)


def _conv3x3_same_pad(
    X: list,       # [N][C][H][W]
    W: list,       # [C_out=C][C_in=C][3][3]
    b: list,       # [C]
    N: int, C: int, H: int, Width: int,
) -> list:
    # Pre-flatten bias into a pythonic lookup for speed.
    out = [[[[0.0] * Width for _ in range(H)] for _ in range(C)] for _ in range(N)]
    for n in range(N):
        Xn = X[n]
        for cout in range(C):
            Wcout = W[cout]
            bias_val = b[cout]
            out_c = out[n][cout]
            for y in range(H):
                # Pre-compute valid ky range for this y under same-padding:
                # iy = y + ky - 1 must satisfy 0 <= iy < H  =>  ky in [max(0, 1 - y), min(3, H - y + 1)].
                ky_lo = 0 if y >= 1 else 1
                ky_hi = 3 if (H - y) >= 2 else 2
                for x in range(Width):
                    kx_lo = 0 if x >= 1 else 1
                    kx_hi = 3 if (Width - x) >= 2 else 2
                    acc = bias_val
                    for cin in range(C):
                        Xncin = Xn[cin]
                        Wci = Wcout[cin]
                        for ky in range(ky_lo, ky_hi):
                            iy = y + ky - 1
                            Xrow = Xncin[iy]
                            Wky = Wci[ky]
                            for kx in range(kx_lo, kx_hi):
                                ix = x + kx - 1
                                acc += Xrow[ix] * Wky[kx]
                    out_c[y][x] = acc
    return out


def _group_norm_silu_residual(
    conv_out: list,    # [N][C][H][W]
    X: list,           # [N][C][H][W] (residual input)
    gamma: list,       # [C]
    beta: list,        # [C]
    N: int, C: int, H: int, Width: int,
) -> list:
    group_size = 32 if C >= 32 else C
    num_groups = max(1, C // group_size)
    eps = 1e-5
    elements_per_group = group_size * H * Width

    out = [[[[0.0] * Width for _ in range(H)] for _ in range(C)] for _ in range(N)]
    for n in range(N):
        for g in range(num_groups):
            c_start = g * group_size
            c_end = c_start + group_size
            # Pass 1: mean + sum-of-squares.
            s = 0.0
            s2 = 0.0
            for c in range(c_start, c_end):
                conv_c = conv_out[n][c]
                for y in range(H):
                    row = conv_c[y]
                    for x in range(Width):
                        v = row[x]
                        s += v
                        s2 += v * v
            mean = s / elements_per_group
            var = s2 / elements_per_group - mean * mean
            if var < 0.0:
                var = 0.0
            rstd = 1.0 / math.sqrt(var + eps)
            # Pass 2: normalize, scale+shift, SiLU, residual.
            for c in range(c_start, c_end):
                gm = gamma[c]
                bt = beta[c]
                conv_c = conv_out[n][c]
                Xc = X[n][c]
                out_c = out[n][c]
                for y in range(H):
                    conv_row = conv_c[y]
                    X_row = Xc[y]
                    out_row = out_c[y]
                    for x in range(Width):
                        normed = (conv_row[x] - mean) * rstd
                        scaled = normed * gm + bt
                        # SiLU = scaled * sigmoid(scaled)
                        sig = 1.0 / (1.0 + math.exp(-scaled))
                        out_row[x] = scaled * sig + X_row[x]
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    """FP32 reference for `vae_conv_gn_silu_residual_bf16`.

    Pipeline: 3x3 same-padding conv -> GroupNorm (32 channels/group) ->
    SiLU -> residual-add. Output shape = `(batch, channels, height, width)`.

    Contract: when `channels >= 32`, `channels % 32 == 0` is REQUIRED
    because the GroupNorm partitions channels into fixed 32-channel
    groups. `channels = 33` would silently leave channel 32 outside any
    group (floor-divided `num_groups = 1`) and the CUDA kernel's final
    pass would read `mean/var` out of bounds for channel 32.
    Refusing upfront avoids that silent corruption.
    """
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    N = int(axes["batch"])
    C = int(axes["channels"])
    H = int(axes["height"])
    Width = int(axes["width"])

    if C >= 32 and C % 32 != 0:
        raise ValueError(
            f"groupnorm_channels_not_divisible_by_32: channels={C} "
            f"(conv/VAE row uses fixed 32-channel groups; channels must "
            f"satisfy channels % 32 == 0 when channels >= 32)"
        )

    X = deterministic_seeded_tensor((N, C, H, Width), seed,       low=-0.5, high=0.5)
    W = deterministic_seeded_tensor((C, C, 3, 3),      seed + 1,  low=-0.5, high=0.5)
    b = deterministic_seeded_tensor((C,),              seed + 2,  low=-0.5, high=0.5)
    gamma = deterministic_seeded_tensor((C,),          seed + 3,  low=-0.5, high=0.5)
    beta  = deterministic_seeded_tensor((C,),          seed + 4,  low=-0.5, high=0.5)

    conv_out = _conv3x3_same_pad(X, W, b, N, C, H, Width)
    return _group_norm_silu_residual(conv_out, X, gamma, beta, N, C, H, Width)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
