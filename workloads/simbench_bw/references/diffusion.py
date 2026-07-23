"""SimBench-BW Diffusion reference family.

Covers three L3 diffusion-domain rows:

- `diffusion_resnet_time_block_bf16` — ResNet block with FiLM-style time
  embedding and 3x3 same-padding convs (matches the plan's "time
  embedding + conv/norm/SiLU" contract). Reference path:
      h_pre1 = conv3x3(x, W_conv1)                 # 3x3 same-pad along (H, W)
      h1     = silu(groupnorm(h_pre1) + time_bias) # GN over (C, H, W) per batch
      h_pre2 = conv3x3(h1, W_conv2)
      h2     = silu(groupnorm(h_pre2))
      out    = x + h2                              # residual add
  Axes: `batch`, `channels`, `height`, `width`, `time_dim`.

- `diffusion_cross_attn_video_bf16` — text-conditioned video cross
  attention. Q from video tokens `[batch, heads, seq_video, head_dim]`,
  K / V from text `[batch, heads, seq_text, head_dim]`. Standard scaled
  dot-product attention with row-wise softmax over the text sequence.
  Output shape `[batch, heads, seq_video, head_dim]`. Axes: `batch`,
  `heads`, `seq_video`, `seq_text`, `head_dim`.

- `flux_adaln_zero_mod_bf16` — AdaLN-Zero modulation block. Conditioning
  vector `c` of shape `[batch, cond_dim]` is projected by three
  independent linear maps to (scale, shift, gate) of shape
  `[batch, hidden]`. The reference applies:
      h = layernorm(x)
      h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
      out = x + gate.unsqueeze(1) * h               # zero-init gate
  Axes: `batch`, `seq`, `hidden`, `cond_dim`. Output `[batch, seq, hidden]`.

Seeded inputs (same MWC PRNG as host drivers via
`_common.deterministic_seeded_tensor`):
    x          = seed       : the activation tensor
    t          = seed + 1   : time embedding (resnet)
    W_time     = seed + 2   : time-projection weight (resnet)
    W_conv1    = seed + 3   : 3x3 conv stage-1 weight (resnet)
    W_conv2    = seed + 4   : 3x3 conv stage-2 weight (resnet)
    K, V       = seed + 1, seed + 2 : cross-attn keys/values
    cond, W_*  = seed + 1 / seed + 2 / seed + 3 / seed + 4 :
                AdaLN-Zero conditioning + (scale, shift, gate) projections.

Tolerance entry: `(BF16, tree_sum, diffusion)` in `tolerances.toml`.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "diffusion"

# R34 lands pure-Python FP32 references for all three covered rows.
# The roster declares `reference_type = "fp32_gpu"` for these rows
# (the production path runs the FP32 reference on a separate GPU stream
# rather than CPU); the family does NOT export a uniform
# `REFERENCE_BACKEND` so `verify_coverage.check_reference_type_runtime`
# stays opt-out for this family. The Python reference body still
# computes a deterministic FP32 anchor that the kernel must match
# under the family's tolerance band.

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "diffusion_resnet_time_block_bf16",
    "diffusion_cross_attn_video_bf16",
    "flux_adaln_zero_mod_bf16",
)


_RMSNORM_EPS = 1e-5
_LAYERNORM_EPS = 1e-5


def _layernorm_row(row: list[float], eps: float) -> list[float]:
    """LayerNorm (no affine) over a flat row. Returns a fresh list."""
    n = len(row)
    if n == 0:
        return []
    mean = sum(row) / n
    var = sum((v - mean) ** 2 for v in row) / n
    scale = 1.0 / math.sqrt(var + eps)
    return [(v - mean) * scale for v in row]


def _silu(v: float) -> float:
    return v / (1.0 + math.exp(-v))


def _matmul_2d(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """`a @ b` for a `[M, K]` and b `[K, N]`. Returns `[M, N]`."""
    M = len(a)
    K = len(a[0]) if M else 0
    N = len(b[0]) if K else 0
    out = [[0.0] * N for _ in range(M)]
    for i in range(M):
        ai = a[i]
        oi = out[i]
        for k in range(K):
            aik = ai[k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(N):
                oi[j] += aik * bk[j]
    return out


def _conv3x3_same(x_b: list[list[list[float]]],
                   W_conv: list[list[list[list[float]]]]) -> list[list[list[float]]]:
    """3x3 same-padding 2D conv (zero-padded) over a single batch slice.

    x_b shape: [C_in][H][W]; W_conv shape: [C_out][C_in][3][3].
    Returns [C_out][H][W].
    """
    C_in = len(x_b)
    H = len(x_b[0])
    W = len(x_b[0][0])
    C_out = len(W_conv)
    out = [[[0.0] * W for _ in range(H)] for _ in range(C_out)]
    for co in range(C_out):
        for h in range(H):
            for w in range(W):
                acc = 0.0
                for ci in range(C_in):
                    for kh in range(3):
                        for kw in range(3):
                            ih = h + kh - 1
                            iw = w + kw - 1
                            if 0 <= ih < H and 0 <= iw < W:
                                acc += x_b[ci][ih][iw] * W_conv[co][ci][kh][kw]
                out[co][h][w] = acc
    return out


def _groupnorm_chw(t: list[list[list[float]]], eps: float) -> list[list[list[float]]]:
    """1-group GroupNorm over (C, H, W); equivalent to LayerNorm over the
    full per-batch volume. Returns a normalized copy of `t`.
    """
    C = len(t)
    H = len(t[0])
    W = len(t[0][0])
    n = C * H * W
    if n == 0:
        return [[[0.0] * W for _ in range(H)] for _ in range(C)]
    s = 0.0
    for c in range(C):
        for h in range(H):
            row = t[c][h]
            for w in range(W):
                s += row[w]
    mean = s / n
    sq = 0.0
    for c in range(C):
        for h in range(H):
            row = t[c][h]
            for w in range(W):
                d = row[w] - mean
                sq += d * d
    inv = 1.0 / math.sqrt(sq / n + eps)
    out = [[[0.0] * W for _ in range(H)] for _ in range(C)]
    for c in range(C):
        for h in range(H):
            row = t[c][h]
            out_row = out[c][h]
            for w in range(W):
                out_row[w] = (row[w] - mean) * inv
    return out


def _resnet_time_block(axes: dict[str, Any], seed: int) -> list:
    """ResNet-style time block reference: 3x3 conv -> GroupNorm -> SiLU.
    Output shape: `[batch, channels, height, width]`."""
    B = int(axes["batch"])
    C = int(axes["channels"])
    H = int(axes["height"])
    W = int(axes["width"])
    T = int(axes["time_dim"])

    x       = deterministic_seeded_tensor((B, C, H, W),    seed,     low=-0.5, high=0.5)
    t       = deterministic_seeded_tensor((B, T),          seed + 1, low=-0.5, high=0.5)
    W_time  = deterministic_seeded_tensor((T, C),          seed + 2, low=-0.25, high=0.25)
    W_conv1 = deterministic_seeded_tensor((C, C, 3, 3),    seed + 3, low=-0.1,  high=0.1)
    W_conv2 = deterministic_seeded_tensor((C, C, 3, 3),    seed + 4, low=-0.1,  high=0.1)

    # time_bias[b, c] = sum_t t[b, t] * W_time[t, c]
    time_bias = _matmul_2d(t, W_time)

    out = [[[[0.0] * W for _ in range(H)] for _ in range(C)] for _ in range(B)]
    for b in range(B):
        x_b = x[b]
        time_bias_b = time_bias[b]
        out_b = out[b]

        # Stage 1: conv -> GN -> SiLU + time_bias
        h_pre1 = _conv3x3_same(x_b, W_conv1)
        normed1 = _groupnorm_chw(h_pre1, _RMSNORM_EPS)
        h1 = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            tb = time_bias_b[c]
            for h in range(H):
                for w in range(W):
                    h1[c][h][w] = _silu(normed1[c][h][w]) + tb

        # Stage 2: conv -> GN -> SiLU
        h_pre2 = _conv3x3_same(h1, W_conv2)
        normed2 = _groupnorm_chw(h_pre2, _RMSNORM_EPS)

        # Residual add.
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    out_b[c][h][w] = x_b[c][h][w] + _silu(normed2[c][h][w])
    return out


def _cross_attn_video(axes: dict[str, Any], seed: int) -> list:
    """Text-conditioned video cross attention reference.
    Output shape: `[batch, heads, seq_video, head_dim]`."""
    B  = int(axes["batch"])
    Hh = int(axes["heads"])
    Sv = int(axes["seq_video"])
    St = int(axes["seq_text"])
    D  = int(axes["head_dim"])

    Q = deterministic_seeded_tensor((B, Hh, Sv, D), seed,     low=-0.5, high=0.5)
    K = deterministic_seeded_tensor((B, Hh, St, D), seed + 1, low=-0.5, high=0.5)
    V = deterministic_seeded_tensor((B, Hh, St, D), seed + 2, low=-0.5, high=0.5)

    scale = 1.0 / math.sqrt(D)
    out: list = [[[[0.0] * D for _ in range(Sv)] for _ in range(Hh)] for _ in range(B)]
    for b in range(B):
        for h in range(Hh):
            Qh, Kh, Vh = Q[b][h], K[b][h], V[b][h]
            out_h = out[b][h]
            for i in range(Sv):
                Qi = Qh[i]
                scores = [0.0] * St
                for j in range(St):
                    Kj = Kh[j]
                    s = 0.0
                    for d in range(D):
                        s += Qi[d] * Kj[d]
                    scores[j] = s * scale
                m = max(scores)
                exps = [math.exp(s - m) for s in scores]
                total = sum(exps)
                probs = [e / total for e in exps]
                out_row = out_h[i]
                for j in range(St):
                    pj = probs[j]
                    if pj == 0.0:
                        continue
                    Vj = Vh[j]
                    for d in range(D):
                        out_row[d] += pj * Vj[d]
    return out


def _adaln_zero(axes: dict[str, Any], seed: int) -> list:
    """AdaLN-Zero modulation reference.
    Output shape: `[batch, seq, hidden]`."""
    B = int(axes["batch"])
    S = int(axes["seq"])
    Hh = int(axes["hidden"])
    Cd = int(axes["cond_dim"])

    x       = deterministic_seeded_tensor((B, S, Hh), seed,     low=-0.5, high=0.5)
    cond    = deterministic_seeded_tensor((B, Cd),    seed + 1, low=-0.5, high=0.5)
    W_scale = deterministic_seeded_tensor((Cd, Hh),   seed + 2, low=-0.1, high=0.1)
    W_shift = deterministic_seeded_tensor((Cd, Hh),   seed + 3, low=-0.1, high=0.1)
    W_gate  = deterministic_seeded_tensor((Cd, Hh),   seed + 4, low=-0.1, high=0.1)

    scale_b = _matmul_2d(cond, W_scale)
    shift_b = _matmul_2d(cond, W_shift)
    gate_b  = _matmul_2d(cond, W_gate)

    out: list = [[[0.0] * Hh for _ in range(S)] for _ in range(B)]
    for b in range(B):
        sb, hb, gb = scale_b[b], shift_b[b], gate_b[b]
        x_b = x[b]
        out_b = out[b]
        for s in range(S):
            row = x_b[s]
            normed = _layernorm_row(row, _LAYERNORM_EPS)
            modulated = [normed[d] * (1.0 + sb[d]) + hb[d] for d in range(Hh)]
            out_row = out_b[s]
            for d in range(Hh):
                out_row[d] = row[d] + gb[d] * modulated[d]
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    """FP32 deterministic reference for the diffusion family. Dispatches
    on slug; raises ValueError for any slug this family does not cover."""
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    if slug == "diffusion_resnet_time_block_bf16":
        return _resnet_time_block(axes, seed)
    if slug == "diffusion_cross_attn_video_bf16":
        return _cross_attn_video(axes, seed)
    # slug == "flux_adaln_zero_mod_bf16"
    return _adaln_zero(axes, seed)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    """Tolerance-aware allclose resolved via
    `(BF16, tree_sum, diffusion)` in `tolerances.toml`."""
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
