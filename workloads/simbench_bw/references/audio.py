"""SimBench-BW Audio reference family.

Covers two L3 audio-domain rows:

- `conformer_depthwise_glu_bf16` — Conformer convolution module:
  pointwise(1x1) up to 2*channels -> GLU split -> depthwise conv1d
  over the time axis with kernel size `kernel_size` -> SiLU ->
  pointwise(1x1) back to channels. Reference shape:
      x:   [batch, time, channels]
      h0 = GLU(x @ W_in)               # split last dim into (a, b); a * sigmoid(b)
      h1 = depthwise_conv1d_time(h0)   # per-channel time conv, kernel_size, zero pad
      h2 = silu(h1)
      out = h2 @ W_out                 # pointwise back to channels
  Axes: `batch`, `time`, `channels`, `kernel_size`. Output
  `[batch, time, channels]`.

- `whisper_encoder_block_bf16` — Whisper encoder block: an audio
  conv front-end followed by a pre-norm self-attention + pre-norm
  2-layer MLP with GELU. The conv front-end is a causal 1D conv along
  the seq axis with kernel_size=3 (standard Whisper conv stem). The
  conv output then feeds the transformer block exactly as in a ViT
  encoder. Axes: `batch`, `seq`, `heads`, `head_dim`, `mlp_dim`. The
  conv kernel size is fixed at 3 (the row's contract; not exposed in
  the shape axes).

Seeded inputs (same MWC PRNG as host drivers via
`_common.deterministic_seeded_tensor`):

Conformer block:
    x        = seed       : [batch, time, channels]
    W_in     = seed + 1   : [channels, 2 * channels] expansion projection
    W_dw     = seed + 2   : [channels, kernel_size] per-channel depthwise weights
    W_out    = seed + 3   : [channels, channels] contraction projection

Whisper encoder block:
    x        = seed       : [batch, seq, hidden]
    W_qkv    = seed + 1   : [hidden, 3 * hidden]
    W_out    = seed + 2   : [hidden, hidden] attn out projection
    W_ff1    = seed + 3   : [hidden, mlp_dim]
    W_ff2    = seed + 4   : [mlp_dim, hidden]
    W_conv   = seed + 5   : [hidden, hidden, kernel=3] audio conv front-end

Tolerance entry: `(BF16, tree_sum, audio)` in `tolerances.toml`.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "audio"


FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "conformer_depthwise_glu_bf16",
    "whisper_encoder_block_bf16",
)


_LAYERNORM_EPS = 1e-5


def _layernorm_row(row: list[float], eps: float) -> list[float]:
    n = len(row)
    if n == 0:
        return []
    mean = sum(row) / n
    var = sum((v - mean) ** 2 for v in row) / n
    scale = 1.0 / math.sqrt(var + eps)
    return [(v - mean) * scale for v in row]


def _gelu(v: float) -> float:
    return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v * v * v)))


def _silu(v: float) -> float:
    return v / (1.0 + math.exp(-v))


def _sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


def _matmul_2d(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
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


def _depthwise_conv1d_time(x_bt: list[list[float]], W_dw: list[list[float]],
                            kernel_size: int) -> list[list[float]]:
    """Per-channel depthwise 1D convolution along the time axis with
    centered zero padding. x_bt shape `[time, channels]`; W_dw shape
    `[channels, kernel_size]`. Output `[time, channels]`."""
    T = len(x_bt)
    C = len(x_bt[0]) if T else 0
    pad = kernel_size // 2
    out = [[0.0] * C for _ in range(T)]
    for t in range(T):
        out_t = out[t]
        for c in range(C):
            wc = W_dw[c]
            acc = 0.0
            for k in range(kernel_size):
                src = t + k - pad
                if 0 <= src < T:
                    acc += x_bt[src][c] * wc[k]
            out_t[c] = acc
    return out


def _conformer_depthwise_glu(axes: dict[str, Any], seed: int) -> list:
    """Conformer convolution module reference.
    Output shape: `[batch, time, channels]`."""
    B  = int(axes["batch"])
    T  = int(axes["time"])
    C  = int(axes["channels"])
    Ks = int(axes.get("kernel_size", 7))

    x     = deterministic_seeded_tensor((B, T, C),       seed,     low=-0.5, high=0.5)
    W_in  = deterministic_seeded_tensor((C, 2 * C),      seed + 1, low=-0.1, high=0.1)
    W_dw  = deterministic_seeded_tensor((C, Ks),         seed + 2, low=-0.1, high=0.1)
    W_out = deterministic_seeded_tensor((C, C),          seed + 3, low=-0.1, high=0.1)

    out: list = [[[0.0] * C for _ in range(T)] for _ in range(B)]
    for b in range(B):
        x_b = x[b]
        # stage 1: pointwise expansion to 2*C, then GLU split.
        expanded = _matmul_2d(x_b, W_in)
        glu = [[0.0] * C for _ in range(T)]
        for t in range(T):
            row = expanded[t]
            for c in range(C):
                a = row[c]
                gate = row[C + c]
                glu[t][c] = a * _sigmoid(gate)

        # stage 2: depthwise conv1d over time.
        conved = _depthwise_conv1d_time(glu, W_dw, Ks)

        # stage 3: SiLU activation.
        for t in range(T):
            row = conved[t]
            for c in range(C):
                row[c] = _silu(row[c])

        # stage 4: pointwise back to C.
        projected = _matmul_2d(conved, W_out)
        out_b = out[b]
        for t in range(T):
            row = projected[t]
            out_t = out_b[t]
            for c in range(C):
                out_t[c] = row[c]
    return out


def _multi_head_self_attention(
    x_b: list[list[float]], W_qkv: list[list[float]], W_out: list[list[float]],
    heads: int, head_dim: int,
) -> list[list[float]]:
    S = len(x_b)
    hidden = heads * head_dim

    qkv = _matmul_2d(x_b, W_qkv)
    Q = [[[0.0] * head_dim for _ in range(S)] for _ in range(heads)]
    K = [[[0.0] * head_dim for _ in range(S)] for _ in range(heads)]
    V = [[[0.0] * head_dim for _ in range(S)] for _ in range(heads)]
    for s in range(S):
        row = qkv[s]
        for h in range(heads):
            base_q = h * head_dim
            base_k = hidden + h * head_dim
            base_v = 2 * hidden + h * head_dim
            for d in range(head_dim):
                Q[h][s][d] = row[base_q + d]
                K[h][s][d] = row[base_k + d]
                V[h][s][d] = row[base_v + d]

    scale = 1.0 / math.sqrt(head_dim)
    attn_out = [[[0.0] * head_dim for _ in range(S)] for _ in range(heads)]
    for h in range(heads):
        Qh, Kh, Vh = Q[h], K[h], V[h]
        for i in range(S):
            Qi = Qh[i]
            scores = [0.0] * S
            for j in range(S):
                Kj = Kh[j]
                s = 0.0
                for d in range(head_dim):
                    s += Qi[d] * Kj[d]
                scores[j] = s * scale
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            total = sum(exps)
            probs = [e / total for e in exps]
            out_row = attn_out[h][i]
            for j in range(S):
                pj = probs[j]
                if pj == 0.0:
                    continue
                Vj = Vh[j]
                for d in range(head_dim):
                    out_row[d] += pj * Vj[d]

    concat = [[0.0] * hidden for _ in range(S)]
    for s in range(S):
        for h in range(heads):
            base = h * head_dim
            for d in range(head_dim):
                concat[s][base + d] = attn_out[h][s][d]
    return _matmul_2d(concat, W_out)


_WHISPER_CONV_KERNEL = 3


def _whisper_audio_conv_front_end(
    x: list[list[list[float]]],
    W_conv: list[list[list[float]]],
    hidden: int,
) -> list[list[list[float]]]:
    """Causal 1D conv along the seq axis.

    x_conv[b, s, h] = sum_{k=0..K-1, c} x[b, s-k, c] * W_conv[h, c, k]
    with zero padding for s - k < 0. Layout matches the conv stem of
    the Whisper audio encoder before the first transformer block.
    """
    B = len(x)
    S = len(x[0])
    K = _WHISPER_CONV_KERNEL
    out = [[[0.0] * hidden for _ in range(S)] for _ in range(B)]
    for b in range(B):
        x_b = x[b]
        out_b = out[b]
        for s in range(S):
            for h in range(hidden):
                acc = 0.0
                for k in range(K):
                    tau = s - k
                    if tau < 0:
                        continue
                    x_row = x_b[tau]
                    w_h = W_conv[h]
                    for c in range(hidden):
                        acc += x_row[c] * w_h[c][k]
                out_b[s][h] = acc
    return out


def _whisper_encoder_block(axes: dict[str, Any], seed: int) -> list:
    """Whisper encoder block: audio conv front-end then pre-norm
    transformer block. Output shape: `[batch, seq, hidden]`."""
    B = int(axes["batch"])
    S = int(axes["seq"])
    Hh = int(axes["heads"])
    D = int(axes["head_dim"])
    M = int(axes["mlp_dim"])
    hidden = Hh * D
    K = _WHISPER_CONV_KERNEL

    x      = deterministic_seeded_tensor((B, S, hidden),         seed,     low=-0.5, high=0.5)
    W_qkv  = deterministic_seeded_tensor((hidden, 3 * hidden),   seed + 1, low=-0.1, high=0.1)
    W_out  = deterministic_seeded_tensor((hidden, hidden),       seed + 2, low=-0.1, high=0.1)
    W_ff1  = deterministic_seeded_tensor((hidden, M),            seed + 3, low=-0.1, high=0.1)
    W_ff2  = deterministic_seeded_tensor((M, hidden),            seed + 4, low=-0.1, high=0.1)
    W_conv = deterministic_seeded_tensor((hidden, hidden, K),    seed + 5, low=-0.1, high=0.1)

    x_conv = _whisper_audio_conv_front_end(x, W_conv, hidden)

    out: list = [[[0.0] * hidden for _ in range(S)] for _ in range(B)]
    for b in range(B):
        x_b = x_conv[b]
        normed_x = [_layernorm_row(x_b[s], _LAYERNORM_EPS) for s in range(S)]
        attn = _multi_head_self_attention(normed_x, W_qkv, W_out, Hh, D)
        h1 = [[x_b[s][d] + attn[s][d] for d in range(hidden)] for s in range(S)]

        normed_h1 = [_layernorm_row(h1[s], _LAYERNORM_EPS) for s in range(S)]
        ff_inner = _matmul_2d(normed_h1, W_ff1)
        for s in range(S):
            row = ff_inner[s]
            for k in range(M):
                row[k] = _gelu(row[k])
        ff_out = _matmul_2d(ff_inner, W_ff2)

        out_b = out[b]
        for s in range(S):
            for d in range(hidden):
                out_b[s][d] = h1[s][d] + ff_out[s][d]
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    """FP32 deterministic reference for the audio family. Dispatches on
    slug; raises ValueError for any slug this family does not cover."""
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    if slug == "conformer_depthwise_glu_bf16":
        return _conformer_depthwise_glu(axes, seed)
    # slug == "whisper_encoder_block_bf16"
    return _whisper_encoder_block(axes, seed)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    """Tolerance-aware allclose resolved via
    `(BF16, tree_sum, audio)` in `tolerances.toml`."""
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
