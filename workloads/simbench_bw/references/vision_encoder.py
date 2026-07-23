"""SimBench-BW Vision Encoder reference family.

Covers two L3 vision-domain rows:

- `vit_encoder_block_bf16` — standard ViT transformer encoder block.
  Pre-norm self-attention + pre-norm 2-layer MLP with GELU. Reference:
      h1 = x + Attention(LN(x))
      out = h1 + MLP(LN(h1))
  with multi-head self-attention `[batch, seq, heads, head_dim]` and
  MLP hidden width `mlp_dim`. Axes: `batch`, `seq`, `heads`,
  `head_dim`, `mlp_dim`. Hidden width = `heads * head_dim`.

- `sam_mask_decoder_cross_attn_bf16` — SAM mask-decoder cross attention.
  Q from mask tokens `[batch, heads, seq_mask, head_dim]`; K / V from
  image tokens `[batch, heads, seq_image, head_dim]`. Standard scaled
  dot-product cross attention with row-wise softmax over the image
  sequence. Output `[batch, heads, seq_mask, head_dim]`. Axes: `batch`,
  `heads`, `seq_mask`, `seq_image`, `head_dim`.

Seeded inputs (same MWC PRNG as host drivers via
`_common.deterministic_seeded_tensor`):

ViT encoder block:
    x        = seed
    W_qkv    = seed + 1   : packed [hidden, 3 * hidden] for Q/K/V projection
    W_out    = seed + 2   : [hidden, hidden] attention output projection
    W_ff1    = seed + 3   : [hidden, mlp_dim] MLP up
    W_ff2    = seed + 4   : [mlp_dim, hidden] MLP down

SAM mask-decoder cross attn:
    Q = seed
    K = seed + 1
    V = seed + 2

Tolerance entry: `(BF16, tree_sum, vision_encoder)` in `tolerances.toml`.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "vision_encoder"


FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "vit_encoder_block_bf16",
    "sam_mask_decoder_cross_attn_bf16",
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
    """Tanh-approximation GELU. Standard ViT activation."""
    return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v * v * v)))


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


def _multi_head_self_attention(
    x_b: list[list[float]], W_qkv: list[list[float]], W_out: list[list[float]],
    heads: int, head_dim: int,
) -> list[list[float]]:
    """Single-batch multi-head self attention.
    x_b: `[seq, hidden]`. Returns `[seq, hidden]`."""
    S = len(x_b)
    hidden = heads * head_dim

    # Pack QKV via a single matmul: x_b @ W_qkv -> [seq, 3 * hidden].
    qkv = _matmul_2d(x_b, W_qkv)

    # Split Q / K / V and reshape to per-head views: [heads, seq, head_dim].
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

    # Per-head SDPA.
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

    # Concat heads back to [seq, hidden] then apply output projection.
    concat = [[0.0] * hidden for _ in range(S)]
    for s in range(S):
        for h in range(heads):
            base = h * head_dim
            for d in range(head_dim):
                concat[s][base + d] = attn_out[h][s][d]
    return _matmul_2d(concat, W_out)


def _vit_encoder_block(axes: dict[str, Any], seed: int) -> list:
    """ViT encoder block. Output shape: `[batch, seq, hidden]`."""
    B = int(axes["batch"])
    S = int(axes["seq"])
    Hh = int(axes["heads"])
    D = int(axes["head_dim"])
    M = int(axes["mlp_dim"])
    hidden = Hh * D

    x     = deterministic_seeded_tensor((B, S, hidden),   seed,     low=-0.5, high=0.5)
    W_qkv = deterministic_seeded_tensor((hidden, 3 * hidden), seed + 1, low=-0.1, high=0.1)
    W_out = deterministic_seeded_tensor((hidden, hidden), seed + 2, low=-0.1, high=0.1)
    W_ff1 = deterministic_seeded_tensor((hidden, M),      seed + 3, low=-0.1, high=0.1)
    W_ff2 = deterministic_seeded_tensor((M, hidden),      seed + 4, low=-0.1, high=0.1)

    out: list = [[[0.0] * hidden for _ in range(S)] for _ in range(B)]
    for b in range(B):
        x_b = x[b]
        # Pre-norm self-attention.
        normed_x = [_layernorm_row(x_b[s], _LAYERNORM_EPS) for s in range(S)]
        attn = _multi_head_self_attention(normed_x, W_qkv, W_out, Hh, D)
        h1 = [[x_b[s][d] + attn[s][d] for d in range(hidden)] for s in range(S)]

        # Pre-norm MLP: GELU(LN(h1) @ W_ff1) @ W_ff2.
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


def _sam_cross_attn(axes: dict[str, Any], seed: int) -> list:
    """SAM mask-decoder cross attention.
    Output shape: `[batch, heads, seq_mask, head_dim]`."""
    B  = int(axes["batch"])
    Hh = int(axes["heads"])
    Sm = int(axes["seq_mask"])
    Si = int(axes["seq_image"])
    D  = int(axes["head_dim"])

    Q = deterministic_seeded_tensor((B, Hh, Sm, D), seed,     low=-0.5, high=0.5)
    K = deterministic_seeded_tensor((B, Hh, Si, D), seed + 1, low=-0.5, high=0.5)
    V = deterministic_seeded_tensor((B, Hh, Si, D), seed + 2, low=-0.5, high=0.5)

    scale = 1.0 / math.sqrt(D)
    out: list = [[[[0.0] * D for _ in range(Sm)] for _ in range(Hh)] for _ in range(B)]
    for b in range(B):
        for h in range(Hh):
            Qh, Kh, Vh = Q[b][h], K[b][h], V[b][h]
            out_h = out[b][h]
            for i in range(Sm):
                Qi = Qh[i]
                scores = [0.0] * Si
                for j in range(Si):
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
                for j in range(Si):
                    pj = probs[j]
                    if pj == 0.0:
                        continue
                    Vj = Vh[j]
                    for d in range(D):
                        out_row[d] += pj * Vj[d]
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    """FP32 deterministic reference for the vision_encoder family.
    Dispatches on slug; raises ValueError for any slug this family does
    not cover."""
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    if slug == "vit_encoder_block_bf16":
        return _vit_encoder_block(axes, seed)
    # slug == "sam_mask_decoder_cross_attn_bf16"
    return _sam_cross_attn(axes, seed)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    """Tolerance-aware allclose resolved via
    `(BF16, tree_sum, vision_encoder)` in `tolerances.toml`."""
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
