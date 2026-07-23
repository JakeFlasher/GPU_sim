"""SimBench-BW Multimodal reference family.

Cross-attention where Q comes from one modality and K / V from another.
Currently covers a single row:

- `clip_text_image_cross_attn_bf16`: CLIP-style text-image cross
  attention. Q is derived from text tokens
  `[batch, heads, seq_text, head_dim]`; K / V are derived from image
  tokens `[batch, heads, seq_image, head_dim]`. Standard scaled
  dot-product attention over a rectangular (seq_text × seq_image)
  attention matrix, with row-wise softmax over the image sequence.

Seeded inputs (same MWC PRNG as host drivers via
`_common.deterministic_seeded_tensor`):
    Q = seed     : (batch, heads, seq_text,  head_dim)
    K = seed + 1 : (batch, heads, seq_image, head_dim)
    V = seed + 2 : (batch, heads, seq_image, head_dim)

Tolerance entry: `(BF16, tree_sum, multimodal)` in `tolerances.toml`.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "multimodal"

# R29 lands a pure-Python cross-attention reference for the one
# covered row; the roster row is relabelled `reference_type = "cpu"`
# in the same commit so this family opts into
# `verify_coverage.py::check_reference_type_runtime`.
REFERENCE_BACKEND: str = "cpu"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "clip_text_image_cross_attn_bf16",
)


def _cross_attention(axes: dict[str, Any], seed: int) -> list:
    """[B, H, St, D] = softmax(Q @ K^T / sqrt(D), axis=j) @ V.

    Q: [B, H, St, D] (text queries).
    K / V: [B, H, Si, D] (image keys / values).
    Output: [B, H, St, D]."""
    B  = int(axes["batch"])
    H  = int(axes["heads"])
    St = int(axes["seq_text"])
    Si = int(axes["seq_image"])
    D  = int(axes["head_dim"])

    Q = deterministic_seeded_tensor((B, H, St, D), seed,     low=-0.5, high=0.5)
    K = deterministic_seeded_tensor((B, H, Si, D), seed + 1, low=-0.5, high=0.5)
    V = deterministic_seeded_tensor((B, H, Si, D), seed + 2, low=-0.5, high=0.5)

    scale = 1.0 / math.sqrt(D)
    out: list = [[[[0.0] * D for _ in range(St)] for _ in range(H)] for _ in range(B)]
    for b in range(B):
        for h in range(H):
            Qh = Q[b][h]
            Kh = K[b][h]
            Vh = V[b][h]
            out_h = out[b][h]
            # Scores[i][j] = Qh[i] dot Kh[j] * scale
            scores = [[0.0] * Si for _ in range(St)]
            for i in range(St):
                Qi = Qh[i]
                row = scores[i]
                for j in range(Si):
                    Kj = Kh[j]
                    s = 0.0
                    for d in range(D):
                        s += Qi[d] * Kj[d]
                    row[j] = s * scale
            # Stable row-wise softmax over the image dim.
            for i in range(St):
                row = scores[i]
                m = max(row)
                exps = [math.exp(x - m) for x in row]
                total = sum(exps)
                probs = [e / total for e in exps]
                # out_h[i][d] = sum_j probs[j] * Vh[j][d]
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
    """FP32 cross-attention reference. Dispatches on slug; raises
    ValueError for any slug this family does not cover."""
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    return _cross_attention(shape_point["axes"], seed)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    """Tolerance-aware allclose resolved via
    `(BF16, tree_sum, multimodal)` in `tolerances.toml`."""
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
