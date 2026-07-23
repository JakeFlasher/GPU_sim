"""SimBench-BW TTS reference family.

Two TTS operator references:

- `tts_duration_predictor_ln_bf16`: typical TTS duration predictor.
  Plan-row 67 contract: "duration predictor conv + LN". Pipeline:
  Conv1d(H->H, kernel=3, causal-style zero-pad) -> +b1 -> SiLU ->
  LayerNorm(over H) -> Linear(H->1) -> exp. Input X is
  [batch, seq, hidden]; output is [batch, seq] giving one predicted
  duration per token.
- `tts_vocoder_upsample_silu_bf16`: learned upsample (per-channel
  per-output-position weights) followed by SiLU. Plan-row 68
  contract: "transposed / upsample conv + SiLU". Input X is
  [batch, channels, length]; output is
  [batch, channels, length * upsample_factor]. The upsample is
  Y[b, c, l*UF + k] = silu(W_upsample[c, k] * X[b, c, l]) — a
  non-overlapping learned 1D transposed conv with stride=UF and
  per-channel kernel of length UF.

Seeded inputs (same MWC PRNG as host drivers via
`_common.deterministic_seeded_tensor`):

Duration predictor:
    X       = seed
    W_conv  = seed + 1 : (H, H, K=3)       conv1d kernel along seq
    b1      = seed + 2 : (H,)              conv bias
    gamma   = seed + 3 : (H,)              LN gain per hidden
    beta    = seed + 4 : (H,)              LN shift per hidden
    W2      = seed + 5 : (H,)              duration predictor projector
    b2      = seed + 6 : scalar

Vocoder:
    X           = seed
    W_upsample  = seed + 1 : (C, UF)       per-channel learned upsample

Tolerance entry: `(BF16, tree_sum, tts)` in `tolerances.toml`.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "tts"

# R29 closes the R28-review blocking tts drift: both tts rows
# in `roster.toml` are now `reference_type = "cpu"` (previously
# `tts_vocoder_upsample_silu_bf16` was advertised as `fp32_gpu` while
# the shipped reference was pure-Python CPU). With the roster side
# aligned, this family now opts into `check_reference_type_runtime`.
REFERENCE_BACKEND: str = "cpu"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "tts_duration_predictor_ln_bf16",
    "tts_vocoder_upsample_silu_bf16",
)


def _silu(x: float) -> float:
    """SiLU(x) = x * sigmoid(x). sigmoid(x) = 1/(1+exp(-x))."""
    return x / (1.0 + math.exp(-x))


_DURATION_CONV_K = 3


def _duration_predictor(axes: dict[str, Any], seed: int) -> list:
    """[B, S] = exp(LN(SiLU(conv1d(X) + b1)) -> Linear -> + b2).

    The conv1d stage is causal (kernel=3, zero-pad on negative indices)
    and convolves along the seq axis: hidden_pre[b, s, h] = sum_{k, c}
    X[b, s-k, c] * W_conv[h, c, k]. This is the plan's row-67 contract
    "duration predictor conv + LN" — the first stage is a conv, not a
    pointwise linear.
    """
    B = int(axes["batch"])
    S = int(axes["seq"])
    H = int(axes["hidden"])
    K = _DURATION_CONV_K
    eps = 1e-5

    X      = deterministic_seeded_tensor((B, S, H),  seed,     low=-0.5, high=0.5)
    W_conv = deterministic_seeded_tensor((H, H, K),  seed + 1, low=-0.3, high=0.3)
    b1     = deterministic_seeded_tensor((H,),       seed + 2, low=-0.5, high=0.5)
    gamma  = deterministic_seeded_tensor((H,),       seed + 3, low=-0.5, high=0.5)
    beta   = deterministic_seeded_tensor((H,),       seed + 4, low=-0.5, high=0.5)
    W2     = deterministic_seeded_tensor((H,),       seed + 5, low=-0.5, high=0.5)
    b2_vec = deterministic_seeded_tensor((1,),       seed + 6, low=-0.5, high=0.5)
    b2 = b2_vec[0]

    out: list = [[0.0] * S for _ in range(B)]
    for b in range(B):
        X_b = X[b]
        for s in range(S):
            # Conv1d along seq with causal zero-padding.
            hidden1 = [0.0] * H
            for h in range(H):
                acc = b1[h]
                for k in range(K):
                    tau = s - k
                    if tau < 0:
                        continue
                    X_tau = X_b[tau]
                    w_h = W_conv[h]
                    for c in range(H):
                        acc += X_tau[c] * w_h[c][k]
                hidden1[h] = _silu(acc)
            # LayerNorm over hidden dim.
            mean = sum(hidden1) / H
            var_sum = 0.0
            for v in hidden1:
                d = v - mean
                var_sum += d * d
            var = var_sum / H
            rstd = 1.0 / math.sqrt(var + eps)
            # Linear projection + bias + exp; fused with LN affine.
            dot = b2
            for h in range(H):
                normed = (hidden1[h] - mean) * rstd
                scaled = normed * gamma[h] + beta[h]
                dot += scaled * W2[h]
            out[b][s] = math.exp(dot)
    return out


def _vocoder_upsample_silu(axes: dict[str, Any], seed: int) -> list:
    """Learned 1D transposed-conv upsample with stride=UF, kernel=UF.

    Y[b, c, l*UF + k] = silu(W_upsample[c, k] * X[b, c, l])

    Per-channel learned weights: each input position l contributes UF
    distinct output values (one per k in 0..UF-1) via a row in
    W_upsample[c]. Collapses to nearest-neighbor replication when
    W_upsample is identically 1.
    """
    B = int(axes["batch"])
    C = int(axes["channels"])
    L = int(axes["length"])
    UF = int(axes["upsample_factor"])

    X          = deterministic_seeded_tensor((B, C, L),  seed,     low=-0.5, high=0.5)
    W_upsample = deterministic_seeded_tensor((C, UF),    seed + 1, low=-1.0, high=1.0)

    out_len = L * UF
    out: list = [[[0.0] * out_len for _ in range(C)] for _ in range(B)]
    for b in range(B):
        Xb = X[b]
        out_b = out[b]
        for c in range(C):
            X_row = Xb[c]
            W_row = W_upsample[c]
            out_row = out_b[c]
            for l in range(L):
                base = l * UF
                xv = X_row[l]
                for k in range(UF):
                    out_row[base + k] = _silu(W_row[k] * xv)
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    """FP32 reference for `tts_duration_predictor_ln_bf16` and
    `tts_vocoder_upsample_silu_bf16`. Dispatches on slug; raises
    ValueError for any slug this family does not cover."""
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    if slug == "tts_duration_predictor_ln_bf16":
        return _duration_predictor(axes, seed)
    # slug == "tts_vocoder_upsample_silu_bf16"
    return _vocoder_upsample_silu(axes, seed)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    """Tolerance-aware allclose resolved via the row's tolerances.toml
    entry. Both tts rows use `(BF16, tree_sum, tts)`."""
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
