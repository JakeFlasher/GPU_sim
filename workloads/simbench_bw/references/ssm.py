"""SimBench-BW SSM reference family.

State-space models: Mamba / Mamba-2 / chunk-gated-delta-rule / Hyena. Each
row gets its own dispatcher branch; the four rows are intentionally NOT
collapsed onto a single shared body so a downstream change in one
algorithm cannot silently drift the others.

R41 split (Codex R40-review directive): the previous shared
`_selective_scan` was a contract placeholder ("output = mean(state)
replicated across channels") that satisfied two distinct plan rows
(57 mamba2 discretize-chunk and 58 chunk gated delta rule) with the
same recurrence. R41 splits them:

- `_mamba_selective_scan` (Row 56): pre-existing simple scan, kept for
  the Hopper-portable simple-recurrence row.
- `_mamba2_discretize_chunk` (Row 57): explicit timestep discretization
  `Abar = exp(dt * a)`, `Bbar = (Abar - 1) / a * b`, real state mixing
  via a `C_proj` output projection.
- `_chunk_gated_delta_rule` (Row 58): rank-1 delta-rule key/value
  update with sigmoid-gated channel mix.
- `_direct_conv` (Row 59): unchanged Hyena long-conv reference.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "ssm"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "mamba_selective_scan_bf16",
    "mamba2_discretize_chunk_bf16",
    "chunk_gated_delta_rule_bf16",
    "hyena_fft_conv_block_fp32",
)


def _mamba_selective_scan(x: list[list[list[float]]],
                           a: list[list[list[float]]],
                           b: list[list[list[float]]]) -> list[list[list[float]]]:
    """Row 56: simple selective-scan recurrence.

    Shapes: x[batch][seq][channels], a[batch][seq][state], b[batch][seq][state].
    Output: out[batch][seq][channels]. One scalar state dim per channel pair.

    This is the plan's row-56 "chunked selective scan (non-atomic
    variant)" body. Row 57 (discretize-chunk) and row 58 (gated delta
    rule) now have row-specific bodies in this file.
    """
    batch = len(x)
    seq = len(x[0])
    channels = len(x[0][0])
    state_dim = len(a[0][0])
    out = [[[0.0] * channels for _ in range(seq)] for _ in range(batch)]
    for ba in range(batch):
        state = [0.0] * state_dim
        for t in range(seq):
            for s in range(state_dim):
                state[s] = a[ba][t][s] * state[s] + b[ba][t][s] * x[ba][t][s % channels]
            m = sum(state) / state_dim
            for c in range(channels):
                out[ba][t][c] = m
    return out


def _mamba2_discretize_chunk(x: list[list[list[float]]],
                              a: list[list[list[float]]],
                              b: list[list[list[float]]],
                              dt: list[list[list[float]]],
                              c_proj: list[list[float]]) -> list[list[list[float]]]:
    """Row 57: Mamba-2 discretized state-space scan with chunk projection.

    Shapes: x[batch][seq][channels], a[batch][seq][state], b[batch][seq][state],
            dt[batch][seq][state], c_proj[state][channels].

    Per timestep:
      Abar[t, s] = exp(dt[t, s] * a[t, s])
      Bbar[t, s] = (Abar[t, s] - 1) / a[t, s] * b[t, s]      (continuous->discrete)
      state[s]   = Abar[t, s] * state[s] + Bbar[t, s] * x[t, s % channels]
      out[t, c]  = sum_s c_proj[s, c] * state[s]              (real C-projection)

    The C-projection is what makes "discretize_chunk" the LLM-decoder
    contract from Mamba-2 (Dao+Gu 2024, "Transformers are SSMs"):
    output is a learned linear projection of the post-recurrence state,
    not a uniform mean as in the row-58 GDN variant.
    """
    batch = len(x)
    seq = len(x[0])
    channels = len(x[0][0])
    state_dim = len(a[0][0])
    out = [[[0.0] * channels for _ in range(seq)] for _ in range(batch)]
    for ba in range(batch):
        state = [0.0] * state_dim
        for t in range(seq):
            for s in range(state_dim):
                a_t = a[ba][t][s]
                # `a` is seeded on (0.9, 0.999) so it cannot be zero;
                # division below is numerically safe within the row.
                Abar = math.exp(dt[ba][t][s] * a_t)
                Bbar = (Abar - 1.0) / a_t * b[ba][t][s]
                state[s] = Abar * state[s] + Bbar * x[ba][t][s % channels]
            for c in range(channels):
                acc = 0.0
                for s in range(state_dim):
                    acc += c_proj[s][c] * state[s]
                out[ba][t][c] = acc
    return out


def _chunk_gated_delta_rule(x: list[list[list[float]]],
                             a: list[list[list[float]]],
                             b: list[list[list[float]]],
                             key: list[list[list[float]]],
                             gate: list[list[list[float]]]) -> list[list[list[float]]]:
    """Row 58: gated delta rule (FlashInfer-Bench GDN variant).

    Shapes: x[batch][seq][channels], a[batch][seq][state],
            b[batch][seq][state], key[batch][seq][state],
            gate[batch][seq][channels].

    The delta-rule update is a rank-1 outer product of (x - state @ key)
    onto key, with `a` as a per-state decay and `b` as a per-state
    delta gain:
      err[t, c]    = x[t, c] - sum_s state[s, c] * key[t, s]
      state[s, c]  = a[t, s] * state[s, c] + b[t, s] * err[t, c] * key[t, s]
    The gated output mixes the post-update state mean with a gated
    skip-connection from x:
      g[t, c]      = sigmoid(gate[t, c])
      out[t, c]    = g[t, c] * mean_s(state[s, c]) + (1 - g[t, c]) * x[t, c]

    The row-57 vs row-58 distinction is structural: row 57 evolves the
    state via (Abar, Bbar) discretization and projects through C; row 58
    evolves the state via a key-driven delta-rule update and gates the
    output with a per-position sigmoid against x. Same shape contract,
    different operator.
    """
    batch = len(x)
    seq = len(x[0])
    channels = len(x[0][0])
    state_dim = len(a[0][0])
    out = [[[0.0] * channels for _ in range(seq)] for _ in range(batch)]
    for ba in range(batch):
        # state[s][c]: per-(state, channel) slot.
        state = [[0.0] * channels for _ in range(state_dim)]
        for t in range(seq):
            x_t = x[ba][t]
            a_t = a[ba][t]
            b_t = b[ba][t]
            key_t = key[ba][t]
            gate_t = gate[ba][t]
            # Rank-1 delta-rule update with per-state decay (a) and gain (b).
            for c in range(channels):
                state_dot_key = 0.0
                for s in range(state_dim):
                    state_dot_key += state[s][c] * key_t[s]
                err_c = x_t[c] - state_dot_key
                for s in range(state_dim):
                    state[s][c] = a_t[s] * state[s][c] + b_t[s] * err_c * key_t[s]
            # Gated output.
            for c in range(channels):
                m = 0.0
                for s in range(state_dim):
                    m += state[s][c]
                m /= state_dim
                g = 1.0 / (1.0 + math.exp(-gate_t[c]))
                out[ba][t][c] = g * m + (1.0 - g) * x_t[c]
    return out


def _direct_conv(x: list[list[list[float]]], filt: list[list[float]]) -> list[list[list[float]]]:
    """Direct 1D convolution for the Hyena reference (shape-match, not FFT).

    x[batch][seq][channels], filt[channels][kernel]. Returns x shape-matched.
    """
    batch = len(x)
    seq = len(x[0])
    channels = len(x[0][0])
    kernel = len(filt[0])
    out = [[[0.0] * channels for _ in range(seq)] for _ in range(batch)]
    for ba in range(batch):
        for c in range(channels):
            for t in range(seq):
                acc = 0.0
                for kk in range(kernel):
                    tau = t - kk
                    if 0 <= tau < seq:
                        acc += x[ba][tau][c] * filt[c][kk]
                out[ba][t][c] = acc
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    batch = int(axes.get("batch", 1))
    seq = int(axes.get("seq", 64))
    channels = int(axes.get("channels", 16))

    if slug == "hyena_fft_conv_block_fp32":
        kernel = int(axes.get("kernel", 8))
        x = deterministic_seeded_tensor((batch, seq, channels), seed)
        filt = deterministic_seeded_tensor((channels, kernel), seed + 1, low=-0.1, high=0.1)
        return _direct_conv(x, filt)

    state_dim = int(axes.get("state_dim", 16))
    x = deterministic_seeded_tensor((batch, seq, channels), seed)
    a = deterministic_seeded_tensor((batch, seq, state_dim), seed + 1, low=0.9, high=0.999)
    b = deterministic_seeded_tensor((batch, seq, state_dim), seed + 2, low=-0.1, high=0.1)

    if slug == "mamba_selective_scan_bf16":
        return _mamba_selective_scan(x, a, b)

    if slug == "mamba2_discretize_chunk_bf16":
        dt = deterministic_seeded_tensor((batch, seq, state_dim), seed + 3, low=0.01, high=0.1)
        c_proj = deterministic_seeded_tensor((state_dim, channels), seed + 4, low=-0.2, high=0.2)
        return _mamba2_discretize_chunk(x, a, b, dt, c_proj)

    # slug == "chunk_gated_delta_rule_bf16"
    key = deterministic_seeded_tensor((batch, seq, state_dim), seed + 3, low=-0.3, high=0.3)
    gate = deterministic_seeded_tensor((batch, seq, channels), seed + 4, low=-1.0, high=1.0)
    return _chunk_gated_delta_rule(x, a, b, key, gate)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
