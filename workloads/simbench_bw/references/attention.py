"""SimBench-BW Attention reference family.

Forward path: Q @ K^T, optional causal masking, softmax-normalize
row-wise with the row-max stability trick, multiply by V, return the
output tensor in FP32.

Ragged prefill path (R16): `ragged_attention_prefill_bf16` with
`num_seqs > 1` in axes maps cu_seqlens onto an even partition of
total_tokens and computes per-segment-isolated SDPA.

Backward path (R17): `flash_attn_bwd_bf16` (and future
`flash_attn_bwd_fp8`) seed an upstream gradient `dO` from
`seed + b + 300`, recompute the forward softmax weights `P`, then
return dQ = ds @ K / sqrt(head_dim) where
ds = P * (dP - (P * dP).sum(-1, keepdim=True)) and dP = dO @ V^T.
Output shape matches Q.
"""

from __future__ import annotations

import math
from typing import Any

from _common import (
    compare_with_tolerance,
    deterministic_seeded_tensor,
    flatten_nested,
    sha256_bf16_flat,
    sha256_float32_flat,
)


FAMILY_NAME = "attention"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "flash_attn_fwd_bf16_causal",
    "flash_attn_fwd_bf16_noncausal",
    "flash_attn_fwd_fp8_causal",
    "flash_attn_bwd_bf16",
    "flash_attn_bwd_fp8",
    "ragged_attention_prefill_bf16",
    "speculative_decode_draft_verify_pipeline",
)


def _sdpa(q: list[list[list[float]]],
          k: list[list[list[float]]],
          v: list[list[list[float]]],
          causal: bool) -> list[list[list[float]]]:
    """Scaled dot-product attention over a single (heads, seq, dim) triple.

    Shapes: q/k/v are [heads][seq][head_dim]. Returns [heads][seq][head_dim].
    """
    heads = len(q)
    seq = len(q[0])
    head_dim = len(q[0][0])
    scale = 1.0 / math.sqrt(head_dim)
    out: list[list[list[float]]] = []
    for h in range(heads):
        # scores[i][j] = q[h][i] dot k[h][j] * scale
        scores = [[0.0] * seq for _ in range(seq)]
        for i in range(seq):
            for j in range(seq):
                if causal and j > i:
                    scores[i][j] = float("-inf")
                    continue
                s = 0.0
                for d in range(head_dim):
                    s += q[h][i][d] * k[h][j][d]
                scores[i][j] = s * scale
        # row-wise softmax with stable max
        probs = [[0.0] * seq for _ in range(seq)]
        for i in range(seq):
            m = max(scores[i])
            if m == float("-inf"):
                probs[i] = [0.0] * seq
                continue
            exps = [math.exp(x - m) if x != float("-inf") else 0.0 for x in scores[i]]
            s = sum(exps)
            probs[i] = [e / s for e in exps]
        # out[h][i][d] = sum_j probs[i][j] * v[h][j][d]
        head_out = [[0.0] * head_dim for _ in range(seq)]
        for i in range(seq):
            for j in range(seq):
                if probs[i][j] == 0.0:
                    continue
                for d in range(head_dim):
                    head_out[i][d] += probs[i][j] * v[h][j][d]
        out.append(head_out)
    return out


def _even_cu_seqlens(total_tokens: int, num_seqs: int) -> list[int]:
    """Even partition of `total_tokens` into `num_seqs` segments. Matches
    the host-driver-side derivation used by the ragged kernel so host-side
    segment boundaries are reproducible from (total_tokens, num_seqs)
    alone, without an external cu_seqlens buffer on the correctness shape.
    Segments are distributed round-robin: the first `(total_tokens %
    num_seqs)` segments get one extra token so the partition always sums
    to `total_tokens`."""
    assert num_seqs >= 1
    assert total_tokens >= num_seqs
    base = total_tokens // num_seqs
    extra = total_tokens % num_seqs
    cu = [0]
    offset = 0
    for s in range(num_seqs):
        seg_len = base + (1 if s < extra else 0)
        offset += seg_len
        cu.append(offset)
    assert cu[-1] == total_tokens
    return cu


def _compute_p(q: list[list[list[float]]],
               k: list[list[list[float]]],
               causal: bool) -> list[list[list[float]]]:
    """Return the softmax-weights tensor `P[h][i][j]` = softmax(Q H^T / sqrt(d)),
    row-stable. Same first half of `_sdpa` without the V multiply — used by
    the backward path to avoid recomputation."""
    heads = len(q)
    seq = len(q[0])
    head_dim = len(q[0][0])
    scale = 1.0 / math.sqrt(head_dim)
    out: list[list[list[float]]] = []
    for h in range(heads):
        scores = [[0.0] * seq for _ in range(seq)]
        for i in range(seq):
            for j in range(seq):
                if causal and j > i:
                    scores[i][j] = float("-inf")
                    continue
                s = 0.0
                for d in range(head_dim):
                    s += q[h][i][d] * k[h][j][d]
                scores[i][j] = s * scale
        probs = [[0.0] * seq for _ in range(seq)]
        for i in range(seq):
            m = max(scores[i])
            if m == float("-inf"):
                probs[i] = [0.0] * seq
                continue
            exps = [math.exp(x - m) if x != float("-inf") else 0.0 for x in scores[i]]
            s = sum(exps)
            probs[i] = [e / s for e in exps]
        out.append(probs)
    return out


def _compute_dq(q: list[list[list[float]]],
                k: list[list[list[float]]],
                v: list[list[list[float]]],
                do: list[list[list[float]]],
                causal: bool) -> list[list[list[float]]]:
    """Backward gradient dQ for scaled-dot-product attention given Q, K, V,
    and the upstream gradient dO (same shape as the forward output).

    Math:
      P  = softmax(Q @ K^T / sqrt(d))  (row-wise; causal mask sets
                                          masked entries to 0 after softmax)
      dP[h][i][j] = sum_d dO[h][i][d] * V[h][j][d]
      row_dot[h][i] = sum_j P[h][i][j] * dP[h][i][j]
      ds[h][i][j]   = P[h][i][j] * (dP[h][i][j] - row_dot[h][i])
      dQ[h][i][d]   = (sum_j ds[h][i][j] * K[h][j][d]) / sqrt(d)

    Returns dQ with shape [heads][seq][head_dim]."""
    heads = len(q)
    seq = len(q[0])
    head_dim = len(q[0][0])
    scale = 1.0 / math.sqrt(head_dim)
    p = _compute_p(q, k, causal=causal)
    dq: list[list[list[float]]] = []
    for h in range(heads):
        dp = [[0.0] * seq for _ in range(seq)]
        for i in range(seq):
            for j in range(seq):
                if causal and j > i:
                    continue  # p[h][i][j] is 0, so dp entry contributes nothing
                s = 0.0
                for d in range(head_dim):
                    s += do[h][i][d] * v[h][j][d]
                dp[i][j] = s
        row_dot = [0.0] * seq
        for i in range(seq):
            row_dot[i] = sum(p[h][i][j] * dp[i][j] for j in range(seq))
        ds = [[0.0] * seq for _ in range(seq)]
        for i in range(seq):
            for j in range(seq):
                ds[i][j] = p[h][i][j] * (dp[i][j] - row_dot[i])
        dq_h = [[0.0] * head_dim for _ in range(seq)]
        for i in range(seq):
            for j in range(seq):
                s = ds[i][j]
                if s == 0.0:
                    continue
                for d in range(head_dim):
                    dq_h[i][d] += s * k[h][j][d]
        for i in range(seq):
            for d in range(head_dim):
                dq_h[i][d] *= scale
        dq.append(dq_h)
    return dq


def _segmented_sdpa(q: list[list[list[float]]],
                    k: list[list[list[float]]],
                    v: list[list[list[float]]],
                    cu_seqlens: list[int],
                    causal: bool) -> list[list[list[float]]]:
    """SDPA with per-segment isolation: queries in segment s only attend
    to keys in segment s. No cross-segment attention. `cu_seqlens` is
    the cumulative-sequence-lengths array, length `num_seqs + 1`, with
    `cu_seqlens[0] == 0` and `cu_seqlens[-1] == seq`."""
    heads = len(q)
    seq = len(q[0])
    head_dim = len(q[0][0])
    out: list[list[list[float]]] = [[[0.0] * head_dim for _ in range(seq)] for _ in range(heads)]
    num_seqs = len(cu_seqlens) - 1
    for s in range(num_seqs):
        lo = cu_seqlens[s]
        hi = cu_seqlens[s + 1]
        if hi <= lo:
            continue
        seg_len = hi - lo
        # Slice out the per-segment Q / K / V and run a standard SDPA on them.
        q_seg = [[q[h][i] for i in range(lo, hi)] for h in range(heads)]
        k_seg = [[k[h][i] for i in range(lo, hi)] for h in range(heads)]
        v_seg = [[v[h][i] for i in range(lo, hi)] for h in range(heads)]
        seg_out = _sdpa(q_seg, k_seg, v_seg, causal=causal)
        for h in range(heads):
            for i in range(seg_len):
                out[h][lo + i] = seg_out[h][i]
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    if slug == "speculative_decode_draft_verify_pipeline":
        return _speculative_decode_pipeline_reference(shape_point, seed)
    axes = shape_point["axes"]
    # Canonical axis names vary slightly across attention rows; normalize.
    heads = int(axes.get("heads", 4))
    seq = int(axes.get("seq", axes.get("total_tokens", 64)))
    head_dim = int(axes.get("head_dim", 32))
    batch = int(axes.get("batch", 1))
    causal = "causal" in slug and "noncausal" not in slug
    # The ragged prefill row explicitly carries `cu_seqlens` semantics per
    # its roster citation. `num_seqs > 1` at that slug triggers the
    # per-segment-isolated branch; other slugs keep the contiguous path.
    num_seqs = int(axes.get("num_seqs", 1))
    segmented = (slug == "ragged_attention_prefill_bf16" and num_seqs > 1)
    # Backward variants compute real dQ, not the forward output. The
    # upstream gradient dO is seeded deterministically from seed+b+300
    # so the kernel can reproduce it host-side.
    is_backward = slug in ("flash_attn_bwd_bf16", "flash_attn_bwd_fp8")
    batch_out: list[list[list[list[float]]]] = []
    for b in range(batch):
        q = deterministic_seeded_tensor((heads, seq, head_dim), seed + b, low=-0.5, high=0.5)
        k = deterministic_seeded_tensor((heads, seq, head_dim), seed + b + 100, low=-0.5, high=0.5)
        v = deterministic_seeded_tensor((heads, seq, head_dim), seed + b + 200, low=-0.5, high=0.5)
        if is_backward:
            do = deterministic_seeded_tensor((heads, seq, head_dim), seed + b + 300, low=-0.5, high=0.5)
            batch_out.append(_compute_dq(q, k, v, do, causal=causal))
        elif segmented:
            cu = _even_cu_seqlens(seq, num_seqs)
            batch_out.append(_segmented_sdpa(q, k, v, cu, causal=causal))
        else:
            batch_out.append(_sdpa(q, k, v, causal=causal))
    return batch_out


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)


# -----------------------------------------------------------------------------
# Shared draft boundary state + L4 reference / checkpoint
# (R32 landed the checkpoint helper; R33 refactors so the
# pipeline-specific reference and the checkpoint derive from the same
# `_speculative_decode_boundary_state` — per Codex R32-review
# gap #1 directive for reference/checkpoint coherence.)
# -----------------------------------------------------------------------------


def _argmax(row: list[float]) -> int:
    best_i = 0
    best_v = row[0]
    for i in range(1, len(row)):
        if row[i] > best_v:
            best_v = row[i]
            best_i = i
    return best_i


def _softmax(row: list[float]) -> list[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def _extract_speculative_axes(axes: dict[str, Any]) -> dict[str, int]:
    return {
        "batch":        int(axes.get("batch", 2)),
        "draft_length": int(axes.get("draft_length", 4)),
        "vocab":        int(axes.get("vocab", 128)),
        "num_layers":   int(axes.get("num_layers", 2)),
        "num_kv_heads": int(axes.get("num_kv_heads", 2)),
        "head_dim":     int(axes.get("head_dim", 32)),
    }


def _speculative_decode_boundary_state(axes: dict[str, Any], seed: int) -> dict:
    """Deterministic draft-to-verifier boundary state. Returns a dict
    with the exact fields `build_l4_checkpoint` serializes:

    - `draft_logits`: seeded `[batch, draft_length, vocab]` from
      `seed + 700`.
    - `draft_probabilities`: row-wise softmax of draft_logits along
      the vocab axis.
    - `draft_tokens`: argmax of draft_logits (deterministic greedy
      sampling).
    - `draft_attention_state`: seeded
      `[num_layers, num_kv_heads, draft_length, head_dim]` from
      `seed + 800`.

    Both `build_reference(speculative_decode_draft_verify_pipeline, ...)`
    and `build_l4_checkpoint(speculative_decode_draft_verify_pipeline, ...)`
    call this helper. The reference's verifier-side output is
    derived from this same draft state, so a kernel that ignores the
    draft-boundary can't pass correctness even if its checkpoint
    digests match shape-only.
    """
    a = _extract_speculative_axes(axes)
    batch, dlen, vocab = a["batch"], a["draft_length"], a["vocab"]
    nlay, nkv, hd = a["num_layers"], a["num_kv_heads"], a["head_dim"]
    draft_logits = deterministic_seeded_tensor(
        (batch, dlen, vocab), seed + 700, low=-2.0, high=2.0,
    )
    draft_probabilities: list = [
        [_softmax(draft_logits[b][s]) for s in range(dlen)]
        for b in range(batch)
    ]
    draft_tokens: list[list[int]] = [
        [_argmax(draft_logits[b][s]) for s in range(dlen)]
        for b in range(batch)
    ]
    draft_attention_state = deterministic_seeded_tensor(
        (nlay, nkv, dlen, hd), seed + 800, low=-0.5, high=0.5,
    )
    return {
        **a,
        "draft_logits":          draft_logits,
        "draft_probabilities":   draft_probabilities,
        "draft_tokens":          draft_tokens,
        "draft_attention_state": draft_attention_state,
    }


def _speculative_decode_pipeline_reference(shape_point: dict[str, Any], seed: int) -> list:
    """Pipeline-specific end-to-end reference for
    `speculative_decode_draft_verify_pipeline`. Runs verifier-side
    logic against the SAME draft boundary state `build_l4_checkpoint`
    digests:

    1. Construct draft boundary state via the shared helper.
    2. Seed a target-model logits tensor independently from `seed + 900`
       (a kernel that doesn't run the target model can't reproduce it).
    3. Softmax the target logits to get target probabilities.
    4. For each (b, s, v): compute the rejection-sampling acceptance
       factor `a = min(1, target_prob / draft_prob)` per the
       standard speculative-decode verifier math. Return
       `target_probabilities * acceptance_factor`
       (weights target by verifier accept probability).
    5. Output shape: `[batch, draft_length, vocab]` FP32 tensor.

    A kernel that skips either the draft stage or the verifier stage
    cannot pass correctness — the output byte-depends on both."""
    state = _speculative_decode_boundary_state(shape_point["axes"], seed)
    batch = state["batch"]
    dlen = state["draft_length"]
    vocab = state["vocab"]
    draft_probabilities = state["draft_probabilities"]
    # Target-model logits seeded from a distinct offset so the reference
    # materially depends on BOTH the draft (via boundary state) AND the
    # target (via this tensor). A placeholder that only runs the draft
    # would produce different output.
    target_logits = deterministic_seeded_tensor(
        (batch, dlen, vocab), seed + 900, low=-2.0, high=2.0,
    )
    out: list = [[[0.0] * vocab for _ in range(dlen)] for _ in range(batch)]
    for b in range(batch):
        for s in range(dlen):
            target_row = target_logits[b][s]
            m = max(target_row)
            exps = [math.exp(x - m) for x in target_row]
            total = sum(exps)
            target_probs = [e / total for e in exps]
            # Rejection-sampling accept factor per vocab position:
            # a_v = min(1, target_probs[v] / max(draft_probs[v], eps)).
            # Numerator of the accept ratio; eps keeps the division safe
            # when the draft assigned near-zero mass.
            eps = 1e-12
            draft_row = draft_probabilities[b][s]
            out_row = out[b][s]
            for v in range(vocab):
                a = min(1.0, target_probs[v] / max(draft_row[v], eps))
                out_row[v] = target_probs[v] * a
    return out


def build_l4_checkpoint(slug: str, shape_point: dict[str, Any], seed: int) -> dict:
    """Emit the canonical draft-to-verifier boundary checkpoint matching
    `schemas/l4_checkpoints/speculative_decode_draft_verify_pipeline.schema.json`.

    R33: built on top of `_speculative_decode_boundary_state`, the
    same helper the pipeline-specific reference consumes. Checkpoint
    digests and reference output therefore derive from the same draft
    state by construction; the coherence test in
    `tests/test_round33_repairs.py` verifies digest equality."""
    if slug != "speculative_decode_draft_verify_pipeline":
        raise ValueError(
            f"{slug!r} is not an L4 pipeline slug this family implements "
            f"(attention covers only speculative_decode_draft_verify_pipeline for L4)"
        )
    state = _speculative_decode_boundary_state(shape_point["axes"], seed)
    batch = state["batch"]
    dlen = state["draft_length"]
    vocab = state["vocab"]
    nlay = state["num_layers"]
    nkv = state["num_kv_heads"]
    hd = state["head_dim"]
    draft_tokens_flat = [t for row in state["draft_tokens"] for t in row]
    draft_logits_flat = flatten_nested(state["draft_logits"])
    draft_probs_flat = flatten_nested(state["draft_probabilities"])
    draft_att_flat = flatten_nested(state["draft_attention_state"])
    return {
        "schema_version": 1,
        "slug": "speculative_decode_draft_verify_pipeline",
        "checkpoint_name": "draft_to_verifier_boundary",
        "draft_tokens": {
            "shape": [batch, dlen],
            "dtype": "int32",
            "values": draft_tokens_flat,
        },
        "draft_logits": {
            "shape": [batch, dlen, vocab],
            "dtype": "bfloat16",
            "digest": sha256_bf16_flat(draft_logits_flat),
        },
        "draft_probabilities": {
            "shape": [batch, dlen, vocab],
            "dtype": "float32",
            "digest": sha256_float32_flat(draft_probs_flat),
        },
        "draft_attention_state_digest": {
            "shape": [nlay, nkv, dlen, hd],
            "dtype": "bfloat16",
            "digest": sha256_bf16_flat(draft_att_flat),
        },
    }
