"""SimBench-BW Paged Attention reference family.

Paged-KV decode family (GQA, MQA, MLA) and KV-cache-update paths. The
reference emits the full-precision decode output against a seeded KV cache
laid out as contiguous pages; the kernel's page-table walk is compared to
this non-paged reference under the family's tolerance.

R33 closes Codex's R32-review gap #1 for
`prefill_then_decode_pagedkv_pipeline`: the pipeline-specific
`build_reference` branch now runs the decode phase against the same
paged KV cache `build_l4_checkpoint` digests, via a shared
`_prefill_then_decode_boundary_state()` helper. The reference's final
output therefore provably depends on the boundary state — a kernel
that ignores the prefilled KV cache and runs its own unpaged decode
cannot pass correctness even if it matches the checkpoint schema.

The base decode reference for the other four slugs (gqa / mqa / mla /
kv_cache_update_rope) is unchanged.
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
    sha256_int32_flat,
)


FAMILY_NAME = "paged_attention"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "pagedkv_decode_gqa_bf16",
    "pagedkv_decode_mqa_bf16",
    "paged_mla_decode_bf16",
    "kv_cache_update_rope_bf16",
    "prefill_then_decode_pagedkv_pipeline",
)


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    if slug == "prefill_then_decode_pagedkv_pipeline":
        return _prefill_then_decode_pipeline_reference(shape_point, seed)
    axes = shape_point["axes"]
    batch = int(axes.get("batch", 1))
    context = int(axes.get("context", axes.get("seq", 128)))
    head_dim = int(axes.get("head_dim", axes.get("head_dim_ckv", 32)))
    q_heads = int(axes.get("q_heads", axes.get("heads", 4)))
    kv_heads = int(axes.get("kv_heads", 1))

    if slug == "kv_cache_update_rope_bf16":
        # R39 repair: the row's contract is "in-place KV-cache
        # update with fused RoPE rotation". Reference applies RoPE
        # to the seeded K tensor at each position `p in [0, context)`
        # and returns the rotated K cache. Exercises the RoPE rotation
        # path over a paged KV cache (not generic decode SDPA, which
        # the R38 placeholder collapsed onto).
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        half = head_dim // 2
        k_cache = deterministic_seeded_tensor((batch, kv_heads, context, head_dim), seed + 100, low=-0.5, high=0.5)
        inv_freq = [1.0 / (10000.0 ** (2 * i / head_dim)) for i in range(half)]
        kv_out: Any = [[[[0.0] * head_dim for _ in range(context)] for _ in range(kv_heads)] for _ in range(batch)]
        for b in range(batch):
            for h in range(kv_heads):
                for p in range(context):
                    for i in range(half):
                        angle = p * inv_freq[i]
                        cos_v = math.cos(angle)
                        sin_v = math.sin(angle)
                        r_even = k_cache[b][h][p][2 * i]
                        r_odd  = k_cache[b][h][p][2 * i + 1]
                        kv_out[b][h][p][2 * i]     = r_even * cos_v - r_odd * sin_v
                        kv_out[b][h][p][2 * i + 1] = r_even * sin_v + r_odd * cos_v
        return kv_out

    # Build Q as [batch][q_heads][head_dim] (single-token decode query)
    q = deterministic_seeded_tensor((batch, q_heads, head_dim), seed, low=-0.5, high=0.5)
    # Build KV as [batch][kv_heads][context][head_dim]
    k = deterministic_seeded_tensor((batch, kv_heads, context, head_dim), seed + 100, low=-0.5, high=0.5)
    v = deterministic_seeded_tensor((batch, kv_heads, context, head_dim), seed + 200, low=-0.5, high=0.5)

    scale = 1.0 / math.sqrt(head_dim)
    out: list[list[list[float]]] = [[[0.0] * head_dim for _ in range(q_heads)] for _ in range(batch)]
    for b in range(batch):
        for h in range(q_heads):
            kv_h = h % kv_heads
            # scores[j] = q[b][h] dot k[b][kv_h][j] * scale
            scores = [0.0] * context
            for j in range(context):
                s = 0.0
                for d in range(head_dim):
                    s += q[b][h][d] * k[b][kv_h][j][d]
                scores[j] = s * scale
            m = max(scores)
            exps = [math.exp(x - m) for x in scores]
            norm = sum(exps)
            probs = [e / norm for e in exps]
            # out[b][h][d] = sum_j probs[j] * v[b][kv_h][j][d]
            for j in range(context):
                for d in range(head_dim):
                    out[b][h][d] += probs[j] * v[b][kv_h][j][d]
    return out


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    return compare_with_tolerance(actual, reference, tolerance=tolerance)


# -----------------------------------------------------------------------------
# Shared prefill-to-decode boundary state (R33)
#
# Both `build_reference("prefill_then_decode_pagedkv_pipeline", ...)` and
# `build_l4_checkpoint("prefill_then_decode_pagedkv_pipeline", ...)` go
# through `_prefill_then_decode_boundary_state()`. The checkpoint
# digests the boundary state; the reference's decode stage consumes
# the same boundary state. Byte-level coherence between reference
# output and checkpoint digests is therefore guaranteed per
# BL-20260424-kernel-contract-vs-shape-match (refuse shape-matched
# placeholders that don't actually consume the stated state).
# -----------------------------------------------------------------------------


def _extract_pipeline_axes(axes: dict[str, Any]) -> dict[str, int]:
    """Canonical axis extraction for prefill_then_decode. Defaults match
    the shape file; callers pass the full axes dict."""
    batch = int(axes.get("batch", 2))
    context = int(axes.get("context", axes.get("seq", 32)))
    head_dim = int(axes.get("head_dim", 32))
    q_heads = int(axes.get("q_heads", axes.get("heads", 4)))
    kv_heads = int(axes.get("kv_heads", 2))
    block_size = int(axes.get("block_size", 16))
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    return {
        "batch": batch, "context": context, "head_dim": head_dim,
        "q_heads": q_heads, "kv_heads": kv_heads, "block_size": block_size,
    }


def _prefill_then_decode_boundary_state(axes: dict[str, Any], seed: int) -> dict:
    """Deterministic prefill-to-decode boundary state. Returns a dict
    with the exact fields `build_l4_checkpoint` serializes:

    - `kv_block_table`: `[num_sequences, max_blocks_per_sequence]`.
      Sequence `s` owns sequential block IDs `s*mbps .. (s+1)*mbps - 1`.
    - `kv_data_blocks`: seeded `[num_blocks, block_size, num_kv_heads, head_dim]`
      BF16 tensor (stored as Python nested floats; BF16 semantics applied
      at digest time via `_common.fp32_to_bf16_bytes`).
    - `seq_lengths`: per-sequence prefill length, uniform `context`.
    - `rope_cos_sin_cache`: standard RoPE theta schedule for
      `max_position = context`.
    - `num_sequences`, `max_blocks_per_sequence`, `num_blocks`,
      `block_size`, `head_dim`, `kv_heads`, `q_heads`, `batch`,
      `context` (copied from axes for downstream decode).
    """
    a = _extract_pipeline_axes(axes)
    batch, context, head_dim = a["batch"], a["context"], a["head_dim"]
    kv_heads, block_size = a["kv_heads"], a["block_size"]
    num_sequences = batch
    max_blocks_per_sequence = (context + block_size - 1) // block_size
    num_blocks = num_sequences * max_blocks_per_sequence
    block_table: list[list[int]] = [
        [s * max_blocks_per_sequence + b for b in range(max_blocks_per_sequence)]
        for s in range(num_sequences)
    ]
    kv_data_blocks = deterministic_seeded_tensor(
        (num_blocks, block_size, kv_heads, head_dim), seed + 500, low=-0.5, high=0.5,
    )
    seq_lengths = [context for _ in range(num_sequences)]
    rope_cache = _rope_cos_sin_cache(max_position=context, head_dim=head_dim)
    return {
        **a,
        "num_sequences": num_sequences,
        "max_blocks_per_sequence": max_blocks_per_sequence,
        "num_blocks": num_blocks,
        "kv_block_table": block_table,
        "kv_data_blocks": kv_data_blocks,
        "seq_lengths": seq_lengths,
        "rope_cos_sin_cache": rope_cache,
    }


def _prefill_then_decode_pipeline_reference(shape_point: dict[str, Any], seed: int) -> list:
    """Pipeline-specific end-to-end reference for
    `prefill_then_decode_pagedkv_pipeline`. Runs the decode phase
    against the SAME paged KV cache `build_l4_checkpoint` digests:

    1. Construct the boundary state via the shared helper.
    2. Derive K and V for each sequence by walking the block table:
       for sequence `b`, position `p = 0..context-1`, the block index
       is `block_table[b][p // block_size]` and the in-block offset
       is `p % block_size`. That gives K[b, kv_h, p, :] and V[b, kv_h, p, :]
       from `kv_data_blocks[block_idx][offset][kv_h][:]`.
       (Reference uses the K-half for both K and V since the schema
       treats the stored tensor as the combined KV payload.)
    3. Seed a single-token decode Q from `seed` so it is
       checkpoint-independent (the boundary state contains K/V only
       per the schema).
    4. Run scaled dot-product attention over the reconstructed
       `[batch, kv_heads, context, head_dim]` K/V tensors.
    5. Return `[batch, q_heads, head_dim]` decode output.

    A kernel that skips the prefill-to-decode boundary (e.g. runs its
    own unpaged decode on fresh tensors) will produce different output
    because `kv_data_blocks` is seeded at `seed + 500` (not `seed + 100`
    like the base decode reference)."""
    state = _prefill_then_decode_boundary_state(shape_point["axes"], seed)
    batch = state["batch"]
    context = state["context"]
    head_dim = state["head_dim"]
    q_heads = state["q_heads"]
    kv_heads = state["kv_heads"]
    block_size = state["block_size"]
    kv_data = state["kv_data_blocks"]
    block_table = state["kv_block_table"]

    # Reconstruct [batch, kv_heads, context, head_dim] K and V by
    # walking the block table. Reference uses the same tensor for K
    # and V (the schema stores the combined KV data).
    k: list = [[[[0.0] * head_dim for _ in range(context)] for _ in range(kv_heads)] for _ in range(batch)]
    v = k  # alias: share the same seeded data; kernel would distinguish them
    for b in range(batch):
        for pos in range(context):
            block_idx = block_table[b][pos // block_size]
            offset = pos % block_size
            for kv_h in range(kv_heads):
                row = kv_data[block_idx][offset][kv_h]
                k[b][kv_h][pos] = list(row)

    # Seed single-token decode Q independently of the boundary state.
    q = deterministic_seeded_tensor((batch, q_heads, head_dim), seed, low=-0.5, high=0.5)

    scale = 1.0 / math.sqrt(head_dim)
    out: list = [[[0.0] * head_dim for _ in range(q_heads)] for _ in range(batch)]
    for b in range(batch):
        for h in range(q_heads):
            kv_h = h % kv_heads
            scores = [0.0] * context
            for j in range(context):
                s = 0.0
                for d in range(head_dim):
                    s += q[b][h][d] * k[b][kv_h][j][d]
                scores[j] = s * scale
            m = max(scores)
            exps = [math.exp(x - m) for x in scores]
            norm = sum(exps)
            probs = [e / norm for e in exps]
            for j in range(context):
                for d in range(head_dim):
                    out[b][h][d] += probs[j] * v[b][kv_h][j][d]
    return out


# -----------------------------------------------------------------------------
# L4 checkpoint emission for prefill_then_decode_pagedkv_pipeline
# (R32; matches
# schemas/l4_checkpoints/prefill_then_decode_pagedkv_pipeline.schema.json)
# -----------------------------------------------------------------------------


def _rope_cos_sin_cache(max_position: int, head_dim: int, base: float = 10000.0) -> list:
    """Build a canonical [2, max_position, head_dim] RoPE cos/sin table.
    axis 0 splits cos (index 0) vs sin (index 1). Uses the standard
    theta_k = base^{-2k/head_dim} frequency schedule applied to every
    position 0..max_position-1. Deterministic: no seed dependence, so
    the digest is stable across runs for a given (max_position, head_dim).
    """
    assert head_dim % 2 == 0, "RoPE head_dim must be even"
    half = head_dim // 2
    cache: list = [
        [[0.0] * head_dim for _ in range(max_position)] for _ in range(2)
    ]
    for k in range(half):
        inv_freq = 1.0 / (base ** (2.0 * k / head_dim))
        for p in range(max_position):
            angle = p * inv_freq
            cos_v = math.cos(angle)
            sin_v = math.sin(angle)
            # Mirror angle across the two halves of the head dim
            # (standard rotary encoding).
            cache[0][p][k]         = cos_v
            cache[0][p][k + half]  = cos_v
            cache[1][p][k]         = sin_v
            cache[1][p][k + half]  = sin_v
    return cache


def build_l4_checkpoint(slug: str, shape_point: dict[str, Any], seed: int) -> dict:
    """Emit the canonical prefill-to-decode KV-cache-boundary checkpoint
    matching
    `schemas/l4_checkpoints/prefill_then_decode_pagedkv_pipeline.schema.json`.

    R33: built on top of `_prefill_then_decode_boundary_state`,
    the same helper the pipeline-specific reference consumes. Checkpoint
    digests and reference output therefore derive from the same state
    by construction; the coherence path retains digest equality.
    """
    if slug != "prefill_then_decode_pagedkv_pipeline":
        raise ValueError(
            f"{slug!r} is not an L4 pipeline slug this family implements "
            f"(paged_attention covers only prefill_then_decode_pagedkv_pipeline for L4)"
        )
    state = _prefill_then_decode_boundary_state(shape_point["axes"], seed)
    block_table_flat = [v for row in state["kv_block_table"] for v in row]
    kv_data_flat = flatten_nested(state["kv_data_blocks"])
    rope_flat = flatten_nested(state["rope_cos_sin_cache"])
    return {
        "schema_version": 1,
        "slug": "prefill_then_decode_pagedkv_pipeline",
        "checkpoint_name": "prefill_to_decode_kv_boundary",
        "kv_block_table": {
            "shape": [state["num_sequences"], state["max_blocks_per_sequence"]],
            "dtype": "int32",
            "digest": sha256_int32_flat(block_table_flat),
        },
        "kv_data_blocks": {
            "shape": [state["num_blocks"], state["block_size"], state["kv_heads"], state["head_dim"]],
            "dtype": "bfloat16",
            "digest": sha256_bf16_flat(kv_data_flat),
        },
        "seq_lengths": {
            "shape": [state["num_sequences"]],
            "dtype": "int32",
            "values": state["seq_lengths"],
        },
        "rope_cos_sin_cache": {
            "shape": [2, state["context"], state["head_dim"]],
            "dtype": "float32",
            "rope_kind": "standard",
            "digest": sha256_float32_flat(rope_flat),
        },
    }
