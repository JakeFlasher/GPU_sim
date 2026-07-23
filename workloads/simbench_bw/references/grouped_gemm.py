"""SimBench-BW Grouped GEMM reference family.

MoE router + top-k + grouped-GEMM family. The reference runs the top-k
softmax routing, expands tokens into a per-expert permutation via the
resulting expert_ids, then runs one FP32 GEMM per expert and un-permutes
back to the original token ordering. All shapes are deterministic under
the given seed.

R30 extends coverage for the L4 pipeline row
`moe_layer_residual_norm_pipeline` end-to-end: the base router + grouped
GEMM + combine is followed by a residual add against the original input
(requires `hidden == intermediate` for this slug) and a per-token
RMSNorm with a seeded per-dim gain. A sibling helper
`build_l4_checkpoint()` returns the canonical router-to-grouped-gemm
boundary checkpoint matching
`schemas/l4_checkpoints/moe_layer_residual_norm_pipeline.schema.json`
(digests for tensor-valued fields; full values for `expert_offsets`).
"""

from __future__ import annotations

import math
from typing import Any

from _common import (
    compare_with_tolerance,
    deterministic_seeded_tensor,
    sha256_bf16_flat,
    sha256_float32_flat,
    sha256_int32_flat,
)


FAMILY_NAME = "grouped_gemm"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "moe_router_topk_softmax",
    "moe_grouped_gemm_bf16",
    "moe_grouped_gemm_fp8",
    "nvfp4_grouped_gemm_moe",
    "moe_layer_residual_norm_pipeline",
)


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, k = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k == k2
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            aip = a[i][p]
            for j in range(n):
                out[i][j] += aip * b[p][j]
    return out


def _router_topk(logits: list[list[float]], top_k: int) -> tuple[list[list[int]], list[list[float]]]:
    out_idx: list[list[int]] = []
    out_w: list[list[float]] = []
    for row in logits:
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        s = sum(exps)
        probs = [e / s for e in exps]
        idx = sorted(range(len(row)), key=lambda i: (-probs[i], i))[:top_k]
        weights = [probs[i] for i in idx]
        # Re-normalize top-k weights to sum to 1 (standard MoE convention)
        wsum = sum(weights)
        weights = [w / wsum for w in weights]
        out_idx.append(idx)
        out_w.append(weights)
    return out_idx, out_w


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    tokens = int(axes.get("tokens", 64))
    experts = int(axes.get("experts", 4))
    top_k = int(axes.get("top_k", 2))

    logits = deterministic_seeded_tensor((tokens, experts), seed, low=-2.0, high=2.0)
    expert_ids, routing_weights = _router_topk(logits, top_k)

    if slug == "moe_router_topk_softmax":
        return {"expert_ids": expert_ids, "routing_weights": routing_weights}

    hidden = int(axes.get("hidden", 32))
    intermediate = int(axes.get("intermediate", 32))
    x = deterministic_seeded_tensor((tokens, hidden), seed + 1, low=-0.5, high=0.5)
    # Per-expert weight matrix; shared shape across experts for the reference.
    w_experts = [
        deterministic_seeded_tensor((hidden, intermediate), seed + 100 + e, low=-0.1, high=0.1)
        for e in range(experts)
    ]
    # Combine: out[t][c] = sum_{k in top_k} routing_weights[t][k] * (x[t] @ w_experts[expert_ids[t][k]])[c]
    out = [[0.0] * intermediate for _ in range(tokens)]
    for t in range(tokens):
        row = [x[t]]
        for k_slot, eid in enumerate(expert_ids[t]):
            partial = _matmul(row, w_experts[eid])[0]
            w = routing_weights[t][k_slot]
            for c in range(intermediate):
                out[t][c] += w * partial[c]

    if slug == "moe_layer_residual_norm_pipeline":
        # Residual add requires the expert projection to land back in
        # the same dim as `x`. The row's shape file enforces
        # `hidden == intermediate`; refuse if it does not so the
        # contract cannot silently drift.
        if hidden != intermediate:
            raise ValueError(
                f"moe_layer_residual_norm_pipeline requires hidden == intermediate "
                f"for the post-expert residual add (got hidden={hidden}, intermediate={intermediate})"
            )
        # Residual: out += x.
        for t in range(tokens):
            for c in range(hidden):
                out[t][c] += x[t][c]
        # Per-token RMSNorm over the hidden dim with a seeded per-dim gain.
        gamma = deterministic_seeded_tensor((hidden,), seed + 200, low=-0.5, high=0.5)
        eps = 1e-5
        for t in range(tokens):
            ss = 0.0
            for v in out[t]:
                ss += v * v
            rstd = 1.0 / math.sqrt(ss / hidden + eps)
            for c in range(hidden):
                out[t][c] = out[t][c] * rstd * gamma[c]

    return out


# -----------------------------------------------------------------------------
# L4 checkpoint emission (R30, AC23). R33 hoisted the digest
# helpers into `_common.py` so all three L4 reference families share one
# implementation; this file now imports rather than privately duplicates.
# `fp32_to_bf16_bytes` is imported for potential per-call inspection /
# testing; the actual digesting goes through the sha256_* helpers.
# -----------------------------------------------------------------------------


def build_l4_checkpoint(slug: str, shape_point: dict[str, Any], seed: int) -> dict:
    """Return the canonical L4 component-checkpoint dict for `slug`
    matching `schemas/l4_checkpoints/<slug>.schema.json`.

    Currently only `moe_layer_residual_norm_pipeline` is supported;
    other L4 slugs (prefill_then_decode, speculative_decode) will add
    their own branches in future drift-recovery rounds.
    """
    if slug != "moe_layer_residual_norm_pipeline":
        raise ValueError(
            f"{slug!r} is not an L4 pipeline slug with a checkpoint schema that "
            f"this family implements (R30 covers only moe_layer_residual_norm_pipeline)"
        )
    axes = shape_point["axes"]
    tokens = int(axes.get("tokens", 64))
    experts = int(axes.get("experts", 4))
    top_k = int(axes.get("top_k", 2))
    hidden = int(axes.get("hidden", 32))

    # Reproduce the router stage deterministically with the same seed.
    logits = deterministic_seeded_tensor((tokens, experts), seed, low=-2.0, high=2.0)
    expert_ids, routing_weights = _router_topk(logits, top_k)
    x = deterministic_seeded_tensor((tokens, hidden), seed + 1, low=-0.5, high=0.5)

    # Permute tokens by destination expert: stable sort of (eid, t, k_slot)
    # triples places every copy heading to expert 0 first, then expert 1,
    # etc. permuted_tokens[i, :] = x[t] where (t, k_slot) was sorted to i.
    pairs: list[tuple[int, int, int]] = []
    counts = [0] * experts
    for t in range(tokens):
        for k_slot in range(top_k):
            eid = expert_ids[t][k_slot]
            pairs.append((eid, t, k_slot))
            counts[eid] += 1
    pairs.sort(key=lambda p: (p[0], p[1], p[2]))
    permuted_tokens = [x[t] for (_eid, t, _k_slot) in pairs]

    # expert_offsets: prefix-sum vector of length experts+1. offsets[0]=0,
    # offsets[-1] = num_tokens * top_k.
    expert_offsets: list[int] = [0]
    acc = 0
    for e in range(experts):
        acc += counts[e]
        expert_offsets.append(acc)
    assert expert_offsets[-1] == tokens * top_k

    # Flatten tensors for digest.
    expert_ids_flat = [eid for row in expert_ids for eid in row]
    permuted_flat = [v for row in permuted_tokens for v in row]
    routing_weights_flat = [w for row in routing_weights for w in row]

    return {
        "schema_version": 1,
        "slug": "moe_layer_residual_norm_pipeline",
        "checkpoint_name": "router_to_grouped_gemm_boundary",
        "expert_ids_per_token": {
            "shape": [tokens, top_k],
            "dtype": "int32",
            "digest": sha256_int32_flat(expert_ids_flat),
        },
        "permuted_tokens": {
            "shape": [tokens * top_k, hidden],
            "dtype": "bfloat16",
            "digest": sha256_bf16_flat(permuted_flat),
        },
        "expert_offsets": {
            "shape": [experts + 1],
            "dtype": "int32",
            "values": expert_offsets,
        },
        "routing_weights": {
            "shape": [tokens, top_k],
            "dtype": "float32",
            "digest": sha256_float32_flat(routing_weights_flat),
        },
    }


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    if isinstance(actual, dict) and "expert_ids" in actual:
        return (
            actual.get("expert_ids") == reference.get("expert_ids")
            and compare_with_tolerance(actual["routing_weights"], reference["routing_weights"], tolerance=tolerance)
        )
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
