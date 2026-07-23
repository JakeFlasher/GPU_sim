"""SimBench-BW Sampling reference family.

Top-k + temperature softmax sampling. Reference is deterministic top-k + FP32
softmax on CPU. Given a fixed seed and a fixed logit tensor, the output is a
sorted top-k (index, probability) list.
"""

from __future__ import annotations

import math
from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "sampling"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "sampling_topk_temperature_bf16",
)


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    batch = int(axes["batch"])
    vocab = int(axes["vocab"])
    top_k = int(axes["top_k"])
    temperature = float(axes.get("temperature", 1.0))
    logits = deterministic_seeded_tensor((batch, vocab), seed, low=-4.0, high=4.0)

    out_indices: list[list[int]] = []
    out_probs: list[list[float]] = []
    for row in logits:
        scaled = [x / temperature for x in row]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        s = sum(exps)
        probs = [e / s for e in exps]
        idx_sorted = sorted(range(vocab), key=lambda i: (-probs[i], i))[:top_k]
        out_indices.append(idx_sorted)
        out_probs.append([probs[i] for i in idx_sorted])
    return {"indices": out_indices, "probabilities": out_probs}


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    if actual.get("indices") != reference.get("indices"):
        return False
    return compare_with_tolerance(actual["probabilities"], reference["probabilities"], tolerance=tolerance)
