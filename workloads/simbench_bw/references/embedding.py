"""SimBench-BW Embedding reference family.

Vocab-scale gather / bag-scatter references.

- `embedding_gather_vocab128k`: vocab-scale gather.
  For each (b, l) in [batch] x [indices], output[b, l, :] = E[indices[b, l], :].
  Pure per-output-element lookup from an embedding table E[V, D].

- `embedding_bag_scatter_atomic`: bag-sum scatter (EmbeddingBag sum mode).
  For each b, sum E[indices[b, i], :] over i in [0, bag_size) into output[b, :].
  Models the bag-style embedding where many indices collapse per sample,
  accumulated with atomic add on the kernel side.

Seeded inputs (same MWC PRNG as host drivers via
`_common.deterministic_seeded_tensor`):
    E       = seed        : (vocab_size, hidden)          # embedding table
    indices = seed + 1    : (batch, indices | bag_size)    # lookup indices

All float values are FP32 references (BF16 I/O on the kernel side); the
tolerance band absorbs BF16 quantization per `tolerances.toml`
`(BF16, warp_shuffle_sum, embedding)` for gather and
`(BF16, atomic_add, embedding)` for bag-scatter.
"""

from __future__ import annotations

from typing import Any

from _common import compare_with_tolerance, deterministic_seeded_tensor


FAMILY_NAME = "embedding"

# Declares that every slug in this family uses a pure-Python CPU
# reference path. See `verify_coverage.py::check_reference_type_runtime`
# — a roster row with `reference_type != "cpu"` under this family would
# fail the hook. Both embedding rows are already `reference_type = "cpu"`
# in roster.toml.
REFERENCE_BACKEND: str = "cpu"

FAMILY_COVERS_SLUGS: tuple[str, ...] = (
    "embedding_gather_vocab128k",
    "embedding_bag_scatter_atomic",
)


# Default vocab size when a shape point omits `vocab_size`. The row's slug
# advertises "vocab128k"; shape points that need to exercise the full
# 128k-row table pass `vocab_size = 131072` explicitly. Smaller
# correctness / trace shapes use a smaller default to keep the reference
# evaluation tractable in pure-Python.
_DEFAULT_VOCAB_SIZE = 128


def _build_indices(batch: int, width: int, vocab_size: int, seed: int) -> list[list[int]]:
    """Generate `(batch, width)` integer indices into [0, vocab_size) using
    the same MWC PRNG as the host drivers (seed*2654435761+1 init;
    state = state*1103515245 + 12345). Indices are sampled uniformly in
    [0, vocab_size). Shared by the gather and the bag-scatter references."""
    state = (seed * 2654435761 + 1) & 0xFFFFFFFF
    indices: list[list[int]] = []
    for _b in range(batch):
        row: list[int] = []
        for _w in range(width):
            state = (state * 1103515245 + 12345) & 0xFFFFFFFF
            row.append(state % max(1, vocab_size))
        indices.append(row)
    return indices


def _gather_reference(axes: dict[str, Any], seed: int) -> list:
    """[B, L, D] = E[indices[B, L], D]."""
    batch = int(axes["batch"])
    width = int(axes.get("indices", axes.get("bag_size", 1)))
    hidden = int(axes["hidden"])
    vocab_size = int(axes.get("vocab_size", _DEFAULT_VOCAB_SIZE))

    table = deterministic_seeded_tensor((vocab_size, hidden), seed, low=-0.5, high=0.5)
    indices = _build_indices(batch, width, vocab_size, seed + 1)

    out: list = [[[0.0] * hidden for _ in range(width)] for _ in range(batch)]
    for b in range(batch):
        for w in range(width):
            row = table[indices[b][w]]
            out_row = out[b][w]
            for d in range(hidden):
                out_row[d] = row[d]
    return out


def _bag_scatter_reference(axes: dict[str, Any], seed: int) -> list:
    """[B, D] = sum_{i in bag_size} E[indices[B, i], D].

    Reference is a deterministic Python summation; the kernel is atomic-add
    (deterministic=ND) and the tolerance band absorbs the ordering
    difference. The reference is the "clean" sum that the non-deterministic
    kernel must match within tolerance."""
    batch = int(axes["batch"])
    bag_size = int(axes.get("bag_size", axes.get("indices", 1)))
    hidden = int(axes["hidden"])
    vocab_size = int(axes.get("vocab_size", _DEFAULT_VOCAB_SIZE))

    table = deterministic_seeded_tensor((vocab_size, hidden), seed, low=-0.5, high=0.5)
    indices = _build_indices(batch, bag_size, vocab_size, seed + 1)

    out: list = [[0.0] * hidden for _ in range(batch)]
    for b in range(batch):
        out_row = out[b]
        for i in range(bag_size):
            table_row = table[indices[b][i]]
            for d in range(hidden):
                out_row[d] += table_row[d]
    return out


def build_reference(slug: str, shape_point: dict[str, Any], seed: int) -> Any:
    """FP32 reference for `embedding_gather_vocab128k` (gather) and
    `embedding_bag_scatter_atomic` (bag-sum scatter). Dispatches on slug;
    raises ValueError for any slug this family does not cover."""
    if slug not in FAMILY_COVERS_SLUGS:
        raise ValueError(f"{slug!r} is not covered by {FAMILY_NAME!r}")
    axes = shape_point["axes"]
    if slug == "embedding_gather_vocab128k":
        return _gather_reference(axes, seed)
    # slug == "embedding_bag_scatter_atomic"
    return _bag_scatter_reference(axes, seed)


def compare(actual: Any, reference: Any, tolerance: dict) -> bool:
    """Tolerance-aware allclose resolved via the row's tolerances.toml entry.
    `(BF16, warp_shuffle_sum, embedding)` for gather; `(BF16, atomic_add,
    embedding)` for bag-scatter."""
    return compare_with_tolerance(actual, reference, tolerance=tolerance)
