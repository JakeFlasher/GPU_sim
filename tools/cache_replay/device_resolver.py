"""Knowable-L2-capacity resolver for the cache-replay predictor.

Per the plan, the replay engine must reuse the existing device-property /
cache-model surface for the knowable L2 capacity rather than carry a
cache-replay-local constant. This helper routes the sm_120 RTX 5060 Laptop L2
capacity through:

  * ``run_suite._static_card_l2_bytes_for_props`` (the live-suite static-card
    resolver: compute capability 12.x -> 32 MiB), and
  * ``tools.hrd_profiler.cache_model_spec.resolve_cache_model_spec`` (the
    cache-model resolver that turns a capacity input into a fully-populated
    spec with capacity / geometry / slice sources and evidence).

It returns both the capacity in bytes and a provenance block (capacity_source,
geometry_source, num_l2_slices_source, the resolver identity, and the resolver
evidence) so every emitted artifact can record that the capacity came from the
shared resolver, not a bare local literal.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_TOOLS_HRD = _REPO_ROOT / "tools" / "hrd_profiler"
_SIMBENCH = _REPO_ROOT / "workloads" / "simbench_bw"
for _p in (_TOOLS_HRD, _SIMBENCH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cache_model_spec as _cms  # type: ignore[import-not-found]  # noqa: E402

# RTX 5060 Laptop is Blackwell GB206, compute capability 12.0.
SM120_COMPUTE_CAPABILITY = (12, 0)
RESOLVER_IDENTITY = (
    "tools.hrd_profiler.cache_model_spec.resolve_cache_model_spec"
    " + run_suite static card"
)


def _static_card_l2_bytes() -> int | None:
    """The sm_120 L2 capacity from the live-suite static-card resolver. A tiny
    shim object suffices: the function reads only compute_capability_{major,minor}."""
    import run_suite  # type: ignore[import-not-found]

    shim = types.SimpleNamespace(
        compute_capability_major=SM120_COMPUTE_CAPABILITY[0],
        compute_capability_minor=SM120_COMPUTE_CAPABILITY[1],
    )
    return run_suite._static_card_l2_bytes_for_props(shim)


def _static_card_num_l2_slices() -> int | None:
    import run_suite  # type: ignore[import-not-found]

    shim = types.SimpleNamespace(
        compute_capability_major=SM120_COMPUTE_CAPABILITY[0],
        compute_capability_minor=SM120_COMPUTE_CAPABILITY[1],
    )
    return run_suite._static_card_num_l2_slices_for_props(shim)


def resolve_sm120_l2() -> tuple[int, dict[str, Any]]:
    """Resolve the knowable sm_120 L2 capacity (bytes) + a provenance block.

    The capacity is the cache-model resolver's output (fed the static-card
    value); it is never a bare cache-replay-local literal. ``capacity_source``
    is ``"static_device_card"`` (not ``"constant"``), proving the value flowed
    through the shared resolver.
    """
    static_card_l2 = _static_card_l2_bytes()
    static_card_slices = _static_card_num_l2_slices()
    spec, evidence = _cms.resolve_cache_model_spec(
        cuda_l2_cache_size_bytes=None,
        static_card_l2_cache_size_bytes=static_card_l2,
        static_card_num_l2_slices=static_card_slices,
    )
    capacity_bytes = int(spec["capacity_bytes"])
    provenance = {
        "capacity_bytes": capacity_bytes,
        "capacity_source": spec["capacity_source"],
        "geometry_source": spec["geometry_source"],
        "num_l2_slices_source": spec["num_l2_slices_source"],
        "sector_size_bytes": spec["sector_size_bytes"],
        "line_size_bytes": spec["line_size_bytes"],
        "compute_capability": (
            f"{SM120_COMPUTE_CAPABILITY[0]}.{SM120_COMPUTE_CAPABILITY[1]}"
        ),
        "resolver": RESOLVER_IDENTITY,
        "resolver_evidence": evidence,
    }
    return capacity_bytes, provenance
