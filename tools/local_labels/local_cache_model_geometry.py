"""Shared resolver for the local-device cache-model geometry consumed by
the v5 predictor producers.

Two producers feed ``predict_cache_hit_rate``: the headline cohort runner
(``run_headline_cohort_predict_compare.py``) and the per-cell fresh-regen
bundle builder (``build_predictor_bundle_for_cell.py``). Both MUST predict
under the SAME cache model, otherwise a fresh-regen bundle is computed
under different geometry than the trace sidecars / headline runner and can
be admitted with wrong predictions/provenance.

The geometry is sourced from the device-properties manifest entry
(L2 capacity, L2 slice count, CTA-interleave model) rather than hard-coded
constants, so every emitted bundle records the exact geometry it was
predicted under. Only ``sm_86`` (the local RTX 3090 Ti) is populated this
loop; a non-sm_86 arch raises ``UnsupportedLocalCacheModelGeometryArch`` so
the caller fails closed with a typed reason rather than silently falling
back to the walker's single-slice / interleave-off / second-layer-off
default. The sm_120 (RTX 5060 laptop) manifest key is the sm_120 session's
to add when that lane develops; this loop is sm_86-develop-only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
# ``device_properties`` lives under tools/hrd_profiler/; mirror the
# headline runner's import path so the in-process resolution succeeds.
sys.path.insert(0, str(_REPO_ROOT / "tools" / "hrd_profiler"))
import device_properties  # noqa: E402  # type: ignore[import-not-found]

# 5-tuple key for the local sm_86 RTX 3090 Ti device-properties manifest
# entry (mirrors run_headline_cohort_predict_compare._LOCAL_MANIFEST_KEY;
# the cross-producer divergence guard test asserts the two stay equal).
SM86_LOCAL_MANIFEST_KEY: dict[str, str] = {
    "arch": "sm_86",
    "sku": "rtx_3090_ti",
    "driver_version": "595.71.05",
    "cuda_toolkit_version": "13.2",
    "nvbit_build": "1.8",
}

# Coarse CTA/SM interleaving is enabled on sm_86 (the walker's back-compat
# default is False); the HRD second layer is likewise enabled. The
# manifest's ``cta_interleave_model`` must agree (round_robin_per_sm when
# True; sequential_per_tb when False) or resolution fails closed.
_SM86_CTA_INTERLEAVE_ENABLED = True
_SM86_HRD_SECOND_LAYER_ENABLED = True


class UnsupportedLocalCacheModelGeometryArch(Exception):
    """Raised when local cache-model geometry is requested for an arch
    without a populated device-properties manifest key. Only sm_86 is
    populated this loop; callers fail closed with a typed reason instead
    of using the walker's single-slice fallback."""


@dataclass(frozen=True)
class CacheModelGeometry:
    """The resolved cache-model geometry + provenance for one local device.

    ``entry`` carries the authoritative ``l2_capacity_bytes`` and
    ``num_l2_slices``; ``provenance`` is the dict every emitted bundle
    records under ``cache_model_geometry_provenance``.
    """

    entry: "device_properties.DevicePropertiesEntry"
    provenance: dict
    cta_interleave_enabled: bool
    hrd_second_layer_enabled: bool


def resolve_local_cache_model_geometry_for_arch(
    arch: str,
) -> CacheModelGeometry:
    """Resolve the local device-properties cache-model geometry for ``arch``.

    Raises ``UnsupportedLocalCacheModelGeometryArch`` for any arch other
    than sm_86, and ``RuntimeError`` when the manifest's
    ``cta_interleave_model`` disagrees with the producer's interleave
    setting (a downstream consumer would otherwise see a bundle whose
    top-level interleave flag contradicts its own provenance block)."""
    if arch != "sm_86":
        raise UnsupportedLocalCacheModelGeometryArch(arch)
    entry = device_properties.resolve_entry(**SM86_LOCAL_MANIFEST_KEY)
    expected_model = (
        "round_robin_per_sm"
        if _SM86_CTA_INTERLEAVE_ENABLED
        else "sequential_per_tb"
    )
    if entry.cta_interleave_model != expected_model:
        raise RuntimeError(
            f"device-properties manifest cta_interleave_model="
            f"{entry.cta_interleave_model!r} disagrees with producer "
            f"setting cache_model_cta_interleave_enabled="
            f"{_SM86_CTA_INTERLEAVE_ENABLED!r} (expected manifest value: "
            f"{expected_model!r})."
        )
    provenance = entry.to_geometry_provenance(
        num_l2_slices_source="device_properties_manifest",
    )
    return CacheModelGeometry(
        entry=entry,
        provenance=provenance,
        cta_interleave_enabled=_SM86_CTA_INTERLEAVE_ENABLED,
        hrd_second_layer_enabled=_SM86_HRD_SECOND_LAYER_ENABLED,
    )
