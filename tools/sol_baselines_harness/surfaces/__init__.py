"""Counter-free analytics surfaces over the ``sol-execbench`` evaluator.

This package layers analytic surfaces over every *.cu CUTLASS /
CuTe C++ template solution under ``baselines/``:

1. ``static`` — ``cuobjdump`` / ``nvdisasm`` / ``ptxas -v`` reports.
2. ``sanitizer`` — ``compute-sanitizer`` memcheck / racecheck /
   synccheck / initcheck.
3. ``nvbit_metrics`` — counter-free per-launch metric extractor: one
   compact JSON-L summary per accepted kernel launch covering the
   metrics NCU would otherwise provide (instruction-class histogram
   with TCGen05 / TMA / DSMEM separated, global LDG/STG sector
   efficiency, shared-memory bank conflicts, branch active-mask
   histogram). Wall-clock attribution is owned by ``cupti_activity``
   Surface 2 and joined post-hoc on a CUPTI-observable composite key.
4. ``cupti_activity`` — CUPTI Activity wall-clock attribution per
   kernel launch and memcpy phase via an env-gated extension to the
   upstream evaluator's existing in-process callback.
5. ``nsys`` — Nsight Systems CUDA / API timeline with optional NVTX
   ranges.
6. ``cache_prediction`` — counter-free L1 / L2 read-only sector
   hit-rate predictor surface. Registered passive: the plugin
   reports ``unavailable`` while no per-launch prediction sidecar
   producer is wired into the bundle. The surface is NOT
   eval-driving — it never spawns the upstream evaluator and
   never imports NCU. The CLI's stage presets exclude this
   surface; explicit `--surface cache_prediction` selects it.

The architecture is a **Plugin Orchestrator Monolith**: a ``Surface``
ABC + six concrete plugin classes register against a
``SurfaceOrchestrator`` that drives them in dependency order via
native subprocess invocation. The orchestrator emits one
``manifest.json`` per ``(slug, language)`` bundle alongside the
existing per-workload ``trace.json``.

Per CLAUDE.md §"Counter-Free Analytics Framework", the framework is
designed to migrate intact from the local RTX 3090 Ti (sm_86) to a
non-root B200 provider VM (sm_100a), with a future RTX 5060
(sm_120) cross-arch calibration host. Dispatch is single-shape and
native end-to-end — the upstream evaluator runs as a host subprocess
with no container runtime in the loop.
"""

from sol_baselines_harness.surfaces.base import (  # noqa: F401
    HostInfo,
    Surface,
    SurfaceParseResult,
    SurfaceProbeResult,
    SurfaceRunResult,
    SURFACE_STATUSES,
    SURFACE_STATUS_BLOCKED,
    SURFACE_STATUS_FAILED,
    SURFACE_STATUS_OK,
    SURFACE_STATUS_SKIPPED,
    SURFACE_STATUS_UNAVAILABLE,
    VALID_SURFACES,
)
from sol_baselines_harness.surfaces.manifest import (  # noqa: F401
    ManifestBuilder,
    ManifestLintError,
    lint_manifest,
)
from sol_baselines_harness.surfaces.orchestrator import (  # noqa: F401
    SurfaceOrchestrator,
    UnknownSurfaceError,
    assert_b200_fanout_ready,
)
from sol_baselines_harness.surfaces.cache_prediction import (  # noqa: F401
    CachePredictionSurface,
)
from sol_baselines_harness.surfaces.cache_prediction_sidecar import (  # noqa: F401
    CACHE_PREDICTION_SIDECAR_BASENAME,
    CACHE_PREDICTION_SIDECAR_SUBDIR,
    CACHE_PREDICTION_SIDECAR_TYPED_REASONS,
    CachePredictionSidecarError,
    attach_sidecar_to_surface_entry,
    read_cache_prediction_sidecar,
    write_cache_prediction_sidecar,
)
from sol_baselines_harness.surfaces.cupti_activity import (  # noqa: F401
    CUPTIActivitySurface,
)
from sol_baselines_harness.surfaces.nsys import NsysSurface  # noqa: F401
from sol_baselines_harness.surfaces.nvbit_metrics import (  # noqa: F401
    NVBitMetricsSurface,
)
from sol_baselines_harness.surfaces.sanitizer import SanitizerSurface  # noqa: F401
from sol_baselines_harness.surfaces.static import StaticSurface  # noqa: F401


def default_surface_plugins() -> dict:
    """Return the canonical ``{surface_name: plugin_instance}`` mapping.

    Used by the orchestrator's CLI and by tests that need every
    plugin instantiated against the locked vocabulary. Subclassing or
    swapping any plugin requires the caller to assemble its own dict
    and pass it to ``SurfaceOrchestrator(surface_plugins=...)``.
    """
    return {
        "static": StaticSurface(),
        "sanitizer": SanitizerSurface(),
        "nvbit_metrics": NVBitMetricsSurface(),
        "cupti_activity": CUPTIActivitySurface(),
        "nsys": NsysSurface(),
        "cache_prediction": CachePredictionSurface(),
    }


__all__ = [
    "CACHE_PREDICTION_SIDECAR_BASENAME",
    "CACHE_PREDICTION_SIDECAR_SUBDIR",
    "CACHE_PREDICTION_SIDECAR_TYPED_REASONS",
    "CachePredictionSidecarError",
    "CachePredictionSurface",
    "CUPTIActivitySurface",
    "HostInfo",
    "ManifestBuilder",
    "ManifestLintError",
    "NsysSurface",
    "NVBitMetricsSurface",
    "SURFACE_STATUSES",
    "SURFACE_STATUS_BLOCKED",
    "SURFACE_STATUS_FAILED",
    "SURFACE_STATUS_OK",
    "SURFACE_STATUS_SKIPPED",
    "SURFACE_STATUS_UNAVAILABLE",
    "SanitizerSurface",
    "StaticSurface",
    "Surface",
    "SurfaceOrchestrator",
    "SurfaceParseResult",
    "SurfaceProbeResult",
    "SurfaceRunResult",
    "UnknownSurfaceError",
    "VALID_SURFACES",
    "assert_b200_fanout_ready",
    "attach_sidecar_to_surface_entry",
    "default_surface_plugins",
    "lint_manifest",
    "read_cache_prediction_sidecar",
    "write_cache_prediction_sidecar",
]
