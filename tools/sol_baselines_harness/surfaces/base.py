"""``Surface`` ABC and the result dataclasses surface plugins return.

Each surface plugin implements three abstract methods:

* ``probe(host_info)`` — read-only host capability check. Returns a
  ``SurfaceProbeResult`` whose ``status`` is one of the locked
  vocabulary (``ok | unavailable | blocked | failed | skipped``).
* ``run(baseline_dir, problem_dir, bundle_dir, build_mode, ...)`` —
  invoke the surface against one ``(slug, language)`` baseline.
  Returns a ``SurfaceRunResult`` capturing the launched command,
  exit code, stdout/stderr tails, and start/end UTC timestamps.
* ``parse(bundle_dir)`` — validate the on-disk artifact the ``run``
  step emitted. Returns a ``SurfaceParseResult`` carrying the final
  status, artifact path, and per-artifact / per-surface byte sizes.

Each plugin also declares three class-level identity fields used by
the orchestrator's cache-identity tuple:

* ``name`` — short surface name (one of ``VALID_SURFACES``).
* ``schema_version`` — version of the manifest schema this plugin
  emits. Bumped when the per-surface manifest entry shape changes.
* ``impl_version`` — version of the plugin implementation. Bumped
  when behavior changes in a way that should invalidate cached
  artifacts (e.g., a tool flag change).

The plugins return their tool version (e.g., ``nsys --version``
output) from ``probe()``; the orchestrator records it on the
manifest entry so a tool upgrade invalidates only that surface's
cache, not the whole bundle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SURFACE_STATUS_OK = "ok"
SURFACE_STATUS_UNAVAILABLE = "unavailable"
SURFACE_STATUS_BLOCKED = "blocked"
SURFACE_STATUS_FAILED = "failed"
SURFACE_STATUS_SKIPPED = "skipped"
# AD-35 required-prefer status: probe / dep imported but does not meet
# the throughput / availability floor; downstream falls back to a
# slower / lower-quality path without blocking. Used by the B200
# preflight required-prefer probes; runtime surfaces emit it only
# when they have a documented graceful-degrade path (none currently
# do — surfaces stick to ok/unavailable/blocked/failed/skipped).
SURFACE_STATUS_DEGRADED = "degraded"

SURFACE_STATUSES = frozenset(
    {
        SURFACE_STATUS_OK,
        SURFACE_STATUS_UNAVAILABLE,
        SURFACE_STATUS_BLOCKED,
        SURFACE_STATUS_FAILED,
        SURFACE_STATUS_SKIPPED,
        SURFACE_STATUS_DEGRADED,
    }
)

VALID_SURFACES = frozenset(
    {
        "static",
        "sanitizer",
        "nvbit_metrics",
        "cupti_activity",
        "nsys",
        "cache_prediction",
    }
)


@dataclass(frozen=True)
class HostInfo:
    """Host environment fingerprint passed into every ``probe()`` call.

    Populated once by the orchestrator at startup from a combination
    of ``nvidia-smi`` device-identity reads (read-only),
    ``nvcc --version``, and ``cudaRuntimeGetVersion``. The ``kind``
    field encodes the
    operator's intent and lands in the manifest's ``host.kind`` slot:

    * ``local_sm_8_6`` — local RTX 3090 Ti (Phase 1 smoke target).
    * ``local_sm_120`` — an SM120 cloud host. The active target is the
      RTX PRO 6000 Workstation; historical RTX 5060 artifacts remain
      explicitly labeled and are not substituted for it.
    * ``b200_sm_100a`` — non-root B200 provider (Phase 2 fanout).
    * ``unknown`` — host probe could not classify; surfaces should
      still attempt their probes and fall back to ``unavailable``
      when toolchain pieces are missing.
    """

    kind: str
    device_name: str
    compute_capability: str
    driver_version: str
    cuda_runtime_version: str
    captured_at_utc: str
    nvtx_header_present: bool = False
    nvtx_library_present: bool = False


@dataclass
class SurfaceProbeResult:
    """Result of a surface's ``probe()`` call."""

    status: str
    tool_version: Optional[str] = None
    waiver_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurfaceRunResult:
    """Result of a surface's ``run()`` call."""

    status: str
    command: List[str] = field(default_factory=list)
    exit_code: int = 0
    stdout_tail: str = ""
    stderr_tail: str = ""
    start_utc: str = ""
    end_utc: str = ""
    waiver_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurfaceParseResult:
    """Result of a surface's ``parse()`` call."""

    status: str
    artifact_path: Optional[str] = None
    waiver_reason: Optional[str] = None
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    substatus: Dict[str, Any] = field(default_factory=dict)


class Surface(ABC):
    """Abstract base for the six counter-free analytic surfaces.

    Concrete subclasses MUST set the three class attributes
    (``name``, ``schema_version``, ``impl_version``) and implement
    ``probe``, ``run``, and ``parse``. Subclasses missing any of the
    three abstract methods fail to instantiate with a ``TypeError``
    raised by ``abc.ABCMeta`` — the orchestrator's manifest lint
    relies on this strictness to reject an incomplete plugin before
    a run begins.
    """

    name: str = ""
    schema_version: str = ""
    impl_version: str = ""

    @abstractmethod
    def probe(self, host_info: HostInfo) -> SurfaceProbeResult:
        """Read-only host capability check.

        MUST NOT modify host state (no clock locks, no power-policy
        changes, no driver mutation). MUST be cheap (target < 1 s).
        """

    @abstractmethod
    def run(
        self,
        baseline_dir: Path,
        problem_dir: Path,
        bundle_dir: Path,
        build_mode: str,
        workload_subset: List[int],
        dispatch_command: Optional[List[str]] = None,
        plan: Optional[Any] = None,
    ) -> SurfaceRunResult:
        """Invoke the surface against one ``(slug, language)`` baseline.

        When ``workload_subset`` carries multiple indices,
        the surface may emit one artifact per index OR one rolled-up
        artifact, at its discretion; the manifest records which.

        Parameters
        ----------
        baseline_dir
            Path to the landed baseline (``baselines/<cat>/<slug>/<lang>/``).
        problem_dir
            Path to the matching benchmark spec
            (``tools/sol_execbench/data/benchmark/<subset>/<NNN_name>/``).
        bundle_dir
            Target ``.harness_traces/<slug>__<lang>/`` directory; the
            surface writes its artifacts under
            ``bundle_dir / <surface.name>/``.
        build_mode
            ``"judge"`` or ``"profiling"`` — affects the staged
            ``solution.json`` the surface dispatches against.
        workload_subset
            List of workload indices to evaluate. Default is the
            lowest-index workload only. The surface treats
            this as authoritative; ``--full-fanout`` produces the
            full list.
        dispatch_command
            When supplied, the orchestrator has already constructed
            the launch command (e.g., for surface groups that share
            a single eval invocation); the surface appends its own
            flags / env overrides. When ``None``, the surface
            constructs its own command.
        """

    @abstractmethod
    def parse(self, bundle_dir: Path) -> SurfaceParseResult:
        """Validate the on-disk artifact the ``run`` step emitted.

        MUST be deterministic and side-effect-free apart from sizing
        the artifact file. The orchestrator's manifest lint depends
        on the returned ``status`` / ``artifact_path`` /
        ``waiver_reason`` invariants .
        """

    def parse_on_failure(
        self,
        bundle_dir: Path,
        run_result: SurfaceRunResult,
    ) -> Optional[SurfaceParseResult]:
        """Optional hook for surfaces whose tracer writes typed
        diagnostics under ``bundle_dir`` even when the eval driver
        exits non-zero. Returns ``None`` (default) to keep the
        orchestrator's existing legacy mapping (run failure ->
        SurfaceParseResult(status=run.status, waiver=run.waiver));
        returns a populated ``SurfaceParseResult`` to overlay the
        tracer-typed waiver onto the manifest entry.

        The NVBit surface uses this hook to honor the AD-21 failure
        precedence: when the tracer wrote ``_invalid_regex.flag`` and
        ``_exit(1)``'d, the subprocess returns non-zero but the
        operator-actionable waiver lives in the flag file, not in
        the eval driver's stderr. Calling ``parse_nvbit_artifacts``
        from this hook lifts that typed waiver onto the manifest.

        The orchestrator preserves the original ``exit_code`` /
        ``stdout_tail`` / ``stderr_tail`` regardless of what this
        hook returns; only ``status`` / ``waiver_reason`` /
        ``substatus`` are overlaid.
        """
        return None
