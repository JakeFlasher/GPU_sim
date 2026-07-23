"""``NVBitMetricsSurface`` - counter-free per-launch metrics via NVBit injection.

Injects the new on-device metric extractor
(``nvbit_tracer/metric_extractor/metric_extractor.so``) into the native
``sol-execbench`` evaluator subprocess via ``CUDA_INJECTION64_PATH``
(curated through ``build_subprocess_env``). Produces:

* ``nvbit_metrics/launches.jsonl``               - one compact JSON-L summary per accepted kernel launch.
* ``nvbit_metrics/filter_stats.json``            - per-process running counters from the user-module filter.

The metric extractor calls **zero** CUPTI APIs by construction - wall-clock
attribution lives entirely in the ``cupti_activity`` Surface 2 and is
joined post-hoc by report code on the cross-surface composite key the
JSON-L carries (``workload_index``, ``demangled_kernel_name``,
``kernel_ordinal_in_workload``, ``grid_xyz``, ``block_xyz``,
``dynamic_smem``, ``stream_ordinal``). This surface therefore cannot
collide with ``sol-execbench``'s in-process CUPTI subscriber at
``sol-execbench/src/sol_execbench/core/bench/timing.py:207-261``, but the
default ``methodology="cupti"`` it would otherwise install still talks
through CUPTI, which is rejected the moment NVBit's CUDA-callback
subscription holds the only allowed CUPTI subscriber slot
(``ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED (39)``). The fix is a
``sitecustomize.py`` shim that monkey-patches ``time_runnable`` to
default ``methodology="cuda_events"`` for the injected eval
subprocess only.

This surface is the canonical Surface 1 (counter-free metric
extractor). Injection lands via ``CUDA_INJECTION64_PATH``; the
orchestrator keeps the surface in its own dispatch group separate
from ``cupti_activity`` and ``nsys`` so a single bundle can run all
three as matched sequential invocations against the same baseline,
never as one in-process collision. The legacy Accel-Sim
trace-dumper this surface replaced was removed in the Scope-B
cleanup (2026-05-16); see project memory entry
``project_nvbit_tracer_replacement.md`` for the full decommission
log.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from baselines_harness.surfaces.base import (
    HostInfo,
    Surface,
    SurfaceParseResult,
    SurfaceProbeResult,
    SurfaceRunResult,
    SURFACE_STATUS_BLOCKED,
    SURFACE_STATUS_FAILED,
    SURFACE_STATUS_OK,
    SURFACE_STATUS_UNAVAILABLE,
)
from baselines_harness.surfaces.manifest import now_utc_iso
from baselines_harness.surfaces.profiling_build import reduced_config_overrides
from baselines_harness.surfaces.subprocess_env import (
    BUILD_TOOLCHAIN_ALLOWED,
    build_subprocess_env,
)

_NVBIT_METRICS_DIR = "nvbit_metrics"
_LAUNCHES_FILENAME = "launches.jsonl"
_FILTER_STATS_FILENAME = "filter_stats.json"

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
METRIC_EXTRACTOR_TOOL_PATH = (
    _REPO_ROOT / "nvbit_tracer" / "metric_extractor" / "metric_extractor.so"
)

# CUPTI surfaces forbidden alongside NVBit injection. Mirrors the legacy
# ``nvbit`` surface's check so the orchestrator-built argv cannot smuggle
# a privileged CUPTI flag in via ``--cupti-*`` from a future dispatch
# helper.
_FORBIDDEN_FLAGS = (
    "--cupti-events",
    "--cupti-metrics",
    "--cupti-pm-sampling",
    "--cupti-range",
    "--cupti-perfworks",
)

# Operator-tunable env vars the metric extractor honours that
# materially change *produced behavior*. These names participate in the
# cache-key allowlist via ``compute_env_filter_hash`` (per
# ``BL-20260515-cache-key-causal-coverage``: every operator-tunable
# input that changes produced behavior must enter the cache key or
# stale cache hits silently mask a re-run with new operator state).
#
# The names map 1:1 to ``getenv`` reads inside
# ``nvbit_tracer/metric_extractor/metric_extractor.cu``:
#
# * ``METRIC_EXTRACTOR_USER_MODULE_REGEX`` - regex matched against
#   ``/proc/self/maps`` entries; controls which loaded modules (user
#   ``.so`` vs torch / cuBLAS / cuDNN) get instrumented. Defaults to
#   ``.*sol_execbench.*`` when unset.
# * ``METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS`` - per-process slot cap on
#   ``LaunchMetrics``. Changes the producer's truncation point so the
#   cache key MUST track it.
#
# Crucially, the artifact-path env vars are NOT in this allowlist -
# they are harness-owned and unconditionally point at the bundle
# directory (see ``_HARNESS_OWNED_ARTIFACT_PATHS`` below). An operator
# value of ``METRIC_EXTRACTOR_LAUNCHES_PATH`` or
# ``METRIC_EXTRACTOR_FILTER_STATS_PATH`` MUST NOT escape the bundle -
# otherwise a leftover env-var from the operator's shell would silently
# redirect the JSON-L to a wrong location, the orchestrator would
# decide the surface is ``blocked`` (no artifact found at the
# bundle-local path), AND the operator-pointed artifact would be
# treated as fresh on the next cache-identity check.
NVBIT_METRICS_USER_ENV_VARS: Tuple[str, ...] = (
    "METRIC_EXTRACTOR_USER_MODULE_REGEX",
    "METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS",
    # Tier-2 address-trace producer env vars. Each value flips the
    # bytes the producer emits (sidecar enable/disable, per-lane
    # debug enable, per-launch budget), so every operator-tunable
    # knob enters the cache key.
    "METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP",
    "METRIC_EXTRACTOR_TIER2_PER_LANE",
    "METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES",
    # The 64-hex-char SHA-256 the producer uses for
    # LaunchHeaderV1.binary_sha256 + deterministic_sampling_seed
    # derivation. This env var is a TEST OVERRIDE only; the
    # production producer hashes the accepted module origin directly
    # on launch-enter (see tier2_load_binary_sha256_for_origin in
    # metric_extractor.cu). The var stays in the allowlist so that
    # operator/test-supplied overrides still invalidate cache identity
    # under BL-20260515-cache-key-causal-coverage; the surface does
    # not auto-set it.
    "METRIC_EXTRACTOR_TIER2_BINARY_SHA256_HEX",
    # Per-cell kernel-name regex filter at the producer's launch
    # acceptance gate. When set, only kernels whose demangled name
    # matches the regex are accepted; non-matching launches are
    # skipped before slot allocation, preserving the per-process
    # slot budget for the target kernel. Behaviour-changing producer
    # knob: setting it admits a different subset of launches into
    # the sidecar / launches.jsonl, so every cache-identity check
    # MUST track it.
    "METRIC_EXTRACTOR_KERNEL_NAME_REGEX",
)

# Harness-owned env vars. The plugin sets these unconditionally to
# bundle-local paths AFTER any operator-env merge so a stray operator
# value in the shell environment cannot redirect the artifacts out of
# the bundle. Listed for documentation only - the values come from
# ``_compose()``'s host_artifact_dir argument.
_HARNESS_OWNED_ARTIFACT_PATHS: Tuple[str, ...] = (
    "METRIC_EXTRACTOR_LAUNCHES_PATH",
    "METRIC_EXTRACTOR_FILTER_STATS_PATH",
)

# Auxiliary env name the harness uses to record whether
# ``METRIC_EXTRACTOR_USER_MODULE_REGEX`` was auto-generated. Excluded
# from the operator-passthrough allowlist above - the harness owns it
# exclusively.
_REGEX_AUTO_SET_ENV = "METRIC_EXTRACTOR_USER_MODULE_REGEX_AUTO_SET"

# Default auto-regex anchor: a narrow pattern matching only the
# ``sol-execbench`` staged user kernel ``.so`` under the harness-owned
# TMPDIR. Mirrors ``nvbit.py::compose_auto_user_module_regex`` so the
# two surfaces apply the same filter scope when matched dispatch
# invocations run against the same baseline.
def compose_auto_user_module_regex(staging_tmp_dir: Path) -> str:
    """Generate the narrow auto-regex matching only the sol-execbench
    staged user kernel ``.so`` under the harness-owned TMPDIR.

    The pattern shape:
      ``^<escaped-real-tmp>/sol_execbench_[^/]+/benchmark_kernel(?:\\.[^/]+)?\\.so(?: \\(deleted\\))?$``

    The optional ``(?:\\.[^/]+)?`` tolerates the early-dlopen window
    before the upstream evaluator renames the platform-suffixed
    ``.cpython-312-x86_64-linux-gnu.so`` to plain ``benchmark_kernel.so``;
    the optional ``(?: \\(deleted\\))?`` tolerates the
    ``/proc/self/maps`` post-unlink suffix. ``re.escape`` on the
    realpath keeps the anchor resilient to TMPDIRs that carry regex
    metacharacters (e.g. ``+``, ``.``, ``(``, ``)``).
    """
    tmp_real = os.path.realpath(str(staging_tmp_dir))
    return (
        rf"^{re.escape(tmp_real)}/sol_execbench_[^/]+/"
        r"benchmark_kernel(?:\.[^/]+)?\.so(?: \(deleted\))?$"
    )


def compute_env_filter_hash(env_view: Dict[str, str]) -> str:
    """Return a SHA-256 hex digest of the canonical key=value join over
    ``NVBIT_METRICS_USER_ENV_VARS``. Drives cache invalidation in
    ``SurfaceOrchestrator.compute_surface_cache_identity`` - any change
    in the active operator env (regex string change, slot-cap change,
    artifact-path change) yields a different hash and forces a re-run.

    Keys are sorted lexicographically; unset vars are omitted from the
    canonical form so a run with an empty value differs from a run
    with the var unset (the tool's behavior is the same in both cases,
    but the cache key keeps them distinguishable so accidental
    unset/set toggling is surfaced). The
    ``METRIC_EXTRACTOR_USER_MODULE_REGEX_AUTO_SET`` flag is excluded
    here - it is a derived classification, not an operator-tunable knob.
    """
    parts: List[str] = []
    for key in sorted(NVBIT_METRICS_USER_ENV_VARS):
        if key in env_view:
            parts.append(f"{key}={env_view[key]}")
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# CUPTI-bypass shim - mirrors ``nvbit.py``'s pattern so the new surface
# can ride the same eval-driver path without re-introducing the
# documented ``ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED (39)`` collision
# when sol-execbench's in-process CUPTI subscriber races NVBit's
# CUDA-callback subscription.
# ---------------------------------------------------------------------------
_CUPTI_BYPASS_SITECUSTOMIZE = '''\
"""Auto-generated by scripts/baselines_harness/surfaces/nvbit_metrics.py.

nvbit_metrics-only timing shim: monkey-patches
``sol_execbench.core.bench.timing.time_runnable`` to default
``methodology="cuda_events"`` so the CUPTI Activity subscription it
would otherwise install does not collide with the NVBit-injected
CUDA-callback subscription the metric extractor's NVBit ``.so``
already holds. Operator-supplied ``methodology`` arguments still win -
only the unset case is overridden.

Removing this file (or removing its parent directory from
``PYTHONPATH``) restores the upstream ``methodology="cupti"`` default.
"""

import os
import sys


def _install_shim() -> None:
    try:
        import sol_execbench.core.bench.timing as _timing
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            "WARNING: nvbit_metrics CUPTI-bypass shim could not import "
            "sol_execbench.core.bench.timing (%r); proceeding without "
            "the methodology override. The eval driver may hit "
            "CUPTI subscriber collision under NVBit injection.\\n"
            % (exc,)
        )
        return
    _orig = _timing.time_runnable

    def _patched_time_runnable(*args, **kwargs):
        kwargs.setdefault("methodology", "cuda_events")
        return _orig(*args, **kwargs)

    _patched_time_runnable.__wrapped__ = _orig
    _patched_time_runnable.__name__ = _orig.__name__
    _patched_time_runnable.__doc__ = _orig.__doc__
    _timing.time_runnable = _patched_time_runnable
    if os.environ.get("SIMBENCH_NVBIT_METRICS_CUPTI_BYPASS_VERBOSE", "") not in ("", "0"):
        sys.stderr.write(
            "INFO: nvbit_metrics CUPTI-bypass shim installed: "
            "sol_execbench.core.bench.timing.time_runnable defaults to "
            "methodology=cuda_events\\n"
        )


_install_shim()
'''


def write_cupti_bypass_sitecustomize(directory: Path) -> Path:
    """Materialize the nvbit_metrics-only ``sitecustomize.py`` shim
    under ``directory``. Idempotent - overwrites any prior content so a
    re-run picks up shim updates without manual cleanup. Returns the
    canonical realpath of the directory (the value that should be
    prepended to ``PYTHONPATH``).
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sitecustomize.py").write_text(_CUPTI_BYPASS_SITECUSTOMIZE)
    return Path(os.path.realpath(str(directory)))


def compose_pythonpath_with_shim(shim_dir: Path,
                                 existing: Optional[str]) -> str:
    """Prepend the shim directory to an existing ``PYTHONPATH`` so the
    NVBit-injected eval subprocess loads the shim BEFORE any
    user-supplied path entries. The realpath of ``shim_dir`` is what
    lands on ``PYTHONPATH``; symlink components are resolved so the
    eval driver's ``site.py`` import sees the same canonical path.
    """
    real = os.path.realpath(str(shim_dir))
    if not existing:
        return real
    return f"{real}{os.pathsep}{existing}"


class NVBitMetricsSurface(Surface):
    """``nvbit_metrics`` counter-free metric-extractor plugin."""

    name = "nvbit_metrics"
    schema_version = "1"
    impl_version = "1"

    def __init__(
        self,
        tool_path: Path = METRIC_EXTRACTOR_TOOL_PATH,
        eval_timeout_s: float = 1800.0,
    ) -> None:
        # ``eval_timeout_s`` matches the legacy ``nvbit`` plugin's
        # 30-minute default - the same wrapped sol-execbench eval
        # driver runs under instrumentation here, and torch + cuTile +
        # cudnn cubin load time dominates the wall-clock budget.
        self.tool_path = tool_path
        self.eval_timeout_s = eval_timeout_s

    # ------------------------------------------------------------------
    # probe()
    # ------------------------------------------------------------------
    def probe(self, host_info: HostInfo) -> SurfaceProbeResult:
        if not self.tool_path.is_file():
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                waiver_reason=(
                    f"metric_extractor tool not built at {self.tool_path}; "
                    f"run `make -C nvbit_tracer/metric_extractor` to build it."
                ),
            )
        # The driver-level injection probe is the same shape as the
        # legacy ``nvbit`` surface: ``CUDA_INJECTION64_PATH`` may be
        # silently dropped under a restrictive non-root ptrace / kernel
        # allowlist policy. The probe can't detect that without a live
        # run, so we report ``ok`` here and let ``run()`` map missing
        # output to ``status: blocked`` with a waiver naming the policy
        # origin.
        del host_info  # signature parity; not needed here
        return SurfaceProbeResult(
            status=SURFACE_STATUS_OK,
            tool_version=f"metric_extractor.so@{_so_mtime(self.tool_path)}",
            extra={"tool_path": str(self.tool_path)},
        )

    # ------------------------------------------------------------------
    # dispatch_argv() / _compose()
    # ------------------------------------------------------------------
    def dispatch_argv(self, plan):
        """Return the cache-identity argv for the native dispatch.

        Called by the orchestrator BEFORE the eval runs so the cache
        key reflects the surface-specific env / mount contributions.
        """
        return self._compose(
            plan,
            host_artifact_dir=Path("/tmp/_dummy_nvbit_metrics"),
        )

    def _compose(self, plan, *, host_artifact_dir: Path):
        # Native dispatch shape:
        #   CUDA_INJECTION64_PATH=<metric_extractor.so>
        #   METRIC_EXTRACTOR_LAUNCHES_PATH=<bundle>/nvbit_metrics/launches.jsonl
        #   METRIC_EXTRACTOR_FILTER_STATS_PATH=<bundle>/nvbit_metrics/filter_stats.json
        #   METRIC_EXTRACTOR_USER_MODULE_REGEX=<auto-regex>|<operator value>
        #   PYTHONPATH=<bundle>/nvbit_metrics/_cupti_bypass:<existing>
        #
        # The auto-regex anchors on the sol-execbench staged TMPDIR so
        # only the user kernel ``.so`` is instrumented; torch /
        # cuBLAS / cuDNN cubins are filtered out by the regex.
        staging_tmp = host_artifact_dir / "staging_tmp"
        try:
            staging_tmp.mkdir(parents=True, exist_ok=True)
        except OSError:
            # The dispatch-argv preview path may run against a
            # non-writable host_artifact_dir; the real run() call
            # below will retry against the writable bundle.
            pass

        filter_env: Dict[str, str] = {}
        for key in NVBIT_METRICS_USER_ENV_VARS:
            if key in os.environ:
                filter_env[key] = os.environ[key]
        operator_regex = filter_env.get("METRIC_EXTRACTOR_USER_MODULE_REGEX", "")
        if not operator_regex:
            filter_env["METRIC_EXTRACTOR_USER_MODULE_REGEX"] = (
                compose_auto_user_module_regex(staging_tmp)
            )
            filter_env[_REGEX_AUTO_SET_ENV] = "1"

        # CUPTI bypass: install a sitecustomize.py under a bundle-local
        # directory that monkey-patches
        # ``sol_execbench.core.bench.timing.time_runnable`` to default
        # ``methodology="cuda_events"``. Without this, the NVBit
        # injection collides with sol-execbench's CUPTI Activity timer
        # (ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, error 39) and the
        # eval driver aborts during the timing call before any user
        # kernel runs. The directory is prepended to PYTHONPATH for
        # the eval subprocess only; other surfaces and canonical
        # judge/scoring invocations are unaffected.
        cupti_bypass_dir = host_artifact_dir / "_cupti_bypass"
        try:
            cupti_bypass_real = write_cupti_bypass_sitecustomize(
                cupti_bypass_dir
            )
        except OSError:
            cupti_bypass_real = Path(
                os.path.realpath(str(cupti_bypass_dir))
            )
        pythonpath_with_shim = compose_pythonpath_with_shim(
            cupti_bypass_real, os.environ.get("PYTHONPATH"),
        )

        # Artifact paths - harness-owned, pinned to the bundle directory.
        # The producer reads METRIC_EXTRACTOR_LAUNCHES_PATH and
        # METRIC_EXTRACTOR_FILTER_STATS_PATH from getenv at first use;
        # the harness pins these values so the JSON-L lands under
        # ``<bundle>/nvbit_metrics/`` regardless of any operator env
        # state. The keys are intentionally NOT in
        # ``NVBIT_METRICS_USER_ENV_VARS`` (the cache-key allowlist), and
        # they are written into the composed env AFTER the operator
        # filter_env merge so an inherited operator value cannot win
        # the redirect.
        launches_path = host_artifact_dir / _LAUNCHES_FILENAME
        filter_stats_path = host_artifact_dir / _FILTER_STATS_FILENAME

        # The surface does NOT auto-set
        # METRIC_EXTRACTOR_TIER2_BINARY_SHA256_HEX from
        # plan.landed_solution_sha256. plan.landed_solution_sha256
        # is the SHA-256 of the staged solution.json: a source-file
        # proxy, NOT the profiled binary image digest the Tier-2
        # sidecar schema requires for LaunchHeaderV1.binary_sha256.
        # The producer in metric_extractor.cu hashes the accepted
        # module origin (the loaded benchmark_kernel.so or
        # calibration executable) directly on launch-enter and
        # populates LaunchContext.binary_sha256, so the schema field
        # carries the actual profiled binary image identity.
        #
        # METRIC_EXTRACTOR_TIER2_BINARY_SHA256_HEX remains in
        # NVBIT_METRICS_USER_ENV_VARS so an operator who explicitly
        # sets the var (e.g., for low-level synthetic-digest tests in
        # test_metric_extractor_tier2_runtime.py) still participates in
        # cache identity; the var is now a TEST OVERRIDE rather than
        # the production source.

        extra_env = {
            "CUDA_INJECTION64_PATH": str(self.tool_path),
            "TMPDIR": os.path.realpath(str(staging_tmp)),
            "PYTHONPATH": pythonpath_with_shim,
            **filter_env,
            # Harness-owned artifact paths land LAST so they always win
            # over any operator-supplied value that the filter_env merge
            # might have surfaced from the shell environment.
            "METRIC_EXTRACTOR_LAUNCHES_PATH": str(launches_path),
            "METRIC_EXTRACTOR_FILTER_STATS_PATH": str(filter_stats_path),
        }
        extra_mounts = [
            (self.tool_path, "metric_extractor_tool"),
            (host_artifact_dir, "nvbit_metrics_artifact"),
        ]
        return plan.build_eval_argv(
            extra_env=extra_env, extra_mounts=extra_mounts,
        )

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------
    def run(
        self,
        baseline_dir: Path,
        problem_dir: Path,
        bundle_dir: Path,
        build_mode: str,
        workload_subset: List[int],
        dispatch_command: Optional[List[str]] = None,
        plan: Any = None,
    ) -> SurfaceRunResult:
        del baseline_dir, problem_dir, build_mode, workload_subset
        del dispatch_command  # composed via plan.build_eval_argv inside _compose
        out_dir = bundle_dir / _NVBIT_METRICS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if plan is None:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=["(no dispatch plan supplied)"],
                exit_code=1,
                start_utc=now_utc_iso(),
                end_utc=now_utc_iso(),
                waiver_reason=(
                    "NVBitMetricsSurface requires a DispatchPlan so the "
                    "tool path + injection env can be composed into the "
                    "native ``sol-execbench`` subprocess."
                ),
            )
        argv, env_overrides, mounts = self._compose(
            plan, host_artifact_dir=out_dir,
        )
        del mounts  # consumed only by container shapes the orchestrator no longer uses
        # Lint pre-flight: the constructed command MUST NOT carry any
        # privileged CUPTI flag. NVBit and CUPTI Events / Metrics / PM
        # Sampling / PerfWorks are mutually exclusive at the CUDA driver
        # layer, and CUPTI's privileged surfaces are forbidden by the
        # framework contract anyway (CLAUDE.md Critical Rule 3).
        for flag in _FORBIDDEN_FLAGS:
            hit = any(
                token == flag or token.startswith(f"{flag}=")
                for token in argv
            )
            if hit:
                return SurfaceRunResult(
                    status=SURFACE_STATUS_FAILED,
                    command=argv,
                    exit_code=1,
                    start_utc=now_utc_iso(),
                    end_utc=now_utc_iso(),
                    waiver_reason=(
                        f"nvbit_metrics surface refuses to launch with "
                        f"forbidden CUPTI flag {flag!r}; NVBit and "
                        f"privileged CUPTI surfaces are mutually "
                        f"exclusive."
                    ),
                )
        start = now_utc_iso()
        subprocess_env = build_subprocess_env(
            plan_env_overrides=plan.env_overrides,
            surface_env_overrides=env_overrides,
            extra_allowed=BUILD_TOOLCHAIN_ALLOWED,
        )
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                env=subprocess_env,
                timeout=self.eval_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return SurfaceRunResult(
                status=SURFACE_STATUS_BLOCKED,
                command=argv,
                exit_code=-1,
                start_utc=start,
                end_utc=now_utc_iso(),
                waiver_reason=(
                    f"nvbit_metrics eval timed out after "
                    f"{self.eval_timeout_s}s wrapped around sol-execbench. "
                    f"tool_path={self.tool_path}; verify the host allows "
                    f"LD_PRELOAD-style CUDA driver injection (ptrace + "
                    f"kernel allowlist)."
                ),
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=argv,
                exit_code=-1,
                start_utc=start,
                end_utc=now_utc_iso(),
                waiver_reason=f"nvbit_metrics eval launch failed: {exc!r}",
            )
        end = now_utc_iso()
        # If the injection succeeded the tool emits at least one
        # launches.jsonl line; absence with a clean returncode usually
        # signals that the runtime suppressed CUDA_INJECTION64_PATH
        # (provider-side ptrace / kernel allowlist policy).
        launches_path = out_dir / _LAUNCHES_FILENAME
        if not launches_path.exists() and proc.returncode == 0:
            return SurfaceRunResult(
                status=SURFACE_STATUS_BLOCKED,
                command=argv,
                exit_code=proc.returncode,
                stdout_tail=(proc.stdout or "")[-500:],
                stderr_tail=(proc.stderr or "")[-500:],
                start_utc=start,
                end_utc=end,
                waiver_reason=(
                    "nvbit_metrics injection appears blocked by host "
                    "policy: CUDA_INJECTION64_PATH was set but no "
                    "launches.jsonl landed under the bundle dir. Verify "
                    "the host allows LD_PRELOAD-style CUDA driver "
                    "injection (ptrace + kernel allowlist)."
                ),
                extra={"env_overrides": env_overrides},
            )
        cupti_bypass_shim_path = str(
            out_dir / "_cupti_bypass" / "sitecustomize.py"
        )
        return SurfaceRunResult(
            status=(
                SURFACE_STATUS_OK
                if proc.returncode == 0
                else SURFACE_STATUS_FAILED
            ),
            command=argv,
            exit_code=proc.returncode,
            stdout_tail=(proc.stdout or "")[-500:],
            stderr_tail=(proc.stderr or "")[-500:],
            start_utc=start,
            end_utc=end,
            extra={
                "env_overrides": env_overrides,
                "config_overrides": reduced_config_overrides(),
                "cupti_bypass_shim_path": cupti_bypass_shim_path,
            },
        )

    # ------------------------------------------------------------------
    # parse()
    # ------------------------------------------------------------------
    def parse(self, bundle_dir: Path) -> SurfaceParseResult:
        out_dir = bundle_dir / _NVBIT_METRICS_DIR
        return parse_nvbit_metrics_artifacts(out_dir)

    def parse_on_failure(
        self,
        bundle_dir: Path,
        run_result: SurfaceRunResult,
    ) -> Optional[SurfaceParseResult]:
        """Overlay a typed waiver onto failed runs when the producer
        wrote a partial artifact tree.

        The producer's ``nvbit_at_term()`` always writes
        ``filter_stats.json`` (even on abnormal eval driver exits, via
        atexit fallback) so the on-disk state often carries
        operator-actionable diagnostics that the eval driver's stderr
        does not. Whenever a ``filter_stats.json`` exists under
        ``<bundle>/nvbit_metrics/``, route through
        ``parse_nvbit_metrics_artifacts`` so the manifest carries the
        typed waiver instead of a generic launch-failure message.
        """
        del run_result  # signature parity; not needed here
        out_dir = bundle_dir / _NVBIT_METRICS_DIR
        if not out_dir.is_dir():
            return None
        if (out_dir / _FILTER_STATS_FILENAME).is_file():
            return parse_nvbit_metrics_artifacts(out_dir)
        return None


def parse_nvbit_metrics_artifacts(out_dir: Path) -> SurfaceParseResult:
    """Validate a finished ``nvbit_metrics`` artifact directory.

    Failure-mode precedence:

    1. ``launches.jsonl`` missing: ``status=failed``,
       ``waiver_reason="launches.jsonl missing"`` - the tool never
       executed past its first cuLaunch decision.
    2. ``launches.jsonl`` parses with errors (mid-line truncation,
       JSON-decode errors, missing required fields): ``status=failed``
       with the first error's reason inlined.
    3. ``launches.jsonl`` empty (zero records, no errors):
       ``status=failed`` with a "no launches captured" waiver. An
       empty file usually means the user-module regex matched nothing.
    4. Otherwise: ``status=ok`` with ``artifact_path="nvbit_metrics/"``
       and substatus carrying the parser counts (record count,
       per-launch_kind tally, ``filter_stats.json`` snapshot if
       present).

    Factored out of ``NVBitMetricsSurface.parse()`` so the failure-
    precedence test can exercise it directly with a synthetic
    artifact tree.
    """
    launches_path = out_dir / _LAUNCHES_FILENAME
    filter_stats_path = out_dir / _FILTER_STATS_FILENAME

    if not launches_path.is_file():
        # Surface the filter_stats.json snapshot when it exists; the
        # producer's atexit fallback writes it even when the eval
        # driver crashed before any user kernel ran, so operators get
        # the running-state evidence on the manifest.
        substatus = _filter_stats_substatus(filter_stats_path)
        return SurfaceParseResult(
            status=SURFACE_STATUS_FAILED,
            waiver_reason="launches.jsonl missing",
            substatus=substatus,
        )

    # Strict parse via the canonical reference parser.
    records, errors = _parse_launches(launches_path)
    if errors:
        substatus = _filter_stats_substatus(filter_stats_path)
        substatus["records_parsed"] = len(records)
        substatus["parse_errors"] = [
            {
                "line_number": e.line_number,
                "reason": e.reason,
                "raw_head": e.raw[:200],
            }
            for e in errors[:5]
        ]
        substatus["parse_error_count"] = len(errors)
        first = errors[0]
        return SurfaceParseResult(
            status=SURFACE_STATUS_FAILED,
            waiver_reason=(
                f"launches.jsonl parse error at line "
                f"{first.line_number}: {first.reason}"
            ),
            substatus=substatus,
        )
    if not records:
        substatus = _filter_stats_substatus(filter_stats_path)
        substatus["records_parsed"] = 0
        return SurfaceParseResult(
            status=SURFACE_STATUS_FAILED,
            waiver_reason=(
                "launches.jsonl emitted zero records; either the "
                "user-module regex matched nothing or the eval "
                "subprocess exited before any cuLaunchKernel "
                "completion fired."
            ),
            substatus=substatus,
        )
    substatus = _filter_stats_substatus(filter_stats_path)
    substatus["records_parsed"] = len(records)
    kind_tally: Dict[str, int] = {}
    for rec in records:
        kind_tally[rec.launch_kind] = kind_tally.get(rec.launch_kind, 0) + 1
    substatus["launch_kind_tally"] = kind_tally
    # Surface the schema_version / tool_version of the first record so
    # the manifest carries the producer's wire-version stamp without
    # the operator having to read the JSON-L manually.
    substatus["schema_version"] = records[0].schema_version
    substatus["tool_version"] = records[0].tool_version
    # Bounded-memory aggregates. Walk every parsed record and
    # roll the per-record substatus flags into manifest-level counters
    # so an operator scanning the manifest sees the bounded-memory
    # signal without having to grep the JSON-L itself. Three rollups:
    #
    # * ``truncated_launches`` - count of records carrying
    #   ``substatus.truncated = true``. Non-zero means at least one
    #   launch hit the per-launch artifact cap and emitted a bounded
    #   replacement record instead of its full metric payload.
    # * ``per_process_budget_skipped_launches`` - count of records
    #   carrying ``substatus.skipped = true`` AND
    #   ``substatus.waiver_reason = "per_process_budget_exceeded"``.
    #   Non-zero means at least one accepted launch landed after the
    #   slot pool was fully consumed and was emitted uninstrumented
    #   with a budget-skip waiver.
    # * ``saturated_counters_total`` - sum of every record's
    #   ``substatus.saturated_counters`` value. Non-zero means at
    #   least one per-launch counter clamped at ``UINT64_MAX`` instead
    #   of wrapping; the operator should investigate whether the
    #   workload genuinely exceeded the counter's representable range
    #   or whether a producer-side regression is double-incrementing.
    #
    # Per-waiver-reason counts surface explicitly so operators see the
    # individual reason distribution rather than a single aggregate
    # number; the canonical waiver-reason values are documented in
    # ``metric_extractor.cu`` and ``JSONL_SCHEMA.md``.
    truncated_launches = 0
    per_process_budget_skipped_launches = 0
    saturated_counters_total = 0
    waiver_reason_counts: Dict[str, int] = {}
    for rec in records:
        rec_substatus = rec.raw.get("substatus")
        if not isinstance(rec_substatus, dict):
            continue
        if bool(rec_substatus.get("truncated")):
            truncated_launches += 1
        if bool(rec_substatus.get("skipped")):
            waiver = rec_substatus.get("waiver_reason")
            if waiver == "per_process_budget_exceeded":
                per_process_budget_skipped_launches += 1
        sat = rec_substatus.get("saturated_counters")
        if isinstance(sat, int) and not isinstance(sat, bool):
            saturated_counters_total += sat
        waiver = rec_substatus.get("waiver_reason")
        if isinstance(waiver, str) and waiver:
            waiver_reason_counts[waiver] = (
                waiver_reason_counts.get(waiver, 0) + 1
            )
    substatus["truncated_launches"] = truncated_launches
    substatus["per_process_budget_skipped_launches"] = (
        per_process_budget_skipped_launches
    )
    substatus["saturated_counters_total"] = saturated_counters_total
    substatus["waiver_reason_counts"] = waiver_reason_counts

    # Filter-stats contradiction detector. The terminal
    # ``filter_stats.json`` MUST not fall behind the live state the
    # last per-launch record's ``filter_stats_snapshot`` saw. A
    # separate atexit process could once overwrite the file with
    # zero counters; the producer's merge-via-flock writer closes
    # the race at the source, but the surface parser still detects
    # and surfaces the contradiction so operators see a regression
    # immediately. The check compares the terminal file's
    # accepted/skipped counters against the FINAL in-record snapshot
    # - they should be >= the in-record state because the terminal
    # write happens after every per-launch record was emitted.
    last_record_snapshot = records[-1].raw.get("filter_stats_snapshot")
    filter_stats_payload = substatus.get("filter_stats")
    if (
        isinstance(last_record_snapshot, dict)
        and isinstance(filter_stats_payload, dict)
    ):
        contradiction_fields = []
        for fld in (
            "accepted_launches",
            "skipped_launches",
            "module_origin_miss_count",
            "cooperative_launches",
            "graph_node_launches",
            "graph_node_accepted",
            "graph_node_skipped",
            "graph_launches_seen",
            "graph_launches_not_observed",
            "per_process_budget_skipped",
            "unknown_opcode_count",
        ):
            in_record_val = last_record_snapshot.get(fld)
            terminal_val = filter_stats_payload.get(fld)
            if (
                isinstance(in_record_val, int)
                and not isinstance(in_record_val, bool)
                and isinstance(terminal_val, int)
                and not isinstance(terminal_val, bool)
                and terminal_val < in_record_val
            ):
                contradiction_fields.append({
                    "field": fld,
                    "in_record_final": in_record_val,
                    "terminal_filter_stats": terminal_val,
                })
        # Path-attribution contradiction: if the in-record final
        # snapshot carries representative_skipped_paths but the
        # terminal file's array is empty (or missing), surface it.
        # An empty terminal array while the live state has framework
        # paths is the audit failure the path-attribution sweep
        # closes.
        in_record_paths = last_record_snapshot.get(
            "representative_skipped_paths"
        )
        terminal_paths = filter_stats_payload.get(
            "representative_skipped_paths"
        )
        if (
            isinstance(in_record_paths, list)
            and len(in_record_paths) > 0
            and (
                not isinstance(terminal_paths, list)
                or len(terminal_paths) == 0
            )
        ):
            contradiction_fields.append({
                "field": "representative_skipped_paths",
                "in_record_final": list(in_record_paths),
                "terminal_filter_stats": (
                    list(terminal_paths)
                    if isinstance(terminal_paths, list) else None
                ),
            })
        if contradiction_fields:
            substatus["terminal_filter_stats_inconsistent"] = (
                contradiction_fields
            )
            # The standalone filter_stats.json MUST reflect the live
            # counter state for the path-attribution audit. The C++
            # merge writer in metric_extractor.cu handles the
            # multi-process race at the source, but only when
            # ``nvbit_at_term`` actually fires - sol-execbench's
            # eval-driver subprocess lifecycle can exit through paths
            # (signals, _exit) that bypass NVBit's termination
            # callback. The Python-side reconciler here closes the
            # remaining gap: when the parser observes the
            # contradiction, it rewrites filter_stats.json from the
            # authoritative in-record final state (taking
            # max(terminal, in_record) per field so a partial-merge
            # state can be reconciled). After the rewrite, the
            # standalone artifact agrees with the JSON-L's last
            # snapshot - which is what the audit contract requires.
            try:
                reconciled = dict(filter_stats_payload)
                for fld in (
                    "accepted_launches",
                    "skipped_launches",
                    "module_origin_miss_count",
                    "cooperative_launches",
                    "graph_node_launches",
                    "graph_node_accepted",
                    "graph_node_skipped",
                    "graph_launches_seen",
                    "graph_launches_not_observed",
                    "per_process_budget_skipped",
                    "unknown_opcode_count",
                ):
                    in_v = last_record_snapshot.get(fld)
                    cur_v = reconciled.get(fld)
                    if (
                        isinstance(in_v, int)
                        and not isinstance(in_v, bool)
                        and isinstance(cur_v, int)
                        and not isinstance(cur_v, bool)
                        and in_v > cur_v
                    ):
                        reconciled[fld] = in_v
                # launch_index is the highest accepted+skipped+...; the
                # in-record snapshot does not carry it separately, so
                # roll it from accepted+skipped as a conservative
                # lower-bound that beats the existing zero.
                inferred_launch_index = (
                    reconciled.get("accepted_launches", 0)
                    + reconciled.get("skipped_launches", 0)
                )
                existing_launch_index = reconciled.get("launch_index", 0)
                if (
                    isinstance(existing_launch_index, int)
                    and not isinstance(existing_launch_index, bool)
                    and inferred_launch_index > existing_launch_index
                ):
                    reconciled["launch_index"] = inferred_launch_index
                # Path-attribution recovery. The terminal
                # filter_stats.json's representative_skipped_paths
                # array can be empty when nvbit_at_term misses the
                # live state; in that case the in-record
                # filter_stats_snapshot is the authoritative source
                # for the framework module paths the user-module
                # filter rejected. Union with any existing
                # terminal-file paths (cap at 10) so the standalone
                # audit artifact carries the right
                # attribution even under the missing-terminal-callback
                # path.
                in_record_paths = last_record_snapshot.get(
                    "representative_skipped_paths"
                )
                if isinstance(in_record_paths, list):
                    existing_paths = reconciled.get(
                        "representative_skipped_paths"
                    )
                    if not isinstance(existing_paths, list):
                        existing_paths = []
                    seen = set(existing_paths)
                    unioned = list(existing_paths)
                    for p in in_record_paths:
                        if not isinstance(p, str):
                            continue
                        if p in seen:
                            continue
                        seen.add(p)
                        unioned.append(p)
                        if len(unioned) >= 10:
                            break
                    if unioned != existing_paths:
                        reconciled["representative_skipped_paths"] = (
                            unioned
                        )
                tmp_path = filter_stats_path.with_suffix(
                    filter_stats_path.suffix + ".reconcile.tmp"
                )
                tmp_path.write_text(
                    json.dumps(reconciled) + "\n", encoding="utf-8"
                )
                tmp_path.replace(filter_stats_path)
                substatus["terminal_filter_stats_reconciled_from_in_record"] = (
                    True
                )
                substatus["filter_stats"] = reconciled
            except OSError as exc:
                # Reconciliation is best-effort. If the file is not
                # writable (read-only mount, permissions), surface the
                # failure but keep status=ok - the contradiction
                # detector already flagged the issue for the operator.
                substatus["terminal_filter_stats_reconcile_failed"] = (
                    f"{type(exc).__name__}: {exc}"
                )
    size = launches_path.stat().st_size
    if filter_stats_path.is_file():
        size += filter_stats_path.stat().st_size
    return SurfaceParseResult(
        status=SURFACE_STATUS_OK,
        artifact_path=f"{_NVBIT_METRICS_DIR}/",
        size_bytes=size,
        compressed_size_bytes=size,
        substatus=substatus,
    )


def _parse_launches(launches_path: Path):
    """Return ``(records, errors)`` via the strict reference parser.

    Lazy-imports the parser so a broken import in the
    ``baselines_harness.metric_extractor`` package does not poison the
    surface plugin's module-load path.
    """
    from baselines_harness.metric_extractor.launches_parser import (
        parse_launches_jsonl,
    )
    return parse_launches_jsonl(launches_path)


def _filter_stats_substatus(filter_stats_path: Path) -> Dict[str, Any]:
    """Return a substatus dict carrying the ``filter_stats.json``
    snapshot when present, else an empty dict. Bad JSON degrades to
    an empty dict with a ``filter_stats_parse_error`` marker so the
    surface stays operator-actionable instead of swallowing the
    failure.
    """
    if not filter_stats_path.is_file():
        return {}
    try:
        payload = json.loads(filter_stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"filter_stats_parse_error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {
            "filter_stats_parse_error": (
                f"top-level value is {type(payload).__name__}, "
                f"expected JSON object"
            ),
        }
    return {"filter_stats": payload}


def _so_mtime(path: Path) -> str:
    """Best-effort tool-version stamp for the built ``.so``."""
    try:
        return str(int(path.stat().st_mtime))
    except OSError:
        return "unknown"
