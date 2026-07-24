"""``NsysSurface`` — Nsight Systems CUDA / API timeline.

Wraps the native ``sol-execbench`` evaluator subprocess with
``nsys profile --trace=cuda,nvtx``
to produce ``<bundle>/nsys/profile.nsys-rep``. NVTX ranges are
**optional**: if ``SOLBENCH_HAS_NVTX`` was injected at the profiling
build (i.e. host probes detected both NVTX header AND library), ranges
appear in the report and the manifest substatus is
``nvtx_ranges_present: true``; otherwise nsys still runs and substatus
is ``nvtx_ranges_present: false``. The surface status stays ``ok`` in
either case.

Judge-build ``solution.json`` never links ``-lnvToolsExt``; the NVTX
flag is exclusively a profiling-build concern.

The plugin uses the framework-wide reduced eval config
(``warmup_runs=1``, ``iterations=1``, ``benchmark_reference=false``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from sol_baselines_harness.surfaces.base import (
    HostInfo,
    Surface,
    SurfaceParseResult,
    SurfaceProbeResult,
    SurfaceRunResult,
    SURFACE_STATUS_FAILED,
    SURFACE_STATUS_OK,
    SURFACE_STATUS_UNAVAILABLE,
)
from sol_baselines_harness.surfaces.manifest import now_utc_iso
from sol_baselines_harness.surfaces.profiling_build import reduced_config_overrides
from sol_baselines_harness.surfaces.subprocess_env import (
    BUILD_TOOLCHAIN_ALLOWED,
    build_subprocess_env,
)

_NSYS_DIR = "nsys"


class NsysSurface(Surface):
    """Nsight Systems plugin."""

    name = "nsys"
    schema_version = "1"
    impl_version = "1"

    def __init__(self, eval_timeout_s: float = 900.0) -> None:
        self.eval_timeout_s = eval_timeout_s

    def probe(self, host_info: HostInfo) -> SurfaceProbeResult:
        # nsys must exist on the host (native dispatch path: the
        # nsys wrapper is prepended to the native ``sol-execbench``
        # argv and runs as a host subprocess. Native dispatch can
        # determine host tool availability up front, so a missing
        # ``nsys`` binary surfaces as ``unavailable`` with a
        # ``waiver_reason`` instead of letting the orchestrator
        # invoke the wrapper and record a misleading ``failed``
        # status.
        if shutil.which("nsys") is None:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                tool_version=None,
                waiver_reason=(
                    "nsys not located on PATH; install Nsight Systems "
                    "on the host to enable this surface."
                ),
            )
        try:
            proc = subprocess.run(
                ["nsys", "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                tool_version=None,
                waiver_reason=(
                    f"nsys --version failed to launch: {exc!r}"
                ),
            )
        if proc.returncode != 0:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                tool_version=None,
                waiver_reason=(
                    f"nsys --version exited with "
                    f"returncode={proc.returncode}; stderr tail: "
                    f"{(proc.stderr or '').strip()[-200:]!r}"
                ),
            )
        version = (proc.stdout or proc.stderr).strip().splitlines()
        return SurfaceProbeResult(
            status=SURFACE_STATUS_OK,
            tool_version=version[0] if version else None,
        )

    def dispatch_argv(self, plan):
        """Return the cache-identity argv for nsys's native wrap."""
        return self._compose(plan, host_artifact_dir=Path("/tmp/_dummy_nsys"))

    def _compose(self, plan, *, host_artifact_dir: Path):
        host_report = str(host_artifact_dir / "profile.nsys-rep")
        wrappers = [
            "nsys", "profile",
            "--trace=cuda,nvtx",
            "--force-overwrite=true",
            f"--output={host_report}",
        ]
        extra_mounts = [(host_artifact_dir, "nsys_artifact")]
        return plan.build_eval_argv(wrappers=wrappers, extra_mounts=extra_mounts)

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
        out_dir = bundle_dir / _NSYS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        start = now_utc_iso()
        if plan is None:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=["(no dispatch plan supplied)"],
                exit_code=1,
                start_utc=start,
                end_utc=now_utc_iso(),
                waiver_reason=(
                    "NsysSurface requires a DispatchPlan to compose "
                    "the host-side nsys profile wrapper around the "
                    "native ``sol-execbench`` argv."
                ),
            )
        argv, env_overrides, mounts = self._compose(plan, host_artifact_dir=out_dir)
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
                status=SURFACE_STATUS_FAILED,
                command=argv,
                exit_code=-1,
                start_utc=start,
                end_utc=now_utc_iso(),
                waiver_reason=f"nsys profile timed out after {self.eval_timeout_s}s",
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=argv,
                exit_code=-1,
                start_utc=start,
                end_utc=now_utc_iso(),
                waiver_reason=f"nsys profile launch failed: {exc!r}",
            )
        # The wrapped ``sol-execbench`` may exit non-zero for reasons
        # orthogonal to nsys's profile capture (e.g. extra timing
        # diagnostics that the eval driver returns 1 for under
        # nsys wrapping but 0 for under a bare invocation). nsys itself
        # produces the ``profile.nsys-rep`` artifact regardless. Treat
        # the surface as ``ok`` whenever the artifact landed non-empty;
        # the inner exit code is recorded in ``exit_code`` for audit.
        rep_path = out_dir / "profile.nsys-rep"
        artifact_ok = rep_path.is_file() and rep_path.stat().st_size > 0
        if proc.returncode != 0 and not artifact_ok:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=argv,
                exit_code=proc.returncode,
                stdout_tail=(proc.stdout or "")[-500:],
                stderr_tail=(proc.stderr or "")[-500:],
                start_utc=start,
                end_utc=now_utc_iso(),
                waiver_reason=(
                    f"nsys exited {proc.returncode} and no usable "
                    f"profile.nsys-rep was produced"
                ),
            )
        return SurfaceRunResult(
            status=SURFACE_STATUS_OK,
            command=argv,
            exit_code=proc.returncode,
            stdout_tail=(proc.stdout or "")[-500:],
            stderr_tail=(proc.stderr or "")[-500:],
            start_utc=start,
            end_utc=now_utc_iso(),
            extra={"config_overrides": reduced_config_overrides()},
        )

    def parse(self, bundle_dir: Path) -> SurfaceParseResult:
        path = bundle_dir / _NSYS_DIR / "profile.nsys-rep"
        if not path.is_file():
            return SurfaceParseResult(
                status=SURFACE_STATUS_FAILED,
                waiver_reason=f"nsys surface expected {path}; missing.",
            )
        size = path.stat().st_size
        # ``nvtx_ranges_present`` is a coarse heuristic: the .nsys-rep
        # is a SQLite blob and a reliable check requires opening it.
        # The orchestrator owns the authoritative bit via
        # ``HostInfo.nvtx_*_present``; the surface records ``False``
        # by default and lets the manifest reflect the orchestrator's
        # readout.
        return SurfaceParseResult(
            status=SURFACE_STATUS_OK,
            artifact_path=f"{_NSYS_DIR}/profile.nsys-rep",
            size_bytes=size,
            compressed_size_bytes=size,
            substatus={"nvtx_ranges_present": False},
        )
