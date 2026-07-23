"""``CUPTIActivitySurface`` — CUPTI Activity wall-clock attribution.

Consumes the env-gated extension to
``sol-execbench/src/sol_execbench/core/bench/timing.py``: when
``SOLBENCH_CUPTI_ACTIVITY_OUT=<path>`` is exported into the native
``sol-execbench`` evaluator subprocess (via
``plan.env_overrides`` curated through ``build_subprocess_env``), the
upstream evaluator's existing CUPTI Activity callback writes each
activity record as JSON to the named path. No second in-process
CUPTI subscription is registered — the framework piggy-backs on the
upstream consumer.

Per the framework contract, the patched ``timing.py`` produces
byte-identical CUPTI behavior to the pre-patch state when the env
var is unset.

The plugin uses the framework-wide reduced eval config
(``warmup_runs=1``, ``iterations=1``, ``benchmark_reference=false``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from baselines_harness.surfaces.base import (
    HostInfo,
    Surface,
    SurfaceParseResult,
    SurfaceProbeResult,
    SurfaceRunResult,
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

_CUPTI_DIR = "cupti"
_SINK_ENV_VAR = "SOLBENCH_CUPTI_ACTIVITY_OUT"


class CUPTIActivitySurface(Surface):
    """CUPTI Activity plugin (env-gated sink in upstream timing.py)."""

    name = "cupti_activity"
    schema_version = "1"
    impl_version = "1"

    def __init__(self, eval_timeout_s: float = 600.0) -> None:
        self.eval_timeout_s = eval_timeout_s

    def probe(self, host_info: HostInfo) -> SurfaceProbeResult:
        # Verify the env-gated patch is present in the upstream
        # timing.py — without it, setting SOLBENCH_CUPTI_ACTIVITY_OUT
        # has no effect and the surface would emit an empty artifact.
        timing_path = _find_upstream_timing_py()
        if timing_path is None:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                waiver_reason=(
                    "sol-execbench/src/sol_execbench/core/bench/timing.py "
                    "not found; cannot verify the env-gated CUPTI sink."
                ),
            )
        try:
            timing_text = timing_path.read_text(encoding="utf-8")
        except OSError as exc:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                waiver_reason=f"cannot read {timing_path}: {exc}",
            )
        if _SINK_ENV_VAR not in timing_text:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                waiver_reason=(
                    f"upstream timing.py at {timing_path} does not carry "
                    f"the {_SINK_ENV_VAR} env-gated sink branch; apply "
                    f"scripts/baselines_harness/patches/cupti_activity_sink.patch."
                ),
            )
        # Negative invariant: the patched callback MUST NOT have
        # registered a second activity_register_callbacks invocation.
        # The pre-patch file has exactly one call; we count them.
        registration_count = timing_text.count("activity_register_callbacks(")
        if registration_count != 1:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_FAILED,
                waiver_reason=(
                    f"upstream timing.py has {registration_count} "
                    f"activity_register_callbacks invocations; expected "
                    f"exactly 1 (the patch adds only a write-side "
                    f"branch, never a second subscription)."
                ),
            )
        return SurfaceProbeResult(
            status=SURFACE_STATUS_OK,
            tool_version="env-gated sink (in-process)",
            extra={"timing_py": str(timing_path)},
        )

    def dispatch_argv(self, plan):
        """Return the argv that the orchestrator would launch.

        Used by the cache-identity hash so the per-surface argv
        accounts for the subprocess env injection of
        ``SOLBENCH_CUPTI_ACTIVITY_OUT`` plus the host-side artifact
        path additions.
        """
        return self._compose(plan, host_activity_path=Path("/tmp/_dummy_cupti_activity.jsonl"))

    def _compose(self, plan, *, host_activity_path: Path):
        return plan.build_eval_argv(
            extra_env={_SINK_ENV_VAR: str(host_activity_path)},
            extra_mounts=[(host_activity_path.parent, "cupti_artifact")],
        )

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
        out_dir = bundle_dir / _CUPTI_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        activity_path = out_dir / "activity.jsonl"
        start = now_utc_iso()
        if plan is None:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=["(no dispatch plan supplied)"],
                exit_code=1,
                start_utc=start,
                end_utc=now_utc_iso(),
                waiver_reason=(
                    "CUPTIActivitySurface requires a DispatchPlan; the "
                    "subprocess env injection of "
                    "``SOLBENCH_CUPTI_ACTIVITY_OUT`` cannot be composed "
                    "without it."
                ),
            )
        # Native dispatch: SOLBENCH_CUPTI_ACTIVITY_OUT points at the
        # host bundle path; the patched upstream timing.py writes
        # activity records there.
        argv, env_overrides, mounts = self._compose(plan, host_activity_path=activity_path)
        # Pre-create the activity file (the upstream callback appends).
        activity_path.write_text("", encoding="utf-8")
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
                waiver_reason=f"CUPTI eval timed out after {self.eval_timeout_s}s",
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=argv,
                exit_code=-1,
                start_utc=start,
                end_utc=now_utc_iso(),
                waiver_reason=f"CUPTI eval launch failed: {exc!r}",
            )
        return SurfaceRunResult(
            status=SURFACE_STATUS_OK if proc.returncode == 0 else SURFACE_STATUS_FAILED,
            command=argv,
            exit_code=proc.returncode,
            stdout_tail=(proc.stdout or "")[-500:],
            stderr_tail=(proc.stderr or "")[-500:],
            start_utc=start,
            end_utc=now_utc_iso(),
            extra={
                "env_overrides": env_overrides,
                "config_overrides": reduced_config_overrides(),
                "wrapper": "cupti env-gated sink (host subprocess)",
            },
        )

    def parse(self, bundle_dir: Path) -> SurfaceParseResult:
        import json as _json
        path = bundle_dir / _CUPTI_DIR / "activity.jsonl"
        if not path.is_file():
            return SurfaceParseResult(
                status=SURFACE_STATUS_FAILED,
                waiver_reason=f"CUPTI surface expected {path}; missing.",
            )
        # Require at least one parseable JSON record carrying CUPTI
        # activity fields (kind, start_ns, end_ns). An empty file
        # signals the env-gated sink in the upstream timing.py never
        # fired in this run — likely because the patch wasn't applied
        # to the installed ``sol-execbench`` package.
        records: List[Dict[str, Any]] = []
        bad_lines = 0
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = _json.loads(raw)
            except _json.JSONDecodeError:
                bad_lines += 1
                continue
            if isinstance(rec, dict) and "kind" in rec:
                records.append(rec)
        if not records:
            return SurfaceParseResult(
                status="blocked",
                waiver_reason=(
                    f"CUPTI surface produced no Activity records in "
                    f"{path}. The env-gated sink in upstream "
                    f"sol-execbench/.../timing.py did not fire; verify "
                    f"the patch at scripts/baselines_harness/patches/"
                    f"cupti_activity_sink.patch is applied to the "
                    f"installed ``sol-execbench`` package's copy of "
                    f"timing.py."
                ),
                size_bytes=path.stat().st_size,
            )
        # Empty activity.jsonl is not automatically failure — a kernel
        # with no driver-visible activities will produce an empty
        # file. The orchestrator records the size and trusts the
        # operator to interpret a 0-byte trace.
        size = path.stat().st_size
        return SurfaceParseResult(
            status=SURFACE_STATUS_OK,
            artifact_path=f"{_CUPTI_DIR}/activity.jsonl",
            size_bytes=size,
            compressed_size_bytes=size,
        )


def _find_upstream_timing_py() -> Optional[Path]:
    """Locate the upstream ``timing.py`` carrying the env-gated sink."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "sol-execbench" / "src" / "sol_execbench" / "core" / "bench" / "timing.py"
        if candidate.is_file():
            return candidate
    return None
