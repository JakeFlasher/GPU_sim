"""``SanitizerSurface`` — compute-sanitizer (memcheck/racecheck/synccheck/initcheck).

Runs ``compute-sanitizer`` on the host by composing the dispatch
plan with a ``compute-sanitizer`` wrapper inserted in front of the
native ``sol-execbench`` argv. Each tool emits to a per-bundle
artifact directory so the XML reports land directly on the host
filesystem.

Per the framework contract, the canonical judge methodology is
**unchanged** — sanitizer staging stays out of the landed
``solution.json`` and the canonical eval config; reduced config
(``warmup_runs=1``, ``iterations=1``, ``benchmark_reference=false``)
is recorded in ``manifest.config_overrides``.

Outputs per ``(slug, language)`` bundle:

* ``sanitizer/memcheck.xml``
* ``sanitizer/racecheck.xml``
* ``sanitizer/synccheck.xml``
* ``sanitizer/initcheck.xml``
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

_TOOLS = ("memcheck", "racecheck", "synccheck", "initcheck")
_SANITIZER_DIR = "sanitizer"


class SanitizerSurface(Surface):
    """compute-sanitizer plugin."""

    name = "sanitizer"
    schema_version = "1"
    impl_version = "2"

    def __init__(self, per_tool_timeout_s: float = 600.0) -> None:
        self.per_tool_timeout_s = per_tool_timeout_s

    def probe(self, host_info: HostInfo) -> SurfaceProbeResult:
        # ``compute-sanitizer`` is a host binary that ships with the
        # CUDA toolkit (the same toolkit ``nvcc`` / ``ptxas`` come
        # from). If it does not resolve on ``PATH``, the surface
        # cannot run — record that honestly as ``unavailable`` with
        # a ``waiver_reason`` so the manifest captures a host gap
        # instead of downgrading otherwise-clean bundles to
        # ``status: failed`` at run-time.
        if shutil.which("compute-sanitizer") is None:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                tool_version=None,
                waiver_reason=(
                    "compute-sanitizer not located on PATH; install "
                    "the host CUDA toolkit's compute-sanitizer binary "
                    "to enable this surface."
                ),
            )
        try:
            proc = subprocess.run(
                ["compute-sanitizer", "--version"],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=5.0,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                tool_version=None,
                waiver_reason=(
                    f"compute-sanitizer --version failed to launch: "
                    f"{exc!r}"
                ),
            )
        if proc.returncode != 0:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                tool_version=None,
                waiver_reason=(
                    f"compute-sanitizer --version exited with "
                    f"returncode={proc.returncode}; stderr tail: "
                    f"{(proc.stderr or '').strip()[-200:]!r}"
                ),
            )
        version = (proc.stdout or proc.stderr).strip().splitlines()
        return SurfaceProbeResult(
            status=SURFACE_STATUS_OK,
            tool_version=version[0] if version else None,
        )

    def dispatch_argv(self, plan) -> Tuple[List[str], Dict[str, str], List[Tuple[str, str]]]:
        """Return the resolved argv + env + mounts for one tool.

        Cache-identity hashing only. Side-effect-free: no directory
        creation, no host-fs writes. The real ``run()`` does the
        ``mkdir`` for ``<bundle>/sanitizer`` before invoking the
        wrapper.

        Uses ``memcheck`` as the representative argv (every tool
        shares the same outer shape; only the ``--tool=<name>`` token
        differs). The hash thus changes when the wrapper's outer
        flags / mounts change.
        """
        # Derive the host artifact dir purely from the plan; do NOT
        # create it here. The path is stable under the bundle because
        # the orchestrator's ``run_all`` writes ``manifest.json`` to
        # the same bundle root.
        bundle_root = self._infer_bundle_root(plan)
        host_sanitizer_dir = bundle_root / _SANITIZER_DIR
        return self._compose_argv(
            plan,
            tool="memcheck",
            host_sanitizer_dir=host_sanitizer_dir,
        )

    def _infer_bundle_root(self, plan) -> Path:
        """Pure inference of the bundle root from the plan.

        The bundle root is one directory above ``_staged`` when the
        plan stages a profiling solution (the typical case for
        sanitizer runs since they ride the reduced-config staging).
        For judge-mode bare runs we fall back to the
        ``staged_solution_path``'s parent, which is the landed
        ``baselines/<cat>/<slug>/<lang>/`` directory — used only by
        synthetic / test plans that don't carry a real bundle.
        """
        staged = Path(plan.staged_solution_path)
        if staged.parent.name == "_staged":
            return staged.parent.parent
        # Tests and synthetic plans: fall back to the staged
        # solution's parent. The directory is not created here.
        return staged.parent

    def _compose_argv(
        self,
        plan,
        *,
        tool: str,
        host_sanitizer_dir: Path,
    ) -> Tuple[List[str], Dict[str, str], List[Tuple[str, str]]]:
        host_xml = str(host_sanitizer_dir / f"{tool}.xml")
        # ``compute-sanitizer`` 2025.4+ uses ``--xml`` as a boolean
        # flag and ``--save <path>`` as the output sink. Together they
        # emit an XML error-record file.
        wrappers = [
            "compute-sanitizer",
            f"--tool={tool}",
            "--xml",
            "--save", host_xml,
        ]
        extra_mounts = [(host_sanitizer_dir, "sanitizer_artifact")]
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
        out_dir = bundle_dir / _SANITIZER_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        if plan is None:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=["(no dispatch plan supplied)"],
                exit_code=1,
                start_utc=now_utc_iso(),
                end_utc=now_utc_iso(),
                waiver_reason=(
                    "SanitizerSurface requires the orchestrator to "
                    "pass a DispatchPlan so the compute-sanitizer "
                    "wrapper can prepend to the native "
                    "``sol-execbench`` argv."
                ),
            )
        # Sanitizer artifacts land directly under the bundle's
        # sanitizer/ dir (host filesystem; no mount step).
        config_overrides = reduced_config_overrides()
        per_tool_argv: List[List[str]] = []
        per_tool_env: List[Dict[str, str]] = []
        per_tool_status: Dict[str, Dict[str, Any]] = {}
        any_failed = False
        start = now_utc_iso()
        for tool in _TOOLS:
            argv, env, _mounts = self._compose_argv(
                plan, tool=tool, host_sanitizer_dir=out_dir,
            )
            per_tool_argv.append(argv)
            per_tool_env.append(env)
            tool_start = now_utc_iso()
            subprocess_env = build_subprocess_env(
                plan_env_overrides=plan.env_overrides,
                surface_env_overrides=env,
                extra_allowed=BUILD_TOOLCHAIN_ALLOWED,
            )
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    check=False,
                    env=subprocess_env,
                    timeout=self.per_tool_timeout_s,
                )
                # compute-sanitizer success criterion: the per-tool
                # XML report exists and is parseable. The wrapped
                # ``sol-execbench`` CLI may exit non-zero on slow /
                # tolerance-failing workloads under sanitizer's
                # instrumentation; the sanitizer report is still
                # valid evidence in that case.
                xml_path = out_dir / f"{tool}.xml"
                report_valid = xml_path.is_file() and xml_path.stat().st_size > 0
                if report_valid:
                    try:
                        head = xml_path.read_text(encoding="utf-8")[:200]
                        report_valid = "ComputeSanitizerOutput" in head
                    except OSError:
                        report_valid = False
                if report_valid:
                    tool_status = SURFACE_STATUS_OK
                elif proc.returncode == 0:
                    tool_status = SURFACE_STATUS_FAILED
                else:
                    tool_status = SURFACE_STATUS_FAILED
                per_tool_status[tool] = {
                    "status": tool_status,
                    "exit_code": proc.returncode,
                    "start_utc": tool_start,
                    "end_utc": now_utc_iso(),
                    "stderr_tail": (proc.stderr or "")[-500:],
                    "command": argv,
                    "report_valid": report_valid,
                }
            except subprocess.TimeoutExpired:
                per_tool_status[tool] = {
                    "status": SURFACE_STATUS_FAILED,
                    "exit_code": -1,
                    "start_utc": tool_start,
                    "end_utc": now_utc_iso(),
                    "waiver_reason": f"compute-sanitizer/{tool} timed out after {self.per_tool_timeout_s}s",
                    "command": argv,
                }
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                per_tool_status[tool] = {
                    "status": SURFACE_STATUS_FAILED,
                    "exit_code": -1,
                    "start_utc": tool_start,
                    "end_utc": now_utc_iso(),
                    "waiver_reason": f"compute-sanitizer launch failed: {exc!r}",
                    "command": argv,
                }
            if per_tool_status[tool]["status"] == SURFACE_STATUS_FAILED:
                any_failed = True
        end = now_utc_iso()
        # Surface aggregation: ``ok`` when at least one tool produced
        # a valid XML report — operator can still derive sanitizer
        # evidence per-tool from substatus. Surface ``failed`` only
        # when EVERY tool failed to land a report (e.g., every wrapper
        # subprocess failed to launch).
        any_report_valid = any(
            ts.get("report_valid") for ts in per_tool_status.values()
        )
        surface_status = (
            SURFACE_STATUS_OK if any_report_valid else SURFACE_STATUS_FAILED
        )
        return SurfaceRunResult(
            status=surface_status,
            command=per_tool_argv[0] if per_tool_argv else [],
            exit_code=0 if any_report_valid else 1,
            start_utc=start,
            end_utc=end,
            waiver_reason=(
                "no compute-sanitizer tool produced a valid XML report; "
                "see manifest.surfaces.sanitizer.substatus per-tool detail."
                if surface_status == SURFACE_STATUS_FAILED else (
                    "partial: one or more tools failed to produce a "
                    "report (see substatus.per_tool_status); the surface "
                    "is still ok because at least one tool succeeded."
                    if any_failed else None
                )
            ),
            extra={
                "config_overrides": config_overrides,
                "per_tool_status": per_tool_status,
                "env_overrides": per_tool_env[0] if per_tool_env else {},
                "wrapper": "compute-sanitizer (host subprocess)",
            },
        )

    def parse(self, bundle_dir: Path) -> SurfaceParseResult:
        out_dir = bundle_dir / _SANITIZER_DIR
        sizes = {}
        missing = []
        for tool in _TOOLS:
            p = out_dir / f"{tool}.xml"
            if not p.is_file():
                missing.append(tool)
                continue
            sizes[tool] = p.stat().st_size
        # At least one tool's XML report must be present; a fully
        # missing set indicates the wrapper never produced any output.
        if not sizes:
            return SurfaceParseResult(
                status=SURFACE_STATUS_FAILED,
                waiver_reason=(
                    f"sanitizer surface produced no XML artifacts under "
                    f"{out_dir}/; expected at least one of {list(_TOOLS)}"
                ),
                size_bytes=0,
            )
        total = sum(sizes.values())
        return SurfaceParseResult(
            status=SURFACE_STATUS_OK,
            artifact_path=f"{_SANITIZER_DIR}/",
            size_bytes=total,
            compressed_size_bytes=total,
            substatus={
                "per_tool_size_bytes": sizes,
                "tools_missing_reports": missing,
            },
        )
