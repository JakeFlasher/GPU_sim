"""Profiling-build mechanism: staged ``solution.json`` copy + NVTX gating.

The framework's profiling surfaces (NVBit, sanitizer, CUPTI Activity,
nsys) need a slightly different build than the canonical judge:

* ``-DSOLBENCH_PROFILING`` is **always** injected when
  ``build_mode == "profiling"``. Source code branches on this macro to
  enable per-kernel ``%clock64`` probes, instrumentation hooks, etc.
* ``-DSOLBENCH_HAS_NVTX`` is **conditionally** injected only when both
  NVTX header AND NVTX library are present on the host. NVTX include
  and call sites in *.cu kernels must guard against
  ``SOLBENCH_HAS_NVTX``, never against ``SOLBENCH_PROFILING`` alone
  — a profiling build on a host without NVTX must still compile
  cleanly (the contract enforced by `host_probe._probe_nvtx_paths`).

The landed ``baselines/<cat>/<slug>/<lang>/solution.json`` file is
**never mutated**: profiling runs stage a separate copy under
``<bundle_dir>/_staged/solution.json`` (or a temporary directory when
the bundle isn't yet materialized) so the landed file's mtime stays
invariant.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from sol_baselines_harness.surfaces.base import HostInfo

PROFILING_MACRO = "-DSOLBENCH_PROFILING"
NVTX_MACRO = "-DSOLBENCH_HAS_NVTX"
NVTX_LIBRARY_FLAG = "-lnvToolsExt"


@dataclass(frozen=True)
class ProfilingBuildResult:
    """Outcome of staging the profiling ``solution.json``."""

    staged_solution_path: Path
    nvtx_header_present: bool
    nvtx_library_present: bool
    profiling_macro_injected: bool
    nvtx_macro_injected: bool
    nvtx_library_linked: bool


def stage_profiling_solution(
    landed_solution_path: Path,
    host_info: HostInfo,
    *,
    target_dir: Optional[Path] = None,
) -> ProfilingBuildResult:
    """Materialize a profiling-build ``solution.json`` next to the landed file.

    The staged copy carries:

    * ``-DSOLBENCH_PROFILING`` in ``compile_options.cuda_cflags`` (always).
    * ``-DSOLBENCH_HAS_NVTX`` in ``compile_options.cuda_cflags`` AND
      ``-lnvToolsExt`` in ``compile_options.ld_flags`` (only when both
      NVTX probes returned True at host probe time).

    The landed ``solution.json`` file at ``landed_solution_path`` is
    NOT modified — we read it, deep-copy the dict in memory, mutate
    the copy, and write the copy to ``target_dir / 'solution.json'``.
    When ``target_dir`` is None, a fresh temporary directory is
    created and the path returned to the caller.

    Caller is responsible for cleanup of the staged directory; the
    convention is one staged directory per bundle so the orchestrator
    can pass ``bundle_dir / '_staged'``.
    """
    text = landed_solution_path.read_text(encoding="utf-8")
    solution_dict = json.loads(text)
    spec = solution_dict.setdefault("spec", {})
    compile_options = spec.setdefault("compile_options", {})
    cuda_cflags = list(compile_options.get("cuda_cflags") or [])
    ld_flags = list(compile_options.get("ld_flags") or [])
    # Profiling macro is unconditional.
    if PROFILING_MACRO not in cuda_cflags:
        cuda_cflags.append(PROFILING_MACRO)
    nvtx_inject = (
        host_info.nvtx_header_present and host_info.nvtx_library_present
    )
    if nvtx_inject:
        if NVTX_MACRO not in cuda_cflags:
            cuda_cflags.append(NVTX_MACRO)
        if NVTX_LIBRARY_FLAG not in ld_flags:
            ld_flags.append(NVTX_LIBRARY_FLAG)
    else:
        # Defensively strip any pre-existing NVTX flags inherited from
        # the landed solution.json so a staged profiling build on a
        # host without NVTX can never link nvToolsExt by accident.
        cuda_cflags = [f for f in cuda_cflags if f != NVTX_MACRO]
        ld_flags = [f for f in ld_flags if f != NVTX_LIBRARY_FLAG]
    compile_options["cuda_cflags"] = cuda_cflags
    compile_options["ld_flags"] = ld_flags
    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="solbench_profiling_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    staged_path = target_dir / "solution.json"
    staged_path.write_text(
        json.dumps(solution_dict, indent=2) + "\n",
        encoding="utf-8",
    )
    return ProfilingBuildResult(
        staged_solution_path=staged_path,
        nvtx_header_present=host_info.nvtx_header_present,
        nvtx_library_present=host_info.nvtx_library_present,
        profiling_macro_injected=True,
        nvtx_macro_injected=nvtx_inject,
        nvtx_library_linked=nvtx_inject,
    )


def stage_reduced_config(
    *,
    target_dir: Path,
    eval_config_path: Optional[Path] = None,
) -> Path:
    """Materialize a reduced eval config for sanitizer / NVBit / nsys / CUPTI.

    Operator-driven decision (recorded in the plan as DEC-2): profiling
    runs trade steady-state telemetry for tractability via:

    * ``warmup_runs=1``
    * ``iterations=1``
    * ``benchmark_reference=false``

    The reduced config is staged as a separate JSON file (one per
    bundle) so the canonical judge eval methodology is unchanged.
    When ``eval_config_path`` is supplied, we read it as a base and
    overlay the three reduced fields; otherwise we emit a minimal
    config carrying only the three overrides (the upstream CLI fills
    in defaults for unspecified fields).

    Returns the on-disk path of the staged config.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    if eval_config_path is not None and eval_config_path.is_file():
        base = json.loads(eval_config_path.read_text(encoding="utf-8"))
    else:
        base = {}
    base["warmup_runs"] = 1
    base["iterations"] = 1
    base["benchmark_reference"] = False
    staged_path = target_dir / "eval_config.reduced.json"
    staged_path.write_text(
        json.dumps(base, indent=2) + "\n",
        encoding="utf-8",
    )
    return staged_path


def reduced_config_overrides() -> Dict[str, Any]:
    """Return the dict of reduced-config overrides for manifest recording."""
    return {
        "warmup_runs": 1,
        "iterations": 1,
        "benchmark_reference": False,
    }


def assert_landed_solution_unchanged(
    landed_solution_path: Path,
    pre_run_mtime_ns: int,
) -> None:
    """Verify the landed ``solution.json`` mtime hasn't moved.

    Per the profiling-build contract: the landed file MUST stay
    byte-identical to its pre-run state. The check is mtime-based
    (cheap; catches the common mutation patterns) — a malicious or
    buggy code path that re-wrote the file with byte-identical bytes
    would silently pass this check, but that scenario doesn't match
    the framework's observed failure modes.

    Raises ``RuntimeError`` when the mtime has changed.
    """
    actual = landed_solution_path.stat().st_mtime_ns
    if actual != pre_run_mtime_ns:
        raise RuntimeError(
            f"landed solution.json at {landed_solution_path} was "
            f"modified during a profiling run "
            f"(pre={pre_run_mtime_ns}, post={actual}). "
            f"Profiling builds MUST stage a copy; the landed file is "
            f"byte-immutable under the framework's contract."
        )
