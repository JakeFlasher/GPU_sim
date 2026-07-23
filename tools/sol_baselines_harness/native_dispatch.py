"""Build and run the official ``sol-execbench`` CLI natively.

The eval driver runs as a host subprocess, driven directly by
``sol-execbench <problem_dir> --solution <path> --json [extras]``.
The upstream package is installed natively (``uv sync`` of the
in-tree ``sol-execbench/`` source); ``sol-execbench`` is on the
host's PATH.

The module exposes ``DispatchRequest`` / ``DispatchResult`` /
``build_command`` / ``run`` as the canonical native dispatch surface
consumed by ``run.py`` and the surface plugins.

Subprocess env hygiene: ``run()`` curates the eval driver's env via
:func:`baselines_harness.surfaces.subprocess_env.build_subprocess_env`
— the same policy the surface plugins use. The curated allow-list
keeps the eval driver's build path resolving (``CUTLASS_DIR`` is
opted in via ``extra_allowed`` because the upstream
``sol-execbench/src/sol_execbench/driver/templates/build_ext.py``
reads it); the denylist scrubs profiling-surface state
(``LD_PRELOAD``, ``CUDA_INJECTION64_PATH``, ``NVBIT_TOOL``,
``TRACES_FOLDER``, ``SOLBENCH_CUPTI_ACTIVITY_OUT``,
``NSYS_OUTPUT_FILE``, and any ``CUPTI_*`` / ``NSYS_*`` key) that
would otherwise leak from a prior surface invocation or the host
shell.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from baselines_harness.surfaces.subprocess_env import (
    BUILD_TOOLCHAIN_ALLOWED,
    build_subprocess_env,
)

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
REPO_ROOT = _SCRIPTS_DIR.parent

# Eval driver's safetensors search-root for FlashInfer-backed
# workloads. The driver reads ``FLASHINFER_TRACE_DIR`` from the
# subprocess env and falls back to skipping these blobs when unset
# — workloads that reference flashinfer-trace tensors then fail or
# produce invalid evidence. The real dataset lives at the repo-root
# ``data/flashinfer-trace`` (gitignored external blobs, commit
# 435d738b); the prior ``sol-execbench/data/flashinfer-trace`` path
# never existed on disk. The surfaces orchestrator path also pins
# this var; we set it in the direct-dispatch path so ``run.py`` /
# ``report.py`` invocations carry the same metadata.
_FLASHINFER_TRACE_DIR = (
    REPO_ROOT / "data" / "flashinfer-trace"
)


class DispatchError(Exception):
    """Subprocess-construction or run failure."""


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of a dispatch call."""

    returncode: int
    stdout: str
    stderr: str
    cmd: List[str]


@dataclass(frozen=True)
class DispatchRequest:
    """Inputs for one harness invocation."""

    problem_dir: Path
    solution_path: Path
    extra_args: Sequence[str] = field(default_factory=tuple)
    timeout_seconds: Optional[float] = None


def build_command(req: DispatchRequest) -> List[str]:
    """Assemble the native ``sol-execbench`` invocation."""
    if not req.problem_dir.is_dir():
        raise DispatchError(
            f"Problem directory not found at {req.problem_dir}. Run "
            f"./sol-execbench/scripts/download_data.sh to populate."
        )
    if not req.solution_path.is_file():
        raise DispatchError(
            f"Solution file not found at {req.solution_path}. Run "
            f"scaffold.py + pack.py first."
        )
    cmd: List[str] = [
        "sol-execbench",
        str(req.problem_dir.resolve()),
        "--solution", str(req.solution_path.resolve()),
        "--json",
    ]
    cmd.extend(req.extra_args)
    return cmd


def run(req: DispatchRequest) -> DispatchResult:
    """Run the constructed command and return stdout/stderr/returncode.

    The caller decides what to do with the return value; this function
    does not raise on non-zero exit (the upstream CLI may exit non-zero
    on benign conditions like a single non-PASSED workload, and the
    harness wants to surface that as a deferred row).

    The subprocess env is curated via
    :func:`baselines_harness.surfaces.subprocess_env.build_subprocess_env`
    so profiling-surface state from a prior surface invocation
    (``LD_PRELOAD``, ``CUDA_INJECTION64_PATH``, ``CUPTI_*``,
    ``NSYS_*``, etc.) cannot leak into the eval driver.
    """
    cmd = build_command(req)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=req.timeout_seconds,
        cwd=str(REPO_ROOT),
        env=build_subprocess_env(
            plan_env_overrides={
                "FLASHINFER_TRACE_DIR": str(_FLASHINFER_TRACE_DIR),
            },
            extra_allowed=BUILD_TOOLCHAIN_ALLOWED,
        ),
    )
    return DispatchResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        cmd=list(cmd),
    )
