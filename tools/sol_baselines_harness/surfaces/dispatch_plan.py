"""Dispatch plan: native eval-driver invocation for one bundle.

The orchestrator turns a ``(baseline_dir, problem_dir, build_mode,
bundle_dir, host_info)`` tuple into a ``DispatchPlan`` that fully
specifies how to invoke the upstream ``sol-execbench`` evaluator on
the host. Each surface that needs the eval consumes the plan via
``DispatchPlan.build_eval_argv(wrappers, extra_env, extra_mounts)``,
which composes a full invocation with surface-specific wrapper
prefixes (compute-sanitizer / nsys / NVBit) and surface-specific
env overrides.

Dispatch is single-shape and native: ``sol-execbench <problem_dir>
--solution <path> --json ...`` runs as a host subprocess driven by
the harness. No Docker, no podman, no container runtime.

The plan captures every decision (argv tokens, env_overrides, staged
solution / config / workload paths, evaluator tree hash, reference-
timing flag) so the manifest's reproducibility fields are derivable
directly from the plan.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from sol_baselines_harness.cache import sol_execbench_source_hash  # noqa: E402
from sol_baselines_harness.native_dispatch import (  # noqa: E402
    _resolve_sol_execbench,
)
from sol_baselines_harness.normalise_benchmark import (  # noqa: E402
    needs_staging,
    stage_benchmark,
)
from sol_baselines_harness.surfaces.base import HostInfo
from sol_baselines_harness.surfaces.profiling_build import (
    reduced_config_overrides,
    stage_profiling_solution,
    stage_reduced_config,
)


_REPO_ROOT = _SCRIPTS_DIR.parent
# Real FlashInfer trace dataset (gitignored external blobs) lives at the
# repo-root ``data/flashinfer-trace``; the prior ``tools/sol_execbench/data/...``
# path never existed on disk.
_FLASHINFER_TRACE_DIR_DEFAULT = _REPO_ROOT / "data" / "flashinfer-trace"


@dataclass
class DispatchPlan:
    """Concrete native eval-driver invocation for one bundle.

    The plan stores the canonical ``eval_argv`` plus a set of base
    env overrides; each surface adds its own wrappers / env / mount
    records via :meth:`build_eval_argv`.

    ``mounts`` is advisory only: host paths recorded alongside their
    surface role so the manifest can audit which artifacts the run
    touched. The list never carries container-only path segments.
    """

    eval_argv: List[str]
    env_overrides: Dict[str, str]
    mounts: List[Tuple[str, str]]
    staged_solution_path: Path
    staged_config_path: Optional[Path]
    staged_workload_path: Optional[Path]
    landed_solution_sha256: str
    evaluator_tree_hash: str
    eval_timeout_s: float
    reference_timing_enabled: bool
    build_mode: str = "judge"
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    workload_subset: List[int] = field(default_factory=list)
    workload_subset_signature: str = ""

    def build_eval_argv(
        self,
        wrappers: Optional[List[str]] = None,
        extra_env: Optional[Dict[str, str]] = None,
        extra_mounts: Optional[List[Tuple[Path, str]]] = None,
    ) -> Tuple[List[str], Dict[str, str], List[Tuple[str, str]]]:
        """Return ``(argv, env_overrides, mounts)`` for one eval invocation.

        Parameters
        ----------
        wrappers
            Tokens prepended to the canonical ``eval_argv`` so the
            wrapper runs natively before ``sol-execbench``. Used by
            sanitizer (``["compute-sanitizer", "--tool=memcheck",
            "--xml=on", ...]``) and nsys (``["nsys", "profile",
            "--trace=cuda,nvtx", "--output=<host_path>"]``).
        extra_env
            Surface-specific env overrides merged on top of the plan's
            base ``env_overrides``. Used by NVBit (``CUDA_INJECTION64_PATH``,
            ``TRACES_FOLDER``), CUPTI (``SOLBENCH_CUPTI_ACTIVITY_OUT``),
            etc.
        extra_mounts
            Surface-specific advisory ``(host_path, role)`` records
            for manifest recording.
        """
        wrappers = list(wrappers or [])
        extra_env = dict(extra_env or {})
        extra_mounts = list(extra_mounts or [])

        merged_env = {**self.env_overrides, **extra_env}
        argv = list(wrappers) + list(self.eval_argv)
        mounts_record = list(self.mounts) + [(str(h), c) for h, c in extra_mounts]
        return argv, merged_env, mounts_record

    @property
    def base_eval_argv(self) -> List[str]:
        """Return the unmodified canonical eval argv (auditing only)."""
        return list(self.eval_argv)

    def as_manifest_dispatch_command(self) -> Dict[str, Any]:
        """Return the dict the manifest's top-level ``dispatch_command`` slot wants."""
        return {
            "argv": list(self.eval_argv),
            "env_overrides": dict(self.env_overrides),
            "mounts": [list(m) for m in self.mounts],
        }


_REDUCED_CONFIG_SURFACES = frozenset(
    {"sanitizer", "nvbit_metrics", "cupti_activity", "nsys"}
)


def build_dispatch_plan(
    *,
    baseline_dir: Path,
    problem_dir: Path,
    bundle_dir: Path,
    host_info: HostInfo,
    build_mode: str,
    workload_subset: List[int],
    workload_count: int,
    eval_timeout_s: float = 900.0,
    reference_timing_enabled: bool = False,
    capture_static_so: bool = False,
    enabled_surfaces: Optional[frozenset] = None,
) -> DispatchPlan:
    """Construct a ``DispatchPlan`` for one bundle (native dispatch).

    Stages the profiling ``solution.json`` + reduced eval config when
    ``build_mode == "profiling"``; stages a workload subset
    ``workload.jsonl`` when the resolved subset is not the full
    workload set; records the landed ``solution.json`` identity once
    for cache metadata.

    Each surface composes its own invocation via
    :meth:`DispatchPlan.build_eval_argv`.
    """
    if not baseline_dir.is_dir():
        raise ValueError(f"baseline_dir not found: {baseline_dir}")
    if not problem_dir.is_dir():
        raise ValueError(f"problem_dir not found: {problem_dir}")
    landed_solution = baseline_dir / "solution.json"
    if not landed_solution.is_file():
        raise ValueError(f"solution.json not found at {landed_solution}")

    if needs_staging(problem_dir):
        problem_dir = stage_benchmark(problem_dir)

    landed_solution_sha256 = _sha256_of_file(landed_solution)

    staged_solution_path = landed_solution
    staged_config_path: Optional[Path] = None
    config_overrides: Dict[str, Any] = {}
    nvtx_inject = False
    if build_mode == "profiling":
        staging_root = bundle_dir / "_staged"
        staging_root.mkdir(parents=True, exist_ok=True)
        result = stage_profiling_solution(
            landed_solution_path=landed_solution,
            host_info=host_info,
            target_dir=staging_root,
        )
        staged_solution_path = result.staged_solution_path
        nvtx_inject = result.nvtx_macro_injected
    elif build_mode != "judge":
        raise ValueError(
            f"build_mode must be 'judge' or 'profiling'; got {build_mode!r}"
        )
    needs_reduced_config = bool(
        (enabled_surfaces or frozenset()) & _REDUCED_CONFIG_SURFACES
    )
    if needs_reduced_config or build_mode == "profiling":
        staging_root = bundle_dir / "_staged"
        staging_root.mkdir(parents=True, exist_ok=True)
        if staged_config_path is None:
            staged_config_path = stage_reduced_config(target_dir=staging_root)
        config_overrides = reduced_config_overrides()

    staged_workload_path: Optional[Path] = None
    if workload_subset and len(workload_subset) < workload_count:
        staged_workload_path = _stage_workload_subset(
            problem_dir=problem_dir,
            target_dir=bundle_dir / "_staged",
            subset_indices=workload_subset,
        )

    workload_signature = (
        ",".join(str(i) for i in sorted(workload_subset))
        if workload_subset else f"all-{workload_count}"
    )

    evaluator_tree_hash = sol_execbench_source_hash()

    torch_extensions_dir: Optional[Path] = None
    if capture_static_so:
        static_staging_host = (bundle_dir / "static" / "staging").resolve()
        static_staging_host.mkdir(parents=True, exist_ok=True)
        torch_extensions_dir = static_staging_host / "torch_extensions"
        torch_extensions_dir.mkdir(parents=True, exist_ok=True)
        (static_staging_host / "kernel_cu").mkdir(parents=True, exist_ok=True)
        (static_staging_host / "build_logs").mkdir(parents=True, exist_ok=True)

    eval_argv = _build_native_eval_argv(
        problem_dir=problem_dir,
        staged_solution_path=staged_solution_path,
        staged_config_path=staged_config_path,
        staged_workload_path=staged_workload_path,
        keep_staging=capture_static_so,
        eval_timeout_s=eval_timeout_s,
    )

    env_overrides: Dict[str, str] = {
        "FLASHINFER_TRACE_DIR": str(_FLASHINFER_TRACE_DIR_DEFAULT),
        "SOL_EXECBENCH_GPU_CLK_MHZ": os.environ.get("SOL_EXECBENCH_GPU_CLK_MHZ", ""),
        "SOL_EXECBENCH_DRAM_CLK_MHZ": os.environ.get("SOL_EXECBENCH_DRAM_CLK_MHZ", ""),
    }
    if torch_extensions_dir is not None:
        env_overrides["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)
        # The vendored ``build_ext.py`` builds the kernel via
        # ``torch.utils.cpp_extension.load(..., build_directory=str(HERE))``
        # where ``HERE = Path.cwd().resolve()`` is the staging dir
        # the upstream CLI creates via
        # ``tempfile.mkdtemp(prefix="sol_execbench_")``. The
        # ``build_directory=`` kwarg overrides
        # ``TORCH_EXTENSIONS_DIR``, so the JIT-built ``.so`` lands
        # under the staging dir, NOT under the torch_extensions
        # path. Route the upstream CLI's ``tempfile.mkdtemp`` to a
        # bundle-rooted location by setting ``TMPDIR``; the
        # staging dir then resolves to
        # ``<bundle>/static/staging/sol_execbench_XXX/`` and
        # ``StaticSurface._locate_kernel_so``'s fallback walk
        # picks up the ``.so`` deterministically. The
        # ``TORCH_EXTENSIONS_DIR`` override stays in place because
        # torch.utils.cpp_extension may still write intermediate
        # build artifacts there.
        env_overrides["TMPDIR"] = str(static_staging_host)
    if nvtx_inject:
        env_overrides["SOLBENCH_HAS_NVTX"] = "1"

    mounts: List[Tuple[str, str]] = [
        (str(problem_dir.resolve()), "problem_dir"),
        (str(staged_solution_path.resolve()), "solution"),
    ]
    if staged_config_path is not None:
        mounts.append((str(staged_config_path.resolve()), "config"))
    if staged_workload_path is not None:
        mounts.append((str(staged_workload_path.resolve()), "workload"))
    if capture_static_so:
        mounts.append((str((bundle_dir / "static" / "staging").resolve()), "static_staging"))

    return DispatchPlan(
        eval_argv=eval_argv,
        env_overrides=env_overrides,
        mounts=mounts,
        staged_solution_path=staged_solution_path,
        staged_config_path=staged_config_path,
        staged_workload_path=staged_workload_path,
        landed_solution_sha256=landed_solution_sha256,
        evaluator_tree_hash=evaluator_tree_hash,
        eval_timeout_s=eval_timeout_s,
        reference_timing_enabled=reference_timing_enabled,
        build_mode=build_mode,
        config_overrides=config_overrides,
        workload_subset=list(workload_subset),
        workload_subset_signature=workload_signature,
    )


def _stage_workload_subset(
    *,
    problem_dir: Path,
    target_dir: Path,
    subset_indices: List[int],
) -> Path:
    """Materialise a staged ``workload.jsonl`` filtered to ``subset_indices``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    source_path = problem_dir / "workload.jsonl"
    if not source_path.is_file():
        raise ValueError(f"workload.jsonl not found under {problem_dir}")
    lines = source_path.read_text(encoding="utf-8").splitlines()
    selected_lines: List[str] = []
    keep = set(subset_indices)
    valid_idx = 0
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if valid_idx in keep:
            selected_lines.append(raw)
        valid_idx += 1
    staged = target_dir / "workload.jsonl"
    staged.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    return staged


def _build_native_eval_argv(
    *,
    problem_dir: Path,
    staged_solution_path: Path,
    staged_config_path: Optional[Path],
    staged_workload_path: Optional[Path],
    keep_staging: bool = False,
    eval_timeout_s: Optional[float] = None,
) -> List[str]:
    """Construct the canonical native ``sol-execbench`` invocation.

    The ``--timeout`` CLI flag bounds the inner eval_driver
    subprocess. sol-execbench's default is 600s; surfaces that wrap
    the evaluator (NVBit, sanitizer, nsys, cupti) need a longer
    inner timeout so the inner driver finishes before the outer
    surface's ``subprocess.run(timeout=...)`` fires. We pass the
    plan's ``eval_timeout_s`` minus a 60s buffer (so the inner
    driver exits cleanly before the outer surface kills it) when
    that value exceeds the CLI default.
    """
    # Resolve the same in-tree evaluator used by the direct-dispatch path.
    # The curated subprocess environment preserves the caller's PATH and does
    # not implicitly activate ``tools/sol_execbench/.venv``; emitting a bare
    # command here therefore made every plan-based surface fail with
    # ``FileNotFoundError`` on a clean checkout.
    cmd: List[str] = [
        _resolve_sol_execbench(),
        str(problem_dir),
        "--solution", str(staged_solution_path),
        "--json",
    ]
    if eval_timeout_s is not None and eval_timeout_s > 600:
        # Leave a 60s buffer for the inner driver to flush its
        # stdout / write the trace / let NVBit's enhanced_tracer
        # finalise before the outer surface timeout fires.
        inner_timeout = max(600, int(eval_timeout_s) - 60)
        cmd.extend(["--timeout", str(inner_timeout)])
    if keep_staging:
        cmd.append("--keep-staging")
    if staged_config_path is not None:
        cmd.extend(["--config", str(staged_config_path)])
    if staged_workload_path is not None:
        cmd.extend(["--workload", str(staged_workload_path)])
    return cmd


def _sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of the file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "DispatchPlan",
    "build_dispatch_plan",
]
