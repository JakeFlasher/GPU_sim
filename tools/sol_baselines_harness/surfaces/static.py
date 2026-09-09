"""``StaticSurface`` — static introspection over the built kernel ``.so``.

Build-time surface: runs ``cuobjdump --dump-sass`` / ``--dump-ptx`` /
``--list-elf`` / ``--print-gpu-summary``, ``nvdisasm`` per-kernel
disassembly, and ``ptxas -v`` register / spill report against the
``.so`` the upstream evaluator emits.

Cubin-first extension: the surface also extracts per-arch cubins
via ``cuobjdump -xelf all -arch=sm_<n>`` into ``static/cubins/<arch>/``
and runs four additional cubin-first invocations against the
extracted cubins:

* ``static/disasm-bbcfg-poff.txt``   — ``nvdisasm -bbcfg -poff``
* ``static/disasm-json.txt``         — ``nvdisasm --format=json --print-code``
* ``static/resource-usage.txt``      — ``cuobjdump --dump-resource-usage``
* ``static/disasm-g.txt``,           — ``nvdisasm -g`` / ``-gi`` / ``-gp``
  ``static/disasm-gi.txt``,
  ``static/disasm-gp.txt``

Each new invocation tolerates non-zero exit (``accept_nonzero=True``)
so a missing ``-lineinfo`` or unsupported arch does not fail the
surface. The per-pilot ``cubin_extraction`` substatus records which
arches were extracted and how many cubins each produced.

``.so`` capture strategy: the upstream evaluator builds the kernel
on the host via ``torch.utils.cpp_extension.load()`` into the
directory pointed at by :envvar:`TORCH_EXTENSIONS_DIR`. The plan
sets :envvar:`TORCH_EXTENSIONS_DIR` to
``<bundle_dir>/static/staging/torch_extensions`` when the static
surface is enabled, so the staged ``.so`` lands at a deterministic
host-rooted path and survives the eval subprocess exit.

The staging layout under ``<bundle_dir>/static/staging/`` is:

* ``torch_extensions/`` — the JIT-built ``.so`` + ``build.ninja``.
* ``kernel_cu/``        — staged copies of the source ``.cu`` files.
* ``build_logs/``       — raw build stderr captured for diagnostics.

When :envvar:`TORCH_EXTENSIONS_DIR` is unset on the eval subprocess,
the surface emits ``status: failed`` with a ``waiver_reason`` naming
the missing variable — never falls through silently to a system-
default path that the harness cannot locate.

The surface produces:

* ``static/sass.txt``                 — ``cuobjdump --dump-sass``.
* ``static/ptx.txt``                  — ``cuobjdump --dump-ptx``.
* ``static/list-elf.txt``             — ``cuobjdump --list-elf``.
* ``static/print-gpu-summary.txt``    — ``cuobjdump --print-gpu-summary``.
* ``static/disasm.txt``               — ``nvdisasm`` per-kernel disassembly.
* ``static/ptxas-v.txt``              — ``ptxas -v`` stats from a native
  ``nvcc --cubin -Xptxas=-v`` recompile of the staged ``kernel.cu``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from sol_baselines_harness.surfaces.subprocess_env import (
    BUILD_TOOLCHAIN_ALLOWED,
    build_subprocess_env,
)

_STATIC_DIR = "static"
_STAGING_SUBDIR = "staging"


class StaticSurface(Surface):
    """Static introspection over the built kernel ``.so``."""

    name = "static"
    schema_version = "1"
    # impl_version bumped from "1" to "2" when the per-arch cubin
    # extraction + four cubin-first invocations were added (six new
    # artifact files plus an extracted-cubin tree under cubins/).
    # The bump is load-bearing because the orchestrator's cache
    # identity hashes impl_version; a pre-bump cache entry without
    # the new artifacts would otherwise satisfy the new contract on
    # cache replay — every input that affects the produced binary's
    # runtime behavior must be in the cache key.
    impl_version = "2"

    def probe(self, host_info: HostInfo) -> SurfaceProbeResult:
        tools = ("cuobjdump", "nvdisasm", "ptxas")
        missing = [t for t in tools if shutil.which(t) is None]
        if missing:
            return SurfaceProbeResult(
                status=SURFACE_STATUS_UNAVAILABLE,
                waiver_reason=(
                    f"static introspection requires the CUDA Binary "
                    f"Utilities on PATH; missing: {missing}"
                ),
            )
        version = _probe_cuobjdump_version()
        return SurfaceProbeResult(
            status=SURFACE_STATUS_OK,
            tool_version=version,
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
        out_dir = bundle_dir / _STATIC_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        # The plan must set TORCH_EXTENSIONS_DIR so the eval subprocess
        # writes its JIT-built .so under a deterministic host-rooted
        # path. Without it, the harness cannot locate the .so and the
        # surface fails closed rather than silently falling through to
        # a system-default torch extensions cache.
        plan_env_overrides = (
            getattr(plan, "env_overrides", {}) if plan is not None else {}
        )
        if "TORCH_EXTENSIONS_DIR" not in plan_env_overrides:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=["(no TORCH_EXTENSIONS_DIR)"],
                exit_code=1,
                start_utc=now_utc_iso(),
                end_utc=now_utc_iso(),
                waiver_reason=(
                    "static surface requires TORCH_EXTENSIONS_DIR in the "
                    "plan's env_overrides so the JIT-built .so lands at a "
                    "deterministic host-rooted path; the plan did not "
                    "set the variable."
                ),
            )
        so_path = _locate_kernel_so(bundle_dir)
        if so_path is None:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=["(no .so capture)"],
                exit_code=1,
                start_utc=now_utc_iso(),
                end_utc=now_utc_iso(),
                waiver_reason=(
                    "no kernel .so found under "
                    f"{bundle_dir / _STATIC_DIR / _STAGING_SUBDIR}/. "
                    "Confirm the eval subprocess ran with --keep-staging "
                    "AND TORCH_EXTENSIONS_DIR set to the bundle's staging "
                    "root before invoking the static surface."
                ),
            )
        start = now_utc_iso()
        # Per-surface env isolation: the static surface's host-tool
        # subprocesses (cuobjdump / nvdisasm / nvcc) must run with a
        # curated env so denylisted variables (CUDA_INJECTION64_PATH,
        # CUPTI_*, NSYS_*, LD_PRELOAD) set by an earlier surface or
        # by the ambient shell never leak into the static path. The
        # static surface declares no overrides of its own.
        static_env = build_subprocess_env(
            plan_env_overrides=plan_env_overrides,
            surface_env_overrides={},
            extra_allowed=BUILD_TOOLCHAIN_ALLOWED,
        )
        commands: List[str] = []
        try:
            commands.append(
                _run_and_capture(
                    ["cuobjdump", "--dump-sass", str(so_path)],
                    out_dir / "sass.txt",
                    env=static_env,
                )
            )
            commands.append(
                _run_and_capture(
                    ["cuobjdump", "--dump-ptx", str(so_path)],
                    out_dir / "ptx.txt",
                    env=static_env,
                )
            )
            commands.append(
                _run_and_capture(
                    ["cuobjdump", "--list-elf", str(so_path)],
                    out_dir / "list-elf.txt",
                    env=static_env,
                )
            )
            # ``--print-gpu-summary`` is supported on newer cuobjdump
            # but missing on older toolchain versions. Tolerate the
            # missing flag by accepting non-zero exit; the artifact
            # text captures the actual stderr so the operator can
            # tell missing-flag from real failure.
            commands.append(
                _run_and_capture(
                    ["cuobjdump", "--print-gpu-summary"],
                    out_dir / "print-gpu-summary.txt",
                    accept_nonzero=True,
                    env=static_env,
                )
            )
            commands.append(
                _run_and_capture(
                    ["nvdisasm", str(so_path)],
                    out_dir / "disasm.txt",
                    accept_nonzero=True,
                    env=static_env,
                )
            )
            # ptxas -v: harvest from the build's stderr / staged log
            # files first, then fall back to a native ``nvcc --cubin
            # -Xptxas=-v`` recompile of the staged ``kernel.cu``.
            # Empty result is treated as a surface failure by the
            # parse step.
            ptxas_v = _harvest_ptxas_v(bundle_dir, env=static_env)
            (out_dir / "ptxas-v.txt").write_text(ptxas_v, encoding="utf-8")
            commands.append("(ptxas-v via native nvcc --cubin -Xptxas=-v recompile)")

            # Per-arch cubin extraction + four cubin-first
            # invocations. Each new invocation tolerates non-zero
            # exit so a missing -lineinfo, unsupported arch, or
            # older cuobjdump that lacks --dump-resource-usage
            # doesn't fail the whole surface. The extraction
            # metadata is persisted as a sidecar that the parser
            # reads back into the surface's manifest substatus —
            # `SurfaceRunResult` doesn't carry substatus directly,
            # so the sidecar is the cross-phase channel.
            cubin_extraction = _extract_cubins_and_disasm(
                so_path=so_path,
                out_dir=out_dir,
                env=static_env,
                commands=commands,
            )
            try:
                (out_dir / "cubin_extraction.json").write_text(
                    json.dumps(cubin_extraction, indent=2, sort_keys=True)
                )
            except OSError:
                # Best-effort; the parser tolerates absence.
                pass
        except subprocess.SubprocessError as exc:
            return SurfaceRunResult(
                status=SURFACE_STATUS_FAILED,
                command=commands,
                exit_code=1,
                start_utc=start,
                end_utc=now_utc_iso(),
                stderr_tail=str(exc)[-1000:],
                waiver_reason=f"static introspection subprocess failed: {exc!r}",
            )
        return SurfaceRunResult(
            status=SURFACE_STATUS_OK,
            command=commands,
            exit_code=0,
            start_utc=start,
            end_utc=now_utc_iso(),
        )

    def parse(self, bundle_dir: Path) -> SurfaceParseResult:
        out_dir = bundle_dir / _STATIC_DIR
        # Artifacts that must be present AND non-empty (real evidence).
        # ``print-gpu-summary.txt`` and ``disasm.txt`` are
        # tool-version-dependent and accept any non-zero content
        # (even an error message); the strict-non-empty list below
        # carries the artifacts the framework actually relies on.
        required_present = (
            "sass.txt",
            "ptx.txt",
            "list-elf.txt",
            "print-gpu-summary.txt",
            "disasm.txt",
            "ptxas-v.txt",
        )
        required_non_empty = ("sass.txt", "ptx.txt", "list-elf.txt")
        sizes = {}
        for name in required_present:
            p = out_dir / name
            if not p.is_file():
                return SurfaceParseResult(
                    status=SURFACE_STATUS_FAILED,
                    waiver_reason=(
                        f"static surface expected artifact {name} under "
                        f"{out_dir}/, but it is missing."
                    ),
                )
            sizes[name] = p.stat().st_size
        for name in required_non_empty:
            if sizes[name] == 0:
                return SurfaceParseResult(
                    status=SURFACE_STATUS_FAILED,
                    waiver_reason=(
                        f"static surface expected non-empty artifact {name} "
                        f"under {out_dir}/; got 0 bytes."
                    ),
                    size_bytes=sum(sizes.values()),
                )
        # ``ptxas-v.txt`` MUST carry real ptxas-info / register / spill
        # / stack-frame evidence (not an empty file, not a placeholder
        # comment block).
        ptxas_text = (out_dir / "ptxas-v.txt").read_text(encoding="utf-8")
        if not ptxas_v_evidence_is_real(ptxas_text):
            return SurfaceParseResult(
                status=SURFACE_STATUS_FAILED,
                waiver_reason=(
                    "static surface's ptxas-v.txt does not carry real "
                    "ptxas -v register/spill evidence (empty or "
                    "placeholder only). The host ``nvcc --cubin "
                    "-Xptxas=-v`` recompile produced no ptxas info "
                    "lines; either ``nvcc`` is not on PATH or the "
                    "kernel carries no __global__ device functions."
                ),
                size_bytes=sum(sizes.values()),
            )

        # Account for the cubin-first extension artifacts: the four
        # new disasm files + extracted per-arch cubins. Each is
        # optional — extraction tolerates non-zero exit, so absent
        # files are not a failure. Per-arch cubins live under
        # ``static/cubins/<arch>/``.
        cubin_first_artifacts = (
            "disasm-bbcfg-poff.txt",
            "disasm-json.txt",
            "resource-usage.txt",
            "disasm-g.txt",
            "disasm-gi.txt",
            "disasm-gp.txt",
        )
        for name in cubin_first_artifacts:
            p = out_dir / name
            if p.is_file():
                sizes[name] = p.stat().st_size
        cubins_root = out_dir / "cubins"
        if cubins_root.is_dir():
            for cubin in cubins_root.rglob("*.cubin"):
                # Key as ``cubins/<arch>/<filename>`` so the operator
                # can see which per-arch tree each cubin came from.
                rel_key = str(cubin.relative_to(out_dir))
                sizes[rel_key] = cubin.stat().st_size

        # Read the cross-phase sidecar that ``run()`` wrote and
        # surface it as a substatus block. Absent sidecar is
        # tolerated (older bundles / extraction didn't run).
        cubin_extraction_sidecar = out_dir / "cubin_extraction.json"
        cubin_extraction_payload: Optional[Dict[str, Any]] = None
        if cubin_extraction_sidecar.is_file():
            try:
                parsed = json.loads(cubin_extraction_sidecar.read_text())
                if isinstance(parsed, dict):
                    cubin_extraction_payload = parsed
            except (OSError, json.JSONDecodeError):
                cubin_extraction_payload = None

        total = sum(sizes.values())
        substatus: Dict[str, Any] = {"per_artifact_size_bytes": sizes}
        if cubin_extraction_payload is not None:
            substatus["cubin_extraction"] = cubin_extraction_payload

        return SurfaceParseResult(
            status=SURFACE_STATUS_OK,
            artifact_path=f"{_STATIC_DIR}/",
            size_bytes=total,
            compressed_size_bytes=total,
            substatus=substatus,
        )


def assert_keep_staging_is_host_mounted(
    eval_argv: List[str],
    plan: Any,
    bundle_dir: Path,
) -> None:
    """Reject bare ``--keep-staging`` without a host-rooted staging path.

    Called by the orchestrator before invoking the eval subprocess to
    enforce the static surface's contract: the JIT-built ``.so`` must
    land under :envvar:`TORCH_EXTENSIONS_DIR`, which the plan sets to
    ``<bundle_dir>/static/staging/torch_extensions`` when ``capture_static_so``
    is True. A bare ``--keep-staging`` without that env var would let
    torch fall back to a system-default path the harness cannot locate.

    Raises ``ValueError`` when ``--keep-staging`` appears without a
    matching :envvar:`TORCH_EXTENSIONS_DIR` override that resolves
    under the bundle's staging root.
    """
    if "--keep-staging" not in eval_argv:
        return
    expected = (Path(bundle_dir) / _STATIC_DIR / _STAGING_SUBDIR / "torch_extensions").resolve()
    plan_env = getattr(plan, "env_overrides", {}) or {}
    candidate = plan_env.get("TORCH_EXTENSIONS_DIR", "")
    if not candidate:
        raise ValueError(
            "--keep-staging requires TORCH_EXTENSIONS_DIR to be set in "
            "the plan's env_overrides so the JIT-built .so lands under "
            f"{expected}. The plan did not supply the variable; either "
            "enable capture_static_so when constructing the plan or "
            "drop --keep-staging from the eval argv."
        )
    candidate_resolved = Path(candidate).resolve()
    expected_str = str(expected)
    candidate_str = str(candidate_resolved)
    if not (
        candidate_str == expected_str
        or expected_str.startswith(candidate_str.rstrip("/") + "/")
        or candidate_str.startswith(expected_str.rstrip("/") + "/")
    ):
        raise ValueError(
            f"--keep-staging present but TORCH_EXTENSIONS_DIR={candidate!r} "
            f"is not under the bundle's expected staging root {expected}. "
            "Set TORCH_EXTENSIONS_DIR via the plan's env_overrides."
        )


def _extract_cubins_and_disasm(
    *,
    so_path: Path,
    out_dir: Path,
    env: Dict[str, str],
    commands: List[str],
) -> Dict[str, Any]:
    """Extract per-arch cubins from ``so_path`` via
    ``cuobjdump -xelf all -arch=sm_<n>`` and run the four cubin-first
    nvdisasm / cuobjdump invocations against the best extracted cubin.

    The arch list comes from the captured ``list-elf.txt`` output the
    surface already produced; we re-parse the file rather than re-running
    ``cuobjdump --list-elf`` so the discovery path is byte-identical to
    what the operator sees in the manifest.

    Each new invocation tolerates non-zero exit. The four output files
    live at ``<out_dir>/disasm-bbcfg-poff.txt``, ``disasm-json.txt``,
    ``resource-usage.txt``, and ``disasm-{g,gi,gp}.txt``. Returns a
    substatus dict the caller stuffs into the SurfaceRunResult so the
    orchestrator manifest carries the extraction evidence.
    """
    import re
    list_elf_path = out_dir / "list-elf.txt"
    arches: List[str] = []
    if list_elf_path.is_file():
        elf_text = list_elf_path.read_text(encoding="utf-8", errors="replace")
        # Each line looks like:
        #   ELF file    N: <basename>.<index>.<arch>.cubin
        # where <arch> is e.g. sm_75, sm_100, sm_100a, sm_120a.
        seen = set()
        for m in re.finditer(r"\.(sm_\d+[a-z]?)\.cubin\b", elf_text):
            arch = m.group(1)
            if arch not in seen:
                seen.add(arch)
                arches.append(arch)

    # Pick the highest-arch as the canonical target for the four
    # new invocations. The plan's primary B200 path is sm_100a; on
    # a local sm_120-only build we fall back to whatever's highest.
    # Sort so sm_100a outranks sm_100, sm_120a outranks sm_120, etc.
    def _arch_rank(a: str) -> tuple[int, str]:
        m = re.match(r"sm_(\d+)([a-z]?)", a)
        if not m:
            return (0, a)
        return (int(m.group(1)), m.group(2))
    arches_sorted = sorted(arches, key=_arch_rank, reverse=True)

    extraction_substatus: Dict[str, Any] = {
        "available_arches": arches,
        "extracted": {},
        "extraction_failed_arches": [],
        "best_arch_for_disasm": None,
        "disasm_outputs": {},
    }

    cuobjdump = shutil.which("cuobjdump")
    nvdisasm = shutil.which("nvdisasm")
    if not cuobjdump or not nvdisasm:
        extraction_substatus["tool_unavailable"] = {
            "cuobjdump": cuobjdump is not None,
            "nvdisasm": nvdisasm is not None,
        }
        return extraction_substatus

    cubins_root = out_dir / "cubins"
    cubins_root.mkdir(parents=True, exist_ok=True)

    best_cubin: Optional[Path] = None
    best_arch: Optional[str] = None
    for arch in arches_sorted:
        per_arch = cubins_root / arch
        per_arch.mkdir(parents=True, exist_ok=True)
        rc, _, _ = _run_cuobjdump_scoped(
            [cuobjdump, "-xelf", "all", "-arch", arch, str(so_path)],
            cwd=per_arch,
            env=env,
        )
        commands.append(
            f"(cuobjdump -xelf all -arch {arch} -> {per_arch})"
        )
        if rc != 0:
            extraction_substatus["extraction_failed_arches"].append(arch)
            continue
        cubin_files = sorted(per_arch.glob("*.cubin"))
        extraction_substatus["extracted"][arch] = [c.name for c in cubin_files]
        if not cubin_files:
            extraction_substatus["extraction_failed_arches"].append(arch)
            continue
        if best_cubin is None:
            # Pick the largest cubin in this arch — typically the
            # one with actual SASS rather than a link-stub.
            best_cubin = max(cubin_files, key=lambda p: p.stat().st_size)
            best_arch = arch

    extraction_substatus["best_arch_for_disasm"] = best_arch

    if best_cubin is None:
        # No cubin successfully extracted; the four new outputs are
        # absent. Surface stays ok because the original invocations
        # succeeded.
        return extraction_substatus

    # Four new invocations. Each tolerates non-zero exit so a
    # missing -lineinfo or unsupported arch does not fail the
    # surface (the original six static invocations still succeeded
    # by the time we reach this code).
    invocations = [
        (
            [nvdisasm, "-bbcfg", "-poff", str(best_cubin)],
            out_dir / "disasm-bbcfg-poff.txt",
        ),
        (
            [nvdisasm, "--format=json", "--print-code", str(best_cubin)],
            out_dir / "disasm-json.txt",
        ),
        (
            [cuobjdump, "--dump-resource-usage", str(best_cubin)],
            out_dir / "resource-usage.txt",
        ),
    ]
    # nvdisasm -g/-gi/-gp share the same disasm pass but differ in
    # the granularity of the per-instruction debug-info output.
    # Each writes its own file.
    for g_flag, suffix in (("-g", "g"), ("-gi", "gi"), ("-gp", "gp")):
        invocations.append(
            (
                [nvdisasm, g_flag, str(best_cubin)],
                out_dir / f"disasm-{suffix}.txt",
            )
        )

    for argv, dest in invocations:
        cmd_str = _run_and_capture(
            argv, dest, accept_nonzero=True, env=env,
        )
        commands.append(cmd_str)
        extraction_substatus["disasm_outputs"][dest.name] = {
            "size_bytes": dest.stat().st_size if dest.is_file() else 0,
        }

    return extraction_substatus


def _run_cuobjdump_scoped(
    argv: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
) -> tuple[int, str, str]:
    """Run cuobjdump scoped to a CWD so extracted .cubin files land
    in the intended per-arch dir. Returns (rc, stdout, stderr).
    cuobjdump's exit code is treated as advisory; the cubin-first
    extension tolerates non-zero so a missing arch on an older
    toolchain doesn't block the surface."""
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
            timeout=30.0,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (FileNotFoundError, subprocess.SubprocessError):
        return 127, "", "cuobjdump not on PATH or failed to launch"


def _probe_cuobjdump_version() -> Optional[str]:
    """Return ``cuobjdump --version`` first line, or ``None``."""
    try:
        proc = subprocess.run(
            ["cuobjdump", "--version"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    out = (proc.stdout or proc.stderr).strip().splitlines()
    return out[0] if out else None


def _run_and_capture(
    cmd: List[str],
    out_path: Path,
    accept_nonzero: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Run a command and dump combined stdout+stderr to ``out_path``.

    Returns the joined command line for manifest recording. When
    ``accept_nonzero`` is True (e.g. ``nvdisasm`` exits non-zero on
    fat binaries with a host-only stub), the non-zero return is
    captured into the artifact without raising.

    ``env`` must be a curated subprocess env produced by
    :func:`build_subprocess_env`; callers (currently
    :meth:`StaticSurface.run`) pass it explicitly so the static
    surface's host-tool subprocesses never inherit the ambient
    shell's denylisted variables.
    """
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=120.0,
        env=env,
    )
    out_path.write_text(
        (proc.stdout or "") + (proc.stderr or ""),
        encoding="utf-8",
    )
    if proc.returncode != 0 and not accept_nonzero:
        raise subprocess.SubprocessError(
            f"{cmd[0]} exited {proc.returncode}; stderr tail: "
            f"{(proc.stderr or '').splitlines()[-3:]}"
        )
    return " ".join(cmd)


def _locate_kernel_so(bundle_dir: Path) -> Optional[Path]:
    """Find the kernel ``.so`` under the bundle's staging directory.

    The eval subprocess builds the kernel via
    ``torch.utils.cpp_extension.load(..., build_directory=str(HERE))``
    where ``HERE = Path.cwd().resolve()`` is the staging dir the
    upstream CLI creates via ``tempfile.mkdtemp(prefix=
    "sol_execbench_")``. The ``build_directory=`` kwarg overrides
    :envvar:`TORCH_EXTENSIONS_DIR`, so the ``.so`` lands under the
    staging dir rather than the torch_extensions tree.

    The plan routes ``TMPDIR`` to ``<bundle_dir>/static/staging``
    when ``capture_static_so`` is set, so the staging dir resolves
    to ``<bundle_dir>/static/staging/sol_execbench_XXX/`` and the
    ``.so`` is captured under the bundle. This function walks the
    staging tree to locate it.

    Order of preference:
      1. ``<bundle_dir>/static/staging/sol_execbench_*/benchmark_kernel.so``
         (the canonical upstream-build output).
      2. Any ``.so`` under the torch_extensions tree (intermediate
         JIT artifacts torch may emit there).
      3. Any other ``.so`` under the staging dir (fallback for
         operator-driven staging layouts).
    """
    staging = bundle_dir / _STATIC_DIR / _STAGING_SUBDIR
    if not staging.is_dir():
        return None
    upstream_builds = sorted(
        (
            p for p in staging.glob("sol_execbench_*/benchmark_kernel.so")
            if p.is_file()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if upstream_builds:
        return upstream_builds[0]
    torch_extensions = staging / "torch_extensions"
    if torch_extensions.is_dir():
        candidates = sorted(
            (p for p in torch_extensions.rglob("*.so") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    for so in sorted(staging.rglob("*.so")):
        if so.is_file():
            return so
    return None


def _harvest_ptxas_v(
    bundle_dir: Path,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Harvest ``ptxas -v`` register / spill stats for the kernel.

    Strategy:
    * First, scan staging ``.log`` files for existing ``ptxas info``
      lines.
    * If no log lines are found, recompile the staged ``kernel.cu``
      natively via ``nvcc --cubin -Xptxas=-v`` under the curated
      ``env`` supplied by the caller.

    Returns the harvested text (or "" when both paths fail). The
    caller's parse step treats empty output as a surface failure.
    """
    staging = bundle_dir / _STATIC_DIR / _STAGING_SUBDIR
    if not staging.is_dir():
        return ""
    fragments: List[str] = []
    for log in sorted(staging.rglob("*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "ptxas info" in line or "Used " in line:
                fragments.append(line)
    if fragments:
        return "\n".join(fragments) + "\n"
    return _ptxas_v_via_recompile(staging, env=env)


def _ptxas_v_via_recompile(
    staging_root: Path,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Re-compile a staged ``kernel.cu`` natively to harvest ``ptxas -v``.

    The kernel includes ``torch/extension.h``; PyTorch's include
    paths are resolved at runtime via
    :func:`torch.utils.cpp_extension.include_paths`. The recompile
    invokes ``nvcc --cubin -Xptxas=-v <cuda_cflags + torch_includes>
    kernel.cu`` and parses stderr for register / spill / stack-frame
    lines.

    Key detail: must use ``--cubin`` (or ``-c``), NOT ``-ptx``. ``-ptx``
    stops at PTX generation BEFORE ptxas runs; only when the toolchain
    advances to a cubin does ``ptxas -v`` actually emit its register /
    spill diagnostics.

    Returns the harvested text. Empty string means no ptxas-info lines
    were found (treated as a surface failure by the caller).
    """
    if shutil.which("nvcc") is None:
        return ""
    candidates = sorted(
        staging_root.rglob("kernel.cu"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return ""
    kernel_cu = candidates[0]
    staging_kernel_dir = kernel_cu.parent
    build_ninja = staging_kernel_dir / "build.ninja"
    cuda_cflags: List[str] = []
    if build_ninja.is_file():
        try:
            text = build_ninja.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("cuda_cflags = "):
                cuda_cflags = _split_ninja_args(stripped[len("cuda_cflags = "):])
                break
    flags = _filter_recompile_flags(cuda_cflags)
    flags.extend(_torch_include_paths_as_flags())
    cubin_out = staging_root / "build_logs" / "__ptxas_harvest.cubin"
    cubin_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvcc", "--cubin", "-Xptxas=-v",
        *flags,
        str(kernel_cu),
        "-o", str(cubin_out),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=180.0,
            cwd=str(staging_kernel_dir),
            env=env,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    output = (proc.stderr or "") + (proc.stdout or "")
    log_path = staging_root / "build_logs" / "ptxas_recompile.log"
    try:
        log_path.write_text(output, encoding="utf-8")
    except OSError:
        pass
    lines = [
        line for line in output.splitlines()
        if "ptxas info" in line
        or "registers" in line
        or "stack frame" in line
        or "spill stores" in line
        or "spill loads" in line
    ]
    return ("\n".join(lines) + "\n") if lines else ""


def _filter_recompile_flags(cuda_cflags: List[str]) -> List[str]:
    """Filter ninja's ``cuda_cflags`` down to nvcc-recompile-safe entries.

    Drops linker-only options (``-fPIC``, ``--compiler-options …``) and
    the ``-c`` flag (we emit a cubin instead). Rewrites container-rooted
    include paths if any survived in the snapshot.
    """
    filtered: List[str] = []
    skip_next = False
    for flag in cuda_cflags:
        if skip_next:
            skip_next = False
            continue
        if flag in ("-c", "-fPIC"):
            continue
        if flag.startswith("--compiler-options"):
            if "=" not in flag and not flag.endswith("'"):
                skip_next = True
            continue
        if flag.startswith("-I/staging_static") or flag.startswith("-I/workspace"):
            continue
        filtered.append(flag)
    return filtered


def _torch_include_paths_as_flags() -> List[str]:
    """Resolve PyTorch's include paths via ``torch.utils.cpp_extension``."""
    try:
        from torch.utils.cpp_extension import include_paths  # type: ignore
    except Exception:
        return []
    flags: List[str] = []
    try:
        for path in include_paths(cuda=True):
            flags.append(f"-I{path}")
    except Exception:
        return []
    return flags


def _split_ninja_args(s: str) -> List[str]:
    """Parse a ninja args string while preserving single-quoted groups.

    ``ninja`` writes ``--compiler-options '-fPIC'`` with the quote
    grouping intact; naive ``str.split()`` would break those. We use
    ``shlex`` for the correct quoted-token splitting.
    """
    import shlex
    return shlex.split(s)


def ptxas_v_evidence_is_real(text: str) -> bool:
    """Return True when ``text`` carries real ptxas -v register/spill evidence.

    Used by :meth:`StaticSurface.parse` to reject empty / placeholder
    ``ptxas-v.txt`` files. A real ptxas-v dump contains at least one
    line matching ``ptxas info`` (the canonical prefix the tool emits)
    or one of the register/spill/stack-frame indicators.
    """
    if not text or not text.strip():
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if (
            "ptxas info" in stripped
            or "registers" in stripped
            or "stack frame" in stripped
            or "spill" in stripped
        ):
            return True
    return False
