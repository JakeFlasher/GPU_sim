"""Native host-environment fingerprint for cache identity.

Computes a SHA-256 digest of a canonical-order JSON over ten
components that meaningfully change the produced binary's runtime
behaviour:

  * ``python_version``               — interpreter major.minor.patch
  * ``torch_version``                — ``torch.__version__``
  * ``nvcc_version``                 — parsed from ``nvcc --version``
  * ``cuda_runtime_version``         — runtime CUDA version linked
                                        against by torch (or stub
                                        compile reading
                                        ``cudaRuntimeGetVersion()``)
  * ``cutlass_tree_hash``            — full-tree hash of the
                                        effective ``CUTLASS_DIR``
  * ``cudnn_version``                — frontend pkg / on-disk header
                                        cascade
  * ``sol_execbench_package_version``— package metadata / pyproject
                                        cascade
  * ``nvcc_ccbin``                   — operator's ``NVCC_CCBIN`` pin
                                        (empty string when unset)
  * ``cxx_compiler``                 — operator's ``CXX`` pin
                                        (empty string when unset)
  * ``c_compiler``                   — operator's ``CC`` pin
                                        (empty string when unset)

Strict identity: any component that the host cannot read returns the
sentinel string ``"unknown"`` from its getter. ``compute_native_env_hash``
still produces a deterministic digest in that case, but the cache layer
treats any ``"unknown"`` component as a degraded identity and refuses
replay — see :class:`DegradedNativeEnvError`. No asymmetric "unknown
matches anything" fallback exists.

The three compiler-override components participate in the cache
identity because ``BUILD_TOOLCHAIN_ALLOWED`` forwards ``NVCC_CCBIN``
/ ``CXX`` / ``CC`` into every surface subprocess that compiles the
kernel: pinning a different host compiler is a deterministic build
input the cache key MUST capture. An unset env var reads ``""`` —
a valid state ("operator did not pin"), distinct from
:data:`UNKNOWN`, so two hosts that both leave the vars unset agree
on these components.

Single source of truth: :class:`DegradedNativeEnvError` is defined
here and imported wherever the strict-rejection rule fires
(``cache.py:derive`` raises it).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1].parent
# The active vendor tree is tracked in the repository.  The explicit
# environment override remains available for an operator-managed external
# tree, but no absent Docker-era path is used as an implicit build input.
CUTLASS_DIR_DEFAULT = str(REPO_ROOT / "third_party" / "cutlass")
SOL_EXECBENCH_PYPROJECT = REPO_ROOT / "sol-execbench" / "pyproject.toml"


def default_cutlass_dir() -> Path:
    """Single source of truth for the harness's native CUTLASS default.

    Returns the tracked active CUTLASS tree at
    ``REPO_ROOT / "third_party" / "cutlass"`` (v4.6.1). The harness
    injects this default into every build subprocess (see
    :func:`baselines_harness.surfaces.subprocess_env.build_subprocess_env`)
    so a fresh checkout uses the same tree as the cache fingerprint without
    a manual operator export.

    An operator that wants a different CUTLASS tree (system install,
    B200 container path) sets ``CUTLASS_DIR`` explicitly; the
    explicit value wins over this default.
    """
    return REPO_ROOT / "third_party" / "cutlass"


def _effective_cutlass_dir() -> Path:
    """Return the CUTLASS tree the eval driver's build path will use.

    Resolves the live build path: ``CUTLASS_DIR`` from the environment
    when set, otherwise :func:`default_cutlass_dir`. The harness's
    subprocess-env builder injects the same default into every build
    subprocess, so this function and the subprocess agree on the
    tree the cache hash represents.
    """
    return Path(os.environ.get("CUTLASS_DIR", str(default_cutlass_dir())))


def _effective_cutlass_version_h() -> Path:
    """Return the ``version.h`` path at the effective CUTLASS root."""
    return _effective_cutlass_dir() / "include" / "cutlass" / "version.h"


# Build-input directory list. The vendored ``build_ext.py:46-48``
# adds ``<cutlass_dir>/include`` and ``<cutlass_dir>/tools/util/
# include`` to nvcc's include paths; the walk-fallback hashes those
# subtrees so any change to a build-input header (not just
# ``version.h``) invalidates the cache identity.
_CUTLASS_BUILD_INPUT_SUBDIRS: Tuple[str, ...] = (
    "include",
    "tools/util/include",
)


def _walk_tree_content_hash(root: Path) -> Optional[str]:
    """Walk the build-input subtrees under ``root`` and produce a
    deterministic SHA-256 over every file's relative path + content.

    The walk visits only the directories the build actually consumes
    (``include/`` and ``tools/util/include/`` per ``build_ext.py``)
    so the hash is bounded by real build inputs and runs in a few
    seconds even on a fresh-checkout CUTLASS tree.

    Returns ``None`` when none of the expected subtrees exist on
    disk under ``root``; the caller then returns
    :data:`UNKNOWN`.
    """
    targets = [root / sub for sub in _CUTLASS_BUILD_INPUT_SUBDIRS]
    existing = [t for t in targets if t.is_dir()]
    if not existing:
        return None
    digest = hashlib.sha256()
    file_count = 0
    for target in existing:
        rel_root = target.relative_to(root)
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            rel = (rel_root / path.relative_to(target)).as_posix()
            try:
                content = path.read_bytes()
            except OSError:
                # Unreadable file is a real cache-identity gap; fail
                # closed so ``cache.derive`` raises
                # ``DegradedNativeEnvError`` rather than producing a
                # silently-incomplete hash.
                return None
            rel_bytes = rel.encode("utf-8")
            digest.update(len(rel_bytes).to_bytes(8, "big"))
            digest.update(rel_bytes)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
            digest.update(b"\x00")
            file_count += 1
    if file_count == 0:
        return None
    return digest.hexdigest()

UNKNOWN = "unknown"

NATIVE_ENV_COMPONENT_ORDER: Tuple[str, ...] = (
    "python_version",
    "torch_version",
    "nvcc_version",
    "cuda_runtime_version",
    "cutlass_tree_hash",
    "cudnn_version",
    "sol_execbench_package_version",
    "nvcc_ccbin",
    "cxx_compiler",
    "c_compiler",
)


class DegradedNativeEnvError(RuntimeError):
    """Raised when one or more native-env components read ``"unknown"``.

    Strict cache identity requires every component to resolve to a real
    value before a cache key can be trusted for replay. A degraded
    fingerprint is never silently reused — callers either fix the host
    environment or run with cache reuse disabled.
    """

    def __init__(self, missing: List[str], components: Dict[str, str]) -> None:
        self.missing = list(missing)
        self.components = dict(components)
        super().__init__(
            "native env fingerprint degraded; missing components: "
            f"{sorted(missing)}; full components: {components}"
        )


def _run(cmd: List[str], timeout: float = 10.0) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"command timed out: {' '.join(cmd)}"
    return proc.returncode, proc.stdout, proc.stderr


def python_version() -> str:
    info = sys.version_info
    return f"{info.major}.{info.minor}.{info.micro}"


def torch_version() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return UNKNOWN
    version = getattr(torch, "__version__", None)
    if not version:
        return UNKNOWN
    return str(version)


def nvcc_version() -> str:
    rc, stdout, _ = _run(["nvcc", "--version"])
    if rc != 0:
        return UNKNOWN
    match = re.search(r"release (\d+\.\d+)", stdout)
    return match.group(1) if match else UNKNOWN


def cuda_runtime_version() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return UNKNOWN
    runtime = getattr(getattr(torch, "version", None), "cuda", None)
    if not runtime:
        return UNKNOWN
    return str(runtime)


def cutlass_tree_hash() -> str:
    """Hash the CUTLASS tree the build path will use.

    The eval driver compiles against the tree at
    ``os.environ["CUTLASS_DIR"]`` when supplied, otherwise the tracked
    ``REPO_ROOT/third_party/cutlass`` default. The cache identity must be
    nvcc consumes — version.h alone is not enough, because a local
    patch or checkout update can change headers under
    ``include/cutlass/`` (or ``tools/util/include/``) without
    bumping ``version.h``, and the cache would replay PASS traces
    from a different binary.

    Strategy:

    1. **Live build-input walk.** Walk ``<root>/include`` and
       ``<root>/tools/util/include`` (the directories
       ``build_ext.py`` adds to nvcc's include path) and produce a
       SHA-256 over every file's relative path + current content.  This
       deliberately does not use a Git ``HEAD`` tree: compilation consumes
       INDEX/WORKTREE bytes, so a staged or unstaged header edit must change
       the fingerprint before cache reuse or capture admission.
    2. **Checkout-stable path identity.** Resolve the effective tree.  When it
       is beneath :data:`REPO_ROOT`, label it by its resolved repo-relative
       POSIX path so two checkouts with identical build inputs share an
       identity.  An external tree retains its resolved absolute path so two
       distinct operator-managed trees never alias merely because their bytes
       currently match.  The label and live content hash are wrapped in one
       canonical-order JSON value.

    Returns :data:`UNKNOWN` when the live walk cannot produce a hash (e.g. the
    include directories do not exist on disk); ``cache.derive`` then
    raises :class:`DegradedNativeEnvError`.
    """
    root = _effective_cutlass_dir()
    tree_hash = _walk_tree_content_hash(root)
    if tree_hash is None:
        return UNKNOWN
    resolved_root = root.resolve()
    resolved_repo_root = REPO_ROOT.resolve()
    try:
        path_label = resolved_root.relative_to(resolved_repo_root).as_posix()
        path_scope = "repo_relative"
    except ValueError:
        path_label = resolved_root.as_posix()
        path_scope = "resolved_absolute"
    payload = json.dumps(
        {
            "cutlass_dir": path_label,
            "cutlass_dir_scope": path_scope,
            "tree_content_sha256": tree_hash,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_cudnn_headers() -> List[Path]:
    candidates: List[Path] = []
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(env_var)
        if base:
            candidates.append(Path(base) / "include" / "cudnn_version.h")
    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_root = Path(nvcc).resolve().parent.parent
        candidates.append(nvcc_root / "include" / "cudnn_version.h")
    candidates.extend([
        Path("/usr/local/cuda/include/cudnn_version.h"),
        Path("/usr/include/cudnn_version.h"),
        Path("/usr/include/x86_64-linux-gnu/cudnn_version.h"),
    ])
    try:
        import nvidia.cudnn  # type: ignore
        for root in getattr(nvidia.cudnn, "__path__", []):
            candidates.append(Path(root) / "include" / "cudnn_version.h")
    except Exception:
        pass
    return candidates


def _parse_cudnn_header(header: Path) -> Optional[str]:
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    major = re.search(r"#define\s+CUDNN_MAJOR\s+(\d+)", text)
    minor = re.search(r"#define\s+CUDNN_MINOR\s+(\d+)", text)
    patch = re.search(r"#define\s+CUDNN_PATCHLEVEL\s+(\d+)", text)
    if not (major and minor):
        return None
    patch_value = patch.group(1) if patch else "0"
    return f"{major.group(1)}.{minor.group(1)}.{patch_value}"


def cudnn_version() -> str:
    try:
        import nvidia.cudnn_frontend as _frontend  # type: ignore
        version = getattr(_frontend, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    for header in _candidate_cudnn_headers():
        if not header.is_file():
            continue
        parsed = _parse_cudnn_header(header)
        if parsed:
            return parsed
    return UNKNOWN


def _read_sol_execbench_pyproject_version() -> Optional[str]:
    if not SOL_EXECBENCH_PYPROJECT.is_file():
        return None
    try:
        with SOL_EXECBENCH_PYPROJECT.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        return None
    return version.strip()


def sol_execbench_package_version() -> str:
    try:
        import sol_execbench  # type: ignore
        version = getattr(sol_execbench, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        from importlib.metadata import version as pkg_version, PackageNotFoundError  # type: ignore
        try:
            return pkg_version("sol-execbench")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    parsed = _read_sol_execbench_pyproject_version()
    if parsed:
        return parsed
    return UNKNOWN


def nvcc_ccbin() -> str:
    """Operator's pinned host compiler for nvcc (``NVCC_CCBIN``).

    ``BUILD_TOOLCHAIN_ALLOWED`` forwards ``NVCC_CCBIN`` into every
    surface subprocess that compiles the kernel; an operator who
    pins a specific host compiler (e.g. on a distro whose default
    ``g++`` is newer than the installed CUDA toolkit supports) is
    selecting a deterministic build input. The cache key MUST
    capture that selection so a swap between two pinned compilers
    invalidates cached PASS evidence.

    Returns the env-var value verbatim, or ``""`` when unset. Empty
    string is a valid state — "operator did not pin a compiler" —
    and is NOT the :data:`UNKNOWN` sentinel; two hosts that both
    leave ``NVCC_CCBIN`` unset agree on this component.
    """
    return os.environ.get("NVCC_CCBIN", "")


def cxx_compiler() -> str:
    """Operator's pinned C++ compiler (``CXX``).

    Forwarded into the eval driver's build path via
    ``BUILD_TOOLCHAIN_ALLOWED``. Same semantics as
    :func:`nvcc_ccbin`: pinned values participate in the cache
    identity; an unset env reads ``""`` and does NOT trigger
    :class:`DegradedNativeEnvError`.
    """
    return os.environ.get("CXX", "")


def c_compiler() -> str:
    """Operator's pinned C compiler (``CC``).

    Forwarded into the eval driver's build path via
    ``BUILD_TOOLCHAIN_ALLOWED``. Same semantics as
    :func:`nvcc_ccbin`: pinned values participate in the cache
    identity; an unset env reads ``""`` and does NOT trigger
    :class:`DegradedNativeEnvError`.
    """
    return os.environ.get("CC", "")


_GETTERS = {
    "python_version": python_version,
    "torch_version": torch_version,
    "nvcc_version": nvcc_version,
    "cuda_runtime_version": cuda_runtime_version,
    "cutlass_tree_hash": cutlass_tree_hash,
    "cudnn_version": cudnn_version,
    "sol_execbench_package_version": sol_execbench_package_version,
    "nvcc_ccbin": nvcc_ccbin,
    "cxx_compiler": cxx_compiler,
    "c_compiler": c_compiler,
}


def native_env_components() -> Dict[str, str]:
    """Probe each component getter; return the raw component dict.

    The returned mapping always has all seven keys in
    :data:`NATIVE_ENV_COMPONENT_ORDER`. Components whose getter cannot
    resolve a value carry the sentinel ``"unknown"``; callers decide
    whether that is acceptable.
    """
    return {name: _GETTERS[name]() for name in NATIVE_ENV_COMPONENT_ORDER}


def compute_native_env_hash(components: Optional[Dict[str, str]] = None) -> str:
    """Return the SHA-256 hex digest of the canonical-order component JSON.

    ``components`` may be supplied (e.g. from a stored manifest) to
    re-derive a hash without re-probing. When omitted, the live
    component values are read.
    """
    raw = components if components is not None else native_env_components()
    ordered = {name: raw.get(name, UNKNOWN) for name in NATIVE_ENV_COMPONENT_ORDER}
    payload = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def missing_components(components: Dict[str, str]) -> List[str]:
    """Return the names of components that read ``"unknown"``."""
    return [name for name in NATIVE_ENV_COMPONENT_ORDER if components.get(name, UNKNOWN) == UNKNOWN]
