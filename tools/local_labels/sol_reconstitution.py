#!/usr/bin/env python3
"""SOL-ExecBench host-runnability determination for the sm_86 capture campaign.

Determines, on THIS host (RTX 3090 Ti, the pinned toolchain), whether each
in-scope SOL-ExecBench cell is RUNNABLE — its scaffolded baseline solution
compiles and every workload reaches the evaluator's PASSED verdict — or must be
typed-deferred. The result is stamped with a native-env fingerprint (toolchain +
CUTLASS-tree identity), because runnability is host/toolchain specific: a cell the
pre-migration ledger recorded as "done" can fail to compile under a newer host
compiler, and that is an honest, recorded outcome — not an assumption.

It drives the upstream ``sol-execbench`` CLI directly (the vendored
``tools/sol_baselines_harness/run.py`` cannot import today — a documented
incomplete-merge package rename, MERGE_PROVENANCE.md), applying the reconstitution
shims WITHOUT editing any validated harness code:
  * ``CUTLASS_DIR = third_party/cutlass``  (the real tree; the harness default
    ``REPO_ROOT/cutlass`` is a pre-migration path skew)
  * the SOL ``.venv`` interpreter executing this repository's
    ``sol_execbench.cli.main`` module after a source-root identity check (the
    editable console script is not trusted as an import-root authority)
  * benchmark data at ``tools/sol_execbench/data/benchmark`` (the default
    ``sol-execbench/data/benchmark`` is skewed)

Targets-never-features boundary: nothing here is an NCU metric. The native-env
hash is a build identity (library/compiler versions + CUTLASS tree); the
runnability verdict is the SOL-ExecBench evaluator's own compile/correctness
status, never a hardware counter.

Usage:
  .venv/bin/python tools/local_labels/sol_reconstitution.py fingerprint --out <f.json>
  .venv/bin/python tools/local_labels/sol_reconstitution.py probe \
      --external-campaign-root /absolute/owned/empty/root \
      --out /absolute/owned/empty/root/runnability.external.v2.json \
      [--outer-timeout 2000]
Exit: 0 ok; 1 a probe/emit invariant failed; 2 setup/IO error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import protected_paths  # noqa: E402

SOL_VENV_PYTHON = REPO_ROOT / "tools" / "sol_execbench" / ".venv" / "bin" / "python"
SOL_EXECBENCH_BIN = REPO_ROOT / "tools" / "sol_execbench" / ".venv" / "bin" / "sol-execbench"
SOL_PACKAGE_SOURCE_ROOT = REPO_ROOT / "tools" / "sol_execbench" / "src"
SOL_CUDA_HOME = Path("/usr/local/cuda-13.2")
SOL_COMPILER_ENVIRONMENT = {
    "CUDA_HOME": str(SOL_CUDA_HOME),
    "NVCC_CCBIN": "g++-14",
    "CC": "gcc-14",
    "CXX": "g++-14",
    "CUDAHOSTCXX": "g++-14",
}
SOL_HAZARD_VARIABLES = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
CUTLASS_DIR = REPO_ROOT / "third_party" / "cutlass"
BENCHMARK_ROOT = REPO_ROOT / "tools" / "sol_execbench" / "data" / "benchmark"
BASELINES_ROOT = REPO_ROOT / "workloads" / "sol_execbench_baselines"
COVERAGE_LEDGER = (REPO_ROOT / "tools" / "sol_baselines_harness" / "calibration"
                   / "sm86_coverage_ledger_v1.toml")
NATIVE_ENV_MODULE_ROOT = REPO_ROOT  # tools.sol_baselines_harness.native_env

# Historical full-correctness protocol recovered from the last complete SOL
# run.  These are correctness-probe settings, not performance settings: one
# warmup and three measured executions are enough to establish evaluator
# correctness while avoiding the old 10/50 repetition memory amplification.
PROBE_PROTOCOL_ID = "sol_full_correctness_v1"
PROBE_CONFIG = REPO_ROOT / "tools" / "local_labels" / "sol_full_correctness_config.v1.json"
PROBE_CONFIG_DOCUMENT = {
    "warmup_runs": 1,
    "iterations": 3,
    "benchmark_reference": False,
    "lock_clocks": False,
}
PROBE_CONFIG_SHA256 = "9ddc69a74e8ce11b4867fd1cd9afe310771e7af7a986bc1d8439dc1390ded6bf"
CLI_TIMEOUT_SECONDS = 1500
DEFAULT_OUTER_TIMEOUT_SECONDS = 2000.0
MIN_OUTER_TIMEOUT_SECONDS = 2000.0
SOL_NATIVE_ENV_FINGERPRINT = (
    REPO_ROOT / "tools/local_labels/authorities"
    / "sol_native_env_fingerprint_sm86.live.v2.json"
)
LIVE_FINGERPRINT_SCHEMA_VERSION = 2
LIVE_FINGERPRINT_ID = "sm86_sol_native_env_fingerprint_live_v2"
LIVE_NATIVE_ENV_COMPONENTS = frozenset({
    "python_version", "torch_version", "nvcc_version",
    "cuda_runtime_version", "cutlass_tree_hash", "cudnn_version",
    "sol_execbench_package_version", "nvcc_ccbin", "cxx_compiler",
    "c_compiler",
})
LIVE_FINGERPRINT_FIELDS = frozenset({
    "schema_version", "fingerprint_id", "arch", "sku",
    "native_env_hash", "components", "missing_components", "degraded",
})

RUNNABILITY_OWNER_MARKER_NAME = ".sol-runnability-owner-v1.json"
RUNNABILITY_OWNER_PURPOSE = "sm86_sol_full_correctness_campaign_input"
HISTORICAL_ARTIFACT_ID = "sm86_sol_runnability_v1"
ARTIFACT_ID = "sm86_sol_runnability_external_v2"
ARTIFACT_SCHEMA_VERSION = 2
RUNNABILITY_SCHEMA = (
    REPO_ROOT / "workloads/simbench_bw/schemas/sol_runnability.external.v2.schema.json"
)


DATA_ANNEX_ROOT = REPO_ROOT / "data" / "annex"
DATA_ANNEX_MANIFEST = REPO_ROOT / "evidence" / "manifests" / "data_annex.sha256"
DATA_ANNEX_MANIFEST_SHA256 = "f76f24cdd48a31444d1c6fe3fe349ffba91e415f62dd92d31b8d876e048deed5"
FLASHINFER_CORPUS_ROOT = DATA_ANNEX_ROOT / "flashinfer_trace"
FLASHINFER_MANIFEST_PREFIX = "./flashinfer_trace/"
FLASHINFER_CORPUS_MANIFEST_SHA256 = "b6d9282c217f71b310f2a0a47eed7746ad9f65e2b1fdfda5a1522b35ba74e2d5"
FLASHINFER_CORPUS_FILE_COUNT = 5244
FLASHINFER_CORPUS_BLOB_COUNT = 2611
FLASHINFER_CAPTURE_CONTRACT_KEYS = frozenset({
    "policy", "category", "environment_variable", "injection_policy",
    "root", "data_annex_manifest_path", "data_annex_manifest_sha256",
    "filtered_manifest_sha256", "file_count", "safetensors_count",
})

SOL_PACKAGE_ROOT = REPO_ROOT / "tools" / "sol_execbench"
SOL_PACKAGE_IDENTITY_FILES = (
    SOL_PACKAGE_ROOT / ".python-version",
    SOL_PACKAGE_ROOT / "pyproject.toml",
    SOL_PACKAGE_ROOT / "uv.lock",
)
UV_LOCK = SOL_PACKAGE_ROOT / "uv.lock"
UV_BINARY = Path("/home/jakeshea/.local/bin/uv")
UV_BINARY_SHA256 = "3b9d43bace955a409eb6f012b0932a4935534bdb54b85b9fd696cabbb15979a0"
UV_VERSION = "uv 0.10.7"

# category -> benchmark data subset dir (upstream generate.py SUBSET_MAP).
SUBSET_MAP = {"L1": "L1", "L2": "L2", "FIB": "FlashInfer-Bench", "Quant": "Quant"}

# Runnability statuses. Exactly one per cell; every non-runnable status is typed.
RUNNABLE = "runnable"
COMPILE_FAILED = "compile_failed"
CORRECTNESS_FAILED = "correctness_failed"
HOST_CAPACITY_OOM = "host_capacity_oom"
EVAL_INCOMPLETE = "eval_incomplete"
EVAL_ERROR = "eval_error"
RESOLVE_ERROR = "resolve_error"
ENVIRONMENT_INVALID = "environment_invalid"
PREREQUISITE_MISSING = "prerequisite_missing"
EVAL_TIMEOUT = "eval_timeout"
STATUSES = (RUNNABLE, COMPILE_FAILED, CORRECTNESS_FAILED, HOST_CAPACITY_OOM,
            EVAL_INCOMPLETE, EVAL_ERROR, RESOLVE_ERROR, ENVIRONMENT_INVALID,
            PREREQUISITE_MISSING, EVAL_TIMEOUT)

# How each non-runnable status maps into the campaign's closed typed-exclusion
# vocabulary when it reaches the roster authority. A compile/eval failure that is
# purely host-toolchain (not an arch gate) is calibration_inconclusive: the cell
# could not be established as capturable on this host under the pinned env.
STATUS_TO_TYPED_REASON = {
    COMPILE_FAILED: "calibration_inconclusive",
    CORRECTNESS_FAILED: "correctness_failed",
    HOST_CAPACITY_OOM: "host_capacity_deferred",
    EVAL_INCOMPLETE: "calibration_inconclusive",
    EVAL_ERROR: "calibration_inconclusive",
    RESOLVE_ERROR: "calibration_inconclusive",
    ENVIRONMENT_INVALID: "calibration_inconclusive",
    PREREQUISITE_MISSING: "calibration_inconclusive",
    EVAL_TIMEOUT: "calibration_inconclusive",
}

# CUDA-runtime OOM signatures only — a bare "out of memory" from the host
# compiler is a COMPILE failure, not a host-capacity deferral, so it must not
# match here (and the compile check runs first regardless).
_OOM_SIGNS = ("cuda out of memory", "cuda_error_out_of_memory",
              "cudaerrormemoryallocation", "torch.cuda.outofmemoryerror")
_COMPILE_SIGNS = ("compilation failed", "ninja: build stopped", "error building extension",
                  "returned non-zero exit status")
_TIMEOUT_SIGNS = ("timeoutexpired", "timed out after", "evaluation timeout",
                  "evaluation subprocess timeout")


class PrerequisiteContractError(RuntimeError):
    """A required read-only corpus/config/package contract did not verify."""


class EnvironmentContractError(RuntimeError):
    """The active execution environment does not match the certified stack."""


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_DIR",
        "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_PREFIX", "GIT_WORK_TREE",
    ):
        environment.pop(name, None)
    return environment


def source_head() -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True, env=_git_environment(),
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise PrerequisiteContractError(
            f"cannot bind SOL source HEAD: {completed.stderr[-400:]}"
        )
    return value


def source_worktree_state() -> list[dict[str, Any]]:
    """Require the live producer to execute from a completely clean worktree."""
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1", "-z",
         "--untracked-files=all"],
        check=False, capture_output=True, env=_git_environment(),
    )
    if completed.returncode != 0:
        raise PrerequisiteContractError(
            f"cannot inspect SOL source worktree: {completed.stderr[-400:]!r}"
        )
    if completed.stdout:
        rendered = completed.stdout.decode("utf-8", errors="replace").replace(
            "\0", "\n"
        )
        raise PrerequisiteContractError(
            "SOL live source worktree must be completely clean; observed:\n"
            f"{rendered}"
        )
    return []


def validate_external_campaign_root(path: Path) -> dict[str, Any]:
    """Return a sole-owner fingerprint for one real external namespace root."""
    requested = Path(path)
    if not requested.is_absolute():
        raise PrerequisiteContractError(
            "external campaign root must be an explicit absolute path"
        )
    lexical = Path(os.path.abspath(os.fspath(requested)))
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise PrerequisiteContractError(
            f"external campaign root is unavailable: {requested}: {exc}"
        ) from exc
    if lexical != resolved or requested.is_symlink() or not resolved.is_dir():
        raise PrerequisiteContractError(
            "external campaign root and every ancestor must be canonical real directories"
        )
    repository = REPO_ROOT.resolve(strict=True)
    try:
        resolved.relative_to(repository)
    except ValueError:
        pass
    else:
        raise PrerequisiteContractError(
            "external campaign root must be outside the repository"
        )
    if protected_paths.is_inside_git_worktree(resolved):
        raise PrerequisiteContractError(
            f"external campaign root is inside a Git worktree: {resolved}"
        )
    try:
        protected_paths.assert_write_allowed(
            resolved, repo_root=repository, operation="publish SOL campaign input",
        )
    except protected_paths.ProtectedPathError as exc:
        raise PrerequisiteContractError(str(exc)) from exc
    metadata = os.stat(resolved, follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise PrerequisiteContractError(
            "external campaign root must be owner-controlled and not group/world writable"
        )
    return {
        "canonical_path": resolved.as_posix(),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "uid": metadata.st_uid,
    }


def external_child_path(root_identity: Mapping[str, Any], name: str) -> Path:
    root = Path(str(root_identity.get("canonical_path", "")))
    if (
        not name or name in {".", ".."} or "/" in name or "\\" in name
        or Path(name).name != name
    ):
        raise PrerequisiteContractError(f"external campaign filename is unsafe: {name!r}")
    return root / name


def open_external_root(root_identity: Mapping[str, Any]) -> int:
    current = validate_external_campaign_root(
        Path(str(root_identity.get("canonical_path", "")))
    )
    if current != dict(root_identity):
        raise PrerequisiteContractError("external campaign root identity changed")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(current["canonical_path"], flags)
    metadata = os.fstat(descriptor)
    if (
        metadata.st_dev != current["device"]
        or metadata.st_ino != current["inode"]
        or metadata.st_uid != current["uid"]
        or stat.S_IMODE(metadata.st_mode) != current["mode"]
    ):
        os.close(descriptor)
        raise PrerequisiteContractError("external campaign root changed while opening")
    return descriptor


def external_root_names(root_identity: Mapping[str, Any]) -> set[str]:
    descriptor = open_external_root(root_identity)
    try:
        return set(os.listdir(descriptor))
    finally:
        os.close(descriptor)


def read_external_child(
    root_identity: Mapping[str, Any], name: str, *, require_readonly: bool = True,
) -> bytes:
    external_child_path(root_identity, name)
    root_fd = open_external_root(root_identity)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(name, flags, dir_fd=root_fd)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or (require_readonly and metadata.st_mode & 0o222)
        ):
            raise PrerequisiteContractError(
                f"external campaign input is not a sole-owner regular file: {name}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise PrerequisiteContractError(
            f"external campaign input is unreadable: {name}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(root_fd)


def publish_external_child(
    root_identity: Mapping[str, Any], name: str, payload: bytes,
    *, allow_identical_existing: bool = False,
) -> Path:
    path = external_child_path(root_identity, name)
    root_fd = open_external_root(root_identity)
    descriptor: int | None = None
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(name, flags, 0o400, dir_fd=root_fd)
        except FileExistsError:
            if not allow_identical_existing:
                raise PrerequisiteContractError(
                    f"refusing to overwrite existing campaign publication: {path}"
                ) from None
        else:
            created = True
            _write_all(descriptor, payload)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.fsync(root_fd)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            try:
                os.unlink(name, dir_fd=root_fd)
                os.fsync(root_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(root_fd)
    observed = read_external_child(root_identity, name)
    if observed != payload:
        raise PrerequisiteContractError(
            f"external campaign input bytes differ: {path}"
        )
    return path


def atomic_write_external_checkpoint(
    root_identity: Mapping[str, Any], name: str, payload: bytes,
) -> Path:
    """Durably replace only a mutable checkpoint through a bound dirfd."""
    path = external_child_path(root_identity, name)
    root_fd = open_external_root(root_identity)
    temporary = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    created = False
    try:
        if name in os.listdir(root_fd):
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            current_fd = os.open(name, flags, dir_fd=root_fd)
            try:
                metadata = os.fstat(current_fd)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                ):
                    raise PrerequisiteContractError(
                        f"checkpoint target is not a sole-owner regular file: {name}"
                    )
            finally:
                os.close(current_fd)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600, dir_fd=root_fd)
        created = True
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        created = False
        os.fsync(root_fd)
        return path
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(temporary, dir_fd=root_fd)
                os.fsync(root_fd)
            except OSError:
                pass
        os.close(root_fd)


def unlink_external_child(root_identity: Mapping[str, Any], name: str) -> None:
    root_fd = open_external_root(root_identity)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=root_fd)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PrerequisiteContractError(
                    f"refusing to unlink nonregular checkpoint: {name}"
                )
        finally:
            os.close(descriptor)
        os.unlink(name, dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace ``path`` without ever exposing a partial JSON artifact."""
    # Do not resolve the final component: resolving an existing symlink would
    # replace its target, which is the opposite of a safe atomic publication.
    path = path.absolute()
    if path.is_symlink():
        raise PrerequisiteContractError(
            f"refusing atomic publication through a symlink: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("exclusive publication made no write progress")
        view = view[written:]


def publish_external_bytes_exclusive(
    path: Path, payload: bytes, *, repo_root: Path = REPO_ROOT,
) -> None:
    """Durably create one immutable external campaign input.

    Emission is intentionally not idempotent: even identical existing bytes are
    refused.  Callers use their separate check mode for byte-identical reruns.
    The parent must already exist as a canonical, non-symlink directory; the
    live-authority producer creates the HEAD-stamped input namespace first.
    """
    requested = Path(path)
    if not requested.is_absolute():
        raise PrerequisiteContractError(
            "campaign publication path must be explicit and absolute"
        )
    parent = requested.parent
    try:
        resolved_parent = parent.resolve(strict=True)
        resolved_repo = Path(repo_root).resolve(strict=True)
    except OSError as exc:
        raise PrerequisiteContractError(
            f"campaign publication parent is unavailable: {parent}: {exc}"
        ) from exc
    if parent != resolved_parent or parent.is_symlink() or not parent.is_dir():
        raise PrerequisiteContractError(
            "campaign publication parent/ancestors must be canonical real directories"
        )
    if (resolved_parent / requested.name).is_relative_to(resolved_repo):
        raise PrerequisiteContractError(
            "campaign publication path must be outside the repository"
        )
    if not requested.name or requested.name in {".", ".."}:
        raise PrerequisiteContractError("campaign publication filename is invalid")
    if os.path.lexists(requested):
        raise PrerequisiteContractError(
            f"refusing to overwrite existing campaign publication: {requested}"
        )

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    parent_fd = os.open(parent, directory_flags)
    descriptor: int | None = None
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(requested.name, flags, 0o400, dir_fd=parent_fd)
        created = True
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(parent_fd)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(requested.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)


def sol_exec_env(
    environ: Mapping[str, str] | None = None,
    *,
    category: str | None = None,
    flashinfer_corpus_root: Path | None = None,
) -> dict[str, str]:
    """Construct the closed local-source CUDA/GCC reconstitution environment.

    ``FLASHINFER_TRACE_DIR`` is never inherited.  It is injected only for a FIB
    cell and only after the caller has verified the preserved corpus.  L1/L2
    probes therefore cannot accidentally depend on an ambient trace checkout.
    """
    env = dict(os.environ if environ is None else environ)
    for name, expected in SOL_COMPILER_ENVIRONMENT.items():
        if name not in env:
            continue
        observed = env[name]
        admitted = observed == expected
        if name in {"CC", "CXX", "CUDAHOSTCXX"}:
            admitted = Path(observed).name == expected
        if not admitted:
            raise RuntimeError(
                f"ambient {name}={observed!r} conflicts with pinned {expected!r}"
            )
    hazards = [name for name in SOL_HAZARD_VARIABLES if name in env]
    if hazards:
        raise RuntimeError(f"forbidden NVCC environment variables: {hazards}")
    for name in (*SOL_COMPILER_ENVIRONMENT, *SOL_HAZARD_VARIABLES,
                 "FLASHINFER_TRACE_DIR"):
        env.pop(name, None)
    env.update(SOL_COMPILER_ENVIRONMENT)
    env["CUTLASS_DIR"] = str(CUTLASS_DIR)
    path_prefixes = [str(SOL_CUDA_HOME / "bin"), str(SOL_VENV_PYTHON.parent)]
    prior_path = [
        entry for entry in env.get("PATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() not in {
            Path(prefix).resolve() for prefix in path_prefixes
        }
    ]
    env["PATH"] = os.pathsep.join([*path_prefixes, *prior_path])
    source_root = str(SOL_PACKAGE_SOURCE_ROOT.resolve())
    prior_pythonpath = [
        entry for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and Path(entry).resolve() != SOL_PACKAGE_SOURCE_ROOT.resolve()
    ]
    env["PYTHONPATH"] = os.pathsep.join([source_root, *prior_pythonpath])
    if category is not None and category not in SUBSET_MAP:
        raise EnvironmentContractError(f"unknown SOL category: {category!r}")
    if category == "FIB":
        if flashinfer_corpus_root is None:
            raise PrerequisiteContractError(
                "verified FlashInfer corpus is required for FIB evaluation"
            )
        env["FLASHINFER_TRACE_DIR"] = str(flashinfer_corpus_root.resolve())
    return env


def require_sol_execution_identity(environment: Mapping[str, str]) -> Path:
    """Fail unless nvcc, GCC 14, and the imported package are repo-local pins."""
    expected_nvcc = (SOL_CUDA_HOME / "bin" / "nvcc").resolve()
    observed_nvcc = shutil.which("nvcc", path=environment.get("PATH"))
    if (
        environment.get("CUDA_HOME") != str(SOL_CUDA_HOME)
        or observed_nvcc is None
        or Path(observed_nvcc).resolve() != expected_nvcc
        or not expected_nvcc.is_file()
        or not os.access(expected_nvcc, os.X_OK)
    ):
        raise RuntimeError(
            "SOL nvcc identity mismatch: "
            f"expected={expected_nvcc} observed={observed_nvcc!r}"
        )
    for name in ("CC", "CXX", "CUDAHOSTCXX", "NVCC_CCBIN"):
        requested = environment.get(name)
        located = shutil.which(requested or "", path=environment.get("PATH"))
        if (
            requested != SOL_COMPILER_ENVIRONMENT[name]
            or located is None
            or not Path(located).resolve().is_file()
            or not os.access(Path(located).resolve(), os.X_OK)
        ):
            raise RuntimeError(
                f"SOL compiler identity mismatch: {name}={requested!r} "
                f"resolved={located!r}"
            )
    identity_script = (
        "from pathlib import Path; import sol_execbench; "
        "print(Path(sol_execbench.__file__).resolve())"
    )
    try:
        identity = subprocess.run(
            [str(SOL_VENV_PYTHON), "-c", identity_script],
            env=dict(environment), cwd=str(REPO_ROOT), capture_output=True,
            text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"SOL package identity probe failed: {exc}") from exc
    lines = identity.stdout.strip().splitlines()
    if identity.returncode != 0 or len(lines) != 1:
        raise RuntimeError(
            "SOL package identity probe failed: "
            f"returncode={identity.returncode} stderr={identity.stderr[-400:]}"
        )
    observed_source = Path(lines[0]).resolve()
    expected_source = (
        SOL_PACKAGE_SOURCE_ROOT / "sol_execbench" / "__init__.py"
    ).resolve()
    if observed_source != expected_source:
        raise RuntimeError(
            "SOL package module identity mismatch: "
            f"expected={expected_source} observed={observed_source}"
        )
    return observed_source


def native_env_fingerprint(
    environ: Mapping[str, str] | None = None,
) -> dict:
    """Compute the native-env fingerprint by running the SOL harness's own
    ``native_env`` inside the SOL ``.venv`` (which carries torch/CUDA) with the
    CUTLASS shim. Returns {components, hash, degraded, missing}; the
    fingerprint is a build identity, not a metric."""
    script = (
        "import json,sys;"
        f"sys.path.insert(0, {str(NATIVE_ENV_MODULE_ROOT)!r});"
        "from tools.sol_baselines_harness import native_env as n;"
        "c=n.native_env_components();"
        "print(json.dumps({'components':c,'hash':n.compute_native_env_hash(c),"
        "'missing':n.missing_components(c)}))"
    )
    environment = sol_exec_env(environ)
    require_sol_execution_identity(environment)
    proc = subprocess.run([str(SOL_VENV_PYTHON), "-c", script],
                          capture_output=True, text=True,
                          env=environment,
                          cwd=str(REPO_ROOT), timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"native_env probe failed: {proc.stderr[-400:]}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload["degraded"] = bool(payload.get("missing"))
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_probe_config(path: Path = PROBE_CONFIG) -> dict[str, Any]:
    """Verify the byte-exact historical 1/3 correctness configuration."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PrerequisiteContractError(f"probe config unreadable: {path}: {exc}") from exc
    observed = _sha256_bytes(payload)
    if observed != PROBE_CONFIG_SHA256:
        raise PrerequisiteContractError(
            f"probe config SHA-256 mismatch: expected={PROBE_CONFIG_SHA256} "
            f"observed={observed} path={path}"
        )
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PrerequisiteContractError(f"probe config is invalid JSON: {exc}") from exc
    if document != PROBE_CONFIG_DOCUMENT or set(document) != set(PROBE_CONFIG_DOCUMENT):
        raise PrerequisiteContractError(
            f"probe config contract mismatch: expected={PROBE_CONFIG_DOCUMENT!r} "
            f"observed={document!r}"
        )
    return document


def sol_package_identity(
    package_root: Path = SOL_PACKAGE_ROOT,
) -> dict[str, Any]:
    """Hash the exact local SOL package/lock/source bytes used by the probe.

    The digest is the SHA-256 of sorted ``<file-sha>  <relative-path>\\n``
    rows.  Generated bytecode and ``__pycache__`` directories are excluded;
    every other regular source-tree file participates.
    """
    package_root = package_root.resolve()
    paths = [package_root / p.name for p in SOL_PACKAGE_IDENTITY_FILES]
    source_root = package_root / "src"
    if source_root.is_dir():
        paths.extend(
            path for path in source_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PrerequisiteContractError(f"SOL package identity inputs missing: {missing}")
    rows: list[bytes] = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(package_root).as_posix()):
        relative = path.relative_to(package_root).as_posix()
        rows.append(f"{_sha256_file(path)}  {relative}\n".encode("utf-8"))
    serialized = b"".join(rows)
    return {
        "sha256": _sha256_bytes(serialized),
        "file_count": len(rows),
    }


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_installed_distribution_manifest(
    manifest: Any, locked_pairs: set[tuple[str, str]]
) -> int:
    """Validate the deterministic exact-installed-file manifest.

    Returns the total number of installed files.  This pure boundary is also
    used by drift-negative tests; ``uv sync --check`` remains the independent
    lock synchronization proof.
    """
    if not isinstance(manifest, list) or not manifest:
        raise EnvironmentContractError("installed distribution manifest is empty")
    seen_names: set[str] = set()
    installed_pairs: set[tuple[str, str]] = set()
    installed_file_count = 0
    for row in manifest:
        if not isinstance(row, dict) or set(row) != {"name", "version", "files"}:
            raise EnvironmentContractError(
                "installed distribution manifest row has an invalid shape"
            )
        name, version, files = row["name"], row["version"], row["files"]
        if (
            not isinstance(name, str)
            or name != _canonical_distribution_name(name)
            or not isinstance(version, str)
            or not version
            or not isinstance(files, list)
        ):
            raise EnvironmentContractError(
                "installed distribution manifest identity is invalid"
            )
        if name in seen_names:
            raise EnvironmentContractError(
                f"duplicate installed distribution name: {name}"
            )
        seen_names.add(name)
        installed_pairs.add((name, version))
        seen_files: set[str] = set()
        for file_row in files:
            if (
                not isinstance(file_row, dict)
                or set(file_row) != {"path", "sha256"}
                or not isinstance(file_row["path"], str)
                or not file_row["path"]
                or not isinstance(file_row["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", file_row["sha256"]) is None
            ):
                raise EnvironmentContractError(
                    f"installed distribution file manifest is invalid: {name}"
                )
            if file_row["path"] in seen_files:
                raise EnvironmentContractError(
                    f"duplicate installed distribution file: "
                    f"{name}:{file_row['path']}"
                )
            seen_files.add(file_row["path"])
        installed_file_count += len(files)
    unlocked = sorted(installed_pairs - locked_pairs)
    if unlocked:
        raise EnvironmentContractError(
            f"installed distributions absent from frozen uv.lock: {unlocked}"
        )
    if installed_file_count <= 0:
        raise EnvironmentContractError(
            "installed distribution manifest contains no files"
        )
    return installed_file_count


def installed_distribution_identity(
    environment: Mapping[str, str],
    *,
    uv_binary: Path | None = None,
    lock_path: Path = UV_LOCK,
) -> dict[str, Any]:
    """Verify and bind the complete installed SOL Python distribution closure.

    ``uv sync --check --locked --no-dev`` proves the exact venv is synchronized
    to the frozen lock without changing it.  A deterministic manifest over every
    installed distribution additionally binds normalized name/version and the
    installed METADATA, RECORD, and direct-url metadata bytes.  The local SOL
    source tree is bound separately by :func:`sol_package_identity`.
    """
    production_uv = uv_binary is None
    uv_path = (UV_BINARY if production_uv else Path(uv_binary)).resolve()
    if not uv_path.is_file() or not os.access(uv_path, os.X_OK):
        raise EnvironmentContractError(f"uv executable is invalid: {uv_path}")
    observed_uv_sha256 = _sha256_file(uv_path)
    if production_uv and observed_uv_sha256 != UV_BINARY_SHA256:
        raise EnvironmentContractError(
            "pinned uv binary SHA-256 mismatch: "
            f"expected={UV_BINARY_SHA256} observed={observed_uv_sha256}"
        )
    if not lock_path.is_file():
        raise PrerequisiteContractError(f"SOL uv lock is absent: {lock_path}")

    check = subprocess.run(
        [str(uv_path), "sync", "--check", "--locked", "--no-dev",
         "--project", str(SOL_PACKAGE_ROOT)],
        capture_output=True, text=True, env=dict(environment),
        cwd=str(REPO_ROOT), timeout=120, check=False,
    )
    if check.returncode != 0:
        raise EnvironmentContractError(
            "SOL installed distribution closure differs from uv.lock: "
            f"stdout={check.stdout[-600:]} stderr={check.stderr[-600:]}"
        )

    script = r'''
import base64, hashlib, importlib.metadata as metadata, json, re
def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()
rows = []
physical_owners = {}
for dist in metadata.distributions():
    name = dist.metadata.get("Name")
    if not isinstance(name, str) or not name:
        raise SystemExit("installed distribution lacks Name metadata")
    files = []
    for declared in sorted(dist.files or (), key=lambda value: value.as_posix()):
        installed = dist.locate_file(declared)
        if not installed.is_file():
            raise SystemExit(
                f"installed distribution file absent: {name}: {declared.as_posix()}"
            )
        payload = installed.read_bytes()
        observed_sha = hashlib.sha256(payload).hexdigest()
        record_hash = declared.hash
        expected_sha = None
        if record_hash is None:
            if not declared.as_posix().endswith(".dist-info/RECORD"):
                raise SystemExit(
                    f"installed distribution file lacks RECORD hash: "
                    f"{name}: {declared.as_posix()}"
                )
        else:
            if record_hash.mode != "sha256":
                raise SystemExit(
                    f"installed distribution RECORD uses unsupported hash: "
                    f"{name}: {declared.as_posix()}: {record_hash.mode}"
                )
            padding = "=" * (-len(record_hash.value) % 4)
            expected_sha = base64.urlsafe_b64decode(
                record_hash.value + padding
            ).hex()
        physical = str(installed.resolve())
        owner = physical_owners.setdefault(physical, {
            "observed_sha256": observed_sha,
            "observed_size": len(payload),
            "claims": [],
        })
        if owner["observed_sha256"] != observed_sha:
            raise SystemExit(f"installed file changed during manifest probe: {physical}")
        owner["claims"].append({
            "distribution": canonical(name),
            "expected_sha256": expected_sha,
            "expected_size": declared.size,
            "path": declared.as_posix(),
        })
        files.append({
            "path": declared.as_posix(),
            "sha256": observed_sha,
        })
    rows.append({
        "name": canonical(name),
        "version": dist.version,
        "files": files,
    })
# Some locked NVIDIA distributions intentionally overlap paths.  The final
# installed byte must match at least one locked owner claim; requiring it to
# match every overwritten RECORD would falsely reject a deterministic uv
# environment, while accepting an arbitrary byte would miss real drift.
for physical, owner in physical_owners.items():
    hashed_claims = [claim for claim in owner["claims"]
                     if claim["expected_sha256"] is not None]
    if hashed_claims and not any(
        claim["expected_sha256"] == owner["observed_sha256"]
        and (claim["expected_size"] is None
             or claim["expected_size"] == owner["observed_size"])
        for claim in hashed_claims
    ):
        names = sorted({claim["distribution"] for claim in hashed_claims})
        raise SystemExit(
            f"installed distribution RECORD ownership mismatch: "
            f"{physical}: owners={names}"
        )
rows.sort(key=lambda row: (row["name"], row["version"]))
if len({row["name"] for row in rows}) != len(rows):
    raise SystemExit("duplicate installed distribution name")
print(json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
'''
    manifest_probe = subprocess.run(
        [str(SOL_VENV_PYTHON), "-c", script],
        capture_output=True, text=True, env=dict(environment),
        cwd=str(REPO_ROOT), timeout=120, check=False,
    )
    if manifest_probe.returncode != 0:
        raise EnvironmentContractError(
            "installed distribution manifest probe failed: "
            f"{manifest_probe.stderr[-600:]}"
        )
    try:
        manifest = json.loads(manifest_probe.stdout)
    except json.JSONDecodeError as exc:
        raise EnvironmentContractError(
            f"installed distribution manifest is invalid JSON: {exc}"
        ) from exc
    lock = tomllib.loads(lock_path.read_text())
    locked_pairs = {
        (_canonical_distribution_name(package["name"]), package["version"])
        for package in lock.get("package", [])
        if isinstance(package, dict)
        and isinstance(package.get("name"), str)
        and isinstance(package.get("version"), str)
    }
    installed_file_count = validate_installed_distribution_manifest(
        manifest, locked_pairs
    )

    version = subprocess.run(
        [str(uv_path), "--version"], capture_output=True, text=True,
        env=dict(environment), cwd=str(REPO_ROOT), timeout=30, check=False,
    )
    if version.returncode != 0 or not version.stdout.strip():
        raise EnvironmentContractError("uv version probe failed")
    observed_uv_version = version.stdout.strip()
    if production_uv and observed_uv_version != UV_VERSION:
        raise EnvironmentContractError(
            "pinned uv version mismatch: "
            f"expected={UV_VERSION!r} observed={observed_uv_version!r}"
        )
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return {
        "installed_distribution_manifest_sha256": _sha256_bytes(manifest_bytes),
        "installed_distribution_count": len(manifest),
        "installed_distribution_file_count": installed_file_count,
        "installed_distribution_record_verified": True,
        "uv_lock_sha256": _sha256_file(lock_path),
        "uv_binary_sha256": observed_uv_sha256,
        "uv_version": observed_uv_version,
    }


def verify_flashinfer_corpus(
    *,
    data_annex_root: Path = DATA_ANNEX_ROOT,
    manifest_path: Path = DATA_ANNEX_MANIFEST,
    expected_manifest_sha256: str = DATA_ANNEX_MANIFEST_SHA256,
    expected_filtered_sha256: str = FLASHINFER_CORPUS_MANIFEST_SHA256,
    expected_file_count: int = FLASHINFER_CORPUS_FILE_COUNT,
    expected_blob_count: int = FLASHINFER_CORPUS_BLOB_COUNT,
) -> dict[str, Any]:
    """Read-only verification of the preserved FlashInfer trace corpus.

    Only the ``./flashinfer_trace/`` rows from the frozen ``data_annex``
    manifest are authoritative.  The function proves exact set equality and
    hashes every file before returning the root that may be exported to FIB.
    """
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise PrerequisiteContractError(
            f"data-annex manifest unreadable: {manifest_path}: {exc}"
        ) from exc
    observed_manifest_sha = _sha256_bytes(manifest_bytes)
    if observed_manifest_sha != expected_manifest_sha256:
        raise PrerequisiteContractError(
            "data-annex manifest SHA-256 mismatch: "
            f"expected={expected_manifest_sha256} observed={observed_manifest_sha}"
        )

    filtered_lines: list[bytes] = []
    entries: dict[str, str] = {}
    prefix = FLASHINFER_MANIFEST_PREFIX.encode("ascii")
    for raw_line in manifest_bytes.splitlines(keepends=True):
        line = raw_line.rstrip(b"\r\n")
        if len(line) < 67 or line[64:66] != b"  " or not line[66:].startswith(prefix):
            continue
        digest_bytes, path_bytes = line[:64], line[66:]
        try:
            digest = digest_bytes.decode("ascii")
            manifest_name = path_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PrerequisiteContractError(
                "FlashInfer manifest row is not decodable"
            ) from exc
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise PrerequisiteContractError(
                f"FlashInfer manifest hash syntax invalid: {digest!r}"
            )
        relative = manifest_name.removeprefix("./")
        posix = PurePosixPath(relative)
        if (
            posix.is_absolute()
            or not posix.parts
            or posix.parts[0] != "flashinfer_trace"
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise PrerequisiteContractError(
                f"FlashInfer manifest path is unsafe: {manifest_name!r}"
            )
        if relative in entries:
            raise PrerequisiteContractError(
                f"duplicate FlashInfer manifest path: {relative}"
            )
        entries[relative] = digest
        filtered_lines.append(raw_line)

    filtered_bytes = b"".join(filtered_lines)
    observed_filtered_sha = _sha256_bytes(filtered_bytes)
    if observed_filtered_sha != expected_filtered_sha256:
        raise PrerequisiteContractError(
            "filtered FlashInfer manifest SHA-256 mismatch: "
            f"expected={expected_filtered_sha256} observed={observed_filtered_sha}"
        )
    blob_count = sum(name.endswith(".safetensors") for name in entries)
    if len(entries) != expected_file_count or blob_count != expected_blob_count:
        raise PrerequisiteContractError(
            "FlashInfer manifest cardinality mismatch: "
            f"files={len(entries)}/{expected_file_count} "
            f"blobs={blob_count}/{expected_blob_count}"
        )

    data_annex_root = data_annex_root.resolve()
    corpus_root = data_annex_root / "flashinfer_trace"
    if not corpus_root.is_dir() or corpus_root.is_symlink():
        raise PrerequisiteContractError(
            f"FlashInfer corpus root is absent/not a real directory: {corpus_root}"
        )
    actual: set[str] = set()
    for directory, dirnames, filenames in os.walk(corpus_root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            candidate = directory_path / dirname
            if candidate.is_symlink():
                raise PrerequisiteContractError(
                    f"FlashInfer corpus contains a symlink directory: {candidate}"
                )
        for filename in filenames:
            candidate = directory_path / filename
            mode = candidate.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise PrerequisiteContractError(
                    f"FlashInfer corpus contains a non-regular file: {candidate}"
                )
            actual.add(candidate.relative_to(data_annex_root).as_posix())
    expected = set(entries)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PrerequisiteContractError(
            "FlashInfer corpus file-set mismatch: "
            f"missing={missing[:10]!r} extra={extra[:10]!r}"
        )
    mismatches: list[str] = []
    for relative in sorted(entries):
        if _sha256_file(data_annex_root / relative) != entries[relative]:
            mismatches.append(relative)
    if mismatches:
        raise PrerequisiteContractError(
            f"FlashInfer corpus content mismatch: {mismatches[:10]!r}"
        )
    return {
        "root": str(corpus_root),
        "data_annex_manifest_sha256": observed_manifest_sha,
        "filtered_manifest_sha256": observed_filtered_sha,
        "file_count": len(entries),
        "safetensors_count": blob_count,
    }


def flashinfer_capture_contract(
    probe_contract: Mapping[str, Any], *, category: str,
) -> dict[str, Any]:
    """Project one verified probe corpus into a closed capture contract.

    All SOL categories bind the same corpus authority so an ambient
    ``FLASHINFER_TRACE_DIR`` can never change capture identity.  Only FIB is
    allowed to receive the verified root at runtime; every other category
    carries an explicit removal policy.
    """
    if category not in SUBSET_MAP:
        raise PrerequisiteContractError(
            f"unknown SOL category for FlashInfer contract: {category!r}"
        )
    corpus_root = probe_contract.get("_flashinfer_corpus_root")
    expected_root = FLASHINFER_CORPUS_ROOT.resolve()
    if (
        probe_contract.get("flashinfer_corpus_verified") is not True
        or not isinstance(corpus_root, Path)
        or corpus_root.resolve() != expected_root
    ):
        raise PrerequisiteContractError(
            "verified FlashInfer corpus operand is absent or noncanonical"
        )
    contract = {
        "policy": "verified_fib_only",
        "category": category,
        "environment_variable": "FLASHINFER_TRACE_DIR",
        "injection_policy": (
            "verified_absolute_root" if category == "FIB"
            else "forbidden_remove_ambient"
        ),
        "root": (
            FLASHINFER_CORPUS_ROOT.relative_to(REPO_ROOT).as_posix()
            if category == "FIB" else None
        ),
        "data_annex_manifest_path": DATA_ANNEX_MANIFEST.relative_to(
            REPO_ROOT
        ).as_posix(),
        "data_annex_manifest_sha256": probe_contract.get(
            "data_annex_manifest_sha256"
        ),
        "filtered_manifest_sha256": probe_contract.get(
            "flashinfer_corpus_manifest_sha256"
        ),
        "file_count": probe_contract.get("flashinfer_corpus_file_count"),
        "safetensors_count": probe_contract.get(
            "flashinfer_corpus_blob_count"
        ),
    }
    # Reuse the verifier below as the single shape/type authority without
    # rehashing the corpus already proved by build_probe_contract.
    _validate_flashinfer_capture_contract_shape(contract)
    return contract


def _validate_flashinfer_capture_contract_shape(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or set(contract) != set(
        FLASHINFER_CAPTURE_CONTRACT_KEYS
    ):
        raise PrerequisiteContractError(
            "FlashInfer capture contract has an open or incomplete shape"
        )
    value = dict(contract)
    category = value.get("category")
    if category not in SUBSET_MAP:
        raise PrerequisiteContractError("FlashInfer capture category is invalid")
    expected_policy = (
        "verified_absolute_root" if category == "FIB"
        else "forbidden_remove_ambient"
    )
    expected_root = (
        FLASHINFER_CORPUS_ROOT.relative_to(REPO_ROOT).as_posix()
        if category == "FIB" else None
    )
    if (
        value.get("policy") != "verified_fib_only"
        or value.get("environment_variable") != "FLASHINFER_TRACE_DIR"
        or value.get("injection_policy") != expected_policy
        or value.get("root") != expected_root
        or value.get("data_annex_manifest_path")
        != DATA_ANNEX_MANIFEST.relative_to(REPO_ROOT).as_posix()
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("data_annex_manifest_sha256", ""))
        ) is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("filtered_manifest_sha256", ""))
        ) is None
        or type(value.get("file_count")) is not int
        or value["file_count"] <= 0
        or type(value.get("safetensors_count")) is not int
        or value["safetensors_count"] <= 0
        or value["safetensors_count"] > value["file_count"]
    ):
        raise PrerequisiteContractError("FlashInfer capture contract is invalid")
    return value


def verify_flashinfer_capture_contract(
    contract: Mapping[str, Any], *, repo_root: Path = REPO_ROOT,
) -> Path | None:
    """Re-prove corpus bytes and return the only runtime root FIB may receive."""
    value = _validate_flashinfer_capture_contract_shape(contract)
    repo_root = repo_root.resolve()
    observed = verify_flashinfer_corpus(
        data_annex_root=repo_root / "data/annex",
        manifest_path=repo_root / value["data_annex_manifest_path"],
        expected_manifest_sha256=value["data_annex_manifest_sha256"],
        expected_filtered_sha256=value["filtered_manifest_sha256"],
        expected_file_count=value["file_count"],
        expected_blob_count=value["safetensors_count"],
    )
    if value["category"] != "FIB":
        return None
    root = (repo_root / value["root"]).resolve()
    if root != Path(observed["root"]).resolve():
        raise PrerequisiteContractError(
            "FlashInfer capture root differs from verified corpus root"
        )
    return root


def _canonical_bytes(document: Any) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _canonical_sha256(document: Mapping[str, Any] | list[Any]) -> str:
    return _sha256_bytes(_canonical_bytes(document))


def certified_native_env_hash(
    path: Path = SOL_NATIVE_ENV_FINGERPRINT,
) -> str:
    """Load the distinct live fingerprint authority; never rewrite history."""
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentContractError(
            f"certified native fingerprint is unreadable: {path}: {exc}"
        ) from exc
    components = document.get("components") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or set(document) != LIVE_FINGERPRINT_FIELDS
        or document.get("schema_version") != LIVE_FINGERPRINT_SCHEMA_VERSION
        or document.get("fingerprint_id") != LIVE_FINGERPRINT_ID
        or document.get("arch") != "sm_86"
        or document.get("sku") != "rtx_3090_ti"
        or not isinstance(components, dict)
        or set(components) != LIVE_NATIVE_ENV_COMPONENTS
        or not all(isinstance(value, str) and value and value != "unknown"
                   for value in components.values())
    ):
        raise EnvironmentContractError("certified native fingerprint shape is invalid")
    observed = document.get("native_env_hash")
    computed = _canonical_sha256(components)
    if (
        observed != computed
        or document.get("degraded") is not False
        or document.get("missing_components") != []
    ):
        raise EnvironmentContractError(
            "certified native fingerprint does not rederive cleanly"
        )
    return computed


CERTIFIED_NATIVE_ENV_HASH = certified_native_env_hash()


def probe_target_identity(cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(
        {
            (cell["category"], cell["slug"], cell["language"])
            for cell in cells
        }
    )
    if len(rows) != len(cells):
        raise PrerequisiteContractError("probe target set contains duplicates")
    serialized = [
        {"category": category, "slug": slug, "language": language}
        for category, slug, language in rows
    ]
    return {
        "probe_target_count": len(serialized),
        "probe_target_set_sha256": _canonical_sha256({"targets": serialized}),
    }


RESUME_IDENTITY_FIELDS = (
    "probe_protocol_id",
    "probe_protocol_sha256",
    "probe_config_path",
    "probe_config_sha256",
    "cli_timeout_seconds",
    "outer_timeout_seconds",
    "sol_execbench_package_sha256",
    "sol_execbench_package_file_count",
    "sol_execbench_module_path",
    "sol_execbench_module_sha256",
    "installed_distribution_manifest_sha256",
    "installed_distribution_count",
    "installed_distribution_file_count",
    "installed_distribution_record_verified",
    "uv_lock_sha256",
    "uv_binary_sha256",
    "uv_version",
    "data_annex_manifest_sha256",
    "flashinfer_corpus_manifest_sha256",
    "flashinfer_corpus_file_count",
    "flashinfer_corpus_blob_count",
    "flashinfer_corpus_verified",
    "native_env_hash",
    "probe_target_count",
    "probe_target_set_sha256",
)


def resume_identity_sha256(contract: Mapping[str, Any]) -> str:
    """Composite resume key; every protocol/package/corpus/toolchain pin binds."""
    missing = [field for field in RESUME_IDENTITY_FIELDS if field not in contract]
    if missing:
        raise PrerequisiteContractError(
            f"resume identity fields missing: {missing}"
        )
    return _canonical_sha256({field: contract[field] for field in RESUME_IDENTITY_FIELDS})


def build_probe_contract(
    *,
    native_env_hash: str,
    outer_timeout: float,
    environ: Mapping[str, str] | None = None,
    target_cells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare and identity-bind every non-GPU probe prerequisite once."""
    if outer_timeout < MIN_OUTER_TIMEOUT_SECONDS:
        raise PrerequisiteContractError(
            f"outer timeout {outer_timeout:g}s is below the required "
            f"{MIN_OUTER_TIMEOUT_SECONDS:g}s"
        )
    if native_env_hash != CERTIFIED_NATIVE_ENV_HASH:
        raise EnvironmentContractError(
            "uncertified SOL native environment: "
            f"expected={CERTIFIED_NATIVE_ENV_HASH} observed={native_env_hash}"
        )
    verify_probe_config()
    package = sol_package_identity()
    environment = sol_exec_env(environ)
    module_path = require_sol_execution_identity(environment)
    distributions = installed_distribution_identity(environment)
    corpus = verify_flashinfer_corpus()
    targets = probe_target_identity(
        list(target_cells) if target_cells is not None else in_scope_cells()
    )

    public = {
        "probe_protocol_id": PROBE_PROTOCOL_ID,
        "probe_protocol_sha256": _sha256_file(Path(__file__)),
        "probe_config_path": PROBE_CONFIG.relative_to(REPO_ROOT).as_posix(),
        "probe_config_sha256": PROBE_CONFIG_SHA256,
        "cli_timeout_seconds": CLI_TIMEOUT_SECONDS,
        "outer_timeout_seconds": outer_timeout,
        "sol_execbench_package_sha256": package["sha256"],
        "sol_execbench_package_file_count": package["file_count"],
        "sol_execbench_module_path": module_path.relative_to(REPO_ROOT).as_posix(),
        "sol_execbench_module_sha256": _sha256_file(module_path),
        **distributions,
        "data_annex_manifest_sha256": corpus["data_annex_manifest_sha256"],
        "flashinfer_corpus_manifest_sha256": corpus["filtered_manifest_sha256"],
        "flashinfer_corpus_file_count": corpus["file_count"],
        "flashinfer_corpus_blob_count": corpus["safetensors_count"],
        "flashinfer_corpus_verified": True,
        "flashinfer_corpus_error": None,
        "native_env_hash": native_env_hash,
        **targets,
    }
    public["resume_identity_sha256"] = resume_identity_sha256(public)
    return {
        **public,
        "_environment": environment,
        "_flashinfer_corpus_root": Path(corpus["root"]),
        "_validated": True,
    }


def public_probe_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the serializable identity/contract fields for the artifact."""
    return {key: value for key, value in contract.items() if not key.startswith("_")}


def require_validated_probe_contract(
    contract: Mapping[str, Any], *, outer_timeout: float
) -> None:
    if contract.get("_validated") is not True:
        raise PrerequisiteContractError("probe contract was not locally validated")
    if outer_timeout < MIN_OUTER_TIMEOUT_SECONDS:
        raise PrerequisiteContractError(
            f"outer timeout {outer_timeout:g}s is below the required "
            f"{MIN_OUTER_TIMEOUT_SECONDS:g}s"
        )
    if float(contract.get("outer_timeout_seconds", -1)) != float(outer_timeout):
        raise PrerequisiteContractError(
            "probe contract/outer-timeout mismatch: "
            f"contract={contract.get('outer_timeout_seconds')} requested={outer_timeout}"
        )
    if contract.get("probe_protocol_id") != PROBE_PROTOCOL_ID:
        raise PrerequisiteContractError("probe protocol identity mismatch")
    if contract.get("probe_config_sha256") != PROBE_CONFIG_SHA256:
        raise PrerequisiteContractError("probe configuration identity mismatch")
    if contract.get("cli_timeout_seconds") != CLI_TIMEOUT_SECONDS:
        raise PrerequisiteContractError("inner CLI timeout identity mismatch")
    if contract.get("native_env_hash") != CERTIFIED_NATIVE_ENV_HASH:
        raise EnvironmentContractError("native environment identity mismatch")
    if contract.get("flashinfer_corpus_verified") is not True:
        detail = contract.get("flashinfer_corpus_error")
        raise PrerequisiteContractError(
            str(detail) if detail else "FlashInfer corpus is not verified"
        )
    expected_resume = resume_identity_sha256(contract)
    if contract.get("resume_identity_sha256") != expected_resume:
        raise PrerequisiteContractError("probe resume identity does not rederive")


# --------------------------------------------------------------------------- #
# Pure helpers (no subprocess / no GPU) — the unit-testable core.
# --------------------------------------------------------------------------- #
def _lower(s) -> str:
    """Lowercased string form of any value (a trace status can arrive as a
    non-string; never raise on it)."""
    return str(s).lower() if s is not None else ""


def classify_cli_outcome(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    expected_workloads: int,
) -> tuple[str, str]:
    """Adjudicate one ``sol-execbench --json`` run to (status, detail).

    Fail-closed and ordered strongest-signal-first: an out-of-memory signature
    (host-capacity, the cell is a shrink candidate) before a generic compile
    failure before correctness. A clean run is ``runnable`` ONLY when the CLI
    exited 0 AND every one of the ``expected_workloads`` traces reports the
    evaluator's PASSED verdict — a nonzero exit, a missing/short trace set, or any
    non-PASSED trace each yields a distinct typed non-runnable status. Pure (no
    I/O) so every branch is unit-tested without a GPU.
    """
    text = _lower(stderr) + "\n" + _lower(stdout)
    traces = _parse_traces(stdout)
    # A build/compile failure (including a host-compiler out-of-memory) is a
    # compile failure, NOT a host-capacity deferral — check it before the
    # CUDA-runtime OOM signature.
    if any(sign in text for sign in _COMPILE_SIGNS) and not traces:
        return COMPILE_FAILED, "solution failed to compile on this host toolchain"
    if any(sign in text for sign in _OOM_SIGNS):
        return HOST_CAPACITY_OOM, "CUDA out-of-memory (host-capacity; shrink candidate)"
    if any(sign in text for sign in _TIMEOUT_SIGNS):
        return EVAL_TIMEOUT, "SOL evaluator reached its configured timeout"

    if returncode != 0 and not traces:
        return EVAL_ERROR, f"sol-execbench exited {returncode} with no parseable traces"

    if not traces:
        return EVAL_INCOMPLETE, "no evaluator traces parsed from --json output"

    if expected_workloads <= 0:
        # A benchmark with no counted workloads (missing/empty workload.jsonl)
        # can never be confirmed runnable — 'N/0 passed' is not a pass.
        return EVAL_INCOMPLETE, "benchmark declares no workloads (cannot confirm runnable)"

    statuses = [_lower(t.get("status")) for t in traces]
    if len(traces) < expected_workloads:
        return EVAL_INCOMPLETE, (
            f"only {len(traces)}/{expected_workloads} workload traces returned")
    non_passed = [s for s in statuses if s != "passed"]
    if non_passed:
        # A non-PASSED verdict that is itself an OOM/compile status routes to the
        # capacity/compile bucket; otherwise it is a correctness failure.
        if any("memory" in s or "oom" in s for s in non_passed):
            return HOST_CAPACITY_OOM, f"non-passed (memory): {sorted(set(non_passed))}"
        if any("compile" in s for s in non_passed):
            return COMPILE_FAILED, f"non-passed (compile): {sorted(set(non_passed))}"
        if any("timeout" in s for s in non_passed):
            return EVAL_TIMEOUT, f"non-passed (timeout): {sorted(set(non_passed))}"
        return CORRECTNESS_FAILED, f"non-passed workloads: {sorted(set(non_passed))}"

    if returncode != 0:
        return EVAL_ERROR, (
            f"all traces passed but CLI exited {returncode} (fail-closed)")
    return RUNNABLE, f"{len(traces)}/{expected_workloads} workloads passed"


def _parse_traces(stdout: str) -> list[dict]:
    """Extract per-workload trace dicts from the CLI's ``--json`` stdout. Accepts
    a JSON list, a top-level object carrying a trace list, or JSONL. Each trace's
    evaluator status is flattened to a top-level ``status`` key. Returns [] when
    nothing parses (a fail-closed signal, never an exception)."""
    out = (stdout or "").strip()
    if not out:
        return []
    candidates: list = []
    try:
        obj = json.loads(out)
        if isinstance(obj, list):
            candidates = obj
        elif isinstance(obj, dict):
            for key in ("traces", "results", "rows", "workloads"):
                if isinstance(obj.get(key), list):
                    candidates = obj[key]
                    break
            else:
                candidates = [obj]
    except json.JSONDecodeError:
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    traces: list[dict] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        status = c.get("status")
        if status is None:
            ev = c.get("evaluation")
            if isinstance(ev, dict):
                st = ev.get("status")
                status = st.get("value") if isinstance(st, dict) else st
        # A directly-nested status object ({"status": {"value": "PASSED"}}) is
        # unwrapped too; any surviving non-string is coerced so the classifier
        # (which lower-cases it) never sees a raw dict/list/int.
        if isinstance(status, dict):
            status = status.get("value")
        if status is not None and not isinstance(status, str):
            status = str(status)
        traces.append({"status": status, "raw": c})
    return traces


def in_scope_cells(ledger_path: Path = COVERAGE_LEDGER) -> list[dict]:
    """The 201 in-scope cells from the coverage ledger: one {slug, category,
    language} per row. This is the iteration source of truth."""
    import tomllib
    doc = tomllib.loads(ledger_path.read_text())
    cells = []
    for r in doc.get("rows", []):
        cells.append({"slug": r["slug"], "category": r["category"],
                      "language": r["language"],
                      "prior_done": bool(r.get("done")),
                      "prior_deferred_correctness": bool(r.get("deferred_correctness"))})
    return cells


def resolve_paths(cell: dict) -> tuple[Path, Path, Path]:
    """(baseline_dir, solution_json, benchmark_dir) for a cell. Raises
    FileNotFoundError if the solution or benchmark dir is absent (fail-closed —
    a resolve_error, never a silent skip)."""
    cat, slug, lang = cell["category"], cell["slug"], cell["language"]
    baseline_dir = BASELINES_ROOT / cat / slug / lang
    solution_json = baseline_dir / "solution.json"
    if not solution_json.is_file():
        raise FileNotFoundError(f"solution.json absent: {solution_json}")
    try:
        solution_document = json.loads(solution_json.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FileNotFoundError(
            f"solution.json invalid/unreadable: {solution_json}: {exc}"
        ) from exc
    if not isinstance(solution_document, dict):
        raise FileNotFoundError(f"solution.json root is not an object: {solution_json}")
    definition = solution_document.get("definition")
    if not definition:
        raise FileNotFoundError(f"solution.json missing 'definition': {solution_json}")
    subset = SUBSET_MAP.get(cat)
    if subset is None:
        raise FileNotFoundError(f"unknown category {cat!r}")
    benchmark_dir = BENCHMARK_ROOT / subset / definition
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"benchmark dir absent: {benchmark_dir}")
    return baseline_dir, solution_json, benchmark_dir


def _path_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _execution_input_path_state(path: Path) -> dict[str, Any]:
    reference = _path_reference(path)
    if path.is_symlink():
        return {
            "path": reference, "state": "invalid", "sha256": None,
            "detail": "symlink",
        }
    if not path.exists():
        return {
            "path": reference, "state": "missing", "sha256": None,
            "detail": None,
        }
    if not path.is_file():
        return {
            "path": reference, "state": "invalid", "sha256": None,
            "detail": "not_regular_file",
        }
    try:
        digest = _sha256_file(path)
    except OSError as exc:
        return {
            "path": reference, "state": "invalid", "sha256": None,
            "detail": f"unreadable_errno_{exc.errno}",
        }
    return {
        "path": reference, "state": "present", "sha256": digest,
        "detail": None,
    }


def execution_input_state(
    cell: Mapping[str, Any], *, workload_path: Path | None = None,
    definition_path: Path | None = None,
) -> dict[str, Any]:
    """Snapshot presence and exact bytes for all three execution inputs.

    The definition/workload paths are unresolved until a valid solution names
    the benchmark definition.  Recording that unresolved state is deliberate:
    if a missing or invalid solution later appears, the snapshot changes and a
    negative checkpoint cannot be reused.
    """
    category = cell.get("category")
    slug = cell.get("slug")
    language = cell.get("language")
    solution_path = BASELINES_ROOT / str(category) / str(slug) / str(language) / "solution.json"
    solution = _execution_input_path_state(solution_path)

    def unresolved(detail: str) -> dict[str, Any]:
        return {
            "path": None, "state": "unresolved", "sha256": None,
            "detail": detail,
        }

    definition_state = unresolved("solution_not_present")
    workload_state = unresolved("solution_not_present")
    if solution["state"] == "present":
        try:
            solution_document = json.loads(solution_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            definition_state = unresolved("solution_json_invalid")
            workload_state = unresolved("solution_json_invalid")
        else:
            definition_name = (
                solution_document.get("definition")
                if isinstance(solution_document, dict) else None
            )
            subset = SUBSET_MAP.get(category)
            if not isinstance(definition_name, str) or not definition_name:
                definition_state = unresolved("solution_definition_invalid")
                workload_state = unresolved("solution_definition_invalid")
            elif subset is None:
                definition_state = unresolved("category_invalid")
                workload_state = unresolved("category_invalid")
            else:
                benchmark = BENCHMARK_ROOT / subset / definition_name
                definition_state = _execution_input_path_state(
                    Path(definition_path) if definition_path is not None
                    else benchmark / "definition.json"
                )
                selected_workload = (
                    Path(workload_path) if workload_path is not None
                    else benchmark / "workload.jsonl"
                )
                workload_state = _execution_input_path_state(selected_workload)
    return {
        "definition": definition_state,
        "workload": workload_state,
        "solution": solution,
    }


def _load_workload_identities(
    benchmark_dir: Path, workload_path: Path | None = None
) -> list[dict[str, Any]]:
    """Load and bind every workload row without discarding its exact identity."""
    path = workload_path if workload_path is not None else benchmark_dir / "workload.jsonl"
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PrerequisiteContractError(f"workload file unreadable: {path}: {exc}") from exc
    identities: list[dict[str, Any]] = []
    seen_uuids: set[str] = set()
    # Split on LF only.  The retained/hash-bound row bytes include every other
    # byte, including leading/trailing spaces and a CR in a CRLF input.
    for source_line, raw_line in enumerate(payload.split(b"\n"), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith(b"//"):
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PrerequisiteContractError(
                f"invalid workload JSON at {path}:{source_line}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise PrerequisiteContractError(
                f"workload row is not an object at {path}:{source_line}"
            )
        uuid = row.get("uuid")
        axes = row.get("axes")
        if not isinstance(uuid, str) or not uuid or not isinstance(axes, dict):
            raise PrerequisiteContractError(
                f"workload identity is incomplete at {path}:{source_line}"
            )
        if uuid in seen_uuids:
            raise PrerequisiteContractError(
                f"duplicate workload UUID at {path}:{source_line}: {uuid}"
            )
        seen_uuids.add(uuid)
        identities.append({
            "workload_index": len(identities),
            "source_line": source_line,
            "uuid": uuid,
            "axes": axes,
            "workload_sha256": _sha256_bytes(raw_line),
            "workload_sha256_scope": "exact_json_line_bytes_excluding_lf_only",
        })
    if not identities:
        raise PrerequisiteContractError(f"workload file contains no rows: {path}")
    return identities


def _diagnostics_from_traces(
    identities: list[dict[str, Any]],
    stdout: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, int, int]:
    """Join full evaluator traces to workload rows by UUID, never position."""
    traces = _parse_traces(stdout)
    by_uuid: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    unexpected: list[dict[str, Any]] = []
    for trace_index, parsed in enumerate(traces):
        raw = parsed["raw"]
        workload = raw.get("workload") if isinstance(raw, dict) else None
        uuid = workload.get("uuid") if isinstance(workload, dict) else None
        if isinstance(uuid, str) and uuid:
            by_uuid.setdefault(uuid, []).append((trace_index, parsed))
        else:
            unexpected.append({
                "trace_index": trace_index,
                "observed_uuid": None,
                "status": parsed.get("status"),
                "trace": raw,
                "problem": "trace_missing_workload_uuid",
            })

    diagnostics: list[dict[str, Any]] = []
    expected_uuids = {identity["uuid"] for identity in identities}
    passed = 0
    for identity in identities:
        matches = by_uuid.get(identity["uuid"], [])
        diagnostic = {
            **identity,
            "diagnostic_present": len(matches) == 1,
            "trace_index": None,
            "status": None,
            "trace": None,
        }
        if len(matches) == 1:
            trace_index, parsed = matches[0]
            diagnostic.update(
                trace_index=trace_index,
                status=parsed.get("status"),
                trace=parsed["raw"],
            )
            if _lower(parsed.get("status")) == "passed":
                passed += 1
        elif len(matches) > 1:
            for trace_index, parsed in matches:
                unexpected.append({
                    "trace_index": trace_index,
                    "observed_uuid": identity["uuid"],
                    "status": parsed.get("status"),
                    "trace": parsed["raw"],
                    "problem": "duplicate_trace_uuid",
                })
        diagnostics.append(diagnostic)
    for uuid, matches in by_uuid.items():
        if uuid in expected_uuids:
            continue
        for trace_index, parsed in matches:
            unexpected.append({
                "trace_index": trace_index,
                "observed_uuid": uuid,
                "status": parsed.get("status"),
                "trace": parsed["raw"],
                "problem": "unexpected_trace_uuid",
            })
    unexpected.sort(key=lambda row: row["trace_index"])
    complete = (
        len(traces) == len(identities)
        and not unexpected
        and all(row["diagnostic_present"] for row in diagnostics)
    )
    return diagnostics, unexpected, complete, len(traces), passed


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _path_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise PrerequisiteContractError(f"execution input is not a regular file: {path}")
    return {"path": _path_reference(path), "sha256": _sha256_file(resolved)}


def _finish_record(
    rec: dict[str, Any],
    identities: list[dict[str, Any]],
    *,
    status: str,
    detail: str,
    command: list[str] | None = None,
    returncode: int | None = None,
    stdout: str | bytes | None = None,
    stderr: str | bytes | None = None,
    cli_timeout_seconds: int = CLI_TIMEOUT_SECONDS,
    outer_timeout_seconds: float = DEFAULT_OUTER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    full_stdout, full_stderr = _as_text(stdout), _as_text(stderr)
    diagnostics, unexpected, complete, returned, passed = _diagnostics_from_traces(
        identities, full_stdout
    )
    rec.update(
        status=status,
        detail=detail,
        expected_workloads=len(identities) if identities else None,
        returned_workloads=returned,
        passed_workloads=passed,
        diagnostics_complete=complete,
        workload_diagnostics=diagnostics,
        unexpected_workload_diagnostics=unexpected,
        command=command,
        returncode=returncode,
        cli_timeout_seconds=cli_timeout_seconds,
        outer_timeout_seconds=outer_timeout_seconds,
        stdout=full_stdout,
        stderr=full_stderr,
        stdout_sha256=_sha256_bytes(full_stdout.encode("utf-8")),
        stderr_sha256=_sha256_bytes(full_stderr.encode("utf-8")),
        # Compatibility field for older readers.  It is no longer the only
        # retained diagnostic: full stdout/stderr and full traces are above.
        stderr_tail=full_stderr[-400:] if full_stderr else None,
    )
    return rec


def run_certified(
    cell: dict,
    timeout: float = DEFAULT_OUTER_TIMEOUT_SECONDS,
    *,
    probe_contract: Mapping[str, Any] | None = None,
    workload_path: Path | None = None,
    definition_path: Path | None = None,
) -> dict:
    """Run native or proxy workloads through one certified execution contract.

    Never raises: resolution, prerequisite, environment, timeout, evaluator,
    and numerical failures remain distinct typed records.
    """
    input_state = execution_input_state(
        cell, workload_path=workload_path, definition_path=definition_path
    )
    rec = {"slug": cell["slug"], "category": cell["category"],
           "language": cell["language"],
           "prior_done": cell.get("prior_done"),
           "prior_deferred_correctness": cell.get("prior_deferred_correctness"),
           "execution_kind": "proxy" if workload_path is not None else "native",
           "definition_input": None, "workload_input": None,
           "solution_input": None, "execution_contract_sha256": None,
           "execution_input_state": input_state,
           "execution_input_state_sha256": _canonical_sha256(input_state)}
    if definition_path is not None and workload_path is None:
        return _finish_record(
            rec, [], status=PREREQUISITE_MISSING,
            detail="definition override requires a workload override",
            outer_timeout_seconds=timeout,
        )
    try:
        _, solution_json, benchmark_dir = resolve_paths(cell)
        selected_definition = (
            Path(definition_path) if definition_path is not None
            else benchmark_dir / "definition.json"
        )
        selected_workload = (
            Path(workload_path).resolve() if workload_path is not None
            else benchmark_dir / "workload.jsonl"
        )
        rec.update(
            definition_input=_path_identity(selected_definition),
            workload_input=_path_identity(selected_workload),
            solution_input=_path_identity(solution_json),
        )
        if execution_input_state(
            cell, workload_path=workload_path, definition_path=definition_path
        ) != input_state:
            raise PrerequisiteContractError(
                "execution input presence/bytes changed during preparation"
            )
    except FileNotFoundError as exc:
        return _finish_record(
            rec, [], status=RESOLVE_ERROR, detail=str(exc),
            outer_timeout_seconds=timeout,
        )
    except PrerequisiteContractError as exc:
        return _finish_record(
            rec, [], status=PREREQUISITE_MISSING, detail=str(exc),
            outer_timeout_seconds=timeout,
        )
    try:
        identities = _load_workload_identities(benchmark_dir, selected_workload)
    except PrerequisiteContractError as exc:
        return _finish_record(
            rec, [], status=PREREQUISITE_MISSING, detail=str(exc),
            outer_timeout_seconds=timeout,
        )

    try:
        if probe_contract is None:
            fp = native_env_fingerprint()
            probe_contract = build_probe_contract(
                native_env_hash=fp["hash"], outer_timeout=timeout
            )
        require_validated_probe_contract(probe_contract, outer_timeout=timeout)
        environment = sol_exec_env(
            probe_contract.get("_environment"),
            category=cell["category"],
            flashinfer_corpus_root=probe_contract.get("_flashinfer_corpus_root"),
        )
        observed_module = require_sol_execution_identity(environment)
        expected_module = probe_contract.get("sol_execbench_module_path")
        if expected_module != observed_module.relative_to(REPO_ROOT).as_posix():
            raise PrerequisiteContractError("execution module differs from probe contract")
        if probe_contract.get("sol_execbench_module_sha256") != _sha256_file(
            observed_module
        ):
            raise PrerequisiteContractError("execution module bytes differ from probe contract")
        verify_probe_config()
        if execution_input_state(
            cell, workload_path=workload_path, definition_path=definition_path
        ) != input_state:
            raise PrerequisiteContractError(
                "execution input presence/bytes changed before launch"
            )
        rec["execution_contract_sha256"] = probe_contract["resume_identity_sha256"]
    except EnvironmentContractError as exc:
        return _finish_record(
            rec, identities, status=ENVIRONMENT_INVALID, detail=str(exc),
            outer_timeout_seconds=timeout,
        )
    except PrerequisiteContractError as exc:
        return _finish_record(
            rec, identities, status=PREREQUISITE_MISSING, detail=str(exc),
            outer_timeout_seconds=timeout,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _finish_record(
            rec, identities, status=PREREQUISITE_MISSING,
            detail=f"invalid probe contract: {exc}",
            outer_timeout_seconds=timeout,
        )
    except RuntimeError as exc:
        return _finish_record(
            rec, identities, status=ENVIRONMENT_INVALID,
            detail=f"environment refusal: {exc}",
            outer_timeout_seconds=timeout,
        )

    execution_command = [
        str(SOL_VENV_PYTHON), "-m", "sol_execbench.cli.main",
        "--definition", str(selected_definition.resolve()),
        "--workload", str(selected_workload),
        "--solution", str(solution_json),
        "--json",
        "--config", str(PROBE_CONFIG),
        "--timeout", str(CLI_TIMEOUT_SECONDS),
    ]
    command = [
        "tools/sol_execbench/.venv/bin/python", "-m", "sol_execbench.cli.main",
        "--definition", _path_reference(selected_definition),
        "--workload", _path_reference(selected_workload),
        "--solution", _path_reference(solution_json),
        "--json", "--config", PROBE_CONFIG.relative_to(REPO_ROOT).as_posix(),
        "--timeout", str(CLI_TIMEOUT_SECONDS),
    ]
    try:
        proc = subprocess.run(
            execution_command,
            capture_output=True, text=True, env=environment,
            cwd=str(REPO_ROOT), timeout=timeout)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return _finish_record(
            rec, identities, status=EVAL_TIMEOUT,
            detail=f"outer evaluation timeout after {timeout:g}s",
            command=command, stdout=exc.stdout, stderr=exc.stderr,
            outer_timeout_seconds=timeout,
        )
    except OSError as exc:
        return _finish_record(
            rec, identities, status=ENVIRONMENT_INVALID,
            detail=f"subprocess spawn failed: {exc}", command=command,
            outer_timeout_seconds=timeout,
        )
    try:
        status, detail = classify_cli_outcome(
            returncode=returncode, stdout=stdout, stderr=stderr,
            expected_workloads=len(identities))
    except Exception as exc:  # noqa: BLE001 — one cell never aborts the sweep
        status, detail = EVAL_ERROR, f"classification error: {exc}"
    finished = _finish_record(
        rec, identities, status=status, detail=detail, command=command,
        returncode=returncode, stdout=stdout, stderr=stderr,
        outer_timeout_seconds=timeout,
    )
    if finished["status"] == RUNNABLE and not finished["diagnostics_complete"]:
        finished["status"] = EVAL_INCOMPLETE
        finished["detail"] = (
            "evaluator statuses passed but workload UUID diagnostics were incomplete"
        )
    return finished


def run_one(
    cell: dict,
    timeout: float = DEFAULT_OUTER_TIMEOUT_SECONDS,
    *,
    probe_contract: Mapping[str, Any] | None = None,
) -> dict:
    """Compatibility wrapper for a native full-workload certified run."""
    return run_certified(
        cell, timeout=timeout, probe_contract=probe_contract, workload_path=None
    )


def summarize(records: list[dict]) -> dict:
    by_status = {s: 0 for s in STATUSES}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return {
        "total": len(records),
        "runnable": by_status[RUNNABLE],
        "by_status": by_status,
    }


def _identity_matches_current(identity: Any) -> bool:
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        return False
    path = identity.get("path")
    if not isinstance(path, str) or not path:
        return False
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        return _path_identity(candidate) == identity
    except PrerequisiteContractError:
        return False


def _record_input_state_matches(
    record: Mapping[str, Any], cell: Mapping[str, Any],
    *, workload_path: Path | None = None,
    definition_path: Path | None = None,
) -> bool:
    state = record.get("execution_input_state")
    if not isinstance(state, dict) or set(state) != {
        "definition", "workload", "solution"
    }:
        return False
    if record.get("execution_input_state_sha256") != _canonical_sha256(state):
        return False
    if state != execution_input_state(
        cell, workload_path=workload_path, definition_path=definition_path
    ):
        return False
    field_map = {
        "definition": "definition_input",
        "workload": "workload_input",
        "solution": "solution_input",
    }
    for state_name, identity_name in field_map.items():
        item = state.get(state_name)
        if not isinstance(item, dict):
            return False
        identity = record.get(identity_name)
        if item.get("state") == "present":
            if identity != {"path": item.get("path"), "sha256": item.get("sha256")}:
                return False
        elif identity is not None:
            return False
    return True


def _require_certified_record_command(
    record: Mapping[str, Any], *, resume_identity: str, key: tuple[Any, ...]
) -> None:
    command = record.get("command")
    if command is None:
        return
    command_paths = (
        [command[4], command[6], command[8]]
        if isinstance(command, list) and len(command) == 14 else []
    )
    inputs = [
        record.get("definition_input"), record.get("workload_input"),
        record.get("solution_input"),
    ]
    if (
        not isinstance(command, list)
        or len(command) != 14
        or command[:4] != [
            "tools/sol_execbench/.venv/bin/python", "-m",
            "sol_execbench.cli.main", "--definition"
        ]
        or command[5] != "--workload"
        or command[7] != "--solution"
        or command[9:14] != [
            "--json", "--config",
            PROBE_CONFIG.relative_to(REPO_ROOT).as_posix(), "--timeout",
            str(CLI_TIMEOUT_SECONDS),
        ]
        or len(command_paths) != 3
        or any(
            not isinstance(identity, dict)
            or identity.get("path") != observed_path
            for identity, observed_path in zip(inputs, command_paths)
        )
        or record.get("execution_contract_sha256") != resume_identity
        or record.get("outer_timeout_seconds", 0) < MIN_OUTER_TIMEOUT_SECONDS
    ):
        raise PrerequisiteContractError(
            f"cell command is outside the certified execution API: {key}"
        )


def _checkpoint_record_is_resumable(
    record: Any, expected_cell: Mapping[str, Any], contract: Mapping[str, Any]
) -> bool:
    try:
        if not isinstance(record, dict):
            return False
        key = _cell_key(record)
        expected_key = (
            expected_cell["category"], expected_cell["slug"],
            expected_cell["language"],
        )
        if (
            key != expected_key
            or record.get("status") not in STATUSES
            or record.get("execution_kind") != "native"
            or record.get("cli_timeout_seconds") != CLI_TIMEOUT_SECONDS
            or record.get("outer_timeout_seconds")
            != contract.get("outer_timeout_seconds")
            or not isinstance(record.get("workload_diagnostics"), list)
            or not isinstance(record.get("unexpected_workload_diagnostics"), list)
            or not _record_input_state_matches(record, expected_cell)
        ):
            return False
        for field in ("definition_input", "workload_input", "solution_input"):
            identity = record.get(field)
            if identity is not None and not _identity_matches_current(identity):
                return False
        _require_certified_record_command(
            record, resume_identity=contract["resume_identity_sha256"], key=key
        )
        if record.get("status") == RUNNABLE:
            diagnostics = record["workload_diagnostics"]
            expected = record.get("expected_workloads")
            uuids = [row.get("uuid") for row in diagnostics if isinstance(row, dict)]
            if (
                record.get("command") is None
                or record.get("diagnostics_complete") is not True
                or not isinstance(expected, int)
                or expected <= 0
                or record.get("returned_workloads") != expected
                or record.get("passed_workloads") != expected
                or len(diagnostics) != expected
                or len(uuids) != expected
                or len(set(uuids)) != expected
                or record.get("unexpected_workload_diagnostics") != []
                or any(
                    row.get("diagnostic_present") is not True
                    or _lower(row.get("status")) != "passed"
                    or row.get("workload_sha256_scope")
                    != "exact_json_line_bytes_excluding_lf_only"
                    for row in diagnostics
                )
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError, PrerequisiteContractError):
        return False


def _validated_checkpoint_records(
    artifact: Any, *, contract: Mapping[str, Any],
    external_contract: Mapping[str, Any],
    expected_by_key: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Validate every checkpoint byte/row or refuse resume without salvage."""
    import jsonschema
    if not isinstance(artifact, dict):
        raise PrerequisiteContractError("runnability checkpoint root is not an object")
    required = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": CERTIFIED_NATIVE_ENV_HASH,
        "native_env_degraded": False,
        "publication_state": "checkpoint",
        "in_scope_total": 201,
        "probe_target_count": 201,
        "status_to_typed_reason": STATUS_TO_TYPED_REASON,
        **public_probe_contract(contract),
        **{
            key: value for key, value in external_contract.items()
            if not key.startswith("_")
        },
    }
    for field, expected in required.items():
        if artifact.get(field) != expected:
            raise PrerequisiteContractError(
                f"runnability checkpoint contract drift at {field}"
            )
    expected_top_keys = set(required) | {"probed", "pending", "summary", "cells"}
    if set(artifact) != expected_top_keys:
        raise PrerequisiteContractError(
            "runnability checkpoint has unknown or missing top-level fields: "
            f"{sorted(set(artifact) ^ expected_top_keys)}"
        )
    rows = artifact.get("cells")
    if not isinstance(rows, list):
        raise PrerequisiteContractError("runnability checkpoint cells are not an array")
    schema = json.loads(RUNNABILITY_SCHEMA.read_text())
    cell_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "#/$defs/cell",
        "$defs": schema["$defs"],
    }
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PrerequisiteContractError(
                f"runnability checkpoint cell {index} is not an object"
            )
        try:
            key = _cell_key(row)
        except KeyError as exc:
            raise PrerequisiteContractError(
                f"runnability checkpoint cell {index} lacks identity"
            ) from exc
        if key in records:
            raise PrerequisiteContractError(
                f"duplicate runnability checkpoint cell: {key}"
            )
        try:
            jsonschema.validate(row, cell_schema)
        except jsonschema.ValidationError as exc:
            raise PrerequisiteContractError(
                f"runnability checkpoint cell violates external-v2 schema: {key}"
            ) from exc
        expected = expected_by_key.get(key)
        if expected is None or not _checkpoint_record_is_resumable(
            row, expected, contract
        ):
            raise PrerequisiteContractError(
                f"stale or invalid runnability checkpoint cell: {key}"
            )
        records[key] = row
    if artifact.get("probed") != len(rows) or artifact.get("pending") != 201 - len(rows):
        raise PrerequisiteContractError(
            "runnability checkpoint probed/pending accounting is invalid"
        )
    if artifact.get("summary") != summarize(rows):
        raise PrerequisiteContractError("runnability checkpoint summary does not rederive")
    return records


def validate_runnability_external_owner(
    artifact: Mapping[str, Any], cells: list[dict[str, Any]], *,
    external_campaign_root: Path,
) -> None:
    owner = artifact.get("external_campaign_owner")
    if not isinstance(owner, dict):
        raise PrerequisiteContractError(
            "runnability external owner binding is absent"
        )
    root_identity = validate_external_campaign_root(external_campaign_root)
    output_name = artifact.get("external_output_name")
    checkpoint_name = artifact.get("external_checkpoint_name")
    if (
        not isinstance(output_name, str)
        or not isinstance(checkpoint_name, str)
        or owner.get("name") != RUNNABILITY_OWNER_MARKER_NAME
    ):
        raise PrerequisiteContractError(
            "runnability external output/owner name binding is invalid"
        )
    external_child_path(root_identity, output_name)
    external_child_path(root_identity, checkpoint_name)
    worktree_state = source_worktree_state()
    source_closure = probe_source_closure(cells)
    for field, expected in {
        "source_commit": source_head(),
        "source_worktree_state": worktree_state,
        "source_worktree_state_sha256": _canonical_sha256(worktree_state),
        **source_closure,
    }.items():
        if artifact.get(field) != expected:
            raise PrerequisiteContractError(
                f"runnability external source identity drift at {field}"
            )
    document = _runnability_owner_document(
        contract=artifact, source_closure=source_closure,
        output_name=output_name,
        checkpoint_name=checkpoint_name,
        source_commit=artifact["source_commit"],
        worktree_state=worktree_state,
    )
    expected_payload = _canonical_bytes(document)
    observed_payload = read_external_child(
        root_identity, RUNNABILITY_OWNER_MARKER_NAME
    )
    if observed_payload != expected_payload or owner != {
        "name": RUNNABILITY_OWNER_MARKER_NAME,
        "sha256": _sha256_bytes(expected_payload),
        "owner_seal_sha256": document["owner_seal_sha256"],
    }:
        raise PrerequisiteContractError(
            "runnability external owner marker does not rederive"
        )


def validate_completed_runnability(
    artifact: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any] | None = None,
    expected_cells: list[dict[str, Any]] | None = None,
    require_external_owner: bool = False,
    external_campaign_root: Path | None = None,
) -> None:
    """Reject anything other than the complete current 201-cell authority."""
    import jsonschema
    cells = list(expected_cells) if expected_cells is not None else in_scope_cells()
    target_identity = probe_target_identity(cells)
    if len(cells) != 201:
        raise PrerequisiteContractError(
            f"current SOL scope is not the required 201 cells: {len(cells)}"
        )
    artifact_pair = (artifact.get("schema_version"), artifact.get("artifact_id"))
    if artifact_pair == (ARTIFACT_SCHEMA_VERSION, ARTIFACT_ID):
        external_artifact = True
    elif artifact_pair == (1, HISTORICAL_ARTIFACT_ID):
        external_artifact = False
    else:
        raise PrerequisiteContractError(
            "runnability schema/artifact identity is unsupported: "
            f"{artifact_pair!r}"
        )
    if require_external_owner and not external_artifact:
        raise PrerequisiteContractError(
            "fresh runnability consumer requires the external-v2 authority"
        )
    schema_path = (
        RUNNABILITY_SCHEMA if external_artifact
        else REPO_ROOT / "workloads/simbench_bw/schemas/sol_runnability.schema.json"
    )
    try:
        jsonschema.validate(dict(artifact), json.loads(schema_path.read_text()))
    except (OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        raise PrerequisiteContractError(
            f"runnability artifact violates {schema_path.name}: {exc}"
        ) from exc
    required_identity = {
        "schema_version": (
            ARTIFACT_SCHEMA_VERSION if external_artifact else 1
        ),
        "artifact_id": (
            ARTIFACT_ID if external_artifact else HISTORICAL_ARTIFACT_ID
        ),
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": CERTIFIED_NATIVE_ENV_HASH,
        "native_env_degraded": False,
        "publication_state": "complete",
        "in_scope_total": 201,
        "probed": 201,
        "pending": 0,
        "flashinfer_corpus_verified": True,
        **target_identity,
    }
    for field, expected in required_identity.items():
        if artifact.get(field) != expected:
            raise PrerequisiteContractError(
                f"runnability {field} mismatch: expected={expected!r} "
                f"observed={artifact.get(field)!r}"
            )
    if artifact.get("status_to_typed_reason") != STATUS_TO_TYPED_REASON:
        raise PrerequisiteContractError("runnability typed-reason mapping drift")
    if artifact.get("resume_identity_sha256") != resume_identity_sha256(artifact):
        raise PrerequisiteContractError("runnability resume identity does not rederive")
    if require_external_owner:
        if external_campaign_root is None:
            raise PrerequisiteContractError(
                "external campaign root is required to validate runnability owner"
            )
        validate_runnability_external_owner(
            artifact, cells, external_campaign_root=external_campaign_root
        )
    if expected_contract is not None:
        expected_public = public_probe_contract(expected_contract)
        for field, expected in expected_public.items():
            if artifact.get(field) != expected:
                raise PrerequisiteContractError(
                    f"runnability probe contract drift at {field}"
                )

    records = artifact.get("cells")
    if not isinstance(records, list) or len(records) != 201:
        raise PrerequisiteContractError("runnability must contain exactly 201 cells")
    expected_keys = {
        (cell["category"], cell["slug"], cell["language"]) for cell in cells
    }
    expected_by_key = {
        (cell["category"], cell["slug"], cell["language"]): cell
        for cell in cells
    }
    observed_keys: set[tuple[str, str, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise PrerequisiteContractError("runnability cell is not an object")
        key = (record.get("category"), record.get("slug"), record.get("language"))
        if key in observed_keys:
            raise PrerequisiteContractError(f"duplicate runnability cell: {key}")
        observed_keys.add(key)
        expected_cell = expected_by_key.get(key)
        if expected_cell is None or not _record_input_state_matches(
            record, expected_cell
        ):
            raise PrerequisiteContractError(
                f"runnability execution-input state drift: {key}"
            )
        if record.get("status") not in STATUSES:
            raise PrerequisiteContractError(f"unknown runnability status: {key}")
        if record.get("execution_kind") != "native":
            raise PrerequisiteContractError(
                f"canonical runnability contains non-native execution: {key}"
            )
        for field in ("definition_input", "workload_input", "solution_input"):
            identity = record.get(field)
            if identity is not None and not _identity_matches_current(identity):
                raise PrerequisiteContractError(
                    f"runnability input identity drift at {field}: {key}"
                )
        diagnostics = record.get("workload_diagnostics")
        if not isinstance(diagnostics, list):
            raise PrerequisiteContractError(f"missing workload diagnostics: {key}")
        if any(not isinstance(row, dict) for row in diagnostics):
            raise PrerequisiteContractError(
                f"invalid workload diagnostic object: {key}"
            )
        uuids = [row.get("uuid") for row in diagnostics]
        if (
            any(not isinstance(uuid, str) or not uuid for uuid in uuids)
            or len(uuids) != len(set(uuids))
            or any(
                row.get("workload_sha256_scope")
                != "exact_json_line_bytes_excluding_lf_only"
                for row in diagnostics
            )
        ):
            raise PrerequisiteContractError(f"duplicate workload diagnostic UUID: {key}")
        if record.get("status") == RUNNABLE:
            expected = record.get("expected_workloads")
            if (
                record.get("diagnostics_complete") is not True
                or not isinstance(expected, int)
                or expected <= 0
                or record.get("returned_workloads") != expected
                or record.get("passed_workloads") != expected
                or len(diagnostics) != expected
                or record.get("unexpected_workload_diagnostics") != []
                or any(
                    row.get("diagnostic_present") is not True
                    or _lower(row.get("status")) != "passed"
                    for row in diagnostics
                )
                or record.get("command") is None
            ):
                raise PrerequisiteContractError(
                    f"runnable cell lacks a complete UUID/PASSED bijection: {key}"
                )
        command = record.get("command")
        if command is not None:
            _require_certified_record_command(
                record,
                resume_identity=artifact["resume_identity_sha256"],
                key=key,
            )
    if observed_keys != expected_keys:
        raise PrerequisiteContractError(
            "runnability target-set mismatch: "
            f"missing={sorted(expected_keys - observed_keys)[:5]} "
            f"extra={sorted(observed_keys - expected_keys)[:5]}"
        )
    expected_summary = summarize(records)
    if artifact.get("summary") != expected_summary:
        raise PrerequisiteContractError("runnability summary does not rederive")


def load_completed_runnability(
    path: Path, *, expected_contract: Mapping[str, Any] | None = None,
    require_external_owner: bool = False,
    external_campaign_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load exact bytes and validate the complete authority for read-only consumers."""
    if require_external_owner:
        if external_campaign_root is None:
            raise PrerequisiteContractError(
                "external campaign root is required for runnability load"
            )
        root_identity = validate_external_campaign_root(external_campaign_root)
        expected_path = external_child_path(root_identity, path.name)
        if path.absolute() != expected_path:
            raise PrerequisiteContractError(
                "runnability path is not a direct child of its owner root"
            )
        payload = read_external_child(root_identity, path.name)
    else:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise PrerequisiteContractError(
                f"completed runnability authority is unreadable: {path}: {exc}"
            ) from exc
    try:
        artifact = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PrerequisiteContractError(
            f"completed runnability authority is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(artifact, dict):
        raise PrerequisiteContractError("runnability authority root is not an object")
    validate_completed_runnability(
        artifact, expected_contract=expected_contract,
        require_external_owner=require_external_owner,
        external_campaign_root=external_campaign_root,
    )
    if require_external_owner and artifact.get("external_output_name") != path.name:
        raise PrerequisiteContractError(
            "loaded runnability filename differs from its sealed owner output name"
        )
    return artifact, _sha256_bytes(payload)


FINGERPRINT_ID = LIVE_FINGERPRINT_ID


def _cell_key(rec: dict) -> tuple:
    return (rec["category"], rec["slug"], rec["language"])


def emit_fingerprint(out: Path) -> dict:
    fp = native_env_fingerprint()
    artifact = {
        "schema_version": LIVE_FINGERPRINT_SCHEMA_VERSION,
        "fingerprint_id": FINGERPRINT_ID,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": fp["hash"],
        "components": fp["components"],
        "missing_components": fp["missing"],
        "degraded": fp["degraded"],
    }
    atomic_write_json(out, artifact)
    return artifact


def probe_source_closure(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind every target input plus the non-package producer/config authority."""
    fixed = (
        Path(__file__),
        PROBE_CONFIG,
        COVERAGE_LEDGER,
        DATA_ANNEX_MANIFEST,
        REPO_ROOT / "tools/sol_baselines_harness/native_env.py",
        RUNNABILITY_SCHEMA,
    )
    fixed_rows = []
    for path in fixed:
        if path.is_symlink() or not path.is_file():
            raise PrerequisiteContractError(
                f"SOL probe source-closure input is nonregular: {path}"
            )
        fixed_rows.append({
            "path": _path_reference(path),
            "sha256": _sha256_file(path),
        })
    target_rows = []
    for cell in sorted(
        cells, key=lambda row: (row["category"], row["slug"], row["language"])
    ):
        target_rows.append({
            "category": cell["category"],
            "slug": cell["slug"],
            "language": cell["language"],
            "execution_input_state": execution_input_state(cell),
        })
    closure = {"fixed_files": fixed_rows, "targets": target_rows}
    return {
        "probe_source_closure": closure,
        "probe_source_closure_file_count": len(fixed_rows),
        "probe_source_closure_target_count": len(target_rows),
        "probe_source_closure_sha256": _canonical_sha256(closure),
    }


def _runnability_owner_document(
    *, contract: Mapping[str, Any], source_closure: Mapping[str, Any],
    output_name: str, checkpoint_name: str, source_commit: str,
    worktree_state: list[dict[str, Any]],
) -> dict[str, Any]:
    document = {
        "schema_version": 1,
        "purpose": RUNNABILITY_OWNER_PURPOSE,
        "source_commit": source_commit,
        "source_worktree_state": worktree_state,
        "source_worktree_state_sha256": _canonical_sha256(worktree_state),
        **dict(source_closure),
        "probe_resume_identity_sha256": contract["resume_identity_sha256"],
        "probe_protocol_sha256": contract["probe_protocol_sha256"],
        "probe_config_sha256": contract["probe_config_sha256"],
        "probe_target_set_sha256": contract["probe_target_set_sha256"],
        "sol_execbench_package_sha256": contract[
            "sol_execbench_package_sha256"
        ],
        "native_env_hash": contract["native_env_hash"],
        "output_name": output_name,
        "checkpoint_name": checkpoint_name,
    }
    document["owner_seal_sha256"] = _canonical_sha256(document)
    return document


def _bind_runnability_external_owner(
    *, contract: Mapping[str, Any], cells: list[dict[str, Any]],
    root: Path, output_name: str, checkpoint_name: str,
) -> dict[str, Any]:
    root_identity = validate_external_campaign_root(root)
    worktree_state = source_worktree_state()
    source_closure = probe_source_closure(cells)
    document = _runnability_owner_document(
        contract=contract, source_closure=source_closure,
        output_name=output_name,
        checkpoint_name=checkpoint_name, source_commit=source_head(),
        worktree_state=worktree_state,
    )
    payload = _canonical_bytes(document)
    names = external_root_names(root_identity)
    if RUNNABILITY_OWNER_MARKER_NAME not in names:
        if names:
            raise PrerequisiteContractError(
                "fresh SOL runnability root is nonempty and has no owner marker: "
                f"{sorted(names)}"
            )
        publish_external_child(
            root_identity, RUNNABILITY_OWNER_MARKER_NAME, payload
        )
    elif read_external_child(
        root_identity, RUNNABILITY_OWNER_MARKER_NAME
    ) != payload:
        raise PrerequisiteContractError(
            "SOL runnability owner marker belongs to another campaign identity"
        )
    return {
        "source_commit": document["source_commit"],
        "source_worktree_state": document["source_worktree_state"],
        "source_worktree_state_sha256": document[
            "source_worktree_state_sha256"
        ],
        **dict(source_closure),
        "_external_campaign_root_identity": root_identity,
        "external_campaign_owner": {
            "name": RUNNABILITY_OWNER_MARKER_NAME,
            "sha256": _sha256_bytes(payload),
            "owner_seal_sha256": document["owner_seal_sha256"],
        },
        "external_output_name": document["output_name"],
        "external_checkpoint_name": document["checkpoint_name"],
    }


def _probe(out: Path, *, limit: int | None, timeout: float,
           slugs: set[str] | None,
           external_campaign_root: Path | None = None) -> dict:
    if limit is not None or slugs is not None:
        raise PrerequisiteContractError(
            "partial canonical SOL probe modes are forbidden; run the exact 201-cell set"
        )
    cells = in_scope_cells()
    if len(cells) != 201:
        raise PrerequisiteContractError(f"SOL probe scope changed from 201: {len(cells)}")
    fp = native_env_fingerprint()
    contract = build_probe_contract(
        native_env_hash=fp["hash"], outer_timeout=timeout, target_cells=cells
    )
    if external_campaign_root is None:
        raise PrerequisiteContractError(
            "fresh canonical SOL probe requires --external-campaign-root"
        )
    root_identity = validate_external_campaign_root(external_campaign_root)
    expected_out = external_child_path(root_identity, out.name)
    if out.absolute() != expected_out:
        raise PrerequisiteContractError(
            "runnability --out must be a direct child of --external-campaign-root"
        )
    checkpoint_name = f"{out.name}.partial"
    checkpoint = external_child_path(root_identity, checkpoint_name)
    external_contract = _bind_runnability_external_owner(
        contract=contract, cells=cells, root=external_campaign_root,
        output_name=out.name, checkpoint_name=checkpoint_name,
    )
    root_identity = external_contract["_external_campaign_root_identity"]

    # A complete matching canonical authority needs no rerun.  A malformed or
    # stale canonical file is never used as a checkpoint.
    names = external_root_names(root_identity)
    allowed_names = {
        RUNNABILITY_OWNER_MARKER_NAME, out.name, checkpoint_name,
    }
    if names - allowed_names:
        raise PrerequisiteContractError(
            f"SOL runnability owner namespace contains unexpected files: "
            f"{sorted(names - allowed_names)}"
        )
    if out.name in names:
        try:
            existing = json.loads(read_external_child(root_identity, out.name))
            validate_completed_runnability(
                existing, expected_contract=contract, expected_cells=cells,
                require_external_owner=True,
                external_campaign_root=external_campaign_root,
            )
            if checkpoint_name in names:
                unlink_external_child(root_identity, checkpoint_name)
            if external_root_names(root_identity) != {
                RUNNABILITY_OWNER_MARKER_NAME, out.name,
            }:
                raise PrerequisiteContractError(
                    "completed SOL runnability namespace is not closed"
                )
            return existing
        except (OSError, json.JSONDecodeError, PrerequisiteContractError) as exc:
            raise PrerequisiteContractError(
                f"refusing to overwrite stale/malformed SOL runnability final: {exc}"
            ) from exc

    # Resume is valid only under the identical protocol, config, local package,
    # verified corpus, outer timeout, and certified native toolchain.  A match
    # on the old native-env hash alone is intentionally insufficient.
    prior: dict[tuple, dict] = {}
    expected_by_key = {
        (cell["category"], cell["slug"], cell["language"]): cell
        for cell in cells
    }
    if checkpoint_name in names:
        try:
            existing = json.loads(read_external_child(
                root_identity, checkpoint_name, require_readonly=False
            ))
            prior = _validated_checkpoint_records(
                existing, contract=contract,
                external_contract=external_contract,
                expected_by_key=expected_by_key,
            )
        except (
            OSError, json.JSONDecodeError, KeyError, TypeError, ValueError,
            PrerequisiteContractError,
        ) as exc:
            raise PrerequisiteContractError(
                "runnability checkpoint is malformed/stale and will not be "
                f"overwritten: {exc}"
            ) from exc

    selected_keys = {
        (cell["category"], cell["slug"], cell["language"]) for cell in cells
    }
    records: dict[tuple, dict] = {
        key: value for key, value in prior.items() if key in selected_keys
    }
    todo = [c for c in cells if (c["category"], c["slug"], c["language"]) not in records]
    for i, cell in enumerate(todo, 1):
        rec = run_one(cell, timeout=timeout, probe_contract=contract)
        records[_cell_key(rec)] = rec
        # Persist after every cell so the sweep is crash-safe / resumable.
        _write_runnability(
            checkpoint, fp, contract, cells, records,
            publication_state="checkpoint",
            external_contract=external_contract,
        )
        print(f"[{i}/{len(todo)}] {cell['category']}/{cell['slug']}/"
              f"{cell['language']}: {rec['status']}", flush=True)

    artifact = _write_runnability(
        out, fp, contract, cells, records, publication_state="complete"
        , external_contract=external_contract
    )
    validate_completed_runnability(
        artifact, expected_contract=contract, expected_cells=cells,
        require_external_owner=True,
        external_campaign_root=external_campaign_root,
    )
    if checkpoint_name in external_root_names(root_identity):
        unlink_external_child(root_identity, checkpoint_name)
    if external_root_names(root_identity) != {
        RUNNABILITY_OWNER_MARKER_NAME, out.name,
    }:
        raise PrerequisiteContractError(
            "completed SOL runnability namespace contains unexpected files"
        )
    return artifact


def _write_runnability(out: Path, fp: dict, contract: Mapping[str, Any],
                       cells: list[dict],
                       records: dict[tuple, dict], *,
                       publication_state: str,
                       external_contract: Mapping[str, Any]) -> dict:
    ordered = [records[k] for k in sorted(records)]
    probed = len(ordered)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "native_env_hash": fp["hash"],
        "native_env_degraded": fp["degraded"],
        "publication_state": publication_state,
        **{
            key: value for key, value in external_contract.items()
            if not key.startswith("_")
        },
        **public_probe_contract(contract),
        "in_scope_total": len(cells),
        "probed": probed,
        "pending": len(cells) - probed,
        "summary": summarize(ordered),
        "status_to_typed_reason": STATUS_TO_TYPED_REASON,
        "cells": ordered,
    }
    if publication_state == "complete" and (probed != 201 or len(cells) != 201):
        raise PrerequisiteContractError(
            f"refusing partial canonical runnability publication: {probed}/{len(cells)}"
        )
    if publication_state == "complete":
        validate_completed_runnability(
            artifact, expected_contract=contract, expected_cells=cells
        )
    root_identity = external_contract["_external_campaign_root_identity"]
    payload = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode()
    if publication_state == "complete":
        publish_external_child(root_identity, out.name, payload)
    else:
        atomic_write_external_checkpoint(root_identity, out.name, payload)
    return artifact


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fingerprint", help="emit the native-env fingerprint")
    fp.add_argument("--out", required=True)
    pr = sub.add_parser("probe", help="probe in-scope SOL cells for runnability")
    pr.add_argument("--out", required=True)
    pr.add_argument(
        "--external-campaign-root", required=True,
        help=("pre-existing empty/owned external directory that exclusively "
              "contains the owner marker, checkpoint, and final authority"),
    )
    pr.add_argument(
        "--outer-timeout", "--timeout", dest="outer_timeout", type=float,
        default=DEFAULT_OUTER_TIMEOUT_SECONDS,
        help=("outer process timeout in seconds; must be >= 2000 "
              "(--timeout retained as a compatibility alias)"),
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out

    try:
        if args.cmd == "fingerprint":
            a = emit_fingerprint(out)
            print(f"native-env fingerprint: hash={a['native_env_hash'][:16]}... "
                  f"degraded={a['degraded']} -> {out}")
            return 1 if a["degraded"] else 0
        a = _probe(
            out, limit=None, timeout=args.outer_timeout, slugs=None,
            external_campaign_root=Path(args.external_campaign_root),
        )
        s = a["summary"]
        print(f"sol runnability: probed={a['probed']}/{a['in_scope_total']} "
              f"runnable={s['runnable']} by_status={s['by_status']} -> {out}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"sol_reconstitution: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
