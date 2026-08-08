"""Compile a golden cell through retained no-follow tool capabilities."""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import ncu_action_executor  # noqa: E402
from tools.local_labels import protected_paths  # noqa: E402


_ACTION_TIMEOUT_SECONDS = 15 * 60
_MAX_ACTION_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_BINARY_BYTES = 1024 * 1024 * 1024
_MAX_PREFLIGHT_BYTES = 4 * 1024 * 1024
_FILE_SIZE_BLOCK_BYTES = 1024
_MAX_BINARY_FILE_BLOCKS = _MAX_BINARY_BYTES // _FILE_SIZE_BLOCK_BYTES
_RETAINED_BASH = Path("/usr/bin/bash")
_GOLDEN_STAGING_ROOT = (REPO_ROOT / "results" / "nvbit_capture_staging").resolve()


class RetainedNvccError(RuntimeError):
    """The admitted NVCC or host compiler could not be retained."""


def _require_staging_output(path: Path) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    if candidate.parent == candidate or candidate.name in {"", ".", ".."}:
        raise RetainedNvccError(f"compiler output path is invalid: {candidate}")
    staged_parent = re.fullmatch(
        r"(?P<slug>[^/]+)\.staged\.[1-9][0-9]*\.[0-9a-f]{32}",
        candidate.parent.name,
    )
    if (
        staged_parent is None
        or candidate.parent.parent
        != _GOLDEN_STAGING_ROOT / "golden_cells"
    ):
        raise RetainedNvccError(
            f"compiler output is outside a closed attempt cell: {candidate}"
        )
    expected_slug = staged_parent.group("slug")
    if re.fullmatch(
        rf"{re.escape(expected_slug)}\.bin\.staged\.[1-9][0-9]*",
        candidate.name,
    ) is None:
        raise RetainedNvccError(
            f"compiler output does not use the closed staged binary name: {candidate}"
        )
    if candidate.resolve(strict=False) != candidate:
        raise RetainedNvccError(
            f"compiler output path contains a symlink: {candidate}"
        )
    try:
        candidate.relative_to(_GOLDEN_STAGING_ROOT)
        protected_paths.assert_write_allowed(
            candidate,
            repo_root=REPO_ROOT,
            operation="retained golden compiler output",
        )
    except (ValueError, protected_paths.ProtectedPathError) as exc:
        raise RetainedNvccError(
            f"compiler output is outside the guarded golden staging root: {candidate}"
        ) from exc
    return candidate


def _file_size_limited_command(
    *,
    bash_executable: str,
    target_executable: str,
    target_arguments: tuple[str, ...],
    max_file_blocks: int = _MAX_BINARY_FILE_BLOCKS,
) -> tuple[str, ...]:
    if type(max_file_blocks) is not int or max_file_blocks <= 0:
        raise RetainedNvccError("compiler file-size block limit is invalid")
    return (
        bash_executable,
        "--noprofile",
        "--norc",
        "-c",
        'ulimit -f "$1" || exit 125; shift; exec "$@"',
        "bounded-retained-nvcc",
        str(max_file_blocks),
        target_executable,
        *target_arguments,
    )


def _strict_json_object(payload: bytes) -> dict:
    try:
        decoded = payload.decode("utf-8", errors="strict")

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
            result: dict = {}
            for key, value in pairs:
                if key in result:
                    raise RetainedNvccError(
                        f"preflight contains duplicate JSON key: {key}"
                    )
                result[key] = value
            return result

        value = json.loads(decoded, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetainedNvccError(f"preflight JSON is invalid: {exc}") from exc
    if type(value) is not dict:
        raise RetainedNvccError("preflight must be an exact JSON object")
    return value


def _require_preflight_source_authority(
    *,
    preflight: object,
    source: object,
    retained_git: object,
) -> str:
    preflight.verify_unchanged()
    source.verify_unchanged()
    manifest = _strict_json_object(preflight.read_bytes())
    source_commit = manifest.get("git_head")
    if manifest.get("schema_version") != 4:
        raise RetainedNvccError(
            "live golden compilation requires preflight schema_version 4"
        )
    if (
        manifest.get("working_tree_clean") is not True
        or manifest.get("capture_admitted") is not True
        or type(source_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise RetainedNvccError(
            "preflight does not authorize one admitted clean Git HEAD"
        )
    try:
        source_relative = source.path.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise RetainedNvccError(
            f"golden source is outside repository: {source.path}"
        ) from exc
    head = protected_paths.run_retained_git(
        retained_git,
        repo_root=REPO_ROOT,
        arguments=("rev-parse", "--verify", "HEAD"),
        text=True,
    )
    if head.returncode != 0 or head.stdout.strip() != source_commit:
        raise RetainedNvccError(
            "current Git HEAD disagrees with admitted preflight git_head"
        )
    status = protected_paths.run_retained_git(
        retained_git,
        repo_root=REPO_ROOT,
        arguments=(
            "status", "--porcelain=v1", "--untracked-files=all", "--",
            source_relative,
        ),
        text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise RetainedNvccError(
            "golden compile source is dirty relative to admitted preflight HEAD"
        )
    committed = protected_paths.run_retained_git(
        retained_git,
        repo_root=REPO_ROOT,
        arguments=("show", f"{source_commit}:{source_relative}"),
    )
    if committed.returncode != 0 or committed.stdout != source.read_bytes():
        raise RetainedNvccError(
            "retained golden compile source differs from preflight clean-HEAD bytes"
        )
    preflight.verify_unchanged()
    source.verify_unchanged()
    return source_commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--nvcc-sha256", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output_descriptor = -1
    try:
        output = _require_staging_output(args.output)
        with (
            immutable.open_retained_executable(args.nvcc) as nvcc,
            immutable.open_retained_executable(
                args.compiler, allow_hardlinks=True,
            ) as compiler,
            immutable.open_retained_executable(_RETAINED_BASH) as retained_bash,
            immutable.open_retained_executable(
                protected_paths.TRUSTED_GIT_EXECUTABLE
            ) as retained_git,
            immutable.open_retained_regular(args.source) as source,
            immutable.open_retained_regular(
                args.preflight, max_bytes=_MAX_PREFLIGHT_BYTES,
            ) as preflight,
            immutable.PinnedDirectoryReader(output.parent) as output_parent,
        ):
            if nvcc.sha256 != args.nvcc_sha256:
                raise RetainedNvccError("NVCC SHA-256 changed before invocation")
            if compiler.sha256 != args.compiler_sha256:
                raise RetainedNvccError("host compiler SHA-256 changed before invocation")
            if preflight.sha256 != args.preflight_sha256:
                raise RetainedNvccError("preflight SHA-256 changed before invocation")
            source_commit = _require_preflight_source_authority(
                preflight=preflight,
                source=source,
                retained_git=retained_git,
            )
            output_parent.verify_path()
            try:
                output_descriptor = os.open(
                    output.name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    0o700,
                    dir_fd=output_parent.descriptor,
                )
            except OSError as exc:
                raise RetainedNvccError(
                    f"cannot create retained compiler output: {exc}"
                ) from exc
            command = _file_size_limited_command(
                bash_executable=retained_bash.proc_path,
                target_executable=nvcc.proc_path,
                target_arguments=(
                    f"-ccbin=/proc/self/fd/{compiler.descriptor}",
                    "-O3",
                    "-std=c++17",
                    "-gencode",
                    "arch=compute_86,code=sm_86",
                    "-x",
                    "cu",
                    source.proc_path,
                    "-o",
                    f"/proc/self/fd/{output_descriptor}",
                ),
            )
            try:
                result = ncu_action_executor.execute_bounded_command(
                    command,
                    environment={
                        "LANG": "C",
                        "LANGUAGE": "C",
                        "LC_ALL": "C",
                        "PATH": "/bin:/usr/bin",
                    },
                    working_directory="/",
                    inherited_descriptors=(
                        nvcc.descriptor, compiler.descriptor,
                        retained_bash.descriptor, source.descriptor,
                        output_descriptor,
                    ),
                    timeout_seconds=_ACTION_TIMEOUT_SECONDS,
                    max_output_bytes=_MAX_ACTION_OUTPUT_BYTES,
                    timeout_code=ncu_action_executor.NCU_ACTION_TIMEOUT,
                    launch_code=ncu_action_executor.NCU_ACTION_LAUNCH_FAILED,
                )
            except ncu_action_executor.NcuActionExecutorError as exc:
                if exc.stdout:
                    sys.stdout.buffer.write(exc.stdout)
                if exc.stderr:
                    sys.stderr.buffer.write(exc.stderr)
                raise
            if result.stdout:
                sys.stdout.buffer.write(result.stdout)
            if result.stderr:
                sys.stderr.buffer.write(result.stderr)
            retained_bash.verify_unchanged()
            nvcc.verify_unchanged()
            compiler.verify_unchanged()
            if _require_preflight_source_authority(
                preflight=preflight,
                source=source,
                retained_git=retained_git,
            ) != source_commit:
                raise RetainedNvccError(
                    "preflight source_commit changed during compilation"
                )
            output_parent.verify_path()
            linked = os.stat(
                output.name, dir_fd=output_parent.descriptor, follow_symlinks=False,
            )
            opened = os.fstat(output_descriptor)
            if (
                stat.S_ISLNK(linked.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or linked.st_dev != opened.st_dev
                or linked.st_ino != opened.st_ino
                or linked.st_nlink != 1
                or opened.st_nlink != 1
            ):
                raise RetainedNvccError(
                    "compiler output path changed during retained invocation"
                )
            if opened.st_size > _MAX_BINARY_BYTES:
                raise RetainedNvccError(
                    "compiler output exceeds the retained binary byte limit: "
                    f"{opened.st_size} > {_MAX_BINARY_BYTES}"
                )
            os.fsync(output_descriptor)
            return result.returncode
    except (
        OSError,
        immutable.ImmutableArtifactError,
        RetainedNvccError,
        ncu_action_executor.NcuActionExecutorError,
    ) as exc:
        print(f"retained_nvcc_compile_refused: {exc}", file=sys.stderr)
        return 2
    finally:
        if output_descriptor >= 0:
            os.close(output_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
