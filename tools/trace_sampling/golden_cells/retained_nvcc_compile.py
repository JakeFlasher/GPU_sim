"""Compile a golden cell through retained no-follow tool capabilities."""
from __future__ import annotations

import argparse
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
_GOLDEN_STAGING_ROOT = (REPO_ROOT / "results" / "nvbit_capture_staging").resolve()


class RetainedNvccError(RuntimeError):
    """The admitted NVCC or host compiler could not be retained."""


def _require_staging_output(path: Path) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    if candidate.parent == candidate or candidate.name in {"", ".", ".."}:
        raise RetainedNvccError(f"compiler output path is invalid: {candidate}")
    if re.fullmatch(
        rf"{re.escape(candidate.parent.name)}\.bin\.staged\.[1-9][0-9]*",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--nvcc-sha256", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output_descriptor = -1
    try:
        output = _require_staging_output(args.output)
        with (
            immutable.open_retained_executable(args.nvcc) as nvcc,
            immutable.open_retained_executable(args.compiler) as compiler,
            immutable.open_retained_regular(args.source) as source,
            immutable.PinnedDirectoryReader(output.parent) as output_parent,
        ):
            if nvcc.sha256 != args.nvcc_sha256:
                raise RetainedNvccError("NVCC SHA-256 changed before invocation")
            if compiler.sha256 != args.compiler_sha256:
                raise RetainedNvccError("host compiler SHA-256 changed before invocation")
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
            command = (
                f"/proc/self/fd/{nvcc.descriptor}",
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
                        source.descriptor, output_descriptor,
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
