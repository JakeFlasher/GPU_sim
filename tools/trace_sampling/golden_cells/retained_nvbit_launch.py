"""Run one golden cell with retained binary and tracer capabilities."""
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


class RetainedNvbitError(RuntimeError):
    """The admitted binary or tracer could not be retained."""


def _require_staging_trace(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(candidate)
    if re.fullmatch(r"trace\.staged\.[1-9][0-9]*", candidate.name) is None:
        raise RetainedNvbitError(
            f"trace output does not use the closed staged trace name: {candidate}"
        )
    try:
        metadata = candidate.lstat()
    except FileNotFoundError:
        pass
    else:
        kind = "directory" if stat.S_ISDIR(metadata.st_mode) else "entry"
        raise RetainedNvbitError(
            f"trace output {kind} already exists and cannot be reused: {candidate}"
        )
    if candidate.resolve(strict=False) != candidate:
        raise RetainedNvbitError(
            f"trace output contains a symlink: {candidate}"
        )
    try:
        candidate.relative_to(_GOLDEN_STAGING_ROOT)
        protected_paths.assert_write_allowed(
            candidate,
            repo_root=REPO_ROOT,
            operation="retained golden trace output",
        )
    except (ValueError, protected_paths.ProtectedPathError) as exc:
        raise RetainedNvbitError(
            f"trace output is outside the guarded golden staging root: {candidate}"
        ) from exc
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--tracer", type=Path, required=True)
    parser.add_argument("--tracer-sha256", required=True)
    parser.add_argument("--pb-lib", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        trace_dir = _require_staging_trace(args.trace_dir)
        immutable.ensure_directory(trace_dir.parent)
        with immutable.PinnedDirectoryReader(trace_dir.parent) as trace_parent:
            trace_parent.verify_path()
            try:
                os.mkdir(trace_dir.name, mode=0o755, dir_fd=trace_parent.descriptor)
            except FileExistsError as exc:
                raise RetainedNvbitError(
                    f"trace output appeared before exclusive creation: {trace_dir}"
                ) from exc
            os.fsync(trace_parent.descriptor)
            trace_parent.verify_path()
        with (
            immutable.open_retained_executable(args.binary) as binary,
            immutable.open_retained_regular(args.tracer) as tracer,
            immutable.PinnedDirectoryReader(trace_dir) as trace_root,
        ):
            if binary.sha256 != args.binary_sha256:
                raise RetainedNvbitError("target binary SHA-256 changed before NVBit")
            if tracer.sha256 != args.tracer_sha256:
                raise RetainedNvbitError("tracer SHA-256 changed before NVBit")
            env = {
                "LANG": "C",
                "LANGUAGE": "C",
                "LC_ALL": "C",
                "PATH": "/bin:/usr/bin",
                "LD_LIBRARY_PATH": str(args.pb_lib),
                "CUDA_INJECTION64_PATH": f"/proc/self/fd/{tracer.descriptor}",
                "USER_DEFINED_FOLDERS": "1",
                "TRACES_FOLDER": f"/proc/self/fd/{trace_root.descriptor}",
            }
            try:
                result = ncu_action_executor.execute_bounded_command(
                    (f"/proc/self/fd/{binary.descriptor}",),
                    environment=env,
                    working_directory="/",
                    inherited_descriptors=(
                        binary.descriptor, tracer.descriptor, trace_root.descriptor,
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
            trace_root.verify_path()
            return result.returncode
    except (
        OSError,
        immutable.ImmutableArtifactError,
        RetainedNvbitError,
        ncu_action_executor.NcuActionExecutorError,
    ) as exc:
        print(f"retained_nvbit_launch_refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
