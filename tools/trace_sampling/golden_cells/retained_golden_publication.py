#!/usr/bin/env python3
"""Retain and atomically publish one golden-cell capture generation."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import os
import re
import stat
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import immutable_artifact as immutable  # noqa: E402
from tools.local_labels import protected_paths  # noqa: E402
from tools.nvbit_tracer import trace_admission  # noqa: E402
from tools.trace_sampling.golden_cells import golden_ncu_capture  # noqa: E402


_ATTEMPT_CELL = re.compile(
    r"(?P<slug>[^/]+)\.staged\.[1-9][0-9]*\.[0-9a-f]{32}"
)
_GOLDEN_ROOT = (
    REPO_ROOT / "results" / "nvbit_capture_staging" / "golden_cells"
).resolve()
_NVBIT_ATTEMPT_NAME = "nvbit_launch_attempt.json"
_MAX_NVBIT_ATTEMPT_BYTES = 1024 * 1024
_RAW_LIMITS = trace_admission.raw_trace_generation_limits(
    trace_admission.DEFAULT_TRACE_ADMISSION_LIMITS,
)
_TRACE_RETAIN_LIMITS = {
    name: _RAW_LIMITS[name]
    for name in (
        "max_files",
        "max_directories",
        "max_total_file_bytes",
        "max_total_relative_path_bytes",
    )
}
_CELL_LIMITS = {
    "max_files": int(_RAW_LIMITS["max_files"]) + 32,
    "max_directories": int(_RAW_LIMITS["max_directories"]) + 2,
    "max_total_file_bytes": int(_RAW_LIMITS["max_total_file_bytes"])
    + 3 * 1024 * 1024 * 1024,
    "max_total_relative_path_bytes": int(
        _RAW_LIMITS["max_total_relative_path_bytes"]
    )
    + 1024 * 1024,
}


class GoldenPublicationError(RuntimeError):
    """A staged golden generation could not be retained or published."""


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _guarded_golden_path(path: Path, *, operation: str) -> Path:
    candidate = _absolute(path)
    if candidate.resolve(strict=False) != candidate:
        raise GoldenPublicationError(
            f"golden publication path contains a symlink: {candidate}"
        )
    try:
        candidate.relative_to(_GOLDEN_ROOT)
        protected_paths.assert_write_allowed(
            candidate,
            repo_root=REPO_ROOT,
            operation=operation,
        )
    except (ValueError, protected_paths.ProtectedPathError) as exc:
        raise GoldenPublicationError(
            f"golden publication path escapes guarded staging: {candidate}"
        ) from exc
    return candidate


def _validate_cell_mapping(stage: Path, final: Path) -> tuple[Path, Path]:
    stage = _guarded_golden_path(stage, operation="golden cell stage")
    final = _guarded_golden_path(final, operation="golden cell publication")
    match = _ATTEMPT_CELL.fullmatch(stage.name)
    if (
        match is None
        or match.group("slug") != final.name
        or stage.parent != _GOLDEN_ROOT
        or final.parent != _GOLDEN_ROOT
        or stage.parent != final.parent
        or stage.resolve(strict=False) != stage
        or final.resolve(strict=False) != final
    ):
        raise GoldenPublicationError(
            "golden cell stage does not map to its closed final slug"
        )
    return stage, final


def _entry_exists(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def prepare_cell_stage(stage: Path, final: Path) -> None:
    """Create one attempt-owned cell stage without reusing an old generation."""
    stage, final = _validate_cell_mapping(stage, final)
    immutable.ensure_directory(stage.parent)
    immutable.ensure_directory(final.parent)
    with immutable.PinnedDirectoryReader(
        stage.parent
    ) as stage_parent, immutable.PinnedDirectoryReader(final.parent) as final_parent:
        stage_parent.verify_path()
        final_parent.verify_path()
        if _entry_exists(final_parent.descriptor, final.name):
            raise GoldenPublicationError(
                f"golden cell publication target already exists: {final}"
            )
        try:
            os.mkdir(stage.name, mode=0o700, dir_fd=stage_parent.descriptor)
        except FileExistsError as exc:
            raise GoldenPublicationError(
                f"golden cell stage already exists and is retained: {stage}"
            ) from exc
        os.fsync(stage_parent.descriptor)
        stage_parent.verify_path()
        final_parent.verify_path()


def _admission_identity(admission: object) -> tuple[object, ...]:
    return tuple(
        getattr(admission, name, None)
        for name in (
            "trace_admission_sha256",
            "launch_count",
            "threadblock_count",
            "instruction_count",
            "expected_record_count",
            "admitted_record_count",
        )
    )


def _require_launch_generation_handoff(
    generation_reader: immutable.PinnedDirectoryReader,
) -> None:
    attempt = immutable._open_relative_regular_at(
        generation_reader.descriptor,
        _NVBIT_ATTEMPT_NAME,
    )
    try:
        payload, _metadata = immutable._read_fd_snapshot(
            attempt.descriptor,
            max_bytes=_MAX_NVBIT_ATTEMPT_BYTES,
        )
        attempt.verify_path()
    finally:
        attempt.close()
    document = immutable.strict_json_loads(
        payload,
        label=_NVBIT_ATTEMPT_NAME,
    )
    if type(document) is not dict:
        raise GoldenPublicationError("golden NVBit launch attempt is not an object")
    if immutable.canonical_json_bytes(document) != payload:
        raise GoldenPublicationError(
            "golden NVBit launch attempt is not canonical JSON"
        )
    immutable.verify_model_seal(document, field="attempt_seal_sha256")
    if (
        document.get("schema_version") != 1
        or document.get("artifact_kind") != "golden_nvbit_launch_attempt"
        or document.get("status") != "succeeded"
    ):
        raise GoldenPublicationError(
            "golden NVBit launch attempt is not a successful v1 handoff"
        )
    identity = document.get("trace_generation")
    if type(identity) is not dict or set(identity) != {"device", "inode"}:
        raise GoldenPublicationError(
            "golden NVBit launch attempt lacks an exact trace generation"
        )
    device = identity.get("device")
    inode = identity.get("inode")
    if (
        type(device) is not int
        or device < 0
        or type(inode) is not int
        or inode <= 0
    ):
        raise GoldenPublicationError(
            "golden NVBit launch trace generation is malformed"
        )
    generation_reader.verify_path()
    metadata = os.fstat(generation_reader.descriptor)
    observed = (metadata.st_dev, metadata.st_ino)
    if observed != (device, inode):
        raise GoldenPublicationError(
            "staged trace generation differs from the NVBit launch handoff"
        )
    generation_reader.verify_path()


def process_trace_generation(
    staged_trace: Path,
    published_trace: Path,
    *,
    derive_ncu: Callable[[Path], None] | None = None,
    prepare_handoff: Callable[[int], "_CellCommitHandoff"] | None = None,
) -> object:
    """Admit, derive, re-admit, and relocate one retained trace inode tree."""
    staged_trace = _guarded_golden_path(
        staged_trace, operation="retained golden trace admission",
    )
    published_trace = _guarded_golden_path(
        published_trace, operation="attempt-local golden trace publication",
    )
    attempt_match = _ATTEMPT_CELL.fullmatch(staged_trace.parent.name)
    if (
        attempt_match is None
        or staged_trace.parent.parent != _GOLDEN_ROOT
        or staged_trace.parent != published_trace.parent
        or staged_trace == published_trace
        or re.fullmatch(r"trace\.staged\.[1-9][0-9]*", staged_trace.name) is None
        or published_trace.name != "trace"
    ):
        raise GoldenPublicationError("golden trace publication paths are invalid")

    handoff: _CellCommitHandoff | None = None
    with contextlib.ExitStack() as handoff_guard:
        with immutable.PinnedDirectoryReader(staged_trace) as generation_reader:
            _require_launch_generation_handoff(generation_reader)
            with contextlib.ExitStack() as generation_guard:
                retained_tree = generation_guard.enter_context(
                    generation_reader.retain_tree_generation(**_TRACE_RETAIN_LIMITS)
                )
                initial_admission = trace_admission.require_trace_admission(
                    staged_trace
                )
                if derive_ncu is not None:
                    derive_ncu(staged_trace)
                closing_admission = trace_admission.require_trace_admission(
                    staged_trace
                )
                if _admission_identity(closing_admission) != _admission_identity(
                    initial_admission
                ):
                    raise GoldenPublicationError(
                        "golden trace admission changed during NCU derivation"
                    )
                generation_guard.close()

            with immutable.PinnedDirectoryReader(staged_trace.parent) as parent:
                generation_reader.relocate_noreplace(
                    source_parent=parent,
                    source_name=staged_trace.name,
                    destination_parent=parent,
                    destination_name=published_trace.name,
                )
            generation_reader.verify_path()
            relocated_tree = generation_reader.snapshot_tree(**_TRACE_RETAIN_LIMITS)
            if relocated_tree.files != retained_tree.files:
                raise GoldenPublicationError(
                    "golden trace file generation changed across publication"
                )
            if tuple(
                row for row in relocated_tree.directories if row[0] != "."
            ) != tuple(
                row for row in retained_tree.directories if row[0] != "."
            ):
                raise GoldenPublicationError(
                    "golden trace child-directory generation changed across publication"
                )
            published_admission = trace_admission.require_trace_admission(
                published_trace
            )
            if _admission_identity(published_admission) != _admission_identity(
                initial_admission
            ):
                raise GoldenPublicationError(
                    "published golden trace admission differs from retained generation"
                )
            generation_reader.verify_path()
            if prepare_handoff is not None:
                handoff = prepare_handoff(generation_reader.descriptor)
                handoff_guard.callback(handoff.close)
        if handoff is not None:
            handoff.complete()
        return published_admission


@dataclasses.dataclass
class _CellCommitHandoff:
    stage: Path
    final: Path
    generation: object
    source_parent: object
    destination_parent: object
    trace_identity: tuple[int, int]
    artifacts: tuple[tuple[Path, Path], ...]
    _closed: bool = False

    @classmethod
    def open(
        cls,
        stage: Path,
        final: Path,
        *,
        expected_trace_descriptor: int,
        artifacts: tuple[tuple[Path, Path], ...] = (),
    ) -> "_CellCommitHandoff":
        stage, final = _validate_cell_mapping(stage, final)
        normalized_artifacts = tuple(
            (
                _guarded_golden_path(source, operation="golden artifact source"),
                _guarded_golden_path(destination, operation="golden artifact publication"),
            )
            for source, destination in artifacts
        )
        if any(
            source.parent != stage
            or destination.parent != stage
            or source == destination
            for source, destination in normalized_artifacts
        ):
            raise GoldenPublicationError(
                "golden derived artifacts escape the retained attempt cell"
            )
        generation = source_parent = destination_parent = None
        try:
            generation = immutable._open_pinned_directory(stage)
            source_parent = immutable._open_pinned_directory(stage.parent)
            destination_parent = immutable._open_pinned_directory(final.parent)
            trace_metadata = os.fstat(expected_trace_descriptor)
            if not stat.S_ISDIR(trace_metadata.st_mode):
                raise GoldenPublicationError(
                    "retained golden trace descriptor is not a directory"
                )
            handoff = cls(
                stage=stage,
                final=final,
                generation=generation,
                source_parent=source_parent,
                destination_parent=destination_parent,
                trace_identity=immutable._node_identity(trace_metadata),
                artifacts=normalized_artifacts,
            )
            handoff._verify_open_generation()
            generation = source_parent = destination_parent = None
            return handoff
        finally:
            for pinned in (generation, source_parent, destination_parent):
                if pinned is not None:
                    pinned.close()

    def _verify_open_generation(self) -> None:
        self.generation.verify_path()
        self.source_parent.verify_path()
        self.destination_parent.verify_path()
        retained_identity = immutable._node_identity(
            os.fstat(self.generation.descriptor)
        )
        source_entry = os.stat(
            self.stage.name,
            dir_fd=self.source_parent.descriptor,
            follow_symlinks=False,
        )
        if immutable._node_identity(source_entry) != retained_identity:
            raise GoldenPublicationError(
                "golden cell stage identity changed during descriptor handoff"
            )
        trace_entry = os.stat(
            "trace",
            dir_fd=self.generation.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(trace_entry.st_mode)
            or immutable._node_identity(trace_entry) != self.trace_identity
        ):
            raise GoldenPublicationError(
                "golden trace identity changed during cell handoff"
            )
        if _entry_exists(self.destination_parent.descriptor, self.final.name):
            raise FileExistsError(self.final)

    def complete(self) -> None:
        if self._closed:
            raise GoldenPublicationError("golden cell handoff is closed")
        self._verify_open_generation()
        for source, destination in self.artifacts:
            immutable.publish_regular_file(
                destination.parent,
                destination.name,
                source,
            )
            self._verify_open_generation()

        retained_identity = immutable._node_identity(
            os.fstat(self.generation.descriptor)
        )
        staged_generation = immutable._freeze_external_directory_generation_at(
            self.generation.descriptor,
            path=self.stage,
            require_nonempty=True,
            max_file_bytes=None,
            max_file_bytes_by_path=None,
            **_CELL_LIMITS,
        )
        self._verify_open_generation()
        frozen_source = os.stat(
            self.stage.name,
            dir_fd=self.source_parent.descriptor,
            follow_symlinks=False,
        )
        if immutable._node_identity(frozen_source) != retained_identity:
            raise GoldenPublicationError(
                "golden cell stage identity changed during freeze"
            )
        os.fsync(self.source_parent.descriptor)
        os.fsync(self.destination_parent.descriptor)
        immutable._rename_noreplace_at(
            self.source_parent.descriptor,
            self.stage.name,
            self.destination_parent.descriptor,
            self.final.name,
        )
        source_after = immutable._entry_metadata(
            self.source_parent.descriptor, self.stage.name,
        )
        destination_after = immutable._entry_metadata(
            self.destination_parent.descriptor, self.final.name,
        )
        if (
            source_after is not None
            or destination_after is None
            or immutable._node_identity(destination_after) != retained_identity
        ):
            raise GoldenPublicationError(
                "golden cell no-replace commit outcome is not authenticated"
            )
        os.fsync(self.source_parent.descriptor)
        os.fsync(self.destination_parent.descriptor)
        committed_generation = immutable._inspect_immutable_directory_generation_at(
            self.generation.descriptor,
            path=self.final,
            require_nonempty=True,
            max_file_bytes=None,
            max_file_bytes_by_path=None,
            **_CELL_LIMITS,
        )
        if (
            committed_generation.files != staged_generation.files
            or committed_generation.content_manifest_sha256
            != staged_generation.content_manifest_sha256
            or committed_generation.total_file_bytes
            != staged_generation.total_file_bytes
        ):
            raise GoldenPublicationError(
                "golden cell generation changed across final publication"
            )
        tree = immutable._snapshot_pinned_tree_at(
            self.generation.descriptor,
            require_readonly_files=True,
            require_readonly_directories=True,
            **_CELL_LIMITS,
        )
        if not tree.files:
            raise GoldenPublicationError("published golden cell is empty")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.generation.close()
        self.source_parent.close()
        self.destination_parent.close()


def commit_cell_stage(stage: Path, final: Path) -> None:
    """Freeze and atomically move one complete cell to its final no-replace name."""
    stage, final = _validate_cell_mapping(stage, final)
    with immutable.PinnedDirectoryReader(stage / "trace") as trace_reader:
        handoff = _CellCommitHandoff.open(
            stage,
            final,
            expected_trace_descriptor=trace_reader.descriptor,
        )
    try:
        handoff.complete()
    finally:
        handoff.close()


def _ncu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--slug")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--binary-sha256")
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--stderr-out", type=Path)
    parser.add_argument("--observation-out", type=Path)
    parser.add_argument("--run-manifest-out", type=Path)
    parser.add_argument("--capture-manifest-out", type=Path)
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--cell-csv-out", type=Path)
    parser.add_argument("--cell-stderr-out", type=Path)
    parser.add_argument("--cell-observation-out", type=Path)
    parser.add_argument("--cell-run-manifest-out", type=Path)
    parser.add_argument("--cell-capture-manifest-out", type=Path)
    parser.add_argument("--cell-receipt-out", type=Path)
    parser.add_argument("--published-binary", type=Path)
    parser.add_argument("--declared-trace", type=Path)
    parser.add_argument("--published-csv-out", type=Path)
    parser.add_argument("--published-stderr-out", type=Path)
    parser.add_argument("--published-observation-out", type=Path)
    parser.add_argument("--published-run-manifest-out", type=Path)
    parser.add_argument("--published-capture-manifest-out", type=Path)
    parser.add_argument("--published-receipt-out", type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare-cell")
    prepare.add_argument("--stage", type=Path, required=True)
    prepare.add_argument("--final", type=Path, required=True)
    process = subparsers.add_parser("process-cell")
    process.add_argument("--staged-trace", type=Path, required=True)
    process.add_argument("--trace-out", type=Path, required=True)
    process.add_argument("--cell-stage", type=Path, required=True)
    process.add_argument("--cell-final", type=Path, required=True)
    process.add_argument("--with-ncu", action="store_true")
    _ncu_arguments(process)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare-cell":
            prepare_cell_stage(args.stage, args.final)
        else:
            derive_ncu = None
            if args.with_ncu:
                required_ncu_arguments = (
                    "preflight", "preflight_sha256", "slug", "binary",
                    "binary_sha256", "csv_out", "stderr_out",
                    "observation_out", "run_manifest_out",
                    "capture_manifest_out", "receipt_out",
                    "cell_csv_out", "cell_stderr_out",
                    "cell_observation_out", "cell_run_manifest_out",
                    "cell_capture_manifest_out", "cell_receipt_out",
                    "published_binary", "declared_trace",
                    "published_csv_out", "published_stderr_out",
                    "published_observation_out", "published_run_manifest_out",
                    "published_capture_manifest_out", "published_receipt_out",
                )
                missing = tuple(
                    name for name in required_ncu_arguments
                    if getattr(args, name) is None
                )
                if missing:
                    raise GoldenPublicationError(
                        "NCU derivation arguments are incomplete: "
                        + ", ".join(missing)
                    )

                def derive_ncu(trace_path: Path) -> None:
                    golden_ncu_capture.collect_golden_ncu(
                        preflight_path=args.preflight,
                        preflight_sha256=args.preflight_sha256,
                        slug=args.slug,
                        binary=args.binary,
                        binary_sha256=args.binary_sha256,
                        trace_dir=trace_path,
                        csv_out=args.csv_out,
                        stderr_out=args.stderr_out,
                        observation_out=args.observation_out,
                        run_manifest_out=args.run_manifest_out,
                        capture_manifest_out=args.capture_manifest_out,
                        receipt_out=args.receipt_out,
                        published_binary=args.published_binary,
                        published_trace_dir=args.declared_trace,
                        published_csv_out=args.published_csv_out,
                        published_stderr_out=args.published_stderr_out,
                        published_observation_out=args.published_observation_out,
                        published_run_manifest_out=args.published_run_manifest_out,
                        published_capture_manifest_out=args.published_capture_manifest_out,
                        published_receipt_out=args.published_receipt_out,
                    )
                artifacts = (
                    (args.csv_out, args.cell_csv_out),
                    (args.stderr_out, args.cell_stderr_out),
                    (args.observation_out, args.cell_observation_out),
                    (args.run_manifest_out, args.cell_run_manifest_out),
                    (args.capture_manifest_out, args.cell_capture_manifest_out),
                    (args.receipt_out, args.cell_receipt_out),
                )
            else:
                artifacts = ()

            def prepare_handoff(trace_descriptor: int) -> _CellCommitHandoff:
                return _CellCommitHandoff.open(
                    args.cell_stage,
                    args.cell_final,
                    expected_trace_descriptor=trace_descriptor,
                    artifacts=artifacts,
                )

            process_trace_generation(
                args.staged_trace,
                args.trace_out,
                derive_ncu=derive_ncu,
                prepare_handoff=prepare_handoff,
            )
    except (
        OSError,
        GoldenPublicationError,
        immutable.ImmutableArtifactError,
        golden_ncu_capture.GoldenNcuCaptureError,
        trace_admission.TraceAdmissionError,
    ) as exc:
        print(f"retained_golden_publication_refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
