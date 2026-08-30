#!/usr/bin/env python3
"""Read-only validation of the active launch/memory replay-pair corpus.

The scan uses the tracked content binding in
``evidence/sm86_active_replay_binding.json`` for replay/input identity; replay
mtimes are not used once that binding is validated.  A separate binary-build
freshness check still protects the simulator executable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from functools import cache
from pathlib import Path
from typing import Any, Iterator, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPECTED_PAIR_COUNT = 5_519
SIMULATOR_CRATES = ("arch", "engine", "mem", "sm", "trace", "direct_run")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load campaign module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPLAY = load_module("active_replay_validation", HERE / "replay_resume.py")
LAUNCH = load_module(
    "validated_launch_cycle_campaign", HERE / "launch_cycle_semantics/campaign.py"
)
RETIREMENT = load_module(
    "validated_retirement_campaign", HERE / "memory_retirement_service/campaign.py"
)
ARRIVAL = load_module(
    "validated_arrival_campaign", HERE / "l1tex_arrival_replay_service/campaign.py"
)
ADMISSION = load_module(
    "validated_admission_campaign", HERE / "l1tex_admission_backpressure/campaign.py"
)
from workloads.research_probes.semantic_replay_validation import (  # noqa: E402
    validate_replay_document,
)
from workloads.research_probes import active_replay_binding as CONTENT_BINDING  # noqa: E402

CONTENT_BINDING_PATH = REPO_ROOT / "evidence/sm86_active_replay_binding.json"
CONTENT_BINDING_SHA256 = "4316083b8b61f53f3b7bb8ff4f9d78eabae2cc21538acaa9bc2ba1a26a6128db"


@cache
def _supplemental_candidate_ids() -> tuple[frozenset[int], ...]:
    return (
        frozenset(ADMISSION.validated_feedback_service_sweep()),
        frozenset(ADMISSION.validated_write_completion_feedback()),
        frozenset(ADMISSION.validated_red_rmw_completion_sweep()),
    )


@cache
def _admission_cell_keys() -> frozenset[str]:
    return frozenset(cell.key for cell in ADMISSION.configured_cells())


def _known_supplemental_replay_directory(path: Path, root: Path) -> bool:
    """Recognize valid full-campaign outputs outside the active matrix.

    The admission campaign can deliberately replay every cell for every
    candidate in its exploratory service/write-completion sweeps.  The active
    5,519-pair matrix keeps only the frozen transfer subset, while those extra
    candidate directories are still useful, deterministic research artifacts.
    Keep the strict active-pair validation unchanged, but do not misclassify a
    known, well-shaped supplemental directory as a stale/unknown replay.
    """

    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    if len(relative.parts) != 2:
        return False
    directory, cell_key = relative.parts
    if cell_key not in _admission_cell_keys():
        return False

    prefixes_and_ids = tuple(
        zip(
            (
                "simulator-memory-admission-feedback-service-",
                "simulator-memory-write-completion-feedback-",
                "simulator-red-rmw-completion-",
            ),
            _supplemental_candidate_ids(),
        )
    )
    for prefix, candidate_ids in prefixes_and_ids:
        if not directory.startswith(prefix):
            continue
        suffix = directory[len(prefix) :]
        try:
            candidate_id = int(suffix)
        except ValueError:
            return False
        return candidate_id in candidate_ids
    return False


def _supplemental_descriptor(
    path: Path, root: Path
) -> tuple[str, int, str] | None:
    """Return the candidate family/id/cell encoded by a supplemental path."""

    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    if len(relative.parts) != 2:
        return None
    directory, cell_key = relative.parts
    if cell_key not in _admission_cell_keys():
        return None
    families = (
        ("service", "simulator-memory-admission-feedback-service-", 0),
        ("write-completion", "simulator-memory-write-completion-feedback-", 1),
        ("red-rmw", "simulator-red-rmw-completion-", 2),
    )
    candidate_sets = _supplemental_candidate_ids()
    for family, prefix, index in families:
        if not directory.startswith(prefix):
            continue
        try:
            candidate_id = int(directory[len(prefix) :])
        except ValueError:
            return None
        if candidate_id not in candidate_sets[index]:
            return None
        return family, candidate_id, cell_key
    return None


def _supplemental_pair_spec(
    path: Path, root: Path, binary: Path
) -> tuple[Path, Path, list[Path], Any, Path] | None:
    """Build the same replay expectation used by the owning campaign."""

    descriptor = _supplemental_descriptor(path, root)
    if descriptor is None:
        return None
    family, candidate_id, cell_key = descriptor
    cell = next(
        (candidate for candidate in ADMISSION.configured_cells() if candidate.key == cell_key),
        None,
    )
    if cell is None:
        return None
    trace = root / "nvbit" / cell_key
    control = root / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    arch_card = REPO_ROOT / "arch_profiles/cards/sm_86.json"
    pchase_card = REPO_ROOT / "data/pchase/sm_86_static_card.json"
    launches = REPLAY.expected_launches_from_trace(
        trace,
        grid=(cell.blocks, 1, 1),
        block=(cell.warps * 32, 1, 1),
    )
    if family == "service":
        source_file, candidate = ADMISSION.validated_feedback_service_sweep()[candidate_id]
        candidate_file = (
            root
            / "metadata/memory_admission_feedback_service_singletons"
            / f"candidate-{candidate_id}.json"
        )
        source_snapshot = root / "metadata" / source_file.name
        dependencies = [
            *_ordinary_dependencies(binary, control, trace),
            source_file,
            source_snapshot,
            candidate_file,
        ]
    elif family == "write-completion":
        candidate_file, candidate = ADMISSION.validated_write_completion_feedback()[candidate_id]
        snapshot = root / f"metadata/memory_write_completion_feedback_{candidate_id}.json"
        dependencies = [
            *_ordinary_dependencies(binary, control, trace),
            candidate_file,
            snapshot,
        ]
    else:
        source_file, candidate = ADMISSION.validated_red_rmw_completion_sweep()[candidate_id]
        candidate_file = (
            root
            / "metadata/red_rmw_completion_singletons"
            / f"candidate-{candidate_id}.json"
        )
        source_snapshot = root / "metadata" / source_file.name
        dependencies = [
            *_ordinary_dependencies(binary, control, trace),
            source_file,
            source_snapshot,
            candidate_file,
        ]
    expectation = REPLAY.ReplayExpectation(
        launches=launches,
        carded_dram=False,
        feedback_candidate=candidate,
    )
    first = path / "replay-1.json"
    second = path / "replay-2.json"
    return first, second, dependencies, expectation, trace


def _pair(
    root: Path,
    directory: str,
    key: str,
) -> tuple[Path, Path]:
    destination = root / directory / key
    return destination / "replay-1.json", destination / "replay-2.json"


def replay_result_directories(root: Path) -> set[Path]:
    return {
        path.parent.resolve()
        for name in ("replay-1.json", "replay-2.json")
        for path in root.rglob(name)
    }


def _load_content_binding(path: Path, binary: Path) -> CONTENT_BINDING.ContentBinding:
    if not path.is_file():
        raise ValueError(f"active replay content binding is missing: {path}")
    if path == CONTENT_BINDING_PATH and CONTENT_BINDING.sha256(path) != CONTENT_BINDING_SHA256:
        raise ValueError("active replay content binding hash does not match the pinned record")
    document = REPLAY.load_json_strict(path)
    if not isinstance(document, dict):
        raise ValueError("active replay content binding root is malformed")
    return CONTENT_BINDING.ContentBinding(document, binary=binary)


def _validate_pair(
    label: str,
    group: str,
    first: Path,
    second: Path,
    dependencies: Sequence[Path],
    expectation: Any,
    trace: Path,
    semantic_rows: list[dict[str, Any]],
    content_binding: CONTENT_BINDING.ContentBinding | None = None,
) -> str | None:
    """Run content-bound structural, twin, and trace-semantic validation.

    Content-bound pairs use direct schema validation rather than replay mtime
    ordering; the binary-build freshness check remains enforced at startup.
    """

    if not first.is_file() or not second.is_file():
        # Let the shared replay validator return its precise missing/empty
        # diagnostic; same-file checks require both paths to exist.
        pass
    else:
        try:
            if first.resolve() == second.resolve() or os.path.samefile(first, second):
                return "replay pair paths resolve to one file"
        except OSError as exc:
            return f"replay pair file identity is unreadable: {exc}"
    if content_binding is not None:
        content_error = content_binding.check_pair(
            group, label, first, second, dependencies
        )
        if content_error is not None:
            return content_error
    if content_binding is not None:
        error = None
        for path in (first, second):
            error = REPLAY.direct_replay_schema_error(path, expectation)
            if error is not None:
                break
        if error is None:
            try:
                if first.read_bytes() != second.read_bytes():
                    error = "replay pair is not byte-identical"
            except OSError as exc:
                error = f"could not read replay pair: {exc}"
    else:
        error = REPLAY.replay_pair_error(first, second, dependencies, expectation)
    if error is not None:
        return error
    try:
        document = REPLAY.load_json_strict(first)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        return f"{first}: result is not strict JSON: {exc}"
    if not isinstance(document, dict):
        return f"{first}: result root is not an object"
    rows, semantic_errors = validate_replay_document(document, trace, label=label)
    try:
        trace_path = str(trace.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        trace_path = str(trace)
    for row in rows:
        row["trace_path"] = trace_path
        expected = row.get("trace_expected")
        if isinstance(expected, dict) and isinstance(expected.get("trace_dir"), str):
            try:
                expected["trace_dir"] = str(
                    Path(expected["trace_dir"])
                    .resolve()
                    .relative_to(REPO_ROOT.resolve())
                )
            except ValueError:
                pass
    semantic_rows.extend(rows)
    if semantic_errors:
        return "; ".join(semantic_errors[:3])
    return None


def _validate_supplemental_directory(
    path: Path,
    root: Path,
    binary: Path,
    semantic_rows: list[dict[str, Any]],
    content_binding: CONTENT_BINDING.ContentBinding | None = None,
) -> str | None:
    """Validate a recognized supplemental directory before classifying it."""

    spec = _supplemental_pair_spec(path, root, binary)
    if spec is None:
        return "supplemental replay path could not be resolved"
    first, second, dependencies, expectation, trace = spec
    return _validate_pair(
        f"supplemental/{path.parent.name}/{path.name}",
        "supplemental",
        first,
        second,
        dependencies,
        expectation,
        trace,
        semantic_rows,
        content_binding,
    )


def _ordinary_dependencies(
    binary: Path,
    control: Path,
    trace: Path,
) -> list[Path]:
    return [
        binary,
        control,
        REPO_ROOT / "arch_profiles/cards/sm_86.json",
        REPO_ROOT / "data/pchase/sm_86_static_card.json",
        trace,
    ]


def launch_pairs(root: Path, binary: Path) -> Iterator[tuple]:
    control = root / "sass/launch_cycle_semantics_trace.control_bits.json"
    for cell in LAUNCH.configured_cells():
        trace = root / "nvbit" / cell.key
        launches = REPLAY.expected_launches_from_trace(
            trace, grid=(cell.blocks, 1, 1), block=(32, 1, 1)
        )
        if len(launches) != cell.chain_length:
            raise RuntimeError(
                f"{cell.key}: trace has {len(launches)} launches, "
                f"expected {cell.chain_length}"
            )
        dependencies = _ordinary_dependencies(binary, control, trace)
        for directory, carded in (
            ("simulator", False),
            ("simulator-carded-dram", True),
        ):
            expectation = REPLAY.ReplayExpectation(
                launches=launches,
                carded_dram=carded,
                cache_scope="stream-l2",
            )
            first, second = _pair(root, directory, cell.key)
            yield f"launch/{directory}/{cell.key}", first, second, dependencies, expectation


def retirement_pairs(root: Path, binary: Path) -> Iterator[tuple]:
    control = root / "sass/memory_retirement_service_trace.control_bits.json"
    for cell in RETIREMENT.configured_cells():
        trace = root / "nvbit" / cell.key
        launches = REPLAY.expected_launches_from_trace(
            trace, grid=(cell.blocks, 1, 1), block=(32, 1, 1)
        )
        dependencies = _ordinary_dependencies(binary, control, trace)
        for directory, carded in (
            ("simulator", False),
            ("simulator-carded-dram", True),
        ):
            expectation = REPLAY.ReplayExpectation(
                launches=launches, carded_dram=carded
            )
            first, second = _pair(root, directory, cell.key)
            yield (
                f"retirement/{directory}/{cell.key}",
                first,
                second,
                dependencies,
                expectation,
            )


def arrival_pairs(root: Path, binary: Path) -> Iterator[tuple]:
    control = root / "sass/l1tex_arrival_replay_service_trace.control_bits.json"
    candidates = REPLAY.candidate_configurations(ARRIVAL.CANDIDATE_FILE)
    for cell in ARRIVAL.configured_cells():
        trace = root / "nvbit" / cell.key
        launches = REPLAY.expected_launches_from_trace(
            trace,
            grid=(cell.blocks, 1, 1),
            block=(cell.warps * 32, 1, 1),
        )
        ordinary_dependencies = _ordinary_dependencies(binary, control, trace)
        for directory, carded in (
            ("simulator", False),
            ("simulator-carded-dram", True),
        ):
            expectation = REPLAY.ReplayExpectation(
                launches=launches, carded_dram=carded
            )
            first, second = _pair(root, directory, cell.key)
            yield (
                f"arrival/{directory}/{cell.key}",
                first,
                second,
                ordinary_dependencies,
                expectation,
            )
        candidate_dependencies = [
            *ordinary_dependencies,
            ARRIVAL.CANDIDATE_FILE,
            root / "metadata/memory_service_candidates.json",
        ]
        expectation = REPLAY.ReplayExpectation(
            launches=launches,
            carded_dram=False,
            service_candidates=candidates,
        )
        first, second = _pair(
            root, "simulator-memory-service-candidates", cell.key
        )
        yield (
            f"arrival/simulator-memory-service-candidates/{cell.key}",
            first,
            second,
            candidate_dependencies,
            expectation,
        )


def admission_pairs(root: Path, binary: Path) -> Iterator[tuple]:
    control = root / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    service_candidates = REPLAY.candidate_configurations(ADMISSION.CANDIDATE_FILE)
    admission_candidates = REPLAY.candidate_configurations(
        ADMISSION.ADMISSION_CANDIDATE_FILE
    )
    for cell in ADMISSION.configured_cells():
        trace = root / "nvbit" / cell.key
        launches = REPLAY.expected_launches_from_trace(
            trace,
            grid=(cell.blocks, 1, 1),
            block=(cell.warps * 32, 1, 1),
        )
        ordinary_dependencies = _ordinary_dependencies(binary, control, trace)

        expectation = REPLAY.ReplayExpectation(
            launches=launches, carded_dram=False
        )
        first, second = _pair(root, "simulator", cell.key)
        yield (
            f"admission/simulator/{cell.key}",
            first,
            second,
            ordinary_dependencies,
            expectation,
        )

        expectation = REPLAY.ReplayExpectation(
            launches=launches,
            carded_dram=False,
            service_candidates=service_candidates,
        )
        first, second = _pair(
            root, "simulator-memory-service-candidates", cell.key
        )
        yield (
            f"admission/simulator-memory-service-candidates/{cell.key}",
            first,
            second,
            [
                *ordinary_dependencies,
                ADMISSION.CANDIDATE_FILE,
                root / "metadata/memory_service_candidates.json",
            ],
            expectation,
        )

        expectation = REPLAY.ReplayExpectation(
            launches=launches,
            carded_dram=False,
            admission_candidates=admission_candidates,
        )
        first, second = _pair(
            root, "simulator-memory-admission-candidates", cell.key
        )
        yield (
            f"admission/simulator-memory-admission-candidates/{cell.key}",
            first,
            second,
            [
                *ordinary_dependencies,
                ADMISSION.ADMISSION_CANDIDATE_FILE,
                root / "metadata/memory_admission_candidates.json",
            ],
            expectation,
        )

        for candidate_id, candidate_file in ADMISSION.FEEDBACK_CANDIDATE_FILES.items():
            candidates = REPLAY.candidate_configurations(
                candidate_file, require_exactly_one=True
            )
            expectation = REPLAY.ReplayExpectation(
                launches=launches,
                carded_dram=False,
                feedback_candidate=candidates[0],
            )
            directory = f"simulator-memory-admission-feedback-{candidate_id}"
            first, second = _pair(root, directory, cell.key)
            yield (
                f"admission/{directory}/{cell.key}",
                first,
                second,
                [
                    *ordinary_dependencies,
                    candidate_file,
                    root / f"metadata/memory_admission_feedback_{candidate_id}.json",
                ],
                expectation,
            )


def admission_service_sweep_pairs(root: Path, binary: Path) -> Iterator[tuple]:
    control = root / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    candidates = ADMISSION.validated_feedback_service_sweep()
    full_ids = set(range(105, 113)) | set(range(121, 129))
    for candidate_id, (source_file, candidate) in candidates.items():
        tail = "store" if candidate["operation"] == "store" else "red"
        cells = (
            ADMISSION.configured_cells()
            if candidate_id in full_ids
            else [
                cell
                for cell in ADMISSION.configured_cells()
                if cell.stage == "burst_depth"
                and cell.depth <= 16
                and cell.tail in {"control", tail}
            ]
        )
        singleton = (
            root
            / "metadata/memory_admission_feedback_service_singletons"
            / f"candidate-{candidate_id}.json"
        )
        source_snapshot = root / "metadata" / source_file.name
        for cell in cells:
            trace = root / "nvbit" / cell.key
            expectation = REPLAY.ReplayExpectation(
                launches=REPLAY.expected_launches_from_trace(
                    trace,
                    grid=(cell.blocks, 1, 1),
                    block=(cell.warps * 32, 1, 1),
                ),
                carded_dram=False,
                feedback_candidate=candidate,
            )
            directory = f"simulator-memory-admission-feedback-service-{candidate_id}"
            first, second = _pair(root, directory, cell.key)
            yield (
                f"admission-service/{candidate_id}/{cell.key}",
                first,
                second,
                [
                    *_ordinary_dependencies(binary, control, trace),
                    source_file,
                    source_snapshot,
                    singleton,
                ],
                expectation,
            )


def write_completion_pairs(root: Path, binary: Path) -> Iterator[tuple]:
    control = root / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    candidates = ADMISSION.validated_write_completion_feedback()
    full_ids = {201, 202, 203, 204, 207, 208, 209}
    for candidate_id, (candidate_file, candidate) in candidates.items():
        snapshot = root / f"metadata/memory_write_completion_feedback_{candidate_id}.json"
        tail = "store" if candidate["operation"] == "store" else "red"
        cells = (
            ADMISSION.configured_cells()
            if candidate_id in full_ids
            else [
                cell
                for cell in ADMISSION.configured_cells()
                if cell.stage == "burst_depth"
                and cell.depth <= 16
                and cell.tail in {"control", tail}
            ]
        )
        for cell in cells:
            trace = root / "nvbit" / cell.key
            expectation = REPLAY.ReplayExpectation(
                launches=REPLAY.expected_launches_from_trace(
                    trace,
                    grid=(cell.blocks, 1, 1),
                    block=(cell.warps * 32, 1, 1),
                ),
                carded_dram=False,
                feedback_candidate=candidate,
            )
            directory = f"simulator-memory-write-completion-feedback-{candidate_id}"
            first, second = _pair(root, directory, cell.key)
            yield (
                f"write-completion/{candidate_id}/{cell.key}",
                first,
                second,
                [
                    *_ordinary_dependencies(binary, control, trace),
                    candidate_file,
                    snapshot,
                ],
                expectation,
            )


def red_rmw_completion_pairs(root: Path, binary: Path) -> Iterator[tuple]:
    control = root / "sass/l1tex_admission_backpressure_trace.control_bits.json"
    candidates = ADMISSION.validated_red_rmw_completion_sweep()
    for candidate_id, (candidate_file, candidate) in candidates.items():
        source_snapshot = root / "metadata" / candidate_file.name
        singleton = (
            root
            / "metadata/red_rmw_completion_singletons"
            / f"candidate-{candidate_id}.json"
        )
        for cell in ADMISSION.configured_cells():
            trace = root / "nvbit" / cell.key
            expectation = REPLAY.ReplayExpectation(
                launches=REPLAY.expected_launches_from_trace(
                    trace,
                    grid=(cell.blocks, 1, 1),
                    block=(cell.warps * 32, 1, 1),
                ),
                carded_dram=False,
                feedback_candidate=candidate,
            )
            directory = f"simulator-red-rmw-completion-{candidate_id}"
            first, second = _pair(root, directory, cell.key)
            yield (
                f"red-rmw-completion/{candidate_id}/{cell.key}",
                first,
                second,
                [
                    *_ordinary_dependencies(binary, control, trace),
                    candidate_file,
                    source_snapshot,
                    singleton,
                ],
                expectation,
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--launch-root", type=Path, default=LAUNCH.DEFAULT_OUTPUT
    )
    result.add_argument(
        "--retirement-root", type=Path, default=RETIREMENT.DEFAULT_OUTPUT
    )
    result.add_argument(
        "--arrival-root", type=Path, default=ARRIVAL.DEFAULT_OUTPUT
    )
    result.add_argument(
        "--admission-root", type=Path, default=ADMISSION.DEFAULT_OUTPUT
    )
    result.add_argument(
        "--semantic-report",
        type=Path,
        help="optional JSON report containing per-pair trace and readiness diagnostics",
    )
    result.add_argument(
        "--content-binding",
        type=Path,
        default=CONTENT_BINDING_PATH,
        help="content-identity aggregate for the retained replay corpus",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    binary = REPO_ROOT / "sim/target/release/sim-run"
    if not binary.is_file() or binary.stat().st_size == 0:
        print(
            "active replay validation: release sim-run is missing; build it first",
            file=sys.stderr,
        )
        return 2
    build_inputs = [REPO_ROOT / "sim/Cargo.toml", REPO_ROOT / "sim/Cargo.lock"]
    for crate in SIMULATOR_CRATES:
        crate_root = REPO_ROOT / "sim" / crate
        build_inputs.extend((crate_root / "Cargo.toml", crate_root / "src"))
        if (crate_root / "build.rs").is_file():
            build_inputs.append(crate_root / "build.rs")
    try:
        newest_source = REPLAY.latest_input_mtime_ns(build_inputs)
    except (OSError, RuntimeError) as error:
        print(f"active replay validation: {error}", file=sys.stderr)
        return 2
    if binary.stat().st_mtime_ns <= newest_source:
        print(
            "active replay validation: release sim-run does not postdate its build inputs; "
            "run `cargo build --release --manifest-path sim/Cargo.toml -p sim-run` first",
            file=sys.stderr,
        )
        return 2
    try:
        content_binding = _load_content_binding(
            args.content_binding.expanduser().resolve(), binary
        )
        groups = (
            ("launch", launch_pairs(args.launch_root.resolve(), binary)),
            (
                "retirement",
                retirement_pairs(args.retirement_root.resolve(), binary),
            ),
            ("arrival", arrival_pairs(args.arrival_root.resolve(), binary)),
            ("admission", admission_pairs(args.admission_root.resolve(), binary)),
            (
                "admission-service-sweep",
                admission_service_sweep_pairs(args.admission_root.resolve(), binary),
            ),
            (
                "write-completion",
                write_completion_pairs(args.admission_root.resolve(), binary),
            ),
            (
                "red-rmw-completion",
                red_rmw_completion_pairs(args.admission_root.resolve(), binary),
            ),
        )
        total = 0
        failures: list[str] = []
        expected_directories: set[Path] = set()
        semantic_rows: list[dict[str, Any]] = []
        group_counts: dict[str, int] = {}
        group_roots = {
            "launch": args.launch_root.resolve(),
            "retirement": args.retirement_root.resolve(),
            "arrival": args.arrival_root.resolve(),
            "admission": args.admission_root.resolve(),
            "admission-service-sweep": args.admission_root.resolve(),
            "write-completion": args.admission_root.resolve(),
            "red-rmw-completion": args.admission_root.resolve(),
        }
        for group_name, pairs in groups:
            group_count = 0
            for label, first, second, dependencies, expectation in pairs:
                group_count += 1
                total += 1
                expected_directories.add(first.parent.resolve())
                trace = group_roots[group_name] / "nvbit" / first.parent.name
                error = _validate_pair(
                    label,
                    group_name,
                    first,
                    second,
                    dependencies,
                    expectation,
                    trace,
                    semantic_rows,
                    content_binding,
                )
                if error is not None:
                    failures.append(f"{label}: {error}")
            group_counts[group_name] = group_count
            print(f"{group_name}: {group_count} replay pairs")
        actual_directories = set().union(
            replay_result_directories(args.launch_root.resolve()),
            replay_result_directories(args.retirement_root.resolve()),
            replay_result_directories(args.arrival_root.resolve()),
            replay_result_directories(args.admission_root.resolve()),
        )
        supplemental = {
            path
            for path in actual_directories - expected_directories
            if _known_supplemental_replay_directory(path, args.admission_root.resolve())
        }
        supplemental_validated = 0
        for extra in sorted(supplemental):
            error = _validate_supplemental_directory(
                extra,
                args.admission_root.resolve(),
                binary,
                semantic_rows,
                content_binding,
            )
            if error is not None:
                failures.append(f"supplemental/{extra.parent.name}/{extra.name}: {error}")
            else:
                supplemental_validated += 1
        for extra in sorted(
            actual_directories - expected_directories - supplemental
        ):
            failures.append(f"unexpected inactive replay directory: {extra}")
        if supplemental:
            print(
                f"validated {supplemental_validated} known supplemental full-campaign replay directories"
            )
        group_counts["supplemental"] = supplemental_validated
        failures.extend(content_binding.finish(group_counts))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"active replay validation: {error}", file=sys.stderr)
        return 2

    if total != EXPECTED_PAIR_COUNT:
        failures.append(
            f"active replay matrix has {total} pairs, expected {EXPECTED_PAIR_COUNT}"
        )
    if args.semantic_report is not None:
        report_path = args.semantic_report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "validator": "workloads/research_probes/validate_active_replay_pairs.py",
            "content_binding": {
                "path": str(args.content_binding.expanduser().resolve()),
                "sha256": CONTENT_BINDING.sha256(args.content_binding.expanduser().resolve()),
                "corpus_root": content_binding.document.get("corpus_root"),
                "active_pair_count": content_binding.document.get("active_pair_count"),
                "supplemental_pair_count": content_binding.document.get("supplemental_pair_count"),
            },
            "active_pair_count": total,
            "group_counts": group_counts,
            "semantic_launch_count": len(semantic_rows),
            "supplemental_directory_count": len(supplemental),
            "passed": not failures,
            "failures": failures,
            "rows": semantic_rows,
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if failures:
        for failure in failures[:20]:
            print(f"FAIL: {failure}", file=sys.stderr)
        if len(failures) > 20:
            print(
                f"FAIL: {len(failures) - 20} additional failures omitted",
                file=sys.stderr,
            )
        return 1
    print(f"validated {total} fresh, schema-matched, byte-identical replay pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
