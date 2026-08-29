#!/usr/bin/env python3
"""Read-only validation of the active launch/memory replay-pair corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from typing import Iterator, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
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
        for group_name, pairs in groups:
            group_count = 0
            for label, first, second, dependencies, expectation in pairs:
                group_count += 1
                total += 1
                expected_directories.add(first.parent.resolve())
                error = REPLAY.replay_pair_error(
                    first, second, dependencies, expectation
                )
                if error is not None:
                    failures.append(f"{label}: {error}")
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
            if _known_supplemental_replay_directory(
                path, args.admission_root.resolve()
            )
        }
        for extra in sorted(
            actual_directories - expected_directories - supplemental
        ):
            failures.append(f"unexpected inactive replay directory: {extra}")
        if supplemental:
            print(
                f"ignored {len(supplemental)} known supplemental full-campaign replay directories"
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"active replay validation: {error}", file=sys.stderr)
        return 2

    if total != EXPECTED_PAIR_COUNT:
        failures.append(
            f"active replay matrix has {total} pairs, expected {EXPECTED_PAIR_COUNT}"
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
