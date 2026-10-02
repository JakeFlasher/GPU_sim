#!/usr/bin/env python3
"""Compare event-driven and dense-reference SM86 replay outputs.

The simulator binary must be built with ``--features reference``.  The dense
lane is a bounded differential oracle; it is not a production timing mode.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import re
import shlex
import subprocess
import time
from typing import Any, Iterator

DEFAULT_MAX_RECORDS = 8_192


DIAGNOSTICS = re.compile(
    r"SIM_REFERENCE_PERF mode=(?P<mode>event|dense) calls=(?P<calls>\d+) "
    r"local_cycle_steps=(?P<local_cycle_steps>\d+) "
    r"idle_jumps=(?P<idle_jumps>\d+) "
    r"skipped_cycles=(?P<skipped_cycles>\d+)"
)


def _discard_partial_trace(path: pathlib.Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise RuntimeError(f"could not remove invalid partial reference trace {path}: {error}") from error


def run_mode(
    command: list[str],
    cwd: pathlib.Path,
    mode: str,
    timeout: float,
    trace_path: pathlib.Path | None,
    max_records: int,
) -> tuple[dict[str, Any], dict[str, int], int]:
    # The child interprets relative paths from ``cwd`` while this process
    # performs cleanup and later comparison from its own working directory.
    # Make the path absolute once at the boundary so both sides address the
    # same artifact without following a caller-supplied symlink.
    if trace_path is not None:
        trace_path = trace_path.absolute()
    environment = os.environ.copy()
    environment["SIM_SM_REFERENCE_MODE"] = mode
    # A caller's tracing environment must not silently redirect either lane.
    environment.pop("SIM_SM_REFERENCE_TRACE", None)
    environment.pop("SIM_SM_REFERENCE_MAX_RECORDS", None)
    if trace_path is not None:
        environment["SIM_SM_REFERENCE_TRACE"] = str(trace_path)
        environment["SIM_SM_REFERENCE_MAX_RECORDS"] = str(max_records)
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        _discard_partial_trace(trace_path)
        raise RuntimeError(
            f"{mode} run timed out after {timeout}s: {' '.join(command)}"
        ) from error
    if completed.returncode != 0:
        _discard_partial_trace(trace_path)
        raise RuntimeError(
            f"{mode} run failed with {completed.returncode}: {' '.join(command)}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        _discard_partial_trace(trace_path)
        raise RuntimeError(f"{mode} run emitted invalid JSON: {error}") from error
    if not isinstance(document, dict):
        _discard_partial_trace(trace_path)
        raise RuntimeError(f"{mode} run emitted a non-object JSON document")
    match = DIAGNOSTICS.search(completed.stderr)
    if match is None:
        _discard_partial_trace(trace_path)
        raise RuntimeError(
            "reference diagnostics are missing; build sim-run with `--features reference`"
        )
    if match.group("mode") != mode:
        _discard_partial_trace(trace_path)
        raise RuntimeError(
            f"reference diagnostics report mode {match.group('mode')!r}, expected {mode!r}"
        )
    diagnostics = {
        key: int(value)
        for key, value in match.groupdict().items()
        if key != "mode"
    }
    return document, diagnostics, time.perf_counter_ns() - started


def command_files(args: argparse.Namespace) -> list[pathlib.Path]:
    files = list(args.command_file)
    directories = args.command_dir or []
    if isinstance(directories, pathlib.Path):
        directories = [directories]
    for directory in directories:
        matches = sorted(directory.glob("*/command.txt"))
        if not matches:
            raise SystemExit(f"no workload command files in directory: {directory}")
        files.extend(matches)
    if not files:
        raise SystemExit("provide --command-file or --command-dir")
    return files


def parse_path_rewrites(values: list[str]) -> list[tuple[str, str]]:
    """Parse explicit ``OLD=NEW`` command-path substitutions.

    Retained command artifacts can contain absolute paths from the host that
    produced them. Rewriting is opt-in and literal: the comparator never
    guesses a replacement root or silently changes a command.
    """
    rewrites: list[tuple[str, str]] = []
    for value in values:
        old, separator, new = value.partition("=")
        if not separator or not old or not new:
            raise SystemExit(
                f"--path-rewrite expects OLD=NEW with non-empty paths; got {value!r}"
            )
        rewrites.append((old, new))
    return rewrites


def rewrite_command(command: list[str], rewrites: list[tuple[str, str]]) -> list[str]:
    """Replace exact path roots in parsed arguments, in declaration order.

    Parsing precedes rewriting so spaces, quotes, and shell metacharacters in
    a replacement stay in one argument. Match a complete root or its children,
    never a sibling whose name happens to start with the same text.
    """
    rewritten = []
    for argument in command:
        option = ""
        if argument.startswith("--") and "=" in argument:
            name, argument = argument.split("=", 1)
            option = name + "="
        for old, new in rewrites:
            root = old.rstrip("/") or "/"
            prefix = root.rstrip("/") + "/"
            if argument == root:
                argument = new
            elif argument.startswith(prefix):
                argument = new.rstrip("/") + "/" + argument[len(prefix):]
        rewritten.append(option + argument)
    return rewritten


def _apply_replacement(state: Any, path: list[str | int], value: Any) -> Any:
    if not path:
        return value
    parent = state
    for segment in path[:-1]:
        parent = _path_child(parent, segment)
    last = path[-1]
    _path_child(parent, last)  # Replacements cannot invent a missing path.
    parent[last] = value
    return state


def _path_child(parent: Any, segment: str | int) -> Any:
    if isinstance(parent, dict) and isinstance(segment, str) and segment in parent:
        return parent[segment]
    if isinstance(parent, list) and type(segment) is int and 0 <= segment < len(parent):
        return parent[segment]
    raise ValueError(f"invalid boundary replacement path component {segment!r}")


def _apply_splice(
    state: Any,
    path: list[str | int],
    start: int,
    delete_count: int,
    values: list[Any],
) -> Any:
    target = state if not path else _path_child_path(state, path)
    if not isinstance(target, list):
        raise ValueError("splice target is not an array")
    if type(start) is not int or type(delete_count) is not int:
        raise ValueError("splice bounds must be integers")
    if start < 0 or delete_count < 0 or start > len(target) or delete_count > len(target) - start:
        raise ValueError("splice bounds exceed array length")
    if not isinstance(values, list):
        raise ValueError("splice values must be an array")
    target[start : start + delete_count] = values
    return state


def _path_child_path(parent: Any, path: list[str | int]) -> Any:
    for segment in path:
        parent = _path_child(parent, segment)
    return parent


def _trace_records(path: pathlib.Path, mode: str) -> Iterator[dict[str, Any]]:
    """Yield one reconstructed record at a time, reusing the event state.

    Consumers must compare each record before advancing this iterator. Memory
    use depends on one boundary, not on the number of events in the workload.
    """
    with path.open() as stream:
        header = json.loads(next(stream, "null"))
        if not isinstance(header, dict) or header.get("kind") != "header":
            raise RuntimeError(f"reference trace is missing its header: {path}")
        if header.get("state_projection_complete") is not False:
            raise RuntimeError(f"reference trace has an unexpected completeness claim: {path}")
        if header.get("mode") != mode:
            raise RuntimeError(f"reference trace mode differs from {mode!r}: {path}")
        encoding = header.get("event_encoding", "full")
        if encoding not in ("full", "full_then_delta_v1", "full_then_delta_v2"):
            raise RuntimeError(f"unsupported reference encoding {encoding!r}: {path}")
        # Mode and storage encoding may differ; all scientific metadata must agree.
        yield {key: value for key, value in header.items() if key not in ("mode", "event_encoding")}
        previous = None
        events = 0
        for index, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object reference record {index}: {path}")
            kind = row.get("kind")
            if kind == "footer":
                if row.get("records_before_footer") != index or row.get("queue_events_popped") != events:
                    raise RuntimeError(f"reference footer count mismatch: {path}")
                if next(stream, None) is not None:
                    raise RuntimeError(f"reference trace has records after its footer: {path}")
                yield row
                return
            if kind == "event":
                previous = row
            elif kind == "event_delta" and encoding == "full_then_delta_v1":
                if previous is None or row.get("base_record") != index - 1:
                    raise RuntimeError(f"reference delta has no matching base at record {index}: {path}")
                replacements = row.get("replacements")
                if not isinstance(replacements, list):
                    raise RuntimeError(f"reference delta has no replacement list: {path}")
                try:
                    for replacement in replacements:
                        if not isinstance(replacement, dict) or set(replacement) != {"path", "value"}:
                            raise ValueError("replacement must contain path and value")
                        if not isinstance(replacement["path"], list):
                            raise ValueError("replacement path must be an array")
                        previous = _apply_replacement(previous, replacement["path"], replacement["value"])
                except (TypeError, ValueError) as error:
                    raise RuntimeError(f"malformed reference delta at record {index}: {path}: {error}") from error
                if not isinstance(previous, dict) or previous.get("kind") != "event":
                    raise RuntimeError(f"reference delta did not reconstruct an event: {path}")
            elif kind == "event_delta" and encoding == "full_then_delta_v2":
                if previous is None or row.get("base_record") != index - 1:
                    raise RuntimeError(f"reference delta has no matching base at record {index}: {path}")
                operations = row.get("operations")
                if not isinstance(operations, list):
                    raise RuntimeError(f"reference delta has no operation list: {path}")
                try:
                    for operation in operations:
                        if not isinstance(operation, dict) or not isinstance(operation.get("path"), list):
                            raise ValueError("operation must contain a path")
                        op = operation.get("op")
                        if op == "replace":
                            if set(operation) != {"op", "path", "value"}:
                                raise ValueError("replace operation has unexpected fields")
                            previous = _apply_replacement(previous, operation["path"], operation["value"])
                        elif op == "splice":
                            if set(operation) != {"op", "path", "start", "delete_count", "values"}:
                                raise ValueError("splice operation has unexpected fields")
                            previous = _apply_splice(
                                previous,
                                operation["path"],
                                operation["start"],
                                operation["delete_count"],
                                operation["values"],
                            )
                        else:
                            raise ValueError(f"unknown delta operation {op!r}")
                except (TypeError, ValueError) as error:
                    raise RuntimeError(f"malformed reference delta at record {index}: {path}: {error}") from error
                if not isinstance(previous, dict) or previous.get("kind") != "event":
                    raise RuntimeError(f"reference delta did not reconstruct an event: {path}")
            else:
                raise RuntimeError(f"unexpected reference record kind {kind!r}: {path}")
            events += 1
            yield previous
    raise RuntimeError(f"reference trace is missing its footer: {path}")


def compare_boundary_traces(event_path: pathlib.Path, dense_path: pathlib.Path) -> dict[str, Any]:
    count = 0
    first_difference = None
    for index, (event, dense) in enumerate(itertools.zip_longest(
        _trace_records(event_path, "event"), _trace_records(dense_path, "dense")
    )):
        if event is not None and event.get("kind") == "event":
            count += 1
        if event != dense and first_difference is None:
            first_difference = index
    return {
        "boundary_trace_equal": first_difference is None,
        "boundary_event_records": count,
        "first_different_boundary_record": first_difference,
    }


def summarize_native_boundary_refinement(path: pathlib.Path, mode: str) -> dict[str, Any]:
    """Summarize typed native-boundary closure for an exact-fallback check.

    The reference stream is an independently replayed ``SmCore`` projection;
    it is not a macro-model oracle.  A record is eligible for preservation only
    when its typed boundary declares ``complete``.  Any omitted or incomplete
    field is counted as an exact-replay fallback reason instead of being
    replaced by a finite estimate.
    """
    records = 0
    complete = 0
    fallback = 0
    unknown_fields: dict[str, int] = {}
    for row in _trace_records(path, mode):
        if row.get("kind") != "event":
            continue
        records += 1
        boundary = row.get("typed_boundary")
        completeness = boundary.get("completeness") if isinstance(boundary, dict) else None
        refinement = row.get("native_boundary_refinement")
        refinement_decision = refinement.get("decision") if isinstance(refinement, dict) else None
        # Older reference streams have no comparator field; their complete
        # typed boundary remains a structural eligibility count.  Once an
        # explicit comparator decision exists, only `preserve` is eligible.
        if refinement_decision == "reify":
            fallback += 1
            reason = refinement.get("reason") if isinstance(refinement, dict) else None
            if isinstance(reason, dict) and len(reason) == 1:
                kind, detail = next(iter(reason.items()))
                fields = [
                    f"native_reification:{kind}:{detail}"
                    if isinstance(detail, (str, int, float, bool))
                    else f"native_reification:{kind}"
                ]
            else:
                fields = ["native_reification:malformed_reason"]
            for field in fields:
                unknown_fields[field] = unknown_fields.get(field, 0) + 1
            continue
        if (refinement_decision in (None, "preserve")) and (
            completeness == "complete" or (
            isinstance(completeness, dict) and set(completeness) == {"complete"}
            )
        ):
            complete += 1
            continue
        fallback += 1
        if isinstance(completeness, dict) and set(completeness) == {"incomplete"}:
            values = completeness.get("incomplete", {}).get("unknown", [])
            if isinstance(values, list):
                fields = [value for value in values if isinstance(value, str)]
            else:
                fields = ["malformed_incomplete_unknown_list"]
        elif isinstance(completeness, dict) and set(completeness) == {"refused"}:
            reason = completeness.get("refused", {}).get("reason")
            fields = [f"refused:{reason}" if isinstance(reason, str) else "refused"]
        else:
            fields = ["typed_boundary_completeness_missing_or_malformed"]
        for field in fields:
            unknown_fields[field] = unknown_fields.get(field, 0) + 1
    return {
        "native_boundary_records": records,
        "complete_boundary_records": complete,
        "exact_fallback_boundary_records": fallback,
        "exact_fallback_required": fallback != 0,
        "unknown_or_refused_fields": dict(sorted(unknown_fields.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-file", type=pathlib.Path, action="append", default=[])
    parser.add_argument(
        "--command-dir", type=pathlib.Path, action="append", default=[],
        help="directory containing workload */command.txt files; repeatable",
    )
    parser.add_argument("--cwd", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--trace-dir", type=pathlib.Path)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="replace an exact path root in parsed command arguments; repeatable",
    )
    args = parser.parse_args()

    if args.max_records <= 1:
        raise SystemExit("--max-records must be greater than 1")
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    path_rewrites = parse_path_rewrites(args.path_rewrite)

    reports: list[dict[str, Any]] = []
    for run_index, command_file in enumerate(command_files(args)):
        command = rewrite_command(shlex.split(command_file.read_text()), path_rewrites)
        if not command:
            raise SystemExit(f"empty command file: {command_file}")
        trace_stem = f"{run_index:04d}_{command_file.parent.name}"
        event_trace = None if args.trace_dir is None else args.trace_dir / f"{trace_stem}.event.jsonl"
        dense_trace = None if args.trace_dir is None else args.trace_dir / f"{trace_stem}.dense.jsonl"
        event, event_diag, event_time_ns = run_mode(
            command, args.cwd, "event", args.timeout, event_trace, args.max_records
        )
        dense, dense_diag, dense_time_ns = run_mode(
            command, args.cwd, "dense", args.timeout, dense_trace, args.max_records
        )
        report: dict[str, Any] = {
                "workload": command_file.parent.name,
                "command_file": str(command_file),
                "command": command,
                "outputs_equal": event == dense,
                "event": event_diag,
                "dense": dense_diag,
                "event_wall_time_ms": round(event_time_ns / 1_000_000, 3),
                "dense_wall_time_ms": round(dense_time_ns / 1_000_000, 3),
                "terminal_event_time": event.get("launches", [{}])[0]
                .get("timing", {})
                .get("terminal_modeled_event_time"),
        }
        if event_trace is not None and dense_trace is not None:
            report.update(compare_boundary_traces(event_trace, dense_trace))
        reports.append(report)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "reference_mode": "dense",
                "cwd": str(args.cwd.resolve()),
                "runs": reports,
                "all_outputs_equal": all(row["outputs_equal"] for row in reports),
                "all_boundary_traces_equal": (
                    all(row.get("boundary_trace_equal", True) for row in reports)
                    if args.trace_dir is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if any(not row["outputs_equal"] or row.get("boundary_trace_equal") is False for row in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
