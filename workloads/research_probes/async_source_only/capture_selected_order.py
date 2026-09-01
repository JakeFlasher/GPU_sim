#!/usr/bin/env python3
"""Capture and replay the bounded SM86 selected async-order campaign."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_wait_prefix import (  # noqa: E402
    CaptureError,
    identity,
    run,
    write_json,
)
from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)

BINARY = HERE / "async_cross_warp_queue_overlap"
ROOT = REPO_ROOT / "results/captures/sm86_async_selected_order_trace_20260831_v1"
PREDECESSOR_ROOT = (
    REPO_ROOT
    / "results/captures/sm86_async_cross_warp_queue_overlap_20260831_v1"
)
CONTROL = PREDECESSOR_ROOT / "static/control_bits.json"
STATIC_SUMMARY = PREDECESSOR_ROOT / "static/summary.json"
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"
ARCH_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"
NVBIT_CAPTURE = (
    REPO_ROOT / "workloads/research_probes/ldgsts_burst_release/capture_nvbit.py"
)
ROUTES = ("l1", "l2")
ARMS = ("queued", "drained")
REPETITIONS = (1, 2)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CaptureError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise CaptureError(f"{path}: expected one JSON object")
    return value


def stage_cells(stage: str) -> Iterable[tuple[int, str, str, int]]:
    if stage == "w4":
        for repetition in REPETITIONS:
            for route in ROUTES:
                for arm in ARMS:
                    yield repetition, route, arm, 4
        return
    if stage == "holdout":
        for repetition in REPETITIONS:
            for arm in ARMS:
                yield repetition, "l2", arm, 8
        return
    raise CaptureError(f"unknown selected-order stage {stage!r}")


def cell_key(route: str, arm: str, active_warps: int) -> str:
    return f"{route}_{arm}_w{active_warps}_g64"


def cell_directory(base: Path, stage: str, repetition: int, key: str) -> Path:
    return base / stage / f"repeat-{repetition}" / key


def expected_selected_count(active_warps: int) -> int:
    # Every background warp contributes 8 LDGSTS + 4 commits + 1 wait;
    # the target contributes 8 LDGSTS + 1 commit + 1 wait per epoch.
    return 64 * ((active_warps - 1) * 13 + 10)


def retained_binary() -> dict[str, Any]:
    summary = read_json(STATIC_SUMMARY)
    retained = summary.get("binary")
    if not isinstance(retained, dict) or not isinstance(retained.get("sha256"), str):
        raise CaptureError(f"{STATIC_SUMMARY}: malformed retained binary identity")
    current = identity(BINARY)
    if current["sha256"] != retained["sha256"]:
        raise CaptureError(
            "the queue-overlap binary differs from the exact-work predecessor; "
            "rebuild/revalidate static and NCU evidence before capture"
        )
    return current


def capture_is_current(destination: Path, binary: dict[str, Any]) -> bool:
    binding_path = destination / "campaign_binding.json"
    if not destination.exists():
        return False
    if not binding_path.is_file():
        raise CaptureError(f"partial capture must be moved aside: {destination}")
    binding = read_json(binding_path)
    environment = read_json(destination / "environment.json")
    logical = environment.get("environment")
    effective = environment.get("effective_environment")
    if (
        binding.get("binary", {}).get("sha256") != binary["sha256"]
        or binding.get("async_order_enabled") is not True
        or not isinstance(logical, dict)
        or logical.get("SIMBENCH_NVBIT_ASYNC_ORDER") != "1"
        or not isinstance(effective, dict)
        or effective.get("SIMBENCH_NVBIT_ASYNC_ORDER") != "1"
    ):
        raise CaptureError(f"stale or non-order capture must be moved aside: {destination}")
    complete = read_json(destination / "complete.json")
    if complete.get("complete") is not True:
        raise CaptureError(f"incomplete capture must be moved aside: {destination}")
    return True


def nvbit_stage(stage: str, timeout: int) -> None:
    binary = retained_binary()
    for repetition, route, arm, active_warps in stage_cells(stage):
        key = cell_key(route, arm, active_warps)
        destination = cell_directory(ROOT / "nvbit", stage, repetition, key)
        if capture_is_current(destination, binary):
            continue
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--property=Delegate=yes",
            "/usr/bin/python3",
            str(NVBIT_CAPTURE),
            "--output",
            str(destination),
            "--cell",
            f"selected_order_{stage}_r{repetition}_{key}",
            "--binary",
            str(BINARY),
            "--repetition",
            str(repetition),
            "--mode",
            "ncu",
            "--timeout",
            str(timeout),
            "--async-order",
            "--target-args",
            "--groups",
            "64",
            "--active-warps",
            str(active_warps),
            "--route",
            route,
            "--mode",
            "endpoint",
            "--arm",
            arm,
            "--markers",
            "on",
        ]
        run(command, timeout=timeout + 30)


def protobuf_closure(trace: Path) -> dict[str, str]:
    closure = {"dynamic_trace.pb": identity(trace / "dynamic_trace.pb")["sha256"]}
    for path in sorted((trace / "threadblocks").rglob("*.pb")):
        closure[str(path.relative_to(trace))] = identity(path)["sha256"]
    if len(closure) == 1:
        raise CaptureError(f"{trace}: no threadblock protobuf payloads")
    return closure


def reject_links(root: Path) -> None:
    links = [str(path) for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise CaptureError(f"{root}: order-stripped input contains symlinks: {links}")


def materialize_order_stripped_trace(
    source: Path, destination: Path, expected_count: int
) -> None:
    source_closure = protobuf_closure(source)
    record_path = destination / "order_stripped.json"
    if destination.exists():
        if not record_path.is_file():
            raise CaptureError(f"partial order-stripped trace must be moved aside: {destination}")
        record = read_json(record_path)
        if (
            record.get("source_protobuf_closure") != source_closure
            or record.get("cleared_instrumentation_order_fields") != expected_count
        ):
            raise CaptureError(f"stale order-stripped trace must be moved aside: {destination}")
        return

    reject_links(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    bindings = load_dynamic_trace_bindings()
    descriptor = bindings.instruction_pb2.instruction.DESCRIPTOR
    if "instrumentation_order" not in descriptor.fields_by_name:
        raise CaptureError("tracked protobuf binding has no instrumentation_order field")
    cleared = 0
    for path in sorted((destination / "threadblocks").rglob("*.pb")):
        threadblock = bindings.threadblock_pb2.threadblock()
        threadblock.ParseFromString(path.read_bytes())
        for warp in threadblock.warps.values():
            for instruction in warp.instructions:
                if instruction.HasField("instrumentation_order"):
                    instruction.ClearField("instrumentation_order")
                    cleared += 1
        path.write_bytes(threadblock.SerializeToString(deterministic=True))
    if cleared != expected_count:
        raise CaptureError(
            f"{source}: cleared {cleared} order fields, expected {expected_count}"
        )
    write_json(
        record_path,
        {
            "schema_version": 1,
            "kind": "sm86_async_order_stripped_trace",
            "source": str(source.resolve()),
            "source_protobuf_closure": source_closure,
            "cleared_instrumentation_order_fields": cleared,
            "derived_protobuf_closure": protobuf_closure(destination),
        },
    )


def replay_command(trace: Path) -> list[str]:
    return [
        str(SIM_RUN),
        "--trace",
        str(trace),
        "--control",
        str(CONTROL),
        "--arch-card",
        str(ARCH_CARD),
        "--pchase-card",
        str(PCHASE_CARD),
        "--mode",
        "l1-l2",
    ]


def parse_replay(payload: bytes, *, key: str, ordered: bool) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise CaptureError(f"{key}: sim-run output is not JSON: {error}") from error
    if not isinstance(value, dict):
        raise CaptureError(f"{key}: sim-run output is not an object")
    launches = value.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise CaptureError(f"{key}: replay must contain exactly one launch")
    timing = launches[0].get("timing")
    readiness = timing.get("readiness") if isinstance(timing, dict) else None
    if not isinstance(readiness, dict):
        raise CaptureError(f"{key}: replay has no readiness object")
    expected_schema = 21 if ordered else 20
    if readiness.get("schema_version") != expected_schema:
        raise CaptureError(
            f"{key}: readiness schema {readiness.get('schema_version')!r}, "
            f"expected {expected_schema}"
        )
    observation = readiness.get("async_instrumentation_order")
    if ordered and not isinstance(observation, dict):
        raise CaptureError(f"{key}: ordered replay has no order observation")
    if not ordered and observation is not None:
        raise CaptureError(f"{key}: stripped replay unexpectedly has order observation")
    return value


def replay_is_current(
    destination: Path,
    trace: Path,
    stripped: Path,
    binary: dict[str, Any],
) -> bool:
    command_path = destination / "command.json"
    if not destination.exists():
        return False
    if not command_path.is_file():
        raise CaptureError(f"partial replay must be moved aside: {destination}")
    command = read_json(command_path)
    outputs = command.get("ordered_outputs")
    baseline = command.get("stripped_output")
    if (
        command.get("binary", {}).get("sha256") != binary["sha256"]
        or command.get("sim_run", {}).get("sha256") != identity(SIM_RUN)["sha256"]
        or command.get("trace_protobuf_closure") != protobuf_closure(trace)
        or command.get("stripped_protobuf_closure") != protobuf_closure(stripped)
        or not isinstance(outputs, list)
        or len(outputs) != 2
        or not isinstance(baseline, dict)
    ):
        raise CaptureError(f"stale replay must be moved aside: {destination}")
    paths = [destination / "replay-1.json", destination / "replay-2.json"]
    if any(not path.is_file() for path in paths) or not (destination / "stripped.json").is_file():
        raise CaptureError(f"incomplete replay must be moved aside: {destination}")
    if identity(paths[0])["sha256"] != identity(paths[1])["sha256"]:
        raise CaptureError(f"{destination}: retained replay twins differ")
    return True


def replay_stage(stage: str, timeout: int) -> None:
    binary = retained_binary()
    for repetition, route, arm, active_warps in stage_cells(stage):
        key = cell_key(route, arm, active_warps)
        trace = cell_directory(ROOT / "nvbit", stage, repetition, key)
        stripped = cell_directory(ROOT / "order-stripped", stage, repetition, key)
        materialize_order_stripped_trace(
            trace, stripped, expected_selected_count(active_warps)
        )
        destination = cell_directory(ROOT / "simulator", stage, repetition, key)
        if replay_is_current(destination, trace, stripped, binary):
            continue
        ordered_command = replay_command(trace)
        stripped_command = replay_command(stripped)
        destination.mkdir(parents=True, exist_ok=False)
        ordered_outputs: list[dict[str, Any]] = []
        for replay_index in (1, 2):
            result = run(ordered_command, timeout=timeout)
            parse_replay(result.stdout, key=key, ordered=True)
            output = destination / f"replay-{replay_index}.json"
            output.write_bytes(result.stdout)
            (destination / f"stderr-{replay_index}.txt").write_bytes(result.stderr)
            ordered_outputs.append(identity(output))
        if ordered_outputs[0]["sha256"] != ordered_outputs[1]["sha256"]:
            raise CaptureError(f"{key}: ordered replay twins differ")
        baseline = run(stripped_command, timeout=timeout)
        parse_replay(baseline.stdout, key=key, ordered=False)
        stripped_output = destination / "stripped.json"
        stripped_output.write_bytes(baseline.stdout)
        (destination / "stripped.stderr.txt").write_bytes(baseline.stderr)
        write_json(
            destination / "command.json",
            {
                "schema_version": 1,
                "kind": "sm86_async_selected_order_replay",
                "stage": stage,
                "repetition": repetition,
                "route": route,
                "arm": arm,
                "active_warps": active_warps,
                "groups": 64,
                "ordered_command": ordered_command,
                "stripped_command": stripped_command,
                "binary": binary,
                "sim_run": identity(SIM_RUN),
                "control": identity(CONTROL),
                "arch_card": identity(ARCH_CARD),
                "pchase_card": identity(PCHASE_CARD),
                "trace_protobuf_closure": protobuf_closure(trace),
                "stripped_protobuf_closure": protobuf_closure(stripped),
                "ordered_outputs": ordered_outputs,
                "stripped_output": identity(stripped_output),
            },
        )


def authorize_holdout() -> dict[str, Any]:
    # Import lazily so ordinary W4 capture does not load protobuf bindings or
    # create any analysis output.
    from analyze_selected_order import build_smoke_authorization  # noqa: WPS433

    authorization_path = ROOT / "analysis/smoke-authorization.json"
    authorization = build_smoke_authorization(
        ROOT, require_holdout_absent=not authorization_path.is_file()
    )
    if authorization.get("holdout_authorized") is not True:
        raise CaptureError("W4 selected-order evidence did not authorize holdout")
    encoded = json.dumps(authorization, indent=2, sort_keys=True) + "\n"
    if authorization_path.is_file():
        if authorization_path.read_text(encoding="utf-8") != encoded:
            raise CaptureError("retained smoke authorization differs from current W4 evidence")
    else:
        authorization_path.parent.mkdir(parents=True, exist_ok=True)
        authorization_path.write_text(encoded, encoding="utf-8")
    return authorization


def holdout_phase(timeout: int) -> None:
    authorize_holdout()
    nvbit_stage("holdout", timeout)
    replay_stage("holdout", timeout)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("nvbit", "replay", "smoke", "holdout", "complete"),
        required=True,
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        if args.phase in ("nvbit", "smoke", "complete"):
            nvbit_stage("w4", args.timeout)
        if args.phase in ("replay", "smoke", "complete"):
            replay_stage("w4", args.timeout)
        if args.phase in ("holdout", "complete"):
            holdout_phase(args.timeout)
    except (CaptureError, OSError, ValueError, KeyError) as error:
        print(f"selected-order capture: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
