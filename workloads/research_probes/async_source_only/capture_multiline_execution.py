#!/usr/bin/env python3
"""Capture static, direct, NVBit, and simulator evidence for the multiline assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = (
    REPO_ROOT / "results/captures/sm86_async_multiline_response_20260831_v1"
)
BINARY = HERE / "async_multiline_response"
CONTROL_TOOL = REPO_ROOT / "tools/sass_control/emit_control_bits_sidecar.py"
NVBIT_CAPTURE = (
    REPO_ROOT
    / "workloads/research_probes/ldgsts_burst_release/capture_nvbit.py"
)
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"
ARCH_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"
CUDA_BIN = Path(os.environ.get("CUDA_BIN", "/usr/local/cuda-13.2/bin"))
CUOBJDUMP = CUDA_BIN / "cuobjdump"

DEPTHS = (1, 2)
LEADS = (0, 4, 8, 16)
VARIANTS = ("span_expensive", "span_cheap")
REPETITIONS = tuple(range(1, 11))
TRACE_GROUPS = 64
DIRECT_GROUPS = 8192


class CaptureError(RuntimeError):
    """A required command or retained-artifact check failed."""


@dataclass(frozen=True)
class Cell:
    depth: int
    lead: int

    @property
    def direct_name(self) -> str:
        return (
            f"async_multiline_response_d{self.depth}_l{self.lead}_"
            f"g{DIRECT_GROUPS}"
        )

    def endpoint_name(self, variant: str) -> str:
        return (
            f"async_multiline_response_{variant}_d{self.depth}_"
            f"l{self.lead}_g{TRACE_GROUPS}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json_text(value))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CaptureError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise CaptureError(f"{path}: expected one JSON object")
    return value


def run(
    command: Sequence[str],
    *,
    timeout: int,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=dict(environment) if environment is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CaptureError(f"command failed before completion: {command}: {error}") from error
    if completed.returncode != 0:
        raise CaptureError(
            f"command exited {completed.returncode}: {command}\n{completed.stderr[-4000:]}"
        )
    return completed


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return value


def require_binary(expected_sha256: str | None = None) -> dict[str, Any]:
    if not BINARY.is_file() or not os.access(BINARY, os.X_OK):
        raise CaptureError(f"probe binary is missing or not executable: {BINARY}")
    observed = sha256(BINARY)
    if expected_sha256 is not None and observed != expected_sha256:
        raise CaptureError(
            f"probe binary identity mismatch: {observed} != {expected_sha256}"
        )
    return {
        "path": str(BINARY),
        "sha256": observed,
        "size_bytes": BINARY.stat().st_size,
    }


def retained_binary(root: Path) -> dict[str, Any]:
    document = read_json(root / "static/command.json")
    binary = document.get("binary")
    if not isinstance(binary, dict) or not isinstance(binary.get("sha256"), str):
        raise CaptureError(f"{root / 'static/command.json'}: missing binary identity")
    current = require_binary(str(binary["sha256"]))
    if current != binary:
        raise CaptureError("retained static binary identity differs from current binary")
    return current


def selected_cells(smoke: bool, selector: str | None) -> list[Cell]:
    if smoke:
        if selector is not None:
            raise CaptureError("--smoke and --cells are mutually exclusive")
        return [Cell(1, 16)]
    configured = [Cell(depth, lead) for depth in DEPTHS for lead in LEADS]
    if selector is None:
        return configured
    by_name = {f"d{cell.depth}l{cell.lead}": cell for cell in configured}
    values: list[Cell] = []
    for raw in selector.split(","):
        key = raw.strip()
        if key not in by_name:
            raise CaptureError(f"unknown cell {key!r}; expected e.g. d1l16")
        if by_name[key] not in values:
            values.append(by_name[key])
    if not values:
        raise CaptureError("--cells selected no cells")
    return values


def selected_repetitions(smoke: bool, selector: str | None) -> list[int]:
    if smoke:
        if selector is not None:
            raise CaptureError("--smoke and --repeats are mutually exclusive")
        return [1]
    if selector is None:
        return list(REPETITIONS)
    values: list[int] = []
    for raw in selector.split(","):
        try:
            repetition = int(raw.strip())
        except ValueError as error:
            raise CaptureError(f"invalid repetition {raw!r}") from error
        if repetition not in REPETITIONS:
            raise CaptureError("repetitions must be in 1..10")
        if repetition not in values:
            values.append(repetition)
    if not values:
        raise CaptureError("--repeats selected no repetitions")
    return values


def capture_static(root: Path, timeout: int) -> None:
    static = root / "static"
    retained = tuple(
        static / name
        for name in (
            "control_bits.json",
            "resources.txt",
            "sass.txt",
            "build.log",
            "command.json",
        )
    )
    if all(path.is_file() and path.stat().st_size > 0 for path in retained):
        retained_binary(root)
        print(f"static existing: {static}", flush=True)
        return
    if static.is_dir() and any(static.iterdir()):
        raise CaptureError(
            f"incomplete static directory must be moved aside before recapture: {static}"
        )
    static.mkdir(parents=True, exist_ok=True)
    build_command = [
        "make",
        "-B",
        "-C",
        str(HERE),
        "async_multiline_response",
    ]
    built = run(build_command, timeout=timeout, environment=environment())
    atomic_write_text(static / "build.log", built.stdout + built.stderr)
    binary = require_binary()

    for option, output in (
        ("--dump-sass", "sass.txt"),
        ("--dump-resource-usage", "resources.txt"),
    ):
        dumped = run(
            [str(CUOBJDUMP), option, str(BINARY)],
            timeout=timeout,
            environment=environment(),
        )
        atomic_write_text(static / output, dumped.stdout + dumped.stderr)

    control_command = [
        sys.executable,
        str(CONTROL_TOOL),
        "--binary",
        str(BINARY),
        "--arch",
        "sm_86",
        "--toolkit-bindir",
        str(CUDA_BIN),
        "--out",
        str(static / "control_bits.json"),
    ]
    controlled = run(control_command, timeout=timeout, environment=environment())
    atomic_write_text(static / "control.log", controlled.stdout + controlled.stderr)
    atomic_write_json(
        static / "command.json",
        {
            "schema_version": 1,
            "build": build_command,
            "control": control_command,
            "binary": binary,
        },
    )
    print(f"static: {static}", flush=True)


def expected_order(repetition: int) -> str:
    return "ab" if repetition % 2 else "ba"


def direct_command(cell: Cell, repetition: int) -> list[str]:
    return [
        str(BINARY),
        "--groups",
        str(DIRECT_GROUPS),
        "--depth",
        str(cell.depth),
        "--lead-samples",
        str(cell.lead),
        "--order",
        expected_order(repetition),
        "--markers",
        "off",
    ]


def validate_direct(
    destination: Path,
    cell: Cell,
    repetition: int,
    binary: Mapping[str, Any],
) -> None:
    path = destination / "result.json"
    payload = read_json(path)
    expected = {
        "workload": "async_multiline_response",
        "mode": "pair",
        "order": expected_order(repetition),
        "depth": cell.depth,
        "lead_clock_samples": cell.lead,
        "groups": DIRECT_GROUPS,
        "correct": True,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise CaptureError(f"{path}: direct binding mismatch: {mismatches}")
    for variant in VARIANTS:
        arm = payload.get(variant)
        if not isinstance(arm, dict) or arm.get("correct") is not True:
            raise CaptureError(f"{path}: invalid {variant} result")
    command = read_json(destination / "command.json")
    expected_command = direct_command(cell, repetition)
    if command.get("command") != expected_command or command.get("binary") != binary:
        raise CaptureError(f"{destination}: direct command/binary binding mismatch")


def capture_direct(
    root: Path, cells: Sequence[Cell], repetitions: Sequence[int], timeout: int
) -> None:
    binary = retained_binary(root)
    for cell in cells:
        for repetition in repetitions:
            destination = (
                root
                / "direct"
                / cell.direct_name
                / f"repeat-{repetition}"
            )
            result_path = destination / "result.json"
            if result_path.is_file():
                validate_direct(destination, cell, repetition, binary)
                print(f"direct existing: {cell.direct_name}/repeat-{repetition}")
                continue
            command = direct_command(cell, repetition)
            completed = run(command, timeout=timeout, environment=environment())
            destination.mkdir(parents=True, exist_ok=True)
            atomic_write_text(result_path, completed.stdout)
            atomic_write_text(destination / "stderr.txt", completed.stderr)
            atomic_write_json(
                destination / "command.json",
                {
                    "schema_version": 1,
                    "command": command,
                    "binary": binary,
                    "environment_overrides": {
                        "CUDA_VISIBLE_DEVICES": "0",
                        "LANG": "C",
                        "LC_ALL": "C",
                    },
                },
            )
            validate_direct(destination, cell, repetition, binary)
            print(f"direct: {cell.direct_name}/repeat-{repetition}", flush=True)


def validate_nvbit(
    destination: Path, cell: Cell, variant: str, expected_sha256: str
) -> None:
    complete = read_json(destination / "complete.json")
    binding = read_json(destination / "campaign_binding.json")
    expected_name = cell.endpoint_name(variant)
    if complete.get("complete") is not True or complete.get("key") != expected_name:
        raise CaptureError(f"{destination}: invalid NVBit completion record")
    if complete.get("binary_sha256") != expected_sha256:
        raise CaptureError(f"{destination}: wrong NVBit binary")
    if binding.get("cell") != expected_name:
        raise CaptureError(f"{destination}: wrong NVBit binding")
    if binding.get("output_contract", {}).get("passed") is not True:
        raise CaptureError(f"{destination}: failed NVBit output contract")


def capture_nvbit(root: Path, cells: Sequence[Cell], timeout: int) -> None:
    binary = retained_binary(root)
    for cell in cells:
        for variant in VARIANTS:
            name = cell.endpoint_name(variant)
            destination = root / "nvbit" / name
            if (destination / "complete.json").is_file():
                validate_nvbit(destination, cell, variant, str(binary["sha256"]))
                print(f"nvbit existing: {name}")
                continue
            capture_command = [
                sys.executable,
                str(NVBIT_CAPTURE),
                "--output",
                str(destination),
                "--cell",
                name,
                "--binary",
                str(BINARY),
                "--mode",
                "ncu",
                "--gpu",
                "0",
                "--timeout",
                str(timeout),
                "--target-args",
                "--groups",
                str(TRACE_GROUPS),
                "--depth",
                str(cell.depth),
                "--lead-samples",
                str(cell.lead),
                "--variant",
                variant,
                "--markers",
                "on",
            ]
            command = [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "-p",
                "Delegate=yes",
                *capture_command,
            ]
            run(command, timeout=timeout + 120, environment=environment())
            validate_nvbit(destination, cell, variant, str(binary["sha256"]))
            print(f"nvbit: {name}", flush=True)


def build_simulator(timeout: int) -> None:
    command = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(REPO_ROOT / "sim/Cargo.toml"),
        "-p",
        "sim-run",
    ]
    run(command, timeout=timeout, environment=environment())
    if not SIM_RUN.is_file() or not os.access(SIM_RUN, os.X_OK):
        raise CaptureError(f"sim-run build did not produce {SIM_RUN}")


def replay_command(root: Path, cell: Cell, variant: str) -> list[str]:
    return [
        str(SIM_RUN),
        "--trace",
        str(root / "nvbit" / cell.endpoint_name(variant)),
        "--control",
        str(root / "static/control_bits.json"),
        "--arch-card",
        str(ARCH_CARD),
        "--pchase-card",
        str(PCHASE_CARD),
        "--mode",
        "l1-l2",
    ]


def validate_replay(
    root: Path, destination: Path, cell: Cell, variant: str
) -> None:
    first = destination / "replay-1.json"
    second = destination / "replay-2.json"
    if not first.is_file() or not second.is_file():
        raise CaptureError(f"{destination}: incomplete replay twins")
    if first.read_bytes() != second.read_bytes():
        raise CaptureError(f"{destination}: replay twins differ")
    read_json(first)
    trace = root / "nvbit" / cell.endpoint_name(variant) / "dynamic_trace.pb"
    control = root / "static/control_bits.json"
    expected = {
        "schema_version": 1,
        "command": replay_command(root, cell, variant),
        "sim_run": {"path": str(SIM_RUN), "sha256": sha256(SIM_RUN)},
        "trace_sha256": sha256(trace),
        "control_sha256": sha256(control),
    }
    if read_json(destination / "command.json") != expected:
        raise CaptureError(f"{destination}: replay input/binary binding mismatch")


def capture_replay(root: Path, cells: Sequence[Cell], timeout: int) -> None:
    build_simulator(timeout)
    for cell in cells:
        for variant in VARIANTS:
            name = cell.endpoint_name(variant)
            trace = root / "nvbit" / name / "dynamic_trace.pb"
            if not trace.is_file() or trace.stat().st_size == 0:
                raise CaptureError(f"missing trace for {name}: {trace}")
            destination = root / "simulator" / name
            if (destination / "replay-1.json").is_file() and (
                destination / "replay-2.json"
            ).is_file():
                validate_replay(root, destination, cell, variant)
                print(f"replay existing: {name}")
                continue
            destination.mkdir(parents=True, exist_ok=True)
            command = replay_command(root, cell, variant)
            for repeat in (1, 2):
                completed = run(command, timeout=timeout, environment=environment())
                atomic_write_text(destination / f"replay-{repeat}.json", completed.stdout)
                atomic_write_text(destination / f"stderr-{repeat}.txt", completed.stderr)
            atomic_write_json(
                destination / "command.json",
                {
                    "schema_version": 1,
                    "command": command,
                    "sim_run": {
                        "path": str(SIM_RUN),
                        "sha256": sha256(SIM_RUN),
                    },
                    "trace_sha256": sha256(trace),
                    "control_sha256": sha256(root / "static/control_bits.json"),
                },
            )
            validate_replay(root, destination, cell, variant)
            print(f"replay: {name}", flush=True)


def validate_selected(root: Path, cells: Sequence[Cell], repetitions: Sequence[int]) -> None:
    binary = retained_binary(root)
    for relative in (
        "static/control_bits.json",
        "static/resources.txt",
        "static/sass.txt",
        "static/build.log",
    ):
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise CaptureError(f"missing static artifact: {path}")
    for cell in cells:
        for repetition in repetitions:
            validate_direct(
                root / "direct" / cell.direct_name / f"repeat-{repetition}",
                cell,
                repetition,
                binary,
            )
        for variant in VARIANTS:
            validate_nvbit(
                root / "nvbit" / cell.endpoint_name(variant),
                cell,
                variant,
                str(binary["sha256"]),
            )
            validate_replay(
                root,
                root / "simulator" / cell.endpoint_name(variant),
                cell,
                variant,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--phase",
        choices=("all", "static", "direct", "nvbit", "replay"),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cells", help="comma-separated subset such as d1l16,d2l4")
    parser.add_argument("--repeats", help="comma-separated subset in 1..10")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        if args.timeout < 1:
            raise CaptureError("--timeout must be positive")
        cells = selected_cells(args.smoke, args.cells)
        repetitions = selected_repetitions(args.smoke, args.repeats)
        root = args.root.resolve()
        if args.validate_only:
            validate_selected(root, cells, repetitions)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "root": str(root),
                        "cells": [f"d{cell.depth}l{cell.lead}" for cell in cells],
                        "repetitions": repetitions,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.phase in {"all", "static"}:
            capture_static(root, args.timeout)
        if args.phase in {"all", "direct"}:
            capture_direct(root, cells, repetitions, args.timeout)
        if args.phase in {"all", "nvbit"}:
            capture_nvbit(root, cells, args.timeout)
        if args.phase in {"all", "replay"}:
            capture_replay(root, cells, args.timeout)
        return 0
    except (CaptureError, OSError, ValueError) as error:
        print(f"multiline execution capture: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
