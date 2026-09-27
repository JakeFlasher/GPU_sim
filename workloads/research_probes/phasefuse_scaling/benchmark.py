#!/usr/bin/env python3
"""Run the bounded PhaseFuse descriptor-scaling matrix.

This is an analyzer/contract experiment, not a native GPU simulation.  Every
case is an isolated child process.  Child output is redirected to temporary
files (rather than pipes) so a malformed or unexpectedly verbose child cannot
deadlock the wait/timeout loop.  A case is successful only when the process,
JSON contract, and declared invariants all pass; missing cases and empty
success sets are failures, never vacuous successes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = ROOT / "sim/target/release/phasefuse-scale"
SIZES = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000)
REGION_LENGTHS = (1, 2, 8, 64)
RESIDENCIES = ("complete", "unknown")
ORDERS = ("sorted", "shuffled")
MATRIX_CASES = len(SIZES) * len(REGION_LENGTHS) * len(RESIDENCIES) * len(ORDERS)
MAX_TIMEOUT_SECONDS = 300.0
MAX_CAPTURE_BYTES = 64 * 1024

CONTRACT_KEYS = {
    "schema",
    "case_key",
    "seed",
    "n_records",
    "program_count",
    "region_len",
    "residency",
    "input_order",
    "build_input_ns",
    "api_wall_ns",
    "planner_time_ns",
    "exact_execution_steps",
    "causal_execution_steps",
    "exact_final_cycle",
    "causal_final_cycle",
    "candidate_regions",
    "collapsed_regions",
    "reified_regions",
    "candidate_instructions",
    "collapsed_instructions",
    "terminal_state_matches",
    "refusal_reasons",
}
NUMERIC_CONTRACT_KEYS = {
    "seed",
    "n_records",
    "program_count",
    "region_len",
    "build_input_ns",
    "api_wall_ns",
    "planner_time_ns",
    "exact_execution_steps",
    "causal_execution_steps",
    "exact_final_cycle",
    "causal_final_cycle",
    "candidate_regions",
    "collapsed_regions",
    "reified_regions",
    "candidate_instructions",
    "collapsed_instructions",
}

# These are the source inputs whose identity must accompany a scaling capture.
# Hashes identify the evidence; they are not approval or trust gates.
SOURCE_FILES = (
    Path("sim/sm/src/bin/phasefuse_scale.rs"),
    Path("sim/run/schema/phasefuse_scale.v1.schema.json"),
    Path("workloads/research_probes/phasefuse_scaling/benchmark.py"),
    Path("workloads/research_probes/phasefuse_scaling/README.md"),
    Path("workloads/research_probes/phasefuse_scaling/test_benchmark.py"),
)


def _content_identity(path: Path) -> dict[str, object]:
    """Return a stable path/size/SHA-256 identity for one input file."""

    resolved = path.resolve()
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "bytes": size, "sha256": digest.hexdigest()}


def content_identities(binary: Path) -> dict[str, object]:
    """Collect the executable and source identities for a capture manifest."""

    return {
        "binary": _content_identity(binary),
        "sources": {str(path): _content_identity(ROOT / path) for path in SOURCE_FILES},
    }


def _read_capture(stream: Any) -> tuple[str, bool]:
    """Read bounded child output and report whether it was truncated."""

    stream.seek(0)
    data = stream.read(MAX_CAPTURE_BYTES + 1)
    truncated = len(data) > MAX_CAPTURE_BYTES
    if truncated:
        data = data[:MAX_CAPTURE_BYTES]
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += "\n...[capture truncated at 65536 bytes]"
    return text, truncated


def _failure(row: dict[str, object], message: str) -> None:
    failures = row.setdefault("failures", [])
    assert isinstance(failures, list)
    failures.append(message)


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def validate_contract(payload: object, expected: dict[str, object]) -> list[str]:
    """Validate the child JSON and cross-field invariants for one invocation."""

    if not isinstance(payload, dict):
        return ["contract is not a JSON object"]
    failures: list[str] = []
    missing = CONTRACT_KEYS.difference(payload)
    extra = set(payload).difference(CONTRACT_KEYS)
    if missing:
        failures.append(f"contract missing fields: {sorted(missing)}")
    if extra:
        failures.append(f"contract has unexpected fields: {sorted(extra)}")
    for key in NUMERIC_CONTRACT_KEYS:
        if key in payload and not _is_nonnegative_int(payload[key]):
            failures.append(f"{key} must be a non-negative integer")
    if payload.get("schema") != "phasefuse_scale_v1":
        failures.append("schema is not phasefuse_scale_v1")
    for key in ("case_key", "residency", "input_order"):
        if key in payload and not isinstance(payload[key], str):
            failures.append(f"{key} must be a string")
    if payload.get("program_count") != 4:
        failures.append("program_count must be exactly four")
    if payload.get("n_records") != expected["n"]:
        failures.append("n_records does not match the requested case")
    if payload.get("seed") != expected["seed"]:
        failures.append("seed does not match the requested case")
    if payload.get("region_len") != expected["region_len"]:
        failures.append("region_len does not match the requested case")
    if payload.get("residency") != expected["residency"]:
        failures.append("residency does not match the requested case")
    if payload.get("input_order") != expected["order"]:
        failures.append("input_order does not match the requested case")
    if payload.get("case_key") != expected["case_key"]:
        failures.append("case_key does not match the requested case")

    n_records_value = payload.get("n_records")
    if not _is_nonnegative_int(n_records_value) or not 4 <= n_records_value <= 1_000_000:
        failures.append("n_records is outside the executable bound 4..1000000")
    if payload.get("region_len") not in REGION_LENGTHS:
        failures.append("region_len is outside the executable choices")
    if payload.get("residency") not in RESIDENCIES:
        failures.append("residency is outside the executable choices")
    if payload.get("input_order") not in ORDERS:
        failures.append("input_order is outside the executable choices")

    if all(key in payload and _is_nonnegative_int(payload[key]) for key in NUMERIC_CONTRACT_KEYS):
        n_records = int(payload["n_records"])
        candidate_regions = int(payload["candidate_regions"])
        collapsed_regions = int(payload["collapsed_regions"])
        reified_regions = int(payload["reified_regions"])
        candidate_instructions = int(payload["candidate_instructions"])
        collapsed_instructions = int(payload["collapsed_instructions"])
        if payload["exact_execution_steps"] != n_records:
            failures.append("exact_execution_steps must equal n_records")
        if payload["causal_execution_steps"] > payload["exact_execution_steps"]:
            failures.append("causal_execution_steps exceeds exact_execution_steps")
        if candidate_regions != collapsed_regions + reified_regions:
            failures.append("candidate_regions must equal collapsed_regions + reified_regions")
        if candidate_instructions > n_records:
            failures.append("candidate_instructions exceeds n_records")
        if collapsed_instructions > candidate_instructions:
            failures.append("collapsed_instructions exceeds candidate_instructions")
        reasons = payload.get("refusal_reasons")
        if not isinstance(reasons, dict):
            failures.append("refusal_reasons must be an object")
        else:
            reason_total = 0
            for reason, count in reasons.items():
                if not isinstance(reason, str) or not _is_nonnegative_int(count):
                    failures.append("refusal_reasons must map strings to non-negative integers")
                else:
                    reason_total += count
            if reason_total != reified_regions:
                failures.append("refusal_reasons total must equal reified_regions")
    if payload.get("terminal_state_matches") is not True:
        failures.append("terminal_state_matches is not true")
    return failures


def run_once(
    command: list[str], timeout: float, expected: dict[str, object] | None = None
) -> dict[str, object]:
    """Run one bounded child and return a non-vacuous, self-describing row."""

    started = time.perf_counter_ns()
    row: dict[str, object] = {
        "command": [str(part) for part in command],
        "exit_code": None,
        "timed_out": False,
        "wall_ns": 0,
        "user_cpu_ns": 0,
        "sys_cpu_ns": 0,
        "peak_rss_kib": 0,
        "stdout": "",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "failures": [],
    }
    status: int | None = None
    usage: Any = None
    try:
        # Files avoid the classic wait4 + PIPE deadlock if a broken child emits
        # more than a pipe buffer before the parent reaps it.
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    close_fds=True,
                )
            except (OSError, ValueError) as error:
                _failure(row, f"child launch failed: {error}")
            else:
                deadline = time.monotonic() + timeout
                while True:
                    waited, candidate_status, candidate_usage = os.wait4(
                        process.pid, os.WNOHANG
                    )
                    if waited == process.pid:
                        status, usage = candidate_status, candidate_usage
                        break
                    if time.monotonic() >= deadline:
                        row["timed_out"] = True
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                        _, status, usage = os.wait4(process.pid, 0)
                        break
                    time.sleep(0.001)
            row["stdout"], row["stdout_truncated"] = _read_capture(stdout_file)
            row["stderr"], row["stderr_truncated"] = _read_capture(stderr_file)
    except (OSError, ValueError) as error:
        _failure(row, f"child wait/capture failed: {error}")

    row["wall_ns"] = time.perf_counter_ns() - started
    if usage is not None:
        row["user_cpu_ns"] = int(usage.ru_utime * 1e9)
        row["sys_cpu_ns"] = int(usage.ru_stime * 1e9)
        row["peak_rss_kib"] = int(usage.ru_maxrss)
    if status is not None:
        row["exit_code"] = os.waitstatus_to_exitcode(status)
    if row["timed_out"]:
        _failure(row, f"child exceeded timeout of {timeout:g}s")
    elif row["exit_code"] != 0:
        _failure(row, f"child exit code was {row['exit_code']}")
    elif row["stdout_truncated"]:
        _failure(row, "child stdout exceeded the bounded capture size")
    else:
        try:
            payload = json.loads(str(row["stdout"]))
        except json.JSONDecodeError as error:
            _failure(row, f"child stdout is not exactly one JSON object: {error}")
        else:
            row["contract"] = payload
            if expected is not None:
                for failure in validate_contract(payload, expected):
                    _failure(row, failure)
    row["ok"] = not row["failures"]
    return row


def _case(n: int, region_len: int, residency: str, order: str, seed: int) -> dict[str, object]:
    return {
        "case_key": f"n{n}-r{region_len}-{residency}-{order}-seed{seed}",
        "n": n,
        "region_len": region_len,
        "residency": residency,
        "order": order,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=Path("/tmp/phasefuse_scale.jsonl"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be in 1..10")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= MAX_TIMEOUT_SECONDS:
        parser.error(f"--timeout must be finite and in (0,{MAX_TIMEOUT_SECONDS:g}]")
    if not 0 <= args.seed <= (1 << 64) - 1:
        parser.error("--seed must be in 0..2^64-1")
    binary = args.binary if args.binary.is_absolute() else ROOT / args.binary
    if not binary.is_file():
        parser.error(f"--binary is not a regular file: {binary}")
    if not os.access(binary, os.X_OK):
        parser.error(f"--binary is not executable: {binary}")
    output_resolved = args.output.resolve()
    protected_paths = {binary.resolve(), *(ROOT / path for path in SOURCE_FILES)}
    if output_resolved in {path.resolve() for path in protected_paths}:
        parser.error("--output must not overwrite the executable or scaling source inputs")
    try:
        identities = content_identities(binary)
    except OSError as error:
        parser.error(f"cannot identify binary/source inputs: {error}")
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("w", encoding="utf-8")
    except OSError as error:
        parser.error(f"cannot open --output: {error}")

    rows_by_case: dict[str, list[dict[str, object]]] = {}
    row_count = 0
    with stream:
        for n in SIZES:
            for region_len in REGION_LENGTHS:
                for residency in RESIDENCIES:
                    for order in ORDERS:
                        expected = _case(n, region_len, residency, order, args.seed)
                        for repeat in range(args.repeats):
                            command = [
                                str(binary),
                                "--n",
                                str(n),
                                "--region-len",
                                str(region_len),
                                "--residency",
                                residency,
                                "--order",
                                order,
                                "--seed",
                                str(args.seed),
                            ]
                            result = run_once(command, args.timeout, expected)
                            row = {
                                "case_key": expected["case_key"],
                                "n": n,
                                "region_len": region_len,
                                "residency": residency,
                                "order": order,
                                "seed": args.seed,
                                "repeat": repeat,
                                "timeout_s": args.timeout,
                                **result,
                            }
                            rows_by_case.setdefault(str(expected["case_key"]), []).append(row)
                            row_count += 1
                            stream.write(json.dumps(row, sort_keys=True) + "\n")
                            stream.flush()

    aggregates: list[dict[str, object]] = []
    for key, group in sorted(rows_by_case.items()):
        good = [row for row in group if row.get("ok") is True]
        case_ok = len(group) == args.repeats and len(good) == args.repeats
        aggregates.append(
            {
                "case_key": key,
                "repeats": len(group),
                "successful_repeats": len(good),
                "ok": case_ok,
                "wall_ns_median": statistics.median(row["wall_ns"] for row in good)
                if good
                else None,
                "peak_rss_kib_max": max((row["peak_rss_kib"] for row in good), default=None),
                "terminal_state_matches": bool(good)
                and all(
                    isinstance(row.get("contract"), dict)
                    and row["contract"].get("terminal_state_matches") is True
                    for row in good
                ),
                "failures": [
                    {"repeat": row["repeat"], "failures": row.get("failures", [])}
                    for row in group
                    if row.get("ok") is not True
                ],
            }
        )
    failed_cases = [aggregate for aggregate in aggregates if aggregate["ok"] is not True]
    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "schema": "phasefuse_scale_results_v1",
        "identity": identities,
        "matrix": {
            "expected_cases": MATRIX_CASES,
            "actual_cases": len(rows_by_case),
            "repeats": args.repeats,
            "rows": row_count,
            "sizes": SIZES,
            "region_lengths": REGION_LENGTHS,
            "residencies": RESIDENCIES,
            "orders": ORDERS,
        },
        "ok": len(rows_by_case) == MATRIX_CASES and not failed_cases,
        "failed_cases": failed_cases,
        "aggregates": aggregates,
    }
    try:
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as error:
        print(json.dumps({"error": f"cannot write summary: {error}"}))
        return 2
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "rows": row_count,
                "cases": len(rows_by_case),
                "failed_cases": len(failed_cases),
                "output": str(args.output),
                "summary": str(summary_path),
            }
        )
    )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
