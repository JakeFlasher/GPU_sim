#!/usr/bin/env python3
"""Measure absolute end-to-end sim-run cost on retained full-coverage traces."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RESULT_ROOT = REPO_ROOT / "results/captures/simulator_event_throughput_20260819_v1"
ARCH_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"
BINARY = REPO_ROOT / "sim/target/release/sim-run"


@dataclass(frozen=True)
class Case:
    key: str
    trace: Path
    control: Path
    extra: tuple[str, ...] = ()
    expected_readiness_schema: int | None = None
    expected_feedback_candidate_id: int | None = None


def retained(relative: str) -> Path:
    return REPO_ROOT / "results/captures" / relative


LAUNCH = retained("sm86_launch_cycle_semantics_20260818_v1")
RETIREMENT = retained("sm86_memory_retirement_service_20260818_v1")
ARRIVAL = retained("sm86_l1tex_arrival_replay_service_20260818_v1")
ADMISSION = retained("sm86_l1tex_admission_backpressure_20260818_v1")


CASES = (
    Case(
        "launch_chain_control_c8",
        LAUNCH / "nvbit/b84_control_i256_c8",
        LAUNCH / "sass/launch_cycle_semantics_trace.control_bits.json",
    ),
    Case(
        "retirement_load_stream_d64",
        RETIREMENT / "nvbit/b84_load_none_line_stream_d64_l32",
        RETIREMENT / "sass/memory_retirement_service_trace.control_bits.json",
    ),
    Case(
        "retirement_store_reuse_d64",
        RETIREMENT / "nvbit/b84_store_none_line_reuse_d64_l32",
        RETIREMENT / "sass/memory_retirement_service_trace.control_bits.json",
    ),
    Case(
        "arrival_store_w8",
        ARRIVAL / "nvbit/b84_w8_store_line_d8_l32_g0",
        ARRIVAL / "sass/l1tex_arrival_replay_service_trace.control_bits.json",
    ),
    Case(
        "admission_control_d128",
        ADMISSION / "nvbit/b84_w8_control_line_unique_d128_g0_all",
        ADMISSION / "sass/l1tex_admission_backpressure_trace.control_bits.json",
    ),
    Case(
        "admission_store_d128",
        ADMISSION / "nvbit/b84_w8_store_line_unique_d128_g0_all",
        ADMISSION / "sass/l1tex_admission_backpressure_trace.control_bits.json",
    ),
    Case(
        "admission_red_d128",
        ADMISSION / "nvbit/b84_w8_red_line_unique_d128_g0_all",
        ADMISSION / "sass/l1tex_admission_backpressure_trace.control_bits.json",
    ),
    Case(
        "admission_atom_d128",
        ADMISSION / "nvbit/b84_w8_atom_line_unique_d128_g0_all",
        ADMISSION / "sass/l1tex_admission_backpressure_trace.control_bits.json",
    ),
    Case(
        "admission_red_gap64",
        ADMISSION / "nvbit/b84_w8_red_line_unique_d32_g64_all",
        ADMISSION / "sass/l1tex_admission_backpressure_trace.control_bits.json",
    ),
    Case(
        "admission_red_d128_schema10",
        ADMISSION / "nvbit/b84_w8_red_line_unique_d128_g0_all",
        ADMISSION / "sass/l1tex_admission_backpressure_trace.control_bits.json",
        (
            "--memory-admission-feedback",
            str(HERE.parent / "l1tex_admission_backpressure/memory_admission_feedback_red_12.json"),
        ),
        10,
        12,
    ),
    Case(
        "admission_red_d128_schema11",
        ADMISSION / "nvbit/b84_w8_red_line_unique_d128_g0_all",
        ADMISSION / "sass/l1tex_admission_backpressure_trace.control_bits.json",
        (
            "--memory-admission-feedback",
            str(HERE.parent / "l1tex_admission_backpressure/memory_write_completion_feedback_red_202.json"),
        ),
        11,
        202,
    ),
    Case(
        "admission_red_d128_schema13",
        ADMISSION / "nvbit/b84_w8_red_line_unique_d128_g0_all",
        ADMISSION / "sass/l1tex_admission_backpressure_trace.control_bits.json",
        (
            "--memory-admission-feedback",
            str(ADMISSION / "metadata/red_rmw_completion_singletons/candidate-214.json"),
        ),
        13,
        214,
    ),
)


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def command(case: Case, cpu: int, binary: Path = BINARY) -> list[str]:
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required for pinned-CPU measurements")
    return [
        taskset,
        "-c",
        str(cpu),
        str(binary),
        "--trace",
        str(case.trace),
        "--control",
        str(case.control),
        "--arch-card",
        str(ARCH_CARD),
        "--pchase-card",
        str(PCHASE_CARD),
        "--mode",
        "l1-l2",
        *case.extra,
    ]


def simulator_environment(
    *,
    disable_batching: bool = False,
    performance_counters: bool = False,
    retain_raw_trace_payload: bool = False,
    stream_trace_payload: bool = False,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("SIM_RUN_DISABLE_L2_ACCESS_BATCHING", None)
    environment.pop("SIM_RUN_PERFORMANCE_COUNTERS", None)
    environment.pop("SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD", None)
    environment.pop("SIM_RUN_STREAM_TRACE_PAYLOAD", None)
    if disable_batching:
        environment["SIM_RUN_DISABLE_L2_ACCESS_BATCHING"] = "1"
    if performance_counters:
        environment["SIM_RUN_PERFORMANCE_COUNTERS"] = "1"
    if retain_raw_trace_payload:
        environment["SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD"] = "1"
    if stream_trace_payload:
        environment["SIM_RUN_STREAM_TRACE_PAYLOAD"] = "1"
    return environment


def run_once(
    case: Case,
    cpu: int,
    timeout_seconds: float,
    *,
    disable_batching: bool = False,
    retain_raw_trace_payload: bool = False,
    stream_trace_payload: bool = False,
    binary: Path = BINARY,
) -> tuple[dict, bytes]:
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        started = time.perf_counter_ns()
        process = subprocess.Popen(
            command(case, cpu, binary),
            cwd=REPO_ROOT,
            env=simulator_environment(
                disable_batching=disable_batching,
                retain_raw_trace_payload=retain_raw_trace_payload,
                stream_trace_payload=stream_trace_payload,
            ),
            stdout=stdout,
            stderr=stderr,
        )
        deadline = time.monotonic() + timeout_seconds
        usage = None
        status = None
        while usage is None:
            waited, candidate_status, candidate_usage = os.wait4(
                process.pid, os.WNOHANG
            )
            if waited == process.pid:
                status = candidate_status
                usage = candidate_usage
                break
            if time.monotonic() >= deadline:
                process.kill()
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
                raise RuntimeError(f"{case.key}: exceeded {timeout_seconds:g}s timeout")
            time.sleep(0.002)
        elapsed_ns = time.perf_counter_ns() - started
        process.returncode = os.waitstatus_to_exitcode(status)
        stdout.seek(0)
        output = stdout.read()
        stderr.seek(0)
        error_output = stderr.read().decode("utf-8", errors="replace")
    if process.returncode != 0:
        raise RuntimeError(
            f"{case.key}: sim-run exited {process.returncode}: {error_output[-1000:]}"
        )
    document = json.loads(output)
    validate_run_document(case, document)
    launches = document.get("launches")
    assert isinstance(launches, list) and launches
    records = sum(int(launch["counters"]["records_seen"]) for launch in launches)
    instructions = sum(
        int(launch["counters"]["instructions_issued"]) for launch in launches
    )
    elapsed_seconds = elapsed_ns / 1_000_000_000
    return (
        {
            "elapsed_seconds": elapsed_seconds,
            "user_cpu_seconds": float(usage.ru_utime),
            "system_cpu_seconds": float(usage.ru_stime),
            "peak_rss_kib": int(usage.ru_maxrss),
            "minor_page_faults": int(usage.ru_minflt),
            "major_page_faults": int(usage.ru_majflt),
            "voluntary_context_switches": int(usage.ru_nvcsw),
            "involuntary_context_switches": int(usage.ru_nivcsw),
            "records_seen": records,
            "instructions_issued": instructions,
            "records_per_second": records / elapsed_seconds,
            "instructions_per_second": instructions / elapsed_seconds,
            "launches": len(launches),
            "result_bytes": len(output),
            "simulator_version": document.get("simulator_version"),
        },
        output,
    )


def validate_run_document(case: Case, document: object) -> None:
    if not isinstance(document, dict):
        raise RuntimeError(f"{case.key}: sim-run root is not an object")
    launches = document.get("launches")
    if not isinstance(launches, list) or not launches:
        raise RuntimeError(f"{case.key}: sim-run emitted no launches")
    for index, launch in enumerate(launches):
        if not isinstance(launch, dict):
            raise RuntimeError(f"{case.key}: launch {index} is not an object")
        if case.expected_readiness_schema is None:
            continue
        timing = launch.get("timing")
        readiness = timing.get("readiness") if isinstance(timing, dict) else None
        if not isinstance(readiness, dict):
            raise RuntimeError(f"{case.key}: launch {index} has no readiness report")
        actual_schema = readiness.get("schema_version")
        if actual_schema != case.expected_readiness_schema:
            raise RuntimeError(
                f"{case.key}: launch {index} readiness schema {actual_schema!r} "
                f"does not match expected {case.expected_readiness_schema}"
            )
        feedback = readiness.get("memory_admission_feedback")
        candidate = feedback.get("candidate") if isinstance(feedback, dict) else None
        actual_candidate_id = (
            candidate.get("candidate_id") if isinstance(candidate, dict) else None
        )
        if actual_candidate_id != case.expected_feedback_candidate_id:
            raise RuntimeError(
                f"{case.key}: launch {index} feedback candidate "
                f"{actual_candidate_id!r} does not match expected "
                f"{case.expected_feedback_candidate_id}"
            )


def require_performance_feature(
    binary: Path, cpu: int, timeout_seconds: float
) -> dict[str, object]:
    binary = binary.resolve()
    completed = subprocess.run(
        command(CASES[0], cpu, binary),
        cwd=REPO_ROOT,
        env=simulator_environment(performance_counters=True),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "performance-feature probe failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    stderr = completed.stderr.decode("utf-8", errors="replace")
    if "SIM_PHASE_PERF setup_ns=" not in stderr or "SIM_EVENT_PERF " not in stderr:
        raise RuntimeError(
            f"{binary}: performance-counters feature is absent or inactive"
        )
    document = json.loads(completed.stdout)
    stat = binary.stat()
    return {
        "path": str(binary),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "simulator_version": document.get("simulator_version"),
        "performance_feature_probe": True,
    }


def verify_binary_identity(identity: dict[str, object]) -> None:
    binary = Path(str(identity["path"]))
    stat = binary.stat()
    actual = {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }
    expected = {key: identity[key] for key in actual}
    if actual != expected:
        raise RuntimeError(
            f"simulator binary changed during measurement: expected {expected}, got {actual}"
        )


def capture_binary_identity(binary: Path, simulator_version: object) -> dict[str, object]:
    binary = binary.resolve()
    stat = binary.stat()
    return {
        "path": str(binary),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "simulator_version": simulator_version,
    }


def paired_percent_changes(
    baseline: Sequence[dict], changed: Sequence[dict], field: str
) -> list[float]:
    baseline_by_repeat = {int(run["repeat"]): float(run[field]) for run in baseline}
    changed_by_repeat = {int(run["repeat"]): float(run[field]) for run in changed}
    if len(baseline_by_repeat) != len(baseline) or len(changed_by_repeat) != len(changed):
        raise RuntimeError(f"{field}: duplicate repeat in matched comparison")
    if set(baseline_by_repeat) != set(changed_by_repeat):
        raise RuntimeError(f"{field}: comparison repeat sets differ")
    result = []
    for repeat in sorted(baseline_by_repeat):
        denominator = baseline_by_repeat[repeat]
        if denominator <= 0:
            raise RuntimeError(f"{field}: repeat {repeat} baseline is not positive")
        result.append(100 * (changed_by_repeat[repeat] / denominator - 1))
    return result


def validate_comparison_evidence(
    comparison_root: Path,
    document: object,
    conditions: tuple[str, str],
) -> bool:
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise RuntimeError(f"{comparison_root}: comparison schema is not 1")
    method = document.get("method")
    if (
        not isinstance(method, dict)
        or method.get("cross_condition_byte_equality_required") is not True
        or not isinstance(method.get("measured_repeats"), int)
        or method["measured_repeats"] < 1
    ):
        raise RuntimeError(f"{comparison_root}: comparison method is malformed")
    if method.get("feature_binary") == "sim-run/performance-counters":
        binary = method.get("binary")
        if not isinstance(binary, dict) or binary.get("performance_feature_probe") is not True:
            raise RuntimeError(f"{comparison_root}: performance feature was not proven")
    elif method.get("binary_mode") == "matched_ordinary_binaries":
        for label in ("baseline_binary", "candidate_binary"):
            identity = method.get(label)
            if not isinstance(identity, dict) or not all(
                key in identity
                for key in (
                    "path",
                    "size_bytes",
                    "mtime_ns",
                    "device",
                    "inode",
                    "simulator_version",
                )
            ):
                raise RuntimeError(
                    f"{comparison_root}: {label} identity is malformed"
                )
    else:
        raise RuntimeError(f"{comparison_root}: comparison binary mode is unknown")
    expected = {case.key for case in CASES}
    cases = document.get("cases")
    summaries = document.get("summaries")
    runs = document.get("runs")
    for label, value in (("cases", cases), ("summaries", summaries), ("runs", runs)):
        if not isinstance(value, dict) or set(value) != expected:
            raise RuntimeError(f"{comparison_root}: {label} case set differs")
    by_key = {case.key: case for case in CASES}
    root = comparison_root.resolve()
    for key in sorted(expected):
        metadata = cases[key]
        if not isinstance(metadata, dict):
            raise RuntimeError(f"{comparison_root}/{key}: case metadata is malformed")
        expected_case = by_key[key]
        expected_metadata = {
            "trace": str(expected_case.trace.relative_to(REPO_ROOT)),
            "control": str(expected_case.control.relative_to(REPO_ROOT)),
            "extra_arguments": list(expected_case.extra),
            "trace_bytes": tree_bytes(expected_case.trace),
            "expected_readiness_schema": expected_case.expected_readiness_schema,
            "expected_feedback_candidate_id": expected_case.expected_feedback_candidate_id,
        }
        if any(metadata.get(name) != value for name, value in expected_metadata.items()):
            raise RuntimeError(f"{comparison_root}/{key}: case binding differs")
        relative = metadata.get("canonical_output")
        declared_bytes = metadata.get("canonical_result_bytes")
        if not isinstance(relative, str) or not isinstance(declared_bytes, int):
            raise RuntimeError(f"{comparison_root}/{key}: canonical output is undeclared")
        canonical_path = (root / relative).resolve()
        try:
            canonical_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"{comparison_root}/{key}: canonical output escapes comparison root"
            ) from error
        payload = canonical_path.read_bytes()
        if len(payload) != declared_bytes:
            raise RuntimeError(f"{comparison_root}/{key}: canonical byte length differs")
        canonical_document = json.loads(payload)
        validate_run_document(expected_case, canonical_document)
        condition_runs = runs[key]
        if not isinstance(condition_runs, dict) or set(condition_runs) != set(conditions):
            raise RuntimeError(f"{comparison_root}/{key}: comparison conditions differ")
        expected_repeats = set(range(1, method["measured_repeats"] + 1))
        for condition in conditions:
            material = condition_runs[condition]
            if not isinstance(material, list) or len(material) != len(expected_repeats):
                raise RuntimeError(
                    f"{comparison_root}/{key}/{condition}: repeat count differs"
                )
            repeats = {run.get("repeat") for run in material if isinstance(run, dict)}
            if repeats != expected_repeats or any(
                run.get("result_bytes") != declared_bytes
                for run in material
                if isinstance(run, dict)
            ):
                raise RuntimeError(
                    f"{comparison_root}/{key}/{condition}: repeat evidence differs"
                )
    return True


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def summarize(case: Case, runs: Sequence[dict]) -> dict:
    stable_fields = (
        "records_seen",
        "instructions_issued",
        "launches",
        "result_bytes",
        "simulator_version",
    )
    for field in stable_fields:
        if len({run[field] for run in runs}) != 1:
            raise RuntimeError(f"{case.key}: {field} changed across repetitions")
    return {
        "trace_bytes": tree_bytes(case.trace),
        **{field: runs[0][field] for field in stable_fields},
        "elapsed_seconds_median": median([run["elapsed_seconds"] for run in runs]),
        "elapsed_seconds_min": min(run["elapsed_seconds"] for run in runs),
        "elapsed_seconds_max": max(run["elapsed_seconds"] for run in runs),
        "instructions_per_second_median": median(
            [run["instructions_per_second"] for run in runs]
        ),
        "records_per_second_median": median(
            [run["records_per_second"] for run in runs]
        ),
        "peak_rss_kib_median": median([run["peak_rss_kib"] for run in runs]),
        "peak_rss_kib_max": max(run["peak_rss_kib"] for run in runs),
        "user_cpu_seconds_median": median(
            [run["user_cpu_seconds"] for run in runs]
        ),
        "system_cpu_seconds_median": median(
            [run["system_cpu_seconds"] for run in runs]
        ),
    }


def selected_cases(selector: str | None) -> list[Case]:
    if selector is None:
        return list(CASES)
    requested = [item for item in selector.split(",") if item]
    by_key = {case.key: case for case in CASES}
    unknown = sorted(set(requested) - set(by_key))
    if unknown:
        raise RuntimeError(f"unknown case(s): {', '.join(unknown)}")
    return [by_key[key] for key in requested]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    result.add_argument("--binary", type=Path, default=BINARY)
    result.add_argument("--cpu", type=int, default=8)
    result.add_argument("--repeats", type=int, default=7)
    result.add_argument("--timeout-seconds", type=float, default=120.0)
    result.add_argument("--cases")
    result.add_argument("--disable-batching", action="store_true")
    result.add_argument("--keys", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.keys:
        print("\n".join(case.key for case in CASES))
        return 0
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.cpu not in os.sched_getaffinity(0):
        raise SystemExit(f"CPU {args.cpu} is outside the current affinity mask")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"release simulator is missing: {binary}")
    cases = selected_cases(args.cases)
    for case in cases:
        for path in (case.trace, case.control, ARCH_CARD, PCHASE_CARD):
            if not path.exists():
                raise SystemExit(f"{case.key}: missing input {path}")

    canonical: dict[str, bytes] = {}
    print("warming " + ", ".join(case.key for case in cases), flush=True)
    for case in cases:
        _, output = run_once(
            case,
            args.cpu,
            args.timeout_seconds,
            disable_batching=args.disable_batching,
            binary=binary,
        )
        canonical[case.key] = output

    measured: dict[str, list[dict]] = {case.key: [] for case in cases}
    order_log: list[list[str]] = []
    rng = random.Random(13_086)
    for repeat in range(1, args.repeats + 1):
        order = list(cases)
        rng.shuffle(order)
        order_log.append([case.key for case in order])
        for case in order:
            run, output = run_once(
                case,
                args.cpu,
                args.timeout_seconds,
                disable_batching=args.disable_batching,
                binary=binary,
            )
            if output != canonical[case.key]:
                raise RuntimeError(f"{case.key}: normal result bytes changed across runs")
            run["repeat"] = repeat
            measured[case.key].append(run)
            print(
                f"repeat {repeat}/{args.repeats} {case.key}: "
                f"{run['elapsed_seconds']:.3f}s, "
                f"{run['instructions_per_second'] / 1_000_000:.3f} M inst/s, "
                f"{run['peak_rss_kib'] / 1024:.1f} MiB",
                flush=True,
            )

    document = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": args.cpu,
            "affinity": sorted(os.sched_getaffinity(0)),
        },
        "binary": {
            "path": str(binary),
            "size_bytes": binary.stat().st_size,
            "mtime_ns": binary.stat().st_mtime_ns,
            "device": binary.stat().st_dev,
            "inode": binary.stat().st_ino,
        },
        "method": {
            "warmups_per_case": 1,
            "measured_repeats": args.repeats,
            "order_seed": 13_086,
            "round_orders": order_log,
            "mode": "l1-l2",
            "stdout_byte_stability_required": True,
            "same_cycle_batching_disabled": args.disable_batching,
        },
        "cases": {
            case.key: {
                "trace": str(case.trace.relative_to(REPO_ROOT)),
                "control": str(case.control.relative_to(REPO_ROOT)),
                "extra_arguments": list(case.extra),
                "summary": summarize(case, measured[case.key]),
                "runs": measured[case.key],
            }
            for case in cases
        },
    }
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "baseline.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["case", *next(iter(document["cases"].values()))["summary"].keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, item in document["cases"].items():
            writer.writerow({"case": key, **item["summary"]})
    print(output / "baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
