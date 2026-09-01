#!/usr/bin/env python3
"""Run the bounded SM86 async-copy cross-warp queue-overlap campaign."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_wait_prefix import (  # noqa: E402
    CaptureError,
    identity,
    now,
    run,
    write_json,
)

BINARY = HERE / "async_cross_warp_queue_overlap"
SOURCE = HERE / "async_cross_warp_queue_overlap.cu"
MAKEFILE = HERE / "Makefile"
ROOT = REPO_ROOT / "results/captures/sm86_async_cross_warp_queue_overlap_20260831_v1"
SIM_RUN = REPO_ROOT / "sim/target/release/sim-run"
CONTROL = ROOT / "static/control_bits.json"
ARCH_CARD = REPO_ROOT / "arch_profiles/cards/sm_86.json"
PCHASE_CARD = REPO_ROOT / "data/pchase/sm_86_static_card.json"
CONTROL_TOOL = REPO_ROOT / "tools/sass_control/emit_control_bits_sidecar.py"
NVBIT_CAPTURE = (
    REPO_ROOT / "workloads/research_probes/ldgsts_burst_release/capture_nvbit.py"
)
CUDA_BIN = Path("/usr/local/cuda-13.2/bin")
CUOBJDUMP = CUDA_BIN / "cuobjdump"
NCU = Path("/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu")
ROUTES = ("l1", "l2")
ARMS = ("queued", "drained")
NCU_METRICS = (
    "smsp__inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_aperture_device_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
)


def parse_target(
    stdout: bytes,
    *,
    mode: str,
    route: str,
    groups: int,
    active_warps: int,
    arm: str | None = None,
) -> dict[str, Any]:
    lines = [line for line in stdout.decode("utf-8", errors="strict").splitlines() if line]
    if len(lines) != 1:
        raise CaptureError(f"target emitted {len(lines)} lines, expected one")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise CaptureError(f"target stdout is not JSON: {error}") from error
    expected = {
        "workload": "async_cross_warp_queue_overlap",
        "mode": mode,
        "route": route,
        "active_warps": active_warps,
        "groups": groups,
        "copies_per_group": 8,
        "background_groups_per_epoch": 4,
        "correct": True,
        "result_store_after_intervals": True,
    }
    if arm is not None:
        expected["arm"] = arm
    for field, value in expected.items():
        if payload.get(field) != value:
            raise CaptureError(f"target {field}={payload.get(field)!r}, expected {value!r}")
    return payload


def static_phase(timeout: int) -> None:
    static = ROOT / "static"
    static.mkdir(parents=True, exist_ok=True)
    build = run(
        ["make", "-B", "-C", str(HERE), f"NVCC={CUDA_BIN / 'nvcc'}", BINARY.name],
        timeout=timeout,
    )
    (static / "build.stdout.txt").write_bytes(build.stdout)
    (static / "build.stderr.txt").write_bytes(build.stderr)
    sass = run([str(CUOBJDUMP), "--dump-sass", str(BINARY)], timeout=timeout)
    resources = run(
        [str(CUOBJDUMP), "--dump-resource-usage", str(BINARY)], timeout=timeout
    )
    (static / "sass.txt").write_bytes(sass.stdout)
    (static / "resources.txt").write_bytes(resources.stdout)
    control = run(
        [
            sys.executable,
            str(CONTROL_TOOL),
            "--binary",
            str(BINARY),
            "--arch",
            "sm_86",
            "--toolkit-bindir",
            str(CUDA_BIN),
            "--out",
            str(CONTROL),
        ],
        timeout=timeout,
    )
    (static / "control.stdout.txt").write_bytes(control.stdout)
    (static / "control.stderr.txt").write_bytes(control.stderr)
    sass_text = sass.stdout.decode("utf-8", errors="strict")
    resources_text = resources.stdout.decode("utf-8", errors="strict")
    required = (
        "queue_overlap_paired_kernel",
        "queue_overlap_endpoint_kernel",
        "LDGSTS.E.128",
        "LDGSTS.E.BYPASS.128",
        "LDGDEPBAR",
        "DEPBAR.LE SB0, 0x0",
        "BAR.SYNC.DEFER_BLOCKING",
    )
    if not all(item in sass_text for item in required):
        raise CaptureError("static SASS is missing a required queue-overlap form")
    if resources_text.count("SHARED:33112") != 4:
        raise CaptureError("queue-overlap target functions do not all use 33,112 shared bytes")
    write_json(
        static / "summary.json",
        {
            "schema_version": 1,
            "kind": "sm86_async_cross_warp_queue_static",
            "generated_at": now(),
            "binary": identity(BINARY),
            "source": identity(SOURCE),
            "makefile": identity(MAKEFILE),
            "control": identity(CONTROL),
            "sass": identity(static / "sass.txt"),
            "resources": identity(static / "resources.txt"),
            "target_function_count": 4,
            "static_shared_bytes": 33112,
            "required_forms": list(required),
        },
    )


def direct_phase(*, smoke: bool, repeats: int, timeout: int) -> None:
    groups = 64 if smoke else 8192
    stage = "smoke" if smoke else "identification"
    for repetition in range(1, repeats + 1):
        order = "qd" if repetition % 2 else "dq"
        for route in ROUTES:
            command = [
                str(BINARY),
                "--groups",
                str(groups),
                "--active-warps",
                "4",
                "--route",
                route,
                "--mode",
                "paired",
                "--order",
                order,
                "--markers",
                "off",
            ]
            started = now()
            result = run(command, timeout=timeout)
            payload = parse_target(
                result.stdout,
                mode="paired",
                route=route,
                groups=groups,
                active_warps=4,
            )
            destination = ROOT / "direct" / stage / f"repeat-{repetition}"
            write_json(destination / f"{route}.json", payload)
            write_json(
                destination / f"{route}.run.json",
                {
                    "schema_version": 1,
                    "command": command,
                    "started_at": started,
                    "finished_at": now(),
                    "binary": identity(BINARY),
                    "return_code": result.returncode,
                    "stderr": result.stderr.decode("utf-8", errors="replace"),
                },
            )


def nvbit_phase(timeout: int) -> None:
    binary_id = identity(BINARY)
    for route in ROUTES:
        for arm in ARMS:
            key = f"{route}_{arm}_w4_g64"
            destination = ROOT / "nvbit" / key
            binding = destination / "campaign_binding.json"
            if binding.is_file():
                value = json.loads(binding.read_text(encoding="utf-8"))
                if value.get("binary", {}).get("sha256") == binary_id["sha256"]:
                    continue
                raise CaptureError(f"stale NVBit capture must be moved aside: {destination}")
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
                key,
                "--binary",
                str(BINARY),
                "--mode",
                "ncu",
                "--timeout",
                str(timeout),
                "--target-args",
                "--groups",
                "64",
                "--active-warps",
                "4",
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


def replay_phase(timeout: int) -> None:
    for route in ROUTES:
        for arm in ARMS:
            key = f"{route}_{arm}_w4_g64"
            trace = ROOT / "nvbit" / key
            command = [
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
            destination = ROOT / "simulator" / key
            outputs = []
            for repetition in (1, 2):
                result = run(command, timeout=timeout)
                try:
                    parsed = json.loads(result.stdout)
                except json.JSONDecodeError as error:
                    raise CaptureError(f"sim-run output is not JSON: {error}") from error
                launches = parsed.get("launches")
                if not isinstance(launches, list) or len(launches) != 1:
                    raise CaptureError(f"{key}: replay must contain one launch")
                readiness = launches[0]["timing"]["readiness"]
                if readiness.get("schema_version") != 20:
                    raise CaptureError(f"{key}: replay readiness schema is not 20")
                if not isinstance(readiness.get("async_issue_overlap"), dict):
                    raise CaptureError(f"{key}: replay has no async issue-overlap observation")
                destination.mkdir(parents=True, exist_ok=True)
                path = destination / f"replay-{repetition}.json"
                path.write_bytes(result.stdout)
                (destination / f"stderr-{repetition}.txt").write_bytes(result.stderr)
                outputs.append(identity(path))
            if outputs[0]["sha256"] != outputs[1]["sha256"]:
                raise CaptureError(f"{key}: replay twins differ")
            write_json(
                destination / "command.json",
                {
                    "schema_version": 1,
                    "command": command,
                    "binary": identity(BINARY),
                    "sim_run": identity(SIM_RUN),
                    "trace": identity(trace / "dynamic_trace.pb"),
                    "control": identity(CONTROL),
                    "arch_card": identity(ARCH_CARD),
                    "pchase_card": identity(PCHASE_CARD),
                    "outputs": outputs,
                },
            )


def parse_ncu_csv(payload: bytes) -> dict[str, float]:
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8", errors="strict"))))
    data = [row for row in rows if row.get("Kernel Name") and row.get("ID") != "ID"]
    if len(data) != 1:
        raise CaptureError(f"NCU import has {len(data)} data rows, expected one")
    try:
        return {metric: float(data[0][metric]) for metric in NCU_METRICS}
    except (KeyError, ValueError) as error:
        raise CaptureError(f"NCU metric parse failed: {error}") from error


def ncu_phase(timeout: int) -> None:
    summaries: dict[tuple[str, str], dict[str, float]] = {}
    for route in ROUTES:
        for arm in ARMS:
            key = f"{route}_{arm}_w4_g64"
            destination = ROOT / "ncu" / key
            destination.mkdir(parents=True, exist_ok=True)
            export = destination / "profile"
            command = [
                str(NCU),
                "--devices",
                "0",
                "--target-processes",
                "application-only",
                "--profile-from-start",
                "off",
                "--launch-count",
                "1",
                "--replay-mode",
                "application",
                "--app-replay-mode",
                "strict",
                "--app-replay-match",
                "grid",
                "--cache-control",
                "none",
                "--clock-control",
                "none",
                "--kernel-name",
                "regex:queue_overlap_endpoint_kernel",
                "--metrics",
                ",".join(NCU_METRICS),
                "--export",
                str(export),
                "--force-overwrite",
                str(BINARY),
                "--groups",
                "64",
                "--active-warps",
                "4",
                "--route",
                route,
                "--mode",
                "endpoint",
                "--arm",
                arm,
                "--markers",
                "on",
            ]
            profiled = run(command, timeout=timeout)
            (destination / "profile.stdout.txt").write_bytes(profiled.stdout)
            (destination / "profile.stderr.txt").write_bytes(profiled.stderr)
            report = destination / "profile.ncu-rep"
            imported = run(
                [
                    str(NCU),
                    "--import",
                    str(report),
                    "--page",
                    "raw",
                    "--csv",
                    "--print-units",
                    "base",
                    "--print-fp",
                ],
                timeout=timeout,
            )
            (destination / "raw.csv").write_bytes(imported.stdout)
            metrics = parse_ncu_csv(imported.stdout)
            summaries[(route, arm)] = metrics
            write_json(
                destination / "summary.json",
                {
                    "schema_version": 1,
                    "kind": "sm86_async_cross_warp_queue_ncu_smoke",
                    "route": route,
                    "arm": arm,
                    "active_warps": 4,
                    "groups": 64,
                    "command": command,
                    "binary": identity(BINARY),
                    "ncu": identity(NCU),
                    "report": identity(report),
                    "raw_csv": identity(destination / "raw.csv"),
                    "metrics": metrics,
                },
            )
    exact = {
        "smsp__inst_executed_op_ldgsts.sum": 2048.0,
        "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum": 32768.0,
        "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum": 8192.0,
    }
    for key, metrics in summaries.items():
        for metric, expected in exact.items():
            if metrics[metric] != expected:
                raise CaptureError(f"{key} {metric}={metrics[metric]}, expected {expected}")
    for route in ROUTES:
        queued = summaries[(route, "queued")]
        drained = summaries[(route, "drained")]
        for metric in exact:
            if queued[metric] != drained[metric]:
                raise CaptureError(f"{route}: queued/drained exact work differs")
        for metric in NCU_METRICS[3:]:
            tolerance = max(16.0, abs(queued[metric]) / 1000.0)
            if abs(queued[metric] - drained[metric]) > tolerance:
                raise CaptureError(
                    f"{route}: {metric} paired drift exceeds {tolerance}"
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("static", "direct", "nvbit", "replay", "ncu", "smoke"),
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.repeats > 10:
        parser.error("--repeats must be in [1, 10]")
    try:
        if args.phase in ("static", "smoke"):
            static_phase(args.timeout)
        if args.phase == "direct":
            direct_phase(
                smoke=args.smoke,
                repeats=1 if args.smoke else args.repeats,
                timeout=args.timeout,
            )
        if args.phase == "smoke":
            direct_phase(smoke=True, repeats=1, timeout=args.timeout)
            nvbit_phase(args.timeout)
            replay_phase(args.timeout)
            ncu_phase(args.timeout)
        elif args.phase == "nvbit":
            nvbit_phase(args.timeout)
        elif args.phase == "replay":
            replay_phase(args.timeout)
        elif args.phase == "ncu":
            ncu_phase(args.timeout)
    except (CaptureError, OSError, ValueError, KeyError) as error:
        print(f"cross-warp queue capture: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
