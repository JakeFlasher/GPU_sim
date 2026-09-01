#!/usr/bin/env python3
"""Run the bounded SM86 asynchronous-copy wait-prefix campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BINARY = HERE / "async_wait_prefix_order"
MAKEFILE = HERE / "Makefile"
ROOT = REPO_ROOT / "results/captures/sm86_async_wait_prefix_order_20260831_v1"
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
VARIANTS = ("l2_old", "l1_old")
NCU_METRICS = (
    "smsp__inst_executed_op_ldgsts.sum",
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum",
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
    "lts__t_sectors_srcunit_tex_aperture_device_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
)


class CaptureError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or stat.st_size <= 0:
        raise CaptureError(f"missing or empty artifact: {resolved}")
    return {"path": str(resolved), "sha256": sha256(resolved), "size_bytes": stat.st_size}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    command: Sequence[str],
    *,
    timeout: int,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    merged = os.environ.copy()
    merged.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if environment:
        merged.update(environment)
    try:
        result = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CaptureError(f"command timed out after {timeout}s: {command}") from error
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise CaptureError(f"command exited {result.returncode}: {command}\n{stderr}")
    return result


def parse_target(stdout: bytes, *, mode: str, variant: str, epochs: int) -> dict[str, Any]:
    lines = [line for line in stdout.decode("utf-8", errors="strict").splitlines() if line]
    if len(lines) != 1:
        raise CaptureError(f"target emitted {len(lines)} nonempty stdout lines, expected one")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise CaptureError(f"target stdout is not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CaptureError("target JSON is not an object")
    expected = {
        "workload": "async_wait_prefix_order",
        "mode": mode,
        "variant": variant,
        "epochs": epochs,
        "correct": True,
        "sum_consistent": True,
        "result_store_after_intervals": True,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise CaptureError(f"target field {field}={payload.get(field)!r}, expected {value!r}")
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
    text = sass.stdout.decode("utf-8", errors="strict")
    summary = {
        "schema_version": 1,
        "kind": "sm86_async_wait_prefix_static",
        "generated_at": now(),
        "binary": identity(BINARY),
        "source": identity(HERE / "async_wait_prefix_order.cu"),
        "makefile": identity(MAKEFILE),
        "control": identity(CONTROL),
        "sass": identity(static / "sass.txt"),
        "resources": identity(static / "resources.txt"),
        "global_counts": {
            "ldgsts": text.count("LDGSTS.E"),
            "commits": text.count("LDGDEPBAR"),
            "wait_group_1": text.count("DEPBAR.LE SB0, 0x1"),
            "wait_group_2": text.count("DEPBAR.LE SB0, 0x2"),
            "wait_group_0": text.count("DEPBAR.LE SB0, 0x0"),
        },
        "required_function_names": [
            "wait_prefix_identification_kernel",
            "wait_prefix_holdout_kernel",
        ],
    }
    if not all(name in text for name in summary["required_function_names"]):
        raise CaptureError("dedicated wait-prefix functions missing from SASS")
    if summary["global_counts"] != {
        "ldgsts": 10,
        "commits": 10,
        "wait_group_1": 1,
        "wait_group_2": 1,
        "wait_group_0": 2,
    }:
        raise CaptureError(f"unexpected static async-control counts: {summary['global_counts']}")
    write_json(static / "summary.json", summary)


def direct_phase(*, smoke: bool, repeats: int, timeout: int) -> None:
    epochs = 64 if smoke else 8192
    mode = "identification"
    stage = "smoke" if smoke else "identification"
    for repetition in range(1, repeats + 1):
        order = VARIANTS if repetition % 2 else tuple(reversed(VARIANTS))
        for variant in order:
            command = [
                str(BINARY),
                "--epochs",
                str(epochs),
                "--mode",
                mode,
                "--variant",
                variant,
                "--markers",
                "off",
            ]
            started = now()
            result = run(command, timeout=timeout)
            payload = parse_target(result.stdout, mode=mode, variant=variant, epochs=epochs)
            destination = ROOT / "direct" / stage / f"repeat-{repetition}"
            write_json(destination / f"{variant}.json", payload)
            write_json(
                destination / f"{variant}.run.json",
                {
                    "schema_version": 1,
                    "command": command,
                    "started_at": started,
                    "finished_at": now(),
                    "return_code": result.returncode,
                    "binary": identity(BINARY),
                    "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                    "stderr": result.stderr.decode("utf-8", errors="replace"),
                },
            )


def nvbit_phase(timeout: int) -> None:
    binary_id = identity(BINARY)
    for variant in VARIANTS:
        destination = ROOT / "nvbit" / f"identification_{variant}_g64"
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
            f"identification_{variant}_g64",
            "--binary",
            str(BINARY),
            "--mode",
            "ncu",
            "--timeout",
            str(timeout),
            "--target-args",
            "--epochs",
            "64",
            "--mode",
            "identification",
            "--variant",
            variant,
            "--markers",
            "on",
        ]
        run(command, timeout=timeout + 30)


def replay_phase(timeout: int) -> None:
    for variant in VARIANTS:
        trace = ROOT / "nvbit" / f"identification_{variant}_g64"
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
        destination = ROOT / "simulator" / f"identification_{variant}_g64"
        outputs = []
        for repetition in (1, 2):
            result = run(command, timeout=timeout)
            try:
                parsed = json.loads(result.stdout)
            except json.JSONDecodeError as error:
                raise CaptureError(f"sim-run output is not JSON: {error}") from error
            readiness = parsed["launches"][0]["timing"]["readiness"]
            if readiness.get("schema_version") != 19:
                raise CaptureError(f"replay readiness schema is not 19: {variant}")
            if not isinstance(readiness.get("async_wait_prefix"), dict):
                raise CaptureError(f"replay has no async wait-prefix observation: {variant}")
            destination.mkdir(parents=True, exist_ok=True)
            replay = destination / f"replay-{repetition}.json"
            replay.write_bytes(result.stdout)
            (destination / f"stderr-{repetition}.txt").write_bytes(result.stderr)
            outputs.append(identity(replay))
        if outputs[0]["sha256"] != outputs[1]["sha256"]:
            raise CaptureError(f"non-deterministic replay twins: {variant}")
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
    summaries: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        destination = ROOT / "ncu" / f"identification_{variant}_g64"
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
            "regex:wait_prefix_identification_kernel",
            "--metrics",
            ",".join(NCU_METRICS),
            "--export",
            str(export),
            "--force-overwrite",
            str(BINARY),
            "--epochs",
            "64",
            "--mode",
            "identification",
            "--variant",
            variant,
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
        summaries[variant] = metrics
        write_json(
            destination / "summary.json",
            {
                "schema_version": 1,
                "kind": "sm86_async_wait_prefix_ncu_smoke",
                "variant": variant,
                "epochs": 64,
                "command": command,
                "binary": identity(BINARY),
                "ncu": identity(NCU),
                "report": identity(report),
                "raw_csv": identity(destination / "raw.csv"),
                "metrics": metrics,
            },
        )
    exact = {
        "smsp__inst_executed_op_ldgsts.sum": 128.0,
        "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum": 4096.0,
        "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum": 1024.0,
    }
    for variant, metrics in summaries.items():
        for metric, expected in exact.items():
            if metrics[metric] != expected:
                raise CaptureError(
                    f"{variant} {metric}={metrics[metric]}, expected {expected}"
                )
    if summaries["l2_old"] != summaries["l1_old"]:
        raise CaptureError("paired NCU smoke metrics differ")


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
            direct_phase(smoke=args.smoke, repeats=1 if args.smoke else args.repeats, timeout=args.timeout)
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
        print(f"wait-prefix capture: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
