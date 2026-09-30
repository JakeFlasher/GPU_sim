#!/usr/bin/env python3
"""Collect native CUPTI PC states without NCU labels, manifests, or tests.

Uses retained portable_lcd binaries and timing-shape arguments. NVIDIA's sample
prepends a context ID to file names, so each process runs in a fresh output
directory with a basename-only file name. Both producer and decoder use the
same explicit CUPTI library directory. Instrumented CUDA-event times are not
performance predictions; acquisition wall time is retained separately.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from sm86_cupti_pc_sampling_summary import run as summarize

REPO = Path(__file__).resolve().parents[1]
SPECS = REPO / "workloads/research_probes/sm86_ncu_demo_benchmark"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute(command, directory, label, env, timeout):
    started = time.perf_counter()
    result = {"command": list(map(str, command)), "timed_out": False}
    with (directory / f"{label}.stdout").open("w") as out, (directory / f"{label}.stderr").open("w") as err:
        proc = subprocess.Popen(result["command"], cwd=directory, env=env,
                                stdout=out, stderr=err, start_new_session=True)
        try:
            result["returncode"] = proc.wait(timeout=timeout)
        except BaseException as exc:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            if not isinstance(exc, subprocess.TimeoutExpired):
                raise
            result.update(returncode=proc.returncode, timed_out=True)
    result["wall_seconds"] = time.perf_counter() - started
    write(directory / f"{label}.process.json", result)
    if result["returncode"] != 0 or result["timed_out"]:
        raise RuntimeError(f"{label}: returncode={result['returncode']}, timeout={result['timed_out']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--injection", type=Path, default=Path("/tmp/pc_sampling_continuous_sm86_20260930/libpc_sampling_continuous.so"))
    parser.add_argument("--decoder", type=Path, default=Path("/tmp/pc_sampling_utility_sm86_20260930/pc_sampling_utility"))
    parser.add_argument("--toolkit", type=Path, default=Path("/usr/local/cuda-13.2"))
    parser.add_argument("--cohort", choices=("all", "base", "large"), default="all")
    parser.add_argument("--workload", action="append", default=[], help="Campaign ID; repeat to select multiple IDs")
    parser.add_argument("--launches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--sampling-period", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=75.0)
    args = parser.parse_args()
    if args.launches <= 0 or args.warmup < 0 or not 5 <= args.sampling_period <= 31:
        parser.error("positive launches, nonnegative warmup, and sampling period in [5,31] required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    base_env = dict(os.environ)
    for name in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "INJECTION_PARAM", "NVBIT_TOOL", "SIMBENCH_BRACKET_REPS"):
        base_env.pop(name, None)
    base_env.update({
        "SIMBENCH_ALPHA_NO_MANIFEST": "1", "NVBIT_ALPHA_NO_HASH": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": f"{args.toolkit}/extras/CUPTI/lib64:{args.toolkit}/lib64:" + base_env.get("LD_LIBRARY_PATH", ""),
    })
    execute(["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,driver_version", "--format=csv,noheader"],
            args.output, "host_observation", base_env, args.timeout)
    workloads = []
    for cohort, name in (("base", "benchmark_spec.json"), ("large", "large_workload_spec.json")):
        if args.cohort not in ("all", cohort):
            continue
        for spec in json.loads((SPECS / name).read_text())["workloads"]:
            identifier = f"{cohort}__{spec['id']}"
            if not args.workload or identifier in args.workload:
                workloads.append((identifier, spec))
    if not workloads or set(args.workload) - {identifier for identifier, _ in workloads}:
        parser.error("empty selection or unknown campaign workload ID")
    write(args.output / "settings.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {key: base_env[key] for key in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "SIMBENCH_ALPHA_NO_MANIFEST", "NVBIT_ALPHA_NO_HASH")},
        "measurement_role": "observation only, not prediction features for the scored workloads",
        "sampler_cache_control": "no explicit flush; repeated-launch history, not NCU cache-control equivalence",
    })
    rows = []
    started = time.perf_counter()
    for identifier, spec in workloads:
        directory = args.output / identifier
        directory.mkdir()
        slug = spec.get("runner_slug", Path(spec["source"]).parent.name)
        binary = args.binary_campaign.resolve() / identifier / slug
        row = {"workload": identifier, "status": "running"}
        try:
            env = {**base_env, "CUDA_INJECTION64_PATH": str(args.injection.resolve()),
                   "INJECTION_PARAM": f"--collection-mode 2 --sampling-period {args.sampling_period} --file-name pcsampling.dat"}
            command = [binary, "--timing_shape", "--allow-any-device", "--warmup", args.warmup,
                       "--reps", args.launches, *spec.get("timing_overrides", [])]
            run = execute(command, directory, "sampling", env, args.timeout)
            data_files = sorted(directory.glob("*_pcsampling.dat"))
            if not data_files:
                raise RuntimeError("producer exited without a PC-sampling file")
            summaries = []
            for data in data_files:
                label = f"decode_{data.stem}"
                # This fresh directory has no numbered cubins. Source correlation
                # is disabled; the stock utility does not invoke module CRC work.
                execute([args.decoder.resolve(), "--file-name", data, "--disable-source-correlation"],
                        directory, label, base_env, args.timeout)
                summary = summarize(argparse.Namespace(input=directory / f"{label}.stdout",
                                                       output=directory / f"{label}.summary.json"))
                summaries.append(summary)
            row.update(status="completed", acquisition_seconds=run["wall_seconds"],
                       sample_count=sum(item["total_samples"] for item in summaries),
                       dropped_samples=sum(item["dropped_samples"] for item in summaries),
                       pc_records=sum(item["pc_record_count"] for item in summaries), summaries=summaries)
            print(f"{identifier}: {row['sample_count']} samples, {row['dropped_samples']} drops", flush=True)
        except (OSError, RuntimeError, ValueError) as exc:
            row.update(status="failed", error=str(exc))
            print(f"{identifier}: {exc}", flush=True)
        write(directory / "result.json", row)
        rows.append(row)
        report = {"schema": "sm86_native_cupti_pc_campaign_v1", "planned_workloads": len(workloads),
                  "completed_workloads": sum(item["status"] == "completed" for item in rows),
                  "wall_seconds": time.perf_counter() - started, "workloads": rows,
                  "ncu_labels_consumed": False, "instrumented_event_times_not_used": True}
        write(args.output / "summary.json", report)
    print(f"Completed {report['completed_workloads']}/{len(workloads)}; {args.output}")
    return int(report["completed_workloads"] != len(workloads))


if __name__ == "__main__":
    raise SystemExit(main())
