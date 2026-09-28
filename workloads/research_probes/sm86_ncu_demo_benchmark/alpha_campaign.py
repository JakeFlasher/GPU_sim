#!/usr/bin/env python3
"""Run every base/expanded workload through the real alpha performance paths.

No correctness mode, test suite, artifact hashing, or scoring gates. NCU is
collected after replay and is never used to select simulator parameters.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from tools.sass_control.extractor import parse_sass_text

METRICS = [
    "gpu__time_duration.sum", "gpc__cycles_elapsed.max", "sm__cycles_active.max",
    "smsp__inst_executed.sum", "dram__sectors_read.sum", "dram__sectors_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "lts__t_sectors_srcunit_tex_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_lookup_miss.sum",
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute(command, directory, label, env, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    command = list(map(str, command))
    started = time.monotonic()
    result = {"command": command, "returncode": None, "timed_out": False}
    with (directory / f"{label}.stdout").open("w") as out, (directory / f"{label}.stderr").open("w") as err:
        try:
            proc = subprocess.Popen(command, cwd=directory, env=env, stdout=out, stderr=err,
                                    start_new_session=True)
            try:
                result["returncode"] = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        pass
                    if sig == signal.SIGTERM:
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                try:
                    result["returncode"] = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    result["returncode"] = -signal.SIGKILL
        except OSError as exc:
            result["error"] = str(exc)
    result["wall_seconds"] = time.monotonic() - started
    result["status"] = "completed" if result["returncode"] == 0 and not result["timed_out"] else "failed"
    write_json(directory / f"{label}.process.json", result)
    return result


def distribution(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None, "stdev": None}
    return {"n": len(values), "median": statistics.median(values), "min": min(values),
            "max": max(values), "stdev": statistics.stdev(values) if len(values) > 1 else None}


def read_stdout_json(path):
    text = path.read_text()
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def read_ncu(path):
    rows = list(csv.reader(io.StringIO(path.read_text())))
    if not rows:
        return []
    try:
        header_index = next(index for index, row in enumerate(rows) if "Kernel Name" in row)
    except StopIteration:
        return []
    header = rows[header_index]
    unit_index = header_index + 1
    unit_values = rows[unit_index] if unit_index < len(rows) else []
    unit_markers = {"nsecond", "nanoseconds", "ns", "usecond", "us", "msecond", "ms", "second", "s", "cycle", "cycles"}
    has_units = any(value.strip().lower() in unit_markers for value in unit_values)
    units = dict(zip(header, unit_values)) if has_units else {}
    data_start = unit_index + 1 if has_units else unit_index
    launches = {}
    for values in rows[data_start:]:
        if len(values) < len(header):
            continue
        raw = dict(zip(header, values))
        key = (raw.get("ID"), raw.get("Kernel Name"))
        if key[1] in (None, ""):
            continue
        launch = launches.setdefault(key, {"id": key[0], "kernel": key[1], "metrics": {}})
        for metric in METRICS:
            text = raw.get(metric, "")
            try:
                value = float(text.replace(",", ""))
                if not math.isfinite(value):
                    value = None
            except ValueError:
                value = None
            unit = units.get(metric)
            entry = {"value": value, "unit": unit}
            if metric == "gpu__time_duration.sum":
                multiplier = {"nsecond": 1.0, "nanoseconds": 1.0, "ns": 1.0, "usecond": 1000.0,
                              "us": 1000.0, "msecond": 1_000_000.0,
                              "ms": 1_000_000.0, "second": 1_000_000_000.0,
                              "s": 1_000_000_000.0}.get(unit)
                entry["nanoseconds"] = None if value is None or multiplier is None else value * multiplier
            launch["metrics"][metric] = entry
    return list(launches.values())


def target_command(binary, workload, warmup, reps):
    return [binary, "--timing_shape", "--warmup", str(warmup), "--reps", str(reps),
            "--allow-any-device", *workload.get("timing_overrides", [])]


def run_workload(workload, args, root, env):
    campaign_id = workload.get("campaign_id", workload["id"])
    directory = root / campaign_id
    directory.mkdir()
    source = REPO / workload["source"]
    slug = workload.get("runner_slug", source.parent.name)
    binary = directory / slug
    row = {"id": campaign_id, "source_id": workload["id"], "shape": workload.get("timing_shape"),
           "status": "running", "processes": {}, "native": [], "replay": [], "ncu": [], "errors": []}

    def step(command, label, run_env=env):
        print(f"{campaign_id}: {label}", flush=True)
        result = execute(command, directory, label, run_env, args.timeout)
        row["processes"][label] = result
        if result["status"] != "completed":
            row["errors"].append({"stage": label, **result})
        write_json(directory / "result.json", row)
        return result["status"] == "completed"

    if not step([args.toolkit / "bin/nvcc", f"-ccbin={args.cxx}", "-std=c++17", "-O2",
                 "-lineinfo", "-gencode=arch=compute_80,code=sm_86",
                 "-I", REPO / "workloads/portable_lcd/cutlass/upstream",
                 source.parent / f"{slug}.cu", source, "-o", binary], "build"):
        row["status"] = "build_failed"
        return row

    for rep in range(args.native_repeats):
        label = f"native_{rep}"
        if step(target_command(binary, workload, args.warmup, args.native_launches), label):
            try:
                row["native"].append(read_stdout_json(directory / f"{label}.stdout"))
            except (ValueError, KeyError) as exc:
                row["errors"].append({"stage": label, "error": str(exc)})

    control = directory / "control.json"
    disassembled = step([args.toolkit / "bin/cuobjdump", "--dump-sass", binary], "sass")
    if disassembled:
        try:
            # Parse the actual sidecar input directly, without a toolkit/version gate.
            parsed = parse_sass_text((directory / "sass.stdout").read_text(), "sm_86")
            summary = {}
            total_decoded = total_unknown = 0
            for function_name, records in parsed["functions"].items():
                decoded = sum(1 for record in records if record.get("status") == "decoded")
                unknown = len(records) - decoded
                total_decoded += decoded
                total_unknown += unknown
                summary[function_name] = {
                    "instructions": len(records), "decoded": decoded,
                    "unknown_encoding": unknown,
                    "stall_cycles_static_sum": sum(
                        record["control"]["stall"] for record in records
                        if record.get("status") == "decoded"
                    ),
                }
            parsed.update({
                "binary": str(binary), "toolkit_version": "unreported-alpha",
                "summary": {"functions": summary, "total_decoded": total_decoded,
                             "total_unknown_encoding": total_unknown},
            })
            write_json(control, parsed)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            row["errors"].append({"stage": "control_parse", "error": str(exc)})
            disassembled = False
    trace_env = dict(env)
    trace_env.update({
        "CUDA_INJECTION64_PATH": str(args.tracer), "NVBIT_ALPHA_NO_HASH": "1",
        "LD_LIBRARY_PATH": str(REPO / ".toolchain/pb/lib") + ":" + env.get("LD_LIBRARY_PATH", ""),
        "TRACES_FOLDER": str(directory / "trace"), "USER_DEFINED_FOLDERS": "1",
        "ACTIVE_FROM_START": "1", "DYNAMIC_KERNEL_LIMIT_START": "0", "DYNAMIC_KERNEL_LIMIT_END": "1",
        "SKIP_ENHANCED_TRACER": "0", "ENHANCED_TRACER_MAX_PARSER_INPUT_KB": "1024",
    })
    traced = step(target_command(binary, workload, 0, 1), "trace", trace_env)
    if traced and disassembled:
        replay_env = dict(env, SIM_RUN_PERFORMANCE_COUNTERS="1")
        for rep in range(args.replay_repeats):
            label = f"replay_{rep}"
            if step([args.simulator,
                     "--trace", directory / "trace", "--control", control,
                     "--arch-card", REPO / "arch_profiles/cards/sm_86.json",
                     "--pchase-card", REPO / "data/pchase/sm_86_static_card.json",
                     "--mode", "l1-l2", "--ncu-shaped-metrics"], label, replay_env):
                try:
                    report = read_stdout_json(directory / f"{label}.stdout")
                    row["replay"].append({"wall_seconds": row["processes"][label]["wall_seconds"],
                                          "peak_rss_kb": None,
                                          "launches": report["launches"]})
                except (OSError, ValueError, KeyError) as exc:
                    row["errors"].append({"stage": label, "error": str(exc)})

    for rep in range(args.ncu_repeats):
        label = f"ncu_{rep}"
        base = directory / label
        if step([args.ncu, "--export", base, "--replay-mode", "application",
                 "--clock-control", "none", "--cache-control", "all",
                 "--launch-skip", args.warmup, "--launch-count", "1", "--metrics", ",".join(METRICS),
                 *target_command(binary, workload, args.warmup, 1)], label):
            export_label = f"{label}_csv"
            if step([args.ncu, "--import", str(base) + ".ncu-rep", "--csv", "--page", "raw"], export_label):
                try:
                    launches = read_ncu(directory / f"{export_label}.stdout")
                    if not launches:
                        raise ValueError("NCU CSV contained no kernel launch rows")
                    row["ncu"].append({"repeat": rep, "launches": launches})
                except (OSError, ValueError, StopIteration) as exc:
                    row["errors"].append({"stage": export_label, "error": str(exc)})
    row["status"] = "completed" if not row["errors"] else "completed_with_errors"
    write_json(directory / "result.json", row)
    return row


def summarize(rows):
    summaries = []
    for row in rows:
        native = distribution([v.get("mean_ms") for v in row["native"]])
        replay = distribution([v["wall_seconds"] for v in row["replay"]])
        ncu_metrics = {}
        for metric in METRICS:
            entries = [launch.get("metrics", {}).get(metric, {})
                       for rep in row.get("ncu", []) for launch in rep.get("launches", [])]
            key = "nanoseconds" if metric == "gpu__time_duration.sum" else "value"
            ncu_metrics[metric] = distribution([entry.get(key) for entry in entries])
        launches = row.get("replay", [])[0].get("launches", []) if row.get("replay") else []
        issued_values = [launch.get("counters", {}).get("instructions_issued") for launch in launches]
        issued = sum(value for value in issued_values if isinstance(value, (int, float))) if issued_values else None
        summaries.append({
            "id": row["id"], "status": row["status"], "shape": row["shape"],
            "native_mean_ms": native, "replay_wall_seconds": replay, "ncu": ncu_metrics,
            "replay_peak_rss_kb": max((v["peak_rss_kb"] for v in row["replay"] if v["peak_rss_kb"] is not None), default=None),
            "warp_instructions": issued,
            "replay_warp_instructions_per_second": issued / replay["median"] if issued is not None and replay["median"] else None,
            "frontiers": [{"kernel": l.get("kernel_name"), "timing": l.get("timing"),
                           "unmodeled_tails": l.get("counters", {}).get("unmodeled_tails"),
                           "coverage": l.get("coverage")} for l in launches],
            "failed_stages": [e.get("stage", "unknown") for e in row.get("errors", [])],
        })
    return summaries


def publish(root, rows, total, elapsed):
    summaries = summarize(rows)
    write_json(root / "summary.json", {
        "schema": "sm86_alpha_campaign_v1", "status": "finished" if len(rows) == total else "running",
        "completed_workloads": len(rows), "planned_workloads": total, "elapsed_seconds": elapsed,
        "measurement_role": "development_performance", "workloads": summaries,
        "limitations": ["no correctness assessment", "no artifact freshness claim", "no drift-free clock claim",
                        "native warm-cache CUDA event timing and cold-cache NCU durations are separate measurements",
                        "simulator issue/event/completion/GPC-envelope frontiers are separate; no full elapsed accuracy score",
                        "no proprietary physical-pipe witness or matched external timing result"],
    })
    flat = []
    for row in summaries:
        front = row["frontiers"][0].get("timing") or {} if row["frontiers"] else {}
        flat.append({"workload": row["id"], "status": row["status"],
                     "native_mean_ms": row["native_mean_ms"]["median"],
                     "ncu_duration_ns": row["ncu"]["gpu__time_duration.sum"]["median"],
                     "ncu_gpc_cycles": row["ncu"]["gpc__cycles_elapsed.max"]["median"],
                     "ncu_warp_instructions": row["ncu"]["smsp__inst_executed.sum"]["median"],
                     "sim_warp_instructions": row["warp_instructions"],
                     "sim_last_issue_cycles": front.get("diagnostic_last_issue_cycles"),
                     "sim_event_cycles": front.get("terminal_modeled_event_time"),
                     "sim_gpc_envelope_cycles": (front.get("gpc_observation_frontier") or {}).get("cycles"),
                     "replay_wall_seconds": row["replay_wall_seconds"]["median"],
                     "replay_warp_instructions_per_second": row["replay_warp_instructions_per_second"],
                     "replay_peak_rss_kb": row["replay_peak_rss_kb"],
                     "ncu_repeats": row["ncu"]["gpu__time_duration.sum"]["n"],
                     "failed_stages": ";".join(row["failed_stages"])})
    if flat:
        with (root / "performance.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--toolkit", type=Path, default=Path("/usr/local/cuda-13.2"))
    parser.add_argument("--cxx", default="/usr/bin/g++-14")
    parser.add_argument("--ncu", default="/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu")
    parser.add_argument("--simulator", type=Path, default=REPO / "sim/target/release/sim-run")
    parser.add_argument("--tracer", type=Path, default=REPO / "tools/nvbit_tracer/tracer_tool/tracer_tool.so")
    parser.add_argument("--ncu-repeats", type=int, default=7)
    parser.add_argument("--native-repeats", type=int, default=5)
    parser.add_argument("--native-launches", type=int, default=100)
    parser.add_argument("--replay-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    args.toolkit, args.simulator, args.tracer = args.toolkit.resolve(), args.simulator.resolve(), args.tracer.resolve()
    if args.output is None:
        parent = REPO / "results/alpha"
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="full_3090ti_", dir=parent))
    else:
        root = args.output.resolve()
        root.mkdir(parents=True)
    workloads = []
    for cohort, name in (("base", "benchmark_spec.json"), ("large", "large_workload_spec.json")):
        for source_workload in json.loads((HERE / name).read_text())["workloads"]:
            workload = dict(source_workload)
            workload["campaign_id"] = f"{cohort}__{workload['id']}"
            workloads.append(workload)
    env = dict(os.environ)
    for key in ("CUDA_INJECTION64_PATH", "LD_PRELOAD", "NVBIT_TOOL", "SIMBENCH_BRACKET_REPS"):
        env.pop(key, None)
    env.update({"CUDA_VISIBLE_DEVICES": "0", "SIMBENCH_ALPHA_NO_MANIFEST": "1",
                "PATH": str(args.toolkit / "bin") + ":" + env.get("PATH", "")})
    print(f"Campaign output: {root}\nWorkloads: {len(workloads)}", flush=True)
    write_json(root / "settings.json", {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                       "workloads": [w["campaign_id"] for w in workloads]})
    rows = []
    started = time.monotonic()
    try:
        for workload in workloads:
            try:
                row = run_workload(workload, args, root, env)
            except Exception as exc:
                row = {"id": workload.get("campaign_id", workload["id"]),
                       "source_id": workload["id"], "shape": workload.get("timing_shape"), "status": "failed",
                       "native": [], "replay": [], "ncu": [], "errors": [{"stage": "workload", "error": str(exc)}]}
                print(f"{workload.get('campaign_id', workload['id'])}: {exc}", flush=True)
            rows.append(row)
            publish(root, rows, len(workloads), time.monotonic() - started)
    finally:
        publish(root, rows, len(workloads), time.monotonic() - started)
    print(f"Performance results: {root / 'performance.csv'}", flush=True)
    return int(any(row["status"] != "completed" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
