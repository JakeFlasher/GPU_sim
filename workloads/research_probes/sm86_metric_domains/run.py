#!/usr/bin/env python3
"""Small then scaled SM86 metric-domain experiment; no model fitting."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

REPO = Path(__file__).resolve().parents[3]
METRICS = [
    "smsp__inst_executed.sum", "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fma.sum", "smsp__inst_executed_pipe_lsu.sum",
    "gpc__cycles_elapsed.max", "sm__cycles_active.avg", "dram__sectors_read.sum",
    "dram__sectors_write.sum", "lts__t_sectors_srcunit_tex.sum",
] + [f"l1tex__t_{quantity}_pipe_lsu_mem_global_op_{op}.sum"
     for op in ("ld", "st", "atom", "red") for quantity in ("requests", "sectors")]


def run(command, directory, label, env=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{label}.command.txt").write_text(shlex.join(map(str, command)) + "\n")
    start = time.monotonic()
    with (directory / f"{label}.stdout").open("w") as out, (directory / f"{label}.stderr").open("w") as err:
        result = subprocess.run(list(map(str, command)), cwd=REPO, env=env,
                                stdout=out, stderr=err, timeout=240)
    (directory / f"{label}.result.json").write_text(json.dumps({
        "exit_code": result.returncode, "seconds": time.monotonic() - start}, indent=2))
    if result.returncode:
        raise RuntimeError(f"{label} failed: {(directory / f'{label}.stderr').read_text()[-2000:]}")


def write_comparison(root, size):
    """Keep DRAM writes, eviction demand, and residency in separate columns."""
    rows = []
    for case in sorted(root.glob(f"{size}_*/case.json")):
        directory = case.parent
        output = json.loads((directory / "replay.stdout").read_text())
        launch, obs = output["launches"][0], output["ncu_observation"]
        metrics = launch["native_metrics"]["metrics"]
        row = json.loads(case.read_text())
        row.update({
            "sim_instructions": launch["counters"]["instructions_issued"],
            "ncu_instructions": obs["warp_instructions"],
            "sim_store_requests": metrics["global_store_requests"]["value"],
            "ncu_store_requests": obs["global_store_requests"],
            "sim_store_sectors": metrics["global_store_sectors"]["value"],
            "ncu_store_sectors": obs["global_store_sectors"],
            "sim_dram_read_sectors": metrics["dram_read_bytes"]["value"] / 32,
            "ncu_dram_read_sectors": obs["dram_read_sectors"],
            "sim_dirty_eviction_sectors": metrics["dirty_eviction_sectors"]["value"],
            "sim_terminal_dirty_sectors": metrics["terminal_dirty_sectors"]["value"],
            "sim_dram_write_status": metrics["dram_write_sectors"]["status"],
            "ncu_dram_write_sectors": obs["dram_write_sectors"],
        })
        rows.append(row)
    if rows:
        with (root / f"{size}.comparison.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", choices=("small", "large"), default="small")
    parser.add_argument("--suite", choices=("domains", "writeback"), default="domains")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    binary = root / "metric_domains"
    source = Path(__file__).with_name(
        "main_writeback.cu" if args.suite == "writeback" else "main.cu"
    )
    # Always rebuild in alpha mode.  Without artifact identity binding, a
    # cached executable would silently mix an old source with a new campaign.
    run(["/usr/local/cuda-13.2/bin/nvcc", "-ccbin=/usr/bin/g++-14", "-arch=sm_86", "-O2", "-lineinfo",
         source, "-o", binary], root, "build")
    hardware = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,driver_version,clocks.current.sm,clocks.current.memory",
        "--format=csv"], text=True)
    (root / "host.csv").write_text(hardware)
    control = root / "control.json"
    run([REPO / ".venv/bin/python", "tools/sass_control/emit_control_bits_sidecar.py", "--binary", binary,
         "--arch", "sm_86", "--toolkit-bindir", "/usr/local/cuda-13.2/bin", "--out", control], root, "control")
    (root / f"{args.size}.inputs.json").write_text(json.dumps({
        "seed": "i*1664525+1013904223 (u32)",
        "measurement_role": "alpha_development_probe",
    }, indent=2))
    blocks, iterations = (1, 8) if args.size == "small" else (168, 32)
    if args.suite == "writeback":
        # Footprints chosen from the 6 MiB architecture capacity, before NCU:
        # repeated hot line, 1 KiB stream; then 4 MiB and 16 MiB streams.
        cases = ([("store", 0, 1, 8), ("store", 4, 1, 8), ("control", 0, 1, 8)]
                 if args.size == "small" else
                 [("store", 4, 256, 128), ("store", 4, 1024, 128)])
    else:
        cases = [(op, p, blocks, iterations) for op in ("read", "atom", "red")
                 for p in ((0, 1, 2) if args.size == "small" else (0, 2, 3))]
        cases += [("control", 0, blocks, iterations)]
    for op, pattern, blocks, iterations in cases:
        name = f"{args.size}_{op}_{pattern}"
        if args.suite == "writeback":
            name += f"_b{blocks}_i{iterations}"
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "case.json").write_text(json.dumps({
            "case": name, "operation": op, "pattern": pattern, "blocks": blocks,
            "iterations": iterations, "suite": args.suite, "role": "development_probe",
        }, indent=2) + "\n")
        command = [binary, op, pattern, str(blocks), str(iterations)]
        run(command, directory, "native")
        run([REPO / ".venv/bin/python", "tools/local_labels/collect_ncu.py", "--output", directory / "ncu",
             "--ncu", "/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu", "--timeout", "180",
             "--launch-count", "1", "--replay-mode", "application", "--clock-control", "none",
             "--cache-control", "all", "--metrics", ",".join(METRICS), "--", *command], directory, "ncu")
        env = dict(os.environ)
        for key in ("CUDA_INJECTION64_PATH", "LD_PRELOAD", "NVBIT_TOOL", "CUDA_VISIBLE_DEVICES"):
            env.pop(key, None)
        extra = {
            "CUDA_INJECTION64_PATH": str(REPO / "tools/nvbit_tracer/tracer_tool/tracer_tool.so"),
            "NVBIT_ALPHA_NO_HASH": "1",
            "LD_LIBRARY_PATH": str(REPO / ".toolchain/pb/lib"),
            "PATH": "/usr/local/cuda-13.2/bin:" + env["PATH"],
            "TRACES_FOLDER": str(directory / "trace"), "USER_DEFINED_FOLDERS": "1",
            "ACTIVE_FROM_START": "1", "DYNAMIC_KERNEL_LIMIT_START": "0", "DYNAMIC_KERNEL_LIMIT_END": "1",
            "SKIP_ENHANCED_TRACER": "0", "ENHANCED_TRACER_MAX_PARSER_INPUT_KB": "1024",
        }
        env.update(extra)
        (directory / "trace.environment.json").write_text(json.dumps(extra, indent=2))
        run(command, directory, "trace", env)
        run(["sim/target/release/sim-run", "--trace", directory / "trace", "--control", control,
             "--arch-card", "arch_profiles/cards/sm_86.json", "--pchase-card", "data/pchase/sm_86_static_card.json",
             "--mode", "l1-l2", "--ncu-shaped-metrics", "--ncu-csv", directory / "ncu/ncu.csv"], directory, "replay")
        (directory / "replay.inputs.json").write_text(json.dumps({
            "measurement_role": "alpha_development_probe",
            "inputs": [str(REPO / "sim/target/release/sim-run"), str(binary), str(control)],
        }, indent=2) + "\n")
        output = json.loads((directory / "replay.stdout").read_text())
        obs, launch = output["ncu_observation"], output["launches"][0]
        print(directory.name, launch["counters"]["instructions_issued"], obs["warp_instructions"],
              {k: v["sectors"] for k, v in launch["trace_metrics"]["memory"].items()},
              {k: obs[k] for k in ("global_atomic_sectors", "global_reduction_sectors", "dram_read_sectors")}, flush=True)
        write_comparison(root, args.size)


if __name__ == "__main__":
    main()
