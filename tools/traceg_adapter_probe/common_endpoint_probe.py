#!/usr/bin/env python3
"""Bounded common-input local/Accel-Sim endpoint probe.

The probe constructs one tiny full-mask SM86 ALU launch from the retained
canary schema, lowers it through the strict adapter, parses/runs it through a
caller-supplied pinned Accel-Sim tree, and replays the corresponding enhanced
protobuf through local ``sim-run``. It is an independent parser/representation
smoke (E10), not the literature lower-bound evaluation (E10-L) and not a
required direct executable timing score (E10-P). The endpoints are deliberately
reported side-by-side and are **not** converted into an NCU score:
Accel-Sim's ``gpu_tot_sim_cycle`` and local ``terminal_modeled_event_time`` are
different model domains.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

# Permit the documented direct `python3 tools/.../common_endpoint_probe.py`
# invocation without requiring callers to set PYTHONPATH manually.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.traceg_adapter_probe.adapter import adapt
from tools.traceg_adapter_probe.external_compare import compare
from tools.nvbit_tracer.proto_binding_loader import load_dynamic_trace_bindings


CANARY = REPO / "sim/run/tests/fixtures/addr_carry_canary_bundle"
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_PROVENANCE_BYTES = 64 * 1024 * 1024


class ProbeRefusal(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(detail)


def _file_identity(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ProbeRefusal("provenance_missing", f"cannot read {path}: {exc}") from exc
    if len(data) > MAX_PROVENANCE_BYTES:
        raise ProbeRefusal("provenance_limit_exceeded", f"{path} exceeds provenance limit")
    return {"path": str(path), "size_bytes": len(data)}


def _copy_canary_inputs(root: Path) -> dict[str, Path]:
    source = CANARY
    paths = {
        "trace": root / "dynamic_trace.pb",
        "threadblock": root / "threadblock.pb",
        "control": root / "control.json",
        "exec_info": root / "enhanced_execution_info.json",
    }
    shutil.copyfile(source / "trace/dynamic_trace.pb", paths["trace"])
    shutil.copyfile(
        source / "trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb",
        paths["threadblock"],
    )
    shutil.copyfile(source / "control/canary_sm86.control_bits.json", paths["control"])
    shutil.copyfile(source / "trace/extra_info/enhanced_execution_info.json", paths["exec_info"])
    return paths


def _make_fixture(root: Path) -> tuple[dict[str, Path], Path]:
    bindings = load_dynamic_trace_bindings()
    paths = _copy_canary_inputs(root)
    trace = bindings.trace_pb2.Trace()
    trace.ParseFromString(paths["trace"].read_bytes())
    trace.trace_contract_id = "sm86_addr_carry_shmem_split_v2"
    del trace.memory_payload_classes[:]
    trace.memory_payload_classes.extend([1, 2, 3, 6, 8])
    kernel = next(iter(next(iter(trace.gpu_device.values())).streams.values())).kernels[0]
    kernel.number_of_registers = 32
    kernel.size_shared_memory = 0
    kernel.shared_memory_static_bytes = 0
    kernel.shared_memory_dynamic_bytes = 0
    paths["trace"].write_bytes(trace.SerializeToString())

    threadblock = bindings.threadblock_pb2.threadblock()
    threadblock.ParseFromString(paths["threadblock"].read_bytes())
    threadblock.block_id.x = 0
    threadblock.block_id.y = 0
    threadblock.block_id.z = 0
    warp = threadblock.warps[0]
    warp.id = 0
    del warp.instructions[:]
    for pc in (0, 16, 32, 48):
        instruction = warp.instructions.add()
        instruction.pc = pc
        # Accel-Sim's explicit EXIT is a zero-mask marker.  Make the source
        # terminal carry the same effective mask so instruction-count
        # alignment is not obtained by silently projecting a full-mask EXIT.
        instruction.active_mask = 0 if pc == 48 else 0xFFFF_FFFF
        instruction.predicate_mask = 0 if pc == 48 else 0xFFFF_FFFF
        instruction.function_unique_id = 1
        instruction.warpid_tb = 0
        instruction.warpid_sm = 0
        instruction.sm_id = 0
        instruction.vpc = pc
    paths["threadblock"].write_bytes(threadblock.SerializeToString())

    control = json.loads(paths["control"].read_text())
    function_name, instructions = next(iter(control["functions"].items()))
    for instruction in instructions:
        if instruction["pc"] == 16:
            instruction["opcode_text"] = "FADD R0, R1, R2"
        elif instruction["pc"] == 48:
            instruction["opcode_text"] = "EXIT"
    control["functions"][function_name] = instructions
    paths["control"].write_text(json.dumps(control, sort_keys=True) + "\n")

    out = root / "adapter"
    adapt(paths["trace"], paths["threadblock"], paths["control"], out, exec_info_path=paths["exec_info"])

    trace_dir = root / "trace"
    threadblock_dir = trace_dir / "threadblocks/device_0/stream_0/kernel_1"
    threadblock_dir.mkdir(parents=True)
    shutil.copyfile(paths["trace"], trace_dir / "dynamic_trace.pb")
    shutil.copyfile(paths["threadblock"], threadblock_dir / "d_0_s_0_k_1_0,0,0.pb")
    extra = trace_dir / "extra_info"
    extra.mkdir()
    shutil.copyfile(paths["exec_info"], extra / "enhanced_execution_info.json")
    shutil.copyfile(CANARY / "trace/extra_info/allocations.json", extra / "allocations.json")
    shutil.copyfile(CANARY / "trace/stats.csv", trace_dir / "stats.csv")
    shutil.copyfile(CANARY / "cards/sm_86.json", root / "sm_86.json")
    shutil.copyfile(CANARY / "cards/sm_86_static_card.json", root / "sm_86_static_card.json")
    return paths, out


def _run_local(root: Path, binary: Path, timeout: float) -> dict[str, Any]:
    command = [
        str(binary),
        "--trace",
        str(root / "trace"),
        "--control",
        str(root / "control.json"),
        "--arch-card",
        str(root / "sm_86.json"),
        "--pchase-card",
        str(root / "sm_86_static_card.json"),
        "--mode",
        "l1-l2",
        "--cache-scope",
        "launch",
    ]
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProbeRefusal("local_sim_timeout", f"local sim-run exceeded {timeout}s") from exc
    if completed.returncode != 0:
        raise ProbeRefusal("local_sim_failed", completed.stderr[-4000:])
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        document = json.loads(completed.stdout)
        launch = document["launches"][0]
        timing = launch["timing"]
        trace_metrics = launch.get("trace_metrics", {})
        instruction_activity = trace_metrics.get("instructions", {})
        by_opcode = trace_metrics.get("by_opcode", {})
        terminal_activity = by_opcode.get("EXIT", {})
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ProbeRefusal("local_output_invalid", str(exc)) from exc
    warp_instructions = instruction_activity.get("warp_instructions")
    active_thread_instructions = instruction_activity.get("active_thread_instructions")
    terminal_warp_instructions = terminal_activity.get("warp_instructions", 0)
    terminal_thread_instructions = terminal_activity.get("active_thread_instructions", 0)
    return {
        "status": "ok",
        "command": command,
        "instructions_issued": document.get("launches", [{}])[0]
        .get("counters", {})
        .get("instructions_issued"),
        "warp_instructions": instruction_activity.get("warp_instructions"),
        "active_thread_instructions": instruction_activity.get(
            "active_thread_instructions"
        ),
        "terminal_record_count": terminal_warp_instructions,
        "nonterminal_dynamic_record_count": (
            None
            if warp_instructions is None
            else max(0, warp_instructions - terminal_warp_instructions)
        ),
        "active_thread_instructions_excluding_terminal": (
            None
            if active_thread_instructions is None
            else max(0, active_thread_instructions - terminal_thread_instructions)
        ),
        "terminal_modeled_event_time": timing.get("terminal_modeled_event_time"),
        "sm_active_terminal_frontier": timing.get("sm_active_terminal_frontier"),
        "diagnostic_last_issue_cycles": timing.get("diagnostic_last_issue_cycles"),
        "completion_frontier": timing.get("completion_frontier"),
        "resource": {
            "wall_seconds": time.monotonic() - started,
            "child_user_seconds": max(0.0, child_after.ru_utime - child_before.ru_utime),
            "child_system_seconds": max(0.0, child_after.ru_stime - child_before.ru_stime),
            "child_max_rss_kb": child_after.ru_maxrss,
        },
    }


def _external_launch_latency(config: Path) -> int | None:
    try:
        raw = config.read_bytes()
    except OSError:
        return None
    if len(raw) > MAX_CONFIG_BYTES:
        return None
    match = re.search(
        rb"(?m)^\s*-?gpgpu_kernel_launch_latency\s+([0-9]+)", raw
    )
    return None if match is None else int(match.group(1))


def run_probe(
    parser_root: Path,
    parser_library: Path,
    accel_sim: Path,
    accel_config: Path,
    trace_config: Path,
    local_binary: Path,
    timeout: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sm86_common_endpoint_") as temp:
        root = Path(temp)
        _paths, adapter_dir = _make_fixture(root)
        traceg = adapter_dir / "kernel-1.traceg"
        adapter_result = adapter_dir / "adapter_result.json"
        kernelslist = adapter_dir / "kernelslist.g"
        external = compare(
            traceg,
            adapter_result,
            parser_root,
            parser_library,
            timeout=timeout,
            accel_sim=accel_sim,
            config=accel_config,
            kernelslist=kernelslist,
            extra_args=("-config", str(trace_config)),
        )
        local = _run_local(root, local_binary, timeout)
        accel_metrics = external.get("accel_sim", {})
        launch_latency = _external_launch_latency(accel_config)
        external_cycle = accel_metrics.get("gpu_tot_sim_cycle")
        post_launch_cycle = (
            None
            if external_cycle is None or launch_latency is None
            else max(0, external_cycle - launch_latency)
        )
        return {
            "schema": "sm86_common_endpoint_probe_v1",
            "status": "ok",
            "scoring": False,
            "comparison_role": "executable_parser_representation_smoke",
            "ground_truth_lane": "local_ncu",
            "literature_floor_manifest": str(
                REPO / "docs/accelsim_literature_lower_bound_manifest_20260928.json"
            ),
            "adapter": json.loads(adapter_result.read_text()),
            "external": external,
            "local": local,
            "provenance": {
                "local_binary": _file_identity(local_binary),
                "local_arch_card": _file_identity(root / "sm_86.json"),
                "local_pchase_card": _file_identity(root / "sm_86_static_card.json"),
                "local_control": _file_identity(root / "control.json"),
                "local_dynamic_trace": _file_identity(root / "trace/dynamic_trace.pb"),
                "local_threadblock": _file_identity(root / "trace/threadblocks/device_0/stream_0/kernel_1/d_0_s_0_k_1_0,0,0.pb"),
                "external_accel_binary": external.get("provenance", {}).get("accel_sim_binary"),
                "external_parser_library": external.get("provenance", {}).get("external_parser_library"),
                "external_accel_config": _file_identity(accel_config),
                "external_trace_config": _file_identity(trace_config),
            },
            "architecture_crosswalk": {
                "schema": "sm86_external_crosswalk_v1",
                "status": "capability_mismatch",
                "scoring": False,
                "local_target": "RTX3090Ti_SM86_local_profile",
                "external_target": "SM86_RTX3070_AccelSim_profile",
                "unmapped_fields": [
                    "instruction_front_end",
                    "fetch_decode_width_and_stages",
                    "SMSP_scheduler_and_issue_slots",
                    "cache_geometry_and_replacement",
                    "DRAM_address_mapping_and_latency",
                    "control_word_stall_yield_reuse_semantics",
                    "launch_envelope_and_full_drain_endpoint",
                ],
                "reason": "same compute capability does not establish SKU/card/front-end equivalence; no timing ratio is admissible",
            },
            "endpoint_mapping": {
                "status": "not_aligned",
                "external": "gpu_tot_sim_cycle",
                "local": "terminal_modeled_event_time",
                "external_launch_latency_cycles": launch_latency,
                "external_post_launch_cycle": post_launch_cycle,
                "instruction_counts": {
                    "external": "gpu_tot_sim_insn (Accel-Sim thread-level counter)",
                    "local": "active_thread_instructions_excluding_terminal (local trace-derived counter)",
                    "external_value": external.get("accel_sim", {}).get(
                        "gpu_tot_sim_insn"
                    ),
                    "local_value": local.get("active_thread_instructions_excluding_terminal"),
                    "status": (
                        "aligned"
                        if external.get("accel_sim", {}).get("gpu_tot_sim_insn")
                        == local.get("active_thread_instructions_excluding_terminal")
                        else "not_aligned"
                    ),
                },
                "reason": "cycle/retirement domains remain different; instruction-count alignment is reported separately; no NCU or silicon score",
            },
        }


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser-root", type=Path, required=True)
    parser.add_argument("--parser-library", type=Path, required=True)
    parser.add_argument("--accel-sim", type=Path, required=True)
    parser.add_argument("--accel-config", type=Path, required=True)
    parser.add_argument("--trace-config", type=Path, required=True)
    parser.add_argument("--local-binary", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_probe(
            args.parser_root,
            args.parser_library,
            args.accel_sim,
            args.accel_config,
            args.trace_config,
            args.local_binary,
            args.timeout,
        )
    except Exception as exc:
        print(json.dumps({"schema": "sm86_common_endpoint_probe_v1", "status": "refused", "detail": str(exc)}, sort_keys=True))
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            print(json.dumps({"status": "refused", "detail": "output exists"}))
            return 2
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
