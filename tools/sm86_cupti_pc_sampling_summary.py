#!/usr/bin/env python3
"""Summarize NVIDIA CUPTI PC-sampling utility text without route inference."""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
import statistics
import time

PREFIX = "smsp__pcsamp_warps_issue_stalled_"
NO_ISSUE_SUFFIX = "_not_issued"


def split_states(counts):
    """All-cycle states and no-issue-cycle subsets are not disjoint counts."""
    states, no_issue = {}, {}
    for name, count in counts.items():
        if not name.startswith(PREFIX):
            continue
        reason = name.removeprefix(PREFIX)
        if reason.endswith(NO_ISSUE_SUFFIX):
            no_issue[reason.removesuffix(NO_ISSUE_SUFFIX)] = count
        else:
            states[reason] = count
    return states, no_issue


def run(args):
    started = time.perf_counter()
    text = args.input.read_text()
    buffers = list(re.finditer(r"Count of PC records: (\d+), Total Samples: (\d+), Total Dropped Samples: (\d+)", text))
    if not buffers:
        raise RuntimeError("CUPTI utility buffer summary not found")
    reason_counts = collections.Counter()
    function_counts = {}
    buffer_samples = []
    current_buffer = None
    pc_offsets = set()
    qualified_pcs = set()
    pc_records = []
    for line in text.splitlines():
        buffer_match = re.search(r"Buffer Number: (\d+), Range Id: (\d+), Count of PC records: (\d+), Total Samples: (\d+), Total Dropped Samples: (\d+)", line)
        if buffer_match:
            current_buffer = {
                "buffer": int(buffer_match.group(1)),
                "range_id": int(buffer_match.group(2)),
                "pc_records": int(buffer_match.group(3)),
                "samples": int(buffer_match.group(4)),
                "dropped": int(buffer_match.group(5)),
                "stall_reason_samples": collections.Counter(),
            }
            buffer_samples.append(current_buffer)
            continue
        match = re.search(r"pcOffset: (\d+).*?stallReasonCount: (\d+), (.*)$", line)
        if not match:
            continue
        pc_offsets.add(int(match.group(1)))
        identity = re.search(r"functionName: (.*?), functionIndex: (\d+), correlationId: (\d+)", line)
        function = identity.group(1) if identity else "unknown_function"
        function_index = int(identity.group(2)) if identity else None
        correlation = int(identity.group(3)) if identity else None
        qualified_pcs.add((function, function_index, int(match.group(1))))
        if function not in function_counts:
            function_counts[function] = collections.Counter()
        if current_buffer is not None:
            current_buffer.setdefault("correlation_ids", set()).add(correlation)
        record = {
            "function": function,
            "function_index": function_index,
            "correlation_id": correlation,
            "pc_offset": int(match.group(1)),
            "stall_reason_samples": {},
        }
        for reason, samples in re.findall(r"(smsp__pcsamp_[A-Za-z0-9_]+): (\d+)", match.group(3)):
            reason_counts[reason] += int(samples)
            function_counts[function][reason] += int(samples)
            record["stall_reason_samples"][reason] = int(samples)
            if current_buffer is not None:
                current_buffer["stall_reason_samples"][reason] += int(samples)
        pc_records.append(record)
    states, no_issue = split_states(reason_counts)
    for item in buffer_samples:
        item["correlation_ids"] = sorted(c for c in item.get("correlation_ids", ()) if c is not None)
        item["scheduler_state_samples"], item["scheduler_no_issue_samples"] = split_states(item.pop("stall_reason_samples"))
    report = {
        "schema": "sm86_cupti_pc_sampling_summary_v2",
        "input": str(args.input),
        "buffer_count": len(buffers),
        "pc_record_count": sum(int(buffer.group(1)) for buffer in buffers),
        "total_samples": sum(int(buffer.group(2)) for buffer in buffers),
        "dropped_samples": sum(int(buffer.group(3)) for buffer in buffers),
        "decoded_pc_offsets": sorted(pc_offsets),
        "decoded_pc_count": len(pc_offsets),
        "qualified_pc_count": len(qualified_pcs),
        "pc_records": pc_records,
        "stall_reason_samples": dict(reason_counts),
        "scheduler_state_samples": states,
        "scheduler_no_issue_samples": no_issue,
        "scheduler_state_sample_sum": sum(states.values()),
        "scheduler_no_issue_sample_sum": sum(no_issue.values()),
        "scheduler_no_issue_sample_fraction": sum(no_issue.values()) / sum(states.values()) if sum(states.values()) else None,
        "functions": {
            function: {"scheduler_state_samples": split_states(counts)[0],
                       "scheduler_no_issue_samples": split_states(counts)[1]}
            for function, counts in sorted(function_counts.items())
        },
        "buffers": buffer_samples,
        "buffer_sample_summary": {
            "buffer_count": len(buffer_samples),
            "min_samples": min((item["samples"] for item in buffer_samples), default=None),
            "median_samples": statistics.median((item["samples"] for item in buffer_samples)) if buffer_samples else None,
            "max_samples": max((item["samples"] for item in buffer_samples), default=None),
            "total_dropped": sum(item["dropped"] for item in buffer_samples),
        },
        "runtime_seconds": time.perf_counter() - started,
        "contract": "decoded aggregate PC/stall counts; no physical pipe, sub-unit, or issue-slot identity",
        "measurement_semantics": {
            "scheduler_state_samples": "all-cycle sampled warp states, including selected/not_selected (not true stalls)",
            "scheduler_no_issue_samples": "subsets sampled on cycles where the scheduler issued nothing; never add to all-cycle states",
            "imc_miss": "immediate constant cache miss, not instruction-cache miss",
            "time": "sample fractions are not unweighted wall-time fractions or a complete cycle trace",
            "v1_note": "v1 base buckets summed overlapping base and _not_issued metrics; do not use those sums as populations",
        },
        "ncu_target_labels_consumed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
