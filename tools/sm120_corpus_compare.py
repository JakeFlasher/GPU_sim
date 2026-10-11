#!/usr/bin/env python3
"""Post-replay, unit-aware comparison of whole-corpus native replay reports.

Reads one capture at a time. Nothing from NCU is sent to a simulator. An APE
between distinct endpoints is a raw discrepancy, not timing-model equivalence.
"""

import argparse
import collections
import json
import math
import re
import statistics
from pathlib import Path

from sm120_replay_bench import join_ncu_rows, parse_ncu_raw


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None}
    return {"n": len(values), "median": statistics.median(values),
            **{name: values[int((len(values) - 1) * q)]
               for name, q in [("p25", .25), ("p75", .75), ("p90", .9)]}}


def shape(value):
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(value)
    numbers = re.findall(r"-?\d+", str(value or ""))
    return tuple(map(int, numbers)) if len(numbers) == 3 else None


def predictions(launch, mode):
    if mode == "full":
        timing = launch.get("timing", {})
        completion = timing.get("completion_frontier", {})
        # Complete frontiers use cycles, incomplete ones lower_bound_cycles.
        terminal = completion.get("lower_bound_cycles", completion.get("cycles"))
        if terminal is None:
            terminal = timing.get("terminal_modeled_event_time")
        return {"gpc__cycles_elapsed.max": terminal,
                "sm__cycles_active.max": timing.get("sm_active_terminal_frontier", {}).get("cycles")}
    summary = launch.get("summary", {})
    return {"gpc__cycles_elapsed.max": summary.get("maximum_summarized_final_cycle"),
            "sm__cycles_active.max": summary.get("maximum_exact_final_cycle")}


def compare(path, mode, profiles, output):
    key = "launches" if mode == "full" else "fast_frontier"
    counts = collections.Counter()
    statuses = collections.Counter()
    joins = collections.Counter()
    shapes = collections.defaultdict(collections.Counter)
    missing_profiles = []
    errors = collections.defaultdict(list)
    equal_capture = collections.defaultdict(list)
    metric_status = collections.defaultdict(collections.Counter)
    with path.open() as stream:
        for line in stream:
            capture = json.loads(line)
            rel = capture.get("canonical_capture", capture.get("capture_name"))
            if not rel:
                raise ValueError("capture identity missing in replay JSONL")
            counts["captures"] += 1
            statuses[capture.get("status", "unspecified")] += 1
            profile = profiles / rel / "raw.csv"
            if not profile.is_file():
                missing_profiles.append(rel)
            ncu = parse_ncu_raw(profile)
            launches = capture.get(key, [])
            counts["launches"] += len(launches)
            local = collections.defaultdict(list)
            for join in join_ncu_rows(launches, ncu):
                joins[join["join_status"]] += 1
                li, ni = join.get("launch_index"), join.get("ncu_row_index")
                row = {"capture": rel, "mode": mode, **join}
                if li is None or ni is None:
                    output.write(json.dumps(row) + "\n")
                    continue
                launch, target = launches[li], ncu[ni]
                for left, right in [("grid", "grid_size"), ("block", "block_size")]:
                    a, b = shape(launch.get(left)), shape(target.get(right))
                    disposition = "unavailable" if a is None or b is None else "match" if a == b else "mismatch"
                    shapes[left][disposition] += 1
                    row[left + "_comparison"] = disposition
                row["ncu_profile"] = str(profile)
                row["ncu_metrics"] = target.get("metrics", {})
                row["model_predictions"] = predictions(launch, mode)
                row["raw_ape_distinct_endpoints"] = {}
                row["completion_status"] = launch.get("timing", {}).get("completion_frontier", {}).get("status")
                for metric, pred in row["model_predictions"].items():
                    observation = target.get("metrics", {}).get(metric, {})
                    truth = observation.get("value")
                    status = "compared_distinct_endpoints"
                    if pred is None:
                        status = "model_unavailable"
                    elif observation.get("unit") != "cycle":
                        status = "target_unit_missing_or_not_cycle"
                    elif truth is None or not math.isfinite(truth) or truth <= 0:
                        status = "target_unavailable_or_nonpositive"
                    else:
                        ape = abs(pred - truth) / truth
                        errors[metric].append(ape)
                        local[metric].append(ape)
                        row["raw_ape_distinct_endpoints"][metric] = ape
                    metric_status[metric][status] += 1
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
            for metric, vals in local.items():
                equal_capture[metric].append(statistics.median(vals))
    return {**counts, "statuses": statuses, "join_statuses": joins,
            "shape_comparisons": shapes, "missing_profiles": missing_profiles,
            "metric_status": metric_status,
            "raw_ape_per_launch": {k: distribution(v) for k, v in errors.items()},
            "raw_ape_equal_capture_median": {k: distribution(v) for k, v in equal_capture.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "post-replay only; NCU values never replay/model inputs",
               "endpoint_note": "full SM-active lower bound vs SM-active.max; full terminal frontier vs GPC elapsed.max; fast summarized and exact frontiers remain distinct endpoints",
               "sources": {"full": str(args.full), "fast": str(args.fast), "profiles": str(args.profiles)}}
    with (args.output / "launch_comparisons.jsonl").open("x") as output:
        for mode in ("full", "fast"):
            summary[mode] = compare(getattr(args, mode), mode, args.profiles, output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
