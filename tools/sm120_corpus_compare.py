#!/usr/bin/env python3
"""Post-replay, unit-aware comparison of whole-corpus native replay reports.

Reads one capture at a time from compressed or legacy JSONL. Its comparison
artifact is zstd-compressed. Nothing from NCU is sent to a simulator. An APE
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
from jsonl_io import (jsonl_compression_description, open_jsonl_reader,
                      open_jsonl_writer, resolve_jsonl_path)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None}
    return {"n": len(values), "median": statistics.median(values),
            **{name: values[int((len(values) - 1) * q)]
               for name, q in [("p25", .25), ("p75", .75), ("p90", .9)]}}


def shape(value):
    """Return a validated three-dimensional launch shape, or ``None``.

    NCU exports strings while simulator rows may contain arrays.  Malformed
    arrays/strings are unavailable evidence, not exceptions that abort a whole
    comparison or accidental partial integer conversions.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            return None
        try:
            if any(isinstance(item, bool) for item in value):
                return None
            result = tuple(int(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if all(item >= 0 for item in result) else None
    if not isinstance(value, str):
        return None
    numbers = re.findall(r"-?\d+", value)
    if len(numbers) != 3:
        return None
    try:
        result = tuple(int(item) for item in numbers)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(item >= 0 for item in result) else None


def _finite_number(value):
    """Keep JSON predictions finite; unavailable/non-finite means ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def predictions(launch, mode):
    if not isinstance(launch, dict):
        return {"gpc__cycles_elapsed.max": None, "sm__cycles_active.max": None}
    if mode == "full":
        timing = launch.get("timing", {})
        if not isinstance(timing, dict):
            timing = {}
        completion = timing.get("completion_frontier", {})
        if not isinstance(completion, dict):
            completion = {}
        # Complete frontiers use cycles, incomplete ones lower_bound_cycles.
        terminal = completion.get("lower_bound_cycles", completion.get("cycles"))
        if terminal is None:
            terminal = timing.get("terminal_modeled_event_time")
        sm_frontier = timing.get("sm_active_terminal_frontier", {})
        if not isinstance(sm_frontier, dict):
            sm_frontier = {}
        return {"gpc__cycles_elapsed.max": _finite_number(terminal),
                "sm__cycles_active.max": _finite_number(sm_frontier.get("cycles"))}
    summary = launch.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    return {"gpc__cycles_elapsed.max": _finite_number(summary.get("maximum_summarized_final_cycle")),
            "sm__cycles_active.max": _finite_number(summary.get("maximum_exact_final_cycle"))}


def compare(path, mode, profiles, output):
    # Report renderers must aggregate metrics with no simulator prediction in
    # either mode into one unavailable-by-family section: never score N/A as
    # zero or repeat that metric's NCU distribution.
    key = "launches" if mode == "full" else "fast_frontier"
    counts = collections.Counter()
    statuses = collections.Counter()
    joins = collections.Counter()
    shapes = collections.defaultdict(collections.Counter)
    missing_profiles = []
    errors = collections.defaultdict(list)
    equal_capture = collections.defaultdict(list)
    metric_status = collections.defaultdict(collections.Counter)
    with open_jsonl_reader(path) as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                capture = json.loads(line)
                if not isinstance(capture, dict):
                    raise ValueError("replay JSONL row is not an object")
                rel = capture.get("canonical_capture", capture.get("capture_name"))
                if not rel:
                    raise ValueError("capture identity missing in replay JSONL")
                relative_path = Path(str(rel))
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError("capture identity is not a relative path")
                rel = relative_path.as_posix()
            except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
                # A malformed source row is source corruption, not a valid
                # comparison row.  Abort so the compressed output is not
                # published as a misleading partial success.
                raise ValueError(f"{path}:{line_number}: malformed replay row: {error}") from error
            counts["captures"] += 1
            statuses[capture.get("status", "unspecified")] += 1
            profile = profiles / rel / "raw.csv"
            if not profile.is_file():
                missing_profiles.append(rel)
            ncu = parse_ncu_raw(profile)
            launches = capture.get(key, [])
            if not isinstance(launches, list) or any(not isinstance(launch, dict) for launch in launches):
                raise ValueError(f"{path}:{line_number}: {key} must be an array of objects")
            counts["launches"] += len(launches)
            local = collections.defaultdict(list)
            for join in join_ncu_rows(launches, ncu):
                joins[join["join_status"]] += 1
                li, ni = join.get("launch_index"), join.get("ncu_row_index")
                row = {"capture": rel, "mode": mode, **join}
                if li is None or ni is None:
                    output.write(json.dumps(row, allow_nan=False) + "\n")
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
                timing = launch.get("timing", {})
                completion = timing.get("completion_frontier", {}) if isinstance(timing, dict) else {}
                row["completion_status"] = completion.get("status") if isinstance(completion, dict) else None
                for metric, pred in row["model_predictions"].items():
                    observation = target.get("metrics", {}).get(metric, {})
                    truth = observation.get("value")
                    status = "compared_distinct_endpoints"
                    if pred is None:
                        status = "model_unavailable"
                    elif not isinstance(pred, (int, float)) or isinstance(pred, bool) or not math.isfinite(pred):
                        status = "model_nonfinite_or_non_numeric"
                    elif observation.get("unit") != "cycle":
                        status = "target_unit_missing_or_not_cycle"
                    elif truth is None or not isinstance(truth, (int, float)) or isinstance(truth, bool) or not math.isfinite(truth) or truth <= 0:
                        status = "target_unavailable_or_nonpositive"
                    else:
                        ape = abs(pred - truth) / truth
                        errors[metric].append(ape)
                        local[metric].append(ape)
                        row["raw_ape_distinct_endpoints"][metric] = ape
                    metric_status[metric][status] += 1
                output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            for metric, vals in local.items():
                equal_capture[metric].append(statistics.median(vals))
    return {**counts,
            "source_reader_error": None,
            "statuses": statuses, "join_statuses": joins,
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
    parser.add_argument("--capture-root", type=Path,
                        default=Path("."),
                        help="capture/profiles root; output is forbidden anywhere beneath it")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing comparison artifact explicitly")
    args = parser.parse_args()
    args.capture_root = args.capture_root.resolve()
    args.output = args.output.resolve()
    if args.output.is_relative_to(args.capture_root):
        parser.error("comparison output must be outside the capture/profiles corpus root")
    if args.output.exists() and not args.output.is_dir():
        parser.error("comparison output path is not a directory")
    existing_artifacts = {
        args.output / "launch_comparisons.jsonl",
        args.output / "launch_comparisons.jsonl.zst",
        args.output / "summary.json",
    }
    if not args.overwrite and any(path.exists() for path in existing_artifacts):
        parser.error("comparison artifacts already exist; pass --overwrite to replace them")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "post-replay only; NCU values never replay/model inputs",
               "endpoint_note": "full SM-active lower bound vs SM-active.max; full terminal frontier vs GPC elapsed.max; fast summarized and exact frontiers remain distinct endpoints",
               "sources": {"full": str(resolve_jsonl_path(args.full)),
                           "fast": str(resolve_jsonl_path(args.fast)),
                           "profiles": str(args.profiles)},
               "launch_comparisons_artifact": str(args.output / "launch_comparisons.jsonl.zst"),
               "jsonl_compression": jsonl_compression_description(),
               "overwrite_requested": bool(args.overwrite)}
    with open_jsonl_writer(args.output / "launch_comparisons.jsonl") as output:
        for mode in ("full", "fast"):
            summary[mode] = compare(getattr(args, mode), mode, args.profiles, output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
