#!/usr/bin/env python3
"""Run the bounded SM86 NCU/NVBit/direct-simulator demo campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import collect_ncu
from tools.local_labels.ncu_csv import NcuCsvError, NcuMetricValue, load_ncu_csv
from tools.nvbit_tracer.runtime_env import NVBIT_CAPTURE_ENVIRONMENT_CONTROLS
from workloads.research_probes import capture_sm86_nvbit
from workloads.research_probes.semantic_replay_validation import (
    validate_replay_document,
)
from workloads.research_probes.sm86_ncu_demo_benchmark import validate_spec
from workloads.research_probes.validate_sm86_live_cohort import (
    _expected_nvbit_environment,
)


DEFAULT_SPEC = HERE / "benchmark_spec.json"
DEFAULT_METRICS = HERE / "metric_schema.json"
DEFAULT_PROFILE = HERE / "host_profile.json"
DEFAULT_GATE = HERE / "accuracy_gate.json"
STAGES = ("correctness", "ncu", "nvbit", "static", "simulator")
SIMULATOR_METRICS = (
    "smsp__inst_executed.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "dram__sectors_read.sum",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "gpc__cycles_elapsed.max",
)
CACHE_LOOKUP_IDENTITIES = (
    (
        "l1_global_read",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    ),
    (
        "l1_unified",
        "l1tex__t_sectors_lookup_hit.sum",
        "l1tex__t_sectors_lookup_miss.sum",
    ),
    (
        "l2_l1tex_source",
        "lts__t_sectors_srcunit_tex_lookup_hit.sum",
        "lts__t_sectors_srcunit_tex_lookup_miss.sum",
    ),
)
TIME_UNITS_TO_NS = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}
SIMULATOR_ENVIRONMENT_CONTROLS = frozenset(
    {
        "SIM_RUN_RETAIN_RAW_TRACE_PAYLOAD",
        "SIM_RUN_STREAM_TRACE_PAYLOAD",
        "SIM_RUN_PERFORMANCE_COUNTERS",
    }
)


class CampaignError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise CampaignError(f"missing or empty artifact: {path}")
    return {
        "path": _portable_path(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = validate_spec.load_object(path)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ensure_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise CampaignError(f"stage destination is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _git(*args: str) -> str:
    completed = collect_ncu._run_bounded(
        ["git", "-C", str(REPO_ROOT), *args],
        environment=dict(os.environ),
        timeout_seconds=30,
    )
    if completed.return_code != 0 or completed.timed_out:
        raise CampaignError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _clean_environment(*, simulator: bool = False) -> dict[str, str]:
    blocked = set(NVBIT_CAPTURE_ENVIRONMENT_CONTROLS) | {
        "BASH_ENV",
        "CUDA_VISIBLE_DEVICES",
        "NVBIT_CONFIG_FILE",
        "NVBIT_TRACE_FOLDER",
    }
    if simulator:
        blocked.update(SIMULATOR_ENVIRONMENT_CONTROLS)
    environment = {
        key: value for key, value in os.environ.items() if key not in blocked
    }
    environment.update(
        {
            "LANG": "C",
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": "/opt/cuda/lib64",
            "PATH": "/usr/local/cuda-13.2/bin:/opt/cuda/bin:/usr/bin:/bin",
        }
    )
    return environment


def _run(
    command: Sequence[str],
    *,
    timeout_seconds: int,
    environment: dict[str, str] | None = None,
) -> collect_ncu.BoundedProcessResult:
    return collect_ncu._run_bounded(
        command,
        environment=dict(os.environ) if environment is None else environment,
        timeout_seconds=timeout_seconds,
    )


def _stage_command(
    spec: Mapping[str, Any], workload: Mapping[str, Any], stage: str, output_root: Path
) -> list[str]:
    binary = str(workload["binary"])
    slug = str(workload["id"])
    if stage == "correctness":
        destination = output_root / "evidence" / "raw" / "correctness" / slug
        args = [str(destination) if token == "{output}" else str(token) for token in workload["correctness_args"]]
        return [binary, *args]
    if stage == "ncu":
        destination = output_root / "evidence" / "raw" / "ncu" / "rep-1" / slug
        ncu = spec["ncu"]
        return [
            ".venv/bin/python",
            "tools/local_labels/collect_ncu.py",
            "--ncu",
            str(ncu["tool"]),
            "--timeout",
            str(ncu["timeout_seconds"]),
            "--output",
            str(destination),
            "--metrics",
            ",".join(ncu["metrics"]),
            "--launch-skip",
            str(ncu["launch_skip"]),
            "--launch-count",
            str(ncu["launch_count"]),
            "--replay-mode",
            str(ncu["replay_mode"]),
            "--clock-control",
            str(ncu["clock_control"]),
            "--cache-control",
            str(ncu["cache_control"]),
            "--",
            binary,
            *map(str, workload["ncu_args"]),
        ]
    if stage == "nvbit":
        return [binary, *map(str, workload["trace_args"])]
    if stage == "static":
        destination = output_root / "evidence" / "raw" / "static" / slug
        control = destination / f"{slug}.control_bits.json"
        return [
            ".venv/bin/python",
            "tools/sass_control/emit_control_bits_sidecar.py",
            "--binary",
            binary,
            "--arch",
            "sm_86",
            "--out",
            str(control),
            "--toolkit-bindir",
            str(spec["static_control"]["toolkit_bindir"]),
        ]
    if stage == "simulator":
        trace = output_root / "evidence" / "raw" / "nvbit" / slug
        control = output_root / "evidence" / "raw" / "static" / slug / f"{slug}.control_bits.json"
        simulator = spec["simulator"]
        return [
            str(simulator["binary"]),
            "--trace",
            str(trace),
            "--control",
            str(control),
            "--arch-card",
            str(simulator["arch_card"]),
            "--pchase-card",
            str(simulator["pchase_card"]),
            "--mode",
            str(simulator["mode"]),
            "--cache-scope",
            str(simulator["cache_scope"]),
            "--ncu-shaped-metrics",
        ]
    raise CampaignError(f"unknown stage {stage!r}")


def _manifest_identity(path: Path) -> dict[str, object]:
    document = _load_json(path)
    return {
        "path": _portable_path(path),
        "sha256": _sha256(path),
        "schema_version": document.get("schema_version"),
    }


def _host_snapshot(profile: Mapping[str, Any]) -> dict[str, Any]:
    observed = validate_spec.validate_profile(profile, live_host=True)
    if observed is None:
        raise CampaignError("live host validation returned no observation")
    return observed


def _write_process_artifacts(destination: Path, command: Sequence[str], result: collect_ncu.BoundedProcessResult) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (destination / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    (destination / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")


def _require_success(stage: str, result: collect_ncu.BoundedProcessResult) -> None:
    if result.timed_out:
        raise CampaignError(f"{stage} timed out")
    if result.return_code != 0:
        raise CampaignError(f"{stage} exited {result.return_code}: {result.stderr.strip()}")
    if not result.process_group_cleanup:
        raise CampaignError(f"{stage} process-group cleanup was incomplete")


def _run_correctness(
    spec: Mapping[str, Any], workload: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    slug = str(workload["id"])
    destination = output_root / "evidence" / "raw" / "correctness" / slug
    _ensure_empty(destination)
    command = _stage_command(spec, workload, "correctness", output_root)
    result = _run(
        command,
        timeout_seconds=int(spec["correctness"]["timeout_seconds"]),
        environment=_clean_environment(),
    )
    _write_process_artifacts(destination, command, result)
    _require_success(f"{slug}.correctness", result)
    manifests = sorted(destination.rglob("manifest.json"))
    if len(manifests) != 1:
        raise CampaignError(f"{slug}: expected one correctness manifest, found {len(manifests)}")
    manifest = _load_json(manifests[0])
    expected_runner_slug = workload.get("runner_slug", slug)
    if (
        manifest.get("mode") != "correctness"
        or manifest.get("slug") != expected_runner_slug
        or manifest.get("binary_sha256") != workload["binary_sha256"]
    ):
        raise CampaignError(f"{slug}: correctness manifest identity mismatch")
    return {
        "command": command,
        "wall_seconds": result.wall_seconds,
        "return_code": result.return_code,
        "manifest": _manifest_identity(manifests[0]),
    }


def _wide_ncu_identity(path: Path) -> dict[str, object]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.lstrip().startswith('"')]
    rows = list(csv.reader(lines))
    if len(rows) < 3 or "Kernel Name" not in rows[0]:
        raise CampaignError(f"{path}: expected NCU 2026 wide CSV")
    header = rows[0]
    by_name = {name: index for index, name in enumerate(header)}
    data = [row for row in rows[2:] if len(row) == len(header) and row[by_name["Kernel Name"]].strip()]
    if len(data) != 1:
        raise CampaignError(f"{path}: expected exactly one launch row, found {len(data)}")
    row = data[0]
    required = ("ID", "Process Name", "Kernel Name", "Grid Size", "Block Size", "Device", "CC")
    if any(name not in by_name for name in required):
        raise CampaignError(f"{path}: NCU identity columns are incomplete")
    return {name: row[by_name[name]].strip() for name in required}


def _canonical_ncu_value(metric: str, value: NcuMetricValue) -> tuple[float, str]:
    raw = value.metric_value
    unit = value.metric_unit
    if metric == "gpu__time_duration.sum":
        if unit not in TIME_UNITS_TO_NS:
            raise CampaignError(f"{metric}: unsupported duration unit {unit!r}")
        return raw * TIME_UNITS_TO_NS[unit], "ns"
    expected = "%" if metric.endswith(".pct") else "cycle" if "cycles" in metric else "inst" if metric.startswith("smsp__inst") else "sector"
    if unit != expected:
        raise CampaignError(f"{metric}: expected unit {expected!r}, got {unit!r}")
    return raw, unit


def _parse_ncu_metrics(path: Path, requested: Sequence[str]) -> tuple[dict[str, object], dict[str, object]]:
    try:
        values = load_ncu_csv(path)
    except (OSError, NcuCsvError) as error:
        raise CampaignError(f"{path}: NCU CSV parse failed: {error}") from error
    selected: dict[str, object] = {}
    launch_names: set[tuple[int | None, str]] = set()
    for metric in requested:
        rows = [value for value in values if value.metric_name == metric]
        if len(rows) != 1:
            raise CampaignError(f"{path}: metric {metric!r} has {len(rows)} rows")
        row = rows[0]
        canonical, canonical_unit = _canonical_ncu_value(metric, row)
        if not math.isfinite(canonical) or canonical < 0:
            raise CampaignError(f"{path}: metric {metric!r} is invalid")
        if metric.endswith(".pct") and canonical > 100:
            raise CampaignError(f"{path}: metric {metric!r} exceeds 100 percent")
        selected[metric] = {
            "status": "measured",
            "value": canonical,
            "unit": canonical_unit,
            "raw_value": row.metric_value,
            "raw_unit": row.metric_unit,
        }
        launch_names.add((row.kernel_id, row.kernel_name))
    if len(launch_names) != 1:
        raise CampaignError(f"{path}: requested metrics span multiple launch identities")
    return selected, _wide_ncu_identity(path)


def _derive_ncu_metrics(
    collected: Mapping[str, Mapping[str, object]],
    metric_schema: Mapping[str, Any],
) -> dict[str, object]:
    derived: dict[str, object] = {}
    for row in metric_schema["metrics"]:
        metric = str(row["ncu_metric"])
        sources = row.get("ncu_sources")
        if sources is None:
            derived[metric] = collected[metric]
            continue
        if not isinstance(sources, list) or len(sources) != 2:
            raise CampaignError(f"{row['id']}: NCU hit-rate sources are malformed")
        hit_source, miss_source = map(str, sources)
        hit = float(collected[hit_source]["value"])
        miss = float(collected[miss_source]["value"])
        denominator = hit + miss
        value = None if denominator == 0 else 100.0 * hit / denominator
        derived[metric] = {
            "status": "unavailable_zero_denominator" if value is None else "measured",
            "value": value,
            "unit": "%",
            "formula": row["ncu_formula"],
            "hit_sectors": hit,
            "miss_sectors": miss,
            "sources": {
                hit_source: collected[hit_source],
                miss_source: collected[miss_source],
            },
        }
    return derived


def _run_ncu(
    spec: Mapping[str, Any],
    metric_schema: Mapping[str, Any],
    workload: Mapping[str, Any],
    output_root: Path,
    repeats: int,
) -> dict[str, Any]:
    slug = str(workload["id"])
    rows: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        destination = output_root / "evidence" / "raw" / "ncu" / f"rep-{repeat}" / slug
        _ensure_empty(destination)
        command = _stage_command(spec, workload, "ncu", output_root)
        output_index = command.index("--output") + 1
        command[output_index] = str(destination)
        timeout = int(spec["ncu"]["timeout_seconds"])
        result = _run(
            command,
            timeout_seconds=timeout + 60,
            environment=_clean_environment(),
        )
        _require_success(f"{slug}.ncu.rep-{repeat}", result)
        metadata = _load_json(destination / "metadata.json")
        collected, identity = _parse_ncu_metrics(
            destination / "ncu.csv", spec["ncu"]["metrics"]
        )
        metrics = _derive_ncu_metrics(collected, metric_schema)
        if identity["Process Name"] != Path(str(workload["binary"])).name:
            raise CampaignError(f"{slug}: NCU process name differs from workload binary")
        if identity["Device"] != "0" or identity["CC"] != "8.6":
            raise CampaignError(f"{slug}: NCU device identity mismatch")
        capture_wall = metadata.get("capture_wall_seconds")
        if not isinstance(capture_wall, (int, float)) or not math.isfinite(capture_wall) or capture_wall <= 0:
            raise CampaignError(f"{slug}: NCU capture wall time is missing")
        normalized = {
            "schema_version": 1,
            "repeat": repeat,
            "identity": identity,
            "metrics": metrics,
            "collection_metrics": collected,
            "ncu_profile_wall_seconds": float(capture_wall),
            "collector_wall_seconds": result.wall_seconds,
            "report": _artifact(destination / "ncu-report.ncu-rep"),
            "csv": _artifact(destination / "ncu.csv"),
        }
        _write_json(destination / "normalized.json", normalized)
        rows.append(normalized)
    identities = {
        json.dumps(row["identity"], sort_keys=True, separators=(",", ":"))
        for row in rows
    }
    if len(identities) != 1:
        raise CampaignError(f"{slug}: NCU launch identity changed across repeats")
    return {"repeats": rows, "repeat_count": repeats}


def _run_nvbit(
    spec: Mapping[str, Any], workload: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    slug = str(workload["id"])
    command = _stage_command(spec, workload, "nvbit", output_root)
    destination = output_root / "evidence" / "raw" / "nvbit" / slug
    logical = _expected_nvbit_environment(
        {"id": slug}, str(destination), root=REPO_ROOT
    )
    started_ns = time.perf_counter_ns()
    capture_sm86_nvbit.capture_argv(
        slug,
        command,
        logical,
        output_root,
        int(spec["nvbit"]["timeout_seconds"]),
    )
    wall_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
    for sidecar in (
        "unsupported_memory_path.json",
        "unmapped_memory_range.json",
        "cross_allocation_lane_span.json",
    ):
        path = destination / "extra_info" / sidecar
        value = json.loads(path.read_text(encoding="utf-8"))
        if value != []:
            raise CampaignError(f"{slug}: {sidecar} must be an empty list")
    environment = _load_json(destination / "environment.json")
    if environment.get("output_contract_passed") is not True:
        raise CampaignError(f"{slug}: NVBit output contract did not pass")
    return {
        "command": command,
        "wall_seconds": wall_seconds,
        "trace": _artifact(destination / "dynamic_trace.pb"),
        "stats": _artifact(destination / "stats.csv"),
        "environment": _artifact(destination / "environment.json"),
        "payload_file_count": environment.get("payload_file_count"),
        "sidecars_empty": True,
    }


def _run_static(
    spec: Mapping[str, Any], workload: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    slug = str(workload["id"])
    destination = output_root / "evidence" / "raw" / "static" / slug
    _ensure_empty(destination)
    command = _stage_command(spec, workload, "static", output_root)
    result = _run(
        command,
        timeout_seconds=int(spec["static_control"]["timeout_seconds"]),
        environment=_clean_environment(simulator=True),
    )
    _write_process_artifacts(destination, command, result)
    _require_success(f"{slug}.static", result)
    control = destination / f"{slug}.control_bits.json"
    return {
        "command": command,
        "wall_seconds": result.wall_seconds,
        "control": _artifact(control),
    }


def _normalized_kernel_name(name: str) -> str:
    return re.sub(r"___[0-9]+$", "", name)


def _validate_simulator_document(
    document: Mapping[str, Any], trace: Path, slug: str
) -> dict[str, Any]:
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1 or not isinstance(launches[0], Mapping):
        raise CampaignError(f"{slug}: simulator must emit exactly one launch")
    rows, errors = validate_replay_document(document, trace, label=slug)
    if errors:
        raise CampaignError(f"{slug}: semantic replay failed: {'; '.join(errors[:3])}")
    launch = launches[0]
    projections = launch.get("metric_predictions")
    if not isinstance(projections, list) or len(projections) != len(SIMULATOR_METRICS):
        raise CampaignError(f"{slug}: simulator metric projection count differs")
    names = tuple(row.get("ncu_metric") for row in projections if isinstance(row, Mapping))
    if names != SIMULATOR_METRICS:
        raise CampaignError(f"{slug}: simulator metric projection order differs")
    if any(row.get("arch") != "sm_86" for row in projections):
        raise CampaignError(f"{slug}: simulator metric architecture differs")
    cache_counts = launch.get("cache_lookup_predictions")
    if not isinstance(cache_counts, list) or len(cache_counts) != len(
        CACHE_LOOKUP_IDENTITIES
    ):
        raise CampaignError(f"{slug}: simulator cache count projection count differs")
    identities = tuple(
        (
            row.get("scope"),
            row.get("ncu_hit_metric"),
            row.get("ncu_miss_metric"),
        )
        for row in cache_counts
        if isinstance(row, Mapping)
    )
    if identities != CACHE_LOOKUP_IDENTITIES:
        raise CampaignError(f"{slug}: simulator cache count projection identity differs")
    for row in cache_counts:
        observer = row.get("observer")
        timing_state = row.get("timing_state")
        if (
            row.get("status") != "predicted"
            or not isinstance(observer, Mapping)
            or not isinstance(timing_state, Mapping)
        ):
            raise CampaignError(f"{slug}: simulator cache count projection unavailable")
        hits = int(observer.get("hits", -1))
        misses = int(observer.get("misses", -1))
        lookups = int(observer.get("lookups", -1))
        resident = int(timing_state.get("resident_data_hits", -1))
        remaining = int(timing_state.get("remaining_lookups", -1))
        reclassified = int(timing_state.get("observer_reclassified_lookups", -1))
        if (
            min(hits, misses, lookups, resident, remaining, reclassified) < 0
            or hits + misses != lookups
            or resident + remaining != lookups
            or resident + reclassified != hits
        ):
            raise CampaignError(f"{slug}: simulator cache count conservation differs")
    return {
        "launch": {
            "kernel_name": launch.get("kernel_name"),
            "grid": launch.get("grid"),
            "block": launch.get("block"),
            "device_id": launch.get("device_id"),
            "stream_id": launch.get("stream_id"),
            "kernel_id": launch.get("kernel_id"),
        },
        "semantic": rows,
        "metric_count": len(projections),
        "cache_lookup_count": len(cache_counts),
        "completion_frontier": launch.get("timing", {}).get("completion_frontier"),
    }


def _cross_check_ncu_simulator(output_root: Path, workload: Mapping[str, Any], sim: Mapping[str, Any]) -> None:
    slug = str(workload["id"])
    ncu_path = output_root / "evidence" / "raw" / "ncu" / "rep-1" / slug / "normalized.json"
    if not ncu_path.is_file():
        return
    ncu = _load_json(ncu_path)["identity"]
    launch = sim["launch"]
    if _normalized_kernel_name(str(launch["kernel_name"])) != ncu["Kernel Name"]:
        raise CampaignError(f"{slug}: NCU and simulator kernel names differ")
    if str(tuple(launch["grid"])) != ncu["Grid Size"] or str(tuple(launch["block"])) != ncu["Block Size"]:
        raise CampaignError(f"{slug}: NCU and simulator launch geometry differ")
    if launch["device_id"] != int(ncu["Device"]):
        raise CampaignError(f"{slug}: NCU and simulator device indices differ")


def _run_simulator(
    spec: Mapping[str, Any],
    workload: Mapping[str, Any],
    output_root: Path,
    repeats: int,
) -> dict[str, Any]:
    slug = str(workload["id"])
    destination = output_root / "evidence" / "raw" / "simulator" / slug
    _ensure_empty(destination)
    command = _stage_command(spec, workload, "simulator", output_root)
    simulator_binary = REPO_ROOT / str(spec["simulator"]["binary"])
    if not simulator_binary.is_file():
        raise CampaignError(f"simulator binary is missing: {simulator_binary}")
    outputs: list[Path] = []
    walls: list[float] = []
    observations: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        result = _run(
            command,
            timeout_seconds=int(spec["simulator"]["timeout_seconds"]),
            environment=_clean_environment(simulator=True),
        )
        _require_success(f"{slug}.simulator.repeat-{repeat}", result)
        output = destination / f"replay-{repeat}.json"
        stderr = destination / f"replay-{repeat}.stderr.txt"
        output.write_text(result.stdout, encoding="utf-8")
        stderr.write_text(result.stderr, encoding="utf-8")
        try:
            document = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise CampaignError(f"{slug}: simulator output is not JSON: {error}") from error
        if not isinstance(document, Mapping):
            raise CampaignError(f"{slug}: simulator output root is not an object")
        observations.append(_validate_simulator_document(document, Path(command[2]), slug))
        outputs.append(output)
        walls.append(result.wall_seconds)
    if any(path.read_bytes() != outputs[0].read_bytes() for path in outputs[1:]):
        raise CampaignError(f"{slug}: simulator repeats are not byte-identical")
    if any(observation != observations[0] for observation in observations[1:]):
        raise CampaignError(f"{slug}: simulator repeat observations differ")
    _cross_check_ncu_simulator(output_root, workload, observations[0])
    (destination / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    wall_mean = statistics.fmean(walls)
    wall_relative_range = (
        0.0 if wall_mean == 0 else (max(walls) - min(walls)) / wall_mean * 100.0
    )
    return {
        "command": command,
        "simulator_binary": _artifact(simulator_binary),
        "outputs": [_artifact(path) for path in outputs],
        "repeat_wall_seconds": walls,
        "repeat_count": repeats,
        "simulator_wall_seconds_median": statistics.median(walls),
        "simulator_wall_seconds_mean": wall_mean,
        "simulator_wall_seconds_minimum": min(walls),
        "simulator_wall_seconds_maximum": max(walls),
        "simulator_wall_relative_range_pct": wall_relative_range,
        "deterministic_repeats": True,
        "observation": observations[0],
    }


def _execute_stage(
    spec: Mapping[str, Any],
    metric_schema: Mapping[str, Any],
    profile: Mapping[str, Any],
    workload: Mapping[str, Any],
    stage: str,
    output_root: Path,
    ncu_repeats: int,
    simulator_repeats: int,
) -> dict[str, Any]:
    before = _host_snapshot(profile)
    started = _now()
    try:
        if stage == "correctness":
            result = _run_correctness(spec, workload, output_root)
        elif stage == "ncu":
            result = _run_ncu(
                spec, metric_schema, workload, output_root, ncu_repeats
            )
        elif stage == "nvbit":
            result = _run_nvbit(spec, workload, output_root)
        elif stage == "static":
            result = _run_static(spec, workload, output_root)
        elif stage == "simulator":
            result = _run_simulator(spec, workload, output_root, simulator_repeats)
        else:
            raise CampaignError(f"unknown stage {stage!r}")
    finally:
        after = _host_snapshot(profile)
    return {
        "stage": stage,
        "workload_id": workload["id"],
        "started_at": started,
        "finished_at": _now(),
        "host_pre": before,
        "host_post": after,
        "result": result,
    }


def _selected_workloads(
    spec: Mapping[str, Any], smoke: bool, requested: Sequence[str]
) -> list[Mapping[str, Any]]:
    rows = list(spec["workloads"])
    if smoke:
        rows = [row for row in rows if row["stage"] == "smoke"]
    if requested:
        wanted = set(requested)
        known = {str(row["id"]) for row in rows}
        missing = wanted - known
        if missing:
            raise CampaignError(f"unknown or out-of-scope workload ids: {sorted(missing)}")
        rows = [row for row in rows if row["id"] in wanted]
    if not rows:
        raise CampaignError("no workloads selected")
    return rows


def _new_manifest(
    spec_path: Path,
    metrics_path: Path,
    profile_path: Path,
    gate_path: Path,
    timing_crosswalk_path: Path | None,
    spec: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_id": spec["benchmark_id"],
        "created_at": _now(),
        "output_root": _portable_path(output_root),
        "repository": {
            "revision": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
        },
        "inputs": {
            "spec": _manifest_identity(spec_path),
            "metrics": _manifest_identity(metrics_path),
            "profile": _manifest_identity(profile_path),
            "gate": _manifest_identity(gate_path),
            **(
                {"timing_crosswalk": _manifest_identity(timing_crosswalk_path)}
                if timing_crosswalk_path is not None
                else {}
            ),
        },
        "records": [],
        "scientific_boundary": "NCU values are retained only as comparison targets and never passed to simulator commands or profile inputs.",
    }


def normalize_manifest_paths(manifest_path: Path) -> None:
    """Rewrite only artifact path fields to repository-relative spellings."""

    manifest = _load_json(manifest_path)

    def normalize(value: object, key: str | None = None) -> object:
        if isinstance(value, dict):
            return {name: normalize(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if key in {"path", "output_root"} and isinstance(value, str):
            return _portable_path(Path(value))
        return value

    _write_json(manifest_path, normalize(manifest))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "all", *STAGES), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--ncu-repeats", type=int)
    parser.add_argument("--simulator-repeats", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        os.chdir(REPO_ROOT)
        spec_path = args.spec.resolve()
        metrics_path = args.metrics.resolve()
        profile_path = args.profile.resolve()
        gate_path = args.gate.resolve()
        output_root = args.output.resolve()
        spec = _load_json(spec_path)
        metrics = _load_json(metrics_path)
        profile = _load_json(profile_path)
        gate = _load_json(gate_path)
        timing_crosswalk_path = (
            None
            if spec.get("timing_crosswalk") is None
            else (REPO_ROOT / str(spec["timing_crosswalk"])).resolve()
        )
        if timing_crosswalk_path is not None:
            validate_spec.validate_timing_crosswalk(_load_json(timing_crosswalk_path))
        validate_spec.validate_spec(spec, metrics, profile)
        validate_spec.validate_accuracy_gate(gate)
        workloads = _selected_workloads(spec, args.smoke, args.workload)
        ncu_repeats = args.ncu_repeats
        if ncu_repeats is None:
            ncu_repeats = int(spec["ncu"]["repeats_smoke" if args.smoke else "repeats_final"])
        if not 1 <= ncu_repeats <= int(spec["ncu"]["repeats_final"]):
            raise CampaignError("--ncu-repeats is outside the frozen 1..final-repeat range")
        simulator_repeats = args.simulator_repeats
        if simulator_repeats is None:
            simulator_repeats = int(
                spec["simulator"]["repeats_smoke" if args.smoke else "repeats_final"]
            )
        if not 1 <= simulator_repeats <= int(spec["simulator"]["repeats_final"]):
            raise CampaignError(
                "--simulator-repeats is outside the frozen 1..final-repeat range"
            )
        stages = STAGES if args.stage == "all" else () if args.stage == "plan" else (args.stage,)
        plan = {
            "benchmark_id": spec["benchmark_id"],
            "workloads": [row["id"] for row in workloads],
            "stages": list(stages or STAGES),
            "ncu_repeats": ncu_repeats,
            "simulator_repeats": simulator_repeats,
            "commands": {
                row["id"]: {
                    stage: _stage_command(spec, row, stage, output_root)
                    for stage in STAGES
                }
                for row in workloads
            },
        }
        if args.dry_run or args.stage == "plan":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "campaign_manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            expected_inputs = {
                "spec": spec_path,
                "metrics": metrics_path,
                "profile": profile_path,
                "gate": gate_path,
                **(
                    {"timing_crosswalk": timing_crosswalk_path}
                    if timing_crosswalk_path is not None
                    else {}
                ),
            }
            for name, path in expected_inputs.items():
                if manifest.get("inputs", {}).get(name, {}).get("sha256") != _sha256(path):
                    raise CampaignError(
                        f"existing campaign manifest is bound to another {name} input"
                    )
        else:
            manifest = _new_manifest(
                spec_path,
                metrics_path,
                profile_path,
                gate_path,
                timing_crosswalk_path,
                spec,
                output_root,
            )
        for stage in stages:
            for workload in workloads:
                record = _execute_stage(
                    spec,
                    metrics,
                    profile,
                    workload,
                    stage,
                    output_root,
                    ncu_repeats,
                    simulator_repeats,
                )
                manifest["records"].append(record)
                _write_json(manifest_path, manifest)
                print(f"completed {stage}: {workload['id']}", flush=True)
        print(manifest_path)
        return 0
    except (
        CampaignError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        validate_spec.ValidationError,
    ) as error:
        print(f"sm86_ncu_demo_campaign: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
