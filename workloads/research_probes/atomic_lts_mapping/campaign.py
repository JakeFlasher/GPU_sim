#!/usr/bin/env python3
"""Capture exact GA102 address-to-LTS equivalence classes with weighted RED probes."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SERVICE = HERE.parent / "atomic_multism_service"
DRIVER = SERVICE / "atomic_multism_service_driver"
DRIVER_SOURCE = SERVICE / "atomic_multism_service_driver.cpp"
CUBIN = SERVICE / "atomic_multism_service_trace_red32.cubin"
FAST_DRIVER = HERE / "atomic_lts_mapping_driver"
FAST_DRIVER_SOURCE = HERE / "atomic_lts_mapping_driver.cpp"
FAST_CUBIN = HERE / "atomic_lts_mapping.cubin"
DEFAULT_OUTPUT = REPO_ROOT / "results/captures/sm86_atomic_lts_mapping_20260820_v1"

BLOCKS = 168
GENERATIONS = 64
REGION_STRIDE_BYTES = 128
FIXED_DATA_BYTES = 32 * 1024 * 1024
ANCHOR_BLOCKS = 105
CANDIDATE_BLOCKS = (1, 2, 4, 8, 16, 32)
GROUP_BLOCK_COUNTS = (ANCHOR_BLOCKS, *CANDIDATE_BLOCKS)
DISCOVERY_REGIONS = 256
EXPECTED_LTS_SLICES = 48


@dataclass(frozen=True, slots=True)
class Phase:
    key: str
    allocation_prefix_bytes: int
    classify_start: int
    classify_stop: int
    cohort: str


PHASES = (
    Phase("train", 0, 0, 4096, "fit"),
    Phase("transfer_4m", 4 * 1024 * 1024, 0, 1024, "strict_transfer"),
    Phase("transfer_12m", 12 * 1024 * 1024, 2048, 3072, "strict_transfer"),
)


@dataclass(frozen=True, slots=True)
class Query:
    anchor: int
    candidates: tuple[int, int, int, int, int, int]

    @property
    def regions(self) -> tuple[int, ...]:
        return (self.anchor, *self.candidates)


def load_service_campaign():
    path = SERVICE / "campaign.py"
    spec = importlib.util.spec_from_file_location("lts_mapping_service_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SERVICE_CAMPAIGN = load_service_campaign()
BASE = SERVICE_CAMPAIGN.BASE


def phase_by_key(key: str) -> Phase:
    for phase in PHASES:
        if phase.key == key:
            return phase
    raise RuntimeError(f"unknown phase {key!r}")


def chunks(values: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def discovery_queries() -> list[Query]:
    queries: list[Query] = []
    for anchor in range(DISCOVERY_REGIONS - 1):
        candidates = list(range(anchor + 1, DISCOVERY_REGIONS))
        for group in chunks(candidates, len(CANDIDATE_BLOCKS)):
            if len(group) < len(CANDIDATE_BLOCKS):
                padding = [
                    value
                    for value in range(DISCOVERY_REGIONS)
                    if value != anchor and value not in group
                ]
                group.extend(padding[: len(CANDIDATE_BLOCKS) - len(group)])
            queries.append(Query(anchor, tuple(group)))
    expected_pairs = DISCOVERY_REGIONS * (DISCOVERY_REGIONS - 1) // 2
    covered = {
        tuple(sorted((query.anchor, candidate)))
        for query in queries
        for candidate in query.candidates
    }
    if len(covered) != expected_pairs:
        raise RuntimeError(
            f"discovery query set covers {len(covered)} of {expected_pairs} pairs"
        )
    return queries


def classification_queries(
    representatives: Sequence[int], start: int, stop: int
) -> list[Query]:
    return classification_queries_for_regions(
        representatives, range(start, stop)
    )


def classification_queries_for_regions(
    representatives: Sequence[int], regions: Iterable[int]
) -> list[Query]:
    if len(representatives) != EXPECTED_LTS_SLICES:
        raise RuntimeError(
            f"expected {EXPECTED_LTS_SLICES} representatives, got {len(representatives)}"
        )
    representative_set = set(representatives)
    groups = list(chunks(list(representatives), len(CANDIDATE_BLOCKS)))
    if len(groups) != 8 or any(len(group) != 6 for group in groups):
        raise RuntimeError("representatives must form eight exact six-way groups")
    return [
        Query(region, tuple(group))
        for region in regions
        if region not in representative_set
        for group in groups
    ]


def wide_regions() -> list[int]:
    maximum = FIXED_DATA_BYTES // REGION_STRIDE_BYTES
    selected: set[int] = set()
    for bit in range(12, 18):
        boundary = 1 << bit
        selected.update(
            region
            for region in range(boundary - 128, boundary + 129)
            if 4096 <= region < maximum
        )
    span = maximum - 4096
    selected.update(4096 + (index * 104729) % span for index in range(2048))
    state = 0x1357_9BDF
    while len(selected) < 4096:
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
        selected.add(4096 + state % span)
    if len(selected) != 4096:
        raise RuntimeError(f"wide-region matrix has {len(selected)} entries")
    return sorted(selected)


def dense_fit_regions() -> list[int]:
    return list(range(65_535))


def dense_holdout_regions() -> list[int]:
    lower = 65_535
    upper = FIXED_DATA_BYTES // REGION_STRIDE_BYTES
    selected: set[int] = set()
    for bit in (16, 17):
        boundary = 1 << bit
        selected.update(range(boundary - 2048, boundary + 2049))
    state = 0x2468_ACE1
    while len(selected) < 32_768:
        state = (1_103_515_245 * state + 12_345) & 0x7FFF_FFFF
        selected.add(lower + state % (upper - lower))
    selected = {region for region in selected if lower <= region < upper}
    while len(selected) < 32_768:
        state = (1_103_515_245 * state + 12_345) & 0x7FFF_FFFF
        selected.add(lower + state % (upper - lower))
    return sorted(selected)[:32_768]


def ensure_inputs() -> None:
    completed = subprocess.run(
        ["make", "atomic_multism_service_driver"],
        cwd=SERVICE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"driver build failed:\n{completed.stderr}")
    for path in (DRIVER_SOURCE, DRIVER, CUBIN):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required input is missing: {path}")
    if DRIVER.stat().st_mtime_ns <= DRIVER_SOURCE.stat().st_mtime_ns:
        raise RuntimeError("driver is not newer than its source after make")
    completed = subprocess.run(
        ["make", "all"],
        cwd=HERE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"encoded mapping probe build failed:\n{completed.stderr}")
    for path in (FAST_DRIVER_SOURCE, FAST_DRIVER, FAST_CUBIN):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required encoded mapping input is missing: {path}")
    if FAST_DRIVER.stat().st_mtime_ns <= FAST_DRIVER_SOURCE.stat().st_mtime_ns:
        raise RuntimeError("encoded mapping driver is not newer than its source")


def emit_manifest(output: Path) -> None:
    ensure_inputs()
    BASE.write_json(
        output / "metadata/matrix.json",
        {
            "schema_version": 1,
            "architecture": "sm_86",
            "gpu": "NVIDIA GeForce RTX 3090 Ti",
            "operation": "RED.E.ADD.STRONG.GPU",
            "metric": "lts__t_sectors_op_red.sum",
            "blocks": BLOCKS,
            "generations": GENERATIONS,
            "region_stride_bytes": REGION_STRIDE_BYTES,
            "fixed_data_bytes": FIXED_DATA_BYTES,
            "anchor_blocks": ANCHOR_BLOCKS,
            "candidate_blocks": list(CANDIDATE_BLOCKS),
            "discovery_regions": DISCOVERY_REGIONS,
            "expected_lts_slices": EXPECTED_LTS_SLICES,
            "wide_same_allocation_regions": len(wide_regions()),
            "dense_encoded_fit_regions": len(dense_fit_regions()),
            "dense_encoded_holdout_regions": len(dense_holdout_regions()),
            "phases": [asdict(phase) for phase in PHASES],
            "method": (
                "The anchor owns 105 of 168 CTAs; six candidates own binary "
                "weights 1,2,4,8,16,32. Because 105 exceeds their combined 63, "
                "the LTS maximum minus 105 encodes exactly which candidates "
                "share the anchor's slice."
            ),
            "encoded_method": (
                "A dedicated 49-CTA kernel assigns the anchor 48 RED generations "
                "and the 48 known-distinct representatives 0..47 generations. "
                "The LTS maximum divided by 32 minus 48 identifies the matching "
                "representative in one launch."
            ),
            "invariants": [
                "one warp per CTA and one RED instruction per generation",
                "one 128-byte cache-line step per region ID",
                "one profiler pass and one retained metric row per query launch",
                "all query launches use the same 32 MiB allocation within a phase",
                "discovery observes every pair among the first 256 cache lines",
                "classification compares every address against 48 distinct representatives",
                "wide holdout samples power-of-two boundaries and the full 32 MiB allocation",
                "encoded classification conserves 37,632 RED sectors per query",
            ],
        },
    )


def ncu_report_module():
    candidates = sorted(
        Path("/usr/local").glob("cuda-*/nsight-compute-*/extras/python"),
        reverse=True,
    )
    for directory in candidates:
        if (directory / "ncu_report.py").is_file():
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
            return importlib.import_module("ncu_report")
    raise RuntimeError("Nsight Compute Python report interface was not found")


def extract_metrics(report: Path, expected: int) -> list[dict[str, int]]:
    ncu_report = ncu_report_module()
    context = ncu_report.load_report(report)
    rows: list[dict[str, int]] = []
    for range_index in range(context.num_ranges()):
        current_range = context.range_by_idx(range_index)
        for action_index in range(current_range.num_actions()):
            action = current_range.action_by_idx(action_index)
            rows.append(
                {
                    "id": len(rows),
                    "maximum_sectors": int(
                        action.metric_by_name("lts__t_sectors_op_red.max").value()
                    ),
                    "total_sectors": int(
                        action.metric_by_name("lts__t_sectors_op_red.sum").value()
                    ),
                }
            )
    if len(rows) != expected:
        raise RuntimeError(f"{report}: expected {expected} actions, got {len(rows)}")
    return rows


def target(
    phase: Phase, queries: Sequence[Query], variant: str, batch_path: Path
) -> list[str]:
    if not queries:
        raise RuntimeError("empty weighted-query batch")
    return [
        str(DRIVER),
        str(CUBIN),
        str(BLOCKS),
        str(GENERATIONS),
        str(len(queries)),
        variant,
        "0",
        "0",
        "0",
        "7",
        "0",
        str(FIXED_DATA_BYTES),
        ",".join(map(str, queries[0].regions)),
        ",".join(map(str, GROUP_BLOCK_COUNTS)),
        str(phase.allocation_prefix_bytes),
        f"@{batch_path}",
        str(REGION_STRIDE_BYTES),
    ]


def run_batch(
    output: Path,
    phase: Phase,
    stage: str,
    queries: Sequence[Query],
    force: bool,
) -> tuple[dict[str, object], list[dict[str, int]]]:
    destination = output / "profiles" / phase.key / stage
    report_base = destination / "ncu-report"
    report = destination / "ncu-report.ncu-rep"
    metrics_path = destination / "metrics.json"
    measurement_path = destination / "measurement.json"
    queries_path = destination / "queries.json"
    batch_path = destination / "batch-regions.txt"
    if (
        not force
        and report.is_file()
        and metrics_path.is_file()
        and measurement_path.is_file()
        and queries_path.is_file()
    ):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["rows"]
        if len(metrics) == len(queries):
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
            return measurement, metrics

    ncu = os.environ.get("NCU", "ncu")
    variant = f"lts_{phase.key}_{stage}"
    destination.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(
        ";".join(",".join(map(str, query.regions)) for query in queries),
        encoding="utf-8",
    )
    completed = BASE.run_captured(
        [
            ncu,
            "--force-overwrite",
            "--export",
            str(report_base),
            "--profile-from-start",
            "on",
            "--launch-count",
            str(len(queries)),
            "--replay-mode",
            "application",
            "--clock-control",
            "base",
            "--cache-control",
            "none",
            "--metrics",
            "lts__t_sectors_op_red.sum",
            *target(phase, queries, variant, batch_path),
        ],
        destination,
        timeout=3600,
    )
    if not report.is_file():
        raise RuntimeError(f"NCU did not produce {report}")
    measurement = BASE.measurement_json_line(completed.stdout, variant)
    expected_regions = [list(query.regions) for query in queries]
    if (
        measurement.get("correct") is not True
        or measurement.get("batch_region_ids") != expected_regions
        or measurement.get("group_block_counts") != list(GROUP_BLOCK_COUNTS)
        or measurement.get("region_stride_bytes") != REGION_STRIDE_BYTES
        or measurement.get("kernel_launches") != len(queries)
        or measurement.get("allocation_prefix_bytes", 0)
        != phase.allocation_prefix_bytes
    ):
        raise RuntimeError(f"{phase.key}/{stage}: malformed batch measurement")
    rows = extract_metrics(report, len(queries))
    BASE.write_json(
        queries_path,
        {
            "schema_version": 1,
            "phase": asdict(phase),
            "stage": stage,
            "queries": [asdict(query) for query in queries],
        },
    )
    BASE.write_json(measurement_path, measurement)
    BASE.write_json(
        metrics_path,
        {"schema_version": 1, "metric": "lts__t_sectors_op_red.sum", "rows": rows},
    )
    return measurement, rows


def decode_mask(row: dict[str, int]) -> int:
    sectors_per_block = 32 * GENERATIONS
    expected_total = BLOCKS * sectors_per_block
    if row["total_sectors"] != expected_total:
        raise RuntimeError(f"query {row['id']}: total sectors {row['total_sectors']}")
    maximum = row["maximum_sectors"]
    if maximum % sectors_per_block != 0:
        raise RuntimeError(f"query {row['id']}: fractional block maximum {maximum}")
    mask = maximum // sectors_per_block - ANCHOR_BLOCKS
    if mask < 0 or mask > sum(CANDIDATE_BLOCKS):
        raise RuntimeError(f"query {row['id']}: invalid weighted mask {mask}")
    return mask


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def decode_discovery(
    output: Path,
    phase: Phase,
    measurement: dict[str, object],
    queries: Sequence[Query],
    rows: Sequence[dict[str, int]],
) -> list[int]:
    relation: dict[tuple[int, int], bool] = {}
    union = UnionFind(DISCOVERY_REGIONS)
    for query, row in zip(queries, rows, strict=True):
        mask = decode_mask(row)
        for bit, candidate in enumerate(query.candidates):
            pair = tuple(sorted((query.anchor, candidate)))
            collides = bool(mask & (1 << bit))
            prior = relation.setdefault(pair, collides)
            if prior != collides:
                raise RuntimeError(f"{phase.key}: inconsistent repeated pair {pair}")
            if collides:
                union.union(*pair)
    expected_pairs = DISCOVERY_REGIONS * (DISCOVERY_REGIONS - 1) // 2
    if len(relation) != expected_pairs:
        raise RuntimeError(f"{phase.key}: incomplete pair relation")
    for pair, collides in relation.items():
        if collides != (union.find(pair[0]) == union.find(pair[1])):
            raise RuntimeError(f"{phase.key}: non-transitive collision relation at {pair}")
    classes: dict[int, list[int]] = {}
    for region in range(DISCOVERY_REGIONS):
        classes.setdefault(union.find(region), []).append(region)
    ordered = sorted(classes.values(), key=lambda values: values[0])
    if len(ordered) != EXPECTED_LTS_SLICES:
        raise RuntimeError(
            f"{phase.key}: discovered {len(ordered)} classes, expected {EXPECTED_LTS_SLICES}"
        )
    representatives = [values[0] for values in ordered]
    BASE.write_json(
        output / "profiles" / phase.key / "representatives.json",
        {
            "schema_version": 1,
            "phase": asdict(phase),
            "data_device_address": measurement["data_device_address"],
            "classes": ordered,
            "representatives": representatives,
            "observed_pairs": len(relation),
        },
    )
    return representatives


def run_discovery(output: Path, phase: Phase, force: bool) -> list[int]:
    queries = discovery_queries()
    measurement, rows = run_batch(output, phase, "discovery", queries, force)
    representatives = decode_discovery(
        output, phase, measurement, queries, rows
    )
    print(
        f"{phase.key}: {len(queries)} discovery launches -> "
        f"{len(representatives)} LTS classes",
        flush=True,
    )
    return representatives


def load_representatives(output: Path, phase: Phase) -> list[int]:
    path = output / "profiles" / phase.key / "representatives.json"
    if not path.is_file():
        raise RuntimeError(f"run discovery first: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(value) for value in payload["representatives"]]


def run_classification(output: Path, phase: Phase, force: bool) -> None:
    representatives = load_representatives(output, phase)
    queries = classification_queries(
        representatives, phase.classify_start, phase.classify_stop
    )
    measurement, rows = run_batch(output, phase, "classification", queries, force)
    by_region: dict[int, int] = {
        representative: index
        for index, representative in enumerate(representatives)
        if phase.classify_start <= representative < phase.classify_stop
    }
    hits: dict[int, list[int]] = {}
    for query, row in zip(queries, rows, strict=True):
        mask = decode_mask(row)
        for bit, representative in enumerate(query.candidates):
            if mask & (1 << bit):
                hits.setdefault(query.anchor, []).append(representative)
    representative_to_class = {
        representative: index for index, representative in enumerate(representatives)
    }
    for region in range(phase.classify_start, phase.classify_stop):
        if region in representative_to_class:
            by_region[region] = representative_to_class[region]
            continue
        matches = hits.get(region, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"{phase.key}: region {region} matched {len(matches)} representatives"
            )
        by_region[region] = representative_to_class[matches[0]]
    base_address = int(measurement["data_device_address"])
    records = [
        {
            "region": region,
            "byte_address": base_address + region * REGION_STRIDE_BYTES,
            "line_address": (base_address >> 7) + region,
            "observed_class": by_region[region],
        }
        for region in range(phase.classify_start, phase.classify_stop)
    ]
    BASE.write_json(
        output / "profiles" / phase.key / "classification.json",
        {
            "schema_version": 1,
            "phase": asdict(phase),
            "data_device_address": base_address,
            "representatives": representatives,
            "query_launches": len(queries),
            "records": records,
        },
    )
    print(
        f"{phase.key}: {len(queries)} classification launches -> "
        f"{len(records)} exact address classes",
        flush=True,
    )


def run_wide_classification(output: Path, force: bool) -> None:
    phase = phase_by_key("train")
    representatives_path = output / "profiles" / phase.key / "representatives.json"
    representatives_payload = json.loads(
        representatives_path.read_text(encoding="utf-8")
    )
    representatives = [int(value) for value in representatives_payload["representatives"]]
    regions = wide_regions()
    queries = classification_queries_for_regions(representatives, regions)
    measurement, rows = run_batch(
        output, phase, "wide-classification", queries, force
    )
    discovery_base = int(representatives_payload["data_device_address"])
    base_address = int(measurement["data_device_address"])
    if base_address != discovery_base:
        raise RuntimeError(
            f"wide classification allocation base changed: {base_address} != {discovery_base}"
        )
    hits: dict[int, list[int]] = {}
    for query, row in zip(queries, rows, strict=True):
        mask = decode_mask(row)
        for bit, representative in enumerate(query.candidates):
            if mask & (1 << bit):
                hits.setdefault(query.anchor, []).append(representative)
    representative_to_class = {
        representative: index for index, representative in enumerate(representatives)
    }
    records = []
    for region in regions:
        matches = hits.get(region, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"wide classification region {region} matched {len(matches)} representatives"
            )
        records.append(
            {
                "region": region,
                "byte_address": base_address + region * REGION_STRIDE_BYTES,
                "line_address": (base_address >> 7) + region,
                "observed_class": representative_to_class[matches[0]],
            }
        )
    BASE.write_json(
        output / "profiles" / phase.key / "wide-classification.json",
        {
            "schema_version": 1,
            "cohort": "wide_same_allocation_holdout",
            "data_device_address": base_address,
            "representatives": representatives,
            "query_launches": len(queries),
            "records": records,
        },
    )
    print(
        f"train-wide: {len(queries)} classification launches -> "
        f"{len(records)} exact address classes",
        flush=True,
    )


def run_encoded_classification(
    output: Path, stage: str, regions: Sequence[int], force: bool
) -> None:
    phase = phase_by_key("train")
    representatives_payload = json.loads(
        (output / "profiles" / phase.key / "representatives.json").read_text(
            encoding="utf-8"
        )
    )
    representatives = [int(value) for value in representatives_payload["representatives"]]
    if len(representatives) != EXPECTED_LTS_SLICES:
        raise RuntimeError("encoded classification requires 48 representatives")
    destination = output / "profiles" / phase.key / stage
    report_base = destination / "ncu-report"
    report = destination / "ncu-report.ncu-rep"
    metrics_path = destination / "metrics.json"
    measurement_path = destination / "measurement.json"
    regions_path = destination / "query-regions.txt"
    result_path = output / "profiles" / phase.key / f"{stage}.json"
    if (
        not force
        and report.is_file()
        and metrics_path.is_file()
        and measurement_path.is_file()
        and regions_path.is_file()
        and result_path.is_file()
    ):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if len(payload.get("records", ())) == len(regions):
            print(f"train-{stage}: {len(regions)} encoded launches (existing)")
            return

    destination.mkdir(parents=True, exist_ok=True)
    regions_path.write_text(",".join(map(str, regions)), encoding="utf-8")
    ncu = os.environ.get("NCU", "ncu")
    variant = f"lts_train_{stage.replace('-', '_')}"
    completed = BASE.run_captured(
        [
            ncu,
            "--force-overwrite",
            "--export",
            str(report_base),
            "--profile-from-start",
            "on",
            "--kernel-name",
            "atomic_lts_mapping_probe",
            "--launch-count",
            str(len(regions)),
            "--replay-mode",
            "application",
            "--clock-control",
            "base",
            "--cache-control",
            "none",
            "--metrics",
            "lts__t_sectors_op_red.sum",
            str(FAST_DRIVER),
            str(FAST_CUBIN),
            str(regions_path),
            ",".join(map(str, representatives)),
            str(FIXED_DATA_BYTES),
            str(phase.allocation_prefix_bytes),
            str(REGION_STRIDE_BYTES),
            variant,
        ],
        destination,
        timeout=3600,
    )
    if not report.is_file():
        raise RuntimeError(f"NCU did not produce {report}")
    measurement = BASE.measurement_json_line(completed.stdout, variant)
    discovery_base = int(representatives_payload["data_device_address"])
    if (
        measurement.get("correct") is not True
        or measurement.get("query_count") != len(regions)
        or measurement.get("blocks_per_query") != 49
        or measurement.get("anchor_generations") != 48
        or measurement.get("operations") != 37_632
        or measurement.get("data_device_address") != discovery_base
    ):
        raise RuntimeError(f"{stage}: malformed encoded measurement")
    rows = extract_metrics(report, len(regions))
    records = []
    for region, row in zip(regions, rows, strict=True):
        if row["total_sectors"] != 37_632:
            raise RuntimeError(
                f"{stage} region {region}: total sectors {row['total_sectors']}"
            )
        if row["maximum_sectors"] % 32 != 0:
            raise RuntimeError(
                f"{stage} region {region}: fractional generation maximum"
            )
        observed_class = row["maximum_sectors"] // 32 - 48
        if not 0 <= observed_class < EXPECTED_LTS_SLICES:
            raise RuntimeError(
                f"{stage} region {region}: invalid class {observed_class}"
            )
        records.append(
            {
                "region": region,
                "byte_address": discovery_base + region * REGION_STRIDE_BYTES,
                "line_address": (discovery_base >> 7) + region,
                "observed_class": observed_class,
            }
        )
    BASE.write_json(measurement_path, measurement)
    BASE.write_json(
        metrics_path,
        {"schema_version": 1, "metric": "lts__t_sectors_op_red.sum", "rows": rows},
    )
    BASE.write_json(
        result_path,
        {
            "schema_version": 1,
            "cohort": stage,
            "data_device_address": discovery_base,
            "representatives": representatives,
            "query_launches": len(regions),
            "records": records,
        },
    )
    print(
        f"train-{stage}: {len(regions)} encoded launches -> "
        f"{len(records)} exact address classes",
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        choices=(
            "manifest",
            "discover",
            "classify",
            "wide",
            "encoded-fit",
            "encoded-holdout",
            "all",
        ),
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--phase", choices=tuple(phase.key for phase in PHASES) + ("all",), default="all"
    )
    result.add_argument("--force", action="store_true")
    return result


def selected_phases(selector: str) -> Sequence[Phase]:
    return PHASES if selector == "all" else (phase_by_key(selector),)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output_root.resolve()
    try:
        emit_manifest(output)
        if args.action == "manifest":
            print(output / "metadata/matrix.json")
            return 0
        if args.action == "wide":
            run_wide_classification(output, args.force)
            return 0
        if args.action == "encoded-fit":
            run_encoded_classification(
                output, "encoded-fit", dense_fit_regions(), args.force
            )
            return 0
        if args.action == "encoded-holdout":
            run_encoded_classification(
                output, "encoded-holdout", dense_holdout_regions(), args.force
            )
            return 0
        for phase in selected_phases(args.phase):
            if args.action in {"discover", "all"}:
                run_discovery(output, phase, args.force)
            if args.action in {"classify", "all"}:
                if args.action == "classify":
                    load_representatives(output, phase)
                run_classification(output, phase, args.force)
        if args.action == "all" and args.phase == "all":
            run_wide_classification(output, args.force)
            run_encoded_classification(
                output, "encoded-fit", dense_fit_regions(), args.force
            )
            run_encoded_classification(
                output, "encoded-holdout", dense_holdout_regions(), args.force
            )
    except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f"atomic LTS mapping campaign: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
