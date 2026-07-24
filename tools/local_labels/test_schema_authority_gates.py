"""Fail-closed gates for active and historical label-schema authority."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

import jsonschema
import pytest


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SCHEMAS = _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
_HISTORICAL = _SCHEMAS / "historical"
_HISTORICAL_DURATION = _HISTORICAL / "duration_admitted.v1.schema.json"
_HISTORICAL_CUPTI = _HISTORICAL / "cupti_activity_sidecar.v1.schema.json"
_HISTORICAL_SOL_DONE = (
    _HISTORICAL / "sol_workload_capture_chunk_done.v2.schema.json"
)
_HISTORICAL_MANIFEST = _HISTORICAL / "MANIFEST.sha256"
_ACTIVE_MANIFEST = _SCHEMAS / "measurement_manifest.schema.json"
_V5_SIDECAR = _SCHEMAS / "local_label_sidecar.schema.json"
_V5_ADMITTED = _SCHEMAS / "local_label_admitted.schema.json"
_METRIC_REGISTRY = _HERE / "metric_registry.json"

_OLD_DURATION_SCHEMA_BASENAME = "duration_admitted" + ".schema.json"
_HISTORICAL_DURATION_SHA256 = (
    "5a555d714cab25f1203f561e8027d589ea4329aabe6826337ae68cfbf2b3f62f"
)
_HISTORICAL_CUPTI_SHA256 = (
    "836c0db4f6f5a2a582a9b334be02c829b40a094f709259d8008f05f200faaeb4"
)
_HISTORICAL_SOL_DONE_SHA256 = (
    "5e9bbb94107541fa522e9070972e81916a7be984802493f414d0951a705f0f4c"
)
_HISTORICAL_MANIFEST_SHA256 = (
    "702ef1c2b07d25cc772e7a160a3ce430910b3403a8a19a579bc7e6c03b45479e"
)
_HISTORICAL_FILE_SET = {
    "README.md",
    "MANIFEST.sha256",
    "cupti_activity_sidecar.v1.schema.json",
    "duration_admitted.v1.schema.json",
    "sol_workload_capture_chunk_done.v2.schema.json",
}
_MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9._/-]+)$")
_SHA = "a" * 64


class HistoricalSchemaHashError(RuntimeError):
    """A frozen historical schema no longer matches its manifest."""


class HistoricalSchemaDispatchError(RuntimeError):
    """Evidence did not declare a supported historical schema version."""


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    assert isinstance(value, dict), f"expected JSON object at {path}"
    return value


def _parse_hash_manifest(path: Path) -> dict[str, str]:
    """Parse the closed three-schema historical manifest without normalization."""
    if path.is_symlink() or not path.is_file():
        raise HistoricalSchemaHashError(
            f"historical schema manifest is not a regular file: {path}"
        )
    payload = path.read_bytes()
    if not payload.endswith(b"\n") or b"\r" in payload:
        raise HistoricalSchemaHashError(
            f"historical schema manifest must use LF-terminated rows: {path}"
        )
    try:
        lines = payload[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError as exc:
        raise HistoricalSchemaHashError(
            f"historical schema manifest is not ASCII: {path}"
        ) from exc
    if len(lines) != 3:
        raise HistoricalSchemaHashError(
            f"historical schema manifest must contain exactly three entries: {path}"
        )

    entries: dict[str, str] = {}
    for line in lines:
        match = _MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise HistoricalSchemaHashError(
                f"malformed historical schema manifest entry: {line!r}"
            )
        expected_hash, relative = match.groups()
        parts = relative.split("/")
        candidate = PurePosixPath(relative)
        if (
            candidate.is_absolute()
            or "\\" in relative
            or any(part in ("", ".", "..") for part in parts)
            or any(ord(char) < 32 or ord(char) == 127 for char in relative)
        ):
            raise HistoricalSchemaHashError(
                f"unsafe historical schema manifest path: {relative!r}"
            )
        if relative in entries:
            raise HistoricalSchemaHashError(
                f"duplicate historical schema manifest path: {relative!r}"
            )
        entries[relative] = expected_hash

    if list(entries) != sorted(entries):
        raise HistoricalSchemaHashError(
            f"historical schema manifest entries are not bytewise sorted: {path}"
        )
    return entries


def verify_historical_schema_directory(schema_dir: Path = _HISTORICAL) -> None:
    manifest = schema_dir / "MANIFEST.sha256"
    entries = _parse_hash_manifest(manifest)
    canonical_dir = schema_dir.resolve(strict=True)
    actual_schema_names: set[str] = set()
    for schema_path in schema_dir.rglob("*.schema.json"):
        if schema_path.is_symlink() or not schema_path.is_file():
            raise HistoricalSchemaHashError(
                f"historical schema target is not a regular file: {schema_path}"
            )
        try:
            schema_path.resolve(strict=True).relative_to(canonical_dir)
        except ValueError as exc:
            raise HistoricalSchemaHashError(
                f"historical schema target escapes its directory: {schema_path}"
            ) from exc
        actual_schema_names.add(schema_path.relative_to(schema_dir).as_posix())
    if actual_schema_names != set(entries):
        raise HistoricalSchemaHashError(
            "historical schema directory does not match its manifest: "
            f"manifest={sorted(entries)} actual={sorted(actual_schema_names)}"
        )

    for relative, expected_hash in entries.items():
        schema_path = schema_dir / relative
        if schema_path.is_symlink() or not schema_path.is_file():
            raise HistoricalSchemaHashError(
                f"historical schema manifest target is missing: {schema_path}"
            )
        actual_hash = hashlib.sha256(schema_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise HistoricalSchemaHashError(
                "historical schema hash mismatch: "
                f"expected={expected_hash} actual={actual_hash} path={schema_path}"
            )


_HISTORICAL_DISPATCH = {
    ("cupti_activity_sidecar", 1): _HISTORICAL_CUPTI,
    ("duration_admitted", 1): _HISTORICAL_DURATION,
    ("sol_workload_capture_chunk_done", 2): _HISTORICAL_SOL_DONE,
}


def resolve_historical_schema(family: str, declared_version: object) -> Path:
    if not isinstance(declared_version, int) or isinstance(declared_version, bool):
        raise HistoricalSchemaDispatchError(
            f"missing or invalid historical schema version: {declared_version!r}"
        )
    try:
        return _HISTORICAL_DISPATCH[(family, declared_version)]
    except KeyError as exc:
        raise HistoricalSchemaDispatchError(
            f"unsupported historical schema family/version: "
            f"{family!r}/{declared_version!r}"
        ) from exc


def _valid_duration_record() -> dict:
    return {
        "schema_version": 1,
        "slug": "synthetic_duration",
        "shape_point": "row_0",
        "arch": "sm_86",
        "binary_sha256": _SHA,
        "source_commit": "b" * 40,
        "kernel_signature_sha256": "c" * 64,
        "cupti_collector_binary_sha256": "d" * 64,
        "cupti_sidecar_sha256s": ["e" * 64],
        "duration_label_source": "cupti_dual_source",
        "n_reconciled_repeats": 5,
        "median_duration_ns": 1000.0,
        "robust_cv": 0.01,
        "stddev_duration_ns": 5.0,
        "mad_duration_ns": 4.0,
        "short_kernel": False,
        "gate_basis": "robust_cv",
        "nvml_covariate_summary": {
            "nvml_available": False,
            "n_repeats_with_nvml": 0,
            "temp_c_median": None,
            "power_mw_median": None,
            "sm_clock_mhz_median": None,
            "mem_clock_mhz_median": None,
        },
        "evidence_kind": "synthetic",
    }


def _duration_record_schema() -> dict:
    schema = _load_json(_HISTORICAL_DURATION)
    record_schema = copy.deepcopy(schema["$defs"]["duration_admitted_cell"])
    record_schema.setdefault("$defs", {})["nvml_covariate_summary"] = copy.deepcopy(
        schema["$defs"]["nvml_covariate_summary"]
    )
    return record_schema


def _valid_measurement_manifest() -> dict:
    return {
        "schema_version": 2,
        "slug": "synthetic_manifest",
        "mode": "timing",
        "warm_up_iters": 1,
        "repeat_iters": 5,
        "cv_pct": 0.5,
        "cuda_version": "13.2",
        "driver_version": "610.43.02",
        "gpu_sku": "rtx_3090_ti",
        "gpu_uuid": "synthetic-uuid",
        "sm_count": 82,
        "total_global_mem_bytes": 1,
        "device_name": "synthetic device",
        "compute_capability_major": 8,
        "compute_capability_minor": 6,
        "binary_sha256": _SHA,
        "compiler_flags": ["-gencode=arch=compute_86,code=sm_86"],
        "kb_capability_card": (
            "references/NVIDIA_Manuals/blackwell_compatibility/"
            "01_blackwell-architecture-compatibility.md"
        ),
        "timestamp_utc": "2026-07-11T00:00:00Z",
    }


def _valid_v5_sidecar() -> dict:
    return {
        "schema_version": 5,
        "device": "sm_86",
        "slug": "synthetic_label",
        "shape_point": "row_0",
        "repeat_index": 0,
        "launch_index": 0,
        "binary_sha256": _SHA,
        "profiler_family": "nsight_compute",
        "profiler_version": "ncu_2026.1.x",
        "cuda_version": "13.2",
        "driver_version": "610.43.02",
        "gpu_uuid_sha256": "f" * 64,
        "replay_mode": "kernel",
        "aggregation": "kernel",
        "warmup_count": 0,
        "cache_hit_rate_l2_sector": 0.5,
        "confidence_interval_l2_sector": {
            "lower": 0.49,
            "upper": 0.51,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        },
        "unit_tagged_metrics": [
            {
                "ncu_metric": "gpc__cycles_elapsed.max",
                "unit": "cycles",
                "value": 1000.0,
            }
        ],
    }


def test_g1_historical_manifest_and_contract_are_pinned() -> None:
    verify_historical_schema_directory()
    manifest_payload = _HISTORICAL_MANIFEST.read_bytes()
    assert len(manifest_payload) == 316
    assert hashlib.sha256(manifest_payload).hexdigest() == (
        _HISTORICAL_MANIFEST_SHA256
    )
    assert _parse_hash_manifest(_HISTORICAL_MANIFEST) == {
        _HISTORICAL_CUPTI.name: _HISTORICAL_CUPTI_SHA256,
        _HISTORICAL_DURATION.name: _HISTORICAL_DURATION_SHA256,
        _HISTORICAL_SOL_DONE.name: _HISTORICAL_SOL_DONE_SHA256,
    }
    assert {path.name for path in _HISTORICAL.iterdir()} == _HISTORICAL_FILE_SET

    duration_schema = _load_json(_HISTORICAL_DURATION)
    assert duration_schema["$id"] == "simbench_bw/schemas/" + _OLD_DURATION_SCHEMA_BASENAME
    assert duration_schema["schema_version"] == 1
    assert duration_schema["properties"]["schema_version"]["const"] == 1
    source = duration_schema["$defs"]["duration_admitted_cell"]["properties"][
        "duration_label_source"
    ]
    assert source["const"] == "cupti_dual_source"

    cupti_schema = _load_json(_HISTORICAL_CUPTI)
    assert cupti_schema["$id"] == (
        "simbench_bw/schemas/" + "cupti_activity_" + "sidecar.schema.json"
    )
    assert cupti_schema["schema_version"] == 1
    assert cupti_schema["properties"]["schema_version"]["const"] == 1

    sol_done_schema = _load_json(_HISTORICAL_SOL_DONE)
    assert sol_done_schema["$id"] == (
        "simbench_bw/schemas/sol_workload_capture_chunk_done.schema.json"
    )
    assert sol_done_schema["properties"]["schema_version"]["const"] == 2
    assert sol_done_schema["properties"]["manifest_kind"]["const"] == (
        "sol_workload_capture_chunk_done"
    )


def test_g2_one_byte_mutation_fails_hash_check(tmp_path: Path) -> None:
    for index, schema_name in enumerate(
        (
            _HISTORICAL_CUPTI.name,
            _HISTORICAL_DURATION.name,
            _HISTORICAL_SOL_DONE.name,
        )
    ):
        staged = tmp_path / f"mutated-{index}"
        shutil.copytree(_HISTORICAL, staged)
        verify_historical_schema_directory(staged)
        schema_path = staged / schema_name
        mutated = bytearray(schema_path.read_bytes())
        mutated[0] ^= 1
        schema_path.write_bytes(mutated)
        with pytest.raises(HistoricalSchemaHashError, match="hash mismatch"):
            verify_historical_schema_directory(staged)

    valid_manifest = _HISTORICAL_MANIFEST.read_bytes()
    rows = valid_manifest.splitlines(keepends=True)
    malformed_payloads = (
        valid_manifest.rstrip(b"\n"),
        valid_manifest.replace(b"  ", b" ", 1),
        valid_manifest.replace(b"  ", b"\t", 1),
        valid_manifest.replace(b"c", b"C", 1),
        valid_manifest.replace(b"\n", b"\r\n"),
        rows[1] + rows[0] + rows[2],
        rows[0] + rows[0] + rows[2],
        (
            rows[0].split(b"  ", 1)[0]
            + b"  ../escape.schema.json\n"
            + rows[1]
            + rows[2]
        ),
    )
    for index, malformed in enumerate(malformed_payloads):
        staged = tmp_path / f"manifest-{index}"
        shutil.copytree(_HISTORICAL, staged)
        (staged / _HISTORICAL_MANIFEST.name).write_bytes(malformed)
        with pytest.raises(HistoricalSchemaHashError):
            verify_historical_schema_directory(staged)

    staged = tmp_path / "unmanifested-schema"
    shutil.copytree(_HISTORICAL, staged)
    (staged / "extra.schema.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(HistoricalSchemaHashError, match="does not match"):
        verify_historical_schema_directory(staged)

    staged = tmp_path / "symlinked-manifest"
    shutil.copytree(_HISTORICAL, staged)
    staged_manifest = staged / _HISTORICAL_MANIFEST.name
    staged_manifest.unlink()
    staged_manifest.symlink_to(_HISTORICAL_MANIFEST)
    with pytest.raises(HistoricalSchemaHashError, match="not a regular file"):
        verify_historical_schema_directory(staged)


@pytest.mark.parametrize("version", [None, 0, 2, True])
def test_g3_historical_v1_dispatch_fails_closed(version: object) -> None:
    for family in ("cupti_activity_sidecar", "duration_admitted"):
        with pytest.raises(HistoricalSchemaDispatchError):
            resolve_historical_schema(family, version)


@pytest.mark.parametrize("version", [None, 0, 1, 3, True])
def test_g3_historical_sol_done_v2_dispatch_fails_closed(version: object) -> None:
    with pytest.raises(HistoricalSchemaDispatchError):
        resolve_historical_schema("sol_workload_capture_chunk_done", version)


def test_g3_historical_dispatch_accepts_only_frozen_versions() -> None:
    assert resolve_historical_schema("cupti_activity_sidecar", 1) == _HISTORICAL_CUPTI
    assert resolve_historical_schema("duration_admitted", 1) == _HISTORICAL_DURATION
    assert resolve_historical_schema(
        "sol_workload_capture_chunk_done", 2
    ) == _HISTORICAL_SOL_DONE
    assert not any(
        family == "measurement_manifest" for family, _ in _HISTORICAL_DISPATCH
    )


def test_g4_frozen_v1_accepts_synthetic_cupti_record() -> None:
    jsonschema.validate(instance=_valid_duration_record(), schema=_duration_record_schema())


@pytest.mark.parametrize(
    "mutation",
    [
        {"duration_label_source": "ncu"},
        {"gpu__time_duration.sum": 1000.0},
    ],
)
def test_g4_frozen_v1_rejects_ncu_shaped_records(mutation: dict) -> None:
    record = _valid_duration_record()
    record.update(mutation)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=_duration_record_schema())


def test_g5_measurement_manifest_v2_remains_active() -> None:
    schema = _load_json(_ACTIVE_MANIFEST)
    assert schema["schema_version"] == 2
    assert schema["properties"]["schema_version"]["const"] == 2
    assert schema["properties"]["mode"]["enum"] == [
        "correctness",
        "timing",
        "trace",
    ]
    assert schema["additionalProperties"] is False
    jsonschema.validate(instance=_valid_measurement_manifest(), schema=schema)


def test_g5_measurement_manifest_rejects_ncu_mode() -> None:
    manifest = _valid_measurement_manifest()
    manifest["mode"] = "ncu"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=manifest, schema=_load_json(_ACTIVE_MANIFEST))


def test_g6_three_part_v5_label_authority_is_in_lockstep() -> None:
    sidecar = _load_json(_V5_SIDECAR)
    admitted = _load_json(_V5_ADMITTED)
    registry = _load_json(_METRIC_REGISTRY)

    assert sidecar["$id"] == "simbench_bw/schemas/local_label_sidecar.schema.json"
    assert sidecar["properties"]["schema_version"]["const"] == 5
    assert "unit_tagged_metrics" in sidecar["properties"]
    assert admitted["$id"] == "simbench_bw/schemas/local_label_admitted.schema.json"
    assert admitted["properties"]["schema_version"]["const"] == 5
    assert admitted["$defs"]["admitted_cell"]["properties"]["schema_version"][
        "const"
    ] == 5
    assert "unit_tagged_metrics" in admitted["$defs"]["admitted_cell"]["properties"]
    assert registry["registry_version"] == 2
    assert registry["label_schema_version"] == 5
    assert len(registry["metrics"]) == 61
    by_ncu_metric = {entry["ncu_metric"]: entry for entry in registry["metrics"]}
    assert by_ncu_metric["gpc__cycles_elapsed.max"]["unit"] == "cycles"
    assert by_ncu_metric["gpu__time_duration.sum"]["unit"] == "ns"


def test_g7_cupti_duration_record_fails_active_v5_admitted_schema() -> None:
    active_schema = _load_json(_V5_ADMITTED)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance=_valid_duration_record(),
            schema=active_schema["$defs"]["admitted_cell"],
        )


def test_g7_cupti_fields_fail_active_v5_sidecar_schema() -> None:
    schema = _load_json(_V5_SIDECAR)
    sidecar = _valid_v5_sidecar()
    jsonschema.validate(instance=sidecar, schema=schema)
    sidecar["duration_ns"] = 1000.0
    sidecar["dual_source_reconciliation_passed"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=sidecar, schema=schema)


def _tracked_old_duration_schema_mentions() -> Counter[str]:
    result = subprocess.run(
        ["git", "grep", "-n", "-o", "-I", "-F", _OLD_DURATION_SCHEMA_BASENAME],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or result.stdout)
    paths: list[str] = []
    for line in result.stdout.splitlines():
        path, _line_number, _content = line.split(":", 2)
        paths.append(path)
    return Counter(paths)


def test_g8_old_duration_schema_reference_allowlist() -> None:
    assert not (_SCHEMAS / _OLD_DURATION_SCHEMA_BASENAME).exists()
    assert _tracked_old_duration_schema_mentions() == Counter(
        {
            "workloads/simbench_bw/schemas/historical/"
            "duration_admitted.v1.schema.json": 1,
            "workloads/simbench_bw/schemas/historical/README.md": 1,
        }
    )
    readme = (_REPO_ROOT / "workloads" / "simbench_bw" / "README.md").read_text(
        encoding="utf-8"
    )
    begin = "HISTORICAL-D8-BEGIN"
    end = "HISTORICAL-D8-END"
    assert readme.count(begin) == readme.count(end) == 1
    outside = readme[: readme.index(begin)] + readme[readme.index(end) + len(end) :]
    for retired_token in (
        "--enable-cupti-activity",
        "cupti-collector-lib",
        "cupti_activity_collector",
        "cupti_dual_source_live_admission",
        "calibrate_dual_source_tolerance",
        "capture_timing_deploy_ns_sidecars.py",
        "collect_portable_lcd_duration.py",
        "duration_repeatability_aggregator.py",
        "duration_trace_goldilocks_sweep.py",
        "run_local_dual_source_lcd.py",
        "run_cloud_arm_evidence.sh",
        "verify_cloud_arm_evidence.py",
        "5060_3090ti_cross_dev",
        "current RLCR status",
    ):
        assert retired_token not in outside


def test_g9_tracked_duration_evidence_uses_frozen_v1() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", "data/local_labels"],
        cwd=_REPO_ROOT,
        text=True,
    ).splitlines()
    evidence_paths = [
        _REPO_ROOT / relative
        for relative in tracked
        if Path(relative).name == "duration_admitted.json"
    ]
    schema = _load_json(_HISTORICAL_DURATION)
    for path in evidence_paths:
        evidence = _load_json(path)
        try:
            version = evidence["schema_version"]
        except KeyError as exc:
            raise HistoricalSchemaDispatchError(
                f"tracked historical duration evidence lacks schema_version: {path}"
            ) from exc
        assert (
            resolve_historical_schema("duration_admitted", version)
            == _HISTORICAL_DURATION
        )
        jsonschema.validate(instance=evidence, schema=schema)


def test_g15_frozen_duration_schema_accepts_v1_and_rejects_ncu() -> None:
    schema = _duration_record_schema()
    jsonschema.validate(instance=_valid_duration_record(), schema=schema)
    record = _valid_duration_record()
    record["duration_label_source"] = "ncu_gpu__time_duration"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)
