"""Fail-closed NCU and first-launch provenance for the real golden set."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from golden_provenance import (
    GoldenProvenanceError,
    build_golden_provenance,
    parse_ncu_l2_csv,
    trace_provenance,
)


HEADER = (
    '"ID","Kernel Name","Grid Size","Metric Name","Metric Unit","Metric Value"'
)


def _ncu_csv(
    *,
    metric: str = "lts__t_sector_hit_rate.pct",
    unit: str = "%",
    value: str = "50.39",
    grid: str = "(36864, 1, 1)",
    kernel: str = "gs_copy(const float *, float *, int)",
) -> str:
    return (
        "==WARNING== Running with unmodified GPU clocks.\n"
        "==PROF== Connected to process 42 (/tmp/gs_stream_copy.bin)\n"
        "==PROF== Disconnected from process 42\n"
        f"{HEADER}\n"
        f'"0","{kernel}","{grid}","{metric}","{unit}","{value}"\n'
    )


def _write_ncu(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ncu_l2.csv"
    path.write_text(text)
    return path


def _trace_fixture(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    trace = tmp_path / "trace"
    first = trace / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
    nested = first / "nested"
    nested.mkdir(parents=True)
    (trace / "dynamic_trace.pb").write_bytes(b"dynamic-wire")
    (trace / "stats.csv").write_bytes(b"header\nrow\n")
    extra = trace / "extra_info"
    extra.mkdir()
    (extra / "allocations.json").write_bytes(b'{"allocations": []}\n')
    paths = [first / "block_b.pb", nested / "block_a.pb"]
    paths[0].write_bytes(b"payload-b")
    paths[1].write_bytes(b"payload-a")
    return trace, first, paths


def test_parse_ncu_csv_after_profiler_preamble_and_hash_raw_bytes(tmp_path):
    text = _ncu_csv()
    path = _write_ncu(tmp_path, text)

    parsed = parse_ncu_l2_csv(
        path,
        expected_grid=(36864, 1, 1),
        expected_kernel_name="gs_copy(const float *, float *, int)",
    )

    assert parsed == {
        "csv_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "metric_name": "lts__t_sector_hit_rate.pct",
        "metric_unit": "%",
        "value_pct": 50.39,
        "grid": [36864, 1, 1],
        "kernel_name": "gs_copy(const float *, float *, int)",
    }


def test_duplicate_matching_metric_fails(tmp_path):
    text = _ncu_csv() + (
        '"1","gs_copy(const float *, float *, int)","(36864, 1, 1)",'
        '"lts__t_sector_hit_rate.pct","%","51.0"\n'
    )
    with pytest.raises(GoldenProvenanceError, match="ncu_metric_row_count"):
        parse_ncu_l2_csv(
            _write_ncu(tmp_path, text), expected_grid=(36864, 1, 1)
        )


def test_wrong_metric_fails(tmp_path):
    path = _write_ncu(tmp_path, _ncu_csv(metric="lts__t_sectors.sum"))
    with pytest.raises(GoldenProvenanceError, match="ncu_metric_row_count"):
        parse_ncu_l2_csv(path, expected_grid=(36864, 1, 1))


def test_wrong_unit_fails(tmp_path):
    path = _write_ncu(tmp_path, _ncu_csv(unit="ratio"))
    with pytest.raises(GoldenProvenanceError, match="ncu_metric_unit_mismatch"):
        parse_ncu_l2_csv(path, expected_grid=(36864, 1, 1))


def test_wrong_grid_and_kernel_fail(tmp_path):
    path = _write_ncu(tmp_path, _ncu_csv(grid="(1024, 1, 1)"))
    with pytest.raises(GoldenProvenanceError, match="ncu_grid_mismatch"):
        parse_ncu_l2_csv(path, expected_grid=(36864, 1, 1))

    path.write_text(_ncu_csv(kernel="another_kernel()"))
    with pytest.raises(GoldenProvenanceError, match="ncu_kernel_mismatch"):
        parse_ncu_l2_csv(
            path,
            expected_grid=(36864, 1, 1),
            expected_kernel_name="gs_copy(const float *, float *, int)",
        )


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_metric_fails(tmp_path, value):
    path = _write_ncu(tmp_path, _ncu_csv(value=value))
    with pytest.raises(GoldenProvenanceError, match="ncu_metric_nonfinite"):
        parse_ncu_l2_csv(path, expected_grid=(36864, 1, 1))


@pytest.mark.parametrize("value", ["-0.01", "100.01"])
def test_out_of_range_metric_fails(tmp_path, value):
    path = _write_ncu(tmp_path, _ncu_csv(value=value))
    with pytest.raises(GoldenProvenanceError, match="ncu_metric_out_of_range"):
        parse_ncu_l2_csv(path, expected_grid=(36864, 1, 1))


def test_trace_hash_is_deterministic_for_directory_or_explicit_paths(tmp_path):
    trace, first, paths = _trace_fixture(tmp_path)

    from_dir = trace_provenance(trace, first_launch_dir=first)
    from_paths = trace_provenance(
        trace, first_launch_paths=list(reversed(paths))
    )

    assert from_dir == from_paths
    assert from_dir["first_launch"]["relative_path"] == (
        "threadblocks/device_0/stream_0/kernel_1"
    )
    assert from_dir["first_launch"]["block_count"] == 2
    assert from_dir["first_launch"]["byte_count"] == 18
    assert "files" not in from_dir["first_launch"]

    detailed = trace_provenance(trace, first_launch_dir=first, include_files=True)
    assert [
        item["relative_path"] for item in detailed["first_launch"]["files"]
    ] == ["first_launch/block_b.pb", "first_launch/nested/block_a.pb"]


def test_trace_payload_change_changes_file_and_aggregate_hashes(tmp_path):
    trace, first, paths = _trace_fixture(tmp_path)
    before = trace_provenance(trace, first_launch_dir=first, include_files=True)

    paths[1].write_bytes(b"payload-A")  # same size: digest, not size, detects it
    after = trace_provenance(trace, first_launch_dir=first, include_files=True)

    assert before["first_launch"]["block_count"] == after["first_launch"]["block_count"]
    assert before["first_launch"]["byte_count"] == after["first_launch"]["byte_count"]
    assert (
        before["first_launch"]["files"][1]["sha256"]
        != after["first_launch"]["files"][1]["sha256"]
    )
    assert (
        before["first_launch"]["content_sha256"]
        != after["first_launch"]["content_sha256"]
    )
    assert before["content_sha256"] != after["content_sha256"]


@pytest.mark.parametrize("artifact", ["dynamic_trace.pb", "stats.csv"])
def test_trace_required_artifact_missing_or_empty_fails(tmp_path, artifact):
    trace, first, _ = _trace_fixture(tmp_path)
    target = trace / artifact
    target.unlink()
    with pytest.raises(GoldenProvenanceError, match="_missing"):
        trace_provenance(trace, first_launch_dir=first)
    target.write_bytes(b"")
    with pytest.raises(GoldenProvenanceError, match="_empty"):
        trace_provenance(trace, first_launch_dir=first)


def test_empty_first_launch_or_present_empty_allocations_fails(tmp_path):
    trace, first, paths = _trace_fixture(tmp_path)
    for path in paths:
        path.unlink()
    with pytest.raises(GoldenProvenanceError, match="first_launch_payload_missing"):
        trace_provenance(trace, first_launch_dir=first)

    paths[0].write_bytes(b"payload")
    (trace / "extra_info" / "allocations.json").write_bytes(b"")
    with pytest.raises(GoldenProvenanceError, match="allocations_json_empty"):
        trace_provenance(trace, first_launch_dir=first)


def test_allocations_is_optional_but_an_empty_block_is_not(tmp_path):
    trace, first, paths = _trace_fixture(tmp_path)
    (trace / "extra_info" / "allocations.json").unlink()
    provenance = trace_provenance(trace, first_launch_dir=first)
    assert provenance["allocations_json"] is None

    paths[0].write_bytes(b"")
    with pytest.raises(GoldenProvenanceError, match="first_launch_payload_empty"):
        trace_provenance(trace, first_launch_dir=first)


def test_builder_records_explicit_version_strings(tmp_path):
    trace, first, _ = _trace_fixture(tmp_path)
    ncu = _write_ncu(tmp_path, _ncu_csv())
    provenance = build_golden_provenance(
        ncu_csv_path=ncu,
        trace_dir=trace,
        first_launch_dir=first,
        expected_grid=(36864, 1, 1),
        tracer_version="nvbit-enhanced-tracer-v5",
        stats_version="trace-sampling-counting-pass-v1",
        estimator_version="ghost-wave-segment-ht-v1",
    )
    assert provenance["schema_version"] == 1
    assert provenance["versions"] == {
        "tracer": "nvbit-enhanced-tracer-v5",
        "stats": "trace-sampling-counting-pass-v1",
        "estimator": "ghost-wave-segment-ht-v1",
    }
    assert "files" not in provenance["trace"]["first_launch"]

    with pytest.raises(GoldenProvenanceError, match="estimator_version_invalid"):
        build_golden_provenance(
            ncu_csv_path=ncu,
            trace_dir=trace,
            first_launch_dir=first,
            expected_grid=(36864, 1, 1),
            tracer_version="nvbit-enhanced-tracer-v5",
            stats_version="trace-sampling-counting-pass-v1",
            estimator_version="",
        )
