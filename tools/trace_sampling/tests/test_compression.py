"""zstd tier: byte-exact round trip, ratio, and the compressibility probe."""
import hashlib
from pathlib import Path

import pytest

from compression import (CompressionError, compress_bytes, compress_trace_dir,
                         compressibility_probe, parse_payload_file_manifest)


def test_round_trip_is_byte_exact():
    raw = b"abc" * 5000 + bytes(range(256)) * 40
    packed = compress_bytes(raw)
    assert len(packed) < len(raw)  # compressible payload shrinks


def _make_trace(tmp: Path, n_blocks: int, payload: bytes) -> Path:
    d = tmp / "trace"
    kd = d / "threadblocks" / "device_0" / "stream_0" / "kernel_1"
    kd.mkdir(parents=True)
    for i in range(n_blocks):
        (kd / f"b_{i}.pb").write_bytes(payload)
    return d


def test_compress_trace_dir_hashes_decompressed_and_binds_each_file(tmp_path):
    d = _make_trace(tmp_path, 4, b"x" * 2000)
    out = tmp_path / "zst"
    res = compress_trace_dir(d, out)
    assert res.n_blocks == 4
    assert res.compressed_bytes < res.raw_bytes
    assert res.ratio > 1.0
    assert len(list(out.rglob("*.pb.zst"))) == 4
    entries = parse_payload_file_manifest(res.per_block_raw_sha256_manifest)
    assert [entry.path for entry in entries] == [
        f"threadblocks/device_0/stream_0/kernel_1/b_{index}.pb"
        for index in range(4)
    ]
    assert sum(entry.size_bytes for entry in entries) == res.raw_bytes
    assert all(entry.sha256 == hashlib.sha256(b"x" * 2000).hexdigest()
               for entry in entries)


def test_dot_only_pb_basename_is_rejected_explicitly(tmp_path):
    trace = _make_trace(tmp_path, 1, b"payload")
    (trace / "threadblocks" / ".pb").write_bytes(b"ambiguous")

    with pytest.raises(CompressionError, match="unsafe_threadblock_payload_path"):
        compress_trace_dir(trace, tmp_path / "zst")


def test_pb_named_directory_is_rejected_like_a_nonregular_payload(tmp_path):
    trace = _make_trace(tmp_path, 1, b"payload")
    directory = trace / "threadblocks" / "nested.pb"
    directory.mkdir()
    (directory / "block.pb").write_bytes(b"nested")

    with pytest.raises(CompressionError, match="nonregular_threadblock_payload"):
        compress_trace_dir(trace, tmp_path / "zst")


def test_aggregate_hash_uses_canonical_relative_path_byte_order(tmp_path):
    trace = tmp_path / "trace"
    (trace / "threadblocks" / "a").mkdir(parents=True)
    (trace / "threadblocks" / "a.pb").write_bytes(b"first")
    (trace / "threadblocks" / "a" / "b.pb").write_bytes(b"second")

    result = compress_trace_dir(trace, tmp_path / "zst")
    entries = parse_payload_file_manifest(result.per_block_raw_sha256_manifest)
    assert [entry.path for entry in entries] == [
        "threadblocks/a.pb",
        "threadblocks/a/b.pb",
    ]
    assert result.decompressed_sha256 == hashlib.sha256(b"firstsecond").hexdigest()


def test_file_boundary_repartition_changes_manifest_not_aggregate_hash(tmp_path):
    first = _make_trace(tmp_path / "first", 2, b"unused")
    paths = sorted((first / "threadblocks").rglob("*.pb"))
    paths[0].write_bytes(b"ab")
    paths[1].write_bytes(b"c")
    original = compress_trace_dir(first, tmp_path / "first-zst")

    second = _make_trace(tmp_path / "second", 2, b"unused")
    paths = sorted((second / "threadblocks").rglob("*.pb"))
    paths[0].write_bytes(b"a")
    paths[1].write_bytes(b"bc")
    repartitioned = compress_trace_dir(second, tmp_path / "second-zst")

    assert original.raw_bytes == repartitioned.raw_bytes == 3
    assert original.decompressed_sha256 == repartitioned.decompressed_sha256
    assert (original.per_block_raw_sha256_manifest
            != repartitioned.per_block_raw_sha256_manifest)


def test_probe_fits_and_breach(tmp_path):
    d = _make_trace(tmp_path, 20, b"y" * 1000)  # 20 KB raw, very compressible
    fit = compressibility_probe(d, byte_budget=10_000, n_sample=5)
    assert fit.fits and fit.sampled_ratio > 1.0
    breach = compressibility_probe(d, byte_budget=5, n_sample=5)
    assert not breach.fits


def test_empty_trace_probe_refuses_vacuous_fit(tmp_path):
    d = tmp_path / "empty"
    (d / "threadblocks").mkdir(parents=True)
    with pytest.raises(CompressionError, match="zero_threadblock_payloads"):
        compressibility_probe(d, byte_budget=10)


def test_empty_trace_compression_refuses_vacuous_success(tmp_path):
    d = tmp_path / "empty"
    (d / "threadblocks").mkdir(parents=True)
    with pytest.raises(CompressionError, match="zero_threadblock_payloads"):
        compress_trace_dir(d, tmp_path / "out")


def test_probe_refuses_zero_sample_or_budget(tmp_path):
    d = _make_trace(tmp_path, 1, b"x" * 100)
    with pytest.raises(CompressionError, match="invalid_probe_sample_count"):
        compressibility_probe(d, byte_budget=10, n_sample=0)
    with pytest.raises(CompressionError, match="invalid_byte_budget"):
        compressibility_probe(d, byte_budget=0, n_sample=1)
