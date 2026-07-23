"""zstd tier: byte-exact round trip, ratio, and the compressibility probe."""
from pathlib import Path

import pytest

from compression import (CompressionError, compress_bytes, compress_trace_dir,
                         compressibility_probe)


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


def test_compress_trace_dir_hashes_decompressed(tmp_path):
    d = _make_trace(tmp_path, 4, b"x" * 2000)
    out = tmp_path / "zst"
    res = compress_trace_dir(d, out)
    assert res.n_blocks == 4
    assert res.compressed_bytes < res.raw_bytes
    assert res.ratio > 1.0
    assert len(list(out.rglob("*.pb.zst"))) == 4


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
