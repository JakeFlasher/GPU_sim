"""Tier-1 lossless compression + a compressibility probe.

zstd `-19 --long=27` per-block `.pb` -> `.pb.zst` (preserves lazy per-block
reads). The contract hash is taken over the DECOMPRESSED bytes and the round trip
is verified on every block, so compression can never silently corrupt the wire.
The probe compresses a small sample of blocks and extrapolates the whole-cell
ratio, so a cell can be routed (fits-after-compression vs still-breaching) before
paying to compress it all.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ZSTD = shutil.which("zstd") or "zstd"
LEVEL = 19
LONG = 27


class CompressionError(RuntimeError):
    """A trace cannot be admitted to the lossless-compression tier."""


def _zstd(data: bytes, *, decompress: bool = False) -> bytes:
    args = [ZSTD, "-q", f"--long={LONG}", "-c"]
    args.append("-d" if decompress else f"-{LEVEL}")
    p = subprocess.run(args, input=data, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"zstd {'-d' if decompress else 'compress'} failed: "
                           f"{p.stderr.decode(errors='replace')}")
    return p.stdout


def compress_bytes(raw: bytes) -> bytes:
    """Compress + verify the round trip; raise if decompression is not exact."""
    packed = _zstd(raw, decompress=False)
    if _zstd(packed, decompress=True) != raw:
        raise RuntimeError("zstd round trip is not byte-exact — refusing to emit")
    return packed


@dataclass
class CompressionResult:
    raw_bytes: int
    compressed_bytes: int
    n_blocks: int
    decompressed_sha256: str  # over the concatenated decompressed block bytes

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.compressed_bytes if self.compressed_bytes else 0.0


def compress_trace_dir(trace_dir: Path, out_dir: Path) -> CompressionResult:
    """Compress every `threadblocks/**/*.pb` into `<out_dir>/...pb.zst`,
    mirroring the tree. Hash is over the decompressed (original) bytes.

    A present-but-empty ``threadblocks/`` tree is not a zero-byte trace that
    happens to compress well.  It is missing simulator input, so refuse before
    creating an apparently valid compression result.
    """
    raw_total = comp_total = n = 0
    digest = hashlib.sha256()
    blocks = sorted((trace_dir / "threadblocks").rglob("*.pb"))
    if not blocks:
        raise CompressionError(
            f"zero_threadblock_payloads: {trace_dir}/threadblocks contains no .pb files"
        )
    for pb in blocks:
        raw = pb.read_bytes()
        packed = compress_bytes(raw)
        rel = pb.relative_to(trace_dir)
        dst = out_dir / rel.with_suffix(rel.suffix + ".zst")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(packed)
        raw_total += len(raw)
        comp_total += len(packed)
        digest.update(raw)
        n += 1
    return CompressionResult(raw_bytes=raw_total, compressed_bytes=comp_total,
                             n_blocks=n, decompressed_sha256=digest.hexdigest())


@dataclass
class ProbeResult:
    sampled_blocks: int
    total_blocks: int
    sampled_ratio: float
    estimated_full_compressed_bytes: int
    fits: bool  # under the byte budget after compression?


def compressibility_probe(trace_dir: Path, *, byte_budget: int,
                          n_sample: int = 10) -> ProbeResult:
    """Compress up to `n_sample` blocks (largest first — the conservative
    ratio), extrapolate the whole-cell compressed size, and decide whether the
    cell fits under `byte_budget` after compression."""
    if type(byte_budget) is not int or byte_budget <= 0:
        raise CompressionError(f"invalid_byte_budget: {byte_budget!r}")
    if type(n_sample) is not int or n_sample <= 0:
        raise CompressionError(f"invalid_probe_sample_count: {n_sample!r}")
    blocks = sorted((trace_dir / "threadblocks").rglob("*.pb"),
                    key=lambda p: p.stat().st_size, reverse=True)
    total = len(blocks)
    if total == 0:
        raise CompressionError(
            f"zero_threadblock_payloads: {trace_dir}/threadblocks contains no .pb files"
        )
    sample = blocks[:min(n_sample, total)]
    raw = comp = 0
    for pb in sample:
        b = pb.read_bytes()
        raw += len(b)
        comp += len(compress_bytes(b))
    ratio = raw / comp if comp else 0.0
    total_raw = sum(p.stat().st_size for p in blocks)
    est_comp = int(total_raw / ratio) if ratio else total_raw
    return ProbeResult(sampled_blocks=len(sample), total_blocks=total,
                       sampled_ratio=ratio,
                       estimated_full_compressed_bytes=est_comp,
                       fits=est_comp <= byte_budget)
