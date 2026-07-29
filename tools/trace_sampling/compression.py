"""Tier-1 lossless compression + a compressibility probe.

zstd `-19 --long=27` per-block `.pb` -> `.pb.zst` (preserves lazy per-block
reads). The legacy contract hash is taken over the DECOMPRESSED bytes; a
canonical path/length/raw-SHA manifest additionally binds every file boundary.
The round trip is verified on every block, so compression can never silently
corrupt the wire.
The probe compresses a small sample of blocks and extrapolates the whole-cell
ratio, so a cell can be routed (fits-after-compression vs still-breaching) before
paying to compress it all.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

ZSTD = shutil.which("zstd") or "zstd"
LEVEL = 19
LONG = 27


class CompressionError(RuntimeError):
    """A trace cannot be admitted to the lossless-compression tier."""


class PayloadManifestError(ValueError):
    """A per-file payload manifest is absent or noncanonical."""


PAYLOAD_FILE_MANIFEST_HEADER = "trace-payload-file-manifest-v1"
_MAX_U64 = (1 << 64) - 1
_MAX_U64_TEXT = str(_MAX_U64)


@dataclass(frozen=True)
class PayloadManifestEntry:
    path: str
    size_bytes: int
    sha256: str


def _validate_manifest_entry(entry: PayloadManifestEntry) -> None:
    path = entry.path
    if (not isinstance(path, str) or not path or not path.isascii()
            or any(not character.isprintable() or character.isspace()
                   for character in path)
            or "\\" in path):
        raise PayloadManifestError(
            "payload manifest path must be a nonempty ASCII-graphic POSIX path")
    pure = PurePosixPath(path)
    basename = pure.name
    payload_stem = basename[:-3] if basename.endswith(".pb") else ""
    if (pure.is_absolute() or pure.as_posix() != path
            or any(part in ("", ".", "..") for part in pure.parts)
            or not pure.parts or pure.parts[0] != "threadblocks"
            or not payload_stem or all(character == "." for character in payload_stem)):
        raise PayloadManifestError(
            f"payload manifest path is not a canonical threadblocks .pb path: {path!r}")
    if (type(entry.size_bytes) is not int or entry.size_bytes < 0
            or entry.size_bytes > _MAX_U64):
        raise PayloadManifestError(
            f"payload manifest size must be a u64: {entry.size_bytes!r}")
    if (not isinstance(entry.sha256, str) or len(entry.sha256) != 64
            or any(character not in "0123456789abcdef" for character in entry.sha256)):
        raise PayloadManifestError("payload manifest hash must be lowercase SHA-256")


def build_payload_file_manifest(
        entries: Iterable[PayloadManifestEntry]) -> str:
    """Return the canonical v1 path/size/SHA-256 payload manifest."""
    ordered = sorted(entries, key=lambda entry: entry.path)
    if not ordered:
        raise PayloadManifestError("payload manifest must contain at least one .pb file")
    previous = None
    lines = [PAYLOAD_FILE_MANIFEST_HEADER]
    for entry in ordered:
        _validate_manifest_entry(entry)
        if previous == entry.path:
            raise PayloadManifestError(
                f"payload manifest contains duplicate path {entry.path!r}")
        previous = entry.path
        lines.append(f"{entry.path}\t{entry.size_bytes}\t{entry.sha256}")
    return "\n".join(lines) + "\n"


def parse_payload_file_manifest(value: object) -> tuple[PayloadManifestEntry, ...]:
    """Parse a canonical v1 manifest without normalizing attacker input."""
    if not isinstance(value, str):
        raise PayloadManifestError("payload manifest must be a string")
    prefix = PAYLOAD_FILE_MANIFEST_HEADER + "\n"
    if not value.startswith(prefix) or not value.endswith("\n"):
        raise PayloadManifestError(
            "payload manifest must use the canonical v1 header and trailing newline")
    body = value[len(prefix):-1]
    if not body:
        raise PayloadManifestError("payload manifest must contain at least one .pb file")

    entries = []
    previous = None
    for line in body.split("\n"):
        fields = line.split("\t")
        if len(fields) != 3:
            raise PayloadManifestError(
                "payload manifest entries must be path<TAB>size<TAB>sha256")
        path, size_text, sha256 = fields
        if (not size_text.isascii() or not size_text.isdigit()
                or (len(size_text) > 1 and size_text.startswith("0"))):
            raise PayloadManifestError(
                f"payload manifest size is not canonical decimal: {size_text!r}")
        if (len(size_text) > len(_MAX_U64_TEXT)
                or (len(size_text) == len(_MAX_U64_TEXT)
                    and size_text > _MAX_U64_TEXT)):
            raise PayloadManifestError(
                f"payload manifest size exceeds u64: {size_text!r}")
        entry = PayloadManifestEntry(
            path=path, size_bytes=int(size_text), sha256=sha256)
        _validate_manifest_entry(entry)
        if previous is not None and path <= previous:
            raise PayloadManifestError(
                "payload manifest paths must be unique and byte-sorted")
        previous = path
        entries.append(entry)

    canonical = build_payload_file_manifest(entries)
    if canonical != value:
        raise PayloadManifestError("payload manifest is not canonically encoded")
    return tuple(entries)


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
    per_block_raw_sha256_manifest: str  # canonical path/size/hash file identity

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.compressed_bytes if self.compressed_bytes else 0.0


def compress_trace_dir(trace_dir: Path, out_dir: Path) -> CompressionResult:
    """Compress every `threadblocks/**/*.pb` into `<out_dir>/...pb.zst`,
    mirroring the tree. The legacy aggregate hash covers the concatenated
    decompressed bytes; the canonical per-file manifest additionally binds each
    safe relative path, exact byte length, and raw SHA-256.

    A present-but-empty ``threadblocks/`` tree is not a zero-byte trace that
    happens to compress well.  It is missing simulator input, so refuse before
    creating an apparently valid compression result.
    """
    raw_total = comp_total = n = 0
    digest = hashlib.sha256()
    candidates = []
    for pb in (trace_dir / "threadblocks").rglob("*.pb"):
        if pb.is_symlink() or not pb.is_file():
            raise CompressionError(f"nonregular_threadblock_payload: {pb}")
        rel = pb.relative_to(trace_dir).as_posix()
        try:
            _validate_manifest_entry(PayloadManifestEntry(rel, 0, "0" * 64))
        except PayloadManifestError as exc:
            raise CompressionError(f"unsafe_threadblock_payload_path: {exc}") from exc
        candidates.append((rel, pb))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise CompressionError(
            f"zero_threadblock_payloads: {trace_dir}/threadblocks contains no .pb files"
        )
    manifest_entries = []
    for rel, pb in candidates:
        raw = pb.read_bytes()
        packed = compress_bytes(raw)
        dst = out_dir / Path(rel).with_suffix(".pb.zst")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(packed)
        raw_total += len(raw)
        comp_total += len(packed)
        digest.update(raw)
        manifest_entries.append(PayloadManifestEntry(
            path=rel,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        ))
        n += 1
    return CompressionResult(
        raw_bytes=raw_total,
        compressed_bytes=comp_total,
        n_blocks=n,
        decompressed_sha256=digest.hexdigest(),
        per_block_raw_sha256_manifest=build_payload_file_manifest(manifest_entries),
    )


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
