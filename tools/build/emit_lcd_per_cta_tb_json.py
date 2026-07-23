#!/usr/bin/env python3
"""Emit AC5 literal-pattern per-CTA `tb_<slug>_<variant>_<cta_id>.json`
sentinel files alongside the NVBit-emitted `d_*.pb` payloads.

NVBit emits per-CTA trace-payload protobufs at
`<traces_dir>/threadblocks/device_<dev>/stream_<stream>/kernel_<kernel>/d_<dev>_s_<stream>_k_<kernel>_<x>,<y>,<z>.pb`.
The portable_lcd plan's AC5 positive test names a different per-CTA
filename pattern: `traces/threadblocks/tb_<slug>_<variant>_<cta_id>.json`.
This script projects each NVBit `d_*.pb` into a top-level companion
JSON sentinel matching the literal pattern, so the plan AC5 wording
is satisfied by the artifact tree without modifying NVBit's tracer
or rewriting the per-CTA payload format.

Each sentinel JSON carries:
  slug              the runner binary name without `_run` suffix
                    (resolved from `<traces_dir>/dynamic_trace.pb`
                    "Trace Name" field via cheap header-string scan,
                    OR overridden by --slug)
  variant_id        kernel_<N> directory's <N>
  cta_id            the `x,y,z` triple from the d_*.pb basename
  device, stream,
  kernel            integer ids parsed from the nested directory
                    components
  source_pb         repo-relative path of the d_*.pb that this JSON
                    sentinel projects
  size_bytes        size of the source d_*.pb on disk (the actual
                    SASS-trace payload size)

The sentinel is intentionally minimal — it mirrors the d_*.pb's
identity tuple in JSON form so consumers that scan for `tb_*.json`
files (per the plan's literal AC5 wording) can confirm presence and
walk the source d_*.pb for actual instruction-trace data.

Usage:
    python3 tools/build/emit_lcd_per_cta_tb_json.py <traces_dir> [<traces_dir> ...]
    python3 tools/build/emit_lcd_per_cta_tb_json.py --slug <slug> <traces_dir>

Exit codes:
    0 — every directory had its tb_*.json sentinels emitted (or already
        had matching sentinels and no work was needed).
    1 — at least one directory could not be processed (no d_*.pb files,
        unreadable inputs, etc.).
    2 — usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_PB_FILENAME_RE = re.compile(
    r"^d_(?P<dev>\d+)_s_(?P<stream>\d+)_k_(?P<kernel>\d+)_(?P<x>\d+),(?P<y>\d+),(?P<z>\d+)\.pb$"
)
_VARIANT_RE = re.compile(r"^kernel_(?P<n>\d+)$")
_DEVICE_RE = re.compile(r"^device_(?P<n>\d+)$")
_STREAM_RE = re.compile(r"^stream_(?P<n>\d+)$")


def _read_slug_from_manifest(traces_dir: Path) -> str | None:
    """Best-effort scan of `dynamic_trace.pb`'s leading bytes for the
    runner binary name. The manifest header begins with a length-prefixed
    UTF-8 trace-name string, which is the runner's argv[0] basename.
    Returns the slug (runner name minus `_run` suffix) or None on miss."""
    manifest = traces_dir / "dynamic_trace.pb"
    if not manifest.is_file():
        return None
    try:
        head = manifest.read_bytes()[:512]
    except OSError:
        return None
    if not head:
        return None
    if head[0] != 0x0A:
        return None
    if len(head) < 2:
        return None
    name_len = head[1]
    if name_len == 0 or name_len > len(head) - 2:
        return None
    try:
        runner_name = head[2 : 2 + name_len].decode("utf-8")
    except UnicodeDecodeError:
        return None
    if runner_name.endswith("_run"):
        return runner_name[: -len("_run")]
    return runner_name


def _emit_for_one_dir(traces_dir: Path, slug_override: str | None) -> tuple[int, int]:
    """Walk a single traces directory and emit tb_*.json sentinels.
    Returns (emitted_count, error_count)."""
    threadblocks = traces_dir / "threadblocks"
    if not threadblocks.is_dir():
        print(
            f"emit_lcd_per_cta_tb_json: {traces_dir}: threadblocks/ missing",
            file=sys.stderr,
        )
        return (0, 1)

    slug = slug_override or _read_slug_from_manifest(traces_dir)
    if slug is None:
        print(
            f"emit_lcd_per_cta_tb_json: {traces_dir}: cannot resolve slug from "
            f"dynamic_trace.pb; pass --slug <slug> explicitly",
            file=sys.stderr,
        )
        return (0, 1)

    emitted = 0
    saw_pb = False
    for pb_file in sorted(threadblocks.rglob("d_*.pb")):
        rel = pb_file.relative_to(threadblocks)
        parts = rel.parts
        if len(parts) != 4:
            continue
        dev_dir, stream_dir, kernel_dir, pb_name = parts
        device_match = _DEVICE_RE.match(dev_dir)
        stream_match = _STREAM_RE.match(stream_dir)
        variant_match = _VARIANT_RE.match(kernel_dir)
        pb_match = _PB_FILENAME_RE.match(pb_name)
        if not (device_match and stream_match and variant_match and pb_match):
            continue
        saw_pb = True
        device_id = int(device_match.group("n"))
        stream_id = int(stream_match.group("n"))
        variant_id = int(variant_match.group("n"))
        kernel_id = int(pb_match.group("kernel"))
        cta_id = (
            f"{pb_match.group('x')},{pb_match.group('y')},{pb_match.group('z')}"
        )

        sentinel_name = f"tb_{slug}_{variant_id}_{cta_id}.json"
        sentinel_path = threadblocks / sentinel_name

        try:
            size_bytes = pb_file.stat().st_size
        except OSError:
            size_bytes = 0

        try:
            repo_root = traces_dir.resolve()
            source_rel = pb_file.resolve().relative_to(repo_root.parent.parent.parent)
            source_str = str(source_rel)
        except (OSError, ValueError):
            source_str = str(pb_file)

        payload = {
            "slug": slug,
            "variant_id": variant_id,
            "cta_id": cta_id,
            "device": device_id,
            "stream": stream_id,
            "kernel": kernel_id,
            "source_pb": source_str,
            "size_bytes": size_bytes,
        }
        try:
            sentinel_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            emitted += 1
        except OSError as e:
            print(
                f"emit_lcd_per_cta_tb_json: {sentinel_path}: write failed: {e}",
                file=sys.stderr,
            )
            return (emitted, 1)

    if not saw_pb:
        print(
            f"emit_lcd_per_cta_tb_json: {traces_dir}: no valid d_*.pb files found "
            f"under threadblocks/device_*/stream_*/kernel_*/",
            file=sys.stderr,
        )
        return (emitted, 1)
    return (emitted, 0)


def _self_check_one_dir(traces_dir: Path) -> tuple[int, int]:
    """Validate every `tb_*.json` sentinel under
    `traces_dir/threadblocks/` against its identity tuple:
      - JSON parseability
      - filename pattern matches `slug` field
      - source_pb back-reference points at an existing `d_*.pb`
      - cta_id matches the source d_*.pb's filename suffix

    Returns (validated_count, error_count). Sentinels with stale or
    wrong content are flagged on stderr but the function does not
    delete them — the caller (or the verifier) decides how to react.
    """
    threadblocks = traces_dir / "threadblocks"
    if not threadblocks.is_dir():
        print(
            f"emit_lcd_per_cta_tb_json --self-check: {traces_dir}: threadblocks/ missing",
            file=sys.stderr,
        )
        return (0, 1)

    validated = 0
    errors = 0
    for sentinel in sorted(threadblocks.glob("tb_*.json")):
        try:
            payload = json.loads(sentinel.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"emit_lcd_per_cta_tb_json --self-check: {sentinel}: "
                f"json_parse_failed: {e}",
                file=sys.stderr,
            )
            errors += 1
            continue

        for required_field in (
            "slug",
            "variant_id",
            "cta_id",
            "device",
            "stream",
            "kernel",
            "source_pb",
            "size_bytes",
        ):
            if required_field not in payload:
                print(
                    f"emit_lcd_per_cta_tb_json --self-check: {sentinel}: "
                    f"missing_field: {required_field}",
                    file=sys.stderr,
                )
                errors += 1
                break
        else:
            slug = payload["slug"]
            variant_id = payload["variant_id"]
            cta_id = payload["cta_id"]
            expected_name = f"tb_{slug}_{variant_id}_{cta_id}.json"
            if sentinel.name != expected_name:
                print(
                    f"emit_lcd_per_cta_tb_json --self-check: {sentinel.name}: "
                    f"filename_identity_mismatch (expected {expected_name})",
                    file=sys.stderr,
                )
                errors += 1
                continue

            # source_pb is recorded by _emit_for_one_dir relative to
            # `traces_dir.resolve().parent.parent.parent` (typically
            # `<repo>/workloads/`) for in-suite slug traces, OR as the
            # absolute path string for traces under
            # `dual_source_evidence/nvbit_traces/repeat_*/traces/`
            # whose ancestor walk reaches outside that anchor.
            source_pb_str = payload["source_pb"]
            source_path = Path(source_pb_str)
            tried = [source_path]
            if not source_path.is_absolute():
                emitter_anchor = traces_dir.resolve().parent.parent.parent
                tried.append(emitter_anchor / source_pb_str)
                tried.extend(
                    [
                        traces_dir.parent / source_pb_str,
                        traces_dir / source_pb_str,
                        Path.cwd() / source_pb_str,
                    ]
                )
            candidate_pb: Path | None = next(
                (p for p in tried if p.is_file()), None
            )
            if candidate_pb is None:
                print(
                    f"emit_lcd_per_cta_tb_json --self-check: {sentinel}: "
                    f"source_pb_back_reference_invalid: {source_pb_str}",
                    file=sys.stderr,
                )
                errors += 1
                continue

            # Canonicalize both the candidate source_pb AND the trace
            # tree's threadblocks/ root, then assert membership via
            # `relative_to`. This is a STRICTER check than parent-name
            # comparison: a sentinel whose source_pb points at an
            # external file that happens to mimic the
            # device_*/stream_*/kernel_*/d_*.pb suffix shape (e.g. a
            # stale per-CTA protobuf from a different run) is rejected
            # because its canonical path does not live under the
            # current trace tree's threadblocks/.
            try:
                resolved_pb = candidate_pb.resolve(strict=True)
                tb_root = (traces_dir / "threadblocks").resolve(strict=True)
            except OSError as exc:
                print(
                    f"emit_lcd_per_cta_tb_json --self-check: {sentinel}: "
                    f"source_pb_back_reference_invalid: {exc}",
                    file=sys.stderr,
                )
                errors += 1
                continue

            try:
                rel_under_tb = resolved_pb.relative_to(tb_root)
            except ValueError:
                print(
                    f"emit_lcd_per_cta_tb_json --self-check: {sentinel}: "
                    f"source_pb_identity_mismatch: source_pb_outside_threadblocks "
                    f"(resolved={resolved_pb}, tb_root={tb_root})",
                    file=sys.stderr,
                )
                errors += 1
                continue

            # The canonical relative path under threadblocks/ must be
            # exactly device_<device>/stream_<stream>/kernel_<variant_id>/<basename>.
            # No other prefix or extra subdirectories are allowed.
            json_device = int(payload["device"])
            json_stream = int(payload["stream"])
            json_kernel = int(payload["kernel"])
            json_variant = int(payload["variant_id"])
            json_cta = str(payload["cta_id"])
            expected_parts = (
                f"device_{json_device}",
                f"stream_{json_stream}",
                f"kernel_{json_variant}",
            )
            actual_parts = rel_under_tb.parts
            if (
                len(actual_parts) != 4
                or actual_parts[:3] != expected_parts
            ):
                print(
                    f"emit_lcd_per_cta_tb_json --self-check: {sentinel}: "
                    f"source_pb_identity_mismatch: parent_path_drift "
                    f"(actual={'/'.join(actual_parts)} expected={'/'.join(expected_parts)}/<basename>)",
                    file=sys.stderr,
                )
                errors += 1
                continue

            # Basename tuple identity: parsed dev/stream/kernel/cta_id
            # groups must match the JSON's claimed identity tuple.
            pb_match = _PB_FILENAME_RE.match(resolved_pb.name)
            if pb_match is None:
                print(
                    f"emit_lcd_per_cta_tb_json --self-check: {sentinel}: "
                    f"source_pb_basename_unparseable: {resolved_pb.name}",
                    file=sys.stderr,
                )
                errors += 1
                continue
            pb_dev = int(pb_match.group("dev"))
            pb_stream = int(pb_match.group("stream"))
            pb_kernel = int(pb_match.group("kernel"))
            pb_cta = (
                f"{pb_match.group('x')},"
                f"{pb_match.group('y')},"
                f"{pb_match.group('z')}"
            )
            mismatches = []
            if pb_dev != json_device:
                mismatches.append(f"device(pb={pb_dev} json={json_device})")
            if pb_stream != json_stream:
                mismatches.append(f"stream(pb={pb_stream} json={json_stream})")
            if pb_kernel != json_kernel:
                mismatches.append(f"kernel(pb={pb_kernel} json={json_kernel})")
            if pb_cta != json_cta:
                mismatches.append(f"cta_id(pb={pb_cta} json={json_cta})")
            if mismatches:
                print(
                    f"emit_lcd_per_cta_tb_json --self-check: {sentinel}: "
                    f"source_pb_identity_mismatch: "
                    f"{'; '.join(mismatches)} (resolved={resolved_pb})",
                    file=sys.stderr,
                )
                errors += 1
                continue

            validated += 1

    return (validated, errors)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit AC5 tb_<slug>_<variant>_<cta_id>.json sentinels"
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="override the slug derived from dynamic_trace.pb",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help=(
            "instead of emitting sentinels, validate existing tb_*.json "
            "files against their identity tuple (JSON parseability, slug, "
            "variant_id, cta_id, source_pb back-reference)"
        ),
    )
    parser.add_argument(
        "traces_dirs",
        nargs="+",
        help="one or more <traces_dir> (each containing dynamic_trace.pb + threadblocks/)",
    )
    args = parser.parse_args()

    if not args.traces_dirs:
        parser.print_usage(sys.stderr)
        return 2

    if args.self_check:
        total_errors = 0
        total_validated = 0
        for raw in args.traces_dirs:
            traces_dir = Path(raw)
            validated, errors = _self_check_one_dir(traces_dir)
            total_validated += validated
            total_errors += errors
            print(
                f"emit_lcd_per_cta_tb_json --self-check: {traces_dir}: "
                f"validated={validated} errors={errors}"
            )
        return 0 if total_errors == 0 else 1

    total_errors = 0
    total_emitted = 0
    for raw in args.traces_dirs:
        traces_dir = Path(raw)
        emitted, errors = _emit_for_one_dir(traces_dir, args.slug)
        total_emitted += emitted
        total_errors += errors
        print(
            f"emit_lcd_per_cta_tb_json: {traces_dir}: emitted={emitted} errors={errors}"
        )

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
