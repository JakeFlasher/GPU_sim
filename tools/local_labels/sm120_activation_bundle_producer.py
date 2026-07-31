"""Historical sm_120 predictor-bundle producer.

The source and its retained artifacts document the former AC-4 production
lane, but live sm_120 bundle emission is disabled by the shared
sm120-template safety gate. ``main`` returns a typed-unavailable report before
loading labels, planning output, compiling CUDA helpers, or invoking probes.
The lower-level helpers remain available for source/evidence archaeology and
focused contract tests; they are individually guarded before any binary, trace,
CUDA, or output access.

Historical contract (not an active production claim): the former implementation
consumed the admitted sm_120 rows in ``data/local_labels/label_admitted.json``
and emitted one predictor bundle per cell. Existing evidence is retained
byte-for-byte; this module never rewrites it while the gate is active.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SWEEPS_ROOT = _REPO_ROOT / "data" / "local_labels" / "sweeps"
_ADMITTED_PATH = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
_CONSUME_TRACE = (
    _REPO_ROOT / "tools" / "sector_stream" / "consume_trace.py"
)

sys.path.insert(0, str(_REPO_ROOT / "workloads" / "simbench_bw"))
sys.path.insert(0, str(_REPO_ROOT))
from tools.local_labels.roster_path_resolver import (  # noqa: E402
    _row_for, resolve_cell_root,
)
from tools.local_labels.protected_paths import (  # noqa: E402
    ProtectedPathError,
    assert_external_output,
    assert_write_allowed,
)
from tools.local_labels.sm120_template_safety import (  # noqa: E402
    Sm120TemplateUnavailable,
    is_sm120_template_arch,
    unavailable_result,
)
from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    ensure_directory,
    replace_regular_bytes,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    AdmittedTraceSources,
    TraceAdmissionError,
    open_admitted_trace_sources,
    require_trace_admission,
)
from tools.sector_stream.consume_trace import (  # noqa: E402
    AdmittedSectorStream,
    consume_admitted_trace_sources,
)

_HARD_TRACE_SOURCE_ERRORS = (
    TraceAdmissionError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    ImmutableArtifactError,
)

# RTX 5060 Laptop GPU device-name string. Mirrors the value
# emitted by `run_suite.py --mode trace` for the dram_stream
# baseline so the bundle's device field matches existing
# sm_120-host evidence.
_SM120_LAPTOP_DEVICE_NAME = "NVIDIA GeForce RTX 5060 Laptop GPU"
_STAGING_ROOT = _REPO_ROOT / "results" / "nvbit_capture_staging"
_DEFAULT_OUTPUT_BASE = _STAGING_ROOT / "sm120_cohort_bundles"


def _assert_staging_output(path: Path | str, *, operation: str) -> Path:
    candidate = assert_write_allowed(
        path, repo_root=_REPO_ROOT, operation=operation,
    )
    staging = _STAGING_ROOT.resolve(strict=False)
    try:
        candidate.relative_to(staging)
    except ValueError as exc:
        raise ProtectedPathError(
            f"refusing to {operation} outside {staging}: {candidate}"
        ) from exc
    return candidate


def _bundle_output_plan(
    *,
    candidates: list[tuple[str, str]],
    output_root: Path | str,
    helper_output: Path | str | None = None,
) -> dict:
    """Clear every bundle/derived/helper destination before live probes."""

    root = _assert_staging_output(output_root, operation="select bundle output root")
    bundles_dir = _assert_staging_output(
        root / "predictor_bundles", operation="prepare predictor bundle directory",
    )
    helper_candidate = (
        Path(helper_output)
        if helper_output is not None
        else Path(tempfile.gettempdir()) / "gpu_simulator_sm120_get_l2_cache_size"
    )
    helper = assert_external_output(
        helper_candidate,
        repo_root=_REPO_ROOT,
        operation="build sm120 CUDA helper",
    )
    cells: dict[tuple[str, str], dict[str, Path]] = {}
    for slug, shape_point in candidates:
        source_cell = resolve_cell_root(
            slug=slug,
            shape_point=shape_point,
            sku="rtx_5060_laptop",
            arch_dir="sm_120",
        )
        try:
            relative_cell = source_cell.resolve(strict=False).relative_to(_REPO_ROOT)
        except ValueError as exc:
            raise ProtectedPathError(
                f"roster cell root escapes repository: {source_cell}"
            ) from exc
        derived = root / "derived_sector_streams" / relative_cell
        cells[(slug, shape_point)] = {
            "bundle_path": _assert_staging_output(
                bundles_dir / f"sm_120__{slug}__{shape_point}.json",
                operation="prepare predictor bundle",
            ),
            "derived_sector_stream": _assert_staging_output(
                derived / f"trace_{shape_point}" / "sector_stream.jsonl",
                operation="prepare derived sector stream",
            ),
        }
    return {
        "output_root": root,
        "predictor_bundles_dir": bundles_dir,
        "helper": helper,
        "cells": cells,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _query_l2_cache_size_bytes(
    *, helper_bin: Path | None = None, arch: str = "sm_120",
) -> int:
    """Query the live ``cudaDeviceProp.l2CacheSize`` via a small
    compiled CUDA helper. The helper is built on demand at
    ``/tmp/sm120_get_l2_cache_size``. Returns the byte count for
    device 0 (the only GPU on this single-laptop host)."""
    if is_sm120_template_arch(arch):
        raise Sm120TemplateUnavailable(
            operation="sm120_activation_bundle_producer.query_l2_cache_size",
            detail=(
                f"runtime geometry for {arch} is not a production "
                "sm_120 predictor input"
            ),
        )
    helper_src = _REPO_ROOT / "tools" / "local_labels" / "_sm120_get_l2_cache_size.cu"
    helper_bin = assert_external_output(
        helper_bin
        if helper_bin is not None
        else Path(tempfile.gettempdir()) / "gpu_simulator_sm120_get_l2_cache_size",
        repo_root=_REPO_ROOT,
        operation="build sm120 CUDA helper",
    )
    if not helper_src.is_file():
        raise FileNotFoundError(
            f"tracked CUDA helper source missing: {helper_src}; restore it "
            "byte-exactly rather than creating source in the worktree"
        )
    if not helper_bin.is_file() or helper_bin.stat().st_mtime < helper_src.stat().st_mtime:
        subprocess.run(
            ["nvcc", "-ccbin=/usr/bin/g++-14", "-O0",
             str(helper_src), "-o", str(helper_bin)],
            check=True, capture_output=True,
        )
    out = subprocess.run(
        [str(helper_bin)], check=True, capture_output=True, text=True,
    )
    return int(out.stdout.strip())


def _git_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
        cwd=str(_REPO_ROOT),
    )
    return out.stdout.strip()


def _load_admitted_sm120(
    path: Path = _ADMITTED_PATH,
) -> dict[tuple[str, str], dict]:
    """Load the admitted sm_120 cells from ``path`` (default: the global
    ``data/local_labels/label_admitted.json``). A caller that selected a
    non-default label corpus (e.g. the headline runner's
    ``--label-admitted-path`` for a pure-sm_120 cohort) MUST pass it here
    so the producer builds the SELECTED corpus rather than the global
    admitted set — otherwise the runner's downstream bundle-set gate sees
    extra/missing bundles."""
    with path.open("r", encoding="utf-8") as fh:
        d = json.load(fh)
    out: dict[tuple[str, str], dict] = {}
    for cell in d.get("cells", []):
        if cell.get("arch") != "sm_120":
            continue
        key = (cell.get("slug"), cell.get("shape_point"))
        out[key] = cell
    return out


def _generate_sector_stream(
    *,
    sources: AdmittedTraceSources,
    output_path: Path,
) -> tuple[AdmittedSectorStream, Path]:
    """Generate admission-bound sectors and serialize those exact bytes."""
    output_path = _assert_staging_output(
        output_path, operation="write derived sector stream",
    )
    result = consume_admitted_trace_sources(
        sources=sources,
        output_root=None,
        write_jsonl=False,
    )
    if not result.schema_admits_all:
        raise TraceAdmissionError(
            "sector_stream_schema_invalid",
            "sm120 helper sector-stream producer generated a schema-invalid "
            f"record: {result.schema_first_failure}",
        )
    result.verify_for(sources)
    ensure_directory(output_path.parent)
    replace_regular_bytes(output_path, result.jsonl_bytes())
    return result, output_path


def _resolve_binary_path(slug: str, shape_point: str) -> Path:
    """Resolve the per-cell binary by parsing the matched roster
    row's ``binary_path`` field. Round-13 widening: simbench_bw L1
    cells (e.g., ``warm_cold_pair_relaunch``) use
    ``workloads/simbench_bw/l1/<slug>/<slug>_run`` while
    portable_lcd cells use
    ``workloads/portable_lcd/<slug>/<slug>_run``. The frozen
    roster carries the canonical ``binary_path`` per row; this
    helper returns the absolute path.
    """
    row = _row_for(slug, shape_point)
    if row is None:
        raise FileNotFoundError(
            f"no roster row for ({slug!r}, {shape_point!r}); the "
            f"producer requires the cell to appear in the frozen "
            f"roster"
        )
    binary_path = row.get("binary_path")
    if not binary_path:
        raise FileNotFoundError(
            f"roster row for ({slug!r}, {shape_point!r}) has "
            f"no binary_path field"
        )
    return _REPO_ROOT / binary_path


def _emit_bundle_for_cell(
    *,
    slug: str,
    shape_point: str,
    admitted: dict,
    l2_cache_size_bytes: int,
    output_dir: Path,
    head_commit: str,
    bundle_path: Path | None = None,
    derived_sector_stream: Path | None = None,
    arch: str = "sm_120",
) -> Path:
    """Run cache_predictor against the EXISTING per-shape trace and
    emit a bundle JSON for one admitted sm_120 cohort cell. Returns
    the path to the emitted bundle.

    Round-13 widening: the producer consumes existing artifacts
    (dynamic_trace.pb + sector_stream.jsonl) rather than
    re-capturing the trace via NVBit. Re-capture would risk
    binary_sha256 drift against the admitted row; the existing
    trace was captured at the same binary HEAD as the admission."""
    if is_sm120_template_arch(arch):
        raise Sm120TemplateUnavailable(
            operation="sm120_activation_bundle_producer.emit_bundle",
            detail=(
                f"bundle emission for {arch} is disabled; retained sm_120 "
                "traces and labels are evidence only"
            ),
        )

    binary = _resolve_binary_path(slug, shape_point)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"activation binary missing or not executable: {binary}"
        )
    binary_sha = _sha256_file(binary)
    if binary_sha != admitted["binary_sha256"]:
        raise RuntimeError(
            f"binary SHA mismatch for {slug}: on-disk={binary_sha} "
            f"admitted={admitted['binary_sha256']}; the producer "
            "refuses to emit a bundle that would not join the "
            "admitted 5-tuple. Rebuild the binary against the same "
            "source HEAD as the admitted cell, or re-admit the cell."
        )

    cell_root = resolve_cell_root(
        slug=slug, shape_point=shape_point,
        sku="rtx_5060_laptop", arch_dir="sm_120",
    )
    trace_dir = cell_root / f"trace_{shape_point}"
    if derived_sector_stream is None:
        raise RuntimeError(
            "sm120 helper requires an explicitly cleared staging destination "
            "for regenerated sector-stream bytes"
        )

    import cache_predictor as cp  # type: ignore[import-not-found]

    manifest_pairing_index = {
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": binary_sha,
        "kernel_signature_sha256": admitted["kernel_signature_sha256"],
        "source_commit": admitted["source_commit"],
        # Strict-arch ratchet: explicit sm_120 marker so the fitter's
        # arch-verification guard admits the join.
        "arch": "sm_120",
    }

    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        sector_stream, sector_stream_path = _generate_sector_stream(
            sources=sources,
            output_path=derived_sector_stream,
        )
        bundle = cp.predict_cache_hit_rate_from_admitted_sources(
            sources=sources,
            sector_stream=sector_stream,
            trace_dir=trace_dir,
            sector_stream_path=sector_stream_path,
            cache_model_capacity_bytes=l2_cache_size_bytes,
            manifest_pairing_index=manifest_pairing_index,
            device=_SM120_LAPTOP_DEVICE_NAME,
            sharing_locality_path=(
                trace_dir / "sharing_locality" / "sharing_locality.json"
            ),
            hrd_profile_path=(
                trace_dir / "hrd_profile" / "hrd_profile.json"
            ),
        )

    # `predict_cache_hit_rate` does not write the bundle; serialize
    # it here under the sm_120 cohort-bundles dir's
    # predictor_bundles/ subtree. The filename convention mirrors
    # the legacy `sm_86__<slug>__<shape>.json` shape, swapping the
    # arch prefix to `sm_120`.
    output_dir = _assert_staging_output(
        output_dir, operation="create predictor bundle directory",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = _assert_staging_output(
        bundle_path
        if bundle_path is not None
        else output_dir / f"sm_120__{slug}__{shape_point}.json",
        operation="write predictor bundle",
    )
    with bundle_path.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return bundle_path


def _sm120_template_unavailable_summary() -> dict:
    """Return the non-mutating result for this retired production path."""
    return unavailable_result(
        operation="sm120_activation_bundle_producer",
        arch="sm_120",
        output_root=None,
        n_admitted_cells=0,
        n_bundles_written=0,
        n_failures=0,
        bundles=[],
        failures=[],
        device=_SM120_LAPTOP_DEVICE_NAME,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help=(
            "Optional explicit output root below "
            "results/nvbit_capture_staging."
        ),
    )
    parser.add_argument(
        "--helper-output", type=Path, default=None,
        help="Optional CUDA helper path outside every Git worktree.",
    )
    parser.add_argument(
        "--label-admitted-path", type=Path, default=_ADMITTED_PATH,
        help=(
            "Path to the label_admitted.json driving the sm_120 cell set "
            "(default: data/local_labels/label_admitted.json). A caller "
            "(e.g. run_headline_cohort_predict_compare) that selected a "
            "non-default label corpus threads its own path here so the "
            "producer builds the SELECTED corpus, not the global admitted "
            "set."
        ),
    )
    args = parser.parse_args(argv)

    # Safety boundary: this module is retained as a source/evidence reader,
    # but its former production sm_120 emission lane is disabled. The gate is
    # deliberately before label loading, output planning, CUDA helper build,
    # and Git/probe subprocesses so a call cannot mutate or overwrite evidence.
    result = _sm120_template_unavailable_summary()
    sys.stderr.write(
        "sm120_activation_bundle_producer: "
        f"{result['typed_diagnostic']}: {result['typed_reason']}\n"
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 2

    # The code below is retained for historical reconstruction only. It is

    # Round-13 widening: drive the producer from every admitted
    # sm_120 row in `label_admitted.json` rather than the prior
    # 5-cell hardcoded `_AMENDED_COHORT_CELLS` list. This closes
    # AC-4's "30 joinable sm_120 L2 predictor bundles"
    # requirement (the cohort target the loop has reached).
    candidates = sorted(admitted.keys())  # deterministic order

    if args.output_root is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        output_root = _DEFAULT_OUTPUT_BASE / f"sm120_cohort_bundles_{ts}"
    else:
        output_root = args.output_root
    try:
        plan = _bundle_output_plan(
            candidates=candidates,
            output_root=output_root,
            helper_output=args.helper_output,
        )
    except (ProtectedPathError, ValueError) as exc:
        sys.stderr.write(
            f"sm120_activation_bundle_producer: output refusal: {exc}\n"
        )
        return 2
    # All output, derived-stream, and helper paths are cleared before live
    # CUDA or Git subprocesses.
    l2_size = _query_l2_cache_size_bytes(helper_bin=plan["helper"])
    head = _git_head()
    output_root = plan["output_root"]
    predictor_bundles_dir = plan["predictor_bundles_dir"]

    written: list[Path] = []
    failures: list[str] = []
    for slug, shape in candidates:
        admitted_row = admitted[(slug, shape)]
        cell_plan = plan["cells"][(slug, shape)]
        try:
            bp = _emit_bundle_for_cell(
                slug=slug,
                shape_point=shape,
                admitted=admitted_row,
                l2_cache_size_bytes=l2_size,
                output_dir=predictor_bundles_dir,
                head_commit=head,
                bundle_path=cell_plan["bundle_path"],
                derived_sector_stream=cell_plan["derived_sector_stream"],
            )
        except _HARD_TRACE_SOURCE_ERRORS:
            raise
        except (FileNotFoundError, RuntimeError) as exc:
            failures.append(f"{slug}/{shape}: {type(exc).__name__}: {exc}")
            sys.stderr.write(
                f"sm120_activation_bundle_producer: FAIL "
                f"{slug}/{shape}: {exc}\n"
            )
            continue
        written.append(bp)
        sys.stderr.write(f"sm120_activation_bundle_producer: wrote {bp}\n")
    if failures:
        sys.stderr.write(
            f"\nsm120_activation_bundle_producer: "
            f"{len(failures)} cell(s) failed:\n"
        )
        for f in failures:
            sys.stderr.write(f"  {f}\n")

    def _rel(p: Path) -> str:
        """Render ``p`` relative to the repo root when possible;
        fall back to the absolute path for off-repo (test) output
        directories."""
        try:
            return str(p.relative_to(_REPO_ROOT))
        except ValueError:
            return str(p)

    summary = {
        "output_root": _rel(output_root),
        "l2_cache_size_bytes": l2_size,
        "head_commit": head,
        "n_admitted_cells": len(candidates),
        "n_bundles_written": len(written),
        "n_failures": len(failures),
        "bundles": [_rel(p) for p in written],
        "failures": failures,
        "device": _SM120_LAPTOP_DEVICE_NAME,
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
