"""Synthetic-fixture builders for the comparison-runner and
verifier-hook test surfaces.

Every fixture this module emits is SYNTHETIC. Real NCU label cells
under ``data/local_labels/label_admitted.json`` are out of scope
for this round; the fixture builders construct hand-authored
admitted-shape predictor bundles and admitted-shape NCU label
cells so the runner / verifier can be exercised before
label stabilization lands. Tests that consume these fixtures MUST
include the literal substring ``synthetic`` in their function
name to make the synthetic-only nature explicit at the report
level (see plan synthetic-fixture discipline).

The bundles' shapes match
``workloads/simbench_bw/schemas/cache_predictor.schema.json`` and
the label cells' shapes match the admitted-cell shape that
``tools/local_labels/repeatability_aggregator.py`` emits, with the
addition of the synthetic ``profiler_version`` field that the
runner cross-checks against the contract's pinned
``ncu_metric_version`` (per the resolved contract-side pinning of
the metric version).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


_SYNTHETIC_BINARY_SHA = "0" * 64
_SYNTHETIC_KERNEL_SIG_SHA = "1" * 64
_SYNTHETIC_SOURCE_COMMIT = "deadbeef" * 5
_SYNTHETIC_PROFILER_VERSION = "ncu_2026.1.x"


def _make_synthetic_manifest_pairing_index(
    *,
    slug: str = "synthetic_kernel",
    shape_point: str = "trace_00",
    binary_sha256: str = _SYNTHETIC_BINARY_SHA,
    source_commit: str = _SYNTHETIC_SOURCE_COMMIT,
    kernel_signature_sha256: str = _SYNTHETIC_KERNEL_SIG_SHA,
) -> dict[str, str]:
    return {
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": binary_sha256,
        "source_commit": source_commit,
        "kernel_signature_sha256": kernel_signature_sha256,
    }


def _write_synthetic_sector_stream(
    tmp_path: Path,
    *,
    name: str = "sector_stream.jsonl",
    payload: bytes = b"{}\n{}\n",
) -> tuple[Path, str]:
    """Write a synthetic sector_stream sidecar at
    ``tmp_path / name`` and return ``(path, sha256_hex)``. The
    parent directory is created if absent so callers can pass a
    fresh per-fixture subdirectory without preflight mkdir."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / name
    p.write_bytes(payload)
    return p, hashlib.sha256(payload).hexdigest()


def _make_synthetic_input_sidecar_digests(
    tmp_path: Path,
) -> dict[str, dict]:
    """Mint a synthetic ``input_sidecar_digests`` dict whose
    ``sector_stream.sha256`` matches the on-disk file's actual
    sha256 (so the verifier's digest cross-check admits)."""
    p, sha = _write_synthetic_sector_stream(tmp_path)
    return {
        "sector_stream": {
            "path": str(p),
            "sha256": sha,
        }
    }


def _make_synthetic_admitted_predictor_bundle(
    tmp_path: Path,
    *,
    result_kind: str = "scalar",
    predicted_hit_rate: float = 0.5,
    hit_rate_lower: float = 0.4,
    hit_rate_upper: float = 0.6,
    underconfigured_bin_count: int = 0,
    admitted_bin_count: int = 5,
    typed_reason: str | None = None,
    manifest_pairing_index: dict | None = None,
    input_sidecar_digests: dict | None = None,
    device: str = "NVIDIA GeForce RTX 5060 Laptop GPU",
    cache_model_capacity_bytes: int = 33554432,
    bootstrap_seed: int = 0,
    epsilon: float = 0.01,
) -> dict[str, Any]:
    """Construct a synthetic predictor bundle that schema-validates
    against ``cache_predictor.schema.json`` for the requested arm.

    For the scalar arm the bootstrap CI is collapsed around the
    point estimate (``ci_low == ci_high == predicted_hit_rate``,
    ``ci_half_width == 0``); the synthetic shape exercises only the
    schema and provenance binding, not bootstrap variance. For the
    bounded arm the lower / upper / underconfigured / admitted
    fields are configurable. For the typed_unavailable arm the
    ``typed_reason`` is mandatory and the digests dict can be empty
    (the per-arm ``input_sidecar_digests.sector_stream`` clause
    relaxes for this arm)."""
    if manifest_pairing_index is None:
        manifest_pairing_index = _make_synthetic_manifest_pairing_index()
    if input_sidecar_digests is None:
        if result_kind == "typed_unavailable":
            input_sidecar_digests = {}
        else:
            input_sidecar_digests = (
                _make_synthetic_input_sidecar_digests(tmp_path)
            )
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "smith_eklund_lru_v1",
        "manifest_pairing_index": dict(manifest_pairing_index),
        "input_sidecar_digests": dict(input_sidecar_digests),
        "device": device,
        "cache_model_capacity_bytes": cache_model_capacity_bytes,
        "sector_size_bytes": 32,
        "bootstrap_resamples": 2000,
        "bootstrap_seed": bootstrap_seed,
        "epsilon": epsilon,
        "result_kind": result_kind,
    }
    if result_kind == "scalar":
        bundle["predicted_hit_rate"] = predicted_hit_rate
        bundle["ci_low"] = predicted_hit_rate
        bundle["ci_high"] = predicted_hit_rate
        bundle["ci_half_width"] = 0.0
        bundle["ci_confidence"] = 0.95
    elif result_kind == "bounded":
        bundle["hit_rate_lower"] = hit_rate_lower
        bundle["hit_rate_upper"] = hit_rate_upper
        bundle["underconfigured_bin_count"] = underconfigured_bin_count
        bundle["admitted_bin_count"] = admitted_bin_count
    elif result_kind == "typed_unavailable":
        if typed_reason is None:
            typed_reason = "cache_model_underconfigured"
        bundle["typed_reason"] = typed_reason
        bundle["underconfigured_bin_count"] = underconfigured_bin_count
        bundle["admitted_bin_count"] = admitted_bin_count
    else:
        raise ValueError(f"unknown result_kind {result_kind!r}")
    return bundle


def _make_synthetic_ncu_label_cell(
    *,
    slug: str = "synthetic_kernel",
    shape_point: str = "trace_00",
    binary_sha256: str = _SYNTHETIC_BINARY_SHA,
    source_commit: str = _SYNTHETIC_SOURCE_COMMIT,
    kernel_signature_sha256: str = _SYNTHETIC_KERNEL_SIG_SHA,
    arch: str = "sm_120",
    profiler_version: str = _SYNTHETIC_PROFILER_VERSION,
    ground_truth_pct: float = 50.0,
    cache_hit_rate_l1tex_sector_pct: float | None = None,
    l1tex_label_status: str = "mean_absent_label_missing",
    evidence_kind: str = "synthetic",
) -> dict[str, Any]:
    """Construct a synthetic admitted NCU label cell shape. The
    cell carries the seven join-key fields, ``profiler_version``
    (cross-checked against the contract's pinned
    ``ncu_metric_version``), ``cache_hit_rate_l2_sector_pct`` in
    [0, 100], the v4 ``cache_hit_rate_l1tex_sector_pct`` (nullable)
    + ``l1tex_label_status`` (closed enum) pair, and the
    ``evidence_kind`` classifier (default ``"synthetic"``). The
    runner normalizes the percent to a fraction in [0, 1] at point
    of use. Tests that exercise the live-evidence path explicitly
    override ``evidence_kind`` to ``"live"`` so the live-only
    smoke-test floor counts those cells; the default keeps
    synthetic cells from accidentally satisfying the floor. The
    default L1 shape is ``(None, "mean_absent_label_missing")`` so
    L2-focused tests don't have to specify L1 inputs; tests that
    need the L1 mean pass it explicitly and adjust
    ``l1tex_label_status`` to ``"mean_present"``."""
    return {
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": binary_sha256,
        "source_commit": source_commit,
        "kernel_signature_sha256": kernel_signature_sha256,
        "arch": arch,
        "profiler_version": profiler_version,
        "ncu_metric_name": "lts__t_sector_hit_rate.pct",
        "cache_hit_rate_l2_sector_pct": ground_truth_pct,
        "cache_hit_rate_l1tex_sector_pct": (
            cache_hit_rate_l1tex_sector_pct
        ),
        "l1tex_label_status": l1tex_label_status,
        "evidence_kind": evidence_kind,
    }


def _make_synthetic_run_summary_with_cache_predictor_path(
    tmp_path: Path,
    *,
    bundle: dict | None = None,
    bundle_filename: str = "cache_predictor_bundle.json",
    summary_filename: str | None = None,
    slug: str = "synthetic_kernel",
    shape_point: str = "trace_00",
    binary_sha256: str = _SYNTHETIC_BINARY_SHA,
    source_commit: str = _SYNTHETIC_SOURCE_COMMIT,
    kernel_signature_sha256: str = _SYNTHETIC_KERNEL_SIG_SHA,
    write_bundle: bool = True,
) -> tuple[Path, Path, dict]:
    """Build a synthetic ``trace_<shape>.json`` run summary that
    references a synthetic cache_predictor bundle.

    Layout (mirrors the canonical ``run_suite.py --mode trace``
    tree under ``results/<slug>/<sku>/sm_<N>/``)::

        <tmp_path>/
            trace_<shape>/
                cache_predictor/
                    cache_predictor_bundle.json   # the bundle
                    sector_stream.jsonl           # corroborating witness
            trace_<shape>.json                    # the run summary

    Returns ``(summary_path, bundle_path, bundle_dict)``. When
    ``write_bundle=False`` the ``bundle_path`` is set on the summary
    but the file is NOT written (exercises the
    ``cache_predictor_run_summary_missing_path`` defect surface)."""
    trace_dir = tmp_path / f"trace_{shape_point}"
    bundle_dir = trace_dir / "cache_predictor"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if bundle is None:
        bundle = _make_synthetic_admitted_predictor_bundle(
            bundle_dir,
            manifest_pairing_index=_make_synthetic_manifest_pairing_index(
                slug=slug,
                shape_point=shape_point,
                binary_sha256=binary_sha256,
                source_commit=source_commit,
                kernel_signature_sha256=kernel_signature_sha256,
            ),
        )
    bundle_path = bundle_dir / bundle_filename
    if write_bundle:
        import json
        bundle_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    summary = {
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": binary_sha256,
        "source_commit": source_commit,
        "kernel_signature_sha256": kernel_signature_sha256,
        "cache_predictor_bundle_path": str(bundle_path),
        "cache_predictor_source_commit": source_commit,
    }
    if summary_filename is None:
        summary_filename = f"trace_{shape_point}.json"
    summary_path = tmp_path / summary_filename
    import json
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    return summary_path, bundle_path, bundle


def _make_synthetic_hrd_profile_sibling(
    bundle_dir: Path,
    *,
    bundle_underconfigured: bool = False,
    typed_reason: str = "cache_model_underconfigured",
    underconfigured_bin_count: int = 0,
    admitted_bin_count: int = 5,
) -> Path:
    """Drop a sibling ``hrd_profile/hrd_profile.json`` next to the
    bundle so the verifier's bundle_underconfigured cross-check
    has something to read. The sibling contains ONLY structural
    fields (no modeled hit_rate / hit_count); the verifier reads
    only ``bundle_underconfigured`` and the bin-count metadata."""
    hrd_dir = bundle_dir.parent / "hrd_profile"
    hrd_dir.mkdir(parents=True, exist_ok=True)
    hrd_path = hrd_dir / "hrd_profile.json"
    payload: dict[str, Any] = {
        "bundle_underconfigured": bundle_underconfigured,
        "underconfigured_bin_count": underconfigured_bin_count,
        "admitted_bin_count": admitted_bin_count,
        "evidence": [
            {
                "scope": "bundle",
                "typed_reason": typed_reason,
                "underconfigured_bin_count": underconfigured_bin_count,
                "admitted_bin_count": admitted_bin_count,
            }
        ],
    }
    import json
    hrd_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    return hrd_path
