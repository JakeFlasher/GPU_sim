"""Stage operator measurement manifests from predictor trace JSONs.

The SWEEP_RUNBOOK requires `--measurement-manifests` JSON files carrying the
operator-pinned `kernel_signature_sha256` (and implicit `binary_sha256`).
Canonical roster paths are read-only inputs.  Active manifests are mirrored at:

    results/nvbit_capture_staging/measurement_manifests/
        workloads/<roster_source>/results/<slug>/<sku>/<arch_dir>/
        measurement_manifest_<shape_point>.json

The roster-aware path resolution is shared with
``activation_trace_orchestrator.py`` via
``tools/local_labels/roster_path_resolver.py``: portable_lcd rows
land under ``workloads/portable_lcd/results/...``; simbench_bw rows
land under ``workloads/simbench_bw/results/...``.

The predictor trace JSON sidecars at the same cell-root carry the
join-key fields (``slug``, ``shape_point``, ``binary_sha256``,
``kernel_signature_sha256``) plus the orchestrator's full device-
property field set when the orchestrator wrote them.

This script's contract (post-Round-7):

1. If a full ``measurement_manifest_<shape>.json`` already exists
   at the target path (the orchestrator wrote it), PRESERVE it.
   Merge only the join-key fields from the trace JSON if any are
   missing from the manifest. Do NOT replace the rich payload with
   a 4-field stub.

2. If no manifest exists, BUILD a full schema-v2 manifest from the
   trace JSON plus the SKU device-property table. Include
   ``cuda_version``, ``driver_version``, ``gpu_sku``, ``gpu_uuid``,
   ``sm_count``, ``total_global_mem_bytes``, ``device_name``,
   ``compute_capability_major``, ``compute_capability_minor``,
   ``compiler_flags``, ``kb_capability_card``, ``timestamp_utc``,
   ``shape_point``, ``binary_sha256``, ``kernel_signature_sha256``.

3. A register-cap build-VARIANT candidate
   (``<slug>/<shape_point>/register_cap_<cap>``) stages under a
   NON-COLLIDING per-variant subdir
   ``<cell_root>/variants/register_cap_<cap>/`` (reading
   ``trace_<shape>.json`` and writing
   ``measurement_manifest_<shape>.json`` there), so two caps of the
   same ``(slug, shape_point)`` never share a path and a base cell's
   cell-root paths stay byte-compatible. The variant trace JSON must
   declare the SAME canonical ``register_cap`` / ``variant_id`` as the
   requested cap (validated via the shared ``variant_identity``
   normalizer with a strict type pre-check); the staged variant
   manifest carries the canonical pair (declared optional in
   ``measurement_manifest.schema.json``).

Usage:

    python3 tools/local_labels/stage_measurement_manifests_from_predictor_trace.py \\
        --candidates <slug>/<shape>[/register_cap_<cap>][,<slug>/<shape>...] \\
        --sku rtx_5060_laptop \\
        --arch-dir sm_120

The script is idempotent for a given trace input and SKU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.local_labels.roster_path_resolver import resolve_cell_root  # noqa: E402
from tools.local_labels.variant_identity import variant_identity  # noqa: E402
from tools.local_labels.protected_paths import (  # noqa: E402
    ProtectedPathError,
    assert_write_allowed,
)


_STAGING_ROOT = _REPO_ROOT / "results" / "nvbit_capture_staging"
_DEFAULT_OUTPUT_ROOT = _STAGING_ROOT / "measurement_manifests"


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


def _staging_artifact_dir(
    source_artifact_dir: Path, *, output_root: Path | str,
) -> Path:
    """Mirror a read-only roster artifact path under active staging."""

    root = _assert_staging_output(output_root, operation="select manifest staging root")
    try:
        relative = source_artifact_dir.resolve(strict=False).relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise ProtectedPathError(
            f"source artifact path escapes repository: {source_artifact_dir}"
        ) from exc
    return _assert_staging_output(
        root / relative, operation="prepare staged manifest directory",
    )


# Mirror of `activation_trace_orchestrator._DEVICE_PROPS_BY_SKU` for
# the staging path. The activation orchestrator emits the rich
# manifest via the same constants; this staging path uses them only
# when NO orchestrator-written manifest already exists at the
# target path.
_RTX_3090_TI_PROPS = {
    "cuda_version": "13.2",
    # NVIDIA driver version observed on the 3090 Ti host. The CUDA
    # toolkit string `"13.2"` belongs in `cuda_version`, NOT here; the
    # earlier `driver_version = "13.2"` was that toolkit value misrouted
    # into the driver slot. The committed sm_86 label sidecars + the
    # device-properties manifest's 3090 Ti entry carry `595.71.05`
    # (mirrors the 5060 entry below); a staged manifest's
    # consumer-facing `driver_version` must match.
    "driver_version": "595.71.05",
    "gpu_uuid": "GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c",
    "sm_count": 84,
    "total_global_mem_bytes": 25282478080,
    "compute_capability_major": 8,
    "compute_capability_minor": 6,
    "device_name": "NVIDIA GeForce RTX 3090 Ti",
    "gpu_sku": "rtx_3090_ti",
}

_RTX_5060_LAPTOP_PROPS = {
    "cuda_version": "13.2",
    "driver_version": "595.71.05",
    "gpu_uuid": "GPU-7106d751-e907-1dd4-6c9a-ba17bb7a606c",
    "sm_count": 26,
    "total_global_mem_bytes": 8082096128,
    "compute_capability_major": 12,
    "compute_capability_minor": 0,
    "device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "gpu_sku": "rtx_5060_laptop",
}

_DEVICE_PROPS_BY_SKU: dict[str, dict] = {
    "rtx_3090_ti": _RTX_3090_TI_PROPS,
    "rtx_5060_laptop": _RTX_5060_LAPTOP_PROPS,
}

# Canonical portable_lcd compiler flags (LCD discipline, see
# `workloads/portable_lcd/common.mk:14-103`). Identical to the
# orchestrator's `_build_activation_manifest_payload` to keep both
# staging surfaces emitting the same flag set.
_COMPILER_FLAGS_LCD_FULL: list[str] = [
    "-gencode", "arch=compute_80,code=sm_86",
    "-gencode", "arch=compute_80,code=sm_90",
    "-gencode", "arch=compute_80,code=sm_120",
    "-gencode", "arch=compute_80,code=compute_80",
]

_KB_CAPABILITY_CARD_LCD: str = (
    "references/NVIDIA_Manuals/blackwell_compatibility/"
    "01_blackwell-architecture-compatibility.md"
)


def _build_full_manifest_from_trace(
    *, trace: dict, sku: str, shape_point: str,
) -> dict:
    """Construct a full schema-v2 manifest payload from the trace
    JSON plus the SKU's device-property record. Used only when no
    full manifest exists at the target path.
    """
    if sku not in _DEVICE_PROPS_BY_SKU:
        raise ValueError(
            f"unknown sku {sku!r}; expected one of "
            f"{sorted(_DEVICE_PROPS_BY_SKU.keys())}"
        )
    props = _DEVICE_PROPS_BY_SKU[sku]
    return {
        "schema_version": 2,
        "slug": str(trace["slug"]),
        "mode": "trace",
        "warm_up_iters": 0,
        "repeat_iters": 1,
        "cv_pct": 0.0,
        "cuda_version": props["cuda_version"],
        "driver_version": props["driver_version"],
        "gpu_sku": props["gpu_sku"],
        "gpu_uuid": props["gpu_uuid"],
        "sm_count": props["sm_count"],
        "total_global_mem_bytes": props["total_global_mem_bytes"],
        "device_name": props["device_name"],
        "compute_capability_major": props["compute_capability_major"],
        "compute_capability_minor": props["compute_capability_minor"],
        "binary_sha256": str(trace["binary_sha256"]),
        "kernel_signature_sha256": str(trace["kernel_signature_sha256"]),
        "compiler_flags": list(_COMPILER_FLAGS_LCD_FULL),
        "kb_capability_card": _KB_CAPABILITY_CARD_LCD,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "shape_point": shape_point,
    }


_RICH_MANIFEST_DEVICE_FIELDS: tuple[str, ...] = (
    "cuda_version", "driver_version", "gpu_sku", "device_name",
)


def _existing_manifest_is_rich(existing: dict) -> bool:
    """A pre-Round-7 staging stub carries only the four join-key
    fields plus ``schema_version``. A rich orchestrator-written
    manifest carries the device-property field set. Returns True
    iff the manifest carries the rich device fields; the staging
    step uses this to decide between preserve+merge and full
    rebuild.
    """
    return all(field in existing for field in _RICH_MANIFEST_DEVICE_FIELDS)


def _merge_join_keys_into_existing(
    *, existing: dict, trace: dict, shape_point: str,
) -> dict:
    """Return a copy of ``existing`` with the join-key fields filled
    from ``trace`` when missing. The existing manifest's device
    fields (`driver_version`, `gpu_sku`, etc.) are preserved
    verbatim — the orchestrator's rich manifest IS the source of
    truth for those, and the staging step MUST NOT overwrite it
    with a minimal stub.
    """
    merged = dict(existing)
    merged.setdefault("schema_version", 2)
    merged["slug"] = str(trace["slug"])
    merged["shape_point"] = shape_point
    merged["binary_sha256"] = str(trace["binary_sha256"])
    merged["kernel_signature_sha256"] = str(trace["kernel_signature_sha256"])
    return merged


def _canonical_variant_or_raise(register_cap: int, where: str) -> str:
    """Return the canonical ``register_cap_<cap>`` string for ``register_cap``,
    routing through the SINGLE shared normalizer so a negative / out-of-domain
    cap fails closed identically to every other producer + consumer. ``where``
    names the caller for the diagnostic."""
    variant_id, _cap, err = variant_identity({"register_cap": register_cap})
    if err is not None:
        raise ValueError(
            f"stage_measurement_manifests: {where} register_cap "
            f"{register_cap!r} is not a valid build-variant cap: {err}"
        )
    return variant_id


def _trace_variant_matches_or_raise(
    *, trace: dict, register_cap: int, canonical_variant_id: str,
    trace_json: Path,
) -> None:
    """Fail closed unless the variant trace JSON declares the SAME canonical
    build-variant identity as the requested cap.

    The trace JSON is operator input, so a STRICT TYPE PRE-CHECK fronts the
    tolerate-absence normalizer (mirrors the label_driver.sh manifest gate): a
    present-but-malformed ``register_cap`` (JSON string / bool) or
    ``variant_id`` (non-string / empty) is a declared-but-INVALID value that
    must ERROR, not be read as absent and silently treated as a base trace. A
    base trace (no variant fields) placed at a variant path, a foreign cap, an
    inconsistent pair, or a noncanonical alias all fail closed BEFORE any
    manifest is written."""
    if "register_cap" in trace and (
        not isinstance(trace["register_cap"], int)
        or isinstance(trace["register_cap"], bool)
    ):
        raise ValueError(
            f"stage_measurement_manifests: trace JSON at {trace_json} has a "
            f"register_cap that is present but not an integer: "
            f"{trace['register_cap']!r}"
        )
    if "variant_id" in trace and (
        not isinstance(trace["variant_id"], str) or not trace["variant_id"]
    ):
        raise ValueError(
            f"stage_measurement_manifests: trace JSON at {trace_json} has a "
            f"variant_id that is present but not a non-empty string: "
            f"{trace['variant_id']!r}"
        )
    record: dict = {}
    if "register_cap" in trace:
        record["register_cap"] = trace["register_cap"]
    if "variant_id" in trace:
        record["variant_id"] = trace["variant_id"]
    trace_variant_id, trace_cap, err = variant_identity(record)
    if err is not None:
        raise ValueError(
            f"stage_measurement_manifests: trace JSON at {trace_json} has an "
            f"inconsistent build-variant identity: {err}"
        )
    if trace_cap is None:
        raise ValueError(
            f"stage_measurement_manifests: requested build-variant "
            f"{canonical_variant_id!r} but the trace JSON at {trace_json} "
            f"declares no build-variant identity (a base trace at a variant "
            f"path); refusing to stage a variant manifest from a base trace"
        )
    if trace_variant_id != canonical_variant_id:
        raise ValueError(
            f"stage_measurement_manifests: trace JSON at {trace_json} declares "
            f"build-variant {trace_variant_id!r} but the requested variant is "
            f"{canonical_variant_id!r}; refusing to stage a manifest for a "
            f"mismatched build variant"
        )


def _prepare_one(
    slug: str, shape_point: str, sku: str, arch_dir: str,
    register_cap: int | None = None,
    *, output_root: Path | str = _DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, dict]:
    """Read one canonical trace and prepare a staging payload without writing.

    A base cell (``register_cap is None``) stages ``trace_<shape>.json`` ->
    ``measurement_manifest_<shape>.json`` directly under the cell-root, exactly
    as before (byte-compatible). A register-cap build-variant cell stages under
    a NON-COLLIDING per-variant subdir ``variants/register_cap_<cap>/`` so two
    caps of the same ``(slug, shape_point)`` never share a trace/manifest path;
    the staged variant manifest additionally carries the canonical
    ``register_cap`` / ``variant_id`` copied from the variant trace JSON."""
    cell_root = resolve_cell_root(
        slug=slug, shape_point=shape_point, sku=sku, arch_dir=arch_dir,
    )
    if register_cap is None:
        source_artifact_dir = cell_root
        canonical_variant_id: str | None = None
    else:
        canonical_variant_id = _canonical_variant_or_raise(
            register_cap, "candidate"
        )
        source_artifact_dir = cell_root / "variants" / canonical_variant_id
    trace_json = source_artifact_dir / f"trace_{shape_point}.json"
    if not trace_json.is_file():
        raise FileNotFoundError(
            f"stage_measurement_manifests: trace JSON not found at "
            f"{trace_json}; the cell's predictor pipeline must run "
            f"before manifest staging"
        )
    with trace_json.open(encoding="utf-8") as fh:
        trace = json.load(fh)
    for key in (
        "slug", "shape_point", "binary_sha256",
        "kernel_signature_sha256",
    ):
        if not trace.get(key):
            raise ValueError(
                f"stage_measurement_manifests: trace JSON at "
                f"{trace_json} is missing or empty for field "
                f"{key!r}; the trace JSON must carry all four join-"
                f"key fields"
            )
    # Fail closed on trace-identity drift: the trace JSON found at the
    # requested cell-root MUST describe the REQUESTED (slug, shape_point).
    # A stale or foreign sidecar (e.g. copied from another cell, or whose
    # payload shape_point no longer matches the requested name) would
    # otherwise stage a manifest for the WRONG cell — or under the wrong
    # measurement_manifest_<shape>.json name, since shape_point is taken from
    # the payload below — letting the live sweep run with missing/stale
    # provenance. Verifying identity here makes that reassignment a no-op.
    if (
        str(trace["slug"]) != slug
        or str(trace["shape_point"]) != shape_point
    ):
        raise ValueError(
            f"stage_measurement_manifests: trace JSON at {trace_json} has "
            f"identity (slug={trace['slug']!r}, "
            f"shape_point={trace['shape_point']!r}) that does not match the "
            f"requested (slug={slug!r}, shape_point={shape_point!r}); refusing "
            f"to stage a manifest for a mismatched/foreign trace sidecar"
        )
    shape_point = str(trace["shape_point"])
    # The trace at a variant path must declare the SAME canonical build-variant
    # identity as the requested cap; a base / foreign / malformed / noncanonical
    # trace variant fails closed BEFORE any manifest is written.
    if register_cap is not None:
        assert canonical_variant_id is not None  # set in the variant branch
        _trace_variant_matches_or_raise(
            trace=trace, register_cap=register_cap,
            canonical_variant_id=canonical_variant_id, trace_json=trace_json,
        )
    source_manifest_path = (
        source_artifact_dir / f"measurement_manifest_{shape_point}.json"
    )
    target_artifact_dir = _staging_artifact_dir(
        source_artifact_dir, output_root=output_root,
    )
    manifest_path = _assert_staging_output(
        target_artifact_dir / f"measurement_manifest_{shape_point}.json",
        operation="prepare staged measurement manifest",
    )
    if source_manifest_path.is_file():
        with source_manifest_path.open(encoding="utf-8") as fh:
            existing = json.load(fh)
        if _existing_manifest_is_rich(existing):
            # PRESERVE the orchestrator-written rich manifest;
            # merge in any missing join-key fields without
            # disturbing device provenance. Round-6 regression
            # (Codex Round 6 review): pre-fix the staging step
            # replaced the rich manifest with a 4-field stub,
            # dropping `driver_version` / `gpu_sku` / device fields
            # the sweep harness consumes.
            manifest = _merge_join_keys_into_existing(
                existing=existing, trace=trace, shape_point=shape_point,
            )
        else:
            # Existing manifest is a pre-Round-7 4-field stub; the
            # device fields are missing. UPGRADE it in place by
            # building the full payload from the trace + SKU
            # device-props. The join-key fields stay byte-identical
            # because both forms read them from the same trace.
            manifest = _build_full_manifest_from_trace(
                trace=trace, sku=sku, shape_point=shape_point,
            )
    else:
        # No prior manifest at the target path: build the full
        # schema-v2 payload from the trace JSON + SKU device-props
        # so the sweep harness sees a complete manifest.
        manifest = _build_full_manifest_from_trace(
            trace=trace, sku=sku, shape_point=shape_point,
        )
    if register_cap is not None:
        # Copy the canonical build-variant identity into the staged manifest so
        # the driver's --measurement-manifest gate + the admission/join path see
        # the cap (validated against the schema's optional register_cap /
        # canonical variant_id). Set AFTER the build/preserve/upgrade dispatch
        # so a variant manifest always carries the pair regardless of any
        # pre-existing manifest at the target path.
        manifest["register_cap"] = register_cap
        manifest["variant_id"] = canonical_variant_id
    return manifest_path, manifest


def _write_prepared_manifest(path: Path, manifest: dict) -> Path:
    path = _assert_staging_output(path, operation="write staged measurement manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _stage_one(
    slug: str, shape_point: str, sku: str, arch_dir: str,
    register_cap: int | None = None,
    *, output_root: Path | str = _DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Compatibility wrapper for a single guarded staging write."""

    path, manifest = _prepare_one(
        slug,
        shape_point,
        sku,
        arch_dir,
        register_cap,
        output_root=output_root,
    )
    return _write_prepared_manifest(path, manifest)


def _parse_arch_subpath(arch_subpath: str) -> tuple[str, str]:
    """Parse the legacy ``--arch-dir`` value into ``(sku, arch_dir)``.

    Round 7 accepts BOTH the legacy combined form
    (``rtx_5060_laptop/sm_120``) and the split form via separate
    ``--sku`` and ``--arch-dir`` flags. The combined form is kept
    for backwards compatibility with prior round commands.
    """
    parts = arch_subpath.strip("/").split("/")
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 1 and parts[0] in _DEVICE_PROPS_BY_SKU:
        # ``--arch-dir rtx_5060_laptop`` alone implies the cohort's
        # canonical sm_120 directory for the laptop SKU.
        return parts[0], "sm_120" if parts[0] == "rtx_5060_laptop" else "sm_86"
    raise ValueError(
        f"stage_measurement_manifests: --arch-dir {arch_subpath!r} "
        f"could not be parsed as <sku>/<arch_dir> or a known sku; "
        f"pass --sku and --arch-dir separately instead."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage operator measurement manifests for the SWEEP_"
            "RUNBOOK live sweep by reading the predictor trace "
            "JSONs and emitting/merging the full schema-v2 "
            "measurement_manifest_<shape>.json."
        )
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help=(
            "Comma-separated list of <slug>/<shape_point> tuples, each "
            "optionally suffixed with a register-cap build-variant token "
            "(<slug>/<shape_point>/register_cap_<cap>) to stage a per-cap "
            "variant manifest under variants/register_cap_<cap>/. Example: "
            "dram_stream_stride_mix/trace_l2_accuracy_xl,"
            "register_spill_local_mem/trace_00/register_cap_64"
        ),
    )
    parser.add_argument(
        "--sku",
        type=str,
        default=None,
        choices=sorted(_DEVICE_PROPS_BY_SKU.keys()),
        help=(
            "Target GPU SKU. When omitted, the legacy combined "
            "`--arch-dir <sku>/<arch>` value is parsed for the "
            "SKU and arch components."
        ),
    )
    parser.add_argument(
        "--arch-dir",
        type=str,
        default="rtx_3090_ti/sm_86",
        help=(
            "Architecture-bound subpath under the roster-resolved "
            "cell-root. Legacy combined form: "
            "`rtx_5060_laptop/sm_120`. Split form (preferred): "
            "pass `--sku rtx_5060_laptop --arch-dir sm_120`."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT,
        help=(
            "Guarded staging root below results/nvbit_capture_staging. "
            "Roster-resolved portable/simbench result trees are read-only."
        ),
    )
    args = parser.parse_args(argv)

    if args.sku is None:
        sku, arch_dir = _parse_arch_subpath(args.arch_dir)
    else:
        sku = args.sku
        arch_dir = args.arch_dir
        if "/" in arch_dir:
            sys.stderr.write(
                f"stage_measurement_manifests: when --sku is "
                f"provided, --arch-dir must be the arch component "
                f"only (e.g., sm_120), got {arch_dir!r}\n"
            )
            return 2

    candidates: list[tuple[str, str, int | None]] = []
    for entry in args.candidates.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split("/")
        register_cap: int | None
        if len(parts) == 2:
            slug, shape_point = parts[0].strip(), parts[1].strip()
            register_cap = None
        elif len(parts) == 3:
            slug, shape_point, suffix = (p.strip() for p in parts)
            # Normalize the optional register-cap build-variant suffix through
            # the SINGLE shared normalizer; fail closed on a noncanonical alias
            # (register_cap_064), negative, or otherwise malformed suffix. An
            # empty / non-register_cap suffix yields register_cap is None, which
            # is also a malformed 3-part candidate here.
            _vid, register_cap, err = variant_identity({"variant_id": suffix})
            if err is not None or register_cap is None:
                detail = f": {err}" if err is not None else ""
                sys.stderr.write(
                    f"stage_measurement_manifests: invalid build-variant "
                    f"suffix in candidate {entry!r}; expected "
                    f"register_cap_<cap> (canonical decimal, no leading "
                    f"zeros){detail}\n"
                )
                return 2
        else:
            sys.stderr.write(
                f"stage_measurement_manifests: invalid candidate "
                f"{entry!r}; expected <slug>/<shape_point> or "
                f"<slug>/<shape_point>/register_cap_<cap>\n"
            )
            return 2
        if not slug or not shape_point:
            sys.stderr.write(
                f"stage_measurement_manifests: invalid candidate "
                f"{entry!r}; slug and shape_point must be non-empty\n"
            )
            return 2
        candidates.append((slug, shape_point, register_cap))

    prepared: list[tuple[Path, dict]] = []
    for slug, shape_point, register_cap in candidates:
        try:
            prepared.append(
                _prepare_one(
                    slug,
                    shape_point,
                    sku,
                    arch_dir,
                    register_cap,
                    output_root=args.output_root,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(
                f"stage_measurement_manifests: {exc}\n"
            )
            return 2
    destinations = [path for path, _manifest in prepared]
    if len(destinations) != len(set(destinations)):
        sys.stderr.write(
            "stage_measurement_manifests: duplicate candidates resolve to "
            "the same staging destination; refusing before any write\n"
        )
        return 2
    # Every input, payload, and destination is now known and cleared.  Only
    # this second phase mutates staging.
    written = [
        _write_prepared_manifest(path, manifest)
        for path, manifest in prepared
    ]
    sys.stderr.write(
        f"stage_measurement_manifests: wrote "
        f"{len(written)} manifest(s):\n"
    )
    for path in written:
        sys.stderr.write(f"  {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
