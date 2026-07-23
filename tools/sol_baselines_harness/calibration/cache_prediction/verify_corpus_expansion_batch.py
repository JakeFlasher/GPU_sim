"""Operator smoke script for `batch-NNN.toml` manifests.

Loads a corpus-expansion batch manifest from a caller-passed
path and validates it via the canonical batch loader
(`load_corpus_expansion_batch_manifest`) with the
sm_86-compatible slug set drawn from the in-tree
`host_arch_availability_v1.toml`. Prints the validated
manifest's batch_id, slug count, family stratification,
runtime sum, and the per-slug promotion entries on success;
exits non-zero with a typed reason on validation failure.

The verifier ALSO loads the operator-local
`.humanize/corpus-expansion/verified_dispatch_families.toml`
and passes its `family_status_map` to the batch loader so v1.1
`friendly_multi_slug` batches whose `dispatch_family_signature`
is unknown or `pending_verification` are rejected at the
operator gate. Legacy v1.0 batches remain accepted when the
verified-families manifest is absent or invalid; v1.1 batches
hard-fail with exit 3 in that case.

Usage:

    python verify_corpus_expansion_batch.py <path-to-batch-NNN.toml>

Acts as durable evidence for downstream rounds: a fresh
checkout can re-run the script against an operator-authored
`.humanize/corpus-expansion/batch-NNN.toml` without any
batch-specific glue.

This module is calibration-only and never imported by the
deployed harness path. The NCU quarantine scanner enforces
that contract.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from baselines_harness.calibration.cache_prediction.corpus_expansion_batch import (  # noqa: E402
    CORPUS_EXPANSION_BATCH_SCHEMA_VERSION,
    CorpusExpansionBatchError,
    load_corpus_expansion_batch_manifest,
)
from baselines_harness.calibration.cache_prediction.verified_dispatch_families import (  # noqa: E402
    VerifiedDispatchFamiliesError,
    load_verified_dispatch_families,
)


HOST_ARCH_MANIFEST_RELATIVE_PATH: str = (
    "scripts/baselines_harness/calibration/cache_prediction/"
    "host_arch_availability_v1.toml"
)

VERIFIED_DISPATCH_FAMILIES_RELATIVE_PATH: str = (
    ".humanize/corpus-expansion/verified_dispatch_families.toml"
)


def _load_host_arch_slug_sets(
    host_arch_manifest_path: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    """Return `(canonical_slug_set, host_arch_unsupported_slug_set)`
    parsed from `host_arch_availability_v1.toml`.

    `canonical_slug_set` is the set of canonical SOL-ExecBench
    slugs (the union of all 235 rows, regardless of host-arch
    availability). `host_arch_unsupported_slug_set` is the
    subset whose status differs from `sm86_compatible_cohort`.
    """
    data = tomllib.loads(
        host_arch_manifest_path.read_text(encoding="utf-8"),
    )
    canonical: set[str] = set()
    unsupported: set[str] = set()
    for row in data.get("problems", ()):
        slug = row["slug"]
        canonical.add(slug)
        if row["status"] != "sm86_compatible_cohort":
            unsupported.add(slug)
    return frozenset(canonical), frozenset(unsupported)


def verify_batch_manifest(
    batch_path: Path,
    host_arch_manifest_path: Path,
    verified_dispatch_families_path: Path,
) -> int:
    """Validate `batch_path` against the in-tree canonical +
    host-arch slug sets and print a structured summary.

    Cadence-uplift gating: the verified-dispatch-families
    manifest at `verified_dispatch_families_path` is loaded and
    its `family_status_map` is passed into the batch loader so
    that v1.1 `friendly_multi_slug` batches with an unknown or
    `pending_verification` `dispatch_family_signature` are
    rejected at this operator gate. A missing or invalid
    verified-families manifest is acceptable only for v1.0
    legacy batches; a v1.1 batch hits a hard exit-3 error when
    the verified-families manifest cannot be loaded.

    Returns `0` on success, `2` on any
    `CorpusExpansionBatchError`, `3` on IO failures for either
    the host-arch manifest or the verified-families manifest
    (the latter only when the batch is v1.1).
    """
    try:
        canonical, unsupported = _load_host_arch_slug_sets(
            host_arch_manifest_path,
        )
    except (OSError, tomllib.TOMLDecodeError, KeyError) as exc:
        print(
            f"error: failed to load host-arch manifest at "
            f"{host_arch_manifest_path!s}: {exc}",
            file=sys.stderr,
        )
        return 3

    family_status_map: dict[str, str] | None = None
    verified_load_error: VerifiedDispatchFamiliesError | None = None
    try:
        vdf_manifest = load_verified_dispatch_families(
            verified_dispatch_families_path,
        )
        family_status_map = dict(vdf_manifest.family_status_map)
    except VerifiedDispatchFamiliesError as exc:
        verified_load_error = exc

    try:
        manifest = load_corpus_expansion_batch_manifest(
            batch_path,
            canonical_slug_set=canonical,
            host_arch_unsupported_slug_set=unsupported,
            verified_dispatch_families=family_status_map,
        )
    except CorpusExpansionBatchError as exc:
        print(
            f"error: batch manifest validation failed for "
            f"{batch_path!s}: {exc.typed_reason}: "
            f"{exc.message}",
            file=sys.stderr,
        )
        return 2

    if (
        manifest.schema_version
        == CORPUS_EXPANSION_BATCH_SCHEMA_VERSION
        and verified_load_error is not None
    ):
        print(
            f"error: verified-dispatch-families manifest at "
            f"{verified_dispatch_families_path!s} could not be "
            f"loaded ({verified_load_error.typed_reason}: "
            f"{verified_load_error.message}); v1.1 batch "
            f"manifests require a valid verified-families "
            f"manifest for cadence-uplift gating.",
            file=sys.stderr,
        )
        return 3

    runtime_sum = sum(
        entry.estimated_runtime_minutes for entry in manifest.slugs
    )

    print(f"batch manifest: {batch_path}")
    print(f"  schema_version = {manifest.schema_version}")
    print(f"  batch_id = {manifest.batch_id}")
    print(f"  max_slugs = {manifest.max_slugs}")
    print(
        f"  max_estimated_runtime_minutes = "
        f"{manifest.max_estimated_runtime_minutes}"
    )
    print(f"  len(slugs) = {len(manifest.slugs)}")
    print(f"  per-slug runtime sum = {runtime_sum}")
    print(
        f"  family_stratification = "
        f"{dict(manifest.family_stratification)}"
    )
    print(f"  per-slug entries:")
    for entry in manifest.slugs:
        print(
            f"    - {entry.slug} | "
            f"cohort={entry.target_cohort} | "
            f"language={entry.target_language} | "
            f"runtime_min={entry.estimated_runtime_minutes}"
        )
    print(f"  promotion entries:")
    for promotion in manifest.predictor_design_coverage_promotions:
        print(
            f"    - {promotion.slug} -> "
            f"{promotion.target_cohort}"
        )
    print("  validation: OK")
    return 0


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a corpus-expansion batch manifest against "
            "the in-tree canonical + host-arch slug sets."
        ),
    )
    parser.add_argument(
        "batch_path",
        type=Path,
        help="path to batch-NNN.toml",
    )
    parser.add_argument(
        "--host-arch-manifest-path",
        type=Path,
        default=_REPO_ROOT / HOST_ARCH_MANIFEST_RELATIVE_PATH,
        help=(
            "path to host_arch_availability_v1.toml "
            "(default: in-tree manifest)"
        ),
    )
    parser.add_argument(
        "--verified-dispatch-families-path",
        type=Path,
        default=_REPO_ROOT / VERIFIED_DISPATCH_FAMILIES_RELATIVE_PATH,
        help=(
            "path to verified_dispatch_families.toml used to "
            "gate v1.1 friendly_multi_slug batches (default: "
            "in-tree operator-local "
            ".humanize/corpus-expansion/verified_dispatch_families.toml)"
        ),
    )
    args = parser.parse_args(argv)
    return verify_batch_manifest(
        args.batch_path,
        args.host_arch_manifest_path,
        args.verified_dispatch_families_path,
    )


def main(argv: list[str] | None = None) -> int:
    return _main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
