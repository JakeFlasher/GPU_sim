"""Batch driver that walks the landed `baselines/` tree, builds
`LedgerCellSpec` rows for every discovered `*.cu` source, and
invokes `run_calibration_sweep()` to produce a `LabelLedger`
with full NCU-provenance fields populated on every captured
entry.

The driver is calibration-only. It must NOT be imported from
the deployed harness path (`scripts/baselines_harness/run.py`,
`scripts/baselines_harness/report.py`, the surface plugins, or
anywhere outside `calibration/cache_prediction/`); the NCU
quarantine scanner enforces that contract.

CLAUDE.md Rule 1 carve-out: NCU may be invoked here as the
offline ground-truth source on the sm_86 calibration host. The
emitted ledger is consumed by the deployed cache predictor's
isotonic calibrator at deployment time; the deployed pipeline
does not depend on NCU.

CLAUDE.md Rule 2: This driver never mutates clock policy. The
`clock_policy_observed` provenance field records the actual
observed state (currently the `--clock-control none` argv pin)
rather than altering it.

Discovery convention: `baselines/<category>/<slug>/<language>/`
where `<category>` ∈ {`L1`, `L2`, `Quant`, `FIB`} and
`<language>` ∈ {`cuda_cpp`, `cutlass`}. A landed source is one
that carries at least a `kernel.cu` (or any `*.cu`) and a
`solution.json`. The driver discovers them without compiling;
the operator-built probe binary for cell `<slug>` is expected
at `<probe_binary_root>/<slug>` (default:
`nvbit_tracer/metric_extractor/tests/calibration_cohort_b/`).

For discovered slugs that do not have a corresponding probe
binary on disk, the driver emits a `LedgerCellSpec` with
`binary_path=None` and an explicit
`probe_binary_missing_for_slug` unavailable reason; the sweep
folds those into `[[unavailable]]` rows rather than aborting
the entire batch.
"""

from __future__ import annotations

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from sol_baselines_harness.calibration.cache_prediction.accuracy_diagnostic import (
    STATUS_HOST_ARCH_UNSUPPORTED,
    HostArchAvailabilityEntry,
    PerProblemRecordSchemaError,
    load_host_arch_availability_manifest,
)
from sol_baselines_harness.calibration.cache_prediction.calibration_sweep import (
    CalibrationSweepError,
    LabelLedger,
    LedgerCellSpec,
    LedgerEntry,
    LedgerUnavailable,
    build_sweep_specs_from_manifest,
    load_ledger_from_toml,
    run_calibration_sweep,
    serialize_ledger_to_toml,
    validate_label_ledger,
)
from sol_baselines_harness.calibration.cache_prediction.corpus_manifest import (
    load_calibration_corpus_manifest,
)


_HOST_ARCH_SM86_COMPATIBLE: str = "sm86_compatible_cohort"
_HOST_ARCH_MANIFEST_RELATIVE_SUBPATH: str = (
    "scripts/baselines_harness/calibration/"
    "cache_prediction/host_arch_availability_v1.toml"
)
_CALIBRATION_CORPUS_RELATIVE_SUBPATH: str = (
    "scripts/baselines_harness/calibration/"
    "cache_prediction/calibration_corpus.toml"
)


HARVEST_BATCH_TYPED_REASONS: frozenset[str] = frozenset({
    "harvest_batch_no_landed_sources",
    "harvest_batch_baselines_root_missing",
    "probe_binary_missing_for_slug",
    "harvest_batch_host_arch_manifest_missing",
    "harvest_batch_slug_not_in_host_arch_manifest",
    "harvest_batch_duplicate_cell_name",
    "label_stability_drift_exceeded",
    "label_stability_sentinel_cell_missing",
    "label_stability_sentinel_spec_unavailable",
    "label_stability_duplicate_sentinel_spec",
})


# The predeclared 14-cell cohort-A label-stability sentinel
# subset. Comparison against this fixed set on a fresh harvest
# rerun flags hit-rate drift that exceeds 1 percentage point
# (0.01 on the [0, 1] fraction), surfacing host/toolchain
# regressions before they corrupt the full ledger.
#
# Membership is drawn from the measured cohort-A entries in the
# in-tree `label_ledger_v1_full_partition.toml`. The 14 cells
# below are the foundational kernels (7 original calibration
# kernels + 3 smoke loops + 3 stride/vector variants + 1
# hostile-cohort smoke). The set is intentionally smaller than
# the full measured cohort-A so that a sentinel rerun is fast
# (operator-budget friendly) and that the choice of cells is
# pinned in code rather than re-derived per-run.
LABEL_STABILITY_SENTINEL_CELLS: tuple[str, ...] = (
    "calib_broadcast_global",
    "calib_cg_cache_bypass",
    "calib_coalesced_32x4B",
    "calib_coalesced_32x8B",
    "calib_cpasync_global_to_shared",
    "calib_divergent_lanes",
    "calib_hostile_gemm_stream_smoke",
    "calib_layernorm_fwd_bf16",
    "calib_smoke_broadcast_loop",
    "calib_smoke_capacity_overflow_loop",
    "calib_smoke_coalesced_loop",
    "calib_stride2_32x4B",
    "calib_vector_32x16B",
    "calib_vector_cross_32B",
)


# Default drift tolerance for `verify_label_stability()`: 1
# percentage point on the [0, 1] hit-rate fraction. The plan
# pins this; the helper accepts a parameter override only for
# unit-test convenience.
_DEFAULT_DRIFT_TOLERANCE_FRACTION: float = 0.01


class HarvestBatchError(Exception):
    """Typed error raised by the batch driver. Mirrors the
    typed-error pattern used elsewhere in the calibration
    package. The `typed_reason` is drawn from the closed set
    in `HARVEST_BATCH_TYPED_REASONS`.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in HARVEST_BATCH_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; "
                f"expected one of "
                f"{sorted(HARVEST_BATCH_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


_CANONICAL_CATEGORIES: tuple[str, ...] = ("L1", "L2", "Quant", "FIB")
_CANONICAL_LANGUAGES: tuple[str, ...] = ("cuda_cpp", "cutlass")
_DEFAULT_PROBE_BINARY_SUBPATH: str = (
    "nvbit_tracer/metric_extractor/tests/calibration_cohort_b"
)


@dataclass(frozen=True)
class LandedKernelSource:
    """A landed *.cu baseline discovered by the batch driver.

    Each instance carries enough provenance for the sweep to
    construct a `LedgerCellSpec`: the canonical problem slug,
    the category that owns it, the implementation language, and
    the absolute path to the kernel source file. The path
    `solution_json_path` is the in-tree `solution.json` next to
    the kernel; `kernel_source_path` is the `*.cu` file that
    will be hashed for `kernel_signature_sha256` provenance.
    """

    slug: str
    category: str
    language: str
    kernel_source_path: Path
    solution_json_path: Path


def discover_landed_baselines(
    repo_root: Path,
) -> tuple[LandedKernelSource, ...]:
    """Walk `<repo_root>/baselines/{L1,L2,Quant,FIB}/<slug>/{cuda_cpp,cutlass}/`
    and return one `LandedKernelSource` per directory containing
    both a `*.cu` source and a `solution.json`.

    Skips category dirs that do not exist (a fresh checkout may
    not have every category populated). Each `<slug>` directory
    may carry one or both language variants; both are emitted
    when present. Within a language directory the kernel source
    is preferred in this order: `kernel.cu`, then any single
    `*.cu`; multiple `*.cu` files trigger no special handling
    (the first one in `sorted()` order is used).

    Raises `HarvestBatchError("harvest_batch_baselines_root_missing", ...)`
    when `<repo_root>/baselines/` does not exist.
    """
    baselines_root = repo_root / "baselines"
    if not baselines_root.is_dir():
        raise HarvestBatchError(
            "harvest_batch_baselines_root_missing",
            f"baselines root does not exist on disk: "
            f"{str(baselines_root)!r}",
        )
    landed: list[LandedKernelSource] = []
    for category in _CANONICAL_CATEGORIES:
        category_root = baselines_root / category
        if not category_root.is_dir():
            continue
        for slug_dir in sorted(category_root.iterdir()):
            if not slug_dir.is_dir():
                continue
            slug = slug_dir.name
            for language in _CANONICAL_LANGUAGES:
                language_dir = slug_dir / language
                if not language_dir.is_dir():
                    continue
                kernel_source = _find_kernel_source(language_dir)
                if kernel_source is None:
                    continue
                solution_json = language_dir / "solution.json"
                if not solution_json.is_file():
                    continue
                landed.append(LandedKernelSource(
                    slug=slug,
                    category=category,
                    language=language,
                    kernel_source_path=kernel_source,
                    solution_json_path=solution_json,
                ))
    return tuple(landed)


def _find_kernel_source(language_dir: Path) -> Path | None:
    """Return the canonical `*.cu` source for a language dir.

    Preference order:
      1. `kernel.cu` (the convention from the seed templates).
      2. The lexicographically-first `*.cu` if `kernel.cu` is
         absent (some hand-ports split the kernel across
         multiple `*.cu` files).
    Returns None when no `*.cu` is present.
    """
    canonical = language_dir / "kernel.cu"
    if canonical.is_file():
        return canonical
    cus = sorted(language_dir.glob("*.cu"))
    if not cus:
        return None
    return cus[0]


def build_sweep_specs_from_landed_sources(
    landed_sources: Iterable[LandedKernelSource],
    *,
    probe_binary_root: Path,
) -> tuple[LedgerCellSpec, ...]:
    """For each discovered `LandedKernelSource`, produce a
    `LedgerCellSpec` that the calibration sweep can consume.

    The probe binary for slug `<slug>` is expected at
    `<probe_binary_root>/<slug>`; when absent the spec is
    emitted with `binary_path=None` and the unavailable reason
    `probe_binary_missing_for_slug`, which the sweep folds into
    `[[unavailable]]` rows rather than aborting.

    The driver carries cuda_cpp + cutlass language variants as
    distinct sources under the same `slug`. The calibration
    sweep keys on `cell_name`, so when both language variants
    exist for the same slug only the first discovered one
    (cuda_cpp before cutlass per `_CANONICAL_LANGUAGES`) is
    emitted; the operator can run a second batch targeted at
    the alternate language by passing a filtered
    `landed_sources` iterable.
    """
    seen_slugs: set[str] = set()
    specs: list[LedgerCellSpec] = []
    for source in landed_sources:
        if source.slug in seen_slugs:
            continue
        seen_slugs.add(source.slug)
        probe_binary = probe_binary_root / source.slug
        if probe_binary.is_file():
            specs.append(LedgerCellSpec(
                cell_name=source.slug,
                binary_path=probe_binary,
                launch_skip_count=5,
                warmup_protocol_name="ncu_launch_skip_n_count_1",
                shape_point=source.slug,
                kernel_source_path=source.kernel_source_path,
            ))
            continue
        specs.append(LedgerCellSpec(
            cell_name=source.slug,
            binary_path=None,
            launch_skip_count=0,
            warmup_protocol_name="ncu_unavailable",
            unavailable_reason="probe_binary_missing_for_slug",
            shape_point=source.slug,
            kernel_source_path=None,
        ))
    return tuple(specs)


def filter_landed_sources_by_host_arch(
    landed_sources: Iterable[LandedKernelSource],
    host_arch_manifest: Iterable[HostArchAvailabilityEntry] | dict[str, HostArchAvailabilityEntry],
) -> tuple[LandedKernelSource, ...]:
    """Filter `LandedKernelSource` rows by the host-arch
    availability manifest, keeping only sources whose slug
    carries `status == "sm86_compatible_cohort"`.

    `host_arch_manifest` is the mapping returned by
    `load_host_arch_availability_manifest()` (a
    `Mapping[str, HostArchAvailabilityEntry]`). The helper
    duck-types on `.status` so the caller can pass either the
    dict view directly or a dataclass-iterable form.

    Sources whose slug is `host_arch_unsupported` in the
    manifest are dropped silently — that is the carveout
    contract. Sources whose slug is absent from the manifest
    entirely raise `HarvestBatchError(
    "harvest_batch_slug_not_in_host_arch_manifest", ...)`:
    every landed baseline MUST appear in the canonical manifest
    (which enumerates all 235 canonical SOL-ExecBench problems),
    so an unknown landed slug is either a renamed problem (the
    manifest needs updating) or an off-corpus port (which does
    not belong in the batch).
    """
    by_slug: dict[str, HostArchAvailabilityEntry] = (
        dict(host_arch_manifest)
        if isinstance(host_arch_manifest, dict)
        else {entry.slug: entry for entry in host_arch_manifest}
    )
    kept: list[LandedKernelSource] = []
    for source in landed_sources:
        entry = by_slug.get(source.slug)
        if entry is None:
            raise HarvestBatchError(
                "harvest_batch_slug_not_in_host_arch_manifest",
                f"landed source slug {source.slug!r} is not "
                f"present in the host-arch availability "
                f"manifest; every landed baseline must appear "
                f"in the canonical manifest. Either update the "
                f"manifest to cover the slug or remove the "
                f"landed source from the batch.",
            )
        if entry.status == _HOST_ARCH_SM86_COMPATIBLE:
            kept.append(source)
        # host_arch_unsupported slugs are silently dropped:
        # they cannot run on sm_86 by definition, so the
        # batch carves them out of the harvest.
    return tuple(kept)


def _load_required_host_arch_manifest(
    host_arch_manifest_path: Path,
) -> Mapping[str, HostArchAvailabilityEntry]:
    """Load the host-arch manifest and fail closed if it's
    missing or malformed.

    Wraps `load_host_arch_availability_manifest()` (which
    returns `None` on missing path) to enforce that the batch
    driver always sees a real manifest. The batch harvest is
    scoped to available sm_86-compatible hand-ports, so a
    missing manifest is operator error rather than a degrade-
    to-empty signal.
    """
    if not host_arch_manifest_path.is_file():
        raise HarvestBatchError(
            "harvest_batch_host_arch_manifest_missing",
            f"host-arch availability manifest does not exist "
            f"at {str(host_arch_manifest_path)!r}; the batch "
            f"driver requires it to filter landed sources to "
            f"the sm_86-compatible cohort. Either point at the "
            f"in-tree manifest "
            f"({_HOST_ARCH_MANIFEST_RELATIVE_SUBPATH}) or "
            f"create the manifest at the chosen path.",
        )
    try:
        manifest = load_host_arch_availability_manifest(
            host_arch_manifest_path,
        )
    except PerProblemRecordSchemaError as exc:
        raise HarvestBatchError(
            "harvest_batch_host_arch_manifest_missing",
            f"host-arch availability manifest at "
            f"{str(host_arch_manifest_path)!r} failed strict "
            f"validation ({exc.typed_reason}: {exc.message}); "
            f"the batch driver cannot proceed without a valid "
            f"manifest",
        ) from exc
    if manifest is None:
        # Defensive: file exists per the `is_file()` check
        # above but the loader returned None anyway. Treat as
        # missing-manifest fail-closed.
        raise HarvestBatchError(
            "harvest_batch_host_arch_manifest_missing",
            f"host-arch availability manifest at "
            f"{str(host_arch_manifest_path)!r} returned None "
            f"from the loader despite existing on disk",
        )
    return manifest


def _merge_new_into_existing_ledger(
    *,
    new_ledger: LabelLedger,
    existing_ledger: LabelLedger,
) -> LabelLedger:
    """Merge `new_ledger` entries + unavailable rows into the
    existing ledger by `cell_name`. The default duplicate
    policy is REJECT: any new row whose `cell_name` already
    appears in the existing ledger (as either an entry or an
    unavailable row) raises `HarvestBatchError(
    "harvest_batch_duplicate_cell_name", ...)`.

    The merged ledger inherits the existing ledger's meta
    block (schema_version, captured_at_utc, host_arch,
    host_descriptor_hash, ncu_version, driver_version) — the
    new ledger's meta is discarded to keep one source of
    truth for the batch's host fingerprint. Callers who want
    to overwrite a row must drop it from the existing ledger
    first (a separate explicit operation, not silent merge
    semantics).
    """
    existing_cell_names: set[str] = set()
    for entry in existing_ledger.entries:
        existing_cell_names.add(entry.cell_name)
    for u in existing_ledger.unavailable:
        existing_cell_names.add(u.cell_name)

    conflicts: list[str] = []
    for entry in new_ledger.entries:
        if entry.cell_name in existing_cell_names:
            conflicts.append(entry.cell_name)
    for u in new_ledger.unavailable:
        if u.cell_name in existing_cell_names:
            conflicts.append(u.cell_name)
    if conflicts:
        raise HarvestBatchError(
            "harvest_batch_duplicate_cell_name",
            f"new batch ledger has {len(conflicts)} cell "
            f"name(s) already present in the existing ledger: "
            f"{sorted(set(conflicts))!r}. The default merge "
            f"policy rejects duplicates; drop the existing row "
            f"first to overwrite, or filter the new batch to "
            f"exclude the conflicting cells.",
        )

    merged_entries = tuple(existing_ledger.entries) + tuple(new_ledger.entries)
    merged_unavailable = (
        tuple(existing_ledger.unavailable) + tuple(new_ledger.unavailable)
    )
    return LabelLedger(
        schema_version=existing_ledger.schema_version,
        captured_at_utc=existing_ledger.captured_at_utc,
        host_arch=existing_ledger.host_arch,
        host_descriptor_hash=existing_ledger.host_descriptor_hash,
        ncu_version=existing_ledger.ncu_version,
        driver_version=existing_ledger.driver_version,
        entries=merged_entries,
        unavailable=merged_unavailable,
    )


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.run_ncu_harvest_batch.run_batch_harvest')
def run_batch_harvest(
    *,
    repo_root: Path,
    ncu_metric_version: str,
    host_arch: str,
    host_descriptor_hash: str,
    ncu_version: str,
    driver_version: str,
    probe_binary_root: Path | None = None,
    host_arch_manifest_path: Path | None = None,
    existing_ledger_path: Path | None = None,
    output_ledger_path: Path | None = None,
    runner=None,
    clock=None,
    sha256_of_file=None,
    source_commit_resolver=None,
    ncu_cli_path: str | None = None,
    ncu_cli_sha256_reader=None,
    ncu_version_runner=None,
    nvidia_smi_runner=None,
    probe_path: Path | str | None = None,
) -> LabelLedger:
    """Discover landed `*.cu` baselines under
    `<repo_root>/baselines/`, filter them by the host-arch
    availability manifest to the sm_86-compatible cohort,
    build sweep specs, and run the calibration sweep. When
    `existing_ledger_path` is provided the helper loads and
    merges the freshly harvested ledger into it via the
    default reject-duplicates policy. When
    `output_ledger_path` is provided the resulting (merged or
    fresh) ledger is serialized to disk through the strict
    serializer. Returns the resulting `LabelLedger`.

    Defaults:
      * `probe_binary_root`: `<repo_root>/nvbit_tracer/metric_extractor/tests/calibration_cohort_b/`.
      * `host_arch_manifest_path`: `<repo_root>/scripts/baselines_harness/calibration/cache_prediction/host_arch_availability_v1.toml`.

    Fail-closed semantics:
      * `harvest_batch_baselines_root_missing`: `<repo_root>/baselines/` does not exist.
      * `harvest_batch_no_landed_sources`: discovery returned an empty tuple.
      * `harvest_batch_host_arch_manifest_missing`: manifest path does not exist or fails validation.
      * `harvest_batch_slug_not_in_host_arch_manifest`: a landed slug is not enumerated in the manifest.
      * `harvest_batch_duplicate_cell_name`: a newly-harvested cell already appears in the existing ledger.

    All probe-injection seams flow through to the underlying
    `run_calibration_sweep()` for unit testability.
    """
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.run_ncu_harvest_batch.run_batch_harvest')
    if probe_binary_root is None:
        probe_binary_root = repo_root / _DEFAULT_PROBE_BINARY_SUBPATH
    if host_arch_manifest_path is None:
        host_arch_manifest_path = (
            repo_root / _HOST_ARCH_MANIFEST_RELATIVE_SUBPATH
        )
    landed = discover_landed_baselines(repo_root)
    if not landed:
        raise HarvestBatchError(
            "harvest_batch_no_landed_sources",
            f"discover_landed_baselines({str(repo_root)!r}) "
            f"returned zero LandedKernelSource entries; "
            f"refusing to emit an empty calibration ledger",
        )
    manifest = _load_required_host_arch_manifest(
        host_arch_manifest_path,
    )
    sm86_landed = filter_landed_sources_by_host_arch(
        landed, manifest,
    )
    specs = build_sweep_specs_from_landed_sources(
        sm86_landed, probe_binary_root=probe_binary_root,
    )
    if not specs:
        # All landed sources were host_arch_unsupported. The
        # batch is empty by carveout, not by operator error;
        # fail closed via the existing reason so the operator
        # sees the actionable message.
        raise HarvestBatchError(
            "harvest_batch_no_landed_sources",
            f"after filtering by host_arch_availability_v1.toml, "
            f"zero sm_86-compatible landed sources remain in "
            f"{str(repo_root / 'baselines')!r}; refusing to "
            f"emit an empty calibration ledger",
        )
    new_ledger = run_calibration_sweep(
        specs,
        ncu_metric_version=ncu_metric_version,
        host_arch=host_arch,
        host_descriptor_hash=host_descriptor_hash,
        ncu_version=ncu_version,
        driver_version=driver_version,
        runner=runner,
        clock=clock,
        sha256_of_file=sha256_of_file,
        source_commit_resolver=source_commit_resolver,
        ncu_cli_path=ncu_cli_path,
        ncu_cli_sha256_reader=ncu_cli_sha256_reader,
        ncu_version_runner=ncu_version_runner,
        nvidia_smi_runner=nvidia_smi_runner,
        probe_path=probe_path,
    )

    final_ledger = new_ledger
    if existing_ledger_path is not None and existing_ledger_path.is_file():
        existing_ledger = load_ledger_from_toml(existing_ledger_path)
        final_ledger = _merge_new_into_existing_ledger(
            new_ledger=new_ledger,
            existing_ledger=existing_ledger,
        )
        validate_label_ledger(final_ledger)

    if output_ledger_path is not None:
        serialize_ledger_to_toml(final_ledger, output_ledger_path)

    return final_ledger


def verify_label_stability(
    new_ledger: LabelLedger,
    existing_ledger_path: Path,
    *,
    sentinel_cell_names: tuple[str, ...] = LABEL_STABILITY_SENTINEL_CELLS,
    drift_tolerance_fraction: float = _DEFAULT_DRIFT_TOLERANCE_FRACTION,
) -> None:
    """Compare the new ledger against the existing on-disk
    ledger for every sentinel cell, and fail closed when the
    absolute drift on either `l1_global_load_hit_rate` or
    `l2_read_hit_rate` exceeds `drift_tolerance_fraction`.

    A "sentinel rerun" — re-harvesting the predeclared
    cohort-A subset after a host or toolchain change — is the
    canary for label-corrupting regressions. The default
    sentinel is `LABEL_STABILITY_SENTINEL_CELLS` (14 cells);
    the default tolerance is `0.01` (1 percentage point on the
    [0, 1] hit-rate fraction).

    Both ledgers MUST contain a matching `[[entries]]` row for
    every cell in `sentinel_cell_names`; a missing row in
    either ledger raises
    `HarvestBatchError("label_stability_sentinel_cell_missing", ...)`.
    A drift overshoot raises
    `HarvestBatchError("label_stability_drift_exceeded", ...)`
    with a message naming every drifted cell and its per-field
    drift values, so the operator can pinpoint the regression
    without re-running the harvest.

    `existing_ledger_path` is loaded via
    `load_ledger_from_toml()`; the strict loader fails closed
    on malformed input so this helper does not duplicate
    validation. Returns `None` on a clean compare; the helper
    is invoked for its side-effect of raising on drift.
    """
    if not existing_ledger_path.is_file():
        raise HarvestBatchError(
            "label_stability_sentinel_cell_missing",
            f"existing ledger does not exist at "
            f"{str(existing_ledger_path)!r}; cannot run a "
            f"label-stability sentinel without a baseline",
        )
    existing_ledger = load_ledger_from_toml(existing_ledger_path)

    new_by_name: dict[str, LedgerEntry] = {
        entry.cell_name: entry for entry in new_ledger.entries
    }
    existing_by_name: dict[str, LedgerEntry] = {
        entry.cell_name: entry for entry in existing_ledger.entries
    }
    missing_in_new: list[str] = []
    missing_in_existing: list[str] = []
    for cell_name in sentinel_cell_names:
        if cell_name not in new_by_name:
            missing_in_new.append(cell_name)
        if cell_name not in existing_by_name:
            missing_in_existing.append(cell_name)
    if missing_in_new or missing_in_existing:
        raise HarvestBatchError(
            "label_stability_sentinel_cell_missing",
            f"sentinel cells missing from one or both ledgers:"
            f"\n  missing in new ledger: {sorted(missing_in_new)!r}"
            f"\n  missing in existing ledger: "
            f"{sorted(missing_in_existing)!r}",
        )

    drifted: list[tuple[str, float, float]] = []
    for cell_name in sentinel_cell_names:
        new_entry = new_by_name[cell_name]
        existing_entry = existing_by_name[cell_name]
        l1_drift = abs(
            new_entry.label.l1_global_load_hit_rate
            - existing_entry.label.l1_global_load_hit_rate
        )
        l2_drift = abs(
            new_entry.label.l2_read_hit_rate
            - existing_entry.label.l2_read_hit_rate
        )
        if (
            l1_drift > drift_tolerance_fraction
            or l2_drift > drift_tolerance_fraction
        ):
            drifted.append((cell_name, l1_drift, l2_drift))
    if drifted:
        diff_lines = "\n".join(
            f"  {cell}: l1_drift={l1:.6f} l2_drift={l2:.6f}"
            for cell, l1, l2 in drifted
        )
        raise HarvestBatchError(
            "label_stability_drift_exceeded",
            f"{len(drifted)} sentinel cell(s) drifted by more "
            f"than {drift_tolerance_fraction} on L1 or L2 "
            f"read-only hit-rate fraction:\n{diff_lines}",
        )


def _build_sentinel_sweep_specs(
    *,
    repo_root: Path,
    sentinel_cell_names: tuple[str, ...],
    corpus_manifest_path: Path | None = None,
    cohort_b_binary_root: Path | None = None,
) -> tuple[LedgerCellSpec, ...]:
    """Build the `LedgerCellSpec` subset for the sentinel cells.

    Loads `calibration_corpus.toml`, builds the full sweep spec
    set via `build_sweep_specs_from_manifest()`, and filters to
    exactly `sentinel_cell_names`. Fails closed BEFORE any NCU
    invocation when:

      * Any sentinel cell is absent from the manifest-derived
        specs (`label_stability_sentinel_cell_missing`).
      * Any sentinel spec carries `binary_path=None` (the
        operator's probe binary build is incomplete; the spec
        cannot be NCU-harvested):
        `label_stability_sentinel_spec_unavailable`.

    This is an internal helper for `run_label_stability_rerun()`;
    tests inject fabricated manifest paths or `cohort_b_binary_root`
    to exercise each failure mode without touching the real
    calibration corpus.
    """
    if corpus_manifest_path is None:
        corpus_manifest_path = repo_root / _CALIBRATION_CORPUS_RELATIVE_SUBPATH
    manifest = load_calibration_corpus_manifest(corpus_manifest_path)
    all_specs = build_sweep_specs_from_manifest(
        manifest,
        repo_root=repo_root,
        cohort_b_binary_root=cohort_b_binary_root,
    )
    by_name: dict[str, LedgerCellSpec] = {}
    duplicates: list[str] = []
    for spec in all_specs:
        if spec.cell_name in by_name:
            duplicates.append(spec.cell_name)
            continue
        by_name[spec.cell_name] = spec
    if duplicates:
        raise HarvestBatchError(
            "label_stability_duplicate_sentinel_spec",
            f"manifest-derived sweep specs at "
            f"{str(corpus_manifest_path)!r} contain duplicate "
            f"cell_name(s): {sorted(set(duplicates))!r}. The "
            f"calibration corpus must enumerate each sentinel "
            f"cell exactly once.",
        )
    missing = [name for name in sentinel_cell_names if name not in by_name]
    if missing:
        raise HarvestBatchError(
            "label_stability_sentinel_cell_missing",
            f"sentinel cell(s) absent from the manifest-derived "
            f"sweep specs at {str(corpus_manifest_path)!r}: "
            f"{sorted(missing)!r}",
        )
    unavailable_specs: list[str] = []
    for cell_name in sentinel_cell_names:
        spec = by_name[cell_name]
        if spec.binary_path is None:
            unavailable_specs.append(cell_name)
    if unavailable_specs:
        raise HarvestBatchError(
            "label_stability_sentinel_spec_unavailable",
            f"sentinel cell(s) carry binary_path=None and cannot "
            f"be NCU-harvested; the operator's probe binary "
            f"build is incomplete for: "
            f"{sorted(unavailable_specs)!r}",
        )
    return tuple(by_name[cell_name] for cell_name in sentinel_cell_names)


@retired_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.run_ncu_harvest_batch.run_label_stability_rerun')
def run_label_stability_rerun(
    *,
    repo_root: Path,
    existing_ledger_path: Path,
    ncu_metric_version: str,
    host_arch: str,
    host_descriptor_hash: str,
    ncu_version: str,
    driver_version: str,
    sentinel_cell_names: tuple[str, ...] = LABEL_STABILITY_SENTINEL_CELLS,
    drift_tolerance_fraction: float = _DEFAULT_DRIFT_TOLERANCE_FRACTION,
    corpus_manifest_path: Path | None = None,
    cohort_b_binary_root: Path | None = None,
    runner=None,
    clock=None,
    sha256_of_file=None,
    source_commit_resolver=None,
    ncu_cli_path: str | None = None,
    ncu_cli_sha256_reader=None,
    ncu_version_runner=None,
    nvidia_smi_runner=None,
    probe_path: Path | str | None = None,
) -> LabelLedger:
    """Re-harvest the predeclared sentinel subset and compare
    against the existing ledger.

    The operator-callable entry point for the label-stability
    sentinel rerun. Builds sweep specs from
    `calibration_corpus.toml`, filters to exactly
    `sentinel_cell_names` (default
    `LABEL_STABILITY_SENTINEL_CELLS`, 14 cells), fails closed
    BEFORE the sweep when any sentinel is missing from the
    manifest or carries `binary_path=None`, invokes
    `run_calibration_sweep()` on exactly the sentinel specs
    with every probe-injection seam threaded through, then
    calls `verify_label_stability()` on the resulting ledger
    + `existing_ledger_path`.

    Returns the freshly harvested sentinel `LabelLedger` on
    success. Raises `HarvestBatchError` with one of the
    closed-vocabulary typed reasons on any failure
    (sentinel missing / spec unavailable / drift exceeded
    / sentinel cell missing from one of the ledgers in the
    comparator stage).
    """
    refuse_legacy_ncu_calibration('sol_baselines_harness.calibration.cache_prediction.run_ncu_harvest_batch.run_label_stability_rerun')
    sentinel_specs = _build_sentinel_sweep_specs(
        repo_root=repo_root,
        sentinel_cell_names=sentinel_cell_names,
        corpus_manifest_path=corpus_manifest_path,
        cohort_b_binary_root=cohort_b_binary_root,
    )
    new_ledger = run_calibration_sweep(
        sentinel_specs,
        ncu_metric_version=ncu_metric_version,
        host_arch=host_arch,
        host_descriptor_hash=host_descriptor_hash,
        ncu_version=ncu_version,
        driver_version=driver_version,
        runner=runner,
        clock=clock,
        sha256_of_file=sha256_of_file,
        source_commit_resolver=source_commit_resolver,
        ncu_cli_path=ncu_cli_path,
        ncu_cli_sha256_reader=ncu_cli_sha256_reader,
        ncu_version_runner=ncu_version_runner,
        nvidia_smi_runner=nvidia_smi_runner,
        probe_path=probe_path,
    )
    verify_label_stability(
        new_ledger,
        existing_ledger_path,
        sentinel_cell_names=sentinel_cell_names,
        drift_tolerance_fraction=drift_tolerance_fraction,
    )
    return new_ledger


__all__ = [
    "HARVEST_BATCH_TYPED_REASONS",
    "HarvestBatchError",
    "LABEL_STABILITY_SENTINEL_CELLS",
    "LandedKernelSource",
    "build_sweep_specs_from_landed_sources",
    "discover_landed_baselines",
    "filter_landed_sources_by_host_arch",
    "run_batch_harvest",
    "run_label_stability_rerun",
    "verify_label_stability",
]
