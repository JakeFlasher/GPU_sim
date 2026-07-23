"""L1 truth-label restoration utility for admitted cells whose
prior v3-to-v4 schema migration dropped L1 mean while preserving
the per-repeat sidecars.

Per the existing L1 admission gate (n_l1tex_repeats >=
repeats_required), ``repeatability_aggregator._summarize_cell`` is
the single source of truth for computing per-cell L1 mean + stddev
from per-repeat sidecars. The earlier schema migration populated
the L2 fields on `data/local_labels/label_admitted.json` but
recorded `cache_hit_rate_l1tex_sector_pct=null` +
`l1tex_label_status=mean_absent_provenance_missing` on every
pre-migration sm_86 cell, because the original per-repeat sidecars
were said to be not reproducible from canonical paths.

This utility supports TWO source-modes per cell:

  - ``canonical_admission_cfg``: the cfg-dir's L2 aggregate
    matches the admitted row's L2 mean / stddev. Used for cells
    whose admission cfg-dir is still on disk and whose L1 mean
    needs back-fill alone.
  - ``path_b_l1_only_cfg``: the cfg-dir is a fresh Path-B local
    NCU sweep on the current binary whose L2 stddev may exceed
    the 0.03 admission gate but whose L1 evidence is otherwise
    valid. The admitted row is already admitted (has a non-null
    L2 mean from a prior sweep); the utility ONLY writes the L1
    fields and does NOT overwrite L2.

Before mutating any admitted row, the utility validates:

  - slug / shape_point / arch equality between the cfg-dir's
    per-repeat sidecars + run manifests and the target admitted
    row.
  - Consistent per-repeat ``binary_sha256`` across all sidecars,
    matching the admitted row's recorded ``binary_sha256``.
  - Consistent per-repeat ``kernel_signature_sha256`` across all
    run manifests, matching the admitted row's recorded
    ``kernel_signature_sha256``.
  - source_commit on every sidecar/manifest belongs to the
    documented post-commit accept window (``{HEAD, HEAD^}``).
  - ``n_l1tex_repeats >= repeats_required``.
  - L1 stddev ``<=`` the documented L1-repeatability threshold
    (mirrors the L2-side ``0.03`` gate; this is the L1-side
    discipline pin).
  - For ``canonical_admission_cfg``: aggregated L2 mean / stddev
    match the admitted row's L2 fields within float tolerance.
  - For ``path_b_l1_only_cfg``: the admitted row has a non-null
    L2 mean (i.e., already admitted) and the utility preserves
    those L2 fields byte-identical.

Run manifests must exist for every per-repeat sidecar (the
contract precedent in ``repeatability_aggregator._collect_manifests``).

Any violation raises ``_SemanticBindingViolation`` and exits
with a typed stderr diagnostic. The mutated admission dict is
schema-validated in memory BEFORE the file is replaced.

CLI:

    python3 tools/local_labels/restore_admitted_l1_mean_from_source_config.py
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tools" / "local_labels"))

import repeatability_aggregator as ra  # noqa: E402


# L1-side repeatability threshold (the L1 stddev gate). Mirrors
# the L2-side ``0.03`` admission gate per the project's
# repeatability discipline; documented as the L1 stddev ceiling
# the restore utility enforces before overwriting an admitted
# row's L1 fields.
_DEFAULT_L1_STDDEV_MAX: float = 0.03

# Aggregated L2 mean/stddev tolerance for canonical_admission_cfg
# match. Float-precision aware.
_L2_MATCH_TOL: float = 1e-6


_SourceMode = Literal["canonical_admission_cfg", "path_b_l1_only_cfg"]


@dataclass(frozen=True)
class _SourceRecord:
    """Structured per-cell source mapping for L1 restoration.

    The ``arch`` field is the explicit per-arch lockstep per
    BL-20260507: hardcoding the target arch in the validator was
    a multi-surface lockstep violation that admitted a synthetic
    sm_120 row through an sm_86 cfg-dir; the per-cell map key
    carries arch so the lookup binds the admitted row's arch to
    the source-cfg's arch at indexing time.
    """
    arch: str
    cfg_dir: str
    source_mode: _SourceMode


class _SemanticBindingViolation(Exception):
    """Raised when a cfg-dir's per-repeat evidence fails any of
    the semantic-binding checks. Distinct from
    ``ra._SchemaUnavailableError`` and the generic ``ValueError``
    raised by the v4 schema validator."""


# Canonical (cell -> source-record) mapping for the cells whose
# prior schema migration dropped L1 mean. Each cfg-dir contains
# the per-repeat sidecars whose aggregation reproduces the L2
# mean + stddev currently recorded on ``label_admitted.json``
# (for ``canonical_admission_cfg``) OR a fresh Path-B local NCU
# sweep whose L1 evidence is valid but whose L2 stddev may
# exceed the admission gate (for ``path_b_l1_only_cfg``).
_DEFAULT_CELL_TO_SOURCE_RECORD: dict[tuple[str, str, str], _SourceRecord] = {
    ("sm_86", "layernorm_fwd_bf16_sm80", "timing_shape"): _SourceRecord(
        arch="sm_86",
        cfg_dir=(
            "data/local_labels/sweeps/sweep_20260514T093846Z_b1f6d0854d22/"
            "sweep_20260514T093846Z_18d39d0b18cb/"
            "cell=sm_86__layernorm_fwd_bf16_sm80__timing_shape/"
            "cfg=warmup5_cache_all"
        ),
        source_mode="canonical_admission_cfg",
    ),
    ("sm_86", "rmsnorm_fwd_bf16_sm80", "timing_shape"): _SourceRecord(
        arch="sm_86",
        cfg_dir=(
            "data/local_labels/sweeps/sweep_20260514T162932Z_ebe60dda69fe/"
            "sweep_20260514T162932Z_d2ec0a599caf/"
            "cell=sm_86__rmsnorm_fwd_bf16_sm80__timing_shape/"
            "cfg=warmup1_cache_all"
        ),
        source_mode="canonical_admission_cfg",
    ),
    ("sm_86", "silu_fwd_bf16_sm80", "correctness_shape"): _SourceRecord(
        arch="sm_86",
        cfg_dir=(
            "data/local_labels/sweeps/sweep_20260514T162932Z_ebe60dda69fe/"
            "sweep_20260514T162932Z_d2ec0a599caf/"
            "cell=sm_86__silu_fwd_bf16_sm80__correctness_shape/"
            "cfg=warmup1_cache_all"
        ),
        source_mode="canonical_admission_cfg",
    ),
    ("sm_86", "silu_fwd_bf16_sm80", "timing_shape"): _SourceRecord(
        arch="sm_86",
        cfg_dir=(
            "data/local_labels/sweeps/sweep_20260514T093846Z_b1f6d0854d22/"
            "sweep_20260514T093846Z_18d39d0b18cb/"
            "cell=sm_86__silu_fwd_bf16_sm80__timing_shape/"
            "cfg=warmup5_cache_all"
        ),
        source_mode="canonical_admission_cfg",
    ),
    ("sm_86", "softmax_safe_bf16_sm80", "timing_shape"): _SourceRecord(
        arch="sm_86",
        cfg_dir=(
            "data/local_labels/sweeps/sweep_20260514T093846Z_b1f6d0854d22/"
            "sweep_20260514T093846Z_18d39d0b18cb/"
            "cell=sm_86__softmax_safe_bf16_sm80__timing_shape/"
            "cfg=warmup5_cache_all"
        ),
        source_mode="canonical_admission_cfg",
    ),
    ("sm_86", "l2_thrash_pointer_chase", "trace_l2_accuracy_xl"): _SourceRecord(
        arch="sm_86",
        cfg_dir=(
            "data/local_labels/sweeps/sweep_20260518T162537Z_5033bc28f80f/"
            "cell=sm_86__l2_thrash_pointer_chase__trace_l2_accuracy_xl/"
            "cfg=warmup0_cache_all"
        ),
        source_mode="path_b_l1_only_cfg",
    ),
    ("sm_86", "register_spill_local_mem", "trace_l2_accuracy_xl"): _SourceRecord(
        arch="sm_86",
        cfg_dir=(
            "data/local_labels/sweeps/sweep_20260518T162537Z_5033bc28f80f/"
            "cell=sm_86__register_spill_local_mem__trace_l2_accuracy_xl/"
            "cfg=warmup0_cache_all"
        ),
        source_mode="path_b_l1_only_cfg",
    ),
}


def _git_accept_source_commits(repo_root: Path) -> frozenset[str]:
    """Return ``{HEAD, HEAD^}`` per BL-20260508. Falls back to
    an empty set when git is unavailable; callers may pass an
    explicit accept set in test environments."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return frozenset()
    out = {head}
    try:
        head_caret = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD^"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        out.add(head_caret)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return frozenset(out)


def _load_run_manifests(cfg_dir: Path) -> list[tuple[Path, dict]]:
    """Return ``[(path, manifest_dict), ...]`` for every
    ``label_run_manifest_*_*.json`` under ``cfg_dir`` (recursive).
    Raises ``_SemanticBindingViolation`` on JSON parse error."""
    out: list[tuple[Path, dict]] = []
    for path in sorted(cfg_dir.rglob("label_run_manifest_*_*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise _SemanticBindingViolation(
                f"could not parse run manifest at "
                f"{path.relative_to(cfg_dir.parent)}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise _SemanticBindingViolation(
                f"run manifest at "
                f"{path.relative_to(cfg_dir.parent)} is not a "
                f"JSON object"
            )
        out.append((path, manifest))
    return out


def _validate_cfg_dir_semantics(
    *,
    slug: str,
    shape_point: str,
    admitted_row: dict,
    source_record: _SourceRecord,
    repeats_required: int,
    accept_source_commits: frozenset[str],
    l1_stddev_max: float,
    cfg_dir_abs: Path,
) -> tuple[float, float, int, float, float]:
    """Run all semantic-binding checks; return aggregated L1
    + L2 statistics on success. Raises ``_SemanticBindingViolation``
    on any failure.

    Returns ``(mean_l1, stddev_l1, n_l1tex_repeats,
    mean_l2, stddev_l2)`` -- the L2 fields are used by callers
    to enforce canonical_admission_cfg's L2-match check; for
    path_b_l1_only_cfg they are returned but not consumed.
    """
    # target_arch is derived from the admitted row's arch field,
    # NOT a hard-coded "sm_86" literal. Hard-coding the arch in
    # the validator violated BL-20260507 (the arch token is a
    # multi-surface lockstep) and admitted a synthetic sm_120
    # admitted row through an sm_86 cfg-dir under the prior
    # validator shape. The _SourceRecord's own arch is bound
    # to the requested target at indexing time in main(); here
    # we verify the admitted row's arch matches the source
    # record.
    admitted_arch = str(admitted_row.get("arch", ""))
    if not admitted_arch:
        raise _SemanticBindingViolation(
            f"admitted row {slug}/{shape_point} has empty "
            f"arch; cannot verify identity binding"
        )
    if admitted_arch != source_record.arch:
        raise _SemanticBindingViolation(
            f"admitted row {slug}/{shape_point} arch "
            f"{admitted_arch!r} != source_record.arch "
            f"{source_record.arch!r} (the per-cell _SourceRecord "
            f"is bound to a specific arch; admitting a different "
            f"arch through the same cfg-dir would silently borrow "
            f"another arch's L1 evidence)"
        )
    if str(admitted_row.get("slug", "")) != slug:
        raise _SemanticBindingViolation(
            f"admitted row's slug {admitted_row.get('slug')!r} != "
            f"requested target slug {slug!r}"
        )
    if str(admitted_row.get("shape_point", "")) != shape_point:
        raise _SemanticBindingViolation(
            f"admitted row's shape_point "
            f"{admitted_row.get('shape_point')!r} != requested "
            f"target shape_point {shape_point!r}"
        )
    target_arch = admitted_arch

    # Load per-repeat sidecars via the canonical aggregator
    # helpers so the parse/admission contract stays single-source.
    sidecar_paths = ra._collect_sidecars(cfg_dir_abs)
    if not sidecar_paths:
        raise _SemanticBindingViolation(
            f"cfg_dir {cfg_dir_abs} contains zero per-repeat "
            f"label_*_*.json sidecars for cell {slug}/{shape_point}"
        )
    by_cell, _rejected = ra._load_sidecars_by_cell(sidecar_paths)
    target_key = (target_arch, slug, shape_point)
    if target_key not in by_cell:
        raise _SemanticBindingViolation(
            f"cfg_dir {cfg_dir_abs} does not contain per-repeat "
            f"sidecars for cell {target_key}; available cells: "
            f"{sorted(by_cell)}"
        )
    cell_sidecars = by_cell[target_key]

    # Identity: per-repeat sidecars must agree on slug / shape /
    # device / binary_sha256 with the admitted row.
    admitted_binary = str(admitted_row.get("binary_sha256", ""))
    if not admitted_binary:
        raise _SemanticBindingViolation(
            f"admitted row {slug}/{shape_point} has empty "
            f"binary_sha256; cannot verify identity binding"
        )
    for sidecar in cell_sidecars:
        if str(sidecar.get("slug")) != slug:
            raise _SemanticBindingViolation(
                f"cfg_dir {cfg_dir_abs} sidecar slug "
                f"{sidecar.get('slug')!r} != target slug "
                f"{slug!r}"
            )
        if str(sidecar.get("shape_point")) != shape_point:
            raise _SemanticBindingViolation(
                f"cfg_dir {cfg_dir_abs} sidecar shape_point "
                f"{sidecar.get('shape_point')!r} != target "
                f"shape_point {shape_point!r}"
            )
        if str(sidecar.get("device")) != target_arch:
            raise _SemanticBindingViolation(
                f"cfg_dir {cfg_dir_abs} sidecar device "
                f"{sidecar.get('device')!r} != target arch "
                f"{target_arch!r}"
            )
        if str(sidecar.get("binary_sha256")) != admitted_binary:
            raise _SemanticBindingViolation(
                f"cfg_dir {cfg_dir_abs} sidecar binary_sha256 "
                f"{sidecar.get('binary_sha256')!r} != admitted "
                f"row binary_sha256 {admitted_binary!r} for cell "
                f"{slug}/{shape_point}"
            )

    # Run-manifest existence + per-launch identity binding.
    #
    # The prior loose check only required >=1 manifest exist
    # and iterated all loaded manifests against the admitted
    # row. That admitted a cfg-dir where 4 of 5 per-repeat
    # manifests were deleted (the surviving manifest matched
    # the admitted row, so the loop returned success).
    #
    # The contract requires ONE matched manifest per per-repeat
    # sidecar, keyed by (repeat_index, launch_index). This is
    # the per-launch grain precedent in
    # ``repeatability_aggregator._validate_provenance`` (the
    # ``label_provenance_per_launch_missing`` typed exclusion per
    # BL-20260512-per-launch-provenance-coverage-must-be-enforced).
    manifest_pairs = _load_run_manifests(cfg_dir_abs)
    if not manifest_pairs:
        raise _SemanticBindingViolation(
            f"cfg_dir {cfg_dir_abs} has zero "
            f"label_run_manifest_*_*.json files; the L1-restore "
            f"contract requires a manifest per sidecar"
        )
    admitted_kernel = str(
        admitted_row.get("kernel_signature_sha256", ""),
    )
    if not admitted_kernel:
        raise _SemanticBindingViolation(
            f"admitted row {slug}/{shape_point} has empty "
            f"kernel_signature_sha256; cannot verify identity"
        )

    # Build the (repeat_index, launch_index) -> manifest map for
    # the target cell only. A manifest whose arch/slug/shape do
    # not match the target cell is NOT silently dropped --
    # it's an active binding violation, since the cfg-dir is
    # bound to one (arch, slug, shape) per its on-disk layout
    # and an off-cell manifest indicates a polluted cfg-dir.
    manifests_by_key: dict[tuple[int, int], tuple[Path, dict]] = {}
    for path, manifest in manifest_pairs:
        m_arch = str(manifest.get("arch", ""))
        m_slug = str(manifest.get("slug", ""))
        m_shape = str(manifest.get("shape_point", ""))
        if (m_arch, m_slug, m_shape) != target_key:
            raise _SemanticBindingViolation(
                f"run manifest {path.name} declares "
                f"(arch, slug, shape_point)=({m_arch!r}, "
                f"{m_slug!r}, {m_shape!r}) which does not equal "
                f"the target cell {target_key}; the cfg-dir is "
                f"polluted with off-cell manifest evidence"
            )
        # Per-launch key. Missing or non-int (repeat_index,
        # launch_index) fails closed.
        try:
            repeat_idx = int(manifest.get("repeat_index", -1))
            launch_idx = int(manifest.get("launch_index", -1))
        except (TypeError, ValueError):
            raise _SemanticBindingViolation(
                f"run manifest {path.name} has non-integer "
                f"repeat_index/launch_index"
            ) from None
        if repeat_idx < 0 or launch_idx < 0:
            raise _SemanticBindingViolation(
                f"run manifest {path.name} has missing or "
                f"negative repeat_index/launch_index"
            )
        per_launch_key = (repeat_idx, launch_idx)
        if per_launch_key in manifests_by_key:
            prior_path = manifests_by_key[per_launch_key][0]
            raise _SemanticBindingViolation(
                f"label_manifest_duplicate_per_launch: cfg_dir "
                f"{cfg_dir_abs} has two run manifests for "
                f"(repeat_index, launch_index)={per_launch_key}: "
                f"{prior_path.name} and {path.name}"
            )
        manifests_by_key[per_launch_key] = (path, manifest)

    # For every loaded sidecar, require exactly one matching
    # manifest. Verify the manifest's identity fields bind to
    # BOTH the sidecar AND the admitted row. Collect the
    # sidecar per-launch keys for the inverse coverage check
    # below.
    sidecar_keys: set[tuple[int, int]] = set()
    for sidecar in cell_sidecars:
        try:
            s_repeat = int(sidecar.get("repeat_index", -1))
            s_launch = int(sidecar.get("launch_index", -1))
        except (TypeError, ValueError):
            raise _SemanticBindingViolation(
                f"sidecar in {cfg_dir_abs} has non-integer "
                f"repeat_index/launch_index"
            ) from None
        if s_repeat < 0 or s_launch < 0:
            raise _SemanticBindingViolation(
                f"sidecar in {cfg_dir_abs} has missing or "
                f"negative repeat_index/launch_index"
            )
        per_launch_key = (s_repeat, s_launch)
        # Duplicate-sidecar detection: a second loaded
        # target-cell sidecar declaring the same
        # (repeat_index, launch_index) tuple as a prior
        # sidecar is a polluted cfg-dir. The set-based
        # `sidecar_keys` membership would collapse the
        # duplicate silently, so we test for prior presence
        # BEFORE adding to the set. This enforces a
        # one-to-one binding (not just set equality) at the
        # per-launch grain per
        # BL-20260512-per-launch-provenance-coverage-must-be-enforced.
        if per_launch_key in sidecar_keys:
            raise _SemanticBindingViolation(
                f"label_sidecar_duplicate_per_launch: cfg_dir "
                f"{cfg_dir_abs} has two target-cell sidecars "
                f"declaring the same (repeat_index, "
                f"launch_index)={per_launch_key}; the "
                f"L1-restore contract requires one-to-one "
                f"sidecar/manifest binding at the per-launch "
                f"grain (duplicate tuples would silently share "
                f"a manifest)"
            )
        sidecar_keys.add(per_launch_key)
        if per_launch_key not in manifests_by_key:
            raise _SemanticBindingViolation(
                f"label_manifest_per_launch_missing: cfg_dir "
                f"{cfg_dir_abs} sidecar (repeat_index, "
                f"launch_index)={per_launch_key} has no matching "
                f"label_run_manifest_*_*.json (the L1-restore "
                f"contract requires one manifest per sidecar)"
            )
        path, manifest = manifests_by_key[per_launch_key]
        manifest_kernel = str(
            manifest.get("kernel_signature_sha256", ""),
        )
        if manifest_kernel != admitted_kernel:
            raise _SemanticBindingViolation(
                f"run manifest {path.name} "
                f"kernel_signature_sha256 {manifest_kernel!r} "
                f"!= admitted row's kernel_signature_sha256 "
                f"{admitted_kernel!r} for cell {slug}/{shape_point}"
            )
        manifest_binary = str(manifest.get("binary_sha256", ""))
        if not manifest_binary:
            raise _SemanticBindingViolation(
                f"run manifest {path.name} is missing "
                f"binary_sha256; provenance unbound"
            )
        if manifest_binary != admitted_binary:
            raise _SemanticBindingViolation(
                f"run manifest {path.name} binary_sha256 "
                f"{manifest_binary!r} != admitted row's "
                f"binary_sha256 {admitted_binary!r}"
            )
        # Per-launch sidecar/manifest binary agreement: the
        # sidecar must also carry the same binary_sha256 as the
        # matched manifest (the per-sidecar-vs-admitted check
        # above is at the cell-aggregate grain; this is the
        # explicit per-launch link of the chain).
        sidecar_binary = str(sidecar.get("binary_sha256", ""))
        if sidecar_binary != manifest_binary:
            raise _SemanticBindingViolation(
                f"sidecar (repeat, launch)={per_launch_key} "
                f"binary_sha256 {sidecar_binary!r} != matched "
                f"manifest {path.name} binary_sha256 "
                f"{manifest_binary!r}"
            )

    # Inverse-direction coverage: every loaded target-cell
    # manifest MUST have a matching sidecar. The forward
    # direction (sidecar -> manifest) is already enforced
    # above; this closes the per-launch grain as an EXACT
    # set equality rather than a one-sided coverage check.
    # Without this check, an extra target-cell manifest with
    # no matching sidecar is silently accepted -- the cfg-dir
    # evidence set is not exact at the per-launch grain.
    extra_manifest_keys = sorted(
        set(manifests_by_key) - sidecar_keys
    )
    if extra_manifest_keys:
        # Name the first extra tuple + its manifest filename
        # in the typed diagnostic so stderr is auditable; the
        # full sorted list of extras is also recorded for
        # auditor scan-out of multi-extra cases.
        first_extra = extra_manifest_keys[0]
        first_path = manifests_by_key[first_extra][0]
        raise _SemanticBindingViolation(
            f"label_manifest_per_launch_extra: cfg_dir "
            f"{cfg_dir_abs} has {len(extra_manifest_keys)} "
            f"target-cell manifest(s) with no matching "
            f"sidecar: first extra (repeat_index, "
            f"launch_index)={first_extra} at "
            f"{first_path.name}; full extras list = "
            f"{extra_manifest_keys}"
        )

    # source_commit acceptance:
    #
    # Each per-repeat manifest must carry a real SHA-1 form
    # source_commit (40 hex chars) so provenance is recorded.
    # The strict ``{HEAD, HEAD^}`` window is OPT-IN via
    # ``accept_source_commits``; it applies only when the
    # caller wants to enforce that the sweep is RECENT
    # (e.g., a fresh Path-B sweep just captured against the
    # current commit). Historical canonical_admission_cfg
    # cfg-dirs are by definition not recent; their
    # source_commit was captured when the sweep ran and
    # does not drift after subsequent commits.
    #
    # When the caller passes a non-empty ``accept_source_commits``,
    # the strict window applies. When the set is empty (the
    # default), only the SHA-1 form check fires.
    for path, manifest in manifest_pairs:
        sc = manifest.get("source_commit")
        if sc is None:
            raise _SemanticBindingViolation(
                f"run manifest {path.name} is missing "
                f"source_commit; provenance unrecorded"
            )
        sc_str = str(sc)
        if not (len(sc_str) == 40 and all(
            c in "0123456789abcdef" for c in sc_str
        )):
            raise _SemanticBindingViolation(
                f"run manifest {path.name} source_commit "
                f"{sc!r} is not a 40-hex SHA-1 form; "
                f"provenance recorded but malformed"
            )
        if accept_source_commits and sc_str not in accept_source_commits:
            raise _SemanticBindingViolation(
                f"run manifest {path.name} source_commit "
                f"{sc!r} not in accept window "
                f"{sorted(accept_source_commits)}"
            )

    # Aggregate L1 + L2 statistics via the canonical
    # ``_summarize_cell`` helper (this is the same code path
    # ``repeatability_aggregator.aggregate`` uses, so the
    # admission-floor semantics stay identical).
    summary = ra._summarize_cell(
        target_key,
        cell_sidecars,
        repeats_required=repeats_required,
    )

    if summary.n_l1tex_repeats < repeats_required:
        raise _SemanticBindingViolation(
            f"cfg_dir {cfg_dir_abs} cell {slug}/{shape_point}: "
            f"n_l1tex_repeats={summary.n_l1tex_repeats} < "
            f"repeats_required={repeats_required}"
        )
    if summary.mean_l1tex_sector is None:
        raise _SemanticBindingViolation(
            f"cfg_dir {cfg_dir_abs} cell {slug}/{shape_point}: "
            f"aggregator returned mean_l1tex_sector=None"
        )
    if summary.stddev_l1tex_sector is None:
        raise _SemanticBindingViolation(
            f"cfg_dir {cfg_dir_abs} cell {slug}/{shape_point}: "
            f"aggregator returned stddev_l1tex_sector=None"
        )

    mean_l1 = float(summary.mean_l1tex_sector)
    stddev_l1 = float(summary.stddev_l1tex_sector)
    mean_l2 = float(summary.mean_l2_sector)
    stddev_l2 = float(summary.stddev_l2_sector) if summary.stddev_l2_sector is not None else 0.0

    if stddev_l1 > l1_stddev_max:
        raise _SemanticBindingViolation(
            f"cfg_dir {cfg_dir_abs} cell {slug}/{shape_point}: "
            f"L1 stddev {stddev_l1!r} > l1_stddev_max "
            f"{l1_stddev_max!r}"
        )

    # source-mode-specific invariants.
    if source_record.source_mode == "canonical_admission_cfg":
        # Aggregated L2 mean/stddev must match the admitted row
        # within float tolerance. This is the post-hoc verifier
        # per BL-20260511: semantic-bind not just hash.
        admitted_l2_pct = admitted_row.get("cache_hit_rate_l2_sector_pct")
        admitted_l2_stddev = admitted_row.get("stddev_l2_sector")
        if admitted_l2_pct is None or admitted_l2_stddev is None:
            raise _SemanticBindingViolation(
                f"canonical_admission_cfg requires admitted row "
                f"{slug}/{shape_point} to carry non-null L2 "
                f"mean + stddev; got mean={admitted_l2_pct!r}, "
                f"stddev={admitted_l2_stddev!r}"
            )
        admitted_l2 = float(admitted_l2_pct) / 100.0
        if not math.isclose(mean_l2, admitted_l2, abs_tol=_L2_MATCH_TOL):
            raise _SemanticBindingViolation(
                f"canonical_admission_cfg L2 mean mismatch for "
                f"{slug}/{shape_point}: cfg-dir aggregate "
                f"{mean_l2!r} != admitted row {admitted_l2!r} "
                f"(tolerance {_L2_MATCH_TOL!r})"
            )
        if not math.isclose(
            stddev_l2, float(admitted_l2_stddev),
            abs_tol=_L2_MATCH_TOL,
        ):
            raise _SemanticBindingViolation(
                f"canonical_admission_cfg L2 stddev mismatch "
                f"for {slug}/{shape_point}: cfg-dir aggregate "
                f"{stddev_l2!r} != admitted row "
                f"{admitted_l2_stddev!r} (tolerance "
                f"{_L2_MATCH_TOL!r})"
            )
    elif source_record.source_mode == "path_b_l1_only_cfg":
        # Path-B requires the admitted row is ALREADY admitted
        # (non-null L2 mean from a prior sweep). The L1 fields
        # are written; L2 fields are PRESERVED.
        admitted_l2_pct = admitted_row.get("cache_hit_rate_l2_sector_pct")
        if admitted_l2_pct is None:
            raise _SemanticBindingViolation(
                f"path_b_l1_only_cfg requires admitted row "
                f"{slug}/{shape_point} to already carry a "
                f"non-null L2 mean (i.e., already admitted in a "
                f"prior sweep); got "
                f"cache_hit_rate_l2_sector_pct=None"
            )
    else:
        raise _SemanticBindingViolation(
            f"unknown source_mode "
            f"{source_record.source_mode!r} for cell "
            f"{slug}/{shape_point}"
        )

    return mean_l1, stddev_l1, summary.n_l1tex_repeats, mean_l2, stddev_l2


def _restore_one(
    *,
    slug: str,
    shape_point: str,
    source_record: _SourceRecord,
    admitted_row: dict,
    repeats_required: int,
    accept_source_commits: frozenset[str],
    l1_stddev_max: float,
    repo_root: Path,
) -> tuple[float, float, int]:
    """Restore one cell's L1 fields. Returns
    ``(mean_l1, stddev_l1, n_l1tex_repeats)`` on success.
    Raises ``_SemanticBindingViolation`` on any binding
    failure."""
    cfg_dir_abs = repo_root / source_record.cfg_dir
    mean_l1, stddev_l1, n_l1tex, _mean_l2, _stddev_l2 = (
        _validate_cfg_dir_semantics(
            slug=slug,
            shape_point=shape_point,
            admitted_row=admitted_row,
            source_record=source_record,
            repeats_required=repeats_required,
            accept_source_commits=accept_source_commits,
            l1_stddev_max=l1_stddev_max,
            cfg_dir_abs=cfg_dir_abs,
        )
    )
    return mean_l1, stddev_l1, n_l1tex


def _validate_admitted_in_memory(
    admitted: dict, *, schema_path: Path,
) -> None:
    """Validate the in-memory mutated admission dict against the
    v4 schema. Raises ``jsonschema.ValidationError`` on
    failure. This runs BEFORE the file is replaced, so a
    schema-invalid mutation cannot land on disk even briefly."""
    import jsonschema  # local import: schema-dependent
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(admitted, schema)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Restore L1 mean / stddev / l1tex_label_status on "
            "label_admitted.json from per-repeat sidecars under "
            "the canonical (or Path-B) cfg-dir for each named "
            "cell. Semantic-binding checks fire before any "
            "mutation."
        ),
    )
    parser.add_argument(
        "--label-admitted-path",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "label_admitted.json",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=(
            _REPO_ROOT / "workloads" / "simbench_bw" / "schemas"
            / "local_label_admitted.schema.json"
        ),
    )
    parser.add_argument(
        "--repeats-required", type=int,
        default=ra.DEFAULT_REPEATS_REQUIRED,
    )
    parser.add_argument(
        "--l1-stddev-max", type=float,
        default=_DEFAULT_L1_STDDEV_MAX,
    )
    parser.add_argument(
        "--strict-source-commit-window", action="store_true",
        help=(
            "Require every per-repeat run manifest's "
            "source_commit to be in {HEAD, HEAD^}. Default off: "
            "historical canonical_admission_cfg cfg-dirs have "
            "stable source_commits from when their sweeps ran "
            "and do not drift. Pass this flag when you need to "
            "force fresh-sweep enforcement (e.g., when re-running "
            "a Path-B sweep on the current commit)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.label_admitted_path.is_file():
        sys.stderr.write(
            f"restore_admitted_l1_mean_from_source_config: "
            f"label_admitted not found at {args.label_admitted_path}\n"
        )
        return 2

    with args.label_admitted_path.open("r", encoding="utf-8") as fh:
        admitted = json.load(fh)

    # The admitted-row map is keyed by (arch, slug, shape_point):
    # a synthetic admitted row whose arch differs from the
    # per-cell _SourceRecord's arch MUST NOT silently match an
    # entry whose (slug, shape_point) collides. The 9-cell
    # production cohort is currently all sm_86, but the key
    # shape generalizes for future cross-arch admission entries.
    cells_by_key: dict[tuple[str, str, str], dict] = {}
    for cell in admitted["cells"]:
        cells_by_key[(
            str(cell.get("arch", "")),
            str(cell.get("slug", "")),
            str(cell.get("shape_point", "")),
        )] = cell

    if args.strict_source_commit_window:
        accept_source_commits = _git_accept_source_commits(_REPO_ROOT)
    else:
        accept_source_commits = frozenset()

    failures: list[tuple[str, str, str, str]] = []
    restored: list[tuple[str, str, str, float, float, int, str]] = []
    for (arch, slug, shape_point), source_record in (
        _DEFAULT_CELL_TO_SOURCE_RECORD.items()
    ):
        cell = cells_by_key.get((arch, slug, shape_point))
        if cell is None:
            failures.append(
                (arch, slug, shape_point,
                 f"cell not present in {args.label_admitted_path} "
                 f"under (arch, slug, shape_point)="
                 f"({arch!r}, {slug!r}, {shape_point!r})")
            )
            continue
        try:
            mean_l1, stddev_l1, n_l1tex = _restore_one(
                slug=slug,
                shape_point=shape_point,
                source_record=source_record,
                admitted_row=cell,
                repeats_required=args.repeats_required,
                accept_source_commits=accept_source_commits,
                l1_stddev_max=args.l1_stddev_max,
                repo_root=_REPO_ROOT,
            )
        except _SemanticBindingViolation as exc:
            failures.append((arch, slug, shape_point, str(exc)))
            continue
        cell["cache_hit_rate_l1tex_sector_pct"] = round(mean_l1 * 100, 6)
        cell["stddev_l1tex_sector"] = stddev_l1
        cell["l1tex_label_status"] = "mean_present"
        restored.append((
            arch, slug, shape_point, mean_l1, stddev_l1, n_l1tex,
            source_record.source_mode,
        ))

    if failures:
        sys.stderr.write(
            f"\nrestore_admitted_l1_mean_from_source_config: "
            f"{len(failures)} failures (semantic binding)\n"
        )
        for arch, slug, shape_point, reason in failures:
            sys.stderr.write(
                f"  {arch}/{slug}/{shape_point}: {reason}\n"
            )
        return 2

    # Validate the mutated dict in memory BEFORE replacing the
    # file on disk. A schema-invalid mutation cannot land even
    # briefly.
    try:
        _validate_admitted_in_memory(
            admitted, schema_path=args.schema_path,
        )
    except Exception as exc:  # pylint: disable=broad-except
        sys.stderr.write(
            f"\nrestore_admitted_l1_mean_from_source_config: "
            f"in-memory schema validation FAILED before write: "
            f"{exc}\n"
        )
        return 2

    # Serialize with the project's canonical formatting (matching
    # repeatability_aggregator._write semantics: indent=2, sort_keys=True).
    with args.label_admitted_path.open("w", encoding="utf-8") as fh:
        json.dump(admitted, fh, indent=2, sort_keys=True)
        fh.write("\n")

    sys.stderr.write(
        f"restore_admitted_l1_mean_from_source_config: restored "
        f"{len(restored)} cells\n"
    )
    for arch, slug, shape_point, mean_l1, stddev_l1, n_l1tex, mode in restored:
        sys.stderr.write(
            f"  {arch}/{slug}/{shape_point}: "
            f"mean_l1_pct={mean_l1 * 100:.4f}, "
            f"stddev_l1={stddev_l1:.6f}, n_l1tex={n_l1tex}, "
            f"source_mode={mode!r}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
