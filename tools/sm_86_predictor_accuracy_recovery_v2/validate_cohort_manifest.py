"""Strict validator for the sm_86 submission-track cohort manifest.

The schema lives at
``tools/sm_86_predictor_accuracy_recovery_v2/schemas/cohort_manifest.schema.json``.
The validator wraps ``jsonschema.validate(...)`` with three
additional cross-cell invariants the schema cannot express:

  - **uniqueness**: every cell's ``(slug, shape_point)`` pair is
    unique across the array.
  - **admitted-count accounting**: the top-level
    ``admitted_cell_count`` equals the number of cells whose
    ``admission_status == "admitted"``.
  - **attempted-count accounting**: the top-level
    ``attempted_cell_count`` equals ``len(cells)`` (every cell, both
    admitted and attempted_excluded).

The validator raises typed exceptions so callers can recover
gracefully. ``validate_manifest(path) -> CohortManifest`` returns a
strongly-typed dataclass on success; the dataclass exposes the cells
list + the counters + the frozen-at-commit + the replacement-history
list for downstream consumers (the manifest-driven cohort runner,
the post-hoc audit tools, etc.).

Per BL-20260506-schema-validator-probe-driven-tightening, the
validator surfaces each violated invariant with a precise typed
error so the test suite can enumerate every clause independently.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import jsonschema

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared mixed-arity corpus-cell identity helper: cell_variant_id yields "" for
# a base cell and the canonical register_cap_<n> for a register-cap build
# variant, so the (slug, shape_point) uniqueness check widens to
# (slug, shape_point, variant_id) and sibling build variants are NOT falsely
# flagged as duplicate cells.
_LOCAL_LABELS_DIR = _REPO_ROOT / "tools" / "local_labels"
if str(_LOCAL_LABELS_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LABELS_DIR))
from corpus_cell_identity import cell_variant_id, variant_lockstep_error  # noqa: E402
_SCHEMA_PATH = (
    Path(__file__).resolve().parent
    / "schemas" / "cohort_manifest.schema.json"
)


# ---------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------


class CohortManifestValidationError(Exception):
    """Base class for cohort-manifest validation failures."""


class CohortManifestSchemaError(CohortManifestValidationError):
    """Wraps ``jsonschema.ValidationError`` with a docstring-anchored
    message. The validator re-raises a typed sentinel so callers can
    match on the exception type rather than parsing the
    ``jsonschema.ValidationError`` message."""

    def __init__(self, detail: str, *, json_pointer: str | None = None):
        self.detail = detail
        self.json_pointer = json_pointer
        super().__init__(detail)


class CohortManifestDuplicateCellError(CohortManifestValidationError):
    """Raised when two or more cells share the same
    ``(slug, shape_point)`` pair."""

    def __init__(self, *, slug: str, shape_point: str, count: int):
        self.slug = slug
        self.shape_point = shape_point
        self.count = count
        super().__init__(
            f"duplicate cohort cell: ({slug!r}, {shape_point!r}) "
            f"appears {count} times"
        )


class CohortManifestAccountingDriftError(CohortManifestValidationError):
    """Raised when the top-level ``attempted_cell_count`` or
    ``admitted_cell_count`` does not match the actual cells array."""

    def __init__(
        self, *, counter_name: str, declared: int, observed: int,
    ):
        self.counter_name = counter_name
        self.declared = declared
        self.observed = observed
        super().__init__(
            f"cohort manifest accounting drift: declared "
            f"{counter_name}={declared} but observed {observed} cells"
        )


class CohortManifestLabelPresenceError(CohortManifestValidationError):
    """Raised when a cell with ``admission_status="admitted"`` cannot
    be matched against any row in the canonical
    ``data/local_labels/label_admitted.json`` aggregate on the
    5-field identity key (``arch``, ``slug``, ``shape_point``,
    ``binary_sha256``, ``kernel_signature_sha256``).

    The check is the orthogonal counterpart to the file-existence
    reconciliation in ``_reconcile_cell_paths``: that helper only
    verifies that the label file exists on disk; this exception
    closes the additional false-positive arm where a cell points at
    an existing aggregate file that contains no row for the cell's
    own identity tuple. Without this check, an admitted manifest
    cell whose trace landed but whose NCU label sweep has not yet
    been aggregated into ``label_admitted.json`` would pass the
    validator on file existence alone.
    """

    def __init__(
        self, *, arch: str, slug: str, shape_point: str,
        binary_sha256: str, kernel_signature_sha256: str,
        label_admitted_path: str,
    ):
        self.arch = arch
        self.slug = slug
        self.shape_point = shape_point
        self.binary_sha256 = binary_sha256
        self.kernel_signature_sha256 = kernel_signature_sha256
        self.label_admitted_path = label_admitted_path
        super().__init__(
            f"cohort manifest label-presence violation: admitted cell "
            f"({slug!r}, {shape_point!r}, arch={arch!r}) has no matching "
            f"identity-tuple row in {label_admitted_path!r}; the cell's "
            f"binary_sha256={binary_sha256[:16]}... and "
            f"kernel_signature_sha256={kernel_signature_sha256[:16]}... "
            f"are not jointly present in the label corpus"
        )


class CohortManifestIdentityLockstepError(CohortManifestValidationError):
    """Raised when a cell's identity-tuple disagrees with the cell's
    own top-level fields. Lockstep fields: ``slug``, ``shape_point``,
    ``arch``, ``device``. The cohort manifest is a drift-sensitive
    artifact; an internal disagreement about which kernel a
    binary/kernel-signature/source-commit tuple belongs to is a
    contract violation that the runner must reject before iterating
    the admitted cohort."""

    def __init__(
        self, *, cell_slug: str, cell_shape_point: str,
        field: str, cell_value: str, identity_value: str,
    ):
        self.cell_slug = cell_slug
        self.cell_shape_point = cell_shape_point
        self.field = field
        self.cell_value = cell_value
        self.identity_value = identity_value
        super().__init__(
            f"cohort manifest identity-tuple lockstep violation: "
            f"cell ({cell_slug!r}, {cell_shape_point!r}) carries "
            f"top-level {field}={cell_value!r} but identity_tuple."
            f"{field}={identity_value!r}"
        )


class CohortManifestVariantLockstepError(CohortManifestValidationError):
    """Raised when a cell's TOP-LEVEL build-variant identity
    (``register_cap`` / ``variant_id``) disagrees with the build-variant
    identity carried inside its nested ``identity_tuple``. Sibling of
    :class:`CohortManifestIdentityLockstepError`: the schema permits the
    register-cap variant axis at BOTH levels (additive, for build-variant
    cells), so the validator must enforce that a base cell omits the pair at
    both levels and a variant cell normalizes to the SAME canonical
    ``register_cap_<n>`` at both levels. Without this, a regenerated variant
    manifest could drift to top-level ``register_cap_64`` while its nested
    identity_tuple says ``register_cap_32`` and still pass."""

    def __init__(self, *, cell_slug: str, cell_shape_point: str, detail: str):
        self.cell_slug = cell_slug
        self.cell_shape_point = cell_shape_point
        self.detail = detail
        super().__init__(
            f"cohort manifest build-variant lockstep violation: "
            f"cell ({cell_slug!r}, {cell_shape_point!r}): {detail}"
        )


# ---------------------------------------------------------------------
# Loaded manifest representation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CohortManifestCell:
    """In-memory representation of a single cell entry."""

    slug: str
    shape_point: str
    arch: str
    device: str
    family_bucket: str
    split_assignment: str
    expected_trace_path: str
    expected_label_path: str
    admission_status: str
    exclusion_typed_reason: str | None
    identity_tuple: Mapping[str, str]
    # Canonical build-variant id ("" for a base cell, register_cap_<n> for a
    # register-cap variant). Additive; default "" keeps base manifests
    # unchanged. Derived from the cell's register_cap/variant_id via the shared
    # normalizer (which raises on an inconsistent pair).
    variant_id: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CohortManifestCell":
        return cls(
            slug=str(payload["slug"]),
            shape_point=str(payload["shape_point"]),
            arch=str(payload["arch"]),
            device=str(payload["device"]),
            family_bucket=str(payload["family_bucket"]),
            split_assignment=str(payload["split_assignment"]),
            expected_trace_path=str(payload["expected_trace_path"]),
            expected_label_path=str(payload["expected_label_path"]),
            admission_status=str(payload["admission_status"]),
            exclusion_typed_reason=(
                str(payload["exclusion_typed_reason"])
                if payload["exclusion_typed_reason"] is not None
                else None
            ),
            identity_tuple=dict(payload["identity_tuple"]),
            variant_id=cell_variant_id(payload),
        )


@dataclass(frozen=True)
class CohortManifest:
    """In-memory representation of a validated cohort manifest."""

    schema_version: int
    frozen_at_commit: str
    attempted_cell_count: int
    admitted_cell_count: int
    replacement_history: tuple[Mapping[str, Any], ...]
    cells: tuple[CohortManifestCell, ...]
    raw: Mapping[str, Any] = field(default_factory=dict)

    def admitted_cells(self) -> tuple[CohortManifestCell, ...]:
        """Return the admitted-cell subset (used by the
        manifest-driven runner)."""
        return tuple(
            cell for cell in self.cells
            if cell.admission_status == "admitted"
        )

    def attempted_excluded_cells(self) -> tuple[CohortManifestCell, ...]:
        return tuple(
            cell for cell in self.cells
            if cell.admission_status == "attempted_excluded"
        )


# ---------------------------------------------------------------------
# Loader + validator
# ---------------------------------------------------------------------


def _load_schema() -> Mapping[str, Any]:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_json_pointer(error: jsonschema.ValidationError) -> str:
    return "/" + "/".join(str(part) for part in error.absolute_path)


def validate_manifest(path: Path | str) -> CohortManifest:
    """Read ``path`` as a cohort manifest and validate it against the
    schema + the three cross-cell invariants the schema cannot
    express.

    Returns a :class:`CohortManifest` on success.

    Raises:
        :class:`CohortManifestSchemaError` for any
        ``jsonschema.ValidationError`` (missing field, enum miss,
        wrong type, pattern miss, additionalProperties violation,
        conditional schema violation).
        :class:`CohortManifestDuplicateCellError` when two cells
        share a ``(slug, shape_point)`` pair.
        :class:`CohortManifestAccountingDriftError` when the
        ``attempted_cell_count`` or ``admitted_cell_count`` does not
        match the actual cells array.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    schema = _load_schema()

    try:
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.ValidationError as exc:
        raise CohortManifestSchemaError(
            detail=str(exc.message),
            json_pointer=_format_json_pointer(exc),
        ) from exc

    # Build-variant top-level <-> nested identity_tuple lockstep runs FIRST,
    # over the RAW payload cells, BEFORE CohortManifestCell.from_dict (which
    # itself calls the shared normalizer and would otherwise raise a BARE
    # ValueError on a malformed / internally-inconsistent variant pair). Running
    # the typed check first guarantees every variant-identity violation surfaces
    # as CohortManifestVariantLockstepError, not an untyped exception.
    _check_variant_lockstep(payload["cells"])
    _check_replacement_history_variant_consistency(payload)
    cells: list[CohortManifestCell] = [
        CohortManifestCell.from_dict(raw_cell)
        for raw_cell in payload["cells"]
    ]
    _check_unique_cells(cells)
    _check_identity_lockstep(cells)
    _check_accounting(payload, cells)

    return CohortManifest(
        schema_version=int(payload["schema_version"]),
        frozen_at_commit=str(payload["frozen_at_commit"]),
        attempted_cell_count=int(payload["attempted_cell_count"]),
        admitted_cell_count=int(payload["admitted_cell_count"]),
        replacement_history=tuple(payload["replacement_history"]),
        cells=tuple(cells),
        raw=payload,
    )


def _check_identity_lockstep(cells: list[CohortManifestCell]) -> None:
    """For each cell, assert the top-level (slug, shape_point, arch,
    device) tuple matches the cell's identity_tuple values byte-for-byte.

    The manifest is a drift-sensitive artifact: if cell.slug says
    one kernel but identity_tuple.slug says another, downstream
    consumers (cascade-recap, the predictor join key, the comparison
    runner) cannot tell which row is authoritative. The validator
    raises :class:`CohortManifestIdentityLockstepError` on the first
    mismatch with the offending field name + the two divergent values.
    """
    for cell in cells:
        for field_name in ("slug", "shape_point", "arch", "device"):
            cell_value = getattr(cell, field_name)
            identity_value = cell.identity_tuple.get(field_name)
            if cell_value != identity_value:
                raise CohortManifestIdentityLockstepError(
                    cell_slug=cell.slug,
                    cell_shape_point=cell.shape_point,
                    field=field_name,
                    cell_value=cell_value,
                    identity_value=(
                        str(identity_value)
                        if identity_value is not None
                        else "<missing>"
                    ),
                )


def _check_variant_lockstep(raw_cells: list[Mapping[str, Any]]) -> None:
    """For every cell, require its TOP-LEVEL build-variant identity
    (``register_cap`` / ``variant_id``) to agree with the build-variant
    identity inside its nested ``identity_tuple``: a base cell omits the pair
    at both levels; a register-cap variant normalizes to the SAME canonical
    ``register_cap_<n>`` at both levels. Operates on the RAW cell payloads
    (which carry the top-level + nested pairs verbatim) and routes both
    through the shared ``variant_lockstep_error`` (single canonical normalizer
    -- no re-inlined parsing). Raises
    :class:`CohortManifestVariantLockstepError` on the first mismatch."""
    for cell in raw_cells:
        err = variant_lockstep_error(cell, cell.get("identity_tuple"))
        if err is not None:
            raise CohortManifestVariantLockstepError(
                cell_slug=str(cell.get("slug")),
                cell_shape_point=str(cell.get("shape_point")),
                detail=err,
            )


def _check_replacement_history_variant_consistency(
    payload: Mapping[str, Any],
) -> None:
    """Each ``replacement_history`` item's ``old_identity_tuple`` /
    ``new_identity_tuple`` may carry the register-cap variant axis (the schema
    ``$ref``s the same identity_tuple def that permits register_cap /
    variant_id). A single identity tuple has no top-level pair to lockstep
    against, but its OWN register_cap vs variant_id must still be internally
    consistent + well-typed -- so route each through the shared normalizer
    (which raises on an inconsistent / malformed / noncanonical pair) and fail
    closed. This closes the audit-record arm of the same variant-identity
    integrity class the cohort cells enforce, so a contradictory variant tuple
    can never be recorded in the manifest's swap history."""
    for i, item in enumerate(payload.get("replacement_history", []) or []):
        for key in ("old_identity_tuple", "new_identity_tuple"):
            it = item.get(key) or {}
            try:
                cell_variant_id(it)
            except ValueError as exc:
                raise CohortManifestVariantLockstepError(
                    cell_slug=str(it.get("slug")),
                    cell_shape_point=str(it.get("shape_point")),
                    detail=f"replacement_history[{i}].{key}: {exc}",
                ) from exc


def _check_unique_cells(cells: list[CohortManifestCell]) -> None:
    # Mixed-arity uniqueness: a base cell keys on (slug, shape_point); a
    # register-cap build variant widens to (slug, shape_point, variant_id) so
    # sibling variants of one (slug, shape_point) are distinct cells, not
    # duplicates. Byte-identical for a base-only manifest (variant_id == "").
    counts: dict[tuple[str, ...], int] = {}
    for cell in cells:
        key = (
            (cell.slug, cell.shape_point) if not cell.variant_id
            else (cell.slug, cell.shape_point, cell.variant_id)
        )
        counts[key] = counts.get(key, 0) + 1
    for key, count in counts.items():
        if count > 1:
            raise CohortManifestDuplicateCellError(
                slug=key[0], shape_point=key[1], count=count,
            )


def _check_accounting(
    payload: Mapping[str, Any], cells: list[CohortManifestCell],
) -> None:
    declared_attempted = int(payload["attempted_cell_count"])
    if declared_attempted != len(cells):
        raise CohortManifestAccountingDriftError(
            counter_name="attempted_cell_count",
            declared=declared_attempted,
            observed=len(cells),
        )
    declared_admitted = int(payload["admitted_cell_count"])
    observed_admitted = sum(
        1 for cell in cells if cell.admission_status == "admitted"
    )
    if declared_admitted != observed_admitted:
        raise CohortManifestAccountingDriftError(
            counter_name="admitted_cell_count",
            declared=declared_admitted,
            observed=observed_admitted,
        )


# ---------------------------------------------------------------------
# Path-reconciliation helper (missing trace / label fallback)
# ---------------------------------------------------------------------


def _reconcile_cell_paths(
    cell: CohortManifestCell, repo_root: Path,
) -> CohortManifestCell:
    """Inspect ``cell.expected_trace_path`` and
    ``cell.expected_label_path``. If either resolves to a non-existent
    file on disk under ``repo_root``, return a copy of the cell whose
    ``admission_status`` is ``attempted_excluded`` and whose
    ``exclusion_typed_reason`` is ``trace_missing`` or
    ``label_missing`` (whichever path is absent first). The trace
    path is checked first; if it is present, the label path is
    checked next.

    The reconciliation only fires for cells already marked
    ``admitted`` in the manifest — already-excluded cells preserve
    their declared `exclusion_typed_reason` because the manifest's
    pre-existing reason is more specific than a generic missing-path
    fallback.
    """
    if cell.admission_status != "admitted":
        return cell
    trace_path = repo_root / cell.expected_trace_path
    if not trace_path.is_file():
        return _replace_cell_admission(
            cell,
            admission_status="attempted_excluded",
            exclusion_typed_reason="trace_missing",
        )
    label_path = repo_root / cell.expected_label_path
    if not label_path.is_file():
        return _replace_cell_admission(
            cell,
            admission_status="attempted_excluded",
            exclusion_typed_reason="label_missing",
        )
    return cell


def _replace_cell_admission(
    cell: CohortManifestCell,
    *,
    admission_status: str,
    exclusion_typed_reason: str | None,
) -> CohortManifestCell:
    """Return a copy of ``cell`` with the new admission_status and
    exclusion_typed_reason. The frozen dataclass requires
    ``dataclasses.replace`` for mutation."""
    import dataclasses
    return dataclasses.replace(
        cell,
        admission_status=admission_status,
        exclusion_typed_reason=exclusion_typed_reason,
    )


def validate_manifest_with_path_reconciliation(
    path: Path | str,
    repo_root: Path | str | None = None,
    *,
    label_admitted_path: Path | str | None = None,
    enforce_label_presence: bool = True,
) -> CohortManifest:
    """Load and validate the cohort manifest, then reconcile each
    admitted cell's `expected_trace_path` / `expected_label_path`
    against the filesystem. Admitted cells whose paths are missing
    on disk are flipped to `attempted_excluded` with the
    `trace_missing` or `label_missing` typed reason.

    When ``enforce_label_presence`` is True (the default), every
    cell that REMAINS admitted post-reconciliation is additionally
    cross-referenced against the canonical
    ``data/local_labels/label_admitted.json`` aggregate on the
    5-field identity key (arch, slug, shape_point, binary_sha256,
    kernel_signature_sha256). Missing identity-tuple rows raise
    :class:`CohortManifestLabelPresenceError`. Override the
    aggregate path via ``label_admitted_path``; pass
    ``enforce_label_presence=False`` to skip the check entirely
    (e.g. for path-reconciliation-only tests that explicitly do
    not exercise the label-presence guard).

    Returns a new :class:`CohortManifest` whose `cells` list reflects
    the reconciled admission states AND whose
    `admitted_cell_count` matches the count of cells that remain
    admitted post-reconciliation. The `attempted_cell_count` is
    unchanged (the cohort's total size does not depend on
    on-disk presence). `replacement_history` and other top-level
    fields carry through.

    The path-reconciliation is the negative-test surface the
    schema cannot express: a manifest whose `expected_trace_path`
    does not exist falls back to `attempted_excluded` with
    `exclusion_typed_reason="trace_missing"`. The runner consumes
    this reconciled manifest and iterates only the reconciled
    admitted set.
    """
    base = validate_manifest(path)
    repo_root_path = (
        Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
    )
    reconciled_cells = tuple(
        _reconcile_cell_paths(cell, repo_root_path)
        for cell in base.cells
    )
    if enforce_label_presence:
        # Honor each admitted cell's OWN expected_label_path: the cell's
        # 5-field identity tuple must be present in the label artifact IT
        # declares. This is strictly stronger than checking one global
        # aggregate — a cell cannot claim admission against a label file
        # that does not actually contain its row. When all cells point at
        # data/local_labels/label_admitted.json (the historical state)
        # this is byte-identical to the prior single-aggregate check.
        # ``label_admitted_path`` still overrides the per-cell path for
        # callers that intentionally pin one aggregate.
        _check_label_presence_per_cell(
            reconciled_cells, repo_root_path, label_admitted_path,
        )
    admitted_count = sum(
        1 for cell in reconciled_cells
        if cell.admission_status == "admitted"
    )
    import dataclasses
    return dataclasses.replace(
        base,
        cells=reconciled_cells,
        admitted_cell_count=admitted_count,
    )


def _load_label_admitted_identity_keys(
    label_admitted_path: Path,
) -> set[tuple[str, str, str, str, str]]:
    """Return the set of 5-field identity keys present in the
    canonical ``label_admitted.json`` aggregate. The key tuple is
    ``(arch, slug, shape_point, binary_sha256, kernel_signature_sha256)``.

    Missing label_admitted.json is treated as an empty set (no admitted
    cell can be label-backed when the corpus does not exist).
    """
    if not label_admitted_path.is_file():
        return set()
    import json as _json
    payload = _json.loads(label_admitted_path.read_text(encoding="utf-8"))
    return {
        (
            str(row.get("arch", "")),
            str(row.get("slug", "")),
            str(row.get("shape_point", "")),
            str(row.get("binary_sha256", "")),
            str(row.get("kernel_signature_sha256", "")),
        )
        for row in payload.get("cells", ())
    }


def _check_label_presence(
    cell: CohortManifestCell,
    label_identity_keys: set[tuple[str, str, str, str, str]],
    label_admitted_path: str,
) -> None:
    """For an admitted cell, require its 5-field identity tuple to be
    present in ``label_identity_keys`` (built from the canonical
    ``label_admitted.json`` aggregate). Excluded cells pass unconditionally.

    Raises :class:`CohortManifestLabelPresenceError` on mismatch.
    """
    if cell.admission_status != "admitted":
        return
    identity_tuple = cell.identity_tuple
    binary_sha256 = str(identity_tuple.get("binary_sha256", ""))
    kernel_signature_sha256 = str(
        identity_tuple.get("kernel_signature_sha256", "")
    )
    key = (
        cell.arch,
        cell.slug,
        cell.shape_point,
        binary_sha256,
        kernel_signature_sha256,
    )
    if key not in label_identity_keys:
        raise CohortManifestLabelPresenceError(
            arch=cell.arch,
            slug=cell.slug,
            shape_point=cell.shape_point,
            binary_sha256=binary_sha256,
            kernel_signature_sha256=kernel_signature_sha256,
            label_admitted_path=label_admitted_path,
        )


def _check_label_presence_per_cell(
    cells,
    repo_root_path: Path,
    label_admitted_path: Path | str | None = None,
) -> None:
    """Check every admitted cell's identity tuple against the label
    artifact named by ITS OWN ``expected_label_path`` (resolved under
    ``repo_root_path``). Identity-key sets are cached per resolved path
    so a label file is read once. When ``label_admitted_path`` is given
    it overrides the per-cell path for all cells (the historical
    single-aggregate behavior). Excluded cells pass unconditionally.

    Raises :class:`CohortManifestLabelPresenceError` on the first miss.
    """
    override = (
        Path(label_admitted_path).resolve()
        if label_admitted_path is not None
        else None
    )
    key_cache: dict[Path, set[tuple[str, str, str, str, str]]] = {}
    for cell in cells:
        if cell.admission_status != "admitted":
            continue
        resolved = override or (repo_root_path / cell.expected_label_path).resolve()
        if resolved not in key_cache:
            key_cache[resolved] = _load_label_admitted_identity_keys(resolved)
        _check_label_presence(cell, key_cache[resolved], str(resolved))


def validate_manifest_with_label_presence(
    path: Path | str,
    label_admitted_path: Path | str | None = None,
) -> CohortManifest:
    """Load and validate the cohort manifest, then cross-reference
    every admitted cell against the canonical
    ``data/local_labels/label_admitted.json`` aggregate on the
    5-field identity key (arch, slug, shape_point, binary_sha256,
    kernel_signature_sha256). Admitted cells whose identity tuple
    is absent from the label corpus raise
    :class:`CohortManifestLabelPresenceError`.

    This closes the false-positive arm where an admitted manifest
    cell points at an existing aggregate file that contains no row
    for the cell's own identity tuple — a state that
    ``_reconcile_cell_paths`` cannot detect because the path-
    existence check passes whenever any non-empty
    ``label_admitted.json`` is present on disk.

    Returns the validated manifest unchanged on success (does NOT
    reconcile admission status; the runner consumes the original
    cohort intact). For automatic flip-to-excluded behavior on
    label miss, compose this validator with the existing
    replacement-history admission-transition path at the caller.
    """
    base = validate_manifest(path)
    # Honor each admitted cell's OWN expected_label_path (strictly
    # stronger; byte-identical when every cell points at the canonical
    # aggregate). ``label_admitted_path`` still pins one aggregate for
    # all cells when a caller passes it explicitly.
    _check_label_presence_per_cell(base.cells, _REPO_ROOT, label_admitted_path)
    return base


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Validate a cohort_manifest.json against the strict "
            "schema + cross-cell invariants + the identity-bound "
            "label-presence check (every admitted cell's 5-field "
            "identity tuple must be present in the canonical "
            "data/local_labels/label_admitted.json aggregate)."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
            / "cohort_manifest.json"
        ),
        help=(
            "Path to the cohort manifest JSON file (default: "
            "data/sm_86_predictor_accuracy_recovery_v2/cohort_manifest.json)."
        ),
    )
    parser.add_argument(
        "--label-admitted",
        type=Path,
        default=None,
        help=(
            "Path to the canonical label_admitted.json aggregate "
            "(default: data/local_labels/label_admitted.json). "
            "Identity-bound label-presence check requires every "
            "admitted manifest cell to have a matching identity-tuple "
            "row in this aggregate."
        ),
    )
    parser.add_argument(
        "--no-label-presence-check",
        action="store_true",
        help=(
            "Skip the identity-bound label-presence check; run "
            "schema + uniqueness + accounting + identity-lockstep "
            "validation only. Use only for path-reconciliation tests "
            "that explicitly do not exercise the label-presence guard."
        ),
    )
    args = parser.parse_args(argv)
    try:
        manifest = validate_manifest_with_path_reconciliation(
            args.manifest,
            label_admitted_path=args.label_admitted,
            enforce_label_presence=not args.no_label_presence_check,
        )
    except CohortManifestSchemaError as exc:
        print(
            f"cohort_manifest_schema_error at {exc.json_pointer!s}: "
            f"{exc.detail}",
            file=sys.stderr,
        )
        return 1
    except CohortManifestDuplicateCellError as exc:
        print(
            f"cohort_manifest_duplicate_cell: ({exc.slug!r}, "
            f"{exc.shape_point!r}) appears {exc.count} times",
            file=sys.stderr,
        )
        return 1
    except CohortManifestAccountingDriftError as exc:
        print(
            f"cohort_manifest_accounting_drift: "
            f"{exc.counter_name}={exc.declared}, observed={exc.observed}",
            file=sys.stderr,
        )
        return 1
    except CohortManifestLabelPresenceError as exc:
        print(
            f"cohort_manifest_label_presence: "
            f"({exc.slug!r}, {exc.shape_point!r}, arch={exc.arch!r}) "
            f"has no matching identity-tuple row in "
            f"{exc.label_admitted_path}",
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: cohort manifest valid; "
        f"attempted={manifest.attempted_cell_count} "
        f"admitted={manifest.admitted_cell_count} "
        f"cells={len(manifest.cells)}",
    )
    return 0


__all__ = [
    "CohortManifest",
    "CohortManifestCell",
    "CohortManifestValidationError",
    "CohortManifestSchemaError",
    "CohortManifestDuplicateCellError",
    "CohortManifestAccountingDriftError",
    "CohortManifestIdentityLockstepError",
    "CohortManifestLabelPresenceError",
    "validate_manifest",
    "validate_manifest_with_path_reconciliation",
    "validate_manifest_with_label_presence",
]


if __name__ == "__main__":
    sys.exit(_main())
