"""Cross-study framework contract (deployed scope).

Single canonical home for the 7-tuple cell join key, SHA-256
validation helpers, post-hoc drift detection, and bootstrap
constants that the sibling project at
`/home/jakeshea/3090ti_5060laptop_cross_study` ships as its
comparison contract. Downstream deployed-scope modules
(`tiered_gate_evaluator`, the future predictor bundle factory,
the future report tables) import from here so a single
authoritative version of the contract exists.

## Surfaces

  * `CellJoinKey` — frozen dataclass with the 7 fields
    `(slug, shape_point, binary_sha256, source_commit,
    kernel_signature_sha256, arch, ncu_metric_version)`. Order
    matches the sibling project's
    `comparison_contract.toml [join_key].fields` array verbatim.

  * `validate_cell_join_key(key)` — field-presence + SHA-256
    shape check. Raises `framework_contract_join_key_missing_field`
    for empty / non-string fields and
    `framework_contract_sha256_malformed` for invalid digests.

  * `validate_sha256_digest(value, field_name)` — accepts a
    64-character lowercase hex string ONLY. Rejects non-strings,
    empty strings, uppercase hex (sibling project's convention
    is lowercase to match `hashlib.sha256(...).hexdigest()`),
    wrong-length hex, and any non-hex character.

  * `detect_post_hoc_key_drift(keys_by_source)` — accepts a
    mapping `{source_label: CellJoinKey}` and raises
    `framework_contract_post_hoc_key_drift` when two sources
    carry the same `(slug, arch)` pair but disagree on any
    other 7-tuple field. The typed error message names the
    sources and the disagreeing field so a downstream report
    consumer can fix the drift without parsing strings.

  * `BOOTSTRAP_RESAMPLES = 2000` and `CI_CONFIDENCE = 0.95` —
    canonical bootstrap constants. Matches the sibling
    project's `geomean.bootstrap_resamples` and
    `geomean.ci_confidence` keys verbatim. Other deployed
    modules import these from here rather than redefining.

  * `assert_bootstrap_constants_match(resamples, confidence)`
    — drift guard for downstream modules that need to assert
    they are using the canonical values. Raises
    `framework_contract_bootstrap_constant_drift` on any
    mismatch.

  * `FRAMEWORK_CONTRACT_TYPED_REASONS` — closed frozenset of
    the 8 typed reasons the module raises (4 cell-join
    reasons + 4 sidecar reasons).

  * `FrameworkContractError` — carries `typed_reason` +
    `message` attributes so downstream callers compose
    typed-unavailable bundles or report-table errors without
    string parsing.

Imports only stdlib. No calibration-subtree imports; the NCU
quarantine boundary is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping


# Canonical bootstrap constants. Other deployed-scope modules
# import these from here rather than redefining; the sibling
# project pins the same values in
# `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/comparison_contract.toml`
# under `[geomean]`.
BOOTSTRAP_RESAMPLES: int = 2000
CI_CONFIDENCE: float = 0.95


# Hex digit set used by the SHA-256 validation helper. Lowercase
# only — the sibling project's convention matches
# `hashlib.sha256(...).hexdigest()` output, which is lowercase
# by definition. Accepting uppercase would silently admit a
# downstream record that hashed by a non-canonical path.
_LOWERCASE_HEX: frozenset[str] = frozenset("0123456789abcdef")
_SHA256_HEX_LENGTH: int = 64


FRAMEWORK_CONTRACT_TYPED_REASONS: frozenset[str] = frozenset({
    "framework_contract_join_key_missing_field",
    "framework_contract_sha256_malformed",
    "framework_contract_post_hoc_key_drift",
    "framework_contract_bootstrap_constant_drift",
    "framework_contract_sidecar_digest_invalid",
    "framework_contract_sidecar_digest_required_missing",
    "framework_contract_sidecar_digest_duplicate",
    "framework_contract_sidecar_circular_input",
})


# Field names whose values must satisfy the SHA-256 shape
# (64-character lowercase hex). The remaining 7-tuple fields
# are free-form strings checked only for presence + non-empty.
_SHA256_FIELDS: frozenset[str] = frozenset({
    "binary_sha256",
    "kernel_signature_sha256",
})


# Canonical sidecar-name vocabulary. Mirrors the sibling
# project's `input_sidecar_digests` schema at
# `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/schemas/cache_predictor.schema.json`
# where `additionalProperties: false` pins the per-input
# allowed names to `sector_stream` (required) and
# `dynamic_trace_pb` (optional cross-check). Names outside
# this allowed set are rejected by
# `validate_input_sidecar_digests` so a downstream
# predictor bundle cannot smuggle an arbitrary input under
# the canonical contract.
INPUT_SIDECAR_ALLOWED_NAMES: frozenset[str] = frozenset({
    "sector_stream",
    "dynamic_trace_pb",
})

# Canonical default required-name contract: every predictor
# bundle MUST list `sector_stream` as an input. A caller
# may explicitly pass `required_names=()` to opt out (for
# example a test that only needs the unknown-name check),
# but the default reflects the sibling project's hard
# requirement.
INPUT_SIDECAR_REQUIRED_NAMES: tuple[str, ...] = ("sector_stream",)

# Sibling-project circular-output names. A predictor that
# lists any of these as an input is recreating the
# sibling's `circular_predictor_input_rejected` failure
# mode: the predictor would consume its own modeled
# output and bootstrap a non-existent ground truth. The
# validator rejects these names independently of any
# `forbidden_self_name` the caller might pass.
CIRCULAR_PREDICTOR_OUTPUT_NAMES: frozenset[str] = frozenset({
    "cache_prediction",
    "cache_predictor",
    "hrd_profile",
    "hit_count",
    "hit_rate",
})


class FrameworkContractError(Exception):
    """Raised on framework-contract violations.

    Carries the typed-reason string so downstream callers (the
    bundle factory, the report tables) can compose typed-
    unavailable bundles or report-table errors without parsing
    message strings.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class CellJoinKey:
    """7-tuple cell join key matching the sibling project's
    `comparison_contract.toml [join_key].fields` verbatim.

    The field ordering is contract-pinned; downstream
    consumers iterate via `dataclasses.fields(CellJoinKey)`
    so any reordering is mechanically caught at construction
    time.
    """

    slug: str
    shape_point: str
    binary_sha256: str
    source_commit: str
    kernel_signature_sha256: str
    arch: str
    ncu_metric_version: str


def validate_sha256_digest(value: object, *, field_name: str) -> str:
    """Accept a 64-character lowercase hex string. Reject any
    other shape with `framework_contract_sha256_malformed`.

    Returns the validated string unchanged.

    Validation rules (each a separate failure path so a
    diagnostic can name the exact violation):

      * Non-string: type error.
      * Wrong length: must be exactly 64 characters.
      * Uppercase or mixed-case hex: must be lowercase.
      * Non-hex character: must use `0-9` + `a-f` only.
    """
    if not isinstance(value, str):
        raise FrameworkContractError(
            "framework_contract_sha256_malformed",
            f"{field_name} must be a string; got "
            f"{type(value).__name__}",
        )
    if len(value) != _SHA256_HEX_LENGTH:
        raise FrameworkContractError(
            "framework_contract_sha256_malformed",
            f"{field_name} must be {_SHA256_HEX_LENGTH} "
            f"characters; got {len(value)} ({value!r})",
        )
    for char in value:
        if char not in _LOWERCASE_HEX:
            raise FrameworkContractError(
                "framework_contract_sha256_malformed",
                f"{field_name} contains non-lowercase-hex "
                f"character {char!r} (the sibling project's "
                f"convention matches hashlib.sha256 "
                f"hexdigest output, which is lowercase): "
                f"{value!r}",
            )
    return value


def validate_cell_join_key(key: CellJoinKey) -> None:
    """Validate every field of a `CellJoinKey`.

    Free-form fields (`slug`, `shape_point`, `source_commit`,
    `arch`, `ncu_metric_version`) must be non-empty strings.
    SHA-256 fields (`binary_sha256`,
    `kernel_signature_sha256`) must satisfy
    `validate_sha256_digest`.

    Raises `framework_contract_join_key_missing_field` for
    missing / empty fields and
    `framework_contract_sha256_malformed` for invalid
    digests.
    """
    if not isinstance(key, CellJoinKey):
        raise FrameworkContractError(
            "framework_contract_join_key_missing_field",
            f"argument must be a CellJoinKey; got "
            f"{type(key).__name__}",
        )
    for field in fields(CellJoinKey):
        value = getattr(key, field.name)
        if not isinstance(value, str) or not value.strip():
            raise FrameworkContractError(
                "framework_contract_join_key_missing_field",
                f"CellJoinKey.{field.name} must be a non-"
                f"empty string; got {value!r}",
            )
        if field.name in _SHA256_FIELDS:
            validate_sha256_digest(value, field_name=field.name)


def cell_join_keys_match(a: CellJoinKey, b: CellJoinKey) -> bool:
    """Return `True` iff every 7-tuple field is equal between
    `a` and `b`. Used as a building block for the post-hoc
    drift detector; an exact equality returns True regardless
    of which sources produced the keys.
    """
    if not isinstance(a, CellJoinKey) or not isinstance(b, CellJoinKey):
        raise FrameworkContractError(
            "framework_contract_join_key_missing_field",
            f"both arguments must be CellJoinKey; got "
            f"{type(a).__name__}, {type(b).__name__}",
        )
    for field in fields(CellJoinKey):
        if getattr(a, field.name) != getattr(b, field.name):
            return False
    return True


def detect_post_hoc_key_drift(
    keys_by_source: Mapping[str, CellJoinKey],
) -> None:
    """Detect post-hoc drift between source records.

    Inputs: a mapping `{source_label: CellJoinKey}` carrying
    one entry per record source (e.g., `{"manifest": ...,
    "ledger": ..., "prediction": ...}`). The detector
    iterates over the sources pairwise; when two keys share
    the same `(slug, arch)` pair but disagree on any other
    7-tuple field, raises
    `framework_contract_post_hoc_key_drift` whose message
    names BOTH source labels AND the disagreeing field.

    A common-key relation requires both `slug` AND `arch`
    matches because cross-arch records legitimately share a
    slug (the same `(slug, sm_86)` and `(slug, sm_100a)` are
    paired records, not drift). Drift is only meaningful
    within the same `(slug, arch)` cell.

    Empty / single-source mappings are no-ops (no pairwise
    comparison possible).

    Source labels must be non-empty strings; an empty key
    raises `framework_contract_join_key_missing_field`
    BEFORE the pairwise sweep so the diagnostic is precise.
    """
    if not isinstance(keys_by_source, Mapping):
        raise FrameworkContractError(
            "framework_contract_join_key_missing_field",
            f"keys_by_source must be a Mapping; got "
            f"{type(keys_by_source).__name__}",
        )
    for source_label, key in keys_by_source.items():
        if not isinstance(source_label, str) or not source_label.strip():
            raise FrameworkContractError(
                "framework_contract_join_key_missing_field",
                f"source label must be a non-empty string; "
                f"got {source_label!r}",
            )
        validate_cell_join_key(key)
    sources = sorted(keys_by_source.keys())
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            label_a = sources[i]
            label_b = sources[j]
            key_a = keys_by_source[label_a]
            key_b = keys_by_source[label_b]
            if (
                key_a.slug != key_b.slug
                or key_a.arch != key_b.arch
            ):
                # Different cell or different arch — not a drift.
                continue
            for field in fields(CellJoinKey):
                value_a = getattr(key_a, field.name)
                value_b = getattr(key_b, field.name)
                if value_a != value_b:
                    raise FrameworkContractError(
                        "framework_contract_post_hoc_key_drift",
                        f"join-key drift between "
                        f"{label_a!r} and {label_b!r} on "
                        f"cell (slug={key_a.slug!r}, "
                        f"arch={key_a.arch!r}): "
                        f"field {field.name!r} differs "
                        f"({value_a!r} vs {value_b!r})",
                    )


def assert_bootstrap_constants_match(
    *,
    resamples: int,
    confidence: float,
) -> None:
    """Assert that a caller's bootstrap constants match the
    canonical values. Downstream modules call this as a
    self-check when they cache local copies of the
    constants.

    Raises `framework_contract_bootstrap_constant_drift`
    naming the disagreeing field on any mismatch.
    """
    if resamples != BOOTSTRAP_RESAMPLES:
        raise FrameworkContractError(
            "framework_contract_bootstrap_constant_drift",
            f"resamples disagrees with canonical "
            f"BOOTSTRAP_RESAMPLES: got {resamples!r}, "
            f"expected {BOOTSTRAP_RESAMPLES!r}",
        )
    if confidence != CI_CONFIDENCE:
        raise FrameworkContractError(
            "framework_contract_bootstrap_constant_drift",
            f"confidence disagrees with canonical "
            f"CI_CONFIDENCE: got {confidence!r}, expected "
            f"{CI_CONFIDENCE!r}",
        )


@dataclass(frozen=True)
class InputSidecarDigest:
    """Per-input sidecar SHA-256 digest record.

    Carries `(name, path, sha256)` for one sidecar input
    consumed by a downstream record (e.g., a predictor
    bundle declaring which sector-stream files it
    consumed). Field shape matches the sibling project's
    `input_sidecar_digests` schema at
    `/home/jakeshea/3090ti_5060laptop_cross_study/workloads/simbench_bw/schemas/cache_predictor.schema.json`
    object form: `name` is the canonical sidecar
    identifier (e.g., `sector_stream`,
    `calibration_corpus`), `path` is the repo-relative
    path the SHA was computed against, `sha256` is the
    64-character lowercase hex digest.
    """

    name: str
    path: str
    sha256: str


_REQUIRED_NAMES_SENTINEL = object()


def validate_input_sidecar_digests(
    digests,
    *,
    required_names = _REQUIRED_NAMES_SENTINEL,
    forbidden_self_name: str = "",
) -> None:
    """Validate a sequence of `InputSidecarDigest` records.

    Failure modes (each its own typed reason so a caller
    can branch on the cause):

      * `framework_contract_sidecar_digest_invalid`:
        non-`InputSidecarDigest` element, empty `name`,
        empty `path`, malformed SHA-256, unknown sidecar
        name (not in `INPUT_SIDECAR_ALLOWED_NAMES`), or
        unsafe path (absolute, or contains `..`
        component).
      * `framework_contract_sidecar_digest_duplicate`:
        two entries share the same `name`.
      * `framework_contract_sidecar_digest_required_missing`:
        one of the canonical `INPUT_SIDECAR_REQUIRED_NAMES`
        (or a caller-supplied override) is absent from
        the digest list.
      * `framework_contract_sidecar_circular_input`:
        a digest's `name` matches a sibling-project
        circular-output name (`hrd_profile`, `hit_rate`,
        `cache_prediction`, `cache_predictor`,
        `hit_count`) OR matches the caller's
        `forbidden_self_name`. Either path recreates
        the sibling project's
        `circular_predictor_input_rejected` failure
        mode.

    `required_names` defaults to
    `INPUT_SIDECAR_REQUIRED_NAMES` (i.e., `sector_stream`
    is required by the canonical contract). Pass
    `required_names=()` to opt out for tests that don't
    need it. `forbidden_self_name` is optional; the
    canonical `CIRCULAR_PREDICTOR_OUTPUT_NAMES` rejection
    fires independently.
    """
    if required_names is _REQUIRED_NAMES_SENTINEL:
        required_names = INPUT_SIDECAR_REQUIRED_NAMES
    if not hasattr(digests, "__iter__"):
        raise FrameworkContractError(
            "framework_contract_sidecar_digest_invalid",
            f"digests must be iterable; got "
            f"{type(digests).__name__}",
        )
    seen_names: set[str] = set()
    materialized: list[InputSidecarDigest] = []
    for digest in digests:
        if not isinstance(digest, InputSidecarDigest):
            raise FrameworkContractError(
                "framework_contract_sidecar_digest_invalid",
                f"every element must be an "
                f"InputSidecarDigest; got "
                f"{type(digest).__name__}",
            )
        if (
            not isinstance(digest.name, str)
            or not digest.name.strip()
        ):
            raise FrameworkContractError(
                "framework_contract_sidecar_digest_invalid",
                f"sidecar name must be a non-empty "
                f"string; got {digest.name!r}",
            )
        # Circular-output rejection runs BEFORE the
        # allowed-name check so the diagnostic explains
        # the cause as "circular self-input" rather than
        # "unknown name" — those names are syntactically
        # valid but semantically forbidden.
        if digest.name in CIRCULAR_PREDICTOR_OUTPUT_NAMES:
            raise FrameworkContractError(
                "framework_contract_sidecar_circular_input",
                f"sidecar name {digest.name!r} is in the "
                f"canonical predictor-output set "
                f"{sorted(CIRCULAR_PREDICTOR_OUTPUT_NAMES)}; "
                f"a predictor that lists its own modeled "
                f"output as an input would recreate the "
                f"sibling project's "
                f"circular_predictor_input_rejected "
                f"failure mode",
            )
        if digest.name not in INPUT_SIDECAR_ALLOWED_NAMES:
            raise FrameworkContractError(
                "framework_contract_sidecar_digest_invalid",
                f"sidecar name {digest.name!r} is not in "
                f"the canonical allowed set "
                f"{sorted(INPUT_SIDECAR_ALLOWED_NAMES)}; "
                f"the sibling contract pins "
                f"`additionalProperties: false` on the "
                f"input_sidecar_digests object",
            )
        if (
            not isinstance(digest.path, str)
            or not digest.path.strip()
        ):
            raise FrameworkContractError(
                "framework_contract_sidecar_digest_invalid",
                f"sidecar path must be a non-empty "
                f"string (name={digest.name!r}); got "
                f"{digest.path!r}",
            )
        # Repo-relative path discipline: reject absolute
        # paths AND `..` traversal so a downstream
        # bundle cannot reference an arbitrary
        # filesystem location or escape the repo root.
        if digest.path.startswith("/"):
            raise FrameworkContractError(
                "framework_contract_sidecar_digest_invalid",
                f"sidecar path must be repo-relative "
                f"(no leading `/`); got "
                f"{digest.path!r} for name "
                f"{digest.name!r}",
            )
        path_parts = digest.path.replace("\\", "/").split("/")
        if ".." in path_parts:
            raise FrameworkContractError(
                "framework_contract_sidecar_digest_invalid",
                f"sidecar path must not contain `..` "
                f"path components (no parent traversal); "
                f"got {digest.path!r} for name "
                f"{digest.name!r}",
            )
        validate_sha256_digest(
            digest.sha256,
            field_name=f"sidecar({digest.name!r}).sha256",
        )
        if digest.name in seen_names:
            raise FrameworkContractError(
                "framework_contract_sidecar_digest_duplicate",
                f"sidecar name {digest.name!r} appears "
                f"more than once; each name must be "
                f"unique within a digest list",
            )
        if (
            forbidden_self_name
            and digest.name == forbidden_self_name
        ):
            raise FrameworkContractError(
                "framework_contract_sidecar_circular_input",
                f"sidecar name {digest.name!r} matches "
                f"the producer's own output sidecar "
                f"(forbidden_self_name={forbidden_self_name!r}); "
                f"a producer that lists its own output "
                f"as an input would recreate the sibling "
                f"project's circular_predictor_input_rejected "
                f"failure mode",
            )
        seen_names.add(digest.name)
        materialized.append(digest)
    for required_name in required_names:
        if required_name not in seen_names:
            raise FrameworkContractError(
                "framework_contract_sidecar_digest_required_missing",
                f"required sidecar name "
                f"{required_name!r} is absent from the "
                f"digest list; observed names: "
                f"{sorted(seen_names)!r}",
            )


def cell_join_key_from_ledger_entry(entry) -> CellJoinKey:
    """Build a `CellJoinKey` from a `LedgerEntry`-shaped
    object. The entry must expose the 7 fields the join
    key requires; this helper looks them up by attribute
    name so the call site does not couple to the
    calibration-subtree dataclass.

    The constructed key is validated before return; a
    malformed entry raises the same typed reason a direct
    `validate_cell_join_key` call would.
    """
    key = CellJoinKey(
        slug=getattr(entry, "cell_name", ""),
        shape_point=getattr(entry, "shape_point", ""),
        binary_sha256=getattr(entry, "binary_sha256", ""),
        source_commit=getattr(entry, "source_commit", ""),
        kernel_signature_sha256=getattr(
            entry, "kernel_signature_sha256", "",
        ),
        arch=getattr(entry, "arch", ""),
        ncu_metric_version=getattr(entry, "ncu_metric_version", ""),
    )
    validate_cell_join_key(key)
    return key


def cell_join_key_from_calibration_input(record) -> CellJoinKey:
    """Build a `CellJoinKey` from a
    `CalibrationInputRecord`-shaped object. The record
    must carry a `cell_join_key` attribute of type
    `CellJoinKey`; this helper validates the carried
    key (rather than re-deriving from individual
    fields) so the record's typed key field is the
    single source of truth.
    """
    key = getattr(record, "cell_join_key", None)
    if not isinstance(key, CellJoinKey):
        raise FrameworkContractError(
            "framework_contract_join_key_missing_field",
            f"CalibrationInputRecord.cell_join_key must "
            f"be a CellJoinKey; got {type(key).__name__}",
        )
    validate_cell_join_key(key)
    return key


def cell_join_key_from_loocv_residual(residual) -> CellJoinKey:
    """Build a `CellJoinKey` from a `LOOCVResidual`-shaped
    object. Mirrors
    `cell_join_key_from_calibration_input`: the residual
    must carry a typed `cell_join_key` populated by
    `run_loocv()` from the held-out record.
    """
    key = getattr(residual, "cell_join_key", None)
    if not isinstance(key, CellJoinKey):
        raise FrameworkContractError(
            "framework_contract_join_key_missing_field",
            f"LOOCVResidual.cell_join_key must be a "
            f"CellJoinKey; got {type(key).__name__}",
        )
    validate_cell_join_key(key)
    return key


def validate_join_key_consistency_across_sources(
    keys_by_source,
) -> None:
    """Gate-side helper: drives
    `detect_post_hoc_key_drift` over a
    `{source_label: CellJoinKey}` mapping that downstream
    callers (the predictor bundle factory, the report
    tables) have already extracted from their respective
    records via the `cell_join_key_from_*` factories.

    This is a thin re-export so the consistency-check
    call site reads naturally at the bundle-factory
    boundary: build keys from every source, then call
    this helper to validate them as a coherent join.
    """
    detect_post_hoc_key_drift(keys_by_source)


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "CIRCULAR_PREDICTOR_OUTPUT_NAMES",
    "CI_CONFIDENCE",
    "CellJoinKey",
    "FRAMEWORK_CONTRACT_TYPED_REASONS",
    "FrameworkContractError",
    "INPUT_SIDECAR_ALLOWED_NAMES",
    "INPUT_SIDECAR_REQUIRED_NAMES",
    "InputSidecarDigest",
    "assert_bootstrap_constants_match",
    "cell_join_key_from_calibration_input",
    "cell_join_key_from_ledger_entry",
    "cell_join_key_from_loocv_residual",
    "cell_join_keys_match",
    "detect_post_hoc_key_drift",
    "validate_cell_join_key",
    "validate_input_sidecar_digests",
    "validate_join_key_consistency_across_sources",
    "validate_sha256_digest",
]
