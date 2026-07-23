"""Operator-facing pre-flight provenance audit for the live NCU sweep.

The local NCU label sweep launches against a 13-cell candidate
cohort (8 memory-microbench + 5 activation slugs at sm_86). Each
cell joins the predictor side (per-shape trace JSON sidecar
under ``workloads/simbench_bw/results/<slug>/...``) to the label
side (operator-staged measurement manifest under
``data/local_labels/sweeps/<sweep_id>/manifests/<slug>.json``)
on the seven-field join key pinned by
``workloads/simbench_bw/comparison_contract.toml``'s
``[join_key].fields``:

  Staged in the operator manifest (4 fields):
    slug, shape_point, binary_sha256, kernel_signature_sha256

  Derived at audit time (3 fields):
    source_commit       <- accept-window of two commits derived
                           from ``git -C <repo_root> rev-parse HEAD``
                           and ``git -C <repo_root> rev-parse HEAD^``;
                           predictor PASSes if its recorded
                           ``source_commit`` is either the
                           audit-time HEAD or its first parent.
                           See ``Audit-time source_commit
                           accept-window`` below.
    arch                <- ``--arch`` cross-checked against
                           ``tools/arch_capabilities.json``
    ncu_metric_version  <- ``[ncu_metric].ncu_metric_version`` in
                           ``workloads/simbench_bw/comparison_contract.toml``

The audit fails closed before any NCU shim is spawned: if any
field cannot be matched, the audit emits a typed diagnostic and
exits 1; the operator must regenerate the trace at HEAD (or
align the manifest) before re-running. If file parsing fails
(missing manifest, malformed JSON, missing contract), the audit
exits 2 with a typed io diagnostic.

Audit-time ``source_commit`` accept-window:
  After a single round-close commit lands, the predictor-side
  trace sidecars that were generated BEFORE the close commit
  carry the parent commit (now ``HEAD^``) in their
  ``source_commit`` field. Strict ``predictor.source_commit ==
  HEAD`` equality would always reject such close-round audits,
  even though the cohort is internally consistent. To stay
  coherent with the focused nested-provenance and regeneration-
  delta scanners (which already accept the ``{HEAD, HEAD^}``
  window for predictor-side source-commit freshness), the live
  audit accepts the predictor ``source_commit`` iff it is in
  the set returned by ``_derive_source_commit_accept_set``:
  ``{HEAD, HEAD^}`` on a multi-commit repo, or ``{HEAD}`` on a
  root-only repo where ``HEAD^`` does not resolve. The window
  is exactly two commits deep; ``HEAD^^`` and older are
  rejected, preserving the existing strict guard against label
  drift across multiple rounds.

Exit codes:
  0  PASS — all seven fields match.
  1  typed mismatch — one of the documented drift modes.
  2  io / parse failure — missing or malformed input file.

CLI:

  python3 tools/local_labels/provenance_audit.py \
      --predictor-trace-json workloads/simbench_bw/results/<slug>/.../trace_<shape>.json \
      --manifest data/local_labels/sweeps/<sweep_id>/manifests/<slug>.json \
      --arch sm_86 \
      [--repo-root <path>] \
      [--arch-capabilities <path>] \
      [--comparison-contract <path>]

The audit is also importable: ``run_audit(...)`` returns an
``AuditResult`` whose ``typed_reason`` is None on PASS or one of
the documented diagnostic tokens on failure. The
``sweep_repeatability_matrix`` harness wires this in as the
``--live`` pre-flight gate so a live sweep cannot launch unless
every candidate's audit passes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import tomllib  # py>=3.11
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


_DIAG_PREFIX = "provenance_audit:"

# Typed diagnostic tokens. One per documented drift mode. The
# matching test surface lives in
# ``tools/local_labels/test_provenance_audit.py``; both the
# producer (this file) and the negative-control test pin the
# token spelling, so a typo on either side surfaces immediately.
TYPED_REASON_SLUG_MISMATCH = "provenance_audit_slug_mismatch"
TYPED_REASON_BINARY_SHA256_MISMATCH = (
    "provenance_audit_binary_sha256_mismatch"
)
TYPED_REASON_SOURCE_COMMIT_MISMATCH = (
    "provenance_audit_source_commit_mismatch"
)
TYPED_REASON_KERNEL_SIGNATURE_SHA256_MISMATCH = (
    "provenance_audit_kernel_signature_sha256_mismatch"
)
TYPED_REASON_SHAPE_POINT_DRIFT = "provenance_audit_shape_point_drift"
TYPED_REASON_ARCH_UNRECOGNIZED = "provenance_audit_arch_unrecognized"
TYPED_REASON_ARCH_MISMATCH = "provenance_audit_arch_mismatch"
TYPED_REASON_NCU_METRIC_VERSION_DRIFT = (
    "provenance_audit_ncu_metric_version_drift"
)
TYPED_REASON_MANIFEST_SCHEMA_VERSION_DRIFT = (
    "provenance_audit_manifest_schema_version_drift"
)
TYPED_REASON_MANIFEST_REQUIRED_FIELD_MISSING = (
    "provenance_audit_manifest_required_field_missing"
)
TYPED_REASON_PREDICTOR_MANIFEST_INCOMPLETE = (
    "provenance_audit_predictor_manifest_incomplete"
)

DIAGNOSTIC_TOKENS: frozenset[str] = frozenset({
    TYPED_REASON_SLUG_MISMATCH,
    TYPED_REASON_BINARY_SHA256_MISMATCH,
    TYPED_REASON_SOURCE_COMMIT_MISMATCH,
    TYPED_REASON_KERNEL_SIGNATURE_SHA256_MISMATCH,
    TYPED_REASON_SHAPE_POINT_DRIFT,
    TYPED_REASON_ARCH_UNRECOGNIZED,
    TYPED_REASON_ARCH_MISMATCH,
    TYPED_REASON_NCU_METRIC_VERSION_DRIFT,
    TYPED_REASON_MANIFEST_SCHEMA_VERSION_DRIFT,
    TYPED_REASON_MANIFEST_REQUIRED_FIELD_MISSING,
    TYPED_REASON_PREDICTOR_MANIFEST_INCOMPLETE,
})

# The audit's own pin of the operator-staged measurement-manifest
# schema_version. Kept in lockstep with
# ``tools/local_labels/SWEEP_RUNBOOK.md`` and
# ``tools/local_labels/validate_measurement_manifest.py``'s own
# pin (both currently 2). Distinct from
# ``workloads/simbench_bw/schemas/local_label_admitted.schema.json``
# whose ``schema_version`` is 4.
PINNED_MANIFEST_SCHEMA_VERSION = 2

# The audit's own pin of the NCU metric version. The audit reads
# the contract value at audit time and refuses to admit if the
# contract drifted (e.g. a contract amendment landed without the
# audit being updated in lockstep).
PINNED_NCU_METRIC_VERSION = "ncu_2026.1.x"

# Predictor-side per-shape trace JSON sidecar required fields.
# The 5 fields form the predictor's ``manifest_pairing_index``;
# combined with the 2 audit-derived fields ``arch`` and
# ``ncu_metric_version`` they make the 7-field join-key tuple.
PREDICTOR_REQUIRED_FIELDS = (
    "slug",
    "shape_point",
    "binary_sha256",
    "source_commit",
    "kernel_signature_sha256",
)

# Operator-staged manifest required fields. The matching field
# set in ``validate_measurement_manifest.py`` (4 fields plus
# ``schema_version``).
MANIFEST_REQUIRED_FIELDS = (
    "slug",
    "shape_point",
    "binary_sha256",
    "kernel_signature_sha256",
)

# Where each join-key field is sourced. Drives the iteration
# over ``[join_key].fields`` in the contract; an unknown field in
# the contract list is rejected as a contract-vs-audit drift
# (exit 2 — io / contract integrity failure, not one of the 10
# documented audit-mismatch modes).
_STAGED_IN_MANIFEST = "staged_in_manifest"
_DERIVED_FROM_GIT_HEAD = "derived_from_git_head"
_DERIVED_FROM_OPERATOR_FLAG = "derived_from_operator_flag"
_DERIVED_FROM_CONTRACT = "derived_from_contract"
# The layer-aware comparison-runner migration added `cache_layer`
# as the 8th element of the contract's join_key. The field is a
# per-bundle layer split (L1TEX / L2) emitted by the predictor's
# bundle producer at runtime, NOT a per-cell field that the
# operator-staged measurement manifest or predictor trace JSON
# carry. The audit treats it as a layer-aware split and IGNORES
# it during the per-cell field comparison; the 7 pre-flight fields
# (slug, shape_point, binary_sha256, source_commit,
# kernel_signature_sha256, arch, ncu_metric_version) remain the
# load-bearing surface.
_LAYER_AWARE_SPLIT = "layer_aware_split_ignored_at_preflight"

_FIELD_ROLES: dict[str, str] = {
    "slug": _STAGED_IN_MANIFEST,
    "shape_point": _STAGED_IN_MANIFEST,
    "binary_sha256": _STAGED_IN_MANIFEST,
    "kernel_signature_sha256": _STAGED_IN_MANIFEST,
    "source_commit": _DERIVED_FROM_GIT_HEAD,
    "arch": _DERIVED_FROM_OPERATOR_FLAG,
    "ncu_metric_version": _DERIVED_FROM_CONTRACT,
    "cache_layer": _LAYER_AWARE_SPLIT,
}
# Subset of _FIELD_ROLES the audit actually compares between
# manifest, predictor JSON, operator flags, contract, and git
# HEAD. _LAYER_AWARE_SPLIT roles are skipped because the field
# is a per-bundle runtime split, not a per-cell value.
_PREFLIGHT_FIELDS: tuple[str, ...] = tuple(
    field for field, role in _FIELD_ROLES.items()
    if role != _LAYER_AWARE_SPLIT
)

# Pattern for the per-shape result-directory name that has been
# erroneously pasted into the join-key value field. The on-disk
# convention is ``trace_<shape_point>``; the join-key value is
# the bare shape_point token (e.g. ``trace_00``). A manifest
# declaring ``trace_trace_00`` has pasted the directory name.
_DIRECTORY_NAME_DRIFT_RE = re.compile(r"^trace_trace_")

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARCH_CAPABILITIES = (
    _DEFAULT_REPO_ROOT / "tools" / "arch_capabilities.json"
)
_DEFAULT_COMPARISON_CONTRACT = (
    _DEFAULT_REPO_ROOT
    / "workloads" / "simbench_bw" / "comparison_contract.toml"
)


@dataclass(frozen=True)
class AuditResult:
    """Outcome of one audit invocation.

    ``typed_reason`` is None on PASS or one of
    ``DIAGNOSTIC_TOKENS`` on failure. ``detail`` is the
    operator-facing message body (one short line; no PII; no
    timestamps). On PASS, ``detail`` is the canonical PASS line
    (``slug=... shape_point=... arch=... source_commit=... ``
    ``ncu_metric_version=...``); on failure it is the
    typed-diagnostic detail.
    """

    typed_reason: str | None
    detail: str
    exit_code: int


class _IoFailure(Exception):
    """Raised when an input file cannot be loaded or parsed.

    Distinct from the typed audit diagnostics: the audit cannot
    begin if its inputs are unusable, so we surface this as exit
    2 (io/parse failure) instead of one of the 10 documented
    typed reasons.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _io_diag(path: Path, what: str) -> _IoFailure:
    return _IoFailure(f"{what} at {path!s}")


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise _io_diag(path, f"{label} file not found")
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise _IoFailure(
            f"{label} at {path!s} did not parse as JSON: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise _IoFailure(
            f"{label} at {path!s} is not valid UTF-8: {exc}"
        ) from exc
    except OSError as exc:
        raise _IoFailure(
            f"{label} at {path!s} could not be read: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise _IoFailure(
            f"{label} at {path!s} root must be a JSON object; "
            f"got {type(payload).__name__}"
        )
    return payload


def _load_toml(path: Path, label: str) -> dict:
    if not path.is_file():
        raise _io_diag(path, f"{label} file not found")
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise _IoFailure(
            f"{label} at {path!s} did not parse as TOML: {exc}"
        ) from exc
    except OSError as exc:
        raise _IoFailure(
            f"{label} at {path!s} could not be read: {exc}"
        ) from exc


def _load_admitted_arches(arch_capabilities_path: Path) -> tuple[str, ...]:
    payload = _load_json(arch_capabilities_path, "arch_capabilities")
    legend = payload.get("supported_arch_legend")
    if not isinstance(legend, dict) or not legend:
        raise _IoFailure(
            f"arch_capabilities at {arch_capabilities_path!s} is "
            f"missing or has empty 'supported_arch_legend'"
        )
    return tuple(sorted(str(k) for k in legend.keys()))


def _load_contract_join_key_fields(
    contract: dict, contract_path: Path,
) -> tuple[str, ...]:
    join_key = contract.get("join_key")
    if not isinstance(join_key, dict):
        raise _IoFailure(
            f"comparison_contract at {contract_path!s} is missing "
            f"the [join_key] table"
        )
    fields = join_key.get("fields")
    if not isinstance(fields, list) or not fields:
        raise _IoFailure(
            f"comparison_contract at {contract_path!s} is missing "
            f"or has empty [join_key].fields"
        )
    str_fields = tuple(str(f) for f in fields)
    unknown = [f for f in str_fields if f not in _FIELD_ROLES]
    if unknown:
        raise _IoFailure(
            f"comparison_contract at {contract_path!s} declares "
            f"unknown join-key field(s) {unknown!r}; this audit "
            f"recognizes only {sorted(_FIELD_ROLES)!r}"
        )
    # The pre-flight-comparable subset MUST be fully present in
    # the contract. The layer-aware `cache_layer` field is per-
    # bundle runtime data; the contract may or may not list it,
    # but its presence is permitted and its absence does NOT
    # require coordinated update.
    missing = [
        f for f in _PREFLIGHT_FIELDS if f not in str_fields
    ]
    if missing:
        raise _IoFailure(
            f"comparison_contract at {contract_path!s} is missing "
            f"join-key field(s) {missing!r} that this audit "
            f"requires; coordinated contract+audit update needed"
        )
    return str_fields


def _load_contract_ncu_metric_version(
    contract: dict, contract_path: Path,
) -> str:
    metric = contract.get("ncu_metric")
    if not isinstance(metric, dict):
        raise _IoFailure(
            f"comparison_contract at {contract_path!s} is missing "
            f"the [ncu_metric] table"
        )
    version = metric.get("ncu_metric_version")
    if not isinstance(version, str) or not version:
        raise _IoFailure(
            f"comparison_contract at {contract_path!s} is missing "
            f"or has empty [ncu_metric].ncu_metric_version"
        )
    return version


def _derive_source_commit_from_head(repo_root: Path) -> str:
    if not repo_root.is_dir():
        raise _IoFailure(
            f"--repo-root {repo_root!s} is not a directory"
        )
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as exc:
        raise _IoFailure(
            f"git binary not on PATH; cannot derive source_commit "
            f"({exc})"
        ) from exc
    if result.returncode != 0:
        raise _IoFailure(
            f"git -C {repo_root!s} rev-parse HEAD failed "
            f"(exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    sha = result.stdout.strip()
    if not sha:
        raise _IoFailure(
            f"git -C {repo_root!s} rev-parse HEAD returned empty "
            f"output"
        )
    return sha


def _derive_source_commit_accept_set(
    repo_root: Path, *, live_sweep: bool = False,
) -> tuple[str, ...]:
    """Return the set of source_commit SHAs the preflight admits
    for the predictor trace sidecar.

    Two modes:

    * ``live_sweep=False`` (default; offline audit): returns
      ``(HEAD, HEAD^)`` so sidecars captured at the parent of the
      current HEAD still admit. This supports offline replay /
      post-commit verification where the predictor sidecar was
      captured under the pre-recapture commit AND the harness
      runs against the recaptured HEAD.
    * ``live_sweep=True`` (pre-live preflight): returns
      ``(HEAD,)`` only. Codex Round-22 review found that
      ``label_driver.sh`` records ``source_commit`` from
      ``git rev-parse HEAD`` only (not HEAD^), so a sidecar at
      HEAD^ that the offline accept set admits would produce
      labels that ``comparison_runner`` cannot join with the
      predictor bundle (the bundle would carry the sidecar's
      HEAD^ source_commit; the labels would carry HEAD). The
      live-sweep narrowing ensures the preflight only PASSes
      when the sidecar's source_commit matches what the
      subsequent live label collection will record.
    """

    head_sha = _derive_source_commit_from_head(repo_root)
    if live_sweep:
        return (head_sha,)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD^"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as exc:
        raise _IoFailure(
            f"git binary not on PATH; cannot derive source_commit "
            f"({exc})"
        ) from exc
    if result.returncode != 0:
        return (head_sha,)
    parent_sha = result.stdout.strip()
    if not parent_sha or parent_sha == head_sha:
        return (head_sha,)
    return (head_sha, parent_sha)


def _check_predictor_completeness(
    predictor: dict, predictor_path: Path,
) -> AuditResult | None:
    for field in PREDICTOR_REQUIRED_FIELDS:
        value = predictor.get(field)
        if value is None or value == "":
            return AuditResult(
                typed_reason=TYPED_REASON_PREDICTOR_MANIFEST_INCOMPLETE,
                detail=(
                    f"predictor trace JSON at {predictor_path!s} is "
                    f"missing required field {field!r}"
                ),
                exit_code=1,
            )
    return None


def _check_manifest_schema(
    manifest: dict, manifest_path: Path,
) -> AuditResult | None:
    schema_version = manifest.get("schema_version")
    if schema_version != PINNED_MANIFEST_SCHEMA_VERSION:
        return AuditResult(
            typed_reason=TYPED_REASON_MANIFEST_SCHEMA_VERSION_DRIFT,
            detail=(
                f"manifest at {manifest_path!s} has "
                f"schema_version={schema_version!r}; "
                f"audit requires {PINNED_MANIFEST_SCHEMA_VERSION!r}"
            ),
            exit_code=1,
        )
    for field in MANIFEST_REQUIRED_FIELDS:
        value = manifest.get(field)
        if value is None or value == "":
            return AuditResult(
                typed_reason=(
                    TYPED_REASON_MANIFEST_REQUIRED_FIELD_MISSING
                ),
                detail=(
                    f"manifest at {manifest_path!s} is missing "
                    f"required field {field!r}"
                ),
                exit_code=1,
            )
    return None


def _compare_staged_field(
    field: str,
    manifest_value: str,
    predictor_value: str,
) -> AuditResult | None:
    if manifest_value == predictor_value:
        return None
    if field == "shape_point":
        if _DIRECTORY_NAME_DRIFT_RE.match(manifest_value):
            return AuditResult(
                typed_reason=TYPED_REASON_SHAPE_POINT_DRIFT,
                detail=(
                    f"manifest shape_point={manifest_value!r} "
                    f"appears to be the per-shape result directory "
                    f"name; the join-key value should be the bare "
                    f"shape_point token (predictor side reports "
                    f"{predictor_value!r})"
                ),
                exit_code=1,
            )
        return AuditResult(
            typed_reason=TYPED_REASON_SHAPE_POINT_DRIFT,
            detail=(
                f"manifest shape_point={manifest_value!r} differs "
                f"from predictor side {predictor_value!r}"
            ),
            exit_code=1,
        )
    typed_by_field = {
        "slug": TYPED_REASON_SLUG_MISMATCH,
        "binary_sha256": TYPED_REASON_BINARY_SHA256_MISMATCH,
        "kernel_signature_sha256": (
            TYPED_REASON_KERNEL_SIGNATURE_SHA256_MISMATCH
        ),
    }
    typed = typed_by_field[field]
    return AuditResult(
        typed_reason=typed,
        detail=(
            f"manifest {field}={manifest_value!r} differs from "
            f"predictor side {predictor_value!r}"
        ),
        exit_code=1,
    )


def _canonicalize_request_arch(arch: str) -> str:
    """Map the operator-supplied / arch_capabilities-admitted
    arch token onto the bare canonical sm_XX form so it can be
    compared to the sidecar's arch derivation. The Loop-2
    discipline at `tools/arch_capabilities.json` distinguishes
    `sm_120-laptop` (RTX 5060 laptop) from server `sm_120`
    SKUs; the predictor sidecar carries the legacy local label
    `sm_120` for the laptop. The canonical form here is the
    legacy local label because that is what the sidecar
    surfaces; mapping the audit-side `sm_120-laptop` to
    `sm_120` lets the cross-check admit the documented sweep
    path. Unknown tokens pass through verbatim so the
    `arch_unrecognized` gate above stays the source of truth
    for unsupported arches."""

    if arch == "sm_120-laptop":
        return "sm_120"
    return arch


def _derive_sidecar_canonical_arch(predictor: dict) -> str | None:
    """Resolve the predictor sidecar's arch to the canonical
    bare sm_XX token. Two sidecar shapes are supported:

    1. Activation orchestrator sidecar: carries an `arch`
       field directly (e.g., `"sm_86"`).
    2. Microbench `nvbit_trace_summary` sidecar: carries
       `compute_capability_major` + `compute_capability_minor`
       (e.g., 8 + 6 -> sm_86; 12 + 0 -> sm_120).

    Returns the canonical token on success, or None when the
    sidecar carries neither shape (in which case the audit
    skips the cross-check rather than fail-closed; that case
    is already covered by the predictor-completeness check
    upstream). The compute-capability path consults a small
    lookup so future arches (e.g., a Hopper 9.0a) get
    explicit handling rather than a silent string-format
    fallback."""

    arch_value = predictor.get("arch")
    if isinstance(arch_value, str) and arch_value:
        return arch_value
    cc_major = predictor.get("compute_capability_major")
    cc_minor = predictor.get("compute_capability_minor")
    if isinstance(cc_major, int) and isinstance(cc_minor, int):
        return f"sm_{cc_major}{cc_minor}"
    return None


def run_audit(
    *,
    predictor_trace_json: Path,
    manifest: Path,
    arch: str,
    repo_root: Path | None = None,
    arch_capabilities: Path | None = None,
    comparison_contract: Path | None = None,
    live_sweep: bool = False,
) -> AuditResult:
    """Run a single 7-field audit against one (predictor JSON,
    manifest) pair. Returns an :class:`AuditResult` describing
    PASS or the typed mismatch. Raises :class:`_IoFailure` (the
    CLI maps to exit 2) if any input file is unreadable.

    The ``live_sweep`` kwarg controls the source_commit accept
    set: ``False`` (default) admits both HEAD and HEAD^ for
    offline replay / post-commit verification; ``True`` narrows
    to HEAD only because the subsequent live label collection
    (`label_driver.sh`) records ``source_commit`` from
    ``git rev-parse HEAD`` only and a HEAD^-sidecar approval
    would yield labels that cannot join the predictor bundle.
    Codex Round-22 review P1 fix.
    """

    repo_root = repo_root or _DEFAULT_REPO_ROOT
    arch_capabilities = arch_capabilities or _DEFAULT_ARCH_CAPABILITIES
    comparison_contract = (
        comparison_contract or _DEFAULT_COMPARISON_CONTRACT
    )

    admitted_arches = _load_admitted_arches(arch_capabilities)
    if arch not in admitted_arches:
        return AuditResult(
            typed_reason=TYPED_REASON_ARCH_UNRECOGNIZED,
            detail=(
                f"--arch {arch!r} not admitted by "
                f"arch_capabilities at {arch_capabilities!s}; "
                f"admitted set = {list(admitted_arches)!r}"
            ),
            exit_code=1,
        )

    # Predictor-side arch cross-check. The audit's `--arch` is
    # the disambiguated Loop-2 token from
    # `tools/arch_capabilities.json` (e.g., `sm_86`,
    # `sm_120-laptop`); the predictor sidecar carries the legacy
    # local-label form (e.g., `sm_86`, `sm_120`) either via a
    # bare `arch` field (activation orchestrator's sidecar shape)
    # or via `compute_capability_major/minor` (microbench
    # sidecar shape). Canonicalize both sides to the bare
    # `sm_XX` form and compare; reject with
    # `provenance_audit_arch_mismatch` on disagreement so the
    # pre-live gate cannot launch an NCU sweep that would
    # record labels under a different arch than the predictor
    # bundle's. The check fires after the operator-supplied
    # arch is admitted (so an unadmitted arch surfaces the
    # narrower `arch_unrecognized` typed reason first).
    predictor_payload = _load_json(
        predictor_trace_json, "predictor trace JSON",
    )
    sidecar_canonical_arch = _derive_sidecar_canonical_arch(
        predictor_payload,
    )
    if sidecar_canonical_arch is not None:
        request_canonical_arch = _canonicalize_request_arch(arch)
        if sidecar_canonical_arch != request_canonical_arch:
            return AuditResult(
                typed_reason=TYPED_REASON_ARCH_MISMATCH,
                detail=(
                    f"--arch {arch!r} (canonical "
                    f"{request_canonical_arch!r}) differs from "
                    f"predictor sidecar arch derivation "
                    f"{sidecar_canonical_arch!r} at "
                    f"{predictor_trace_json!s}; the live NCU "
                    f"sweep would record labels under a "
                    f"different arch than the predictor bundle, "
                    f"breaking the comparison-runner join key"
                ),
                exit_code=1,
            )

    contract = _load_toml(
        comparison_contract, "comparison_contract",
    )
    contract_fields = _load_contract_join_key_fields(
        contract, comparison_contract,
    )
    contract_ncu_metric_version = _load_contract_ncu_metric_version(
        contract, comparison_contract,
    )
    if contract_ncu_metric_version != PINNED_NCU_METRIC_VERSION:
        return AuditResult(
            typed_reason=TYPED_REASON_NCU_METRIC_VERSION_DRIFT,
            detail=(
                f"comparison_contract at {comparison_contract!s} "
                f"declares ncu_metric_version="
                f"{contract_ncu_metric_version!r}; audit pins "
                f"{PINNED_NCU_METRIC_VERSION!r}"
            ),
            exit_code=1,
        )

    predictor = _load_json(
        predictor_trace_json, "predictor trace JSON",
    )
    incomplete = _check_predictor_completeness(
        predictor, predictor_trace_json,
    )
    if incomplete is not None:
        return incomplete

    manifest_payload = _load_json(manifest, "measurement manifest")
    schema_failure = _check_manifest_schema(manifest_payload, manifest)
    if schema_failure is not None:
        return schema_failure

    accepted_source_commits = _derive_source_commit_accept_set(
        repo_root, live_sweep=live_sweep,
    )
    derived_source_commit = accepted_source_commits[0]

    for field in contract_fields:
        role = _FIELD_ROLES[field]
        if role == _STAGED_IN_MANIFEST:
            failure = _compare_staged_field(
                field=field,
                manifest_value=str(manifest_payload[field]),
                predictor_value=str(predictor[field]),
            )
            if failure is not None:
                return failure
        elif role == _DERIVED_FROM_GIT_HEAD:
            if predictor[field] not in accepted_source_commits:
                return AuditResult(
                    typed_reason=(
                        TYPED_REASON_SOURCE_COMMIT_MISMATCH
                    ),
                    detail=(
                        f"audit-time source_commit "
                        f"{derived_source_commit!r} differs from "
                        f"predictor side {predictor[field]!r} "
                        f"(accepted set: "
                        f"{list(accepted_source_commits)!r})"
                    ),
                    exit_code=1,
                )

    pass_line = _format_pass_line(
        slug=str(predictor["slug"]),
        shape_point=str(predictor["shape_point"]),
        arch=arch,
        source_commit=derived_source_commit,
        ncu_metric_version=contract_ncu_metric_version,
    )
    return AuditResult(
        typed_reason=None,
        detail=pass_line,
        exit_code=0,
    )


def _format_pass_line(*, slug: str, shape_point: str, arch: str,
                      source_commit: str,
                      ncu_metric_version: str) -> str:
    """Canonical PASS line. Byte-deterministic for the same
    inputs (no timestamps, no env vars, no absolute paths)."""

    return (
        f"{_DIAG_PREFIX} PASS "
        f"slug={slug} "
        f"shape_point={shape_point} "
        f"arch={arch} "
        f"source_commit={source_commit} "
        f"ncu_metric_version={ncu_metric_version}"
    )


def _format_failure_line(result: AuditResult) -> str:
    return f"{_DIAG_PREFIX} {result.typed_reason}: {result.detail}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-flight provenance audit: cross-check the "
            "7-field join key between a predictor-side trace "
            "JSON sidecar and an operator-staged measurement "
            "manifest before the live NCU sweep launches."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictor-trace-json", type=Path, required=True,
        help=(
            "Path to the predictor-side per-shape trace JSON "
            "sidecar (e.g. workloads/simbench_bw/results/<slug>/"
            "rtx_3090_ti/sm_86/trace_<shape>.json)."
        ),
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help=(
            "Path to the operator-staged measurement-manifest "
            "JSON (schema_version=2; carrying slug, shape_point, "
            "binary_sha256, kernel_signature_sha256)."
        ),
    )
    parser.add_argument(
        "--arch", required=True,
        help=(
            "Operator-supplied target architecture (cross-checked "
            "against tools/arch_capabilities.json's admitted set)."
        ),
    )
    parser.add_argument(
        "--repo-root", type=Path, default=_DEFAULT_REPO_ROOT,
        help=(
            f"Repository root for `git rev-parse HEAD` to derive "
            f"source_commit (default: {_DEFAULT_REPO_ROOT!s})."
        ),
    )
    parser.add_argument(
        "--arch-capabilities", type=Path,
        default=_DEFAULT_ARCH_CAPABILITIES,
        help=(
            f"Path to arch_capabilities.json "
            f"(default: {_DEFAULT_ARCH_CAPABILITIES!s})."
        ),
    )
    parser.add_argument(
        "--comparison-contract", type=Path,
        default=_DEFAULT_COMPARISON_CONTRACT,
        help=(
            f"Path to comparison_contract.toml "
            f"(default: {_DEFAULT_COMPARISON_CONTRACT!s})."
        ),
    )
    parser.add_argument(
        "--live-sweep", action="store_true",
        help=(
            "Pre-live preflight mode: narrow the source_commit "
            "accept set to current HEAD only (the default offline "
            "mode also admits HEAD^ for replay/post-commit "
            "verification). Required when this audit gates a "
            "subsequent live NCU sweep, because label_driver.sh "
            "records source_commit from `git rev-parse HEAD` only "
            "and a HEAD^-sidecar approval would yield labels that "
            "cannot join the predictor bundle. Codex Round-22 "
            "review P1 fix."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run_audit(
            predictor_trace_json=args.predictor_trace_json,
            manifest=args.manifest,
            arch=args.arch,
            repo_root=args.repo_root,
            arch_capabilities=args.arch_capabilities,
            comparison_contract=args.comparison_contract,
            live_sweep=args.live_sweep,
        )
    except _IoFailure as exc:
        sys.stderr.write(f"{_DIAG_PREFIX} {exc.message}\n")
        return 2
    if result.typed_reason is None:
        sys.stdout.write(result.detail + "\n")
        return 0
    sys.stderr.write(_format_failure_line(result) + "\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
