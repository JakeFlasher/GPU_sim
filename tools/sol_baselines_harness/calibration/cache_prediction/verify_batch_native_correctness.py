# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native-correctness reconciliation verifier for a batch manifest.

Inputs:

    --batch-manifest         Path to a TOML batch manifest holding a list
                             of ``[[slugs]] slug = "..."`` entries.
    --trace-dir              Directory carrying one
                             ``<slug>.trace.jsonl`` (machine-readable
                             trace rows) and one ``<slug>.log``
                             (human-readable Rich-table snapshot +
                             ``EXIT=N`` line) per slug.
    --host-resource-manifest Path to the host-resource availability
                             carve-out manifest (the AC-compatible
                             non-evaluable workload manifest).

For every slug in the batch manifest the verifier:

    1. Resolves the canonical ``workload.jsonl`` (via
       ``baselines/<category>/<slug>/`` → ``SUBSET_MAP`` →
       ``sol-execbench/data/benchmark/<subset>/<def>/``) and enumerates
       every workload UUID.
    2. Loads ``<trace-dir>/<slug>.trace.jsonl`` and the sibling
       ``<slug>.log`` evidence file.
    3. For each canonical UUID, requires either:
         (a) a ``PASSED`` trace row carrying that UUID, OR
         (b) a per-workload carve-out entry in the host-resource
             manifest matching ``(slug, uuid)`` whose ``typed_reason``
             is bound (via ``CARVEOUT_REASON_EVIDENCE``) to a
             compatible trace status + log substring set. A per-
             workload carve-out without a trace row, with a PASSED
             trace row, or with an incompatible status/log fails the
             gate.
    4. Whole-slug ``[[unavailable_slugs]]`` entries are rejected
       outright with the closed typed reason
       ``whole_slug_carveout_requires_per_workload_entries``. Whole-
       slug carve-outs cannot bind a ``typed_reason`` to per-workload
       evidence the way per-workload entries can; multiple workloads
       must be expanded into per-workload ``[[unavailable_workloads]]``
       entries instead.

Failure modes are emitted with closed-vocabulary typed reasons (see
``TYPED_REASONS``). Exits 0 on full reconciliation; exits 2 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# The fail-closed manifest loader lives next to this script. Importing
# it makes the reconciliation verifier reject every manifest that the loader
# itself rejects (invalid typed reasons, duplicate workload UUIDs,
# missing required fields, unsupported schema versions). Direct TOML
# parsing would bypass that closure.
from host_resource_availability import (  # noqa: E402
    HostResourceAvailabilityError,
    HostResourceAvailabilityManifest,
    HostResourceUnavailableEntry,
    HostResourceUnavailableSlugEntry,
    load_host_resource_availability_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


SUBSET_MAP = {
    "L1": "L1",
    "L2": "L2",
    "Quant": "Quant",
    "FIB": "FlashInfer-Bench",
}


# Closed-vocabulary typed reasons. Every failure path emits exactly
# one of these — adding a new failure path requires extending this
# constant (and the closure-pin regression test).
TYPED_REASONS: frozenset[str] = frozenset({
    # Batch / slug resolution failures
    "batch_manifest_unparseable",
    "batch_manifest_no_slugs",
    "baseline_dir_missing_for_slug",
    "ambiguous_baseline_dir_for_slug",
    "unknown_baseline_category",
    "slug_missing_problem_prefix",
    "canonical_problem_dir_missing",
    "canonical_workload_jsonl_missing",
    "canonical_workload_jsonl_unparseable",
    "canonical_workload_uuid_missing",
    # Trace file shape failures
    "missing_trace_file",
    "empty_trace_file",
    "missing_trace_log_file",
    "empty_trace_log_file",
    "unparseable_trace_row",
    "trace_row_missing_uuid",
    "trace_row_missing_status",
    "unrecognised_status",
    "trace_uuid_not_in_canonical_workload",
    "duplicate_uuid_in_trace",
    # Reconciliation failures
    "canonical_uuid_lacks_evidence",
    "carveout_uuid_lacks_trace_row",
    "carveout_uuid_is_passed_in_trace",
    "carveout_reason_evidence_mismatch",
    # Carve-out manifest validation failures
    "host_resource_manifest_unparseable",
    "host_resource_manifest_load_failed",
    "manifest_slug_unknown",
    "manifest_uuid_unknown",
    "manifest_uuid_slug_mismatch",
    "manifest_axes_drift",
    "whole_slug_carveout_slug_unknown",
    "whole_slug_carveout_without_per_workload_evidence",
    # Whole-slug carve-outs are rejected outright by the verifier
    # because they cannot bind a `typed_reason` to per-workload
    # evidence the way per-workload entries can. If a slug needs
    # multiple workloads carved out, list each one as a
    # ``[[unavailable_workloads]]`` row so the reconciliation gate
    # can verify status + log evidence per UUID. The two legacy
    # whole-slug reasons (`whole_slug_carveout_slug_unknown`,
    # `whole_slug_carveout_without_per_workload_evidence`) are kept
    # in the closure for back-compat with any external caller that
    # introspects the set; the verifier no longer emits them.
    "whole_slug_carveout_requires_per_workload_entries",
})


# Carve-out typed-reason -> (expected EvaluationStatus, [required log
# substrings]) binding. The verifier rejects any per-workload carve-out
# whose trace status / log content disagrees with this mapping. A
# carve-out is documented host-resource evidence; the manifest is NOT
# a kernel-correctness escape hatch, so the trace must demonstrate
# the SAME failure mode the manifest claims.
CARVEOUT_REASON_EVIDENCE: dict[str, tuple[str, tuple[str, ...]]] = {
    "timing_oom_exceeds_local_vram": (
        "RUNTIME_ERROR",
        ("Timing failed", "out of memory"),
    ),
    "reference_oom_exceeds_local_vram": (
        "INVALID_REFERENCE",
        ("Reference run", "out of memory"),
    ),
    "reference_timeout_exceeds_local_eval_window": (
        "TIMEOUT",
        (),
    ),
    "correctness_check_oom_exceeds_local_vram": (
        "RUNTIME_ERROR",
        ("Correctness check failed", "out of memory"),
    ),
}


# Valid EvaluationStatus values (mirrors
# ``sol_execbench.core.data.EvaluationStatus``). Pinned as a local
# constant so the verifier has no runtime dependency on the
# evaluation package — the verifier ships in any environment where
# the traces themselves can be read.
KNOWN_STATUSES: frozenset[str] = frozenset({
    "PASSED",
    "INVALID_REFERENCE",
    "INCORRECT_SHAPE",
    "INCORRECT_NUMERICAL",
    "INCORRECT_DTYPE",
    "RUNTIME_ERROR",
    "COMPILE_ERROR",
    "TIMEOUT",
    "REWARD_HACK",
})


@dataclass(frozen=True)
class Failure:
    """A single reconciliation failure."""

    reason: str
    slug: str
    detail: str

    def __post_init__(self) -> None:
        if self.reason not in TYPED_REASONS:
            raise ValueError(
                f"Failure.reason={self.reason!r} is not in the closed "
                f"typed-reason vocabulary; extend TYPED_REASONS first."
            )


class _ResolveError(Exception):
    """Internal: raised by the slug→workload.jsonl resolver."""

    def __init__(self, reason: str, slug: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {slug}")
        self.reason = reason
        self.slug = slug
        self.detail = detail


def resolve_canonical_workload_jsonl(
    slug: str, repo_root: Optional[Path] = None
) -> Path:
    """Locate the canonical ``workload.jsonl`` for *slug*.

    Resolves slug → category (via ``baselines/<category>/<slug>/``) →
    canonical
    ``sol-execbench/data/benchmark/<subset>/<def>/workload.jsonl``.
    Raises :class:`_ResolveError` with a typed reason on failure.
    """
    if repo_root is None:
        repo_root = _REPO_ROOT
    if not slug.startswith("problem_"):
        raise _ResolveError("slug_missing_problem_prefix", slug)
    baselines = list((repo_root / "baselines").glob(f"*/{slug}"))
    if not baselines:
        raise _ResolveError("baseline_dir_missing_for_slug", slug)
    if len(baselines) > 1:
        raise _ResolveError(
            "ambiguous_baseline_dir_for_slug",
            slug,
            f"matched: {[str(p) for p in baselines]}",
        )
    category = baselines[0].parent.name
    subset = SUBSET_MAP.get(category)
    if subset is None:
        raise _ResolveError("unknown_baseline_category", slug, f"category={category!r}")
    bench_name = slug[len("problem_"):]
    canonical_dir = (
        repo_root / "sol-execbench" / "data" / "benchmark" / subset / bench_name
    )
    if not canonical_dir.is_dir():
        raise _ResolveError(
            "canonical_problem_dir_missing", slug, f"expected at {canonical_dir}"
        )
    workload_jsonl = canonical_dir / "workload.jsonl"
    if not workload_jsonl.is_file():
        raise _ResolveError(
            "canonical_workload_jsonl_missing", slug, f"expected at {workload_jsonl}"
        )
    return workload_jsonl


def load_canonical_uuids(workload_jsonl: Path, slug: str) -> list[str]:
    """Parse the canonical workload.jsonl and return the UUID list in
    file order. UUIDs are unique by construction (sol-execbench's own
    invariant), so this preserves the workload index → UUID mapping.
    """
    return [u for u, _ in _load_canonical_uuid_axes(workload_jsonl, slug)]


def _load_canonical_uuid_axes(
    workload_jsonl: Path, slug: str
) -> list[tuple[str, dict[str, int]]]:
    """Return ``[(uuid, axes_dict), ...]`` in file order."""
    out: list[tuple[str, dict[str, int]]] = []
    text = workload_jsonl.read_text(encoding="utf-8")
    for idx, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _ResolveError(
                "canonical_workload_jsonl_unparseable",
                slug,
                f"line {idx + 1}: {exc}",
            ) from exc
        uuid = row.get("uuid")
        if not uuid or not isinstance(uuid, str):
            raise _ResolveError(
                "canonical_workload_uuid_missing", slug, f"line {idx + 1}"
            )
        axes = row.get("axes") or {}
        if not isinstance(axes, dict):
            axes = {}
        out.append((uuid, axes))
    return out


def _parse_axes_descriptor(text: str) -> dict[str, int]:
    """Parse a free-form workload_axes string into a ``{name: int}`` dict.

    The carve-out manifest emits human-readable axes strings like
    ``"batch_size=2 seq_len=8192 num_heads=32 head_dim=128"``. We extract
    every ``<word>=<int>`` token; anything that doesn't match (decimal,
    quoted, comment) is silently skipped — the descriptor is human
    documentation, not a complete machine spec. The drift check just
    validates that every parsed name/value pair agrees with the
    canonical axes dict.
    """
    import re

    out: dict[str, int] = {}
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+)\b", text):
        out[match.group(1)] = int(match.group(2))
    return out


def _parse_trace_jsonl(path: Path) -> tuple[list[dict], Optional[str]]:
    """Return (rows, unparseable_detail). On parse error the second
    element is non-None and rows holds the rows parsed so far."""
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8")
    for idx, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            return rows, f"line {idx + 1}: {exc}"
    return rows, None


def _verify_slug(
    slug: str,
    trace_dir: Path,
    per_workload_carveouts: dict[tuple[str, str], HostResourceUnavailableEntry],
    repo_root: Optional[Path] = None,
) -> list[Failure]:
    """Verify a single slug's traces against canonical + carve-outs.

    Returns the list of failures (empty when the slug reconciles).
    """
    if repo_root is None:
        repo_root = _REPO_ROOT
    failures: list[Failure] = []

    try:
        workload_jsonl = resolve_canonical_workload_jsonl(slug, repo_root)
        canonical_uuids = load_canonical_uuids(workload_jsonl, slug)
    except _ResolveError as exc:
        return [Failure(exc.reason, exc.slug, exc.detail)]

    canonical_uuid_set: set[str] = set(canonical_uuids)
    canonical_index_by_uuid: dict[str, int] = {
        u: i for i, u in enumerate(canonical_uuids)
    }

    trace_jsonl = trace_dir / f"{slug}.trace.jsonl"
    trace_log = trace_dir / f"{slug}.log"

    if not trace_jsonl.is_file():
        failures.append(Failure("missing_trace_file", slug, str(trace_jsonl)))
        return failures
    if trace_jsonl.stat().st_size == 0:
        failures.append(Failure("empty_trace_file", slug, str(trace_jsonl)))
        return failures
    if not trace_log.is_file():
        failures.append(Failure("missing_trace_log_file", slug, str(trace_log)))
    elif trace_log.stat().st_size == 0:
        failures.append(Failure("empty_trace_log_file", slug, str(trace_log)))

    log_text: str = ""
    if trace_log.is_file() and trace_log.stat().st_size > 0:
        log_text = trace_log.read_text(encoding="utf-8", errors="replace")

    rows, unparseable = _parse_trace_jsonl(trace_jsonl)
    if unparseable is not None:
        failures.append(Failure("unparseable_trace_row", slug, unparseable))

    status_by_uuid: dict[str, str] = {}
    for row_idx, row in enumerate(rows):
        workload = row.get("workload") or {}
        uuid = workload.get("uuid")
        if not uuid or not isinstance(uuid, str):
            failures.append(
                Failure("trace_row_missing_uuid", slug, f"row {row_idx}")
            )
            continue
        evaluation = row.get("evaluation") or {}
        status = evaluation.get("status")
        if not status:
            failures.append(
                Failure(
                    "trace_row_missing_status",
                    slug,
                    f"row {row_idx} uuid={uuid}",
                )
            )
            continue
        if status not in KNOWN_STATUSES:
            failures.append(
                Failure(
                    "unrecognised_status",
                    slug,
                    f"row {row_idx} status={status!r}",
                )
            )
            continue
        if uuid not in canonical_uuid_set:
            failures.append(
                Failure(
                    "trace_uuid_not_in_canonical_workload",
                    slug,
                    f"row {row_idx} uuid={uuid}",
                )
            )
            continue
        if uuid in status_by_uuid:
            failures.append(
                Failure(
                    "duplicate_uuid_in_trace", slug, f"row {row_idx} uuid={uuid}"
                )
            )
            continue
        status_by_uuid[uuid] = status

    passed_uuids = {u for u, s in status_by_uuid.items() if s == "PASSED"}

    for uuid in canonical_uuids:
        if uuid in passed_uuids:
            # A PASSED workload that ALSO has a carve-out entry is a
            # bug in the manifest — the carve-out should be removed.
            if (slug, uuid) in per_workload_carveouts:
                failures.append(
                    Failure(
                        "carveout_uuid_is_passed_in_trace",
                        slug,
                        f"uuid={uuid} has PASSED trace row + carve-out entry",
                    )
                )
            continue
        carve_entry = per_workload_carveouts.get((slug, uuid))
        if carve_entry is not None:
            # Per-workload carve-out: require an actual trace row AND
            # require the trace status / log content match the carve-
            # out's typed_reason. The manifest is documented host-
            # resource evidence, not a kernel-correctness escape hatch.
            row_status = status_by_uuid.get(uuid)
            if row_status is None:
                failures.append(
                    Failure(
                        "carveout_uuid_lacks_trace_row",
                        slug,
                        f"uuid={uuid} typed_reason={carve_entry.typed_reason}",
                    )
                )
                continue
            ev = _check_carveout_evidence(
                slug=slug,
                uuid=uuid,
                workload_index=canonical_index_by_uuid[uuid],
                trace_status=row_status,
                typed_reason=carve_entry.typed_reason,
                log_text=log_text,
            )
            if ev is not None:
                failures.append(ev)
            continue
        failures.append(
            Failure(
                "canonical_uuid_lacks_evidence",
                slug,
                f"uuid={uuid} status_in_trace={status_by_uuid.get(uuid)!r}",
            )
        )

    # Whole-slug carve-outs are rejected up-front in verify() and
    # never reach this function. The two legacy whole-slug typed
    # reasons (whole_slug_carveout_slug_unknown,
    # whole_slug_carveout_without_per_workload_evidence) remain in
    # TYPED_REASONS for back-compat with anyone introspecting the
    # closed set; the verifier no longer emits them.

    return failures


def _check_carveout_evidence(
    *,
    slug: str,
    uuid: str,
    workload_index: int,
    trace_status: str,
    typed_reason: str,
    log_text: str,
) -> Optional[Failure]:
    """Bind a per-workload carve-out's typed_reason to compatible
    evidence in the trace status + log text.

    Returns a Failure when the trace status disagrees with the
    expected status, or when the log lacks the required substrings.
    Returns None when evidence matches (or no binding is registered
    for the typed_reason).
    """
    binding = CARVEOUT_REASON_EVIDENCE.get(typed_reason)
    if binding is None:
        # The manifest loader has already enforced the closed
        # vocabulary; an unmapped reason here means the binding table
        # is intentionally permissive for that reason (currently:
        # nothing — the table covers every documented reason).
        return None
    expected_status, required_substrings = binding
    if trace_status != expected_status:
        return Failure(
            "carveout_reason_evidence_mismatch",
            slug,
            (
                f"uuid={uuid} workload_index={workload_index} "
                f"typed_reason={typed_reason} requires status="
                f"{expected_status!r} but trace status is {trace_status!r}"
            ),
        )
    # Search for required substrings in the log section that mentions
    # this workload index. The sol-execbench CLI emits runtime logs as
    # `Workload <N> (STATUS):` blocks; search after that marker for
    # the next `Workload ` block boundary.
    workload_block = _extract_workload_log_block(log_text, workload_index)
    missing = [s for s in required_substrings if s not in workload_block]
    if missing:
        return Failure(
            "carveout_reason_evidence_mismatch",
            slug,
            (
                f"uuid={uuid} workload_index={workload_index} "
                f"typed_reason={typed_reason} log for this workload "
                f"is missing required substrings: {missing}"
            ),
        )
    return None


def _extract_workload_log_block(log_text: str, workload_index: int) -> str:
    """Return the slice of *log_text* that documents *workload_index*.

    The sol-execbench CLI emits runtime-log blocks as

        Workload <N> (STATUS):
        <error / traceback lines>

    separated by blank lines and bounded by the next ``Workload `` marker
    or end-of-file. We return the block for the requested index; an
    empty string when no block is found. The slice is a search window,
    not a strict parser — duplicate substrings outside the requested
    block do NOT count as evidence.
    """
    needle = f"Workload {workload_index} ("
    start = log_text.find(needle)
    if start < 0:
        return ""
    # Walk forward looking for the next ``Workload <K> (`` marker (K != index)
    cursor = start + len(needle)
    while True:
        next_marker = log_text.find("Workload ", cursor)
        if next_marker < 0:
            return log_text[start:]
        # Confirm this is a new workload block (digit follows ``Workload ``)
        tail = log_text[next_marker + len("Workload "):][:8]
        if tail and tail[0].isdigit():
            return log_text[start:next_marker]
        cursor = next_marker + len("Workload ")


def _validate_manifest_against_canonical(
    per_workload_carveouts: dict[tuple[str, str], HostResourceUnavailableEntry],
    repo_root: Optional[Path] = None,
) -> list[Failure]:
    """Carve-out entries must reference real slugs / real UUIDs."""
    if repo_root is None:
        repo_root = _REPO_ROOT
    failures: list[Failure] = []

    referenced_slugs = {s for s, _ in per_workload_carveouts.keys()}

    canonical_axes_by_slug: dict[str, dict[str, dict[str, int]]] = {}
    for slug in sorted(referenced_slugs):
        try:
            workload_jsonl = resolve_canonical_workload_jsonl(slug, repo_root)
            canonical_axes_by_slug[slug] = dict(
                _load_canonical_uuid_axes(workload_jsonl, slug)
            )
        except _ResolveError as exc:
            failures.append(Failure("manifest_slug_unknown", slug, exc.reason))

    for (slug, uuid), entry in per_workload_carveouts.items():
        canonical_axes = canonical_axes_by_slug.get(slug)
        if canonical_axes is None:
            continue
        if uuid not in canonical_axes:
            failures.append(
                Failure("manifest_uuid_unknown", slug, f"uuid={uuid}")
            )
            continue
        descriptor = entry.workload_axes or ""
        if not descriptor:
            continue
        parsed = _parse_axes_descriptor(descriptor)
        canonical_axes_for_uuid = canonical_axes[uuid]
        drifts: list[str] = []
        for k, v in parsed.items():
            if k in canonical_axes_for_uuid and canonical_axes_for_uuid[k] != v:
                drifts.append(
                    f"{k}: manifest={v}, canonical={canonical_axes_for_uuid[k]}"
                )
        if drifts:
            failures.append(
                Failure(
                    "manifest_axes_drift",
                    slug,
                    f"uuid={uuid} drift={'; '.join(drifts)}",
                )
            )

    return failures


def verify(
    batch_manifest: Path,
    trace_dir: Path,
    host_resource_manifest: Path,
    repo_root: Optional[Path] = None,
) -> list[Failure]:
    """Verify all slugs in *batch_manifest* and return the failure list.

    This is the testable functional surface; the CLI wrapper calls
    this and exits 0 / 2 based on the result.
    """
    # Resolve at call time so test fixtures can monkeypatch the
    # module-level _REPO_ROOT.
    if repo_root is None:
        repo_root = _REPO_ROOT
    failures: list[Failure] = []

    try:
        batch_data = tomllib.loads(batch_manifest.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [Failure("batch_manifest_unparseable", batch_manifest.name, str(exc))]

    slugs_block: Iterable[dict] = batch_data.get("slugs") or []
    slugs: list[str] = [s.get("slug", "") for s in slugs_block if s.get("slug")]
    if not slugs:
        return [Failure("batch_manifest_no_slugs", batch_manifest.name, "")]

    # Use the validated manifest loader so the same closed-schema
    # validation that gates the manifest writer (closed typed_reason
    # vocabulary, no duplicate workload UUIDs, no duplicate slug
    # entries, required fields present, supported schema version)
    # also gates the reconciliation step. Direct tomllib.loads would
    # accept anything that's valid TOML, which is fail-open.
    try:
        host_manifest: HostResourceAvailabilityManifest = (
            load_host_resource_availability_manifest(host_resource_manifest)
        )
    except HostResourceAvailabilityError as exc:
        return [
            Failure(
                "host_resource_manifest_load_failed",
                host_resource_manifest.name,
                f"{exc.typed_reason}: {exc.detail}",
            )
        ]
    except tomllib.TOMLDecodeError as exc:
        return [
            Failure(
                "host_resource_manifest_unparseable",
                host_resource_manifest.name,
                str(exc),
            )
        ]

    # Build per-workload carve-out dict from the validated dataclass
    # entries, not from raw TOML dictionaries. The loader already
    # rejected duplicates and invalid typed_reasons.
    per_workload_carveouts: dict[tuple[str, str], HostResourceUnavailableEntry] = {
        (entry.slug, entry.workload_uuid): entry
        for entry in host_manifest.unavailable_workloads
    }

    # Whole-slug carve-outs are rejected outright: they cannot bind a
    # `typed_reason` to compatible per-workload trace evidence the
    # way per-workload entries can. A whole-slug `correctness_check_oom`
    # carve-out paired with all-`INCORRECT_NUMERICAL` trace rows is
    # logically a kernel-correctness failure, not a host-resource
    # exception. To carve out multiple workloads for a slug, list
    # each one as a ``[[unavailable_workloads]]`` row so the gate
    # can verify status + log evidence per UUID.
    for entry in host_manifest.unavailable_slugs:
        failures.append(
            Failure(
                "whole_slug_carveout_requires_per_workload_entries",
                entry.slug,
                (
                    f"whole-slug carve-out with typed_reason="
                    f"{entry.typed_reason!r} cannot be reconciled; "
                    f"expand into per-workload [[unavailable_workloads]] "
                    f"entries so each UUID's trace status + log evidence "
                    f"can be verified"
                ),
            )
        )

    failures += _validate_manifest_against_canonical(
        per_workload_carveouts, repo_root
    )

    for slug in slugs:
        failures += _verify_slug(
            slug,
            trace_dir,
            per_workload_carveouts,
            repo_root,
        )

    return failures


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconciliation verifier: native-correctness traces × "
            "canonical workload.jsonl × host-resource carve-out manifest."
        )
    )
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument(
        "--host-resource-manifest", type=Path, required=True
    )
    args = parser.parse_args(argv)

    failures = verify(
        args.batch_manifest, args.trace_dir, args.host_resource_manifest
    )
    if failures:
        for f in failures:
            print(f"FAIL [{f.reason}] {f.slug}: {f.detail}", file=sys.stderr)
        print(f"\nTOTAL FAILURES: {len(failures)}", file=sys.stderr)
        return 2
    print(
        f"OK: {args.batch_manifest.name} reconciled against trace dir "
        f"{args.trace_dir} and host-resource manifest "
        f"{args.host_resource_manifest.name}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
