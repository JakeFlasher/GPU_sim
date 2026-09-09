"""Post-hoc wall-clock join: ``nvbit_metrics/launches.jsonl`` × ``cupti_activity/activity.jsonl``.

The ``nvbit_metrics`` surface ships every per-launch JSON-L record with
a ``cross_surface_join_key`` composite tuple but emits NO wall-clock /
timing field — the on-device atomic aggregation perturbs instrumented-
run timing by construction, so wall-clock attribution belongs to a
matched uninstrumented ``cupti_activity`` Surface 2 run instead. This
module joins the two artifacts post-hoc and assigns every launch from
either side to exactly one of five buckets:

* ``matched`` — exactly one NVBit record and exactly one CUPTI record
  share the key.
* ``ambiguous`` — two or more NVBit records share the same key within
  the matched run.
* ``unmatched_nvbit_only`` — NVBit recorded a launch the CUPTI side
  did not see (graph-launch waiver path, or the CUPTI surface coverage
  gap on the same kernel).
* ``unmatched_cupti_only`` — CUPTI recorded a launch the NVBit side
  did not see (filtered by the user-module regex, or skipped due to
  the slot-pool budget cap).
* ``unjoinable`` — a key field is missing from either side, so the
  joiner cannot decide which bucket the launch belongs in.

The join records each side's identity namespace as provenance and retains
this composite diagnostic tuple:
``(workload_index, evaluator_config_hash, kernel_name_mangled,
kernel_ordinal_in_workload, grid_xyz, block_xyz, dynamic_smem,
stream_ordinal, context_id)``. CUPTI Activity normally emits the
**mangled** symbol in its ``name`` slot, so the joiner matches by
mangled name (NVBit's ``kernel_name_mangled`` top-level field) and
records the demangled name on the matched row as auxiliary
information. Numeric stream/context IDs are diagnostics only and are never
compared across processes. This keeps the join deterministic across hosts whose
demangler libraries diverge.

The CUPTI side does not natively emit ``kernel_ordinal_in_workload``.
Its correlation ID is local to the CUPTI process and is not compared
with the NVBit launch ordinal. The joiner retains a local arrival index
for diagnostics only and pairs records by singleton launch signature.

The diagnostic tuple fields are:

* ``workload_index`` (int|null) — None-tolerant: when either side
  hasn't populated it the field is treated as agreeing.
* ``evaluator_config_hash`` (str|null) — None-tolerant identically.
* ``kernel_name_mangled`` (str) — required; matched strictly. NVBit reads
  its raw mangled field; CUPTI reads ``name``/``raw_name`` with demangled
  text as a fallback.
* ``kernel_ordinal_in_workload`` (int) — retained as a local diagnostic.
* ``grid_xyz`` / ``block_xyz`` ([3]int) — required; matched strictly.
* ``dynamic_smem`` (int) — required; matched strictly.
* ``identity_namespace`` (string) — required provenance; it is never used as
  a launch key.
* ``stream_ordinal`` / ``context_id`` (int) — local diagnostics, never joined.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# CUPTI Activity record kinds that correspond to kernel launches. CUDA's
# stable enum values are KERNEL=3 and CONCURRENT_KERNEL=10; runtime API
# records are kind=5 and must never be interpreted as kernels.
_KERNEL_RECORD_KINDS = frozenset({3, 10})


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_xyz_int_triple(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(_is_nonnegative_int(item) for item in value)
    )

def _is_kernel_launch_record(rec: Dict[str, Any]) -> bool:
    """True when ``rec`` is a CUPTI kernel-launch record."""
    return rec.get("kind") in _KERNEL_RECORD_KINDS


# Locked-vocabulary status enum. Both halves of the join machinery
# (joiner + summary emitter) refer to these symbols rather than
# stringly-typed comparisons.
JOIN_STATUS_MATCHED = "matched"
JOIN_STATUS_AMBIGUOUS = "ambiguous"
JOIN_STATUS_UNMATCHED_NVBIT_ONLY = "unmatched_nvbit_only"
JOIN_STATUS_UNMATCHED_CUPTI_ONLY = "unmatched_cupti_only"
JOIN_STATUS_UNJOINABLE = "unjoinable"
JOIN_STATUSES = frozenset({
    JOIN_STATUS_MATCHED,
    JOIN_STATUS_AMBIGUOUS,
    JOIN_STATUS_UNMATCHED_NVBIT_ONLY,
    JOIN_STATUS_UNMATCHED_CUPTI_ONLY,
    JOIN_STATUS_UNJOINABLE,
})


@dataclass
class JoinedRow:
    """One row of the post-hoc cross-surface join.

    Carries the assigned ``join_status``, the join key tuple (when it
    could be computed), and the per-side raw evidence (NVBit dict
    and/or CUPTI dict). A row with ``join_status=unjoinable`` carries
    at least one of the two sides; an ``unmatched_*_only`` row carries
    exactly one side; ``matched`` and ``ambiguous`` rows carry both.
    The duration in ns is set from the CUPTI side when present.
    """

    join_status: str
    join_key: Optional[Tuple[Any, ...]] = None
    kernel_name_mangled: Optional[str] = None
    duration_ns: Optional[int] = None
    nvbit_record: Optional[Dict[str, Any]] = None
    cupti_records: List[Dict[str, Any]] = field(default_factory=list)
    nvbit_extras: List[Dict[str, Any]] = field(default_factory=list)
    waiver_reason: Optional[str] = None


@dataclass
class JoinSummary:
    """Bundle-level summary of the join.

    Per Codex review #5 directive: every launch from either side must
    land in exactly one ``join_status`` bucket; the summary lists the
    counts for each bucket so the operator can audit cross-surface
    coverage from the manifest alone without re-running the join.
    """

    bucket_counts: Dict[str, int]
    nvbit_total: int
    cupti_total: int
    rows: List[JoinedRow]


def _read_jsonl_lines(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON-L file into a list of dicts, tolerating empty lines.

    A malformed or non-object line is an input-integrity failure.  Silently
    dropping it would make the later bucket totals look complete while
    omitting a launch from one surface.
    """
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: invalid UTF-8 JSONL input") from exc
    except OSError as exc:
        raise ValueError(f"{path}: cannot read JSONL input: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number}: malformed JSONL record: {exc.msg}"
            ) from exc
        if not isinstance(obj, dict):
            raise ValueError(
                f"{path}:{line_number}: JSONL record must be an object"
            )
        out.append(obj)
    return out


def nvbit_join_key(rec: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """Return the composite key tuple from an NVBit launch record.

    The tuple's name field uses the raw mangled kernel symbol. The
    demangled spelling remains auxiliary evidence because demangling
    implementations can differ between the two processes.

    Returns ``None`` (signalling ``unjoinable``) when any required
    field is missing OR when the record's launch_kind is
    ``graph_launch_not_observed`` (those records intentionally carry
    no per-node identity that the CUPTI side could match against).
    """
    if rec.get("launch_kind") == "graph_launch_not_observed":
        return None
    cs = rec.get("cross_surface_join_key")
    if not isinstance(cs, dict):
        return None
    if cs.get("identity_namespace") not in {"cuda_driver", "cupti_activity"}:
        return None
    name = rec.get("kernel_name_mangled")
    if not isinstance(name, str) or not name:
        name = cs.get("demangled_kernel_name")
    ordinal = cs.get("kernel_ordinal_in_workload")
    grid = cs.get("grid_xyz")
    block = cs.get("block_xyz")
    smem = cs.get("dynamic_smem")
    stream = cs.get("stream_ordinal")
    context = cs.get("context_id")
    if (
        not isinstance(name, str)
        or not name
        or not _is_nonnegative_int(ordinal)
        or not _is_nonnegative_int(smem)
    ):
        return None
    if not _is_xyz_int_triple(grid) or not _is_xyz_int_triple(block):
        return None
    return (
        cs.get("workload_index"),
        cs.get("evaluator_config_hash"),
        name,
        int(ordinal),
        tuple(grid),
        tuple(block),
        int(smem),
        int(stream) if _is_nonnegative_int(stream) else -1,
        int(context) if _is_nonnegative_int(context) else -1,
    )


def _join_spine(key: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Stable launch signature shared across separate processes."""
    return (key[2], key[4], key[5], key[6])


def _cupti_kernel_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter CUPTI activity rows to kernel-launch records.

    The local arrival index is retained for diagnostics only; it is not
    used as a cross-process identity.
    """
    out: List[Dict[str, Any]] = []
    ordinal = 0
    for rec in rows:
        if not _is_kernel_launch_record(rec):
            continue
        annotated = dict(rec)
        annotated["_synthetic_kernel_ordinal"] = ordinal
        out.append(annotated)
        ordinal += 1
    return out


def _cupti_full_join_key(
    rec: Dict[str, Any],
) -> Tuple[Optional[Tuple[Any, ...]], Optional[str]]:
    """Build the composite tuple from a CUPTI activity record.

    Returns ``(tuple, None)`` when every required field resolves;
    ``(None, "<missing-field>")`` when any required field is absent,
    so the joiner can classify the record as ``unjoinable`` with a
    typed waiver. ``workload_index`` and ``evaluator_config_hash`` are
    optional (the CUPTI sink does not natively expose them); they
    enter the tuple as ``None`` so the matcher can compare them
    against the NVBit-side values directly.

    The required fields are: raw kernel name,
    ``kernel_ordinal_in_workload`` (local diagnostic index), ``grid_xyz``,
    ``block_xyz``, ``dynamic_smem``, ``stream_ordinal``, ``context_id``.
    """
    # Prefer the raw mangled CUPTI name; demangled text is only a fallback
    # for activity bindings that do not expose the raw symbol.
    name = rec.get("name") or rec.get("raw_name")
    if not isinstance(name, str) or not name:
        name = rec.get("demangled_kernel_name")
    if not isinstance(name, str) or not name:
        return None, "missing kernel name"
    if rec.get("identity_namespace") != "cupti_activity":
        return None, "incompatible CUPTI identity namespace"
    ordinal = rec.get("_synthetic_kernel_ordinal")
    if not _is_nonnegative_int(ordinal):
        return None, "missing synthetic kernel ordinal"
    grid = rec.get("grid_xyz")
    if not _is_xyz_int_triple(grid):
        return None, "missing grid_xyz"
    block = rec.get("block_xyz")
    if not _is_xyz_int_triple(block):
        return None, "missing block_xyz"
    smem = rec.get("dynamic_smem")
    if not _is_nonnegative_int(smem):
        return None, "missing dynamic_smem"
    # Stream/context IDs are retained as diagnostics only. They are local to
    # each process and are deliberately excluded from the launch signature.
    stream_id = rec.get("stream_id")
    if not _is_nonnegative_int(stream_id):
        stream_id = -1
    context_id = rec.get("context_id")
    if not _is_nonnegative_int(context_id):
        context_id = -1
    return (
        rec.get("workload_index"),  # None until harness injects
        rec.get("evaluator_config_hash"),  # None until harness injects
        name,
        int(ordinal),
        tuple(grid),
        tuple(block),
        int(smem),
        int(stream_id),
        int(context_id),
    ), None


def _full_keys_match(
    nvbit_key: Tuple[Any, ...],
    cupti_key: Tuple[Any, ...],
) -> Optional[str]:
    """Return ``None`` when the two tuples represent the same launch,
    otherwise a string naming the first disagreement.

    The matching is field-by-field with a None-tolerance rule for the
    two optional fields (``workload_index``,
    ``evaluator_config_hash``): when EITHER side has ``None`` the
    fields are treated as agreeing (we can't disprove the match from
    no-info); when BOTH sides have concrete values they must equal.
    Name, grid, block, and dynamic shared memory are strict. Launch
    ordinal and numeric stream/context IDs are process-local diagnostics
    and are excluded from this comparison.
    """
    # Optional fields — None-tolerant.
    for idx, name in ((0, "workload_index"), (1, "evaluator_config_hash")):
        n_val = nvbit_key[idx]
        c_val = cupti_key[idx]
        if n_val is None or c_val is None:
            continue
        if n_val != c_val:
            return f"{name} nvbit={n_val!r} cupti={c_val!r}"
    # Stable launch signature — strict equality.
    if nvbit_key[2] != cupti_key[2]:
        return (
            f"kernel_name_mangled nvbit={nvbit_key[2]!r} "
            f"cupti={cupti_key[2]!r}"
        )
    if tuple(nvbit_key[4]) != tuple(cupti_key[4]):
        return (
            f"grid_xyz nvbit={list(nvbit_key[4])} "
            f"cupti={list(cupti_key[4])}"
        )
    if tuple(nvbit_key[5]) != tuple(cupti_key[5]):
        return (
            f"block_xyz nvbit={list(nvbit_key[5])} "
            f"cupti={list(cupti_key[5])}"
        )
    if int(nvbit_key[6]) != int(cupti_key[6]):
        return (
            f"dynamic_smem nvbit={nvbit_key[6]} cupti={cupti_key[6]}"
        )
    return None


def join_launches_against_cupti(
    nvbit_records: List[Dict[str, Any]],
    cupti_records: List[Dict[str, Any]],
) -> JoinSummary:
    """Perform the post-hoc wall-clock join using the full
    diagnostic tuple.

    Returns a ``JoinSummary`` whose ``rows`` partition every NVBit
    and CUPTI launch into exactly one bucket. The function is
    side-effect-free and tolerant of partial inputs.

    The matching key is the launch signature
    ``(kernel_name_mangled, grid_xyz, block_xyz, dynamic_smem)``.
    The remaining tuple fields are retained for diagnostics:

    * NVBit side: ``kernel_name_mangled`` + the remaining tuple slots from the
      JSON-L record's
      ``cross_surface_join_key`` object.
    * CUPTI side: raw ``name``/``raw_name`` + launch shape fields / local IDs from the
      sink's shape-field extensions.

    Missing required fields on either side land in ``unjoinable`` with
    a typed waiver naming the missing field. A repeated signature is
    ambiguous; it is never paired by arrival order.
    """
    kernel_cupti = _cupti_kernel_records(cupti_records)

    # Build an index of NVBit records keyed by the stable launch signature.
    nvbit_index: Dict[
        Tuple[Any, ...],
        List[Tuple[Dict[str, Any], Tuple[Any, ...]]],
    ] = {}
    nvbit_unjoinable: List[Dict[str, Any]] = []
    nvbit_graph_not_observed: List[Dict[str, Any]] = []
    for rec in nvbit_records:
        if rec.get("launch_kind") == "graph_launch_not_observed":
            nvbit_graph_not_observed.append(rec)
            continue
        key = nvbit_join_key(rec)
        if key is None:
            nvbit_unjoinable.append(rec)
            continue
        # Spine = stable launch signature; process-local ordinals and IDs
        # are deliberately excluded.
        spine = _join_spine(key)
        nvbit_index.setdefault(spine, []).append((rec, key))

    cupti_index: Dict[
        Tuple[Any, ...],
        List[Tuple[Dict[str, Any], Tuple[Any, ...]]],
    ] = {}
    cupti_unjoinable: List[Tuple[Dict[str, Any], str]] = []
    for rec in kernel_cupti:
        cupti_key, reason = _cupti_full_join_key(rec)
        if cupti_key is None:
            cupti_unjoinable.append((rec, reason or "missing key field"))
            continue
        spine = _join_spine(cupti_key)
        cupti_index.setdefault(spine, []).append((rec, cupti_key))

    rows: List[JoinedRow] = []
    bucket_counts: Dict[str, int] = {s: 0 for s in JOIN_STATUSES}

    # Emit one row per NVBit spine; classify against the CUPTI index.
    for spine, nvbit_for_spine in nvbit_index.items():
        cupti_for_spine = cupti_index.pop(spine, [])
        first_nvbit, full_key = nvbit_for_spine[0]
        nvbit_extras = [r for r, _ in nvbit_for_spine[1:]]
        cupti_rec_list = [r for r, _ in cupti_for_spine]
        if len(nvbit_for_spine) > 1 or len(cupti_for_spine) > 1:
            rows.append(JoinedRow(
                join_status=JOIN_STATUS_AMBIGUOUS,
                join_key=full_key,
                kernel_name_mangled=first_nvbit.get("kernel_name_mangled"),
                nvbit_record=first_nvbit,
                nvbit_extras=nvbit_extras,
                cupti_records=cupti_rec_list,
                waiver_reason=(
                    f"launch signature {spine!r} is repeated "
                    f"(nvbit={len(nvbit_for_spine)}, "
                    f"cupti={len(cupti_for_spine)}); pairing is ambiguous"
                ),
            ))
            bucket_counts[JOIN_STATUS_AMBIGUOUS] += 1
        elif not cupti_for_spine:
            rows.append(JoinedRow(
                join_status=JOIN_STATUS_UNMATCHED_NVBIT_ONLY,
                join_key=full_key,
                kernel_name_mangled=first_nvbit.get("kernel_name_mangled"),
                nvbit_record=first_nvbit,
                waiver_reason=(
                    "NVBit launch has no CUPTI counterpart "
                    "(graph-launch waiver path or surface coverage gap)"
                ),
            ))
            bucket_counts[JOIN_STATUS_UNMATCHED_NVBIT_ONLY] += 1
        else:
            cupti_first_rec, cupti_first_key = cupti_for_spine[0]
            mismatch = _full_keys_match(full_key, cupti_first_key)
            if mismatch is not None:
                rows.append(JoinedRow(
                    join_status=JOIN_STATUS_UNJOINABLE,
                    join_key=full_key,
                    kernel_name_mangled=first_nvbit.get(
                        "kernel_name_mangled"
                    ),
                    nvbit_record=first_nvbit,
                    cupti_records=cupti_rec_list,
                    waiver_reason=(
                        f"composite-tuple mismatch between NVBit "
                        f"and CUPTI: {mismatch}"
                    ),
                ))
                bucket_counts[JOIN_STATUS_UNJOINABLE] += 1
                continue
            duration_ns = _duration_from_cupti(cupti_first_rec)
            rows.append(JoinedRow(
                join_status=JOIN_STATUS_MATCHED,
                join_key=full_key,
                kernel_name_mangled=first_nvbit.get("kernel_name_mangled"),
                duration_ns=duration_ns,
                nvbit_record=first_nvbit,
                cupti_records=cupti_rec_list,
            ))
            bucket_counts[JOIN_STATUS_MATCHED] += 1

    # Whatever's left in cupti_index is unmatched on the NVBit side.
    for spine, cupti_for_spine in cupti_index.items():
        cupti_rec_list = [r for r, _ in cupti_for_spine]
        _, cupti_first_key = cupti_for_spine[0]
        rows.append(JoinedRow(
            join_status=JOIN_STATUS_UNMATCHED_CUPTI_ONLY,
            join_key=cupti_first_key,
            kernel_name_mangled=spine[0],
            cupti_records=cupti_rec_list,
            duration_ns=_duration_from_cupti(cupti_rec_list[0]),
            waiver_reason=(
                "CUPTI launch has no NVBit counterpart "
                "(filtered by user-module regex or skipped due to "
                "slot budget)"
            ),
        ))
        bucket_counts[JOIN_STATUS_UNMATCHED_CUPTI_ONLY] += 1

    # Unjoinable NVBit (missing key) + unjoinable CUPTI (missing
    # required composite-tuple field). graph_launch_not_observed
    # records also count as unjoinable on the NVBit side — they
    # exist for graph-launch manifest substatus, not for joining
    # with CUPTI.
    for rec in nvbit_unjoinable:
        rows.append(JoinedRow(
            join_status=JOIN_STATUS_UNJOINABLE,
            nvbit_record=rec,
            waiver_reason=(
                "NVBit record missing required composite-tuple "
                "field(s) in cross_surface_join_key"
            ),
        ))
        bucket_counts[JOIN_STATUS_UNJOINABLE] += 1
    for rec in nvbit_graph_not_observed:
        rows.append(JoinedRow(
            join_status=JOIN_STATUS_UNJOINABLE,
            nvbit_record=rec,
            waiver_reason="graph_launch_not_observed record has no CUPTI counterpart by construction",
        ))
        bucket_counts[JOIN_STATUS_UNJOINABLE] += 1
    for rec, missing in cupti_unjoinable:
        rows.append(JoinedRow(
            join_status=JOIN_STATUS_UNJOINABLE,
            cupti_records=[rec],
            waiver_reason=(
                f"CUPTI record missing required composite-tuple "
                f"field: {missing}"
            ),
        ))
        bucket_counts[JOIN_STATUS_UNJOINABLE] += 1

    return JoinSummary(
        bucket_counts=bucket_counts,
        nvbit_total=len(nvbit_records),
        cupti_total=len(kernel_cupti),
        rows=rows,
    )


def _duration_from_cupti(rec: Dict[str, Any]) -> Optional[int]:
    """Return ``end_ns - start_ns`` from a CUPTI record, or ``None``."""
    start = rec.get("start_ns")
    end = rec.get("end_ns")
    if not _is_nonnegative_int(start) or not _is_nonnegative_int(end):
        return None
    if end < start:
        return None
    return end - start


def join_bundle(bundle_dir: Path) -> Optional[JoinSummary]:
    """Convenience entrypoint: read the bundle's NVBit + CUPTI JSON-L
    artifacts and return the join summary.

    Returns ``None`` when the NVBit ``launches.jsonl`` is absent — the
    join is meaningful only when the NVBit side ran.
    """
    nvbit_path = bundle_dir / "nvbit_metrics" / "launches.jsonl"
    cupti_path = bundle_dir / "cupti_activity" / "activity.jsonl"
    if not nvbit_path.is_file():
        # Fall back to the legacy ``cupti/`` directory name the older
        # surface plugin uses. The two layouts coexist during the
        # transition window.
        legacy_cupti = bundle_dir / "cupti" / "activity.jsonl"
        if not legacy_cupti.is_file():
            return None
        cupti_path = legacy_cupti
    cupti_records: List[Dict[str, Any]] = []
    if cupti_path.is_file():
        cupti_records = _read_jsonl_lines(cupti_path)
    else:
        legacy_cupti = bundle_dir / "cupti" / "activity.jsonl"
        if legacy_cupti.is_file():
            cupti_records = _read_jsonl_lines(legacy_cupti)
    if not nvbit_path.is_file():
        return None
    nvbit_records = _read_jsonl_lines(nvbit_path)
    return join_launches_against_cupti(nvbit_records, cupti_records)
