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

The join key is the eight-field composite tuple:
``(workload_index, evaluator_config_hash, kernel_name_mangled,
kernel_ordinal_in_workload, grid_xyz, block_xyz, dynamic_smem,
stream_ordinal)``. The schema doc uses ``demangled_kernel_name`` as
the canonical name field; in practice CUPTI Activity emits the
**mangled** symbol in its ``name`` slot, so the joiner matches by
mangled name (NVBit's ``kernel_name_mangled`` top-level field) and
records the demangled name on the matched row as auxiliary
information. This keeps the join deterministic across hosts whose
demangler libraries diverge.

The CUPTI side does not natively emit ``kernel_ordinal_in_workload``
(it has ``correlation_id``, which is a per-CUDA-driver-call counter
distinct from the workload-local launch ordinal). The joiner
synthesises an ordinal by walking the ``activity.jsonl`` in arrival
order and assigning 0..N to kernel-launch records (CUPTI activity
kind ``CONCURRENT_KERNEL`` = 10, with kind=5 + ``cbid == None`` as
a fallback). The NVBit producer's ``kernel_ordinal_in_workload``
counts every observed kernel-launch CBID (including filter-skipped
framework kernels), so the synthetic CUPTI ordinal and the NVBit
ordinal share the same counting domain and align by construction.

The eight-field tuple this joiner indexes on:

* ``workload_index`` (int|null) — None-tolerant: when either side
  hasn't populated it the field is treated as agreeing.
* ``evaluator_config_hash`` (str|null) — None-tolerant identically.
* ``demangled_kernel_name`` (str) — required; matched strictly.
  NVBit reads ``cross_surface_join_key.demangled_kernel_name``;
  CUPTI reads the sink's ``demangled_kernel_name`` field with
  ``raw_name`` / ``name`` as fallbacks.
* ``kernel_ordinal_in_workload`` (int) — required; matched strictly.
* ``grid_xyz`` / ``block_xyz`` ([3]int) — required; matched strictly.
* ``dynamic_smem`` (int) — required; matched strictly.
* ``stream_ordinal`` (int) — required; matched strictly. The NVBit
  producer derives this from ``cuStreamGetId()``, the same CUDA
  stream identity CUPTI Activity's ``streamId`` field carries.
  Both surfaces therefore share the stream-ID namespace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# CUPTI Activity record kinds that correspond to kernel launches the
# matched join cares about. CONCURRENT_KERNEL (kind=10) is what
# sol-execbench's subscriber emits for every kernel; KERNEL (kind=5) is
# a fallback emitted by older CUPTI versions. Other kinds (memcpy=1,
# memset=4, runtime=5/cbid present) are unrelated to kernel launches
# and are filtered out of the join.
_KERNEL_RECORD_KINDS = frozenset({10})

# CUPTI records carry ``cbid``-bearing API records under kind=5 too.
# The disambiguator is the ``cbid`` field: kernel-launch records
# carry ``cbid=null``, while CUDA API records carry an integer cbid
# (e.g. ``cuModuleLoadFatBinary`` = 17, ``cuLaunchKernel`` = 392).
# Kind=5 records with ``cbid == null`` are the kernel half.
def _is_kernel_launch_record(rec: Dict[str, Any]) -> bool:
    """True when ``rec`` is a CUPTI kernel-launch record."""
    kind = rec.get("kind")
    if kind in _KERNEL_RECORD_KINDS:
        return True
    # kind == 5 with cbid == None covers older CUPTI versions that
    # filed kernel records under KERNEL (5) instead of CONCURRENT_KERNEL
    # (10). The cbid disambiguates against CUDA API records.
    if kind == 5 and rec.get("cbid") is None and rec.get("name"):
        return True
    return False


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
    """Read a JSON-L file into a list of dicts, tolerating empty lines."""
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _nvbit_join_key(rec: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """Return the composite key tuple from an NVBit launch record.

    The tuple's name field uses
    ``cross_surface_join_key.demangled_kernel_name`` as the
    canonical name field. NVBit's top-level ``kernel_name_mangled``
    is auxiliary evidence only — the join key is on the demangled
    name so it matches CUPTI's ``demangled_kernel_name`` field the
    sink emits.

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
    demangled = cs.get("demangled_kernel_name")
    ordinal = cs.get("kernel_ordinal_in_workload")
    grid = cs.get("grid_xyz")
    block = cs.get("block_xyz")
    smem = cs.get("dynamic_smem")
    stream = cs.get("stream_ordinal")
    if (
        not isinstance(demangled, str)
        or not demangled
        or ordinal is None
        or smem is None
        or stream is None
    ):
        return None
    if not isinstance(grid, list) or not isinstance(block, list):
        return None
    return (
        cs.get("workload_index"),
        cs.get("evaluator_config_hash"),
        demangled,
        int(ordinal),
        tuple(grid),
        tuple(block),
        int(smem),
        int(stream),
    )


def _cupti_kernel_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter CUPTI activity rows to kernel-launch records, in arrival
    order. Adds a synthetic ``kernel_ordinal_in_workload`` field so the
    joiner can build a comparable composite key without modifying
    cupti_activity surface emission.
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

    The required fields are: ``demangled_kernel_name``,
    ``kernel_ordinal_in_workload`` (synthesized via
    ``_synthetic_kernel_ordinal``), ``grid_xyz``, ``block_xyz``,
    ``dynamic_smem``, ``stream_ordinal``.
    """
    demangled = rec.get("demangled_kernel_name")
    if not isinstance(demangled, str) or not demangled:
        # Fall back to the raw mangled name if the sink-side demangle
        # failed; without a name there is nothing to match against.
        raw = rec.get("name") or rec.get("raw_name")
        if not isinstance(raw, str) or not raw:
            return None, "missing kernel name"
        demangled = raw
    ordinal = rec.get("_synthetic_kernel_ordinal")
    if not isinstance(ordinal, int):
        return None, "missing synthetic kernel ordinal"
    grid = rec.get("grid_xyz")
    if grid is None:
        return None, "missing grid_xyz"
    block = rec.get("block_xyz")
    if block is None:
        return None, "missing block_xyz"
    smem = rec.get("dynamic_smem")
    if smem is None:
        return None, "missing dynamic_smem"
    # The NVBit producer emits the raw CUDA stream ID (from
    # ``cuStreamGetId()``); CUPTI's ``stream_id`` field carries the
    # same identity. We compare them directly without an
    # intermediate normalizer — both surfaces share the CUDA
    # stream-ID namespace by construction.
    stream_id = rec.get("stream_id")
    if stream_id is None:
        return None, "missing stream_ordinal"
    return (
        rec.get("workload_index"),  # None until harness injects
        rec.get("evaluator_config_hash"),  # None until harness injects
        demangled,
        int(ordinal),
        tuple(grid),
        tuple(block),
        int(smem),
        int(stream_id),
    ), None


def _full_keys_match(
    nvbit_key: Tuple[Any, ...],
    cupti_key: Tuple[Any, ...],
) -> Optional[str]:
    """Return ``None`` when the two 8-tuples represent the same launch,
    otherwise a string naming the first disagreement.

    The matching is field-by-field with a None-tolerance rule for the
    two optional fields (``workload_index``,
    ``evaluator_config_hash``): when EITHER side has ``None`` the
    fields are treated as agreeing (we can't disprove the match from
    no-info); when BOTH sides have concrete values they must equal.
    The remaining six fields are strict: the eight-field tuple is
    the composite key, so they must match exactly.
    """
    # Optional fields — None-tolerant.
    for idx, name in ((0, "workload_index"), (1, "evaluator_config_hash")):
        n_val = nvbit_key[idx]
        c_val = cupti_key[idx]
        if n_val is None or c_val is None:
            continue
        if n_val != c_val:
            return f"{name} nvbit={n_val!r} cupti={c_val!r}"
    # Required fields — strict equality.
    if nvbit_key[2] != cupti_key[2]:
        return (
            f"demangled_kernel_name nvbit={nvbit_key[2]!r} "
            f"cupti={cupti_key[2]!r}"
        )
    if nvbit_key[3] != cupti_key[3]:
        return (
            f"kernel_ordinal_in_workload nvbit={nvbit_key[3]} "
            f"cupti={cupti_key[3]}"
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
    if int(nvbit_key[7]) != int(cupti_key[7]):
        return (
            f"stream_ordinal nvbit={nvbit_key[7]} cupti={cupti_key[7]}"
        )
    return None


def join_launches_against_cupti(
    nvbit_records: List[Dict[str, Any]],
    cupti_records: List[Dict[str, Any]],
) -> JoinSummary:
    """Perform the post-hoc wall-clock join using the full
    eight-field tuple.

    Returns a ``JoinSummary`` whose ``rows`` partition every NVBit
    and CUPTI launch into exactly one bucket. The function is
    side-effect-free and tolerant of partial inputs.

    The matching key is the FULL composite tuple
    ``(workload_index, evaluator_config_hash, demangled_kernel_name,
    kernel_ordinal_in_workload, grid_xyz, block_xyz, dynamic_smem,
    stream_ordinal)``. Both halves of the tuple come from comparable
    fields:

    * NVBit side: ``cross_surface_join_key.demangled_kernel_name`` +
      the seven other tuple slots from the JSON-L record's
      ``cross_surface_join_key`` object.
    * CUPTI side: ``demangled_kernel_name`` from the sol-execbench
      sink's demangle output + synthesized global ordinal from
      arrival order + ``grid_xyz`` / ``block_xyz`` /
      ``dynamic_smem`` / normalized ``stream_id`` from the sink's
      shape-field extensions.

    Missing required fields on either side land in ``unjoinable`` with
    a typed waiver naming the missing field. A spine match
    (``demangled_kernel_name`` + ``kernel_ordinal_in_workload``) that
    fails the full-tuple check lands in ``unjoinable`` rather than
    ``matched``.
    """
    kernel_cupti = _cupti_kernel_records(cupti_records)

    # Build an index of NVBit records keyed by the matching spine
    # (demangled name, synthetic ordinal). Multiple NVBit records with
    # the same spine count toward ``ambiguous``. The full eight-field
    # tuple lives alongside each entry so the post-spine match check
    # can use it without re-deriving.
    nvbit_index: Dict[
        Tuple[str, int],
        List[Tuple[Dict[str, Any], Tuple[Any, ...]]],
    ] = {}
    nvbit_unjoinable: List[Dict[str, Any]] = []
    nvbit_graph_not_observed: List[Dict[str, Any]] = []
    for rec in nvbit_records:
        if rec.get("launch_kind") == "graph_launch_not_observed":
            nvbit_graph_not_observed.append(rec)
            continue
        key = _nvbit_join_key(rec)
        if key is None:
            nvbit_unjoinable.append(rec)
            continue
        # Spine = (demangled_kernel_name, ordinal). The remaining six
        # tuple fields are verified after the spine lookup.
        spine: Tuple[str, int] = (key[2], int(key[3]))
        nvbit_index.setdefault(spine, []).append((rec, key))

    cupti_index: Dict[
        Tuple[str, int],
        List[Tuple[Dict[str, Any], Tuple[Any, ...]]],
    ] = {}
    cupti_unjoinable: List[Tuple[Dict[str, Any], str]] = []
    for rec in kernel_cupti:
        cupti_key, reason = _cupti_full_join_key(rec)
        if cupti_key is None:
            cupti_unjoinable.append((rec, reason or "missing key field"))
            continue
        spine = (cupti_key[2], int(cupti_key[3]))
        cupti_index.setdefault(spine, []).append((rec, cupti_key))

    rows: List[JoinedRow] = []
    bucket_counts: Dict[str, int] = {s: 0 for s in JOIN_STATUSES}

    # Emit one row per NVBit spine; classify against the CUPTI index.
    for spine, nvbit_for_spine in nvbit_index.items():
        cupti_for_spine = cupti_index.pop(spine, [])
        first_nvbit, full_key = nvbit_for_spine[0]
        nvbit_extras = [r for r, _ in nvbit_for_spine[1:]]
        cupti_rec_list = [r for r, _ in cupti_for_spine]
        if len(nvbit_for_spine) > 1:
            rows.append(JoinedRow(
                join_status=JOIN_STATUS_AMBIGUOUS,
                join_key=full_key,
                kernel_name_mangled=first_nvbit.get("kernel_name_mangled"),
                nvbit_record=first_nvbit,
                nvbit_extras=nvbit_extras,
                cupti_records=cupti_rec_list,
                waiver_reason=(
                    f"{len(nvbit_for_spine)} NVBit launches share the "
                    f"join spine {spine!r} within one matched run"
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
    if not isinstance(start, int) or not isinstance(end, int):
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
