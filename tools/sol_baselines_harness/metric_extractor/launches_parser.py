"""Strict reference parser for the metric extractor's per-launch JSON-L.

The on-disk format is one JSON object per line, terminated by a single
``\\n`` byte. Every line MUST carry both ``schema_version`` (integer)
and ``tool_version`` (string); the parser rejects any line missing
either. Mid-line truncation (a line that does not end in ``}\\n``) is
also rejected so a partially-written final record cannot silently
corrupt the consumer's view.

Schema: ``nvbit_tracer/metric_extractor/JSONL_SCHEMA.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


@dataclass
class LaunchRecord:
    """One parsed JSON-L record. ``raw`` keeps the original mapping so
    consumers can read forward-compatible fields the dataclass does not
    enumerate yet (the schema is intentionally extensible)."""

    schema_version: int
    tool_version: str
    launch_kind: str
    launch_index: int
    ctx_id: int
    stream_id: int
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseError:
    """Description of a single malformed line. Bundle-level integrity
    check should treat the presence of any error as ``surface: failed``
    per the per-surface manifest contract — the JSON-L is NEVER reported
    as ``ok`` with partial records."""

    line_number: int
    reason: str
    raw: str


_REQUIRED_TOP_FIELDS = (
    "schema_version",
    "tool_version",
    "launch_kind",
    "launch_index",
    "ctx_id",
    "stream_id",
    "grid_dim",
    "block_dim",
    "cluster_dim",
    "dynamic_smem",
    "instruction_class_histogram",
    "tma_descriptor",
    "sector_efficiency_global_load",
    "sector_efficiency_global_store",
    "bank_conflicts",
    "branch_active_lanes_hist",
    "confidence",
    "static_sidecar",
    "cross_surface_join_key",
    "substatus",
    "filter_stats_snapshot",
    "host",
)

# Forbidden top-level field names. The metric extractor's on-device
# atomic aggregation perturbs instrumented-run timing by construction,
# so wall-clock attribution MUST come from the matched ``cupti_activity``
# Surface 2 run, never from the NVBit-injected eval. Any timing field
# in the JSON-L is a producer-side regression that breaks the
# wall-clock-attribution negative-test contract — the parser rejects
# records carrying any of
# these field names with a typed ``ParseError`` so a malformed record
# can never reach the surface manifest as ``status: ok``.
_FORBIDDEN_TOP_FIELDS = (
    "duration_us",
    "start_ns",
    "end_ns",
)
_FORBIDDEN_TOP_FIELD_PREFIXES = ("wall_clock",)

# The branch active-mask histogram is a fixed-length array of 33 buckets,
# one bucket per active-lane count in [0, 32] inclusive. The producer
# emits exactly this length on every accepted launch; a record claiming
# a different length is malformed by construction (a producer-side
# regression that breaks the bucket-indexing contract).
_BRANCH_ACTIVE_LANES_HIST_LEN = 33

# Sub-field invariants documented in JSONL_SCHEMA.md. Each entry maps
# a top-level field name to the (sub_field, py_type) pairs the parser
# enforces. None of these may be silently dropped between schema
# revisions — every new sub-field added in the producer must land
# here in the same commit, paired with a schema-doc update, so a
# downstream consumer cannot silently see a partial record as
# well-formed.
_REQUIRED_CROSS_SURFACE_JOIN_KEY_INT_OR_NULL = (
    "workload_index",
)
_REQUIRED_CROSS_SURFACE_JOIN_KEY_STR_OR_NULL = (
    "evaluator_config_hash",
    "demangled_kernel_name",
)
_REQUIRED_CROSS_SURFACE_JOIN_KEY_INT = (
    "kernel_ordinal_in_workload",
    "dynamic_smem",
    "stream_ordinal",
)
_REQUIRED_CROSS_SURFACE_JOIN_KEY_XYZ = (
    "grid_xyz",
    "block_xyz",
)

_REQUIRED_SUBSTATUS_BOOL = (
    "truncated",
    "skipped",
    "is_graph_node",
    "is_cooperative",
)
_REQUIRED_SUBSTATUS_INT = (
    "saturated_counters",
)
_REQUIRED_SUBSTATUS_STR_OR_NULL = (
    "waiver_reason",
)

_REQUIRED_FILTER_STATS_SNAPSHOT_INT = (
    "accepted_launches",
    "skipped_launches",
    "module_origin_miss_count",
    "cooperative_launches",
    "graph_node_launches",
    "graph_node_accepted",
    "graph_node_skipped",
    "graph_launches_seen",
    "graph_launches_not_observed",
    "per_process_budget_skipped",
    "unknown_opcode_count",
)

_REQUIRED_HOST_STR = (
    "compute_capability",
    "nvbit_tool_version",
    "arch_label",
)
_REQUIRED_HOST_INT = (
    "cuda_runtime_version",
)

# The producer's static_sidecar slot. Currently always emitted as
# ``null`` (the harness-side static-surface join lands separately —
# see T16); future revisions may populate this with
# ``registers_per_thread`` / ``dynamic_smem_per_block`` /
# ``theoretical_occupancy_warps`` / ``local_memory_spill_bytes``
# sub-fields. When the slot is non-null, the parser still tolerates
# any of those four sub-fields being ``null`` (so the harness can
# fill them in incrementally) but the keys must exist if the slot
# is present.
_OPTIONAL_STATIC_SIDECAR_SUBFIELDS = (
    "registers_per_thread",
    "dynamic_smem_per_block",
    "theoretical_occupancy_warps",
    "local_memory_spill_bytes",
)

# Sentinel returned when graph_launch_not_observed records intentionally
# carry ``[0,0,0]`` / ``0`` instead of real launch dims. The parser
# accepts the sentinel; the strict shape (list of three non-negative
# integers) still applies.
_ZERO_XYZ = (0, 0, 0)

# Sub-fields required when `tma_descriptor` is non-null. Mirrors the
# JSONL_SCHEMA.md row introduced when the on-device TMA descriptor sample
# + host-side decode landed: the launch-exit emitter populates every
# member of this set whenever a TMA instruction fired during the launch,
# so a record claiming a non-null descriptor without one of these keys is
# malformed by construction.
_REQUIRED_TMA_DESCRIPTOR_SUBFIELDS = (
    "sample_count",
    "sample_opcode",
    "sample_handle_size",
    "is_tensor",
    "is_bulk",
    "is_prefetch",
    "is_multicast",
    "src_memspace",
    "dst_memspace",
    "transfer_count",
    "transfer_size",
    "byte_count",
    "multicast_cta_mask",
)


def _looks_truncated(raw_bytes: bytes) -> bool:
    """Return True when the line's byte sequence does not end in the
    expected ``}\\n`` terminator. A POSIX writer that crashed mid-line
    typically leaves a partial JSON object without the closing brace
    and the trailing newline; either symptom counts as truncation.
    """
    if not raw_bytes.endswith(b"\n"):
        return True
    body = raw_bytes[:-1]
    # Strip CR for tolerant CRLF handling, but only the trailing one.
    if body.endswith(b"\r"):
        body = body[:-1]
    body = body.rstrip()
    return not body.endswith(b"}")


def _is_strict_int(value: Any) -> bool:
    """``isinstance(True, int)`` is ``True`` in Python (bool is a subclass
    of int), but a JSON-L record using ``true`` / ``false`` for a
    numeric field is malformed by the schema. Reject bool explicitly.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _is_xyz_int_triple(value: Any) -> bool:
    """True when ``value`` is a list of exactly three non-negative
    non-bool integers — the canonical ``grid_dim``/``block_dim``/
    ``cluster_dim``/``grid_xyz``/``block_xyz`` shape.
    """
    if not isinstance(value, list) or len(value) != 3:
        return False
    return all(_is_strict_int(v) and v >= 0 for v in value)


def _validate_record(rec: Dict[str, Any]) -> Optional[str]:
    """Return ``None`` when the record satisfies every mandatory field,
    or a human-readable failure reason otherwise. The validator must
    catch every type mismatch BEFORE ``parse_launches_jsonl`` blindly
    casts to int / str, so the parser never raises an uncaught
    ``ValueError``/``TypeError`` on adversarial input.
    """
    # Wall-clock-attribution negative-test invariant: the producer
    # never emits wall-clock / timing fields because the on-device
    # atomic aggregation perturbs instrumented-run timing by
    # construction. Any timing field is a producer-side regression
    # and must reach the surface manifest as ``status: failed``,
    # never as ``ok``. The forbidden-field check fires BEFORE the
    # required-field loop so a malformed timing contamination
    # surfaces its own typed waiver instead of being masked by a
    # generic "missing required field" message.
    for forbidden in _FORBIDDEN_TOP_FIELDS:
        if forbidden in rec:
            return (
                f"forbidden timing field present: {forbidden!r}; "
                f"the metric extractor calls zero CUPTI APIs by "
                f"construction, so wall-clock fields are a "
                f"contract violation."
            )
    for key in rec:
        for prefix in _FORBIDDEN_TOP_FIELD_PREFIXES:
            if key.startswith(prefix):
                return (
                    f"forbidden timing field prefix: {key!r} starts "
                    f"with {prefix!r}; the metric extractor calls zero "
                    f"CUPTI APIs by construction, so wall-clock fields "
                    f"are a contract violation."
                )
    for key in _REQUIRED_TOP_FIELDS:
        if key not in rec:
            return f"missing required field: {key!r}"
    if not _is_strict_int(rec["schema_version"]):
        return (
            f"schema_version must be a non-bool integer, got "
            f"{type(rec['schema_version']).__name__}"
        )
    if not isinstance(rec["tool_version"], str) or not rec["tool_version"]:
        return "tool_version must be a non-empty string"
    if not isinstance(rec["launch_kind"], str) or not rec["launch_kind"]:
        return "launch_kind must be a non-empty string"
    if not _is_strict_int(rec["launch_index"]):
        return (
            f"launch_index must be a non-bool integer, got "
            f"{type(rec['launch_index']).__name__}: {rec['launch_index']!r}"
        )
    if not _is_strict_int(rec["ctx_id"]):
        return (
            f"ctx_id must be a non-bool integer, got "
            f"{type(rec['ctx_id']).__name__}: {rec['ctx_id']!r}"
        )
    if not _is_strict_int(rec["stream_id"]):
        return (
            f"stream_id must be a non-bool integer, got "
            f"{type(rec['stream_id']).__name__}: {rec['stream_id']!r}"
        )
    # tma_descriptor must be null OR an object with every documented
    # sub-field present. Numbers / strings / arrays are rejected; an
    # object missing any sub-field is also rejected so a downstream
    # consumer cannot silently drop fields between schema revisions.
    tma = rec["tma_descriptor"]
    if tma is not None:
        if not isinstance(tma, dict):
            return (
                f"tma_descriptor must be null or an object, got "
                f"{type(tma).__name__}"
            )
        for sub in _REQUIRED_TMA_DESCRIPTOR_SUBFIELDS:
            if sub not in tma:
                return (
                    f"tma_descriptor missing required sub-field: {sub!r}"
                )
    # confidence object must carry the per-field labels mentioned in the
    # schema doc. For tma_descriptor specifically the consistency rule is
    # `absent` iff tma_descriptor is null, `high` iff non-null — any other
    # combination is a producer/consumer drift the parser must surface.
    conf = rec["confidence"]
    if not isinstance(conf, dict):
        return (
            f"confidence must be an object, got {type(conf).__name__}"
        )
    if "tma_descriptor" not in conf:
        return "confidence missing required label: 'tma_descriptor'"
    tma_conf = conf["tma_descriptor"]
    if not isinstance(tma_conf, str):
        return (
            f"confidence.tma_descriptor must be a string, got "
            f"{type(tma_conf).__name__}: {tma_conf!r}"
        )
    if tma is None and tma_conf != "absent":
        return (
            f"confidence.tma_descriptor must be 'absent' when "
            f"tma_descriptor is null, got {tma_conf!r}"
        )
    if tma is not None and tma_conf != "high":
        return (
            f"confidence.tma_descriptor must be 'high' when "
            f"tma_descriptor is a non-null object, got {tma_conf!r}"
        )
    # sector_efficiency_global_{load,store} must be null OR a numeric
    # value in [0.0, 1.0]. The producer emits 6 decimal places; the
    # parser accepts both int (e.g. exactly 0 or 1) and float so a
    # downstream rewrite that strips trailing zeros doesn't break
    # round-tripping. Bool is rejected (int subclass quirk).
    for field_name in ("sector_efficiency_global_load",
                       "sector_efficiency_global_store"):
        value = rec[field_name]
        if value is None:
            continue
        if isinstance(value, bool):
            return (
                f"{field_name} must be null or numeric, got bool"
            )
        if not isinstance(value, (int, float)):
            return (
                f"{field_name} must be null or a numeric value, got "
                f"{type(value).__name__}: {value!r}"
            )
        if not (0.0 <= float(value) <= 1.0):
            return (
                f"{field_name} must be in [0.0, 1.0], got {value!r}"
            )
    # bank_conflicts must be a non-negative integer. Bool is rejected
    # explicitly (Python int-subclass quirk). The producer accumulates a
    # non-negative uint64 counter on the device side and emits it as an
    # integer; negative values are impossible by construction, so the
    # parser surfaces any negative as a ParseError to flag producer-side
    # regression.
    bank_conflicts = rec["bank_conflicts"]
    if isinstance(bank_conflicts, bool) or not _is_strict_int(bank_conflicts):
        return (
            f"bank_conflicts must be a non-bool integer, got "
            f"{type(bank_conflicts).__name__}: {bank_conflicts!r}"
        )
    if bank_conflicts < 0:
        return (
            f"bank_conflicts must be non-negative, got {bank_conflicts!r}"
        )
    # branch_active_lanes_hist must be a fixed-length list of 33
    # non-negative integers. Bool is rejected explicitly (Python
    # int-subclass quirk). The producer accumulates non-negative uint64
    # counters on the device side and emits them as integers; negative
    # values are impossible by construction, so the parser surfaces any
    # negative as a ParseError to flag producer-side regression. The
    # length check guards against a producer-side bucket-count change
    # silently breaking the bucket-indexing contract.
    hist = rec["branch_active_lanes_hist"]
    if not isinstance(hist, list):
        return (
            f"branch_active_lanes_hist must be a list, got "
            f"{type(hist).__name__}: {hist!r}"
        )
    if len(hist) != _BRANCH_ACTIVE_LANES_HIST_LEN:
        return (
            f"branch_active_lanes_hist must have exactly "
            f"{_BRANCH_ACTIVE_LANES_HIST_LEN} buckets, got {len(hist)}"
        )
    for i, bucket in enumerate(hist):
        if isinstance(bucket, bool) or not _is_strict_int(bucket):
            return (
                f"branch_active_lanes_hist[{i}] must be a non-bool "
                f"integer, got {type(bucket).__name__}: {bucket!r}"
            )
        if bucket < 0:
            return (
                f"branch_active_lanes_hist[{i}] must be non-negative, "
                f"got {bucket!r}"
            )

    # launch_kind == "graph_launch_not_observed" intentionally omits
    # the kernel identity triple (kernel_name / kernel_name_mangled /
    # function_uid) per the schema doc, because the individual graph
    # nodes are exactly what the tool failed to observe; the remaining
    # launch kinds (regular / cooperative / graph_node_traced) MUST
    # carry the triple. The validator enforces both halves so a
    # producer-side mistake — emitting the triple on a graph_launch_not_observed
    # record or omitting it on a regular record — is caught here.
    launch_kind = rec["launch_kind"]
    graph_not_observed = launch_kind == "graph_launch_not_observed"
    if graph_not_observed:
        for forbidden_for_graph in (
            "kernel_name",
            "kernel_name_mangled",
            "function_uid",
        ):
            if forbidden_for_graph in rec:
                return (
                    f"graph_launch_not_observed record must NOT carry "
                    f"{forbidden_for_graph!r}; the graph nodes are the "
                    f"exact entities the tool failed to observe."
                )
    else:
        if not isinstance(rec.get("kernel_name"), str) or not rec["kernel_name"]:
            return (
                f"kernel_name must be a non-empty string for "
                f"launch_kind={launch_kind!r}; got "
                f"{type(rec.get('kernel_name')).__name__}: "
                f"{rec.get('kernel_name')!r}"
            )
        if (
            not isinstance(rec.get("kernel_name_mangled"), str)
            or not rec["kernel_name_mangled"]
        ):
            return (
                f"kernel_name_mangled must be a non-empty string for "
                f"launch_kind={launch_kind!r}"
            )
        if not _is_strict_int(rec.get("function_uid")):
            return (
                f"function_uid must be a non-bool integer for "
                f"launch_kind={launch_kind!r}; got "
                f"{type(rec.get('function_uid')).__name__}"
            )

    # Launch dimensions. grid_dim / block_dim / cluster_dim are each
    # exactly three non-negative integers; graph_launch_not_observed
    # records carry the [0,0,0] sentinel triple. The validator accepts
    # the sentinel naturally because every entry is required to be a
    # non-negative integer, and (0, 0, 0) satisfies that.
    for dim_name in ("grid_dim", "block_dim", "cluster_dim"):
        if not _is_xyz_int_triple(rec[dim_name]):
            return (
                f"{dim_name} must be a list of exactly three non-bool "
                f"non-negative integers, got {rec[dim_name]!r}"
            )
    if isinstance(rec["dynamic_smem"], bool) or not _is_strict_int(
        rec["dynamic_smem"]
    ):
        return (
            f"dynamic_smem must be a non-bool integer, got "
            f"{type(rec['dynamic_smem']).__name__}"
        )
    if rec["dynamic_smem"] < 0:
        return (
            f"dynamic_smem must be non-negative, got {rec['dynamic_smem']!r}"
        )

    # instruction_class_histogram is a sparse map from class-name to a
    # non-negative integer count. Buckets with zero counts are omitted
    # from the wire format, so an empty object is a legitimate
    # observation (no instrumented instructions fired this launch — the
    # canonical case for graph_launch_not_observed records).
    hist_obj = rec["instruction_class_histogram"]
    if not isinstance(hist_obj, dict):
        return (
            f"instruction_class_histogram must be an object, got "
            f"{type(hist_obj).__name__}"
        )
    for k, v in hist_obj.items():
        if not isinstance(k, str) or not k:
            return (
                f"instruction_class_histogram key must be a non-empty "
                f"string, got {k!r}"
            )
        if isinstance(v, bool) or not _is_strict_int(v):
            return (
                f"instruction_class_histogram[{k!r}] must be a non-bool "
                f"integer count, got {type(v).__name__}: {v!r}"
            )
        if v < 0:
            return (
                f"instruction_class_histogram[{k!r}] must be "
                f"non-negative, got {v!r}"
            )

    # cross_surface_join_key — composite key for the post-hoc CUPTI
    # Activity join. Every sub-field listed in JSONL_SCHEMA.md must be
    # present with the correct type so the report-side joiner can
    # assign every launch to exactly one ``join_status`` bucket
    # (matched / ambiguous / unmatched_nvbit_only / unmatched_cupti_only /
    # unjoinable) without resorting to inference.
    cskey = rec["cross_surface_join_key"]
    if not isinstance(cskey, dict):
        return (
            f"cross_surface_join_key must be an object, got "
            f"{type(cskey).__name__}"
        )
    for sub in _REQUIRED_CROSS_SURFACE_JOIN_KEY_INT_OR_NULL:
        if sub not in cskey:
            return (
                f"cross_surface_join_key missing required sub-field: {sub!r}"
            )
        if cskey[sub] is not None and not _is_strict_int(cskey[sub]):
            return (
                f"cross_surface_join_key.{sub} must be null or a non-bool "
                f"integer, got {type(cskey[sub]).__name__}: {cskey[sub]!r}"
            )
    for sub in _REQUIRED_CROSS_SURFACE_JOIN_KEY_STR_OR_NULL:
        if sub not in cskey:
            return (
                f"cross_surface_join_key missing required sub-field: {sub!r}"
            )
        if cskey[sub] is not None and not isinstance(cskey[sub], str):
            return (
                f"cross_surface_join_key.{sub} must be null or a string, "
                f"got {type(cskey[sub]).__name__}: {cskey[sub]!r}"
            )
    for sub in _REQUIRED_CROSS_SURFACE_JOIN_KEY_INT:
        if sub not in cskey:
            return (
                f"cross_surface_join_key missing required sub-field: {sub!r}"
            )
        if not _is_strict_int(cskey[sub]):
            return (
                f"cross_surface_join_key.{sub} must be a non-bool integer, "
                f"got {type(cskey[sub]).__name__}: {cskey[sub]!r}"
            )
    for sub in _REQUIRED_CROSS_SURFACE_JOIN_KEY_XYZ:
        if sub not in cskey:
            return (
                f"cross_surface_join_key missing required sub-field: {sub!r}"
            )
        if not _is_xyz_int_triple(cskey[sub]):
            return (
                f"cross_surface_join_key.{sub} must be a list of three "
                f"non-bool non-negative integers, got {cskey[sub]!r}"
            )

    # substatus — operator-actionable flags. The validator enforces
    # the type of each documented sub-field; the meaning of each
    # field is documented in JSONL_SCHEMA.md.
    sub_status = rec["substatus"]
    if not isinstance(sub_status, dict):
        return (
            f"substatus must be an object, got {type(sub_status).__name__}"
        )
    for sub in _REQUIRED_SUBSTATUS_BOOL:
        if sub not in sub_status:
            return f"substatus missing required sub-field: {sub!r}"
        if not isinstance(sub_status[sub], bool):
            return (
                f"substatus.{sub} must be a boolean, got "
                f"{type(sub_status[sub]).__name__}: {sub_status[sub]!r}"
            )
    for sub in _REQUIRED_SUBSTATUS_INT:
        if sub not in sub_status:
            return f"substatus missing required sub-field: {sub!r}"
        if (
            isinstance(sub_status[sub], bool)
            or not _is_strict_int(sub_status[sub])
        ):
            return (
                f"substatus.{sub} must be a non-bool integer, got "
                f"{type(sub_status[sub]).__name__}: {sub_status[sub]!r}"
            )
        if sub_status[sub] < 0:
            return (
                f"substatus.{sub} must be non-negative, got "
                f"{sub_status[sub]!r}"
            )
    for sub in _REQUIRED_SUBSTATUS_STR_OR_NULL:
        if sub not in sub_status:
            return f"substatus missing required sub-field: {sub!r}"
        if (
            sub_status[sub] is not None
            and not isinstance(sub_status[sub], str)
        ):
            return (
                f"substatus.{sub} must be null or a string, got "
                f"{type(sub_status[sub]).__name__}: {sub_status[sub]!r}"
            )
    # graph_launch_not_observed canonical-waiver invariant: the
    # producer always sets ``substatus.waiver_reason`` to the
    # canonical string the harness manifest expects (per
    # ``write_graph_launch_not_observed_record`` in the producer).
    # The validator does NOT pin the string verbatim because future
    # producer-side waiver-message edits should not require a
    # parser-side schema bump; it just enforces that the slot is
    # non-null for graph_launch_not_observed records.
    if graph_not_observed and sub_status["waiver_reason"] is None:
        return (
            "graph_launch_not_observed record must carry a non-null "
            "substatus.waiver_reason"
        )

    # filter_stats_snapshot — running counters at emit time. Every
    # field is a non-negative integer counter; a non-integer or
    # negative value indicates a producer-side regression.
    fss = rec["filter_stats_snapshot"]
    if not isinstance(fss, dict):
        return (
            f"filter_stats_snapshot must be an object, got "
            f"{type(fss).__name__}"
        )
    for sub in _REQUIRED_FILTER_STATS_SNAPSHOT_INT:
        if sub not in fss:
            return (
                f"filter_stats_snapshot missing required sub-field: {sub!r}"
            )
        if isinstance(fss[sub], bool) or not _is_strict_int(fss[sub]):
            return (
                f"filter_stats_snapshot.{sub} must be a non-bool "
                f"integer, got {type(fss[sub]).__name__}: {fss[sub]!r}"
            )
        if fss[sub] < 0:
            return (
                f"filter_stats_snapshot.{sub} must be non-negative, "
                f"got {fss[sub]!r}"
            )

    # representative_skipped_paths — wire-versioned per the schema's
    # version-discipline rule. Schema v2 (tool_version >= 0.0.14)
    # requires the field; schema v1 (older historical bundles)
    # defaults to [] so the round trip on captured pre-bump evidence
    # still succeeds. Requiring the field unconditionally under
    # schema_version=1 would break every preserved older bundle. The
    # version-aware path here closes that gap while keeping the
    # contract strict for fresh emissions.
    schema_v = int(rec["schema_version"])
    if "representative_skipped_paths" not in fss:
        if schema_v >= 2:
            return (
                "filter_stats_snapshot missing required sub-field: "
                "'representative_skipped_paths' (required by schema_version >= 2)"
            )
        # Schema v1 forward-compat: default to empty list. Mutating
        # the record in-place keeps downstream consumers (reconciler,
        # aggregator) on a single code path regardless of source
        # schema version.
        fss["representative_skipped_paths"] = []
    paths = fss["representative_skipped_paths"]
    if not isinstance(paths, list):
        return (
            f"filter_stats_snapshot.representative_skipped_paths must be "
            f"a list, got {type(paths).__name__}"
        )
    if len(paths) > 10:
        return (
            f"filter_stats_snapshot.representative_skipped_paths must be "
            f"capped at 10 entries, got {len(paths)}"
        )
    for i, p in enumerate(paths):
        if not isinstance(p, str):
            return (
                f"filter_stats_snapshot.representative_skipped_paths"
                f"[{i}] must be a string, got {type(p).__name__}: {p!r}"
            )

    # host snapshot — captured at first emit.
    host = rec["host"]
    if not isinstance(host, dict):
        return f"host must be an object, got {type(host).__name__}"
    for sub in _REQUIRED_HOST_STR:
        if sub not in host:
            return f"host missing required sub-field: {sub!r}"
        if not isinstance(host[sub], str):
            return (
                f"host.{sub} must be a string, got "
                f"{type(host[sub]).__name__}: {host[sub]!r}"
            )
    for sub in _REQUIRED_HOST_INT:
        if sub not in host:
            return f"host missing required sub-field: {sub!r}"
        if isinstance(host[sub], bool) or not _is_strict_int(host[sub]):
            return (
                f"host.{sub} must be a non-bool integer, got "
                f"{type(host[sub]).__name__}: {host[sub]!r}"
            )

    # static_sidecar may be null (the current producer's default) OR an
    # object carrying any subset of the four documented sub-fields,
    # each individually null or a non-bool integer. The parser
    # explicitly tolerates the null shape so the harness-side T16 join
    # can fill the slot incrementally without breaking parse.
    sidecar = rec["static_sidecar"]
    if sidecar is not None:
        if not isinstance(sidecar, dict):
            return (
                f"static_sidecar must be null or an object, got "
                f"{type(sidecar).__name__}"
            )
        for sub in _OPTIONAL_STATIC_SIDECAR_SUBFIELDS:
            if sub in sidecar:
                v = sidecar[sub]
                if v is None:
                    continue
                if isinstance(v, bool) or not _is_strict_int(v):
                    return (
                        f"static_sidecar.{sub} must be null or a "
                        f"non-bool integer, got {type(v).__name__}: {v!r}"
                    )

    # confidence labels — the JSONL_SCHEMA.md table enumerates the
    # current set of label slots. The parser already validated
    # confidence.tma_descriptor above; the remaining six slots must be
    # present and string-typed so the harness-side T25 tolerance gate
    # can flip individual labels from "placeholder" to "high" without
    # the parser losing track of which fields are calibrated.
    for label_field in (
        "instruction_class_histogram",
        "sector_efficiency_global_load",
        "sector_efficiency_global_store",
        "bank_conflicts",
        "branch_active_lanes_hist",
        "static_sidecar",
    ):
        if label_field not in conf:
            return f"confidence missing required label: {label_field!r}"
        if not isinstance(conf[label_field], str):
            return (
                f"confidence.{label_field} must be a string, got "
                f"{type(conf[label_field]).__name__}: {conf[label_field]!r}"
            )

    return None


def parse_launches_jsonl(
    source: Union[Path, str, Iterable[bytes]],
) -> Tuple[List[LaunchRecord], List[ParseError]]:
    """Parse the metric extractor's per-launch JSON-L.

    ``source`` is either a filesystem path (``Path`` or ``str``) or an
    iterable of byte-strings (one per line, terminator preserved).

    Returns ``(records, errors)``. ``errors`` is empty when the file
    parsed cleanly; any non-empty list signals the bundle should be
    flagged ``failed`` per the per-surface manifest contract.
    """
    records: List[LaunchRecord] = []
    errors: List[ParseError] = []

    line_iter: Iterable[bytes]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return records, errors
        line_iter = path.open("rb").readlines()
    else:
        line_iter = list(source)

    for line_no, raw_bytes in enumerate(line_iter, start=1):
        if not raw_bytes.strip():
            continue
        if _looks_truncated(raw_bytes):
            errors.append(
                ParseError(
                    line_number=line_no,
                    reason=(
                        "mid-line truncation: line does not end with "
                        "'}\\n'"
                    ),
                    raw=raw_bytes.decode("utf-8", errors="replace").rstrip(
                        "\n"
                    ),
                )
            )
            continue
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                ParseError(
                    line_number=line_no,
                    reason=f"json decode error: {exc}",
                    raw=raw_bytes.decode("utf-8", errors="replace").rstrip(
                        "\n"
                    ),
                )
            )
            continue
        if not isinstance(payload, dict):
            errors.append(
                ParseError(
                    line_number=line_no,
                    reason=f"top-level value is {type(payload).__name__},"
                    f" expected JSON object",
                    raw=raw_bytes.decode("utf-8", errors="replace").rstrip(
                        "\n"
                    ),
                )
            )
            continue
        why = _validate_record(payload)
        if why is not None:
            errors.append(
                ParseError(
                    line_number=line_no,
                    reason=why,
                    raw=raw_bytes.decode("utf-8", errors="replace").rstrip(
                        "\n"
                    ),
                )
            )
            continue
        try:
            records.append(
                LaunchRecord(
                    schema_version=int(payload["schema_version"]),
                    tool_version=str(payload["tool_version"]),
                    launch_kind=str(payload["launch_kind"]),
                    launch_index=int(payload["launch_index"]),
                    ctx_id=int(payload["ctx_id"]),
                    stream_id=int(payload["stream_id"]),
                    raw=payload,
                )
            )
        except (TypeError, ValueError, KeyError) as exc:
            # Defensive: _validate_record above should have caught
            # everything that could trip the conversions here, but
            # never let an exception escape the parser. The harness
            # surface plugin uses any error in the returned list as
            # signal to mark the surface `failed`.
            errors.append(
                ParseError(
                    line_number=line_no,
                    reason=(
                        f"defensive: unexpected exception constructing "
                        f"LaunchRecord ({type(exc).__name__}: {exc})"
                    ),
                    raw=raw_bytes.decode("utf-8", errors="replace").rstrip(
                        "\n"
                    ),
                )
            )

    return records, errors
