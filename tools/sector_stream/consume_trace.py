"""Sector-stream live-trace consumer (C-AC-F live-evidence path).

Walks a simbench NVBit trace directory, reconstructs each
per-warp memory access record, runs it through
:func:`tools.sector_stream.generator.compute_sector_stream`, and
writes one JSONL line per record to
``<trace_dir>/sector_stream/sector_stream.jsonl``.

The capability core consumes only an ``AdmittedTraceSources`` allowlist and
pinned snapshots. ``consume_trace_dir`` preserves path-based CLI/direct-caller
compatibility by requiring admission, opening the capability, and invoking the
core.

Three classes of input record map to one of three output
dispositions:

1. **Memory-class records the generator admits.**
   ``GLOBAL`` and ``LOCAL`` records with non-empty addresses
   produce admitted JSONL lines via
   :func:`compute_sector_stream`.

2. **Memory-class records the generator typed-excludes.**
   ``SHARED``, ``TEXTURE``, ``SURFACE``, ``ATOMIC``, ``TMA``,
   ``ASYNC_COPY``, and ``CLUSTER_DSMEM`` produce JSONL lines
   with ``typed_reason=sector_stream_class_unsupported``.

3. **Non-memory records.**
   Instructions whose ``memory_op_class`` is absent on the
   wire (ALU, branch, control flow) and that carry no
   addresses are skipped — they are not memory accesses and
   the sector-stream generator does not apply.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.local_labels.immutable_artifact import (  # noqa: E402
    ensure_directory,
    read_regular,
    replace_regular_bytes,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    AdmittedTraceSources,
    TraceAdmissionError,
    open_admitted_trace_sources,
    require_trace_admission,
)

_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "sector_stream.schema.json"
)


def _freeze_json(value: Any) -> Any:
    """Return a recursively immutable copy of one JSON value."""
    if isinstance(value, dict):
        return MappingProxyType({
            str(key): _freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(
        "sector-stream records must contain only JSON-compatible values; "
        f"found {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True, init=False)
class AdmittedSectorStream:
    """Immutable sector records bound to one admitted trace generation.

    Construction is intentionally private to this module's capability-backed
    producer. Downstream consumers receive this object rather than an unbound
    list and must call :meth:`verify_for` before using its records.
    """

    records: tuple[Mapping[str, Any], ...]
    provenance: tuple[Mapping[str, Any], ...]
    trace_admission_sha256: str
    trace_filesystem_generation_sha256: str
    contract_ok: bool
    contract_typed_reason: str | None
    contract_detail: str | None
    total_records: int
    admitted_records: int
    typed_exclusions: Mapping[str, int]
    schema_admits_all: bool
    schema_first_failure: str | None
    jsonl_path: Path | None
    _jsonl_payload: bytes

    def __new__(cls) -> "AdmittedSectorStream":
        raise TypeError(
            "AdmittedSectorStream can only be constructed by the "
            "capability-backed sector-stream producer"
        )

    @classmethod
    def _from_admitted_sources(
        cls,
        *,
        sources: AdmittedTraceSources,
        records: list[dict[str, Any]],
        provenance: list[dict[str, Any]],
        admitted_records: int,
        typed_exclusions: dict[str, int],
        schema_admits_all: bool,
        schema_first_failure: str | None,
        jsonl_path: Path | None,
        jsonl_payload: bytes,
    ) -> "AdmittedSectorStream":
        if not isinstance(sources, AdmittedTraceSources):
            raise TypeError("sources must be an AdmittedTraceSources capability")
        sources.verify_unchanged()
        admission = sources.admission
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "records",
            tuple(_freeze_json(record) for record in records),
        )
        object.__setattr__(
            instance,
            "provenance",
            tuple(_freeze_json(record) for record in provenance),
        )
        object.__setattr__(
            instance,
            "trace_admission_sha256",
            admission.trace_admission_sha256,
        )
        object.__setattr__(
            instance,
            "trace_filesystem_generation_sha256",
            admission.trace_filesystem_generation_sha256,
        )
        object.__setattr__(instance, "contract_ok", True)
        object.__setattr__(instance, "contract_typed_reason", None)
        object.__setattr__(instance, "contract_detail", None)
        object.__setattr__(instance, "total_records", len(records))
        object.__setattr__(instance, "admitted_records", admitted_records)
        object.__setattr__(
            instance,
            "typed_exclusions",
            MappingProxyType(dict(typed_exclusions)),
        )
        object.__setattr__(instance, "schema_admits_all", schema_admits_all)
        object.__setattr__(instance, "schema_first_failure", schema_first_failure)
        object.__setattr__(instance, "jsonl_path", jsonl_path)
        object.__setattr__(instance, "_jsonl_payload", bytes(jsonl_payload))
        return instance

    def verify_for(self, sources: AdmittedTraceSources) -> None:
        """Bind both portable and local-generation seals to ``sources``."""
        if not isinstance(sources, AdmittedTraceSources):
            raise TypeError("sources must be an AdmittedTraceSources capability")
        sources.verify_unchanged()
        admission = sources.admission
        mismatches: list[str] = []
        if self.trace_admission_sha256 != admission.trace_admission_sha256:
            mismatches.append(
                "trace_admission_sha256 "
                f"sector={self.trace_admission_sha256} "
                f"source={admission.trace_admission_sha256}"
            )
        if (
            self.trace_filesystem_generation_sha256
            != admission.trace_filesystem_generation_sha256
        ):
            mismatches.append(
                "trace_filesystem_generation_sha256 "
                f"sector={self.trace_filesystem_generation_sha256} "
                f"source={admission.trace_filesystem_generation_sha256}"
            )
        if mismatches:
            raise TraceAdmissionError(
                "trace_snapshot_changed",
                "sector-stream result does not belong to the admitted trace "
                "capability: " + "; ".join(mismatches),
            )
        sources.verify_unchanged()

    def jsonl_bytes(self) -> bytes:
        """Return the canonical generated JSONL bytes without filesystem I/O."""
        return self._jsonl_payload

    def mutable_records(self) -> list[dict[str, Any]]:
        """Return detached JSON records after capability identity is verified."""
        records: list[dict[str, Any]] = []
        for line in self._jsonl_payload.splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):  # construction invariant
                raise TypeError("sector-stream JSONL record must be an object")
            records.append(value)
        return records


# Lazy-importable references that the body of consume_trace_dir
# resolves at call time. Keeping them out of the module-load chain
# means importing :mod:`consume_trace` does not require the
# protobuf bindings to be on disk (the unit tests patch the
# protobuf module path before invoking the consumer).
def _import_helpers() -> tuple[Any, Any, Any]:
    """Import the trace-contract helpers and the generator.

    Returns ``(import_trace_contract, generator,
    instruction_pb2_module)``. The third element comes from the
    project's ``_import_pb_modules()`` helper which compiles the
    proto bindings from
    ``tools/nvbit_tracer/traces_enhanced/dynamic_trace/`` on
    first call and caches them.
    """
    from tools.nvbit_tracer import import_trace_contract
    from tools.sector_stream import generator

    _, instruction_pb2, _ = import_trace_contract._import_pb_modules()
    return import_trace_contract, generator, instruction_pb2


def _proto_class_to_string(value: int, instruction_pb2: Any) -> str:
    """Map a proto ``MemoryOpClass`` enum value to the generator's
    uppercase string vocabulary.

    The proto enum names are ``MEMORY_OP_CLASS_<CLASS>`` (e.g.
    ``MEMORY_OP_CLASS_GLOBAL``); the generator expects the class
    suffix only (``GLOBAL``). The mapping is derived from the
    proto descriptor so a future enum widening on the producer
    side automatically widens this mapping without a code edit
    here.
    """
    # `Name(value)` returns the proto enum-symbol name, e.g.
    # `MEMORY_OP_CLASS_GLOBAL`. Strip the prefix to get the
    # generator's vocabulary.
    enum_name: str = (
        instruction_pb2.MemoryOpClass.Name(value)
    )
    prefix = "MEMORY_OP_CLASS_"
    if enum_name.startswith(prefix):
        return enum_name[len(prefix):]
    # Defensive: a proto-side rename would surface here as a
    # caller bug — return the raw name so the generator's
    # `sector_stream_class_unsupported` typed exclusion fires.
    return enum_name


def _record_for_address(
    *,
    inst: Any,
    addr: Any,
    import_trace_contract: Any,
    generator: Any,
    instruction_pb2: Any,
) -> dict[str, Any] | None:
    """Compute the sector-stream record for one
    ``(instruction, address)`` pair.

    Returns ``None`` if the pair has no usable lane vector
    (e.g. base-stride record with structural malformation that
    the upstream validator already accepted but
    ``_reconstruct_lane_vector`` rejects). Real malformations
    are caught by ``validate_threadblock`` before this function
    runs; the ``None`` path is a defensive fallback.
    """
    and_mask = inst.active_mask & inst.predicate_mask
    address_mask = import_trace_contract._address_lane_mask(addr, and_mask)

    lane_vector, recon_reason = (
        import_trace_contract._reconstruct_lane_vector(addr, address_mask)
    )
    if lane_vector is None:
        # The validator's contract walk already runs before this
        # path; a structural lane-vector failure here means the
        # validator widened acceptance without updating the
        # consumer. Surface the reason rather than crashing.
        return {
            "result_kind": "typed_exclusion",
            "typed_reason": "sector_stream_class_unsupported",
            "memory_op_class": _proto_class_to_string(
                inst.memory_op_class, instruction_pb2
            ),
            "lane_vector_reconstruction_failure": recon_reason,
        }

    # Replace the inactive-lane sentinel with 0 so the generator
    # can treat the lane_addresses sequence uniformly. Inactive
    # lanes do not contribute (their bits are clear in
    # ``and_mask``); the address values for inactive lanes are
    # ignored by the generator.
    lane_addresses = [
        addr if addr != import_trace_contract.LANE_INACTIVE_SENTINEL
        else 0
        for addr in lane_vector
    ]

    memory_op_class = _proto_class_to_string(
        inst.memory_op_class, instruction_pb2
    )

    # Use the address-level data_width when present; fall back to
    # the instruction-level field. The wire contract pins both,
    # but `address.data_width` is the per-record value and is the
    # one the generator should consume.
    data_width = (
        addr.data_width if addr.data_width else inst.data_width
    )

    # Per-address memory space (address-carrying contract). Present only on
    # post-transition traces; absent -> None -> the generator falls back to the
    # instruction-level class. Authoritative over memory_op_class when present.
    address_space = (
        addr.address_space if addr.HasField("address_space") else None
    )
    return generator.compute_sector_stream(
        lane_addresses=lane_addresses,
        active_mask=address_mask,
        predicate_mask=address_mask,
        data_width=data_width,
        memory_op_class=memory_op_class,
        address_space=address_space,
    )


def _consume_threadblock(
    tb: Any,
    *,
    label: str,
    import_trace_contract: Any,
    generator: Any,
    instruction_pb2: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk one threadblock and yield sector-stream records.

    Returns ``(records, provenance)`` where the two lists align
    by index. The records are strictly schema-conformant (the
    on-disk JSONL artifact); the provenance entries are
    in-process debug structures with ``label``, ``warp_index``,
    ``inst_index``, ``vpc`` for callers that want to map a
    record back to its trace location. The split is the R7
    correction for the R6 ``_origin``-on-disk bug Codex
    flagged.
    """
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for warp_index in sorted(tb.warps.keys()):
        warp = tb.warps[warp_index]
        for inst_index, inst in enumerate(warp.instructions):
            if not inst.HasField("memory_op_class"):
                # ALU / branch / control flow — no memory access.
                continue
            if not inst.addresses:
                # Memory instruction whose dynamic instance had
                # all lanes predicated off; the producer emitted
                # the class but no per-lane addresses. Skip
                # because we cannot reconstruct a lane vector.
                continue
            for addr in inst.addresses:
                rec = _record_for_address(
                    inst=inst,
                    addr=addr,
                    import_trace_contract=import_trace_contract,
                    generator=generator,
                    instruction_pb2=instruction_pb2,
                )
                if rec is None:
                    continue
                records.append(rec)
                provenance.append(
                    {
                        "label": label,
                        "warp_index": warp_index,
                        "inst_index": inst_index,
                        "vpc": (
                            inst.vpc if inst.HasField("vpc") else None
                        ),
                    }
                )
    return records, provenance


def _admits_against_schema(
    record: dict[str, Any], schema: dict[str, Any]
) -> tuple[bool, str | None]:
    """Validate one sector-stream record against the schema's
    ``oneOf``. Hand-rolled (no ``jsonschema`` dependency) to
    match the project's test convention.

    Validates the EXACT record passed in. No field stripping,
    no copy-then-prune — the function operates on the same
    dict that callers will write to disk, so a discrepancy
    between in-memory schema admission and on-disk artifact
    cannot recur (the R7 correction for the R6
    ``_origin``-on-disk bug).
    """
    branches = schema["oneOf"]
    discriminator = record.get("result_kind")
    branch: dict[str, Any] | None = None
    for b in branches:
        if (
            b["properties"]["result_kind"].get("const")
            == discriminator
        ):
            branch = b
            break
    if branch is None:
        return False, f"unknown result_kind={discriminator!r}"

    for field in branch["required"]:
        if field not in record:
            return False, f"missing required field {field!r}"

    if not branch.get("additionalProperties", True):
        allowed = set(branch["properties"].keys())
        extra = set(record.keys()) - allowed
        if extra:
            return False, f"unexpected fields: {sorted(extra)}"

    properties = branch["properties"]
    for field, value in record.items():
        if field not in properties:
            continue
        spec = properties[field]
        if "const" in spec and value != spec["const"]:
            return False, (
                f"field {field!r}={value!r} != "
                f"const {spec['const']!r}"
            )
        if "enum" in spec and value not in spec["enum"]:
            return False, (
                f"field {field!r}={value!r} not in enum "
                f"{spec['enum']}"
            )
    return True, None


def consume_admitted_trace_sources(
    *,
    sources: AdmittedTraceSources,
    output_root: Path | None = None,
    write_jsonl: bool = True,
) -> AdmittedSectorStream:
    """Consume one already-admitted trace through the sector generator.

    ``sources`` is mandatory and keyword-only. The core enumerates only
    ``sources.threadblock_paths`` and reads every payload through
    ``read_threadblock_snapshot``; it never validates, inventories, or reopens
    the trace by pathname. ``output_root`` is used only as an output
    destination when ``write_jsonl`` is true.

    The immutable result carries the exact generated records plus both the
    portable trace-admission seal and the local filesystem-generation seal.
    Downstream consumers must verify the result against their capability before
    consuming records; an unbound bare-list core contract is deliberately not
    exposed.
    """
    if not isinstance(sources, AdmittedTraceSources):
        raise TypeError("sources must be an AdmittedTraceSources capability")
    if type(write_jsonl) is not bool:
        raise TypeError("write_jsonl must be boolean")
    if write_jsonl and output_root is None:
        raise ValueError("output_root is required when write_jsonl is true")

    sources.verify_unchanged()
    admission = sources.admission
    pb_paths = sources.threadblock_paths
    if len(pb_paths) != admission.threadblock_count:
        raise TraceAdmissionError(
            "trace_snapshot_changed",
            "admitted threadblock path count disagrees with admission",
        )

    import_trace_contract, generator, instruction_pb2 = _import_helpers()
    schema = json.loads(read_regular(_SCHEMA_PATH))
    if not isinstance(schema, dict):
        raise TypeError("sector-stream schema must be a JSON object")
    threadblock_pb2 = import_trace_contract._import_pb_modules()[0]

    all_records: list[dict[str, Any]] = []
    all_provenance: list[dict[str, Any]] = []
    for relative_path in pb_paths:
        snapshot = sources.read_threadblock_snapshot(relative_path)
        tb = threadblock_pb2.threadblock()
        tb.ParseFromString(snapshot.payload)
        records, provenance = _consume_threadblock(
            tb,
            label=relative_path,
            import_trace_contract=import_trace_contract,
            generator=generator,
            instruction_pb2=instruction_pb2,
        )
        all_records.extend(records)
        all_provenance.extend(provenance)

    admitted = 0
    typed_exclusions: dict[str, int] = {}
    schema_admits_all = True
    schema_first_failure: str | None = None
    for rec in all_records:
        if rec.get("result_kind") == "sector_stream":
            admitted += 1
        else:
            reason = rec.get("typed_reason", "<missing>")
            typed_exclusions[reason] = (
                typed_exclusions.get(reason, 0) + 1
            )
        ok, why = _admits_against_schema(rec, schema)
        if not ok and schema_admits_all:
            schema_admits_all = False
            schema_first_failure = why

    payload = b"".join(
        (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8")
        for rec in all_records
    )
    jsonl_path: Path | None = None
    if write_jsonl:
        if not schema_admits_all:
            raise TraceAdmissionError(
                "sector_stream_schema_invalid",
                "refusing to publish schema-invalid sector-stream bytes: "
                f"{schema_first_failure}",
            )
        if output_root is None:  # guarded above; keep failure explicit
            raise ValueError("output_root is required when write_jsonl is true")
        out_dir = Path(output_root) / "sector_stream"
        ensure_directory(out_dir)
        jsonl_path = out_dir / "sector_stream.jsonl"
        replace_regular_bytes(jsonl_path, payload)

    sources.verify_unchanged()
    return AdmittedSectorStream._from_admitted_sources(
        sources=sources,
        records=all_records,
        provenance=all_provenance,
        admitted_records=admitted,
        typed_exclusions=typed_exclusions,
        schema_admits_all=schema_admits_all,
        schema_first_failure=schema_first_failure,
        jsonl_path=jsonl_path,
        jsonl_payload=payload,
    )


def consume_trace_dir(
    trace_dir: Path,
    *,
    write_jsonl: bool = True,
) -> AdmittedSectorStream:
    """Path-compatible wrapper around the admitted-source core."""
    admission = require_trace_admission(trace_dir)
    with open_admitted_trace_sources(
        trace_dir, expected_admission=admission
    ) as sources:
        return consume_admitted_trace_sources(
            sources=sources,
            output_root=trace_dir,
            write_jsonl=write_jsonl,
        )


__all__ = [
    "AdmittedSectorStream",
    "consume_admitted_trace_sources",
    "consume_trace_dir",
]


def main(argv: list[str] | None = None) -> int:
    """CLI surface: consume one trace directory and print the
    counts dict (one JSON object) to stdout. Exit code is 0 on
    success, 1 on contract failure or schema-rejection of any
    emitted record, 2 on usage error.
    """
    parser = argparse.ArgumentParser(
        prog="python tools/sector_stream/consume_trace.py",
        description=(
            "Consume one simbench NVBit trace directory through "
            "the sector-stream generator and write JSONL to "
            "<trace_dir>/sector_stream/sector_stream.jsonl. "
            "Validates the wire contract first via "
            "tools/nvbit_tracer/import_trace_contract.py."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        required=True,
        type=Path,
        help=(
            "Path to the simbench trace directory containing "
            "threadblocks/**/*.pb (e.g., a directory produced "
            "by run_suite.py --mode trace)."
        ),
    )
    parser.add_argument(
        "--no-write-jsonl",
        action="store_true",
        help=(
            "Skip writing the JSONL output (useful for "
            "validation-only consumption)."
        ),
    )
    args = parser.parse_args(argv)

    result = consume_trace_dir(
        args.trace_dir,
        write_jsonl=not args.no_write_jsonl,
    )

    summary = {
        "contract_ok": result.contract_ok,
        "contract_typed_reason": result.contract_typed_reason,
        "contract_detail": result.contract_detail,
        "total_records": result.total_records,
        "admitted_records": result.admitted_records,
        "typed_exclusions": dict(result.typed_exclusions),
        "schema_admits_all": result.schema_admits_all,
        "schema_first_failure": result.schema_first_failure,
        "jsonl_path": (
            str(result.jsonl_path) if result.jsonl_path is not None else None
        ),
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not result.contract_ok:
        return 1
    if not result.schema_admits_all:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
