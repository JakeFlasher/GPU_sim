"""Sector-stream live-trace consumer (C-AC-F live-evidence path).

Walks a simbench NVBit trace directory, reconstructs each
per-warp memory access record, runs it through
:func:`tools.sector_stream.generator.compute_sector_stream`, and
writes one JSONL line per record to
``<trace_dir>/sector_stream/sector_stream.jsonl``.

The consumer is the offline-checkable path that closes the
C-AC-F live-evidence requirement: a sm_120 simbench trace flows
end-to-end through the generator and produces schema-valid
output. The wire-format contract (``threadblocks/**/*.pb`` plus
``extra_info/allocations.json``) is validated by the existing
:mod:`tools.nvbit_tracer.import_trace_contract` validator before
this module walks the records, so a malformed trace is caught at
its surface rather than producing nonsense sector-stream output.

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
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_TOOLS_NVBIT = _REPO_ROOT / "tools" / "nvbit_tracer"
_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "sector_stream.schema.json"
)


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
    if str(_TOOLS_NVBIT) not in sys.path:
        sys.path.insert(0, str(_TOOLS_NVBIT))
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import import_trace_contract  # type: ignore[import-not-found]
    import generator  # type: ignore[import-not-found]

    _, instruction_pb2, _ = (
        import_trace_contract._import_pb_modules()
    )
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

    lane_vector, recon_reason = (
        import_trace_contract._reconstruct_lane_vector(addr, and_mask)
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
        active_mask=inst.active_mask,
        predicate_mask=inst.predicate_mask,
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


def consume_trace_dir(
    trace_dir: Path,
    *,
    write_jsonl: bool = True,
) -> dict[str, Any]:
    """Consume one simbench trace directory through the
    sector-stream generator.

    Layout assumed: ``trace_dir/threadblocks/**/*.pb`` plus
    optional ``trace_dir/extra_info/allocations.json``. The
    function:

    1. Validates the wire contract via
       :func:`tools.nvbit_tracer.import_trace_contract.validate_trace_dir`.
       If the trace fails the contract, returns immediately with
       ``contract_ok=False``; downstream sector-stream output is
       not produced.
    2. Walks every threadblock and emits one JSONL line per
       address record (admitted or typed exclusion).
    3. Validates every emitted line against the sector-stream
       JSON Schema.

    Returns a dict with:

    * ``contract_ok``: bool — overall AC4 contract result.
    * ``contract_typed_reason``: str | None — the typed-reason
      identifier if the contract failed.
    * ``contract_detail``: str | None — the failing record path.
    * ``total_records``: int — number of sector-stream records
      produced (admitted + typed exclusions).
    * ``admitted_records``: int — number with
      ``result_kind=sector_stream``.
    * ``typed_exclusions``: dict[str, int] — counts keyed by
      typed_reason.
    * ``schema_admits_all``: bool.
    * ``schema_first_failure``: str | None — schema-rejection
      reason from the first failing record.
    * ``jsonl_path``: Path | None — written file path
      (``None`` if ``write_jsonl=False``).
    * ``records``: list[dict] — strictly schema-conformant
      record dicts (the same content the JSONL artifact
      contains). Returned in-process for callers that want to
      inspect output without re-reading the file.
    * ``provenance``: list[dict] — per-record debug provenance
      (``label``, ``warp_index``, ``inst_index``, ``vpc``)
      aligned with ``records`` by index. NOT written to disk;
      only available in-process. The R7 split keeps the on-disk
      JSONL strictly schema-conformant while preserving the
      debug pathway for callers who want to map a record back
      to its trace location.
    """
    import_trace_contract, generator, instruction_pb2 = (
        _import_helpers()
    )

    contract_result = import_trace_contract.validate_trace_dir(
        trace_dir
    )
    if not contract_result.ok:
        return {
            "contract_ok": False,
            "contract_typed_reason": contract_result.typed_reason,
            "contract_detail": contract_result.detail,
            "total_records": 0,
            "admitted_records": 0,
            "typed_exclusions": {},
            "schema_admits_all": False,
            "schema_first_failure": None,
            "jsonl_path": None,
            "records": [],
            "provenance": [],
        }

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    threadblock_pb2, _, _ = (
        import_trace_contract._import_pb_modules()
    )

    pb_paths = sorted(
        (trace_dir / "threadblocks").rglob("*.pb")
    )
    all_records: list[dict[str, Any]] = []
    all_provenance: list[dict[str, Any]] = []
    for pb_path in pb_paths:
        tb = threadblock_pb2.threadblock()
        tb.ParseFromString(pb_path.read_bytes())
        records, provenance = _consume_threadblock(
            tb,
            label=str(pb_path.relative_to(trace_dir)),
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

    jsonl_path: Path | None = None
    if write_jsonl:
        out_dir = trace_dir / "sector_stream"
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "sector_stream.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for rec in all_records:
                fh.write(
                    json.dumps(rec, sort_keys=True) + "\n"
                )

    return {
        "contract_ok": True,
        "contract_typed_reason": None,
        "contract_detail": None,
        "total_records": len(all_records),
        "admitted_records": admitted,
        "typed_exclusions": typed_exclusions,
        "schema_admits_all": schema_admits_all,
        "schema_first_failure": schema_first_failure,
        "jsonl_path": jsonl_path,
        "records": all_records,
        "provenance": all_provenance,
    }


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

    # Print only the count/status fields on the CLI; the
    # `records` and `provenance` fields are large and only
    # useful in-process. Callers that want them should use the
    # library API.
    summary_keys = {
        "contract_ok",
        "contract_typed_reason",
        "contract_detail",
        "total_records",
        "admitted_records",
        "typed_exclusions",
        "schema_admits_all",
        "schema_first_failure",
        "jsonl_path",
    }
    sys.stdout.write(
        json.dumps(
            {k: (str(v) if isinstance(v, Path) else v)
             for k, v in result.items() if k in summary_keys},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if not result["contract_ok"]:
        return 1
    if not result["schema_admits_all"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
