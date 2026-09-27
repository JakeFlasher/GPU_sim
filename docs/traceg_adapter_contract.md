# Enhanced protobuf → legacy `.traceg` adapter contract

Updated 2026-09-27.  This is a bounded representation probe, not a native
simulator input path and not a timing or silicon-validation result.  The
implementation is `tools/traceg_adapter_probe/adapter.py` and emits schema
`sm86_traceg_adapter_probe_v2` with `scoring: false`.

## Admitted projection

The adapter fails closed unless all of the following are true:

- The dynamic root has exactly one device (device id `0`), one stream, and one
  kernel.  The launch is SM86 (`binary_version = 86`), grid `(1,1,1)`, block
  `(32,1,1)`, and a non-cluster launch (`cluster_dim` absent/zero or
  `(1,1,1)`).  Positive kernel/function IDs, explicit register count, and
  explicit static and dynamic shared-memory components are required.  Input
  files are bounded before protobuf/JSON decoding.
- The threadblock is block `(0,0,0)` with exactly warp `0`, at most 64 dynamic
  records, and no empty payload.  Every dynamic record has explicit active and
  predicate masks, an effective full-warp mask, warp-0 identity, a function
  ID matching the launch, and a present 64-bit `vpc` whose low 32 bits equal
  `pc`.
- The official SM86 control-bits JSON has an `arch` of `sm_86` (or its exact
  aliases), a non-empty `functions` map, and exactly one decoded static record
  per function-relative PC.  `opcode_text` is authoritative; the dynamic
  protobuf's normalized opcode fields are never consulted.  Static records
  must be unpredicated and use only `IADD3` or `FADD`, with at most one regular
  destination and four regular sources.  Integer `IADD3` literals are bounded
  signed 64-bit values.  `RZ` is deliberately canonicalized to legacy `R255`.
- A static `EXIT` may be present only as the final dynamic record.  Its static
  text must be exactly `EXIT`, its effective mask is full or zero, and it has
  no address or memory metadata.  A capture ending at protobuf EOF is
  `termination_incomplete`; the adapter never appends an inferred or synthetic
  instruction.  The optional enhanced-execution-info JSON is an identity
  cross-check (`function_unique_id`, function name, and PC set), not an opcode
  source.

Memory, branches, barriers, predicates, special/uniform registers, atomics,
texture/surface/TMA/async operations, and all other opcodes are outside this
contract.  Even a memory enum value that appears to identify global memory is
not interpreted as an address projection; any address vector or memory field on
an admitted ALU/`EXIT` record is a typed refusal.

## Output and identity

A successful run creates a new output directory (existing directories are
refused, so prior output cannot be overwritten) containing:

- `kernel-<id>.traceg`, one canonical block/warp, full-mask ALU records, and an
  explicit terminal `EXIT` copied from the dynamic record;
- `kernelslist.g`, naming that trace; and
- `adapter_result.json`, which records `status: "ok"`, `scoring: false`, the
  dynamic-exit termination kind, canonical records, and SHA-256 digests of the
  dynamic protobuf, threadblock protobuf, control sidecar, optional identity
  sidecar, traceg bytes, and kernels list.

The digest manifest binds the files consumed by this invocation.  It does not
turn a self-authored fixture into independent hardware evidence and does not
assert that legacy scheduling or completion semantics equal native SM86.

## Refusal vocabulary

`AdapterRefusal` is a normal (non-frozen) exception carrying a stable
`RefusalCode`, detail, and optional dynamic record index.  Important classes
include:

| Condition | Code |
| --- | --- |
| Missing pinned protobuf runtime, malformed root/threadblock, or input limit | `protobuf_runtime_missing`, `root_decode_failed`, `threadblock_decode_failed`, `input_limit_exceeded` |
| Wrong launch cardinality/SM/geometry/metadata | `root_cardinality_not_one`, `root_not_sm86`, `launch_shape_unsupported`, `launch_metadata_missing` |
| Missing/mismatched masks, PC, warp or function join | `mask_metadata_missing`, `mask_not_full_warp`, `vpc_missing`, `vpc_pc_mismatch`, `function_join_mismatch` |
| Missing/duplicate/malformed control records or unsupported static text | `sidecar_*`, `unsupported_opcode`, `predicated_static_instruction`, `unsupported_operand`, `register_*` |
| Memory or address information that cannot be represented | `memory_projection_unsupported`, `memory_metadata_invalid`, `address_*` |
| Missing, non-final, or malformed dynamic termination | `termination_incomplete`, `termination_malformed` |
| Malformed emitted grammar or attempted overwrite | `traceg_parse_failed`, `roundtrip_mismatch`, `output_exists`, `output_write_failed` |

## Parser and evidence boundary

`parse_traceg` is a strict structural reader for the tiny grammar emitted here:
it checks one closed TB, required metadata, bounded counts, full masks,
register tokens, explicit terminal `EXIT`, count equality, and trailing data.  It
is in the same module as the emitter, so this round trip is implementation
(equivalence) evidence, **not an independent Accel-Sim/MGPUSim parser result**.
No external simulator or GPU is invoked by this probe.  A future external
reader check must be reported separately and must not be promoted to an NCU or
elapsed-time score.

## Reproduction and tests

From the repository root:

```text
python -m tools.traceg_adapter_probe.adapter --help
pytest -q tools/traceg_adapter_probe/test_adapter.py
uv run --with protobuf==7.34.1 pytest -q tools/traceg_adapter_probe/test_adapter.py
```

The first test command runs grammar/refusal tests and skips protobuf cases when
the pinned runtime is absent.  The pinned-runtime command exercises a real
protobuf bundle derived from
`sim/run/tests/fixtures/addr_carry_canary_bundle` in a temporary directory,
adds an explicit dynamic `EXIT` for a positive three-record projection, and
checks that the untouched canary refuses at its first `LDG.E` with
`memory_projection_unsupported`.  It also checks EOF/non-final termination
refusal, control-join mismatch, duplicate static PC,
malformed/trailing/count/header input, memory metadata, and output overwrite
refusal.  These are local contract tests only; they make no claim of
native completion, external timing, or silicon accuracy.
