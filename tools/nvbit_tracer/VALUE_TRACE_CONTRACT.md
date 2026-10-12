# Finite value-dependent execution-trace contract

Alpha invocations are unbound by default. Use `--bound` only for the
historical manifest promotion form; no hash is emitted by the default
development result.

`register_value_adapter.py --dynamic-index` emits
`nvbit_value_trace_capture_v1` only when every text-capture sequence joins to
an exact sidecar record:

```json
{
  "sequence": 0,
  "function_unique_id": 7,
  "pc": 32,
  "opcode": "IADD3 R0, R1, 1"
}
```

The result is a **dynamic value-capture** artifact.  It is not a new cubin,
not static SASS regeneration, and not a timing input.  A complete SM86 value
trace still needs, for every admitted dynamic record:

- binary/toolchain/function/PC identity and active/predicate masks;
- source and destination register/predicate values;
- effective addresses and byte masks, load-return bytes, and store payloads;
- atomic old/new values and fence scope/order/read-from identity;
- initial memory bytes, overflow/drop counters, and output-byte checksums.

The current bounded producer supplies only lane register snapshots. Missing
fields remain explicit refusals rather than defaults. The current finite
scalar slice is implemented by `sm86_value_trace.py` for lane ID,
immediate/register moves, `IADD3`, `ISETP` predicates, predicated arithmetic,
`LOP3.LUT` with three GPR/immediate/constant inputs, an 8-bit truth table, and
the SM86 data-only or predicate-plus-register destination form with `!PT`,
bounded `SHF.L/R.U32` funnel shifts, and the observed immediate-multiplier
`IMAD.SHL.U32` plus multiplier-1 `IMAD.IADD` forms,
initialized `LDG.E`, `STG.E`, numeric `BRA`, `NOP`, and explicit `EXIT`; input-varying lanes,
addresses, and predicate paths produce different dynamic values. The next native slice is one
CTA/warp, a pinned SM86 cubin, and a larger `IADD3`/`ISETP`/`BRA`/32-bit
`LDG`/`STG` subset differentially checked against an independent scalar
interpreter. `PLOP3.LUT`, BoolOp/predicate-input variants, and dynamic
uniform/predicate sources remain refused. Calls, special/uniform registers, indirect branches, async/TMA/
tensor instructions, races, and atomics without old/new capture are outside
that admission contract.

The interpreter now refuses `register_value_missing` when an ordinary GPR is
read before a captured or explicitly supplied value exists. Unknown register
state is never defaulted to zero. Use `sm86_value_trace.py --registers` for an
explicit initial register map when a captured trace requires one.
