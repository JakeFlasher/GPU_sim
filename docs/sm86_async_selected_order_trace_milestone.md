# SM86 selected asynchronous-copy instrumentation-order milestone

**Status:** complete 2026-08-31 as
`complete_selected_order_trace_fidelity`

## Outcome

The NVBit → protobuf → Rust reader → simulator path now preserves an opt-in,
timing-free order witness for selected asynchronous-copy issue, commit, and
wait instructions. The witness classifies every forced queued/drained epoch in
both the W4 identification smoke and the untouched W8 holdout. Readiness schema
21 consumes it only as an observation; removing the diagnostic and normalizing
the schema reproduces the same tag-14-stripped schema-20 replay byte-for-byte.

No simulator scheduling, memory request, completion frontier, queue parameter,
architecture card, or typed-tail state changes. This milestone repairs trace
fidelity; it does not promote timing.

## Producer contract

`dynamic_trace.instruction` adds optional uint64 tag 14,
`instrumentation_order`. It is enabled only with:

```text
SIMBENCH_NVBIT_ASYNC_ORDER=1
```

The static producer selects exactly:

- `LDGSTS*` issue anchors;
- `LDGDEPBAR` commit anchors; and
- `DEPBAR.LE SB0,*` wait anchors.

Ordinary SB1--SB5 waits remain unnumbered. The accepted launch's existing main
channel receiver numbers selected protobuf instruction creations densely from
one. A multi-MREF LDGSTS receives one token because only the first operand
record creates the instruction. No device instruction, atomic, clock read, or
additional channel push was added.

The token means callback/channel receive order. It is neither a hardware cycle
nor source, destination-service, or completion time.

Python and Rust wire preflights enforce the tag's varint wire type and reject
duplicates before protobuf allocation. Readers reject present-zero tokens,
missing same-record SM/warp identity, non-increasing warp-local order,
duplicate/global non-dense streams, missing selected records, or tokens on
non-selected instructions.

The rebuilt tracer is 16,039,392 bytes with SHA-256:

```text
0989e5ec62625c5143dfd2e134eadbbcd69ce65ec1c43c305c0ffd5f63aebf75
```

Its exact 55-DSO runtime manifest is retained and validated.

## Observe-only schema 21

Selected records remain in a cold sparse warp sidecar. After extraction and
before result publication—but never during modeled scheduling—the simulator
sorts tokens and reconstructs only the async group state needed for diagnostic
counts:

- selected, issue, commit, and wait events;
- issue events that observe committed groups owned by other warps;
- summed and maximum other-warp committed-group population; and
- protocol contradictions.

The hot compact instruction remains 20 bytes. `SmCore` never reads the order
sidecar. Schema-21 state is attached after simulation, so timing and traffic
cannot depend on it.

Example W4 repeat-1 diagnostics show the structural separation that schema 20
could not recover:

| Cell | Issues with other-warp groups | Population sum | Maximum |
|---|---:|---:|---:|
| L1 queued | 1,680 | 11,016 | 12 |
| L1 drained | 1,167 | 4,905 | 8 |
| L2 queued | 1,682 | 11,032 | 12 |
| L2 drained | 1,179 | 4,895 | 8 |

Raw unconstrained interleavings may differ across repeats and are not used as
hash gates. Every stream has zero protocol contradictions.

## W4 identification smoke

Two independent captures of L1/L2 queued/drained produced eight complete W4
traces. Every endpoint contains exactly:

```text
2,048 LDGSTS issues
  832 commits
  256 waits
3,136 selected records
```

Tokens are exactly dense `1..3136` and strictly increasing within every warp.
For all 512 route/arm/repeat epochs:

- queued has all 12 background commits before the target's first LDGSTS and
  the target wait before all three background waits;
- drained has all three background waits before target issue; and
- the token-only classifier selects the correct arm.

The passing W4 authorization was retained before W8 was opened.

## W8 holdout

The holdout uses the same binary with target warp 7 and seven backgrounds.
Two independent L2 queued/drained repeats produced four traces. Every endpoint
contains exactly:

```text
4,096 LDGSTS issues
1,856 commits
  512 waits
6,464 selected records
```

All 256 holdout epochs classify correctly: queued has all 28 background
commits before target issue and target wait before all seven background waits;
drained has all seven waits before target issue. Tokens are dense `1..6464`
and per-warp ordered.

Across W4 and W8, the final evidence contains 12 order-enabled traces, 24
ordered replay outputs, and 12 tag-14-stripped baselines. Every ordered pair is
deterministic, and every normalized ordered replay equals its stripped
baseline exactly. Four predecessor NCU reports are reused because the CUDA
binary and exact work are unchanged.

## Default-disabled compatibility

A deterministic one-warp wait-prefix canary contains 384 dense selected
records when ordering is enabled and zero tag-14 fields when default-disabled.
Independent capture bytes are not compared because SM/warp placement is
captured state. A retained tag-14-absent schema-18 replay remains byte-identical
at:

```text
61338a8d446d7fd699d2462ff3817908736323bc294cfcedf94997f0427c555e
```

## Evidence identity

The canonical root is:

```text
results/captures/sm86_async_selected_order_trace_20260831_v1/
```

Salient identities are:

```text
report           47409ba78b40ac5d7200cff6914668505ec6e2a39b0c9b183cf5351b9f22e891
smoke auth       6c028c94d9c5fdbbfc60e65f89c3963ea0f5db702c730c0435cc67510c431dc6
tracer DSO       0989e5ec62625c5143dfd2e134eadbbcd69ce65ec1c43c305c0ffd5f63aebf75
runtime manifest 4b51eca80f403ee28dcda1ff7d71141f63dd5a119d59e0eca4625e0d2a17cd4a
sim-run          a7bb98d1cafca83b20388acd6ec3563c427e1ff032d52424681c7b88fcc99a92
probe binary     9944a80c075e823f914a9adb1416acc34be6bb30f6e7e1dab2d03849379d2dff
proto schema     f8204fb93df6c65f96762cc0ffbebf1f9a0cfd78265a844695be1ce2dc54e8ad
```

The prospective contract began at commit `312ec4b6`; its compatibility oracle
was clarified before final reduction. The exact producer, consumer, live
campaign tooling, and tracked decision are commit
`7496273138aaa8e81b8e953c097da02c39a73bb1`.

## Reproduction

The pinned protobuf runtime is required for parsing and stripping tag 14:

```bash
.venv/bin/python \
  workloads/research_probes/async_source_only/capture_selected_order.py \
  --phase smoke --timeout 1800

.venv/bin/python \
  workloads/research_probes/async_source_only/analyze_selected_order.py \
  --stage smoke

.venv/bin/python \
  workloads/research_probes/async_source_only/capture_selected_order.py \
  --phase holdout --timeout 1800

.venv/bin/python \
  workloads/research_probes/async_source_only/analyze_selected_order.py \
  --stage complete

.venv/bin/python \
  workloads/research_probes/async_source_only/analyze_selected_order.py \
  --stage complete --validate-only
```

## Scientific boundary and next action

The workload's shared handshakes force the causal relations validated here, so
the token is a valid fidelity witness for those relations. It is not an
independent silicon service-order measurement and token differences have no
duration meaning.

Retain tag 14 and schema 21. Do not use either to fit timing or force arbitrary
warp schedules. A later queue-mechanism milestone may use the witness to verify
that a synchronization-derived happens-before constraint is preserved while
evaluating an independently specified service mechanism; silicon target clocks
and NCU remain comparison-only.
