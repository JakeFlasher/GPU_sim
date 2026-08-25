# Direct-replay trace snapshot lifetime milestone

**Completed:** 2026-08-20  
**Predecessor:** [same-cycle event batching](../docs/simulator_same_cycle_event_batch_milestone.md)  
**Goal:** reduce direct `sim-run` trace-load peak memory while preserving complete
semantic validation, deterministic block order, and typed error precedence

## Measured target

After queue batching, the gap-heavy RED case still reaches about 571 MiB peak
RSS. Its retained trace is about 179 MiB and feature phase timing attributes
roughly half of runtime to `TraceDir::open_direct`. Deep ordinary STORE/RED also
spend about 40% of runtime in trace loading.

`TraceDir::open_direct` currently reads every threadblock protobuf into a raw
`BTreeMap`, then validates/compacts the complete tree into a second map. Raw
bytes remain resident because public diagnostic APIs can decode the original
protobuf again. Campaign `TraceDirSnapshot` requires immutable raw bytes and is
out of scope; direct replay needs only the compact validated view.

## Candidate design

1. Add a distinct direct-replay compact loader rather than weakening
   `TraceDir::open`, `TraceDirSnapshot`, or existing raw-payload APIs.
2. Descriptor-enumerate the complete canonical payload roster and resource
   bounds first, preserving current root/tree/path/count/size error precedence.
3. Read, wire-preflight, decode, semantically validate, and compact one
   threadblock file at a time in deterministic path order; release its raw bytes
   before reading the next file.
4. Retain compact blocks plus the validated path roster. APIs that require raw
   protobuf bytes remain available only on raw-retaining loaders and fail with a
   specific typed capability error on the compact loader.
5. Use the compact loader only in direct `sim-run` after complete parity tests;
   historical campaign consumers retain their snapshot path unchanged.

If descriptor APIs cannot expose a stable bounded roster without reading the
tree, first factor that enumeration primitive. Do not reopen files by ambient
path after validation and do not trade memory for TOCTOU ambiguity.

## Acceptance

- Synthetic differential tests cover every existing trace/root/payload error
  family and assert identical variants plus responsible paths/indices.
- Compact and raw-retaining direct loaders expose identical launch identities,
  validation summaries, allocation joins, compact threadblocks, and replay
  outputs on all retained cohorts.
- Peak RSS materially falls on `admission_red_gap64`; deep STORE/RED and arrival
  cases also report memory and load-phase changes.
- Median wall time does not regress more than 3% on any 12-case cohort arm.
- Normal JSON is byte-identical and all 5,519 active replay pairs validate after
  the final default build.

This is lifetime reduction, not sampling: every payload remains validated and
every traced record remains replayed.

## Outcome

Direct `sim-run` now releases raw threadblock protobuf bytes after the unchanged
complete validation/compaction pass. Raw-retaining and campaign loaders remain
unchanged. Peak RSS falls 5.5–29.5% across all 12 cases, including 571.2→402.6
MiB on gap64, with only 0.35% maximum timing regression and byte-identical
outputs. See the
[completed milestone](../docs/simulator_compact_trace_lifetime_milestone.md).
Closure includes the full Rust/feature/campaign trace suites, 60 affected Python
tests, and 5,519 fresh byte-identical replay pairs.
