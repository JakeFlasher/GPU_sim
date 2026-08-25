# Completed milestone: compact direct trace-payload lifetime

**Status:** complete 2026-08-20; compact direct loader retained  
**Predecessor:** [same-cycle event batching](simulator_same_cycle_event_batch_milestone.md)  
**Evidence:**
`results/captures/simulator_event_throughput_20260819_v1/trace_lifetime.md`

## Goal and boundary

After event batching, the gap-heavy trace still reached about 571 MiB peak RSS.
Direct trace admission held both immutable raw threadblock protobuf bytes and
the compact validated representation throughout replay. Raw bytes are useful to
diagnostic callers that intentionally decode the original protobuf again, but
direct `sim-run` replays only the compact view.

`TraceDir::open_direct_compact` now executes the existing complete direct loader:
descriptor-anchored reads, resource bounds, wire preflight, protobuf decode,
allocation joins, launch/block/warp/instruction semantics, and compaction. Only
after that succeeds does it release the raw threadblock byte map. The canonical
path roster and every `ValidatedThreadblock` remain available.

Existing `TraceDir::open_direct`, campaign `TraceDirSnapshot`, and raw decoding
APIs retain their bytes and behavior. Calling `read_threadblock` on a known path
from the compact loader returns a specific `RawThreadblockPayloadUnavailable`
error; an unknown path still returns the established snapshot-missing error.
Thus the optimization does not silently reopen mutable paths or weaken
campaign snapshot semantics.

`validated_threadblocks_in_launch_order` now provides an explicit compact-safe
ordered API. The raw `read_threadblocks_in_launch_order` documentation names
its raw-retaining-loader requirement instead of directing compact callers to an
operation that must fail.

## Matched result

One feature binary alternated raw-retaining and compact conditions on pinned CPU
8, with one warm-up and seven measured runs per condition. Every pair had to
emit byte-identical normal JSON.

| case | raw peak | compact peak | RSS change | time change |
|---|---:|---:|---:|---:|
| RED gap64 | 571.2 MiB | 402.6 MiB | -29.5% | -0.9% |
| ordinary RED d128 | 397.2 MiB | 304.3 MiB | -23.4% | -0.3% |
| ordinary STORE d128 | 391.1 MiB | 299.8 MiB | -23.3% | +0.1% |
| ATOMG d128 | 346.8 MiB | 250.4 MiB | -27.8% | +0.2% |
| schema 13 | 332.0 MiB | 239.3 MiB | -27.9% | -0.7% |
| arrival STORE | 89.4 MiB | 67.4 MiB | -24.6% | -0.7% |
| retirement LOAD | 41.0 MiB | 33.9 MiB | -17.3% | +0.4% |

Every cohort case reduces peak RSS by 5.5–29.5%. Maximum paired-median timing
regression is 0.35%, far below the 3% bound.

## Validation and successor

Focused tests compare raw and compact validation summaries, payload contracts,
kernel/path rosters, launch-ordered compact block contents, and typed raw-access
errors. The harness now rejects a non-feature binary before measurement,
retains independently decodable canonical output bytes, binds feedback case
labels to schemas 10/11/13 and candidate IDs 12/202/214, and uses adjacent
paired changes. The complete Rust workspace, normal and feature suites,
campaign-enabled trace suite, formatting, and 60 affected Python tests pass.
All 5,519 active replay pairs were regenerated against the final compact-loader
binary and pass freshness, nested schema/configuration identity,
deterministic-twin, byte, and no-extra-output checks.

This is lifetime reduction, not sampling: every payload is still read, decoded,
and validated. The successor built the descriptor-rostered one-file path but
found only a 4.6% gap-heavy reduction, then localized and reduced the dominant
validated-instruction storage instead. See
[compact validated identity](simulator_compact_identity_milestone.md).

No sudo or GPU recapture was required.
