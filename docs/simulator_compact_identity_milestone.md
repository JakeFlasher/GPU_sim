# Completed milestone: compact validated-instruction identity

**Status:** complete 2026-08-20; 20-byte identity layout retained  
**Predecessor:** [compact direct trace lifetime](simulator_compact_trace_lifetime_milestone.md)  
**Evidence:**
`results/captures/simulator_event_throughput_20260819_v1/compact_identity.md`

## Localization and design

The compact-loader successor first implemented a complete bounded
descriptor-relative payload roster. Each `.pb` leaf is opened and bound to its
path, size, inode, and generation before root semantic decoding. A later
one-file read goes through the retained root descriptor, requires the exact
rostered size and generation, and detects mutation during the read. Raw and
campaign snapshot loaders are unchanged.

The three-round pilot showed that streaming alone is not the dominant-case
answer. It cuts the small launch-chain arm substantially but reduces gap-heavy
peak RSS only 4.6%, below the 10% materiality criterion. Streaming therefore
remains an opt-in performance-build research path and is not the default.

The pilot instead localized the peak to the validated record array. The gap
trace contains 6,804,000 instructions. Each retained 32 bytes even though two
fields are validation-proven invariants:

- `function_unique_id` equals the root launch identity; and
- `warpid_tb` equals the enclosing protobuf warp map key.

Those values now live once per `ValidatedWarp`. Per-record `sm_id` and
`warpid_sm` remain exact i32 values, with an explicit presence byte rather than
a sentinel. This preserves every signed value and `None` while reducing the
record from 32 to exactly 20 bytes. Replay accessors and error checks retain the
same values.

## Matched result

Two ordinary release binaries alternated baseline-32 and compact-20 conditions
on pinned CPU 8. Every case received one warm-up and seven paired repetitions;
both binary identities remained stable during capture and every normal JSON
result matched retained canonical bytes.

| case | 32-byte MiB | 20-byte MiB | RSS change | time change |
|---|---:|---:|---:|---:|
| RED gap64 | 403.6 | 329.4 | -18.4% | -0.5% |
| ordinary RED d128 | 304.2 | 259.5 | -14.7% | +0.8% |
| ordinary STORE d128 | 299.7 | 255.6 | -14.7% | -2.5% |
| ATOMG d128 | 250.4 | 203.8 | -18.6% | -0.1% |
| schema 10/11/13 | 239.2–239.4 | 194.6–194.7 | -18.7% | +0.4–0.8% |
| admission control | 205.6 | 159.5 | -22.4% | -0.8% |
| arrival STORE | 67.2 | 57.7 | -14.0% | -0.7% |
| launch chain | 45.3 | 37.6 | -17.2% | +1.4% |

All 12 arms reduce RSS by 7.8–22.4%. The maximum paired timing regression is
1.98%, inside the 3% acceptance bound. The representation is retained.

## Validation

Focused coverage pins exact 20-byte size, `None` versus all signed i32 values,
raw/compact/streaming block parity, roster hard-link rejection, generation
replacement rejection, and exact CLI JSON. The full Rust workspace, explicit
performance/campaign feature suites, formatting, Python tests and compilation,
all predecessor/successor reducers, and the active replay validator pass.

The final default release regenerated exactly 5,519 active replay pairs. Every
pair is fresh, nested-schema/configuration matched, byte-identical, and the
active roots contain no surplus replay directory. No sudo or new GPU/NCU/NVBit
capture was required.

