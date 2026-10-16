# Simulator status — SM120 alpha

Updated 2026-10-16. The Rust simulator and Python replay tools build from the
`sim/` workspace. The complete real corpus is 339 roots / 11,558 launches per
mode; full replay precedes fast and complete-corpus runs use exactly 12 workers.

Implemented source behavior separates TMA submissions from legacy async-copy
commit state, routes LDGSTS explicitly, preserves barrier scope/precision
uncertainty, bounds diagnostics, and fails closed on unsupported directions or
malformed input. `completion_clock` is intentionally unavailable. Effective L1
and L2 scenarios remain explicit conditional configurations; no physical cache,
MSHR, scheduler, or engine claim is made.

Replay and post-replay generated JSONL use `.jsonl.zst` through
`tools/jsonl_io.py`. Raw capture sidecars and workload manifests are input data
and remain outside that generated-artifact rule. See the proposal `PLAN.md` and
compact reports for current evidence; superseded engineering history is in the
local adversarial dossier.

The current effective-model slice also emits model-relative SM/SMSP IPC rows
(`01307b1c`, refined by `6a634f84`). The complete 339-root full/fast replay
completed after the change; IPC is an effective projection and does not claim
physical issue width or a clock crosswalk.

Q75 is the current active-IPC projection. Its pure-TMA arm applies only to
multi-transfer, payload-bearing, zero-LDGSTS launches outside the two-SM scope;
it uses an 8/5 effective lifetime arm and leaves all other launches on Q74.
The complete Q75 replay passed 339/339 roots and 11,558 launches per mode with
full integrated L1+L2 first, fast flat second, exactly 12 workers, and compressed
artifacts. Full active-IPC mean APE is 9.5537% (median 6.2304%, P90 17.0335%),
with unchanged maximum/family maxima and no paired-launch regression.

Q76 L1/L2 diagnosis (2026-10-16) tested stream-L2 lifetime and a 4096-line
effective-L1 capacity on four real captures; neither was promoted. Stream-L2
worsened the selected unified L1 hit-rate residual and the larger capacity was
non-discriminating. Remaining hit-rate residuals are therefore still typed as
scope/client/join uncertainty rather than a justified geometry constant.

## Q79 — partial-TMA source-TEX projection (2026-10-16)

Commit `05442f3e` adds a narrowly guarded observation-side projection for
partial TMA address coverage. Complete address coverage keeps the existing
shadow. For complete, aligned payload summaries with incomplete mixed
addresses, the projection admits only a per-direction load shadow whose
request cardinality matches payload-derived sectors and counts payload-bearing
partial stores as cold/write misses. It never imports aliased partial-store
shadow hits; missing/malformed payload summaries keep the ordinary fallback.
Replay cache state, timing, DRAM demand, IPC, and `completion_clock` are
unchanged.

The exact four-capture gate completed 243 launches per mode and the five-capture
probe 246 launches per mode, full integrated L1+L2 first and fast flat second,
with 12 workers and all rows `ok`. The complete run is
`replay_results/q79_partial_tma_source_full_20261016/`: 339/339 roots and
11,558/11,558 launches per mode, all `ok`, compressed artifacts.

Against Q75 on identical full-corpus joins, source-TEX hit mean APE improved
60.1535% → 59.9332% (P90/max unchanged) and source-TEX miss mean APE improved
2.57315% → 2.30313% (P90 3.03030% → 2.80561%, maximum unchanged). There were
zero worsened rows and every family maximum was unchanged or lower; all other
scored metrics were bit-for-bit unchanged. The quant_005 512-sector read-hit
gap remains unavailable because descriptor/OOB padding is not cache-hit evidence.

## Q80/Q81/Q82 follow-up (2026-10-16)

The proposed TMA-shadow write-tag reinterpretation and zero-byte coverage fix
were tested on real complete/partial captures and rejected: complete source-TEX
hit/miss rows worsened, while the zero-byte-only arm made quant_005 source
misses 1,809 versus 2,580 NCU misses. The proposed TMA-L2-miss-to-DRAM-read
projection was evaluated offline and worsened 30 launches, so neither change
was landed or replayed on the full corpus.

Commit `6c1b5a05` instead lands quarantined typed sidecar provenance: transfer
product checks, tensor-map summary, coordinates, raw valid/OOB references,
unique sectors/lines, duplicate-sector counts, and nominal-box overage. A
five-capture, 246-launch/mode, 12-worker full/fast probe passed with every
scored estimate bit-for-bit identical to Q79. These fields are diagnostics only;
`completion_clock` and physical topology remain unavailable.

The proposed shared-memory-carveout L1 refinement was also stopped before
replay: in the existing Q79 artifact, 816 of 830 positive-target global-load
L1 rows with nonzero launch shared memory already have simulator hits at or
below NCU hits, so reducing effective capacity cannot satisfy the tail-safe
sign condition.

## Q83 — TMA reuse decomposition and clock-sandwich diagnostics (2026-10-16)

Commit `408381b1` adds launch-level decomposition of per-record unique sectors
and lines into inter-record reuse/shared-line packing versus intra-record
address fanout, plus read-only hardware-clock sandwich span sums when paired
producer/query clocks exist. A five-capture, 246-launch/mode probe passed with
all scored estimates bit-for-bit identical to Q82. Quant_005 launch 1 has zero
inter-record duplicate sectors and shared lines, 54,784 intra-record duplicate
references, and zero usable clock pairs; these are evidence-coverage facts, not
cache hits, timing constants, or completion timestamps.

## Q84 — join/client-scope provenance tags (2026-10-16)

Report commit `84200733` adds metadata-only launch association tiers and metric
client-scope classes to post-replay comparison rows. The Q83 five-capture report
was regenerated with the tags; scored estimates, joins, row counts, and APE
summaries remain unchanged. Tags never filter workloads or alter replay.

## Q85 — directional TMA reuse diagnostics (2026-10-16)

Commit `5dc6827a` partitions Q83 reuse provenance into load-load,
store-store, and cross-direction sector/line overlap. The five-capture probe
passed with all scored estimates bit-for-bit identical to Q84. Quant_005 launch
1 has load 1,536 sectors/384 lines, store 1,024 sectors/256 lines, and zero
load-load, store-store, or cross-direction inter-record overlap. These fields
remain structural diagnostics only.
