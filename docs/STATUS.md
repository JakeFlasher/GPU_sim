# Simulator status — SM86 / RTX 3090 Ti branch

**Branch override (2026-10-15):** this checkout is `3090ti_full_replay`,
targeting the local RTX 3090 Ti / GA102 / `sm_86` and corpus
`../sm86_rtx_3090_capture/`. The SM120 material below is retained historical
context and does not define this branch's active target or defaults. The active
implementation plan is `plans/3090ti_full_replay_modeling_plan.md`; SM120
behavior remains preserved unless a scoped SM86 refactor replaces a shared seam.

## Current SM86 implementation slice — 2026-10-15

The architecture-neutral sidecar adapter and SM86 corpus runner are operational.
The real three-root vertical completed full 3/3 and fast 3/3 successfully.
The complete-corpus full run discovered 356 roots and completed 286 roots
`ok`, with 62 explicit mixed-code control dispositions and 8 retained
replay/input failures; 12,392 launches were replayed in successful rows in
32.845 seconds using 12 workers. Full output is
`../sm86_replay_results/full_corpus_20261015_v2/full.jsonl.zst`.

This is runner/coverage evidence, not an NCU accuracy claim. Successful full
rows retain conditional dirty-writeback/coherence tails; metric joining and
mechanism accuracy refinement remain active.

## Current SM86 refinement slice — L2 write observer

The full-mode L2 observer now uses operation-qualified source lanes and a
conditional all-write-hit arm for ordinary stores, supported by independent
Ampere write-validate evidence. Physical L2 state, DRAM demand and timing are
unchanged. The post-replay full-corpus rerun completed 356 roots (286 `ok`, 62
mixed-code controls, 8 replay/input failures) and 12,392 successful launches in
33.625 seconds. Exact NCU identity comparison remained 107 rows. L2 lookup-hit
APE improved 54.52% → 35.04%; lookup-miss APE improved 84.20% → 27.00%.
Atomic/RED, local-memory, constant/GCC, DRAM and IPC residuals remain open.

## Current SM86 refinement slice — H2 constant-domain demand — 2026-10-15

The trace reader now retains field-15 active-lane bank-local CBANK offsets,
field-16 bank identity, and explicit field-17 metadata-only records. The
observer emits checked 32-B packet, 32-B offset-domain, 64-B candidate-line,
active-reference, and metadata-only counters. These are source-domain
observations only: they do not enter L1/L2 state, DRAM timing, warp readiness,
or IPC, and are not labeled as GCC requests or DRAM sectors.

Four representative real full replays completed 4/4. The corrected complete
full replay completed 356 roots (286 `ok`, 62 mixed-code controls, 8 retained
replay/input failures) and 12,392 successful launches in 34.263 s with 12
workers. Artifact:
`../sm86_replay_results/full_corpus_h2_constant_20261015_v2/full.jsonl.zst`.

All 107 exact-identity rows with constant observations retain active offset
vectors (2,845,702 references); 3,840 field-17-only metadata records are
reported separately. H2 leaves scored L2/DRAM/IPC residuals unchanged:
35.04% L2 hit APE, 27.00% L2 miss APE, 40.91% DRAM-read APE, 14.57% SM-active
IPC APE, and 12.41% elapsed-IPC APE. The compact report is
`docs/sm86_constant_demand_slice_20261015.md`.

H2 is therefore coverage and semantic protection, not sub-10% closure. GCC
constant requests remain a separate static `LDG.*.CONSTANT`/L1.5 source-class
problem. The next active arm is H3, a bounded SM86 fixed-latency register-read
and reuse-cache envelope for IPC accuracy; it must be transactional and typed
gap-aware, with cache/DRAM state untouched.

Updated 2026-10-15. The Rust simulator and Python replay tools build from the
`sim/` workspace. The active SM86 corpus is 356 roots (354 contract-clean plus
2 typed-invalid dispositions); full replay precedes fast and complete-corpus
runs use exactly 12 workers. The historical SM120 counts below do not override
the branch override above.

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

## Current SM86 refinement slice — H3 fixed register-read/RFC envelope

Full-mode SM86 now has a bounded per-SMSP effective register-read arm: two
regular banks, one effective read port per bank, a three-cycle reservation
window, and a deterministic three-entry-per-bank RFC queue keyed by traced
block/warp/register/operand position. Gap, not-applicable, and unsupported
vector shapes do not mutate RF/RFC state. The arm does not touch cache,
DRAM, or memory request ordering.

Five representative real roots completed full replay 5/5. The corrected
complete full replay completed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 successful launches, in 34.431 s with 12 workers:
`../sm86_replay_results/full_corpus_h3_rf_20261015_v2/full.jsonl.zst`.

The arm recorded 27,167,662 projected reservations, 3,622,126 RFC hits,
6,908,102 misses, and 133,572 typed unsupported-shape records across the
successful launches. Exact-identity cache/DRAM residuals are unchanged; IPC
mean APE moves slightly from 18.08% to 18.26% for `sm__cycles_active.avg` and
12.41% to 12.53% for elapsed IPC. H3 is retained as causal coverage but is not
claimed as an accuracy improvement. Compact report:
`docs/sm86_register_read_slice_20261015.md`.

The next arm is H4: persistent fetch-last-issued state separate from issue hold
across Yield, evaluated with H3 in the default full composition.

## Current SM86 refinement slice — H4 fetch-last-issued / Yield

H4 separates fetch/decode continuation from issue hold. Full-mode fetch uses a
persistent per-SMSP `fetch_last_issued` warp while its instruction buffer has
capacity; Yield still clears issue hold for the next issue cycle. Five real
roots completed 5/5, with observed continuations and Yield/hold separations.

The complete full-first corpus completed 356 roots (286 `ok`, 62 controls, 8
failures), 12,392 successful launches in 34.826 s with 12 workers:
`../sm86_replay_results/full_corpus_h4_fetch_yield_20261015_v1/full.jsonl.zst`.
H4 leaves timing/cache/DRAM predictions unchanged on the exact 107-row join:
DRAM 40.91%, `sm__cycles_active.avg` 18.26%, and elapsed IPC 12.53% mean
APE. It is retained as front-end semantic coverage, not an IPC accuracy gain.
Compact report: `docs/sm86_fetch_yield_slice_20261015.md`.

The next timing-active arm targets static `LDG.CONSTANT`/GCC source
classification or another independently evidenced IPC denominator seam; NCU
remains post-replay only.

## Current SM86 refinement slice — H5 static LDG.CONSTANT source class

H5 emits static-opcode-qualified `LDG.*.CONSTANT` global-load instruction,
sector, per-instruction 128-B line-candidate, and CTA/warp-scoped unique-line
counters. It is separate from H2 CBANK offsets and does not change cache/DRAM
state or request timing.

Representative real full replay completed 5/5. Complete full replay completed
356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 34.915 s
with 12 workers:
`../sm86_replay_results/full_corpus_h5_ldg_constant_20261015_v1/full.jsonl.zst`.
Successful rows emitted 6,807,596 source-class instructions and 7,142,629
request candidates. Exact-identity DRAM and IPC APE remain 40.91%, 18.26%, and
12.53%, unchanged from H4. H5 is source attribution, not an accuracy gain.
Compact report: `docs/sm86_ldg_constant_slice_20261015.md`.

The next arm targets an independently evidenced GCC/L1.5 response-scope
condition or bounded DRAM service relation; NCU remains post-replay only.

## Current SM86 refinement slice — H6 GCC/L1.5 scope candidate

H6 adds CTA-scoped static `LDG.*.CONSTANT` line candidates beside the existing
warp-qualified candidates. This makes replication scope explicit without
pretending either set is GCC request truth; no timing/cache/DRAM state changed.

Four real full roots completed 4/4 and the complete full-first corpus completed
356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 35.194 s
with 12 workers:
`../sm86_replay_results/full_corpus_h6_gcc_scope_20261015_v1/full.jsonl.zst`.
Exact DRAM/IPC mean APE remains 40.91%, 18.26%, and 12.53%; H6 is not an
accuracy gain. Compact report: `docs/sm86_gcc_scope_slice_20261015.md`.

A universal GCC scope/multiplier is rejected. Continue to the next bounded
DRAM service or IPC denominator arm.

## SM86 IPC denominator probe — rejected H7 hypothesis

A real four-root full probe tested substituting the SM-active observation
frontier for the GPC launch-envelope denominator. It increased elapsed-IPC
mean APE to 51.81% on the exact joins versus the retained 12.53% GPC endpoint,
so the change was reverted before complete-corpus promotion. Evidence is
retained in `docs/sm86_ipc_denominator_negative_20261015.md`; no NCU-derived
parameter was retained.

## Current SM86 refinement slice — H8 DRAM partition pressure

H8 adds observer-only GA102 DRAM partition/slice pressure candidates: line-ID
modulo-12, XOR-fold modulo-12, and modulo-6 L2-slice domains. The NVIDIA GA102
whitepaper supports the controller/slice counts, but physical address hashing
is undisclosed; no candidate affects the aggregate G/D/1 timing server.

Five representative real roots completed 5/5. Complete full replay completed
356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 35.409 s
with 12 workers:
`../sm86_replay_results/full_corpus_h8_dram_partition_20261015_v1/full.jsonl.zst`.
Exact DRAM/L1/L2/IPC residuals are unchanged (DRAM 40.91%, L2 hit/miss/source
35.04%/27.00%/28.24%, active/elapsed IPC 18.26%/12.53%). H8 is observability,
not a timing-active model. Compact report:
`docs/sm86_dram_partition_pressure_slice_20261015.md`.

## Current SM86 refinement slice — H9 near/far pressure uncertainty

H9 adds a typed near/far candidate interval over H8's modulo/XOR pressure maps;
no physical swizzle or latency is selected. Independent RTX 3090 Ti memtest
repeats support a future [810,850] GB/s forced-DRAM service-rate scenario, but
unlocked-clock variance prevents adopting it as default timing.

Five representative roots completed full replay 5/5. Complete full replay
completed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in
35.059 s with 12 workers:
`../sm86_replay_results/full_corpus_h9_near_far_20261015_v1/full.jsonl.zst`.
Exact DRAM/L1/L2/IPC residuals remain unchanged from H8. Compact report:
`docs/sm86_near_far_pressure_slice_20261015.md`.

Timing-active partition/row service remains deferred pending a locked local
3090 Ti mapping/service probe.

## Current SM86 refinement slice — H10 effective DRAM service rate

Independent repeated RTX 3090 Ti memtest runs support a forced-DRAM warm-rate
scenario near 825 GB/s (range ~810–850 GB/s under an unlocked protocol), while
the card vendor peak remains 1008 GB/s. The runner now defaults SM86 full
carded replay to 825,000,000,000 B/s and records the defaulted hypothesis;
explicit overrides remain available.

Four representative full roots completed the 825 GB/s scenario. Complete
full-first replay completed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches in 35.094 s with 12 workers:
`../sm86_replay_results/full_corpus_h10_dram_rate_825_20261015_v1/full.jsonl.zst`.
On 107 exact joins, GPC/SM timing mean APE moved 13.14→13.05%, 18.26→18.19%,
and SM active max 18.11→18.00%; DRAM traffic APE stayed 40.91%. Compact
report: `docs/sm86_dram_service_rate_slice_20261015.md`.

## Current SM86 refinement slice — H11 source-composed DRAM demand

H11 emits a typed candidate demand envelope combining L2-missing data,
compulsory instruction sectors, static LDG.CONSTANT line candidates, and H2
CBANK offset candidates. It does not claim achieved DRAM traffic or alter
service timing. Representative full replay contained 3/4 NCU DRAM observations
inside the candidate interval.

Complete full replay (with the default recorded 825 GB/s service scenario)
completed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in
35.846 s with 12 workers:
`../sm86_replay_results/full_corpus_h11_dram_demand_20261015_v1/full.jsonl.zst`.
DRAM mean APE remains 40.91%; no universal constant/GCC multiplier is
supported. Compact report: `docs/sm86_dram_demand_envelope_slice_20261015.md`.

## Current SM86 refinement slice — H12 atomic conflict exposure

H12 emits exact traced same-sector atomic alias exposure (conflict-bearing and
distinct instruction counts plus alias sectors) without imposing a latency or
DRAM multiplier. Four atomic real roots completed 4/4; complete full replay
completed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in
35.578 s with 12 workers:
`../sm86_replay_results/full_corpus_h12_atomic_conflict_20261015_v1/full.jsonl.zst`.
The full corpus observed 58 conflict instructions, 622 distinct instructions,
and 198 alias sectors. Exact DRAM/L1/L2/IPC residuals are unchanged. Compact
report: `docs/sm86_atomic_conflict_slice_20261015.md`.

Timing-active atomic serialization is deferred until a locked local
microbenchmark identifies a per-transaction relation.

## SM86 atomic service prior — H13 negative promotion

Independent local RTX 3090 Ti warm-L2 atomic microbenchmarks show same-word
conflict overhead of roughly 10–18% versus distinct addresses, but do not
provide a safe per-transaction DRAM/queue constant. H12 traces preserve exact
alias exposure; H13 therefore rejects a timing multiplier pending a locked
probe and lossless request conflict identity. Current residuals remain DRAM
40.91%, active IPC 18.19%, elapsed IPC 12.42%.

## Current SM86 refinement slice — H14 local-memory width observer

H14 retains producer `data_width` for local-memory records and emits
predicate-lane×width 32-B candidate sectors without changing coalesced traffic,
cache state, or timing. Five local/vector roots completed 5/5; complete replay
completed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in
35.671 s with 12 workers:
`../sm86_replay_results/full_corpus_h14_local_width_20261015_v1/full.jsonl.zst`.
It observed 4,892 width records, zero unknown widths, 19,824 bytes/sectors.
Exact residuals remain unchanged; compact report:
`docs/sm86_local_width_slice_20261015.md`.

## Current SM86 refinement slice — H15 RF Control/Allocate decoupling

H15 keeps transactional RF/RFC reservation diagnostics but removes RF-port
conflict from scheduler eligibility, matching Huerta's Ampere Control/Allocate
semantics. High-RF roots completed 4/4. Complete full replay completed 356
roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in 35.109 s with
12 workers:
`../sm86_replay_results/full_corpus_h15_rf_allocate_20261015_v1/full.jsonl.zst`.

Exact mean APE improves: GPC 13.05→12.91%, SM active 18.19→18.01%, SMSP
17.69→17.50%, elapsed IPC 12.42→12.31%; DRAM remains 40.91%. Compact report:
`docs/sm86_rf_allocate_slice_20261015.md`.

## Current SM86 refinement slice — H16 L1 tag-lookup observer — 2026-10-15

H16 corrected the primary L1 lookup projection. It now counts resident hits and
merge observations, then removes one tag miss per fresh line allocation; it does
not alter cache state, timing, L2, DRAM, or readiness. Four real roots completed
full-first and fast replay 4/4 each. The complete corpus completed 356 roots
(286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392 launches in
36.054 s with 12 workers:
`../sm86_replay_results/full_corpus_h16_l1_lookup_20261015_v1/full.jsonl.zst`.

On 107 exact identity rows, L1 global-load lookup-hit mean APE improved
33.43%→30.71%; lookup-miss APE improved 206.41%→58.44%. L2 (28.24%), DRAM
(40.91%), SM active cycles (18.01%), and elapsed IPC (12.31%) were unchanged.
Compact report: `docs/sm86_l1_lookup_slice_20261015.md`.

This is a bounded tag-observer correction, not physical L1 closure. Async-copy
packetization and the unresolved SM86 set/way/hash state remain explicit residuals;
the next implementation slice is the card-backed 256-set × 4-way effective L1
state arm, evaluated full-first on working-set, thrash, vector-copy, and SGEMM
roots.

## Current SM86 refinement slice — H17 card-backed L1 set/way state — 2026-10-15

H17 activates a card-backed effective 256-set × 4-way modulo-indexed L1 state
for SM86 full replay. The p-chase 98,304-B active-capacity point remains a
separate global capacity scenario; GA102's proprietary hash remains typed
unknown and fully-associative state remains available for sensitivity. Fill,
MSHR, generation, dirty, and observer semantics remain transactional.

Five real roots completed full-first and fast replay 5/5 each. Complete replay
completed 356 roots (286 `ok`, 62 mixed-code controls, 8 replay/input failures),
12,392 launches in 36.517 s with 12 workers:
`../sm86_replay_results/full_corpus_h17_l1_setway_20261015_v1/full.jsonl.zst`.

Versus H16, exact mean APE moved L1 lookup hit 30.71%→30.39%, L1 lookup miss
58.44%→57.74%, GPC 12.91%→12.75%, SM active 18.01%→17.86%, SMSP active
17.50%→17.36%, and elapsed IPC 12.31%→12.12%; L2 source-sector APE regressed
28.24%→32.43%, while DRAM remained 40.91%. Compact report:
`docs/sm86_l1_setway_slice_20261015.md`.

H17 is promoted as an effective topology scenario, not a recovered GA102 hash.
The next slice must adversarially isolate the L2 regression and implement the
source-qualified device-scope store/fence completion tail without fitting NCU.

## Current SM86 refinement slice — H18 L2/hash uncertainty — 2026-10-15

Adversarial comparison isolated H17's L2 regression to fixed modulo-set aliasing:
three high-stride real streaming roots rose from 32,896 to 82,048 L2 lookup
sectors while DRAM stayed 32,768 sectors. H18 adds a deterministic
pressure-bounded modulo/XOR-fold unknown-hash scenario that uses only traced
line-domain occupancy and carded 256-set×4-way geometry. NCU targets are not
inputs and no physical GA102 hash is claimed.

Five representative real full replays completed 5/5. Complete full-first replay
completed 356 roots (286 `ok`, 62 mixed-code controls, 8 replay/input failures),
12,392 launches in 36.751 s with 12 workers:
`../sm86_replay_results/full_corpus_h18_l1_hashfallback_20261015_v2/full.jsonl.zst`.
The final fallback code separately completed the working-set, L2-thrash,
vector-copy, and both vendor-SGEMM roots 5/5:
`../sm86_replay_results/h18_l1_hashfallback_20261015_v2_representative/full.jsonl.zst`.

Against H17, H18 reduces L2 source-sector APE 32.43%→28.24% and retains DRAM
40.91%; L1 hit/miss moves 30.39/57.74%→30.71/58.44%, GPC/SM/elapsed IPC moves
12.75/17.86/12.12%→12.91/18.01/12.31%. Compact report:
`docs/sm86_l2_hash_uncertainty_slice_20261015.md`.

H18 is promoted as the default effective unknown-hash scenario; fixed modulo and
fixed XOR remain sensitivity interpretations. Next work is the independent
source-qualified device-scope store/fence completion tail for IPC/cycles.

## Current SM86 refinement slice — H19 source-qualified store/fence tail — 2026-10-15

H19 extends the existing memory-admission/write-ack seam and broadens the static
fence recognizer to `MEMBAR.GL`, `MEMBAR.GPU`, and `MEMBAR.SC.GPU`. The runner now
auto-selects `arch_profiles/candidates/sm86_store_fence_candidate_v1.json` for
SM86 full replay. Static gating requires traced global STORE/RED work plus a
device-scope fence or ERRBAR/CCTL marker; nonqualified captures are unchanged.
Dirty writeback and coherence remain typed incomplete tails.

Five representative store/fence and memory roots completed full 5/5. The
complete default full-first replay completed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 launches in 36.577 s with 12 workers:
`../sm86_replay_results/full_corpus_h19_store_fence_default_20261015_v2/full.jsonl.zst`.
The run observed 1,069 qualified launches, 103,737 acknowledgements, 378,742
write-completion requests, and 104,805 completion busy cycles. Compact report:
`docs/sm86_store_fence_slice_20261015.md`.

H19 leaves exact L1/L2/DRAM/IPC APE unchanged at 30.71%/58.44%/28.24%/40.91%
for the cache/traffic rows and 12.31% elapsed IPC. This is a completed,
source-qualified effective tail, not an accuracy closure; next work targets why
its completion frontier is not critical on most scored launches.

## Current SM86 refinement slice — H20 dependent-SFU frontier — 2026-10-15

H20 adds a narrow same-register MUFU/RCP/RSQ/EX2/LG2/SIN/COS effective latency
frontier (30 cycles) on SM86, sourced from the independent RTX 3090 Ti clock64
dependent-chain probe. The existing four-cycle SMSP SFU initiation interval and
all non-self-register SFU forms are unchanged.

Five SFU-bearing roots completed full/fast replay 5/5. Complete default replay
completed 356 roots (286 `ok`, 62 mixed-code controls, 8 replay/input failures),
12,392 launches in 37.530 s with 12 workers:
`../sm86_replay_results/full_corpus_h20_sfu_dependent_20261015_v1/full.jsonl.zst`.
It observed 269 launches with nonzero SFU exposure, 970,993 limiting cycles and
1,345,369 exposure cycles. L1/L2/DRAM/IPC APE stayed at 30.71/58.44/28.24/40.91%
and 12.31% elapsed IPC (neutral aggregate). Compact report:
`docs/sm86_sfu_dependent_slice_20261015.md`.

H20 is retained as a bounded effective mechanism, not universal SFU closure.
Next work must find a stronger critical-path IPC source or refine the dependency
gate without importing NCU deltas.

## Current SM86 refinement slice — H21 causal IPC frontier decomposition — 2026-10-15

H21 adds source-derived `ipc_critical_path_issue_frontier_cycles`, terminal
frontier, and interval-width rows to every native launch. They remain diagnostic
frontiers, never NCU selectors or hidden timing crosswalks. Five real critical
path roots completed full 5/5; complete default full-first replay completed 356
roots (286 `ok`, 62 mixed-code controls, 8 replay/input failures), 12,392
launches in 38.560 s with 12 workers:
`../sm86_replay_results/full_corpus_h21_critical_frontier_20261015_v1/full.jsonl.zst`.

Exact L1/L2/DRAM/SM/IPC residuals remain unchanged from H20 (30.71/58.44/28.24/
40.91/18.01% and 12.31% elapsed IPC). Compact report:
`docs/sm86_critical_frontier_slice_20261015.md`.

H21 is a neutral critical-path observability slice; the next arm must use these
frontiers to identify a source-qualified timing mechanism rather than fit NCU.

## Current SM86 refinement slice — H22 L0C miss-switch window — 2026-10-15

H22 consumes the already carded SM86 four-cycle immediate-constant-cache
miss-switch window. Huerta's public Ampere source and released SM86 RTX3090
configuration retain the greedy warp while ordinary readiness is satisfied,
then allow the youngest sibling after four L0C miss opportunities. The Rust
implementation uses a per-SMSP countdown, excludes only the IMC response
frontier from the ordinary-ready test, and keeps memory-admission feedback and
all other dependencies authoritative.

Representative real full-first replay completed 4/4 discovered roots (485 full
launches). The authoritative complete replay completed 356 roots (286 `ok`, 62
mixed-code controls, 8 replay/input failures), 12,392 launches in 37.195 s with
12 workers. Artifacts:
`../sm86_replay_results/full_corpus_h22_l0c_switch_20261015_v2/full.jsonl.zst`
and complementary `fast.jsonl.zst`; post-replay report:
`../sm86_replay_results/metric_report_h22_l0c_switch_20261015_v2/`.

The arm is live on high-pressure multi-warp traces: full replay reports
3,125,368 IMC miss-wait events and 1,489,412 bounded greedy-switch-block
cycles, while tag misses remain 1,174,632 and L1/L2/DRAM traffic is unchanged.
Only six common terminal/GPC frontier rows move (one or two cycles; aggregate
−6 cycles). Mean APE is neutral versus H20: L1 lookup hit/miss
30.71%/58.44%, L2 28.24%, DRAM 40.91%, SM active 18.01%, elapsed IPC 12.31%,
and GPC 12.91%. Compact report: `docs/sm86_l0c_switch_slice_20261015.md`.

H22 is a promoted bounded scheduler mechanism and diagnostic, not IPC closure.
The remaining scheduler residual is that a non-greedy candidate that misses in
L0C is parked without scanning the next ready sibling; the next slice should
implement that transactional candidate scan. The public 46-cycle ERRBAR/
post-EXIT special remains deferred because no dynamic `ERRBAR` record executes
in this corpus.

## Current SM86 refinement slice — H23 non-greedy L0C candidate scan — 2026-10-15

H23 completes the public Huerta issue-loop behavior after H22: an expired
greedy warp is probed first, then ordinary-ready non-greedy siblings are
scanned after each IMC miss. Every probe stages the L0C request in the cloned
cache; only the final effective candidate is admitted and all parked misses,
wait frontiers, and diagnostics commit together. The default SM86 full replay
path now includes this bounded scan.

The post-ordering representative batch completed 10/10 real roots (2,728 full
launches). The complete full-first replay completed 356 roots (286 `ok`, 62
mixed-code controls, 8 replay/input failures), 12,392 launches in 40.715 s with
12 workers. Artifacts:
`../sm86_replay_results/full_corpus_h23_l0c_scan_20261015_v2/full.jsonl.zst`
and complementary `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h23_l0c_scan_20261015_v2/`.

The complete run observed 21,574,287 scan attempts, 21,348,688 scan misses,
225,599 scan successes, and 111,823,998 overlapping IMC wait events. L2 and
DRAM traffic are unchanged from H22; L1 hits/misses/merges move only −104/+104/
+234 sectors due to staged probe side effects. Only 74 common launches move
GPC/terminal frontiers (aggregate +392 cycles, range −147…+505). Mean APE is
neutral at reported precision: L1 lookup hit/miss 30.71%/58.44%, L2 28.24%,
DRAM 40.91%, SM active 18.01%, elapsed IPC 12.31%, GPC 12.91%. Compact report:
`docs/sm86_l0c_scan_slice_20261015.md`.

H23 is a source-aligned bounded mechanism, not IPC closure. The next slice must
attack the remaining typed terminal/writeback/coherence or issue-resource
residual rather than fit the large overlapping probe counts. The public
46-cycle ERRBAR/post-EXIT special remains deferred because no dynamic ERRBAR
executes in the corpus.

## Current SM86 refinement slice — H24 consecutive-BAR generation re-arm — 2026-10-15

H24 adds a 28-cycle SM86 CTA-barrier generation re-arm, scoped only to a later
`BAR.SYNC` generation of the same CTA. A fresh local RTX 3090 Ti clock64 probe
from the retained barrier-pipeline source measured 688,134 cycles for 16,384
one-barrier generations and 1,146,885 cycles for two consecutive barriers,
which is 28.000 cycles per extra generation. Arrival throughput remains the
independent two cycles per warp.

Five ordinary barrier/control and five high-BAR real roots completed full-first
replay. The complete replay then completed 356 roots (286 `ok`, 62 mixed-code
controls, 8 replay/input failures), 12,392 launches in 40.694 s with 12 workers:
`../sm86_replay_results/full_corpus_h24_barrier_rearm_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h24_barrier_rearm_20261015_v1/`.

Only 29 launches move versus H23 (maximum +482 cycles). GPC/terminal frontier
sum rises +1,847 cycles. Mean APE shifts slightly: GPC 12.90596%→12.90078%,
SM active 18.00767%→18.02024%, elapsed IPC 12.31350%→12.30731%; L1/L2/DRAM
remains 30.71%/58.44%/28.24%/40.91%. Compact report:
`docs/sm86_barrier_rearm_slice_20261015.md`.

H24 is a promoted conditional barrier mechanism, not accuracy closure. Keep it
scoped to consecutive CTA generations; the next slice should pursue dominant
eligible-not-selected/front-end/L1TEX dependency residuals or a typed fixed-
result-queue observer without inventing execution-to-writeback timing.

## Current SM86 refinement slice — H25 dynamic-youngest priority audit — 2026-10-15

H25 retains the trace's block-local dynamic warp identity and uses it as the
deterministic tie-break for the existing arrival-order youngest selection. A
within-CTA arrival-vs-dynamic-ID inversion counter makes the Huerta
greedy-then-highest-ID assumption explicit without changing timing.

An independent local RTX 3090 Ti clock64 fixed-result-queue probe reproduced a
multiwarp W8 threshold (+0.163 then +1.118 cycles/generation) but showed null
W1/W4 deltas and a non-monotonic collision control. Queue capacity and
execution-to-writeback release remain unidentified, so no result-queue timing
arm was activated.

Five high-BAR roots passed representative full-first replay. Complete replay
passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in
40.456 s with 12 workers:
`../sm86_replay_results/full_corpus_h25_priority_audit_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h25_priority_audit_20261015_v1/`.

All 12,392 launches report zero priority inversions and H25 is bit-for-bit
neutral versus H24 for L1/L2/DRAM/IPC/frontier outputs. Readiness schema is 29.
Compact report: `docs/sm86_priority_queue_audit_slice_20261015.md`.

H25 promotes the explicit audit and records a negative result for timing-active
fixed-result queues. The next slice must target the dominant
eligible-not-selected/front-end/L1TEX dependency residuals with an independent
mechanism prior; no unsupported queue release delay is allowed.

## H26 branch-stream redirect experiment — rejected — 2026-10-15

H26 tested flushing per-SMSP fall-through instruction-stream buffers on traced
non-fall-through branches. Five branch-heavy real roots passed full replay, and
the complete A/B replay completed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches with 12 workers:
`../sm86_replay_results/full_corpus_h26_branch_redirect_20261015_v1/full.jsonl.zst`.

The arm is rejected from the default path: GPC APE worsened 12.90%→14.37%, SM
active 18.02%→22.03%, and elapsed IPC 12.31%→12.73%; L1/L2/DRAM did not improve.
The timing change was removed after the A/B run; compact negative report:
`docs/sm86_branch_redirect_slice_20261015.md`.

This is a falsified front-end hypothesis, not a blocker. Future front-end work
must model static branch-path prediction/target state more narrowly or remain
observer-only; the active default returns to H25/H24 timing.

## Current SM86 refinement slice — H27 L1TEX/front-end redirect observer — 2026-10-15

H27 retains H26's rejected branch-flush timing result but adds a native
observer for dynamic non-fall-through control transfers. It does not mutate
stream buffers or timing; readiness schema 30 carries the count.

Five branch-heavy roots passed representative full-first replay. Complete replay
passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches in
40.452 s with 12 workers:
`../sm86_replay_results/full_corpus_h27_l1tex_observer_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h27_l1tex_observer_20261015_v1/`.

The corpus contains 1,882,174 dynamic non-fall-through transfers and 85,461,673
front-end fetches. H27 is bit-for-bit neutral versus H25 for L1/L2/DRAM, IPC,
frontiers, and APE. Compact report: `docs/sm86_l1tex_observer_slice_20261015.md`.

Use this observer to partition future front-end/L1TEX models by static target
shape; do not convert transfer counts into a universal delay.

## Current SM86 refinement slice — H28 constant-pressure L1TEX observer — 2026-10-15

H28 makes the existing source-qualified SM86 constant-pressure memory path
observable: static SASS streams beyond the released five-entry dispatch window
are counted dynamically, while the established `11 + 39` effective frontier
and all timing remain unchanged. No NCU values are used as inputs.

One direct streaming root and four high-IMC/pressure roots passed representative
full-first replay. Complete replay passed 356 roots (286 `ok`, 62 controls, 8
failures), 12,392 launches in 40.672 s with 12 workers:
`../sm86_replay_results/full_corpus_h28_constant_pressure_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h28_constant_pressure_20261015_v1/`.

The corpus observes 6,336,451 constant-pressure instructions and 1,882,174
dynamic control-flow redirects. Readiness schema is 31. H28 is bit-for-bit
neutral versus H27 for L1/L2/DRAM, IPC/frontiers, and APE. Compact report:
`docs/sm86_constant_pressure_slice_20261015.md`.

Keep the pressure count as a residual partition; a timing arm requires fresh
clock64 load-return/queue evidence rather than a static threshold assumption.

## H29 constant-pressure clock64 differential — negative — 2026-10-15

Fresh no-NCU RTX 3090 Ti clock64 runs of the retained constant-load probe
(dependent chain versus four independent reads, 128 iterations, 1/4/16 blocks)
passed all functional checks. Dependent runs are ~9.7k cycles; independent
pressure runs are ~11.6–12.9k cycles. The differential mixes issue throughput,
overlap, and operand return, so it cannot identify a new L1TEX delay. No timing
change was promoted; H28's 6,336,451-instruction pressure observer remains the
active model. Compact negative report:
`docs/sm86_constant_dependency_probe_slice_20261015.md`.

## Current SM86 refinement slice — H30 six-slice L2 routing observer — 2026-10-15

H30 adds transactionally preserved per-slice L2 request-sector counts using the
active six-slice effective geometry/hash. It makes no proprietary GA102 hash
claim, adds no queue timing, and never uses NCU to select routing.

Five high-stride/L2-thrash/working-set/SGEMM roots passed representative replay.
Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 40.089 s with 12 workers:
`../sm86_replay_results/full_corpus_h30_l2_slice_observer_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h30_l2_slice_observer_20261015_v1/`.

Full-corpus slice sectors are 1,718,579 / 1,653,401 / 1,717,493 /
1,652,955 / 1,717,655 / 1,653,918. H30 is bit-for-bit neutral versus H29 for
L1/L2/DRAM counters, IPC/frontiers, and APE. Compact report:
`docs/sm86_l2_slice_observer_slice_20261015.md`.

Next L2 work requires independent slice-queue/service evidence; do not infer a
physical GA102 hash or timing from this balanced observer alone.

## Current SM86 refinement slice — H31 L2 slice-pressure observer — 2026-10-15

H31 extends H30 with per-launch effective L2 max-slice, min-slice, spread, and
max/min imbalance observers. It does not recover a GA102 hash, model hidden
queues, or feed NCU/timing selection.

The retained RTX 3090 Ti clock64 L2 queue probe passed unique/reuse smoke
points across 1/2/4/8 warps and 1/8/32-load bursts; pressure growth is stable
as an observer but not a transferable fixed latency. Five high-stride/L2 roots
passed representative full replay. Complete replay passed 356 roots (286 `ok`,
62 controls, 8 failures), 12,392 launches in 39.887 s with 12 workers:
`../sm86_replay_results/full_corpus_h31_l2_pressure_observer_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h31_l2_pressure_observer_20261015_v1/`.

Per-launch imbalance ratio: minimum 1.0, median 1.133, p90 3.0, maximum 38.0.
H31 is bit-for-bit neutral versus H30 for L1/L2/DRAM, IPC/frontiers, and APE.
Compact report: `docs/sm86_l2_pressure_observer_slice_20261015.md`.

No L2 queue timing arm is promoted; future timing requires independent service
evidence beyond pressure observation.

## Current SM86 refinement slice — H32 L2 pressure strata — 2026-10-15

H32 extends H31 with active-slice count and maximum-slice demand fraction,
alongside max/min/spread/imbalance observers. No hidden GA102 hash, queue depth,
or NCU-fitted latency is introduced.

Complete full-first replay passed 356 roots (286 `ok`, 62 controls, 8 failures),
12,392 launches in 40.152 s with 12 workers:
`../sm86_replay_results/full_corpus_h32_l2_pressure_strata_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h32_l2_pressure_strata_20261015_v1/`.

Pressure ratios remain min 1.0, median 1.133, p90 3.0, max 38.0; max-slice
fraction is min 16.67%, median 18.75%, p90 50%, max 100%. Active-slice counts:
6 (9,186 launches), 2 (1,658), 3 (488), 4 (487), 1 (358), 5 (215).
H32 is bit-for-bit neutral versus H31 for L1/L2/DRAM, IPC/frontiers, and APE.
Compact report: `docs/sm86_l2_pressure_strata_slice_20261015.md`.

## Current SM86 refinement slice — H33 L2 route concentration observer — 2026-10-15

H33 adds a normalized sum-of-squares concentration metric over effective
six-slice request demand. It distinguishes balanced traffic from single-slice
concentration without assuming a proprietary hash or queue latency.

Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 39.868 s with 12 workers:
`../sm86_replay_results/full_corpus_h33_l2_concentration_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h33_l2_concentration_20261015_v1/`.

Concentration is min 166,667 ppm, median 168,405 ppm, p90 500,000 ppm, max
1,000,000 ppm; imbalance remains min 1.0, median 1.133, p90 3.0, max 38.0.
H33 is timing/traffic neutral versus H32. Compact report:
`docs/sm86_l2_concentration_slice_20261015.md`.

## Current SM86 refinement slice — H34 L2 route strata — 2026-10-15

H34 converts H33 concentration into deterministic route-local strata:
incomplete (<6 active slices), balanced (all six and <250,000 ppm), or
concentrated (all six and >=250,000 ppm). These are descriptive observers only.

Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 40.574 s with 12 workers:
`../sm86_replay_results/full_corpus_h34_route_strata_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h34_route_strata_20261015_v1/`.

Strata counts: 3,206 incomplete, 9,135 balanced, 51 concentrated. H34 is
timing/traffic neutral versus H33 for L1/L2/DRAM, IPC/frontiers, and APE.
Compact report: `docs/sm86_l2_route_strata_slice_20261015.md`.

## Current SM86 refinement slice — H35 route-strata correlation — 2026-10-15

H35 joined H34 route strata to independent post-replay NCU comparisons without
using NCU as a model input. Concentrated launches show higher cycle residuals,
but only 4–8 joined rows per metric; incomplete launches show high DRAM error
on a small subset while L2-sector error is near zero. The correlation is too
sparse/confounded to identify service timing.

Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 40.529 s with 12 workers:
`../sm86_replay_results/full_corpus_h35_route_correlation_20261015_v1/full.jsonl.zst`;
report:
`../sm86_replay_results/metric_report_h35_route_correlation_20261015_v1/`.

H35 preserves the typed route-strata ledger and does not activate L2 timing.
Compact report: `docs/sm86_l2_route_correlation_slice_20261015.md`.

## Current SM86 refinement slice — H36 sectored L2 fill ownership — 2026-10-15

H36 exposes the active event-mode coalescing rule as a checked, cumulative
observer: each read sector is classified as either a new fill owner or a
pending merge onto the matching allocation/sector. A pending merge remains a
lookup miss and reuses the existing ready frontier; no timing constant, hidden
pending-table capacity, or NCU fit is introduced. Full replay emits
`l2_pending_merge_sectors`, `l2_new_fill_sectors`, and
`l2_pending_merge_fraction_pct`.

Five L2-thrash/high-stride/working-set/SGEMM roots passed representative
full-first replay (13 launches):
`../sm86_replay_results/h36_fill_ownership_observer_20261015_v1/full.jsonl.zst`.

Complete full-first replay passed 356 roots (286 `ok`, 62 controls, 8
replay/input failures), 12,392 launches in 40.379 s with exactly twelve
workers. Artifacts:
`../sm86_replay_results/full_corpus_h36_fill_ownership_observer_20261015_v1/full.jsonl.zst`
and adjacent `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h36_fill_ownership_observer_20261015_v1/`.

Corpus totals are 1,727,757 pending-merge sectors and 4,823,652 new-fill
sectors. Pending merges occur in 1,822 launches; the merge fraction among
nonempty launches is 0% minimum, 0% median, 2.273% p90, and 97.196% maximum.
An exact launch-key adversarial comparison versus H35 found zero differences in
all prior counters, metric predictions, DRAM fields, timing/frontier values,
and trace fields after excluding `sidecar_join_wall_ns` and the three new
observer rows. H36 is therefore traffic/timing/IPC neutral and is promoted as
the measurement surface for the next independently justified conditional arm.

Compact report: `docs/sm86_l2_fill_ownership_slice_20261015.md`.

## Current SM86 refinement slice — H37 source-qualified PRT/fill pressure — 2026-10-15

Huerta's Ampere model and modern GPGPU-Sim source support finite memory queues,
PRT/MSHR waiter merging, and explicit fill/response service, but the local
RTX 3090 Ti pressure probes do not identify a stable transferable queue delay.
An experimental one-cycle-per-merged-sector drain arm changed 1,670 frontiers,
improved GPC mean APE by only 0.004 percentage points, and worsened SM-active
APE; it was removed before promotion.

H37 instead adds a source-qualified pending-sector pressure ledger. Before each
shared-L2 event request it samples pending fill sectors by L2 source and emits
sample count, pressure sum, and bounded mean rows. It is observer-only and does
not change cache state, lookup classification, DRAM, IPC, or timing.

Five real L2/high-stride roots passed representative full-first replay. Complete
full-first replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 40.638 s with 12 workers:
`../sm86_replay_results/full_corpus_h37_prt_pressure_observer_20261015_v1/full.jsonl.zst`
and adjacent compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h37_prt_pressure_observer_20261015_v1/`.

Pressure totals are L1-fill 2,727,211 samples/3,253,767,441 sector-samples,
bypass 31,433/1,933,553, untagged 32/4,666, and store 3,279,482/198,743,493.
The H36-vs-H37 keyed adversarial comparison found zero old-field differences
after excluding only new pressure rows and `sidecar_join_wall_ns`. Compact
report: `docs/sm86_l2_prt_pressure_slice_20261015.md`.

## Current SM86 refinement slice — H38 L1 MSHR-full pressure — 2026-10-15

H38 instruments the existing finite L1 MSHR admission path with checked
occupancy samples/sums whenever a miss finds the MSHR set full. It applies no
delay, changes no retry/cache behavior, and keeps the unified pressure gauge
separate from global/local traffic lanes. The rows are observer-only; the
carded MSHR count is not claimed as a proprietary GA102 PRT size.

Five high-MSHR/high-stride/L2 roots passed representative full-first replay.
Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 40.241 s with exactly twelve workers:
`../sm86_replay_results/full_corpus_h38_l1_mshr_pressure_20261015_v1/full.jsonl.zst`
and compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h38_l1_mshr_pressure_20261015_v1/`.

The corpus contains 953,785 full-set MSHR stall samples in 1,120 launches, with
30,978,868 occupancy entries (maximum sample count 124,766; maximum sum
3,992,512). Streaming/DSMEM representatives each record 10,924 samples at
occupancy 32; vendor SGEMM records 36; pointer-chase/working-set roots record
none. A keyed H37 comparison found zero old-field differences after excluding
new MSHR rows and `sidecar_join_wall_ns`; all L1/L2/DRAM/IPC/frontier outputs
and APE remain unchanged. Compact report:
`docs/sm86_l1_mshr_pressure_slice_20261015.md`.

## H39 negative result — L1 lookup retry correction — 2026-10-15

H39 tested whether MSHR-full miss attempts should be added to the L1
lookup-sector denominator. Using H38's complete full-first corpus and 107 exact
independent NCU joins, correlations between MSHR-stall samples and APE are
−0.015 for global-load sectors, −0.041 for lookup hits, −0.018 for lookup
misses, 0.046 for GPC cycles, and −0.032 for SM-active cycles. High-stall
quartiles have no consistent lookup residual direction.

No lookup correction or timing change was implemented. The H38 pressure
observer remains active; future retry modeling requires a dedicated locked
request-level probe. Negative report:
`docs/sm86_l1_lookup_retry_negative_20261015.md`.

## Current SM86 refinement slice — H40 L1 global-load scope audit — 2026-10-15

H40 adds a post-replay diagnostic that flags gross (>4×) disagreement between
trace-side logical memory-instruction counts and independent NCU opcode
metrics. It never changes replay, prediction selection, APE scoring, or NCU
model inputs.

Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 40.199 s with twelve workers:
`../sm86_replay_results/full_corpus_h40_l1_scope_audit_20261015_v1/full.jsonl.zst`
and compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h40_l1_scope_audit_20261015_v2/`.

Exactly one full exact-identity scope warning appears:
`sm86_adapted_l2_074_sam_iou_backward_sm120` has 1,024 trace global-load
instructions/4,352 sectors versus 16 NCU opcode instructions/40 sectors (64×
scope ratio), while warp instructions agree exactly at 32,736. No correction is
promoted; old replay fields and APE are unchanged versus H38. Compact report:
`docs/sm86_l1_scope_audit_slice_20261015.md`.

## Current SM86 refinement slice — H41 L2/DRAM scope audit — 2026-10-15

H41 extends the post-replay-only scope diagnostic to simulator L2 source-sector
and DRAM read-sector counters versus independent NCU operation metrics. It uses
fixed gross-ratio warnings and never changes replay, prediction selection, APE,
or NCU model inputs.

Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 39.934 s with twelve workers:
`../sm86_replay_results/full_corpus_h41_l2_dram_scope_audit_20261015_v1/full.jsonl.zst`
and compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h41_l2_dram_scope_audit_20261015_v1/`.

Full exact-identity joins contain 121 DRAM read-sector warnings, 3 L2
source-sector warnings, and the one H40 global-load opcode warning. The SAM-IoU
join alone has 2,464 modeled L2 source sectors versus 168 NCU sectors and 1,076
modeled DRAM reads versus 564 NCU sectors. No multiplier or timing correction
is promoted; old replay fields and APE are unchanged. Compact report:
`docs/sm86_l2_dram_scope_audit_slice_20261015.md`.

## Current SM86 refinement slice — H42 scope-aware NCU residuals — 2026-10-15

H42 makes explicit H40/H41 scope warnings operational in post-replay reporting.
Predictions covered by a gross trace-vs-NCU scope warning remain in compressed
rows with status `scope_mismatch` and in the warning inventory, but are excluded
from primary APE. Replay, prediction selection, timing, and NCU model inputs do
not change.

H42 reduces the H41 full/fast artifacts and publishes:
`../sm86_replay_results/metric_report_h42_scope_aware_20261015_v1/`.
The full mode has 125 scope warnings and 67 excluded predictions across 107
exact joins. Secondary like-for-like mean APE is L1 global-load sectors
100.96%→0.21%, L2 source sectors 28.24%→6.76%, DRAM read sectors
40.91%→26.84%, while GPC cycles remains 12.90%. The scope-aware values are a
**secondary like-for-like view**; raw/all-row H41 residuals
(100.96%/28.24%/40.91%) remain authoritative, and excluded rows are unresolved
scope/model gaps, not completed mechanisms.
The remaining 26.84% scope-compatible DRAM residual is still active; no closure
claim is made. Compact report:
`docs/sm86_scope_aware_report_slice_20261015.md`.

## H43 DRAM residual ledger — 2026-10-15

H43 joins H42 scope-aware DRAM rows to H37 route/fill-pressure and H38 L1 MSHR
observers. The retained corpus has 58 scope-compatible exact DRAM joins with
26.84% mean APE. All are in the balanced route stratum; concentrated/incomplete
classes have no exact DRAM joins.

Pressure partitions are non-monotonic: low/high pending-merge quartiles are
28.69%/12.47% APE, low/high new-fill quartiles 40.92%/8.59%, and low/high MSHR
groups 28.59%/23.25%. This is confounded by family, shape, and demand volume,
not a transferable service law. No queue delay, partition/hash selection, or
bandwidth correction is promoted. Compact report:
`docs/sm86_dram_residual_ledger_slice_20261015.md`.

## H44 DRAM dirty-tail residual ledger — 2026-10-15

H44 joins the 58 scope-compatible exact DRAM rows with dirty residency,
dirty-eviction, and typed tail flags. All rows carry both `dirty_writeback` and
`coherence` tails; none has dirty L2 eviction sectors, so terminal L2 dirty
residency is the varying demand.

Dirty-residency APE is non-monotonic: low/middle/high quartiles are 21.29%,
35.82%, and 12.10%. The relation is confounded by read demand, family, and
shape; no writeback bandwidth, queue capacity, or coherence law is identified.
Default writeback timing remains disabled. Compact report:
`docs/sm86_dram_dirty_tail_ledger_slice_20261015.md`.

## H45 IPC/cycle residual ledger — 2026-10-15

H45 joins exact cycle predictions to causal frontiers, H37/H38 pressure, route
strata, and typed tails. For 107 exact joins, GPC mean APE is 12.90% and
SM-active mean APE 18.02%. Balanced/incomplete/concentrated route classes are
11.67%/16.08%/28.03% GPC APE (n=88/15/4) and 15.07%/31.76%/31.32% SM-active
APE. Frontier gaps and pressure partitions are non-monotonic; sparse route
classes cannot identify timing. No IPC/cycle correction is promoted. Compact
report: `docs/sm86_ipc_residual_ledger_slice_20261015.md`.

## H46 DRAM demand interval audit — 2026-10-15

The existing trace-derived DRAM demand interval brackets 34/58 (58.6%) H42
scope-compatible exact NCU DRAM read rows. Median low/high bounds are 262/392
sectors and target median is 392; high residuals often lie above the interval,
but family, demand, writeback/coherence, and operation scope remain confounded.
No bound expansion, multiplier, or timing change is promoted. Compact report:
`docs/sm86_dram_demand_interval_slice_20261015.md`.

## H47 negative result — dirty-writeback scenario A/B — 2026-10-15

The existing bounded `CardedWriteback` path was exercised at a predeclared
4,096-sector capacity on five high-terminal-dirty real traces. GPC APE improved
for qkv-rope (29.19%→24.92%), RMSNorm (12.34%→11.51%), and Hyena FFT
(8.38%→7.26%), was unchanged for Flash attention (23.72%), and worsened for
fused linear/GELU/GRN (24.69%→26.16%). SM-active frontiers never changed.

No default writeback timing promotion is justified; the existing optional
scenario and typed tail flags remain explicit. Compact report:
`docs/sm86_writeback_scenario_negative_20261015.md`.

## Current SM86 refinement slice — H48 LDG constant-form observer — 2026-10-16

H48 adds static `LTC128B`, `U16`, and other/plain form counters for
`LDG.*.CONSTANT` global loads. Forms remain global traffic; no L0C/GCC reroute,
lookup correction, timing, or NCU fit is introduced.

Complete replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392
launches in 40.899 s with twelve workers:
`../sm86_replay_results/full_corpus_h48_constant_form_observer_20261016_v1/full.jsonl.zst`
and compressed `fast.jsonl.zst`; report:
`../sm86_replay_results/metric_report_h48_constant_form_observer_20261016_v1/`.

Form totals: LTC128B 199,446 instructions/828,751 sectors (141 launches), U16
5,515,663/5,612,737 (71), other/plain 1,092,487/4,375,547 (688). SAM-IoU's
outlier is LTC128B; RMSNorm/qkv are U16; streaming roots are other/plain. Old
traffic, timing, IPC, frontier, and APE values remain unchanged. Compact report:
`docs/sm86_constant_form_observer_slice_20261016.md`.

## Current SM86 refinement slice — H49 constant-form microprobe — 2026-10-16

The designated notes project probe
`/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/constant_form_scope`
builds two identical-work SM86 kernels. Real execution passes identical
functional checksums, and `cuobjdump` confirms `LDG.E.CONSTANT` for 32-bit
`__ldg` versus `LDG.E.U16.CONSTANT` for uint16 `__ldg`. This validates form
identity as an observer discriminator without proving an L0C/constant-space
route. No lookup/timing correction or NCU fit is promoted. Compact report:
`docs/sm86_constant_form_probe_slice_20261016.md`.

## Current SM86 refinement slice — H50 constant-form reuse/warp sweep — 2026-10-16

The notes probe swept 24 real RTX 3090 Ti points: plain versus uint16 `__ldg`,
1/4/8 warps, and reuse distances 1/4/32/256 lines. All points pass identical
functional checksums per pair; SASS remains `LDG.E.CONSTANT` versus
`LDG.E.U16.CONSTANT`. Clock64 grows with reuse/pressure but form differences
are small and non-monotonic (W1/reuse1 9,257 vs 9,258; W4/reuse32 23,089 vs
21,574; W8/reuse256 39,123 vs 39,108). No form-specific timing correction is
promoted. Compact report: `docs/sm86_constant_form_sweep_slice_20261016.md`.

## H51 constant-form L1 lookup residual ledger — 2026-10-16

H51 joins H48 static form counters to exact independent NCU L1 lookup rows.
Other/plain forms show 5.85% hit and 31.53% miss mean APE (32/43 joins), U16
shows 51.36% hit and 34.90% miss (42/50), LTC128B has no positive-target hit
joins and 26.68% miss (3), and no-form rows show 6.89% hit/25.25% miss. Forms
remain global traffic observers; no correction or fitting is promoted. Compact
report: `docs/sm86_constant_form_residual_ledger_slice_20261016.md`.

## H52 integrated non-regression audit — 2026-10-16

H48 has 12,392 launch keys, exactly matching H38; an adversarial comparison
finds zero old-field differences after removing only new form rows and
`sidecar_join_wall_ns`. H42/H48 report inventories each retain compressed
`metric_comparisons.jsonl.zst`, `summary.json`, and `summary.md`, with 125 scope
warnings and 67 explicit exclusions. Workspace cargo checks, Python compilation,
and diff checks pass. Scope exclusions remain unresolved, not closure claims.
Their summary headings predate H63's raw-versus-secondary correction; use the
regenerated H63 report for that view. Compact report:
`docs/sm86_integrated_audit_slice_20261016.md`.

## H53 L2 line/sector residual ledger — 2026-10-16

On 104 H42 scope-compatible exact joins, L2 lookup-miss APE is exactly 0%.
Lookup-hit mean APE is 11.77% overall: balanced 12.50% (85 rows), incomplete
1.02% (15), concentrated 36.77% (4). Line/sector miss counts and pending/new
pressure are non-monotonic; no L2 multiplier or route timing arm is promoted.
Compact report: `docs/sm86_l2_semantic_residual_ledger_slice_20261016.md`.

## H54 independent NCU constant-form probe — 2026-10-16

NCU independently profiled plain 32-bit and uint16 `__ldg` controls with
identical work. At 1 warp/64 iterations/reuse1 both report 64 global-load
instructions/requests, 256 sectors, and 252 hits/4 misses; at 8 warps/64
iterations/reuse256 both report 512 instructions/requests, 2,048 sectors, and
1,764 hits/284 misses. No form-specific lookup differential appears in these
controls; results were not replay inputs or fitted parameters. Compact report:
`docs/sm86_constant_form_ncu_probe_slice_20261016.md`.

## H55 consolidated accuracy dashboard — 2026-10-16

`docs/sm86_accuracy_dashboard_20261016.md` consolidates raw/all-row H41 APE and
secondary H42 scope-compatible APE, excluded counts, corpus coverage, and
unresolved tails. Raw values remain authoritative; secondary values are not
closure claims. The dashboard records 356 roots/12,392 launches/107 identity
joins and the current L1/L2/DRAM/IPC residuals.

## H56 scope-warning threshold sensitivity — 2026-10-16

On H41 exact joins, L2 scope exclusions are 6/3/3 and DRAM exclusions 70/49/29
at 2×/4×/8× thresholds; DRAM secondary APE is 19.15%/26.84%/32.81%. The 4×
threshold remains a fixed gross-scope diagnostic, not an optimization knob;
raw/all-row APE remains authoritative. Compact report:
`docs/sm86_scope_threshold_sensitivity_slice_20261016.md`.

## H57 SOTA DRAM queue prior — 2026-10-16

Source-inspection notes from GPGPU-Sim/Accel-Sim describe per-partition
private/shared credits, finite L2↔DRAM queues, and round-robin arbitration;
Huerta adds finite memory-subcore/PRT/coalescer structures. No Accel-Sim
binary/runtime or imported constant is used. These generic priors do not identify
GA102 partition hashes/capacities or completion ownership. H43/H47 evidence is
confounded/mixed, so no partitioned timing arm or imported capacity is enabled.
Compact report: `docs/sm86_dram_sota_queue_prior_slice_20261016.md`.

## H58 branch integration audit — 2026-10-16

Release `sim-trace`/`sim-run` builds, workspace all-features checks, Python
compilation, and diff checks pass. Five current real traces replay full-first
5/5 (99 launches) with compressed output and the twelve-worker runner. H48/H42
artifacts/reports and the H58 smoke artifact are present; active maps contain
no stale SM120 default/corpus path. Compact report:
`docs/sm86_integration_audit_slice_20261016.md`.

## H59 GPC observation-envelope residual — 2026-10-16

The measured SM86 envelope (`1776 + 8·floor(log2(grid))`) is non-monotonic
against 107 exact NCU joins: log₂ 0/2/3/8/10 groups show 12.11%/9.56%/17.51%/
21.76%/29.19% APE with sparse high-grid samples. Active-SM partitions are also
non-monotonic. No grid/SM multiplier or NCU correction is promoted. Compact
report: `docs/sm86_gpc_envelope_residual_slice_20261016.md`.

## H60 unresolved mechanism gap tracker — 2026-10-16

The explicit tracker records raw/secondary L1/L2/DRAM/IPC residuals, typed
writeback/coherence tails, 125 scope warnings/67 exclusions, sparse route/grid
strata, and four concrete probe requirements. No gap is renamed, silently
excluded, or declared physically recovered. Report:
`docs/sm86_unresolved_gap_tracker_20261016.md`.

## H61 LTC128B probe boundary — 2026-10-16

Direct `ld.global.LTC128B.u32` PTX is rejected by CUDA 13.2 `ptxas` with
`Unknown modifier '.LTC128B'`. The SASS-only form is therefore retained from
H48 traces/cuobjdump as an observer; no invalid inline opcode or binary patch is
attempted. Compact report:
`docs/sm86_ltc128b_probe_blocker_slice_20261016.md`.

## H62 full-mode capability contract audit — 2026-10-16

The active full runner enables all implemented/evidence-backed SM86 observers
and models by default: integrated L1/L2, aggregate carded DRAM/read service,
source/route pressure, MSHR pressure, form observers, IPC/frontier projections,
and scope diagnostics. Partition candidates remain observer-only; writeback,
coherence, and visibility arms activate only when typed trace events make them
applicable and explicit scenario flags configure them. Otherwise tails remain
typed unavailable rather than silently invented.
Plan wording was reconciled with this contract. No runtime behavior changed.

## H63 adversarial plan/source consistency audit — 2026-10-16

The H30–H62 review corrected two concrete contract defects. H42 prose and the
reducer had called scope-compatible rows “primary”; they now explicitly retain
scope-mismatch rows in compressed output, exclude them from raw/all-row
comparison APE, and label the compared table secondary like-for-like. Raw
H41/all-row values remain authoritative. The plan's DRAM section also now
matches the landed source: full replay uses aggregate `DramBw::Carded` read
service plus observer-only H30–H37 partition/route covariates and the H19
store/fence candidate; finite partition/bank timing is a future
evidence-gated arm, while dirty writeback/coherence/visibility remain optional
or typed unavailable by default. H57–H62 entries are chronologically ordered.
Modeled cache-state/trace line-domain observers no longer use physical-truth
wording; GPGPU-Sim/Accel-Sim references are source inspection only; and the
825 GB/s value is documented as an unlocked absolute-rate hypothesis mapped
onto the locked simulation clock.

The corpus runner now has an explicit `--no-memory-admission-feedback` ablation,
defaults H19 feedback only for launch scope, suppresses it (and records the
suppression) for stream-L2 scope, and per-launch DRAM counters retain the
configured service rate.

The corrected reporter reduced the retained real H41 full/fast artifacts at
`../sm86_replay_results/metric_report_h63_scope_label_audit_20261016_v1/` with
125 full warnings/67 exclusions and 0/0 fast warnings/exclusions; its summary
machine fields identify the secondary scope view and raw-APE exclusion policy.
A real stream-L2 full replay completed `ok` for one capture with candidate
suppression and compressed output at
`../sm86_replay_results/h63_stream_l2_scope_20261016_v2/full.jsonl.zst`.
No cache/DRAM timing law, NCU fitting, observer promotion, or accuracy claim
changed. Compact report:
`docs/sm86_plan_consistency_audit_slice_20261016.md`.

The unresolved-mechanism consultation bundle for an external model is
`docs/sm86_external_consult_prompt_20261016.txt`; it is a plain-text copy/paste
prompt and is not an execution or acceptance gate.

## H64 L2 queue-frontier timing arm — negative decision — 2026-10-16

The independent notes `l2_queue_frontier` probe was rebuilt on the identified
RTX 3090 Ti (UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, CC 8.6, driver
615.71.09). A 288-point unique/reuse matrix passed all correctness checks.
Twenty-repeat clock64 medians at eight warps/64 loads rise from 20,765 to
28,137 to 32,334 cycles for strides 1/8/32, but selected NCU points change
L2-active/XBAR observations only about 4% and show zero selected request-stall
cycles. The differential is confounded by issue, occupancy, overlap, and return
visibility.

No L2 queue capacity/hash/stride delay or merged-sector timing law is promoted;
H30–H37 remain observer-only. The next required probe is a locked one-factor
DRAM demand/completion separation including dirty writeback/coherence. Report:
`docs/sm86_l2_queue_frontier_negative_20261016.md`.

## H65 P1 L1 wavefront lookup semantics — negative decision — 2026-10-16

A rebuilt same-binary L1TEX probe confirmed cold line loads produce zero lookup
hits and all requested sectors as lookup misses, while output-wavefront counts
remain a distinct NCU endpoint. A full 356-root A/B experiment replacing the
H16 projection with `hits + pending_lookup_hits` reduced zero-target hit false
positives (22/29→1/29) but worsened lookup-hit mean APE 30.71%→33.43% and
lookup-miss mean APE 58.44%→110.05%. The experiment was rejected and the
source restored; no default timing/cache/DRAM behavior changed. Report:
`docs/sm86_l1_wavefront_lookup_negative_20261016.md`.

## H65 supported LTC128B PTX probe — 2026-10-16

CUDA 13.2 `ptxas` accepts `.L2::128B` and emits `LDG.E.LTC128B.STRONG.SM`.
A matched RTX 3090 Ti control produced 43,008 plain versus 172,032 LTC128B
L2 TEX read-miss sectors, 0 versus 129,024 promoted sectors, and 43,176 versus
172,672 DRAM read sectors. This is independent probe evidence only; no timing
promotion or corpus scope reassignment is active. Report:
`docs/sm86_ltc128b_ptx_probe_slice_20261016.md`.

## H66 GCC/L1.5 demand observer — 2026-10-16

An independent RTX 3090 Ti probe confirmed GCC/L1.5 traffic distinct from TEX:
control/constant arms reported GCC L2 sectors 952/1,960 and DRAM reads
3,100/276. Full replay now emits observer-only GCC instruction/constant demand
candidates and a multi-client low demand row; it does not alter L2, DRAM, or
frontiers. Complete full-first replay passed 356 roots (286 `ok`, 62 controls,
8 failures), 12,392 launches, 12 workers, compressed output:
`../sm86_replay_results/full_corpus_h66_gcc_demand_observer_20261016_v1/`.
Report: `docs/sm86_gcc_demand_observer_slice_20261016.md`.

## H67 L2 write-lookup split — negative decision — 2026-10-16

A cold-store NCU probe reported zero lookup-hit sectors, but a full 356-root
experiment replacing the corpus write-tag observer with present-sector hits
worsened secondary L2 lookup-hit APE 11.77%→33.67% and lookup-miss APE
0.0%→53.16%; source-sector APE stayed 6.76%. The change was reverted. The
probe and scored source-unit endpoints are not yet like-for-like; operation-
split NCU/TEX and NVBit crosswalk evidence is required before write-coalescing
or lookup timing changes. Report:
`docs/sm86_l2_write_lookup_negative_20261016.md`.

## H68 no-fence store-feedback qualification — negative decision — 2026-10-16

Broadening H19 feedback to all static STORE/RED kernels was tested on five real
full roots (20 launches). All frontiers and admission exposure were unchanged;
the candidate alone cannot model implicit CTA-EXIT drain. The qualification was
restored. Next is a separately gated EXIT-drain arm based on matched no-fence/
fence probe evidence. Report:
`docs/sm86_store_feedback_no_fence_negative_20261016.md`.

## H69 DRAM demand/completion decomposition — negative decision — 2026-10-16

The retained same-binary probe evidence (102 memory-retirement cells, 30
store-ack cells, launch-cycle controls, and direct DRAM streams) does not
separate read-service overlap, dirty residency, coherence, implicit EXIT drain,
or profiler envelope in one endpoint domain. Lower-bound results are mixed and
no hardware completion token/global order is observed. No timing-active model is
promoted; aggregate carded DRAM, typed tails, H66 GCC observers, and raw H41
authority remain unchanged. Report:
`docs/sm86_dram_demand_completion_negative_20261016.md`.

## H70 per-request L2 frontier observer — 2026-10-16

The shared-L2 event path now records bounded issue-to-fill, issue-to-response,
and model service-delay observer sums/maxima per launch. Full-first real-corpus
replay passed 356 roots (286 `ok`, 62 controls, 8 failures), 12,392 launches,
12 workers, and compressed artifacts at
`../sm86_replay_results/full_corpus_h70_l2_frontier_observer_20261016_v1/`.
The observer is timing-neutral and not NCU-scored. Report:
`docs/sm86_l2_frontier_observer_slice_20261016.md`.

## H71 terminal-EXIT store-drain arm — 2026-10-16

A matched same-binary RTX 3090 Ti no-fence/explicit-`MEMBAR.GL` probe isolated a
real stall-domain difference: no-fence stores report `sm__cycles_active.max`
10,345, `smsp__warps_issue_stalled_drain.sum` 253,415, and zero MEMBAR stalls;
the explicit-fence arm reports 11,730 cycles, 1,915 drain stalls, and 488,950
MEMBAR stalls. A bounded terminal-`EXIT` gate was implemented over the existing
typed store-ack stream, with stale-wake and past-ack event safeguards.

The experimental arm completed full-first replay over all 356 roots with 12
workers and compressed artifacts at
`../sm86_replay_results/full_corpus_h71_exit_drain_20261016_v2/`, restoring the
baseline 286/62/8 coverage. It worsened secondary mean APE for
`sm__cycles_active.avg` (18.0202%→18.1389%), `smsp__cycles_active.avg`
(17.5137%→17.6580%), and IPC projection (14.4865%→14.5529%), despite a tiny
GPC improvement. The card therefore sets
`exit_waits_for_service_completion=false`; default full replay is verified
non-regressing in
`../sm86_replay_results/full_corpus_h71_exit_drain_20261016_v3/` with an exact
H70 metric baseline. The implementation remains an explicitly disabled,
store-only experimental seam. Report:
`docs/sm86_exit_drain_slice_20261016.md`.

## H72 LDGSTS/LTC128B scope crosswalk — 2026-10-16

Full replay now retains static-SASS form partitions (`global_load_ldg`,
`global_load_constant`, `global_load_ltc128b`, `global_load_strong_gpu`, and
`ldgsts_async_copy`) and emits plain-LDG/LDGSTS opcode candidates. The
post-replay reporter distinguishes plain opcode mismatches from LTC128B scope
mismatches; it never relocates broad hierarchy traffic. Real controls confirm
LDGSTS is separate at the opcode endpoint but included in L1TEX global-op
sectors, while a SAM LTC128B.CONSTANT kernel is a 64× scope outlier.

Full-first replay completed 356 roots (286 `ok`, 62 controls, 8 replay errors),
12,392 full launches, exactly 12 workers, compressed at
`../sm86_replay_results/full_corpus_h72_ldgsts_scope_20261016_v1/`. Report:
`../sm86_replay_results/metric_report_h72_ldgsts_scope_20261016_v2/`. The H74
outlier is isolated as one LTC-specific secondary exclusion; LDGSTS opcode APE
is 0% (107/107 exact), while broad L1/L2/DRAM counters and timing are
unchanged. Report:
`docs/sm86_ldgsts_ltc128b_scope_slice_20261016.md`.

## H73 GCC/TEX source-client crosswalk — 2026-10-16

The same-binary CUDA13.2 `gcc_demand_h65` probe and independent NCU exports
confirm GCC/L1.5 traffic distinct from TEX, but two fresh NVBit capture attempts
(CUDA13.4 and CUDA13.2/g++-14 binaries) segfaulted during tracer initialization,
so no dynamic GCC protobuf crosswalk was produced. A post-replay warning seam
now compares H66 trace-derived GCC instruction/constant candidates with NCU GCC
endpoints without changing replay, DRAM, or metric selection. On the full H72
artifact it reports 286 instruction and 200 constant gross mismatches, falsifying
the candidates as direct GCC request predictors. Aggregate carded DRAM and H66
observers remain unchanged. Report:
`docs/sm86_gcc_tex_crosswalk_slice_20261016.md`.

## H74 L2 operation-split source observers — 2026-10-16

Full replay now emits read/write/atom-dot-alu/reduction L2 TEX sector, lookup
hit, and derived-miss rows from `L2SourceSplit`; no lookup policy, L2 state,
DRAM service, or timing changed. Full-first replay passed 356 roots (286 `ok`,
62 controls, 8 replay errors), 12,392 launches, exactly 12 workers, compressed at
`../sm86_replay_results/full_corpus_h74_l2_operation_split_20261016_v1/`.

The report `../sm86_replay_results/metric_report_h74_l2_operation_split_20261016_v2/`
scores read sectors 1.90% APE, read hits 5.44%, read misses 0%; write sectors
39.86%, write hits 39.88%, write misses exact/zero-target; atom-dot-alu and RED
rows exact. Promote the observer surface, but keep P3 write-coalescing/lookup
timing disabled pending same-binary XBAR crosswalk. Report:
`docs/sm86_l2_operation_split_slice_20261016.md`.

H73 dimensional correction: GCC candidate warnings now compare sector-domain
candidates only to `lts__t_sectors_srcunit_gcc.sum`; request-domain GCC metrics
are not mixed with sector candidates. Corrected report:
`../sm86_replay_results/metric_report_h73_gcc_crosswalk_20261016_v2/` (286
instruction and 264 constant-sector gross mismatches). A successful rebase
tracer capture exists in `/home/jakeshea/GPU_sims_notes/.../gcc_demand_h73/`,
but dynamic records lack constant-bank/address fields; static PC→SASS joining
is not endpoint-equivalent to NCU GCC requests. Current-branch tracer segfaults
remain a compatibility blocker.

## H75 L1TEX/XBAR write-coalescing probe — 2026-10-16

A locked RTX 3090 Ti CUDA13.2/g++-14 probe swept distinct-line, same-sector
cross-warp, and spaced same-sector stores (84 blocks, four warps, depth64).
NCU reports L2 `op_write`/XBAR sectors 86,016/86,016 for distinct lines,
5,699/5,699 for same-sector writes, and 6,651/6,651 for spaced writes; lookup
hits are 6,213 and 5,888 with zero misses. L1 global-store sectors remain
21,504 on same-sector arms. This proves coalescing exposure but falsifies a
simple lookup/tag or fixed-window timing law. No simulator timing/state changed.
Report: `docs/sm86_l1tex_write_coalescing_probe_slice_20261016.md`.

## H76 constant-client dynamic interval observer — 2026-10-16

Native trace metrics now retain dynamic static-SASS LDC/ULDC populations,
separate from CBANK address-domain sectors and GCC endpoint counters. Successful
rebase-tracer captures show 16,128 CBANK-tagged arithmetic records and 360,192
constant-mode records versus NCU GCC constant requests 1,008 and 562,212; these
are not equivalent. Full-first replay passed 356 roots (286 `ok`, 62 controls,
8 replay errors), 12,392 launches, 12 workers, compressed at
`../sm86_replay_results/full_corpus_h76_constant_client_interval_20261016_v1/`.
No DRAM/L2/timing change. Report:
`docs/sm86_constant_client_interval_slice_20261016.md`.

## H77 dynamic CBANK tracer repair prototype — 2026-10-16

The rebase tracer was repaired locally: immediate LDC/ULDC CBANK operands are
now instrumented, dynamic constant offsets/bank IDs use additive protobuf tags
15–17, and the replay-bench adapter accepts enhanced execution-info control
metadata. Real mode-1 and mode-0 captures completed and replayed one full
launch each. Mode 1 reports 344,064 LDC records, 2,688 ULDC records,
11,526,144 addresses, and 147 unique constant sectors; mode 0 is a valid
no-LDC control with 2,688 ULDC records and no metadata rejection. The runner
now falls back narrowly to `extra_info/enhanced_execution_info.json` when the
native sidecar is absent. The required full-first corpus replay after this
loader repair remains 286/62/8 over 12,392 full launches with exactly twelve
workers and compressed artifacts at
`../sm86_replay_results/full_corpus_h77_adapter_20261016_v2/`; report:
`../sm86_replay_results/metric_report_h77_adapter_20261016_v2/`. All H76
scored residuals are unchanged. This remains a notes/rebase tracer prototype;
no active-branch NVBit binary, GCC request collapse, L2 state, or DRAM timing
was promoted. Report: `docs/sm86_dynamic_cbank_tracer_slice_20261016.md`.

## H78 metric-unit crosswalk — 2026-10-16

The post-replay reporter now recognizes the native unit spellings emitted by
SM86 rows (`%`, `byte`, `request`, `thread/block`, `block/SM`, and
`register/thread`) and applies explicit Kbyte/Mbyte-to-byte scaling only for
named DRAM/shared-memory metrics. Re-reporting the H77 full-first artifact
increased compared metrics from 40 to 63 and left only the intentional
`gpu__time_duration.sum` clock mismatch. Full replay coverage remains
286/62/8 over 12,392 launches with twelve workers; no simulator/cache/DRAM/
timing behavior changed. Report:
`docs/sm86_metric_unit_crosswalk_slice_20261016.md`.

## H79 packet-aware L1 lookup observer — 2026-10-16

The existing L1 request-boundary audit now separates ordinary-global
same-request merges from older pending and LDGSTS merges. Full replay emits a
bounded packet-aware hit/miss candidate and `[candidate, baseline]` interval
without changing cache state, MSHR ownership, L2/DRAM service, or timing. The
full-first v2 replay remains 286/62/8 over 12,392 launches with twelve workers:
`../sm86_replay_results/full_corpus_h79_packet_observer_20261016_v2/`; report:
`../sm86_replay_results/metric_report_h79_packet_observer_20261016_v2/`.
On 106 scope-compatible joins, hit APE improves 30.71%→17.50% and miss APE
32.39%→19.53%; false-positive zero-target hits fall 22/29→8/29. The
authoritative raw/all-row miss view regresses on the LTC128B scope outlier, so
the candidate is retained observer-only and the baseline remains the selected
default prediction. Report:
`docs/sm86_l1_packet_aware_lookup_slice_20261016.md`.

## H80 L2 write-packet coalescing — 2026-10-16

Full replay now selects a bounded per-SM line/sector write-packet candidate
within an 8-cycle issue window only for
`lts__t_sectors_srcunit_tex_op_write.sum`; the original one-sector-per-request
observer is retained separately. L2 state, lookup-hit/miss rows, DRAM demand,
writeback, and timing are unchanged. Full-first v2 replay remains 286/62/8 over
12,392 launches with twelve workers at
`../sm86_replay_results/full_corpus_h80_write_packet_observer_20261016_v2/`;
report:
`../sm86_replay_results/metric_report_h80_write_packet_observer_20261016_v2/`.
The selected write-sector row improves mean APE 39.86%→9.37%, P90 100%→0%,
and exact nonzero joins 77→100. Report:
`docs/sm86_l2_write_packet_observer_slice_20261016.md`.

## H81 L2 write lookup packet composition — 2026-10-16

The H80 packet candidate now composes with operation-split write lookup rows:
candidate hits are bounded by packet sectors and misses are the checked
remainder; request-domain baselines remain diagnostic. Full-first replay is
286/62/8 over 12,392 launches with twelve workers at
`../sm86_replay_results/full_corpus_h81_write_lookup_packet_20261016_v1/`;
report:
`../sm86_replay_results/metric_report_h81_write_lookup_packet_20261016_v1/`.
Write-sector mean APE is 9.37%, write lookup-hit APE 9.36%, and write lookup
misses remain exact zero-target rows. Aggregate L2, DRAM, timing, and IPC are
unchanged. Report: `docs/sm86_l2_write_lookup_packet_slice_20261016.md`.

## H82 DRAM route-concentration diagnostic — 2026-10-16

Full replay now emits normalized maximum candidate partition/slice demand
fractions and a joint GPC-frontier observer; aggregate carded DRAM, L2 state,
and all timing remain unchanged. Full-first replay v1 remains 286/62/8 over
12,392 launches with twelve workers at
`../sm86_replay_results/full_corpus_h82_route_concentration_20261016_v1/`;
report:
`../sm86_replay_results/metric_report_h82_route_concentration_20261016_v1/`.
The diagnostic is present on 286 full launches (median 175,631 ppm, P90
500,000 ppm, 23 above 500,000); GPC/SM APE is bit-equivalent to H81. Independent
review rejects partition/queue timing promotion until locked one-factor demand
and exact critical-path joins exist. Report:
`docs/sm86_dram_route_concentration_diagnostic_slice_20261016.md`.

## H83 IPC readiness concentration observer — 2026-10-16

Full replay now emits dominant exclusive readiness-cause fraction and total
limiting-cycle observers alongside the existing IPC frontier interval; no
scheduler/cache/DRAM/timing behavior changes. Full-first replay v1 remains
286/62/8 over 12,392 launches with twelve workers at
`../sm86_replay_results/full_corpus_h83_ipc_readiness_20261016_v1/`;
report:
`../sm86_replay_results/metric_report_h83_ipc_readiness_20261016_v1/`.
The observer is present on 286 full launches (median 425,114 ppm, P90 613,776,
103 above 500,000); GPC/SM APE is unchanged. No IPC timing promotion is
justified without held-out endpoint-matched probes. Report:
`docs/sm86_ipc_readiness_concentration_slice_20261016.md`.

## H84 integrated SM86 accuracy dashboard — 2026-10-16

The latest H83 full-first artifact consolidates the default composition and
residuals: L1 sectors 0.21%, L2 TEX sectors 6.76%, L2 op-write sectors 9.37%,
L2 op-write lookup hits 9.36%, DRAM reads 26.84%, GPC 12.90%, and SM-active
18.02% mean APE on their retained identity/scope-compatible joins. H80/H81
write endpoints are selected; H82/H83 route/readiness surfaces remain
observer-only. All frontiers and baseline counters are unchanged. Report:
`docs/sm86_integrated_accuracy_dashboard_20261016.md`.

## H85 independent GCC/constant demand matrix — 2026-10-16

Locked RTX 3090 Ti CUDA13.2/g++-14 probe matrix (arithmetic/one-LDC/two-LDC,
repetitions, footprints) shows GCC requests scaling 7,143→590,880 while L2
GCC sectors move only 2,128→2,240 and DRAM reads 296→316. Two-LDC reaches
2,127,353 requests but only 3,248 L2 GCC sectors and 452 DRAM sectors. This
falsifies a request-count-to-DRAM rule; no P4 multi-client timing or DRAM
selection change is made. Raw matrix and environment are in the notes probe
directory. Report: `docs/sm86_gcc_demand_matrix_slice_20261016.md`.

## H86 integrated adversarial review — 2026-10-16

Workspace Cargo checks/release build, Python compilation, diff checks, zstd
artifact tests, coverage invariants, and all 12,392 promoted write conservation
rows pass on the latest H83 artifact. Review confirms NCU remains post-replay
only; H80/H81 alter only operation-split write predictions; H82/H83 are
observer-only; H85 blocks request-count-to-DRAM fitting. Remaining gates are
L1 lookup scope, GCC/multi-client DRAM ownership, and IPC retirement timing.
Report: `docs/sm86_integrated_adversarial_review_slice_20261016.md`.

## H87 L1 lookup form-scope review — 2026-10-16

Independent H83 trace/NCU stratification shows H79 improves the dominant
`global_load_constant` stratum (hit 30.86%→16.95%, miss 31.34%→17.53%) but
worsens four LTC128B rows (miss 725.01%→1,683.10%); mixed LDGSTS forms are also
distinct. The baseline remains selected and the packet candidate remains an
observer/interval pending per-request client identity. Report:
`docs/sm86_l1_lookup_form_scope_review_slice_20261016.md`.

## H88 constant-client reuse observers — 2026-10-16

Full replay now emits dynamic constant address-record coverage and reuse ratios
without entering hierarchy state or timing. Full-first replay v1 remains
286/62/8 over 12,392 launches with twelve workers at
`../sm86_replay_results/full_corpus_h88_constant_reuse_20261016_v1/`;
report:
`../sm86_replay_results/metric_report_h88_constant_reuse_20261016_v1/`.
Observers cover 286 full launches; GPC APE 12.9008% and DRAM-read APE 26.8371%
are unchanged. Report: `docs/sm86_constant_reuse_observer_slice_20261016.md`.

## H89 repaired dynamic-CBANK H85 crosswalk — 2026-10-16

The repaired external CUDA13.2 NVBit tracer captured real H85 mode-0/mode-1
launches with dynamic CBANK payloads; both replayed successfully through the
active adapter. Mode 1 reports 344,064 LDC, 5,376 ULDC, 712,320 references,
22,794,240 addresses, and 147 unique sectors versus H85 NCU 590,041 GCC
constant requests and 2,240 GCC L2 sectors. Transport is now validated, but
domains remain non-equivalent; no P4 request conversion or timing is enabled.
Report: `docs/sm86_repaired_cbank_crosswalk_slice_20261016.md`.

## H90 complete simulator-vs-NCU accuracy comparison — 2026-10-16

Published a complete family-grouped table for every finite comparable H88
metric, including mean/median/P90 APE, compared/exact/scope counts, zero-target
semantics, unavailable inventory, and the distinction between 12,392 replayed
launches and 107 exact NCU identity joins. Report:
`docs/sm86_accuracy_comparison_report_20261016.md`.

## H91 expanded accuracy report with ordered-fallback joins — 2026-10-16

Per user authorization, the 179 ordered-fallback pairs are now included in a
separate expanded report. These are same-workload pairings despite differing
trace/NCU kernel-name strings; the strict exact-only report remains unchanged.
The expanded report scores 286 successful root pairs (107 exact + 179 ordered
fallback) and is at:
`../sm86_replay_results/metric_report_h91_ordered_fallback_20261016_v1/`.
Human-readable complete table:
`docs/sm86_accuracy_comparison_ordered_fallback_20261016.md`.

## Current SM86 refinement slice — H92 form-qualified L1 lookup selection

H92 selects a narrow trace-only endpoint mixture for SM86 global-load L1
lookup rows: packet-aware constant-only, shape-qualified LDGSTS all-miss, and
cold plain-LDG packet arms. Mixed clients/LTC128B retain baseline selection;
no cache, DRAM, timing, or IPC state changes.

Four real roots and the complete full-first corpus passed. The full corpus has
356 roots (286 `ok`, 62 controls, 8 errors), 12,392 launches, exactly twelve
workers, and compressed artifacts at
`../sm86_replay_results/full_corpus_h92_form_lookup_20261016_115517/`; report:
`../sm86_replay_results/metric_report_h92_form_lookup_20261016_v1/`. Only the
L1 global lookup hit/miss and derived hit-rate rows changed versus H88. On
scope-compatible exact joins, hit APE is 18.0702% and miss APE is 18.8472%,
both below 20%, with no other scored metric regression. Raw/all-row hit APE
is 18.0702%; raw miss is 45.0262% because the known LTC128B/scope-mismatch
row remains unresolved. A same-binary per-request form/client crosswalk is the
next promotion gate.

## Current SM86 refinement slice — H93 aggregate L2 TEX packet conservation

H93 composes the H80/H81 write-packet endpoint into aggregate SM86 L2 TEX
sector/hit rows, retaining baseline request-domain diagnostics and exact miss
conservation. Full-first replay completed 356 roots (286 `ok`, 62 controls,
8 errors), 12,392 launches, twelve workers, and compressed artifacts at
`../sm86_replay_results/full_corpus_h93_l2_packet_conservation_20261016_123337/`;
report: `../sm86_replay_results/metric_report_h93_l2_packet_conservation_20261016_v1/`.
L2 TEX sectors improved to 2.9180% APE, lookup hits to 3.8618%, and misses
remain 0.0%; no other scored metric regressed. The next active seam is a
same-binary U16/partial-line L1 probe and per-request form identity.

## Current SM86 refinement slice — H94 per-client L1 lookup provenance

H94 carries same-request merge provenance by ordinary-global, constant, and
async-copy client. Mixed constant/plain launches now use the constant-client
packet candidate; LTC128B and mixed async forms retain baseline selection.
Replay state/timing/DRAM/IPC are unchanged.

Four real roots and the complete corpus passed: 356 roots (286 `ok`, 62
controls, 8 errors), 12,392 launches, twelve workers, compressed output at
`../sm86_replay_results/full_corpus_h94_l1_client_lookup_20261016_125135/`;
report `../sm86_replay_results/metric_report_h94_l1_client_lookup_20261016_v1/`.
L1 lookup hit/miss scope-compatible APE is 17.7036%/18.7117%, with no other
scored regression. The independent U16 half-line probe falsified a universal
U16 sector multiplier; broader per-sector-validity evidence is required next.

## Current SM86 refinement slice — H99 LTC128B and H102 local frame/coalescing

The complete full-first H99 replay (356 roots: 286 `ok`, 62 controls, 8
errors; 12 workers; 12,392 launches) promotes an observer-only all-miss L1
endpoint for `LTC128B`. Global L1 lookup-miss APE improves to 18.1662% from
18.7117%; no other scored metric changes.

H102 then composes a source-derived local endpoint. Dynamic-index local
records use `raw + (frame256 - transfer_width)` only when their raw sector
count exceeds request count; scalar volatile/register-spill records use the
validated payload candidate. Complete replay artifacts are
`../sm86_replay_results/full_corpus_h102_local_hybrid_20261016_143358/` with
report `../sm86_replay_results/metric_report_h102_local_hybrid_20261016_v1/`.
Local load sectors, local store sectors, and local-load lookup hits are now
1.7972%, 0.3353%, and 1.7972% APE respectively; local lookup misses are exact
on all 107 joins. All non-local metrics remain bit-equivalent to H99.

The remaining global L1 lookup residual is 17.7036% hit / 18.1662% miss and
elapsed IPC is 12.3073%; no sub-10% claim is made for those unresolved seams.
Rejected H95d/H97/H98 evidence is consolidated in
`docs/sm86_rejected_l1_local_experiments_20261016.md`.

H103's sector-fill-owner subtraction was rejected after complete replay
regressed global L1 lookup hit/miss APE to 33.74%/62.44%. Its owner-kind audit
fields remain observer-only; H102/H99 endpoint selection is retained.

Final-tree confirmation: after rebuilding the release binary with H103 rolled
back, H105 full-first replay completed 356 roots (286/62/8), 12,392 launches,
twelve workers. Artifact:
`../sm86_replay_results/full_corpus_h105_final_release_20261016_145305/`;
report: `../sm86_replay_results/metric_report_h105_final_release_20261016_v1/`.
It reproduces H102 local accuracy and H99/H93 non-local accuracy.

## Current SM86 global-L1 slice — H108

H106 resident-sector clamping plus H107/H108 exact cold CONSTANT/LDGSTS and
U16 geometry rules were evaluated on complete replay. H108 reaches global L1
lookup hit/miss APE 10.126%/10.231%; local H102, L2 H93, H99 LTC128B, DRAM,
and IPC rows do not regress. The remaining gap is isolated to mixed/high-reuse
U16 forms and is not closed by fitting. Final H108 artifacts:
`../sm86_replay_results/full_corpus_h108_l1_u16_geometry_20261016_152102/` and
`../sm86_replay_results/metric_report_h108_l1_u16_geometry_20261016_v1/`.

H111's initial GCC constant-sector DRAM addition was rejected after complete
replay (DRAM read APE 203.56%); it was rolled back. A correct Proposal-2
implementation requires per-GPC shared-L2 deduplication and is not yet
promoted.

## H113-H117 final continuation state — 2026-10-16

H113 replaced the SM86 DRAM endpoint's executed 32-byte instruction-sector
term with a dynamic aligned-128B instruction-line term. The complete replay
preserved 356/286/62/8 coverage and 12,392 full launches, but DRAM-read APE
only moved 26.84% -> 25.55%. H114 then added a separate static-SASS aligned
128B instruction-line envelope, still observer/state neutral. Its complete
replay reduced scope-compatible DRAM-read APE to **19.47%** (58 joins), with
all L1/L2/cycle surfaces unchanged. This is the retained SM86 DRAM point
term; it is a source-qualified GCC instruction-footprint hypothesis, not a
request-count multiplier or NCU fit.

H115 promoted a narrow mixed-client L1 lookup ownership rule. For the traced
one-sector CONSTANT + plain-LDG shape, ordinary-client same-request and
older-sector-pending merges are added to the packet candidate and clamped to
the global lookup denominator. Full-first replay preserved 356/286/62/8 and
12,392 launches. Global L1 lookup APE is now **9.155% hit / 9.828% miss**;
L2 TEX remains **2.918% sectors / 3.862% hits / 0% misses**, and local H102
remains unchanged. This closes the global L1 target without changing cache
state, DRAM, or timing.

H116 added an observer-only warp-uniform `LDG.*.CONSTANT` witness to the
validated trace memory record. It reports uniform/divergent instruction and
sector populations but is not selected for DRAM or cache predictions. The
complete replay again preserved 356/286/62/8 and 12,392 launches; H116 report
is `../sm86_replay_results/metric_report_h116_uniform_constant_observer_20261016_v1/`.
The witness is not a direct GCC predictor: grouped-GEMM launches carry
thousands of uniform sectors while the independent GCC residual remains nearly
fixed, so uniform-sector DRAM promotion is rejected.

H117 tested the architecture-card peak DRAM service scenario (1008 GB/s)
against the default independently measured 825 GB/s hypothesis on the full
corpus. It regressed GPC APE **12.90% -> 12.97%**, SM-active APE
**18.02% -> 18.10%**, and elapsed/active IPC **12.31%/14.49% ->
12.40%/14.58%**. The peak scenario is rejected; the 825 GB/s default remains.
The residual IPC/GPC error is not closed by a trace-identifiable service-rate
choice and remains bounded by the locked-clock/DVFS identifiability boundary.

## H121-H122 split store-drain timing — negative decision — 2026-10-17

An opt-in candidate added a separate SMSP/L1TEX admission-drain horizon for
terminal `EXIT`, distinct from the existing L2 visibility acknowledgement.
H121 full replay preserved 356/286/62/8 and 12,392 launches, but improved
elapsed IPC only 12.307%→12.204% and GPC 12.901%→12.875% while worsening
SM-active/SMSP to 18.218%/17.753%. H122 then gated the horizon on a full
five-entry admission queue, reducing the regression but still worsening
SM-active/SMSP to 18.114%/17.630%; elapsed IPC/GPC were 12.286%/12.889%.
Both are rejected under the no-regression rule. The drain field remains an
explicit opt-in research seam; the default H116 candidate and all traffic
metrics are unchanged.
