# Long-term session notes

The immutable session rules are in [`SESSION_RULES.md`](SESSION_RULES.md).
This file is deliberately a current-state note rather than a duplicate.

* Workspace: `unified_gpu_simulator`, Git branch `main`.
* Source reference: `/home/jakeshea/5090_branch/` remains strictly read-only;
  all builds, tests, and edits happen in this workspace.
* Supported lanes: SM86 / RTX 3090 Ti and SM120 / RTX 5090, selected by card
  and capture metadata.
* Full replay: the required SM120 lane is the 339-capture DRAM-enabled/card
  configuration; SM120 DRAM-disabled replay is out of scope. SM86 has 356
  attempted captures with 286 successful rows. Complete runs use twelve
  workers and compressed JSONL.
* Latest audit: `FULL_NUMERIC_COMPARISON.json` reports zero missing/extra rows,
  required-metadata differences, and common numeric differences in all three
  campaigns.
* Current docs: `docs/CONTRACT.md`, `docs/REPLAY.md`, `docs/RESULTS.md`, and
  `docs/HISTORY.md`. Old plans and experiment diaries are not active policy.
* Known limitation: SM120 effective geometry and completion boundaries are
  typed conditional models, not proprietary RTL recovery; NCU remains an
  independent post-replay target.
* Operating split (user directive): GPT-6 Astra is executioner/orchestrator only—implements, runs, and code-reviews; Gemini 4 Argon High invoked through `agy --dangerously-skip-permissions --model gemini-4-argon-high` is the read-only brain for algorithms, ideas, and insight/adversarial review. Gemini must not code. Any blocker, ambiguity, or uncertainty requires consultation with Gemini before proceeding; do not rely on instinct.
* Active objective: reduce APE below 10% for major IPC and L1/L2-related NCU metrics across complete SM120 (339 captures) and successful SM86 (286 captures) full replays, preserving independent-target separation and twelve-worker replay.
* Latest evidence: the required DRAM-enabled SM120 full lane remains green at
  339/339 `ok` with twelve workers; historical disabled-mode artifacts are
  retained but no new disabled replay is run. SM86 remains 286 `ok`/62
  control/8 replay errors. The SM86 LDG.CONSTANT opcode scope fix reduced that
  endpoint from 44.8% to 1.36% APE. Unified-L1, generic-L2, DRAM-client, and
  timing residuals remain above 10% because their NCU endpoints include scope
  or runtime state absent from the trace contract.
* Do not promote a generic-L2 source-lane sum or clamp: Gemini's read-only review and the launch-66 accounting identity show the 96 excess store hits originate upstream in simulated L1/TEX-to-L2 store traffic, not in the report selector. Missing SM86 shared-memory carveout metadata is an explicit unidentifiable input gap.
* Latest full-enabled progress: trace-semantic SM120 store-packet and TMA-store-cold-miss observers completed 339-capture and 286-successful-SM86 full replays with 12 workers. Generic SM120 L2 hit APE improved from ~57% to ~24.3%; operation-read, unified-L1, DRAM, and IPC residuals remain above 10%.
* Replay scope directive: do not run fast replay. Accuracy and correctness work uses only complete full replay with all supported features enabled by default (integrated `l1-l2`, card-selected architecture behavior, and twelve workers); fast-mode artifacts are not required.
* SM120 replay scope refinement: do not run the SM120 DRAM-disabled full replay. The required SM120 accuracy lane is the full DRAM-enabled/card-feature configuration; continue SM86 full replay as required.
* 2026-10-18 reuse-model consultation: Gemini advised a shadow-only,
  trace-derived reuse-distance/reuse-interval observer using 128-byte line
  identities plus four 32-byte sector masks, with separate per-SM and L2
  reference streams, explicit cold/pending/partial-sector states, and
  report-only MRCs before any promotion. This is design advice, not hardware
  truth: do not hard-code SM120 geometry/hash/replacement, MSHR windows, or
  store/TMA policy; validate them against cards/traces and keep NCU out of
  fitting. See `docs/REUSE_MRC_CONSULTATION.md` and the appended
  `consultation_prompt.txt` for the full record and literature links.
* Follow-up Gemini gate: do not promote the per-warp RI proxy or a sampled
  fully-associative RD curve into active L1/L2 scoring/timing. The live
  cycle-interleaved L1 and event-driven L2 models are more faithful; a future
  SHARDS/Fenwick observer is capacity-bound diagnostic only until chronology,
  TMA timing, carveout, and set/hash/replacement metadata are proven.
* 2026-10-18 async-client scope slice: a trace-qualified SM120 report-only
  exclusion of `l1_async_older_pending_merges_global` from external global-op_ld
  L1 hit projections passed complete 12-worker full replays (SM120 339/339;
  SM86 286/62/8). It slightly improved SM120 unified/global L1 APE while
  leaving L2/DRAM/IPC unchanged; physical state/timing and SM86 were untouched.
  Do not broaden it to ordinary pending merges or promote LTC128B/LDGSTS L2
  sibling/volume corrections without a new Gemini-reviewed crosswalk.
* 2026-10-18 typed L2 shadow: the event path now reports static client-scope
  request/resident/pending/fill partitions and a non-promoted LTC128B
  potential-sibling volume. The final refinement tags write, atomic,
  reduction, and async-bypass requests at emission; unqualified bypasses stay
  `Unknown`. It leaves physical L2/DRAM/timing intact and passed both
  twelve-worker full lanes (SM120 339/339, SM86 286/62/8); see
  `docs/L2_SCOPE_SHADOW.md` and `FULL_NUMERIC_COMPARISON_L2_SHADOW3.json`.
  Read-scope masks conserve request sectors; write-class pending/fill arms are
  zero by construction, and same-cycle folded sectors use the lead request's
  diagnostic scope. These are attribution boundaries, not hardware-hit claims.
  Gemini's closure gate freezes active L2/DRAM changes: remaining residuals
  need unobservable SM120 geometry/fill-buffer/TMA/carveout or endpoint-scope
  evidence, not NCU fitting or count balancing.
* Follow-up candidate audit: TMA-shadow L2 misses must not replace raw TMA
  payload DRAM demand (full-row evaluation worsened). A pure-LTC resident plus
  older-pending L1 numerator is mechanistically plausible but remains
  shadow-only: 300 zero-target rows and two false positives, plus unproven
  sector-unit conservation, fail the promotion gate.
* Gemini definitive stop gate: no remaining active trace-only IPC/L1/L2
  projection is defensible. Resume only after TMA descriptor/coordinate and
  launch-cache metadata, disaggregated tag/MSHR/fill/stall counters, or
  isolated LTC128B/TMA microbenchmarks are obtained. Preserve complete
  twelve-worker replay artifacts and report residuals/SM86 coverage boundaries
  without tuning against NCU.
* Follow-up unified-L1 numerator-consistency experiment was rejected after
  full 12-worker replay: common SM120 NCU mean APE worsened 0.5882→0.5966
  (median 0.5641→0.5752). Code was reverted; the hardware ratio identity does
  not make mixed endpoint/client terms interchangeable.
