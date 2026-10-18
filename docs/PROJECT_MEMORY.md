# Current project memory

This is a two-lane simulator, not two independent executables:

* **SM86 / RTX 3090 Ti** uses `arch_profiles/cards/sm_86.json`, the P-chase
  card, and SM86-scoped candidates/observers.
* **SM120 / RTX 5090** uses `arch_profiles/cards/sm_120-rtx-5090.json`, TMA and
  async sidecars, and effective conditional L1/L2 models.

The common path is:

```text
trace + sidecars -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run
```

Architecture identity comes from the card and must agree with capture metadata.
NCU values are post-replay comparison data only. Full replay is integrated
`l1-l2`, emits compressed JSONL, and uses twelve workers for complete corpora;
fast replay is a separate trace-derived mode.

Read [`CONTRACT.md`](CONTRACT.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), and
[`REPLAY.md`](REPLAY.md) before changing the model. Current verification is
summarized in [`RESULTS.md`](RESULTS.md). Do not treat old branch plans or
dated experiment diaries as active instructions.

## Operating split for accuracy work

GPT-6 Astra is the executioner: it edits code, runs experiments, and performs
code review. Gemini 4 Argon High is the read-only insight reviewer, invoked via
`agy --dangerously-skip-permissions --model gemini-4-argon-high --effort high --mode plan`;
it supplies algorithm design and adversarial reasoning but must not code. Any
blocker, ambiguity, or uncertainty must be consulted with Gemini before action.
The active accuracy target is <10% APE for major IPC and L1/L2-related NCU
metrics across all 339 SM120 captures and 286 successful SM86 captures in full
replay.

The latest 12-worker accuracy campaign is complete and reproducible. SM86's
form-qualified `LDG.CONSTANT` opcode projection now agrees closely with NCU
(1.36% full-report APE), while unified-L1, generic-L2, DRAM-client, and timing
endpoints remain explicit scope/effective-model residuals. Do not hide the
generic-L2 store-traffic discrepancy with a count-balancing observer or fit
against NCU labels; the current evidence points to an upstream write
coalescing/cross-launch semantic gap. Per-launch SM86 shared-memory carveout
is not present in trace sidecars and cannot be inferred honestly from the
card maximum.

The latest full-enabled-only observer slice completed both required corpora
with twelve workers. SM120 generic source-TEX L2 hit APE fell to about 24.3%
by using decoded store-packet identity and treating complete TMA-store payload
sectors as cold misses; replay timing, cache state, and DRAM demand were not
changed. This is a real full-replay improvement, but the <10% objective is not
yet met because ordinary operation-read/client scope, unified-L1 scope, DRAM
clients, and IPC frontiers remain unresolved.

For future replay work, run **full replay only**. Do not spend time producing
fast-mode artifacts; use the complete integrated `l1-l2` path with all
supported features enabled by default and twelve workers.
The SM120 DRAM-disabled full-replay variant is also out of scope; use the
DRAM-enabled/card-feature SM120 full lane and the SM86 full lane.

On 2026-10-18 Gemini's read-only reuse-distance/reuse-interval consultation
recommended a trace-derived, shadow-only MRC observer (128-byte lines plus
four 32-byte sector masks, separate per-SM/L2 streams, and explicit cold/
pending/partial-sector states). It did **not** authorize coding or promotion:
SM120 geometry/hash/replacement, MSHR horizons, and store/TMA policy remain
unverified research gaps, and NCU values must not fit the model. The complete
record is `REUSE_MRC_CONSULTATION.md` and `../consultation_prompt.txt`.

The final Gemini gate rejected promotion of the per-warp interval proxy (or a
sampled fully-associative RD curve) into scored L1/L2 predictions or timing:
the live cycle-interleaved L1 and event-driven L2 models are more faithful.
Future SHARDS/Fenwick work is capacity-bound diagnostic only until cross-warp
chronology, TMA timing, carveout, and set/hash/replacement metadata are proven.
