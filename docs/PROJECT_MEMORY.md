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

For future replay work, run **full replay only**. Do not spend time producing
fast-mode artifacts; use the complete integrated `l1-l2` path with all
supported features enabled by default and twelve workers.
