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
