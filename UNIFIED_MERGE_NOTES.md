# Unified merge notes

## Inputs

* SM120 implementation base: `GPU_simulator_5090` HEAD
  `818a1dbd7583c273ee1996caa75a18c3a4ac90a5`.
* SM86 implementation: `GPU_simulator` HEAD
  `0c8791f5d2b87c45d409f23ce6f0ad67b708e50b`.
* Shared history: `f3ca5b2f0d3b12ec9c720df408de7908810d8617`.

The imported merge deliberately excludes paper/thesis/manuscript paths. Capture
and final-result binaries are not part of the source tree; only four small
copied traces are retained under `verification_inputs/` for reproducible
smoke checks.

## Architecture dispatch

`arch_profiles/loader.py` validates `(arch_id, sku, build_arch)` identity.
`sim-run` keeps one common execution path and dispatches architecture-specific
behavior through the card and `RunConfig`:

* SM86 selects the measured compatibility card, P-chase geometry, SM86 L1 set/
  way scenarios, SM86 observer/candidate counters, and the bounded 825 GB/s
  effective service-rate scenario used by the full replay runner.
* SM120 selects the effective L1/L2 model, TMA/async sidecar projections,
  conditional IPC/latent envelope, and the card's explicitly unvalidated
  1792 GB/s prototype service hypothesis. SM120 physical geometry remains a
  typed gap; no SM86 geometry is used as a fallback.
* Full replay emits architecture-scoped metric projections for both lanes;
  fast replay remains trace-derived and emits no native metric projection.
* SM120 store lookup projection retains the original SM120 present-sector
  behavior; the SM86 store lookup candidate is selected only for SM86. This
  preserves the 3090/5090 baseline distinction in the shared L2 code.

The replay runner is architecture-neutral despite its historical filename:
`--target-arch auto|sm_86|sm_120` must agree with the selected card and the
capture metadata. It writes compressed `.jsonl.zst` outputs and supports the
standard twelve-worker corpus run or an explicit one-worker timing baseline.

## Verification

* `cargo check --manifest-path sim/Cargo.toml --workspace` — passed.
* `cargo test --manifest-path sim/Cargo.toml --workspace` — passed (real-trace
  tests remain ignored unless their environment variables are supplied).
* Python syntax checks for architecture loader, replay runner, metric report,
  replay bench/compare, and JSONL I/O — passed.
* Representative copied SM86 scalar/cp.async and SM120 scalar/TMA full replays
  — all `ok`; fast frontier replays also `ok`.
* `VERIFICATION_REPORT.md` compares selected counters against the corresponding
  `final_results` rows; all four representative full-replay cases match on
  status, architecture, launch count, instruction/record counts, front-end
  counters, L1/L2 core counters, DRAM read sectors, and metric prediction count.

The complete 339-row SM120 corpus was rerun in both DRAM-disabled and
carded-DRAM-enabled full modes, and all 356 SM86 attempts were rerun in full
mode. Final status/launch totals match the retained baselines exactly. The
loop’s root-cause fixes and final zero-numeric-difference audit are recorded in
`FULL_REPLAY_AUDIT.md` and `FULL_NUMERIC_COMPARISON.json`.
