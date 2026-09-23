# SM86 current-source replay check — 2026-09-23

The current release binary was run against a retained post-merge SM86 trace.
The replay exited zero. Retained and current outputs had identical `counters`
and `coverage`. `metric_predictions` changed only for `exec_cycles_gpc_max`:
the active source now emits the model-relative `binding_cycles` result instead
of the historical `required_unimplemented` marker. Internal readiness/timing
diagnostics also differ because the active worktree includes later
scheduler/front-end changes. This is a model-revision check, not event/dense
equivalence evidence or a hardware accuracy claim.

After the final adversarial fixes, `sim-run` was rebuilt in release mode and
the command below was rerun. It again exited zero with identical counters and
coverage; the emitted model-relative binding-cycle value was `11102`. Timing
and readiness diagnostics remain revision-relative rather than byte-identical
historical output.

Machine-readable comparison: `docs/sm86_current_source_replay_20260923.json`.

Command shape:

```bash
cd /home/jakeshea/GPU_simulator_rebase
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
sed 's#sim/target/release/sim-run#/home/jakeshea/GPU_simulator_rebase/sim/target/release/sim-run#' \
  results/captures/sm86_postmerge_campaign_small_20260909_v1/evidence/raw/simulator/gemm_hmma_bf16_sm80/command.txt \
  | sed 's# --ncu-csv .*##' | bash > /tmp/sm86_current_replay.json
```
