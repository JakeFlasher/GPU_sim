# SM120 host runbook

The active lane is the RTX 5090 (`sm_120`) full-trace replay.  The complete
capture corpus is `/root/sm120_rtx_5090_capture`; independent NCU/CUPTI records
are joined after replay and are never simulator inputs.  Full replay is always
run before the complementary fast frontier.

## Build

```bash
cd /root/GPU_simulator_rebase/sim
cargo build --workspace --all-features
cargo build --workspace --all-features --release
```

## Replay the complete corpus

Use the retained-corpus driver from the repository root:

```bash
cd /root/GPU_simulator_rebase
python3 tools/sm120_replay_bench.py \
  --capture-root /root/sm120_rtx_5090_capture \
  --work-root /dev/shm/sm120_rtx_5090_work \
  --mode full
python3 tools/sm120_replay_bench.py \
  --capture-root /root/sm120_rtx_5090_capture \
  --work-root /dev/shm/sm120_rtx_5090_work \
  --mode fast
```

The full run must report every captured row and launch.  The fast run is a
performance diagnostic only.  Keep replay rows, joins, residuals, and timings
under the work root; do not add small synthetic fixtures or acceptance gates.

## Direct replay

```bash
sim/target/release/sim-run --sm120-alpha --mode flat \
  --trace TRACE_DIR --control CONTROL_BITS.json \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json
sim/target/release/sim-run --sm120-alpha --fast-frontier-only \
  --trace TRACE_DIR --control CONTROL_BITS.json \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json
```

## Independent NCU comparison

Use the NCU/trace artifacts already captured in
`/root/sm120_rtx_5090_capture`.  Join by launch identity, report the timing
domain and unit explicitly, and keep unmatched rows and native timing gaps
visible.  Never fit, offset, or feed NCU values into a replay.

## Tracer maintenance

The NVBit producer remains under `tools/nvbit_tracer`.  Build it with the
pinned CUDA/protobuf environment when a new real workload capture is required;
write new captures directly to the corpus/work roots.  Do not create local
fixture bundles or small acceptance workloads.
