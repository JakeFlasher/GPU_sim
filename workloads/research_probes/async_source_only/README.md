# Store-free async-source probe

This small SM86 probe is the next discriminator for the simulator's typed
retirement tails. Its target kernel is one warp with fixed-depth
`cp.async.ca` (L1) or `cp.async.cg` (streaming/DRAM) positions, one
`commit_group`/`wait_group 0`, volatile shared reads, and no target-kernel
global store or CTA barrier. Host source initialization is outside the
profiler markers. The empty inline asm at kernel end keeps the clock/token
live without creating a result store.

Build and run a bounded marker-scoped smoke:

```bash
make -C workloads/research_probes/async_source_only clean all
CUDA_VISIBLE_DEVICES=0 \
  workloads/research_probes/async_source_only/async_source_only \
  --groups 64 --depth 4 --route l1 --markers on
```

For an NVBit capture, use the shared `capture_nvbit.py` helper (the capture is a pipeline
smoke, not a replacement for the frozen LDGSTS matrix). NCU values remain
comparison-only; the only simulator inputs are the complete NVBit trace, its
static control sidecar, and the SM86 cards. A clean result can remove the
global-store/coherence confound from the simulator comparison, but it cannot
prove a physical queue release event or justify a timing constant by itself.

The retained depth-4/depth-8 L1 and DRAM slice is reduced with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze.py \
  --root results/captures/sm86_ldgsts_burst_release_20260830_v4 \
  --output results/captures/sm86_ldgsts_burst_release_20260830_v4/analysis/async_source_only_report.json
```

The capture helper is shared with the LDGSTS probe as
`workloads/research_probes/ldgsts_burst_release/capture_nvbit.py`; pass
`--binary .../async_source_only`, `--mode ncu`, and the probe arguments after
`--target-args`.
