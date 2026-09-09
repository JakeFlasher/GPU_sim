# Async-copy / CTA-barrier overlap probe

This one-CTA `sm_86` probe isolates asynchronous-copy completion, independent
work, and CTA rendezvous. The five protocols are compile-time kernel
specializations in one binary so predicated-off barriers never enter a replay
trace. Locality, gap, and CTA width remain runtime arguments, so locality
comparisons use the same specialized kernel and address-generation
instructions.

The final v4 campaign is complete under
`results/captures/sm86_async_barrier_readiness_20260817_v4`. Its reproducible
`analysis/README.md` reports all 36 cells, 112 NCU profiles, seven
SourceCounters sentinels, 36 separate NVBit captures, replay diagnostics, and
the held-out decision. The result retains causal readiness but rejects a fixed
CTA-barrier latency; see
`docs/STATUS.md` for the configured
successor.

Build and run an ordinary correctness sample:

```bash
make -C workloads/research_probes/async_barrier_overlap
workloads/research_probes/async_barrier_overlap/async_barrier_overlap \
  c11 l1 0 4 7
```

Arguments are `PROTOCOL LOCALITY GAP WARPS [TRIALS]`:

- protocols: `c00`, `c01`, `c10`, `c11`, `c12`;
- locality: `l1`, `l2`, `dram` (`na` aliases the control-cell L1 setup);
- gap: `0` or `64`;
- warps: 1 through 8; and
- trials: 1 through 32, default 7.

The long binary uses 16,384 committed groups. The `_trace` binary uses 64
groups for a bounded NVBit capture. Each asynchronous group contains eight
cache-all 512-byte `cp.async` operations. The JSON result reports per-warp
`clock64()` samples and a host-checked final payload. The `gap=64` path is a
14-SASS-instruction dependent nonlinear chain calibrated to approximately 64
hardware cycles, rather than an assumed source-level operation count.

The checked-in campaign driver materializes the configured 36-cell matrix and
can resume individual collection stages:

```bash
.venv/bin/python workloads/research_probes/async_barrier_overlap/campaign.py manifest
.venv/bin/python workloads/research_probes/async_barrier_overlap/campaign.py correctness
.venv/bin/python workloads/research_probes/async_barrier_overlap/campaign.py ncu
.venv/bin/python workloads/research_probes/async_barrier_overlap/campaign.py source
.venv/bin/python workloads/research_probes/async_barrier_overlap/campaign.py nvbit
.venv/bin/python workloads/research_probes/async_barrier_overlap/campaign.py static
.venv/bin/python workloads/research_probes/async_barrier_overlap/campaign.py simulator
.venv/bin/python workloads/research_probes/async_barrier_overlap/analyze.py
```

NCU and NVBit are always separate process launches. Both honor the explicit
`cudaProfilerStart`/`cudaProfilerStop` range so prewarming is outside measured
traffic. NCU uses base clocks, application replay, and uncontrolled caches so
each pass recreates the explicit prewarm/stream construction. NVBit uses the
64-group binary.

## Two-launch L2 handoff capture

The ordinary campaign deliberately captures only the profiler-selected target.
To retain both the prewarm and target for launch-chain replay, run one trial
with tracing active from process start:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
TRACE_OUT="$REPO_ROOT/results/captures/sm86_launch_chain_cache_20260817_v1/nvbit/w4_c10_l2_g0"
mkdir -p "$TRACE_OUT"

LD_LIBRARY_PATH="$REPO_ROOT/.toolchain/pb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_INJECTION64_PATH="$REPO_ROOT/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 \
TRACES_FOLDER="$TRACE_OUT" \
ACTIVE_FROM_START=1 \
SKIP_ENHANCED_TRACER=0 \
  "$REPO_ROOT/workloads/research_probes/async_barrier_overlap/async_barrier_overlap_trace" \
  c10 l2 0 4 1
```

Replay the same trace cold per launch or with drained same-stream L2 handoff by
selecting `--cache-scope launch` or `--cache-scope stream-l2`. Do not pass
`--kernel-id` in stream-L2 mode because skipping the seed would change the
declared launch chain. The first live result and exact comparison are recorded
in
`results/captures/sm86_launch_chain_cache_20260817_v1/analysis/README.md`.
