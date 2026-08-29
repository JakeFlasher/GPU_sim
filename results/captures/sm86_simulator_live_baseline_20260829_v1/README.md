# SM86 simulator live baseline — 2026-08-29

This is a bounded, current-host smoke for the GPU-simulator live-baseline
milestone. It is evidence for the execution path, not a cycle-accuracy claim
and not a replacement for the retained six-workload study.

## Environment

| Item | Observation |
|---|---|
| Repository revision | `7e11418` (`Define SM86 simulator live baseline milestone`) |
| GPU | NVIDIA GeForce RTX 3090 Ti, compute capability 8.6 |
| Driver | 610.57.04 |
| Clock state | 1350 MHz graphics, 10501 MHz memory (already set; no new lock) |
| CUDA compiler | 13.3.73 (`/opt/cuda/bin/nvcc`) |
| NCU | 2026.1.0.0 (`/usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu`) |
| Rust | 1.96.1 |
| Passwordless sudo | unavailable (`sudo: a password is required`) |

The machine and tool versions were read at `2026-08-29T07:17:25-04:00`.

## Matched smoke

The normal CUDA binary passed a three-trial correctness smoke:

```text
workloads/research_probes/async_barrier_overlap/async_barrier_overlap c11 l1 0 4 1
exit=0, correct=true
```

The bounded NVBit binary then ran separately with:

```bash
LD_LIBRARY_PATH="$PWD/.toolchain/pb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_INJECTION64_PATH="$PWD/tools/nvbit_tracer/tracer_tool/tracer_tool.so" \
USER_DEFINED_FOLDERS=1 TRACES_FOLDER=<this-dir>/async_barrier_trace \
ACTIVE_FROM_START=0 SKIP_ENHANCED_TRACER=0 \
  workloads/research_probes/async_barrier_overlap/async_barrier_overlap_trace \
  c11 l1 0 4 1
```

It exited 0 with `correct=true`, 64 logical async groups, and a primary trace
containing one full threadblock payload, `dynamic_trace.pb`, `stats.csv`, and
the expected `extra_info` metadata. The enhanced auxiliary parser reported a
typed `enhanced_tracer_cubin_parse_timeout` because the generated SASS/RFU
inputs exceeded its 64 KiB limit; this smoke therefore proves primary trace
capture, not a fresh static-control admission. The simulator used the matching
checked control sidecar from the retained async-barrier campaign.

NCU was run in a third, separate process against the normal (16,384-group)
binary. `--launch-skip 1` skips the untimed prewarm launch:

```bash
.venv/bin/python tools/local_labels/collect_ncu.py \
  --output <this-dir>/ncu-target \
  --metrics gpc__cycles_elapsed.max,sm__cycles_active.max,smsp__inst_executed.sum \
  --launch-skip 1 --launch-count 1 --replay-mode application \
  --clock-control none -- \
  workloads/research_probes/async_barrier_overlap/async_barrier_overlap c11 l1 0 4 3
```

The profiled target row reports:

| Metric | Value |
|---|---:|
| `gpc__cycles_elapsed.max` | 4,460,056 cycles |
| `sm__cycles_active.max` | 4,458,462 cycles |
| `smsp__inst_executed.sum` | 2,293,904 instructions |

The first exploratory NCU call used `--launch-skip 0` and profiled the prewarm
kernel; it was retained only in `/tmp` and is not used here.

Finally, the current release simulator consumed the fresh NVBit trace:

```bash
sim/target/release/sim-run \
  --trace <this-dir>/async_barrier_trace \
  --control results/captures/sm86_async_barrier_readiness_20260817_v4/sass/async_barrier_overlap_trace.control_bits.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json \
  --mode l1-l2
```

It exited 0 with schema 1, one full-coverage launch, 9,104 issued/observed
records, L1 `32256/512` hit/miss sectors, L2 `54/34` hits/misses, 16 DRAM read
sectors, and a deterministic terminal modeled time of 20,522 cycles. The
completion frontier is explicitly
`incomplete_unmodeled_retirement_tails` with `dirty_writeback=true` and
`coherence=true`; it is a lower bound, not `gpc__cycles_elapsed.max`.

Running that exact command twice produced identical 6,596-byte JSON outputs
(SHA-256 `0fe08416424fb88d78c329658981f6211aa3e8003cb04354005f39c93e7041a8`)
with the same 9,104 records, 20,522-cycle terminal frontier, and incomplete
tail flags.

The trace binary has 64 groups while the NCU binary has 16,384 groups, so raw
cycle totals from these two executables must not be compared without a stated
work-normalization. The saved simulator JSON, NCU report/CSV, workload logs,
and trace metadata are the machine-readable evidence for this smoke.

The active replay validator still reports retained historical pairs as stale
because their JSON mtimes precede the current checkout's simulator input mtimes.
One representative `launch_cycle_semantics` pair was regenerated with the
current release binary; both repeats are byte-identical and pass the validator's
schema/identity helper. This establishes timestamp staleness, not yet a full
5,519-pair refresh.

## Second live slice: barrier pipeline

The same separated path was exercised with
`barrier_pipeline_trace b10 8 0 1` (64 generations in the trace binary) and
`barrier_pipeline b10 8 0 3` (the normal 16,384-generation binary). Both
exited with `correct=true`. The NVBit enhanced parser completed for the one
kernel when its explicit input ceiling was raised to 256 KiB; no parser error
was suppressed. NCU used `--launch-skip 0`, `--launch-count 1`, application
replay, and `--clock-control none` and reported:

| Metric | Value |
|---|---:|
| `gpc__cycles_elapsed.max` | 822,279 cycles |
| `sm__cycles_active.max` | 820,222 cycles |
| `smsp__inst_executed.sum` | 917,688 instructions |

The current release `sim-run` replayed the fresh trace twice with identical
JSON (SHA-256
`33fe138e67194a58f91d889932ee1f28438093006590fa8b8647fa1456a09030`). It
observed 3,768 instructions, a 2,979-cycle modeled terminal lower bound,
readiness schema 6, and typed `dirty_writeback`/`coherence` incomplete tails.
These are diagnostic values for the 64-generation trace and are not directly
the normal binary's NCU totals.

## Next action

Run the same bounded stages for a scheduler/memory representative, then
refresh or explicitly reclassify the stale retained replay pairs. Only after
that comparison should a simulator-core timing seam be selected. No JADE/HETR,
CPU-GPU bridge, `sm_120`, or timing candidate was changed for this baseline.
