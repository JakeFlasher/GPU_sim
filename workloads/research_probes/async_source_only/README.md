# Store-free async-source and destination-layout probe

This SM86 probe isolates asynchronous global-to-shared copy behavior without
a target-kernel global store or CTA barrier. The current visibility pair keeps
one 32-thread warp, source addresses, source work, 32 KiB shared allocation,
registers, and static instruction shape fixed. `--layout compact` uses a
32-byte shared lane stride; `--layout conflict` uses 64 bytes. Both use a
4 KiB destination-position stride. The sparse but identical 32-byte source
lane stride makes source-sector work equal while the destination wavefront
topology changes.

Each loop group issues fixed-depth `cp.async.ca` (L1) or `cp.async.cg`
(bypass/DRAM) positions, commits, executes `wait_group 0`, and performs the
same warp-uniform shared broadcast. Host source initialization is outside the
profiler markers. The empty inline asm at kernel end keeps clocks/token live
without adding a result store.

Build and run a bounded marker-scoped smoke:

```bash
make -C workloads/research_probes/async_source_only clean all
CUDA_VISIBLE_DEVICES=0 \
  workloads/research_probes/async_source_only/async_source_only \
  --groups 64 --depth 4 --route l1 --layout compact --markers on
CUDA_VISIBLE_DEVICES=0 \
  workloads/research_probes/async_source_only/async_source_only \
  --groups 64 --depth 4 --route l1 --layout conflict --markers on
```

For an NVBit capture, use the shared `capture_nvbit.py` helper. NCU values
remain comparison-only; the simulator receives only the complete NVBit trace,
static control sidecar, and SM86 cards. The layout result identifies a bypass
destination effect, but whole-kernel active cycles do not distinguish issue
replay from post-source visibility and therefore are not a timing constant.

Reduce the retained five-repeat depth-4/depth-8 layout slice with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_visibility.py \
  --root results/captures/sm86_async_source_visibility_20260830_v2 \
  --output results/captures/sm86_async_source_visibility_20260830_v2/analysis/report.json
```

The original contiguous-source discriminator remains immutable under its
recorded binary hash and is reduced from the predecessor evidence root with:

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

## Three-frontier phase assay

`async_visibility_phase` uses the same fixed-source layouts but records three
live clock frontiers per copy group: before LDGSTS issue, after the last
LDGSTS, and after `wait_group 0`. Its only global result store occurs after all
measured intervals. Routes are cache-all `l1`, bypass with a repeated L2-hit
footprint (`l2`), and streaming bypass (`dram`).

```bash
make -C workloads/research_probes/async_source_only async_visibility_phase
CUDA_VISIBLE_DEVICES=0 \
  workloads/research_probes/async_source_only/async_visibility_phase \
  --groups 8192 --depth 4 --route l2 --layout compact --markers off
CUDA_VISIBLE_DEVICES=0 \
  workloads/research_probes/async_source_only/async_visibility_phase \
  --groups 8192 --depth 4 --route l2 --layout conflict --markers off

PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_phase.py \
  --root results/captures/sm86_async_visibility_phase_20260830_v3 \
  --output results/captures/sm86_async_visibility_phase_20260830_v3/analysis/report.json
```

The phase assay is live diagnostic evidence, not a simulator input. It
localizes the bypass layout effect after issue, while its depth shape rejects
reuse of the cache-all group-aggregate completion rule.

The simulator-side per-copy observation is reduced with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_per_copy.py \
  --root results/captures/sm86_async_copy_per_copy_observation_20260830_v2 \
  --baseline-root results/captures/sm86_async_source_visibility_20260830_v2 \
  --output results/captures/sm86_async_copy_per_copy_observation_20260830_v2/analysis/report.json
```

Schema-14 bypass observations retain per-instruction source completion,
payload work, and trace-projected destination wavefronts. They do not change
timing or clear the typed visibility tail.

`--layout severe` adds a 128-byte destination lane stride for the follow-on
projection discriminator. Reduce its three-layout L1/L2 campaign with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_projection.py \
  --root results/captures/sm86_async_bypass_projection_20260830_v1 \
  --output results/captures/sm86_async_bypass_projection_20260830_v1/analysis/report.json
```

`--layout skew` adds the orthogonal 96-byte stride. The final multidimensional
campaign separates source-cohort destination-line fanout from bank
multiplicity and reduces with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  workloads/research_probes/async_source_only/analyze_multidim.py \
  --root results/captures/sm86_async_bypass_multidim_20260830_v1 \
  --baseline-root results/captures/sm86_async_source_visibility_20260830_v2 \
  --output results/captures/sm86_async_bypass_multidim_20260830_v1/analysis/report.json
```

Schema 15 reports the separated topology and a timing-neutral per-copy local
candidate. The candidate fails depth transfer, so production timing and the
typed bypass tail remain unchanged.
