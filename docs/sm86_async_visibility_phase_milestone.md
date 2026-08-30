# SM86 async-visibility phase discriminator

**Status:** complete 2026-08-30; bypass effect localized to drain, no group timing promotion

**Architecture:** `sm_86` / NVIDIA GeForce RTX 3090 Ti (device 0)

**Evidence root:** `results/captures/sm86_async_visibility_phase_20260830_v3/`

**Machine report:** `analysis/report.json`

## Question and assay

The fixed-source visibility pair proved that `.cg` compact and conflict
layouts have different live runtimes, but whole-kernel active cycles could not
locate the difference. This assay records three live warp-clock frontiers for
each serial async-copy group:

1. immediately before the LDGSTS sequence;
2. after the last LDGSTS has dispatched, with compiler waits restricted to its
   operand-read barrier; and
3. after `LDGDEPBAR` / `DEPBAR.LE SB0, 0x0`.

The only global result stores occur after every sampled interval. Compact and
conflict retain identical source addresses, source work, registers, shared
allocation, and instruction/control shape. Depths 1, 2, 4, and 8 expose the
work scaling. The three source routes are cache-all L1, `.cg` with an 8 KiB
repeated L2-resident footprint, and streaming `.cg`/DRAM.

The final static sidecar contains 24 functions, 2,048 decoded instructions,
and zero unknown encodings. Every pair has identical resources and static
stall sum. Each function has exactly three ordered `SR_CLOCKLO` reads; the two
issue-bracketing clocks wait on at most LDGSTS read-barrier 1, never SB0, the
post-wait clock has wait mask zero, and all `STG` instructions follow it.

## Live result

Nine long runs per cell use 8,192 copy groups, for 216 direct GPU executions.
The table reports the median compact→conflict delta divided by group count.

| source route | depth | issue delta / group | post-issue drain delta / group | positive drain repeats |
|---|---:|---:|---:|---:|
| L1 `.ca` | 1 | 0.000 | 15.999 | 9/9 |
| L1 `.ca` | 2 | 0.000 | 31.998 | 9/9 |
| L1 `.ca` | 4 | 0.000 | 63.996 | 9/9 |
| L1 `.ca` | 8 | 0.000 | 127.991 | 9/9 |
| L2-hit `.cg` | 1 | 0.000 | 12.039 | 9/9 |
| L2-hit `.cg` | 2 | 0.000 | 11.403 | 9/9 |
| L2-hit `.cg` | 4 | 0.000 | 14.421 | 9/9 |
| L2-hit `.cg` | 8 | 0.000 | 11.766 | 9/9 |
| DRAM `.cg` | 1 | 0.000 | 16.838 | 9/9 |
| DRAM `.cg` | 2 | 0.000 | 18.824 | 9/9 |
| DRAM `.cg` | 4 | 0.000 | 35.872 | 9/9 |
| DRAM `.cg` | 8 | 0.000 | 41.221 | 9/9 |

All 108 paired issue deltas are exactly zero. All 108 paired drain deltas are
positive, and total delta equals drain delta exactly. Raw runs retain two
large L2 conflict pauses rather than clipping them; the nine-repeat medians
remain in the same 11–14 cycle band.

The separate depth-4 NCU control verifies the L2 route rather than inferring
it from timing. Both layouts execute 256 LDGSTS instructions, 8,192 source
sectors, 8,192 L1 misses, 8,192 LTS hits, and zero LTS misses. Destination
wavefronts alone change from 2,048 to 8,192. NCU remains comparison evidence
and is not consumed by the simulator.

## Decision

The bypass topology effect is not an LDGSTS issue/replay penalty: it appears
only after the operand-read frontier and before the completed async wait. The
cache-all control independently reproduces the current model's exact
16-cycles-per-depth layout law. Bypass does not have that shape. With L2 hits,
an eightfold depth increase leaves the differential approximately constant;
with streaming sources, the differential grows but sublinearly and varies with
source service.

This rejects both a cache-all rule import and a new group-aggregate bypass
constant. The data instead support a per-copy completion mechanism that can
overlap destination work with individual source responses before the group
wait observes their maximum. The current simulator collapses a group to one
maximum source timestamp plus summed delivery work, so it cannot express that
ordering for bypass copies.

Keep `async_copy_shared_visibility` as a typed lower bound. The next core
slice is structural: preserve per-copy source completion and destination work
inside an async group, expose the resulting observational frontier, and test
it on the retained compact/conflict traces before selecting any architecture
timing. JADE/HETR, CPU-GPU co-simulation, `sm_120`, and scored workload defaults
remain unchanged.

## Reproduction

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

## Content identities

- Phase binary: `e642bafdd97607778b725e6e4323e0b42ecee7020949f8ce049244238cdf7833`
- Static control sidecar: `22037de752bbb319f087f2c62d92b1e74a825d370122cef64e7664ac10eea185`
- Machine report: `871b2c353f604fb57c2f48e3c7d01319385822070aae6d53db7ea6724b12848c`
- GPU UUID: `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`
- Driver: `610.57.04`; observed P0 SM clock: 1350 MHz
