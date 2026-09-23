# Architecture

The simulator is an event driven replay engine. A workload is captured once,
then replayed against an architecture card without executing the original CUDA
program.

```text
NVBit
  | dynamic protobuf + control sidecars
  v
sim-trace  -- decode, validate, compact
  v
sim-sm     -- warp state, dependencies, issue, barriers, async copies
  v
sim-mem    -- L1/TEX, L2, DRAM request and completion state
  v
sim-engine -- event queue and system progress
  v
  sim-run-campaign -- manifest-driven CLI, JSON result/error output
  sim-run           -- direct trace replay CLI
```

`sim-arch` owns architecture capabilities and defaults. `sim-trace` owns wire
formats and trace directory readers. `sim-sm` owns per-warp and per-CTA
scheduling. `sim-mem` models cache and memory service. `sim-engine` provides
shared event ordering. `sim-run` and `sim-direct-run` provide command line
entry points: `sim-run-campaign` for manifest workflows and `sim-run` for
direct trace replay.

Architecture dependent values live in `arch_profiles/cards`. The SM86 card is
the production path. Unsupported capabilities fail explicitly instead of
silently borrowing SM86 constants.

NCU values enter only through comparison tools. They do not alter trace
replay, architecture cards, or simulator parameters.

## User-defined functional seam

`sim-arch::spec` is separate from production `ArchCard`. The bounded
`simt_u32_v1` executor demonstrates lane arithmetic and resource-relative
scheduling; the Curry-inspired `simt_its_v1` executor demonstrates per-lane
PCs with BSSY/BSYNC/BREAK/EXIT semantics. Both are functional feasibility
backends only. They do not alter the NVBit/SM86 replay and make no timing,
compiler, RTL, or silicon claim.
