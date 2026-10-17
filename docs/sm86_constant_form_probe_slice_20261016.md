# SM86 constant-form microprobe — 2026-10-16

## Probe

The retained notes project now contains
`/home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/constant_form_scope`.
It compiles two identical-work SM86 kernels: 32-bit `__ldg` and uint16
`__ldg`, with the same blocks, threads, iterations, addresses, and functional
checksum. NCU is not used as a fitting input.

Build:

```bash
make -C /home/jakeshea/GPU_sims_notes/GPU_simulator_snapshot/workloads/research_probes/constant_form_scope
```

Both modes execute successfully with identical checksums. `cuobjdump --dump-sass`
shows:

- 32-bit path: `LDG.E.CONSTANT`
- uint16 path: `LDG.E.U16.CONSTANT`

## Decision

Static form identity is a reproducible source/client discriminator for H48's
observer. It does not prove a constant-space/L0C route: modern source and
Nsight semantics keep LDG global traffic distinct from LDC/CBANK constant
space. No lookup correction, latency, cache relocation, or NCU fit is promoted.
The next form-specific probe should vary reuse distance/warp count while
retaining functional and SASS checks.
