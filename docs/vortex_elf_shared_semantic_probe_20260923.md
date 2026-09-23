# Vortex ELF → shared semantic-event probe — 2026-09-23

The retained Vortex integer `vecadd.elf` now crosses the custom path into the
Rust `sim-semantics` event representation. The bridge remains deliberately
narrow: it recognizes a zero-offset register-consistent `lw/lw/add/sw` idiom,
emits a custom SIMT program with caller-supplied synthetic addresses, and lets
`custom_simt_timing` construct and validate the typed events.

```bash
python tools/vortex_elf_shared_semantic_probe.py \
  --adapter tools/vortex_vecadd_elf_adapter.py \
  --timing-binary sim/target/debug/custom_simt_timing \
  --objdump /tmp/vortex-tools-host/riscv64-gnu-toolchain/riscv64-unknown-elf/bin/objdump \
  --elf /tmp/vortex/build/tests/kernel/vecadd/vecadd.elf \
  --spec arch_profiles/specs/simt_u32_vecadd.json \
  --architecture example_simt16 --width 16 \
  --source-revision ab31c0c2f66977fceea05ab282c518bb26b6964b \
  --output docs/vortex_elf_shared_semantic_probe_20260923.json
```

Observed output:

- six validated semantic events;
- operation counts: one special-register, one arithmetic, three global-memory,
  and one no-op event;
- 16-lane width and hypothetical final cycle 112;
- all memory events are typed `global` events; the deferred toy async-store
  mechanism is not relabeled as Vortex `cp.async`.

This is stronger than a hand-authored semantic stream because the ELF pattern
must be recognized first, but it is **not** a general RISC-V decoder, ELF
pointer/dataflow proof, RTL timing match, or silicon result. The synthetic
address binding and unsupported-instruction boundary are retained in the JSON
artifact.
