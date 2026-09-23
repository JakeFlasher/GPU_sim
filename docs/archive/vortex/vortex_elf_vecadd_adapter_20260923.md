# Vortex ELF vecadd adapter — 2026-09-23

The static Vortex ELF path now has a deliberately narrow executable adapter.
`tools/vortex_vecadd_elf_adapter.py` recognizes the compiled
`_Z13vecadd_kernel` (or Vortex's `__vx_kentry_kernel_main` entry) `lw/lw/add/sw`
idiom (zero-offset word loads/add/store with register-consistent operands) and
emits a custom SIMT program with caller-supplied synthetic byte-address
bindings. Any
ELF without a recognized kernel symbol or operation shape is rejected; floating
point `flw/fadd.s/fsw` kernels are rejected explicitly, and no generic RISC-V
instruction is silently mapped.

```bash
python tools/vortex_vecadd_elf_adapter.py \
  --objdump /tmp/vortex-tools-host/riscv64-gnu-toolchain/riscv64-unknown-elf/bin/objdump \
  --elf /tmp/vortex/build/tests/kernel/vecadd/vecadd.elf \
  --architecture example_simt16 --width 16 \
  --output /tmp/vortex_vecadd_adapted.json
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin custom_simt -- \
  arch_profiles/specs/simt_u32_vecadd.json /tmp/vortex_vecadd_adapted.json
```

This is the first real-binary pattern-to-custom-program bridge, but only for
the declared idiom. The adapter does **not** decode pointer arguments, derive
the runtime addresses, or prove that the generated synthetic bindings are the
ELF's actual memory locations. It is not a general Vortex/RISC-V decoder or
binary-compatibility claim.

The regression float kernel is a retained negative case:

```bash
python tools/vortex_vecadd_elf_adapter.py \
  --objdump /tmp/vortex-tools-host/riscv64-gnu-toolchain/riscv64-unknown-elf/bin/objdump \
  --elf /tmp/vortex/build/tests/regression/vecadd/kernel.elf \
  --architecture example_simt16 --width 16 \
  --output /tmp/vortex_vecadd_float_reject.json
# exits nonzero: floating-point vecadd is outside the integer adapter
```

The adapted program passed `custom_simt_differential.py` for both functional
state and the hypothetical timing schedule (`functional_match=true`,
`timing_match=true`).

Machine-readable result: `docs/vortex_elf_adapted_differential_20260923.json`.
