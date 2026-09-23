# Vortex ELF static adapter — 2026-09-23

The Vortex open-ISA seam now has a bounded static ingestion probe. It invokes
the RISC-V `objdump` from the independently installed Vortex toolchain,
classifies standard operations, and preserves custom/unknown encodings instead
of mapping them to generic ALU operations.

```bash
python tools/vortex_elf_summary.py \
  --objdump /tmp/vortex-tools-host/riscv64-gnu-toolchain/riscv64-unknown-elf/bin/objdump \
  --elf /tmp/vortex/build/tests/kernel/vecadd/vecadd.elf \
  --simt-width 4 \
  --source-revision ab31c0c2f66977fceea05ab282c518bb26b6964b \
  --output docs/vortex_elf_summary_20260923.json
```

This is static open-ISA evidence only. Functional execution still comes from
Vortex SimX/RTL; timing, memory effects, and a direct Vortex program adapter
remain future work.
