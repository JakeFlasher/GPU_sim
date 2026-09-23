# Custom SIMT independent differential — 2026-09-23

The bounded custom executor has an independent Python functional/timing oracle.
The oracle reimplements the tiny program semantics rather than importing Rust
code. This is a feasibility correctness check, not an RTL or silicon oracle.

## Command

```bash
cargo build --release --manifest-path sim/Cargo.toml \
  -p sim-arch --bin custom_simt --bin custom_simt_timing
python tools/custom_simt_differential.py \
  --binary sim/target/release/custom_simt \
  --timing-binary sim/target/release/custom_simt_timing \
  --spec arch_profiles/specs/simt_u32_minimal.json \
  --program arch_profiles/programs/simt_u32_async_store.json \
  --output docs/custom_simt_differential_20260923.json
```

The check compares architecture/ISA identity, lane width, instruction count,
all register state, all stores, final cycle, and every timing event (operation,
resource, issue cycle, and completion cycle).

Memory addresses are byte-addressed, but this feasibility ISA admits only
four-byte-aligned u32 words. Async submit/wait operations are carried as typed
custom token effects; the submit event remains a global store (it is not
silently relabeled as a global-to-shared copy). The u32 word probe uses a
four-byte lane stride so lane stores cannot overlap; the executor rejects
unaligned/overlapping word ranges and address-domain wraparound.
