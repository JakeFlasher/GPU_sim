# User-defined architecture specifications

Files in this directory describe hypothetical or open/custom GPU shapes for
the experimental `sim-arch::spec` path. They are **not** replacements for
`arch_profiles/cards`, which remain the evidence-bound inputs for the
production SM86 replay.

Each specification declares an opaque architecture/ISA token, an admitted
instruction-name set, SIMT width, resident-group bound, named resources,
optional timing values, and an evidence status. `unvalidated` means that the values are legal experiment inputs but
are not claims about physical silicon. A timing backend must treat an absent
latency as unknown rather than inventing a default.

Validate an example with:

```bash
cargo run --manifest-path sim/Cargo.toml -p sim-arch --bin architecture_spec \
  -- arch_profiles/specs/simt_u32_minimal.json
```

The current example is intentionally small. It is a feasibility fixture for
the custom-architecture seam, not Curry GPU compatibility or a benchmark. A
matching functional program is under `arch_profiles/programs`.
