# sm_86 atomic completion and serialization probe

This probe crosses returning `ATOMG` and non-returning `RED` with four address
layouts: one 128-byte line per lane, one word per lane within a warp line, one
word shared by each warp, and one word shared by the full CTA. Final memory and
the sum of all returned pre-increment values provide deterministic correctness
oracles even when the order of contending atomics is nondeterministic.

The long cubins execute 1,024 atomic generations for live timing and NCU. The
trace cubins execute 64 generations for bounded NVBit capture and deterministic
simulator replay. Widths 1, 4, 8, and 16 separate single-warp return latency,
one-warp-per-SMSP behavior, and latency-hiding transfer.

The first three layouts are the primary static comparisons: all returning
cubins use 14 registers, all reduction cubins use 12, and every function has 40
instructions. The returning full-CTA word case uses 18 registers and 48
instructions, so it is retained only as an extreme-contention transfer case.

The later `atomic_alias_adversarial` and `atomic_multism_service` campaigns
supersede this probe's offline timing candidate. They retain the exact sector
fix, add exact L2 request topology and CLOCK-bounded replay diagnostics, and
leave timing inactive because the distributed atomic-unit hash is unknown.

Run the complete campaign from this directory:

```bash
make -j"$(nproc)"
../../../.venv/bin/python campaign.py static
../../../.venv/bin/python campaign.py correctness
../../../.venv/bin/python campaign.py trace-correctness
../../../.venv/bin/python campaign.py ncu
../../../.venv/bin/python campaign.py source
../../../.venv/bin/python campaign.py nvbit
cargo build --release --manifest-path ../../../sim/Cargo.toml -p sim-run
../../../.venv/bin/python campaign.py simulator
```

The default output root is
`results/captures/sm86_atomic_completion_serialization_20260818_v1`.
