# Custom SIMT pipeline-overlap probe — 2026-09-24

The bounded custom timing executor now separates the program-issue frontier
from register-ready and per-resource frontiers. An independent ALU sequence
can issue while a memory operation is still in flight; dependent consumers wait
for the producer's completion timestamp. This is still a single-group,
single-program timing model, not a production GPU scheduler.

## Command

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-arch
sim/target/release/custom_simt_timing \
  arch_profiles/specs/simt_u32_minimal.json \
  arch_profiles/programs/simt_u32_pipeline_overlap.json
```

## Observed result

The eight-lane overlap program produced `final_cycle=37` and the following
relevant events:

| instruction | operation | start | complete |
|---:|---|---:|---:|
| 1 | load | 1 | 29 |
| 2 | independent `set_lane_id` | 2 | 4 |
| 3 | dependent `add_imm` | 4 | 6 |
| 4 | store | 9 | 37 |

The ALU operation starts before the load completes, while the store waits for
its register producer. The independent Python differential oracle reports
`functional_match=true` and `timing_match=true` for this program.

This closes the custom seam's basic pipeline-overlap limitation. It does not
claim multi-warp scheduling, arbitrary ISA semantics, or silicon timing.
