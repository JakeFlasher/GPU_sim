> **Dated semantic adapter record.** Retained for trace/semantic context; it does not establish hardware accuracy or universal decoder coverage.

# SM86 semantic-event round-trip validation

The new `sim-semantics` crate and `sim-trace` SM86 adapter were checked against
the retained post-merge campaign. The adapter preserves lane-byte addresses,
alias multiplicity, widths, optional masks, allocation IDs, cache-operator
fields, placement identity, and the dynamic memory class. A 32-byte sector
footprint is emitted only as a named locality projection.
The canonical control sidecar join also carries stall, yield, read/write
barrier, wait-mask, and reuse fields as timing-visible evidence; these fields
are not used to tune the replay during this probe.

The probe is built with:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
cargo build --release --manifest-path sim/Cargo.toml -p sim-trace --bin semantic-probe
```

The focused comparator is:

```bash
python3 tools/semantic_events/sm86_roundtrip_compare.py \
  --trace TRACE_DIR \
  --control CONTROL_SIDECAR \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json \
  --recorded-replay RECORDED_REPLAY \
  --semantic-probe sim/target/release/semantic-probe \
  --sim-run sim/target/release/sim-run
```

The comparator runs native replay and replay after a canonical semantic-event
serialization round-trip, then compares counters, timing, and coverage. For
large-campaign rows that use the retained finite admission feedback input, add
`--memory-admission-feedback workloads/research_probes/l1tex_admission_backpressure/memory_write_completion_feedback_store_201.json`.

Results on 2026-09-12:

| Cohort | Rows | Result |
|---|---:|---|
| Small | 8 | 8/8 round-trip matches; each has explicit static-only constant-cache metadata gaps |
| Large | 7 | 7/7 round-trip matches; each has explicit static-only constant-cache metadata gaps |
| Total | 15 | 15/15 semantic round-trip replays match counters, timing, and coverage; no opcode classifications are unresolved |

The dynamic trace omits `memory_op_class` for static constant-cache loads. The
adapter reports these records as static-only gaps instead of routing them
through global memory, preserving the absent dynamic class in the canonical
round-trip. Across the retained rows, the gap counts are explicit and bounded
(for example, `conv2d_implicit_gemm_bf16_sm80` has 2,176 and
`swiglu_fwd_bf16_sm80_n1048576` has 98,304). Native replay remains unchanged;
no simulator timing default is introduced here. A future producer update can
close the metadata gap by emitting a dedicated constant-memory class.

The RMSNorm canary round-trip matched `264704` instructions and terminal modeled
time `17119` cycles. The causal-frontier probe remains `26` exact events, `19`
causal events, one collapsed region, and equal final cycle `9`.
