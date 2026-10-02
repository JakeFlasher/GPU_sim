# R9.0 native reference-boundary closure (2026-10-02)

This result records a bounded native reference canary used to investigate the
latest review's production-dispatch blocker. It is a diagnostic replay result,
not a hardware measurement, approval artifact, or execution gate.

## Reproduction

```bash
cargo build --manifest-path sim/Cargo.toml --release -p sim-run --features reference
SIM_SM_REFERENCE_TRACE=/tmp/r9_ref.jsonl \
SIM_SM_REFERENCE_MAX_RECORDS=100000 \
sim/target/release/sim-run \
  --trace results/alpha/full_3090ti_20260930_temporal_cold_live2/base__embedding_gather_int32_sm80/trace \
  --control results/alpha/full_3090ti_20260930_temporal_cold_live2/base__embedding_gather_int32_sm80/control.json \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json \
  --mode l1-l2 --cache-scope launch
```

## Result

- Before the fix, the bounded canary stopped while validating a typed
  response-owner/queue join (`SM 2`, warp 0, instruction 17, sector 0, due
  cycle 1686).
- The projection now treats a due-owner/queued-response mismatch as unresolved
  ownership state and widens the typed boundary instead of aborting replay.
  This is conservative unknown propagation, not a guessed queue pairing.
- The rerun completed successfully with return code 0 and a footer: 10,032
  reference event records, 5,016 pre/post pairs, zero complete typed-boundary
  records, and exact fallback for all records. No native macro dispatch or
  speedup was observed.
- The remaining open state is expected: the front-end/cache/response tails are
  not closed, and the comparator has no production macro plan to consume.

## Interpretation

R9.0 closes the canary termination/typed-projection failure by preserving an
explicit unknown. It does **not** close R8.2 production comparator-selected
dispatch. The next native milestone still requires a genuinely closed fixture
through the real event loop, with queue/response, completion, and front-end
continuation state independently matched before mutation.
