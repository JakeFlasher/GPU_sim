# SM86 replay determinism probe

The release `sim-run` binary was invoked twice in each reference mode with the
same retained address-carry canary command:

```text
sim/target/release/sim-run --trace sim/run/tests/fixtures/addr_carry_canary_bundle/trace --control sim/run/tests/fixtures/addr_carry_canary_bundle/control/canary_sm86.control_bits.json --arch-card sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86.json --pchase-card sim/run/tests/fixtures/addr_carry_canary_bundle/cards/sm_86_static_card.json --mode l1-l2 --cache-scope launch --ncu-shaped-metrics
```

With `SIM_SM_REFERENCE_MODE=event`, both processes exited zero and produced
byte-identical stdout (12,813 bytes) and stderr (91 bytes). With
`SIM_SM_REFERENCE_MODE=dense`, both processes exited zero and produced
byte-identical stdout (12,813 bytes) and stderr (90 bytes).

This probe tests process-level deterministic replay independently of the
event-versus-dense differential. It does not establish determinism for future
parallel execution extensions; those must retain explicit sequence and phase
ordering.

The same process-level check was repeated on the retained large
`flash_attn_bwd_bf16_sm80` command. Two event-mode runs and two dense-mode
runs produced identical stdout and stderr, independently of the canary.
