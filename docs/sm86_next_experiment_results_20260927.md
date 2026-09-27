# SM86 bounded next-experiment results

Updated 2026-09-27. This is the local Phase-0/Phase-1 evidence ledger for
[`plans/sm86_next_experiment_campaign_20260927.md`](../plans/sm86_next_experiment_campaign_20260927.md).
It includes a bounded live RTX 3090 Ti/NCU development rerun of the existing
counter matrix and a tiny external compatibility smoke, but no independent
held-out workload, locked-clock timing/cache score, or full external simulator
run. Counts below are implementation/litmus evidence or explicitly
non-scoring counter evidence, not silicon timing accuracy.

## Input identity and environment

| Field | Value |
| --- | --- |
| Branch | `GPUSim_main` |
| Native architecture | SM86 (`arch_profiles/cards/sm_86.json`) |
| Retained canary | `sim/run/tests/fixtures/addr_carry_canary_bundle` (54 records, one block/warp) |
| Retained fast-only GEMM | `results/captures/sm86_carveout_capture_20260924_v3/gemm_hmma_bf16_sm80` (512 blocks/warps) |
| Retained fast-only LayerNorm | `results/captures/sm86_ncu_demo_benchmark_20260901_full_v2/evidence/raw/nvbit/layernorm_fwd_bf16_sm80` (64 blocks/512 warps) |
| Output identity | Fast canary canonical JSON SHA-256 `39f028a845c1f16e9ced65cca173715abc6d230a7c3d810668f2d72f2010838a` (three repeats) |
| External workload policy | No large Accel-Sim/GPGPU-Sim/MGPUSim/FlashGPU-Sim execution |

## Results

| ID | Bounded check | Denominator / accepted | Refused / result | Evidence |
| --- | --- | ---: | ---: | --- |
| E0 | Fast-only deterministic replay | 3 repeated canary runs / 3 byte-identical | 0 mismatches | identical SHA-256 above |
| E1 | Queue reference/litmus suite | 156 `sim-mem` tests pass | 0 failures | `cargo test -p sim-mem --lib --quiet` |
| E2 | Refusal matrix: sampled coverage, missing masks, texture domain, invalid configs | typed errors in targeted replay/metrics/config tests | 0 silent completions | `sampled_replay_rejects*`, `coherence_replay_refuses_missing_lane_masks`, texture and config tests |
| E3 | Coherence litmus: sharing, owner transfer, pending visibility, unrelated line | queue tests + cross-SM synthetic system transfer pass | 0 false visibility/HOL observations | `coherence.rs` tests; `opt_in_coherence_directory_services_cross_sm_owner_transfer` |
| E4 | Frontier attribution with a final service-only tail | 1 synthetic two-SM tail | service completion excluded from SM-active frontier | `service_only_coherence_tail_is_excluded_from_sm_active_frontier` |
| E5 | PhaseFuse boundary/register/residency guard | independent projected pair accepted; RAW/WAW, discarded-prefix, sampled, non-full-mask, and multi-roster contexts refuse | 0 false certificate in unit cases | `fast_frontier_requires_static_register_evidence_and_rejects_raw_waw`, `fast_frontier_refuses_sampled_coverage_before_certification` |
| E6 | PhaseFuse analyzer scaling | 112 synthetic descriptor cases × 5 repeats = 560 process/contract rows; all exits/contracts/invariants pass and terminal states match | **Process success only:** 112/112 cases, 0 failures/timeouts/truncations; 1,168,400 candidate-region observations across rows; 584,200 collapsed under complete residency, 584,200 reified under unknown residency; 210 rows collapse (42 unique cases), 210 rows fully reify (42 unique cases), 140 rows have zero candidates (`region_len=1`). Latest binary SHA `781af9cc2ff6f804e78fbb0d90281eaf2cfcb32ba56e6a41c2360ca997331b0c` | `phasefuse-scale` + hardened wrapper + [`phasefuse_scale.v1`](../sim/run/schema/phasefuse_scale.v1.schema.json), raw capture `results/captures/phasefuse_scaling_hardened_20260927T161924Z/results.jsonl` (ignored locally), summary identities recorded by wrapper |
| E7 | Route/PTX malformed-input matrix | 55 route tests, 75 arch tests, 21 engine tests pass | 0 new failures | focused Cargo tests |
| E8 | Default-off differential | retained carded vs dirty-writeback canary launch objects | all launch counters/timing/metrics byte-identical; only top-level opt-in field differs | unbound launch SHA `7be451161ab6e5a7a558e526534d5669a06acf38d2f7a9c3d04ff73b6792e453`; binary/card/trace/control identity was not retained with that ad-hoc comparison, so treat it as engineering evidence rather than a reproducible claim |
| E9 | Live SM86 counter-domain development rerun | 13 one-block/one-warp reruns of the existing metric-domain source/case matrix; instruction, store, atomic and reduction counters match NCU | not independent/held-out; timing/cache score refused: clocks unlocked, DRAM domains differ, no dirty flush completion | [`sm86_independent_holdout_20260927.md`](sm86_independent_holdout_20260927.md) |
| E10 | Matched external microtrace | strict `sm86_traceg_adapter_probe_v2` self-parser round-trip passes a two-record full-mask fixture plus dynamic EXIT; real holdout refuses unsupported opcode; separate Accel-Sim smoke reports 5,024 cycles/96 instructions | T1 representation/executable compatibility only; self-parser is not an independent legacy-parser check and no common SM86/NCU timing cell exists | [`sm86_external_baseline_contract_20260927.md`](sm86_external_baseline_contract_20260927.md) |

## Fast-only retained traces

The strengthened register/residency contract intentionally produces no
certificates on these retained captures:

| Trace | Wall time (shell clock) | Candidate regions/instructions | Certified | Main refusal |
| --- | ---: | ---: | ---: | --- |
| GEMM | bounded refusal | 0 / 0 | 0 | multi-block/multi-warp and mask/launch-boundary contract |
| LayerNorm | bounded refusal | 0 / 0 | 0 | multi-block/multi-warp and mask/launch-boundary contract |
| Canary | bounded refusal | 0 / 0 | 0 | non-full active/predicate population; no certificate is emitted |

These wall values include compact trace loading, schedule joins and planning,
but do not include native timing replay. Peak RSS for the descriptor-only E6
probe was collected with Python `resource.getrusage`; the host image lacks
`/usr/bin/time`. The zero-certificate outcome is a conservative refusal, not a
timing-speedup or NCU-accuracy claim.

The E6 probe is intentionally **descriptor-only**: it exercises the public
causal analyzer and its O(N log N) canonicalization on generated descriptors,
not compact-trace parsing or native SM timing. Its wall-time fields are
run-dependent and are not used as a deterministic output identity. “112/112”
means process/contract success, not 112 exact certificates; only the
candidate/collapsed/reified populations above describe analyzer outcomes.

## Interpretation and remaining gates

The local gates support T0 deterministic implementation and bounded T1
mechanism/refusal behavior. They do not support T2 SM86 empirical accuracy, T3
SOTA numeric comparison, or T4 architecture decisions. Remaining required work
is compact-trace PhaseFuse scaling with producer parsing included, locked-clock
held-out SM86/NCU captures with complete ownership/mask/sector contracts, and
E10B external-parser integration/common-endpoint scoring (or an explicit
mismatch record). Same-launch L1-hit invalidation remains unmodeled in the coherence
directory; it is a typed limitation rather than an accuracy claim. The
directory still
starts cold; native resident dirty flush, fences, protocol ownership and
physical pipe selection remain outside the trace contract.
