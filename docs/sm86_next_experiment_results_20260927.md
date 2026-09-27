# SM86 bounded next-experiment results

Updated 2026-09-27. This is the local Phase-0/Phase-1 evidence ledger for
[`plans/sm86_next_experiment_campaign_20260927.md`](../plans/sm86_next_experiment_campaign_20260927.md).
It contains no independent RTX 3090 Ti/NCU holdout and no full external
simulator run. Counts below are implementation/litmus evidence, not silicon
accuracy.

## Input identity and environment

| Field | Value |
| --- | --- |
| Branch | `GPUSim_main` |
| Native architecture | SM86 (`arch_profiles/cards/sm_86.json`) |
| Retained canary | `sim/run/tests/fixtures/addr_carry_canary_bundle` (54 records, one block/warp) |
| Retained fast-only GEMM | `results/captures/sm86_carveout_capture_20260924_v3/gemm_hmma_bf16_sm80` (512 blocks/warps) |
| Retained fast-only LayerNorm | `results/captures/sm86_ncu_demo_benchmark_20260901_full_v2/evidence/raw/nvbit/layernorm_fwd_bf16_sm80` (64 blocks/512 warps) |
| Output identity | Fast canary canonical JSON SHA-256 `679f72f214bb678a8c83fac336c70031d6c76c7f8a02a9ba839e0fa51464fad7` (three repeats) |
| External workload policy | No large Accel-Sim/GPGPU-Sim/MGPUSim/FlashGPU-Sim execution |

## Results

| ID | Bounded check | Denominator / accepted | Refused / result | Evidence |
| --- | --- | ---: | ---: | --- |
| E0 | Fast-only deterministic replay | 3 repeated canary runs / 3 byte-identical | 0 mismatches | identical SHA-256 above |
| E1 | Queue reference/litmus suite | 156 `sim-mem` tests pass | 0 failures | `cargo test -p sim-mem --lib --quiet` |
| E2 | Refusal matrix: sampled coverage, missing masks, texture domain, invalid configs | typed errors in targeted replay/metrics/config tests | 0 silent completions | `sampled_replay_rejects*`, `coherence_replay_refuses_missing_lane_masks`, texture and config tests |
| E3 | Coherence litmus: sharing, owner transfer, pending visibility, unrelated line | queue tests + cross-SM synthetic system transfer pass | 0 false visibility/HOL observations | `coherence.rs` tests; `opt_in_coherence_directory_services_cross_sm_owner_transfer` |
| E4 | Frontier attribution with a final service-only tail | 1 synthetic two-SM tail | service completion excluded from SM-active frontier | `service_only_coherence_tail_is_excluded_from_sm_active_frontier` |
| E5 | PhaseFuse register/residency guard | independent projected pair accepted as candidate; RAW/WAW/unknown projection refused | 0 false certificate in unit cases | `fast_frontier_requires_static_register_evidence_and_rejects_raw_waw` |
| E6 | PhaseFuse analyzer scaling | 112 synthetic descriptor cases: N=1k..64k × region length {1,2,8,64} × complete/unknown residency × sorted/shuffled, one repeat; all exits 0 and terminal states match | 42 complete/L≥2 cases collapse; 0 unknown-residency collapses; 0 L=1 collapses; peak child RSS 22,536 KiB; aggregate child wall ~1.11 s | `phasefuse-scale` + wrapper + [`phasefuse_scale.v1`](../sim/run/schema/phasefuse_scale.v1.schema.json), raw JSONL SHA `c149ebd890b6c969b55ae2b1bdcdbe5803f46f72ced9df5d3d182f79408298ef` |
| E7 | Route/PTX malformed-input matrix | 55 route tests, 75 arch tests, 21 engine tests pass | 0 new failures | focused Cargo tests |
| E8 | Default-off differential | retained carded vs dirty-writeback canary launch objects | all launch counters/timing/metrics byte-identical; only top-level opt-in field differs | launch SHA `7be451161ab6e5a7a558e526534d5669a06acf38d2f7a9c3d04ff73b6792e453` |
| E9 | Independent NCU holdout | not available in this environment | blocked, not approximated | requires frozen RTX 3090 Ti capture/card/driver |
| E10 | Matched external microtrace | not built/run | capability-match decision still open | no numeric baseline claimed |

## Fast-only retained traces

The strengthened register/residency contract intentionally produces no
certificates on these retained captures:

| Trace | Wall time (shell clock) | Candidate regions/instructions | Certified | Main refusal |
| --- | ---: | ---: | ---: | --- |
| GEMM | ~0.22 s | 512 / 1,024 | 0 | causal guard after static evidence/residency checks |
| LayerNorm | ~0.29 s | 88 / 176 | 0 | 469 memory/control/non-scalar; remaining causal guards |
| Canary | <0.01 s | 0 / 0 | 0 | one memory/control/non-scalar warp |

These wall values include compact trace loading, schedule joins and planning,
but do not include native timing replay. Peak RSS for the descriptor-only E6
probe was collected with Python `resource.getrusage`; the host image lacks
`/usr/bin/time`. The zero-certificate outcome is a conservative refusal, not a
timing-speedup or NCU-accuracy claim.

The E6 probe is intentionally **descriptor-only**: it exercises the public
causal analyzer and its O(N log N) canonicalization on generated descriptors,
not compact-trace parsing or native SM timing. Its wall-time fields are
run-dependent and are not used as a deterministic output identity.

## Interpretation and remaining gates

The local gates support T0 deterministic implementation and bounded T1
mechanism/refusal behavior. They do not support T2 SM86 empirical accuracy, T3
SOTA numeric comparison, or T4 architecture decisions. Remaining required work
is compact-trace PhaseFuse scaling with producer parsing included, independent held-out SM86/NCU
captures with complete ownership/mask/sector contracts, and a capability-matched
small external microtrace (or an explicit mismatch report). The directory still
starts cold; native resident dirty flush, fences, protocol ownership and
physical pipe selection remain outside the trace contract.
