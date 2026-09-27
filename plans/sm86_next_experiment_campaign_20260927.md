# SM86 next-experiment campaign

Updated 2026-09-27 after the adversarial code/scientific/publication review of
`9614f62b`; bounded correctness and holdout follow-ups landed in `e9d750f6`
and `506c6d6e`. This is a technical experiment plan, not a workflow gate.
Root `AGENTS.md` remains policy. No large Accel-Sim/MGPUSim workload is
required for the local phases.

## Latest bounded progress (2026-09-27)

The first correctness slice has landed: sampled/full and lane-mask admission
checks are fail-closed; texture and local-sector requests cannot enter the
global directory; physical `sm_id` is retained; pending directory transitions
publish only at completion; unrelated zero-service lines avoid global
head-of-line delay; DRAM read/writeback service is independent of coherence;
service-only events are excluded from the SM-active terminal frontier; and
PhaseFuse requires a complete traced-SM roster plus static register RAW/WAW
evidence. The local E0/E1/E2/E3/E4/E5/E7/E8 checks below are now executable in
bounded Rust/fixture tests, including a synthetic final service-only tail for
E4. A descriptor-only E6 scale probe now covers the full 112-case
sorted/shuffled N=1k..64k × region/residency matrix with child RSS;
compact-trace parsing-inclusive scaling remains open. Independent SM86/NCU
progress now includes a 13-case one-block/one-warp counter-domain development
rerun with content-bound trace/NCU/replay artifacts; it reuses the existing
metric-domain source/case population. Unlocked clocks and DRAM-domain
mismatch keep it non-scoring for timing. This is a development rerun of the
existing metric-domain source/case matrix, not an independent holdout. The
ordered stream session now rejects coherence (directory state is not persisted)
and sampled carded writeback is refused. E10 now has a pinned capability audit,
a strict parser/round-trip probe, and a three-record Accel-Sim executable
compatibility smoke, not a numeric baseline. Full locked-clock timing/cache
holdouts and E10B parser integration/common-endpoint scoring remain open and
must not be represented by these results.
The reproducible local ledger is [SM86 bounded next-experiment results](../docs/sm86_next_experiment_results_20260927.md).

## Claim tiers

Do not merge these tiers in one headline:

| Tier | Claim permitted | Evidence required |
| --- | --- | --- |
| T0 | Deterministic implementation and declared recurrence | Rust/property/metamorphic tests, exact output comparison, bounded resource limits |
| T1 | Bounded writeback/directory/PhaseFuse mechanism behavior | Complete synthetic topology, hand oracle, zero false certificates, explicit refusals |
| T2 | SM86 empirical behavior | Independent RTX 3090 Ti/NVBit/NCU holdouts, frozen cards/parameters, same metric domains |
| T3 | SOTA comparison | Matched traces/workloads/frontends, versions/configurations, cycle/counter/wall/RSS results |
| T4 | Architecture decisions | Frozen interventions, detailed/native reference ranking stability and confidence intervals |

The current repository has T0 evidence and bounded T1 seams. The 13-case
holdout is narrow T2 counter-contract evidence only; it is not T2 timing/cache
accuracy. The repository does not yet have full T2, T3, or T4 evidence.

## Phase 0 — correctness blockers before scoring

These are required before reporting coherence, writeback or fast-frontier
accuracy:

1. Reject `ExtractedCoverage::Sampled` and any incomplete launch when an
   opt-in timing mechanism needs complete topology.
2. Reject missing active/predicate masks for coherence/CFG certification; do
   not let missing masks silently mean zero unknown records.
3. Filter `local_sector_mask` from the global coherence domain, or refuse a
   mixed local/global request whose sector origin cannot be separated.
4. Exclude texture/surface/non-global domains from the global directory unless
   a separate visibility contract is supplied.
5. Make initial ownership and L1-hidden accesses explicit: either ingest all
   relevant global accesses or refuse the coherence model when initial state is
   not complete.
6. Replace global coherence head-of-line behavior with per-line visibility
   readiness; do not delay unrelated read/read sharing behind another line.
7. Do not expose directory state to later requests before a pending transfer is
   visible, or label the model explicitly as an admission reservation rather
   than visibility completion.
8. Exclude writeback/coherence-only events from the SM-active terminal frontier.
9. Make carded-read, carded-writeback, and carded-coherence service ordering
   orthogonal; add a clean-read perturbation differential.
10. Reject coherence in stream-L2 replay unless directory state is persisted and
    validated across launches.
11. For PhaseFuse, require the complete SM resident roster, incoming pipeline
    state, and actual register RAW/WAW evidence before certifying a region.
12. Fast-only CLI must reject or omit all timing/resource/coherence settings it
    does not consume.

## Phase 1 — bounded local experiments (hours, no GPU required)

Every experiment publishes: input identity, denominator, accepted count,
refusal count, output digest, wall time including parsing/planning, peak RSS,
and failure reason. No silent drops.

| ID | Experiment | Oracle / invariant | Pass condition |
| --- | --- | --- | --- |
| E0 | Determinism: 10 repeated runs of each retained fixture | Canonical JSON/output and event/frontier hashes | Bit-identical outputs; no nondeterministic ordering |
| E1 | Writeback/coherence queue reference model | Small Python/Rust hand state machine for fractional service, masks, capacity, stale plans, time order | Exact state/completion agreement; transactional failures leave state unchanged |
| E2 | Refusal matrix | Synthetic missing masks, sampled coverage, unknown SM, sparse SM, local-only/mixed sectors, texture, invalid config | Each case returns the typed refusal; zero completion counters on refusal |
| E3 | Coherence litmus suite | Cold read/read; write→remote-read; remote write invalidation; unrelated lines; pending transfer; clean/dirty victim; writeback+coherence | Expected service sectors/frontiers; no unrelated HOL; no pre-completion visibility |
| E4 | Frontier attribution | Final response, writeback completion, coherence completion, SM issue-only tails | SM-active frontier excludes service-only events; full event frontier includes them |
| E5 | PhaseFuse soundness | Two CTAs on one SM/SMSP, incoming pipeline work, scalar RAW/WAW, barrier/control/async-copy boundaries | Zero false certificates; refusal reasons identify each missing boundary |
| E6 | PhaseFuse accounting/scaling | Generated traces at 1K/10K/100K records; detailed vs fast | Report planner-inclusive wall/RSS and actual per-op work; no asymptotic speed claim from collapsed counters |
| E7 | Route/PTX malformed-input matrix | Overflowing alternate route, zero work, malformed punctuation/entry/exit, atomic temp sentinel | Typed rejection; valid fixtures preserve outputs; default scalar route unchanged |
| E8 | Default-off differential | Flat, carded, writeback-disabled, coherence-disabled replay on retained canary | Timing/counters/frontiers byte-identical except explicitly additive omitted fields |
| E9 | Architecture-decision stability | Frozen cache/issue/occupancy/memory interventions and detailed reference ranking | Kendall/Spearman and top-k agreement reported; no NCU-driven selection |
| E10 | Matched microtrace baseline | Small compatible Accel-Sim/GPGPU-Sim subset, or capability-mismatch record | Numeric comparison only when frontend/ISA/domain match; otherwise explicit mismatch |

Recommended bounded commands:

```bash
cargo test --manifest-path sim/Cargo.toml --workspace --quiet
cargo test --manifest-path sim/Cargo.toml -p sim-sm --features campaign --quiet
cargo test --manifest-path sim/Cargo.toml -p sim-mem --lib --quiet
cargo clippy --manifest-path sim/Cargo.toml -p sim-arch --all-targets -- -D warnings
cargo clippy --manifest-path sim/Cargo.toml -p sim-engine --all-targets -- -D warnings
cargo clippy --manifest-path sim/Cargo.toml -p sim-mem --lib -- -D warnings
/usr/bin/time -v sim/target/release/sim-run ...
```

### PhaseFuse-specific protocol

The fast lane is not accepted on final-cycle equality alone. Use the causal
frontier API and the fast producer with deterministic small descriptors:

1. **Complete residency:** two CTAs on one SM/SMSP, full flattened roster,
   block-only roster, unknown roster, extra stalled warp, slot reuse and
   population-change boundaries. A certificate with an omitted co-resident CTA
   is a false certificate.
2. **Incoming pipeline state:** pending work at `start-1`, exactly `start`, a
   different pipe, multiple pending completions, and sibling-CTA work. Compare
   every issue/completion observation and terminal state.
3. **Register dependencies:** SASS RAW, WAW, predicate and 64-bit-span pairs
   with clean control bits. Use native scoreboard/readiness as the oracle;
   unknown register provenance must refuse rather than certify.
4. **Negative sweep:** up to 256 seeded descriptors covering stalls, yields,
   reuse, barriers, async copy, completion units, unknown timing, duplicate
   residents, equal-time ordering and near-overflow arithmetic. Every refusal
   needs a mechanism-specific reason.
5. **Risk coverage:** publish candidate/certified/reified/collapsed counts and
   reason histograms by mechanism. A generic refusal bucket must not hide a
   known risk class.
6. **Planner-inclusive scaling:** N={1k,2k,4k,8k,16k,32k,64k}, 1–4 SMSPs,
   region lengths 1/2/8/64, five repetitions, `/usr/bin/time -v`, wall/user/sys
   time and peak RSS. Measure sorted and shuffled inputs separately because the
   current canonicalization sort is O(N log N). Never call collapsed event
   counts a speedup.

Acceptance requires zero false certificates, complete residency and register
evidence, exact terminal-state equality for every certificate, and measured
planner-inclusive scaling. High refusal is acceptable; silent candidate loss is
not.

## Phase 2 — independent SM86 holdout (requires RTX 3090 Ti + NCU)

This phase is blocked unless the hardware and independent captures are
available. Freeze the card, driver, UUID, clocks, CUDA/toolchain versions and
model parameters before scoring. Keep development traces separate from test
traces.

Capture at least five feature families and approximately 20 independent
microbenchmarks:

1. Scalar dependent/independent issue and pipe candidates.
2. L1 hit, L2 hit, DRAM miss, stride/coalescing and bypass paths.
3. Dirty eviction and explicit flush/writeback probes.
4. Cross-SM producer/consumer with release/acquire, fences and atomics.
5. Occupancy/residency and divergence/value-dependent loops.

Use held-out kernel families, five repeats, locked clocks, and independently
recorded GPU name/UUID/compute capability/driver/device index. Report per-domain
MAE/MAPE/bias/RMSE, median and 95% bootstrap cycle error, max error, rank
correlation, coverage and refusal rate. NCU values may only be targets after
replay; they cannot select parameters or model branches.

## Phase 3 — matched external baselines (capability matrix first)

Do not run full suites. Use a small matched microtrace subset only when the
frontend, ISA and metric domain are compatible:

- Accel-Sim/GPGPU-Sim: possible trace-driven comparison, with exact revision,
  config, conversion cost and host wall/RSS recorded.
- MGPUSim: capability reference; its public NVIDIA path is not a valid SM86
  numerical baseline without a verified frontend.
- Vortex/OpenVADL: compiler/ISA extensibility references, not NVIDIA-SM86
  timing baselines.
- FlashGPU-Sim: modern execution-driven feature reference; its RTX 5090/H100/
  B200 results are not SM86 numbers.

If a matched baseline cannot be built or mapped, report “capability mismatch”
instead of fabricating a numeric comparison.

## Phase 4 — publication analyses

Only after Phases 0–2:

- Ablate dense, sparse-event, PhaseFuse, writeback and directory lanes.
- Plot risk/coverage and refusal curves.
- Include preprocessing, parsing, planning, execution, CPU time and peak RSS.
- Measure scaling by records, warps, blocks and accepted regions.
- Evaluate design-decision stability with Kendall/Spearman rank correlation and
  top-k agreement under frozen interventions.

## Go/no-go criteria

**Go for bounded artifact/demo:** all Phase-0 checks pass, zero false
certificates in E2/E3/E5, deterministic E0 outputs, explicit refusal coverage,
and reproducible scripts/artifacts.

**Go for SM86 empirical claim:** independent held-out captures across the five
families, complete trace contracts, no NCU/model circularity, confidence
intervals, and transparent refusal/coverage reporting.

**No-go for “orders-of-magnitude faster with NCU accuracy”:** any hidden L1 or
ownership state, sampled coherence input, mismatched NCU domain, development-
only corpus, incomplete PhaseFuse boundary state, or speed measurement that
excludes planning/I/O/RSS.

Primary references: [Accel-Sim ISCA'20](https://engineering.purdue.edu/tgrogers/papers/khairy.isca2020.pdf),
[FlashGPU-Sim](https://arxiv.org/abs/2609.15311),
[MGPUSim ISCA'19](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf),
[Vortex](https://arxiv.org/abs/2110.10857),
[OpenVADL](https://openvadl.github.io/openvadl/), and the
[ACM artifact-review policy](https://prod-www.acm.bloomreach.cloud/publications/policies/artifact-review-and-badging-current).
