# Current status

**Updated 2026-09-30.** This is the authoritative capability boundary. It
supersedes the long dated handoffs formerly in `docs/`; those reports remain
under [`archive/research_20260930`](archive/research_20260930/). Read
[`PROJECT_MEMORY`](PROJECT_MEMORY.md) for the compact resume context.

## Decision in one paragraph

The repository has a usable native SM86 replay and a fast, metric-oriented
projection lane. The projection can emit a candidate point and uncertainty
interval for every row in the current campaign contract, including unmatched
shapes through an explicit fallback. It is not a cycle-accurate silicon model,
an architecture-wide OOD guarantee, or proof that the four physical boundaries
are observable. NCU values remain independent comparison targets. The research
goal is therefore **blocked, not complete**, at the exact physical boundaries;
metric approximation may continue without pretending those boundaries closed.

## Execution surfaces

| Surface | Contract | Status |
| --- | --- | --- |
| Native replay | NVBit/SASS and architecture cards through `sim-run` | Runnable SM86 alpha; frontiers remain typed and separate |
| Custom resource model | `simt_u32_v1` and `sim-sm::custom` | Runnable, model-relative timing only |
| Functional ITS example | `simt_its_v1`, one group, bounded PTX subset | Functional example, not a timing backend |
| NCU projection | Trace/card features to NCU GPC cycles and duration | Fast research lane; related-cohort evidence only |
| CUPTI/NAK/PMA probes | Independent launch/resource observations | Useful feature studies; no physical closure |

Build and capture commands live in [`runbook_host.md`](runbook_host.md). Core
architecture and input contracts are [`ARCHITECTURE.md`](ARCHITECTURE.md),
[`COMPONENTS.md`](COMPONENTS.md), [`archprofile.md`](archprofile.md), and
[`traceg_adapter_contract.md`](traceg_adapter_contract.md).

## Retained reference projection

The reference implementation is
`tools/sm86_ncu_projection_surrogate.py`, configured by
[`sm86_refined_model_config_20260929.json`](sm86_refined_model_config_20260929.json).
Its role is `retained_reference_experimental`: a label-disciplined, fast
projection for the current related base/large cohorts. It is not promoted to a
universal simulator model.

The config records the current experimental committee and unbound result paths.
Its policy is:

- keep NCU fields out of features and route selection for the scored row;
- preserve issue/event/completion/launch/typed-tail frontiers;
- prefer exact shape/occupancy anchors only inside a known calibration hull;
- otherwise emit `candidate_fallback`, widen the interval with distance, and
  keep the distance diagnostic visible;
- report end-to-end speed only when acquisition, replay, and projection costs
  are measured together. Existing millisecond figures are post-processing
  measurements, not simulator speedups.

### Current matrix observations

The four cells below are retained from the temporal campaigns. Each cell has
15/15 emitted rows; the cohorts are related and the NCU labels are score-only.

| Phase | Regime route | GPC MAPE | Duration MAPE | Interpretation |
| --- | --- | ---: | ---: | --- |
| cold | low/unknown neutral | 8.36% | 8.08% | fixed neutral event/frontier route |
| warm | low/unknown neutral | 8.02% | 7.90% | fixed neutral event/frontier route |
| cold | independently witnessed boost elasticity | 9.88% | 9.67% | bounded clock response |
| warm | independently witnessed boost elasticity | 9.36% | 9.76% | bounded clock response |

Source index: `results/alpha/sm86_regime_matrix_goal_20260930/matrix.json`.
These values are development observations, not a universal accuracy target or a
claim that every workload is in-distribution.

### Committee alternatives

The following tools are retained as explicit, non-promoted alternatives rather
than hidden fallbacks: `sm86_tile_lattice_projection.py`,
`sm86_tile_pipeline_envelope.py`, `sm86_frontier_ratio_bundle.py`,
`sm86_stage_interval_projection.py`, `sm86_pm_state_envelope.py`,
`sm86_cache_reuse_signature.py`, `sm86_warm_cache_scenario_interval.py`,
`sm86_nak_stall_prior_projection.py`, `sm86_native_cupti_surface_projection.py`,
`sm86_native_event_frontier_crosswalk.py`,
`sm86_endpoint_domain_committee.py`, `sm86_endpoint_regime_committee.py`, and
`sm86_endpoint_regime_router_v2.py`. They model effects of missing state; none
establishes the missing mechanism.

The corrected interaction-envelope paths are under
`results/alpha/sm86_interaction_scope_corrected_20260930/`. They use
opposite-cohort calibration and empirical scenario coverage. The older pooled
v1 report is historical and must not be used as the current interval claim.

## Evidence and corrections

- Native CUPTI PC collection covers 15/15 current rows: 470,506 samples /
  21,161 PC records for the no-warmup campaign and 920,753 samples / 42,500
  records for the warmup-inclusive campaign, with zero reported drops. The
  latter includes eight warmups; neither campaign is identical to an NCU cold
  flush. `imc_miss` is immediate-constant-cache miss, not instruction-cache
  miss.
- Independent Mesa NAK SM86 operation families, CUDA-ELF template bridging,
  direct PMA telemetry, repeated CUPTI sampling, and public RM flush flags are
  retained as feature/semantics evidence. They do not reveal exact issue tags,
  unrestricted ABI/ISA generation, cache read-from/DRAM ownership, or one
  synchronized timing endpoint.
- The previous “1,500--2,050x” comparison is withdrawn as scope-mismatched.
  Projection-only post-processing time cannot be divided into replay time for a
  larger set and called an end-to-end simulator speedup.
- Prior 8--10% summaries are temporal/development repetitions, not untouched
  new-family validation. Base/large are related cohorts, not universal OOD
  evidence. Feature construction can be label-free while model selection in a
  historical campaign was not fully independent; do not restate it as blind
  selection.
- A 10% burst fraction is a known-campaign heuristic for clock regime routing,
  not a proven launch-local clock witness. NCU-local frequency and external
  asynchronous clock samples remain different domains.
- All-cycle CUPTI states and `_not_issued` subsets overlap; the v2 decoder
  documents that distinction. Older opcode-surface joins are negative
  sensitivity results, not a universal scheduler implementation.

## Four unresolved boundaries

| Boundary | What is observable | Safe replacement |
| --- | --- | --- |
| Physical pipe/issue identity | aggregate counters, semantic opcode groups, trace issue frontier | typed candidate domains, resource envelopes, interval/partial-order constraints |
| General CUDA ABI/ISA generation | bounded NAK bodies and template ELF bridge | operation-family cards, independent templates, conservative fallback |
| Cache ownership/read-from/DRAM completion | addresses, reuse signatures, PMA/resource changes, public flush effects | cold/warm scenario intervals, reuse-distance and completion-state models |
| Unified timing endpoint | NCU, CUPTI, CUDA-event, host and replay clocks separately | endpoint-specific transforms and explicit uncertainty, never silent scalar merge |

These replacements are sufficient to improve a fast metric approximation, but
they do not close the physical research goal. A future external witness could
be proprietary RM/firmware telemetry, a new public endpoint, or a genuinely
matched device/simulator intervention; absent that, keep the fields unknown.

## OOD and fallback contract

“Covered” means the tool produced a bounded candidate row for the declared input
schema. It does **not** mean accuracy is guaranteed for arbitrary workloads.
For an unmatched shape, missing metadata, or out-of-hull feature vector:

1. retain the observation and emit `candidate_fallback`;
2. widen the interval using the documented distance/scenario rule;
3. expose the reason and distance in output;
4. never silently substitute an NCU label or refuse solely because shape is
   unmatched.

No model can promise zero OOD risk for every future workload without a defined
input distribution and independent observations. The current contract avoids
an empty result for known campaign rows while keeping uncertainty visible.

## Source and configuration anchors

- Native model: `sim/trace`, `sim/sm`, `sim/mem`, `sim/engine`, `sim/run`.
- Capture/value schema: `workloads/research_probes/sm86_value_differential/`.
- Alpha campaign: `workloads/research_probes/sm86_ncu_demo_benchmark/alpha_campaign.py`.
- Projection config: `docs/sm86_refined_model_config_20260929.json`.
- Experimental tools: `tools/sm86_*.py` and `tools/nvidia_rm/`;
  `tools/nvptx_compile_sm86/` contains the bounded compiler/ELF probe.
- Retained external literature floor: `docs/accelsim_literature_lower_bound_manifest_20260928.json`
  and its schema. It is a separate accuracy/throughput reference, not local
  same-card validation.

## Next useful seam (optional, not a queue)

If the goal is resumed, choose one falsifiable seam: a counter-free,
family-disjoint holdout with total-cost accounting; a public endpoint that
couples cache state to completion; or a native replay result that exercises the
new v2 payload metadata. Do not widen the workload set merely to make a score
look in-distribution, and do not call the blocked goal complete by renaming a
surrogate or diagnostic.

The 2026-09-30 documentation consolidation intentionally ran no tests,
hashing, or verification workflows. That is a record of this maintenance pass,
not a permanent prohibition on ordinary scientific validation.
