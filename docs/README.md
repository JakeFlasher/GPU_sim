# Documentation map

Start with [PROJECT_MEMORY](PROJECT_MEMORY.md) and [STATUS](STATUS.md), then
open only the relevant technical page.

| Need | Document |
|---|---|
| Modeling scope and evaluation | [Predictive contract](PREDICTIVE_CLOSURE.md) |
| Next implementation | [Latest-review R8 refinement program](../plans/latest_review_refinement_20261002.md) · [Second-round history](../plans/second_round_refinement_20261001.md) · [first-round history](../plans/sm86_reification_refactor_20261001.md) |
| Build, trace, replay | [Runbook](runbook_host.md) |
| Focused checks | [Testing](testing.md) |
| Review history | [Review/refactor history](REVIEW_HISTORY.md) |
| Architecture and source map | [Architecture](ARCHITECTURE.md), [components](COMPONENTS.md) |
| Cards and predictive profiles | [Configuration](archprofile.md) |
| Prediction/observation schemas | [`reification_prediction_manifest_v1`](reification_prediction_manifest_v1.schema.json), [`reification_prediction_v2`](reification_prediction_v2.schema.json), [`ncu_observation_v2`](ncu_observation_v2.schema.json), [`reification_prediction_evaluation_v2`](reification_prediction_evaluation_v2.schema.json), [`reification_native_macro_witness_v1`](reification_native_macro_witness_v1.schema.json) |
| Accuracy-first NCU protocol | [Declared SM86 NCU accuracy protocol](sm86_ncu_accuracy_protocol_v1.json) · [Protocol schema](sm86_ncu_accuracy_protocol_v1.schema.json) |
| A8 live-campaign plan | [Accuracy-first A8 plan](../plans/accuracy_first_live_campaign_a8_20261002.md) · [bounded-run schema](sm86_workload_run_v1.schema.json) |
| Completion frontier program | [Writeback/coherence and native-dispatch plan](../plans/sm86_completion_frontier_and_native_dispatch_20261002.md) |
| Completion frontier contract | `sm86_completion_frontier_contract_v1.json` · `tools/report_sm86_completion_frontier.py` |
| Completion frontier C1 live evidence | `sm86_completion_frontier_c1_small_summary_v1.json` · `sm86_completion_frontier_c1_visibility_summary_v1.json` |
| Completion frontier C3 probe | `sm86_completion_frontier_c3_service_probe_v1.json` |
| Completion frontier C3 holdout | `sm86_completion_frontier_c3_store2_holdout_v1.json` |
| NCU identity capture | [Identity schema](sm86_ncu_identity_v1.schema.json) · `tools/capture_sm86_ncu_identity.py` |
| NCU campaign preflight | [Preflight schema](sm86_ncu_campaign_preflight_v1.schema.json) · `tools/preflight_sm86_ncu_campaign.py` |
| First accuracy campaign | `sm86_accuracy_campaign_v1_{manifest,predictions_pre_ncu,observations_post_ncu,evaluation_join,repetition_summary}.json` |
| Intervention contrasts | `sm86_ncu_intervention_plan_v1.json` · `sm86_accuracy_intervention_comparison_v1.json` |
| Real selective-path result | `sm86_accuracy_a7_real_selective_result_v1.json` · [schema](sm86_accuracy_a7_real_selective_result_v1.schema.json) |
| A8 vertical live slice | [summary](sm86_a8_vertical_slice_v1.json) · [prediction](sm86_a8_vertical_predictions_pre_ncu_v1.json) · [observation](sm86_a8_vertical_observations_post_ncu_v1.json) · [evaluation](sm86_a8_vertical_evaluation_v1.json) |
| Third-round result | [R6 corpus and observer evaluation](review_round3_20261002/README.md) · [corpus manifest](sm86_reification_corpus_v1_20261002.json) |
| R9.0 native boundary probe | [Canary termination and unknown-join result](review_round4_20261002/README.md) |
| R9.1 production witness attempt | [Negative native-dispatch result](review_round5_20261002/README.md) |
| Live SM86 development probe | [RTX 3090 Ti small-suite result](review_round6_20261002/README.md) |
| Live SM86 large campaign | [168-block RTX 3090 Ti result](review_round7_20261002/README.md) |
| Live SM86 writeback campaign | [Large dirty-stream result](review_round8_20261002/README.md) |
| Live cache-policy contrast | [All-cache versus none-cache result](review_round9_20261002/README.md) |
| Full retained-trace cost campaign | [Detailed versus analyzer utility result](review_round10_20261002/README.md) |
| Latent and payload APIs | [Contract](sm86_factorized_latent_contract_20260930.md) |
| Experimental semantics | [Diagnostics](experimental_diagnostics.md) |
| Prior measurements | preserved under results/ and experiment inputs |

`AGENTS.md` governs execution. Experiment outputs are data, not queued work.
SOL/campaign source, datasets, environments, schemas, and knowledge cards remain
in place; this index does not require loading them for unrelated development.
