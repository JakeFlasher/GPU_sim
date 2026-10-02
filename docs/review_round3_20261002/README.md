# R6 corpus and independent observer evaluation (2026-10-02)

This directory is a **research result**, not an execution gate or an authority/provenance artifact. It records the completed R6 stage for the third-round revision. The files deliberately keep the pre-observation prediction document separate from the post-prediction NCU observation document.

## Evidence boundary

- Native input: retained SM86 traces/control sidecars from `results/alpha/full_3090ti_20260930_temporal_cold_live2/`.
- The manifest intentionally references those retained working-tree inputs
  rather than copying a large trace corpus into the source commit; the three
  JSON artifacts below preserve the completed run and can be rejoined without
  target leakage when the retained inputs are present.
- Prediction command: `tools/run_prediction_manifest.py` with rebuilt `sim/target/release/sim-run` and `docs/sm86_service_rate_profile_20261001.json`.
- The pre-observation prediction document carries source bindings for each workload's trace, control sidecar, architecture card, pchase card, and declared launch identity. These are semantic input paths, not hashes or execution gates.
- Observation command: `tools/build_ncu_observation_set.py`; it reads only the retained NCU CSV files and does not read predictions.
- Join command: `tools/evaluate_prediction_splits.py`; it checks metric, endpoint, clock domain, cache policy, launch identity, repetition count, GPU UUID, compute capability, SM clock, device index, workload identity, and family-held-out split closure.
- The joined document is checked against `docs/reification_prediction_evaluation_v2.schema.json` in addition to the manifest, prediction, and observation schemas.
- The observation document is explicitly labeled `manifest_labeled_historical`: the builder copies the retained campaign declaration (GPU UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, 1350 MHz, device 0) and does not parse a live device/NCU identity proof. It is not a live-device assertion.
- The simulator output calls the endpoint quantity a declared observer **candidate**. Its role is `ncu_observer_elapsed_interval_not_hardware_endpoint_identity`; no simulator-to-NCU timing conversion is claimed.

## Corpus closure

- 15 workload rows, 15 joined rows, and 30 observation samples (two per workload).
- 6 operation families; each family is assigned to exactly one split: `attention` → `held_out`, `convolution` → `held_out`, `elementwise_reduction` → `held_out`, `memory_embedding` → `held_out`, `normalization` → `calibration`, `tensor_gemm` → `development`.
- Split counts: calibration 3, development 4, held-out 8.
- No workload or family crosses split boundaries. The held-out rows are therefore an architectural-family holdout, not a random row split.
- All 15 rows produced a declared lower-bound point. The native launch path remained detailed replay with a latent sidecar; `macro_execution_observed=false` and `step_reduction=null` in the prediction document.

## Joined results

| Split | Rows | Finite intervals | Open intervals | Finite coverage | Open lower-bound consistency | Mean APE (%) |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 3 | 0 | 3 | — | 0.667 | 14.640 |
| development | 4 | 0 | 4 | — | 1.000 | 28.232 |
| held_out | 8 | 0 | 8 | — | 0.750 | 25.938 |

Interpretation:

- Every native row has an open upper because the replay reports active dirty-writeback/coherence or other unresolved retirement state. Consequently finite-interval coverage is **not applicable**, and no 100% “coverage” claim is made.
- `open_lower_bound_consistency` is only a diagnostic check that the declared simulator lower value is below the median NCU value; it is not interval calibration and is not used for fitting or selection. The values vary by family (0.667, 1.000, and 0.750), so even this diagnostic is not a hardware lower-bound theorem.
- Mean APE compares the declared observer candidate lower point with the independent NCU median only as a negative/model-distance diagnostic. It is not an accuracy result for a finite observer interval.
- The result is intentionally negative for native compression: no row reports a macro dispatch or execution-step reduction. Runtime and memory columns measure the detailed replay-plus-reporting process, not a macro speedup.

## What this closes

1. Prediction is materialized before observation; the prediction JSON contains no NCU value.
2. The corpus is larger than the three earlier canaries and has explicit family-held-out semantics.
3. Open, finite, and refused strata are disjoint in the evaluator; open rows are not counted as finite interval coverage.
4. Row-level observer and declared hardware metadata are mandatory, so a duration CSV or mismatched declared contract cannot silently join a cycle prediction. The historical identity label is not independent host verification.
5. Record coverage and execution-step reduction remain separate; the native corpus reports no step reduction because the front-end exact-fallback boundary is still active.

## What remains unproven

- A native front-end-enabled region with closed issue, response, barrier, visibility, writeback, and completion state.
- A finite native observer upper on real traces; the correct current behavior is an open upper and exact fallback.
- Silicon-level accuracy, a scheduler/queue reconstruction, cache ownership/read-from causality, or an architectural design ranking.
- Any compression/runtime benefit on the native front-end path.

The closed two-instruction scalar `SmCore` issue-model differential remains a separate level-3 model/native-seam result documented in `docs/REVIEW_HISTORY.md`; it must not be conflated with this 15-workload level-4 comparison.

## Reproduction

```bash
cargo build --manifest-path sim/Cargo.toml --release -p sim-run
python3 tools/run_prediction_manifest.py \
  docs/sm86_reification_corpus_v1_20261002.json \
  --sim-run sim/target/release/sim-run \
  --profile docs/sm86_service_rate_profile_20261001.json \
  --timeout 60 \
  --output /tmp/predictions_pre_ncu.json
PYTHONPATH=tools python3 tools/build_ncu_observation_set.py \
  docs/sm86_reification_corpus_v1_20261002.json \
  --campaign-root results/alpha/full_3090ti_20260930_temporal_cold_live2 \
  --source full_3090ti_20260930_temporal_cold_live2/ncu_csv \
  --output /tmp/observations_post_ncu.json
PYTHONPATH=tools python3 tools/evaluate_prediction_splits.py \
  /tmp/predictions_pre_ncu.json /tmp/observations_post_ncu.json
```
