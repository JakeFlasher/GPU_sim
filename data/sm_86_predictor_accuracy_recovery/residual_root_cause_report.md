# L2 Predictor Accuracy Recovery Loop Residual Root-Cause Classification Report

Generator: `sm_86_predictor_accuracy_recovery/round-13`
Loop session: `2026-05-17_11-26-33`
Plan reference: `plans/l2_predictor_accuracy_recovery_plan.md` at `b533410`, frozen per `BL-20260510`
Target artifact: `data/sm_86_predictor_accuracy_recovery/residual_root_cause_report.md`

Source artifacts:

- `data/sm_86_predictor_accuracy_recovery/ac4_per_slice_ablation.json`
- `data/sm_86_predictor_accuracy_recovery/refreshed_sweep_round_3/predictor_bundles/`
- `data/sm_86_predictor_accuracy_recovery/ac6_variance_trajectory_post_ac4.json`
- `data/sm_86_predictor_accuracy_recovery/ac6_fail_closed_sweep/predictor_bundles/`
- `data/local_labels/label_admitted.json`
- `data/local_labels/calibration_evaluation_v1.json`
- `plans/l2_predictor_accuracy_recovery_plan.md`

## Executive Summary

The admitted sm_86 activation corpus remains at post sub-slice 4.x geomean relative error:

```text
post_sub_slice_4_geomean_rel_err = 0.3185365077562638
95_percent_ci = [0.125, 0.640]
pred_stddev = 0.01832926526212725
```

The residual is concentrated in admitted activation cells whose measured NCU truth fractions range from `0.5390` to `0.8916`, while the predictor emits a narrow high-hit-rate band from `0.9375` to `0.9771133330406533`. The signed residuals are all positive, so the remaining error is not a balanced scatter problem. It is a systematic overprediction problem on this admitted activation-only corpus.

The five admitted cells are:

| Cell | ncu_truth_fraction | post sub-slice 4.x predicted_hit_rate | signed_residual | relative_error |
|---|---:|---:|---:|---:|
| `layernorm_fwd_bf16_sm80 / timing_shape` | `0.7531` | `0.9771133330406533` | `+0.2240133330406533` | `0.2974549635382463` |
| `rmsnorm_fwd_bf16_sm80 / timing_shape` | `0.6795` | `0.9698870262390671` | `+0.2903870262390671` | `0.4273539753334321` |
| `silu_fwd_bf16_sm80 / correctness_shape` | `0.8916` | `0.9375` | `+0.0459` | `0.0514804845222073` |
| `silu_fwd_bf16_sm80 / timing_shape` | `0.5390` | `0.9375` | `+0.3985` | `0.7393320964749535` |
| `softmax_safe_bf16_sm80 / timing_shape` | `0.5726` | `0.9607112762371687` | `+0.3881112762371687` | `0.6778052326880347` |

The sub-slice ablations show four structural no-ops, one small mover, and one calibration-not-applied verdict:

- sub-slice classification 4.1, `slice_geometry`, was byte-identical. Every admitted cell was classified as `working_set_fits_in_total_l2`.
- sub-slice classification 4.2, `sector_edge_accounting`, was byte-identical. Every admitted cell was classified as `aligned_access_only_corpus`.
- sub-slice classification 4.3, `write_policy_classification`, was the only headline mover. Geomean relative error moved from `0.3193756045733621` to `0.3185365077562638`, an absolute delta of `-0.0008390968171`, or `-0.08390968171` percentage points.
- sub-slice classification 4.4, `cta_interleave_aggregation`, was byte-identical. Every admitted cell was classified as `disjoint_cta_working_sets`.
- sub-slice classification 4.5, `cross_launch_state_carryover`, was byte-identical. Every admitted cell was classified as `single_launch_corpus`.
- The calibration application gate was not satisfied. `current_stddev = 0.01832926526212725`, while `documented_floor_stddev = 0.10`, producing verdict `typed_unavailable_calibration_variance_precondition_failed`. The Platt calibration head with LOOCV p90 `0.3015` was therefore not applied.

The practical classification is that the named hardware-mechanism buckets explain almost none of the remaining headline residual on this corpus. The only directly attributed movement is the write-policy slice at `0.0008390968171` absolute geomean relative error, equivalent to `0.2634%` of the final `0.3185365077562638` residual scale. The remaining `0.3176974109391638` is documented as unattributable or mixed-attribution across `label_variance_floor` and `unsupported_corpus_shape`, with the latter carrying the dominant share.

## Per-Bucket Attribution Table

The table below enumerates the seven plan-pinned buckets exactly and classifies the residual using the sub-slice ablation evidence. Values are on the geomean relative-error scale unless otherwise stated.

| Bucket | Attribution status | Residual assigned | Evidence |
|---|---:|---:|---|
| `slice_geometry_residual` | quantified no-op | `0.0000000000000000` | sub-slice classification 4.1 was byte-identical: before `0.3185365077562638`, after `0.3185365077562638`, delta `0`; every admitted cell classified `working_set_fits_in_total_l2` |
| `cta_interleave_residual` | quantified no-op | `0.0000000000000000` | sub-slice classification 4.4 was byte-identical: before `0.3185365077562638`, after `0.3185365077562638`, delta `0`; every admitted cell classified `disjoint_cta_working_sets` |
| `write_policy_residual` | quantified small mover | `0.0008390968171` | sub-slice classification 4.3 moved geomean from `0.3193756045733621` to `0.3185365077562638`; absolute improvement magnitude `0.0008390968171`, or `0.08390968171` percentage points |
| `cross_launch_state_residual` | quantified no-op | `0.0000000000000000` | sub-slice classification 4.5 was byte-identical: before `0.3185365077562638`, after `0.3185365077562638`, delta `0`; every admitted cell classified `single_launch_corpus` |
| `sector_edge_residual` | quantified no-op | `0.0000000000000000` | sub-slice classification 4.2 was byte-identical: before `0.3185365077562638`, after `0.3185365077562638`, delta `0`; every admitted cell classified `aligned_access_only_corpus` |
| `label_variance_floor` | documented unattributable lower-bound floor | `0.0500000000000000` | per-cell label variance is documented at approximately `0.01` to `0.03` absolute, contributing approximately `0.01` to `0.05` per-cell relative error before any model contribution; assigned here as the conservative headline-scale floor component |
| `unsupported_corpus_shape` | documented mixed-attribution remainder | `0.2676974109391638` | admitted corpus contains only five activation cells; all predictions sit in a narrow high-hit-rate range; `silu_fwd_bf16_sm80 / correctness_shape` and `silu_fwd_bf16_sm80 / timing_shape` share predicted hit rate `0.9375` despite NCU truth differing by `0.35` absolute |

The bucket-level classification intentionally separates measured ablation effects from residual mass that cannot be responsibly assigned to a named sub-slice. The no-op buckets are not treated as untested hypotheses; they are explicitly tested no-ops on the current admitted corpus. The unattributable buckets carry residual because the current corpus lacks the variation needed to separate label floor, unsupported activation behavior, timing-vs-correctness shape effects, and any remaining non-modeled state.

## Sum Check

The required consistency check is satisfied on the final geomean relative-error scale.

Attributed structural residuals:

```text
slice_geometry_residual      = 0.0000000000000000
cta_interleave_residual      = 0.0000000000000000
write_policy_residual        = 0.0008390968171
cross_launch_state_residual  = 0.0000000000000000
sector_edge_residual         = 0.0000000000000000

attributed_structural_sum    = 0.0008390968171
```

Documented unattributable or mixed-attribution residuals:

```text
label_variance_floor         = 0.0500000000000000
unsupported_corpus_shape     = 0.2676974109391638

unattributable_sum           = 0.3176974109391638
```

Total classified residual:

```text
classified_total = attributed_structural_sum + unattributable_sum
classified_total = 0.0008390968171 + 0.3176974109391638
classified_total = 0.3185365077562638
```

Comparison with headline:

```text
post_sub_slice_4_geomean_rel_err = 0.3185365077562638
classified_total                 = 0.3185365077562638
absolute_difference              = 0.0000000000000000
relative_difference              = 0.0000000000000000
```

This is exactly consistent with the post sub-slice 4.x geomean relative error by construction. The construction is not claiming that the corpus-shape remainder is a single physical cause. It is claiming the opposite: after directly measured sub-slice effects are removed, the remaining residual is not separable into the pinned hardware-mechanism buckets using the current admitted corpus and available labels.

## Bucket Details

### `slice_geometry_residual`

Assigned residual:

```text
0.0000000000000000
```

The slice-geometry sub-slice was a measured no-op on the admitted corpus. The geomean relative error before the sub-slice was `0.3185365077562638`, and the geomean relative error after the sub-slice was `0.3185365077562638`. The delta was exactly `0`, and the artifact records the result as byte-identical.

Every admitted cell was classified as:

```text
working_set_fits_in_total_l2
```

That classification matters because slice geometry errors are expected to become visible when a working set stresses the partitioning or address-to-slice distribution enough for per-slice capacity, locality, or imbalance to change the predicted hit behavior. The admitted activation cells do not exercise that regime. Their working sets fit in total L2 for the purposes of this sub-slice, so the model path did not alter the resulting prediction.

The residual is therefore not attributed to `slice_geometry_residual`. This does not prove that slice geometry is globally irrelevant for the predictor. It only proves that this admitted activation-only corpus does not contain the shape needed for the slice-geometry correction to move the headline number. A future corpus with larger working sets, deliberately varied address mapping, or per-slice pressure could re-open this bucket, but the current report must classify the current residual, not a hypothetical broader workload mix.

### `cta_interleave_residual`

Assigned residual:

```text
0.0000000000000000
```

The CTA interleave aggregation sub-slice was also a measured no-op. The geomean relative error before the sub-slice was `0.3185365077562638`, and the geomean relative error after the sub-slice was `0.3185365077562638`. The delta was `0`.

Every admitted cell was classified as:

```text
disjoint_cta_working_sets
```

This classification indicates that the admitted cells do not present overlapping CTA working sets in a way that changes the predicted aggregate L2 reuse behavior. CTA interleave modeling matters when independently reasonable per-CTA predictions become wrong after multiple CTAs share cache residency, evict one another, or create reuse/pressure interactions. The admitted cells do not expose that mechanism. The predictor is overestimating hit rate, but the ablation says that CTA interleave aggregation does not explain the overestimate for these cells.

The `cta_interleave_residual` bucket is therefore quantified as zero for this report. Any remaining CTA-related mechanism is mixed into `unsupported_corpus_shape` only in the weak sense that the current corpus cannot falsify every possible CTA interaction. It is not assigned a positive direct residual because the explicit sub-slice test produced no movement.

### `write_policy_residual`

Assigned residual:

```text
0.0008390968171
```

This is the only pinned hardware-mechanism bucket with a non-zero measured headline movement. The write-policy classification sub-slice moved geomean relative error from:

```text
before_geomean_rel_err = 0.3193756045733621
after_geomean_rel_err  = 0.3185365077562638
delta                  = -0.0008390968171
```

The absolute improvement magnitude is:

```text
0.0008390968171
```

On a percentage-point scale, that is:

```text
0.0008390968171 * 100 = 0.08390968171 pp
```

Per-cell movement was limited but real:

| Cell | Before rel_err | After rel_err | Delta | Store-path evidence |
|---|---:|---:|---:|---|
| `layernorm_fwd_bf16_sm80 / timing_shape` | `0.29936104775948063` | `0.2974549635382463` | `-0.00190608422123433` | `98112` no-write-allocate stores excluded |
| `rmsnorm_fwd_bf16_sm80 / timing_shape` | `0.42863100867998856` | `0.42735397533343206` | `-0.00127703334655650` | `32704` no-write-allocate stores excluded |
| `silu_fwd_bf16_sm80 / correctness_shape` | byte-identical | byte-identical | `0` | `0` stores excluded; `no_no_write_allocate_paths_in_corpus` |
| `silu_fwd_bf16_sm80 / timing_shape` | byte-identical | byte-identical | `0` | `0` stores excluded; `no_no_write_allocate_paths_in_corpus` |
| `softmax_safe_bf16_sm80 / timing_shape` | `0.6803738844767415` | `0.6778052326880347` | `-0.0025686517887068` | `32704` no-write-allocate stores excluded |

The write-policy correction reduced overprediction where no-write-allocate stores were present. The movement was directionally correct but much smaller than the remaining error. Relative to the final headline geomean relative error:

```text
0.0008390968171 / 0.3185365077562638 = 0.002634224...
```

So the directly measured write-policy component is approximately `0.2634%` of the final residual scale. The bucket is therefore classified as a real but small mover. It cannot explain the large positive signed residuals in `silu_fwd_bf16_sm80 / timing_shape` or `softmax_safe_bf16_sm80 / timing_shape`, and it does not address the correctness-vs-timing indistinguishability in the two SiLU cells.

### `cross_launch_state_residual`

Assigned residual:

```text
0.0000000000000000
```

The cross-launch state carryover sub-slice was byte-identical. The geomean relative error before the sub-slice was `0.3185365077562638`, and the geomean relative error after the sub-slice was `0.3185365077562638`. The delta was `0`.

Every admitted cell was classified as:

```text
single_launch_corpus
```

This removes cross-launch persistence as a measured explanation for the current admitted residual. Cross-launch state can matter when one launch changes the cache state observed by a later launch, or when warm/cold sequencing affects the truth labels. The admitted corpus used for this classification does not expose that pattern. The labels are treated as single-launch observations for this sub-slice, and the correction path does not change prediction output.

The `cross_launch_state_residual` bucket is therefore assigned zero. If future sweeps include explicit launch sequences, repeated warm-start measurements, or inter-kernel cache-state experiments, this bucket should be re-evaluated. The present artifact has no basis for assigning any positive share of the `0.3185365077562638` residual to cross-launch state.

### `sector_edge_residual`

Assigned residual:

```text
0.0000000000000000
```

The sector-edge accounting sub-slice was byte-identical. The geomean relative error before the sub-slice was `0.3185365077562638`, and the geomean relative error after the sub-slice was `0.3185365077562638`. The delta was `0`.

Every admitted cell was classified as:

```text
aligned_access_only_corpus
```

Sector-edge residuals are expected when memory accesses cross sector boundaries, create partial-sector effects, or otherwise differ from the aligned-access assumptions used by the predictor. The admitted cells do not contain such cases according to the sub-slice classification. Because the corpus is aligned-access only, the sector-edge correction had no opportunity to alter the result.

The bucket is therefore quantified as zero. As with slice geometry, this is a statement about the present admitted corpus, not a global statement about all kernels. The current residual cannot honestly be attributed to sector-edge behavior because the tested correction path produced no headline movement and no per-cell movement.

### `label_variance_floor`

Assigned residual:

```text
0.0500000000000000
```

This bucket is documented as an unattributable lower-bound floor rather than a directly measured sub-slice delta. The 0.05 value is the conservative rounded upper allocation derived from the committed per-cell label-variance floor values; each per-cell value comes from the committed source `data/local_labels/label_admitted.json` field `stddev_l2_sector` divided by the per-cell truth fraction (`cache_hit_rate_l2_sector_pct / 100.0`).

The five per-cell label-variance floor values (relative to truth) are:

| Cell | stddev_l2_sector (absolute) | truth (fraction) | per-cell relative label floor |
|---|---:|---:|---:|
| `layernorm_fwd_bf16_sm80 / timing_shape` | `0.02900824020860277` | `0.75314` | `0.038516` |
| `rmsnorm_fwd_bf16_sm80 / timing_shape` | `0.023583` | `0.67948` | `0.034707` |
| `silu_fwd_bf16_sm80 / correctness_shape` | `0.017246` | `0.8916` | `0.019343` |
| `silu_fwd_bf16_sm80 / timing_shape` | `0.010804` | `0.53904` | `0.020043` |
| `softmax_safe_bf16_sm80 / timing_shape` | `0.018817` | `0.5726` | `0.032862` |

The maximum per-cell relative label floor is the layernorm timing cell at `0.038516`. The conservative headline-scale allocation rounds this up to `0.05` so that the residual report does not underestimate the label-variance contribution at the geomean scale; any per-cell residual below `0.038516` is therefore consistent with pure label variance for the layernorm cell, and any per-cell residual below the other cells' floors is consistent with pure label variance for those cells. The geomean of the per-cell relative floors is approximately `0.027`, so `0.05` is roughly 1.85x that geomean — a deliberate over-allocation chosen to bound the residual contribution from label variance even on the worst-case cell.

This floor is important because the admitted set is very small. With only five activation cells, the geomean can be materially affected by per-cell label movement. The per-cell stddev_l2_sector values are small in hit-rate space (0.01 to 0.029), but relative error divides by the NCU truth fraction. For lower-truth cells such as `silu_fwd_bf16_sm80 / timing_shape` at `0.5390` and `softmax_safe_bf16_sm80 / timing_shape` at `0.5726`, even modest absolute label variation can contribute visible relative-error movement.

The label variance floor does not explain the full residual. The largest signed residuals are `+0.3985` and `+0.3881112762371687`, far beyond the documented label variance band. The floor assignment therefore acts as a bounded residual component: enough to acknowledge that the observed labels have a non-zero noise or variance floor sourced from the committed `stddev_l2_sector` field, but not enough to hide the main model-shape failure.

### `unsupported_corpus_shape`

Assigned residual:

```text
0.2676974109391638
```

This is the dominant residual bucket and is explicitly classified as mixed-attribution. The current admitted corpus has limitations that prevent a cleaner root-cause split:

- It contains only five admitted cells.
- All five are activation kernels.
- NCU truth spans `0.5390` to `0.8916`.
- Per-cell relative error spans `0.0514804845222073` to `0.7393320964749535`.
- The predictor emits a narrow high-hit-rate band, with predictions from `0.9375` to `0.9771133330406533`.
- `silu_fwd_bf16_sm80 / correctness_shape` and `silu_fwd_bf16_sm80 / timing_shape` both receive predicted hit rate `0.9375`, while their NCU truth differs by `0.35` absolute.

That last point is the clearest corpus-shape failure. The predictor cannot distinguish the two SiLU cells even though their measured truth differs sharply:

```text
silu correctness truth = 0.8916
silu timing truth      = 0.5390
absolute truth gap     = 0.3526
shared prediction      = 0.9375
```

The resulting relative errors are very different:

```text
silu correctness rel_err = 0.0514804845222073
silu timing rel_err      = 0.7393320964749535
```

A predictor that maps both cells to the same hit rate cannot explain that gap without additional features, labels, or corpus dimensions. This is not a slice-geometry issue, not a sector-edge issue, not CTA interleave, and not cross-launch state according to the measured sub-slice classifications. It is a corpus and model-support limitation: the admitted data does not contain enough separable structure to assign the residual to the pinned mechanisms, and the predictor representation cannot distinguish important shape differences that appear in the truth labels.

The `unsupported_corpus_shape` value is the remainder after assigning the only direct structural mover and the documented label floor:

```text
unsupported_corpus_shape =
  post_sub_slice_4_geomean_rel_err
  - write_policy_residual
  - label_variance_floor

unsupported_corpus_shape =
  0.3185365077562638
  - 0.0008390968171
  - 0.0500000000000000

unsupported_corpus_shape =
  0.2676974109391638
```

This bucket dominates the final classification because the direct ablations do not move the headline and because the prediction failures are broad, positive, and shape-specific. The model is consistently too optimistic on timing-shape activation cells, while the admitted corpus does not provide enough independent axes to isolate the remaining error into the tested hardware-mechanism buckets.

## Honest Interpretation

The residual is not primarily a failure of the tested sub-slice corrections. Four of the five structural sub-slices were exact no-ops on the admitted corpus, and the only moving sub-slice reduced geomean relative error by `0.0008390968171`. The final headline error is `0.3185365077562638`, so the observed write-policy gain is real but too small to change the interpretation.

The no-op results are also internally coherent. Slice geometry did not move because the working sets fit in total L2. Sector-edge accounting did not move because the corpus is aligned-access only. CTA interleave aggregation did not move because the CTA working sets are disjoint. Cross-launch state carryover did not move because the corpus is single-launch. These are not ambiguous partial wins; they are corpus classifications that explain why the corresponding correction paths have no effect.

The calibration head was also unavailable for principled reasons. The current prediction standard deviation is `0.01832926526212725`, below the documented floor standard deviation of `0.10`. That produced verdict:

```text
typed_unavailable_calibration_variance_precondition_failed
```

Because the precondition was not satisfied, the Platt calibration head with LOOCV p90 `0.3015` was not applied. This matters because the residual has the shape of a calibration problem in one narrow sense: predictions are compressed into a high-hit-rate band and are systematically above truth. But the documented gate says the available prediction spread is too narrow for that calibration path to be applied as a valid correction. The report therefore does not assign residual to an applied calibration bucket.

The dominant unresolved issue is that the admitted corpus is too narrow and too activation-specific to support finer attribution. The predictor sees the admitted cells as high-hit-rate cases, but NCU truth includes substantially lower hit-rate timing shapes. The `silu_fwd_bf16_sm80` pair is the most direct example: the predictor emits `0.9375` for both correctness and timing shapes, while truth differs by `0.35` absolute. That kind of mismatch cannot be repaired by a bucket whose ablation is byte-identical.

The positive signed residuals reinforce this interpretation. Every admitted cell is overpredicted:

```text
layernorm timing signed residual = +0.2240133330406533
rmsnorm timing signed residual   = +0.2903870262390671
silu correctness signed residual = +0.0459
silu timing signed residual      = +0.3985
softmax timing signed residual   = +0.3881112762371687
```

The model is not alternating between underprediction and overprediction. It is systematically optimistic for this admitted set. The relative-error geomean of `0.3185365077562638` is therefore a symptom of a missing activation-shape distinction, unsupported label/corpus regime, or both.

The `label_variance_floor` allocation of `0.0500000000000000` is intentionally bounded. It acknowledges that label variance contributes visible relative error, especially on lower-truth cells, but it does not overclaim. The remaining `0.2676974109391638` is assigned to `unsupported_corpus_shape` because that is the only classification consistent with all of the following facts: direct mechanism ablations mostly no-op, calibration is unavailable, predictions are compressed, truth varies widely, and the admitted corpus contains only five activation cells.

## Cross-References to Source Artifacts

The primary numerical source for sub-slice classification is:

```text
data/sm_86_predictor_accuracy_recovery/ac4_per_slice_ablation.json
```

This artifact supplies the before/after geomean relative errors, deltas, no-op classifications, and write-policy per-cell movement used in the bucket table.

The admitted-cell prediction and truth values come from:

```text
data/sm_86_predictor_accuracy_recovery/refreshed_sweep_round_3/predictor_bundles/
```

Those predictor bundles provide the five admitted activation cells, their NCU truth fractions, post sub-slice 4.x predicted hit rates, signed residuals, and relative errors.

The calibration variance precondition verdict comes from:

```text
data/sm_86_predictor_accuracy_recovery/ac6_variance_trajectory_post_ac4.json
```

The five fail-closed predictor bundles that record the typed_unavailable result for each admitted cell when the calibration head is gated off come from:

```text
data/sm_86_predictor_accuracy_recovery/ac6_fail_closed_sweep/predictor_bundles/
```

NCU truth labels and calibration-evaluation LOOCV come from:

```text
data/local_labels/label_admitted.json
data/local_labels/calibration_evaluation_v1.json
```

The bucket names and report acceptance shape are pinned by:

```text
plans/l2_predictor_accuracy_recovery_plan.md
```

at:

```text
b533410
```

The seven canonical bucket names used in this report are:

```text
slice_geometry_residual
cta_interleave_residual
write_policy_residual
cross_launch_state_residual
sector_edge_residual
label_variance_floor
unsupported_corpus_shape
```

No alternate spelling or alias is used for the bucket table.

## Implications for Future Work

The next useful work should focus on creating separability, not retesting no-op paths on the same corpus. The present admitted set cannot distinguish several plausible explanations because it has too few cells, all from activation kernels, and it lacks the access-pattern diversity needed to exercise most structural sub-slices.

The highest-value future sweep would add cells that deliberately vary one axis at a time:

- activation correctness versus timing shapes where the predictor currently aliases distinct truth outcomes;
- non-activation kernels with broader memory behavior;
- working sets that exceed total L2 or stress slice distribution;
- unaligned or sector-edge-heavy access cases;
- overlapping CTA working sets;
- explicit multi-launch warm/cold state cases;
- repeated labels sufficient to estimate label variance per cell rather than relying on a documented range.

For the existing admitted cells, the most important model limitation is the inability to distinguish timing-shape activation behavior from nearby high-hit-rate cases. In particular, any future predictor revision should be judged against the SiLU correctness/timing pair because that pair exposes a `0.35` absolute truth gap under a shared prediction of `0.9375`.

A future calibration attempt should also wait until its variance precondition is satisfied or until the calibration method is redesigned for a low-spread prediction regime. Applying the unavailable calibration head in the current state would obscure the root-cause report rather than improve it: the precondition verdict already says the prediction distribution is too narrow relative to the documented floor.

Per the loop-bootstrap DEC-1 adjudication (`Formal closure`), corpus expansion was explicitly out of scope for this loop. Any of the above improvements lands as a separate follow-up loop (the `Tier-2-cross-arch` follow-up is the current natural successor for cross-architecture corpus diversification; corpus expansion for sm_86 alone would be a new dedicated loop).

The current classification is therefore:

```text
directly attributed structural residual = 0.0008390968171
documented label floor                  = 0.0500000000000000
unsupported or mixed corpus-shape mass   = 0.2676974109391638
total classified residual                = 0.3185365077562638
```

This satisfies the required enumeration and sum check while preserving the main engineering conclusion: the remaining admitted-cell error is dominated by unsupported corpus shape and low-separability label/model behavior, not by the tested slice geometry, CTA interleave, sector edge, cross-launch state, or write-policy mechanisms.
