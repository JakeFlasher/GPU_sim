# SM86 branch-stream redirect experiment — 2026-10-15

## Hypothesis and experiment

H26 tested a source-motivated instruction-front-end hypothesis: when a dynamic
branch transfers to a non-fall-through PC, invalidate the per-SMSP fall-through
stream-buffer lines before fetching the traced target. The dynamic trace already
supplies the target path, so the experiment changed only instruction-side
prefetch state; L1/L2/DRAM data state was untouched.

Five branch-heavy real roots completed full-first replay. The complete A/B run
completed 356 roots (286 `ok`, 62 mixed-code controls, 8 replay/input failures),
12,392 launches with exactly 12 workers:

`../sm86_replay_results/full_corpus_h26_branch_redirect_20261015_v1/full.jsonl.zst`

Post-replay reduction:
`../sm86_replay_results/metric_report_h26_branch_redirect_20261015_v1/`.

## Falsifying result

The flush arm is rejected from the default model. Compared with H25, full-corpus
mean APE moved in the wrong direction:

| Metric | H25 | H26 branch flush | Delta |
|---|---:|---:|---:|
| GPC elapsed cycles | 12.90% | 14.37% | +1.47 pp |
| SM active cycles | 18.02% | 22.03% | +4.01 pp |
| elapsed IPC | 12.31% | 12.73% | +0.42 pp |

The representative roots showed 404–1,293 extra cycles, and the complete run
did not improve L1/L2/DRAM accuracy. The source mechanism is therefore not
promoted without a more precise branch-predictor/path-prior model; the timing
change has been removed from the active code path. The compressed artifact is
retained as negative evidence, not as the default result.

## Next seam

Do not globally flush stream buffers on every traced non-fall-through branch.
Future front-end work must distinguish compiler/static path prediction and
branch-target fetch state, or remain an observer. The current default returns
to H25/H24 timing with the explicit dynamic-youngest audit and barrier re-arm.
