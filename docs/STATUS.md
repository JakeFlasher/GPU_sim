# Unified simulator status

## Architecture lanes

| Lane | Card | Capture root | Status |
|---|---|---|---|
| SM86 / RTX 3090 Ti | `arch_profiles/cards/sm_86.json` | `sm86_rtx_3090_capture` | Compatibility lane with card-backed L1/L2/P-chase and SM86 observer candidates. |
| SM120 / RTX 5090 | `arch_profiles/cards/sm_120-rtx-5090.json` | `sm120_rtx_5090_capture` | Effective/conditional lane with typed geometry gaps, TMA sidecars, and optional prototype DRAM service. |

## Runtime contract

`sim-run` resolves architecture-scoped behavior from `--arch-card`. SM86 may
use `--pchase-card`, an explicit DRAM service-rate hypothesis, and bounded
store/fence admission candidates. SM120 may use effective L1/L2, latent
frontier, TMA sidecars, and the explicitly marked unvalidated carded-DRAM
prototype. No SM86 value is silently used for an SM120 research gap.

`tools/sm120_corpus_replay.py` is retained as the historical entry point but
is architecture-neutral: use `--target-arch auto|sm_86|sm_120`, matching card
and capture root, and either the standard twelve-worker replay or the explicit
sequential one-worker timing baseline.

## Scientific boundaries

NCU data is post-replay comparison only. Native SM86 controls and explicitly
labeled SM86 substitutes are not relabeled as native SM120 features. Effective
models report residuals and typed unavailable fields instead of inventing
proprietary topology.

Detailed historical evidence remains in `docs/SM86_PROJECT_MEMORY.md`,
`docs/SM120_PROJECT_MEMORY.md`, and the lane-specific `sm86_*` report files.
