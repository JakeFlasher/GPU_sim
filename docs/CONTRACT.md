# Unified simulator contract

## Deployment lanes

| Lane | Card | Meaning |
| --- | --- | --- |
| RTX 3090 Ti / SM86 | `arch_profiles/cards/sm_86.json` | Compatibility lane with card-backed L1/L2/P-chase inputs and explicitly scoped SM86 observers. |
| RTX 5090 / SM120 | `arch_profiles/cards/sm_120-rtx-5090.json` | Effective/conditional Blackwell lane with typed gaps, TMA/async sidecars, and explicit effective L1/L2 assumptions. |

The card is the architecture authority. Capture metadata must agree with the
card; a capture directory name never selects a model. SM86 candidate documents
are rejected for SM120 until an architecture-qualified candidate schema exists.

## Execution and evidence boundaries

```text
NVBit/SASS + sidecars -> sim-trace -> sim-sm <-> sim-mem -> sim-engine -> sim-run
NCU/CUDA observations ------------------------------------------------> post-replay join
```

Native replay consumes trace and sidecar evidence; it does not execute the
original CUDA program. NCU values are independent comparison targets, never
replay or model inputs. Issue, modeled-event, completion, launch-envelope, and
typed-tail frontiers remain separate.

Full replay uses integrated `l1-l2` memory and compressed `.jsonl.zst` output.
Fast replay is a separate trace-derived frontier mode and is not required for
the merge verification. Complete-corpus runs use twelve simulator workers.

SM120 cache and timing values are effective scenarios, not claims of recovered
proprietary topology. SM86 card/P-chase values and candidates remain scoped to
SM86. Unsupported or unmeasured behavior is represented as a typed gap rather
than a workload-specific correction.

## Scientific and repository boundaries

* `/home/jakeshea/5090_branch/` is a read-only source/reference checkout.
* Only the five permitted source subtrees were used for the merge research;
  paper/manuscript paths are not simulator inputs.
* Generated captures and full replay artifacts stay outside Git. Keep only
  compact summaries and reproducible commands in this repository.
* The normal release path is `sim/target/release/sim-run`; architecture cards,
  control sidecars, and capture roots must be supplied explicitly.
