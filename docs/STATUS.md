# Unified simulator status

## Supported lanes

| Lane | Card | Runtime status |
| --- | --- | --- |
| SM86 / RTX 3090 Ti | `arch_profiles/cards/sm_86.json` | Full compatibility replay with card-backed memory inputs and scoped SM86 candidates. |
| SM120 / RTX 5090 | `arch_profiles/cards/sm_120-rtx-5090.json` | Full effective replay with typed geometry gaps, TMA/async sidecars, and optional carded DRAM. |

Both lanes use the architecture-neutral `tools/sm120_corpus_replay.py` entry
point and the same Rust workspace. Full-corpus results and expected
dispositions are in [`RESULTS.md`](RESULTS.md).

## Boundaries

NCU data is joined after replay and never selects a model. SM86 controls and
substitutes are not relabeled as SM120 features. Effective SM120 values are
conditional scenarios, not proprietary topology recovery. Unsupported behavior
remains a typed gap. Full replay must use `l1-l2`, compressed `.jsonl.zst`
outputs, and twelve workers; the fast frontier is separate.

## Maintenance

Build/test commands and the portable replay recipe are in the root
[`README.md`](../README.md) and [`REPLAY.md`](REPLAY.md). This page is the
current status contract; historical experiment notes are not execution policy.
