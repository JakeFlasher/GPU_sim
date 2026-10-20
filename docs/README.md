# Documentation map

The active documentation surface is intentionally small:

* [`CONTRACT.md`](CONTRACT.md) — architecture lanes, scientific boundaries,
  and supported runtime behavior.
* [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`COMPONENTS.md`](COMPONENTS.md) —
  execution flow and crate/tool ownership.
* [`REPLAY.md`](REPLAY.md) — architecture-neutral full-replay commands and
  the twelve-worker policy.
* [`RESULTS.md`](RESULTS.md) — retained full-corpus counts and comparison
  interpretation.
* [`HISTORY.md`](HISTORY.md) — concise merge provenance and resolved conflicts.
* [`TERMINOLOGY.md`](TERMINOLOGY.md) — public vocabulary and metric
  definitions.
* [`ACADEMIC_RELEASE_PLAN.md`](ACADEMIC_RELEASE_PLAN.md) — release rewrite,
  regression, and documentation-shipping plan.
* [`ACADEMIC_STYLE_REVIEW.md`](ACADEMIC_STYLE_REVIEW.md) — terminology
  decisions grounded in HOTL and modern GPU-simulation literature.

`SESSION_RULES.md` and `LONG_TERM_NOTES.md` at the repository root are the
session contract and current continuation memory. The source checkout,
captures, and baseline result trees are external inputs; this repository never
modifies them.

Older experiment diaries and branch-specific planning files were removed from
the active tree. Their Git history remains available for archaeology, but they
are not execution instructions.

The release uses conventional computer-architecture terminology. Historical
experiment names remain in dated records only; current contracts and reports
define the mechanism, metric population, aggregation, and evidence boundary.
