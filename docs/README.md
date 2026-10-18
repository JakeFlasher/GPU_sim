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

`SESSION_RULES.md` and `LONG_TERM_NOTES.md` at the repository root are the
session contract and current continuation memory. The source checkout,
captures, and baseline result trees are external inputs; this repository never
modifies them.

Older experiment diaries and branch-specific planning files were removed from
the active tree. Their Git history remains available for archaeology, but they
are not execution instructions.
