# Long-term session notes

The immutable session rules are in [`SESSION_RULES.md`](SESSION_RULES.md).
This file is deliberately a current-state note rather than a duplicate.

* Workspace: `unified_gpu_simulator`, Git branch `main`.
* Source reference: `/home/jakeshea/5090_branch/` remains strictly read-only;
  all builds, tests, and edits happen in this workspace.
* Supported lanes: SM86 / RTX 3090 Ti and SM120 / RTX 5090, selected by card
  and capture metadata.
* Full replay: SM120 339 captures in each DRAM mode; SM86 356 attempted with
  286 successful rows. Complete runs use twelve workers and compressed JSONL.
* Latest audit: `FULL_NUMERIC_COMPARISON.json` reports zero missing/extra rows,
  required-metadata differences, and common numeric differences in all three
  campaigns.
* Current docs: `docs/CONTRACT.md`, `docs/REPLAY.md`, `docs/RESULTS.md`, and
  `docs/HISTORY.md`. Old plans and experiment diaries are not active policy.
* Known limitation: SM120 effective geometry and completion boundaries are
  typed conditional models, not proprietary RTL recovery; NCU remains an
  independent post-replay target.
