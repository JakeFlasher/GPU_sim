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
* Operating split (user directive): GPT-6 Astra is executioner/orchestrator only—implements, runs, and code-reviews; Gemini 4 Argon High invoked through `agy --dangerously-skip-permissions --model gemini-4-argon-high` is the read-only brain for algorithms, ideas, and insight/adversarial review. Gemini must not code. Any blocker, ambiguity, or uncertainty requires consultation with Gemini before proceeding; do not rely on instinct.
* Active objective: reduce APE below 10% for major IPC and L1/L2-related NCU metrics across complete SM120 (339 captures) and successful SM86 (286 captures) full replays, preserving independent-target separation and twelve-worker replay.
* Latest evidence: full replay remains green at 12 workers (SM120 339/339 `ok` in both DRAM modes; SM86 286 `ok`/62 control/8 replay errors). The SM86 LDG.CONSTANT opcode scope fix reduced that endpoint from 44.8% to 1.36% APE. Unified-L1, generic-L2, DRAM-client, and timing residuals remain above 10% because their NCU endpoints include scope or runtime state absent from the trace contract.
* Do not promote a generic-L2 source-lane sum or clamp: Gemini's read-only review and the launch-66 accounting identity show the 96 excess store hits originate upstream in simulated L1/TEX-to-L2 store traffic, not in the report selector. Missing SM86 shared-memory carveout metadata is an explicit unidentifiable input gap.
* Latest full-enabled progress: trace-semantic SM120 store-packet and TMA-store-cold-miss observers completed 339-capture and 286-successful-SM86 full replays with 12 workers. Generic SM120 L2 hit APE improved from ~57% to ~24.3%; operation-read, unified-L1, DRAM, and IPC residuals remain above 10%.
* Replay scope directive: do not run fast replay. Accuracy and correctness work uses only complete full replay with all supported features enabled by default (integrated `l1-l2`, card-selected architecture behavior, and twelve workers); fast-mode artifacts are not required.
* SM120 replay scope refinement: do not run the SM120 DRAM-disabled full replay. The required SM120 accuracy lane is the full DRAM-enabled/card-feature configuration; continue SM86 full replay as required.
