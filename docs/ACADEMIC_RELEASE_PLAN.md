# Academic Release and Terminology Plan

**Status:** completed; release terminology and regression checks verified  
**Scope:** unified SM86/SM120 simulator, documentation, comments, configuration
descriptions, history, and persistent project memory  
**Constraint:** no accepted performance regression

## 1. Objective

Prepare the simulator for external academic release by replacing internal
engineering language with terminology used in GPU architecture and memory
systems papers. The release must describe what is modeled, what is measured,
and what remains unidentifiable without implying proprietary-hardware
recovery or post-hoc NCU fitting.

The release is complete only when:

1. the simulator builds and tests successfully;
2. all documentation uses consistent names and metric definitions;
3. configuration and code comments describe mechanisms rather than internal
   implementation trivia;
4. historical records distinguish superseded experiments from the current
   contract;
5. the existing full-replay artifacts remain numerically unchanged; and
6. a final 12-worker full replay of both lanes shows no regression in any
   scored metric, timing frontier, cache/DRAM counter, or launch disposition.

## 2. Source-study conclusions

### 2.1 Higher-Order Theory of Locality

The terminology should follow Ding and Xiang:

- **footprint**: average number of distinct data items in windows of a given
  length;
- **lifetime**: inverse footprint at a cache capacity;
- **miss ratio**: inverse inter-miss interval, or equivalently the miss-ratio
  curve derived under the stated correctness condition;
- **reuse distance**: number of distinct items between two references to the
  same item;
- **reuse-window hypothesis/correctness condition**: footprint conversion is
  valid only when reuse-window and all-window footprint statistics agree.

Our `sector_mrc_observer_v1` is an exact **trace-local Mattson/Fenwick
reference** for its declared within-warp stream. `reuse_interval_observer_v1`
is an **interval-based footprint-to-MRC estimator**. Neither is a direct NCU
measurement, and neither should be described as recovering a hardware MRC.

### 2.2 Modern GPU-core modeling

Huerta et al. use conventional terms such as:

- compiler-guided greedy-then-youngest warp scheduling (**CGGTY**);
- sub-core or SMSP pipeline;
- static instruction-control fields;
- scoreboard/dependence management;
- register-file cache;
- instruction prefetching;
- validation against hardware counters;
- mean absolute percentage error (MAPE), correlation, and simulation speed.

We should use these terms directly. Internal names such as `observer_store`,
`frontier`, `tap point`, and `unpriced request` belong in code or machine
schemas only when necessary, not in the scientific narrative.

### 2.3 Conventional evaluation language

The release should use:

- **MAPE/Mean APE** for the arithmetic mean of per-row absolute percentage
  error;
- **median APE** and **P90 APE** for skewed distributions;
- **WAPE** only when explicitly defined as aggregate absolute error divided by
  aggregate target magnitude;
- **exact identity join**, **ordered fallback join**, and **unmatched row** for
  NCU association quality;
- **effective model** for a bounded approximation that is not a claim about
  proprietary topology;
- **report-only attribution** for a counter definition that does not mutate
  timing state;
- **conditional interval/envelope** for an uncertainty range;
- **scope limitation** when a metric's hardware client population cannot be
  identified from retained evidence.

## 3. Controlled terminology

| Internal or inflated wording | Release wording |
| --- | --- |
| closed-loop calibration | end-to-end hardware-counter validation |
| open-loop prior | isolated microbenchmark parameter |
| counter tap-point projection | hardware-counter attribution |
| projection layer | counter-semantics model |
| causal horizon | conservative cross-SM lookahead horizon |
| unpriced L1 miss | in-flight L1 miss awaiting arbitration |
| fast frontier mode | core-only/fixed-latency diagnostic mode |
| frontier (when a completion time is meant) | completion stage/timestamp |
| shadow-only observer | report-only observer |
| Class-A | trace-local locality analysis |
| Class-B | simulator-conditioned effective model |
| scope gap | client-scope limitation |
| prior | parameter or measurement source |
| residual frontier | remaining error sources |
| hardware truth/recovery | independent NCU target or effective model |
| store packet | coalesced store request at the named interface |
| hit prediction | attributed lookup hit count |
| zero-target FP | nonzero prediction on a zero-target row |

Terms such as **observer**, **effective**, and **conditional** remain in
machine-readable field names where changing them would break consumers. Their
human-facing definitions must appear once in the contract.

## 4. Work phases

### Phase A — Inventory and terminology control

- inventory Markdown, comments, CLI help, architecture-card descriptions, and
  report schemas;
- classify each occurrence as scientific terminology, implementation detail,
  historical record, or schema compatibility;
- add `docs/TERMINOLOGY.md` as the single glossary;
- do not rename serialized fields or CLI flags without a compatibility plan.

### Phase B — Contract and configuration rewrite

- rewrite `README.md`, `CONTRACT.md`, `STATUS.md`, `PROJECT_MEMORY.md`,
  `ARCHITECTURE.md`, and `REPLAY.md`;
- describe SM86 and SM120 as deployment lanes with separate architecture
  cards;
- state that NCU profiles are post-replay targets, never model inputs;
- replace configuration prose such as “priors” with measured parameters,
  effective parameters, or card values;
- preserve JSON keys and add aliases only when a machine reader requires them.

### Phase C — Code-comment and API language

- rewrite public Rust/Python comments to describe the hardware mechanism and
  invariant first, implementation second;
- retain internal identifiers in code where renaming would create needless
  churn, but remove them from user-facing reports and prose;
- replace “frontier” with a specific issue, modeled-event, response, or
  completion timestamp;
- define every observer at its measurement interface and state whether it can
  affect timing.

### Phase D — Results and historical records

- consolidate current results into `RESULTS.md` and `RESIDUAL_FRONTIER.md`;
- mark experiment diaries and rejected hypotheses as historical;
- remove duplicate milestone narratives from active status documents;
- retain commit IDs and artifact paths only where they explain reproducibility;
- report both exact-join and fallback-join results, never combine them silently.

### Phase E — Paper-aligned release documentation

- write a concise architecture overview using “event-driven cycle-level
  simulation,” “conservative lookahead,” “sectored cache,” and “hardware
  counter validation”;
- state HOTL's correctness condition and our within-warp scope;
- distinguish simulator timing outputs from counter-attribution outputs;
- use tables for metric definitions, aggregation, scope, and limitations;
- remove Rust identifiers, CLI recipes, filesystem paths, and internal
  milestone language from scientific summaries.

### Phase F — Regression verification and shipping

- run formatting/static checks and `cargo build --release --workspace`;
- run `cargo test --workspace`;
- run bounded real-capture probes;
- run full integrated `l1-l2` replays with `--workers 12` for SM120 and SM86;
- run the reuse/MRC, NCU comparison, and terminology/report consistency checks;
- compare final artifacts with the current closure artifacts and reject any
  numeric difference outside explicitly documented observer fields;
- commit the release documentation and implementation together.

## 5. Golden regression rule

No documentation or terminology change may alter simulator behavior. Any code
refactor must be accompanied by before/after comparisons for:

- active and elapsed IPC;
- instruction and memory traffic;
- L1 and L2 lookup hits/misses;
- DRAM sectors and timing frontiers;
- launch status and launch counts;
- HOTL/MRC observer outputs.

If a refactor changes a metric, stop, identify the mechanism, consult Gemini,
and either restore the previous behavior or document a scientifically
supported improvement with full-corpus evidence.

## 6. Current known blockers

The present closure artifacts meet the identifiable cache-locality gates.
Remaining IPC residuals and SM86 fallback-join error are limited by missing
launch chronology, runtime resource metadata, TMA synchronization metadata,
and cross-surface NCU launch identity. These are documented limitations, not
targets for count balancing or NCU fitting.

## 7. Release verification record (2026-10-20)

The first release pass completed the terminology glossary, literature/style
review, active-document map, history consolidation, persistent-memory update,
and comments/API wording pass. The Rust formatter, release build, and complete
workspace tests pass.

Fresh full integrated replays were run with twelve workers after the code
comment/internal-name refactor:

- `full_replay_runs/academic_release2_sm120/full.jsonl.zst`: 339/339 `ok`,
  11,558 launches;
- `full_replay_runs/academic_release2_sm86/full.jsonl.zst`: 286 `ok`, 62
  `control_error`, 8 `replay_error`, 12,392 launches.

`compare_numeric_full.py` reports zero missing/extra rows, zero metadata
changes, and zero common numeric-field differences against the M7 closure
artifacts for both lanes. The reuse invariant checker also passes both lanes.
This is the required no-regression evidence for the terminology and comment
refactor. Generated replay artifacts remain outside Git; compact reports and
commands are retained in the repository.

The release remains intentionally conservative: serialized field names and CLI
compatibility names such as `fast_frontier_only` are not renamed merely for
style. Human-facing documentation calls that mode a core-only diagnostic, and
`TERMINOLOGY.md` records the compatibility exception.
