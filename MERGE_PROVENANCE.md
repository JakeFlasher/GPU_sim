# Merge Provenance & Decision Record

> **CURRENT STATUS — HISTORICAL MERGE LEDGER; NON-OPERATIVE (2026-07-18).**
> This body is preserved as merge provenance. Its retired 41-entry census,
> dual-source timing path, predictor/deployment decisions, and Q3 description of
> cycles as “DVFS-stable” are historical, not current authority. Cycles are
> operating-state- and protocol-bound and are never claimed DVFS-stable. Current
> authority is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
> [`docs/STATUS.md`](docs/STATUS.md),
> [`docs/COMPONENTS.md`](docs/COMPONENTS.md), and the accepted
> [`ADRs`](docs/decisions/README.md), especially ADR 0001 and ADR 0002.

How four trees became one, what was carried vs left behind, and the reasoning
behind every merge decision. This is the "don't guess" ledger: each seam was
re-verified against the *canonical* trees (not the stale dossier) before a choice
was made, and the genuine forks were escalated to the operator.

---

## 1. Source trees

All four are clones of `github.com/JakeFlasher/non_root_gpu_profiler` (solbench
shares the origin but is a fully divergent project).

| Tree | Branch | HEAD | Role |
|---|---|---|---|
| `non_root_gpu_profiler_codex` | `cleanup/pre-merge-simplification` | `9aa3c84a` | **Canonical** sm_86 (247 commits ahead of Tree A) |
| `non_root_gpu_profiler` | `cleanup/pre-merge-simplification` | `d832d13b` | Tree A reference snapshot (dossier baseline) |
| `non_root_gpu_profiler_5060_branch` | `3090ti_cache_sims` | `c4435599` | Tree B — sm_120 lane |
| `solbench_baseline` | `solbench_baseline` | `e1824340` | SOL-ExecBench (dissolved) |

The merge baseline is `non_root_gpu_profiler_codex`; its git history (including the
frozen commit `b1646ef7`) was preserved so the git-anchored cohort guards pass
unedited.

---

## 2. What was carried (payload = source + curated frozen artifacts)

| Carried | Left in the original trees |
|---|---|
| All git-tracked **source** (tools/, workloads/ code, scripts/, tests, configs, schemas) | `references/` corpus — 452 MB papers/manuals/patents |
| Curated **frozen artifacts** the guards pin (heads, cohorts, closures, manifests) | `data/local_labels/sweeps/` — evidence by-products (846 MB on disk; only the frozen cohort-bundle dirs the guards read were carried) |
| The sha-pinned `data/local_labels/*.json` + the 2 frozen `sm120_cohort_bundles_*` dirs | Compiled binaries (`*.so`, `*.o`), per-slug executables, live-trace dumps (all gitignored) |
| Canonical git history (`.git`, 400 MB) so git-anchored guards resolve `b1646ef7` | `.harness_traces/` CUPTI `activity.jsonl` (430 MB of solbench trace evidence) |

Result: **~115 MB source** (+ history). The exclusion of bulk data is intentional and
lossless for the guards — they read the canonical top-level manifests, not the sweeps.

### Integration mechanics
1. **Base:** codex tracked source, minus `references/` and `sweeps/`, preserving layout.
2. **Tree B overlay:** the *set difference* (files tracked in B but not in codex),
   filtered to source — so a codex-canonical file is **never** overwritten by B's
   regressed stub (e.g. B's 1168-byte `sm_86_v5_head.json` with `admitted_count:0` is
   *not* carried; codex's validated head is). This brought in `tools/cache_replay/`,
   `ncu_metric_registry.py`, `typed_ncu_label_path.py`, `duration_admitted.json`, and
   the `sm120_*_closure` artifacts.
3. **solbench dissolve:** each component relocated as a whole unit —
   `baselines/`→`workloads/sol_execbench_baselines/`, `sol-execbench/`→`tools/sol_execbench/`,
   `SOLAR/`→`tools/solar/`, `scripts/baselines_harness/`→`tools/sol_baselines_harness/`,
   `scripts/kb/`→`tools/sol_kb/`, `nvbit_tracer/metric_extractor/`→`tools/nvbit_tracer/metric_extractor/`,
   `cutlass/`+`torchview/`→`third_party/`.

   The `scripts/kb/`→`tools/sol_kb/` line records the historical dissolve
   provenance, not the current hook wiring. The active router and relevance
   gate remain under `scripts/kb/`; the retained `tools/sol_kb/` files are
   SOL-specific authoring utilities. Superseded runtime-fork files were
   removed only after their bytes were verified into the external cleanup
   relocation archive.
4. **New:** `arch_profiles/`, unified `.claude/knowledge/`, `docs/`, the four top-level docs.

---

## 3. Operator decisions (the genuine forks)

Four decisions the trees could not settle were escalated and answered:

| Decision | Choice | Consequence |
|---|---|---|
| **Merge scope** | Source-consolidation only (no live capture) | Unsolved sm_120 items become **research challenges**, not deferred to-dos → `OPEN_RESEARCH_CHALLENGES.md` |
| **Wall-clock** | Labels + honest refutation only | The cycles-first subsystem is carried but makes **no** positive predictor claim |
| **Deploy target** | Local only | ArchProfile has first-class cards for sm_86 + sm_120-laptop only; no GB202/sm_100 cloud SKU cards |
| **solbench** | Fully dissolved | SOL-ExecBench shares the unified `arch_profiles/`, `tools/nvbit_tracer/`, `.claude/knowledge/`, references — no walled module |

---

## 4. The ten seam decisions (Q1–Q10)

Each was re-verified against the canonical codex tree + Tree B. The headline: **the
dossier (written vs Tree A) is materially stale — the canonical tree had already built
most of what it framed as open.**

| # | Seam | Decision (grounded) |
|---|---|---|
| **Q1** | Metric registry | **Canonical = codex `registry_v2`** — a strict superset of Tree B's cache/memory registry (41 vs 22 entries; contains all 15 base NCU tokens; already arch-aware via `per_arch_availability`). Scope-tag the cache/memory subset; re-home B's kind-routed typing as a projection. `gpu__time`/occupancy/IPC stay `label_only`, guarded by the leakage guard. |
| **Q2** | Wall-clock label | **Adopt the codex typed dual-role** already built: NCU `gpu__time` = reference, CUPTI `cupti_dual_source` = deploy observable, bridged by a crosswalk that **passes 13/13** on sm_86. B's `duration_admitted` becomes the deploy-observable leg. |
| **Q3** | Wall-clock predictor | Target = codex's DVFS-stable **cycles** (`gpc__cycles_elapsed.max`); retire B's refuted OLS. **Per operator: presented as labels + honest refutation only** (no positive claim). |
| **Q4** | `label_admitted.json` | **Adopt codex wholesale** — v5 / 74 cells {sm_86:44, sm_120:30}; sm_120 byte-stable @ `b1646ef7`; head refit to 40 cells (sha `d28665f6`). Codex independently recaptured its own 44 sm_86 cells (they are *not* Tree B's). B's 17 unique sm_86 kernels (flash-attn/int8-IMMA/timing-shape) are **not** carried (would need a live recapture + refit — out of scope). |
| **Q5** | sm_90a vs DEC-2 | **Keep sm_90a** — it is a compile-only Hopper fatbin slice, orthogonal to the DEC-2 deploy ban. Formalised as `build_arch_families.json`, kept separate from the deploy `arch_id` (which the DEC-2 guard governs; build Makefiles are out of its scope by design). |
| **Q6** | Blackwell classifier | Lift `_BARE_LDG_E_DEFAULTS_TO_CA` into the per-arch card (`bare_ldg_e_default`): sm_86=`ca` (byte-identical), sm_120=`unknown_typed_unavailable`. `cache_replay` carried as ablation. Gate-clearing = live 5060 work → Challenge 1. |
| **Q7** | comparison_runner | Codex's **3-tuple arch-keyed roster split** (a regression fix over B's 2-tuple, which leaks cross-arch holdout) is canonical; B's up-front predictor-bundle narrowing is a compatible add. Roster path/arch to be resolved via ArchProfile. |
| **Q8** | cache_layers | **Per-arch card:** sm_86 = `["L1","L2"]`, sm_120 = `["L2"]` (sm_120 physically has no L1 head; a uniform `["L1","L2"]` would flood sm_120 with `join_key_unmatched`). |
| **Q9** | Token vocabulary | **Formalised** — canonical bare `sm_XX` `arch_id` + `aliases` + per-SKU HW cards + `build_arch_family`; `resolve_arch_id()` collapses all 8 spellings; a lockstep guard binds the cards to the live constants. Fixes the `sm_120`/`sm_120-laptop` landmine by construction. |
| **Q10** | Provenance slug bug | **Cosmetic** — the wrong-slug `dram_stream_stride_mix__*` files are gitignored and never join the comparison runner (no corrupt join key). Fully wiring the slug through `build.py` is a bounded post-merge engineering step, not carried into the byte-identical baseline. |

---

## 5. Dossier staleness corrections (verified live)

The pre-existing `MERGE_RESEARCH_DOSSIER.md` was written against Tree A @ `083c178a`.
Re-verified against the canonical codex tree, these facts were **stale** and are
corrected here (this is why re-verification, not dossier-trust, drove the merge):

| Dossier said (Tree A) | Reality at canonical HEAD |
|---|---|
| v5 L2 head sha `9597457b…`, **18 cells**, frozen @ `2bb7ac68` | sha **`d28665f6…`**, refit over **40 cells**, re-pinned (`test_frozen_l2_v5_regression_guard.py`) |
| L2 median-abs **0.03732** | **0.05923** (still passes ≤0.06) |
| v5-vs-v4 falsification **confirmed**, attribution 43.78% | **indeterminate**, attribution 14.33% |
| `label_admitted.json` = v5 / 50 cells / 12-timing | v5 / **74 cells** {44,30} / **21-timing** |
| L1 head 6 admitted / CV-geomean 0.00192 | **30 admitted** / CV-geomean **0.071** |
| NCU probe 23/29 | **35/43** |
| Wall-clock "unsolved on both arches" | sm_86 has a *built, gate-scored* cycles predictor (near-miss) + passing deploy-ns crosswalk; only sm_120 has the refuted OLS |
| Q2 "typed enum of both" is hypothetical | already **built and committed** (crosswalk passes 13/13) |
| `tools/cache_replay/` is in "the merged project" | Tree-B-**only**; the codex spine has 0 files there |

NVML reference (1350/2130/37.9 MHz) and the sm_120 hardcoded-50-vs-measured-26 conflict
**were** confirmed accurate; the latter is resolved by the ArchProfile (§Q&challenges).

---

## 6. Byte-identity proof (the validated lane is untouched)

The sm_86 frozen artifacts were carried byte-for-byte; the guards pass unedited:

```
data/local_labels/sm_86_v5_head.json                       sha256 d28665f6…  (test_frozen_l2_v5_regression_guard.py)
data/local_labels/calibration_head_v5_baseline_comparison   sha256 f1ad9fe9…  (same guard)
data/local_labels/calibration_head_v5_accuracy_gates        sha256 ef6e873a…  (same guard)
data/local_labels/sm_86_l1_head.json                        sha256 c2caf894…  (test_l1_head_and_closure.py)
data/local_labels/label_admitted.json                       sha256 2c1634c7…  (v5 / 74 cells)
data/local_labels/sm120_cohort_expansion_roster.toml        sha256 0aaf1645…  (roster validator)
30-cell sm_120 cohort                                       0 identity drift vs git b1646ef7
```

The `arch_profiles/` lockstep test (29 green) AST-extracts the live constants and
asserts each sm_86 card field equals them — the ArchProfile **mirrors, never mutates**.

---

## 7. Guard reconciliation (two NCU-aware projects, one tree)

Dissolving SOL-ExecBench brought a second NCU-quarantine model into the tree. Rather
than weaken either, the guards **each own their subtree**:

* The profiler's `test_no_root_or_ncu_static_guard.py` gained merge exclusions for
  `third_party/` (vendored CUTLASS/torchview) and the dissolved SOL-ExecBench subtrees,
  delegating SOL-ExecBench's NCU enforcement to SOL-ExecBench's own guard.
* SOL-ExecBench's `test_no_ncu_in_deployed_path.py` (comment-aware, deployed-path-scoped)
  was carried and its scope **repointed** to the relocated harness (`tools/sol_baselines_harness/`,
  `tools/nvbit_tracer/metric_extractor/`) so it enforces non-vacuously (11 tests green,
  scope resolves to real files).

Both NCU contracts are green and genuinely enforced. See `docs/ncu-quarantine.md`.

---

## 8. What remains staged (engineering, distinct from research challenges)

Bounded, low-risk follow-ups — *not* the research gaps in `OPEN_RESEARCH_CHALLENGES.md`:

* **ArchProfile wiring depth.** The cards + loader + lockstep guard are complete and
  additive (zero behaviour change). Repointing each live call site
  (`_SM_COUNT_BY_ARCH`, `_GRIDDIM_PENALTY_ALPHA`, `_STATIC_CARD_L2_BYTES`, the roster
  path) to *read* the card is the next mechanical step; the lockstep guard guarantees it
  is safe because the cards already equal the constants.
* **solbench package import rename.** `baselines_harness.*` imports still assume the old
  package root; running solbench's *harness* (not its safety guard) in the new layout
  needs a package-path pass. The safety-critical NCU guard already works.
* **Q10 slug wiring** in `perturbation_harness/build.py` (cosmetic; post-Phase-0).
* **Q1 registry projection** (collapse B's `ncu_metric_registry` into a generated
  projection of the JSON) vs the current peer+drift-test landing.

## 9. Known non-hermetic guards (expected on a source-only checkout)

Two tests in `tools/local_labels/test_sm120_l1l2_frozen_grain_closure.py`
(`test_committed_artifact_validates`, `test_committed_closure_validates_with_recursive_duration_check`)
**fail in this tree and that is expected, not a merge regression.** They validate a
Tree-B **sm_120 ablation-lane** engineering closure whose trace-coverage recompute
counts a cell as covered by checking that its runtime `dynamic_trace.pb` exists on
disk under `workloads/simbench_bw/results/…`. Those `.pb` files are **gitignored,
0-byte runtime placeholders** — present on the live RTX 5060 dev host, absent from
any source checkout (a fresh `git clone` of Tree B lacks them too). Verified: the same
two tests **pass in the original Tree B** (where the runtime `.pb` are on disk) and
recompute `trace: 0` here purely because the source-only payload excludes runtime
trace dumps. Carrying 0-byte placeholders to force a green would be gaming a
non-hermetic check, so it is documented instead. Everything else — the validated
sm_86 lane, the frozen heads, both NCU guards, the ArchProfile lockstep, and 154+
other offline tests — is green. This guard belongs to the refuted sm_120 lane tracked
in `OPEN_RESEARCH_CHALLENGES.md`.

## 10. Test suite removed for a clean refactor (post-verification)

After the merge was verified green (§6 + the consolidated **207 passed / 3 skipped /
187 subtests** offline run), the entire unit/CI test suite was **intentionally deleted**
from this tree by operator request, to be refactored later. Removed: **818 files** — all
`test_*.py` / `*_test.py` (249 loose + those inside 9 `tests/` directories), every
`tests/` directory (fixtures + conftest included), native `test_*.cc` / `test_*.cu`
(6), and `pytest.ini`. No CI configs were present to remove.

**What was kept (source, not tests):** the guard *implementation* modules that other
code imports — `no_deployability_leakage_guard.py`, `artifact_tree_guard.py`,
`composition_dag.py`, the ArchProfile `loader.py` + cards + schema, the frozen heads and
cohort artifacts. Only the `test_*.py` that *exercised* them were deleted; the
contracts' data and logic remain.

**Consequence:** guards that were implemented *purely as tests* (the no-root/NCU static
guard, the frozen-head/cohort regression guards, the ArchProfile lockstep) no longer
execute until the refactored suite re-establishes them. The verification results above
are the recorded baseline the new suite should reproduce; all removed tests are
recoverable from the preserved git history.

## 11. Completeness audit (12-agent, content-hash) + carried misses

An exhaustive path-independent (sha256) audit ran 12 parallel agents across all four
trees + every file-type/config/schema/KB/references/docs/subsystem dimension. Result:
**codex, Tree A, and Tree B = 0 genuine misses** (every hash-miss verified as a
test, a bulk artifact, a superseded merge file, or a relocated file). solbench surfaced
**5 unique genuine misses, all now carried:**

| Miss | Why it mattered | Carried to |
|---|---|---|
| `sm120_ncu_query_metrics_raw.txt.gz` (2.4 MB) | A carried `.toml`'s `sidecar_path=` pointed at it (I'd over-pruned it as `*.txt.gz`) | `tools/sol_baselines_harness/calibration/cache_prediction/` |
| `.githooks/pre-commit` | Guards the vendored NVBit SDK (which is carried) | `.githooks/pre-commit` (guard path rewritten `nvbit_tracer/…` → `tools/nvbit_tracer/…`) |
| solbench `nvbit_tracer/README.md` | Documents the metric_extractor Tier-2 env gate (distinct from codex's tracer README) | `tools/nvbit_tracer/metric_extractor/README.md` |
| `.humanize/bitlesson.md` (BitLesson KB) | A real project lesson knowledge base (gitignored, on-disk only) — codex 1006L canonical + solbench 305L SOL | `.claude/knowledge/bitlesson.md` + `bitlesson-sol-execbench.md` (now **routed** — indexed in the manifest, 355 cards) |
| `.humanize/plans/sol_pattern_mapping.md` | `sol_baselines_scaffold/generate.py:47` hardcodes this path | `.humanize/plans/` (whole dir, 8 planning docs) |

Also confirmed by the audit (not misses, documented so): the suite's `references/NVIDIA_Manuals`
pages are the **newer** (2026-05-12) solbench-crawl versions at shared paths where codex's
were older (2026-03-27) — the content-union kept the newer/expanded prose (no loss); and
the 259 codex reference "hash-misses" are TOC-renumber / image-embed / finer-grained-split
variants of the same manuals at identical paths.
