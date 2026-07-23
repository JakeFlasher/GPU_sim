# Open Research Challenges

> **CURRENT STATUS — HISTORICAL DUPLICATE; NON-OPERATIVE (2026-07-18).**
> Current authority is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
> [`docs/STATUS.md`](docs/STATUS.md),
> [`docs/COMPONENTS.md`](docs/COMPONENTS.md), and the accepted
> [`ADRs`](docs/decisions/README.md), especially ADR 0001 and ADR 0002. All text
> below that presents retired predictor/no-root/no-simulator-revival policy,
> claims cycles are DVFS-stable, or describes a then-current project state is
> historical and non-operative. It must not override the active simulator
> mission, cycles-only publication contract, or current campaign gates. The
> historical body is intentionally preserved.

These are the **hard problems prior development rounds could not close** — the
frontier of the non-root, counter-free measurement program. They are recorded here
as first-class research challenges, *not* as deferred engineering to-dos. Each is
paired with the live evidence that establishes it, why it resists an easy fix, and
what closing it would take. The `arch_profiles/cards/sm_120-laptop.json` card links
its typed-`research_gap` fields directly to the anchors below.

Guiding honesty principle (from `CLAUDE.md`, the Two-Device model): a result is
only "verified" with matching live evidence on the specific arch. **None of these
challenges can be closed from source alone or from an sm_86 host** — they require
live Blackwell (RTX 5060 sm_120) capture that is out of scope for a source merge.
That is exactly why they remain open.

---

## Challenge 1 — sm_120 (Blackwell) L2 cache-hit prediction is unsolved
<a id="sm_120-blackwell-l2-cache-hit-prediction-unsolved"></a>

**Status:** every sm_120 L2 verdict refutes. This is the flagship open problem.

**Evidence (in-tree, `data/local_labels/`):**
| Path (L2 on the 30-cell sm_120 cohort) | median-abs error | gate |
|---|---|---|
| L1-miss-filtered replay (`sm120_cache_metric_closure.json`) | **0.276** | 0.055 → REFUTED |
| standalone calibrated `cache_replay` (`cache_replay_comparison.json`) | **0.358** | 0.055 → REFUTED |
| learned v5 head on the *same* cohort | **0.076** | 0.055 → still REFUTED |

**Root cause — one frozenset.** `tools/cache_predictor/l1_cache_op_classifier.py:309`:
```python
_BARE_LDG_E_DEFAULTS_TO_CA: frozenset[str] = frozenset({"sm_86"})
```
On Blackwell, bare `LDG.E`, `LDG.E.U16.CONSTANT`, and descriptor-based (`desc[UR..]`)
loads are *not* in this set, so they route to `OTHER`/bypass. The result:
`total_l1_eligible_misses = 0`, `l1_filter_inert = true` — the L1→L2 miss filter that
feeds the parametric L2 model has nothing to filter, so every cache verdict collapses.

**Why it is hard (why the fix is not obvious):**
1. The naive fix (add `"sm_120"` to the frozenset) is **forbidden without evidence**
   — the sm_120 tree's own plan (AC-12) says a blind add fails; unknown Blackwell
   load forms silently classified as eligible fail. The correct fix needs a live
   `cuobjdump` inventory + cache-op microbench on the 5060 to *establish* whether each
   Blackwell load form is `ca`, `bypass`, or genuinely `unknown`.
2. Even if the classifier is fixed, it is **unknown whether the 0.055 gate clears.**
   The *learned* head (which does not use the L1 filter at all) already sits at 0.076
   on this cohort — evidence that other error sources (coalescing, prefetch, memory
   compression, ECC) may dominate, not just the classifier.
3. The refuted set is 22 cache/memory claims, of which only the *hit-rate* legs are
   gated by the filter; the *sector-count* legs are dominated by trace fidelity the
   classifier does not touch.

**What closing it needs:** a live RTX 5060 session → cuobjdump/microbench of Blackwell
load semantics → set `arch_profiles/cards/sm_120-laptop.json:bare_ldg_e_default` to the
measured value → re-run the closure → the 0.055 gate decides load-bearing status. The
merge already provides the typed hook (`bare_ldg_e_default` is a per-arch card field);
what is missing is live silicon, and success is not guaranteed.

**How the merge represents it honestly:** `bare_ldg_e_default = "unknown_typed_unavailable"`
on the sm_120 card; `cache_replay` carried as a non-load-bearing ablation; the sm_120
L2 predictor state reads `REFUTED`.

---

## Challenge 2 — sm_120 L1/L2 hardware geometry cards are absent
<a id="sm_120-l1l2-geometry-cards-absent"></a>

**Status:** the single biggest per-arch *data* gap.

**Evidence:** `tools/cache_predictor/l1_static_card_table.json` has **no** sm_120
row, and `tools/hrd_profiler/device_properties.json` has only the sm_86 entry. The L1
walker therefore fail-closes on sm_120 (typed `predictor_l1_sass_provenance_missing`).
L2 capacity (32 MiB) and slice count (12) are runtime-derived, but the full
replacement/associativity geometry and the entire L1/TEX geometry (ways, sets, default
carve-out, max shared) were never characterised on GB206.

**Why it is hard:** Blackwell consumer L1/L2 geometry is not vendor-documented at the
granularity the walker needs; it must be reverse-engineered with pointer-chase
(`tools/pchase/`) and associativity probes on the live 5060. The 3090-Ti host cannot
produce it.

**What closing it needs:** a live 5060 pointer-chase + associativity sweep → populate
`arch_profiles/cards/sm_120-laptop.json:{l1_geometry,l2_geometry.policy}` (currently
`research_gap` / `null`). This is a prerequisite for any honest sm_120 L1 prediction
(Challenge 3) and feeds Challenge 1.

---

## Challenge 3 — sm_120 L1/TEX prediction is uncalibrated
<a id="sm_120-l1tex-uncalibrated"></a>

**Status:** 0 of 26 sm_120 cells score an L1 prediction.

**Evidence:** there is no sm_120 L1 head anywhere in the tree (all L1 head/prediction
artifacts are sm_86-scoped: `data/local_labels/sm_86_l1_head.json`). The sm_120 card
carries `cache_layers = ["L2"]` precisely because L1 cannot be honestly emitted there.

**Why it is hard:** depends on Challenge 2 (no L1 geometry → the walker has no
denominator) *and* Challenge 1 (the classifier is inert). It is downstream of both.

**What closing it needs:** Challenges 2 and 1 first, then fit an sm_120 L1 head and flip
the card's `cache_layers` to `["L1","L2"]`.

> Even on **sm_86**, L1 is a *secondary* claim: it is scored and coverage-gated but has
> **no accuracy gate**, and its relative-error geomean is 0.89 (the median absolute
> residual is a healthy ~4%, but streaming kernels with near-zero true L1 hit rate
> inflate the relative metric). "L1 load-bearing" therefore means *scored*, not
> *accuracy-validated to the L2 standard*. Whether L1 should carry a hard accuracy gate
> before being presented as a contribution is itself an open question.

---

## Challenge 4 — wall-clock / duration prediction is unsolved on both arches
<a id="wall-clock-prediction-unsolved"></a>

**Status:** merged as **labels + honest refutation only** — the suite makes *no*
positive wall-clock predictor claim.

**Evidence:**
* **sm_120:** the OLS-on-log10(duration) model (`build_duration_predictor_report.py`)
  scores held-out SMAPE **1.766**, which does **not** beat the `shape_size_only`
  baseline **0.891**; the NVML clock covariates are the *entire* harmful delta. The
  DVFS span is **15.34×** (180–2762 MHz) — the confound the model cannot see through.
* **sm_86:** a cycles-first predictor (`timing_cycle_predictor.py`, target
  `gpc__cycles_elapsed.max`, DVFS-stable) is an **honest near-miss** — median abs-log
  error 0.041 (passes ≤0.1) but geomean-relative 0.158 (fails ≤0.15) and bootstrap
  upper bound 0.310 (fails ≤0.25). The failure mode is cohort poverty (n=13,
  GEMM-heavy) widening the bootstrap CI, not the target choice.

**One genuine positive, kept as a LABEL result (not a predictor claim):** the deploy-ns
crosswalk (`timing_closure.py`) proves the deploy-legal CUPTI `duration_ns` observable
agrees with the NCU `gpu__time` reference within a 110 µs / 20% noise floor on
**13/13** sm_86 cells. That validates the *measurement*, not a *prediction*.

**Why it is hard:** wall-clock at a counter-less, DVFS-varying deploy target is the
original motivating problem. Cycles factor out DVFS but need (a) a per-arch reference
clock (exists for sm_86 as a 1890 MHz short-burst anchor; **absent for sm_120**) and
(b) a larger, more diverse cohort to shrink the bootstrap CI.

**What closing it needs:** a live 5060 NVML short-burst anchor + sm_120 cycle labels;
and on sm_86, cohort expansion (new kernels) to pass the geomean/bootstrap gates. Both
are live-capture / kernel-authoring efforts beyond a source merge.

---

## Challenge 5 — cross-architecture transfer is unproven
<a id="cross-arch-transfer-unproven"></a>

**Status:** the validated sm_86 lane does **not** transfer to sm_120, and the trees
hardwire that separation.

**Evidence:** the sm_120 closure registry pins `cross_arch_precondition = false` on
every record, and its validator *rejects* `true`. The comparison runner carries
bidirectional per-arch transfer *diagnostics* but no passing cross-arch transfer. The
project's own evidence vocabulary reserves `Cross-arch-verified` for results with live
witness on *both* arches — which no predictor has.

**Why it is hard:** everything above (Challenges 1–4) is a per-arch wall. A cross-arch
claim requires the sm_120 lane to first *work*, then be shown to share a mechanism with
sm_86 — two live campaigns, not one.

**What closing it needs:** close Challenges 1–4 on sm_120 with live evidence, then
re-adjudicate `cross_arch_precondition` to a tri-state and demonstrate transfer.

---

## Corrected in this merge (were latent defects, now resolved)

Not open problems — recording them so the fixes are traceable:

* **sm_120 SM count 50 → 26.** `cache_predictor._SM_COUNT_BY_ARCH["sm_120"]` still
  carries the *guessed* 50; `arch_profiles/cards/sm_120-laptop.json` carries the
  **measured 26** (from the 5060 live manifests) and the lockstep test asserts the
  correction. The source constant is superseded by the card.
* **Provenance slug bug (cosmetic).** `tools/perturbation_harness/build.py` hardcodes
  `KERNEL_SLUG="dram_stream_stride_mix"`, mislabelling artifacts for other kernels — but
  the affected files are gitignored and never join the comparison runner (no corrupt
  join key). Fully wiring the slug through `build.py` is a bounded engineering step
  (see `MERGE_PROVENANCE.md` Q10), distinct from the research challenges above.

---

## The one-line summary

**sm_86 cache prediction works and is frozen; everything Blackwell is an open research
problem gated on live 5060 silicon; wall-clock is unsolved everywhere and shipped as an
honest refutation.** The merge's job was to consolidate the evidence and make the gaps
*legible and typed* (via the ArchProfile `research_gap` markers) — not to paper over them.
