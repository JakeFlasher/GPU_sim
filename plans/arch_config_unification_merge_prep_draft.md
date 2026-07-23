# Arch-Config Unification — Pre-Merge Refactor Map (DRAFT)

Status: DRAFT scoping artifact produced during the pre-merge simplification on
branch `cleanup/pre-merge-simplification`. Groundwork for merging the
3090ti/sm_86 and 5060/sm_120 branches and refactoring core components (cache
replay, predictor algorithm, benchmark suite, timing, cache geometry,
repeatability) to read an arch-config file instead of hardcoded per-arch configs.
NOT an RLCR plan; not yet wired to any AC ladder.

## Provenance
Built from a read-only 7-agent deep-dive (workflow `wf_08953c97`, ~1.86M tokens,
560 tool uses) that mapped **160 hardcoded arch-coupled values** across 6
subsystems, then cross-checked against the source and an independent Codex review.
Every claim below was spot-verified against the working tree.

## Headline (verified)
There is NO unified arch-config today and NO `arch_profile_*` file exists yet
(an early agent draft asserted one — that was wrong; corrected here). Instead
there are **four independent, separately-evolved per-arch surfaces**, all
confirmed present:
- `tools/hrd_profiler/device_properties.json` — L2 geometry (5-tuple keyed, one sm_86 row)
- `tools/cache_predictor/l1_static_card_table.json` — L1 geometry (bare-arch keyed, one sm_86 row)
- `tools/local_labels/metric_registry.json` — repeatability thresholds (single global block 0.03/0.05/0.05)
- `workloads/simbench_bw/comparison_contract.toml` — cohort floors + join key + arch_set
  (+ the sm_86 submission overlay `comparison_contract_sm86_submission.toml`)

Around these sits a large belt of hardcoded sm_86 constants in Python/Make that
should READ these surfaces but instead re-declare them. The refactor introduces
ONE canonical per-arch profile + loader and makes every consumer
resolve-config-then-fall-back-to-current-literal, so sm_86 stays byte-identical.

## Refactor surface (quantified, verified)
- 160 hardcoded arch-coupled values: geometry 45, predictor 29, repeatability 25,
  timing/runner 21, classifier 21, benchmark 19.
- ~30 are sm_86-NCU-derived correction constants in `cache_predictor.py` (highest
  risk; must be re-measured per arch — the 5060 branch owns the sm_120 values).
- `arch_capabilities.json`: 17 "RESTRICTED TO sm_86" + 17 "queued" notes — a
  pre-existing cross-arch-parameterization TODO ledger.
- Token-naming landmine: **189 .py files reference `sm_120`, 29 use `sm_120-laptop`**.
  The label schema enum is `["sm_86","sm_120"]`. → canonical token = `sm_120`.
- The `locked_thresholds.json` reader is a forward-reference only (no file, no
  loader); live correction constants are hardcoded in `cache_predictor.py`.

## Codex-locked design decisions
1. **L2 high-footprint 4-criterion thresholds** (`cache_predictor.py:~3415`): the
   predicate is tied to an sm_86 sweep (`l2_high_footprint_correction_calibration.json`)
   → keep the rule SHAPE global, store the NUMERIC thresholds PER-ARCH (seed sm_86).
2. **Cohort floors** (smoke=5, publication=10; submission 30 slugs/31 cells in the
   overlay): POLICY, not hardware → leave in the contract/overlay TOML files, NOT
   in the arch profile.
3. **Canonical arch token = `sm_120`**. Keep `sm_120-laptop` only as an input alias
   normalized at the CLI/capability boundary (already translated in
   `label_driver.sh:221`); SKU/device fields carry the laptop-vs-server distinction.
4. **ArchProfile source of truth**: ArchProfile READS per-arch JSON cards directly
   (matching the existing `l1_static_card.py` "JSON table + schema + loader"
   pattern). Do NOT add a generator-owned parent config / second source of truth.
5. **Format = JSON + `.schema.json`** (the repo has broad JSON-Schema validation;
   TOML stays for human-facing contracts/rosters). Do NOT introduce TOML profiles.
6. **`locked_thresholds.json`**: supersede with ArchProfile — migrate the
   correction factors into the profile; if back-compat is ever needed, make a
   locked-threshold reader a thin deprecated shim over ArchProfile, not a 2nd source.

## Global vs per-arch split
- GLOBAL (arch-invariant; single source): unit enum {pct,count,ns,cycles};
  repeatability rule vocabulary + unit→rule map; dispersion estimator formulas;
  valid ranges; wire `schema_version=5`; `ns_per_mhz_cycle=1000.0`; warp lane = 32;
  sector/line primitives (32/128) as derivation base; bundle analysis params
  (bootstrap=2000, ci=0.95, ci_half_width=0.05, min_bin=50); cohort floors (POLICY,
  stay in contract TOML, not the profile per Codex Q2).
- PER-ARCH: L1/L2 byte capacity + carveouts + l1_ways + num_l2_slices + sm_count;
  gridDim alpha (86=0.110 / 120=0.0 uncalibrated); all L1/L2 correction-factor
  dicts + the L2 high-footprint structural thresholds (Codex Q1); classifier
  bare-LDG default + token sets + Blackwell memory-touching op classes; the NVML
  reference-operating-point file + clock band; SKU/device-name/gencode/CC.
- PER-ARCH OVERRIDABLE w/ global default (start == global to keep frozen verdicts):
  the CV repeatability threshold (clock/DVFS variance differs Ampere↔Blackwell);
  l1/l2 repeats floors.

## Migration sequencing (frozen-head-preserving)
- Phase 0 (additive, zero behavior change): add the per-arch JSON card(s) +
  `.schema.json` seeded BYTE-IDENTICALLY from current sm_86 literals; add an
  ArchProfile loader (natural home: extend `tools/arch_capabilities.py`) +
  `resolve_arch_id()` alias resolver; round-trip self-test (card sm_86 == live
  constants). No consumer wired. Guards stay green. Commit.
- Phase 1 (parity assertions): per-consumer test asserting card.sm_86 == the live
  hardcoded value (predictor dicts, device_properties row, l1_static_card row,
  metric_registry block, classifier token sets, NVML band). Makes the config a
  provably-equal shadow before any switchover.
- Phase 2 (low-risk invariant collapse): switch single-shared-value geometry/units
  consumers to read the global block (128/32, warp=32, ns_per_mhz, vocab).
- Phase 3 (per-arch facts, fail-soft): switch cache_predictor / classifier / L1+L2
  geometry / NVML / comparison_runner to read the profile; sm_86 resolves to today's
  exact values; unknown-device fail-CLOSED (no silent sm_86 fallback). Assert frozen
  v5 head sha256 == 9597457b... + per-bucket geomeans unchanged.
- Phase 4 (repeatability triplication collapse): route the 0.03 through config;
  UPDATE the static guard that pins the literal to assert `card[sm_86].stddev_abs==0.03`.
- Phase 5 (token reconciliation + enum widening in lockstep): wire the alias map;
  add the sm_120 cards only after the 5060 host verifies sm_count / num_l2_slices /
  l1_ways / the NVML reference point live.
- Phase 6 (build + paths): externalize gencode + COLLECTOR_ARCH + results path
  builder; migrate the 17 "RESTRICTED TO sm_86" orchestrators off hardcoded
  rtx_3090_ti/sm_86 literals; resolve the sm_90a build-vs-capability contradiction.

## Backward-compat / frozen-head strategy
Seed the sm_86 card with EXACT current literals (verified: device_properties
l2=6291456/slices=6, l1 l1tex=131072/ways=4/default=32768, metric_registry
0.03/0.05/0.05, coverage gate 5/10/20, gridDim alpha 0.110, sm_count 84). Each
consumer migrates as "read card; if absent/null, use the existing literal" so a
half-migrated tree still works and any step is revertible. Preserve the drift-lock
(`comparison_runner` _PINNED_* vs contract — both fed from the card). Update guards
in lockstep, never bypass. Wire `schema_version` stays 5 (additive widening only).
data/local_labels values are never rewritten — the card SELECTS the per-arch NVML
reference file; the file keeps owning the operating-point values + provenance.

## Merge mapping (two branches → one)
- sm_86 branch (this one) OWNS the refactor: card schema + loader + alias map +
  wiring all 6 subsystems while keeping sm_86 byte-identical.
- 5060/sm_120 branch OWNS measured sm_120 facts: real GB20x sm_count (replacing the
  GUESSED 50), real L2 capacity/num_l2_slices, the full sm_120 L1 static card
  (replacing the fail-closed null), the sm_120 NVML reference point + clock band,
  the sm_120 classifier capability extension, and sm_120 correction factors.
- Merge rule: code seams come from the sm_86 refactor branch; per-arch VALUES come
  from each arch's card. **Land the card + loader on the merge base FIRST as a no-op
  overlay**; then each branch contributes only its own card block + consumer-side
  reads — turning the classic "both branches edited the same `_SM_COUNT_BY_ARCH` /
  `_DEVICE_TO_ARCH` dict" conflict into additive config blocks.
- gridDim is already the correct model: sm_120 → alpha 0.0 (fail-soft, tagged
  uncalibrated) rather than reusing the sm_86 0.110. Generalize that to every
  correction so a sm_120 cell never silently inherits an sm_86 factor.

## Still-open (route to Codex/operator before implementing Phase 3+)
- Per-consumer fail mode for the uncalibrated sm_120 facts (sm_count guessed 50,
  num_l2_slices null, l1 null, loocv null): typed-unavailable/fail-closed vs
  documented no-op. Specifically: should a sm_120 gridDim prediction with the
  guessed sm_count=50 be allowed at all, or fail-closed until the 5060 host verifies?
- sm_90a gencode: remove from simbench Makefiles to match the arch_capabilities
  DEC-2 server-arch ban, or relax the manifest? (build vs capability contradict.)
