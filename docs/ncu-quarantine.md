# The NCU Quarantine & the Two-Guard Model

> **RETIRED POLICY — HISTORICAL ONLY.** This page preserves the former
> non-root/NCU-free deployment contract and its two-guard reconciliation. The
> current project permits NCU as a silicon-label target and permits the pinned
> host measurement protocol; it still forbids NCU values from simulator inputs.
> Do not use this page to block the active simulator or to override root
> `CLAUDE.md`, `docs/README.md`, or accepted ADRs.

This page records the predecessor suite's static guard model and historical
operator-authorized carve-outs. It remains useful when interpreting old artifacts,
but it is never an active operational gate and must not be reactivated by a local
owner or imported skill. Current capture, label, privilege, and simulator policy
comes only from root `CLAUDE.md`, active schemas, tools, and accepted ADRs.

Related reading:
[`../CLAUDE.md`](../CLAUDE.md) (current charter, not this page's historical rules),
[`./sol_execbench/CLAUDE.md`](./sol_execbench/CLAUDE.md) (dissolved
SOL-ExecBench rules),
[`../MERGE_PROVENANCE.md`](../MERGE_PROVENANCE.md) §7 (guard
reconciliation),
[`../ARCHITECTURE.md`](../ARCHITECTURE.md),
[`../OPEN_RESEARCH_CHALLENGES.md`](../OPEN_RESEARCH_CHALLENGES.md).

---

## 1. The hard contract

The following reproduces the predecessor root snapshot's five-rule contract; it is
not a statement about the current root `CLAUDE.md` (which has a different active
charter). Four historical rules defined a forbidden measurement surface and the
fifth addressed simulator revival. The old static guard enforced that snapshot:

| Rule | Forbidden on cloud/server | Enforcement |
|---|---|---|
| **1** | `ncu` / `nv-nsight-cu-cli` invocation; `.ncu-rep` / `.ncu-repz` artifacts | repo-wide token + artifact-path scan |
| **2** | `nvidia-smi` clock/persistence/power/MIG/compute-mode mutation | `-lgc`/`-lmc`/`-pm`/`-pl`/`--lock-*-clocks` patterns |
| **3** | privileged CUPTI (Events, Metrics, Range Profiling, PM Sampling, PerfWorks) | header + entrypoint + gated-Activity-kind patterns |
| **4** | host-driver work (`modprobe nvidia`, `dkms`, `mokutil`, `NVreg_RestrictProfilingToAdminUsers`) | kernel-module operation patterns |

Rule 3 has a critical nuance: **CUPTI Activity is allowed** — it is the
non-root contract's wall-clock telemetry surface. Only the *profiling*
CUPTI APIs (which internally require the admin profiling permission) are
banned. The guard therefore carries a paired **blocklist** and
**allowlist** so an over-broad regex can never accidentally ban the
legitimate telemetry path (§4).

The prohibition is textual, not just behavioral: the guard flags a
forbidden token even inside a code comment or docstring, because the
user contract is *"never allow use of ncu and gpu hardware counters or
gpu root privilege driver settings"*
(`test_no_root_or_ncu_static_guard.py:169-170`).

---

## 2. The forbidden-surface scan

The repo-wide guard lives at
`workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`. It
walks the whole tree, drops excluded subtrees, filters to source
extensions, and matches every remaining line against the compiled
forbidden-pattern table (`_FORBIDDEN_PATTERNS`,
`test_no_root_or_ncu_static_guard.py:171-239`).

```
                     repo root (rglob "*")
                            │
         ┌──────────────────┴──────────────────┐
         │  EXCLUDED_PREFIXES   (lines 98-130)  │  drop whole subtrees:
         │                                      │   references/{NVIDIA_Manuals,papers}/
         │   ── merge additions ──              │   tools/nvbit_tracer/nvbit_release/
         │   third_party/                       │   tools/local_labels/   ┐ local-label
         │   tools/sol_baselines_harness/       │   data/local_labels/    ┘ QUARANTINE
         │   tools/sol_baselines_scaffold/      │
         │   tools/sol_kb/  tools/solar/        │  ── delegated to solbench's
         │   tools/sol_execbench/               │     OWN comment-aware guard
         │   tools/nvbit_tracer/metric_extractor/
         │   workloads/sol_execbench_baselines/ │
         └──────────────────┬──────────────────┘
                            │  EXCLUDED_FILES (149-162): the guard files
                            │  themselves + runbook-verifier (self-reference)
                            ▼
              suffix ∈ {.sh .py .cu .cuh .h .cpp .hpp .c}   (line 95)
                            │
                            ▼
         ┌───────────────────────────────────────────────┐
         │  for each line: _COMPILED_PATTERNS.search()    │
         │                                                │
         │   ncu / nv-nsight-cu-cli / .ncu-rep(z)  ──► ✗  │  Rule 1
         │   nvidia-smi -lgc/-lmc/-pm/-pl/--lock-* ──► ✗  │  Rule 2
         │   cupti_profiler* / cuptiPmSampling*    ──► ✗  │  Rule 3
         │   CUPTI_ACTIVITY_KIND_PC_SAMPLING/EVENT/ ──► ✗ │  Rule 3
         │       METRIC/INSTRUCTION_*/SASS_METRIC          │
         │   modprobe nvidia / dkms / mokutil      ──► ✗  │  Rule 4
         │                                                │
         │   cupti.h / -lcupti / cuptiActivity*    ──► ✓  │  ALLOWLIST
         │   CUPTI_ACTIVITY_KIND_{KERNEL,MEMCPY,…}  ──► ✓  │  (telemetry)
         └───────────────────────────────────────────────┘
                            │
                    any ✗ → assertion fails with
                    <file>:<line>: <label>: <text>   (line 288)
```

A separate **artifact-path walker**
(`test_ncu_artifact_paths_outside_quarantine_rejected`, lines 743-772)
covers what the source-content scan cannot see: binary `.ncu-rep*`
reports are not in `SCANNED_EXTENSIONS`, so a second pass walks *every*
committed file by path only and rejects any `.ncu-rep[^/]*$` outside the
`data/local_labels/` quarantine (pattern at line 701; classifier
fixtures at lines 779-830 pin `data_local_labels/…` and
`data/local_labels_x/…` as *outside* the precise prefix boundary).

---

## 3. The profiler carve-out: the local-label quarantine

NCU is the ground-truth oracle. The predictor is *trained against* NCU
labels; it just may never *invoke* NCU on a cloud/server target. Rule 1
(`../CLAUDE.md:16-28`) therefore permits NCU **on local devices only**
(RTX 3090 Ti sm_86, RTX 5060 laptop sm_120) and **only** to write into
the quarantine path `data/local_labels/`, driven exclusively by code
under `tools/local_labels/`.

The static guard implements this as an **exclude-then-re-scan** pattern:

1. Both prefixes appear in `EXCLUDED_PREFIXES` (lines 104-105) so the
   repo-wide no-NCU scan skips them —
   `test_quarantine_prefixes_are_excluded_from_repo_wide_scan`
   (lines 600-614) fails if a refactor drops them.
2. `LOCAL_LABEL_QUARANTINE_PREFIXES` (lines 142-145) re-scopes those
   same subtrees under a **derived, scoped pattern set**
   (`_QUARANTINE_FORBIDDEN_PATTERNS`, lines 421-425) that is the
   repo-wide table **minus exactly the NCU labels** —
   `test_local_label_quarantine_remains_bounded` (lines 467-497).

So inside the quarantine, **Rule 1 (NCU) is relaxed, but Rules 2–4 stay
absolutely binding**:

```
   repo-wide pattern table            local-label scoped table
   (all Rules 1-4)                    (_QUARANTINE_FORBIDDEN_PATTERNS)
   ┌───────────────────────┐          ┌───────────────────────┐
   │ ncu / nv-nsight-cu-cli │  drop   │  (relaxed — permitted) │
   │ .ncu-rep / .ncu-repz   │ ──────► │  (relaxed — permitted) │
   ├───────────────────────┤  labels ├───────────────────────┤
   │ nvidia-smi -lgc/-pl/…  │  kept   │ nvidia-smi -lgc/-pl/…  │ ✗
   │ modprobe nvidia / dkms │ ──────► │ modprobe nvidia / dkms │ ✗
   │ cupti_profiler* / PC_… │  kept   │ cupti_profiler* / PC_… │ ✗
   └───────────────────────┘          └───────────────────────┘
```

The relaxation is proven **real and bounded** by four cross-checks so it
can never silently become a no-op or an over-relaxation:

- `test_local_label_permit_list_actually_relaxes_ncu_patterns`
  (500-515): each permitted NCU token must actually trip a repo-wide
  pattern (else filtering it relaxes nothing).
- `test_local_label_quarantine_filter_admits_ncu_tokens` (518-536): the
  NCU tokens must *not* trip the scoped set.
- `test_local_label_quarantine_still_blocks_non_ncu_violations`
  (581-597): every Rules 2–4 sample (`nvidia-smi -lgc`, `modprobe
  nvidia`, `dkms`, `cupti_events.h`, `CUPTI_ACTIVITY_KIND_PC_SAMPLING`,
  …) must still fail inside the quarantine.
- `test_quarantine_pattern_set_drops_only_permitted_labels` (617-654):
  the full-table set-difference assertion — no non-permitted label was
  dropped, no permitted label survived, no orphan permit exists.

Because the quarantine directories may not yet contain source, the scan
logic itself is exercised by synthetic fixtures fed to
`_scan_text_for_quarantine_violations`
(`test_quarantine_scanner_text_fixtures`, lines 951-981) — e.g. *"pure
NCU invocation is permitted"* vs *"nvidia-smi clock-locking is rejected
even alongside NCU"*.

---

## 4. The CUPTI Activity allowlist

CUPTI Activity is the deployed wall-clock surface, so the guard must
prove its blocklist is not over-broad. Two positive controls back this:

- `test_cupti_profiling_blocklist_patterns_actually_match` (328-342):
  every admin-required token (`cupti_events.h`, `cuptiPmSamplingStart`,
  `CUPTI_ACTIVITY_KIND_METRIC`, …) must be caught — a typo'd regex that
  silently stopped matching would fail here, not pass vacuously.
- `test_cupti_activity_allowlist_not_caught_by_blocklist` (377-394):
  the allowlist tokens
  (`cupti.h`, `libcupti.so`, `-lcupti`, `cuptiActivityRegisterCallbacks`,
  `cuptiActivityFlushAll`, `CUPTI_ACTIVITY_KIND_KERNEL/CONCURRENT_KERNEL/
  MEMCPY/…`, `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`) must **never** trip a
  forbidden pattern.

The distinction is surgical: `cupti_profiler` is banned, `cupti.h` is
admissible; `CUPTI_ACTIVITY_KIND_PC_SAMPLING` is banned (it internally
needs profiling permission), `CUPTI_ACTIVITY_KIND_KERNEL` is the
timestamp surface the collector relies on. When a required kind is
unavailable the collector surfaces the NVIDIA-documented
`CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` as the typed
`CuptiKernelTimestampUnavailable` (lines 65-67) rather than escalating
privilege.

> **Honesty posture.** The sm_86 cache-hit predictor is validated
> against these NCU labels; the sm_120 (Blackwell) predictors are
> **refuted** and framed as research challenges
> ([`../OPEN_RESEARCH_CHALLENGES.md`](../OPEN_RESEARCH_CHALLENGES.md));
> wall-clock is carried as **labels + honest refutation only**, making
> no positive predictor claim ([`../MERGE_PROVENANCE.md`](../MERGE_PROVENANCE.md)
> §3, Q3). The quarantine is what lets those labels exist without ever
> contaminating the deployed path.

The path-boundary regression
`test_path_boundary_tracer_and_cupti_collector_pinned` (1030-1113)
additionally SHA-256-pins every tracked file under
`tools/nvbit_tracer/` plus the two CUPTI Activity collector sources, so
the measurement stack cannot drift byte-for-byte without an explicit
plan amendment.

---

## 5. The SOL-ExecBench carve-out: calibration only

The dissolved SOL-ExecBench project ([`./sol_execbench/CLAUDE.md`](./sol_execbench/CLAUDE.md))
carries the *same* spirit with a *different* shape. Its Critical Rule 1
(lines 96-119) bans NCU on the **deployed framework path** (the five
counter-free analytic surfaces driven by `baselines_harness/` —
`surfaces/`, `report.py`, `run.py`) and on the **NVBit metric extractor
that ships to the B200 provider** (`nvbit_tracer/metric_extractor/`),
while permitting NCU as an **offline ground-truth source** for the
Tier-2 cache-prediction calibration under
`calibration/cache_prediction/` on sm_86/sm_120 only.

That carve-out harvests **read-only** sector hit-rate metrics
(`l1tex__t_sector_op_read_hit_rate.pct`,
`lts__t_sector_op_read_hit_rate.pct`, and the underlying `…_read_hit.sum`
/ `…_read.sum` counters) — deliberately *not* the combined
`…_sector_hit_rate.pct` family, which conflates store traffic under
L2's write-back / no-write-allocate policy and would bias labels
(`./sol_execbench/CLAUDE.md:110-116`). The calibrated model + per-arch
ledger it produces needs no NCU at deployment; every B200 invocation
stays NCU-free.

SOL-ExecBench's guard,
`tools/sol_baselines_harness/tests/test_no_ncu_in_deployed_path.py`, is
**comment-aware and deployed-path-scoped** — a structurally different
scanner from the profiler's textual one:

- **Python** sources in `_DEPLOYED_PYTHON_SCOPE` (`surfaces/`,
  `report.py`, `run.py`, `b200_preflight.py`, lines 60-65) are parsed as
  an **AST** (`_PythonNcuVisitor`, lines 157-232): it flags `import ncu`,
  `from ncu import …`, and any `subprocess.*` / `os.system` whose argv
  resolves to an `ncu` / `nv-nsight-cu-cli` binary. Because it walks the
  AST, **comments and docstrings are excluded by construction** — the
  opposite policy to the profiler guard, and deliberate: deployed
  SOL-ExecBench code *may legitimately explain in a comment* that NCU is
  forbidden.
- **Native** sources under `nvbit_tracer/metric_extractor/`
  (lines 71-73) get a **comment-stripping regex** scan
  (`_strip_native_comments` → `_scan_native_source`, lines 246-281): it
  removes `//` and `/* … */` first, then matches `\bncu\b` /
  `nv-nsight-cu-cli` on the surviving code. `test_planted_native_*`
  (391-434) pin that a commented ban is *not* a violation, an in-code
  string *is*, and `incurred` never false-positives.

Critically, the calibration carve-out, `tests/`, and `fixtures/` are
**out of scope** for this guard (lines 98-136), so calibration NCU use
is permitted exactly where Rule 1 authorizes it.

---

## 6. Two guards, each owning their subtree

Merging four trees folded a second NCU-quarantine model into one repo.
Rather than weaken either, the reconciliation
([`../MERGE_PROVENANCE.md`](../MERGE_PROVENANCE.md) §7) makes each guard
**own its own subtree and delegate the other's**:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  cross_arch_gpu_profiler_suite/  (one tree, two NCU contracts)     │
  │                                                                    │
  │   ┌───────────────────────────┐        ┌───────────────────────┐  │
  │   │  PROFILER GUARD            │        │  SOLBENCH GUARD        │  │
  │   │  workloads/simbench_bw/    │        │  tools/sol_baselines_  │  │
  │   │  tests/test_no_root_or_    │        │  harness/tests/test_no_│  │
  │   │  ncu_static_guard.py       │        │  ncu_in_deployed_path.py│ │
  │   │                            │        │                        │  │
  │   │  scans: whole tree,        │        │  scans: surfaces/,     │  │
  │   │    textual, comment-       │        │    report.py, run.py,  │  │
  │   │    INCLUSIVE               │        │    metric_extractor/;  │  │
  │   │                            │        │    AST + comment-AWARE │  │
  │   │  owns Rules 1-4 repo-wide  │        │  owns SOL-ExecBench    │  │
  │   └────────────┬──────────────┘        │  deployed-path NCU ban │  │
  │                │                        └───────────┬───────────┘  │
  │   EXCLUDED_PREFIXES (109-130):                      │              │
  │     third_party/                     delegates ─────┘              │
  │     tools/sol_baselines_harness/  ───────────►  enforced THERE,    │
  │     tools/sol_baselines_scaffold/               non-vacuously      │
  │     tools/sol_kb/  tools/solar/                 (scope repointed   │
  │     tools/sol_execbench/                         to relocated      │
  │     tools/nvbit_tracer/metric_extractor/         harness paths)    │
  │     workloads/sol_execbench_baselines/                             │
  │                                                                    │
  │   ── same delegation pattern as the local-label quarantine ──      │
  │      (exclude here → re-enforce with a scope-appropriate guard)    │
  └──────────────────────────────────────────────────────────────────┘
              BOTH GREEN — neither weakened, both non-vacuous
```

The profiler guard gained the merge exclusions above so it never
double-scans vendored CUTLASS/torchview (`third_party/`) or the
SOL-ExecBench subtrees, whose examples/comments legitimately mention
`nv-nsight-cu-cli` (`test_no_root_or_ncu_static_guard.py:109-130`). It
**delegates** SOL-ExecBench enforcement to SOL-ExecBench's own guard,
whose scope was **repointed** at merge time from the old
`scripts/baselines_harness/` to the relocated
`tools/sol_baselines_harness/` root (`_HARNESS_ROOT`,
`test_no_ncu_in_deployed_path.py:47-73`) so it still resolves to real
files and enforces non-vacuously.

This is the **exact same delegation pattern** the profiler already used
for its local-label quarantine internally: *exclude the subtree from the
broad scan, then re-enforce it with a scope-appropriate guard*. The
merge just applied it at tree scale — the profiler guard excludes the
SOL-ExecBench subtrees the way it excludes `tools/local_labels/`, and a
second guard owns each excluded region.

Result ([`../MERGE_PROVENANCE.md`](../MERGE_PROVENANCE.md) §7): both NCU
contracts are green and genuinely enforced — the profiler guard covers
the profiler spine repo-wide, the solbench guard covers the dissolved
SOL-ExecBench deployed path, and no line of active code on any
cloud/server path can invoke NCU, mutate clocks, install a driver, or
open a privileged CUPTI surface.
