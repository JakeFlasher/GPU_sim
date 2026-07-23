# Cross-Arch `.harness_traces/` Trace-Layout Contract (DEC-B RESOLVED)

> Companion to the cross-arch plan at
> `.humanize/plans/crossarch-predictor-validity-restore-sm86-primary-20260528-052258.md`.
> This file documents the resolution of DEC-B (RESOLVED 2026-05-28).
> The contract here is the canonical reference for `report.py`,
> `test_evidence_label_consistency.py`, `test_trace_layout_contract_v1.py`,
> and any future per-arch consumer of `.harness_traces/` bundle paths.

## Decision

`scripts/baselines_harness/.harness_traces/` bundles use
**arch-qualified sibling directories** when a per-arch evidence
artifact is needed alongside an existing un-suffixed bundle. Bundle
directory names follow exactly one of these three forms:

| Form | Example | When used |
|---|---|---|
| **Legacy un-suffixed** (`<slug>__<lang>/`) | `problem_002_fused_add_rmsnorm_h4096__cuda_cpp/` | The 86 sm_120 R0-R23 bundles. the design contract byte-immutability locks every tracked file under these paths. New sm_86 work MUST NOT write into them. |
| **Arch-qualified sm_86** (`<slug>__<lang>__sm86/`) | `problem_002_fused_add_rmsnorm_h4096__cuda_cpp__sm86/` | All new sm_86 evidence bundles (M2 pilot harvest, M4 expansion). |
| **Arch-qualified sm_120** (`<slug>__<lang>__sm120/`) | `problem_002_fused_add_rmsnorm_h4096__cuda_cpp__sm120/` | Reserved for future sm_120 re-runs on a future 5060 host (not used in this plan; the existing un-suffixed sm_120 bundles are the locked authority). |
| **the design contract coverage variant** (`<slug>__<lang>__sm86_cov/`) | `problem_040_conv2d_residual_block__cuda_cpp__sm86_cov/` | The fresh, provenance-valid the design contract coverage bundle for a cell whose canonical `__sm86/` path is occupied by a frozen R18-R30 pilot bundle. `BUNDLE_NAME_RE` parses it with the base `arch=sm86` (the `_cov` marker is informational), so the overwrite guard + host routing are unaffected; only the coverage ledger points at this path. Non-seed the design contract cells (no frozen pilot bundle) use the plain `__sm86/` path. |

A `<slug>__<lang>__sm86/` and a `<slug>__<lang>/` MAY coexist for the
same slug+lang — the un-suffixed one is the locked sm_120 evidence;
the suffixed one is the new sm_86 evidence. Co-location is the
explicit goal of DEC-B: per-arch evidence stays separable on disk
without inner-schema changes.

## What DEC-B did NOT choose

The alternative (rejected) was an **arch-indexed inner manifest
schema** that would have stored both arches' evidence under a single
canonical `<slug>__<lang>/` bundle with arch keys inside the
`manifest.json`. That option was rejected because it would have
required schema changes in `report.py`, `surface_orchestrator.py`,
`test_evidence_label_consistency.py`, and the manifest emitters
across multiple surface plugins — too many simultaneous edits for a
contract change with no algorithmic benefit over the simpler
sibling-dir form.

## Consumer contract — shared helper API

The canonical implementation lives in
`scripts/baselines_harness/trace_layout.py`. All consumers
(producer side: `surfaces/cli.py`, `report.py`; test side:
`test_evidence_label_consistency.py`, `test_trace_layout_contract_v1.py`)
import from that module rather than duplicating the regex.

### Module-level constants

- `BUNDLE_NAME_RE`: canonical regex (groups `slug`, `lang`, `arch`).
  The `arch` group is optional. A match with `arch=None` is the
  legacy un-suffixed form (sm_120 R0-R23 bundles). A match with
  `arch="sm86"` is the new sm_86 sibling form; `arch="sm120"` is
  reserved.
- `TRACE_LAYOUT_TYPED_REASONS`: closed-vocabulary frozenset of
  typed reasons emitted by the guards
  (`trace_layout_overwrite_sm120_evidence`,
  `evidence_label_arch_dimension_missing`,
  `trace_layout_bundle_name_unparseable`,
  `trace_layout_arch_suffix_manifest_disagreement`).

### `TraceLayoutViolation`

Exception class with a `typed_reason` attribute drawn from
`TRACE_LAYOUT_TYPED_REASONS`. Constructor refuses unknown reasons
to enforce the closed vocabulary.

### `parse_bundle_name(name) -> Optional[Tuple[str, str, Optional[str]]]`

Returns `(slug, language, arch_suffix_or_None)` or `None` if the
name does not match `BUNDLE_NAME_RE`. Never raises.

### `arch_suffix_from_host_kind(host_kind) -> Optional[str]`

Maps a `host.kind` token (e.g., `"local_sm_8_6"`, `"local_sm_120"`,
`"b200_sm_100a"`) to the corresponding bundle-name arch suffix.
Returns `"sm86"` for `local_sm_8_6`; `None` for `local_sm_120` (the
legacy un-suffixed path IS the locked sm_120 canonical), B200, and
unknown.

### `default_bundle_dir(traces_root, slug, language, host_kind=None) -> Path`

Computes the default bundle directory for a fresh emission, routing
through `arch_suffix_from_host_kind`. The active 3090 Ti host
(`host_kind="local_sm_8_6"`) writes to
`<slug>__<lang>__sm86/`; sm_120 / B200 / `None` use the legacy
un-suffixed path. Called by both `surfaces/cli.py:_default_bundle_dir`
and `report.py:_bundle_dir`.

### `assert_no_sm120_overwrite(bundle_dir, host_kind) -> None`

Fail-closed guard for the `--bundle-dir <explicit>` path. Fires
only when (1) `host_kind == "local_sm_8_6"`, (2) `bundle_dir.name`
parses as the legacy un-suffixed form, and (3) the target bundle's
`manifest.json` declares `host.kind=local_sm_120`. Raises
`TraceLayoutViolation("trace_layout_overwrite_sm120_evidence", …)`.

### `evidence_key_for_bundle(bundle_dir, manifest_dict=None) -> Tuple[str, str, str]`

Returns the `(slug, language, arch)` evidence key. Used by
`test_evidence_label_consistency.py` to enumerate bundles by full
arch dimension (no silent drop of arch-suffixed bundles). Raises
with the appropriate typed reason when the arch dimension cannot be
resolved or when name suffix and manifest host disagree.

### `effective_arch_for_bundle(bundle_name, manifest_dict=None) -> Optional[str]`

Convenience wrapper returning just the arch tag or `None`. Never
raises; callers needing the typed-reason failure modes use
`evidence_key_for_bundle`.

### `report.py` integration

`report.py:_bundle_dir(baseline_dir, host_kind=None)` delegates to
`trace_layout.default_bundle_dir`. Existing call sites that walk
already-landed bundles pass `host_kind=None` and find the legacy
un-suffixed paths (the 86 sm_120 R0-R23 bundles). New emissions
pass the live host's `host.kind`.

### `surfaces/cli.py` integration

`surfaces/cli.py:_default_bundle_dir(baseline_dir, host_kind=None)`
delegates to the same helper. The `run` subcommand at the call site
passes `host_info.kind` from `probe_host()`. Explicit
`--bundle-dir <path>` is guarded by `assert_no_sm120_overwrite`
before the orchestrator writes.

### `test_evidence_label_consistency.py` integration

`_load_landed_manifest_labels` now uses `evidence_key_for_bundle`
to produce `(slug, lang, arch)`-keyed labels. Arch-suffixed bundles
are NOT silently dropped — they carry distinct keys
(`<slug>__<lang>__sm86`) and are subject to the same exclusion
ledger discipline as legacy un-suffixed bundles. The smoke sm_86
bundle is explicitly excluded with the M3+ per-arch-README deferral
rationale.

### Migration of the session-bridge smoke bundle

The session-bridge sm_86 smoke bundle at
`.harness_traces/_sm86_smoke_problem_002_fused_add_rmsnorm_h4096__cuda_cpp/`
(committed at `826967ae`) was named with a leading `_sm86_smoke_`
prefix to avoid the legacy `problem_*__*` glob. It is renamed to
`.harness_traces/problem_002_fused_add_rmsnorm_h4096__cuda_cpp__sm86/`
under DEC-B; the new name parses as
`slug=problem_002_fused_add_rmsnorm_h4096`, `lang=cuda_cpp`,
`arch=sm86`. The migration is a `git mv` that preserves all 22 tracked
files inside the bundle.

## Negative-test typed reasons

The `test_trace_layout_contract_v1.py` planted negatives use:

- `trace_layout_overwrite_sm120_evidence` — fired when code attempts
  to write a sm_86 manifest into a `<slug>__<lang>/` path that
  currently holds a sm_120 manifest. (The the design contract byte-immutability
  fixture also catches this via SHA mismatch; the typed reason here
  exists for the layout-side guard.)
- `evidence_label_arch_dimension_missing` — fired when a bundle's
  name carries no `__sm<arch>` suffix AND its `manifest.json` has no
  `host.kind` field, leaving the arch dimension undetermined.

## When this contract changes

If a future round needs the arch-indexed inner-schema variant (e.g.,
to support more than two arches without N-fold dir explosion),
update this file in-place with a versioned schema bump and ensure
both `report.py` + `test_evidence_label_consistency.py` accept the
new form before the bundle emitters change. Do NOT mix layouts mid-
loop without an explicit DEC reopening.
