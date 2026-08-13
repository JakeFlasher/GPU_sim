# NCU collection and retained local-label campaign

For a new experiment, use the direct wrapper:

```bash
.venv/bin/python tools/local_labels/collect_ncu.py \
  --output results/my_run \
  --metrics gpc__cycles_elapsed.max,smsp__inst_executed.sum \
  -- /absolute/path/to/workload [arguments]
```

It accepts an ordinary output directory and requires no campaign descriptor,
receipt, admission state, or preflight. `metrics.py` and `ncu_csv.py` provide
small reusable metric and CSV helpers.

## Retained campaign pipeline

The remainder of this page documents the older repeatability/label format,
which remains available through the opt-in `run-sol-campaign` skill. Its
quarantine, admission, provenance, receipt, hash, and recovery vocabulary is
format-specific and does not limit direct NCU execution or unrelated work.

This subtree produces ground-truth cache hit-rate labels on local GPU
devices (RTX 3090 Ti sm_86 and RTX 5060 laptop sm_120) using Nsight
Compute. This compatibility pipeline writes beneath `data/local_labels/`.

## Components

| Path | Role |
|---|---|
| `label_driver.sh` | Bash wrapper that invokes Nsight Compute on a local kernel binary, captures provenance, and emits a per-launch JSON sidecar plus a `.ncu-rep*` artifact. |
| `import_label.py` | Validator for the per-launch JSON sidecar: schema check, cross-field confidence-interval ordering check, forbidden-field-name denylist (raw HW-counter event identifiers and similar tokens). |
| `repeatability_aggregator.py` | Reads N >= 5 sidecars per cell, checks `binary_sha256` consistency, computes per-target standard deviation, and emits per-cell admission verdicts. |
| `test_*.py` | Offline pytest suites that exercise every positive and negative case without invoking the profiler or touching local hardware. |

## Output Layout

The driver writes per-launch sidecars under the quarantine root in this
shape:

```
data/local_labels/
  <device>/                      # e.g. sm_86, sm_120
    <slug>/                      # kernel slug from the workload roster
      <shape_point>/             # shape-file row id
        label_<repeat>_<launch>.json
        run_<repeat>_<launch>.ncu-rep   # gitignored
```

The `.gitignore` at the quarantine root keeps `.ncu-rep*` artifacts
out of version control; only the JSON sidecars and the per-cell
indices are committed.

The repeatability aggregator emits three indices in the quarantine
root (or in a separate `--out` directory):

| File | Content |
|---|---|
| `label_repeatability.json` | Per-cell summary with `n_sidecars`, `n_repeats`, mean and stddev for each target, `binary_sha256_consistent`, and the admission verdict. Also lists rejected sidecars (ones the importer refused). |
| `label_admitted.json` | Cells whose stddev fell within the threshold and whose `binary_sha256` is consistent across repeats. |
| `label_excluded.json` | Cells excluded by the gate, each carrying a typed reason code. |

## Sidecar Provenance Contract

Every sidecar must carry the following provenance fields. The importer
rejects a sidecar that omits any of them.

| Field | Purpose |
|---|---|
| `schema_version` | Sidecar schema major version. |
| `device` | Compute capability tag of the local device (e.g. `sm_86`, `sm_120`). |
| `slug` | Kernel slug from the workload roster. |
| `shape_point` | Shape-file row id used by the launch. |
| `repeat_index` | Zero-based repeat index. |
| `launch_index` | Zero-based launch index inside the repeat. |
| `binary_sha256` | SHA-256 of the kernel binary used. |
| `profiler_family` | Identifier for the profiling family. Default `nsight_compute`. |
| `profiler_version` | Version string of the profiler used. |
| `cuda_version` | CUDA toolkit version used to build the binary. |
| `driver_version` | NVIDIA driver version observed during the run. |
| `gpu_uuid_sha256` | SHA-256 of the device UUID (raw UUID is hashed so it does not leak into committed JSON). |
| `replay_mode` | Replay mode used by the profiler. |
| `aggregation` | How the metric was aggregated — pinned to `kernel`. |
| `warmup_count` | Number of warmup launches before the measurement run. |
| `cache_hit_rate_l2_sector` | L2 sector hit rate as a fraction in `[0, 1]`. Mandatory. |
| `confidence_interval_l2_sector` | Confidence interval for the L2 metric. Required. |
| `cache_hit_rate_l1tex_sector` | L1/TEX sector hit rate as a fraction in `[0, 1]`. Mandatory (AC-3 admits a cell only when every repeat carries both L2 and L1/TEX evidence). |
| `confidence_interval_l1tex_sector` | Confidence interval for the L1/TEX metric. Required. |

The confidence interval has a fixed shape: `lower`, `upper`,
`confidence_level=0.95`, `units=sector_hit_rate_fraction`,
`aggregation=kernel`. The importer enforces `lower <= upper`.

## Forbidden Fields

The importer rejects any sidecar whose JSON tree contains a field name
matching the forbidden denylist. The denylist targets raw HW-counter
event identifiers and privileged CUPTI Profiling surfaces, so a label
sidecar cannot smuggle admin-required signal back into the predictor's
data path:

- Raw HW counter classes: `hw_counter_*`, `raw_hw_counter_*`,
  `pm_counter_*`, `pm_sample_*`, `pm_sampling_*`, `counter_id`,
  `counter_name`.
- CUPTI Profiling surfaces: `cupti_event_*`, `cupti_metric_*`,
  `cupti_pc_sampling_*`, `range_profiler_*`, `sass_metric_*`,
  `pc_sampling_*`, `instruction_execution_*`, `instruction_correlation_*`.
- Generic event / metric / signal identifiers: `event_id`, `event_name`,
  `metric_id`, `metric_name`, `signal_id`, `signal_name`.
- NVIDIA performance-tooling surfaces: `nvperf_*`, `perfworks_*`.

The denylist is enforced by walking the entire JSON tree, so a
forbidden name nested inside an unrelated block also trips the check.

## Running the Driver

The driver is invoked on local hardware. Operator path:

```bash
tools/local_labels/label_driver.sh \
    --device sm_86 \
    --slug gemm_hmma_bf16_sm80 \
    --shape-point row_0 \
    --repeat 0 \
    --launch 0 \
    --binary path/to/kernel.elf \
    --warmup 1
```

Repeat the invocation across N >= 5 distinct repeats (and optionally
multiple launches per repeat) before running the aggregator.

## Running the Aggregator

```bash
python tools/local_labels/repeatability_aggregator.py --root data/local_labels
```

The aggregator walks the quarantine root, validates each per-launch
sidecar via `import_label.py`, groups valid sidecars by cell
`(device, slug, shape_point)`, computes per-target standard deviation
across repeats, and writes the three indices to disk.

## Repeatability Threshold

The default threshold is 3 percentage points (0.03 in fraction units)
for the per-target across-repeat standard deviation. Override with
`--stddev-threshold 0.02` to tighten, or `--repeats-required 7` to
demand more repeats per cell.

## Typed Exclusion Reasons

| Reason | When |
|---|---|
| `label_repeatability_missing` | Cell has fewer than the required number of distinct repeats. |
| `label_repeatability_failed` | Per-target standard deviation exceeds the threshold. |
| `label_provenance_inconsistent` | Imported sidecars in the same cell disagree on `binary_sha256`. |
| `forbidden_field_name` | A sidecar's JSON tree contains a denylisted field name. |
| `confidence_interval_inverted` | A sidecar's confidence interval has `lower > upper`. |
| `schema_validation_failed` | Schema check failed for any other reason. |
| `sidecar_unreadable` | The sidecar file could not be opened or parsed as JSON. |

## sm_86 chunk admission and interrupted recovery

`sm86_capture_chunk.py` treats a preflight as a time-local capture admission.
The single source for its window is
`campaign_preflight.PREFLIGHT_FRESHNESS_SECONDS` (900 seconds, inclusive), with
at most two seconds of future clock skew. Every runner invocation that captures
or recovers a cell records a new run ID and admission time. A continuously
running invocation is not re-admitted per cell.

Each active cell is an append-only journal directory. The runner durably writes
`journal_start.json` before GPU work and writes the separate
`journal_complete.json` only after the complete cell artifact inventory
verifies. Every completed cell, its cell result, and the chunk DONE must name
the same capture-preflight SHA-256. A finalization-only invocation (all member
journals complete) uses that journal preflight even after its freshness window
has elapsed; a newly supplied preflight cannot replace it.

Cross-preflight recovery refuses by default. If the journal's original
preflight is still valid, `JOURNAL_PREFLIGHT_MISMATCH` instructs the operator to
re-present it. If it is expired, missing, or invalid,
`JOURNAL_PREFLIGHT_SUPERSEDABLE` reports both paths and hashes. Recapture then
requires `--supersede-stale-journals`. The runner inventories every active
journal, atomically renames the entire journal set beneath the persistent
external `superseded/` root, re-verifies it, writes per-cell `inventory.json`,
and fsyncs one append-only ledger row per cell before opening a new attempt.
The third supersession of a cell additionally requires
`--allow-supersession-overflow` after operator escalation. Evidence under
`superseded/` is never published or counted.

The Round-28 provenance refusal-code registry is
`sm86_capture_chunk.ROUND28_PROVENANCE_REFUSAL_CODES`. DONE verification and
campaign accounting close both sides of every supersession ledger and reject
orphan destinations, dangling back-references, mixed preflights, admission-time
violations, and any published path beneath `superseded/`.

### SOL definition-equivalence V2

`sol_definition_equivalence.py` is the single source for the certifying rule
`SOL_DEF_EQUIV_V2`. It retains `SOL_DEF_EQUIV_V1` only as a frozen,
non-certifying reference for the preserved blocking audit. V2 keeps V1's three
root-field classes (`hf_id`, `op_type`, and `custom_inputs_entrypoint`, where
missing, null, and the empty string normalize to null) and adds two exact nested
classes. The first is `/axes/<axis-key>/description` when root `axes` and the
immediate axis value are JSON objects. The second is
`/(inputs|outputs)/<tensor-key>/description` when the named top-level container
and its immediate TensorSpec value are JSON objects; this matches the
representation-only null materialization performed by
`Definition.model_dump_json()`. At either nested class only missing and null
are equivalent. Empty strings, non-object containers or entries, top-level,
deeper, and displaced `description` positions remain strict. Comparison is
recursively type-strict, including bool versus int and int versus float.

Every SOL stage preserves exact `definition.source.json` and packager-produced
`definition.json` bytes, seals both hashes, and writes the deterministic
`definition_equivalence_attestation.json`. The runner, DONE verifier, and
accounting independently rehash both files, rerun V2, verify the complete state
maps and eligible axis/TensorSpec-key unions, and require the exact V2 rule ID.
Adapter and runner refusal payloads cap displayed differences at 20 while the
comparator and audit artifacts retain the complete deterministic pointer list.

New attestations always contain both TensorSpec state fields. The sole
seven-field compatibility projection is verifier-only for captures sealed at
`e38788ca0a4457570bd562ea1389b07b3f855e22`: the verifier recomputes and
allowlists the exact comparator and capture-adapter Git blob hashes, rederives
the nine-field attestation from the sealed definitions, and requires the
observed object to equal exactly that object minus the two TensorSpec fields.
The generic validator continues to reject every seven-field object.

The V2-specific typed refusals are:

- `SOL_ATTESTATION_RULE_MISMATCH`: a new SOL attestation or journal does not
  carry the exact V2 rule ID.
- `SOL_ATTESTATION_STATE_MISMATCH`: sealed hashes, deterministic bytes, or
  recorded representation states do not rederive.
- `SOL_ATTESTATION_POINTER_INELIGIBLE`: an attestation records an axis or
  TensorSpec state outside the byte-derived immediate-description eligibility
  sets.
- `sol_stage_source_mismatch`: residual source-versus-staged differences remain
  after the closed V2 transformations.

### Pre-GPU preparation-failure archive

`archive_preparation_failure.py` is the only authorized transition for a SOL
attempt that compiled during preparation but refused semantic validation before
any eval driver, NCU, NVBit, or control capture ran. It is distinct from capture
supersession: `supersedes` remains null and the separate preparation-attempt
limit is two per safe cell ID. A third archive requires the explicit
`--override-preparation-attempt-limit` flag after escalation.

The mutating CLI requires `--archive-preparation-failure`. It verifies the
sealed `capture_started` journal and exact campaign pins, the 22-file Round-28
inventory, empty prepare stderr, prepare stdout equal to the validated manifest,
the refusal/phase/no-process records, and an explicit forbidden-artifact scan.
The immutable source inventory hash uses the historical text domain
`relative_path<TAB>size_bytes<TAB>sha256<LF>`, C-locale sorted, with paths
relative to the single cell-journal root; it is not a JSON digest. Only after
those checks pass does the tool require equal `st_dev`,
atomically rename the complete `journals/` parent beneath the external
`preparation_failures/` root, rehash it, copy and hash `chunk_context/`, and
append+fsync one UUID4 ledger row. Evidence in that namespace is permanently
non-authoritative and must never be published or reused.

The stable refusal codes for this lane are:

- `PREPARATION_ARCHIVE_FLAG_REQUIRED`: the explicit mutation flag is absent.
- `PREPARATION_ARCHIVE_SOURCE_MISSING`: a source seal, campaign pin, or required
  evidence file is absent or inconsistent.
- `PREPARATION_ARCHIVE_NOT_PREGPU`: execution evidence exists or the no-GPU
  proof is incomplete.
- `PREPARATION_ARCHIVE_CROSS_DEVICE`: atomic rename cannot be guaranteed.
- `PREPARATION_ARCHIVE_INVENTORY_MISMATCH`: the pinned or post-move inventory
  differs; after a move, the tool never silently rolls back.
- `PREPARATION_ATTEMPT_LIMIT`: a third attempt lacks the distinct override.
- `PREPARATION_LEDGER_INCONSISTENT`: ledger parsing, UUID/path closure,
  inventory rehashing, or archive↔ledger set equality fails.

Consumers should import the side-effect-free
`load_preparation_failure_ledger(..., verify_archives=True)` and
`preparation_failure_backrefs(...)` helpers. The former closes both directions:
every row resolves to one verified archive and every archive directory has
exactly one row. The ledger-entry JSON Schema is
`workloads/simbench_bw/schemas/preparation_failure_ledger_entry.schema.json`.
