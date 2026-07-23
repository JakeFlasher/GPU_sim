# Canonical Serialization Specification

This document fixes the byte-level serialization rules every artifact under
`workloads/simbench_bw/`, `golden/`, and the reviewer package must follow.
Canonical serialization is the load-bearing guarantee that lets reviewer-package
replay detect SCIENTIFIC divergence, not packaging noise. Any reviewer-package
artifact that does not satisfy these rules is rejected at `build.py` time with
`reviewer_artifact_not_canonical`.

This specification is itself a frozen pre-registration artifact. Modifications
must go through the `pre_registration_amendments/` registry per
`pre_registration_amendment.schema.json`.

## JSON Outputs

Every JSON file emitted by a SimBench-BW producer or consumer in this repo MUST
conform to ALL of the following:

1. **Sorted keys.** Object keys are sorted lexicographically at every nesting
   level. In Python, this is `json.dumps(obj, sort_keys=True, ...)`.
2. **Indent of two spaces.** `indent=2` in `json.dumps`. No tabs, no other
   indentation widths.
3. **No trailing whitespace.** No spaces or tabs at the end of any line.
4. **UTF-8 encoding without byte-order mark.** Files start with the JSON
   token, never with the BOM byte sequence `0xEF 0xBB 0xBF`.
5. **Final newline.** Files end with exactly one newline character (`\n`).
   No trailing newline-newline; no missing newline.
6. **No NaN, Infinity, -Infinity.** Floating-point values that would serialize
   to non-standard JSON tokens MUST be rejected at producer time. Producers
   that hit a NaN / Infinity in their inputs raise a typed error; they do NOT
   emit non-standard JSON tokens. Schemas with floating-point fields document
   the admissible range so consumers can rely on standard JSON.
7. **Compact array encoding NOT permitted.** Arrays are encoded with their
   elements on separate lines under the `indent=2` rule, the same as objects.

A reference Python helper that satisfies the above:

```
def write_canonical_json(path, obj):
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False)
    if not text.endswith("\n"):
        text = text + "\n"
    path.write_text(text, encoding="utf-8")
```

## TOML Outputs

Every TOML file emitted in this repo MUST conform to:

1. **Stable key ordering.** Top-level keys are emitted in the order they are
   declared in the producer's data model. Within a table, keys are emitted in
   alphabetical order. Within an array of tables, the array order is
   preserved.
2. **Two-space indent for nested table values.** No tabs.
3. **Double-quoted strings.** Single-quoted (literal) strings are reserved
   for paths that contain backslashes; otherwise double quotes.
4. **No trailing whitespace.** Same rule as JSON.
5. **UTF-8 without BOM.** Same rule as JSON.
6. **Final newline.** Same rule as JSON.
7. **Comments allowed at file head only.** Comments inside the TOML body
   are forbidden at canonical-form time so the file content matches the
   producer's data model exactly. Comments belong in the producer's source
   code, not in committed TOML data.

A reference Python helper using `tomli_w`:

```
def write_canonical_toml(path, obj, header_comment=None):
    body = tomli_w.dumps(obj)
    out = ""
    if header_comment is not None:
        for line in header_comment.splitlines():
            out += f"# {line}\n"
        out += "\n"
    out += body
    if not out.endswith("\n"):
        out = out + "\n"
    path.write_text(out, encoding="utf-8")
```

## Timestamp Normalization

Every artifact field whose name contains `dated_iso8601`, `timestamp_ns`, or
otherwise records a wall-clock timestamp is subject to the following rules:

1. **Originally recorded as a real timestamp.** Producers record the actual
   timestamp at emission time so post-hoc forensic analysis can reconstruct
   the run timeline.
2. **Normalized to a placeholder before checksumming.** The reviewer-package
   checksum step replaces every recognized timestamp field's VALUE with the
   literal token `<TIMESTAMP_NORMALIZED>` before computing the checksum so
   that the same logical artifact emits the same checksum across replays
   that necessarily occur at different wall-clock times. The replacement is
   structural (only the field's value, not the field's name); the field name
   is preserved.
3. **Placeholder mapping is itself frozen.** The field-name-to-placeholder
   mapping lives in `canonical_serialization_spec.md` (this file) and is
   amended via the pre-registration amendment process if a new
   timestamp-bearing field is added.

The recognized timestamp-bearing field names at this revision are:

- `*_dated_iso8601`
- `*_timestamp_ns_*` (any field whose name contains `timestamp_ns`)
- `started_at`
- `finished_at`
- `amendment_dated_iso8601`
- `extracted_dated_iso8601`

## Path Normalization

Every artifact field whose value names a filesystem path is subject to:

1. **Repo-relative paths only.** Absolute paths (e.g. starting with `/`) are
   normalized to repo-relative paths before checksumming. The reviewer
   package's `build.py` performs the normalization automatically; producers
   that emit absolute paths SHOULD be repaired to emit repo-relative paths
   directly so the producer-side artifact and the reviewer-package artifact
   are byte-identical.
2. **Forward slashes.** Path separators are forward slashes regardless of
   the host filesystem.
3. **No trailing slash on file paths.** Directory paths MAY have a trailing
   slash but file paths MAY NOT.

## Build-Environment Field Normalization

Every artifact field whose value would otherwise vary across hosts that
build / run the artifact is normalized to a placeholder before checksumming:

- `hostname` -> `<HOSTNAME_NORMALIZED>`
- `build_id` -> `<BUILD_ID_NORMALIZED>`
- `username` -> `<USERNAME_NORMALIZED>`

The reviewer package's checksum step performs this normalization
structurally. Producers SHOULD avoid recording these fields in the first
place when the field is not load-bearing; when the field IS load-bearing
(e.g. preflight manifest's host capabilities matrix), the unchanged value
is recorded in the artifact and the normalization is applied only inside
the checksum step.

## Bootstrap Seed Sourcing

Every cluster-bootstrap consumer MUST source its RNG seed from the frozen
`pre_registration_seeds.toml`. A bootstrap output computed without sourcing
the frozen seed fails the replay-checksum check AND is flagged with
`bootstrap_seed_not_from_frozen_registry`. The frozen seeds file is itself
a frozen pre-registration artifact; modifications go through the
`pre_registration_amendments/` registry.

## Reviewer Package Replay Discipline

The reviewer-package's `replay.sh` re-emits every published ledger and
fingerprint at the frozen git SHA, applies the canonical-serialization
rules above (sorted-keys JSON, normalized timestamps, normalized paths,
normalized build-environment fields), and computes a SHA-256 over each
artifact. The expected checksums are committed alongside the package; a
mismatch is reported as `reviewer_replay_divergence` naming the divergent
artifact + the canonical-form diff so a reviewer can distinguish a
scientific divergence (different numerical value at a fixed input) from
packaging noise (different timestamp / hostname / path representation).

## Companion Archive Policy

If the reviewer package's total bundled size exceeds 5 GB, raw sidecars
are emitted as a checksum-addressed companion archive with the replay CLI
auto-fetching from a documented URL or local path. The 5 GB threshold is
itself a frozen policy parameter recorded here; modifications go through
the amendment registry.

## nvml_state_divergence

The `dual_run_manifest.schema.json` field `drift_tolerance_ns` carries a
nanosecond budget for the divergence between a CUPTI-side run's NVML
launch state and the paired NVBit-side run's NVML launch state. The
canonical definition of `nvml_state_divergence` referenced from that
schema's description is:

> `nvml_state_divergence(cupti_state, nvbit_state)` is the absolute
> value of the difference between the two snapshots'
> `sample_timestamp_ns_host_monotonic` fields, expressed in
> nanoseconds:
>
> ```
> divergence_ns = abs(
>     cupti_state.sample_timestamp_ns_host_monotonic
>     - nvbit_state.sample_timestamp_ns_host_monotonic
> )
> ```
>
> The other state fields (`sm_clock_mhz`, `mem_clock_mhz`, `temp_c`,
> `power_mw`, `throttle_reasons_bitmask`) are validated by the
> schema's per-field admission (every field is required + non-negative
> per the schema's per-field constraints) but are NOT collapsed into
> the wall-clock divergence metric. Per-field divergences are recorded
> in the manifest for reviewer-side analysis but do not contribute to
> the per-launch drift budget.

The rationale for pinning the divergence to host-monotonic timestamp
delta only is alignment with the non-root measurement contract's non-root measurement
methodology: the dual-run discipline exists because the CUPTI Activity
collector and the NVBit injector cannot share a process, so the two
runs are necessarily back-to-back in wall-clock time. The drift the
join layer must reject is wall-clock drift between the two runs at
the per-launch boundary; per-field clock / temperature / power
divergences are second-order signals already captured by the
NVML covariate-regression layer. Folding them into the per-launch
drift budget would double-count the covariate contribution.

The consumer module `dual_run_manifest_consumer.py` and the emitter
`dual_run_manifest_emitter.py` both implement this rule verbatim;
test coverage in
`tests/test_dual_run_manifest_consumer.py::test_admit_passes_when_nvml_clock_differs_but_timestamps_match`
pins the rule directly.
