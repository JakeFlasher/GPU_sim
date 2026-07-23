# FABLE-5 BINDING ADJUDICATION — Round 28 (Corrected Reissue)

**Session:** `b41e0614-a371-4041-b2ed-0d0aecfda548` · **Round:** 28
**Supersedes:** the unsaved adjudication titled "Missing Historical Context Records Before Preparation-Failure Archive" (amendment ID `ANNEX_C_AMEND_R28_PREP070_RECON_V1`)
**Amendment ID of this adjudication:** `ANNEX_C_AMEND_R28_PREP070_RECON_V2`

---

## 1. DECISION

1. The certifying premise `deterministic_v1_comparator_over_sealed_bytes` is **withdrawn as factually false**. Frozen `SOL_DEF_EQUIV_V1` evaluates the sealed problem-070 source/staged pair as **equal** and can never truthfully ground the historical refusal.
2. The historical refusal authority is rebound to the **legacy comparator at original HEAD `9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a`** (`normalized_definition`, `tools/local_labels/sm86_capture_chunk.py`, lines 1306–1317), under the new frozen algorithm identity `SOL_DEF_LEGACY_COMPARE_9B8E0E40`.
3. The exact replacement certifying literal, everywhere the withdrawn literal appeared, is:
   `deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes`
4. Certifying authority is **conjunctive dual-basis**: `annex_c_adjudication` (classification) **and** the legacy-comparator literal (mechanical derivation). Neither alone suffices. Legacy re-derivation is **certifying**, conditional on the complete identity contract in Section 4 and the now-**required** reproduction record in Section 6.
5. `SOL_DEF_EQUIV_V1` remains frozen with its existing semantics. Artifact 1 (`b6dd22ad…cdcd0`, mode 0444) and its problem-070 `equal` row remain byte-identical and immutable. `SOL_DEF_EQUIV_V2`, the H1 audit (`ea51b0c6…52b52`… as recorded: `ea51b0c6da89f5d339cab365ed9bfc69cb642a44dd73dc2fabe9d5dafafcb252`), H2, all authority cardinalities (344/122/222/0/122/18/74; target 2/120/0/0/0/true; final 122), AC-9, Annex A, and Annex B are **unchanged**.
6. The amendment ID `ANNEX_C_AMEND_R28_PREP070_RECON_V1` is **retired**. All live records, schemas, and ledger entries must carry `ANNEX_C_AMEND_R28_PREP070_RECON_V2`.
7. The flawed adjudication is preserved **verbatim as a superseded, non-authoritative blocking artifact**, alongside this corrected adjudication, with an explicit supersession link (Section 10).
8. H3 is **required**. The five in-flight schema files are **retained and amended in place** (Section 7).
9. The non-defective design of the superseded adjudication — honest late reconstruction, typed non-certifying intervals, late GPU observation, narrow incident allowlist — is **re-enacted** under the V2 amendment ID with the corrected authority.
10. Codex remains on its pinned model; the review path remains `gpt-5.5:high`; on any managed-harness safety or capacity check, wait and retry the identical pinned path. The historical journal remains untouched. Archive, matrix regeneration, certifying V2 audit, preflight, bootstrap, and all GPU work remain blocked until the sequencing contract in Section 10 reaches the corresponding step.

---

## 2. FACTUAL CORRECTION

The following relations are adjudicated as the true, deterministic record:

- **Frozen `SOL_DEF_EQUIV_V1` accepts the sealed pair.** For source SHA `3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa` and staged SHA `e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387`, V1 canonicalizes top-level `op_type` missing→null and yields `equal = true`, `differences = []`. This is confirmed independently by the immutable Artifact 1 row (`verdict: "equal"`, `difference_pointers: []`), by the H1 V2 audit, and by the H2 re-run.
- **The historical refusal was produced by the legacy comparator**, not by V1. At original HEAD `9b8e0e40…`, `normalized_definition` rewrites only a *present* empty-string value to `None` for `hf_id`, `op_type`, `custom_inputs_entrypoint`; it never inserts missing keys and never equates a missing key with explicit null. Over the sealed bytes it yields `equal = false` with differing top-level key set exactly `["op_type"]` (source: missing; staged: null), raising `ChunkRefusal("sol_stage_source_mismatch", "staged definition differs from pinned source")`, recorded historically with reason `STAGE_SEMANTIC_VALIDATION_FAILED`.
- **Therefore** the literal `deterministic_v1_comparator_over_sealed_bytes` describes an impossible event and is withdrawn. Calling the rejecting legacy comparator "V1" would rewrite the meaning of a frozen rule ID and contradict Annexes C and D and both immutable audits; this is forbidden.
- Annexes C and D are **not** contradicted by this correction and are **not** amended. Annex C's statement that "the old runner comparator lacked missing≡null behavior" is exactly the legacy defect confirmed here.

---

## 3. PRIOR-ADJUDICATION CLAUSES AMENDED

Annexes A, B, C, and D: **unamended in every clause.**

The superseded adjudication (`…RECON_V1`) is amended at exactly its six defective loci; all other clauses are re-enacted under `…RECON_V2`:

1. Rationale claim that frozen V1 maps the sealed divergence to `sol_stage_source_mismatch` → **struck**; replaced by the Section 2 factual record.
2. Certifying element name `deterministic_v1_comparator_over_sealed_bytes` → **replaced** by `deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes`.
3. `runner_refusal_reconstruction.json.authority_basis` → **replaced** by the const array in Section 5.
4. `phase_reconstruction.json` refusal evidence basis → **replaced** by the same const array.
5. `no_gpu_attestation.certifying_elements` → **replaced** by the const array in Section 5.
6. Optional V1-based reproduction contract → **struck**; replaced by the required legacy reproduction contract in Section 6.

---

## 4. EXACT AUTHORITY AND COMPARATOR CONTRACT

**Authority basis (order-fixed, conjunctive, exhaustive):**

```json
["annex_c_adjudication", "deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes"]
```

- `annex_c_adjudication`: supplies the adjudicated historical classification (pre-GPU preparation failure; zero campaign rows; zero Annex-B supersession attempts consumed). Citation must carry SHA `b9bc56d06ddbed14ca28df02012077a533c9b846a90f2f08ffdcfc9c6c52b568`.
- `deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes`: supplies the mechanical derivation of the refusal. It is **certifying** only when the full identity contract below is populated with verified values and the required reproduction record exists with the expected verdict. If any identity element fails verification, the element is void, the record is invalid, and Codex must stop and re-escalate; there is no fallback to Annex-C-only authority.

**Frozen legacy algorithm identity — `SOL_DEF_LEGACY_COMPARE_9B8E0E40`:**

Normative semantics (binding, character-exact where quoted): for top-level fields `hf_id`, `op_type`, `custom_inputs_entrypoint`, a **present** value equal to `""` is rewritten to `null`; missing keys are **not** inserted and are **never** equated with `null` or `""`; all other content is compared by strict dict equality; inequality maps to refusal code `sol_stage_source_mismatch` with message `staged definition differs from pinned source`.

This rule ID is a frozen *historical* identity. It is **not** an equivalence-certification rule, must never be emitted into any equivalence attestation, and is valid only in refusal-reconstruction contexts under this amendment.

**Complete identity contract (all items required; answers Question 4 — yes to all nine, plus three additions):**

| Field | Value |
|---|---|
| `rule_id` | `SOL_DEF_LEGACY_COMPARE_9B8E0E40` (const) |
| `head` | `9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a` (const) |
| `path` | `tools/local_labels/sm86_capture_chunk.py` (const) |
| `file_sha256` | SHA-256 of the bytes of `git show 9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a:tools/local_labels/sm86_capture_chunk.py` — computed and sealed by Codex |
| `line_span` | `[1306, 1317]` (const) |
| `line_span_sha256` | SHA-256 of the excerpt produced by the bound extraction command below — computed and sealed by Codex |
| `function` | `normalized_definition` (const) |
| `source_definition_sha256` | `3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa` (const) |
| `staged_definition_sha256` | `e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387` (const) |
| result | `equal: false` (const) |
| differing key | `["op_type"]`, pointer `"/op_type"` (const) |
| refusal | code `sol_stage_source_mismatch`, message `staged definition differs from pinned source`, reason `STAGE_SEMANTIC_VALIDATION_FAILED` (consts) |

**Bound extraction command** (deterministic; excerpt saved as `legacy_comparator_source_excerpt.py.txt`):

```
git -C /home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite \
  show 9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a:tools/local_labels/sm86_capture_chunk.py \
  | sed -n '1306,1317p' > legacy_comparator_source_excerpt.py.txt
```

Sanity guard: the excerpt must contain the tokens `def normalized_definition` and `sol_stage_source_mismatch`; otherwise refuse with `SOL_LEGACY_REPRO_SOURCE_DRIFT`.

**Placeholder convention (binding):** any value below written as a string beginning `TO_BE_SEALED` must be replaced at construction time with the verified value. Schemas enforce hex/timestamp patterns, so placeholder strings can never validate or ship.

**Non-certifying context (mandatory in the refusal reconstruction):** the V1 `equal` result (Artifact 1) and V2 `equal` result (H1 audit) must be recorded under `non_certifying_context` as the intended *repair* semantics. They must never appear in any `authority_basis` or `certifying_elements`.

---

## 5. EXACT JSON/SCHEMA SHAPES

Construction site (binding): `.humanize/rlcr/2026-07-08_11-39-13/prep070_reconstruction/` containing the five records, `legacy_comparator_source_excerpt.py.txt`, `gpu_process_observation_late.raw.txt`, and `scratch/` (retained and archived). All timestamps: RFC 3339 UTC with `Z` suffix, millisecond precision or finer. All SHA fields: pattern `^[0-9a-f]{64}$`. All schemas: `additionalProperties: false` throughout.

### 5.1 `runner_refusal_reconstruction.json`

```json
{
  "record_type": "runner_refusal_reconstruction",
  "amendment_id": "ANNEX_C_AMEND_R28_PREP070_RECON_V2",
  "reconstruction_late": true,
  "contemporaneous": false,
  "reconstructed_at": "TO_BE_SEALED:wall_clock_utc",
  "cell": "sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation",
  "chunk": "sol_execbench-000-bootstrap",
  "original_head": "9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a",
  "session": "f10de1cc-612e-43e3-b51f-ce1c9985e281",
  "matrix_file_sha256": "fd3a1868845ceac309e718da6519f4779824b7fd69eddf1bfa0e7e7847d4e297",
  "preflight_sha256": "3810d2b29a06163f41d6aab73375b273e87cf462a7ad22032088802eef85b63a",
  "historical_refusal": {
    "code": "sol_stage_source_mismatch",
    "message": "staged definition differs from pinned source",
    "reason": "STAGE_SEMANTIC_VALIDATION_FAILED"
  },
  "sealed_inputs": {
    "source_definition_sha256": "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa",
    "staged_definition_sha256": "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387",
    "staged_module_sha256": "017ec9c8c977078c141e8d7d4fb58a67501550aa5540c220dbd97a94b8e1ee96"
  },
  "legacy_comparator_identity": {
    "rule_id": "SOL_DEF_LEGACY_COMPARE_9B8E0E40",
    "head": "9b8e0e403ec0fc2acc2fcf40dbe3feb31572475a",
    "path": "tools/local_labels/sm86_capture_chunk.py",
    "file_sha256": "TO_BE_SEALED:sha256_of_git_blob_bytes",
    "line_span": [1306, 1317],
    "line_span_sha256": "TO_BE_SEALED:sha256_of_bound_extraction_excerpt",
    "function": "normalized_definition",
    "source_excerpt_file": "legacy_comparator_source_excerpt.py.txt",
    "semantics": "For top-level fields hf_id, op_type, custom_inputs_entrypoint: a PRESENT empty-string value is rewritten to null; missing keys are not inserted and are never equated with null or empty string; all other content is compared by strict dict equality; inequality raises ChunkRefusal sol_stage_source_mismatch."
  },
  "rederivation": {
    "equal": false,
    "differing_top_level_keys": ["op_type"],
    "difference_pointers": ["/op_type"],
    "source_op_type_state": "missing",
    "staged_op_type_state": "null",
    "maps_to_refusal_code": "sol_stage_source_mismatch",
    "reproduction_record": "legacy_comparator_reproduction.json",
    "reproduction_record_sha256": "TO_BE_SEALED:sha256"
  },
  "authority_basis": [
    "annex_c_adjudication",
    "deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes"
  ],
  "authority_citations": {
    "annex_c_adjudication_sha256": "b9bc56d06ddbed14ca28df02012077a533c9b846a90f2f08ffdcfc9c6c52b568",
    "corrected_adjudication_sha256": "TO_BE_SEALED:sha256_of_v2_adjudication_file",
    "superseded_adjudication_sha256": "TO_BE_SEALED:sha256_of_superseded_v1_file",
    "supersession_status": "superseded_by_v2"
  },
  "non_certifying_context": {
    "sol_def_equiv_v1": {
      "verdict": "equal",
      "difference_pointers": [],
      "artifact_sha256": "b6dd22ad6e7a87033374262aa8cd6a8a3ea97ac901a5e1c266a3a6d7341cdcd0"
    },
    "sol_def_equiv_v2": {
      "verdict": "equal",
      "audit_sha256": "ea51b0c6da89f5d339cab365ed9bfc69cb642a44dd73dc2fabe9d5dafafcb252"
    },
    "note": "V1/V2 equality is the intended repair semantics and is never a basis for the historical refusal."
  }
}
```

### 5.2 `phase_reconstruction.json`

Uniform phase shape: `{phase, occurred, start, end, certainty, evidence_basis?, refusal_code?, reason?}`; nulls permitted for unknown endpoints.

```json
{
  "record_type": "phase_reconstruction",
  "amendment_id": "ANNEX_C_AMEND_R28_PREP070_RECON_V2",
  "interval_type": "non_certifying_reconstruction",
  "reconstruction_late": true,
  "contemporaneous": false,
  "reconstructed_at": "TO_BE_SEALED:wall_clock_utc",
  "cell": "sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation",
  "chunk": "sol_execbench-000-bootstrap",
  "anchors": [
    {
      "name": "prepare_stdout_mtime",
      "value": "2026-07-10T13:34:52.699087829Z",
      "basis": "filesystem_stat_mtime",
      "evidence_sha256": "ca3249b38c8c5dc3c8d428aed3aa8c650978d629903640befb46517aed7d8cdc",
      "certainty": "exact_observed"
    },
    {
      "name": "journal_start",
      "value": null,
      "basis": "journal_start_sha256",
      "evidence_sha256": "6b6dcce8cb7f4a21385c6e52c693f7c398258877ec702090438ee67bea8faa4f",
      "certainty": "content_sealed_time_unknown"
    }
  ],
  "phases": [
    {
      "phase": "admission",
      "occurred": true, "start": null, "end": null, "certainty": "unknown_interval",
      "evidence_basis": ["admission_run:0ff22c451824e77fa4ec20392866c50a7356b6de4e1e986ba788bd1b5ee1a658"]
    },
    {
      "phase": "prepare_execution",
      "occurred": true, "start": null, "end": "2026-07-10T13:34:52.699087829Z",
      "certainty": "end_bounded_by_mtime",
      "evidence_basis": ["prepare_stdout_sha256:ca3249b38c8c5dc3c8d428aed3aa8c650978d629903640befb46517aed7d8cdc"]
    },
    {
      "phase": "stage_semantic_validation_refusal",
      "occurred": true, "start": null, "end": null, "certainty": "unknown_interval",
      "evidence_basis": [
        "annex_c_adjudication",
        "deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes"
      ],
      "refusal_code": "sol_stage_source_mismatch",
      "reason": "STAGE_SEMANTIC_VALIDATION_FAILED"
    },
    {
      "phase": "gpu_execution",
      "occurred": false, "start": null, "end": null, "certainty": "certified_absent",
      "evidence_basis": ["journal_forensic_absence_scan", "annex_c_adjudication"]
    },
    {
      "phase": "finalization",
      "occurred": false, "start": null, "end": null, "certainty": "certified_absent",
      "evidence_basis": ["journal_complete_json_absent"]
    }
  ]
}
```

### 5.3 `preparation_failure_no_gpu_attestation_v2.json`

```json
{
  "record_type": "preparation_failure_no_gpu_attestation",
  "record_version": 2,
  "amendment_id": "ANNEX_C_AMEND_R28_PREP070_RECON_V2",
  "cell": "sol_execbench__problem_070_mamba2_fused_intra_chunk_diagonal_computation",
  "chunk": "sol_execbench-000-bootstrap",
  "attestation": "No GPU execution occurred in the historical problem-070 preparation attempt; the attempt terminated at pre-GPU staged-definition semantic validation.",
  "certifying_elements": [
    "annex_c_adjudication",
    "deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes",
    "journal_forensic_absence_scan"
  ],
  "journal_forensic_absence_scan": {
    "forbidden_artifact_scan": [],
    "journal_complete_present": false,
    "prepare_stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "prepare_stderr_empty": true,
    "inventory_files": 22,
    "inventory_bytes": 3855797,
    "inventory_domain_sha256": "ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36"
  },
  "non_certifying_elements": ["gpu_process_observation_late"],
  "late_observation_reference": {
    "file": "gpu_process_observation_late.json",
    "sha256": "TO_BE_SEALED:sha256",
    "certifying": false
  },
  "reproduction_reference": {
    "file": "legacy_comparator_reproduction.json",
    "sha256": "TO_BE_SEALED:sha256"
  },
  "campaign_rows": 0,
  "capture_supersession_attempts_consumed": 0
}
```

### 5.4 `legacy_comparator_reproduction.json` — see Section 6.

### 5.5 `gpu_process_observation_late.json`

```json
{
  "record_type": "gpu_process_observation_late",
  "amendment_id": "ANNEX_C_AMEND_R28_PREP070_RECON_V2",
  "observed_at": "TO_BE_SEALED:wall_clock_utc",
  "certifying": false,
  "late": true,
  "method": "nvidia-smi_readonly_query",
  "command": "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv",
  "raw_output_file": "gpu_process_observation_late.raw.txt",
  "raw_output_sha256": "TO_BE_SEALED:sha256",
  "interpretation": "Late observation only; asserts nothing about the historical 2026-07-10 attempt."
}
```

### 5.6 Binding schema rules (apply to the five files in the worktree)

- `runner_refusal_reconstruction.schema.json`: `authority_basis` is a **`const`** of the exact two-element array above (order-fixed). Defense-in-depth: additionally include
  ```json
  "not": { "properties": { "authority_basis": { "contains": { "const": "deterministic_v1_comparator_over_sealed_bytes" } } }, "required": ["authority_basis"] }
  ```
  `legacy_comparator_identity.rule_id`, `head`, `path`, `line_span`, `function` are `const`; `rederivation.equal` const `false`; `differing_top_level_keys` const `["op_type"]`; `difference_pointers` const `["/op_type"]`; `maps_to_refusal_code` const `sol_stage_source_mismatch`; `cell` const (incident allowlist of exactly one); `amendment_id` const `ANNEX_C_AMEND_R28_PREP070_RECON_V2`; `non_certifying_context.sol_def_equiv_v1.verdict` const `"equal"`.
- `phase_reconstruction.schema.json`: `interval_type` const `non_certifying_reconstruction`; the refusal phase's `evidence_basis` const two-element array; `gpu_execution.occurred` const `false`; `amendment_id` const V2; same `not`-guard against the withdrawn literal.
- `preparation_failure_no_gpu_attestation_v2.schema.json`: `certifying_elements` const three-element array exactly as in 5.3; `non_certifying_elements` must contain `gpu_process_observation_late`; `journal_forensic_absence_scan` sub-fields const to the sealed values (`[]`, `false`, `e3b0c4…`, `true`, `22`, `3855797`, `ff0ace…`); `reproduction_reference` required; same `not`-guard.
- `gpu_process_observation_late.schema.json`: `certifying` const `false`; `late` const `true`.
- `preparation_failure_ledger_entry.schema.json`: add required object
  ```json
  "authority": {
    "amendment_id": "ANNEX_C_AMEND_R28_PREP070_RECON_V2",
    "corrected_adjudication_sha256": "<64-hex, required, authoritative>",
    "superseded_adjudication_sha256": "<64-hex, required for this incident>",
    "supersession_status": "superseded_by_v2"
  }
  ```
  with `amendment_id` and `supersession_status` as consts; plus required `reconstruction_records` array referencing all five records by path and SHA-256; `campaign_rows` const 0; `capture_supersession_attempts_consumed` const 0; `supersedes` const null; `prior_preparation_failures` linkage per Annex C. The string `ANNEX_C_AMEND_R28_PREP070_RECON_V1` must never validate anywhere.

---

## 6. REQUIRED REPRODUCTION CONTRACT (formerly optional Option C)

Reproduction is **required**, not optional. Omission is rejected; frozen-V1 invocation is rejected. The contract merges options (a) and (b) of Question 6: a **pure extracted legacy comparator** whose behavior is test-pinned to the original-HEAD source, executed over verified scratch copies, with **no import of capture/runner code**.

**Tool:** new repo module `tools/local_labels/legacy_compare_9b8e0e40.py` (part of H3). Constraints: Python stdlib only (`json`, `hashlib`, `argparse`, `pathlib`, `sys`); no import of `sm86_capture_chunk` or any capture, runner, CUDA, or subprocess-launching module; CPU-only; no network; no writes outside the emitted record path; exit non-zero on any refusal.

**Inputs:** Codex copies, read-only, the journal files whose SHA-256 equal the sealed source/staged definition SHAs into `prep070_reconstruction/scratch/source_definition.json` and `scratch/staged_definition.json`. The tool re-hashes both and refuses with `SOL_LEGACY_REPRO_INPUT_DRIFT` on any mismatch with `--expect-source-sha` / `--expect-staged-sha`.

**Source-identity verification:** the record builder re-runs the bound extraction command (Section 4), recomputes `file_sha256` and `line_span_sha256`, and refuses with `SOL_LEGACY_REPRO_SOURCE_DRIFT` on any mismatch with the sealed identity, or if the excerpt lacks the two sanity tokens.

**Bound command shape:**

```
python3 -m tools.local_labels.legacy_compare_9b8e0e40 \
  --source prep070_reconstruction/scratch/source_definition.json \
  --staged prep070_reconstruction/scratch/staged_definition.json \
  --expect-source-sha 3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa \
  --expect-staged-sha e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387 \
  --emit prep070_reconstruction/legacy_comparator_reproduction.json
```

**Expected verdict (consts in schema and asserted by builder):** `equal: false`; `differing_top_level_keys: ["op_type"]`; `difference_pointers: ["/op_type"]`; `maps_to_refusal_code: "sol_stage_source_mismatch"`; rule ID `SOL_DEF_LEGACY_COMPARE_9B8E0E40`.

**Record shape:**

```json
{
  "record_type": "legacy_comparator_reproduction",
  "amendment_id": "ANNEX_C_AMEND_R28_PREP070_RECON_V2",
  "rule_id": "SOL_DEF_LEGACY_COMPARE_9B8E0E40",
  "executed_at": "TO_BE_SEALED:wall_clock_utc",
  "no_gpu": true,
  "pure_cpu": true,
  "capture_code_imported": false,
  "tool": {
    "path": "tools/local_labels/legacy_compare_9b8e0e40.py",
    "sha256": "TO_BE_SEALED:sha256_of_tool_file",
    "head": "TO_BE_SEALED:h3_commit_hash"
  },
  "inputs": {
    "source_scratch_path": "scratch/source_definition.json",
    "source_sha256_expected": "3e0aceb31c7639d533a7b9c0c9f3e75cc7e6f4fb9beb9e56d57ee73bdbf1c4fa",
    "source_sha256_observed": "TO_BE_SEALED:must_equal_expected",
    "staged_scratch_path": "scratch/staged_definition.json",
    "staged_sha256_expected": "e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387",
    "staged_sha256_observed": "TO_BE_SEALED:must_equal_expected"
  },
  "verdict": {
    "equal": false,
    "differing_top_level_keys": ["op_type"],
    "difference_pointers": ["/op_type"],
    "maps_to_refusal_code": "sol_stage_source_mismatch"
  },
  "python_version": "TO_BE_SEALED",
  "command_argv": "TO_BE_SEALED:exact_argv_string"
}
```

---

## 7. REQUIRED CODE CHANGES, IN ORDER

1. Amend the five schema files in the stopped worktree in place (retain the drafts; do not discard): apply every Section 5.6 rule; purge any occurrence of the withdrawn literal and of amendment ID `…_V1` (they may appear only inside `not`-guards and test fixtures).
2. Add `tools/local_labels/legacy_compare_9b8e0e40.py` per Section 6, embedding the legacy algorithm with semantics character-consistent with Section 4.
3. Extend the archive-tooling record builders/validators: build the five records; enforce the placeholder convention (no `TO_BE_SEALED` value may persist); enforce the incident allowlist (exactly problem-070, refusal `SOL_PREP_RECON_ALLOWLIST_VIOLATION`); enforce conjunctive authority citations; enforce `reconstructed_at`/`observed_at`/`executed_at` strictly greater than `2026-07-10T20:15:42Z` (H2); assert the frozen-V1 problem-070 verdict cited in `non_certifying_context` is `equal` and refuse any record claiming otherwise.
4. Add a guard so equivalence-attestation writers refuse rule ID `SOL_DEF_LEGACY_COMPARE_9B8E0E40` (in addition to the existing Annex-D rule that only `SOL_DEF_EQUIV_V2` certifies and V1 is never emitted).
5. Add the tests of Sections 8 and 9.
6. Run the complete no-GPU validation (all prior suites must remain green) plus the new tests; then commit H3 with subject exactly:
   `fix(local_labels): rebind prep070 refusal authority to legacy 9b8e0e40 comparator`
   Tree must be clean immediately after H3.

---

## 8. REQUIRED POSITIVE TESTS

1. `test_v1_frozen_accepts_problem070_sealed_bytes` — frozen `SOL_DEF_EQUIV_V1` over fixtures matching the sealed SHAs yields `equal`, no pointers.
2. `test_legacy_9b8e0e40_rejects_problem070_sealed_bytes` — the extracted comparator yields `equal=false`, differing keys exactly `["op_type"]`, mapping to `sol_stage_source_mismatch`.
3. Legacy semantics unit tests: present `""`→null normalization for each of the three fields; missing key not inserted; missing vs null unequal; present `""` vs null equal; nested content untouched; non-listed top-level fields strict.
4. `test_excerpt_extraction_matches_git_blob` — bound extraction command output matches sealed `line_span_sha256`; blob bytes match sealed `file_sha256`; sanity tokens present.
5. `test_reproduction_record_expected_verdict` — end-to-end tool run over fixture copies emits a schema-valid record with the Section 6 verdict consts.
6. `test_all_five_records_validate` — fully populated records (no placeholders) validate against their schemas.
7. `test_authority_basis_const_exact_order` — permuted or extended arrays fail validation.

---

## 9. REQUIRED NEGATIVE TESTS

1. Schema rejects `authority_basis`, `evidence_basis`, or `certifying_elements` containing `deterministic_v1_comparator_over_sealed_bytes`.
2. Schema rejects amendment ID `ANNEX_C_AMEND_R28_PREP070_RECON_V1` in any record or ledger entry.
3. Builder refuses any record asserting a `SOL_DEF_EQUIV_V1` verdict other than `equal` for the sealed problem-070 SHAs (frozen-rule-meaning guard).
4. Reproduction tool refuses input SHA drift → `SOL_LEGACY_REPRO_INPUT_DRIFT`.
5. Builder refuses source-identity drift (blob SHA, excerpt SHA, or missing sanity tokens) → `SOL_LEGACY_REPRO_SOURCE_DRIFT`.
6. Static test: `legacy_compare_9b8e0e40.py` imports no capture/runner/CUDA/subprocess modules.
7. Allowlist: any cell other than problem-070 → `SOL_PREP_RECON_ALLOWLIST_VIOLATION`.
8. Timestamp truthfulness: `reconstructed_at`/`observed_at`/`executed_at` ≤ `2026-07-10T20:15:42Z` refused; `reconstructed_at` ≤ prepare-stdout mtime refused; `contemporaneous` may never be true; `reconstruction_late` may never be false.
9. `interval_type` other than `non_certifying_reconstruction` refused.
10. Attestation missing `reproduction_reference` refused; attestation listing `gpu_process_observation_late` among `certifying_elements` refused.
11. Ledger entry citing the superseded adjudication SHA in `corrected_adjudication_sha256`, or with equal corrected/superseded SHAs, refused by the builder.
12. Equivalence attestation carrying `SOL_DEF_LEGACY_COMPARE_9B8E0E40` (or `SOL_DEF_EQUIV_V1`, per Annex D) refused.
13. Any record retaining a `TO_BE_SEALED` placeholder fails validation.

---

## 10. H3 / RECORD-PRESERVATION / SEQUENCING CONTRACT

**H3 is required.** Binding sequence; no step may begin before the prior step completes:

1. Implement Section 7 items 1–5.
2. Full no-GPU validation green (all prior suites plus Sections 8–9).
3. **H3 commit** (bound subject; clean tree after).
4. Save adjudication artifacts, both mode `0444`:
   - Flawed adjudication **verbatim from the session transcript**: `.humanize/rlcr/2026-07-08_11-39-13/adjudication_r28_prep070_reconstruction_superseded_v1.md`
   - This corrected adjudication verbatim: `.humanize/rlcr/2026-07-08_11-39-13/adjudication_r28_prep070_reconstruction_v2.md`
   Compute both SHA-256 values.
5. Append to the RLCR session log (append-only; no rewriting of prior entries):
   ```json
   {"event":"adjudication_saved","artifact":"adjudication_r28_prep070_reconstruction_superseded_v1.md","sha256":"<v1_sha>","status":"superseded_non_authoritative","superseded_by_sha256":"<v2_sha>","defect":"false_certifying_premise:deterministic_v1_comparator_over_sealed_bytes"}
   {"event":"adjudication_saved","artifact":"adjudication_r28_prep070_reconstruction_v2.md","sha256":"<v2_sha>","status":"binding","supersedes_sha256":"<v1_sha>","amendment_id":"ANNEX_C_AMEND_R28_PREP070_RECON_V2","h3":"<h3_hash>"}
   ```
6. Construct live records in `prep070_reconstruction/`, in this order: scratch copies + SHA verification → source-identity verification + excerpt file → reproduction run → `runner_refusal_reconstruction.json` → `phase_reconstruction.json` → `gpu_process_observation_late.json` (+ raw output file; the read-only `nvidia-smi` query is the sole authorized GPU-adjacent action before bootstrap) → `preparation_failure_no_gpu_attestation_v2.json` last (it references the others). Validate all; append record SHAs to the session log.
7. Re-verify the historical journal is byte-identical (22 files / 3,855,797 bytes; inventory-domain SHA `ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36`), then perform the Annex-C **atomic archive** of the journal plus the reconstruction records, with inventory and append-only ledger (ledger `authority` object per Section 5.6). Verify post-archive that the journal's inventory-domain SHA is preserved inside the archive, that no archived artifact is reused, and that no published path exists under `superseded/` or `preparation_failures/`.
8. Matrix regeneration.
9. Certifying H3 V2 audit (a **new** artifact; the H1 V2 audit and Artifact 1 are never regenerated or touched).
10. Preflight (Annex A/B contracts unchanged: fingerprint `0d7c673c…3fe68f`, `NVCC_CCBIN="g++-14"`, ambient tripwire, single-preflight identity, 900.000 s inclusive freshness with 2 s future-skew guard).
11. Repaired one-cell SOL bootstrap → target 2 accounted / 120 missing / 0 duplicates / 0 unexpected / 0 leaks / integrity true.
12. Remaining captures and review under pinned `gpt-5.5:high`; final target exactly 122 terminal published outcomes.

---

## 11. ANSWERS TO QUESTIONS 1–15

1. **Yes.** `deterministic_v1_comparator_over_sealed_bytes` is withdrawn as factually incorrect (Section 2). No live artifact may carry it.
2. Replacement string, character-exact, at every occurrence (authority_basis, phase refusal evidence basis, attestation certifying_elements, rationale/provenance text, reproduction output): `deterministic_legacy_9b8e0e40_normalized_definition_comparator_over_sealed_bytes`.
3. The legacy re-derivation **is certifying**, conjunctively with `annex_c_adjudication`; neither alone suffices. V1/V2 results are mandatory non-certifying context (Section 4).
4. Yes to all nine seal items, plus three additions: refusal message literal, historical reason code, and `line_span_sha256` with the bound extraction command (Section 4 table).
5. Provided in full in Section 5 (5.1–5.3) and Section 6 (reproduction record).
6. Amended: reproduction is **required** and uses the pure extracted legacy comparator, test-pinned to the original-HEAD source, without importing or running capture code. Frozen-V1 invocation and omission are both rejected.
7. Full contract in Section 6: no-GPU/pure-CPU constraints, schema/record shape, expected verdict consts, rule identifier `SOL_DEF_LEGACY_COMPARE_9B8E0E40`, bound command, drift refusal codes, evidence fields.
8. **Confirmed.** `SOL_DEF_EQUIV_V1` remains frozen with existing semantics; Artifact 1 (`b6dd22ad…cdcd0`, 0444) including its problem-070 `equal` row remains byte-identical; V1 is never emitted into new attestations.
9. **Confirmed.** `SOL_DEF_EQUIV_V2`, the H1 audit (`ea51b0c6…afbf5…` as sealed: `ea51b0c6da89f5d339cab365ed9bfc69cb642a44dd73dc2fabe9d5dafafcb252`), H2 (`823f2d60…1abc`), all cardinalities (344/122/222/0; budgets 122; chunks 18; SOL cells 74; bootstrap 1→2 accounted/120 missing/0/0/0/true; final 122), AC-9, Annex A, and Annex B are unchanged.
10. **H3 is required.** Retain and amend the five draft schema files in place; git-add the four untracked files; final bytes must satisfy Section 5.6. Recreation of a file is permitted only if amendment would amount to a rewrite; the result must be identical in effect either way. Do not discard as a substitute for correction.
11. **Save both with an explicit supersession link** (Section 10 steps 4–5): filenames and 0444 modes as bound; SHAs computed at save time and logged; ledger `authority` object carries the corrected SHA as sole authority and the superseded SHA in the dedicated non-authoritative field with `supersession_status: "superseded_by_v2"`. Prior ledger/log entries are never rewritten.
12. Sequencing bound in Section 10, steps 1–12.
13. The amendment ID is **replaced**: `ANNEX_C_AMEND_R28_PREP070_RECON_V2`. The `…_V1` ID is retired and may appear only inside the preserved superseded artifact, supersession metadata/log fields, `not`-guards, and negative-test fixtures. All schema consts pin the V2 ID; no live record may claim the V1 amendment as authority.
14. Complete test delta reissued in Sections 8 and 9, covering all six requested proof obligations (V1 accepts; legacy rejects; schemas cannot conflate; V1-based refusal claims rejected; legacy source/input drift refuses; truth/timestamp/allowlist constraints closed).
15. **Confirmed.** No historical journal byte may be changed or moved before the Section 10 step-7 atomic archive; archive, matrix regeneration, certifying audit, preflight, bootstrap, and all GPU work (except the step-6 read-only late observation) remain blocked until the corrected authority is implemented, tested, H3-committed, and recorded in the session log.

---

## 12. FORBIDDEN SHORTCUTS

- Do not relabel, alias, or reinterpret frozen `SOL_DEF_EQUIV_V1` as a rejecting rule; do not attach rejection semantics to any `SOL_DEF_EQUIV_*` identifier; do not present a V1-equal result as a historical refusal.
- Do not modify, regenerate, or re-hash Artifact 1, the H1 V2 audit, any 0444 artifact, the historical journal, or `/etc/profile.d/cuda.sh`.
- Do not emit `deterministic_v1_comparator_over_sealed_bytes` or `ANNEX_C_AMEND_R28_PREP070_RECON_V1` into any live record, schema value position, ledger, or attestation (permitted only in the superseded artifact, supersession metadata, `not`-guards, this adjudication's withdrawal narrative, and negative-test fixtures).
- Do not fabricate any `TO_BE_SEALED` value; every sealed value must be computed from git or the filesystem and logged. Placeholders must be unshippable by schema.
- Do not run the reproduction via `compare_definition_bytes_v1`, via import of `sm86_capture_chunk.py`, or via any capture/runner entry point.
- Do not begin archive, matrix regeneration, certifying V2 audit, preflight, bootstrap, capture, or any GPU work out of the Section 10 order; the only pre-bootstrap GPU-adjacent action is the single read-only `nvidia-smi` query of step 6.
- Do not publish any outcome under `superseded/` or `preparation_failures/`; do not reuse archived artifacts; do not consume Annex-B capture-supersession attempts for this preparation failure; do not add campaign rows for it.
- Do not switch, downgrade, or substitute models or the `gpt-5.5:high` review path; on managed-harness safety or capacity checks, wait and retry the identical pinned path.
- Do not guess on any future contradiction: stop and re-escalate under `AGENTS.md`.

**End of binding adjudication.** Codex may resume implementation at Section 7, step 1.
