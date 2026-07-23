# ADJUDICATION — Round 28 Nested Axis-Description Serialization Divergence

Binding adjudication follows. Save verbatim to `.humanize/rlcr/2026-07-08_11-39-13/adjudication_r28_axis_description_equivalence.md` and record its SHA-256 in the RLCR session log before implementation begins.

---

## 1. DECISION

**Option A is binding, in its strict form, issued as a new rule `SOL_DEF_EQUIV_V2`.**

The definition-equivalence contract is amended to add exactly one nested eligible class: at positions matching exactly `/axes/<axis-key>/description` — where `axes` is the top-level key, its value is a JSON object, and the axis value is a JSON object — a missing `description` key and an explicit JSON `null` are equivalent, symmetrically, on each side independently. Nothing else changes. Empty string is **not** equivalent at these positions. `/inputs/*/description`, `/outputs/*/description`, top-level `/description`, and every deeper or displaced `description` position remain fully strict; divergence there remains a defect requiring immediate stop and new escalation.

Options B, C, D, and E are rejected. The packager's staged serialization, the staged-definition hash algorithm, all pinned sources, all staged bytes, the 74-cell SOL bridge, and the 344/122/222 authority contract are all unchanged. Problem-070's expected staged SHA remains `e752675533d1d49220488a5df6a107648d9c2e6137820c4dc1d9d8cd0fc5f387`.

Every clause of Annex A, Annex B, and Annex C not expressly named as amended in Section 3 below remains binding verbatim.

## 2. RATIONALE

The audit produced complete, uniform evidence: all 212 divergent pointers across all 31 divergent rows have the identical shape — source omits an optional axis `description`, and Pydantic default-materialization inserts `"description": null`. This is mechanically identical to the already-adjudicated problem-070 top-level `op_type` absent→null case; only the position differs. No execution-bearing field diverged anywhere in the 74-row universe.

Option A is the smallest rule expansion that closes the complete observed production evidence. The campaign's governing posture is evidence-bounded closed classes with mandatory stop-and-escalate on anything outside them. Option B would pre-authorize equivalence at input/output/top-level description positions where zero production divergence exists; the synthetic fixture demonstrates the *mechanism* can reach those positions, but the provenance model deliberately prices a future bounded stop-and-escalate below a present unexamined trust-surface expansion. If a future roster diverges at an input/output/top-level description, that stop is the system working as designed, and this adjudication pre-structures that future escalation (Section 13, Q2).

Option C is rejected because it directly contradicts Annex C's retained "packager serialization unchanged" decision, would alter the actual compile/runtime staged bytes, would invalidate problem-070's expected staged SHA and its equivalence control, and would put the staged artifacts out of alignment with the certified g++-14 runnability/shrink evidence, forcing broad re-certification to fix a comparison-side issue. Option D is evidence/input rewriting and remains categorically forbidden. Option E mutates the sealed authority contract to route around a validator and remains categorically forbidden.

A new rule ID (`V2`) rather than an in-place amendment of `V1` is required because attestations embed rule IDs as certifying provenance; two materially different semantics must never share one identifier. Since no SOL capture has ever sealed under V1, there is no dual-version evidence burden.

## 3. PRIOR-ADJUDICATION CLAUSES AMENDED

Exactly the following clauses are amended. Everything else in Annexes A, B, and C remains binding verbatim.

1. **Annex C, C1, "Canonicalization is top-level only"** — amended. Canonicalization now comprises the three top-level fields (unchanged semantics) plus the single nested eligible class defined in Section 4. The C1 defect clause ("any divergence … outside these closed classes is a defect requiring immediate stop and a new escalation — Codex must not widen the rule unilaterally") is **retained** and now applies to the V2 closed classes.
2. **Annex C, C1, rule identifier** — `SOL_DEF_EQUIV_V1` is superseded by `SOL_DEF_EQUIV_V2` for all attestations, stage seals, and verifier checks. V1 is retained only as a frozen, non-certifying reference (Section 5).
3. **Annex C, C1, mismatch reporting ("first 20 JSON-pointer locations")** — clarified, not weakened: the 20-pointer cap applies only to refusal payloads emitted by the adapter/runner. Audit artifacts must record **complete** pointer lists.
4. **Annex C, C1, stage-seal attestation shape** — extended per Section 7 (adds `axes_container_state`, `axis_description_states`).

Explicitly **not** amended: Annex A in its entirety (environment contract, fingerprint `0d7c673c…`, wrapper mitigation, forbidden environment shortcuts); Annex B in its entirety (single-preflight identity, 900 s freshness, 2 s skew, cross-preflight recovery, capture supersession, DONE/accounting closure); Annex C's C2, C3, C4 (restated under V2 in Section 11), C5; the "packager staged serialization and staged-definition hash algorithm remain unchanged" clause; the model constraint; AC-9.

## 4. EXACT DEFINITION-EQUIVALENCE CONTRACT

Rule ID: `SOL_DEF_EQUIV_V2`. Inputs: exact pinned source definition bytes S and packaged staged definition bytes T. The comparator operates on parsed trees only and never rewrites any file.

**§4.1 Parsing (unchanged from V1).** Each side is decoded as strict UTF-8 and parsed as JSON. Duplicate keys at any depth refuse as `sol_definition_duplicate_key`. NaN/Infinity tokens refuse as `sol_definition_nonfinite`. Any other parse failure, or a non-object root, refuses as `sol_definition_parse_error`. Refusals identify which side failed.

**§4.2 Top-level canonicalization (unchanged from V1).** For exactly the root-object keys `hf_id`, `op_type`, `custom_inputs_entrypoint`, on each side independently: if the key is missing, or its value is JSON null, or its value is exactly the string `""`, the key is set to value null. No other key is touched by this step.

**§4.3 Axis-description canonicalization (new).** Evaluated on each side independently:

- Eligibility preconditions: the root object contains key `axes` **and** its value is a JSON object. If `axes` is missing or non-object on a side, this step does nothing on that side; that is not a refusal.
- For each key `k` of the `axes` object whose value is a JSON object, the position `/axes/k/description` is eligible. If the axis value is not an object, no position under it is eligible.
- At each eligible position: if `description` is **missing**, insert `description` with value null. If present with value null, it is unchanged. No other transformation is performed: `""` is **not** normalized; nonempty strings, numbers, booleans, arrays, and objects are **not** normalized.
- Path shape is exact: `axes` must be the top-level key; `<axis-key>` is exactly one object-key segment (any key value; RFC 6901 escaping applies only to reporting/attestation serialization, never to matching); only the **immediate** `description` child of the axis object is eligible. Ineligible, explicitly: top-level `/description`; `/inputs/*/description`; `/outputs/*/description`; `/axes/*/<anything>/description`; any `…/axes/*/description` where `axes` is not the root key; positions under a non-object `axes` or non-object axis value.

**§4.4 Strict recursive comparison (unchanged from V1).** Type-strict throughout: bool differs from int (implementation note: in Python, test bool before int, since `bool` subclasses `int`); int differs from float; strings compare by decoded Unicode code points; arrays require equal length and order-wise equality; objects require exactly equal key sets and per-key equality; unknown fields participate fully. §4.4 semantics are byte-for-byte those of the existing tested V1 comparator; only §4.3 is added.

**§4.5 Difference reporting.** Differences are collected by deterministic depth-first traversal, object keys visited in lexicographic Unicode-code-point order of the raw key, array indices ascending. The comparator API returns the complete difference list and total count. Adapter/runner refusal payloads report at most the first 20 pointers (RFC 6901 escaped) as `sol_stage_source_mismatch`, plus the total count. Audit artifacts record the complete list.

**§4.6 Result.** Equal iff zero differences after §4.2–§4.3 normalization. On equality, the attestation of Section 7 is emitted and sealed. On inequality, refuse as `sol_stage_source_mismatch`; no stage is certified.

**§4.7 Byte preservation.** `definition.source.json` remains the exact source bytes; the staged definition remains the exact packager bytes; `source_definition_sha256` and `staged_definition_sha256` are computed over those exact bytes. Normalization exists only inside the comparison and attestation.

## 5. RULE ID AND COMPATIBILITY

- The certifying rule ID is exactly the string `SOL_DEF_EQUIV_V2` in every new attestation, stage seal, journal, DONE manifest, and audit artifact. Verifiers refuse any other value for newly sealed SOL evidence as `SOL_ATTESTATION_RULE_MISMATCH`.
- No SOL capture sealed under V1; therefore no dual-version acceptance path exists or may be created. `SOL_DEF_EQUIV_V1` is retained in the comparator module solely as a frozen, clearly marked non-certifying reference implementation, used only for the blocking-artifact regeneration (Section 10) and the V1-regression test (P15). It must never be emitted into any attestation.
- All existing uncommitted test fixtures referencing V1 must be updated to V2 expectations, except tests that intentionally exercise the frozen V1 reference and are named accordingly.
- The archived problem-070 preparation-failure record is historical evidence and is not relabeled, reinterpreted, or migrated (Section 11).

## 6. REQUIRED CODE CHANGES — ORDERED

Codex must execute in exactly this order. Steps 1–8 are no-GPU. Q11 disposition: the current uncommitted implementation may be retained and amended; nothing need be discarded wholesale, but every retained part is uncertified until the full battery passes.

1. **Blocking artifact first, before any comparator amendment.** Using the current (still-V1) comparator, deterministically re-run the 74-definition audit and write the immutable blocking artifact (Section 10, Artifact 1). Its counts and row/pointer content must exactly match the escalated evidence (74 rows; 43 equal; 31 divergent; 212 pointers; the exact 31-row list with the tabled per-row counts). Any mismatch is a new immediate stop and escalation. Mark the file read-only (0444) and record its SHA-256.
2. **Amend `tools/local_labels/sol_definition_equivalence.py`** to implement `SOL_DEF_EQUIV_V2` per Section 4, retaining V1 as a frozen reference function, exposing a complete-difference-list API and the per-position state extraction needed for attestations.
3. **Amend `tools/sol_execbench/src/sol_execbench/driver/capture_adapter.py`**: emit the V2 attestation (Section 7) alongside the retained `definition.source.json` copy and both SHAs; refusal payload capped at 20 pointers.
4. **Amend `tools/local_labels/sm86_capture_chunk.py`** (runner): revalidate by recomputing both SHAs from sealed bytes, re-running V2 on `definition.source.json` vs staged bytes, and cross-checking the attestation. Typed refusals: `SOL_ATTESTATION_RULE_MISMATCH` (wrong/absent rule ID), `SOL_ATTESTATION_STATE_MISMATCH` (recorded states differ from recomputation), `SOL_ATTESTATION_POINTER_INELIGIBLE` (any normalized position outside the §4.3 eligible set), plus existing `sol_stage_source_mismatch`. Complete `prior_preparation_failures` propagation (C3 shape; `supersedes` remains null).
5. **Complete the interrupted DONE/accounting lane** in `campaign_done_manifest.py`, `campaign_accounting.py`, and `workloads/simbench_bw/schemas/campaign_done_manifest.schema.json`: require the V2 attestation block for every SOL cell; enforce rule-ID exactness, normalized-pointer eligibility, source-SHA equality against the matrix-pinned per-row source SHA, staged-SHA equality against stage records; enforce bidirectional preparation-ledger closure; reject any published path under `preparation_failures/` or `superseded/` (existing); retain the sole historical portable-bootstrap allowlist.
6. **Create `archive_preparation_failure.py`** implementing Annex C C3 exactly (it does not yet exist): source-seal and 22-file inventory verification against `ff0ace5cc769431efd55d01eb28d7d6aef10300cb00e5722e703306e0d821a36`, forbidden-artifact scan, same-device check, atomic rename into the C3 destination, post-move reverification, `inventory.json`, `chunk_context/` copy, fsynced UUID4 ledger row, two-per-cell limit with `--override-preparation-attempt-limit` / `PREPARATION_ATTEMPT_LIMIT`.
7. **Update and run all tests** (Sections 8–9), including the fixture updated but never rerun; rerun the **full** previously green battery of §2 of the request (local_labels 683+281; SOL adapter/runtime; NVBit Python 339+128; NVBit host; combined DONE/accounting/runner; focused DONE 80), plus Python compilation, shell syntax, and `git diff --check`.
8. **Pre-commit V2 audit (H1)** over the exact 74-member set of the current sealed matrix `fd3a1868…`: zero unpermitted divergences required; write Artifact 2 (Section 10), marked non-certifying.
9. **Commit H2** as a single commit; tree must be clean afterward.
10. **Archive the preparation failure** (step 6 tool) — after H2 exists, before matrix regeneration or any new preflight, per C3.
11. **Regenerate authority and matrix at H2**; verify exact counts 344/122/222, 18 chunks, 0 blocked, 122 budgets, 74 runnable SOL; verify SOL-74 and full-membership row-id equality with H1 (any difference: stop and escalate).
12. **Certifying V2 audit (H2)** against the newly sealed membership from the clean tree; write Artifact 3 at the mandated path (Section 10).
13. **Fresh preflight** via `tools/local_labels/run_sm86_capture_clean_env.sh` (SHA `77a96b4e…`), then the one-cell SOL bootstrap capture under full Annex A/B discipline, carrying `prior_preparation_failures`; seal, verify, reach exactly 2 accounted / 120 missing / 0 duplicates / integrity true; submit to the pinned `gpt-5.5:high` review. No other chunk begins before that review passes.

## 7. REQUIRED SCHEMA AND PROVENANCE

The stage seals a file `definition_equivalence_attestation.json`, serialized deterministically (UTF-8, keys sorted, compact separators `,`/`:`, `ensure_ascii` false, single trailing LF), with its SHA-256 recorded in the stage seal alongside the existing components. Shape:

- `rule_id`: exactly `"SOL_DEF_EQUIV_V2"`.
- `result`: exactly `"equal"` (attestations exist only on equality).
- `source_definition_sha256`, `staged_definition_sha256`.
- `top_level_field_states`: for each of `hf_id`, `op_type`, `custom_inputs_entrypoint`: `{"source": s, "staged": s, "source_normalized": b, "staged_normalized": b}`.
- `axes_container_state`: `{"source": c, "staged": c}` with `c ∈ {object, missing, non_object}`.
- `axis_description_states`: an object keyed by RFC 6901-escaped axis key, one entry per key in the **union** of both sides' `axes` objects (when at least one side's `axes` is an object), keys ordered by raw-key code-point sort; each value `{"source": s, "staged": s, "source_normalized": b, "staged_normalized": b}`.

Closed state vocabulary `s`: `missing`, `null`, `empty_string`, `nonempty_string`, `non_string`; for axis entries additionally `axis_not_object` (that side's axis value is not an object). `*_normalized` is true iff §4.2/§4.3 altered that side's representation at that position (top-level: `missing` and `empty_string`; axis: `missing` only).

DONE-manifest schema additions (already in-flight, to be completed): the attestation block above per SOL cell; verifier checks per Section 6 step 5. New typed refusal codes: `SOL_ATTESTATION_RULE_MISMATCH`, `SOL_ATTESTATION_STATE_MISMATCH`, `SOL_ATTESTATION_POINTER_INELIGIBLE`. Existing refusal codes are unchanged. Accounting's five-step pre-join validation (matrix schema, canonical seal, generator semantics, exact-HEAD regeneration, exact cardinalities) is unchanged and now additionally enforces attestation validity before closure.

## 8. REQUIRED POSITIVE TESTS

- **P1.** All 31 observed rows: fixtures derived from the real pinned sources compare equal under V2, with normalized pointers exactly matching the audited per-row pointer sets (e.g., problem_003's six tabled pointers).
- **P2.** Full real 74-definition integration: load the sealed roster, read pinned sources, serialize through the **real** `ProblemPackager`/`Definition.model_dump_json()` path, compare under V2: 74/74 equal, exactly 31 rows normalized, exactly 212 normalized pointers.
- **P3.** Problem-070 control: source SHA `3e0aceb3…`, staged SHA `e7526755…`, equal; `op_type` states missing/null with top-level normalization; zero axis normalizations.
- **P4.** `missing` vs `null` at `/axes/k/description`, both directions.
- **P5.** `null` vs `null` and `missing` vs `missing` at eligible positions.
- **P6.** Equal explicit nonempty descriptions.
- **P7.** JSON-escape variants of the same decoded string (e.g., `"\u00e9"` vs `"é"`) compare equal.
- **P8.** Axis keys requiring pointer escaping (`a/b`, `a~b`): correct `~1`/`~0` escaping in attestation and reports.
- **P9.** Multiple axes with mixed states in one definition; attestation records the union, sorted, with correct per-side states and normalization flags.
- **P10.** Top-level three-field normalization retained, including `""` → null.
- **P11.** Attestation serialization is byte-deterministic across repeated runs.
- **P12.** Runner revalidation accepts a well-formed V2 stage end-to-end.
- **P13.** DONE/accounting accept a sealed cell with a valid V2 attestation and closed preparation ledger.
- **P14.** Both sides' `axes` missing, or identically non-object: equal (no eligibility, no refusal).
- **P15.** Frozen V1 regression: V1 still flags the 31 rows/212 pointers; V2 accepts them; V1 and V2 agree on every non-eligible difference.

## 9. REQUIRED NEGATIVE TESTS

- **N1.** Nonempty description vs null → mismatch at the exact pointer.
- **N2.** `""` vs null → mismatch. **N3.** `""` vs missing → mismatch.
- **N4.** Bool, int, float, array, and object each vs null at an eligible description → mismatch.
- **N5.** `1` vs `true` and `1` vs `1.0` anywhere → mismatch (type strictness).
- **N6.** Missing vs null at `/axes/k/nested/description` → mismatch (depth ineligible).
- **N7.** Missing vs null at `/inputs/x/description` → mismatch. **N8.** Same at `/outputs/y/description` → mismatch. **N9.** Same at top-level `/description` → mismatch.
- **N10.** Missing vs null at `/wrapper/axes/k/description` → mismatch (non-root `axes`).
- **N11.** `axes` object on one side vs array/string/number/null on the other → mismatch at `/axes`; no normalization performed.
- **N12.** Axis value non-object on one side → mismatch at `/axes/k`; `axis_not_object` state; no eligibility.
- **N13.** Any unknown-field difference → mismatch.
- **N14.** Duplicate keys → `sol_definition_duplicate_key`; NaN/Infinity → `sol_definition_nonfinite`; malformed/non-object root → `sol_definition_parse_error`.
- **N15.** More than 20 differences: refusal payload contains exactly the first 20 in §4.5 traversal order plus the true total; comparator API returns all.
- **N16.** Runner refuses attestation with rule ID `SOL_DEF_EQUIV_V1` or absent → `SOL_ATTESTATION_RULE_MISMATCH`.
- **N17.** Tampered attestation states → `SOL_ATTESTATION_STATE_MISMATCH`; normalized pointer outside the eligible set → `SOL_ATTESTATION_POINTER_INELIGIBLE`.
- **N18.** DONE/accounting reject: missing attestation; publication path under `preparation_failures/` or `superseded/`; unclosed preparation ledger; third preparation attempt without the override flag (`PREPARATION_ATTEMPT_LIMIT`).

## 10. AUDIT-ARTIFACT CONTRACT

Three artifacts, all written atomically (temp file + rename + fsync), all timestamped UTC ISO-8601 `Z`, all recording HEAD, matrix file SHA-256, matrix seal, comparator-module file SHA-256, and packager-module file SHA-256:

1. **`sol_definition_equivalence_audit_r28_v1_blocking.json`** (immutable, 0444): the preserved blocking discovery under frozen V1 — per-row records (row id, source path, source/staged SHAs, verdict, **complete** pointer list) and the global summary 74/43/31/212. This file is never overwritten or deleted; the passing artifacts must not cause its loss.
2. **`sol_definition_equivalence_audit_r28_precommit_h1.json`**: V2 pre-commit gate audit against the H1 matrix `fd3a1868…`; explicitly marked `"certifying": false` (generated from the uncommitted tree).
3. **`sol_definition_equivalence_audit_r28.json`** (the mandated path): the certifying V2 audit, generated at H2 from the clean tree against the regenerated sealed membership. For each of all 74 definitions, sorted by row id: source path; `source_definition_sha256`; `staged_definition_sha256`; original top-level field states; `axes_container_state`; complete `axis_description_states`; sorted list of normalized pointers; `residual_differences: []` (must be empty); per-row verdict `"equal"`. Global block: `rule_id: "SOL_DEF_EQUIV_V2"`; counts (rows 74, equal 74, divergent 0, normalized_rows, total_normalized_pointers); the problem-070 control block (`3e0aceb3…` / `e7526755…`, equal); `prior_blocking_artifact` (path + SHA-256 of Artifact 1); verdict `"pass"`. Any nonzero residual difference means no artifact with verdict `"pass"` may be written and a new stop/escalation fires.

## 11. PRESERVED-ATTEMPT AND RECOVERY IMPACT

The archived problem-070 preparation-failure attempt is untouched: its journal, 22-file inventory (`ff0ace5c…`), session `f10de1cc…`, admission run `0ff22c45…`, and its historical refusal (`STAGE_SEMANTIC_VALIDATION_FAILED` / `sol_stage_source_mismatch`) remain immutable historical evidence and are not relabeled under V2. No artifact from it may be reused. The Annex C C3 archive protocol, its destination naming, ledger shape, and separate two-per-cell limit are unchanged; archiving occurs after H2 and before matrix regeneration or a new preflight. Problem-070's preparation-attempt count stands at 1. The replacement bootstrap journal carries `prior_preparation_failures` referencing the ledger row's `entry_id`, with `supersedes` null. Annex B capture supersession, cross-preflight recovery, and finalization-only recovery are unaffected: this incident is not capture supersession, and `PREFLIGHT_EXPIRED_BEFORE_DONE` does not apply. DONE/accounting continue to require exact bidirectional preparation-ledger closure and reject publication paths under `preparation_failures/` and `superseded/`.

## 12. UPDATED PRE-CAPTURE CHECKLIST

Before any GPU work, in order: (1) Artifact 1 written, immutable, exactly matching the escalated evidence; (2) V2 implemented; all Section 8–9 tests green; full §2 battery, py-compile, shell syntax, `git diff --check` green; (3) pre-commit H1 audit: 74/74 equal, zero unpermitted divergences, Artifact 2 written; (4) H2 committed, tree clean; (5) preparation failure archived with reverified inventory and fsynced ledger row; (6) authority/matrix regenerated at H2; counts exactly 344/122/222, 18 chunks, 0 blocked, 122 budgets, 74 runnable SOL; membership row-id sets identical to H1; (7) certifying H2 audit passes; Artifact 3 written at the mandated path; (8) ordinary shell still fails the ambient tripwire (`SOL_ENV_AMBIENT_CONFLICT`); dedicated wrapper (`77a96b4e…`) passes; `NVCC_PREPEND_FLAGS`/`NVCC_APPEND_FLAGS` absent; `CC`/`CXX`/`CUDAHOSTCXX` snapshot clean; (9) fingerprint probe non-degraded and exactly `0d7c673c…`; (10) fresh preflight: ≤ 900.000 s, ≤ 2 s future skew, exact HEAD/matrix pins, single-preflight identity for the chunk; (11) one-cell SOL bootstrap capture with full stage sealing (fingerprint, component map, compiler block, matrix pin, stage definition, module identity, V2 attestation, accepted-launch evidence, any AC-9 route) and `prior_preparation_failures`; (12) seal, DONE, accounting: exactly 2 accounted / 120 missing / 0 duplicates / integrity true; (13) pinned `gpt-5.5:high` review passes. Only then may any other chunk begin.

## 13. ANSWERS TO Q1–Q16

**Q1.** Option A, strict form, as `SOL_DEF_EQUIV_V2` (Sections 1, 4).

**Q2.** The algorithm is Section 4 in full. Eligible path is exactly `/axes/*/description`; `axes` must be the top-level key with a JSON-object value; `*` is exactly one object-key segment whose value must be a JSON object; only the immediate `description` child is eligible; only missing and null are equivalent (missing normalizes to explicit null per side); `""` is distinct; explicit nonempty descriptions remain type-strict and decoded-value exact; absent/non-object `axes` or a non-object axis value simply disables eligibility (not itself a refusal — ordinary strict comparison governs); nested `/…/axes/*/description` is ineligible; `/inputs/*/description`, `/outputs/*/description`, and top-level `/description` are ineligible, and any divergence there is a stop-and-escalate defect.

**Q3.** New ID `SOL_DEF_EQUIV_V2` (Section 5). No sealed evidence references V1; fixtures update to V2; V1 survives only as a frozen non-certifying reference for Artifact 1 and test P15; verifiers accept only V2 on new SOL evidence.

**Q4.** Section 7's shape is binding: the three top-level field states retained, plus `axes_container_state` and `axis_description_states` keyed by RFC 6901-escaped axis key over the union of both sides' axes, raw-key code-point ordering, closed state vocabulary {`missing`, `null`, `empty_string`, `nonempty_string`, `non_string`, `axis_not_object`}, per-side normalization booleans, deterministic serialization.

**Q5.** Confirmed unchanged: staged-hash computation and real packager bytes. Problem-070's expected staged SHA remains `e7526755…`.

**Q6.** Confirmed: the certified g++-14 runnability/shrink evidence remains reusable; no re-shrink or re-runnability run. The Annex A one-cell live bootstrap requirement stands.

**Q7.** Confirmed: C3 archive protocol, separate two-attempt limit, and after-H2/before-regeneration sequencing are unchanged.

**Q8.** Sections 8 and 9 are the binding test set, including all 31 observed cases, the full real-74 audit, problem-070's exact SHA control, and real `ProblemPackager` integration.

**Q9.** Section 10 is binding: the mandated path holds the certifying per-row/summary artifact (source and staged SHAs, original states, every normalized pointer, zero residual differences, rule ID, verdict), and yes — the blocking discovery is preserved separately and immutably as Artifact 1 before any comparator amendment.

**Q10.** Yes, twice: pre-H2 against the exact 74-member H1 matrix (gate for the commit) and post-regeneration at H2 against the newly sealed membership (certifying), with membership-equality assertion between them.

**Q11.** Codex may retain and amend the uncommitted implementation, subject to: Artifact 1 is generated with the pre-amendment V1 comparator first; rule re-ID to V2; the interrupted DONE/accounting lane is completed; `archive_preparation_failure.py` is created; the previously updated fixture's suite and the full battery are rerun green. Nothing is certified until Section 12 items 1–4 complete.

**Q12.** Confirmed. No archive, matrix regeneration, preflight, bootstrap, or GPU work until V2 is implemented, all required no-GPU tests pass, the 74-definition audit shows zero unpermitted divergences, and H2 is committed with a clean tree — in the Section 6 order.

**Q13.** Confirmed: universe 344; runnable 122; excluded 222; runnable SOL 74; chunks 18; budgets 122; blocked 0; and after the bootstrap seals, exactly 2 accounted / 120 missing.

**Q14.** Section 7 is the complete list: the attestation block and its schema, the three new typed refusal codes, and the DONE/accounting checks (rule-ID exactness, state recomputation, pointer eligibility, source-SHA-vs-matrix-pin equality, staged-SHA equality, preparation-ledger closure). No other schema or code surface changes are authorized.

**Q15.** Confirmed: AC-9 is untouched; capture evidence remains compression-first whenever budget routing occurs.

**Q16.** Confirmed: the review path remains pinned `gpt-5.5:high`; on any managed-harness safety check or capacity wait, Codex waits and retries the identical pinned path; no model switch or downgrade.

## 14. FORBIDDEN SHORTCUTS

All Annex A4 and Annex C5 prohibitions remain in force verbatim, and additionally: do not normalize `""` at axis descriptions; do not extend equivalence to `/inputs/*/description`, `/outputs/*/description`, top-level `/description`, or any nested/displaced `description` position — divergence there is a stop-and-escalate defect; do not apply the §4.3 clause through non-object `axes` or non-object axis values; do not emit `SOL_DEF_EQUIV_V1` in any new attestation or blend V1/V2 semantics under one ID; do not write, edit, or backdate the blocking artifact after the fact, and do not generate it with the amended comparator; do not truncate pointer lists in audit artifacts; do not treat the pre-commit H1 audit as certifying; do not regenerate the matrix, mint a preflight, archive, or touch the GPU before H2 exists with a clean tree; do not edit pinned definitions, overwrite staged definitions, or mutate staged evidence; do not reuse any artifact of the archived attempt; do not exclude or type-dispose the 31 rows; do not reclassify the preparation failure as capture supersession; do not capture from a dirty tree, under fingerprint inequality, degraded probe status, or a stale matrix; and do not switch, downgrade, or substitute the pinned review model for any reason, including harness safety checks — wait and retry the identical path.

Every clause of Annex A, Annex B, and Annex C not named in Section 3 as amended remains binding. This adjudication is final for Round 28; any divergence discovered outside the V2 closed classes, at any position, is a defect requiring immediate stop and a new escalation to Fable-5.
