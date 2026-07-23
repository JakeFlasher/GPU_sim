# Supplemental SOL workload-row capture integration ledger

Status: **IMPLEMENTED ADDITIVE CONTRACT; LIVE CHECKPOINT PENDING**.
Supplemental validation and row binding live behind the dedicated workload-row
campaign controller; the default campaign path continues to consume only the
canonical 344-cell matrix. This ledger is the boundary between the two
authorities and must remain true during live execution and publication.

## Authority and claim boundary

- `build_sol_workload_capture_matrix.py` reads and SHA-pins the canonical
  `data/local_labels/campaign/sm86_reference_capture_matrix.json` and its
  authority as the unchanged base universe. Supplemental SOL selection comes
  from the complete current 201-cell runnability authority and complete v3
  shrink authority, never from the canonical matrix's historical runnable set.
  It does not write any input authority.
- The supplemental artifact is additive. It must never change the canonical
  universe, runnable/excluded counts, authority cells, existing chunks, or DONE
  accounting.
- The supplemental matrix's top-level `source_commit` is the clean live
  producer HEAD shared by runnability, shrink, the plan owner, the live coverage
  inventory, and campaign execution. Its nested
  `canonical_matrix.source_commit` remains historical canonical-matrix
  provenance and is never substituted for the live execution commit.
- Native rows use cohort `native`, claim scope `native_workload_row`, and
  `native_predictive=true`.
- A PASSED row from a non-runnable problem is admitted only when the problem
  still has a complete expected/returned UUID diagnostic bijection. It uses
  cohort `partial_native_row`, claim scope
  `partial_native_workload_row_only`, `native_shape=true`,
  `native_predictive=false`, and `problem_level_runnable=false`. Missing
  diagnostics never qualify as partial evidence.
- Reduced rows use cohort `reduced_proxy`, claim scope
  `reduced_proxy_bundle_only`, and `native_predictive=false`. They may not
  be reported as native-shape coverage or pooled with native rows without an
  explicitly stratified publication analysis.
- Every row is keyed exactly by `(slug, source_workload_uuid, variant_id)`.
  Full native rows use `variant_id=native`; partial-native and reduced rows use
  deterministic lowercase 64-hex identities.
- Every variant and row seals `launch_scope=all_in_profiler_roi` and the closed
  launch-completeness contract. The runner must inventory the deterministic
  in-ROI launch set before capture and admit a row only when dynamic trace, NCU
  label, and control evidence are complete for every inventory member.
  Ordinal-zero-only auxiliary evidence is typed incomplete and cannot produce
  an admitted row.
- `source_workload_index`, `source_axes`, and `source_row_sha256` always bind
  the original native JSONL row. `capture_workload_index`, `capture_axes`, and
  `capture_row_sha256` bind the effective canonical workload row the adapter
  selects. They are identical for native rows and may also remain identical
  for a definition-only reduced proxy. In that case the bundle identity must
  instead bind a distinct, immutable external definition plus its exact
  native-to-reduced definition delta. A reduced cohort may compact native
  indices (for example native indices 0 and 2 become cohort indices 0 and 1),
  so a runner must never substitute one index for the other.
- Every reduced workload and optional reduced definition is read from the
  same sealed shrink-owner namespace, retained as a SHA-256-bound immutable
  campaign input, and copied once more into the private per-attempt request
  tree. The adapter receives only those per-attempt paths. Native definitions
  continue to resolve from the repository benchmark tree unchanged.
- Reduced variants are accepted only from top-level and cell-level v3
  provenance. The cell must be `reduced_representative`,
  `claim_scope=reduced_shape_only`, non-native-predictive, runnable, and passed.
  Effective cohort rows and `proxy_planned` row plans must form an exact
  UUID/SHA bijection; each plan must be runnable/passed and change at least one
  member of the sealed `(definition bytes, workload-row bytes)` bundle. Axis
  reductions carry the exact axes delta; definition reductions carry the exact
  native-to-reduced JSON-pointer delta. Per-variant files are execution
  intermediates and are not consumer authorities.

## Why the default canonical runner path must refuse this artifact

`tools/local_labels/sm86_capture_chunk.py` assumes all of the following:

1. The input has canonical `cells` and canonical chunks whose members are only
   `{roster_source, slug}`.
2. `select_chunk()` resolves one canonical authority cell per member.
3. SOL validation rederives the canonical invocation whose
   `capture_target.source_workload_index` is zero.
4. DONE accounting and campaign counts are expressed in canonical cells, not
   workload-row keys.

The supplemental schema instead has `variants`, row keys, and row chunks.
Passing it to the existing runner must remain an input/schema refusal. Silent
translation, treating a workload row as a canonical cell, or changing the
canonical matrix to fit this schema is forbidden.

## Supplemental runner integration contract

| ID | Surface | Required implementation | Required rejection |
|---|---|---|---|
| SWR-01 | New entry point | Use `sol_workload_capture_campaign.py`; the default canonical runner remains canonical-only and the low-level workload chunk CLI is not a production entry point. | Supplemental input on the default path; canonical input on the supplemental path; direct chunk execution outside the campaign lock/receipt boundary. |
| SWR-02 | Input pins | Verify the supplemental matrix seal, canonical-matrix SHA/seal, canonical-authority SHA, schema, and clean source paths before creating output. | Any pin, seal, path, or schema mismatch before mutation. |
| SWR-03 | Variant lookup | Resolve exactly one `variant_contract_sha256`; recompute that digest after removing its digest field; rederive the copied adapter contract and canonical invocation bindings. | Missing, duplicate, stale, or cross-slug variant reference. |
| SWR-04 | Source bundle | Re-read the original native definition/JSONL and effective capture definition/JSONL using producer/adapter ordering. Require the exact definition hashes/delta plus source index/UUID/axes/raw SHA and capture index/UUID/axes/raw SHA, each file SHA, and row count. | Any definition, index, UUID, axes, raw-byte, file, or v3 plan-bijection drift; an unchanged workload row is allowed only with a distinct exact external-definition delta. |
| SWR-05 | Adapter request | Copy the variant's base capture target, replace only its `source_workload_index` with the row's `capture_workload_index`, retain `staged_workload_index=0`, and pass the capture index to the existing persistent-stage adapter request. | Use of native `source_workload_index` for a compacted proxy cohort, any other target-field mutation, or selection evidence with a different capture index/UUID/row SHA. |
| SWR-06 | Chunking | Consume the declared row keys in order and verify `membership_sha256`; never mix `native`, `partial_native_row`, or `reduced_proxy` cohorts in one chunk. | Missing/extra/duplicate row, out-of-order membership, or cohort mismatch. |
| SWR-07 | Output namespace | Use a new supplemental campaign root and row-keyed journal path. Existing canonical journals and DONE files are read-only. | Output aliasing any canonical campaign, journal, cell, or final directory. |
| SWR-08 | Result identity | Seal the triple key, variant digest, source-row bindings, adapter selection evidence, trace hashes, and cohort/claim scope into every result. | Result identity that omits or contradicts any binding. |
| SWR-09 | Accounting | Add a separate supplemental DONE schema with counts by cohort, variants, rows, admitted/typed outcomes, and exact chunk closure. | Folding supplemental row counts into canonical 344-cell accounting. |
| SWR-10 | Publication export | Export full-native, partial-native-row, and reduced-proxy tables separately and preserve per-row axes, UUID, exact launch inventory, and completeness provenance. | A summary that promotes partial/reduced evidence to full native, loses row identity, or admits incomplete launch layers. |

## Supplemental runner acceptance tests

The runner remains accepted only while tests prove: positive selection of
a nonzero native `source_workload_index`; a compacted reduced row whose
`capture_workload_index` differs; exact staged capture-row byte identity; every SWR-02
through SWR-10 refusal before mutation; three-way cohort chunk separation; crash
resume at the row key; duplicate-key rejection; no writes beneath an existing
canonical campaign; and unchanged hashes for the canonical 344-cell matrix and
authority before and after a synthetic supplemental run.

Production additionally requires the controller's canonical host-wide GPU
lock, a pre-attempt `--max-new-chunks` frontier for bounded pilots, typed
external storage-stop receipts, relocation-safe offline checkpoint
reconciliation, and source-preserving archive verification. A bounded or
capacity-limited checkpoint remains explicitly incomplete and may not be
described as comprehensive coverage.

The runnability producer invokes the host's exact pinned
`/home/jakeshea/.local/bin/uv` (`uv 0.10.7`, SHA-256
`3b9d43bace955a409eb6f012b0932a4935534bdb54b85b9fd696cabbb15979a0`)
directly when proving the installed SOL distribution closure. It does not
depend on ambient `PATH`, and it refuses a byte or version mismatch before
`uv sync --check` can execute. Provision the detached producer environment
with `uv sync --frozen --offline --no-dev --project tools/sol_execbench`; a
development dependency left in that environment is an intentional closure
mismatch.
