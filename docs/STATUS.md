# Project status

Updated 2026-07-18. This file reports implemented and admitted state, not
aspirational plan completion.

## Simulator

| Area | State |
|---|---|
| Rust event queue | Implemented and deterministic |
| NVBit protobuf reader | Implemented with committed Prost bindings |
| Static control-bit join | Implemented, fail-closed |
| SM replay | Implemented as a library over traced placement |
| L1/shared-L2/DRAM path | Implemented as a library |
| NCU-shaped emission | Seven secondary descriptors implemented |
| Primary cycles output | Required but unimplemented |
| Production CLI | Missing |
| Evaluation join/reporter | Missing |
| Performance benchmark/gate | Missing |
| sm_120 validation | Deferred until final sm_86 M2 passes and the operator explicitly approves migration |

`KernelReplayResult.max_cycles` is a directional diagnostic. It must not be
published or emitted as `gpc__cycles_elapsed.max` until admitted silicon
evidence establishes single-launch mapping and multi-launch composition.

The current Rust workspace gate reports 201 passing tests and three explicitly
ignored machine-local evidence tests. The default Python boundary is
self-contained and excludes explicit `campaign_evidence` and
`hardware_evidence` markers. A green default suite does not claim that external
GPU evidence was exercised.

## sm_86 capture campaign

Current milestone state:

- M0 foundation: complete;
- M1 pre-transition anchor: complete;
- M2 trace-contract transition: complete;
- M3 canary: complete;
- M4 reference capture: in progress;
- M5 calibration: not started;
- M6 repeatability: blocked on M4;
- M7 freeze: blocked on M4, M5, and M6.

The canonical authority contains 344 cells: 122 runnable and 222 excluded.
Current defensible canonical credit is 1/122 runnable.

ADR 0001 (`CONTINUE_PRESERVE_PINNED`) authorizes fresh Task7/H3 continuation
outside RLCR under the current campaign, provenance, and hardware gates. The
historical Task7/H3 capture-defect archive and the stopped 49-row campaign
revision are frozen, zero-credit, and reuse-forbidden. Neither may satisfy a
current gate.

The interrupted 53/201 SOL runnability attempt is preserved in a separate
non-authoritative archive; canonical historical runnability v1 is restored. Its
immutable root predates the closed runnability schema and retains 15 producer
provenance fields outside that schema, so verification checks the source-commit,
hash-pinned schema projection plus rederived cell/status counts and does not call
the partial schema-valid. Its whole-dirty-tree diff digest is retained byte-exact
but cannot be reconstructed because unrelated dirty bytes were not archived.
Supplemental launch-set v3 is a separate cohort and remains live-disabled until
a pinned canary passes. It cannot change canonical M4/M7 accounting.

## Metric scope

ADR 0002 (`CYCLES_ONLY_BINDING`) fixes the metric contract:

- sole publication headline: `exec_cycles_gpc_max` /
  `gpc__cycles_elapsed.max`, currently `required_unimplemented`;
- cycles MAPE ≤10% meets the engineering target, >10% and ≤15% is floor-only,
  and >15% is publication failure;
- seven implemented secondary descriptors are reported but publication-ungated;
- capture ceiling: all 61 entries in
  `tools/local_labels/metric_registry.json`.

Keeping the 61-metric capture superset does not claim 61 simulator outputs and
does not create a fallback publication headline.

## Cleanup

Completed or in progress before this documentation stage:

- CPU/offline Python collection separated from optional SOL/FlashInfer tests;
- evidence roots and Humanize sources hash-manifested;
- durable lessons and SOL pattern mapping moved to tracked paths;
- Round-28 evidence migration and consumer refit in progress;
- historical path aliases introduced without rewriting ledger rows.

Still required before local `.humanize/` deletion:

- migrate and verify the complete 22-file set;
- remove every live tracked `.humanize/` dependency;
- pass a fresh-worktree default suite with `.humanize/` absent;
- finish the explicit hardware/campaign evidence gates;
- keep accepted ADR 0001/0002 dispositions reflected in every consumer;
- recheck byte identity against the protected original-repository archive.

Possibly unique evidence is not deleted in this pass: `data/annex/`,
`data/local_labels/campaign/`, the external capture archive, the preparation-
failure archive, and the 15 GB SOL/FlashInfer store until replication is
proven.

## Immediate technical blockers

- `evidence/path_aliases.v1.json` names
  `docs/decisions/0005-evidence-retention-and-path-aliases.md`, but that ADR is
  absent and ADR number 0005 is already assigned to a different capture-envelope
  decision. The v1 relocation chain is therefore not authority-closed; do not
  invent or renumber an ADR, and require an explicitly accepted versioned repair.
- `evidence/manifests/external_capture.sha256` is the frozen 1,135-file
  2026-07-11 point-in-time inventory, not a current whole-root closure for the
  subsequently expanded external capture tree. Later sealed namespaces retain
  their own manifests, ledgers, and owner verifiers. Do not overwrite the
  historical snapshot or infer whole-root coverage from it.
- The committed canonical capture matrix remains the frozen 344-cell authority
  throughout cleanup. Producer repair does not authorize rewriting it. Any
  successor matrix must be generated to a fresh path in an explicitly authorized
  post-cleanup campaign transition and cannot replace canonical v1 without its
  owning provenance and campaign gates.
- Production M5, M6, and M7 entry points and a distinct M7 completion gate are
  still missing.
- The simulator still needs production admission/identity, corrected memory and
  SM semantics, a runner/CLI, admitted cycle semantics, evaluation, and the
  pinned Accel-Sim speed comparison.

Archive-integrity, offline/frozen-hardware, live-campaign, M7 completion, and
final M2 are independent gates; passing or authorizing one satisfies none of
the others.
