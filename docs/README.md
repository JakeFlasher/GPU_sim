# Project documentation index

This index separates current operating authority from historical records. It is
not an evidence manifest and does not change any schema, alias, or campaign
accounting.

## Authority precedence

1. **Evidence identity:** `evidence/MANIFEST.sha256`, versioned
   `evidence/path_aliases.v*.json`, frozen ledgers, and the owning schema/hash
   contracts govern byte identity and relocation.
2. **Project policy:** root [`CLAUDE.md`](../CLAUDE.md) governs the simulator,
   evidence, architecture, escalation, and privilege rules.
3. **Consultation protocol:** root [`AGENTS.md`](../AGENTS.md) governs only the
   separately scoped Fable-5 consultation and evidence-return protocol.
4. **Current project state:** [`STATUS.md`](STATUS.md) reports implemented and
   admitted state; it is not an aspirational plan.
5. **Current design and inventory:** [`ARCHITECTURE.md`](ARCHITECTURE.md),
   [`COMPONENTS.md`](COMPONENTS.md), and [`runbook_host.md`](runbook_host.md)
   describe the simulator, ownership, deletion classifications, and host
   operations.
6. **Accepted decisions:** [`decisions/README.md`](decisions/README.md) and
   ADRs 0001–0006 override conflicting older plans or prose.
7. **Execution plans:** named plans under `plans/` describe work and remain
   subordinate to policy, status, evidence contracts, and accepted ADRs.

## Current entry points

- [Repository overview](../README.md)
- [Architecture and dataflow](ARCHITECTURE.md)
- [Component/deletion inventory](COMPONENTS.md)
- [Project status and gates](STATUS.md)
- [Host and toolchain runbook](runbook_host.md)
- [Testing boundaries](testing.md)
- [Knowledge routing](knowledge-routing.md)
- [Metric registry](metric-registry.md)
- [Accepted decisions](decisions/README.md)
- [Evidence manifest guide](../evidence/manifests/README.md)

## Historical root records

`../ARCHITECTURE.md`, `../HANDOFF.md`, `../MANIFEST.md`,
`../MERGE_PROVENANCE.md`, and `../OPEN_RESEARCH_CHALLENGES.md` are retained
historical records. Their bodies may describe superseded predictor, non-root,
DVFS, or project-state claims. They are not operational instructions or
current authority. Preserve their paths and Git anchors unless a separately
verified, hash-bound archive/redirect migration closes every inbound reference.

The corpus contains **353 non-index content cards** plus two physical index/router
files (`.claude/knowledge/INDEX.md` and `patents/INDEX.md`); `_template.md` is
excluded. The generated manifest therefore has **355 rows**. Generated manifests
are derived and are not policy authority.
