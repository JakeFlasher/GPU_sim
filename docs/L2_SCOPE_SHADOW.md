# Typed L2 client-scope shadow

The L2 event path now records a passive client-scope decomposition. Static
SASS/access metadata tags requests as ordinary LDG, constant LDG, LTC128B,
LDGSTS cache/bypass, local, store, atomic, reduction, or unknown. At the
transactional L2 commit boundary, each read request is partitioned into
resident, pending-merge, and fill-miss masks. Write-class requests report their
request and resident-sector observations; both pending-merge and fill-miss
arms are structurally zero because the event cache class is `WriteAtomic`.

The observer is strictly diagnostic:

- `L2Counters`, `DramStats`, physical tags/sector masks, replacement, event
  timestamps, response frontiers, and IPC timing are untouched;
- TMA sidecar payloads are not double-counted as timed L2 requests;
- no NCU value, workload shape, or per-workload table selects a rule;
- LTC128B potential sibling volume is reported only when a partial request
  starts a fill, never inserted into physical cache or DRAM accounting.

The arrays are serialized under `counters.l2_by_source` as
`client_scope_*`; index order is documented by `L2ClientScope` in
`sim/sm/src/issue.rs`. They are not scored predictions. The final full replay
artifacts are `full_replay_runs/l2_scope2_sm120_enabled/` and
`full_replay_runs/l2_scope2_sm86/`. They passed the required status profiles,
reuse invariants, and twelve-worker full-only runs. The refined run also
assigns write, atomic, reduction, and async bypass requests at their emission
seams instead of leaving those classes as `Unknown`; ordinary bypasses remain
unknown when the trace has no stronger static qualifier. The combined
comparison is `FULL_NUMERIC_COMPARISON_L2_SHADOW2.json`; its only intended
SM120 differences are the already-approved async L1 scope projection, while
SM86 has zero common numeric differences.

The read-scope masks conserve each request sector (`resident | pending | fill`)
at the event preview. Scope is request-granular: when same-cycle L1 sectors
from different static clients fold into one physical fill, the lead request's
scope owns the folded transaction. This is a deterministic diagnostic
attribution, not a claim of exact per-sector hardware client provenance.

Gemini rejected active LTC128B sibling promotion: current capture request masks
already contain full four-sector lines, SM120 cache geometry is unresolved, and
forcing a physical four-sector fill would inflate miss/DRAM counts. Any future
projection must first show conserved, sign-consistent scope identity on held-out
and full corpora without changing physical state.
