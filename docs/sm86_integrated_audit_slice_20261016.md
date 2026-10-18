# SM86 integrated non-regression audit — 2026-10-16

H52 adversarially compared the H48 constant-form full replay against H38 on
all 12,392 launch keys. There are 12,392 keys in each artifact, no missing
keys, and zero old-field differences after removing only the H48 form observer
rows and nondeterministic `sidecar_join_wall_ns`.

The compressed H42 and H48 metric-report inventories each contain
`metric_comparisons.jsonl.zst`, `summary.json`, and `summary.md`; both retain
125 scope warnings and 67 scope-mismatch exclusions. Workspace `cargo check
--workspace --all-features`, Python compilation, and `git diff --check` pass.

Conclusion: H37/H38 pressure ledgers, H48 form observers, and scope-aware
post-replay reporting compose without altering old cache/DRAM/IPC/frontier
behavior. Scope exclusions remain explicitly unresolved and are not closure
claims.

The retained H48/H42 summary headings predate H63's raw-versus-secondary label
correction. Use
`../sm86_replay_results/metric_report_h63_scope_label_audit_20261016_v1/`
for the corrected machine-readable scope view; the earlier directories remain
historical replay evidence.
