# Evidence manifests

These manifests record immutable or potentially unique evidence before the
repository cleanup. Paths inside each tree manifest are relative to the root
named below:

| Manifest | Verification root |
|---|---|
| `humanize.sha256` | protected original-repository `.humanize/` archive |
| `data_annex.sha256` | `data/annex/` |
| `campaign.sha256` | `data/local_labels/campaign/` |
| `sol_execbench_data.sha256` | `tools/sol_execbench/data/` |
| `references.sha256` | `references/` |
| `toolchain.sha256` | `.toolchain/` |
| `external_capture.sha256` | `/home/jakeshea/sm86_reference_capture_r27/` |

`external_capture.sha256` is specifically the 1,135-file point-in-time snapshot
recorded on 2026-07-11. It is not an active inventory of the subsequently
expanded external root, and it must not be regenerated over or used to claim
whole-root closure. Later sealed namespaces are validated by their own immutable
manifests, ledgers, and owner verifiers.

`toolchain.symlinks` records symlink paths and targets separately. The
toolchain is reproducible, but it remains provisionally active until its
pkg-config relocation has been verified.

`humanize_migration_baseline.sha256` records the 22 canonical destinations and
their byte identities at migration time. The two living destinations
(`docs/lessons.md` and the SOL pattern mapping) may evolve after migration;
their immutable baseline remains this manifest plus Git history.

The permanent frozen Round-28 subset is recorded in
`../MANIFEST.sha256`. Evidence is never accepted by hash alone: Stage 2 adds
tracked-path, clean-tree, and versioned path-alias verification.

`tools/dev/build_evidence_manifests.sh` is the historical reconstruction recipe
that produced these snapshots. It is not an active refresh command: running its
legacy implementation would replace tracked point-in-time authorities with a
new tree view. Use scope-specific verify-only commands instead; any successor
snapshot requires a fresh versioned path and explicit authority.

The Humanize source defaults to
`/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/.humanize` and
may be overridden with `HUMANIZE_ARCHIVE_ROOT`. The local ignored copy is not a
runtime dependency of the builder.

## Frozen M1 trace payload authority

`../path_aliases.v2.json` records the exact 17-row mapping from the two
historical capture prefixes to their repository-relative counterparts.
`m1_trace_payloads.sha256` pins the complete two-prefix current tree: 5,033
payloads, including the 30 portable timing-shape and measurement-manifest
siblings outside the fifteen `trace_timing_shape/` directories. The generator
first verifies the pinned source manifest and exact historical/current file-set
and byte equality; it does not route historical payloads through forward trace
admission.

Verify both tracked authorities without writing:

```bash
.venv/bin/python tools/nvbit_tracer/build_m1_payload_authority.py --verify-only
```

For precommit and postcommit comparisons, emit both complete byte strings to
explicit paths outside every Git worktree and compare them with the staged or
committed blobs. Canonical creation is permitted only once, while both tracked
targets are absent; the generator refuses partial creation or replacement.
