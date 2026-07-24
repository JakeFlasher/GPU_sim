# Historical SimBench schemas

This directory contains retired contracts that remain available only for
hash-pinned validation of historical evidence. They are not active producer
contracts and must not be extended in place.

`duration_admitted.v1.schema.json` is the retired CUPTI dual-source wall-clock
duration-label contract (`schema_version` 1). Migration alias attestation:
`workloads/simbench_bw/schemas/duration_admitted.schema.json` →
`workloads/simbench_bw/schemas/historical/duration_admitted.v1.schema.json`,
SHA-256 `5a555d714cab25f1203f561e8027d589ea4329aabe6826337ae68cfbf2b3f62f`.
NCU-only timing is now authoritative, and timing/duration labels travel
exclusively as `unit_tagged_metrics` under the three-part v5 label authority.

The schema's `$id` deliberately retains its historical value so the migrated
file stays byte-identical. Its SHA-256 is pinned in `MANIFEST.sha256`. Retained
duration evidence validates only against this file through declared-version
dispatch; missing or unknown versions fail closed.

`cupti_activity_sidecar.v1.schema.json` is the frozen per-launch CUPTI Activity
sidecar contract retained for validation and offline replay of historical
evidence. Migration alias attestation:
`workloads/simbench_bw/schemas/cupti_activity_sidecar.schema.json` →
`workloads/simbench_bw/schemas/historical/cupti_activity_sidecar.v1.schema.json`,
SHA-256 `836c0db4f6f5a2a582a9b334be02c829b40a094f709259d8008f05f200faaeb4`.
The migrated file remains byte-identical, including its historical `$id`; it
does not authorize a living CUPTI producer. Retained sidecars validate only
through declared-version dispatch, and missing or unknown versions fail closed.

`sol_workload_capture_chunk_done.v2.schema.json` is the frozen additive SOL
workload-row DONE contract (`schema_version` 2). It retains the historical `$id`
`simbench_bw/schemas/sol_workload_capture_chunk_done.schema.json` and exact
SHA-256 `5e9bbb94107541fa522e9070972e81916a7be984802493f414d0951a705f0f4c`.
Retained SOL DONE evidence declaring integer version 2 dispatches only to this
file. Missing, boolean, non-integer, or unknown versions fail closed. This
historical copy does not replace or authorize the active producer schema.
