# preflight.sh mock fixtures

Each subdirectory is a self-contained `--mock-input` fixture for
`tools/cloud/preflight.sh`. The naming convention encodes the
behavior the fixture targets:

- `cupti_smoke_<status>/` — exercises AC-1's CUPTI Activity runtime
  smoke probe at one of the seven status values:
  `available`, `compile_failed`, `link_failed`, `runtime_failed`,
  `insufficient_privilege`, `cc_unsupported`, `mock_input_missing`.
  All other capability fields are set to passing defaults so the
  fixture isolates the CUPTI runtime status. The
  `cupti_smoke_mock_input_missing/` directory deliberately omits
  `cupti_runtime_probe.txt` (its absence is what triggers
  `mock_input_missing`).
- `deployment_mode_cc_on/` — exercises AC-2's
  `ConfidentialComputingNotSupportedForCollection` rejection by
  setting `cc_mode_probe.txt=on`.
- `deployment_mode_mig_enabled/` — exercises AC-2's
  `MigSliceNotValidForCollectionEvidence` rejection by setting
  `nvidia_smi.csv` mig_mode=Enabled.

## Per-fixture file convention

| File | Purpose |
|---|---|
| `nvidia_smi.csv` | single-line CSV in the `name,uuid,driver_version,compute_cap,clocks.max.sm,clocks.max.mem,clocks.current.sm,mig.mode.current,compute_mode` shape `preflight.sh` reads. |
| `nvcc_version.txt` | one-line CUDA toolkit version (e.g. `13.2`). |
| `nvbit_build_hash.txt` | 64-character lowercase hex string. |
| `cluster_attr_probe.txt` | verdict for the cluster-attribute probe (typically `available`). |
| `cupti_activity_probe.txt` | legacy CUPTI presence probe verdict (typically `available`). |
| `cupti_runtime_probe.txt` | AC-1 CUPTI Activity runtime smoke probe verdict. **Absent** in `cupti_smoke_mock_input_missing/`. |
| `cc_mode_probe.txt` | AC-2 Confidential Computing mode (`off`/`devtools`/`on`/`unsupported_query`/`unknown`). |
| `nvml_capability_matrix_probe.txt` | AC-2 NVML capability tier (`full`/`partial_power_only`/`partial_temp_only`/`denied`). |
| `nvbit_injection_probe.txt` | NVBit injection probe verdict (typically `ok` so the NVBit hard-exit does not fire). |
| `permissions.json` | minimal permissions dict. |

## Adding a fixture

Copy any existing `cupti_smoke_<status>/` directory and change only
the field(s) you want to vary. Fixtures are exercised by
`tools/cloud/test_preflight_probes.sh` (smoke tests around real
preflight invocation) and by `tools/cloud/test_preflight_manifest.py`
(unit tests of the validator against synthetic manifests).
