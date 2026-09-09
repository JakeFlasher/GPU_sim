# `tools/sol_baselines_scaffold/`

Scaffold + pack + validate toolchain for SOL-ExecBench baseline kernels.

Kernel bodies are hand-authored on top of exemplar trees copied from
`tools/sol_execbench/examples/`.

| File | Role |
|------|------|
| `scaffold.py` | Copy an exemplar tree from `tools/sol_execbench/examples/{cutlass,cuda_cpp}/<name>/` into `baselines/<category>/problem_NNN_<slug>/<language>/`; convert the entry to destination-passing style; rename kernel symbols; emit a Pydantic-valid `solution.json` (`sources[]` populated from disk at scaffold time, deterministic order). |
| `pack.py` | Re-pack `solution.json.sources[].content` from the editable files on disk. Disk wins on staleness. Deterministic order (entry source first, remaining lex by `path`); idempotent on re-run; fail-closed when a referenced source is missing. |
| `validate.py` + focused contract tests | Two-layer validation: live `Solution` / `Definition` Pydantic parsing plus project lint for destination-passing style, supported languages, and forbidden surfaces. Exits 0 on pass / 2 on failure. |
| `generate.py` | `--manifest-only` mode plus slug-resolution and subset-mapping primitives consumed by `scaffold.py` and `tools/sol_baselines_harness/`. |

## Prerequisites

Populate the benchmark data first (one-time):

```sh
./tools/sol_execbench/scripts/download_data.sh
```

Subsets land at `tools/sol_execbench/data/benchmark/{L1,L2,Quant,FlashInfer-Bench}/<NNN_name>/`.

## Slug → benchmark-name resolution

Internal slug format is `problem_NNN_<rest>` (full markdown stem from
`references/NVIDIA_Manuals/solbench_problems/`). The downloader writes
each problem to `tools/sol_execbench/data/benchmark/<subset>/NNN_<rest>/`.

Resolution strips `^problem_` (keeping the digit prefix):

| Internal slug | Subset | Benchmark dir name |
|---------------|--------|--------------------|
| `problem_001_fused_add_rmsnorm_h2048` | `FlashInfer-Bench` (FIB) | `001_fused_add_rmsnorm_h2048` |
| `problem_069_rms_norm` | `L1` | `069_rms_norm` |
| `problem_004_gemm_n128_k2048` | `FlashInfer-Bench` (FIB) | `004_gemm_n128_k2048` |

If a candidate name does not match a real benchmark directory, the
scaffolder fails closed and references `./tools/sol_execbench/scripts/download_data.sh`.

## Subset mapping

Internal category → downloader subset name:

| Internal | Downloader |
|---|---|
| `L1` | `L1` |
| `L2` | `L2` |
| `Quant` | `Quant` |
| `FIB` | `FlashInfer-Bench` |

## `tbd_*` pattern names

`tools/sol_baselines_scaffold/sol_pattern_mapping.md` may contain
`tbd_<name>` pattern names. The manifest reader strips that prefix at parse
time; the mapping file is consumed read-only.

## Manifest mode

Without `--slug` + `--language`, `generate.py --manifest-only` prints a
JSON manifest of every row in
`tools/sol_baselines_scaffold/sol_pattern_mapping.md`, including whether each
row's dossier / spec / pattern card / `definition.json` is
present.

```sh
python tools/sol_baselines_scaffold/generate.py --manifest-only
```

## Submission language policy

`--language` argparse choices are restricted to `cutlass` and `cuda_cpp`
(per CLAUDE.md Rule 5). Vendor libraries (cuBLAS, cuDNN, CUB) are linked
via `compile_options.ld_flags` on `["cuda_cpp"]` sources — they are not
valid `spec.languages` values for our submission surface even though the
live Pydantic `SupportedLanguages` enum accepts them.

## Fail-closed semantics

The scaffolder writes nothing on any of these conditions:

- `--slug` not in `tools/sol_baselines_scaffold/sol_pattern_mapping.md`.
- Benchmark `definition.json` missing under the resolved
  `tools/sol_execbench/data/benchmark/<subset>/<benchmark_name>/` path.
- `--exemplar` does not name an existing tree under
  `tools/sol_execbench/examples/{cutlass,cuda_cpp}/`.
- `--language` is anything other than `cutlass` / `cuda_cpp`.

Errors print to stderr with exit code 2.
