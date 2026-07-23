# `scripts/baselines_scaffold/`

Scaffold + pack + validate toolchain for SOL-ExecBench baseline kernels.
Renamed from `scripts/baselines_codegen/` via `git mv` (history preserved).

The module exposes three independent concerns plus a preserved
manifest/metadata builder. The original per-pattern kernel-template
registry (`templates/`) was deliberately render-and-burned during the
first pilot (FIB rmsnorm) and no longer exists in tree; kernel bodies
are now hand-authored on top of an exemplar tree copied from
`sol-execbench/examples/`.

| File | Role |
|------|------|
| `scaffold.py` | Copy an exemplar tree from `sol-execbench/examples/{cutlass,cuda_cpp}/<name>/` into `baselines/<category>/problem_NNN_<slug>/<language>/`; convert the entry to destination-passing style; rename kernel symbols; emit a Pydantic-valid `solution.json` (`sources[]` populated from disk at scaffold time, deterministic order). |
| `pack.py` | Re-pack `solution.json.sources[].content` from the editable files on disk. Disk wins on staleness. Deterministic order (entry source first, remaining lex by `path`); idempotent on re-run; fail-closed when a referenced source is missing. |
| `validate.py` + `test_contract.py` | Two-layer gate. (a) Live `Solution` / `Definition` Pydantic parse. (b) Project lint that the live schema permits: Rule-5 vendor-as-language rejection, destination-passing-style argument-order check against `definition.json`, Rule-1/2/6/7 forbidden-surface scan (NCU, `--lock-clocks`, user CUDA streams, `torch.jit.fork`, `data_ptr`-keyed cache, monkey-patched `Event.elapsed_time`, CUDA Graphs without a per-problem note). Exits 0 on pass / 2 on any failure. |
| `generate.py` | `--manifest-only` mode + the slug-resolution and subset-mapping primitives consumed by `scaffold.py` and by `baselines_harness/`. |

## Prerequisites

Populate the benchmark data first (one-time):

```sh
./sol-execbench/scripts/download_data.sh
```

Subsets land at `sol-execbench/data/benchmark/{L1,L2,Quant,FlashInfer-Bench}/<NNN_name>/`.

## Slug → benchmark-name resolution

Internal slug format is `problem_NNN_<rest>` (full markdown stem from
`references/NVIDIA_Manuals/solbench_problems/`). The downloader writes
each problem to `sol-execbench/data/benchmark/<subset>/NNN_<rest>/`.

Resolution strips `^problem_` (keeping the digit prefix):

| Internal slug | Subset | Benchmark dir name |
|---------------|--------|--------------------|
| `problem_001_fused_add_rmsnorm_h2048` | `FlashInfer-Bench` (FIB) | `001_fused_add_rmsnorm_h2048` |
| `problem_069_rms_norm` | `L1` | `069_rms_norm` |
| `problem_004_gemm_n128_k2048` | `FlashInfer-Bench` (FIB) | `004_gemm_n128_k2048` |

If a candidate name does not match a real benchmark directory, the
scaffolder fails closed and references `./sol-execbench/scripts/download_data.sh`.

## Subset mapping

Internal category → downloader subset name:

| Internal | Downloader |
|---|---|
| `L1` | `L1` |
| `L2` | `L2` |
| `Quant` | `Quant` |
| `FIB` | `FlashInfer-Bench` |

## `tbd_*` pattern names

`tools/sol_baselines_scaffold/sol_pattern_mapping.md` carries some legacy
`tbd_<name>` pattern names (artefacts of the KB build's pattern-naming evolution). The
manifest reader strips the prefix at parse time. The mapping file is
NOT edited by this tool — it is consumed read-only.

## Manifest mode

Without `--slug` + `--language`, `generate.py --manifest-only` prints a
JSON manifest of every row in
`tools/sol_baselines_scaffold/sol_pattern_mapping.md`, including whether each
row's dossier / spec / pattern card / `definition.json` is
present.

```sh
python scripts/baselines_scaffold/generate.py --manifest-only
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
  `sol-execbench/data/benchmark/<subset>/<benchmark_name>/` path.
- `--exemplar` does not name an existing tree under
  `sol-execbench/examples/{cutlass,cuda_cpp}/`.
- `--language` is anything other than `cutlass` / `cuda_cpp`.

Errors print to stderr with exit code 2.
