"""Perturbation harness build orchestrator and operator entry point.

Generates perturbation source files from
`workloads/simbench_bw/l1/dram_stream_stride_mix/dram_stream_stride_mix.cu`
+ `main.cu`, compiles each variant with nvcc against
`sm_120-laptop`, runs the binary to collect output, calls
`sass_confirm` for the SASS modifier audit, calls
`correctness_check` for the per-mode delta, and assembles the
`perturbation_run_ledger.json` via `ledger.build_bundle`.

Loop-scope families (Loop 2 reduced scope per the task14 artifact):
- `cache_modifier_ca`: PTX-rewrite the kernel's two target source loads
  (`in[idx]`, `in[i]`) to use `ld.global.ca.f32` via an `asm volatile`
  helper.
- `cache_modifier_cg`: same shape, `ld.global.cg.f32`.
- `cache_modifier_cs`: same shape, `ld.global.cs.f32` (cache-streaming;
  R45 extension to close the microbench-suite-coverage criterion's
  cache-modifier-variants family).
- `working_set_sweep`: parameter sweep over `{1, 2, 4, 8, 16, 32, 64} MiB`
  using a shape-based runtime parametrization (no source edit
  required). The harness compiles a binary per ladder value so each
  record has its own binary identity.

Output layout:

    perturbation_build/
      sources/<binary_name>.cu
      sass/<binary_name>.sass
      <binary_name>            (the executable)

The CLI is intended to be invoked from the simbench controller's
`_emit_perturbation_run_ledger_for_run_summary` helper as well as
manual operator runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sass_confirm import (  # type: ignore[import-not-found]  # noqa: E402
    SassConfirmResult,
    confirm_binary,
)
from correctness_check import (  # type: ignore[import-not-found]  # noqa: E402
    compute_abs_and_rel_delta,
    delta_exceeds_tolerance,
    flatten,
)
from ledger import (  # type: ignore[import-not-found]  # noqa: E402
    SASS_EXPECTED_MATCH_COUNT_BY_FAMILY,
    SUPPORTED_TARGET_ARCHES,
    TARGET_ARCH,
    WORKING_SET_LADDER_MIB,
    build_bundle,
    per_family_admitted_count,
    per_typed_reason_count,
)

# The kernel under test for the perturbation harness. The R30 task14
# artifact pins the dram_stream_stride_mix slug as the canonical
# loop-scope kernel; cross-arch and per-slug expansion is queued to
# future loops.
KERNEL_SLUG = "dram_stream_stride_mix"
KERNEL_NAME = "dram_stream_stride_mix_kernel"
KERNEL_DTYPE = "fp32"

NVCC = os.environ.get("NVCC", "nvcc")
NVCC_COMMON = ["-std=c++17", "-Xptxas=-v", "-lineinfo"]


def nvcc_gencode_for_target_arch(target_arch: str) -> list[str]:
    """Per-target-arch nvcc -gencode flags. The Ampere-desktop and
    Blackwell-laptop arches use different compute capability tokens;
    the perturbation harness binaries must be built with the matching
    -gencode pair so the resulting cubin runs on the live host. The
    multi-surface-lockstep partner is the schema's `$defs.target_arch_enum`
    and the controller helper `_perturbation_target_arch_for_props` —
    a new value added to one MUST be added to all three."""
    if target_arch == "sm_86":
        return ["-gencode", "arch=compute_86,code=sm_86"]
    if target_arch == "sm_120-laptop":
        return ["-gencode", "arch=compute_120,code=sm_120"]
    raise ValueError(
        f"unknown target_arch {target_arch!r}; supported: "
        f"{tuple(SUPPORTED_TARGET_ARCHES)}"
    )


@dataclass
class WorkloadPaths:
    """Resolved paths for the workload under test."""
    kernel_dir: Path
    kernel_source: Path
    main_source: Path
    baseline_binary: Path

    @classmethod
    def for_slug(cls, repo_root: Path, slug: str) -> "WorkloadPaths":
        kernel_dir = repo_root / "workloads" / "simbench_bw" / "l1" / slug
        return cls(
            kernel_dir=kernel_dir,
            kernel_source=kernel_dir / f"{slug}.cu",
            main_source=kernel_dir / "main.cu",
            baseline_binary=kernel_dir / f"{slug}_run",
        )


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_cache_modifier_source(
    *, baseline_kernel_text: str, modifier: str,
) -> str:
    """Inject the cache-modifier asm volatile helpers and route the
    two target source loads (`in[idx]` / `in[i]`) through them.

    The R30 task14 artifact (Section 2 build surface) pins this exact
    rewrite. The result is a syntactically valid CUDA source whose
    compiled SASS contains `LDG.E.CA` / `LDG.E.CG` / `LDG.E.CS` (or
    the per-arch Blackwell equivalents) on the perturbed loads. R45
    extended the supported set to include `cs` (cache-streaming) so
    the microbench-suite-coverage criterion's cache-modifier-variants
    family closes.
    """
    if modifier not in {"ca", "cg", "cs"}:
        raise ValueError(f"unknown cache modifier: {modifier!r}")
    helpers = (
        "// SIMBENCH PERTURBATION-HARNESS CACHE-MODIFIER HELPERS\n"
        "__device__ __forceinline__ float "
        f"simbench_ld_global_{modifier}_f32(const float *p) {{\n"
        "    float v;\n"
        f'    asm volatile("ld.global.{modifier}.f32 %0, [%1];" '
        ': "=f"(v) : "l"(p));\n'
        "    return v;\n"
        "}\n"
        "// END SIMBENCH PERTURBATION-HARNESS CACHE-MODIFIER HELPERS\n"
        "\n"
    )
    rewritten = baseline_kernel_text
    rewritten = rewritten.replace(
        "acc += in[idx];",
        f"acc += simbench_ld_global_{modifier}_f32(&in[idx]);",
    )
    rewritten = rewritten.replace(
        "for (int i = tid; i < total; i += lanes) out[i] = in[i];",
        "for (int i = tid; i < total; i += lanes) out[i] = "
        f"simbench_ld_global_{modifier}_f32(&in[i]);",
    )
    insertion_anchor = "extern \"C\" __global__"
    insertion_index = rewritten.find(insertion_anchor)
    if insertion_index == -1:
        raise RuntimeError(
            "build.generate_cache_modifier_source: kernel anchor "
            f"{insertion_anchor!r} not found"
        )
    return (
        rewritten[:insertion_index]
        + helpers
        + rewritten[insertion_index:]
    )


def _build_flags_for_family(
    *, family: str, parameter: dict[str, Any], target_arch: str = TARGET_ARCH,
) -> list[str]:
    """Concrete -D macros per family. The values are recorded into
    the ledger so the schema-required `build_flags` array carries the
    full provenance. The `-gencode` pair is selected per-target_arch
    via `nvcc_gencode_for_target_arch`."""
    flags = list(NVCC_COMMON)
    flags.extend(nvcc_gencode_for_target_arch(target_arch))
    flags.append(
        f"-DSIMBENCH_PERTURBATION_FAMILY={family}"
    )
    if family == "cache_modifier_ca":
        flags.append("-DSIMBENCH_CACHE_MODIFIER=ca")
    elif family == "cache_modifier_cg":
        flags.append("-DSIMBENCH_CACHE_MODIFIER=cg")
    elif family == "cache_modifier_cs":
        flags.append("-DSIMBENCH_CACHE_MODIFIER=cs")
    elif family == "working_set_sweep":
        flags.append(
            f"-DSIMBENCH_WORKING_SET_MIB={parameter['working_set_mib']}"
        )
    return flags


def _binary_name(
    family: str, parameter: dict[str, Any], target_arch: str = TARGET_ARCH,
    slug: str = KERNEL_SLUG,
) -> str:
    # Q10 provenance fix (2026-07-06): the slug is threaded from the CLI /
    # orchestrator instead of always stamping KERNEL_SLUG, so artifacts built
    # for other kernels no longer carry the dram_stream_stride_mix name.
    if family == "cache_modifier_ca":
        return f"{slug}__cache_modifier_ca__{target_arch}"
    if family == "cache_modifier_cg":
        return f"{slug}__cache_modifier_cg__{target_arch}"
    if family == "cache_modifier_cs":
        return f"{slug}__cache_modifier_cs__{target_arch}"
    if family == "working_set_sweep":
        mib = parameter["working_set_mib"]
        return f"{slug}__working_set_{mib}mib__{target_arch}"
    raise ValueError(f"unknown family {family!r}")


def _compile(
    *, source_path: Path, main_path: Path, output_binary: Path,
    flags: list[str],
) -> tuple[int, str]:
    """Invoke nvcc; return (returncode, stderr_tail)."""
    cmd = [NVCC, *flags, str(source_path), str(main_path),
           "-o", str(output_binary), "-lcudart"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stderr[-4096:]


def _run_binary_collect_output(
    *, binary: Path, args: list[str],
) -> dict[str, Any]:
    """Run the perturbation binary; parse stdout JSON. The simbench
    main.cu prints `{ "schema_version": 1, ..., "output": [...] }`.
    """
    proc = subprocess.run(
        [str(binary), *args], capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"perturbation binary {binary} exited rc={proc.returncode}; "
            f"stderr_tail={proc.stderr[-1024:]!r}"
        )
    return json.loads(proc.stdout)


def _shape_args_for(family: str, parameter: dict[str, Any]) -> list[str]:
    """Per-family CLI args for the simbench main.cu host driver.

    Cache-modifier variants run the canonical correctness shape
    (`warps=4 accesses=16 stride_classes=2` per `correctness_00`).
    Working-set variants run the pinned task14 ladder per Section 1:
    `warps = 32`, `accesses = working_set_mib * 4096`,
    `stride_classes = 2`. For `working_set_mib in {16, 32, 64}` the
    binary is invoked with `--output-mode sha256_f32le` so the
    perturbation harness consumes a digest header instead of a
    multi-MiB JSON tensor (R32 fix per Codex's R31 review action
    item #1: working_set_sweep must actually exercise the MiB ladder
    rather than over-running the small cache-modifier shape).
    """
    if family in {"cache_modifier_ca", "cache_modifier_cg", "cache_modifier_cs"}:
        return [
            "--warps", "4", "--accesses", "16", "--stride-classes", "2",
            "--seed", "42", "--shape-point", "correctness_00",
        ]
    if family == "working_set_sweep":
        mib = int(parameter["working_set_mib"])
        args = [
            "--warps", "32",
            "--accesses", str(mib * 4096),
            "--stride-classes", "2",
            "--seed", "42",
            "--shape-point", f"working_set_{mib}mib",
        ]
        if _comparison_mode_for(family, parameter) == "sha256_f32le":
            args.extend(["--output-mode", "sha256_f32le"])
        return args
    raise ValueError(f"unknown family {family!r}")


def _comparison_mode_for(family: str, parameter: dict[str, Any]) -> str:
    """Pick the canonical `correctness_delta.comparison_mode` per
    `(family, parameter)`. Cache-modifier variants always use
    `abs_and_rel`. The working_set_sweep ladder uses `abs_and_rel`
    for 1/2/4/8 MiB (the JSON output is small enough to parse) and
    `sha256_f32le` for 16/32/64 MiB (the digest mode the task14
    artifact pins for large outputs)."""
    if family in {"cache_modifier_ca", "cache_modifier_cg", "cache_modifier_cs"}:
        return "abs_and_rel"
    if family == "working_set_sweep":
        mib = int(parameter["working_set_mib"])
        return "sha256_f32le" if mib >= 16 else "abs_and_rel"
    raise ValueError(f"unknown family {family!r}")


def _expected_working_set_element_count(parameter: dict[str, Any]) -> int:
    """Return the canonical element_count for a working_set_sweep
    parameter: `working_set_mib * 262144` per the task14 Section 1
    table (`warps=32 * accesses=mib*4096 * stride_classes=2 =
    mib*262144` FP32 elements). Raises if the parameter does not
    carry `working_set_mib`. R33 helper per Codex's R32 review:
    locks the producer/verifier element_count invariant against the
    R31 wrong-shape failure class."""
    mib = parameter.get("working_set_mib")
    if mib is None:
        raise ValueError(
            "build._expected_working_set_element_count: parameter "
            f"missing 'working_set_mib': {parameter!r}"
        )
    return int(mib) * 262144


def _baseline_reference(
    *, baseline_binary: Path, baseline_args: list[str],
    cache_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Run the baseline binary once per shape signature and cache its
    JSON output (or digest header for `sha256_f32le` runs).

    The cache key is sha256 over the (binary, args) tuple so each
    `(family, parameter)` pair gets its own reference. The cache
    lives under `<perturbation_build>/reference/<sha>.json`. R32 fix
    per Codex's R31 review action item #2: previously every
    perturbation record reused the cache-modifier baseline reference
    even though the working_set_sweep variants run different shapes.

    Returns `(reference_output_dict, cache_path)` so the caller can
    embed the cache path in the per-record `correctness_delta`'s
    `reference_path`.
    """
    cache_key_input = json.dumps(
        {"binary": str(baseline_binary), "args": baseline_args},
        sort_keys=True,
    )
    cache_key = _sha256_of_text(cache_key_input)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        with open(cache_path) as f:
            return json.load(f), cache_path
    if not baseline_binary.is_file():
        raise FileNotFoundError(
            f"baseline binary {baseline_binary} not present; build it "
            f"with `make` in the workload directory before running "
            f"the perturbation harness"
        )
    output = _run_binary_collect_output(
        binary=baseline_binary, args=baseline_args,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output, cache_path


def _build_one_perturbation(
    *, family: str, parameter: dict[str, Any],
    perturbation_dir: Path, paths: WorkloadPaths,
    parent_baseline_sha256: str,
    reference_dir: Path,
    target_arch: str = TARGET_ARCH,
    slug: str = KERNEL_SLUG,
) -> dict[str, Any]:
    """Build, audit, and produce the canonical perturbation record
    for a single (family, parameter) pair. Returns the schema-shaped
    record dict.

    R32 fix per Codex's R31 review: the baseline reference is now
    computed per `(family, parameter)` via the args-keyed cache; the
    `_comparison_mode_for` helper picks `abs_and_rel` (small-shape
    JSON) or `sha256_f32le` (digest mode for the larger working-set
    ladder) per record.

    The `target_arch` parameter selects the per-arch -gencode flag set
    via `nvcc_gencode_for_target_arch` (sm_86 → compute_86/sm_86;
    sm_120-laptop → compute_120/sm_120). The default preserves the
    pre-widening sm_120-laptop behavior.
    """
    binary_name = _binary_name(family, parameter, target_arch, slug=slug)
    sources_dir = perturbation_dir / "sources"
    sass_dir = perturbation_dir / "sass"
    output_dir = perturbation_dir
    sources_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_kernel_text = paths.kernel_source.read_text(encoding="utf-8")
    if family == "cache_modifier_ca":
        kernel_text = generate_cache_modifier_source(
            baseline_kernel_text=baseline_kernel_text, modifier="ca",
        )
    elif family == "cache_modifier_cg":
        kernel_text = generate_cache_modifier_source(
            baseline_kernel_text=baseline_kernel_text, modifier="cg",
        )
    elif family == "cache_modifier_cs":
        kernel_text = generate_cache_modifier_source(
            baseline_kernel_text=baseline_kernel_text, modifier="cs",
        )
    elif family == "working_set_sweep":
        kernel_text = baseline_kernel_text
    else:
        raise ValueError(f"unknown family {family!r}")

    perturbed_kernel_path = sources_dir / f"{binary_name}.cu"
    perturbed_kernel_path.write_text(kernel_text, encoding="utf-8")
    source_sha = _sha256_of_text(kernel_text)
    flags = _build_flags_for_family(
        family=family, parameter=parameter, target_arch=target_arch,
    )
    args = _shape_args_for(family, parameter)
    comparison_mode = _comparison_mode_for(family, parameter)

    binary_path = output_dir / binary_name
    rc, stderr_tail = _compile(
        source_path=perturbed_kernel_path,
        main_path=paths.main_source,
        output_binary=binary_path,
        flags=flags,
    )
    if rc != 0 or not binary_path.is_file():
        return {
            "result_kind": "typed_unavailable",
            "family": family,
            "typed_reason": "perturbation_compile_failed",
            "parameter": parameter,
            "target_arch": target_arch,
            "build_flags": flags,
            "source_sha256": source_sha,
            "parent_baseline_binary_sha256": parent_baseline_sha256,
            "compiler_returncode": rc,
            "compiler_stderr_tail": stderr_tail,
        }

    binary_sha = _sha256_of_file(binary_path)
    expected = SASS_EXPECTED_MATCH_COUNT_BY_FAMILY[family]
    sass_result: SassConfirmResult = confirm_binary(
        binary_path,
        family=family,
        kernel_unmangled_name=KERNEL_NAME,
        expected_match_count=expected,
        sass_dump_dir=sass_dir,
    )

    if sass_result.status == "fail":
        if family == "working_set_sweep":
            # working_set_sweep cannot fire modifier_optimized_away;
            # if its baseline LDG count drops below 1 the failure is
            # treated as compile pathology with no LDG. In practice
            # this is unreachable because the unmodified kernel has
            # at least one LDG per target source load.
            return {
                "result_kind": "typed_unavailable",
                "family": family,
                "typed_reason": "perturbation_compile_failed",
                "parameter": parameter,
                "target_arch": target_arch,
                "build_flags": flags,
                "source_sha256": source_sha,
                "parent_baseline_binary_sha256": parent_baseline_sha256,
                "compiler_returncode": rc,
                "compiler_stderr_tail": (
                    "working_set_sweep produced no matching LDG line "
                    "in the target kernel section"
                ),
            }
        return {
            "result_kind": "typed_unavailable",
            "family": family,
            "typed_reason": "modifier_optimized_away",
            "parameter": parameter,
            "target_arch": target_arch,
            "binary_sha256": binary_sha,
            "build_flags": flags,
            "source_sha256": source_sha,
            "parent_baseline_binary_sha256": parent_baseline_sha256,
            "sass_artifact_path": sass_result.artifact_path,
            "sass_confirmation_hash": sass_result.confirmation_hash,
            "sass_expected_match_count": sass_result.expected_match_count,
            "sass_match_count": sass_result.match_count,
        }

    # Per-record baseline reference (R32). Each `(family, parameter)`
    # pair runs the baseline binary at its own shape; `--output-mode`
    # in `args` already carries `sha256_f32le` for digest-mode
    # records, so the cached reference shape matches the
    # perturbation invocation.
    reference_output, reference_cache_path = _baseline_reference(
        baseline_binary=paths.baseline_binary,
        baseline_args=args,
        cache_dir=reference_dir,
    )
    reference_path_str = str(reference_cache_path)

    actual_output = _run_binary_collect_output(
        binary=binary_path, args=args,
    )
    actual_path = output_dir / f"{binary_name}.actual.json"
    actual_path.write_text(
        json.dumps(actual_output, indent=2), encoding="utf-8",
    )

    if comparison_mode == "sha256_f32le":
        delta = _build_digest_delta(
            actual_output=actual_output,
            reference_output=reference_output,
            dtype=KERNEL_DTYPE,
            reference_path=reference_path_str,
            actual_output_path=str(actual_path),
            expected_element_count=_expected_working_set_element_count(
                parameter,
            ),
        )
    else:
        flat_actual = flatten(actual_output["output"])
        flat_reference = flatten(reference_output["output"])
        delta = compute_abs_and_rel_delta(
            flat_actual, flat_reference,
            dtype=KERNEL_DTYPE,
            reference_path=reference_path_str,
            actual_output_path=str(actual_path),
        )
        # R33: producer-side working_set_sweep element_count
        # invariant. The R32 implementation enforced this for the
        # digest path only; under abs_and_rel mode a producer bug
        # could still emit a record claiming a large
        # working_set_mib while comparing only 128 elements (the
        # exact R31 failure class). Per Codex's R32 review,
        # the producer raises if the flattened JSON length or the
        # emitted delta element_count drifts from the expected
        # `working_set_mib * 262144` value.
        if family == "working_set_sweep":
            expected = _expected_working_set_element_count(parameter)
            if len(flat_actual) != expected:
                raise ValueError(
                    "build._build_one_perturbation: working_set_sweep "
                    f"actual element_count={len(flat_actual)} does not "
                    f"match the expected working_set_mib * 262144 = "
                    f"{expected} for parameter {parameter!r}"
                )
            if len(flat_reference) != expected:
                raise ValueError(
                    "build._build_one_perturbation: working_set_sweep "
                    f"reference element_count={len(flat_reference)} "
                    f"does not match the expected working_set_mib * "
                    f"262144 = {expected} for parameter {parameter!r}"
                )
            if delta["element_count"] != expected:
                raise ValueError(
                    "build._build_one_perturbation: working_set_sweep "
                    f"correctness_delta.element_count="
                    f"{delta['element_count']} does not match the "
                    f"expected working_set_mib * 262144 = {expected} "
                    f"for parameter {parameter!r}"
                )

    if delta_exceeds_tolerance(delta):
        return {
            "result_kind": "typed_unavailable",
            "family": family,
            "typed_reason": "correctness_equivalence_failed",
            "parameter": parameter,
            "target_arch": target_arch,
            "binary_sha256": binary_sha,
            "build_flags": flags,
            "source_sha256": source_sha,
            "parent_baseline_binary_sha256": parent_baseline_sha256,
            "sass_artifact_path": sass_result.artifact_path,
            "sass_confirmation_hash": sass_result.confirmation_hash,
            "sass_confirmation_status": "pass",
            "sass_expected_match_count": sass_result.expected_match_count,
            "sass_match_count": sass_result.match_count,
            "correctness_delta": delta,
        }

    return {
        "result_kind": "ok",
        "family": family,
        "parameter": parameter,
        "target_arch": target_arch,
        "binary_sha256": binary_sha,
        "build_flags": flags,
        "source_sha256": source_sha,
        "parent_baseline_binary_sha256": parent_baseline_sha256,
        "sass_artifact_path": sass_result.artifact_path,
        "sass_confirmation_hash": sass_result.confirmation_hash,
        "sass_confirmation_status": "pass",
        "sass_expected_match_count": sass_result.expected_match_count,
        "sass_match_count": sass_result.match_count,
        "correctness_equivalence_status": "pass",
        "correctness_delta": delta,
    }


def _build_digest_delta(
    *, actual_output: dict[str, Any],
    reference_output: dict[str, Any],
    dtype: str,
    reference_path: str, actual_output_path: str,
    expected_element_count: int,
) -> dict[str, Any]:
    """Build the canonical `correctness_delta` digest arm for
    `sha256_f32le` mode.

    Both `actual_output` and `reference_output` are JSON headers
    emitted by `main.cu --output-mode sha256_f32le`; the headers
    carry `output_sha256` (sha256 over the FP32 LE byte stream) and
    `element_count`. The harness verifies the element count matches
    the parameter table's expected value before emitting the delta;
    a mismatch is a producer bug (the harness invoked the wrong
    shape).
    """
    if actual_output.get("output_mode") != "sha256_f32le":
        raise ValueError(
            "build._build_digest_delta: actual output is not in "
            "sha256_f32le mode (got "
            f"{actual_output.get('output_mode')!r})"
        )
    if reference_output.get("output_mode") != "sha256_f32le":
        raise ValueError(
            "build._build_digest_delta: reference output is not in "
            "sha256_f32le mode (got "
            f"{reference_output.get('output_mode')!r})"
        )
    actual_sha = actual_output["output_sha256"]
    reference_sha = reference_output["output_sha256"]
    actual_count = int(actual_output["element_count"])
    reference_count = int(reference_output["element_count"])
    if actual_count != reference_count:
        raise ValueError(
            "build._build_digest_delta: element_count mismatch between "
            f"actual ({actual_count}) and reference ({reference_count})"
        )
    if actual_count != expected_element_count:
        raise ValueError(
            "build._build_digest_delta: element_count "
            f"{actual_count} does not match the expected "
            f"working_set_mib * 262144 = {expected_element_count}"
        )
    tols = {
        "fp32": (1e-5, 1e-4),
        "fp16": (1e-3, 1e-2),
        "bf16": (1e-2, 5e-2),
    }
    if dtype not in tols:
        raise ValueError(
            f"build._build_digest_delta: dtype {dtype!r} not "
            "supported in sha256_f32le mode"
        )
    abs_tol, rel_tol = tols[dtype]
    return {
        "dtype": dtype,
        "comparison_mode": "sha256_f32le",
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "actual_sha256": actual_sha,
        "reference_sha256": reference_sha,
        "element_count": actual_count,
        "reference_path": reference_path,
        "actual_output_path": actual_output_path,
    }


def loop_scope_record_specs() -> list[tuple[str, dict[str, Any]]]:
    """The default 10 loop-scope (family, parameter) pairs: three
    cache-modifier variants (`.ca`/`.cg`/`.cs`; R45 added `.cs`)
    plus the seven `working_set_sweep` MiB ladder values."""
    specs: list[tuple[str, dict[str, Any]]] = [
        ("cache_modifier_ca", {"modifier": "ca"}),
        ("cache_modifier_cg", {"modifier": "cg"}),
        ("cache_modifier_cs", {"modifier": "cs"}),
    ]
    for mib in WORKING_SET_LADDER_MIB:
        specs.append(
            ("working_set_sweep", {
                "working_set_mib": mib,
                "allocation_bytes": mib * 1048576,
                "warps": 32,
                "accesses": mib * 4096,
                "stride_classes": 2,
            })
        )
    return specs


def run_perturbation_harness(
    *, repo_root: Path, slug: str,
    manifest_pairing_index: dict[str, Any],
    device: dict[str, Any],
    perturbation_build_dir: Path,
    target_arch: str = TARGET_ARCH,
) -> dict[str, Any]:
    """Top-level orchestrator: build all loop-scope records, assemble
    + strict-validate the bundle. Used by the simbench controller's
    `_emit_perturbation_run_ledger_for_run_summary` helper.

    The `target_arch` parameter selects the per-arch -gencode flag set
    threaded through `_build_one_perturbation`. The default preserves
    the pre-widening sm_120-laptop behavior; the simbench controller
    selects per-host via `_perturbation_target_arch_for_props(props)`.
    """
    if target_arch not in SUPPORTED_TARGET_ARCHES:
        raise ValueError(
            f"unknown target_arch {target_arch!r}; supported: "
            f"{tuple(SUPPORTED_TARGET_ARCHES)}"
        )
    paths = WorkloadPaths.for_slug(repo_root, slug)
    if not paths.kernel_source.is_file():
        raise FileNotFoundError(
            f"perturbation harness: kernel source {paths.kernel_source} "
            f"not present"
        )
    if not paths.main_source.is_file():
        raise FileNotFoundError(
            f"perturbation harness: main source {paths.main_source} "
            f"not present"
        )
    if not paths.baseline_binary.is_file():
        raise FileNotFoundError(
            f"perturbation harness: baseline binary "
            f"{paths.baseline_binary} not present (run `make` in "
            f"{paths.kernel_dir})"
        )
    parent_baseline_sha = _sha256_of_file(paths.baseline_binary)
    perturbation_build_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = perturbation_build_dir / "reference"

    perturbations: list[dict[str, Any]] = []
    for family, parameter in loop_scope_record_specs():
        try:
            record = _build_one_perturbation(
                family=family, parameter=parameter,
                perturbation_dir=perturbation_build_dir,
                paths=paths,
                parent_baseline_sha256=parent_baseline_sha,
                reference_dir=reference_dir,
                target_arch=target_arch,
                slug=slug,
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            record = {
                "result_kind": "typed_unavailable",
                "family": family,
                "typed_reason": "perturbation_compile_failed",
                "parameter": parameter,
                "target_arch": target_arch,
                "build_flags": _build_flags_for_family(
                    family=family, parameter=parameter,
                    target_arch=target_arch,
                ),
                "source_sha256": _sha256_of_text(""),
                "parent_baseline_binary_sha256": parent_baseline_sha,
                "compiler_returncode": -1,
                "compiler_stderr_tail": (
                    f"orchestration_error: {exc!r}"
                )[:4096],
            }
        perturbations.append(record)

    bundle = build_bundle(
        manifest_pairing_index=manifest_pairing_index,
        device=device,
        perturbations=perturbations,
    )
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.perturbation_harness.build",
        description=(
            "Build the perturbation harness bundle for the "
            "perturbation-harness clause: 10 loop-scope records "
            "across cache_modifier_ca, cache_modifier_cg, "
            "cache_modifier_cs, and the working_set_sweep ladder."
        ),
    )
    parser.add_argument(
        "--slug", default=KERNEL_SLUG,
        help=f"workload slug (default: {KERNEL_SLUG})",
    )
    parser.add_argument(
        "--arch", default=TARGET_ARCH,
        choices=tuple(SUPPORTED_TARGET_ARCHES),
        help=(
            f"target arch enum (default: {TARGET_ARCH}; "
            f"supported: {tuple(SUPPORTED_TARGET_ARCHES)})"
        ),
    )
    parser.add_argument(
        "--out", required=True,
        help="output path for perturbation_run_ledger.json",
    )
    parser.add_argument(
        "--build-dir", default=None,
        help="perturbation_build directory (default: alongside kernel)",
    )
    parser.add_argument(
        "--repo-root", default=None,
        help="repository root (default: auto-detect)",
    )
    args = parser.parse_args(argv)

    repo_root = (
        Path(args.repo_root).resolve() if args.repo_root
        else _HERE.parent.parent
    )
    paths = WorkloadPaths.for_slug(repo_root, args.slug)
    build_dir = (
        Path(args.build_dir).resolve() if args.build_dir
        else paths.kernel_dir / "perturbation_build"
    )

    manifest_pairing_index = {
        "slug": args.slug,
        "shape_point": "correctness_00",
        "binary_sha256": (
            _sha256_of_file(paths.baseline_binary)
            if paths.baseline_binary.is_file() else "0" * 64
        ),
        "source_commit": _resolve_source_commit(repo_root),
        "kernel_signature_sha256": _sha256_of_text(
            paths.kernel_source.read_text(encoding="utf-8")
        ),
    }
    device = _device_block_for_target_arch(args.arch)

    bundle = run_perturbation_harness(
        repo_root=repo_root,
        slug=args.slug,
        manifest_pairing_index=manifest_pairing_index,
        device=device,
        perturbation_build_dir=build_dir,
        target_arch=args.arch,
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    counts = per_family_admitted_count(bundle["perturbations"])
    typed = per_typed_reason_count(bundle["perturbations"])
    admitted = sum(counts.values())
    typed_unavail = sum(typed.values())
    print(
        f"perturbation_harness: wrote {out_path} "
        f"(admitted={admitted}, typed_unavailable={typed_unavail})"
    )
    return 0


def _resolve_source_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except FileNotFoundError:
        pass
    return "0" * 40


def _device_block_for_target_arch(target_arch: str) -> dict[str, Any]:
    """Static device block for the named target arch. The simbench
    controller overrides this with the live-CUDA `DeviceProps` block
    when invoking from `_emit_perturbation_run_ledger_for_run_summary`;
    the static block is only used by the standalone CLI entry point.

    Multi-surface-lockstep partner of `nvcc_gencode_for_target_arch`,
    the schema's `$defs.target_arch_enum`, and the controller helper
    `_perturbation_target_arch_for_props`. A new value added to one
    MUST be added to all four."""
    if target_arch == "sm_86":
        return {
            "device_name": "NVIDIA GeForce RTX 3090 Ti",
            "compute_capability_major": 8,
            "compute_capability_minor": 6,
            "sm_count": 0,
            "l2_cache_size_bytes": 0,
        }
    if target_arch == "sm_120-laptop":
        return {
            "device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
            "compute_capability_major": 12,
            "compute_capability_minor": 0,
            "sm_count": 0,
            "l2_cache_size_bytes": 0,
        }
    raise ValueError(
        f"unknown target_arch {target_arch!r}; supported: "
        f"{tuple(SUPPORTED_TARGET_ARCHES)}"
    )


__all__ = [
    "KERNEL_SLUG",
    "KERNEL_NAME",
    "KERNEL_DTYPE",
    "WorkloadPaths",
    "generate_cache_modifier_source",
    "loop_scope_record_specs",
    "run_perturbation_harness",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
