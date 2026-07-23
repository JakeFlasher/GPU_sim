#!/usr/bin/env python3
"""Materialise a baseline tree by copying an exemplar and emitting metadata.

Given an internal problem slug, a target language (``cutlass`` or
``cuda_cpp``), and a chosen exemplar tree under
``sol-execbench/examples/{cutlass,cuda_cpp}/<exemplar>/``, this tool:

1. Resolves the benchmark directory under
   ``sol-execbench/data/benchmark/<subset>/<NNN_name>/`` and confirms
   ``definition.json`` exists.
2. Copies the exemplar's editable source files
   (``kernel.h``, ``kernel.cu``, ``main.cpp``) into
   ``baselines/<category>/problem_NNN_<slug>/<language>/``.
3. Renames the exemplar's kernel-symbol prefix (e.g.
   ``rmsnorm_h4096_*``) to one derived from the target slug, so that
   different baselines never collide at link time.
4. Converts the C++ entry function ``run(...)`` to destination-passing
   style: returns ``void``; output tensors are appended to the parameter
   list in ``definition.json.outputs`` insertion order; the original
   ``torch::empty_like`` allocation and ``return <name>;`` are removed.
5. Emits a Pydantic-valid ``solution.json`` whose ``sources[]`` is
   populated from the just-written disk files in deterministic order
   (entry source first, remaining sorted lexicographically by ``path``).

Usage:
    python scripts/baselines_scaffold/scaffold.py \\
        --slug problem_001_fused_add_rmsnorm_h2048 \\
        --language cuda_cpp \\
        --exemplar cuda_cpp/rmsnorm

The scaffolder fails closed (exit code 2, nothing written) when:

- ``--language`` is not ``cutlass`` / ``cuda_cpp`` (argparse choices).
- ``--slug`` is missing from
  ``tools/sol_baselines_scaffold/sol_pattern_mapping.md``.
- The resolved benchmark ``definition.json`` is missing (the error
  message points at ``./sol-execbench/scripts/download_data.sh``).
- The exemplar path does not exist under
  ``sol-execbench/examples/{cutlass,cuda_cpp}/``.
- The target directory already exists (no clobber).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from baselines_scaffold.generate import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MAPPING_PATH,
    REPO_ROOT,
    SUBSET_MAP,
    SUPPORTED_LANGUAGES,
    derive_benchmark_name,
    parse_mapping,
)

DEFAULT_EXAMPLES_ROOT = REPO_ROOT / "sol-execbench" / "examples"


def _detect_local_arch_tag(
    *, override: Optional[str] = None
) -> str:
    """Return the local arch tag for stub-README evidence labels.

    Lazy-imports the orchestrator's ``probe_host`` + ``arch_resolver``
    so the scaffold's plain ``import baselines_scaffold.scaffold``
    path stays decoupled from the live ``nvidia-smi`` / ``nvcc``
    subprocess machinery. Tests can pin a deterministic value through
    the ``override`` kwarg.

    Fallback contract: any probe failure (no GPU visible, ``nvcc``
    absent, the resolver returns ``sm_unknown``) collapses to
    ``sm_unknown``. Callers must treat that as a graceful degraded
    state and not as a fail-closed error — the scaffold is allowed
    to emit a stub README on a host without a GPU (e.g. a CPU-only
    dev container preparing a PR) and the operator is expected to
    hand-edit the evidence label before landing.
    """
    if override is not None:
        return override
    try:
        from baselines_harness.arch_resolver import arch_tag_from_host_kind
        from baselines_harness.surfaces.host_probe import probe_host
    except ImportError:
        return "sm_unknown"
    try:
        host_info = probe_host()
    except Exception:  # noqa: BLE001 - live-probe failure shouldn't abort scaffold
        return "sm_unknown"
    return arch_tag_from_host_kind(getattr(host_info, "kind", None))


# Editable source files we copy from the exemplar. Upstream benchmark
# assets (definition.json, workload.jsonl, reference.py) and the
# exemplar's own solution_*.json are deliberately excluded — the
# scaffolder emits a fresh solution.json instead.
_COPIED_SOURCE_FILES: Tuple[str, ...] = ("kernel.h", "kernel.cu", "main.cpp")

# Output suffix permitted on emitted entry source files (C++ family).
_CPP_SOURCE_SUFFIXES: Tuple[str, ...] = (".cu", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cuh")

# Identifier-safe slug stem used for kernel-symbol renames. We keep the
# full ``problem_NNN_<rest>`` stem — it's a valid C++ identifier and
# guarantees uniqueness across baselines.
_SLUG_IDENT_RE = re.compile(r"^problem_[0-9]+_[A-Za-z_][A-Za-z0-9_]*$")


# --- Errors ---------------------------------------------------------------


class ScaffoldError(Exception):
    """Fail-closed scaffolder failure with an actionable message."""


# --- Slug + exemplar resolution ------------------------------------------


def resolve_category_for_slug(slug: str, mapping_path: Path) -> str:
    """Return the internal category (L1/L2/Quant/FIB) for ``slug``.

    Fails closed if the slug is missing from the mapping or the
    mapping file is unreadable.
    """
    if not mapping_path.is_file():
        raise ScaffoldError(
            f"Mapping not found: {mapping_path}. The mapping is repo-tracked "
            f"and should not be missing."
        )
    rows = parse_mapping(mapping_path)
    row = rows.get(slug)
    if row is None:
        raise ScaffoldError(
            f"Slug {slug!r} is not present in {mapping_path}. Check spelling "
            f"or add a mapping row before scaffolding."
        )
    if row.category not in SUBSET_MAP:
        raise ScaffoldError(
            f"Slug {slug!r} has unknown category {row.category!r}. Known "
            f"categories: {sorted(SUBSET_MAP)}."
        )
    return row.category


def resolve_benchmark_dir(
    slug: str, category: str, benchmark_root: Path
) -> Path:
    """Resolve ``benchmark_root/<subset>/<NNN_name>/`` for ``slug``."""
    subset = SUBSET_MAP[category]
    bench_name = derive_benchmark_name(slug)
    bench_dir = benchmark_root / subset / bench_name
    definition = bench_dir / "definition.json"
    if not definition.is_file():
        raise ScaffoldError(
            f"Benchmark definition missing at {definition}. Run "
            f"./sol-execbench/scripts/download_data.sh to populate "
            f"sol-execbench/data/benchmark/{subset}/, or check the slug "
            f"resolution rule (^problem_ is stripped to yield the "
            f"downloader name)."
        )
    return bench_dir


def resolve_exemplar_dir(exemplar: str, examples_root: Path) -> Path:
    """Resolve and validate ``examples_root/<exemplar>/``.

    ``exemplar`` is a relative path under ``examples_root``, e.g.
    ``cuda_cpp/rmsnorm`` or ``cutlass/gemm``. Path traversal is rejected.
    """
    if exemplar.startswith("/") or ".." in Path(exemplar).parts:
        raise ScaffoldError(
            f"Invalid exemplar path {exemplar!r}: must be relative without "
            f"parent traversal."
        )
    exemplar_dir = examples_root / exemplar
    if not exemplar_dir.is_dir():
        raise ScaffoldError(
            f"Exemplar not found at {exemplar_dir}. Available exemplars "
            f"live under {examples_root} (e.g. cuda_cpp/rmsnorm, "
            f"cutlass/gemm)."
        )
    return exemplar_dir


# --- Kernel-symbol rename -------------------------------------------------


def discover_exemplar_prefix(exemplar_dir: Path) -> Optional[str]:
    """Best-effort guess of the exemplar's kernel-symbol prefix.

    We scan ``kernel.cu`` (and ``kernel.h`` if present) for
    ``__global__`` kernel declarations and ``void <name>_launcher(`` host
    functions, then return the longest common identifier prefix shared
    by ``<name>_kernel`` and ``<name>_launcher``. Falls back to ``None``
    when no match is found — in that case the scaffolder still copies
    the tree but skips symbol renames (the author handles it).
    """
    candidates: List[str] = []
    for fname in ("kernel.cu", "kernel.h"):
        path = exemplar_dir / fname
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # ``__global__ <attrs?> <ret> <name>(``
        for match in re.finditer(
            r"__global__\s+(?:\w[\w\s\(\),]*\s+)*(\w+)\s*\(", text
        ):
            candidates.append(match.group(1))
        for match in re.finditer(r"\bvoid\s+(\w+_launcher)\s*\(", text):
            candidates.append(match.group(1))
    # Strip trailing _kernel/_launcher and pick the most common stem.
    stems: Dict[str, int] = {}
    for ident in candidates:
        for tag in ("_kernel", "_launcher"):
            if ident.endswith(tag):
                stem = ident[: -len(tag)]
                stems[stem] = stems.get(stem, 0) + 1
    if not stems:
        return None
    # Most-frequent stem wins; ties broken by length (longer is more
    # specific and less likely to be an accidental prefix).
    return max(stems.items(), key=lambda kv: (kv[1], len(kv[0])))[0]


def slug_identifier_stem(slug: str) -> str:
    """Derive a C++-safe identifier stem from ``slug``.

    The full ``problem_NNN_<rest>`` slug is itself a valid identifier;
    we use it directly to guarantee uniqueness across baselines.
    """
    if not _SLUG_IDENT_RE.match(slug):
        raise ScaffoldError(
            f"Slug {slug!r} is not a valid C++ identifier "
            f"(expected ^problem_[0-9]+_[A-Za-z_][A-Za-z0-9_]*$)."
        )
    return slug


def rename_kernel_symbols(text: str, old_stem: str, new_stem: str) -> str:
    """Replace ``<old_stem>_<suffix>`` with ``<new_stem>_<suffix>``.

    Only matches at word boundaries so partial-identifier collisions
    don't fire. The replacement applies to both ``_kernel`` and
    ``_launcher`` suffixes (the two surfaces an exemplar typically
    exposes), and to any other ``<old_stem>_<suffix>`` identifier — the
    rename is intentionally aggressive because the exemplar's prefix
    encodes the original problem's shape (e.g. ``rmsnorm_h4096``) which
    must not survive into a baseline targeting a different shape.
    """
    if old_stem == new_stem:
        return text
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old_stem)}(?=_)")
    return pattern.sub(new_stem, text)


# --- Destination-passing-style conversion --------------------------------


def _balance_parens(text: str, open_idx: int) -> int:
    """Return the index of the ``)`` matching ``text[open_idx] == '('``.

    Raises ``ScaffoldError`` if no matching paren is found.
    """
    if text[open_idx] != "(":
        raise ScaffoldError(
            f"Expected '(' at position {open_idx} but got {text[open_idx]!r}."
        )
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ScaffoldError(
        "Unbalanced parentheses while parsing C++ entry function."
    )


def _format_full_run_params(
    input_names: List[str], output_names: List[str]
) -> str:
    """Format the full DPS ``run`` parameter list.

    Inputs come first in ``definition.json.inputs`` insertion order
    (declared ``const torch::Tensor&`` — the host never writes back into
    inputs). Outputs follow in ``definition.json.outputs`` insertion
    order, declared as plain ``torch::Tensor`` (by-value, mutable —
    PyTorch tensors are reference-counted, so by-value preserves the
    harness's in-place write semantics; see CLAUDE.md "Submission
    Format" for the canonical signature shape).
    """
    parts: List[str] = []
    for name in input_names:
        parts.append(f"    const torch::Tensor& {name}")
    for name in output_names:
        parts.append(f"    torch::Tensor {name}")
    if not parts:
        return ""
    return "\n" + ",\n".join(parts) + "\n"


def dps_convert_main(
    main_text: str,
    input_names: List[str],
    output_names: List[str],
) -> Tuple[str, bool]:
    """Rewrite the entry function as ``void run(<inputs>, <outputs>)``.

    Returns ``(converted_text, did_convert)``. ``did_convert`` is True
    when the entry function was found and rewritten. When False, the
    caller must fail closed — an exemplar without a recognisable
    ``run(...)`` signature cannot be safely scaffolded into a baseline.

    Conversion steps:

    1. Locate the entry function signature: ``torch::Tensor run(`` or
       ``void run(``.
    2. Find the ``)`` matching the open paren via balanced-paren parsing
       and REPLACE the entire parameter list with one synthesised from
       ``definition.json.inputs`` + ``definition.json.outputs`` (insertion
       order). The exemplar's parameter names are discarded — the
       scaffolder owns the target signature.
    3. Force the return type to ``void``.
    4. Remove ``auto <name> = torch::empty_like(...)`` and similar
       allocations the exemplar used to construct the return value.
    5. Remove ``return <name>;`` for the previous output names AND for
       any tracked allocation identifiers (belt-and-braces).
    """
    sig_pattern = re.compile(
        r"\b(?:torch::Tensor|void)\s+run\s*\(", re.MULTILINE
    )
    sig_match = sig_pattern.search(main_text)
    if sig_match is None:
        return main_text, False

    # Rewrite the whole ``run(...)`` signature: replace the parameter
    # list with the destination-passing-style (inputs then outputs)
    # ordering derived from ``definition.json``.
    pre = main_text[: sig_match.start()]
    rest = main_text[sig_match.start() :]
    # Find the open-paren position inside ``rest``.
    open_local = rest.index("(")
    close_local = _balance_parens(rest, open_local)
    new_param_list = _format_full_run_params(input_names, output_names)
    new_signature = f"void run({new_param_list})"
    text = pre + new_signature + rest[close_local + 1 :]

    # Remove ``auto <name> = torch::empty_like(...);`` allocations and
    # capture the identifiers so we can also drop the matching
    # ``return <ident>;`` below — outputs are pre-allocated by the
    # evaluator under destination-passing-style.
    alloc_pattern = re.compile(
        r"\n[ \t]*auto\s+(\w+)\s*=\s*torch::empty_like\s*\([^;]*\)\s*;[ \t]*",
        re.MULTILINE,
    )
    alloc_match = alloc_pattern.search(text)
    allocated_names: List[str] = []
    while alloc_match is not None:
        allocated_names.append(alloc_match.group(1))
        text = text[: alloc_match.start()] + text[alloc_match.end() :]
        alloc_match = alloc_pattern.search(text)

    # Drop ``return <name>;`` for every tracked allocation and every
    # target output name. Both forms appear in exemplars; both are
    # wrong under destination-passing-style.
    for name in [*allocated_names, *output_names]:
        return_re = re.compile(
            rf"\n[ \t]*return\s+{re.escape(name)}\s*;[ \t]*",
            re.MULTILINE,
        )
        text = return_re.sub("", text)

    return text, True


# --- Solution.json emission ----------------------------------------------


def _read_definition(bench_dir: Path) -> Dict[str, Any]:
    text = (bench_dir / "definition.json").read_text(encoding="utf-8")
    return json.loads(text)


def _ordered_input_names(definition: Dict[str, Any]) -> List[str]:
    return list(definition.get("inputs", {}).keys())


def _ordered_output_names(definition: Dict[str, Any]) -> List[str]:
    return list(definition.get("outputs", {}).keys())


def _sources_in_canonical_order(
    target_dir: Path, entry_path: str
) -> List[Tuple[str, str]]:
    """Walk ``target_dir`` and return ``[(rel_path, content), ...]``.

    Ordering: entry source first, remaining sources sorted
    lexicographically by relative path.
    """
    seen: Dict[str, str] = {}
    for path in sorted(target_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "solution.json":
            continue
        rel = path.relative_to(target_dir).as_posix()
        seen[rel] = path.read_text(encoding="utf-8")
    if entry_path not in seen:
        raise ScaffoldError(
            f"Entry source {entry_path!r} not present in scaffolded tree "
            f"under {target_dir}. Sources found: {sorted(seen)}."
        )
    ordered: List[Tuple[str, str]] = [(entry_path, seen.pop(entry_path))]
    for rel in sorted(seen):
        ordered.append((rel, seen[rel]))
    return ordered


# Explicit sm_86 gencode for the design contract sm_86 DONE bar. Emitted (as two argv
# tokens nvcc consumes verbatim) only in --local-sm86 mode; the dual-target
# default leaves arch selection to the LOCAL auto-detect / B200 packager.
SM86_GENCODE_CFLAGS: Tuple[str, str] = ("-gencode", "arch=compute_86,code=sm_86")


def _build_solution_dict(
    slug: str,
    language: str,
    bench_name: str,
    entry_path: str,
    sources: List[Tuple[str, str]],
    *,
    local_sm86: bool = False,
) -> Dict[str, Any]:
    """Construct the dict shape the live ``Solution`` Pydantic model expects.

    ``local_sm86`` emits an the design contract-conforming sm_86 spec: ``target_hardware ==
    ["LOCAL"]`` (exactly one target) plus an explicit
    ``-gencode arch=compute_86,code=sm_86`` cuda_cflag, for the sm_86 coverage
    DONE bar. The default keeps the dual ``["LOCAL", "B200"]`` template spec.
    """
    if local_sm86:
        target_hardware = ["LOCAL"]
        cuda_cflags = ["-O3", "--use_fast_math", *SM86_GENCODE_CFLAGS]
    else:
        target_hardware = ["LOCAL", "B200"]
        cuda_cflags = ["-O3", "--use_fast_math"]
    return {
        "name": f"{slug}_{language}_scaffold",
        "definition": bench_name,
        "author": "scaffold",
        "spec": {
            "languages": [language],
            # Default template solutions target both LOCAL (whichever
            # NCU-permitted local host is currently attached) and B200
            # (sm_100a) for Phase-2 profiling on the provider that is
            # also the online judge host. Per CLAUDE.md §"Submission
            # Format": ["LOCAL"] for the local compile-check, ["B200"]
            # for the cloud profiling target; the packager auto-injects
            # the right -gencode for each. --local-sm86 narrows this to
            # the design contract sm_86 DONE bar (LOCAL + explicit sm_86 gencode).
            "target_hardware": target_hardware,
            "entry_point": f"{entry_path}::run",
            "dependencies": [],
            "destination_passing_style": True,
            "binding": "torch",
            "compile_options": {
                "cflags": [],
                "cuda_cflags": cuda_cflags,
                "ld_flags": ["-lcuda"],
            },
        },
        "sources": [
            {"path": path, "content": content}
            for path, content in sources
        ],
        "description": (
            f"Scaffolded baseline for {slug} (language={language}); "
            f"hand-edit the copied exemplar to implement the target "
            f"problem before harness-running. Destination-passing-style "
            f"is set true; outputs appear after inputs in definition.json "
            f"insertion order."
        ),
    }


def conform_existing_solution_to_local_sm86(solution_path: Path) -> bool:
    """Conform an existing ``solution.json`` to the design contract sm_86 DONE bar:
    set ``target_hardware == ["LOCAL"]`` and ensure the explicit
    ``-gencode arch=compute_86,code=sm_86`` cuda_cflag is present. Idempotent;
    re-validates against the live ``Solution`` schema; writes back only on a
    change. Returns ``True`` if the file was modified.

    On sm_86 the explicit gencode is byte-identical to the LOCAL auto-detect
    build, so this conforms the spec without altering the compiled binary.
    """
    solution_path = Path(solution_path)
    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    spec = solution.setdefault("spec", {})
    changed = False

    if spec.get("target_hardware") != ["LOCAL"]:
        spec["target_hardware"] = ["LOCAL"]
        changed = True

    compile_options = spec.setdefault("compile_options", {})
    cuda_cflags = list(compile_options.get("cuda_cflags", []) or [])
    if "arch=compute_86,code=sm_86" not in " ".join(cuda_cflags):
        cuda_cflags.extend(SM86_GENCODE_CFLAGS)
        compile_options["cuda_cflags"] = cuda_cflags
        changed = True

    # Always re-validate against the live schema (even on a no-op) so the
    # docstring's contract holds and a previously-conformed file is confirmed
    # still valid; only write when something actually changed.
    _validate_against_live_schema(solution)
    if changed:
        solution_path.write_text(
            json.dumps(solution, indent=2) + "\n", encoding="utf-8"
        )
    return changed


def _validate_against_live_schema(solution_dict: Dict[str, Any]) -> None:
    """Defer to the live ``Solution`` Pydantic model.

    Imported lazily because ``sol-execbench`` may not be on the
    PYTHONPATH in all invocation contexts; on import failure we surface
    a clear error and abort. (The validator is also exercised by
    ``validate.py`` and ``test_contract.py`` downstream.)
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "sol-execbench" / "src"))
        from sol_execbench.core.data.solution import Solution  # noqa: E402
    except ImportError as exc:
        raise ScaffoldError(
            f"Cannot import live Solution Pydantic model: {exc}. Ensure "
            f"sol-execbench/src is on the PYTHONPATH or invoke from the "
            f"repo root."
        ) from exc
    try:
        Solution.model_validate(solution_dict)
    except Exception as exc:  # pydantic.ValidationError or similar
        raise ScaffoldError(
            f"Emitted solution.json failed Pydantic validation: {exc}"
        ) from exc


# --- Source copy ---------------------------------------------------------


def copy_exemplar_sources(exemplar_dir: Path, target_dir: Path) -> List[str]:
    """Copy editable source files from exemplar to target.

    Returns the list of copied relative paths. Skips upstream benchmark
    assets (``definition.json``, ``workload.jsonl``, ``reference.py``)
    and the exemplar's own ``solution_*.json``.
    """
    copied: List[str] = []
    target_dir.mkdir(parents=True, exist_ok=False)
    for fname in _COPIED_SOURCE_FILES:
        src = exemplar_dir / fname
        if not src.is_file():
            continue
        shutil.copyfile(src, target_dir / fname)
        copied.append(fname)
    if not copied:
        raise ScaffoldError(
            f"Exemplar {exemplar_dir} contains none of "
            f"{_COPIED_SOURCE_FILES}. Pick a different exemplar."
        )
    return copied


def _resolve_entry_path(language: str, copied: List[str]) -> str:
    """Pick the entry source path based on language family.

    For ``cuda_cpp`` / ``cutlass`` exemplars we prefer ``main.cpp``
    (the canonical PyBind11 binding host). Fall back to the first
    copied C++ source if ``main.cpp`` is absent.
    """
    if "main.cpp" in copied:
        return "main.cpp"
    for fname in copied:
        if Path(fname).suffix in _CPP_SOURCE_SUFFIXES:
            return fname
    raise ScaffoldError(
        f"No C++ entry source found among copied files {copied}; expected "
        f"main.cpp."
    )


# --- Top-level driver ----------------------------------------------------


def scaffold(
    slug: str,
    language: str,
    exemplar: str,
    output_root: Path,
    benchmark_root: Path,
    examples_root: Path,
    mapping_path: Path,
    *,
    arch_tag: Optional[str] = None,
    local_sm86: bool = False,
) -> Path:
    """Scaffold one baseline; return the target directory path.

    ``arch_tag`` is an optional override for the local arch tag stamped
    into the stub README's ``Evidence`` field. ``None`` (default)
    triggers the live host probe via ``_detect_local_arch_tag``; tests
    pass a pinned value (e.g. ``"sm_86"``) for deterministic output.

    ``local_sm86`` emits an the design contract-conforming sm_86 solution spec
    (``target_hardware == ["LOCAL"]`` + explicit sm_86 gencode) for the
    sm_86 coverage DONE bar.
    """
    category = resolve_category_for_slug(slug, mapping_path)
    bench_dir = resolve_benchmark_dir(slug, category, benchmark_root)
    exemplar_dir = resolve_exemplar_dir(exemplar, examples_root)
    bench_name = derive_benchmark_name(slug)

    target_dir = output_root / category / slug / language
    if target_dir.exists():
        raise ScaffoldError(
            f"Target directory already exists: {target_dir}. Remove it "
            f"or pick a different slug/language before scaffolding."
        )

    copied = copy_exemplar_sources(exemplar_dir, target_dir)
    entry_path = _resolve_entry_path(language, copied)

    # Kernel-symbol rename.
    old_stem = discover_exemplar_prefix(exemplar_dir)
    new_stem = slug_identifier_stem(slug)
    if old_stem is not None and old_stem != new_stem:
        for fname in copied:
            path = target_dir / fname
            text = path.read_text(encoding="utf-8")
            text = rename_kernel_symbols(text, old_stem, new_stem)
            path.write_text(text, encoding="utf-8")

    # Destination-passing-style conversion on the entry source: rewrite
    # the entry signature to exactly match definition.json.inputs +
    # definition.json.outputs. The exemplar's body remains as-is; the
    # author is expected to hand-edit it to consume the new inputs and
    # write into the new outputs. Fail closed if the exemplar's entry
    # function cannot be located — silently emitting a baseline with a
    # mismatched signature would defeat the contract gate downstream.
    definition = _read_definition(bench_dir)
    input_names = _ordered_input_names(definition)
    output_names = _ordered_output_names(definition)
    entry_file_path = target_dir / entry_path
    entry_text = entry_file_path.read_text(encoding="utf-8")
    entry_text, did_convert = dps_convert_main(
        entry_text, input_names, output_names
    )
    if not did_convert:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise ScaffoldError(
            f"Could not locate a 'torch::Tensor run(' or 'void run(' "
            f"signature in exemplar entry {entry_path}. The scaffolder "
            f"requires a recognisable entry function to rewrite. Either "
            f"pick a different exemplar or add the entry function "
            f"manually before scaffolding."
        )
    entry_file_path.write_text(entry_text, encoding="utf-8")

    # Build the solution.json.
    sources = _sources_in_canonical_order(target_dir, entry_path)
    solution_dict = _build_solution_dict(
        slug=slug,
        language=language,
        bench_name=bench_name,
        entry_path=entry_path,
        sources=sources,
        local_sm86=local_sm86,
    )
    _validate_against_live_schema(solution_dict)
    (target_dir / "solution.json").write_text(
        json.dumps(solution_dict, indent=2) + "\n",
        encoding="utf-8",
    )
    _emit_parent_readme(
        parent_dir=target_dir.parent,
        slug=slug,
        category=category,
        bench_name=bench_name,
        language=language,
        arch_tag=arch_tag,
    )
    return target_dir


def _emit_parent_readme(
    parent_dir: Path,
    slug: str,
    category: str,
    bench_name: str,
    language: str,
    *,
    arch_tag: Optional[str] = None,
) -> None:
    """Write a stub parent README that satisfies the contract lint.

    Under the correctness-only landing contract (2026-05-13 re-baseline):

    - ``Evidence`` starts at ``Compile-checked-<arch>`` where
      ``<arch>`` is detected at scaffold-run time from the live host
      via ``_detect_local_arch_tag`` (override via the ``arch_tag``
      kwarg for tests). The harness later upgrades it to
      ``Ran-local-<arch>`` after a correctness-PASSED local run
      (speedup vs PyTorch reference is diagnostic only, not a landing
      gate). The framework adds ``Profiled-local-<arch>`` once the
      per-surface ``manifest.json`` emits without any ``failed``
      surface. Phase-2 work upgrades to ``Profiled-B200-sm_100a`` and
      finally ``Scored-B200-sm_100a``.
    - ``Local speedup vs reference`` is diagnostic; ``< 1.0`` is
      acceptable for a template solution.
    - ``SOL Score (online judge)`` is filled in only after Phase-2
      submission. Target band: 0.3 – 0.5+, but lower scores still
      land as accepted profiling inputs.
    - ``Counter-free artifacts`` lists which of the five analytic
      surfaces (NVBit, CUPTI Activity, Nsight + NVTX, static,
      sanitizer) emitted cleanly per the bundle's
      ``manifest.json``.

    ``Benchmark problem_dir`` cites the canonical downloader path
    (NOT the staged copy under ``.tmp_staged_benchmarks/``). The
    architecture-coupling caveats cover the required keyword groups:
    HBM3e / GDDR7 bandwidth, DSMEM / cluster-cooperative reduction,
    L2 carveout, and TCGen05 / TMEM divergence. The caveats are
    arch-agnostic — they describe the divergence axes between any
    local NCU-permitted host (sm_86 on the 3090 Ti, sm_120 on the
    5060 Laptop) and the B200 (sm_100a) Phase-2 target.
    """
    if (parent_dir / "README.md").is_file():
        # Pre-existing README (e.g. landed pilot upgraded in place) —
        # do not clobber.
        return
    subset = SUBSET_MAP[category]
    resolved_arch_tag = _detect_local_arch_tag(override=arch_tag)
    content = f"""# {slug}

| Field                       | Value                                                                  |
|-----------------------------|------------------------------------------------------------------------|
| Problem                     | <one-line description from the markdown spec>                          |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/{slug}.md`                |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/{subset}/{bench_name}/`                  |
| Variant                     | `{language}`                                                           |
| Evidence                    | `Compile-checked-{resolved_arch_tag}`                                  |
| Local speedup vs reference  | <diagnostic only; `< 1.0` is acceptable>                               |
| SOL Score (online judge)    | <filled after Phase-2 submission; target band 0.3 – 0.5+>              |
| Counter-free artifacts      | <which of {{nvbit, cupti, nsys, static, sanitizer}} emitted ok>        |

## Approach

<hand-port the official sol-execbench Python reference (and any
matching official Python solution under sol-execbench/examples/) into
this *.cu template; then describe the tiling / schedule / fusion
strategy / precision path here. Template solutions are profiling
targets, not optimized kernels.>

## Architecture-coupling caveats

The local NCU-permitted host (this scaffold detected `{resolved_arch_tag}`)
compiles and smoke-runs this baseline; Phase-2 counter-free profiling
and SOL Score submission run on B200 (sm_100a). Differences worth
noting for this pattern:

- HBM3e vs GDDR6X / GDDR7 bandwidth: bandwidth-bound kernels have a
  different ceiling on B200 than on the local consumer GPU; local
  wall-time is a weak proxy for B200 behavior.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out of
  scope for local consumer-arch builds.
- L2 carveout: B200's larger L2 may absorb tensors the local consumer
  GPU's smaller L2 cannot.
- TCGen05 / TMEM divergence: 5th-generation tensor cores and the
  dedicated TMEM register surface only exist on sm_100a; document any
  divergence here when the kernel touches them.
"""
    (parent_dir / "README.md").write_text(content, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/baselines_scaffold/scaffold.py",
        description=(
            "Copy an exemplar tree, DPS-convert the entry function, "
            "rename kernel symbols, and emit a Pydantic-valid solution.json."
        ),
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Internal slug, e.g. problem_001_fused_add_rmsnorm_h2048.",
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=SUPPORTED_LANGUAGES,
        help=(
            "Submission language. Restricted to cutlass/cuda_cpp per "
            "CLAUDE.md Rule 5; vendor libs (cuBLAS / cuDNN / CUB) link "
            "via compile_options.ld_flags on cuda_cpp sources."
        ),
    )
    parser.add_argument(
        "--exemplar",
        required=True,
        help=(
            "Relative path under sol-execbench/examples/, e.g. "
            "cuda_cpp/rmsnorm or cutlass/gemm."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Where to write baselines/<cat>/<slug>/<lang>/ (default: baselines/).",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root containing <subset>/<NNN_name>/definition.json.",
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=DEFAULT_EXAMPLES_ROOT,
        help="Root containing the exemplar trees (default: sol-execbench/examples).",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=MAPPING_PATH,
        help="Path to tools/sol_baselines_scaffold/sol_pattern_mapping.md.",
    )
    parser.add_argument(
        "--arch-tag",
        type=str,
        default=None,
        help=(
            "Override the arch tag stamped into the stub README's "
            "Evidence label (e.g. sm_86, sm_120). Defaults to the live "
            "host probe via arch_resolver.arch_tag_from_host_kind; "
            "set explicitly to scaffold on a host whose probe would "
            "otherwise return sm_unknown (CI / CPU-only)."
        ),
    )
    parser.add_argument(
        "--local-sm86",
        action="store_true",
        help=(
            "Emit an A" "C-4-conforming sm_86 solution spec: "
            "target_hardware=[\"LOCAL\"] plus an explicit "
            "-gencode arch=compute_86,code=sm_86 cuda_cflag (the sm_86 "
            "coverage DONE bar). Default emits the dual [\"LOCAL\",\"B200\"] "
            "template spec."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        target = scaffold(
            slug=args.slug,
            language=args.language,
            exemplar=args.exemplar,
            output_root=args.output_root,
            benchmark_root=args.benchmark_root,
            examples_root=args.examples_root,
            mapping_path=args.mapping_path,
            arch_tag=args.arch_tag,
            local_sm86=args.local_sm86,
        )
    except ScaffoldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"scaffolded baseline: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
