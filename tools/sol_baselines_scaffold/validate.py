#!/usr/bin/env python3
"""Two-layer contract gate for SOL-ExecBench baselines.

Layer (a) — live Pydantic schema parse via
``sol_execbench.core.data.{solution,definition}``. Catches generic
schema violations the upstream evaluator would itself reject:
lowercase ``target_hardware``, empty ``sources[]``, malformed
``entry_point``, mixed Python/C++ languages, file-suffix mismatch,
duplicate or absent entry source.

Layer (b) — project lint that the live schema permits but our
submission surface forbids:

- **Rule 5 (vendor-as-language)** — only ``cutlass`` and ``cuda_cpp``
  are valid ``spec.languages`` values for our submission surface.
  ``cudnn`` / ``cublas`` / ``pytorch`` / ``triton`` / ``cute_dsl`` /
  ``cutile`` / ``cudnn_frontend`` are rejected; vendor libraries link
  via ``compile_options.ld_flags`` on a ``cuda_cpp`` source instead.
- **Destination-passing-style argument-order check** — when
  ``spec.destination_passing_style`` is true, the entry function's
  parameter names must match the concatenation of
  ``definition.json.inputs`` (insertion order) followed by
  ``definition.json.outputs`` (insertion order). When false, only
  the inputs need to appear in that order.
- **Forbidden-surface scan (Rules 1, 2, 6, 7)** — every emitted
  source file (and ``compile_options.cflags`` / ``cuda_cflags`` /
  ``ld_flags``) is scanned for: NCU invocations (``ncu``,
  ``nv-nsight-cu-cli``); ``nvidia-smi --lock-gpu-clocks`` /
  ``--lock-memory-clocks``; the ``sol-execbench --lock-clocks``
  flag; user CUDA stream creation (``cudaStreamCreate`` /
  ``c10::Stream`` instantiation); ``torch.jit.fork``; ``data_ptr``-
  keyed caches; monkey-patched ``Event.elapsed_time`` /
  ``do_bench``; CUDA Graphs (``cudaGraphInstantiate`` /
  ``torch.cuda.CUDAGraph``) without a co-located per-problem note.

Exit codes:
- ``0`` — every checked baseline passes both layers.
- ``2`` — any failure (the offending baseline and reason print to
  stderr).

Usage:
    python scripts/baselines_scaffold/validate.py \\
        --baseline baselines/FIB/problem_001_fused_add_rmsnorm_h2048/cuda_cpp

    python scripts/baselines_scaffold/validate.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from baselines_scaffold.generate import ( # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    SUBSET_MAP,
)


# Our submission surface; the live Pydantic enum is more permissive.
_PROJECT_LANGUAGES = frozenset({"cutlass", "cuda_cpp"})

# Vendor-language tokens that the live enum accepts but we reject.
_VENDOR_LANGUAGE_TOKENS = frozenset(
    {"cudnn", "cublas", "pytorch", "triton", "cute_dsl", "cutile", "cudnn_frontend"}
)

# Per-issue forbidden-surface scan patterns. Each entry is
# ``(label, compiled_re, applies_to_sources_only)``. ``applies_to``
# distinguishes patterns we ONLY want to fire on source files (so the
# JSON's own description / documentation text in a README can mention
# them) from those we also want to catch in solution.json text.
@dataclass(frozen=True)
class ForbiddenPattern:
    label: str
    pattern: re.Pattern
    rule: str
    detail: str


_FORBIDDEN_PATTERNS: Tuple[ForbiddenPattern, ...] = (
    ForbiddenPattern(
        label="ncu invocation",
        pattern=re.compile(r"(?<![A-Za-z0-9_])(?:ncu|nv-nsight-cu-cli)\b"),
        rule="Rule 1",
        detail=(
            "NCU is forbidden on every device (CLAUDE.md Rule 1). The "
            "5060.yaml characterisation has a scoped one-time exception "
            "that does not extend to baselines or the harness."
        ),
    ),
    ForbiddenPattern(
        label="nvidia-smi --lock-gpu-clocks",
        pattern=re.compile(r"nvidia-smi\s+[^\n]*--lock-gpu-clocks"),
        rule="Rule 2",
        detail=(
            "Clock-policy mutation is forbidden (CLAUDE.md Rule 2). "
            "Clock policy belongs to the leaderboard scoring service."
        ),
    ),
    ForbiddenPattern(
        label="nvidia-smi --lock-memory-clocks",
        pattern=re.compile(r"nvidia-smi\s+[^\n]*--lock-memory-clocks"),
        rule="Rule 2",
        detail="Clock-policy mutation forbidden (CLAUDE.md Rule 2).",
    ),
    ForbiddenPattern(
        label="sol-execbench --lock-clocks",
        pattern=re.compile(r"--lock-clocks\b"),
        rule="Rule 2",
        detail=(
            "The sol-execbench --lock-clocks CLI flag is forbidden "
            "(CLAUDE.md Rule 2); the scoring service handles clock policy."
        ),
    ),
    ForbiddenPattern(
        label="cudaStreamCreate",
        pattern=re.compile(
            r"\bcudaStreamCreate(?:WithFlags|WithPriority)?\s*\("
        ),
        rule="Rule 7",
        detail=(
            "User CUDA stream creation is forbidden (CLAUDE.md Rule 7). "
            "Use at::cuda::getCurrentCUDAStream() exclusively."
        ),
    ),
    ForbiddenPattern(
        label="torch.jit.fork",
        pattern=re.compile(r"\btorch\.jit\.fork\b"),
        rule="Rule 7",
        detail="torch.jit.fork is forbidden (CLAUDE.md Rule 7).",
    ),
    ForbiddenPattern(
        label="data_ptr-keyed cache",
        pattern=re.compile(
            r"\b(?:cache|memo|memoize)\b[^\n]*\bdata_ptr\s*\(",
            re.IGNORECASE,
        ),
        rule="Rule 6",
        detail=(
            "data_ptr()-keyed caches go stale under the harness's "
            "256B pointer-shift on every iteration (CLAUDE.md Rule 6)."
        ),
    ),
    ForbiddenPattern(
        label="monkey-patched Event.elapsed_time",
        pattern=re.compile(r"Event\.elapsed_time\s*=\s*"),
        rule="Rule 6",
        detail=(
            "Reassigning Event.elapsed_time is reward-hacking "
            "(CLAUDE.md Rule 6)."
        ),
    ),
    ForbiddenPattern(
        label="monkey-patched do_bench",
        pattern=re.compile(r"\bdo_bench\s*=\s*lambda"),
        rule="Rule 6",
        detail=(
            "Reassigning do_bench is reward-hacking (CLAUDE.md Rule 6)."
        ),
    ),
    ForbiddenPattern(
        label="cudaGraphInstantiate",
        pattern=re.compile(r"\bcudaGraphInstantiate\b"),
        rule="Rule 7",
        detail=(
            "CUDA Graphs require a co-located per-problem note proving "
            "harness stream-injection survives (CLAUDE.md Rule 7). "
            "Document the workaround in a comment block in the same "
            "source file."
        ),
    ),
    ForbiddenPattern(
        label="torch.cuda.CUDAGraph",
        pattern=re.compile(r"\btorch\.cuda\.CUDAGraph\b"),
        rule="Rule 7",
        detail=(
            "CUDA Graphs require a co-located per-problem note "
            "(CLAUDE.md Rule 7)."
        ),
    ),
    ForbiddenPattern(
        label="docker invocation",
        pattern=re.compile(r"(?<![A-Za-z0-9_])docker\s+(?:run|build|image|info|exec|pull|push)\b"),
        rule="Native dispatch",
        detail=(
            "Docker invocations are forbidden in landed sources and "
            "compile flags: the harness drives the upstream evaluator "
            "natively. Mention of docker-the-string in historical-context "
            "disclaimer blocks is acceptable, but bare invocations are not."
        ),
    ),
    ForbiddenPattern(
        label="--privileged flag",
        pattern=re.compile(r"(?<![A-Za-z0-9_])--privileged\b"),
        rule="Native dispatch",
        detail=(
            "Docker --privileged is forbidden; the native dispatch path "
            "has no container runtime."
        ),
    ),
    ForbiddenPattern(
        label="--gpus flag",
        pattern=re.compile(r"(?<![A-Za-z0-9_])--gpus\b"),
        rule="Native dispatch",
        detail=(
            "Docker --gpus is forbidden; CUDA visibility is host-level "
            "under native dispatch."
        ),
    ),
    ForbiddenPattern(
        label="sol-execbench image reference",
        pattern=re.compile(r"sol-execbench:latest\b"),
        rule="Native dispatch",
        detail=(
            "The sol-execbench:latest image reference is retired; "
            "the evaluator is installed natively via `uv sync` of the "
            "in-tree sol-execbench/ package."
        ),
    ),
)


# Token strings that, when present in the SAME file as a CUDA-Graph
# match, vouch for the workaround note. Both English and German style
# acceptable since these are author-written documentation snippets.
_CUDA_GRAPH_NOTE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"CUDA[\s_-]*Graphs?\b.*\bnote\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"stream[\s_-]*injection\b", re.IGNORECASE),
)


class ValidationError(Exception):
    """Validate failure with an actionable message."""


# --- Layer (a): Pydantic schema -----------------------------------------


def _import_pydantic_models():
    """Import the live ``Solution`` / ``Definition`` Pydantic models."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "sol-execbench" / "src"))
        from sol_execbench.core.data.solution import Solution # noqa: E402
        from sol_execbench.core.data.definition import Definition # noqa: E402
    except ImportError as exc:
        raise ValidationError(
            f"Cannot import live Pydantic models: {exc}. Ensure "
            f"sol-execbench/src is importable."
        ) from exc
    return Solution, Definition


def _normalise_definition_for_pydantic(
    definition_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Coerce upstream-convention empty strings to ``None``.

    The downloaded SOL-ExecBench benchmark assets use ``"hf_id": ""``
    on every problem without a HuggingFace model ID, but the live
    ``Definition`` Pydantic model declares ``hf_id`` as
    ``Optional[NonEmptyString]`` (so empty string is invalid; ``None``
    is the correct sentinel). Normalise to ``None`` so the schema
    accepts the upstream data verbatim.
    """
    normalised = dict(definition_dict)
    for key in ("hf_id", "op_type", "custom_inputs_entrypoint"):
        if normalised.get(key) == "":
            normalised[key] = None
    return normalised


def _pydantic_parse(
    solution_dict: Dict[str, Any],
    definition_dict: Dict[str, Any],
) -> Tuple[Any, Any]:
    Solution, Definition = _import_pydantic_models()
    try:
        sol = Solution.model_validate(solution_dict)
    except Exception as exc:
        raise ValidationError(
            f"Pydantic Solution validation failed: {exc}"
        ) from exc
    try:
        defn = Definition.model_validate(
            _normalise_definition_for_pydantic(definition_dict)
        )
    except Exception as exc:
        raise ValidationError(
            f"Pydantic Definition validation failed: {exc}"
        ) from exc
    return sol, defn


# --- Layer (b.1): vendor-as-language reject -----------------------------


def _check_vendor_languages(solution_dict: Dict[str, Any]) -> None:
    languages = solution_dict.get("spec", {}).get("languages", [])
    if not isinstance(languages, list):
        raise ValidationError(
            f"spec.languages must be a list, got {languages!r}."
        )
    bad = [lang for lang in languages if lang in _VENDOR_LANGUAGE_TOKENS]
    if bad:
        raise ValidationError(
            f"Vendor library names {bad} are not valid spec.languages "
            f"for our submission surface (CLAUDE.md Rule 5). Use "
            f"['cuda_cpp'] and link the vendor library via "
            f"compile_options.ld_flags instead."
        )
    unknown = [lang for lang in languages if lang not in _PROJECT_LANGUAGES]
    if unknown:
        raise ValidationError(
            f"spec.languages {unknown} not in the project submission "
            f"surface ({sorted(_PROJECT_LANGUAGES)}); CLAUDE.md Rule 5 "
            f"restricts to cutlass/cuda_cpp even though the live "
            f"Pydantic enum accepts more."
        )


# --- Layer (b.2): destination-passing-style signature-order check -------


def _parse_run_params(entry_text: str) -> Optional[List[str]]:
    """Extract the parameter-name list from ``void run(...)`` or
    ``torch::Tensor run(...)`` in the entry source.

    Returns ``None`` when no ``run(`` signature is found (the validator
    raises a clear error in that case). Order is preserved.
    """
    match = re.search(
        r"\b(?:void|torch::Tensor)\s+run\s*\(", entry_text
    )
    if match is None:
        return None
    open_idx = entry_text.index("(", match.end() - 1)
    depth = 0
    close_idx: Optional[int] = None
    for i in range(open_idx, len(entry_text)):
        c = entry_text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx is None:
        return None
    params_text = entry_text[open_idx + 1 : close_idx]
    # Each parameter declaration ends with an identifier; strip type
    # decorations and default values.
    names: List[str] = []
    for raw in _split_params_at_commas(params_text):
        raw = raw.strip()
        if not raw:
            continue
        # Drop default-value clauses.
        if "=" in raw:
            raw = raw.split("=", 1)[0].strip()
        ident_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", raw)
        if not ident_match:
            return None
        names.append(ident_match.group(1))
    return names


def _split_params_at_commas(params: str) -> List[str]:
    """Split on top-level commas, ignoring those nested inside template angle brackets or parens."""
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for c in params:
        if c in "(<[":
            depth += 1
            buf.append(c)
        elif c in ")>]":
            depth -= 1
            buf.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    if buf:
        parts.append("".join(buf))
    return parts


def _check_dps_signature_order(
    solution_dict: Dict[str, Any],
    entry_text: str,
    definition_dict: Dict[str, Any],
) -> None:
    spec = solution_dict.get("spec", {})
    dps = spec.get("destination_passing_style", True)
    expected_inputs = list(definition_dict.get("inputs", {}).keys())
    expected_outputs = list(definition_dict.get("outputs", {}).keys())
    expected = expected_inputs + expected_outputs if dps else expected_inputs

    actual = _parse_run_params(entry_text)
    if actual is None:
        raise ValidationError(
            "Could not locate `void run(` or `torch::Tensor run(` in the "
            "entry source — DPS signature-order check cannot proceed."
        )
    if actual != expected:
        raise ValidationError(
            f"Entry-function parameter order mismatch: "
            f"expected {expected} (definition.json inputs"
            f"{'+ outputs' if dps else ''} in insertion order); "
            f"got {actual} from the entry source. "
            f"Destination-passing-style={'on' if dps else 'off'}."
        )


# --- Layer (b.3): forbidden-surface scan --------------------------------


def _scan_text(label: str, text: str) -> List[Tuple[str, str, str]]:
    """Return ``[(label, rule, detail), ...]`` for every forbidden hit.

    Pattern entries decide membership; CUDA-Graph variants additionally
    pass when a co-located note vouches.
    """
    hits: List[Tuple[str, str, str]] = []
    for fp in _FORBIDDEN_PATTERNS:
        if not fp.pattern.search(text):
            continue
        if fp.label in ("cudaGraphInstantiate", "torch.cuda.CUDAGraph"):
            if any(p.search(text) for p in _CUDA_GRAPH_NOTE_PATTERNS):
                continue
        hits.append((f"{label}:{fp.label}", fp.rule, fp.detail))
    return hits


def _check_disk_matches_solution(
    solution_dict: Dict[str, Any],
    baseline_dir: Path,
) -> Dict[str, str]:
    """Confirm every ``sources[].path`` resolves on disk and the disk
    content matches ``sources[].content`` byte-for-byte.

    Returns the ``{rel_path: disk_text}`` mapping so the caller can run
    forbidden-surface lint against the disk text (the canonical source
    of truth — the embedded JSON content is a derived view that
    ``pack.py`` is responsible for regenerating).

    Fails closed on any of:

    - A ``sources[].path`` is missing from disk under ``baseline_dir``.
    - A disk file is not UTF-8 readable.
    - Disk content does not match ``sources[].content`` byte-for-byte
      (this catches forbidden edits the author made to the disk file
      without re-running ``pack.py`` — the project lint layer must
      apply to the disk content rather than the stale embedded copy).
    """
    sources = solution_dict.get("sources", [])
    disk_texts: Dict[str, str] = {}
    drift: List[str] = []
    for src in sources:
        rel = src.get("path", "")
        if not isinstance(rel, str) or not rel:
            raise ValidationError(
                f"solution.json sources[] entry missing 'path' string: {src!r}."
            )
        disk_path = baseline_dir / rel
        if not disk_path.is_file():
            raise ValidationError(
                f"Source file {rel} listed in solution.json.sources[] is "
                f"missing from disk under {baseline_dir}. Re-run "
                f"pack.py after restoring the file."
            )
        try:
            disk_text = disk_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(
                f"Source file {rel} under {baseline_dir} is not UTF-8 "
                f"readable: {exc}."
            ) from exc
        if disk_text != src.get("content", ""):
            drift.append(rel)
        disk_texts[rel] = disk_text
    if drift:
        raise ValidationError(
            f"solution.json.sources[].content is stale for "
            f"{drift} under {baseline_dir}. Re-run pack.py to refresh "
            f"the embedded content from disk before validating."
        )
    return disk_texts


def _check_forbidden_surfaces(
    solution_dict: Dict[str, Any],
    baseline_dir: Path,
    disk_texts: Dict[str, str],
) -> None:
    """Scan disk content (not embedded JSON) for forbidden surfaces.

    Reading from disk is the canonical path because authors may edit
    source files post-pack, and a stale embedded ``content`` would
    otherwise hide a forbidden insertion. ``_check_disk_matches_solution``
    already guarantees disk and JSON content agree before we get here;
    using ``disk_texts`` directly makes the data flow explicit.
    """
    all_hits: List[Tuple[str, str, str]] = []
    for rel_path, content in disk_texts.items():
        all_hits.extend(_scan_text(rel_path, content))
    # Also scan compile_options text fields (these live in the JSON
    # only — there is no on-disk counterpart for them).
    spec = solution_dict.get("spec", {})
    co = spec.get("compile_options") or {}
    for key in ("cflags", "cuda_cflags", "ld_flags"):
        flags = co.get(key) or []
        joined = " ".join(str(f) for f in flags)
        all_hits.extend(_scan_text(f"compile_options.{key}", joined))
    if all_hits:
        lines = [
            f" - [{rule}] {label}: {detail}"
            for (label, rule, detail) in all_hits
        ]
        raise ValidationError(
            f"Forbidden surface(s) detected in {baseline_dir}:\n"
            + "\n".join(lines)
        )


# --- Layer (b.4): parent README lint -------------------------------------


# Fields the parent README's field-table must declare. The values
# associated with each field are validated separately.
#
# Re-anchor 2026-05-13: the field set is aligned with the new
# correctness-only landing contract — `Variant` (not `Best variant`,
# since template solutions are not optimized), `Local speedup vs
# reference` recorded as diagnostic data only, `SOL Score (online
# judge)` filled in only after Phase-2 submission, and `Counter-free
# artifacts` listing which of the six surfaces emitted cleanly in the
# `manifest.json` bundle.
_REQUIRED_README_FIELDS: Tuple[str, ...] = (
    "Problem",
    "Reference spec",
    "Benchmark problem_dir",
    "Variant",
    "Evidence",
    "Local speedup vs reference",
    "SOL Score (online judge)",
    "Counter-free artifacts",
)

# Evidence labels accepted under the post-re-baseline (2026-05-13)
# correctness-only contract. Profiled-* labels are manifest-based
# (see CLAUDE.md §"Counter-free artifact manifest"). Scored-B200-sm_100a
# is no longer forbidden — Phase-2 submissions earn it once the online
# judge returns a SOL Score (target band 0.3–0.5+, but lower scores
# still land as accepted profiling inputs).
_ALLOWED_EVIDENCE_LABELS: Tuple[str, ...] = (
    "Compile-checked-sm_86",
    "Ran-local-sm_86",
    "Profiled-local-sm_86",
    "Compile-checked-sm_90",
    "Ran-local-sm_90",
    "Profiled-local-sm_90",
    "Compile-checked-sm_120",
    "Ran-local-sm_120",
    "Profiled-local-sm_120",
    "Profiled-local-complete",
    "Profiled-B200-sm_100a",
    "Profiled-B200-complete",
    "Scored-B200-sm_100a",
    "Not-B200-validated",
)
_FORBIDDEN_EVIDENCE_LABELS: Tuple[str, ...] = ()

# Required caveat keywords. Each tuple is a set of synonyms; at least
# one of each tuple must appear (case-insensitive substring match) in
# the ``## Architecture-coupling caveats`` section. The HBM3e and GDDR7
# pair is intentionally cross-referenced so authors can phrase the
# bandwidth gap either way.
_REQUIRED_CAVEAT_SYNONYMS: Tuple[Tuple[str, ...], ...] = (
    ("HBM3e", "HBM3"),
    ("GDDR7",),
    ("DSMEM", "cluster"),
    ("L2 carveout", "L2 partition", "L2 cache"),
    ("TCGen05", "TMEM"),
)


_README_FIELD_ROW_RE = re.compile(
    r"^\|\s*(?P<field>[^|]+?)\s*\|\s*(?P<value>[^|]*?)\s*\|\s*$",
    re.MULTILINE,
)
_CAVEATS_HEADING_RE = re.compile(
    r"^##\s+Architecture-coupling caveats\s*$",
    re.MULTILINE,
)


def _parse_readme_fields(text: str) -> Dict[str, str]:
    """Return ``{field_name: value}`` from the README's leading table."""
    fields: Dict[str, str] = {}
    for match in _README_FIELD_ROW_RE.finditer(text):
        name = match.group("field").strip()
        value = match.group("value").strip()
        if not name or name == "Field" or set(name) <= {"-"}:
            # Skip the header row and separator row.
            continue
        fields[name] = value
    return fields


def _extract_caveats_section(text: str) -> Optional[str]:
    """Return the body of ``## Architecture-coupling caveats`` (no header)."""
    match = _CAVEATS_HEADING_RE.search(text)
    if match is None:
        return None
    rest = text[match.end():]
    # Stop at the next top-level heading (``## ...``).
    end_match = re.search(r"^##\s+\S", rest, re.MULTILINE)
    if end_match is None:
        return rest
    return rest[: end_match.start()]


def _expected_canonical_problem_dir(
    solution_dict: Dict[str, Any],
    baseline_dir: Path,
) -> str:
    """Return the canonical benchmark path the README must cite.

    Shape: ``sol-execbench/data/benchmark/<subset>/<NNN_name>/`` (with
    trailing slash). Uses the solution's ``definition`` field for the
    NNN_name and the baseline path's category for the subset.
    """
    parents = list(baseline_dir.resolve().parents)
    category = parents[1].name if len(parents) >= 2 else ""
    subset = SUBSET_MAP.get(category, "")
    bench_name = solution_dict.get("definition", "")
    return f"sol-execbench/data/benchmark/{subset}/{bench_name}/"


def _check_parent_readme(
    solution_dict: Dict[str, Any],
    baseline_dir: Path,
) -> None:
    """Lint ``baselines/<cat>/<slug>/README.md`` for landed baselines.

    Enforces the parent-README contract: every landed baseline has a
    parent README whose field table declares each of the canonical
    fields (``Problem``, ``Reference spec``, ``Benchmark problem_dir``,
    ``Variant``, ``Evidence``, ``Local speedup vs reference``,
    ``SOL Score (online judge)``, ``Counter-free artifacts``);
    ``Benchmark problem_dir`` cites the canonical downloader path
    (NOT a staged copy under ``.tmp_staged_benchmarks/``); ``Evidence``
    is one of the allowed labels in ``_ALLOWED_EVIDENCE_LABELS``
    (Compile-checked-sm_120 / Ran-local-sm_120 / Profiled-local-* /
    Profiled-B200-* / Scored-B200-sm_100a / Not-B200-validated); the
    ``Architecture-coupling caveats`` section names HBM3e/HBM3 vs
    GDDR7, DSMEM / cluster, L2 carveout/partition/cache, and
    TCGen05/TMEM.

    READMEs without a leading field table are rejected; missing
    fields, mis-cited problem_dir paths, disallowed evidence labels,
    and missing caveat keywords each produce a single failure line.
    """
    readme_path = baseline_dir.parent / "README.md"
    if not readme_path.is_file():
        raise ValidationError(
            f"Parent README missing at {readme_path}. Per parent-README contract every "
            f"landed baseline needs a per-problem README following the "
            f"template at baselines/README.md (lines 54-80)."
        )
    text = readme_path.read_text(encoding="utf-8")
    fields = _parse_readme_fields(text)
    failures: List[str] = []

    for required in _REQUIRED_README_FIELDS:
        if required not in fields:
            failures.append(f"missing required README field {required!r}")

    expected_bench = _expected_canonical_problem_dir(
        solution_dict, baseline_dir
    )
    bench_value = fields.get("Benchmark problem_dir", "")
    # Tolerate surrounding backticks or trailing slash absence.
    stripped = bench_value.strip().strip("`").rstrip("/") + "/"
    if stripped != expected_bench:
        failures.append(
            f"Benchmark problem_dir must cite the canonical downloader "
            f"path {expected_bench!r}; got {bench_value!r}. Staged copies "
            f"under .tmp_staged_benchmarks/ are not allowed in the "
            f"parent README (parent-README contract)."
        )

    evidence = fields.get("Evidence", "")
    evidence_stripped = evidence.strip().strip("`")
    if evidence_stripped not in _ALLOWED_EVIDENCE_LABELS:
        if evidence_stripped in _FORBIDDEN_EVIDENCE_LABELS:
            failures.append(
                f"Evidence label {evidence_stripped!r} is rejected this "
                f"plan (leaderboard submission deferred). Use "
                f"{_ALLOWED_EVIDENCE_LABELS} instead."
            )
        else:
            failures.append(
                f"Evidence label {evidence_stripped!r} not in the "
                f"allowed set {_ALLOWED_EVIDENCE_LABELS}."
            )

    caveats = _extract_caveats_section(text)
    if caveats is None:
        failures.append(
            "missing '## Architecture-coupling caveats' section in parent README."
        )
    else:
        caveats_lower = caveats.lower()
        for synonym_group in _REQUIRED_CAVEAT_SYNONYMS:
            if not any(s.lower() in caveats_lower for s in synonym_group):
                failures.append(
                    f"caveats section missing required keyword "
                    f"(one of {list(synonym_group)})."
                )

    if failures:
        lines = [f" - {msg}" for msg in failures]
        raise ValidationError(
            f"Parent README ({readme_path}) failed parent-README contract lint:\n"
            + "\n".join(lines)
        )


# --- Top-level driver ----------------------------------------------------


def _load_definition(
    solution_dict: Dict[str, Any],
    baseline_dir: Path,
    benchmark_root: Path,
) -> Tuple[Path, Dict[str, Any]]:
    """Resolve the matching ``definition.json`` for this baseline.

    Uses the parent path's category and the solution.json ``definition``
    field (the downloader's directory name) — but if those disagree we
    still trust the explicit ``definition`` field; the category comes
    from the path layout (`baselines/<cat>/<slug>/<lang>/`).
    """
    bench_name = solution_dict.get("definition", "")
    if not bench_name:
        raise ValidationError(
            "solution.json is missing 'definition' field (benchmark dir name)."
        )
    parents = list(baseline_dir.resolve().parents)
    # baseline_dir is .../baselines/<category>/<slug>/<language>
    # so parents are [..., <slug>, <category>, baselines, ...]
    if len(parents) < 2:
        raise ValidationError(
            f"Baseline path {baseline_dir} does not match "
            f"baselines/<category>/<slug>/<language>/ layout."
        )
    category = parents[1].name
    subset = SUBSET_MAP.get(category)
    if subset is None:
        raise ValidationError(
            f"Baseline directory parent category {category!r} unknown; "
            f"expected one of {sorted(SUBSET_MAP)}."
        )
    bench_path = benchmark_root / subset / bench_name / "definition.json"
    if not bench_path.is_file():
        raise ValidationError(
            f"Matching benchmark definition.json not found at {bench_path}. "
            f"Run ./sol-execbench/scripts/download_data.sh to populate, or "
            f"check the solution's definition field."
        )
    return bench_path, json.loads(bench_path.read_text(encoding="utf-8"))


def _entry_text_from_solution(
    solution_dict: Dict[str, Any],
    baseline_dir: Path,
) -> str:
    spec = solution_dict.get("spec", {})
    entry_point = spec.get("entry_point", "")
    if "::" not in entry_point:
        raise ValidationError(
            f"spec.entry_point {entry_point!r} missing '::' separator."
        )
    entry_path = entry_point.split("::", 1)[0]
    disk_path = baseline_dir / entry_path
    if not disk_path.is_file():
        raise ValidationError(
            f"Entry source {entry_path} not on disk under {baseline_dir}."
        )
    return disk_path.read_text(encoding="utf-8")


def validate_baseline(
    baseline_dir: Path,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
) -> None:
    """Run both layers; raise ``ValidationError`` on any failure."""
    solution_path = baseline_dir / "solution.json"
    if not solution_path.is_file():
        raise ValidationError(
            f"solution.json not found at {solution_path}; run scaffold.py "
            f"to materialise the baseline first."
        )
    solution_dict = json.loads(solution_path.read_text(encoding="utf-8"))
    _, definition_dict = _load_definition(
        solution_dict, baseline_dir, benchmark_root
    )
    # Layer (a): Pydantic
    _pydantic_parse(solution_dict, definition_dict)
    # Layer (b): project lint. Disk content is the canonical source of
    # truth: confirm sources[].path resolves on disk, the disk content
    # matches sources[].content byte-for-byte, then scan the disk text
    # for vendor-as-language / signature-order / forbidden-surface
    # issues.
    disk_texts = _check_disk_matches_solution(solution_dict, baseline_dir)
    _check_vendor_languages(solution_dict)
    entry_text = _entry_text_from_solution(solution_dict, baseline_dir)
    _check_dps_signature_order(solution_dict, entry_text, definition_dict)
    _check_forbidden_surfaces(solution_dict, baseline_dir, disk_texts)
    _check_parent_readme(solution_dict, baseline_dir)


def _iter_all_baselines(output_root: Path) -> Iterable[Path]:
    if not output_root.is_dir():
        return
    for category_dir in sorted(output_root.iterdir()):
        if not category_dir.is_dir() or category_dir.name not in SUBSET_MAP:
            continue
        for slug_dir in sorted(category_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            for lang_dir in sorted(slug_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                if (lang_dir / "solution.json").is_file():
                    yield lang_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/baselines_scaffold/validate.py",
        description=(
            "Two-layer contract gate: live Pydantic Solution/Definition "
            "parse + project lint (vendor-as-language reject, DPS "
            "signature-order check, forbidden-surface scan)."
        ),
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--baseline",
        type=Path,
        help="Path to one baselines/<cat>/<slug>/<lang>/ directory.",
    )
    grp.add_argument(
        "--all",
        action="store_true",
        help="Walk every baseline under --output-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root containing baselines/<cat>/<slug>/<lang>/.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root containing <subset>/<NNN_name>/definition.json.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.all:
        targets = list(_iter_all_baselines(args.output_root))
        if not targets:
            print(
                f"warning: no baselines found under {args.output_root}",
                file=sys.stderr,
            )
            return 0
    else:
        targets = [args.baseline]
    failures: List[Tuple[Path, str]] = []
    for target in targets:
        try:
            validate_baseline(target, args.benchmark_root)
        except ValidationError as exc:
            failures.append((target, str(exc)))
    if failures:
        for target, msg in failures:
            print(f"FAIL {target}: {msg}", file=sys.stderr)
        return 2
    for target in targets:
        print(f"PASS {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
