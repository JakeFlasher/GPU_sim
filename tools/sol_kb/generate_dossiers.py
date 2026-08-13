#!/usr/bin/env python3
"""
generate_dossiers.py — emit the 231 missing problem dossiers under
`.claude/knowledge/problems/`.

`tools/sol_baselines_scaffold/sol_pattern_mapping.md` has 235 rows
(slug → category → primary_pattern → secondary_patterns → raw_source →
notes). This script:

  1. Parses that mapping.
  2. For each slug not already present under `.claude/knowledge/problems/`,
     emits a dossier driven by the mapping row and a brief description
     derived from the spec under `references/NVIDIA_Manuals/solbench_problems/`.
  3. The emitted dossiers satisfy the hardened linter:
       - 7 required frontmatter fields.
       - Out-of-scope block enumerates ALL canonical Rule-1-through-8
         phrases (matches `tools/sol_kb/lint_cards.py`
         `FORBIDDEN_SURFACE_CLAUSES`).
       - Body mentions both `sm_120` and `sm_100a`, cites
         `CLAUDE.md § Architecture Portability Discipline`, and includes
         the "Evidence-label vocabulary" pointer phrase.
       - No status-field tokens (the status-field rule forbids them in
         problems/ dossiers).
       - No numeric latency / bandwidth / SOL Score values (which would
         force a `Latency authority`).

Usage:  python3 tools/sol_kb/generate_dossiers.py [--root REPO_ROOT]
                                                 [--force]
                                                 [--limit N]

The four Round-1 hand-authored dossiers (`problem_004_gemm_n128_k2048`,
`problem_080_moe_complete_layer_with_shared_expert_backward`,
`problem_004_fp8_moe_expert_linear`,
`problem_012_gqa_paged_decode_h32_kv4_d128_ps1`) are preserved
unless `--force` is passed; the script reports them as `skip (exists)`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Match a mapping row:
# | slug | L1|L2|Quant|FIB | primary | secondary | raw_source | notes |
MAPPING_ROW_RE = re.compile(
    r"^\|\s*(problem_[A-Za-z0-9_]+)\s*"     # slug
    r"\|\s*(L1|L2|Quant|FIB)\s*"            # category
    r"\|\s*([A-Za-z0-9_]+)\s*"              # primary_pattern (may be tbd_*)
    r"\|\s*([A-Za-z0-9_;\s]+?)\s*"          # secondary_patterns ("none" or "a;b")
    r"\|\s*(references/[^\|]+?)\s*"         # raw_source
    r"\|\s*([^\|]+?)\s*\|$"                 # notes
)


_ACTIVE_PATH_REWRITES = (
    (
        re.compile(r"(?<![A-Za-z0-9_./-])baselines/"),
        "workloads/sol_execbench_baselines/",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9_./-])sol-execbench/"),
        "tools/sol_execbench/",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9_./-])cutlass/"),
        "third_party/cutlass/",
    ),
)


def _active_text(text: str) -> str:
    """Translate pre-rebase repository roots without rewriting nested paths."""
    for pattern, replacement in _ACTIVE_PATH_REWRITES:
        text = pattern.sub(replacement, text)
    return text


_NESTED_THIRD_PARTY_CUTLASS_RE = re.compile(
    r"workloads/sol_execbench_baselines/(?:L1|L2|Quant|FIB)/"
    r"[A-Za-z0-9_]+/third_party/cutlass/"
)


def _assert_dossier_cutlass_path_contract(problems_dir: Path) -> None:
    """Reject a third-party CUTLASS root accidentally nested in a solution."""
    offenders = []
    for path in sorted(problems_dir.glob("problem_*.md")):
        if _NESTED_THIRD_PARTY_CUTLASS_RE.search(
            path.read_text(encoding="utf-8")
        ):
            offenders.append(path.name)
    if offenders:
        names = ", ".join(offenders)
        raise RuntimeError(
            f"solution-local CUTLASS path contains third_party/cutlass: {names}"
        )


FORBIDDEN_SURFACE_BLOCK = """\
  - Python / Triton / CuTe DSL Python / cuTile Python / cudnn_frontend Python / torch.compile submissions (CLAUDE.md Rule 5).
  - NCU / `nv-nsight-cu-cli` / `.ncu-rep`/`.ncu-repz` artefacts on any device, including local 5060 (CLAUDE.md Rule 1).
  - `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` / persistence / power / MIG / compute-mode mutation (CLAUDE.md Rule 2).
  - Privileged CUPTI: Events, Metrics, Range Profiling, PM Sampling, PerfWorks (CLAUDE.md Rule 3; CUPTI Activity remains allowed).
  - Host-driver work: NVIDIA driver install, DKMS, MOK, `modprobe nvidia`, kernel-module parameter toggles, reboot workflows (CLAUDE.md Rule 4).
  - User CUDA streams, background host threads, `torch.jit.fork`, null-stream side effects, CUDA Graphs without a per-problem note (CLAUDE.md Rule 7).
  - Evaluator-state exploits: `data_ptr` caching, lazy-eval `__eq__` on `FakeTensor`, monkey-patched `Event.elapsed_time` / `do_bench` (CLAUDE.md Rule 6).
  - Precision-contract violations: precision downgrade (e.g. FP16 compute upcast to FP32 output) to pass tolerance (CLAUDE.md Rule 8)."""


def _strip_tbd(name: str) -> str:
    return name[4:] if name.startswith("tbd_") else name


def _category_dir(cat: str) -> str:
    """Map category label to its baselines/ subdirectory letter."""
    return cat  # L1 / L2 / Quant / FIB are used as-is


def _parse_spec_short_description(spec_path: Path) -> str:
    """Return a one-line short description from the spec's `## Description`
    section, or a fallback message if the spec is unreadable.
    """
    if not spec_path.is_file():
        return "(spec body not available; see raw source)"
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError:
        return "(spec body not readable; see raw source)"
    m = re.search(r"^## Description\s*\n+(.+?)(?:\n\n|\n\|)",
                  text, re.MULTILINE | re.DOTALL)
    if not m:
        return "(no Description section in spec; see raw source)"
    desc = m.group(1).strip()
    desc = re.sub(r"\s+", " ", desc)
    if len(desc) > 240:
        desc = desc[:237] + "..."
    return desc


_PATTERN_DTYPE_HINTS = {
    "gemm_baseline": "BF16 / FP16 / FP32 inputs, FP32 accumulator",
    "fused_add_rmsnorm": "BF16 inputs with FP32 reduction",
    "fp8_blockwise_quant": "FP8 e4m3 / e5m2 / NVFP4 e2m1 inputs, FP32 accumulator",
    "rope_attention": "BF16 / FP16 inputs with FP32 angle math",
    "gqa_paged_decode": "BF16 / FP16 inputs, FP32 softmax reduction",
    "swiglu_mlp": "BF16 / FP16 inputs, FP32 SiLU",
    "moe_dispatch": "BF16 / FP16 / FP8 inputs, FP32 softmax + accumulator",
    "attention_backward": "BF16 / FP16 / FP32 inputs, FP32 reduction",
    "attention_core": "BF16 / FP16 inputs, FP32 softmax reduction",
    "vision_audio_conv_blocks": "BF16 / FP16 / FP32 inputs, FP32 normalization",
    "hyena_mamba_ssm": "BF16 / FP16 / FP32 inputs, FP32 state accumulation",
    "norm_layernorm_groupnorm": "BF16 / FP16 inputs, FP32 reduction",
    "transformer_layer": "BF16 / FP16 / FP8 inputs",
    "rope_embedding_tables": "FP32 internal, BF16 / FP16 output",
    "rmsnorm": "BF16 / FP16 inputs, FP32 reduction",
    "mlp_gelu_geglu": "BF16 / FP16 inputs, FP32 GELU",
    "embedding_norm_projection": "BF16 / FP16 inputs, FP32 normalization",
    "gqa_paged_prefill": "BF16 / FP16 / FP8 inputs, FP32 softmax reduction",
    "activation_elementwise": "BF16 / FP16 / FP32 inputs, FP32 internal compute",
    "sam_mask_decoder": "BF16 / FP16 inputs",
    "attention_masking": "FP32 internal, BF16 / bool output",
    "mla_paged_attention": "BF16 / FP16 inputs, FP32 softmax reduction",
    "sparse_activation": "BF16 / FP16 / FP32 inputs",
    "flash_attention": "BF16 / FP16 inputs, FP32 online-softmax accumulators",
    "multimodal_sequence_packing": "BF16 / FP16 inputs",
    "flow_coupling": "BF16 / FP16 / FP32 inputs, FP32 exp",
    "loss_perceptual": "BF16 / FP16 / FP32 inputs, FP32 reduction",
}


def _slug_to_routing_tags(slug: str, primary: str, secondary: list[str]) -> str:
    """Build a Routing tags line with the slug's words + pattern tags."""
    words = slug.replace("problem_", "").split("_")
    words = [w for w in words if not w.isdigit()]
    base_tags = []
    seen = set()
    for w in words:
        wl = w.lower()
        if wl and wl not in seen and len(wl) > 1:
            base_tags.append(wl)
            seen.add(wl)
    pattern_tags = [primary] + secondary
    for p in pattern_tags:
        pl = p.lower()
        if pl and pl != "none" and pl not in seen:
            base_tags.append(pl)
            seen.add(pl)
    # Deliberately omit `sm_100a` / `sm_120` from indexed routing tags:
    # those tokens are required in the card BODY for the arch-portability
    # rule, but adding them to high-weight `Routing tags` makes every SOL
    # dossier match arch-only prompts like "sm_100a vs sm_120" and crowd
    # out the canonical answer in `sources/blackwell-tuning-docs.md`.
    return "; ".join(base_tags)


def _dossier_supporting_sources(primary: str, secondary: list[str]) -> str:
    parts = [f".claude/knowledge/patterns/{primary}.md"]
    for s in secondary:
        if s and s != "none":
            parts.append(f".claude/knowledge/patterns/{s}.md")
    parts.append(".claude/knowledge/sources/cutlass-cute-docs.md")
    return "; ".join(parts)


def _dossier_repo_touchpoints(category: str, slug: str) -> str:
    base = f"baselines/{_category_dir(category)}/{slug}"
    return f"{base}/cutlass/; {base}/cuda_cpp/"


def _matching_pattern_section(primary: str, secondary: list[str]) -> str:
    """Build the `## Matching pattern` body."""
    primary_link = f"`.claude/knowledge/patterns/{primary}.md`"
    if not secondary or secondary == ["none"]:
        return (
            f"This problem instantiates {primary_link}. The pattern "
            f"card carries the canonical CUTLASS / CuTe / CUDA C++ recipe; "
            f"this dossier records the problem-specific shape and dtype "
            f"constraints."
        )
    secondary_links = ", ".join(
        f"`.claude/knowledge/patterns/{s}.md`" for s in secondary if s != "none"
    )
    return (
        f"This problem composes {primary_link} as its primary pattern, "
        f"with secondary contributions from {secondary_links}. The "
        f"individual pattern cards carry the building-block recipes; "
        f"this dossier records the per-problem composition order and "
        f"shape constraints."
    )


def _entry_point_section(category: str, slug: str, dtype_hint: str) -> str:
    base = f"baselines/{_category_dir(category)}/{slug}"
    return (
        f"Per CLAUDE.md Rule 5 the submission surface is CUTLASS C++ or "
        f"CUDA C++. Author the baseline under one (or both) of:\n\n"
        f"- `{base}/cutlass/{{main.cpp, kernel.cu, kernel.h, solution.json}}` "
        f"for a CUTLASS-template variant.\n"
        f"- `{base}/cuda_cpp/{{kernel.cu, solution.json}}` for a "
        f"from-scratch CUDA C++ variant.\n\n"
        f"Entry-point signature per `sol-execbench/docs/solution.md` "
        f"(by-value `torch::Tensor` for inputs and outputs; the eval "
        f"driver pre-allocates the outputs and passes them in "
        f"destination-passing style):\n\n"
        f"```cpp\n"
        f"void run(/* input torch::Tensor args */,\n"
        f"         /* output torch::Tensor args */) {{\n"
        f"    // launch kernel(s); write into the output tensors "
        f"in-place; no return.\n"
        f"}}\n"
        f"```\n\n"
        f"Dtype contract: {dtype_hint}."
    )


def _arch_portability_section(category: str, slug: str) -> str:
    base = f"baselines/{_category_dir(category)}/{slug}"
    return (
        f"## Architecture portability per `CLAUDE.md` § Architecture Portability Discipline\n\n"
        f"The 5060 (sm_120) compiles and smoke-runs this baseline for "
        f"correctness verification; final SOL scoring is on B200 (sm_100a). "
        f"Treat the two targets as different: TCGen05 / TMA / DSMEM / "
        f"TMEM behaviors are sm_100a-only, and HBM3e bandwidth / L2 size "
        f"differences mean local timing is not a proxy for the cloud B200 "
        f"score. The B200 problem packager auto-injects "
        f"`-gencode arch=compute_100a,code=sm_100a` for "
        f"`target_hardware: [\"B200\"]`; build twice when in doubt.\n\n"
        f"Evidence-label vocabulary per `CLAUDE.md` lives in "
        f"`{base}/README.md`, NOT in this dossier."
    )


def _cross_references_section(primary: str, raw_source: str) -> str:
    return (
        f"- Pattern recipe: `.claude/knowledge/patterns/{primary}.md`.\n"
        f"- Source spec (authoritative): `{raw_source}`.\n"
        f"- CUTLASS submission-format precedent: "
        f"`sol-execbench/examples/cutlass/` and "
        f"`sol-execbench/examples/cuda_cpp/`."
    )


def _render(row: dict, repo_root: Path) -> str:
    slug = row["slug"]
    category = row["category"]
    primary_raw = row["primary_pattern"]
    primary = _strip_tbd(primary_raw)
    secondary = [
        _strip_tbd(s.strip()) for s in re.split(r"[;,]", row["secondary"])
        if s.strip() and s.strip().lower() != "none"
    ]
    raw_source = row["raw_source"]
    notes = row["notes"]

    spec_path = repo_root / raw_source
    short_desc = _parse_spec_short_description(spec_path)
    dtype_hint = _PATTERN_DTYPE_HINTS.get(primary,
                                          "see Primary raw source for dtypes")

    use_when = (
        f"starting the {category} baseline for `{slug}` -- {notes}. "
        f"Spec summary: {short_desc}"
    )

    routing_tags = _slug_to_routing_tags(slug, primary, secondary)

    frontmatter = (
        f"---\n"
        f"Use when: {use_when}\n"
        f"Primary raw source: {raw_source}\n"
        f"Supporting sources: {_dossier_supporting_sources(primary, secondary)}\n"
        f"Repo touchpoints: {_dossier_repo_touchpoints(category, slug)}\n"
        f"Patents anchored here: None\n"
        f"Out of scope:\n"
        f"{FORBIDDEN_SURFACE_BLOCK}\n"
        f"Version family: sol-execbench-2026\n"
        f"Default skill: implementing-sol-baselines\n"
        f"Routing tags: {routing_tags}\n"
        f"---\n"
    )

    body = (
        f"\n# {slug}\n\n"
        f"## Use when\n\n{use_when}\n\n"
        f"## Workload summary\n\n"
        f"| Field | Value |\n"
        f"| --- | --- |\n"
        f"| Category | {category} |\n"
        f"| Primary pattern | `{primary}` |\n"
        f"| Secondary patterns | "
        f"{', '.join(secondary) if secondary else 'none'} |\n"
        f"| Source spec | `{raw_source}` |\n"
        f"| Notes | {notes} |\n\n"
        f"For shapes, dtypes, reference PyTorch `run()`, and per-workload "
        f"hardware-target latencies, consult the source spec listed above. "
        f"This dossier intentionally does not duplicate the spec's numeric "
        f"workload table (which carries paper-authority latencies) to keep "
        f"the dossier free of latency-authority ambiguity.\n\n"
        f"## Matching pattern\n\n"
        f"{_matching_pattern_section(primary, secondary)}\n\n"
        f"## Implementation entry point\n\n"
        f"{_entry_point_section(category, slug, dtype_hint)}\n\n"
        f"{_arch_portability_section(category, slug)}\n\n"
        f"## Cross-references\n\n"
        f"{_cross_references_section(primary, raw_source)}\n"
    )
    return _active_text(frontmatter + body)


def _parse_mapping(mapping_path: Path) -> list[dict]:
    rows = []
    text = mapping_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        m = MAPPING_ROW_RE.match(line)
        if not m:
            continue
        rows.append({
            "slug": m.group(1),
            "category": m.group(2),
            "primary_pattern": m.group(3),
            "secondary": m.group(4),
            "raw_source": m.group(5),
            "notes": m.group(6),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repo root (default: cwd)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing dossiers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N writes (0 = unlimited)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    mapping_path = (
        root / "tools" / "sol_baselines_scaffold" / "sol_pattern_mapping.md"
    )
    problems_dir = root / ".claude" / "knowledge" / "problems"
    if not mapping_path.is_file():
        print(f"error: {mapping_path} not found", file=sys.stderr)
        return 1
    if not problems_dir.is_dir():
        print(f"error: {problems_dir} not found", file=sys.stderr)
        return 1

    rows = _parse_mapping(mapping_path)
    if len(rows) != 235:
        print(f"warning: parsed {len(rows)} rows; expected 235",
              file=sys.stderr)

    written = 0
    skipped = 0
    for row in rows:
        out = problems_dir / f"{row['slug']}.md"
        if out.exists() and not args.force:
            skipped += 1
            continue
        content = _render(row, root)
        out.write_text(content, encoding="utf-8")
        written += 1
        if args.limit and written >= args.limit:
            break

    _assert_dossier_cutlass_path_contract(problems_dir)

    print(f"Summary: {written} written, {skipped} skipped, "
          f"{len(rows)} total mapping rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
