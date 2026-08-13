#!/usr/bin/env python3
"""
lint_cards.py — Enforce SOL-ExecBench KB card schema and content rules.

Walks the SOL authoring subcorpora under `.claude/knowledge/` (excluding files
whose names start with `_`, which are templates/scaffolding), parses YAML-ish frontmatter using
the same logic as `build_manifest.py`, and runs three rule classes plus
several schema checks. Designed to run as a precondition to manifest
build so cards never enter the FTS5 index in a broken state.

Rule classes (all are enforced when `--rule all` is passed; otherwise
only the named rule runs):

- `required-fields` — 7 required + 4 optional frontmatter fields
  (`Use when`, `Primary raw source`, `Supporting sources`,
  `Repo touchpoints`, `Patents anchored here`, `Out of scope`,
  `Version family`; optional `Default skill`, `Also relevant skills`,
  `Routing tags`, `Latency authority`). Blank values fail for all 7
  required fields EXCEPT `Patents anchored here` MAY be the canonical
  placeholder `None`. `Out of scope: None` is rejected (the
  forbidden-surface rule requires enumerated clauses). Additional
  schema checks here: unknown subcorpus, non-canonical `Default skill`,
  invalid `Latency authority`, duplicate problem-spec links across
  `problems/`, missing / non-existent `Primary raw source` for any
  subcorpus, status fields forbidden in `problems/`, indexed-anchor
  presence in `patterns/` and `problems/`, numeric-latency-requires-
  authority detection.

- `forbidden-surface` — every `patterns/` and `problems/` card's `Out
  of scope` must enumerate all 8 forbidden-surface rule classes from
  CLAUDE.md (Rules 1, 2, 3, 4, 5, 6, 7, 8).

- `arch-portability` — every `patterns/` and `problems/` card body must
  mention both `sm_120` and `sm_100a` and cite `CLAUDE.md §
  Architecture Portability Discipline`.

Usage:
  python3 tools/sol_kb/lint_cards.py [--root REPO_ROOT] [--rule RULE] [--quiet]

Exit codes:
  0 — all cards pass.
  1 — at least one lint error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

# Import the wired router's data-only contract. Keeping the ownership map in a
# side-effect-free module avoids importing either CLI and prevents local-name
# shadowing by the retired SOL router fork.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_CANONICAL_KB = _REPO_ROOT / "scripts" / "kb"
if str(_CANONICAL_KB) not in sys.path:
    sys.path.insert(0, str(_CANONICAL_KB))
from routing_contract import (  # noqa: E402
    CANONICAL_SKILLS,
    CANONICAL_SUBCORPORA,
    SUBCORPUS_TO_SKILL,
)

# ---------------------------------------------------------------------------
# Frontmatter schema (mirrors build_manifest.py FRONTMATTER_FIELDS plus the
# `Latency authority` field used by the latency-authority rule).
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "Use when",
    "Primary raw source",
    "Supporting sources",
    "Repo touchpoints",
    "Patents anchored here",
    "Out of scope",
    "Version family",
]

OPTIONAL_FIELDS = [
    "Default skill",
    "Also relevant skills",
    "Routing tags",
    "Latency authority",
]

ALL_FIELDS = set(REQUIRED_FIELDS + OPTIONAL_FIELDS)

# Canonical skill/subcorpus sets come from the same data-only contract as the
# wired router. Add future ownership there rather than creating a linter-local
# exception.

LATENCY_AUTHORITY_VALUES = {"paper", "leaderboard", "local-checkout"}

# Subcorpora whose cards must satisfy forbidden-surface + arch-portability.
# `sources/` cards are exempt from arch-portability (they're citation stubs)
# and the forbidden-surface enumeration is recommended but not required —
# the binding requirement is on `patterns/`, `problems/`, and
# `cutlass_constructs/` (which carry concrete implementation guidance
# requiring the same discipline).
PATTERN_PROBLEM_SUBCORPORA = {"patterns", "problems", "cutlass_constructs"}
SOL_AUTHORING_SUBCORPORA = {
    "cutlass_constructs",
    "patterns",
    "problems",
    "sources",
}

# ---------------------------------------------------------------------------
# Forbidden-surface clause detection.
#
# Each tuple is (rule_id, list_of_required_phrases). ALL phrases in each
# tuple must appear (case-insensitive substring) in the card's `Out of
# scope` value for the card to pass. The hardened ALL-required semantics
# (Round 2 Codex review) prevents a card from enumerating one phrase of a
# rule and silently omitting other reward-hacks within the same rule
# class. The canonical reward-hack taxonomy comes from the SOL-ExecBench
# paper Table 3 and CLAUDE.md Rules 1-8.
# ---------------------------------------------------------------------------

FORBIDDEN_SURFACE_CLAUSES: list[tuple[str, list[str]]] = [
    ("Rule 5 (Python/Triton/DSL submissions)",
     ["Python", "Triton", "CuTe DSL", "torch.compile", "cuTile", "cudnn_frontend"]),
    ("Rule 1 (NCU)",
     ["NCU", "nv-nsight-cu-cli", "ncu-rep"]),
    ("Rule 2 (nvidia-smi clock-policy mutation)",
     ["nvidia-smi --lock", "lock-gpu-clocks", "lock-memory-clocks", "nvidia-smi",
      "persistence", "power", "MIG", "compute-mode"]),
    ("Rule 3 (privileged CUPTI)",
     ["CUPTI", "Events", "Metrics", "Range Profiling", "PM Sampling", "PerfWorks"]),
    ("Rule 4 (host-driver work)",
     ["driver install", "DKMS", "MOK", "modprobe nvidia", "host-driver",
      "kernel-module", "reboot"]),
    ("Rule 7 (default-stream discipline)",
     ["CUDA stream", "torch.jit.fork", "null-stream", "CUDA Graph", "background host thread"]),
    ("Rule 6 (evaluator-state exploits)",
     ["data_ptr", "Event.elapsed_time", "do_bench", "lazy-eval", "FakeTensor"]),
    ("Rule 8 (precision-contract violations)",
     ["precision downgrade", "precision-contract", "upcast to FP32", "FP16 compute"]),
]

# ---------------------------------------------------------------------------
# Architecture-portability detection.
# ---------------------------------------------------------------------------

ARCH_PORTABILITY_REQUIRED_TOKENS = ["sm_120", "sm_100a"]
ARCH_PORTABILITY_CITATION_RE = re.compile(
    r"CLAUDE\.md[^.\n]*?Architecture\s+Portability\s+Discipline",
    re.IGNORECASE | re.DOTALL,
)

# Evidence-label vocabulary phrase. Each patterns/ and problems/ card
# must reference the evidence-label vocabulary so a reader can locate it
# (the authoritative status fields themselves live in
# baselines/<category>/<slug>/README.md per CLAUDE.md:210-213 and may
# NOT appear inside problems/ dossiers per the status-field rule).
# Accept either "Evidence-label conventions" or "Evidence-label
# vocabulary" — both are valid pointers to the same CLAUDE.md section.
EVIDENCE_LABEL_VOCAB_RE = re.compile(
    r"\bEvidence[-\s]label\s+(?:conventions?|vocabulary)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# SOL routing anchors. Cards under `patterns/` and `problems/` MUST
# place at least one of these tokens in an indexed FTS5 field (name,
# title, Use when, or Routing tags) so the router can find them.
# ---------------------------------------------------------------------------

SOL_ROUTING_ANCHORS_RE = re.compile(
    # `problem_[0-9]{3,}` (without trailing word boundary) matches the
    # canonical SOL-ExecBench slug stem inside the full filename. The
    # other tokens keep `\b` boundaries since they're prose anchors.
    # Round 3 additions: `flow|coupling|perceptual|multimodal|packing|
    # hyena|conv|layernorm|groupnorm|geglu|gelu` cover the
    # `tbd_<new_pattern>` families from
    # `tools/sol_baselines_scaffold/sol_pattern_mapping.md`
    # so all pattern cards carry a recognizable family anchor in their
    # indexed FTS5 fields.
    r"\b(?:gemm|attention|rmsnorm|rope|moe|gqa|swiglu|paged.?attention|"
    r"fp8|nvfp4|fused.?add|fused.?rms|block.?scale|mla|ssm|mamba|"
    r"flow|coupling|perceptual|multimodal|packing|hyena|conv|"
    r"layernorm|groupnorm|geglu|gelu|"
    # CUTLASS / CuTe construct-card anchors (cutlass_constructs/ cards).
    r"collective|mainloop|epilogue|tma|tcgen05|tmem|tiled.?mma|mma.?atom)\b"
    r"|\bproblem_[0-9]{3,}",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Numeric-latency / bandwidth / SOL-score detection.
# ---------------------------------------------------------------------------

LATENCY_UNIT_RE = re.compile(
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:ms|us|µs|ns)\b"
)
BANDWIDTH_UNIT_RE = re.compile(
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:GB/s|TB/s|GiB/s|TiB/s)\b"
)
SOL_SCORE_RE = re.compile(r"\bS\s*=\s*[0-9]\.[0-9]+\b")

# Markdown-table-header context: a header row containing `Baseline (ms)` /
# `SOL (ms)` / `Latency` plus a unit. If a card has any such header, every
# numeric cell in the matching column counts as a latency citation.
TABLE_HEADER_RE = re.compile(
    r"\|[^|\n]*?(Baseline|SOL|Latency)\s*\(?(ms|us|µs|ns)\)?[^|\n]*?\|",
    re.IGNORECASE,
)

# Numeric cell within a markdown table row.
TABLE_NUMERIC_CELL_RE = re.compile(r"\|\s*[0-9]+\.[0-9]+\s*\|")

# Local-checkout authority anchor. If a card cites the local
# `sol-execbench/` checkout's timing module as a
# source for numeric latency, the `Latency authority` MUST be
# `local-checkout` — not `paper` or `leaderboard` — because the checkout's
# CUPTI-based timer in `core/bench/timing.py` diverges from the published
# CUDA-event methodology (CLAUDE.md Critical Rule 3 note).
TIMING_PY_SOURCE_RE = re.compile(
    r"sol-execbench/src/sol_execbench/core/bench/timing\.py",
)

# ---------------------------------------------------------------------------
# Status-field detection — forbidden in `problems/` dossiers.
# Status authority is `baselines/<category>/<slug>/README.md`
# per CLAUDE.md:210-213.
# ---------------------------------------------------------------------------

STATUS_FIELD_RE = re.compile(
    r"\b(?:Compile-checked-sm_120|Ran-local-sm_120|Scored-B200-sm_100a|"
    r"Not-B200-validated|best.?variant|leaderboard.?latency|"
    r"\bSOL\s+Score\b)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Frontmatter parser (mirrors build_manifest.py logic).
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[Optional[dict], str]:
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return None, text
    raw = text[4:end]
    body = text[end + 5:]
    fields: dict[str, str] = {}
    current_key: Optional[str] = None
    for line in raw.splitlines():
        m = re.match(r"^([A-Za-z][A-Za-z ]+):\s*(.*)$", line)
        if m is not None:
            key = m.group(1) or ""
            value = m.group(2) or ""
            if key in ALL_FIELDS:
                current_key = key
                fields[key] = value.strip()
                continue
        if current_key is not None:
            cont = line.strip()
            if not cont:
                continue
            bullet = re.match(r"^[-*]\s+(.*)$", cont)
            if bullet:
                item = (bullet.group(1) or "").strip()
                cur = fields[current_key]
                fields[current_key] = f"{cur}; {item}" if cur else item
            else:
                fields[current_key] = (fields[current_key] + " " + cont).strip()
    return fields, body


# ---------------------------------------------------------------------------
# Card representation.
# ---------------------------------------------------------------------------


class Card:
    """In-memory representation of a parsed KB card."""

    __slots__ = ("path", "rel", "subcorpus", "name", "fm", "text", "body")

    def __init__(self, path: Path, repo_root: Path):
        self.path = path
        self.rel = path.relative_to(repo_root)
        self.text = path.read_text(encoding="utf-8")
        self.fm, self.body = _split_frontmatter(self.text)
        self.name = path.stem
        parts = self.rel.parts
        if (
            len(parts) >= 3
            and parts[0] == ".claude"
            and parts[1] == "knowledge"
        ):
            self.subcorpus = parts[2] if len(parts) >= 4 else "_root"
        else:
            self.subcorpus = "_root"

    @property
    def is_index(self) -> bool:
        return self.path.name == "INDEX.md"

    def indexed_text(self) -> str:
        """Concatenate the FTS5-indexed fields (name + title + Use when +
        Routing tags). Anchors must appear here, not in body."""
        title_match = re.search(r"^#\s+(.+?)\s*$", self.body or "", re.MULTILINE)
        title = title_match.group(1).strip() if title_match else self.name
        fm = self.fm or {}
        return " ".join([
            self.name.replace("_", " "),
            title,
            fm.get("Use when", ""),
            fm.get("Routing tags", ""),
        ])


# ---------------------------------------------------------------------------
# Per-card rule implementations.
# ---------------------------------------------------------------------------


def _check_subcorpus_known(card: Card, errors: list[str]) -> None:
    if card.subcorpus not in CANONICAL_SUBCORPORA and card.subcorpus != "_root":
        errors.append(
            f"{card.rel}: unknown subcorpus `{card.subcorpus}` "
            f"(not in scripts/kb/routing_contract.py SUBCORPUS_TO_SKILL)"
        )


def _check_required_fields(card: Card, errors: list[str]) -> None:
    if card.fm is None:
        errors.append(f"{card.rel}: missing or malformed frontmatter")
        return
    fm = card.fm
    for field in REQUIRED_FIELDS:
        if field not in fm:
            errors.append(f"{card.rel}: missing required frontmatter key `{field}`")
            continue
        value = fm[field].strip()
        if not value:
            if field == "Patents anchored here":
                # Empty is rejected; the canonical placeholder is `None`.
                errors.append(
                    f"{card.rel}: required field `{field}` is blank "
                    f"(use the canonical placeholder `None` if not patent-anchored)"
                )
            else:
                errors.append(f"{card.rel}: required field `{field}` is blank")
            continue
        if field == "Out of scope" and value.strip().lower() == "none":
            errors.append(
                f"{card.rel}: `Out of scope: None` not allowed — must enumerate "
                f"the 8 forbidden-surface clauses (CLAUDE.md Rules 1-8)"
            )
    # Latency authority validation (optional field with enumerated values).
    if "Latency authority" in fm:
        value = fm["Latency authority"].strip()
        if value and value not in LATENCY_AUTHORITY_VALUES:
            errors.append(
                f"{card.rel}: `Latency authority: {value!r}` invalid; must be one of "
                f"{sorted(LATENCY_AUTHORITY_VALUES)}"
            )
    # Default skill validation (must be in canonical skill list).
    if "Default skill" in fm:
        value = fm["Default skill"].strip()
        if value and value not in CANONICAL_SKILLS:
            errors.append(
                f"{card.rel}: `Default skill: {value!r}` not in canonical skill list "
                f"(scripts/kb/routing_contract.py SUBCORPUS_TO_SKILL.values() +"
                f"implementing-sol-baselines)"
            )


def _check_forbidden_surface(card: Card, errors: list[str]) -> None:
    if card.subcorpus not in PATTERN_PROBLEM_SUBCORPORA:
        return
    if card.fm is None:
        return
    out_of_scope = card.fm.get("Out of scope", "")
    if not out_of_scope:
        return  # Already flagged by required-fields rule.
    lowered = out_of_scope.lower()
    for rule_id, required in FORBIDDEN_SURFACE_CLAUSES:
        missing = [phrase for phrase in required if phrase.lower() not in lowered]
        if missing:
            errors.append(
                f"{card.rel}: `Out of scope` missing required phrase(s) for "
                f"{rule_id}: {missing} (all canonical phrases must appear)"
            )


def _check_arch_portability(card: Card, errors: list[str]) -> None:
    if card.subcorpus not in PATTERN_PROBLEM_SUBCORPORA:
        return
    body = card.body or ""
    for tok in ARCH_PORTABILITY_REQUIRED_TOKENS:
        if tok not in body:
            errors.append(
                f"{card.rel}: missing required token `{tok}` "
                f"(arch-portability rule: patterns/ and problems/ cards must "
                f"mention both sm_120 and sm_100a)"
            )
    if not ARCH_PORTABILITY_CITATION_RE.search(body):
        errors.append(
            f"{card.rel}: missing citation to "
            f"`CLAUDE.md § Architecture Portability Discipline`"
        )
    if not EVIDENCE_LABEL_VOCAB_RE.search(body):
        errors.append(
            f"{card.rel}: missing evidence-label vocabulary phrase "
            f"(evidence-label rule: must mention `Evidence-label "
            f"conventions` or `Evidence-label vocabulary` to point "
            f"readers at the status fields documented in CLAUDE.md § "
            f"Architecture Portability Discipline; the status fields "
            f"themselves remain forbidden in problems/ dossiers per the "
            f"status-field rule)"
        )


def _check_indexed_anchor(card: Card, errors: list[str]) -> None:
    if card.subcorpus not in PATTERN_PROBLEM_SUBCORPORA:
        return
    text = card.indexed_text()
    if not SOL_ROUTING_ANCHORS_RE.search(text):
        errors.append(
            f"{card.rel}: no SOL routing anchor (gemm/attention/rope/moe/fp8/"
            f"nvfp4/problem_NNN/etc.) in indexed FTS5 fields "
            f"(name/title/Use when/Routing tags). Body-only mentions do NOT count."
        )


def _check_latency_authority(card: Card, errors: list[str]) -> None:
    fm = card.fm or {}
    authority = fm.get("Latency authority", "").strip()
    text = card.text or ""

    # Source-specific authority: if the card cites
    # `sol-execbench/src/sol_execbench/core/bench/timing.py`, the only
    # valid `Latency authority` is `local-checkout`. The checkout's
    # CUPTI-based timing path diverges from the paper's CUDA-event
    # methodology; conflating the two is the false-authority bug that
    # Codex Round-1 flagged.
    cites_timing_py = bool(TIMING_PY_SOURCE_RE.search(text))
    has_numeric = bool(
        LATENCY_UNIT_RE.search(text)
        or BANDWIDTH_UNIT_RE.search(text)
        or SOL_SCORE_RE.search(text)
        or (TABLE_HEADER_RE.search(text) and TABLE_NUMERIC_CELL_RE.search(text))
    )
    if cites_timing_py and has_numeric and authority != "local-checkout":
        errors.append(
            f"{card.rel}: cites `sol-execbench/src/sol_execbench/core/bench/"
            f"timing.py` as a source for numeric latency / bandwidth / SOL "
            f"score, but `Latency authority` is "
            f"{authority!r} — must be `local-checkout` (the checkout's CUPTI "
            f"timer diverges from the paper's CUDA-event methodology; "
            f"CLAUDE.md Critical Rule 3 note)"
        )
        return

    has_authority = authority in LATENCY_AUTHORITY_VALUES
    if has_authority:
        return
    # Pass 1: unit-suffixed values.
    if (
        LATENCY_UNIT_RE.search(text)
        or BANDWIDTH_UNIT_RE.search(text)
        or SOL_SCORE_RE.search(text)
    ):
        errors.append(
            f"{card.rel}: numeric latency / bandwidth / SOL score cited but "
            f"`Latency authority` is not set (must be one of "
            f"{sorted(LATENCY_AUTHORITY_VALUES)})"
        )
        return
    # Pass 2: markdown-table-header context.
    if TABLE_HEADER_RE.search(text) and TABLE_NUMERIC_CELL_RE.search(text):
        errors.append(
            f"{card.rel}: markdown table with `Baseline|SOL|Latency` column "
            f"header contains bare-decimal cells; `Latency authority` must be "
            f"set (must be one of {sorted(LATENCY_AUTHORITY_VALUES)})"
        )


def _check_status_fields_forbidden(card: Card, errors: list[str]) -> None:
    if card.subcorpus != "problems":
        return
    body = card.body or ""
    m = STATUS_FIELD_RE.search(body)
    if m:
        errors.append(
            f"{card.rel}: status field `{m.group(0)}` detected in dossier body "
            f"(status-field rule: status authority is "
            f"`baselines/<category>/<slug>/README.md` per CLAUDE.md:210-213; "
            f"dossiers must be status-free)"
        )


def _check_primary_raw_source_path(
    card: Card, errors: list[str], repo_root: Path
) -> None:
    """Validate `Primary raw source` resolves on disk for every subcorpus,
    and additionally enforce the `references/NVIDIA_Manuals/solbench_problems/
    problem_NNN_*.md` prefix rule for `problems/` cards.

    Cards declaring `Primary raw source: None` (the canonical placeholder
    for cards without a single upstream source — currently unused but
    reserved) are allowed. Paths inside backticks are stripped before
    resolution. Paths containing a template wildcard `<...>` are
    permitted as illustrative placeholders inside `Supporting sources`
    or `Repo touchpoints`; `Primary raw source` itself never carries a
    wildcard.
    """
    if card.fm is None:
        return
    raw_value = card.fm.get("Primary raw source", "").strip()
    if not raw_value or raw_value.lower() == "none":
        return  # Already flagged by required-fields rule when missing.
    candidate = raw_value.strip("`").strip()
    # Wildcard paths are not allowed in Primary raw source — it must be
    # a single concrete reference.
    if "<" in candidate and ">" in candidate:
        errors.append(
            f"{card.rel}: `Primary raw source` contains a `<wildcard>` "
            f"placeholder (`{candidate}`); must be a single concrete path."
        )
        return
    if card.subcorpus == "problems":
        expected_prefix = "references/NVIDIA_Manuals/solbench_problems/"
        if not candidate.startswith(expected_prefix):
            errors.append(
                f"{card.rel}: `Primary raw source` must point under "
                f"`{expected_prefix}problem_NNN_*.md` (problems/ dossier "
                f"rule); got `{candidate}`"
            )
            return
    active_candidate = _active_repo_candidate(candidate)
    if _is_optional_external_path(active_candidate):
        return
    target = repo_root / active_candidate
    # Accept either a file or a directory citation — `sources/` cards
    # sometimes cite a directory (the entire crawled subtree) as their
    # primary upstream reference.
    if not (target.is_file() or target.is_dir()):
        errors.append(
            f"{card.rel}: `Primary raw source` `{candidate}` does not "
            f"resolve on disk (must exist as a file or directory under "
            f"the repo root)."
        )


# ---------------------------------------------------------------------------
# Citation-path validity for `Supporting sources` and `Repo touchpoints`.
# These fields hold `;`-separated lists with a mix of concrete repo paths,
# directory citations, illustrative `<wildcard>` placeholders, and (rarely)
# external identifiers. We validate only the concrete repo-relative ones,
# with a documented exemption for forward-looking `baselines/<category>/
# <slug>/` paths (which are created on demand when a baseline lands per
# CLAUDE.md convention, so dossiers cite them as documented intentions).
# ---------------------------------------------------------------------------

_REPO_RELATIVE_PREFIXES = (
    "third_party/cutlass/",
    "references/",
    "tools/sol_execbench/",
    "workloads/sol_execbench_baselines/",
    "scripts/",
    ".claude/",
    "plans/",
    "tools/",
)

_BASELINE_FUTURE_PATH_RE = re.compile(
    r"^workloads/sol_execbench_baselines/"
    r"(?:L1|L2|Quant|FIB)/[A-Za-z0-9_]+/"
    r"(?:cutlass|cuda_cpp)(?:/|$)",
)

_REPO_ROOT_FILES = ("CLAUDE.md",)
_OPTIONAL_EXTERNAL_PREFIXES = ("SOLAR/",)


def _active_repo_candidate(candidate: str) -> str:
    """Resolve path spellings retained from the pre-rebase SOL checkout."""
    exact = {
        "AVO.pdf": "references/papers/AVO.pdf",
        "SOL-Execbench.pdf": "references/papers/SOL-Execbench.pdf",
    }
    if candidate in exact:
        return exact[candidate]
    prefixes = (
        ("cutlass/", "third_party/cutlass/"),
        ("sol-execbench/", "tools/sol_execbench/"),
        ("baselines/", "workloads/sol_execbench_baselines/"),
    )
    for old, new in prefixes:
        if candidate.startswith(old):
            return new + candidate[len(old):]
    return candidate


def _is_optional_external_path(candidate: str) -> bool:
    return any(candidate.startswith(prefix) for prefix in _OPTIONAL_EXTERNAL_PREFIXES)


def _is_documented_future_baseline_path(candidate: str) -> bool:
    """Return True for forward-looking `baselines/<category>/<slug>/...`
    paths that dossiers cite as documented intentions. CLAUDE.md says
    per-problem subdirectories under `baselines/{L1,L2,Quant,FIB}/` are
    created on demand when a baseline lands; until then the dossier-side
    citation is correct-by-convention even though the directory does not
    yet resolve.
    """
    return bool(_BASELINE_FUTURE_PATH_RE.match(candidate))


def _looks_repo_relative(candidate: str) -> bool:
    """Return True for tokens that look like concrete repo-relative
    paths and should therefore be validated for disk existence. Excludes
    wildcard placeholders, free-text URLs, and external identifiers.
    """
    if not candidate:
        return False
    if "<" in candidate and ">" in candidate:
        return False
    if candidate in _REPO_ROOT_FILES:
        return True
    return any(candidate.startswith(p) for p in _REPO_RELATIVE_PREFIXES)


def _check_supporting_and_touchpoints_paths(
    card: Card, errors: list[str], repo_root: Path
) -> None:
    """For each `;`-separated item in `Supporting sources` and
    `Repo touchpoints`, validate that concrete repo-relative paths
    resolve on disk. Forward-looking baseline paths are exempt; wildcard
    placeholders and external references are skipped silently.
    """
    if card.fm is None:
        return
    for field in ("Supporting sources", "Repo touchpoints"):
        value = card.fm.get(field, "").strip()
        if not value or value.lower() == "none":
            continue
        for raw_item in value.split(";"):
            cand = raw_item.strip().strip("`").strip()
            if not cand:
                continue
            active_candidate = _active_repo_candidate(cand)
            if _is_optional_external_path(active_candidate):
                continue
            if not _looks_repo_relative(active_candidate):
                continue
            if _is_documented_future_baseline_path(active_candidate):
                continue
            target = repo_root / active_candidate
            if not (target.is_file() or target.is_dir()):
                errors.append(
                    f"{card.rel}: `{field}` item `{cand}` does not "
                    f"resolve on disk (must exist as a file or directory "
                    f"under the repo root; use a `<wildcard>` placeholder "
                    f"for illustrative paths or document new baseline "
                    f"locations under the standard "
                    f"`baselines/<category>/<slug>/` convention)."
                )


# ---------------------------------------------------------------------------
# Cross-card rule: no two `problems/` dossiers may share the same spec.
# ---------------------------------------------------------------------------


def _check_unique_problem_spec(cards: list[Card], errors: list[str]) -> None:
    seen: dict[str, Path] = {}
    for card in cards:
        if card.subcorpus != "problems" or card.fm is None:
            continue
        spec = card.fm.get("Primary raw source", "").strip().strip("`").strip()
        if not spec:
            continue
        if spec in seen:
            errors.append(
                f"{card.rel}: duplicate `Primary raw source` `{spec}` "
                f"(already linked from `{seen[spec]}`; duplicate-source rule)"
            )
        else:
            seen[spec] = card.rel


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def _load_cards(knowledge_root: Path, repo_root: Path) -> list[Card]:
    cards: list[Card] = []
    for md_path in sorted(knowledge_root.rglob("*.md")):
        # Skip template stubs (names starting with `_`).
        if md_path.name.startswith("_"):
            continue
        card = Card(md_path, repo_root)
        if card.subcorpus in SOL_AUTHORING_SUBCORPORA:
            cards.append(card)
    return cards


def _run_rules(
    cards: list[Card], rule: str, errors: list[str], repo_root: Path
) -> None:
    apply_all = rule == "all"

    if apply_all or rule == "required-fields":
        for card in cards:
            _check_subcorpus_known(card, errors)
            _check_required_fields(card, errors)
            _check_latency_authority(card, errors)
            _check_indexed_anchor(card, errors)
            _check_status_fields_forbidden(card, errors)
            _check_primary_raw_source_path(card, errors, repo_root)
            _check_supporting_and_touchpoints_paths(card, errors, repo_root)
        _check_unique_problem_spec(cards, errors)

    if apply_all or rule == "forbidden-surface":
        for card in cards:
            _check_forbidden_surface(card, errors)

    if apply_all or rule == "arch-portability":
        for card in cards:
            _check_arch_portability(card, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repo root (default: cwd)")
    parser.add_argument(
        "--rule",
        default="all",
        choices=("all", "required-fields", "forbidden-surface", "arch-portability"),
        help="Which rule class to run (default: all)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()
    knowledge_root = repo_root / ".claude" / "knowledge"
    if not knowledge_root.is_dir():
        print(f"error: {knowledge_root} not found", file=sys.stderr)
        return 1

    cards = _load_cards(knowledge_root, repo_root)
    errors: list[str] = []
    _run_rules(cards, args.rule, errors, repo_root)

    if errors:
        for e in errors:
            print(e)
        print(f"\nFAIL: {len(errors)} lint error(s) across {len(cards)} card(s).", file=sys.stderr)
        return 1

    if not args.quiet:
        by_subcorpus: dict[str, int] = {}
        for c in cards:
            by_subcorpus[c.subcorpus] = by_subcorpus.get(c.subcorpus, 0) + 1
        print(f"OK: {len(cards)} card(s) pass `{args.rule}` rule(s).")
        for s, n in sorted(by_subcorpus.items()):
            print(f"  {s:24s} {n:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
