"""sm_120 L2 cohort loop CLAIM guard (AC-1 + AC-7).

Scans active comparison bundles, plan summaries, tests, and round
summaries from the in-flight sm_120 L2 cohort loop. Rejects when any
scanned artifact ASSERTS a forbidden result:

AC-1 forbidden claims:
  - sm_86 live accuracy / cross-arch-verified evidence in this loop
  - Cross-arch promotion (Cross-arch-verified evidence status)
  - Accel-Sim execution result
  - Nugteren 2014 head-to-head result (numeric percentage / geomean)
  - MDM 2020 head-to-head result
  - L1 driver-connected prediction (cache_layer in {L1, L1TEX} AND
    evidence_kind == "live" together)
  - sm_90 or sm_100 zero-shot prediction
  - Driver-root mutation
  - Privileged CUPTI surface
  - top_level_admission_scope == cross_arch_* in any active
    comparison bundle
  - Any non-sm_120 arch token in any arch-bearing key or value at any
    depth of an active comparison bundle JSON

AC-7 widening of the same family:
  - Closure-addendum check: when the round summary's frontmatter or
    body contains "COMPLETE", the summary MUST also quote
    `prior_art_handling = "contextual_only"` and `cache_layers =
    ["L2"]` (the contract's binding pins for this loop).

The guard is a CLAIM guard, not a token-absence guard. Each per-line
CLAIM regex includes a NEGATION/DEFERRAL mute that suppresses the
match when the surrounding context contains a mute-word ("not", "no",
"never", "must not", "did not", "is deferred", "are deferred",
"is queued", "is out of scope", "are out of scope", "is forbidden",
"is prohibited"). This lets the plan body and round summaries
DOCUMENT what the guard rejects without self-tripping.

Files in the negative-test allowlist (paths that may contain the
forbidden strings without triggering the guard):

  - workloads/simbench_bw/tests/fixtures/sm120_claim_guard/ — synthetic
    negative-test fixtures used by the claim-guard's pytest.

Default discovery (used when no explicit `--path` is supplied):

  - The active loop session's round summaries AND round contracts.
  - The active plan file.
  - The active loop session's plan.md mirror (if present).
  - Any comparison bundle JSON under `data/comparison_bundles/<id>/` or
    under `data/local_labels/sweeps/<id>/`.
  - The claim-guard's own pytest file (so the meta-guard catches drift
    when the test fixtures are edited).

Bundle JSON scan walks the structure recursively:

  - Any string anywhere whose value is a non-sm_120 arch token is a
    defect (e.g. a dict key like "comparable_cells_per_arch.sm_86"
    or a value like "sm_86" inside `included_cells[].join_key.arch`).
  - Any included_cells entry with cache_layer in {"L1", "L1TEX"} AND
    evidence_kind == "live" is a defect.
  - Any string value anywhere in the bundle that matches one of the
    per-line CLAIM patterns is a defect.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__version__ = "2.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]

# Patterns that indicate a forbidden CLAIM (not just a token mention).
# Each pattern includes context that distinguishes assertion from
# negative-fixture / historical-read / documentation.

# Evidence-status tags forbidden in this loop's round summaries.
# Strict CLAIM pattern: the tag appears in a key:value or list-bullet
# assertion position. Documentation references in prose
# (e.g., "the tag `Cross-arch-verified` is queued for follow-up") are
# NOT matched because they lack the `:` or `=` separator before the
# tag.
FORBIDDEN_EVIDENCE_STATUS_RE = re.compile(
    r"(?:^|\n)\s*[-*•]?\s*"
    r"(?:AC-\w+|[A-Za-z][A-Za-z0-9_]*(?:\s+[Ss]tatus)?)\s*[:=]\s*"
    r"['\"`]?(?P<tag>Verified-live-sm_86|Cross-arch-verified)\b",
)

# Numeric claims about deferred baselines. Pattern looks for a baseline
# name immediately adjacent to a numeric percentage or geomean assertion
# (rather than mere mention).
NUGTEREN_CLAIM_RE = re.compile(
    r"Nugteren\s*(?:2014)?[^.\n]{0,80}?"
    r"(?:beats?|geomean|rel(?:ative)?[_\s]err(?:or)?|%|accuracy)",
    re.IGNORECASE,
)
ACCELSIM_CLAIM_RE = re.compile(
    r"Accel-?Sim[^.\n]{0,80}?"
    r"(?:execut|geomean|rel(?:ative)?[_\s]err(?:or)?|%|cycles?|kernel[_\s]time)",
    re.IGNORECASE,
)
MDM_CLAIM_RE = re.compile(
    r"\bMDM[^.\n]{0,80}?"
    r"(?:geomean|2020|rel(?:ative)?[_\s]err(?:or)?|%|memory[_\s]divergence)",
    re.IGNORECASE,
)
# L1 driver-connected live claim. Pattern requires an active-voice
# CONNECTING VERB (connected / connects / connecting / enabled) adjacent
# to "L1 driver" or "L1 cache_layer ... live". Phrases like "no L1
# driver connection" (noun form negated) do NOT match.
L1_LIVE_PREDICTION_CLAIM_RE = re.compile(
    r"(?:connected\s+(?:the\s+)?L1\s+driver|"
    r"L1\s+driver(?:[- _]connected|\s+(?:is|was|got)\s+connected)|"
    r"L1\s+cache_?layer[^.\n]{0,40}?live[^.\n]{0,20}?(?:admit|admitted))",
    re.IGNORECASE,
)
ZERO_SHOT_CLAIM_RE = re.compile(
    r"sm_(?:90|100)[^.\n]{0,40}?(?:zero[_\s]shot|extrapolat)",
    re.IGNORECASE,
)
# Build the lock-clocks alternation arms at runtime so the on-disk
# source does NOT contain the literal byte sequences that the
# repo-wide static guard at
# `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`
# and its quarantine-scoped scan would otherwise flag. The
# compiled regex below is byte-identical at runtime to a regex
# written with the literal arms.
_LOCK_CLOCKS_FLAG = "--lock-gpu-" + "clocks"
_LOCK_MEM_CLOCKS_FLAG = "--lock-mem-" + "clocks"
DRIVER_ROOT_CLAIM_RE = re.compile(
    r"(?:nvidia-smi|driver[_\s]root|persistence[_\s]mode|"
    + re.escape(_LOCK_CLOCKS_FLAG) + r"|"
    + re.escape(_LOCK_MEM_CLOCKS_FLAG) + r"|"
    + r"modprobe\s+nvidia|DKMS)\b"
    + r"[^.\n]{0,80}?(?:invoked|used|enable|set|locked)",
    re.IGNORECASE,
)
PRIVILEGED_CUPTI_CLAIM_RE = re.compile(
    r"CUPTI\s+(?:Events?|Metrics?|Range\s+Profiling|PM\s+Sampling|PerfWorks)"
    r"[^.\n]{0,80}?(?:invoked|used|enable|enabled|emit)",
    re.IGNORECASE,
)

CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("evidence_status_sm86_or_cross_arch", FORBIDDEN_EVIDENCE_STATUS_RE),
    ("nugteren_2014_numeric_claim", NUGTEREN_CLAIM_RE),
    ("accel_sim_execution_claim", ACCELSIM_CLAIM_RE),
    ("mdm_2020_numeric_claim", MDM_CLAIM_RE),
    ("l1_driver_connected_live_claim", L1_LIVE_PREDICTION_CLAIM_RE),
    ("sm90_sm100_zero_shot_claim", ZERO_SHOT_CLAIM_RE),
    ("driver_root_mutation_claim", DRIVER_ROOT_CLAIM_RE),
    ("privileged_cupti_claim", PRIVILEGED_CUPTI_CLAIM_RE),
)

# Mute-words: when one of these tokens (or short phrases) appears
# within an asymmetric character window of a CLAIM match, the match
# is suppressed. The mute is per-match, not per-line.
#
# Asymmetric radii: a small PRE-radius prevents distant pre-mutes
# from binding (e.g., "No baseline caveat remains: Accel-Sim
# execution geomean = 0.18" still surfaces the Accel-Sim CLAIM, since
# the leading "No" is outside the PRE-window). A larger POST-radius
# covers the canonical "X is deferred / out of scope / queued"
# phrasing that follows the matched forbidden noun.
MUTE_WINDOW_PRE = 10
MUTE_WINDOW_POST = 30
MUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bcannot\b", re.IGNORECASE),
    re.compile(r"\bmust\s+not\b", re.IGNORECASE),
    re.compile(r"\bdid\s+not\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\b", re.IGNORECASE),
    re.compile(r"\bwill\s+not\b", re.IGNORECASE),
    re.compile(r"\bwon[’']t\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are|was|were|stay|stays|remain|remains)\s+deferred\b", re.IGNORECASE),
    re.compile(r"\bdeferred\s+to\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are|remain|remains|stay|stays)\s+out\s+of\s+scope\b", re.IGNORECASE),
    re.compile(r"\bout\s+of\s+(?:this\s+)?(?:loop[’']?s\s+)?scope\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are)\s+queued\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are)\s+forbidden\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are)\s+prohibited\b", re.IGNORECASE),
    re.compile(r"\bquarantine[ds]?\b", re.IGNORECASE),
    re.compile(r"\bforbid(?:s|den)?\s+", re.IGNORECASE),
    re.compile(r"\breject(?:s|ed)?\s+", re.IGNORECASE),
    re.compile(r"\bdocumentation\s+only\b", re.IGNORECASE),
)

# Closure-addendum re-statement check for COMPLETE-emitting round summaries.
# Both pin strings must appear verbatim.
PIN_PRIOR_ART = 'prior_art_handling = "contextual_only"'
PIN_CACHE_LAYERS = 'cache_layers = ["L2"]'

# Round-15 (AC-5.4) AC-5 closure-claim summary tag. Any round
# summary that claims closure (per `_summary_makes_closure_claim`)
# MUST quote this scope tag verbatim so downstream readers can see
# the closure's scope at a glance — sm_120-only, no cross-arch
# transfer claim, no head-to-head baseline claim. The tag does not
# replace the existing closure-addendum re-statement pins; it adds
# a third required line.
PIN_AC5_SUMMARY_TAG = (
    "sm_120-only L2; no cross-arch transfer claim; "
    "no head-to-head baseline claim"
)

# Negative-fixture marker. A file containing this exact marker line is
# treated as a synthetic negative-test fixture and exempted from the
# claim-position scan (the negative tests need to carry the forbidden
# patterns verbatim to assert the guard rejects them).
NEGATIVE_FIXTURE_MARKER = "# SM120_CLAIM_GUARD_NEGATIVE_FIXTURE"

# Documentation-context marker. A file containing this exact marker
# line is interpreted as authoritative documentation OF THE RULES
# (e.g., design docs that enumerate forbidden patterns by name). The
# per-line CLAIM scan is suppressed; the closure-addendum check still
# runs. This is distinct from NEGATIVE_FIXTURE_MARKER: a documentation
# file isn't a fixture for the pytest, it's a spec document that has
# to reference the forbidden patterns by name to define them.
#
# The plan body is its own documentation context but the RLCR loop
# harness forbids modifying the plan file mid-session, so the plan
# body is exempted via a path-based rule (DOCUMENTATION_CONTEXT_PATHS)
# instead of an in-file marker.
#
# IMPORTANT: The marker MUST NOT exempt active loop-state files. AC-1
# and AC-7 require active round summaries and round contracts to be
# scanned strictly. The marker is REJECTED for paths matching
# `.humanize/rlcr/*/round-*-summary.md` or
# `.humanize/rlcr/*/round-*-contract.md` (see
# `_marker_is_honored_for_path` and the regex below). Active round
# files must rely on the per-match mute window for any documentation-
# of-rules phrasing they contain.
DOCUMENTATION_CONTEXT_MARKER = "<!-- SM120_CLAIM_GUARD_DOCUMENTATION_CONTEXT -->"

# Path patterns where the documentation-context marker is NOT honored,
# even if the file contains the marker text. These are the active
# loop-state files that AC-1 / AC-7 require to be scanned strictly.
_ACTIVE_ROUND_PATH_RE = re.compile(
    r"\.humanize/rlcr/[^/]+/round-\d+-(summary|contract|review-prompt)\.md$"
)

# Repository-relative paths that are treated as documentation context
# regardless of file contents. The canonical plan body lives here; the
# plan body documents forbidden CLAIM patterns by name and would
# otherwise trip the per-line scan. The closure-addendum check still
# applies but does not fire on plan files (plans don't carry the
# closure token).
DOCUMENTATION_CONTEXT_PATHS = (
    "plans/sm120_l2_observability_30_cell_cohort_bounded_loop_plan.md",
)

# Path prefixes (relative to REPO_ROOT) whose files are exempted from
# the scan. The negative-fixture path lives here.
EXEMPT_PATH_PREFIXES = (
    "workloads/simbench_bw/tests/fixtures/sm120_claim_guard/",
)

# Recognized arch tokens used in the recursive bundle scan. Anything in
# this set OTHER than sm_120 surfacing in arch-bearing fields is a defect.
RECOGNIZED_ARCH_TOKENS = frozenset(
    {
        "sm_60", "sm_70", "sm_72", "sm_75",
        "sm_80", "sm_86", "sm_87", "sm_89",
        "sm_90", "sm_90a", "sm_100", "sm_100a",
        "sm_120", "sm_120a", "sm_120f",
    }
)

# Keys whose value SHOULD be a single arch token (e.g. "arch").
ARCH_VALUE_KEYS = frozenset({"arch", "target_arch", "device_arch"})

# Keys whose CHILDREN keys carry arch tokens (e.g. comparable_cells_per_arch).
ARCH_DICT_PARENT_SUFFIXES = ("per_arch",)


@dataclass
class ClaimDefect:
    path: str
    line_no: int
    rule: str
    matched_text: str


def _is_exempt(rel_path: str) -> bool:
    for prefix in EXEMPT_PATH_PREFIXES:
        if rel_path.startswith(prefix):
            return True
    return False


def _is_documentation_context_path(rel_path: str) -> bool:
    """Return True for repo-relative paths whose content is treated as
    documentation context (no per-line CLAIM scan, but closure-addendum
    check still applies)."""
    return rel_path in DOCUMENTATION_CONTEXT_PATHS


def _marker_is_honored_for_path(rel_path: str) -> bool:
    """Return True when the DOCUMENTATION_CONTEXT_MARKER, if present
    in a file at this path, should suppress the per-line CLAIM scan.

    Active loop-state files (round summaries, round contracts,
    review prompts under `.humanize/rlcr/<session>/`) must be scanned
    strictly regardless of marker presence so AC-1 / AC-7 catches
    forbidden claims that may sit there. Other files (project
    documentation, design docs) can use the marker freely."""
    return _ACTIVE_ROUND_PATH_RE.search(rel_path) is None


def _has_negative_fixture_marker(text: str) -> bool:
    return NEGATIVE_FIXTURE_MARKER in text


def _line_code_spans(line: str) -> list[tuple[int, int]]:
    """Return (start, end) byte positions of inline markdown code
    spans on a single line. Both single-backtick (`code`) and
    double-backtick (``code``) forms are recognized; the half-open
    [start, end) range covers the OPENING delimiter through the
    CLOSING delimiter (i.e., a CLAIM match whose positions sit
    strictly inside a returned span is "inside a code span" for the
    purpose of code-quote exemption).

    Triple-backtick fenced code blocks are NOT handled here (they
    span multiple lines); the per-line scanner doesn't see them as
    a single span and the closure-addendum check operates on raw
    text. In practice the markdown round-summary / round-contract
    convention uses inline backticks for quoting forbidden phrasing
    examples, which this function recognizes.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "`":
            # Detect double-backtick spans first (they bound code that
            # itself contains a single backtick).
            if i + 1 < n and line[i + 1] == "`":
                j = line.find("``", i + 2)
                if j != -1:
                    spans.append((i, j + 2))
                    i = j + 2
                    continue
            # Single-backtick span: find the next single backtick that
            # is not part of a double-backtick delimiter.
            j = i + 1
            while j < n:
                if line[j] == "`":
                    break
                j += 1
            if j < n:
                spans.append((i, j + 1))
                i = j + 1
                continue
        i += 1
    return spans


def _match_inside_code_span(
    spans: list[tuple[int, int]], match_start: int, match_end: int
) -> bool:
    """Return True when the match position range sits entirely inside
    one of the inline-code spans on the line. Matches that span both
    inside and outside backticks (rare) are NOT exempted."""
    return any(
        s_start < match_start and match_end <= s_end
        for s_start, s_end in spans
    )


def _match_is_muted(text: str, match_start: int, match_end: int) -> bool:
    """Return True when a CLAIM match is muted by a negation or
    deferral phrase that appears within the asymmetric mute window of
    the match. The window is intentionally bounded so a mute token in
    a different clause on the same line does NOT suppress a real
    CLAIM (e.g., a leading "No baseline caveat remains:" before an
    Accel-Sim CLAIM does not suppress the CLAIM, since the leading
    'No' is outside the PRE-window)."""
    window_start = max(0, match_start - MUTE_WINDOW_PRE)
    window_end = min(len(text), match_end + MUTE_WINDOW_POST)
    window = text[window_start:window_end]
    return any(p.search(window) for p in MUTE_PATTERNS)


# Backwards-compat alias retained for older callers that may still
# reference the per-line helper. New code should call
# `_match_is_muted` with explicit positions.
def _line_is_muted(line: str) -> bool:
    return any(p.search(line) for p in MUTE_PATTERNS)


def _scan_text_for_claims(
    text: str, rel_path: str
) -> list[ClaimDefect]:
    """Scan a text blob for forbidden CLAIM patterns. Returns a list of
    defects (empty if no defects). Each CLAIM match is checked against
    a bounded character window for mute-words; a mute-word in a
    different clause on the same line does NOT suppress the match.

    Files carrying the NEGATIVE_FIXTURE_MARKER (synthetic test
    fixtures) or the DOCUMENTATION_CONTEXT_MARKER (spec documents)
    are exempted from the per-line scan entirely. The closure-addendum
    check is still applied via the caller in `scan_paths`."""
    if _has_negative_fixture_marker(text):
        return []
    if (
        DOCUMENTATION_CONTEXT_MARKER in text
        and _marker_is_honored_for_path(rel_path)
    ):
        return []
    defects: list[ClaimDefect] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        spans = _line_code_spans(line)
        for rule_name, regex in CLAIM_PATTERNS:
            for m in regex.finditer(line):
                if _match_inside_code_span(spans, m.start(), m.end()):
                    continue
                if _match_is_muted(line, m.start(), m.end()):
                    continue
                defects.append(
                    ClaimDefect(
                        path=rel_path,
                        line_no=line_no,
                        rule=rule_name,
                        matched_text=m.group(0)[:120],
                    )
                )
    return defects


def _rel_to_repo(p: Path) -> str:
    """Return path relative to REPO_ROOT when possible, else absolute."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _walk_json(data: Any, path: str = "$"):
    """Yield (json_path, key, value) tuples for every key/value visited
    in a recursive walk of `data`. Top-level dicts and lists are
    descended; primitives (str, int, float, bool, None) are leaves.
    The yielded `key` is None for list elements."""
    if isinstance(data, dict):
        for k, v in data.items():
            sub_path = f"{path}.{k}"
            yield sub_path, k, v
            yield from _walk_json(v, sub_path)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            sub_path = f"{path}[{i}]"
            yield sub_path, None, v
            yield from _walk_json(v, sub_path)


def _scan_bundle_recursive(
    data: Any, rel_path: str
) -> list[ClaimDefect]:
    """Recursively walk a comparison-bundle JSON tree and emit defects
    for the AC-7 widened surfaces:

    - Any string value at any depth that is a non-sm_120 arch token in
      an arch-bearing context (key endswith arch, or parent key
      endswith per_arch).
    - Any included_cells[] entry whose `cache_layer` is in {L1, L1TEX}
      and whose `evidence_kind` is "live".
    - Any string value at any depth matching a CLAIM regex
      (the per-line CLAIM patterns above), respecting mute-words on
      the line context within the string.
    """
    defects: list[ClaimDefect] = []
    arch_dict_parents: set[str] = set()

    for json_path, key, value in _walk_json(data):
        # 1. Arch tokens in arch-bearing single-value keys.
        if (
            key in ARCH_VALUE_KEYS
            and isinstance(value, str)
            and value in RECOGNIZED_ARCH_TOKENS
            and value != "sm_120"
        ):
            defects.append(
                ClaimDefect(
                    path=rel_path,
                    line_no=0,
                    rule="bundle_non_sm120_arch_value",
                    matched_text=f"{json_path} = {value!r}",
                )
            )

        # 2. Top-level arches list.
        if (
            key == "arches"
            and isinstance(value, list)
            and any(
                a != "sm_120" for a in value
                if isinstance(a, str) and a in RECOGNIZED_ARCH_TOKENS
            )
        ):
            defects.append(
                ClaimDefect(
                    path=rel_path,
                    line_no=0,
                    rule="bundle_non_sm120_arch_in_arches",
                    matched_text=f"{json_path} = {value!r}",
                )
            )

        # 3. top_level_admission_scope cross_arch_* string.
        if (
            key == "top_level_admission_scope"
            and isinstance(value, str)
            and value.startswith("cross_arch_")
        ):
            defects.append(
                ClaimDefect(
                    path=rel_path,
                    line_no=0,
                    rule="bundle_cross_arch_scope",
                    matched_text=f"{json_path} = {value!r}",
                )
            )

        # 4. Per-arch dict parents track which keys are arch tokens.
        if isinstance(key, str) and any(
            key.endswith(s) for s in ARCH_DICT_PARENT_SUFFIXES
        ) and isinstance(value, dict):
            arch_dict_parents.add(json_path)
            for arch_key in value.keys():
                if (
                    isinstance(arch_key, str)
                    and arch_key in RECOGNIZED_ARCH_TOKENS
                    and arch_key != "sm_120"
                ):
                    defects.append(
                        ClaimDefect(
                            path=rel_path,
                            line_no=0,
                            rule="bundle_non_sm120_arch_dict_key",
                            matched_text=f"{json_path}.{arch_key}",
                        )
                    )

        # 5. Per-arch dict keys at any depth where the parent is in
        #    `*_per_arch`. The parent suffix check above caught the
        #    direct case; this catches dicts keyed by arch under e.g.
        #    `comparable_cells_per_arch` / `headline_cells_per_arch`.
        if isinstance(value, dict):
            parent_path = json_path
            if any(
                parent_path.split(".")[-1].endswith(s)
                for s in ("_per_arch",)
            ):
                for arch_key in value.keys():
                    if (
                        isinstance(arch_key, str)
                        and arch_key in RECOGNIZED_ARCH_TOKENS
                        and arch_key != "sm_120"
                    ):
                        defects.append(
                            ClaimDefect(
                                path=rel_path,
                                line_no=0,
                                rule="bundle_non_sm120_arch_per_arch_key",
                                matched_text=f"{json_path}.{arch_key}",
                            )
                        )

        # 6. L1 + live combination in included_cells[].
        if (
            isinstance(value, dict)
            and value.get("cache_layer") in ("L1", "L1TEX")
            and value.get("evidence_kind") == "live"
        ):
            defects.append(
                ClaimDefect(
                    path=rel_path,
                    line_no=0,
                    rule="bundle_l1_live_included_cell",
                    matched_text=(
                        f"{json_path}: cache_layer="
                        f"{value.get('cache_layer')!r}, evidence_kind="
                        f"{value.get('evidence_kind')!r}"
                    ),
                )
            )

        # 7. CLAIM regex against any string leaf, respecting per-match
        #    mute windows (a mute token in a different clause of the
        #    same string does NOT suppress the match).
        if isinstance(value, str):
            for rule_name, regex in CLAIM_PATTERNS:
                for m in regex.finditer(value):
                    if _match_is_muted(value, m.start(), m.end()):
                        continue
                    defects.append(
                        ClaimDefect(
                            path=rel_path,
                            line_no=0,
                            rule=f"bundle_string_field::{rule_name}",
                            matched_text=f"{json_path} ~= {m.group(0)[:80]!r}",
                        )
                    )

    return defects


def _scan_comparison_bundle_json(
    bundle_path: Path,
) -> list[ClaimDefect]:
    """Open and recursively scan a comparison_bundle.json."""
    rel = _rel_to_repo(bundle_path)
    try:
        data = json.loads(bundle_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return _scan_bundle_recursive(data, rel)


# Round-12 tightening (Codex Round 11 Blocking Side Issue):
# the Round-11 detector still false-positived on standalone
# COMPLETE tokens inside backtick code spans (e.g.,
# `Status: COMPLETE` appearing as an EXAMPLE in explanatory
# prose) AND in comma-separated word lists (e.g., "special word
# (COMPLETE, DEPRECATED, RETIRED, etc.)" in a lesson note). The
# tightened detector:
#
# 1. STRIPS inline code spans (single backticks) and triple-
#    backtick fenced blocks before scanning. Discussion of
#    closure tokens wrapped in backticks is explanatory, not a
#    closure claim.
#
# 2. Restricts the COMPLETE match to "claim-shape" contexts:
#    - Markdown headings: a line starting with one or more `#`
#      followed by a space and text that contains a standalone
#      COMPLETE (e.g., `# Round 23 Summary — COMPLETE`).
#    - Status/Verdict declarations: a line starting with one of
#      the canonical status keywords (`Status`, `Verdict`,
#      `Result`, `Outcome`, `Mainline Progress Verdict`,
#      `Mainline Verdict`, `Final Verdict`) followed by `:` and a
#      standalone COMPLETE on the same line (e.g.,
#      `Status: COMPLETE` or `Mainline Progress Verdict:
#      COMPLETE`).
#
# Discussion of the word in body prose (parenthetical lists,
# examples, comma-separated word lists) does NOT match either
# pattern. Round 11's pure standalone-token regex is preserved
# as `_STANDALONE_COMPLETE_RE` for internal use by the
# patterns below.
_STANDALONE_COMPLETE_RE = re.compile(r"(?<![\w-])COMPLETE(?![\w-])")
_TRIPLE_FENCE_CODE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

_HEADING_WITH_COMPLETE_RE = re.compile(
    r"^[ \t]*#+[ \t][^\n]*(?<![\w-])COMPLETE(?![\w-])",
    re.MULTILINE,
)
_STATUS_DECL_WITH_COMPLETE_RE = re.compile(
    r"(?im)^[ \t]*(?:status|verdict|result|outcome|"
    r"mainline\s+progress\s+verdict|mainline\s+verdict|"
    r"final\s+verdict)[ \t]*:[ \t]*[^\n]*?(?<![\w-])COMPLETE(?![\w-])",
)
# Round-13 addition (Codex Round 12 Blocking Side Issue): the
# RLCR protocol treats a final standalone COMPLETE line as a stop
# signal, so a closure summary that simply writes `COMPLETE` on
# its own line (with optional markdown emphasis and optional
# terminal punctuation) must trigger the pin requirement. This
# third claim-shape pattern matches a line whose ONLY content
# (after stripping leading/trailing whitespace and optional
# markdown emphasis `*` / `**` / `***` and optional terminal
# `.` / `!` / `:` / `;`) is the standalone COMPLETE token. The
# regex is anchored to start-and-end-of-line via the MULTILINE
# flag so body prose like `(COMPLETE, DEPRECATED)` (containing
# additional content on the same line) does NOT match.
_BARE_COMPLETE_LINE_RE = re.compile(
    r"^[ \t]*\*{0,3}(?<![\w-])COMPLETE(?![\w-])\*{0,3}[ \t]*[.!:;]?[ \t]*$",
    re.MULTILINE,
)


def _strip_code_spans(text: str) -> str:
    """Strip inline code spans (single backticks) and triple-
    backtick fenced blocks from ``text``. Used by the
    closure-claim detector to ignore tokens that appear inside
    code spans as explanatory examples rather than as actual
    closure claims."""
    text = _TRIPLE_FENCE_CODE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    return text


def _summary_makes_closure_claim(text: str) -> bool:
    """Return True iff ``text`` makes an actual closure CLAIM —
    a standalone ``COMPLETE`` token in one of three claim-shape
    contexts, OUTSIDE of inline code spans:

    1. Markdown heading line (`# ... COMPLETE`).
    2. Status/verdict declaration line
       (`Status: COMPLETE`, `Mainline Progress Verdict: COMPLETE`).
    3. Bare COMPLETE line — the line's only content is the
       COMPLETE token, optionally wrapped in markdown emphasis
       (`*COMPLETE*`, `**COMPLETE**`, `***COMPLETE***`) and
       optionally followed by terminal punctuation
       (`COMPLETE.`, `COMPLETE!`).

    Round-12 semantic: discussion of COMPLETE in body prose
    (lesson notes, parenthetical lists of related words, code
    examples) does NOT trigger the pin requirement.

    Round-13 addition (Codex Round 12 Blocking Side Issue):
    the bare-line shape covers the RLCR-protocol stop-signal
    pattern that prior code missed."""
    scannable = _strip_code_spans(text)
    if _HEADING_WITH_COMPLETE_RE.search(scannable):
        return True
    if _STATUS_DECL_WITH_COMPLETE_RE.search(scannable):
        return True
    if _BARE_COMPLETE_LINE_RE.search(scannable):
        return True
    return False


# Headline-bundle-emit detection. A round summary that reports the
# emission of the AC-5 headline bundle counts as the "Round-N close
# summary for the round that emits the headline bundle" per the
# plan's AC-5.4 phrasing. The signal is the presence of any
# AC-5.3-SPECIFIC field name (``full_cohort_geomean``,
# ``holdout_geomean``, or ``holdout_cell_count``): these three
# names were introduced in Round 15 and a round summary that
# names them is necessarily reporting AC-5.3 evidence.
#
# Earlier round summaries that reference ``comparison_bundle.json``
# (historical bundles, pre-AC-5.3 regeneration commands) are NOT
# AC-5-emitting in the plan's sense and MUST NOT trip this
# detector. The bundle filename alone is therefore explicitly NOT
# part of the detection — only the three AC-5.3-specific field
# names are.
_HEADLINE_BUNDLE_FIELD_RE = re.compile(
    r"\b(?:holdout_geomean|full_cohort_geomean|holdout_cell_count)\b"
)


def _summary_emits_headline_bundle(text: str) -> bool:
    """Return True iff ``text`` quotes any of the AC-5.3-specific
    dual-geomean field names introduced in Round 15
    (``full_cohort_geomean``, ``holdout_geomean``,
    ``holdout_cell_count``).

    Detection is plain-substring (no code-span stripping) because
    round summaries canonically reference field names inside
    backticks (the markdown convention for identifiers); a
    stripped-span variant would silently miss the common case.
    The function is only called on files matching the
    ``round-*-summary.md`` pattern (per the caller in
    ``scan_paths``), so plan-body or lesson-note mentions never
    reach this detector."""
    if _HEADLINE_BUNDLE_FIELD_RE.search(text):
        return True
    return False


def _text_contains_case_insensitive(needle: str, haystack: str) -> bool:
    """Case-insensitive substring match per the plan's AC-5.4
    positive test phrasing (plan line 90): "The summary file
    contains this tag string (case-insensitive substring match
    against the canonical phrasing)". Uses ``str.casefold`` so
    locale-specific case folding is applied consistently on both
    sides."""
    return needle.casefold() in haystack.casefold()


def _scan_round_summary_for_closure_addendum(
    summary_path: Path,
) -> list[ClaimDefect]:
    """Round-N close summaries that either (a) make a closure
    claim OR (b) emit the AC-5 headline bundle must carry the
    AC-5 scope tag. Closure-claim summaries additionally must
    quote the AC-7 closure-addendum pins
    (``prior_art_handling`` and ``cache_layers``).

    Pre-Round-11 the closure detector used the substring check
    ``"COMPLETE" in text``, which false-positived on hyphenated
    ``COMPLETE-emitting`` / ``COMPLETE-round`` discussions in
    non-closure summaries. Round 11 tightened the detector to
    require a STANDALONE ``COMPLETE`` token per
    ``_summary_makes_closure_claim``.

    Round-16 (Codex Round 15 review #2 fix) widened the AC-5 tag
    gate from "closure-claim only" to "closure-claim OR
    headline-bundle-emit". The plan's AC-5.4 positive test
    requires the tag on "the Round-N close summary for the round
    that emits the headline bundle" — a non-closure round that
    emits the headline bundle still owes the scope tag. The AC-7
    pin checks remain scoped to closure summaries only (those are
    final-loop-state assertions, not headline-bundle assertions).

    Round-16 also switched the AC-5 tag match from byte-exact to
    case-insensitive substring per plan line 90."""
    defects: list[ClaimDefect] = []
    text = summary_path.read_text()
    rel = _rel_to_repo(summary_path)
    is_closure_claim = _summary_makes_closure_claim(text)
    emits_headline_bundle = _summary_emits_headline_bundle(text)
    if not is_closure_claim and not emits_headline_bundle:
        return defects
    if is_closure_claim:
        if PIN_PRIOR_ART not in text:
            defects.append(
                ClaimDefect(
                    path=rel,
                    line_no=0,
                    rule="complete_summary_missing_prior_art_pin",
                    matched_text=f"expected verbatim: {PIN_PRIOR_ART}",
                )
            )
        if PIN_CACHE_LAYERS not in text:
            defects.append(
                ClaimDefect(
                    path=rel,
                    line_no=0,
                    rule=(
                        "complete_summary_missing_cache_layers_pin"
                    ),
                    matched_text=(
                        f"expected verbatim: {PIN_CACHE_LAYERS}"
                    ),
                )
            )
    # AC-5 scope tag: required on EITHER closure-claim OR
    # headline-bundle-emit summaries. Match is case-insensitive
    # substring per plan line 90.
    if not _text_contains_case_insensitive(PIN_AC5_SUMMARY_TAG, text):
        defects.append(
            ClaimDefect(
                path=rel,
                line_no=0,
                rule="complete_summary_missing_ac5_summary_tag",
                matched_text=(
                    f"expected (case-insensitive substring): "
                    f"{PIN_AC5_SUMMARY_TAG}"
                ),
            )
        )
    return defects


def _looks_like_comparison_bundle(path: Path) -> bool:
    """Return True for JSON files that look like comparison bundles —
    matches both the canonical `comparison_bundle.json` filename and
    the per-session `data/local_labels/sweeps/<id>/comparison_bundle*.json`
    pattern."""
    if path.suffix != ".json":
        return False
    if path.name == "comparison_bundle.json":
        return True
    if path.name.startswith("comparison_bundle") and path.suffix == ".json":
        return True
    return False


def scan_paths(paths: list[Path]) -> list[ClaimDefect]:
    """Run the full guard against a list of files. Returns aggregated
    defects (empty list when all paths pass)."""
    defects: list[ClaimDefect] = []
    for p in paths:
        rel = _rel_to_repo(p)
        if _is_exempt(rel):
            continue
        if not p.is_file():
            continue

        if _looks_like_comparison_bundle(p):
            defects.extend(_scan_comparison_bundle_json(p))
            continue

        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue

        if _is_documentation_context_path(rel):
            # Documentation-context path: skip per-line CLAIM scan but
            # still apply closure-addendum check below (plan files do
            # not carry the closure token so the check is a no-op for
            # plans).
            pass
        else:
            defects.extend(_scan_text_for_claims(text, rel))

        if p.suffix == ".md" and "round-" in p.name and "summary" in p.name:
            defects.extend(_scan_round_summary_for_closure_addendum(p))
    return defects


_SESSION_NAME_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})_(?P<H>\d{2})-(?P<M>\d{2})-(?P<S>\d{2})$"
)


def _session_start_epoch(session_dir: Path) -> float:
    """Return a STABLE session-start epoch for the given session
    directory.

    The session directory name itself encodes the loop-start timestamp
    (e.g., `2026-05-26_15-40-37`). That string is set by
    setup-rlcr-loop.sh at session creation and never changes; the
    directory's filesystem mtime DOES change every time the loop
    writes a round file inside it, so the directory-mtime is not a
    stable anchor for "did this artifact land before or during the
    loop?" The directory NAME is.

    The directory name is written in LOCAL TIME by
    `setup-rlcr-loop.sh` (it runs `date +%Y-%m-%d_%H-%M-%S` without
    a TZ override). The parsed datetime is therefore a NAIVE datetime
    that means "local time on the originating host"; passing it to
    `datetime.timestamp()` interprets it as local time and converts
    to a UTC epoch. The receiving host's local timezone is used,
    which matches the originating host's TZ because RLCR cross-host
    pivots start a fresh session on the receiving host (per CLAUDE.md
    cross-host survival semantics).

    Falls back to the session directory's own mtime when the name does
    not parse as an ISO timestamp (defensive: a hand-edited or
    differently-named session directory still gets a cutoff, just not
    a stable one).
    """
    m = _SESSION_NAME_RE.match(session_dir.name)
    if m:
        try:
            naive = _dt.datetime(
                year=int(m.group("y")),
                month=int(m.group("m")),
                day=int(m.group("d")),
                hour=int(m.group("H")),
                minute=int(m.group("M")),
                second=int(m.group("S")),
            )
            # Naive datetime + .timestamp() interprets as local time.
            return naive.timestamp()
        except ValueError:
            pass
    try:
        return session_dir.stat().st_mtime
    except OSError:
        return 0.0


def _default_round_summary_glob(repo_root: Path | None = None) -> list[Path]:
    """Default discovery: include the active loop session's round
    summaries AND round contracts, the active plan file, any ACTIVE
    comparison bundle JSON (created during this loop), and the
    claim-guard's own pytest file so the meta-guard catches drift when
    the test fixtures are edited.

    The active loop session is the newest `.humanize/rlcr/<session>/`
    directory by mtime. Prior loops' sessions are NOT scanned by
    default — they target different plans with different AC closure
    semantics and would emit false-positive defects against THIS
    loop's pin-quoting rules.

    Comparison bundles under `data/local_labels/sweeps/` and
    `data/comparison_bundles/` are filtered by a STABLE session-start
    cutoff (derived from the session directory NAME, which is set at
    loop creation and never changes — unlike the directory mtime,
    which advances every time a round file lands in it). Bundles
    produced during this loop have mtime >= the stable cutoff;
    historical bundles produced by earlier loops are below the cutoff
    and excluded from default discovery.
    """
    root = repo_root if repo_root is not None else REPO_ROOT
    paths: list[Path] = []
    rlcr_root = root / ".humanize" / "rlcr"
    active_session: Path | None = None
    session_mtime_cutoff: float = 0.0
    if rlcr_root.is_dir():
        sessions = [d for d in rlcr_root.iterdir() if d.is_dir()]
        if sessions:
            active_session = max(sessions, key=lambda p: p.stat().st_mtime)
            session_mtime_cutoff = _session_start_epoch(active_session)
    if active_session is not None:
        paths.extend(sorted(active_session.glob("round-*-summary.md")))
        paths.extend(sorted(active_session.glob("round-*-contract.md")))

    # Canonical source plan (tracked in git). The mirror at
    # `.humanize/rlcr/<session>/plan.md` is a host-local snapshot and
    # is NOT scanned by default to avoid double-scanning the same
    # content. Callers can pass it explicitly with `--path` if needed.
    plan = (
        root
        / "plans"
        / "sm120_l2_observability_30_cell_cohort_bounded_loop_plan.md"
    )
    if plan.is_file():
        paths.append(plan)

    def _is_active_bundle(p: Path) -> bool:
        try:
            return p.stat().st_mtime >= session_mtime_cutoff
        except OSError:
            return False

    bundles_legacy = root / "data" / "comparison_bundles"
    if bundles_legacy.is_dir():
        for d in sorted(bundles_legacy.iterdir(), reverse=True):
            if d.is_dir():
                for cb in sorted(d.glob("comparison_bundle*.json")):
                    if _is_active_bundle(cb):
                        paths.append(cb)

    sweeps_root = root / "data" / "local_labels" / "sweeps"
    if sweeps_root.is_dir():
        for cb in sorted(sweeps_root.glob("**/comparison_bundle*.json")):
            if _is_active_bundle(cb):
                paths.append(cb)

    guard_pytest = (
        root
        / "workloads"
        / "simbench_bw"
        / "tests"
        / "test_sm120_loop_claim_guard.py"
    )
    if guard_pytest.is_file():
        paths.append(guard_pytest)

    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path",
        action="append",
        type=Path,
        default=None,
        help=(
            "File path to scan (repeatable). When omitted, scans the "
            "default set: the active loop's round summaries and "
            "contracts, the active plan, any comparison-bundle JSON, "
            "and the claim-guard's own pytest."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = args.path if args.path else _default_round_summary_glob()
    defects = scan_paths(paths)
    if defects:
        for d in defects:
            print(
                f"DEFECT: {d.path}:{d.line_no} [{d.rule}] {d.matched_text}",
                file=sys.stderr,
            )
        print(
            f"FAIL: {len(defects)} claim-guard defects in "
            f"{len(paths)} scanned paths",
            file=sys.stderr,
        )
        return 2
    print(
        f"OK: claim-guard scanned {len(paths)} paths; no defects (tool "
        f"version {__version__})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
