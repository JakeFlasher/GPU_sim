#!/usr/bin/env python3
"""Validate and emit standalone Fable-5 consultation packages.

Fable-5 is treated as a stateless reader with no repository or local-system
access.  This module enforces the repository's outbound packaging contract,
verifies embedded repository evidence when requested, validates an independent
review sidecar, and provides a native Codex Stop-hook backstop.

The implementation deliberately uses only the Python standard library so the
guard can run before the project's optional dependencies are available.
It never opens the online chat platform, sends network requests, authenticates,
or reads browser sessions/cookies; delivery is always a user's manual action.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence


EXIT_CONTRACT = 1
EXIT_INPUT = 2
EXIT_SECURITY = 3
EXIT_AUDIT = 4
EXIT_REPOSITORY = 5

SESSION_DECLARATION = """FABLE5-CONSULTATION-V1

This is a new, standalone Fable-5 session. You have no prior conversation
context and no access to the repository, filesystem, shell, Git, tools, local
links, hidden project state, or earlier consultations. Decide solely from the
literal context embedded in this prompt. A filename, path, URL, commit, section
reference, or prior-ruling label is an identifier only and is never a
substitute for its necessary content.
"""

REQUIRED_METADATA = collections.OrderedDict(
    (
        ("FRESH_SESSION", "true"),
        ("PRIOR_CONTEXT_REQUIRED", "false"),
        ("FABLE_REPOSITORY_ACCESS", "none"),
        ("FABLE_SYSTEM_ACCESS", "none"),
        ("STANDALONE_CONTEXT_COMPLETE", "true"),
        ("INDEPENDENT_AUDIT_COMPLETE", "true"),
        ("CONTEXT_CURATED", "true"),
        ("SECRETS_EXCLUDED", "true"),
        ("SERIAL_FOLLOWUPS_PROHIBITED", "true"),
        ("FILE_CONTEXT_MODE", "embedded"),
    )
)

REQUIRED_HEADINGS = (
    "## 1. Objective and requested decision",
    "## 2. Repository and project state",
    "## 3. Safety, authority, and mutation constraints",
    "## 4. Work completed and exact observed results",
    "## 5. Incorporated decision history",
    "## 6. Consolidated blocker index",
    "## 7. Evidence index",
    "## 8. Embedded evidence",
    "## 9. Cross-blocker interactions and alternatives",
    "## 10. Questions for adjudication",
    "## 11. Required response format",
    "## 12. Acceptance criteria and stop conditions",
    "## 13. Security and redaction attestation",
    "## 14. Independent pre-send review",
)

REQUIRED_AUDIT_REVIEWS = {
    "fresh_session_self_containment",
    "decision_completeness",
    "cross_stage_interactions",
    "context_minimization",
    "security_redaction",
}

FILE_EVIDENCE_KINDS = {"file_full", "file_excerpt"}
EVIDENCE_KINDS = FILE_EVIDENCE_KINDS | {
    "command_output",
    "git_state",
    "prior_ruling",
    "binary_metadata",
    "policy",
}

EVIDENCE_RE = re.compile(
    r"<!-- FABLE5-EVIDENCE\n(?P<meta>\{.*?\})\n-->\n"
    r"(?P<body>.*?)\n<!-- FABLE5-EVIDENCE-END (?P<end>E\d{3,}) -->",
    re.DOTALL,
)
BEGIN_MARKER = "<!-- FABLE5_CONSULTATION_BEGIN -->"
END_MARKER = "<!-- FABLE5_CONSULTATION_END -->"
PROMPT_FILE_LABEL = "FABLE5_PROMPT_FILE"
PROMPT_SHA_LABEL = "FABLE5_PROMPT_SHA256"
HANDOFF_SENTENCE = (
    "The validated Fable-5 prompt file is ready for you to paste or upload manually."
)

DANGLING_REFERENCE_RE = re.compile(
    r"(?i)\b(?:as (?:previously|earlier) (?:ruled|discussed)|"
    r"the (?:previous|prior|preceding|returned|accepted|standing) "
    r"(?:ruling|prompt|message|table|decision)|prior prompt|previous message|"
    r"already accepted|incorporated by reference|same as above|"
    r"continue from (?:your|the) last answer|use the earlier consultation as "
    r"(?:the )?baseline|per the accepted decision table|apply the standing decision|"
    r"everything not listed here stays as ruled)\b"
)
GENERIC_DELTA_RE = re.compile(
    r"(?i)\b(?:all accepted rows|all existing rows|existing table plus|"
    r"everything else remains unchanged|insert (?:the )?(?:battery|table|paths?) here)\b"
)
DEPENDENCY_QUOTED_EXAMPLE_RE = re.compile(
    r"(?i)\b(?:never use|must not use|forbidden phrases?|phrase such as|"
    r"generic rows? such as|reject(?:s|ed)? (?:the )?(?:phrase|reference|token))\b"
)


class Diagnostic:
    """One stable, line-aware protocol diagnostic."""

    __slots__ = ("code", "message", "line", "category")

    def __init__(
        self,
        code: str,
        message: str,
        line: int = 1,
        category: str = "contract",
    ) -> None:
        self.code = code
        self.message = message
        self.line = max(1, line)
        self.category = category

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "code": self.code,
            "line": self.line,
            "message": self.message,
        }

    def __str__(self) -> str:
        return f"{self.code} line {self.line}: {self.message}"


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: str) -> bytes:
    if path == "-":
        return sys.stdin.buffer.read()
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {path!r}: {exc}") from exc


def _decode_prompt(data: bytes) -> tuple[str | None, list[Diagnostic]]:
    errors: list[Diagnostic] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, [
            Diagnostic(
                "FBL001",
                f"prompt is not valid UTF-8 (byte offset {exc.start})",
            )
        ]
    if b"\x00" in data:
        errors.append(Diagnostic("FBL002", "NUL bytes are forbidden"))
    if b"\r\n" in data or b"\r" in data:
        errors.append(Diagnostic("FBL003", "prompt must use LF line endings"))
    if data and not data.endswith(b"\n"):
        errors.append(Diagnostic("FBL004", "prompt must end with one LF"))
    elif data.endswith(b"\n\n"):
        errors.append(Diagnostic("FBL005", "prompt must end with exactly one LF"))
    return text, errors


def _strip_fence(
    body: str, *, final_newline: bool | None
) -> tuple[bytes | None, str | None]:
    """Return literal evidence bytes after removing one Markdown fence."""

    lines = body.splitlines(keepends=True)
    if not lines:
        return None, "evidence body must begin with one Markdown fence"
    opening = lines[0].rstrip("\n")
    opening_match = re.fullmatch(r"(?P<fence>`{3,})(?P<language>[A-Za-z0-9_+.-]*)", opening)
    if not opening_match:
        return None, "evidence fence must contain only backticks and an optional language token"
    fence = opening_match.group("fence")
    if len(lines) < 2 or lines[-1].rstrip("\n") != fence:
        return None, "evidence body must end with the matching Markdown fence"
    content = "".join(lines[1:-1])
    for content_line in content.splitlines():
        tick_match = re.match(r"^(`+)", content_line)
        if tick_match and len(tick_match.group(1)) >= len(fence):
            return None, "evidence fence is not longer than a backtick run in its content"
    if final_newline is False and content.endswith("\n"):
        content = content[:-1]
    elif final_newline is not True and final_newline is not False:
        return None, "evidence metadata requires boolean final_newline"
    return content.encode("utf-8"), None


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _metadata_key_diagnostics(metadata: dict[str, Any], line: int) -> list[Diagnostic]:
    base = {"id", "kind", "role", "supports", "selection_reason", "non_operational", "final_newline"}
    by_kind: dict[str, tuple[set[str], set[str]]] = {
        "file_full": (
            {"path", "revision", "file_sha256", "file_total_lines", "lines", "excerpt_sha256", "full_required_justification"},
            {"path", "revision", "file_sha256", "file_total_lines", "lines", "excerpt_sha256", "full_required_justification"},
        ),
        "file_excerpt": (
            {"path", "revision", "file_sha256", "file_total_lines", "lines", "excerpt_sha256", "omitted_ranges", "omission_reason"},
            {"path", "revision", "file_sha256", "file_total_lines", "lines", "excerpt_sha256", "omitted_ranges", "omission_reason"},
        ),
        "command_output": (
            {"content_sha256", "command", "cwd", "exit_code", "stream"},
            {"content_sha256", "command", "cwd", "exit_code", "stream"},
        ),
        "git_state": (
            {"content_sha256", "command", "cwd", "exit_code", "stream"},
            {"content_sha256", "command", "cwd", "exit_code", "stream"},
        ),
        "prior_ruling": ({"content_sha256", "source_label"}, {"content_sha256", "source_label"}),
        "binary_metadata": (
            {"content_sha256", "artifact_path", "artifact_sha256"},
            {"content_sha256", "artifact_path", "artifact_sha256"},
        ),
        "policy": ({"content_sha256", "source_label"}, {"content_sha256", "source_label"}),
    }
    kind = metadata.get("kind")
    if kind not in by_kind:
        return []
    allowed_extra, required_extra = by_kind[kind]
    allowed = base | allowed_extra
    required = base | required_extra
    errors: list[Diagnostic] = []
    unknown = sorted(set(metadata) - allowed)
    missing = sorted(required - set(metadata))
    if unknown:
        errors.append(Diagnostic("FBL088", f"unknown evidence metadata keys: {', '.join(unknown)}", line))
    if missing:
        errors.append(Diagnostic("FBL089", f"missing evidence metadata keys: {', '.join(missing)}", line))
    return errors


def _section_slices(text: str) -> tuple[dict[str, str], list[Diagnostic]]:
    found: list[tuple[int, str]] = []
    errors: list[Diagnostic] = []
    for heading in REQUIRED_HEADINGS:
        matches = list(re.finditer(rf"(?m)^{re.escape(heading)}\s*$", text))
        if len(matches) != 1:
            errors.append(
                Diagnostic(
                    "FBL011",
                    f"required heading must appear exactly once: {heading!r}",
                )
            )
        elif matches:
            found.append((matches[0].start(), heading))
    if len(found) != len(REQUIRED_HEADINGS):
        return {}, errors
    if [heading for _, heading in sorted(found)] != list(REQUIRED_HEADINGS):
        errors.append(Diagnostic("FBL012", "required headings are out of order"))
        return {}, errors
    found.sort()
    sections: dict[str, str] = {}
    for index, (start, heading) in enumerate(found):
        body_start = text.find("\n", start) + 1
        body_end = found[index + 1][0] if index + 1 < len(found) else len(text)
        body = text[body_start:body_end].strip()
        sections[heading] = body
        if len(body) < 8:
            errors.append(
                Diagnostic(
                    "FBL013",
                    f"required section is empty or non-operative: {heading!r}",
                    _line_number(text, start),
                )
            )
    return sections, errors


def _parse_metadata(text: str) -> list[Diagnostic]:
    errors: list[Diagnostic] = []
    if not text.startswith(SESSION_DECLARATION):
        errors.append(
            Diagnostic(
                "FBL010",
                "prompt must begin with the exact standalone-session declaration",
            )
        )
        return errors
    first_heading = text.find(REQUIRED_HEADINGS[0])
    metadata_region = text[len(SESSION_DECLARATION) : first_heading if first_heading >= 0 else len(text)]
    seen_order: list[str] = []
    parsed: dict[str, str] = {}
    for raw_line in metadata_region.splitlines():
        if not raw_line:
            continue
        match = re.fullmatch(r"([A-Z][A-Z0-9_]+): ([^\s].*)", raw_line)
        if not match:
            errors.append(
                Diagnostic(
                    "FBL022",
                    "metadata block may contain only exact, unindented KEY: value lines",
                )
            )
            continue
        key, value = match.groups()
        if key not in REQUIRED_METADATA:
            errors.append(Diagnostic("FBL023", f"unknown metadata key {key}"))
            continue
        if key in parsed:
            errors.append(Diagnostic("FBL015", f"duplicate metadata key {key}"))
        parsed[key] = value.strip()
        seen_order.append(key)
    for key, expected in REQUIRED_METADATA.items():
        if key not in parsed:
            errors.append(Diagnostic("FBL014", f"missing required metadata key {key}"))
        elif expected is not None and parsed[key] != expected:
            errors.append(
                Diagnostic("FBL016", f"metadata {key} must equal {expected!r}")
            )
    expected_order = [key for key in REQUIRED_METADATA if key in parsed]
    if seen_order != expected_order:
        errors.append(Diagnostic("FBL017", "required metadata keys are out of order"))
    return errors


def _file_context_mode(text: str) -> str | None:
    match = re.search(r"(?m)^FILE_CONTEXT_MODE:\s*(\S+)\s*$", text)
    return match.group(1) if match else None


def _parse_evidence(
    text: str,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]], list[Diagnostic]]:
    evidence: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    errors: list[Diagnostic] = []
    starts = [match.start() for match in re.finditer(r"<!-- FABLE5-EVIDENCE\n", text)]
    ends = [match.start() for match in re.finditer(r"<!-- FABLE5-EVIDENCE-END\b", text)]
    matches = list(EVIDENCE_RE.finditer(text))
    if len(starts) != len(matches) or len(ends) != len(matches):
        errors.append(
            Diagnostic("FBL030", "one or more evidence blocks are malformed or unclosed")
        )
    for match in matches:
        line = _line_number(text, match.start())
        spans.append(match.span())
        try:
            metadata = json.loads(
                match.group("meta"), object_pairs_hook=_json_object_without_duplicates
            )
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(
                Diagnostic(
                    "FBL031",
                    f"evidence metadata is invalid JSON: {getattr(exc, 'msg', str(exc))}",
                    line,
                )
            )
            continue
        if not isinstance(metadata, dict):
            errors.append(Diagnostic("FBL032", "evidence metadata must be an object", line))
            continue
        evidence_id = metadata.get("id")
        if not isinstance(evidence_id, str) or not re.fullmatch(r"E\d{3,}", evidence_id):
            errors.append(Diagnostic("FBL033", "evidence id must match E###", line))
        if evidence_id != match.group("end"):
            errors.append(
                Diagnostic("FBL034", "evidence start/end identifiers do not match", line)
            )
        kind = metadata.get("kind")
        if kind not in EVIDENCE_KINDS:
            errors.append(
                Diagnostic(
                    "FBL035",
                    f"unsupported evidence kind {kind!r}",
                    line,
                )
            )
        role = metadata.get("role")
        if role not in {"decision", "context"}:
            errors.append(Diagnostic("FBL090", "evidence role must be decision or context", line))
        supports = metadata.get("supports")
        supports_valid = isinstance(supports, list) and all(
            isinstance(value, str) and re.fullmatch(r"B\d{3,}", value)
            for value in supports
        )
        if not supports_valid or (role == "decision" and not supports) or (role == "context" and supports != []):
            errors.append(
                Diagnostic(
                    "FBL036",
                    "decision evidence needs a nonempty B### list; context evidence needs []",
                    line,
                )
            )
        reason = metadata.get("selection_reason")
        if not isinstance(reason, str) or len(reason.strip()) < 12:
            errors.append(
                Diagnostic("FBL037", "evidence requires a substantive selection_reason", line)
            )
        if metadata.get("non_operational") is not True:
            errors.append(
                Diagnostic("FBL091", "literal evidence requires non_operational:true", line)
            )
        errors.extend(_metadata_key_diagnostics(metadata, line))
        literal, fence_error = _strip_fence(
            match.group("body"), final_newline=metadata.get("final_newline")
        )
        if fence_error:
            errors.append(Diagnostic("FBL038", fence_error, line))
            literal = b""
        metadata["_literal"] = literal or b""
        metadata["_line"] = line
        metadata["_span"] = match.span()
        evidence.append(metadata)
    return evidence, spans, errors


def _normative_text(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Blank literal evidence while preserving offsets and line numbers."""

    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
        placeholder = "[embedded evidence]"
        for offset, value in enumerate(placeholder):
            if start + offset < end and chars[start + offset] != "\n":
                chars[start + offset] = value
    declaration_end = len(SESSION_DECLARATION) if text.startswith(SESSION_DECLARATION) else 0
    for index in range(declaration_end):
        if chars[index] != "\n":
            chars[index] = " "
    return "".join(chars)


def _security_diagnostics(text: str) -> list[Diagnostic]:
    patterns: tuple[tuple[str, str, re.Pattern[str]], ...] = (
        ("SEC001", "private key material", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
        ("SEC002", "AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
        ("SEC003", "Git hosting access token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_-]{20,})\b")),
        ("SEC004", "API access token", re.compile(r"\b(?:sk-ant-|sk-proj-|sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{20,})[A-Za-z0-9_-]*\b")),
        ("SEC005", "authorization value", re.compile(r"(?i)Authorization\s*:\s*(?:Bearer|Basic)\s+\S+")),
        (
            "SEC006",
            "credential-like assignment",
            re.compile(
                r"(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|session[_-]?cookie)\s*[:=]\s*[\"']?[A-Za-z0-9+/_=-]{12,}"
            ),
        ),
        ("SEC010", "session cookie header", re.compile(r"(?i)(?:Set-)?Cookie\s*:\s*[^\n]*(?:session|auth|token)[^\n]*")),
        ("SEC011", "JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{8,}\b")),
        ("SEC007", "role/system transcript", re.compile(r"(?i)(?:<system>|\"role\"\s*:\s*\"system\")")),
        ("SEC012", "chain-of-thought transcript marker", re.compile(r"(?im)^\s*(?:\*?Thinking\.\.\.\*?|<analysis>|analysis channel:)\s*$")),
    )
    errors: list[Diagnostic] = []
    for code, label, pattern in patterns:
        for match in pattern.finditer(text):
            fingerprint = _sha256(match.group(0).encode("utf-8"))[:12]
            errors.append(
                Diagnostic(
                    code,
                    f"forbidden {label} detected (fingerprint {fingerprint})",
                    _line_number(text, match.start()),
                    "security",
                )
            )
    return errors


def _normative_diagnostics(text: str) -> list[Diagnostic]:
    errors: list[Diagnostic] = []
    rules: tuple[tuple[str, str, re.Pattern[str]], ...] = (
        (
            "FBL040",
            "dangling cross-session reference",
            DANGLING_REFERENCE_RE,
        ),
        (
            "FBL041",
            "generic delta/table substitute",
            GENERIC_DELTA_RE,
        ),
        (
            "FBL042",
            "local Markdown link cannot substitute for embedded content",
            re.compile(
                r"\]\(<?(?:file://|vscode://|/|~/|\.\.?/)[^)>]*>?\)"
                r"|\]\((?![A-Za-z][A-Za-z0-9+.-]*://|mailto:|#)[^)]+\)"
            ),
        ),
        (
            "FBL043",
            "Fable-5 cannot be asked to inspect or run against the repository",
            re.compile(
                r"(?i)\b(?:Fable-?5|you)\s+(?:must|should|can|please|need to)?\s*(?:inspect|open|read|run|query|browse|check out|access)\s+(?:the\s+)?(?:repository|repo|filesystem|file|Git|shell|command)\b"
            ),
        ),
        (
            "FBL044",
            "operative placeholder",
            re.compile(r"(?i)<(?:path|sha(?:256)?|commit|command|count|file|insert[^>]*)>|\{\{[^{}\n]+\}\}|\b(?:TBD|TODO|FIXME)\b|\?\?\?|\+\s*X\b"),
        ),
        (
            "FBL045",
            "truncated hexadecimal digest",
            re.compile(r"(?i)\b[0-9a-f]{7,63}(?:…|\.\.\.)"),
        ),
        (
            "FBL046",
            "prose command placeholder",
            re.compile(r"(?i)<staged-tree|run (?:the )?(?:same|full) battery|commands? (?:as|from) above"),
        ),
    )
    for code, label, pattern in rules:
        for match in pattern.finditer(text):
            errors.append(
                Diagnostic(code, label, _line_number(text, match.start()))
            )
    return errors


def _duplicate_diagnostics(normative: str, *, strict: bool) -> list[Diagnostic]:
    paragraphs: dict[str, list[int]] = collections.defaultdict(list)
    offset = 0
    for paragraph in re.split(r"\n\s*\n", normative):
        stripped = re.sub(r"\s+", " ", paragraph).strip()
        found = normative.find(paragraph, offset)
        offset = max(offset, found + len(paragraph))
        if len(stripped) >= 120:
            paragraphs[stripped].append(_line_number(normative, max(0, found)))
    diagnostics: list[Diagnostic] = []
    for value, lines in paragraphs.items():
        if len(lines) > 1:
            diagnostics.append(
                Diagnostic(
                    "FBL047" if strict else "WRN047",
                    f"duplicate normative paragraph appears {len(lines)} times",
                    lines[1],
                    "contract" if strict else "warning",
                )
            )
    return diagnostics


def _defined_ids(section: str, prefix: str) -> list[str]:
    return re.findall(rf"(?m)^\s*(?:[-*]\s*)?({prefix}\d{{3,}})\s*[:(\-]", section)


def _graph_diagnostics(
    text: str,
    sections: dict[str, str],
    evidence: list[dict[str, Any]],
) -> list[Diagnostic]:
    errors: list[Diagnostic] = []
    blockers = _defined_ids(sections.get(REQUIRED_HEADINGS[5], ""), "B")
    questions = _defined_ids(sections.get(REQUIRED_HEADINGS[9], ""), "Q")
    evidence_ids = [value.get("id") for value in evidence if isinstance(value.get("id"), str)]
    for label, ids in (("blocker", blockers), ("question", questions), ("evidence", evidence_ids)):
        for duplicate in sorted(key for key, count in collections.Counter(ids).items() if count > 1):
            errors.append(Diagnostic("FBL050", f"duplicate {label} id {duplicate}"))
    if not blockers:
        errors.append(Diagnostic("FBL051", "consolidated blocker index defines no B### item"))
    if not questions:
        errors.append(Diagnostic("FBL052", "adjudication section defines no Q### item"))
    if not evidence_ids:
        errors.append(Diagnostic("FBL053", "embedded evidence section contains no evidence"))
    blocker_set, evidence_set = set(blockers), set(evidence_ids)
    supported: dict[str, set[str]] = collections.defaultdict(set)
    support_edges: dict[str, set[str]] = {}
    roles: dict[str, str] = {}
    for item in evidence:
        evidence_id = item.get("id")
        if isinstance(evidence_id, str):
            roles[evidence_id] = str(item.get("role"))
            support_edges[evidence_id] = set(
                item.get("supports", []) if isinstance(item.get("supports"), list) else []
            )
        for blocker in item.get("supports", []) if isinstance(item.get("supports"), list) else []:
            if blocker not in blocker_set:
                errors.append(
                    Diagnostic(
                        "FBL054",
                        f"evidence {evidence_id} supports undefined blocker {blocker}",
                        int(item.get("_line", 1)),
                    )
                )
            else:
                supported[blocker].add(str(evidence_id))
    for blocker in sorted(blocker_set - set(supported)):
        errors.append(Diagnostic("FBL055", f"blocker {blocker} has no supporting evidence"))
    evidence_index = sections.get(REQUIRED_HEADINGS[6], "")
    question_section = sections.get(REQUIRED_HEADINGS[9], "")
    for evidence_id in sorted(evidence_set):
        index_match = re.search(
            rf"(?m)^.*\b{re.escape(evidence_id)}\b.*$", evidence_index
        )
        if not index_match:
            errors.append(Diagnostic("FBL056", f"evidence {evidence_id} is absent from the evidence index"))
        elif roles.get(evidence_id) == "decision":
            for blocker in support_edges.get(evidence_id, set()):
                if not re.search(rf"\b{re.escape(blocker)}\b", index_match.group(0)):
                    errors.append(
                        Diagnostic(
                            "FBL092",
                            f"evidence-index row for {evidence_id} omits declared support {blocker}",
                        )
                    )
        if roles.get(evidence_id) == "decision" and not re.search(
            rf"\b{re.escape(evidence_id)}\b", question_section
        ):
            errors.append(Diagnostic("FBL057", f"evidence {evidence_id} is orphaned from all questions"))
    for blocker in sorted(blocker_set):
        if not re.search(rf"\b{re.escape(blocker)}\b", question_section):
            errors.append(Diagnostic("FBL058", f"blocker {blocker} is not cited by a question"))
    for question in questions:
        line_match = re.search(rf"(?m)^.*\b{re.escape(question)}\b.*$", question_section)
        line = line_match.group(0) if line_match else ""
        cited_blockers = set(re.findall(r"\bB\d{3,}\b", line))
        cited_evidence = set(re.findall(r"\bE\d{3,}\b", line))
        if not cited_blockers or not cited_evidence:
            errors.append(
                Diagnostic("FBL059", f"question {question} must cite B### and E### on its defining line")
            )
            continue
        for evidence_id in cited_evidence:
            if evidence_id not in evidence_set:
                continue
            if roles.get(evidence_id) != "decision":
                errors.append(
                    Diagnostic(
                        "FBL093",
                        f"question {question} cites context-only evidence {evidence_id}",
                    )
                )
            elif not (support_edges.get(evidence_id, set()) & cited_blockers):
                errors.append(
                    Diagnostic(
                        "FBL094",
                        f"question {question} crosses an unsupported blocker/evidence edge at {evidence_id}",
                    )
                )
        for blocker in cited_blockers & blocker_set:
            if not any(
                blocker in support_edges.get(evidence_id, set())
                for evidence_id in cited_evidence
            ):
                errors.append(
                    Diagnostic(
                        "FBL095",
                        f"question {question} cites blocker {blocker} without supporting evidence on the same line",
                    )
                )
    normative = _normative_text(
        text, [item.get("_span") for item in evidence if isinstance(item.get("_span"), tuple)]
    )
    for prefix, defined in (("B", blocker_set), ("Q", set(questions)), ("E", evidence_set)):
        referenced = set(re.findall(rf"\b{prefix}\d{{3,}}\b", normative))
        for unresolved in sorted(referenced - defined):
            errors.append(Diagnostic("FBL064", f"unresolved {prefix}-reference {unresolved}"))
    return errors


def _content_contract_diagnostics(sections: dict[str, str]) -> list[Diagnostic]:
    errors: list[Diagnostic] = []
    state = sections.get(REQUIRED_HEADINGS[1], "")
    state_requirements = (
        ("Repository", r"(?m)^Repository:\s*\S"),
        ("Branch", r"(?m)^Branch:\s*\S"),
        ("HEAD", r"(?m)^HEAD:\s*`?[0-9a-f]{40}`?\s*$"),
        ("Index state", r"(?m)^Index state:\s*\S"),
        ("Working tree state", r"(?m)^Working tree state:\s*\S"),
        ("Project/RLCR state", r"(?m)^Project/RLCR state:\s*\S"),
    )
    for label, pattern in state_requirements:
        if not re.search(pattern, state):
            errors.append(Diagnostic("FBL060", f"repository state lacks literal {label} field"))
    response = sections.get(REQUIRED_HEADINGS[10], "").lower()
    response_terms = (
        "consolidated ruling",
        "answer every q",
        "task-specific deliverables",
        "mechanical acceptance predicate",
        "no placeholders",
        "no serial remands",
    )
    for term in response_terms:
        if term not in response:
            errors.append(Diagnostic("FBL061", f"required response format omits {term!r}"))
    security = sections.get(REQUIRED_HEADINGS[12], "")
    for term in ("credentials excluded", "hidden prompts excluded", "decision-critical hashes unredacted"):
        if term not in security.lower():
            errors.append(Diagnostic("FBL062", f"security attestation omits {term!r}"))
    review = sections.get(REQUIRED_HEADINGS[13], "")
    for label in (
        "Author:",
        "Independent reviewer:",
        "Review result:",
        "Decision evidence reviewed:",
        "Review records:",
        "Context removed:",
    ):
        if label not in review:
            errors.append(Diagnostic("FBL063", f"independent review section omits {label}"))
    return errors


def _evidence_schema_diagnostics(evidence: list[dict[str, Any]]) -> list[Diagnostic]:
    errors: list[Diagnostic] = []
    sensitive_parts = {
        ".aws",
        ".env",
        ".git",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".ssh",
        "credentials",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
        "secrets",
        "token.txt",
    }
    literal_hashes: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for item in evidence:
        line = int(item.get("_line", 1))
        kind = item.get("kind")
        literal = item.get("_literal", b"")
        literal_hashes[_sha256(literal)].append((str(item.get("id")), line))
        if kind in FILE_EVIDENCE_KINDS:
            path = item.get("path")
            if not isinstance(path, str) or not path:
                errors.append(Diagnostic("FBL070", "file evidence requires path", line))
            else:
                pure = PurePosixPath(path)
                if pure.is_absolute() or ".." in pure.parts:
                    errors.append(Diagnostic("FBL071", "file evidence path must be repository-relative without '..'", line))
                if any(
                    part.lower() in sensitive_parts
                    or part.lower().startswith(".env.")
                    for part in pure.parts
                ):
                    errors.append(
                        Diagnostic("SEC008", "credential-bearing evidence path is forbidden", line, "security")
                    )
            revision = item.get("revision")
            if not isinstance(revision, str) or not re.fullmatch(
                r"(?:HEAD|INDEX|WORKTREE|COMMIT:[0-9a-f]{40})", revision
            ):
                errors.append(Diagnostic("FBL072", "file evidence has invalid revision", line))
            for key in ("file_sha256", "excerpt_sha256"):
                if not isinstance(item.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", item[key]):
                    errors.append(Diagnostic("FBL073", f"file evidence requires full {key}", line))
            excerpt_hash = item.get("excerpt_sha256")
            if isinstance(excerpt_hash, str) and re.fullmatch(r"[0-9a-f]{64}", excerpt_hash):
                if _sha256(literal) != excerpt_hash:
                    errors.append(
                        Diagnostic(
                            "FBL082",
                            "embedded file evidence does not match excerpt_sha256",
                            line,
                        )
                    )
            ranges = item.get("lines")
            total_lines = item.get("file_total_lines")
            if not isinstance(total_lines, int) or total_lines < 1:
                errors.append(
                    Diagnostic("FBL083", "file evidence requires positive file_total_lines", line)
                )
            if (
                not isinstance(ranges, list)
                or len(ranges) != 2
                or not all(isinstance(value, int) and value >= 1 for value in ranges)
                or ranges[0] > ranges[1]
            ):
                errors.append(Diagnostic("FBL074", "file evidence lines must be [start,end]", line))
            elif isinstance(total_lines, int) and ranges[1] > total_lines:
                errors.append(Diagnostic("FBL084", "file evidence range exceeds file_total_lines", line))
            if isinstance(ranges, list) and len(ranges) == 2 and all(
                isinstance(value, int) for value in ranges
            ):
                literal_line_count = len(literal.splitlines(keepends=True))
                if literal_line_count != ranges[1] - ranges[0] + 1:
                    errors.append(
                        Diagnostic(
                            "FBL096",
                            "embedded file line count does not equal the declared range",
                            line,
                        )
                    )
            if kind == "file_excerpt":
                omitted_ranges = item.get("omitted_ranges")
                if not isinstance(omitted_ranges, list):
                    errors.append(Diagnostic("FBL075", "file excerpt requires omitted_ranges", line))
                elif (
                    isinstance(ranges, list)
                    and len(ranges) == 2
                    and isinstance(total_lines, int)
                    and all(isinstance(value, int) for value in ranges)
                ):
                    expected_omissions: list[list[int]] = []
                    if ranges[0] > 1:
                        expected_omissions.append([1, ranges[0] - 1])
                    if ranges[1] < total_lines:
                        expected_omissions.append([ranges[1] + 1, total_lines])
                    if omitted_ranges != expected_omissions:
                        errors.append(
                            Diagnostic(
                                "FBL085",
                                "omitted_ranges must exactly cover every line outside the excerpt",
                                line,
                            )
                        )
                if not isinstance(item.get("omission_reason"), str) or len(item["omission_reason"].strip()) < 12:
                    errors.append(Diagnostic("FBL076", "file excerpt requires substantive omission_reason", line))
            else:
                if (
                    isinstance(ranges, list)
                    and len(ranges) == 2
                    and isinstance(total_lines, int)
                    and ranges != [1, total_lines]
                ):
                    errors.append(
                        Diagnostic(
                            "FBL086",
                            "file_full lines must be exactly [1,file_total_lines]",
                            line,
                        )
                    )
                if not isinstance(item.get("full_required_justification"), str) or len(item["full_required_justification"].strip()) < 12:
                    errors.append(Diagnostic("FBL077", "full file requires full_required_justification", line))
                if (
                    isinstance(item.get("file_sha256"), str)
                    and isinstance(item.get("excerpt_sha256"), str)
                    and item["file_sha256"] != item["excerpt_sha256"]
                ):
                    errors.append(
                        Diagnostic(
                            "FBL097",
                            "file_full requires identical file_sha256 and excerpt_sha256",
                            line,
                        )
                    )
        else:
            content_hash = item.get("content_sha256")
            if not isinstance(content_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", content_hash):
                errors.append(Diagnostic("FBL078", "non-file evidence requires full content_sha256", line))
            elif _sha256(literal) != content_hash:
                errors.append(Diagnostic("FBL079", "evidence body does not match content_sha256", line))
            if not literal and kind != "git_state":
                errors.append(
                    Diagnostic("FBL098", "non-file decision evidence may not be empty", line)
                )
            if kind == "git_state" and item.get("command") != "git status --porcelain=v1 --untracked-files=all":
                errors.append(
                    Diagnostic(
                        "FBL081",
                        "git_state evidence command must be exactly 'git status --porcelain=v1 --untracked-files=all'",
                        line,
                    )
                )
            if kind in {"git_state", "command_output"}:
                if not isinstance(item.get("command"), str) or not item["command"].strip():
                    errors.append(Diagnostic("FBL099", "command evidence requires an exact command", line))
                if not isinstance(item.get("cwd"), str) or not item["cwd"].strip():
                    errors.append(Diagnostic("FBL100", "command evidence requires cwd", line))
                if not isinstance(item.get("exit_code"), int):
                    errors.append(Diagnostic("FBL101", "command evidence requires integer exit_code", line))
                if item.get("stream") not in {"stdout", "stderr", "combined"}:
                    errors.append(Diagnostic("FBL102", "command evidence requires a valid stream", line))
            if kind == "git_state" and item.get("role") != "context":
                errors.append(Diagnostic("FBL103", "git_state evidence must have context role", line))
            if kind == "git_state" and (
                item.get("cwd") != "."
                or item.get("exit_code") != 0
                or item.get("stream") != "stdout"
            ):
                errors.append(
                    Diagnostic(
                        "FBL104",
                        "git_state evidence requires cwd '.', exit_code 0, and stream stdout",
                        line,
                    )
                )
            if kind == "binary_metadata" and (
                not isinstance(item.get("artifact_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["artifact_sha256"])
                or not isinstance(item.get("artifact_path"), str)
                or not item["artifact_path"].strip()
            ):
                errors.append(
                    Diagnostic(
                        "FBL105",
                        "binary_metadata requires artifact_path and full artifact_sha256",
                        line,
                    )
                )
        if re.search(rb"(?i)ignore (?:all )?(?:previous|prior) instructions", literal) and item.get("non_operational") is not True:
            errors.append(
                Diagnostic(
                    "SEC009",
                    "instruction-like literal evidence requires non_operational:true",
                    line,
                    "security",
                )
            )
    for occurrences in literal_hashes.values():
        if len(occurrences) > 1:
            ids = ", ".join(value[0] for value in occurrences)
            errors.append(
                Diagnostic(
                    "FBL080",
                    f"duplicate literal evidence must be consolidated under one ID: {ids}",
                    occurrences[1][1],
                )
            )
    return errors


def _decision_evidence_dependency_diagnostics(
    evidence: list[dict[str, Any]],
) -> list[Diagnostic]:
    """Reject decision prose that still depends on an unembedded session.

    Repository source files and command output may legitimately contain words
    such as "prior" in code, tests, or diagnostics.  Literal ruling/policy
    blocks are different: when they carry a decision edge, deictic prose is an
    unresolved operand, even if the outer prompt itself is well formed.
    """

    errors: list[Diagnostic] = []
    rules = (
        (
            "FBL106",
            "decision evidence contains a dangling cross-session reference",
            DANGLING_REFERENCE_RE,
        ),
        (
            "FBL107",
            "decision evidence substitutes a generic delta for literal content",
            GENERIC_DELTA_RE,
        ),
    )
    for item in evidence:
        if item.get("role") != "decision" or item.get("kind") not in {
            "policy",
            "prior_ruling",
        }:
            continue
        literal = item.get("_literal", b"")
        if not isinstance(literal, bytes):
            continue
        text = literal.decode("utf-8", errors="replace")
        searchable_parts: list[str] = []
        for paragraph in re.split(r"\n\s*\n", text):
            if DEPENDENCY_QUOTED_EXAMPLE_RE.search(paragraph):
                paragraph = re.sub(r'"[^"\n]*(?:\n[^"\n]*)*"|`[^`\n]+`', "[quoted example]", paragraph)
            searchable_parts.append(paragraph)
        searchable = "\n\n".join(searchable_parts)
        for code, message, pattern in rules:
            match = pattern.search(searchable)
            if match is not None:
                errors.append(
                    Diagnostic(
                        code,
                        f"{message}: {item.get('id')}",
                        int(item.get("_line", 1)),
                    )
                )
    return errors


def lint_prompt(data: bytes, *, strict: bool = True) -> tuple[str | None, list[Diagnostic], list[dict[str, Any]], dict[str, str]]:
    text, errors = _decode_prompt(data)
    if text is None:
        return None, errors, [], {}
    errors.extend(_parse_metadata(text))
    evidence, spans, evidence_errors = _parse_evidence(text)
    errors.extend(evidence_errors)
    errors.extend(_security_diagnostics(text))
    normative = _normative_text(text, spans)
    sections, section_errors = _section_slices(normative)
    errors.extend(section_errors)
    errors.extend(_normative_diagnostics(normative))
    errors.extend(_duplicate_diagnostics(normative, strict=strict))
    if sections:
        errors.extend(_graph_diagnostics(text, sections, evidence))
        errors.extend(_content_contract_diagnostics(sections))
        evidence_start = normative.find(REQUIRED_HEADINGS[7])
        evidence_end = normative.find(REQUIRED_HEADINGS[8])
        for item in evidence:
            span = item.get("_span")
            if (
                not isinstance(span, tuple)
                or evidence_start < 0
                or evidence_end < 0
                or span[0] < evidence_start
                or span[1] > evidence_end
            ):
                errors.append(
                    Diagnostic(
                        "FBL087",
                        f"evidence {item.get('id')} must be physically embedded in section 8",
                        int(item.get("_line", 1)),
                    )
                )
    errors.extend(_evidence_schema_diagnostics(evidence))
    errors.extend(_decision_evidence_dependency_diagnostics(evidence))
    mode = _file_context_mode(text)
    has_file_evidence = any(item.get("kind") in FILE_EVIDENCE_KINDS for item in evidence)
    if mode == "embedded" and not has_file_evidence:
        errors.append(Diagnostic("FBL020", "FILE_CONTEXT_MODE embedded requires file evidence"))
    errors.sort(key=lambda value: (value.line, value.code, value.message))
    return text, errors, evidence, sections


def _git_bytes(repo_root: Path, revision: str, path: str) -> bytes:
    if revision == "WORKTREE":
        candidate_path = repo_root / path
        if candidate_path.is_symlink() or not candidate_path.is_file():
            raise ValueError("WORKTREE evidence path must be a regular nonsymlink file")
        candidate = candidate_path.resolve(strict=True)
        root = repo_root.resolve(strict=True)
        if candidate != root and root not in candidate.parents:
            raise ValueError("resolved evidence path escapes repository root")
        return candidate_path.read_bytes()
    spec = f":{path}" if revision == "INDEX" else f"HEAD:{path}"
    if revision.startswith("COMMIT:"):
        spec = f"{revision.partition(':')[2]}:{path}"
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "show", spec],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise ValueError(f"git cannot resolve {revision}:{path}")
    return completed.stdout


def verify_repository_evidence(repo_root: Path, evidence: list[dict[str, Any]]) -> list[Diagnostic]:
    errors: list[Diagnostic] = []
    try:
        root = repo_root.resolve(strict=True)
    except OSError:
        return [Diagnostic("REP001", "repository root does not exist", category="repository")]
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode or Path(probe.stdout.strip()).resolve() != root:
        return [Diagnostic("REP002", "--repo-root is not the Git worktree root", category="repository")]
    for item in evidence:
        if item.get("kind") not in FILE_EVIDENCE_KINDS:
            continue
        line = int(item.get("_line", 1))
        path, revision = item.get("path"), item.get("revision")
        if not isinstance(path, str) or not isinstance(revision, str):
            continue
        try:
            source = _git_bytes(root, revision, path)
        except (OSError, ValueError) as exc:
            errors.append(Diagnostic("REP003", str(exc), line, "repository"))
            continue
        if _sha256(source) != item.get("file_sha256"):
            errors.append(Diagnostic("REP004", f"full-file bytes drifted for {path}", line, "repository"))
            continue
        ranges = item.get("lines")
        if not isinstance(ranges, list) or len(ranges) != 2:
            continue
        source_lines = source.splitlines(keepends=True)
        start, end = ranges
        if item.get("file_total_lines") != len(source_lines):
            errors.append(
                Diagnostic(
                    "REP012",
                    f"file_total_lines drifted for {path}",
                    line,
                    "repository",
                )
            )
            continue
        if end > len(source_lines):
            errors.append(Diagnostic("REP005", f"line range exceeds {path}", line, "repository"))
            continue
        expected = source if item.get("kind") == "file_full" else b"".join(source_lines[start - 1 : end])
        if _sha256(expected) != item.get("excerpt_sha256"):
            errors.append(Diagnostic("REP006", f"excerpt hash drifted for {path}", line, "repository"))
        if expected != item.get("_literal"):
            errors.append(Diagnostic("REP007", f"embedded excerpt bytes differ from {revision}:{path}", line, "repository"))
    return errors


def verify_repository_context(
    repo_root: Path,
    text: str,
    sections: dict[str, str],
    evidence: list[dict[str, Any]],
) -> list[Diagnostic]:
    """Verify declared repository identity/state and every file excerpt."""

    errors = verify_repository_evidence(repo_root, evidence)
    try:
        root = repo_root.resolve(strict=True)
    except OSError:
        return errors
    state = sections.get(REQUIRED_HEADINGS[1], "")
    fields: dict[str, str] = {}
    for label in ("Repository", "Branch", "HEAD"):
        match = re.search(rf"(?m)^{label}:\s*`?([^`\n]+)`?\s*$", state)
        if match:
            fields[label] = match.group(1).strip()
    try:
        declared_root = Path(fields.get("Repository", "")).resolve(strict=True)
    except OSError:
        declared_root = Path("/")
    if declared_root != root:
        errors.append(Diagnostic("REP008", "declared Repository does not equal --repo-root", category="repository"))
    commands = {
        "HEAD": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "Branch": ["git", "-C", str(root), "symbolic-ref", "--short", "-q", "HEAD"],
    }
    for label, command in commands.items():
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        observed = completed.stdout.strip() if completed.returncode == 0 else (
            "DETACHED" if label == "Branch" else ""
        )
        if fields.get(label) != observed:
            errors.append(Diagnostic("REP009", f"declared {label} differs from repository", category="repository"))
    status_items = [
        item
        for item in evidence
        if item.get("kind") == "git_state"
        and item.get("command") == "git status --porcelain=v1 --untracked-files=all"
    ]
    if len(status_items) != 1:
        errors.append(
            Diagnostic(
                "REP010",
                "exactly one current git-status evidence block is required",
                category="repository",
            )
        )
    else:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode or completed.stdout != status_items[0].get("_literal"):
            errors.append(Diagnostic("REP011", "embedded git status drifted", category="repository"))
        else:
            status_lines = completed.stdout.decode("utf-8", errors="strict").splitlines()
            index_dirty = any(
                len(line) >= 2 and line[0] not in {" ", "?", "!"}
                for line in status_lines
            )
            worktree_dirty = any(
                len(line) >= 2 and (line[1] not in {" ", "!"} or line.startswith("??"))
                for line in status_lines
            )
            expected_states = {
                "Index state": "dirty" if index_dirty else "clean",
                "Working tree state": "dirty" if worktree_dirty else "clean",
            }
            for label, expected in expected_states.items():
                match = re.search(rf"(?m)^{re.escape(label)}:\s*(clean|dirty)\b", state)
                if not match or match.group(1) != expected:
                    errors.append(
                        Diagnostic(
                            "REP013",
                            f"declared {label} contradicts embedded Git status",
                            category="repository",
                        )
                    )
    for item in evidence:
        if item.get("kind") != "binary_metadata":
            continue
        artifact_value = item.get("artifact_path")
        if not isinstance(artifact_value, str):
            continue
        artifact = Path(artifact_value)
        if not artifact.is_absolute():
            if ".." in PurePosixPath(artifact_value).parts:
                errors.append(
                    Diagnostic(
                        "REP016",
                        "binary artifact path traversal is forbidden",
                        int(item.get("_line", 1)),
                        "repository",
                    )
                )
                continue
            artifact = root / artifact
        try:
            if artifact.is_symlink() or not artifact.is_file():
                actual_hash = ""
            else:
                actual_hash = _sha256(artifact.read_bytes())
        except OSError:
            actual_hash = ""
        if actual_hash != item.get("artifact_sha256"):
            errors.append(
                Diagnostic(
                    "REP017",
                    f"binary artifact bytes drifted for {artifact_value}",
                    int(item.get("_line", 1)),
                    "repository",
                )
            )
    return errors


def _evidence_state_digest(repo_root: Path, prompt: bytes) -> str:
    """Bind ignored/machine-local artifacts explicitly cited by the prompt."""

    _, diagnostics, evidence, _ = lint_prompt(prompt, strict=True)
    if diagnostics:
        raise ValueError("cannot digest evidence state for an invalid prompt")
    root = repo_root.resolve(strict=True)
    records: list[tuple[str, str]] = []
    for item in evidence:
        kind = item.get("kind")
        value: str | None = None
        expected: str | None = None
        if kind in FILE_EVIDENCE_KINDS and item.get("revision") == "WORKTREE":
            value = item.get("path") if isinstance(item.get("path"), str) else None
            expected = item.get("file_sha256") if isinstance(item.get("file_sha256"), str) else None
        elif kind == "binary_metadata":
            value = item.get("artifact_path") if isinstance(item.get("artifact_path"), str) else None
            expected = item.get("artifact_sha256") if isinstance(item.get("artifact_sha256"), str) else None
        if value is None or expected is None:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            if candidate.is_symlink():
                actual = "SYMLINK"
            elif not candidate.is_file():
                actual = "MISSING_OR_NONREGULAR"
            else:
                actual = "REGULAR:" + _sha256(candidate.read_bytes())
        except OSError:
            actual = "MISSING"
        records.append((f"{kind}:{value}", actual))
    rendered = b"FABLE5-EVIDENCE-STATE-V1\0" + b"".join(
        os.fsencode(path) + b"\0" + digest.encode("ascii") + b"\0"
        for path, digest in sorted(records)
    )
    return _sha256(rendered)


def _working_state_digest(repo_root: Path) -> str:
    """Hash index identity plus every dirty/untracked nonignored worktree byte."""

    root = repo_root.resolve(strict=True)

    def git_bytes(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode:
            raise ValueError(f"Git state command failed: {' '.join(arguments)}")
        return completed.stdout

    index = git_bytes("ls-files", "--stage", "-z")
    dirty = git_bytes("diff-files", "--name-only", "-z", "--")
    untracked = git_bytes("ls-files", "--others", "--exclude-standard", "-z", "--")
    paths = sorted(set(dirty.rstrip(b"\0").split(b"\0") + untracked.rstrip(b"\0").split(b"\0")))
    paths = [value for value in paths if value]
    digest = hashlib.sha256()
    digest.update(b"FABLE5-WORKING-STATE-V1\0INDEX\0")
    digest.update(index)
    for encoded_path in paths:
        relative = os.fsdecode(encoded_path)
        candidate = root / relative
        digest.update(b"PATH\0" + encoded_path + b"\0")
        try:
            stat_result = candidate.lstat()
        except FileNotFoundError:
            digest.update(b"MISSING\0")
            continue
        digest.update(f"MODE:{stat_result.st_mode:o}\0".encode("ascii"))
        if candidate.is_symlink():
            digest.update(b"SYMLINK\0" + os.fsencode(os.readlink(candidate)) + b"\0")
        elif candidate.is_file():
            digest.update(b"FILE\0" + _sha256(candidate.read_bytes()).encode("ascii") + b"\0")
        elif candidate.is_dir():
            submodule = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            value = submodule.stdout.strip() if submodule.returncode == 0 else b""
            digest.update(b"DIRECTORY\0" + value + b"\0")
        else:
            digest.update(b"SPECIAL\0")
    return digest.hexdigest()


def _git_observation(repo_root: Path) -> tuple[Path, str, str, bytes, str]:
    root = repo_root.resolve(strict=True)
    git_dir_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    head_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    branch_result = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "--short", "-q", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if git_dir_result.returncode or head_result.returncode or status_result.returncode:
        raise ValueError("cannot observe Git identity for consultation receipt")
    git_dir = Path(git_dir_result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = root / git_dir
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "DETACHED"
    return (
        git_dir.resolve(),
        head_result.stdout.strip(),
        branch,
        status_result.stdout,
        _working_state_digest(root),
    )


def _receipt_path(repo_root: Path, prompt_sha256: str) -> Path:
    git_dir, _, _, _, _ = _git_observation(repo_root)
    del prompt_sha256
    return git_dir / "fable5-consultation-receipts" / "active.json"


def _write_emit_receipt(
    repo_root: Path,
    prompt: bytes,
    audit: bytes,
    output_path: Path,
    observation: tuple[Path, str, str, bytes, str] | None = None,
) -> Path:
    git_dir, head, branch, status, working_state = observation or _git_observation(repo_root)
    prompt_sha = _sha256(prompt)
    receipt = {
        "schema_version": 1,
        "prompt_sha256": prompt_sha,
        "audit_sha256": _sha256(audit),
        "repo_root": str(repo_root.resolve(strict=True)),
        "head": head,
        "branch": branch,
        "status_sha256": _sha256(status),
        "working_state_sha256": working_state,
        "evidence_state_sha256": _evidence_state_digest(repo_root, prompt),
        "validator_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "output_path": str(output_path.resolve(strict=True)),
        "output_sha256": prompt_sha,
        "emitted_unix_time": int(time.time()),
    }
    directory = git_dir / "fable5-consultation-receipts"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "active.json"
    for stale in directory.glob("*.json"):
        if stale != destination:
            stale.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination


def verify_emit_receipt(
    repo_root: Path, prompt: bytes, output_path: Path
) -> tuple[bool, str]:
    prompt_sha = _sha256(prompt)
    try:
        resolved_output = output_path.resolve(strict=True)
        if resolved_output.read_bytes() != prompt:
            return False, "the emitted prompt file bytes changed after validation"
        receipt_path = _receipt_path(repo_root, prompt_sha)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _, head, branch, status, working_state = _git_observation(repo_root)
    except (OSError, ValueError, json.JSONDecodeError):
        return False, "no valid emit receipt exists for these exact prompt bytes"
    expected = {
        "schema_version": 1,
        "prompt_sha256": prompt_sha,
        "repo_root": str(repo_root.resolve(strict=True)),
        "head": head,
        "branch": branch,
        "status_sha256": _sha256(status),
        "working_state_sha256": working_state,
        "evidence_state_sha256": _evidence_state_digest(repo_root, prompt),
        "validator_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "output_path": str(resolved_output),
        "output_sha256": prompt_sha,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        return False, "the emit receipt is stale for the current repository or validator"
    if not isinstance(receipt.get("audit_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", receipt["audit_sha256"]
    ):
        return False, "the emit receipt lacks an audit binding"
    return True, ""


def _write_single_prompt_file(output_path: Path, repo_root: Path, data: bytes) -> Path:
    """Atomically create the one user-delivered text artifact outside the repo."""

    if not output_path.is_absolute():
        raise ValueError("--output must be an absolute .txt path")
    output_text = str(output_path)
    if any(ord(value) < 32 for value in output_text) or "`" in output_text:
        raise ValueError("--output may not contain control characters or backticks")
    if output_path.suffix.lower() != ".txt":
        raise ValueError("--output must name one plain-text .txt file")
    root = repo_root.resolve(strict=True)
    parent = output_path.parent.resolve(strict=True)
    resolved = parent / output_path.name
    worktree_result = subprocess.run(
        ["git", "-C", str(root), "worktree", "list", "--porcelain"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if worktree_result.returncode:
        raise ValueError("cannot enumerate repository worktrees for output clearance")
    worktree_paths = [
        Path(line.partition(" ")[2]).resolve(strict=True)
        for line in worktree_result.stdout.splitlines()
        if line.startswith("worktree ")
    ]
    if any(resolved == worktree or worktree in resolved.parents for worktree in worktree_paths):
        raise ValueError("the emitted Fable-5 prompt file must stay outside every repository worktree")
    if resolved.exists():
        raise ValueError("--output must not already exist; emit creates the one new prompt file")
    with tempfile.NamedTemporaryFile("wb", dir=parent, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    try:
        os.link(temporary, resolved)
    except FileExistsError as exc:
        raise ValueError("--output appeared concurrently and was not overwritten") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def validate_audit(
    data: bytes, audit_path: str, *, audit_data: bytes | None = None
) -> list[Diagnostic]:
    try:
        raw_audit = _read_bytes(audit_path) if audit_data is None else audit_data
        audit = json.loads(raw_audit.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [Diagnostic("AUD001", f"cannot read independent audit: {exc}", category="audit")]
    if not isinstance(audit, dict):
        return [Diagnostic("AUD002", "audit root must be an object", category="audit")]
    errors: list[Diagnostic] = []
    required_keys = {
        "schema_version",
        "prompt_sha256",
        "author_id",
        "reviewer_id",
        "result",
        "reviewed",
        "decision_evidence_ids",
        "review_records",
        "unresolved_findings",
        "orphan_evidence",
        "duplicate_blocks",
        "context_removed_chars",
        "reviewer_summary",
    }
    missing = sorted(required_keys - set(audit))
    if missing:
        errors.append(Diagnostic("AUD003", f"audit lacks keys: {', '.join(missing)}", category="audit"))
        return errors
    if audit.get("schema_version") != 3:
        errors.append(Diagnostic("AUD004", "audit schema_version must equal 3", category="audit"))
    if audit.get("prompt_sha256") != _sha256(data):
        errors.append(Diagnostic("AUD005", "audit prompt_sha256 is stale", category="audit"))
    author, reviewer = audit.get("author_id"), audit.get("reviewer_id")
    if not isinstance(author, str) or not author.strip() or not isinstance(reviewer, str) or not reviewer.strip():
        errors.append(Diagnostic("AUD006", "audit author_id and reviewer_id must be nonempty", category="audit"))
    elif author.strip().rstrip("/") == reviewer.strip().rstrip("/"):
        errors.append(Diagnostic("AUD007", "audit author and reviewer must be different agents", category="audit"))
    if audit.get("result") != "pass":
        errors.append(Diagnostic("AUD008", "independent audit result must be pass", category="audit"))
    reviewed = audit.get("reviewed")
    if (
        not isinstance(reviewed, list)
        or not all(isinstance(value, str) for value in reviewed)
        or not REQUIRED_AUDIT_REVIEWS.issubset(set(reviewed))
    ):
        errors.append(Diagnostic("AUD009", "audit does not cover every mandatory review dimension", category="audit"))
    if audit.get("unresolved_findings") != []:
        errors.append(Diagnostic("AUD010", "audit has unresolved findings", category="audit"))
    if audit.get("orphan_evidence") != 0 or audit.get("duplicate_blocks") != 0:
        errors.append(Diagnostic("AUD011", "audit reports orphan or duplicate context", category="audit"))
    removed = audit.get("context_removed_chars")
    if not isinstance(removed, int) or removed < 0:
        errors.append(Diagnostic("AUD012", "context_removed_chars must be a nonnegative integer", category="audit"))
    summary = audit.get("reviewer_summary")
    if not isinstance(summary, str) or len(summary.strip()) < 20 or summary.strip().upper() == "PENDING":
        errors.append(
            Diagnostic(
                "AUD019",
                "reviewer_summary must contain the independent review conclusion",
                category="audit",
            )
        )
    text, prompt_errors = _decode_prompt(data)
    if text is not None:
        evidence, spans, _ = _parse_evidence(text)
        sections, _ = _section_slices(_normative_text(text, spans))
        expected_decision_ids = sorted(
            str(item.get("id"))
            for item in evidence
            if item.get("role") == "decision" and isinstance(item.get("id"), str)
        )
        declared_decision_ids = audit.get("decision_evidence_ids")
        if declared_decision_ids != expected_decision_ids:
            errors.append(
                Diagnostic(
                    "AUD020",
                    "audit decision_evidence_ids do not equal the prompt's exact decision evidence set",
                    category="audit",
                )
            )
        records = audit.get("review_records")
        record_ids: list[str] = []
        record_coverage: set[str] = set()
        records_valid = isinstance(records, list) and bool(records)
        if records_valid:
            for record in records:
                if not isinstance(record, dict) or set(record) != {
                    "id",
                    "status",
                    "summary",
                    "evidence_ids",
                }:
                    records_valid = False
                    break
                record_id = record.get("id")
                evidence_ids = record.get("evidence_ids")
                if (
                    not isinstance(record_id, str)
                    or not re.fullmatch(r"RR\d{3,}", record_id)
                    or record.get("status") not in {"no_issue", "resolved_issue"}
                    or not isinstance(record.get("summary"), str)
                    or len(record["summary"].strip()) < 20
                    or not isinstance(evidence_ids, list)
                    or not evidence_ids
                    or not all(value in expected_decision_ids for value in evidence_ids)
                ):
                    records_valid = False
                    break
                record_ids.append(record_id)
                record_coverage.update(evidence_ids)
        if not records_valid or len(record_ids) != len(set(record_ids)):
            errors.append(
                Diagnostic(
                    "AUD021",
                    "audit review_records must be unique RR### no_issue/resolved_issue records with substantive summaries and valid evidence_ids",
                    category="audit",
                )
            )
        elif record_coverage != set(expected_decision_ids):
            errors.append(
                Diagnostic(
                    "AUD022",
                    "audit review_records do not cover every decision evidence block",
                    category="audit",
                )
            )
        review = sections.get(REQUIRED_HEADINGS[13], "")
        author_match = re.search(r"(?m)^Author:\s*(\S.*?)\s*$", review)
        reviewer_match = re.search(r"(?m)^Independent reviewer:\s*(\S.*?)\s*$", review)
        result_match = re.search(r"(?mi)^Review result:\s*(pass)\b", review)
        removed_match = re.search(r"(?m)^Context removed:\s*(\d+)\b", review)
        evidence_match = re.search(
            r"(?m)^Decision evidence reviewed:\s*(E\d{3,}(?:,\s*E\d{3,})*)\s*$",
            review,
        )
        records_match = re.search(
            r"(?m)^Review records:\s*(RR\d{3,}(?:,\s*RR\d{3,})*)\s*$",
            review,
        )
        prompt_author = author_match.group(1).strip().rstrip("/") if author_match else None
        prompt_reviewer = reviewer_match.group(1).strip().rstrip("/") if reviewer_match else None
        audit_author = author.strip().rstrip("/") if isinstance(author, str) else None
        audit_reviewer = reviewer.strip().rstrip("/") if isinstance(reviewer, str) else None
        if prompt_author != audit_author or prompt_reviewer != audit_reviewer:
            errors.append(
                Diagnostic(
                    "AUD013",
                    "section-14 author/reviewer identities do not match the audit",
                    category="audit",
                )
            )
        if result_match is None:
            errors.append(
                Diagnostic("AUD014", "section-14 review result is not pass", category="audit")
            )
        if removed_match is None or not isinstance(removed, int) or int(removed_match.group(1)) != removed:
            errors.append(
                Diagnostic(
                    "AUD015",
                    "section-14 context-removed count does not match the audit",
                    category="audit",
                )
            )
        section_decision_ids = (
            sorted(value.strip() for value in evidence_match.group(1).split(","))
            if evidence_match
            else []
        )
        if section_decision_ids != expected_decision_ids:
            errors.append(
                Diagnostic(
                    "AUD023",
                    "section-14 decision-evidence inventory does not match the prompt",
                    category="audit",
                )
            )
        section_record_ids = (
            sorted(value.strip() for value in records_match.group(1).split(","))
            if records_match
            else []
        )
        if section_record_ids != sorted(record_ids):
            errors.append(
                Diagnostic(
                    "AUD024",
                    "section-14 review-record inventory does not match the audit",
                    category="audit",
                )
            )
        evidence_ids = {
            str(item.get("id"))
            for item in evidence
            if item.get("role") == "decision"
        }
        questions = sections.get(REQUIRED_HEADINGS[9], "")
        computed_orphans = sum(
            not re.search(rf"\b{re.escape(evidence_id)}\b", questions)
            for evidence_id in evidence_ids
        )
        hashes = collections.Counter(_sha256(item.get("_literal", b"")) for item in evidence)
        computed_duplicates = sum(count - 1 for count in hashes.values() if count > 1)
        if audit.get("orphan_evidence") != computed_orphans:
            errors.append(Diagnostic("AUD016", "audit orphan_evidence is not recomputed truth", category="audit"))
        if audit.get("duplicate_blocks") != computed_duplicates:
            errors.append(Diagnostic("AUD017", "audit duplicate_blocks is not recomputed truth", category="audit"))
    elif prompt_errors:
        errors.append(Diagnostic("AUD018", "audit cannot bind an undecodable prompt", category="audit"))
    return errors


def context_report(data: bytes) -> dict[str, Any]:
    text, diagnostics, evidence, sections = lint_prompt(data, strict=False)
    if text is None:
        return {"bytes": len(data), "diagnostics": [value.as_dict() for value in diagnostics]}
    section_sizes = {heading: len(body.encode("utf-8")) for heading, body in sections.items()}
    evidence_sizes = {
        str(item.get("id")): len(item.get("_literal", b"")) for item in evidence
    }
    duplicate_hashes: dict[str, list[str]] = collections.defaultdict(list)
    for item in evidence:
        duplicate_hashes[_sha256(item.get("_literal", b""))].append(str(item.get("id")))
    duplicates = [values for values in duplicate_hashes.values() if len(values) > 1]
    return {
        "bytes": len(data),
        "characters": len(text),
        "approximate_tokens": (len(text) + 3) // 4,
        "section_bytes": section_sizes,
        "evidence_bytes": evidence_sizes,
        "largest_evidence": sorted(evidence_sizes.items(), key=lambda pair: (-pair[1], pair[0]))[:10],
        "duplicate_evidence_groups": duplicates,
        "diagnostics": [value.as_dict() for value in diagnostics],
    }


RESPONSE_FENCE_RE = re.compile(
    r"(?ms)^(?P<fence>`{3,})(?P<language>[^\n`]*)\n"
    r"(?P<body>.*?)^(?P=fence)[ \t]*$"
)
RESPONSE_PLACEHOLDER_RE = re.compile(
    r"(?i)<(?:paste|insert|fill|replace|each\s+direct|path/hash|"
    r"[A-Za-z][A-Za-z0-9 _./-]{0,80})>"
    r"|\{\{[^{}\n]+\}\}|\b(?:TBD|TODO|FIXME)\b"
)
RESPONSE_TRUNCATED_HASH_RE = re.compile(r"(?i)\b[0-9a-f]{7,63}(?:…|\.\.\.)")
RESPONSE_CONDITIONAL_ACTION_RE = re.compile(
    r"(?m)^\s*(?:[-*]\s*)?(?:\|\s*)?AE(?:\s*\||\s|[|:])"
)
RESPONSE_PSEUDOCODE_RE = re.compile(
    r"(?i)\b(?:prior count\s*\+|record(?:ed)?(?:\s+\w+){0,3}\s+at implementation time|"
    r"message per commit contract|stage via explicit git operations per table)\b"
)
RESPONSE_SERIAL_REMAND_RE = re.compile(
    r"(?i)\b(?:need (?:more|additional) context|please provide|"
    r"ask Codex to (?:inspect|run|check)|need (?:the )?(?:repository|repo|file contents))\b"
    r"|\b(?:cannot adjudicate|ask Codex for another consultation|"
    r"request (?:a|another) (?:consultation|prompt|context package)|"
    r"no reliable adjudication is possible|prepare a fresh package)\b"
)
RESPONSE_DEFERRED_EXACTNESS_RE = re.compile(
    r"(?i)\b(?:defer\b[^\n.]{0,80}\b(?:exact values?|implementation step)|"
    r"(?:use|using|run|running) the usual battery|"
    r"(?:derive|deriving) (?:the )?remaining rows mechanically)\b"
)
SHELL_COMMENT_PLACEHOLDER_RE = re.compile(
    r"(?im)^\s*#.*\b(?:paste|verify each|one line per|same as|literal calls|"
    r"commands? from|selector below)\b"
)


def _fence_records(response: str) -> list[tuple[int, str, str, bool, int, int]]:
    """Return Markdown fence records, including an unterminated final fence."""

    records: list[tuple[int, str, str, bool, int, int]] = []
    opened: tuple[str, int, int, int] | None = None
    offset = 0
    for line_number, line in enumerate(response.splitlines(keepends=True), 1):
        raw = line.rstrip("\r\n")
        if opened is None:
            match = re.fullmatch(r"[ \t]{0,3}((?:`{3,}|~{3,}))([^`~]*)", raw)
            if match is not None:
                fence = match.group(1)
                opened = (fence, line_number, offset, offset + len(line))
        else:
            fence, start_line, start_offset, body_offset = opened
            marker = re.escape(fence[0]) + "{" + str(len(fence)) + ",}"
            if re.fullmatch(rf"[ \t]{{0,3}}{marker}[ \t]*", raw):
                language_line = response[start_offset:body_offset].rstrip("\r\n")
                language = language_line.lstrip(" \t`~").strip().lower()
                records.append(
                    (start_line, language, response[body_offset:offset], True, start_offset, offset + len(line))
                )
                opened = None
        offset += len(line)
    if opened is not None:
        fence, start_line, start_offset, body_offset = opened
        language_line = response[start_offset:body_offset].rstrip("\r\n")
        language = language_line.lstrip(" \t`~").strip().lower()
        records.append((start_line, language, response[body_offset:], False, start_offset, len(response)))
    return records


def _response_prose(response: str) -> str:
    chars = list(response)
    for _, _, _, _, start, end in _fence_records(response):
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _shell_blocks(response: str) -> list[tuple[int, str]]:
    return [
        (line + 1, body)
        for line, language, body, closed, _, _ in _fence_records(response)
        if closed and language in {"bash", "sh", "shell"}
    ]


def _cargo_test_positionals(command: str) -> list[str]:
    try:
        tokens = shlex.split(command, comments=True, posix=True)
    except ValueError:
        return []
    cargo_index = next(
        (index for index, value in enumerate(tokens) if Path(value).name == "cargo"),
        None,
    )
    if cargo_index is None or cargo_index + 1 >= len(tokens) or tokens[cargo_index + 1] != "test":
        return []
    option_values = {
        "--bench",
        "--bin",
        "--color",
        "--config",
        "--exclude",
        "--example",
        "--features",
        "--jobs",
        "--manifest-path",
        "--message-format",
        "--package",
        "--profile",
        "--target",
        "--target-dir",
        "--test",
        "-F",
        "-Z",
        "-j",
        "-p",
    }
    positionals: list[str] = []
    index = cargo_index + 2
    shell_boundaries = {"|", "||", "&&", ";", "&"}
    while index < len(tokens) and tokens[index] != "--":
        value = tokens[index]
        if value in shell_boundaries or value.startswith((">", "1>", "2>")):
            break
        if value.startswith("-"):
            option = value.partition("=")[0]
            if option in option_values and "=" not in value and index + 1 < len(tokens):
                index += 2
                continue
            index += 1
            continue
        positionals.append(value)
        index += 1
    return positionals


def validate_response(prompt_data: bytes, response_data: bytes) -> list[Diagnostic]:
    """Conservatively lint a user-returned Fable answer before implementation.

    This is a mechanical rejection layer, not a semantic correctness proof.
    Codex must still validate every returned repository claim under AGENTS.md.
    """

    prompt_text, prompt_errors, _, sections = lint_prompt(prompt_data, strict=True)
    if prompt_errors or prompt_text is None:
        return [
            Diagnostic(
                "RSP001",
                "cannot validate a response against an invalid consultation prompt",
                category="response",
            )
        ]
    try:
        response = response_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [
            Diagnostic(
                "RSP002",
                f"response is not valid UTF-8 at byte offset {exc.start}",
                category="response",
            )
        ]

    errors: list[Diagnostic] = []
    if "\x00" in response:
        errors.append(Diagnostic("RSP003", "response contains NUL bytes", category="response"))
    errors.extend(_security_diagnostics(response))

    fence_records = _fence_records(response)
    response_prose = _response_prose(response)
    for line, language, body, closed, _, _ in fence_records:
        if not closed:
            errors.append(
                Diagnostic("RSP017", "response contains an unterminated Markdown code fence", line, "response")
            )
        shell_like = bool(
            re.search(
                r"(?m)^\s*(?:\$\s+|set\s+-|git\s+|cargo\s+|python\d*\s+|"
                r"[A-Z_][A-Z0-9_]*=|if\s+|for\s+|while\s+|make\s+|sha256sum\s+|cd\s+|export\s+)",
                body,
            )
        )
        if language in {"console", "zsh", "ksh", "fish", "shell-session"} or (not language and shell_like):
            errors.append(
                Diagnostic(
                    "RSP018",
                    "shell-like code must use a closed bash/sh/shell fence for syntax validation",
                    line,
                    "response",
                )
            )

    questions = _defined_ids(sections.get(REQUIRED_HEADINGS[9], ""), "Q")
    for question in questions:
        question_present = bool(re.search(rf"\b{re.escape(question)}\b", response_prose))
        dispositions = list(
            re.finditer(
                rf"(?mi)^\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?{re.escape(question)}\s*"
                rf"(?::|—|-)\s*(?P<decision>ACCEPT|AMEND|REJECT|STOP)\b(?P<after>[^\n]*)$",
                response_prose,
            )
        )
        if not question_present:
            errors.append(
                Diagnostic(
                    "RSP004",
                    f"response does not answer required question {question}",
                    category="response",
                )
            )
            continue
        if not dispositions:
            errors.append(
                Diagnostic(
                    "RSP014",
                    f"response lacks an exact '{question}: ACCEPT|AMEND|REJECT|STOP' disposition",
                    category="response",
                )
            )
            continue
        if len(dispositions) > 1:
            errors.append(
                Diagnostic(
                    "RSP020",
                    f"response contains multiple dispositions for {question}",
                    _line_number(response_prose, dispositions[1].start()),
                    "response",
                )
            )
            continue
        disposition = dispositions[0]
        following = response_prose[disposition.end() :]
        next_question = re.search(r"(?mi)^\s*(?:#{1,6}\s*)?(?:[-*]\s*)?Q\d{3,}\s*(?::|—|-)", following)
        rationale = disposition.group("after") + following[: next_question.start() if next_question else None]
        substance = re.sub(r"[^A-Za-z0-9]+", "", rationale)
        if len(substance) < 12:
            errors.append(
                Diagnostic(
                    "RSP016",
                    f"response disposition for {question} lacks a substantive decision or rationale",
                    _line_number(response_prose, disposition.start()),
                    "response",
                )
            )

    for match in RESPONSE_SERIAL_REMAND_RE.finditer(response_prose):
        errors.append(
            Diagnostic(
                "RSP015",
                "response attempts a prohibited serial remand for context or repository access",
                _line_number(response, match.start()),
                "response",
            )
        )

    for match in RESPONSE_DEFERRED_EXACTNESS_RE.finditer(response_prose):
        errors.append(
            Diagnostic(
                "RSP019",
                "response defers exact contract content instead of adjudicating it",
                _line_number(response_prose, match.start()),
                "response",
            )
        )

    placeholder_scan = response.replace("<br>", "[html-break]")
    placeholder_scan = re.sub(
        r"(?<=:)<[A-Za-z][A-Za-z0-9_-]*>",
        "[dynamic-operand]",
        placeholder_scan,
    )
    for code, message, pattern, scan_text in (
        ("RSP005", "response contains an operative placeholder", RESPONSE_PLACEHOLDER_RE, placeholder_scan),
        ("RSP006", "response contains a truncated hexadecimal digest", RESPONSE_TRUNCATED_HASH_RE, response),
        ("RSP007", "response contains a conditional AE path action", RESPONSE_CONDITIONAL_ACTION_RE, response),
        ("RSP008", "response defers an exact result to pseudocode or implementation time", RESPONSE_PSEUDOCODE_RE, response),
    ):
        for match in pattern.finditer(scan_text):
            errors.append(
                Diagnostic(code, message, _line_number(response, match.start()), "response")
            )

    for block_line, block in _shell_blocks(response):
        completed = subprocess.run(
            ["bash", "-n"],
            input=block.encode("utf-8"),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
            summary = detail[-1] if detail else "bash -n rejected the block"
            errors.append(
                Diagnostic(
                    "RSP009",
                    f"shell block is not syntactically valid: {summary}",
                    block_line,
                    "response",
                )
            )
        placeholder = RESPONSE_PLACEHOLDER_RE.search(block)
        if placeholder is not None:
            errors.append(
                Diagnostic(
                    "RSP010",
                    "shell block contains an unexpanded operative placeholder",
                    block_line + block[: placeholder.start()].count("\n"),
                    "response",
                )
            )
        comment_placeholder = SHELL_COMMENT_PLACEHOLDER_RE.search(block)
        if comment_placeholder is not None:
            errors.append(
                Diagnostic(
                    "RSP011",
                    "shell comment substitutes prose for a required command",
                    block_line + block[: comment_placeholder.start()].count("\n"),
                    "response",
                )
            )

        logical = re.sub(r"\\\n[ \t]*", " ", block)
        for offset, line in enumerate(logical.splitlines()):
            positionals = _cargo_test_positionals(line)
            if len(positionals) > 1:
                errors.append(
                    Diagnostic(
                        "RSP012",
                        "cargo test accepts at most one positional TESTNAME before '--': "
                        + ", ".join(positionals),
                        block_line + offset,
                        "response",
                    )
                )

        definitions = set(
            re.findall(
                r"(?m)^\s*(?:(?:export|readonly|local|declare)(?:\s+-[A-Za-z]+)?\s+)?"
                r"([A-Za-z_][A-Za-z0-9_]*)=",
                block,
            )
        ) | set(re.findall(r"(?m)^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b", block))
        definitions.update(
            re.findall(r"(?m)^\s*read(?:\s+-[A-Za-z]+)*\s+([A-Za-z_][A-Za-z0-9_]*)\b", block)
        )
        for read_line in re.findall(r"(?m)^\s*read\s+([^\n]+)$", block):
            definitions.update(
                value for value in shlex.split(read_line) if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
            )
        definitions.update(
            re.findall(r"(?m)^\s*printf\s+-v\s+([A-Za-z_][A-Za-z0-9_]*)\b", block)
        )
        uses = list(
            re.finditer(
                r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?P<modifier>[^}]*)\}|"
                r"(?P<plain>[A-Za-z_][A-Za-z0-9_]*))",
                block,
            )
        )
        safe_environment = {
            "BASH_SOURCE", "EUID", "HOME", "IFS", "LINENO", "LOGNAME", "OLDPWD",
            "OPTARG", "OPTIND", "PATH", "PIPESTATUS", "PPID", "PWD", "RANDOM",
            "REPLY", "SECONDS", "SHELL", "TMPDIR", "UID", "USER",
        }
        defaulted = {
            match.group("braced")
            for match in uses
            if match.group("braced") and match.group("modifier").startswith((":-", ":=", ":?", ":+"))
        }
        flattened_uses = {match.group("braced") or match.group("plain") for match in uses}
        self_references = set(
            re.findall(
                r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)=\$(?:\{\1\}|\1)\s*$",
                block,
            )
        )
        for variable in sorted(flattened_uses - definitions - safe_environment - defaulted):
            errors.append(
                Diagnostic(
                    "RSP013",
                    f"shell block uses undefined variable ${variable}",
                    block_line,
                    "response",
                )
            )
        for variable in sorted(self_references):
            errors.append(
                Diagnostic(
                    "RSP013",
                    f"shell block self-references ${variable} before a value is established",
                    block_line,
                    "response",
                )
            )

    unique: dict[tuple[str, int, str], Diagnostic] = {}
    for error in errors:
        unique[(error.code, error.line, error.message)] = error
    return sorted(unique.values(), key=lambda value: (value.line, value.code, value.message))


def _exit_for(diagnostics: Sequence[Diagnostic]) -> int:
    categories = {value.category for value in diagnostics}
    if "security" in categories:
        return EXIT_SECURITY
    if "audit" in categories:
        return EXIT_AUDIT
    if "repository" in categories:
        return EXIT_REPOSITORY
    return EXIT_CONTRACT


def _print_diagnostics(diagnostics: Sequence[Diagnostic], *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps([value.as_dict() for value in diagnostics], indent=2, sort_keys=True))
        return
    for value in diagnostics:
        print(str(value), file=sys.stderr)


def _last_assistant_text(transcript_path: Path) -> str:
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        if (
            item.get("type") == "response_item"
            and isinstance(payload, dict)
            and payload.get("type") == "message"
            and payload.get("role") == "assistant"
            and payload.get("phase") == "final_answer"
        ):
            content = payload.get("content", [])
            return "".join(
                value.get("text", "")
                for value in content
                if isinstance(value, dict) and value.get("type") == "output_text"
            )
    return ""


def _block_stop(reason: str) -> None:
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": reason,
                "systemMessage": "Fable-5 handoff blocked by the repository standalone-context guard.",
            },
            sort_keys=True,
        )
    )


def run_stop_hook(repo_root: Path) -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        _block_stop(
            "The repository Fable-5 guard received malformed native-hook input and failed closed."
        )
        return 0
    cwd = Path(str(payload.get("cwd", ""))).resolve()
    configured_root = repo_root.resolve()

    def git_path(path: Path, argument: str) -> Path | None:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", argument],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode:
            return None
        value = Path(completed.stdout.strip())
        if not value.is_absolute():
            value = path / value
        return value.resolve()

    configured_common = git_path(configured_root, "--git-common-dir")
    active_common = git_path(cwd, "--git-common-dir")
    active_root = git_path(cwd, "--show-toplevel")
    if configured_common is None:
        if cwd != configured_root and configured_root not in cwd.parents:
            return 0
        root = configured_root
    else:
        if active_common is None or active_root is None or active_common != configured_common:
            return 0
        root = active_root
    answer = payload.get("last_assistant_message")
    if not isinstance(answer, str) or not answer.strip():
        transcript = payload.get("transcript_path")
        answer = _last_assistant_text(Path(transcript)) if isinstance(transcript, str) else ""
        if not answer:
            _block_stop(
                "The outbound guard could not inspect the final assistant message; retry after preserving a readable transcript."
            )
            return 0
    fable_expression = r"Fable(?:[-\s\u2010-\u2015]?5)"
    visible_answer = re.sub(r"\]\([^)]+\)", "]", answer)
    fable = re.search(rf"(?i)\b{fable_expression}\b", visible_answer)
    intent = re.search(
        rf"(?is)(?:ready[- ]to[- ]paste consultation|\b(?:send|copy|hand|forward|share|submit|give|paste|upload)\s+(?:this|it|the following|the prompt|the file|this prompt|this file)\b.{{0,100}}{fable_expression}|\b(?:ask|consult)\s+{fable_expression}\s+(?:this|with this|about this)\b|wait(?:ing)? for {fable_expression}(?:'s)? ruling|here is (?:the )?consultation prompt)",
        visible_answer,
    )
    explicit_handoff = re.search(
        rf"(?is)(?:\b(?:this|the following|the attached)\s+(?:prompt|consultation|file|package)\s+(?:is\s+)?(?:for|to)\s+{fable_expression}\b|\b{fable_expression}\s+(?:prompt|consultation|handoff|package)\s+(?:is\s+ready|is\s+at|is\s+in|follows|:))",
        visible_answer,
    )
    file_line_matches = re.findall(
        rf"(?m)^{PROMPT_FILE_LABEL}: ([^`\n]+)$", answer
    )
    sha_line_matches = re.findall(
        rf"(?m)^{PROMPT_SHA_LABEL}: ([0-9a-f]{{64}})$", answer
    )
    triggered = (
        SESSION_DECLARATION.splitlines()[0] in answer
        or BEGIN_MARKER in answer
        or END_MARKER in answer
        or bool(file_line_matches or sha_line_matches)
        or bool(
            re.search(
                rf"(?m)^(?:{PROMPT_FILE_LABEL}|{PROMPT_SHA_LABEL}):",
                answer,
            )
        )
        or bool(fable and intent)
        or bool(explicit_handoff)
    )
    if not triggered:
        return 0
    if len(file_line_matches) != 1 or len(sha_line_matches) != 1:
        _block_stop(
            "A Fable-5 handoff must identify exactly one emit-validated .txt file with FABLE5_PROMPT_FILE and FABLE5_PROMPT_SHA256 lines."
        )
        return 0
    output_path = Path(file_line_matches[0])
    if not output_path.is_absolute() or output_path.suffix.lower() != ".txt":
        _block_stop("FABLE5_PROMPT_FILE must be one absolute plain-text .txt path.")
        return 0
    try:
        prompt_bytes = output_path.resolve(strict=True).read_bytes()
    except OSError:
        _block_stop("The declared Fable-5 prompt file is missing or unreadable.")
        return 0
    if _sha256(prompt_bytes) != sha_line_matches[0]:
        _block_stop("FABLE5_PROMPT_SHA256 does not match the declared prompt file bytes.")
        return 0
    if SESSION_DECLARATION.splitlines()[0] in answer or BEGIN_MARKER in answer or END_MARKER in answer:
        _block_stop(
            "The Fable-5 prompt must be delivered only as the declared .txt file, not duplicated inline."
        )
        return 0
    remainder = re.sub(
        rf"(?m)^{PROMPT_FILE_LABEL}:.*$|^{PROMPT_SHA_LABEL}:.*$", "", answer
    )
    link_pattern = re.compile(r"\[[^\]\n]*\]\((?P<target><?[^)]+>?)\)")
    for link_match in list(link_pattern.finditer(remainder)):
        target = link_match.group("target").strip().strip("<>")
        if target.startswith(("file://", "vscode://")):
            target = target.partition("://")[2]
        if target.startswith(("/", "~/", "./", "../")):
            try:
                if Path(target).expanduser().resolve(strict=True) != output_path.resolve(strict=True):
                    _block_stop("A Fable-5 handoff may link only to the one declared prompt file.")
                    return 0
            except OSError:
                _block_stop("A Fable-5 handoff contains a missing additional local artifact.")
                return 0
            remainder = remainder.replace(link_match.group(0), "")
    if remainder.strip() != HANDOFF_SENTENCE:
        _block_stop(
            "A Fable-5 file handoff may contain only the fixed manual-delivery sentence, the two receipt labels, and an optional link to that same file."
        )
        return 0
    _, diagnostics, _, _ = lint_prompt(prompt_bytes, strict=True)
    if diagnostics:
        summary = "; ".join(f"{value.code}@{value.line}" for value in diagnostics[:8])
        _block_stop(f"Fable-5 prompt file failed validation: {summary}. Run the required check/emit workflow.")
        return 0
    receipt_valid, receipt_reason = verify_emit_receipt(
        repo_root, prompt_bytes, output_path
    )
    if not receipt_valid:
        _block_stop(
            f"Fable-5 prompt file lacks a current successful check/emit receipt: {receipt_reason}."
        )
    return 0


def install_stop_hook(repo_root: Path, hooks_file: Path) -> bool:
    try:
        root = repo_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"repository root does not exist: {repo_root}") from exc
    probe = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode or Path(probe.stdout.strip()).resolve() != root:
        raise ValueError("--repo-root must be the exact Git worktree root")
    python = root / ".venv/bin/python"
    script = root / "tools/local_labels/fable5_consultation_protocol.py"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"required interpreter is missing or non-executable: {python}")
    if not script.is_file():
        raise ValueError(f"protocol script is missing: {script}")
    hooks_file = hooks_file.expanduser()
    if hooks_file.exists():
        try:
            document = json.loads(hooks_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot parse {hooks_file}: {exc}") from exc
    else:
        document = {}
    if not isinstance(document, dict):
        raise ValueError("Codex hooks file root must be an object")
    hooks = document.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("Codex hooks field must be an object")
    stop = hooks.setdefault("Stop", [])
    if not isinstance(stop, list):
        raise ValueError("Codex Stop hooks field must be a list")
    command = " ".join(
        shlex.quote(str(value))
        for value in (python, script, "stop-hook", "--repo-root", root)
    )
    replacement = {
        "type": "command",
        "command": command,
        "timeout": 30,
        "statusMessage": "GPU simulator Fable-5 outbound guard",
    }
    for group in stop:
        if not isinstance(group, dict):
            continue
        group_hooks = group.get("hooks", [])
        for index, hook in enumerate(group_hooks):
            if not isinstance(hook, dict):
                continue
            if (
                hook.get("command") == command
                or hook.get("statusMessage") == replacement["statusMessage"]
            ):
                if hook == replacement:
                    return False
                group_hooks[index] = replacement
                break
        else:
            continue
        break
    else:
        stop.append({"hooks": [replacement]})
    hooks_file.parent.mkdir(parents=True, exist_ok=True)
    mode = hooks_file.stat().st_mode if hooks_file.exists() else None
    rendered = json.dumps(document, indent=2, sort_keys=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=hooks_file.parent, delete=False) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    if mode is not None:
        os.chmod(temporary, mode)
    os.replace(temporary, hooks_file)
    return True


def _load_and_lint(
    path: str, *, strict: bool
) -> tuple[bytes, str | None, list[Diagnostic], list[dict[str, Any]], dict[str, str]]:
    data = _read_bytes(path)
    text, diagnostics, evidence, sections = lint_prompt(data, strict=strict)
    return data, text, diagnostics, evidence, sections


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lint_parser = subparsers.add_parser("lint", help="validate prompt structure")
    lint_parser.add_argument("prompt")
    lint_parser.add_argument("--strict", action="store_true")
    lint_parser.add_argument("--json", action="store_true")

    report_parser = subparsers.add_parser("context-report", help="report context size and duplication")
    report_parser.add_argument("prompt")
    report_parser.add_argument("--json", action="store_true")

    response_parser = subparsers.add_parser(
        "check-response",
        help="mechanically reject an incomplete or pseudocode Fable-5 answer",
    )
    response_parser.add_argument("prompt")
    response_parser.add_argument("response")
    response_parser.add_argument("--json", action="store_true")

    for name in ("check", "emit"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("prompt")
        command_parser.add_argument("--audit", required=True)
        command_parser.add_argument("--repo-root", required=True)
        command_parser.add_argument(
            "--verify-repo",
            action="store_true",
            help="compatibility flag; repository verification is always mandatory",
        )
        command_parser.add_argument(
            "--allow-repo-local-inputs",
            action="store_true",
            help="use only when the user explicitly authorized tracking/local draft inputs",
        )
        command_parser.add_argument("--json", action="store_true")
        if name == "emit":
            command_parser.add_argument(
                "--output",
                required=True,
                help="absolute outside-repository .txt path for the one user-delivered prompt",
            )

    hook_parser = subparsers.add_parser("hook", help="validate one staged prompt from stdin")
    hook_parser.add_argument("--label", default="<staged prompt>")

    audit_parser = subparsers.add_parser("audit-template", help="write a review sidecar template")
    audit_parser.add_argument("prompt")
    audit_parser.add_argument("--author-id", required=True)
    audit_parser.add_argument("--reviewer-id", required=True)
    audit_parser.add_argument("--context-removed-chars", type=int, default=0)

    stop_parser = subparsers.add_parser("stop-hook", help="native Codex Stop-hook entry point")
    stop_parser.add_argument("--repo-root", required=True)

    install_parser = subparsers.add_parser("install-stop-hook", help="idempotently install native Stop guard")
    install_parser.add_argument("--repo-root", required=True)
    install_parser.add_argument("--hooks-file", default=str(Path.home() / ".codex/hooks.json"))

    args = parser.parse_args(argv)
    try:
        if args.command == "stop-hook":
            return run_stop_hook(Path(args.repo_root))
        if args.command == "install-stop-hook":
            changed = install_stop_hook(Path(args.repo_root).resolve(), Path(args.hooks_file))
            state = "installed" if changed else "already installed"
            print(
                f"{state}; activation requires Codex restart and explicit trust for the new Stop hook"
            )
            return 0
        if args.command == "audit-template":
            if args.author_id.strip().rstrip("/") == args.reviewer_id.strip().rstrip("/"):
                raise ValueError("audit author and reviewer must be different agents")
            data = _read_bytes(args.prompt)
            _, _, prompt_evidence, _ = lint_prompt(data, strict=True)
            decision_evidence_ids = sorted(
                str(item.get("id"))
                for item in prompt_evidence
                if item.get("role") == "decision" and isinstance(item.get("id"), str)
            )
            document = {
                "schema_version": 3,
                "prompt_sha256": _sha256(data),
                "author_id": args.author_id,
                "reviewer_id": args.reviewer_id,
                "result": "pending",
                "reviewed": [],
                "decision_evidence_ids": decision_evidence_ids,
                "review_records": [
                    {
                        "id": "RR001",
                        "status": "pending",
                        "summary": "Independent reviewer must replace this template review record.",
                        "evidence_ids": decision_evidence_ids,
                    }
                ],
                "unresolved_findings": [
                    "Independent reviewer must complete every review dimension and replace this pending record."
                ],
                "orphan_evidence": 0,
                "duplicate_blocks": 0,
                "context_removed_chars": args.context_removed_chars,
                "reviewer_summary": "PENDING",
            }
            print(json.dumps(document, indent=2, sort_keys=True))
            return 0
        if args.command == "context-report":
            report = context_report(_read_bytes(args.prompt))
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(f"bytes: {report.get('bytes', 0)}")
                print(f"characters: {report.get('characters', 0)}")
                print(f"approximate tokens: {report.get('approximate_tokens', 0)}")
                print("section bytes:")
                for heading, size in report.get("section_bytes", {}).items():
                    print(f"  {size:8d}  {heading}")
                print("largest evidence blocks:")
                for evidence_id, size in report.get("largest_evidence", []):
                    print(f"  {size:8d}  {evidence_id}")
                print(f"duplicate evidence groups: {len(report.get('duplicate_evidence_groups', []))}")
                print(f"diagnostics: {len(report.get('diagnostics', []))}")
            return 0
        if args.command == "check-response":
            prompt_data = _read_bytes(args.prompt)
            response_data = _read_bytes(args.response)
            diagnostics = validate_response(
                prompt_data,
                response_data,
            )
            if diagnostics:
                _print_diagnostics(diagnostics, json_output=args.json)
                return _exit_for(diagnostics)
            if args.json:
                print(
                    json.dumps(
                        {
                            "diagnostics": [],
                            "prompt_sha256": _sha256(prompt_data),
                            "response_sha256": _sha256(response_data),
                            "valid": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print("valid")
                print(f"FABLE5_RESPONSE_PROMPT_SHA256: {_sha256(prompt_data)}")
                print(f"FABLE5_RESPONSE_SHA256: {_sha256(response_data)}")
            return 0
        if args.command == "hook":
            data = sys.stdin.buffer.read()
            _, diagnostics, _, _ = lint_prompt(data, strict=True)
            if diagnostics:
                print(f"Fable-5 staged prompt rejected: {args.label}", file=sys.stderr)
                _print_diagnostics(diagnostics)
                return _exit_for(diagnostics)
            return 0
        if args.command == "lint":
            data, text, diagnostics, _, _ = _load_and_lint(args.prompt, strict=args.strict)
            del data, text
            if not args.strict:
                diagnostics = [value for value in diagnostics if value.category != "warning"]
            if diagnostics:
                _print_diagnostics(diagnostics, json_output=args.json)
                return _exit_for(diagnostics)
            if args.json:
                print("[]")
            else:
                print("valid")
            return 0
        if args.command in {"check", "emit"}:
            if not args.allow_repo_local_inputs:
                root = Path(args.repo_root).resolve(strict=True)
                for label, value in (("prompt", args.prompt), ("audit", args.audit)):
                    candidate = Path(value).resolve(strict=True)
                    if candidate == root or root in candidate.parents:
                        raise ValueError(
                            f"{label} input must stay outside the repository unless --allow-repo-local-inputs was explicitly authorized"
                        )
            data, text, diagnostics, evidence, sections = _load_and_lint(args.prompt, strict=True)
            audit_data = _read_bytes(args.audit)
            diagnostics.extend(
                validate_audit(data, args.audit, audit_data=audit_data)
            )
            observation_before: tuple[Path, str, str, bytes, str] | None = None
            if args.command == "emit":
                try:
                    observation_before = _git_observation(Path(args.repo_root))
                except (OSError, ValueError):
                    diagnostics.append(
                        Diagnostic(
                            "REP014",
                            "cannot capture pre-emission repository state",
                            category="repository",
                        )
                    )
            diagnostics.extend(
                verify_repository_context(
                    Path(args.repo_root), text or "", sections, evidence
                )
            )
            diagnostics.sort(key=lambda value: (value.line, value.code, value.message))
            if diagnostics:
                _print_diagnostics(diagnostics, json_output=args.json)
                return _exit_for(diagnostics)
            if args.command == "emit":
                output_preexisted = Path(args.output).exists()
                output = _write_single_prompt_file(
                    Path(args.output), Path(args.repo_root), data
                )
                observation_after = _git_observation(Path(args.repo_root))
                if observation_before != observation_after:
                    if not output_preexisted:
                        output.unlink(missing_ok=True)
                    drift = Diagnostic(
                        "REP015",
                        "repository state changed between verification and prompt emission",
                        category="repository",
                    )
                    _print_diagnostics([drift], json_output=args.json)
                    return EXIT_REPOSITORY
                _write_emit_receipt(
                    Path(args.repo_root),
                    data,
                    audit_data,
                    output,
                    observation_after,
                )
                print(f"{PROMPT_FILE_LABEL}: {output}")
                print(f"{PROMPT_SHA_LABEL}: {_sha256(data)}")
            elif args.json:
                print("[]")
            else:
                print("valid")
            return 0
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    return EXIT_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
