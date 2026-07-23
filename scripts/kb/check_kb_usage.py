#!/usr/bin/env python3
"""
check_kb_usage.py — Stop hook: did the agent consult the KB?

Reads a Claude Code Stop event JSON from stdin. The event contains
`transcript_path` pointing at a JSONL transcript of the session. We:

  1. Parse the most-recent NON-META user prompt from the transcript.
     (Meta entries with `isMeta: true` are system reminders / hook
     injections, NOT the human's prompt — phase 2-rev critical fix.)
  2. Decide whether the prompt was KB-relevant via the shared classifier
     in `scripts/kb/_relevance.py` (single source of truth for the
     Group A / B / C gate; the bash hook regenerates from the same).
  3. Walk the transcript's tool_use blocks looking for evidence of KB
     consultation:
       - Read tool calls against `.claude/knowledge/**` or `references/**`
       - Bash tool calls that grep/cat/head those paths
       - Grep tool calls with `path` containing knowledge/references
       - Agent's own reply containing a "## Knowledge Consulted" section
         (case-insensitive; accepts heading + bold-only + colon-suffix
          variants and the "Sources Consulted" / "KB Consulted" synonyms)
  4. Append a metric record to `.claude/metrics/kb_usage.jsonl` and rotate
     to .jsonl.1 when the file exceeds 10 MB.
  5. In WARNING mode (default): if the prompt was KB-relevant and no
     consultation evidence was found, print a polite reminder to stderr
     but always exit 0 (do not block).
     In BLOCK mode (env CLAUDE_KB_USAGE_MODE=block): emit a Stop-blocking
     JSON to stdout when the check fails.

Stop-hook contract: emitting `{"decision":"block","reason":"..."}` to
stdout asks Claude Code to keep working. Default behavior is no output.

Usage:
  echo '{"transcript_path":"...","stop_hook_active":false}' | \\
    python3 scripts/kb/check_kb_usage.py [--mode warning|block]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add scripts/kb/ to import path so we can import _relevance.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _relevance import is_kb_relevant  # noqa: E402


# Match `.claude/knowledge/...md`, `references/...`, or the SOL source
# trees cited by the source cards. All are repository-owned KB evidence.
_KB_PATH_RE = re.compile(
    r"(?:(?:^|[^\w./])\.claude/knowledge/[\w./_-]+\.md"
    r"|(?:^|[^\w./])references/[\w./_-]+\.(?:md|pdf|cu|cuh|h|cpp|hpp|c|py|"
    r"sh|toml|json|ptx|sass)"
    r"|(?:^|[^\w./])(?:baselines|sol-execbench|cutlass|SOLAR)/"
    r"[\w./_-]+\.(?:md|cu|cuh|h|cpp|hpp|c|py|sh|toml|json|yaml|yml|cmake|"
    r"jsonl|txt))"
)

# Loosened: accept "## Knowledge Consulted" heading, bold-only
# "**Knowledge Consulted**", colon-suffix "Knowledge Consulted:", and the
# common synonyms "Sources Consulted" / "KB Consulted" / "Citations".
_KNOWLEDGE_CONSULTED_RE = re.compile(
    r"(?:^|\n)\s*(?:#+\s*|\*\*\s*|__\s*)?"
    r"(?:Knowledge\s+Consulted|Sources\s+Consulted|KB\s+Consulted|"
    r"Citations)\b",
    re.IGNORECASE,
)

# Path predicate for Read/Grep tools: accept absolute and relative variants
# of the KB cards, references, and source trees covered above.
_KB_PATH_PREDICATE = re.compile(
    r"(?:^|/)\.claude/knowledge/|(?:^|/)references/"
    r"|(?:^|/)baselines/|(?:^|/)sol-execbench/|(?:^|/)cutlass/|(?:^|/)SOLAR/"
)

_METRIC_ROTATION_BYTES = 10 * 1024 * 1024  # 10 MB


def _read_transcript(transcript_path: Path) -> list[dict]:
    if not transcript_path.is_file():
        return []
    out = []
    try:
        with transcript_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def _last_user_prompt(events: list[dict]) -> tuple[str, int]:
    """Return (text, event_index) of the most recent NON-META user prompt.

    phase 2-rev critical fix: real Claude Code transcripts mark
    system-reminder injections as `type:"user"` with `isMeta: true`. The
    most recent such event is typically the LAST event of the session, so
    a naive reverse-walk returns the reminder text instead of the human's
    prompt. We filter `isMeta: true` and skip user events whose `content`
    contains only tool_result blocks.

    Codex-review fix: also return the event index so the KB-read scan
    can window to events AFTER this prompt — earlier turns' reads must
    not count as evidence for the current answer.
    """
    for i in range(len(events) - 1, -1, -1):
        ev = events[i]
        if ev.get("type") != "user":
            continue
        if ev.get("isMeta") is True:
            continue
        msg = ev.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content, i
        if isinstance(content, list):
            text_parts: list[str] = []
            non_tool_result = False
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    text_parts.append(c.get("text", ""))
                    non_tool_result = True
                elif c.get("type") == "tool_result":
                    continue
                else:
                    non_tool_result = True
            if non_tool_result and text_parts:
                return "\n".join(text_parts), i
    return "", -1


def _kb_reads_in_transcript(
    events: list[dict], start_index: int = 0
) -> tuple[set[str], bool, bool]:
    """Return (set of KB paths read after start_index,
    consulted_section_has_evidence, had_consulted_heading).

    `had_consulted_heading` is True whenever a "## Knowledge Consulted"
    (or recognized synonym) heading appears, regardless of whether any
    repo-relative path follows it. Surfaced separately so callers can
    distinguish the two failure modes — block missing entirely vs
    block present but pathless — and produce a repair message that
    points at the actual problem.
    """
    paths: set[str] = set()
    consulted_section_has_evidence = False
    had_consulted_heading = False
    for i, ev in enumerate(events):
        if i < start_index:
            continue
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message") or {}
        content = msg.get("content") or []
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "tool_use":
                name = c.get("name", "")
                inp = c.get("input") or {}
                if name == "Read":
                    fp = inp.get("file_path", "")
                    if isinstance(fp, str) and _KB_PATH_PREDICATE.search(fp):
                        paths.add(fp)
                elif name == "Bash":
                    cmd = inp.get("command", "")
                    if isinstance(cmd, str):
                        for m in _KB_PATH_RE.finditer(cmd):
                            hit = m.group(0).lstrip()
                            hit = re.sub(r"^[^\w./]+", "", hit)
                            paths.add(hit)
                elif name == "Grep":
                    path = inp.get("path", "")
                    if isinstance(path, str) and _KB_PATH_PREDICATE.search(path):
                        paths.add(path)
            elif c.get("type") == "text":
                text = c.get("text", "")
                if not isinstance(text, str):
                    continue
                m = _KNOWLEDGE_CONSULTED_RE.search(text)
                if m:
                    had_consulted_heading = True
                    if _KB_PATH_RE.search(text[m.end():]):
                        consulted_section_has_evidence = True
    return paths, consulted_section_has_evidence, had_consulted_heading


def _append_metric(repo_root: Path, record: dict) -> None:
    metrics_dir = repo_root / ".claude" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "kb_usage.jsonl"
    record.setdefault(
        "ts", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    try:
        # Rotate if the file exceeds the size cap.
        if out.is_file() and out.stat().st_size > _METRIC_ROTATION_BYTES:
            rotated = metrics_dir / "kb_usage.jsonl.1"
            try:
                if rotated.exists():
                    rotated.unlink()
                out.rename(rotated)
            except OSError:
                pass
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _stop_hook_already_active(event: dict) -> bool:
    return bool(event.get("stop_hook_active"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("warning", "block"),
        default=os.environ.get("CLAUDE_KB_USAGE_MODE", "warning"),
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.root).resolve()

    raw = sys.stdin.read()
    if not raw.strip():
        return 0
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    if _stop_hook_already_active(event):
        return 0

    transcript_path = event.get("transcript_path")
    if not transcript_path:
        return 0

    transcript = Path(transcript_path)
    events = _read_transcript(transcript)
    if not events:
        return 0

    prompt, prompt_index = _last_user_prompt(events)
    kb_relevant = is_kb_relevant(prompt)

    # Window the KB-read scan to events AFTER the last non-meta user
    # prompt — earlier turns' reads must not count as evidence for the
    # current answer (Codex-review fix).
    paths_read, consulted, had_heading = _kb_reads_in_transcript(
        events, start_index=max(prompt_index + 1, 0)
    )
    consulted_evidence = bool(paths_read) or consulted
    skipped = kb_relevant and not consulted_evidence

    record = {
        "kb_relevant": kb_relevant,
        "consulted": consulted_evidence,
        "n_kb_paths_read": len(paths_read),
        "had_knowledge_consulted_section_with_paths": consulted,
        "had_bare_consulted_heading": had_heading and not consulted,
        "prompt_event_index": prompt_index,
        "skipped": skipped,
        "transcript": str(transcript),
        "prompt_preview": prompt[:200] if prompt else "",
        "mode": args.mode,
    }
    _append_metric(repo_root, record)

    if not skipped:
        return 0

    # Case-aware message: distinguish "block missing entirely" from
    # "block present but pathless" so the agent's repair loop targets
    # the actual gap. The N/A literal is intentionally NOT offered as
    # an escape hatch — the prompt was classified KB-relevant by the
    # gate, and a self-claim of N/A does not override the classifier.
    if had_heading and not consulted:
        diagnosis = (
            "the reply contains a `## Knowledge Consulted` heading but "
            "no repo-relative `.claude/knowledge/...` or `references/...` "
            "path inside the block (a bare heading does not satisfy the "
            "check), and no such file was opened in this turn"
        )
    else:
        diagnosis = (
            "the transcript shows no read of any `.claude/knowledge/` "
            "or `references/` file in this turn, and no `## Knowledge "
            "Consulted` block was produced in the reply"
        )
    msg = (
        f"[KB-USAGE] This prompt was classified KB-relevant (gate "
        f"matched), but {diagnosis}. Repair: (a) Read a "
        f"`.claude/knowledge/**/*.md` or `references/**/*` file in this "
        f"turn before replying, OR (b) include a `## Knowledge "
        f"Consulted` block listing at least one repo-relative path you "
        f"actually consulted. The `N/A -- task not KB-relevant this "
        f"round` literal does NOT bypass this check on classifier-"
        f"relevant prompts; it is reserved for prompts the gate already "
        f"excluded. If you believe the gate over-fired, file a false-"
        f"positive entry under `negative:` in `scripts/kb/"
        f"_relevance_evals.yaml`."
    )

    if args.mode == "block":
        print(json.dumps({"decision": "block", "reason": msg}))
        return 0
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Hook contract: never break Claude Code on unexpected error.
        sys.exit(0)
