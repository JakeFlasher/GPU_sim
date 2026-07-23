"""Parse JSONL ``Trace`` lines emitted by the official sol-execbench CLI.

The upstream evaluator (invoked with ``--json``) prints one
``Trace``-shaped JSON object per line to stdout. This module reads
those lines through the live ``Trace`` Pydantic schema at
``sol-execbench/src/sol_execbench/core/data/trace.py`` — we do NOT
re-define the schema, only wrap the parse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
_REPO_ROOT = _SCRIPTS_DIR.parent


def _import_trace_model():
    """Import the live ``Trace`` Pydantic model (lazy)."""
    try:
        sys.path.insert(0, str(_REPO_ROOT / "sol-execbench" / "src"))
        from sol_execbench.core.data.trace import Trace  # noqa: E402
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import live Trace Pydantic model: {exc}. Ensure "
            f"sol-execbench/src is importable."
        ) from exc
    return Trace


def parse_jsonl_lines(
    lines: Iterable[str],
) -> Iterator[Tuple[int, Optional[Any], Optional[str]]]:
    """Iterate ``(line_no, trace, raw_line_if_failed)`` from JSONL lines.

    Yields ``(line_no, trace, None)`` for a successfully parsed trace and
    ``(line_no, None, raw_line)`` for an unparseable line (the caller
    decides whether to surface or skip those).

    The function is generator-shaped so a long-running harness can
    stream traces and surface partial progress.
    """
    Trace = _import_trace_model()
    for line_no, raw in enumerate(lines, start=1):
        raw_stripped = raw.strip()
        if not raw_stripped:
            continue
        try:
            obj = json.loads(raw_stripped)
        except json.JSONDecodeError:
            yield line_no, None, raw_stripped
            continue
        try:
            trace = Trace.model_validate(obj)
        except Exception:  # pydantic.ValidationError or similar
            yield line_no, None, raw_stripped
            continue
        yield line_no, trace, None


def parse_jsonl_text(text: str) -> List[Any]:
    """Parse a full JSONL blob and return the list of validated traces.

    Unparseable lines are silently skipped — use ``parse_jsonl_lines``
    directly when you need to surface them.
    """
    return [
        trace
        for _, trace, raw in parse_jsonl_lines(text.splitlines())
        if trace is not None
    ]
