#!/usr/bin/env bash
# .claude/hooks/kb_reminder.sh
#
# Bash-side fallback for the UserPromptSubmit hook. The primary surface
# is `scripts/kb/route_query.py` (FTS5-backed). This script runs when:
#   - The Python router falls through (missing manifest / zero FTS hits).
#   - A non-Claude tool reads the KB and wants the same gating logic.
#
# Single source of truth for the gate is `scripts/kb/_relevance.py`.
# This script SOURCES the auto-generated `_gate_regexes.sh` rather than
# carrying its own copy of the regex. To regenerate after editing
# `_relevance.py`:
#
#   python3 scripts/kb/_relevance.py --emit-bash > .claude/hooks/_gate_regexes.sh
#
# Hook contract: read JSON event from stdin, optionally print a reminder
# block to stdout, exit 0 unconditionally.

set -u  # keep the fallback fail-quiet when optional inputs are unavailable

prompt="$(jq -r '.prompt // empty' 2>/dev/null || true)"
[ -z "$prompt" ] && exit 0

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE_FILE="$HERE/_gate_regexes.sh"

# If the generated wrapper is missing, exit silently. This fallback never
# regenerates or mutates repository files; regenerate the checked-in artifact
# explicitly with the command in the header after changing `_relevance.py`.
if [ ! -r "$GATE_FILE" ]; then
  exit 0
fi

# shellcheck disable=SC1090
source "$GATE_FILE"

# The generated gate delegates to the canonical Python classifier so bash and
# Python cannot drift on regex syntax. Missing/malformed generated helpers fail
# silent because the primary router already ran.
if ! declare -F kb_gate_is_relevant >/dev/null; then
  exit 0
fi
if ! kb_gate_is_relevant "$prompt"; then
  exit 0
fi

# Compressed fallback reminder. The primary router may provide ranked cards when
# its manifest is available; this path deliberately emits only a generic
# consultation reminder and does not claim routing parity.
printf '%s\n' \
  '[KB] Prompt looks KB-relevant. Open + cite under `## Knowledge Consulted`:' \
  '  - `.claude/knowledge/INDEX.md` (start here if topic is unclear)' \
  '  - skill cards under `.claude/skills/<name>/SKILL.md` for the matching specialty' \
  '  - distilled cards under `.claude/knowledge/<subcorpus>/*.md`'

exit 0
