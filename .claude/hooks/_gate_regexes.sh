# Auto-generated from scripts/kb/_relevance.py — DO NOT EDIT BY HAND.
# Regenerate via: python3 scripts/kb/_relevance.py --emit-bash > .claude/hooks/_gate_regexes.sh
#
# This file is meant to be sourced by a bash hook:
#     source .claude/hooks/_gate_regexes.sh
#     if kb_gate_is_relevant "$PROMPT"; then ...; fi
#
# The Python classifier in scripts/kb/_relevance.py is the canonical source of
# truth. Bash delegates to it through stdin so long prompts do not collide with
# the operating system's argv-size limit and no regex dialect can drift.

kb_gate_is_relevant() {
    # Silent predicate. Returns exit status 0 when the prompt is KB-relevant,
    # 1 otherwise (including an empty prompt or unavailable Python helper).
    # No stdout output: UserPromptSubmit hooks show stdout to the agent, so a
    # stray true/false line would pollute every reminder.
    local prompt="${1-}"
    if [ -z "$prompt" ]; then
        return 1
    fi
    # Resolve the Python script path relative to this sourced file:
    # <repo>/.claude/hooks/_gate_regexes.sh ->
    # <repo>/scripts/kb/_relevance.py. Avoids depending on git being available
    # or on a particular working directory.
    local _hook_dir _repo_root _script result
    _hook_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    _repo_root="$(dirname "$(dirname "$_hook_dir")")"
    _script="$_repo_root/scripts/kb/_relevance.py"
    result="$(printf '%s' "$prompt" | python3 "$_script" --classify-stdin 2>/dev/null)"
    [ "$result" = "true" ]
}
