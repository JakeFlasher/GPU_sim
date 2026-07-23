#!/usr/bin/env bash
# Run tests that validate immutable, machine-local campaign evidence.

set -euo pipefail

repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
python=${PYTHON:-"$repo_root/.venv/bin/python"}
archive_root=${SM86_REFERENCE_CAPTURE_ROOT:-/home/jakeshea/sm86_reference_capture_r27}

if [[ ! -x "$python" ]]; then
    echo "ERROR: project Python is unavailable: $python" >&2
    exit 2
fi

if [[ ! -d "$archive_root" ]]; then
    echo "ERROR: campaign evidence archive is unavailable: $archive_root" >&2
    exit 2
fi

cd "$repo_root"
exec "$python" -m pytest -q -o addopts= -m campaign_evidence "$@"
