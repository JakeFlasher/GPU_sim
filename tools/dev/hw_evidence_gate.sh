#!/usr/bin/env bash
# Authoritative, read-only machine-local evidence gate for the sim workspace.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd -P)
PY="$ROOT/.venv/bin/python"
OWNER="$ROOT/tools/nvbit_tracer/test_sim_rust_machine_local.py"
M1_BUILDER="$ROOT/tools/nvbit_tracer/build_m1_payload_authority.py"
ALIASES="$ROOT/evidence/path_aliases.v2.json"

missing() {
    printf 'hardware_evidence_prerequisite_missing: %s\n' "$1" >&2
    exit 2
}

fail() {
    printf 'hardware_evidence_gate_failed: %s\n' "$1" >&2
    exit 1
}

test -x "$PY" || missing "$PY"
command -v cargo >/dev/null 2>&1 || missing "cargo on PATH"
test -f "$OWNER" || missing "$OWNER"
test -f "$M1_BUILDER" || missing "$M1_BUILDER"
test -f "$ALIASES" || missing "$ALIASES"

TMP_ROOT=$(mktemp -d /tmp/gpu-sim-hw-evidence.XXXXXX) || missing "/tmp"
cleanup() {
    rm -rf -- "$TMP_ROOT"
}
trap cleanup EXIT HUP INT TERM
export CARGO_TARGET_DIR="$TMP_ROOT/cargo-target"
export CARGO_TERM_COLOR=never
export PYTHONDONTWRITEBYTECODE=1

# The builder verifies the source pin, all 17 current mappings, and all 5,033
# payloads without writing.  Preserve its missing-vs-mismatch taxonomy.
set +e
"$PY" "$M1_BUILDER" --verify-only
verify_rc=$?
set -e
case "$verify_rc" in
    0) ;;
    2) missing "M1 payload-authority verification input" ;;
    *) fail "M1 payload-authority verification" ;;
esac

# Parse only current_root values. Historical absolute roots are never emitted
# into the gate environment or a Rust command.
"$PY" - "$ALIASES" <<'PY'
import json
import sys
from pathlib import PurePosixPath

path = sys.argv[1]
doc = json.load(open(path, encoding="utf-8"))
rows = doc.get("rows")
if not isinstance(rows, list) or len(rows) != 17:
    raise SystemExit("path_aliases.v2.json must contain exactly 17 rows")
slugs = []
roots = []
prefixes = ("data/annex/m1_trace_capture_direct/", "workloads/portable_lcd/results/")
for row in rows:
    if not isinstance(row, dict) or set(row) != {"slug", "historical_root", "current_root"}:
        raise SystemExit("path-alias row shape mismatch")
    slug = row["slug"]
    root = row["current_root"]
    if not isinstance(slug, str) or not slug or not isinstance(root, str):
        raise SystemExit("path-alias row scalar mismatch")
    pure = PurePosixPath(root)
    if pure.is_absolute() or "\\" in root or any(p in ("", ".", "..") for p in pure.parts):
        raise SystemExit("unsafe current M1 root")
    if not root.startswith(prefixes):
        raise SystemExit("current M1 root outside declared prefixes")
    slugs.append(slug)
    roots.append(root)
if slugs != sorted(slugs) or len(set(slugs)) != 17 or len(set(roots)) != 17:
    raise SystemExit("path-alias rows are not 17 unique sorted current roots")
print("M1_CURRENT_ROOTS=17")
PY

IGNORED_LOG="$TMP_ROOT/ignored-list.log"
if ! cargo test --locked --release --manifest-path "$ROOT/sim/Cargo.toml" \
    --workspace --all-targets -- --list --ignored >"$IGNORED_LOG" 2>&1; then
    cat "$IGNORED_LOG" >&2
    fail "Rust ignored-test inventory command"
fi
"$PY" - "$IGNORED_LOG" <<'PY'
import sys
from pathlib import Path

expected = sorted([
    "real_sidecars_all_load: test",
    "real_trace_full_walk: test",
    "real_trace_replays_under_issue_model: test",
])
actual = sorted(
    line.strip()
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip().endswith(": test")
)
if actual != expected:
    raise SystemExit(f"ignored Rust test set mismatch: expected={expected!r} actual={actual!r}")
print("RUST_IGNORED_NAME_SET=3")
PY

BASELINE_LOG="$TMP_ROOT/rust-baseline.log"
if ! cargo test --locked --release --manifest-path "$ROOT/sim/Cargo.toml" \
    --workspace --all-targets >"$BASELINE_LOG" 2>&1; then
    cat "$BASELINE_LOG" >&2
    fail "Rust release workspace baseline"
fi
"$PY" - "$BASELINE_LOG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
rows = re.findall(
    r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;",
    text,
    flags=re.MULTILINE,
)
if len(rows) != 13:
    raise SystemExit(f"expected 13 libtest result lines, got {len(rows)}")
totals = tuple(sum(int(row[i]) for row in rows) for i in range(3))
if totals != (201, 0, 3):
    raise SystemExit(f"Rust libtest totals mismatch: expected=(201, 0, 3) actual={totals}")
print(f"RUST_LIBTEST_LINES={len(rows)}")
print("RUST_TOTALS=201/0/3")
PY

COLLECT_LOG="$TMP_ROOT/python-collect.log"
if ! "$PY" -m pytest --collect-only -q -p no:cacheprovider \
    -m hardware_evidence "$OWNER" >"$COLLECT_LOG" 2>&1; then
    cat "$COLLECT_LOG" >&2
    fail "Python hardware-owner collection"
fi
"$PY" - "$COLLECT_LOG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
if re.search(r"\b21 tests collected\b", text) is None:
    raise SystemExit(f"hardware owner collection is not exactly 21:\n{text}")
print("PYTHON_HARDWARE_COLLECTED=21")
PY

RUN_LOG="$TMP_ROOT/python-hardware.log"
if ! HW_EVIDENCE_GATE=1 "$PY" -m pytest -q -s -p no:cacheprovider \
    -m hardware_evidence "$OWNER" >"$RUN_LOG" 2>&1; then
    cat "$RUN_LOG" >&2
    fail "Python hardware-owner sentinel run"
fi
"$PY" - "$RUN_LOG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
if re.search(
    r"^21 passed in [0-9.]+s(?: \([0-9:]+\))?$",
    text,
    flags=re.MULTILINE,
) is None:
    raise SystemExit(f"hardware owner did not report exactly 21 passes:\n{text}")
required_smokes = ("smoke:", "smoke: replay", "smoke: l1", "smoke: l2")
missing = [token for token in required_smokes if token not in text]
if missing:
    raise SystemExit(f"hardware owner output is vacuous; missing {missing!r}")
print("PYTHON_HARDWARE_TOTALS=21/0/0")
print("HARDWARE_SMOKE_OUTPUT=NONVACUOUS")
PY

printf 'HARDWARE_EVIDENCE_GATE=PASS\n'
