#!/usr/bin/env bash
# Authenticate or regenerate sim/trace/src/pb/dynamic_trace.rs from the tracer's
# authoritative .proto schema. Generation always occurs outside the repository;
# --check (the default) diffs without modifying tracked bytes, while --write
# atomically installs a differing result after that external generation succeeds.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
MODE="${1:---check}"
if [[ $# -gt 1 || ( "$MODE" != "--check" && "$MODE" != "--write" ) ]]; then
    echo "usage: $0 [--check|--write]" >&2
    exit 2
fi

# Same pin as tools/nvbit_tracer/pinned_build_env.sh ($PB_PREFIX/bin/protoc);
# the two entry points must agree on the protoc version.
PINNED_PROTOC="$REPO_ROOT/.toolchain/pb/bin/protoc"
PINNED_VERSION="libprotoc 34.1"
if [[ ! -x "$PINNED_PROTOC" ]]; then
    echo "FATAL: pinned protoc not found at $PINNED_PROTOC" >&2
    echo "(host-local .toolchain/pb — recreate via tools/nvbit_tracer/pinned_build_env.sh)" >&2
    exit 2
fi
GOT_VERSION="$("$PINNED_PROTOC" --version)"
if [[ "$GOT_VERSION" != "$PINNED_VERSION" ]]; then
    echo "FATAL: protoc at $PINNED_PROTOC reports '$GOT_VERSION', pin is '$PINNED_VERSION'" >&2
    echo "(the .toolchain prefix moved — update the pin HERE and in pinned_build_env.sh together)" >&2
    exit 2
fi

TMP_OUT="$(mktemp -d "${TMPDIR:-/tmp}/sim-trace-pbgen.XXXXXX")"
TARGET_TMP=""
cleanup() {
    rm -rf -- "$TMP_OUT"
    if [[ -n "$TARGET_TMP" ]]; then
        rm -f -- "$TARGET_TMP"
    fi
}
trap cleanup EXIT

case "$TMP_OUT" in
    "$REPO_ROOT"|"$REPO_ROOT"/*)
        echo "FATAL: temporary generator output resolved inside the repository: $TMP_OUT" >&2
        exit 2
        ;;
esac

# pbgen/Cargo.lock is COMMITTED so regeneration resolves the same prost-build
# on every host; --locked refuses an accidental resolution drift.
PROTOC="$PINNED_PROTOC" cargo run --locked \
    --manifest-path "$HERE/pbgen/Cargo.toml" --release -- \
    "$REPO_ROOT" "$TMP_OUT"

GENERATED="$TMP_OUT/dynamic_trace.rs"
COMMITTED="$HERE/src/pb/dynamic_trace.rs"
if [[ ! -f "$GENERATED" || -L "$GENERATED" ]]; then
    echo "FATAL: generator did not produce one regular dynamic_trace.rs" >&2
    exit 1
fi
if [[ ! -f "$COMMITTED" || -L "$COMMITTED" ]]; then
    echo "FATAL: committed binding is missing or not a regular file: $COMMITTED" >&2
    exit 1
fi

if cmp -s -- "$COMMITTED" "$GENERATED"; then
    echo "ok: committed Prost binding exactly matches pinned external regeneration"
    exit 0
fi

if [[ "$MODE" == "--check" ]]; then
    echo "FATAL: committed Prost binding differs from pinned external regeneration" >&2
    diff -u --label committed/dynamic_trace.rs --label regenerated/dynamic_trace.rs \
        "$COMMITTED" "$GENERATED" || true
    exit 1
fi

TARGET_TMP="$HERE/src/pb/.dynamic_trace.rs.tmp.$$"
cp -- "$GENERATED" "$TARGET_TMP"
chmod 0644 "$TARGET_TMP"
mv -f -- "$TARGET_TMP" "$COMMITTED"
TARGET_TMP=""
echo "updated $COMMITTED from pinned external regeneration"
