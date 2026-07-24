#!/usr/bin/env bash
# Regenerate sim/trace/src/pb/dynamic_trace.rs from the tracer's .proto schema.
# Uses the repo's pinned protoc (34.1) so descriptor parsing matches the C++
# tracer build. Re-run whenever tools/nvbit_tracer/traces_enhanced/dynamic_trace/
# changes, then commit the regenerated file in the SAME commit as the .proto
# change (multi-surface lockstep rule).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
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
# pbgen/Cargo.lock is COMMITTED so regeneration resolves the same prost-build
# on every host; --locked refuses an accidental resolution drift.
PROTOC="$PINNED_PROTOC" cargo run --locked \
    --manifest-path "$HERE/pbgen/Cargo.toml" --release -- \
    "$REPO_ROOT" "$HERE/src/pb"
