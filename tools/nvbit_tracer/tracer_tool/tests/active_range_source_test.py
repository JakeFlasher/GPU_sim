#!/usr/bin/env python3
"""Static consumer-path regression for the CUDA-only callback surface.

The state machine itself has a host C++ test.  This check complements it by
proving that tracer_tool.cu's real launch/receiver consumers invoke the gate
before any output-producing path; no GPU is required.
"""

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "tracer_tool.cu").read_text()


def ordered(*needles: str) -> None:
    cursor = 0
    for needle in needles:
        position = SOURCE.find(needle, cursor)
        assert position >= 0, f"missing consumer marker: {needle}"
        cursor = position + len(needle)


ordered(
    "active_launch_gate.BeginLaunch()",
    "ActiveLaunchDecision::kConcurrentLimitViolation",
    "_exit(EXIT_FAILURE);",
    "ActiveLaunchDecision::kLimitReached",
    "nvbit_enable_instrumented(ctx, kernel_func, false);",
    "pthread_mutex_unlock(&mutex);",
    "return;",
    "ActiveLaunchDecision::kInactive",
    "nvbit_enable_instrumented(ctx, kernel_func, false);",
    "pthread_mutex_unlock(&mutex);",
    "return;",
    "pending_accepted_launch.accepted = true;",
    "ts_scope_dims_captured = true;",
    "statsFile = fopen(stats_location.c_str(), \"a\");",
    "ctx_state->recv_thread_receiving.store(true",
)

ordered(
    "The matching launch entry was outside the active profiler region.",
    "if (!pending_accepted_launch.accepted)",
    "pthread_mutex_unlock(&mutex);",
    "return;",
    "cudaDeviceSynchronize();",
)

ordered(
    "const bool batch_is_accepted =",
    "if (!batch_is_accepted)",
    "inactive_channel_records_dropped.fetch_add",
    "continue;",
    "accepted_channel_records.fetch_add",
    "get_threadblock(tb_string_id)",
)

assert "if (ts_scope_device < 0)" not in SOURCE, (
    "receiver must not let the first process launch claim sampling scope"
)
assert "MAX_ACCEPTED_ACTIVE_LAUNCHES=1" in SOURCE
assert "write_module_origin_sidecar(extrainfo_path);" in SOURCE
assert r'\"stable_key\": \"sha256\"' in SOURCE
assert r'\"path_semantics\": \"path_at_capture\"' in SOURCE
assert r'\"rebind_contract\": \"sha256_required_basename_hint\"' in SOURCE
assert r'\"path_at_capture\"' in SOURCE and r'\"basename\"' in SOURCE

print("active_range_source_test: PASS")
