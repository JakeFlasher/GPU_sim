// Unit tests for the `resolve_output_path` helper in
// `path_resolver.h`. Compiles with `g++ -std=c++17` only — no NVBit,
// no CUDA, no nvcc. Driven by `make -C tools/nvbit_tracer/tracer_tool
// unit-tests`.
//
// R52 added this helper to fix the absolute-`TRACES_FOLDER` bug
// Codex's R51 review surfaced. The pre-R52 sites at
// tracer_tool.cu:858, :1004, :3505 built `cwd + "/" + sass_path`
// (resp. register_usage_path, cubin_path) via raw std::string
// concatenation; with the R51 USER_DEFINED_FOLDERS=1 redirect
// pointing those globals at an absolute TRACES_FOLDER (which
// `run_suite.py::handle_trace` produces by default), the result
// was bogus paths like `<cwd>//abs/results/.../sass/`. The
// resolver makes the absolute case correct while preserving the
// legacy relative behavior.

#include "../path_resolver.h"

#include <cassert>
#include <cstdio>
#include <string>

int main() {
  // ============================================================
  // Absolute input is preserved verbatim.
  // ============================================================
  // The most important contract: when the orchestrator passes an
  // absolute TRACES_FOLDER (via run_suite.py's `--out`-derived
  // path resolution), the tracer must NOT prefix `cwd` to it.
  {
    std::string result = resolve_output_path("/some/cwd",
                                             "/abs/results/slug/sm_120/trace_00");
    assert(result == "/abs/results/slug/sm_120/trace_00");
  }
  {
    // Trailing slash on absolute input is preserved (the helper
    // does not normalize, only chooses between concat and
    // verbatim).
    std::string result = resolve_output_path("/cwd",
                                             "/abs/results/slug/");
    assert(result == "/abs/results/slug/");
  }
  {
    // The `cwd` argument is irrelevant when input is absolute —
    // any cwd, including an empty one, leaves the absolute
    // input unchanged.
    std::string result = resolve_output_path("",
                                             "/abs/path");
    assert(result == "/abs/path");
  }

  // ============================================================
  // Relative input is prefixed with `cwd` via std::filesystem.
  // ============================================================
  // The legacy (pre-R52) tracer always wrote outputs at
  // `cwd / "traces/extra_info/sass"` etc. That behavior must be
  // preserved when callers pass relative paths (the case for
  // unit tests that don't set TRACES_FOLDER, and for any
  // legacy operator workflow that runs the tracer without
  // USER_DEFINED_FOLDERS=1).
  {
    std::string result = resolve_output_path("/some/cwd",
                                             "traces/extra_info/sass");
    // std::filesystem::path::operator/ produces single-slash
    // joins on POSIX; the result is a normalized absolute
    // path rooted at cwd.
    assert(result == "/some/cwd/traces/extra_info/sass");
  }
  {
    // Legacy default: bare "traces" subdir under cwd.
    std::string result = resolve_output_path("/cwd", "traces");
    assert(result == "/cwd/traces");
  }
  {
    // Multi-component relative input.
    std::string result = resolve_output_path(
        "/home/jakeshea/repo",
        "traces/extra_info/register_usage");
    assert(result ==
           "/home/jakeshea/repo/traces/extra_info/register_usage");
  }

  // ============================================================
  // Edge cases
  // ============================================================
  {
    // Empty relative input + non-empty cwd: behaves like
    // std::filesystem::path's behavior of `cwd /
    // empty_path()`, which is `cwd` itself (no trailing slash
    // forced). The exact form depends on the implementation;
    // both `/cwd` and `/cwd/` are acceptable in practice. The
    // test pins the more common `/cwd` form; if the
    // implementation diverges, this test must be updated in
    // lockstep with the helper.
    std::string result = resolve_output_path("/cwd", "");
    // `(std::filesystem::path("/cwd") / "").string()` is
    // implementation-defined but in practice yields "/cwd/"
    // on libstdc++; we accept either form.
    assert(result == "/cwd" || result == "/cwd/");
  }
  {
    // `.` as relative input is a no-op: the resolver returns
    // the cwd path unchanged (modulo the trailing `/.` that
    // std::filesystem may append). Callers do not pass `.`
    // in practice, but this case exercises the relative-path
    // branch with a degenerate input.
    std::string result = resolve_output_path("/cwd", ".");
    // Either "/cwd" (normalized) or "/cwd/." (literal join)
    // is acceptable; the helper does not normalize.
    assert(result == "/cwd" || result == "/cwd/." ||
           result == "/cwd/");
  }

  std::printf("path_resolver_test: all assertions passed\n");
  return 0;
}
