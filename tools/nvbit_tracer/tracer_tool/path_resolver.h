// Host-only path resolver used by the NVBit tracer + tested
// in isolation by tests/path_resolver_test.cc.
//
// `resolve_output_path(cwd, p)` returns `p` verbatim when `p` is
// an absolute filesystem path, else returns `cwd / p`. R52 added
// this helper to fix the absolute-`TRACES_FOLDER` bug Codex's R51
// review surfaced: the pre-R52 sites at tracer_tool.cu:858, :1004,
// :3505 built `cwd + "/" + sass_path` (resp. register_usage_path,
// cubin_path) via raw std::string concatenation, so when the R51
// `USER_DEFINED_FOLDERS=1` redirect set those globals to absolute
// paths (which `run_suite.py::handle_trace` produces by default
// because `--out` resolves to `str(SUITE_ROOT / "results")`),
// the resulting filesystem operations targeted bogus paths like
// `<cwd>//abs/results/.../sass/`. POSIX collapses double slashes
// in the middle of a path, but the `<cwd>` prefix is still
// nonsense. This helper preserves the legacy relative-path
// behavior unchanged while making the absolute-input case
// correct.
//
// The header is intentionally self-contained (no NVBit / CUDA /
// project includes) so it can be linked into a g++-only host
// unit test without bringing the rest of the tracer's
// translation-unit machinery along — same pattern as
// allocation_registry, exclusion_aggregator,
// phase_classifier, memory_op_class.
#ifndef NVBIT_TRACER_TOOL_PATH_RESOLVER_H
#define NVBIT_TRACER_TOOL_PATH_RESOLVER_H

#include <filesystem>
#include <string>

inline std::string resolve_output_path(const std::string &cwd,
                                       const std::string &p) {
  std::filesystem::path fp(p);
  if (fp.is_absolute()) {
    return p;
  }
  return (std::filesystem::path(cwd) / fp).string();
}

#endif  // NVBIT_TRACER_TOOL_PATH_RESOLVER_H
