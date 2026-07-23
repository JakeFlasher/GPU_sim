#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "../module_origin_resolver.h"

static bool expect(bool value, const char* message) {
  if (value) return true;
  std::fprintf(stderr, "FAIL: %s\n", message);
  return false;
}

int main() {
  bool ok = true;
  {
    std::istringstream maps(
        "00001000-00002000 r-xp 00000000 00:00 1 /tmp/lib target.so\n");
    auto result = nvbit_tracer::LookupAddressInMapsStream(
        reinterpret_cast<void*>(0x1800), maps);
    ok &= expect(result.kind == nvbit_tracer::MapsLookupKind::kMapped,
                 "mapped address classification");
    ok &= expect(result.path == "/tmp/lib target.so",
                 "maps parser must preserve pathname spaces");
  }
  {
    std::istringstream maps(
        "00001000-00002000 rw-p 00000000 00:00 0 [heap]\n");
    auto result = nvbit_tracer::LookupAddressInMapsStream(
        reinterpret_cast<void*>(0x1800), maps);
    ok &= expect(result.kind == nvbit_tracer::MapsLookupKind::kInline,
                 "heap mapping must be typed inline/JIT");
  }

  char path[] = "/tmp/nvbit_module_origin_XXXXXX";
  int fd = mkstemp(path);
  ok &= expect(fd >= 0, "mkstemp");
  if (fd >= 0) {
    const char payload[] = "abc";
    ok &= expect(write(fd, payload, 3) == 3, "write hash fixture");
    close(fd);
    auto evidence = nvbit_tracer::EvaluateModuleOrigin(path);
    ok &= expect(evidence.status == "stable_static_module",
                 "regular file must resolve stable module");
    ok &= expect(evidence.reason == "resolved_file", "stable reason");
    ok &= expect(evidence.basename ==
                     std::string(path).substr(std::string(path).find_last_of('/') + 1),
                 "stable origin must retain relocation basename hint");
    ok &= expect(evidence.sha256 ==
                     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                 "SHA-256 must bind exact module bytes");
    unlink(path);
  }

  auto deleted = nvbit_tracer::EvaluateModuleOrigin(
      "/tmp/gone.so (deleted)");
  ok &= expect(deleted.status == "no_stable_static_module",
               "deleted module must never be presented as stable");
  ok &= expect(deleted.reason == "module_path_deleted",
               "deleted module typed reason");
  ok &= expect(deleted.basename == "gone.so",
               "deleted suffix must not leak into basename hint");

  auto jit = nvbit_tracer::EvaluateModuleOrigin("<inline:0x1234>");
  ok &= expect(jit.status == "no_stable_static_module",
               "anonymous/JIT origin must be explicitly unavailable");
  ok &= expect(jit.reason == "jit_or_anonymous_mapping", "JIT typed reason");
  ok &= expect(jit.path.empty() && jit.sha256.empty(),
               "JIT evidence must not fabricate path/hash");

  ok &= expect(nvbit_tracer::JsonEscape("a\n\"b") == "a\\n\\\"b",
               "deterministic JSON escaping");

  if (!ok) return 1;
  std::puts("module_origin_resolver_test: PASS");
  return 0;
}
