// Host-only loaded-module origin resolver.  The /proc/self/maps parser is
// adapted from metric_extractor/module_origin_resolver.h; this version also
// turns a path into typed, hash-bound evidence for the exact accepted launch.

#ifndef NVBIT_TRACER_TRACER_TOOL_MODULE_ORIGIN_RESOLVER_H_
#define NVBIT_TRACER_TRACER_TOOL_MODULE_ORIGIN_RESOLVER_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "../metric_extractor/tier2_sha256.h"

namespace nvbit_tracer {

enum class MapsLookupKind {
  kMapped,
  kInline,
  kNotFound,
  kBadInput,
};

struct MapsLookupResult {
  std::string path;
  MapsLookupKind kind = MapsLookupKind::kNotFound;
};

inline MapsLookupResult LookupAddressInMapsStream(const void* host_ptr,
                                                  std::istream& maps) {
  MapsLookupResult result;
  if (host_ptr == nullptr) {
    result.kind = MapsLookupKind::kBadInput;
    return result;
  }
  const uintptr_t target = reinterpret_cast<uintptr_t>(host_ptr);
  std::string line;
  while (std::getline(maps, line)) {
    std::istringstream fields(line);
    std::string range, perms, offset, dev, inode;
    if (!(fields >> range >> perms >> offset >> dev >> inode)) continue;
    const size_t dash = range.find('-');
    if (dash == std::string::npos) continue;
    char* end = nullptr;
    const uintptr_t lo = std::strtoull(range.c_str(), &end, 16);
    if (end == range.c_str() || *end != '-') continue;
    const char* hi_text = range.c_str() + dash + 1;
    const uintptr_t hi = std::strtoull(hi_text, &end, 16);
    if (end == hi_text || *end != '\0' || hi <= lo) continue;
    if (target < lo || target >= hi) continue;

    std::string pathname;
    std::getline(fields, pathname);
    const size_t first = pathname.find_first_not_of(" \t");
    if (first == std::string::npos || pathname[first] == '[') {
      char marker[64];
      std::snprintf(marker, sizeof(marker), "<inline:0x%lx>",
                    static_cast<unsigned long>(target));
      result.path = marker;
      result.kind = MapsLookupKind::kInline;
      return result;
    }
    result.path = pathname.substr(first);
    result.kind = MapsLookupKind::kMapped;
    return result;
  }
  return result;
}

inline MapsLookupResult LookupAddressInProcessMaps(const void* host_ptr) {
  std::ifstream maps("/proc/self/maps");
  if (!maps) return MapsLookupResult{};
  return LookupAddressInMapsStream(host_ptr, maps);
}

struct ModuleOriginEvidence {
  // "stable_static_module" or the explicit typed terminal status required by
  // the capture consumer: "no_stable_static_module".
  std::string status = "no_stable_static_module";
  std::string reason = "origin_unresolved";
  std::string path;
  std::string basename;
  std::string sha256;
};

inline bool HasDeletedSuffix(const std::string& path) {
  static const std::string suffix = " (deleted)";
  return path.size() >= suffix.size() &&
         path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::string OriginBasename(const std::string& path) {
  std::string clean = path;
  if (HasDeletedSuffix(clean))
    clean.resize(clean.size() - std::string(" (deleted)").size());
  const size_t slash = clean.find_last_of('/');
  return slash == std::string::npos ? clean : clean.substr(slash + 1);
}

inline bool Sha256File(const std::string& path, std::string* digest_hex) {
  if (digest_hex == nullptr) return false;
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  tier2::Sha256Ctx digest;
  char buffer[64 * 1024];
  while (input) {
    input.read(buffer, sizeof(buffer));
    const std::streamsize count = input.gcount();
    if (count > 0) digest.Update(buffer, static_cast<size_t>(count));
  }
  if (!input.eof()) return false;
  uint8_t bytes[32];
  char hex[65];
  digest.Finalize(bytes);
  tier2::HexEncode32(bytes, hex);
  *digest_hex = hex;
  return true;
}

inline ModuleOriginEvidence EvaluateModuleOrigin(const std::string& raw_path) {
  ModuleOriginEvidence evidence;
  const char* alpha = std::getenv("NVBIT_ALPHA_NO_HASH");
  if (alpha != nullptr && std::string(alpha) == "1") {
    evidence.reason = "alpha_path_only";
    evidence.path = raw_path;
    evidence.basename = OriginBasename(raw_path);
    return evidence;
  }
  if (raw_path.empty()) return evidence;
  if (raw_path.rfind("<inline:", 0) == 0) {
    evidence.reason = "jit_or_anonymous_mapping";
    return evidence;
  }
  evidence.path = raw_path;
  evidence.basename = OriginBasename(raw_path);
  if (HasDeletedSuffix(raw_path)) {
    evidence.reason = "module_path_deleted";
    return evidence;
  }

  char* canonical = realpath(raw_path.c_str(), nullptr);
  if (canonical == nullptr) {
    evidence.reason = "module_path_missing";
    return evidence;
  }
  evidence.path = canonical;
  std::free(canonical);
  evidence.basename = OriginBasename(evidence.path);

  struct stat st {};
  if (stat(evidence.path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
    evidence.reason = "module_not_regular_file";
    return evidence;
  }
  if (!Sha256File(evidence.path, &evidence.sha256)) {
    evidence.reason = "module_hash_failed";
    return evidence;
  }
  evidence.status = "stable_static_module";
  evidence.reason = "resolved_file";
  return evidence;
}

inline std::string JsonEscape(const std::string& input) {
  std::ostringstream escaped;
  for (unsigned char c : input) {
    switch (c) {
      case '\"': escaped << "\\\""; break;
      case '\\': escaped << "\\\\"; break;
      case '\b': escaped << "\\b"; break;
      case '\f': escaped << "\\f"; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default:
        if (c < 0x20) {
          char hex[7];
          std::snprintf(hex, sizeof(hex), "\\u%04x", c);
          escaped << hex;
        } else {
          escaped << static_cast<char>(c);
        }
    }
  }
  return escaped.str();
}

}  // namespace nvbit_tracer

#endif  // NVBIT_TRACER_TRACER_TOOL_MODULE_ORIGIN_RESOLVER_H_
