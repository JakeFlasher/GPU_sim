// nvbit_tracer/metric_extractor/metric_extractor.cu
//
// Host-side entry point for the on-device metric extractor — a counter-free
// NVBit 1.8 instrumentation tool that emits one compact JSON-L summary per
// kernel launch. Stands in for the metrics a hardware-counter profiler would
// otherwise emit on a non-root counter-free B200 (sm_100a) host: instruction-
// class histogram (with TCGen05 / TMA / DSMEM in their own buckets), 32B-
// sector efficiency for global LDG/STG, shared-bank-conflict tally, and a
// branch active-mask histogram.
//
// Wall-clock attribution is intentionally NOT collected here. CUPTI Activity
// owns wall-clock. This tool queries only CUDA Driver context/stream IDs;
// calling CUPTI here conflicts with NVBit's instrumentation subscriber.
//
// Build:
//   make ARCH=sm_86       — local 3090 Ti.
//   make ARCH=sm_120      — cloud RTX PRO 6000 Workstation.
//   make ARCH=sm_100a     — cloud B200 cross-compile.
//
// Build-up of the tool, oldest first:
//   - skeleton with a no-op launch hook (build target only).
//   - per-launch state machine — fixed-cap LaunchMetrics slot pool
//     allocated lazily in nvbit_tool_init(); slot index handed to
//     instrumented device functions via nvbit_set_at_launch();
//     cudaDeviceSynchronize() before per-launch slot read on exit.
//   - launch-foundation slice: SASS-walk + nvbit_insert_call so
//     instrumented kernels actually receive the per-launch slot
//     pointer; user-module regex filter (via the local header-only
//     /proc/self/maps lookup at module_origin_resolver.h) so only
//     user kernels are instrumented; cooperative + graph-node
//     launch coverage; per-process filter_stats.json on terminate.
//   - (next) JSON-L per-launch emit, then metric counter families.

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <fstream>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

// every nvbit tool needs to include this once
#include "nvbit_tool.h"

// nvbit interface
#include "nvbit.h"

#include "common.h"
#include "tier2_sha256.h"
#include "tier2_frame_serializer.h"

// Header-only /proc/self/maps lookup helper. Pure host-side, no NVBit or
// CUDA surface; builds into a standalone host unit test. Resolves a
// host-side pointer to the pathname of the .so / cubin / mapped file
// that contains it via /proc/self/maps. Used by the cuModuleLoadData /
// DataEx / FatBinary at-exit arms to capture the on-disk origin of an
// in-memory image at load time, before torch's JIT tempdir .so files
// can be unlinked. The pathname tail is preserved verbatim (including
// any trailing " (deleted)" suffix) so the caller's filter regex can
// match either the live path or the post-unlink path explicitly.
#include "module_origin_resolver.h"

// ISA classification headers. The Blackwell map plus the shared opcode /
// operation-type enums cover sm_120 / sm_100a SASS. The Ampere map remains
// available for sm_86 calibration runs; the metric extractor's lookup
// prefers Blackwell entries and falls back through the ampere map for any
// keys that exist only in the older ISA. The headers are header-only
// (constant std::unordered_map literals), with no link-time deps beyond
// the local translation unit.
#include "isa_def/blackwell_opcode.h"
#include "isa_def/hopper_opcode.h"
#include "isa_def/ampere_opcode.h"
#include "isa_def/operation_type.h"
#include "isa_def/trace_opcode.h"

// Pure host-side instruction-class router shared without NVBit's CUDA
// dependencies.
#include "classifier.h"

// --------------------------------------------------------------------------
// Per-launch slot pool — lazily allocated in nvbit_tool_init() so the
// CUDA-managed allocation lives in the only NVBit lifecycle slot where
// CUDA-API calls are documented as safe (nvbit.h:278-282 forbids them in
// nvbit_at_ctx_init). The slot pool is shared across all CUDA contexts in
// the host process; the launch mutex serialises slot reset / read so two
// in-flight launches do not stomp on each other.
// --------------------------------------------------------------------------

static pthread_mutex_t g_launch_mutex = PTHREAD_MUTEX_INITIALIZER;
static LaunchMetrics*  g_slot_pool         = nullptr;
static uint32_t        g_max_launch_slots  = 0;
// Parallel pool of Tier-2 address-dump per-launch buffers. Allocated
// only when METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP=1 is on at startup
// (see nvbit_tool_init). NULL when the gate is off; in that case the
// host-side launch-enter leaves LaunchMetrics::tier2_buffer_ptr at
// the post-memset zero (NULL) and the device callback short-circuits
// without writing into the records array.
static TierTwoBuffer*  g_tier2_pool        = nullptr;

// Global observed-launch counter. Increments on every kernel-launch CBID
// the hook observes, including launches skipped by the user-module filter.
// This preserves the launch order needed by any independently captured
// CUPTI activity run; process-local stream IDs remain namespace-tagged.
static uint64_t        g_launch_index      = 0;

// Dense accepted-launch counter. Used ONLY for the ring-buffer slot
// index so the per-launch ``LaunchMetrics`` slot pool stays compact —
// the slot pool's job is bounded per-launch device memory, and only
// accepted (instrumented) launches actually allocate a slot. Skipped
// launches NEVER read or write a slot, so this counter must stay
// independent of ``g_launch_index``. The slot index for an accepted
// launch is therefore ``g_accepted_slot_index % g_max_launch_slots``,
// not ``g_launch_index % g_max_launch_slots`` as the pre-Round-2
// implementation incorrectly aliased the two roles.
static uint64_t        g_accepted_slot_index = 0;

// Set by launch-enter, consulted by launch-exit. Because the launch mutex
// is held from enter to exit, this is race-free across launch boundaries.
// A `false` value tells launch-exit to skip the cudaDeviceSynchronize()
// barrier (the launch was filtered out and never instrumented, so there
// is nothing in flight that the host-side slot read could race against).
static bool g_last_launch_accepted = false;
// True when the launch-enter branch tagged the kernel as a
// per-process-budget skip. The launch-exit branch consumes the flag
// to decide whether to emit a bounded skipped record (no
// cudaDeviceSynchronize needed because the kernel ran uninstrumented).
static bool g_pending_budget_skipped_record = false;

// --------------------------------------------------------------------------
// Module-origin registry — populated by the cuModuleLoadData / Ex /
// FatBinary / cuLibraryLoadData / cuLibraryLoadFromFile / cuModuleGetFunction
// callbacks; consulted on launch-enter to decide whether the function lives
// in a user module or in a framework / library module that the regex filter
// rejects.
// --------------------------------------------------------------------------

namespace {

struct ModuleKey {
    CUcontext ctx;
    CUmodule  module;
    bool operator==(const ModuleKey& other) const {
        return ctx == other.ctx && module == other.module;
    }
};

struct LibraryKey {
    CUcontext  ctx;
    CUlibrary  library;
    bool operator==(const LibraryKey& other) const {
        return ctx == other.ctx && library == other.library;
    }
};

struct FuncKey {
    CUcontext  ctx;
    CUfunction func;
    bool operator==(const FuncKey& other) const {
        return ctx == other.ctx && func == other.func;
    }
};

// Key into the mangled-name fallback registry. NVBit-instrumented
// CUfunction handles can differ from the CUDA RUNTIME's handle for
// the same kernel — the only invariant is the mangled symbol name —
// so when the (ctx, func) lookup misses, we retry by name.
struct FuncNameKey {
    CUcontext   ctx;
    std::string mangled_name;
    bool operator==(const FuncNameKey& other) const {
        return ctx == other.ctx && mangled_name == other.mangled_name;
    }
};

struct ModuleKeyHash {
    size_t operator()(const ModuleKey& k) const noexcept {
        return std::hash<void*>()(k.ctx) ^
               (std::hash<void*>()(k.module) << 1);
    }
};

struct LibraryKeyHash {
    size_t operator()(const LibraryKey& k) const noexcept {
        return std::hash<void*>()(k.ctx) ^
               (std::hash<void*>()(k.library) << 1);
    }
};

struct FuncKeyHash {
    size_t operator()(const FuncKey& k) const noexcept {
        return std::hash<void*>()(k.ctx) ^
               (std::hash<void*>()(k.func) << 1);
    }
};

struct FuncNameKeyHash {
    size_t operator()(const FuncNameKey& k) const noexcept {
        return std::hash<void*>()(k.ctx) ^
               (std::hash<std::string>()(k.mangled_name) << 1);
    }
};

}  // namespace

static std::mutex g_origin_mu;
static std::unordered_map<ModuleKey,   std::string, ModuleKeyHash>
    g_module_origin;
static std::unordered_map<LibraryKey,  std::string, LibraryKeyHash>
    g_library_origin;
static std::unordered_map<FuncKey,     std::string, FuncKeyHash>
    g_func_origin;
static std::unordered_map<FuncNameKey, std::string, FuncNameKeyHash>
    g_func_name_origin;

// --------------------------------------------------------------------------
// Filter regex — initialised lazily from METRIC_EXTRACTOR_USER_MODULE_REGEX
// env var. Default matches in-tree user-build prefix `sol_execbench_*` so
// the filter is useful out of the box on this repo's harness fanout.
// --------------------------------------------------------------------------

static std::mutex   g_regex_mu;
static std::regex   g_user_module_regex;
static std::string  g_user_module_regex_source;
static bool         g_user_module_regex_initialized = false;

static const char* DEFAULT_USER_MODULE_REGEX = ".*sol_execbench.*";

static void init_user_module_regex_locked() {
    if (g_user_module_regex_initialized) return;
    const char* env = getenv("METRIC_EXTRACTOR_USER_MODULE_REGEX");
    g_user_module_regex_source =
        (env != nullptr && env[0] != '\0') ? env : DEFAULT_USER_MODULE_REGEX;
    try {
        g_user_module_regex = std::regex(g_user_module_regex_source);
    } catch (const std::regex_error& e) {
        fprintf(stderr,
                "[metric_extractor] METRIC_EXTRACTOR_USER_MODULE_REGEX "
                "invalid (%s); falling back to default %s\n",
                e.what(), DEFAULT_USER_MODULE_REGEX);
        g_user_module_regex_source = DEFAULT_USER_MODULE_REGEX;
        g_user_module_regex = std::regex(g_user_module_regex_source);
    }
    g_user_module_regex_initialized = true;
}

static bool should_accept_origin(const std::string& origin) {
    std::lock_guard<std::mutex> lk(g_regex_mu);
    init_user_module_regex_locked();
    return std::regex_search(origin, g_user_module_regex);
}

// --------------------------------------------------------------------------
// Kernel-name regex — sibling filter to the module-origin regex above.
// When METRIC_EXTRACTOR_KERNEL_NAME_REGEX is set, only kernels whose
// demangled name matches the regex are accepted. When unset, every
// kernel that passes should_accept_origin() is accepted (back-compat).
//
// Use-case: with Tier-2 address-dump enabled, the per-slot Tier-2 buffer
// is ~503 MiB, so the default 512 MiB per-process budget admits only ONE
// instrumented launch. If a PyTorch host script runs a curand RNG kernel
// to initialise input tensors BEFORE the target pilot kernel, the RNG
// kernel grabs the single slot and the actual pilot kernel is skipped
// with per_process_budget_exceeded. The kernel-name filter sidesteps
// this by rejecting launches that do not match the regex, leaving the
// slot pool free for the target kernel.
// --------------------------------------------------------------------------

static std::regex   g_kernel_name_regex;
static std::string  g_kernel_name_regex_source;
static bool         g_kernel_name_regex_enabled = false;
static bool         g_kernel_name_regex_initialized = false;

static void init_kernel_name_regex_locked() {
    if (g_kernel_name_regex_initialized) return;
    const char* env = getenv("METRIC_EXTRACTOR_KERNEL_NAME_REGEX");
    if (env != nullptr && env[0] != '\0') {
        g_kernel_name_regex_source = env;
        try {
            g_kernel_name_regex = std::regex(g_kernel_name_regex_source);
            g_kernel_name_regex_enabled = true;
        } catch (const std::regex_error& e) {
            fprintf(stderr,
                    "[metric_extractor] METRIC_EXTRACTOR_KERNEL_NAME_REGEX "
                    "invalid (%s); disabling kernel-name filter\n",
                    e.what());
            g_kernel_name_regex_enabled = false;
        }
    }
    g_kernel_name_regex_initialized = true;
}

static bool should_accept_kernel_name(const std::string& demangled_name) {
    std::lock_guard<std::mutex> lk(g_regex_mu);
    init_kernel_name_regex_locked();
    if (!g_kernel_name_regex_enabled) return true;
    return std::regex_search(demangled_name, g_kernel_name_regex);
}

// --------------------------------------------------------------------------
// Tier-2 address-dump env gates. Read once on first need; default off
// for the address-dump gate, and default 64 MiB for the per-launch
// trace-byte budget. The instrumentation pass at the global LDG / STG
// hook site consults the address-dump gate to decide whether to
// install the additional ``count_tier2_global_mem_access`` callback
// alongside the existing ``count_global_mem_access``. When the gate
// is off no extra callback is wired into the instrumented function,
// no extra device-side work runs, and no sidecar bytes are produced —
// Tier-1 byte-identity holds by construction.
//
// The trace-byte budget is enforced by the host-side sidecar writer:
// during serialization it stops emitting compact records once the
// running byte count would exceed the budget, then emits a
// ``TruncationRecordV1`` whose ``budget_bytes_limit`` records the
// operator's value. JSONL emits ``tier2_trace_truncated=true`` and
// ``tier2_trace_typed_reason="trace_budget_exceeded"`` in that case.
// All three env names are entered into ``NVBIT_METRICS_USER_ENV_VARS``
// on the host plugin side so flipping any value invalidates the
// surface cache identity.
// --------------------------------------------------------------------------

#define METRIC_EXTRACTOR_TIER2_DEFAULT_BUDGET_BYTES ((uint64_t)67108864ULL)

static std::mutex   g_tier2_address_dump_mu;
static bool         g_tier2_address_dump_enabled = false;
static bool         g_tier2_address_dump_initialized = false;
static bool         g_tier2_per_lane_enabled = false;
static uint64_t     g_tier2_trace_budget_bytes =
    METRIC_EXTRACTOR_TIER2_DEFAULT_BUDGET_BYTES;
// Round 18 (Codex Round-17 review Finding 3): the producer sources the
// profiled-binary identity digest by hashing the bytes of the accepted
// module origin on launch-enter. ``g_tier2_binary_sha256`` is now used
// only as the env-var TEST OVERRIDE path (Round 17's behavior); the
// production path hashes the loaded .so once per origin and caches the
// result. The launch-enter hook populates ``LaunchContext.binary_sha256``
// using the env-override when explicitly set, otherwise the per-origin
// hash. The host serializer reads ``lc.binary_sha256`` directly.
static uint8_t      g_tier2_binary_sha256[32] = {0};
static bool         g_tier2_binary_sha256_env_explicitly_set = false;

static bool tier2_hex_to_byte(char hi, char lo, uint8_t* out) {
    auto nibble = [](char c, int* v) -> bool {
        if (c >= '0' && c <= '9') { *v = c - '0'; return true; }
        if (c >= 'a' && c <= 'f') { *v = 10 + (c - 'a'); return true; }
        if (c >= 'A' && c <= 'F') { *v = 10 + (c - 'A'); return true; }
        return false;
    };
    int h, l;
    if (!nibble(hi, &h) || !nibble(lo, &l)) return false;
    *out = static_cast<uint8_t>((h << 4) | l);
    return true;
}

static void init_tier2_address_dump_locked() {
    if (g_tier2_address_dump_initialized) return;
    const char* env = getenv("METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP");
    g_tier2_address_dump_enabled =
        (env != nullptr && env[0] == '1' && env[1] == '\0');
    const char* per_lane_env =
        getenv("METRIC_EXTRACTOR_TIER2_PER_LANE");
    g_tier2_per_lane_enabled =
        (per_lane_env != nullptr && per_lane_env[0] == '1' &&
         per_lane_env[1] == '\0');
    const char* budget_env =
        getenv("METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES");
    if (budget_env != nullptr && budget_env[0] != '\0') {
        char* endp = nullptr;
        const unsigned long long parsed =
            strtoull(budget_env, &endp, 10);
        if (endp != nullptr && *endp == '\0' && parsed > 0ULL) {
            g_tier2_trace_budget_bytes = (uint64_t)parsed;
        } else {
            fprintf(stderr,
                    "[metric_extractor] WARNING: ignoring invalid "
                    "METRIC_EXTRACTOR_TIER2_TRACE_BUDGET_BYTES=%s; "
                    "falling back to default %llu bytes.\n",
                    budget_env,
                    (unsigned long long)
                        METRIC_EXTRACTOR_TIER2_DEFAULT_BUDGET_BYTES);
        }
    }
    // Round 17: parse METRIC_EXTRACTOR_TIER2_BINARY_SHA256_HEX. 64 hex chars
    // exactly decode to 32 bytes. Malformed input (wrong length, non-hex
    // chars) logs a typed-reason warning and falls back to 32 zeros so the
    // surface keeps producing; the typed reason surfaces the misconfiguration
    // without blocking the launch.
    const char* sha_env =
        getenv("METRIC_EXTRACTOR_TIER2_BINARY_SHA256_HEX");
    if (sha_env != nullptr && sha_env[0] != '\0') {
        size_t n = strlen(sha_env);
        bool ok = (n == 64);
        if (ok) {
            for (size_t i = 0; i < 32; ++i) {
                if (!tier2_hex_to_byte(
                        sha_env[i * 2], sha_env[i * 2 + 1],
                        &g_tier2_binary_sha256[i])) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            memset(g_tier2_binary_sha256, 0, 32);
            fprintf(stderr,
                    "[metric_extractor] WARNING: "
                    "tier2_binary_sha256_hex_invalid: "
                    "METRIC_EXTRACTOR_TIER2_BINARY_SHA256_HEX must be 64 "
                    "lowercase or uppercase hex characters; got %zu chars; "
                    "falling back to 32-zero placeholder.\n",
                    n);
        } else {
            // Round 18: only honor the env-var override when it parsed
            // cleanly. The launch-enter hook checks this flag to decide
            // between the env-supplied digest (test override) and the
            // per-origin file hash (production path).
            g_tier2_binary_sha256_env_explicitly_set = true;
        }
    }
    g_tier2_address_dump_initialized = true;
}

static uint64_t tier2_trace_budget_bytes() {
    std::lock_guard<std::mutex> lk(g_tier2_address_dump_mu);
    init_tier2_address_dump_locked();
    return g_tier2_trace_budget_bytes;
}

static bool tier2_address_dump_enabled() {
    std::lock_guard<std::mutex> lk(g_tier2_address_dump_mu);
    init_tier2_address_dump_locked();
    return g_tier2_address_dump_enabled;
}

// Returns true when METRIC_EXTRACTOR_TIER2_PER_LANE=1 is set at
// process startup. Independent of `tier2_address_dump_enabled()` --
// the per-lane writer is meaningful only when the address-dump env
// gate is also on (it extends the compact-record sidecar), but the
// two env vars are read independently so the runtime can detect a
// misconfiguration (per_lane=1 with address_dump=0 emits nothing).
static bool tier2_per_lane_enabled() {
    std::lock_guard<std::mutex> lk(g_tier2_address_dump_mu);
    init_tier2_address_dump_locked();
    return g_tier2_per_lane_enabled;
}

// Round 17 / Codex Round-16 review Finding 2: copy the parsed
// METRIC_EXTRACTOR_TIER2_BINARY_SHA256_HEX bytes into the caller's
// 32-byte buffer. Round 18 keeps this as a TEST OVERRIDE path; the
// production path goes through ``tier2_load_binary_sha256_for_origin``
// which hashes the loaded .so on launch-enter.
static void tier2_load_binary_sha256(uint8_t out[32]) {
    std::lock_guard<std::mutex> lk(g_tier2_address_dump_mu);
    init_tier2_address_dump_locked();
    memcpy(out, g_tier2_binary_sha256, 32);
}

static bool tier2_binary_sha256_env_explicitly_set() {
    std::lock_guard<std::mutex> lk(g_tier2_address_dump_mu);
    init_tier2_address_dump_locked();
    return g_tier2_binary_sha256_env_explicitly_set;
}

// Round 18 (Codex Round-17 review Finding 3): per-origin SHA-256 cache
// for the profiled-binary identity digest. The launch-enter hook resolves
// the accepted module origin path (the .so / executable containing the
// kernel cubin), the producer hashes it once per origin path, and the
// cached 32-byte digest is attached to LaunchContext on every subsequent
// launch from the same origin.
static std::mutex g_tier2_binary_sha256_cache_mu;
static std::unordered_map<std::string, std::array<uint8_t, 32>>
    g_tier2_binary_sha256_cache;
static std::unordered_map<std::string, bool>
    g_tier2_binary_sha256_unresolved_warned;

static bool tier2_compute_binary_sha256_of_file(
    const std::string& path, uint8_t out[32]) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    tier2::Sha256Ctx ctx;
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        ctx.Update(buf, n);
    }
    const bool read_error = (ferror(f) != 0);
    fclose(f);
    if (read_error) {
        return false;
    }
    ctx.Finalize(out);
    return true;
}

// Look up the profiled-binary digest for ``origin``. Cached after the
// first call. On filesystem failure (file removed, no read permission,
// unrecoverable I/O), zeroes the buffer, logs ``tier2_binary_sha256_unresolved``
// exactly once per origin path, and returns false. The launch-enter hook
// stores the bool on ``LaunchContext.binary_sha256_valid`` so downstream
// consumers can distinguish "real digest" from "fallback zero".
static bool tier2_load_binary_sha256_for_origin(
    const std::string& origin, uint8_t out[32]) {
    {
        std::lock_guard<std::mutex> lk(g_tier2_binary_sha256_cache_mu);
        auto it = g_tier2_binary_sha256_cache.find(origin);
        if (it != g_tier2_binary_sha256_cache.end()) {
            memcpy(out, it->second.data(), 32);
            return true;
        }
    }
    std::array<uint8_t, 32> digest;
    const bool ok =
        tier2_compute_binary_sha256_of_file(origin, digest.data());
    if (ok) {
        std::lock_guard<std::mutex> lk(g_tier2_binary_sha256_cache_mu);
        g_tier2_binary_sha256_cache[origin] = digest;
        memcpy(out, digest.data(), 32);
        return true;
    }
    // Failure: log once per origin, fall back to zero.
    bool already_warned;
    {
        std::lock_guard<std::mutex> lk(g_tier2_binary_sha256_cache_mu);
        auto it = g_tier2_binary_sha256_unresolved_warned.find(origin);
        already_warned = (it != g_tier2_binary_sha256_unresolved_warned.end());
        g_tier2_binary_sha256_unresolved_warned[origin] = true;
    }
    if (!already_warned) {
        fprintf(stderr,
                "[metric_extractor] WARNING: "
                "tier2_binary_sha256_unresolved: cannot hash origin %s; "
                "falling back to 32-zero digest. LaunchHeaderV1.binary_sha256 "
                "will be zero for launches from this origin.\n",
                origin.c_str());
    }
    memset(out, 0, 32);
    return false;
}

// --------------------------------------------------------------------------
// Filter stats — accumulated across the process lifetime; written to
// METRIC_EXTRACTOR_FILTER_STATS_PATH (default ./metric_extractor_filter_stats.json)
// in nvbit_at_term().
// --------------------------------------------------------------------------

static std::atomic<uint64_t> g_accepted_launches{0};
static std::atomic<uint64_t> g_skipped_launches{0};
static std::atomic<uint64_t> g_module_origin_misses{0};
static std::atomic<uint64_t> g_cooperative_launches{0};
static std::atomic<uint64_t> g_graph_node_launches{0};
// Split graph-node total into accepted vs skipped so a later JSON-L /
// manifest emitter can serialise per-node `traced` vs `not_observed`
// substatus without losing the original total.
static std::atomic<uint64_t> g_graph_node_accepted{0};
static std::atomic<uint64_t> g_graph_node_skipped{0};
// Count of launches that passed the origin / accept filters but landed
// AFTER the slot pool was fully consumed. These launches run
// uninstrumented and emit a bounded JSON-L record carrying
// ``substatus.skipped = true`` and
// ``waiver_reason = per_process_budget_exceeded``. The parser surfaces
// the count into the manifest ``substatus`` so the operator sees the
// over-budget cohort explicitly instead of silently shrinking the
// instrumented kernel set.
static std::atomic<uint64_t> g_per_process_budget_skipped{0};
// graph_launches_seen / graph_launches_not_observed / unknown_opcode_count
// were previously declared further down the source. Move them up here
// so write_filter_stats_json() (defined later but referenced at
// nvbit_at_term) can snapshot+merge them alongside the rest of the
// filter-stats counter set. Codex's Round-7 review caught that the
// terminal merge writer was OMITTING these three even though the
// per-record filter_stats_snapshot emits them; the Round-8 fix
// closes the coverage gap. The original declaration site (next to
// the JSON-L writer's filter_stats_snapshot helper) is removed.
static std::atomic<uint64_t> g_graph_launches_seen{0};
static std::atomic<uint64_t> g_graph_launches_not_observed{0};
static std::atomic<uint64_t> g_unknown_opcode_count{0};

static std::mutex                g_skipped_paths_mu;
static std::vector<std::string>  g_representative_skipped_paths;
static std::unordered_set<std::string> g_skipped_paths_seen;
static const size_t              MAX_REPRESENTATIVE_SKIPPED_PATHS = 10;

static void record_skipped_path(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_skipped_paths_mu);
    if (!g_skipped_paths_seen.insert(path).second) return;
    if (g_representative_skipped_paths.size() >=
        MAX_REPRESENTATIVE_SKIPPED_PATHS) {
        return;
    }
    g_representative_skipped_paths.push_back(path);
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

#define ME_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr,                                                    \
                    "[metric_extractor] CUDA error at %s:%d: %s (%d)\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(_err), (int)_err);  \
        }                                                                      \
    } while (0)

// Resolve the per-process slot-pool budget cap. Defaults to
// METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES (compile-time constant);
// honors the METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES_ENV env var
// when set to a positive integer (full-string parse) so an operator
// running a Tier-2 capture pipeline can lift the cap above the
// default when the per-slot Tier-2 footprint × the number of
// launches in the targeted workload exceeds the default 512 MiB.
// Rejected env-var values (non-numeric, out-of-range, zero) fall
// back to the compile-time default with a stderr warning so the
// misconfiguration is visible.
static size_t resolve_per_process_budget_bytes() {
    const char* env =
        getenv("METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES");
    if (env == nullptr || env[0] == '\0') {
        return (size_t)METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES;
    }
    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = strtoull(env, &end, 10);
    const bool full_string_consumed =
        (end != nullptr && *end == '\0');
    const bool in_range = (errno != ERANGE) && (parsed > 0ULL);
    if (full_string_consumed && in_range) {
        return (size_t)parsed;
    }
    fprintf(stderr,
            "[metric_extractor] METRIC_EXTRACTOR_PER_PROCESS_BUDGET_"
            "BYTES=%s rejected (need full-string positive integer); "
            "falling back to compile-time default %llu bytes\n",
            env,
            (unsigned long long)
                METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES);
    return (size_t)METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES;
}

static uint32_t resolve_max_launch_slots() {
    const char* env = getenv(METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS_ENV);
    if (env == nullptr || env[0] == '\0') {
        return (uint32_t)METRIC_EXTRACTOR_DEFAULT_MAX_LAUNCH_SLOTS;
    }
    char* end = nullptr;
    errno = 0;
    long parsed = strtol(env, &end, 10);
    const bool full_string_consumed = (end != nullptr && *end == '\0');
    const bool in_range =
        (errno != ERANGE) && (parsed >= 1) && (parsed <= (long)UINT32_MAX);
    if (full_string_consumed && in_range) {
        return (uint32_t)parsed;
    }
    fprintf(stderr,
            "[metric_extractor] %s=%s rejected (need full-string positive "
            "integer in [1, %u]); falling back to default %u\n",
            METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS_ENV,
            env,
            (unsigned)UINT32_MAX,
            (unsigned)METRIC_EXTRACTOR_DEFAULT_MAX_LAUNCH_SLOTS);
    return (uint32_t)METRIC_EXTRACTOR_DEFAULT_MAX_LAUNCH_SLOTS;
}

static std::string host_addr_to_so_path(const void* host_ptr) {
    if (host_ptr == nullptr) return std::string();
    std::ifstream maps("/proc/self/maps");
    if (!maps.good()) return std::string();
    auto r = nvbit_tracer::lookup_addr_in_maps_stream(host_ptr, maps);
    return r.path;
}

// Per-launch lookup: try the (ctx, func) registry first; fall back to
// the (ctx, mangled-name) registry for kernels whose CUfunction handle
// the host application uses differs from the handle the registry was
// populated against (NVBit-instrumented variant handles routinely
// differ from CUDA RUNTIME's). Caller MUST hold g_origin_mu.
static std::string lookup_func_origin_locked(CUcontext ctx, CUfunction func,
                                             const std::string& mangled_name) {
    auto it = g_func_origin.find(FuncKey{ctx, func});
    if (it != g_func_origin.end()) return it->second;
    if (!mangled_name.empty()) {
        auto nit =
            g_func_name_origin.find(FuncNameKey{ctx, mangled_name});
        if (nit != g_func_name_origin.end()) return nit->second;
    }
    return std::string();
}

// Register `origin` against `f` plus every related function NVBit can
// reach from `f`, keying both by (ctx, func) and by (ctx, mangled-name).
// Caller MUST hold g_origin_mu.
static void register_func_and_related_origin_locked(
    CUcontext ctx, CUfunction f, const std::string& origin) {
    if (f == nullptr) return;
    g_func_origin[FuncKey{ctx, f}] = origin;
    const char* name = nvbit_get_func_name(ctx, f, /*mangled=*/true);
    if (name != nullptr && name[0] != '\0') {
        g_func_name_origin[FuncNameKey{ctx, std::string(name)}] = origin;
    }
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, f);
    for (CUfunction rf : related) {
        if (rf == nullptr) continue;
        g_func_origin[FuncKey{ctx, rf}] = origin;
        const char* rname =
            nvbit_get_func_name(ctx, rf, /*mangled=*/true);
        if (rname != nullptr && rname[0] != '\0') {
            g_func_name_origin[FuncNameKey{ctx, std::string(rname)}] =
                origin;
        }
    }
}

// Enumerate every CUfunction in a freshly-loaded CUmodule and seed
// g_func_origin / g_func_name_origin with the module's origin. This is
// the path the per-launch decision uses when the host application
// never resolves a function via cuModuleGetFunction (some torch /
// cuLibrary paths).
//
// Caller MUST hold g_origin_mu. The CUDA driver APIs called here
// (cuModuleGetFunctionCount + cuModuleEnumerateFunctions) are
// thread-safe per CUDA's documented driver-API contract; they do not
// re-enter g_origin_mu.
static void populate_module_func_origins_locked(CUcontext ctx,
                                                CUmodule module) {
    if (module == nullptr) return;
    auto orig_it = g_module_origin.find(ModuleKey{ctx, module});
    if (orig_it == g_module_origin.end()) return;
    const std::string origin = orig_it->second;

    unsigned int count = 0;
    if (cuModuleGetFunctionCount(&count, module) != CUDA_SUCCESS ||
        count == 0) {
        return;
    }
    std::vector<CUfunction> funcs(count, nullptr);
    if (cuModuleEnumerateFunctions(funcs.data(), count, module) !=
        CUDA_SUCCESS) {
        return;
    }
    for (CUfunction f : funcs) {
        register_func_and_related_origin_locked(ctx, f, origin);
    }
}

// Enumerate every CUkernel in a freshly-loaded CUlibrary, resolve each
// to its CUfunction, and seed g_func_origin / g_func_name_origin. The
// CUDA RUNTIME's lazy-load path (cuLibraryLoadData) does NOT make
// cuKernelGetFunction calls visible at the driver-callback layer for
// every kernel, so this manual enumeration is the only way to bridge
// library-resident kernels to the per-launch decision.
//
// Caller MUST hold g_origin_mu.
static void populate_library_func_origins_locked(CUcontext ctx,
                                                  CUlibrary library,
                                                  const std::string& origin) {
    if (library == nullptr) return;
    unsigned int count = 0;
    if (cuLibraryGetKernelCount(&count, library) != CUDA_SUCCESS ||
        count == 0) {
        return;
    }
    std::vector<CUkernel> kernels(count, nullptr);
    if (cuLibraryEnumerateKernels(kernels.data(), count, library) !=
        CUDA_SUCCESS) {
        return;
    }
    for (CUkernel k : kernels) {
        if (k == nullptr) continue;
        CUfunction f = nullptr;
        if (cuKernelGetFunction(&f, k) != CUDA_SUCCESS || f == nullptr) {
            continue;
        }
        register_func_and_related_origin_locked(ctx, f, origin);
    }
}

// --------------------------------------------------------------------------
// Instruction-class classifier.
//
// Maps a SASS instruction mnemonic to one of the JSONL_SCHEMA.md sparse-map
// buckets (FFMA / MUFU / LDG / STG / LDS / STS / HMMA / WGMMA / TCGEN05 /
// TMA / DSMEM / BRA / BARRIER / WARPSYNC / UNKNOWN). The classifier
// disambiguates buckets that share the same operation-type category in
// isa_def/operation_type.h (e.g. HMMA and the Hopper warp-group GMMAs both
// land in TENSOR_CORE_OP, but the histogram keeps them separate per AC-7)
// by checking the specific TraceInstrOpcode id first.
//
// The mnemonic lookup is two-stage: the full mnemonic is tried first so
// multi-token Blackwell entries like "TCGEN05.MMA.KIND::MXF4" match by exact
// string, then a token-0 fallback (split on '.') handles the conventional
// per-opcode mnemonics. The Blackwell map is consulted first; the Ampere
// map is the fallback so sm_86 calibration runs (which may emit opcodes
// only present in the older catalog) still classify cleanly.
// --------------------------------------------------------------------------

static InstructionClass classify_instr(Instr* instr) {
    if (instr == nullptr) return INSTR_CLASS_UNKNOWN;
    // Memory-space-side classification runs first. An instruction whose
    // runtime memory space is DISTRIBUTED_SHARED is a DSMEM op
    // regardless of how its mnemonic looks — the same SASS LDS/STS
    // spelling can target either plain shared memory (lds/sts bucket) or
    // distributed shared memory (dsmem bucket) depending on the
    // instruction's memory-space attribute, and the bucket-naming
    // contract pins it to dsmem in the DISTRIBUTED_SHARED case. The
    // helper returns the sentinel INSTR_CLASS_UNKNOWN for every other
    // memory space; in that case we fall through to mnemonic routing.
    const InstructionClass ms_class =
        classify_memory_space(instr->getMemorySpace());
    if (ms_class != INSTR_CLASS_UNKNOWN) {
        return ms_class;
    }
    const char* opc_c = instr->getOpcode();
    if (opc_c == nullptr || *opc_c == '\0') return INSTR_CLASS_UNKNOWN;
    // Use the shared mnemonic lookup and opcode/category router.
    return classify_mnemonic(std::string(opc_c));
}

// Bypass-marker for instrumentation passes that must skip DSMEM ops.
// The 32-bank shared-memory bank-conflict model is designed for plain
// shared memory (`LDS` / `STS` targeting `MemorySpace::SHARED`); DSMEM
// ops use distinct banking semantics and would produce meaningless
// counts if processed by the same model. Any future Tier-1 callback
// installer that walks shared-memory instructions MUST consult this
// helper and skip when it returns true; the bucket routing in
// `classify_instr` already places DSMEM in `INSTR_CLASS_DSMEM`, so the
// runtime instrumentation just needs to avoid the shared-bank callback
// for these instructions.
static bool is_distributed_shared(Instr* instr) {
    if (instr == nullptr) return false;
    return instr->getMemorySpace() ==
           InstrType::MemorySpace::DISTRIBUTED_SHARED;
}

// Returns the schema address_space code for an atomic SASS opcode, or
// 255 (unknown sentinel) for non-atomics. The recognized atomic
// family across Ampere / Hopper / Blackwell ISA maps:
//   ATOM   -> global  (legacy short opcode, returns value)
//   ATOMG  -> global  (explicit global atomic, returns value)
//   ATOMS  -> shared  (explicit shared atomic, returns value)
//   RED    -> global  (legacy reduction, no return)
//   REDG   -> global  (Hopper+ explicit global reduction)
//   REDAS  -> shared  (Hopper async shared reduction)
// NVBit's `getOpcodeShort()` returns the first dot-delimited token of
// the SASS mnemonic (e.g. `RED.E.ADD.STRONG.GPU` -> `RED`,
// `REDG.E.ADD.STRONG.GPU` -> `REDG`), so an exact-strcmp set covers
// the entire family. Surface-image atomics (`SUATOM`, `SURED`) are
// deliberately excluded because they target surface memory, which
// the predictor does not model. TMA-class atomics (`UTMAREDG`) flow
// through the dedicated TMA path; their classification is handled
// by `instr->isTMAMem()` upstream of this helper.
static uint32_t atomic_address_space(Instr* instr) {
    if (instr == nullptr) return 255u;
    const char* opc = instr->getOpcodeShort();
    if (opc == nullptr || *opc == '\0') return 255u;
    if (strcmp(opc, "ATOMS") == 0) return 2u;
    if (strcmp(opc, "ATOMG") == 0) return 1u;
    if (strcmp(opc, "ATOM") == 0) return 1u;
    if (strcmp(opc, "RED") == 0) return 1u;
    if (strcmp(opc, "REDG") == 0) return 1u;
    if (strcmp(opc, "REDAS") == 0) return 2u;
    return 255u;
}

static bool is_atomic_instruction(Instr* instr) {
    return atomic_address_space(instr) != 255u;
}

// Returns the schema-v1.0 `op_kind` enum for an atomic SASS opcode:
//   3 = atomic_load   — read-only atomic. NO opcode in the ATOM /
//                       RED family produces this on Ampere /
//                       Hopper / Blackwell; PTX `atom.load` lowers
//                       to `LDG.E.STRONG.GPU` / `LDS.STRONG`, which
//                       this code path does not see (those flow
//                       through the regular global/shared installs
//                       with op_kind=1). Reserved for a future
//                       strong-ordered-load classifier; emitted by
//                       no producer code today.
//   4 = atomic_store  — write-only atomic. The RED family is the
//                       schema match: RED / REDG (Hopper+ explicit
//                       global reduction) / REDAS (Hopper async
//                       shared reduction) — atomic reductions
//                       without a return value (the result is not
//                       communicated to the issuer, so the
//                       memory-traffic shape is store-only with an
//                       atomic ordering guarantee).
//   5 = atomic_rmw    — read-modify-write atomic. The ATOM family:
//                       ATOM (legacy) / ATOMG (explicit global) /
//                       ATOMS (explicit shared). These return the
//                       prior memory value to the issuer; the
//                       memory-traffic shape is load+store.
// Returns 5 (atomic_rmw) as the conservative default for any
// atomic opcode not yet enumerated above, so a future family
// member is encoded as RMW (the most general memory-traffic
// model) until a classifier extension lands.
static uint32_t tier2_atomic_op_kind(Instr* instr) {
    if (instr == nullptr) return 5u;
    const char* opc = instr->getOpcodeShort();
    if (opc == nullptr || *opc == '\0') return 5u;
    if (strcmp(opc, "ATOM") == 0)  return 5u;
    if (strcmp(opc, "ATOMG") == 0) return 5u;
    if (strcmp(opc, "ATOMS") == 0) return 5u;
    if (strcmp(opc, "RED") == 0)   return 4u;
    if (strcmp(opc, "REDG") == 0)  return 4u;
    if (strcmp(opc, "REDAS") == 0) return 4u;
    return 5u;
}

// --------------------------------------------------------------------------
// Instrumentation pass — walks the kernel's SASS instructions and inserts
// the device-side count_metric callback before each. Tracked per-CUfunction
// so a function instrumented once stays instrumented across subsequent
// launches.
//
// Per-instruction dispatch order in instrument_function_if_needed:
//   1. TMA ops (`instr->isTMAMem()`): routed via the
//      `sample_tma_descriptor` callback that ALSO captures the descriptor
//      handle for host-side decoding on launch-exit. The class id pushed
//      to count_metric is `INSTR_CLASS_TMA` regardless of the classifier
//      output.
//   2. Everything else: routed via `count_metric` with `classify_instr`
//      output. `classify_instr` checks `instr->getMemorySpace()` first
//      via `classify_memory_space()`, so DSMEM ops automatically land in
//      `INSTR_CLASS_DSMEM` without a separate top-level branch here.
//
// Bypass-marker contract for DSMEM: when a future Tier-1 callback
// installer (e.g. T11 32-bank shared-memory bank-conflict tally) walks
// shared-memory instructions, it MUST call `is_distributed_shared(instr)`
// and skip when it returns true. The bucket routing for histogram
// counting is independent of the bank-conflict callback; both must
// honour the same marker to satisfy the AC-7 negative-test invariant
// that "DSMEM ops are not processed by the 32-bank shared model".
//
// The instrumentation arguments are:
//   1. predicate guard via nvbit_add_call_arg_guard_pred_val(): so the
//      device function can branch on per-thread predicate state without
//      needing the per-warp ballot at this stage.
//   2. instruction-class id (uint32) derived per-instruction by
//      classify_instr() against the Blackwell / Ampere ISA maps. Out-of-
//      range values are clamped to INSTR_CLASS_UNKNOWN by count_metric.
//   3. per-launch slot pointer (uint64) loaded from the value the host
//      pushed via nvbit_set_at_launch(): the device-side aggregator
//      atomicAdds into this slot.
// --------------------------------------------------------------------------

// Keyed on (CUcontext, CUfunction) so two contexts that reuse the same
// CUfunction handle do not cause the second context's instrumentation
// pass to be silently skipped. Reusing the existing FuncKey + FuncKeyHash
// keeps the discipline consistent with g_func_origin / g_func_module.
static std::unordered_set<FuncKey, FuncKeyHash> g_instrumented_funcs;

// Per-process SASS mnemonic ↔ opcode-id mapping. Populated lazily during
// the instrumentation pass so the TMA descriptor sample slot stores a
// compact uint32 instead of a string; on launch-exit the host re-resolves
// the mnemonic via `g_id_to_opcode[opcode_id]` and feeds it to
// `nvbit_parse_tma_transfer_info`. Mirrors the upstream `mem_trace_tma.cu`
// opcode_to_id_map / id_to_opcode_map pair. Lives behind its own mutex —
// the SASS walk does not hold any other lock when it allocates an id.
static std::mutex                                g_opcode_id_mu;
static std::unordered_map<std::string, uint32_t> g_opcode_id_map;
static std::vector<std::string>                  g_id_to_opcode;

static uint32_t resolve_opcode_id(const std::string& mnemonic) {
    std::lock_guard<std::mutex> lk(g_opcode_id_mu);
    auto it = g_opcode_id_map.find(mnemonic);
    if (it != g_opcode_id_map.end()) return it->second;
    const uint32_t id = static_cast<uint32_t>(g_id_to_opcode.size());
    g_opcode_id_map[mnemonic] = id;
    g_id_to_opcode.push_back(mnemonic);
    return id;
}

static std::string lookup_opcode_by_id(uint32_t id) {
    std::lock_guard<std::mutex> lk(g_opcode_id_mu);
    if (id >= g_id_to_opcode.size()) return std::string();
    return g_id_to_opcode[id];
}

static void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    // Walk the kernel itself plus its related (callee) functions so any
    // device-side helpers reachable from the kernel are also covered.
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!g_instrumented_funcs.insert(FuncKey{ctx, f}).second) continue;

        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (Instr* instr : instrs) {
            // TMA instructions get a dedicated callback that captures the
            // descriptor handle into the per-launch slot for host-side
            // decode via nvbit_parse_tma_transfer_info. Non-TMA
            // instructions keep the original count_metric callback. Both
            // paths push the predicate guard, instruction-class id, and
            // per-launch slot pointer in the same order; the TMA path
            // appends opcode id + TMA descriptor handle/size args.
            const bool is_tma = instr->isTMAMem();
            const bool is_atomic = is_atomic_instruction(instr);
            const InstructionClass cls =
                is_tma ? INSTR_CLASS_TMA : classify_instr(instr);
            if (is_tma) {
                const char* opc_c = instr->getOpcode();
                const std::string mnemonic =
                    (opc_c != nullptr) ? std::string(opc_c) : std::string();
                const uint32_t opcode_id = resolve_opcode_id(mnemonic);
                nvbit_insert_call(instr, "sample_tma_descriptor",
                                  IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr,
                                               static_cast<uint32_t>(cls));
                nvbit_add_call_arg_const_val32(instr, opcode_id);
                nvbit_add_call_arg_launch_val64(instr, 0);
                nvbit_add_call_arg_tma_param_handle_and_size(instr, ctx);
                // Tier-2 sector emission for TMA bulk / tensor
                // transfers MUST NOT go through count_tier2_global_
                // mem_access with nvbit_add_call_arg_mref_addr64.
                // TMA instructions carry a descriptor handle, not a
                // single MREF address; per-element global source /
                // destination addresses are recovered host-side via
                // nvbit_parse_tma_src_addrs / _dst_addrs against the
                // descriptor captured by sample_tma_descriptor
                // above. The full TMA Tier-2 emission path runs at
                // launch-exit and is deferred to a follow-up round.
                // The sample_tma_descriptor Tier-1 install above is
                // unconditional and preserves AC-13 byte-identity.
            } else {
                nvbit_insert_call(instr, "count_metric", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr,
                                               static_cast<uint32_t>(cls));
                nvbit_add_call_arg_launch_val64(instr, 0);
            }
            // Additional callback for global LDG / STG: per-warp distinct
            // 32B sector counting feeds the sector-efficiency aggregator.
            // DSMEM is excluded by the `== GLOBAL` filter (DSMEM ops have
            // memory space `DISTRIBUTED_SHARED`, which is the marker T11's
            // bank-conflict callback also uses to skip them via the
            // positive `== SHARED` filter below).
            // The instrument-class histogram still counts LDG / STG via
            // the count_metric path above; this callback layers an
            // address-aware aggregation on top, so the same instruction
            // contributes to both the histogram and the sector counters.
            if (!is_tma &&
                instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL &&
                (instr->isLoad() || instr->isStore())) {
                const int bytes_per_access = instr->getSize();
                const uint32_t is_store =
                    instr->isStore() ? 1u : 0u;
                // Tier-1 sector-efficiency callback. Always installed
                // (unconditional on the env gate). The aggregator
                // counts per-warp distinct 32 B sectors against the
                // useful-byte numerator to derive sector_efficiency_
                // global_{load,store}. The Tier-2 address-dump
                // callback below is installed in addition when the
                // env gate is on; the unconditional install here is
                // what guarantees Tier-1 byte identity at env=0.
                nvbit_insert_call(instr, "count_global_mem_access",
                                  IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, is_store);
                nvbit_add_call_arg_const_val32(
                    instr, static_cast<uint32_t>(bytes_per_access));
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_launch_val64(instr, 0);
                // Tier-2 address-dump callback. Installed only when
                // the env gate is on. With the gate off the call
                // below is not emitted; the instrumented function
                // carries identical device-side work to the no-
                // Tier-2 build and Tier-1 output is byte-identical by
                // construction. With the gate on, the companion
                // device function ``count_tier2_global_mem_access``
                // in ``inject_funcs.cu`` captures one
                // ``CompactAccessRecord`` per warp-collective access
                // into the per-launch ``TierTwoBuffer`` reachable
                // through ``LaunchMetrics::tier2_buffer_ptr``. The
                // const-val arg ordering matches the callback's
                // signature: op_kind, address_space, cache_modifier,
                // bytes_per_access, instruction_id.
                if (tier2_address_dump_enabled() && !is_atomic) {
                    // Atomic-class globals (ATOM / ATOMG / RED) flow
                    // through the dedicated atomic Tier-2 install below
                    // with op_kind=5; gating here keeps the same
                    // instruction from producing two Tier-2 records.
                    // op_kind: 1 = load, 2 = store (matches the schema's
                    // CompactAccessRecordV1 op_kind enum). The default-or-
                    // unknown bucket is 0; not used at this site because
                    // the surrounding filter guarantees the instruction is
                    // a load or a store.
                    const uint32_t tier2_op_kind = is_store ? 2u : 1u;
                    // Address space: 1 = global. The schema enum admits
                    // shared / local / dsmem in future producer-coverage
                    // expansions; this hook is global-only by filter.
                    const uint32_t tier2_address_space = 1u;
                    // Cache modifier extracted from the SASS opcode
                    // string. NVBit exposes the opcode as a string
                    // (e.g. ``LDG.E.CG``) via ``Instr::getOpcode()``;
                    // the suffix tokens encode the modifier. Mapping follows
                    // CompactAccessRecordV1's cache_modifier enum in
                    // dynamic_trace/trace.proto:
                    // 0 = default_or_none, 1 = ca, 2 = cg, 3 = cs,
                    // 4 = cv, 5 = lu, 255 = unknown. The default
                    // discipline for an opcode with no recognized
                    // modifier suffix is 0 (default_or_none); an
                    // opcode whose modifier byte is present but not
                    // in the recognized set maps to 255 (unknown).
                    uint32_t tier2_cache_modifier = 0u;
                    const std::string opcode_str = instr->getOpcode();
                    if (opcode_str.find(".CG") != std::string::npos) {
                        tier2_cache_modifier = 2u;
                    } else if (opcode_str.find(".CA") != std::string::npos) {
                        tier2_cache_modifier = 1u;
                    } else if (opcode_str.find(".CS") != std::string::npos) {
                        tier2_cache_modifier = 3u;
                    } else if (opcode_str.find(".CV") != std::string::npos) {
                        tier2_cache_modifier = 4u;
                    } else if (opcode_str.find(".LU") != std::string::npos) {
                        tier2_cache_modifier = 5u;
                    }
                    // L1 access policy classification. The legacy
                    // ``cache_modifier`` byte is unreliable on
                    // Ampere sm_86 because ptxas folds ``.cg`` into
                    // the strong-ordered SASS form
                    // ``LDG.E.STRONG.GPU`` without an explicit
                    // ``.CG`` suffix — so ``cache_modifier=0``
                    // conflates default policy with folded
                    // ``.cg``. The new ``l1_access_policy`` enum
                    // disambiguates: explicit ``.CG`` AND the
                    // ``STRONG.GPU`` Ampere fold both stamp
                    // ``l1_bypassed`` (value 2); any other global
                    // load with a recognized opcode stamps
                    // ``l1_admitted`` (value 1). Atomic /
                    // cp.async / TMA / spill paths stamp
                    // ``l1_unknown`` (value 0) — their L1 policy
                    // is governed by ``op_kind``, not by the
                    // global-load policy field. Routing-layer
                    // consumers consult this field ahead of the
                    // legacy ``cache_modifier`` byte.
                    uint32_t tier2_l1_access_policy = 1u;
                    if (opcode_str.find(".CG") != std::string::npos
                        || opcode_str.find("STRONG.GPU")
                           != std::string::npos) {
                        tier2_l1_access_policy = 2u;
                    }
                    // Instruction id: NVBit's per-function instruction
                    // ordinal (Instr::getIdx). Stable across runs of
                    // the same SASS because NVBit walks instructions
                    // in a deterministic order.
                    const uint32_t tier2_instruction_id =
                        static_cast<uint32_t>(instr->getIdx());
                    nvbit_insert_call(instr,
                                      "count_tier2_global_mem_access",
                                      IPOINT_BEFORE);
                    nvbit_add_call_arg_guard_pred_val(instr);
                    nvbit_add_call_arg_const_val32(instr, tier2_op_kind);
                    nvbit_add_call_arg_const_val32(instr,
                                                   tier2_address_space);
                    nvbit_add_call_arg_const_val32(instr,
                                                   tier2_cache_modifier);
                    nvbit_add_call_arg_const_val32(
                        instr, static_cast<uint32_t>(bytes_per_access));
                    nvbit_add_call_arg_const_val32(instr,
                                                   tier2_instruction_id);
                    nvbit_add_call_arg_const_val32(
                        instr, tier2_l1_access_policy);
                    nvbit_add_call_arg_mref_addr64(instr, 0);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                }
            }
            // Additional callback for plain SHARED LDS / STS: per-warp
            // 32-bank conflict tally feeds the bank-conflict counter.
            // DSMEM (DISTRIBUTED_SHARED) is excluded by the positive
            // `== SHARED` filter; the AC-7 negative-test invariant
            // "DSMEM ops are not processed by the 32-bank shared model"
            // holds by construction. The is_distributed_shared(Instr*)
            // bypass marker from Round 4 is the documented contract for
            // any future shared-memory callback installer to honour;
            // here, the positive selection makes the marker redundant
            // but the contract stays documented.
            if (!is_tma &&
                instr->getMemorySpace() == InstrType::MemorySpace::SHARED &&
                (instr->isLoad() || instr->isStore())) {
                nvbit_insert_call(instr, "count_shared_bank_conflicts_cb",
                                  IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr, 0);
                nvbit_add_call_arg_launch_val64(instr, 0);
                // Tier-2 address-dump callback for shared LDS / STS.
                // Installed only when the env gate is on. Reuses the
                // address-space-agnostic count_tier2_global_mem_access
                // device function with address_space=2 (shared). The
                // shared memory space has no cache-modifier suffix in
                // the SASS opcode (the per-bank model handles policy);
                // cache_modifier is fixed at 0 (default_or_none).
                // Atomic-class shared instructions (ATOMS) flow
                // through the dedicated atomic install below with
                // op_kind=5; gating here prevents a double record.
                if (tier2_address_dump_enabled() && !is_atomic) {
                    const int bytes_per_access_shared = instr->getSize();
                    const uint32_t tier2_op_kind_shared =
                        instr->isStore() ? 2u : 1u;
                    const uint32_t tier2_instruction_id_shared =
                        static_cast<uint32_t>(instr->getIdx());
                    nvbit_insert_call(instr,
                                      "count_tier2_global_mem_access",
                                      IPOINT_BEFORE);
                    nvbit_add_call_arg_guard_pred_val(instr);
                    nvbit_add_call_arg_const_val32(instr,
                                                   tier2_op_kind_shared);
                    nvbit_add_call_arg_const_val32(instr, 2u);
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_const_val32(
                        instr,
                        static_cast<uint32_t>(bytes_per_access_shared));
                    nvbit_add_call_arg_const_val32(
                        instr, tier2_instruction_id_shared);
                    // l1_access_policy = l1_unknown (shared
                    // memory is not L1-cached on Ampere; the
                    // shared-bank model handles policy).
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_mref_addr64(instr, 0);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                }
            }
            // Tier-2 address-dump callback for plain LOCAL LDL / STL.
            // Local memory carries per-thread spill traffic plus the
            // PTX `.local` allocation path. The metric_extractor has
            // no unconditional Tier-1 local counter (the local-memory
            // class is dominated by spills, which the SASS instruction
            // histogram already records via the LDL / STL buckets);
            // this block records ordered 32 B sector sets for each
            // LDL / STL the device executes when the env gate is on.
            // When the env gate is off no callback is installed, so
            // Tier-1 byte-identity holds by construction.
            // address_space=3 (local) and cache_modifier=0 (local
            // opcodes carry no `.cg`-style suffix in SASS; the
            // `.LU` modifier is observed on GLOBAL not LOCAL).
            if (!is_tma && !is_atomic &&
                instr->getMemorySpace() == InstrType::MemorySpace::LOCAL &&
                (instr->isLoad() || instr->isStore())) {
                if (tier2_address_dump_enabled()) {
                    const int bytes_per_access_local = instr->getSize();
                    const uint32_t tier2_op_kind_local =
                        instr->isStore() ? 2u : 1u;
                    const uint32_t tier2_instruction_id_local =
                        static_cast<uint32_t>(instr->getIdx());
                    nvbit_insert_call(instr,
                                      "count_tier2_global_mem_access",
                                      IPOINT_BEFORE);
                    nvbit_add_call_arg_guard_pred_val(instr);
                    nvbit_add_call_arg_const_val32(instr,
                                                   tier2_op_kind_local);
                    nvbit_add_call_arg_const_val32(instr, 3u);
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_const_val32(
                        instr,
                        static_cast<uint32_t>(bytes_per_access_local));
                    nvbit_add_call_arg_const_val32(
                        instr, tier2_instruction_id_local);
                    // l1_access_policy = l1_unknown (local
                    // memory carries spill traffic; the L1
                    // admit/bypass decision is governed by
                    // op_kind, not by the global-load policy
                    // field).
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_mref_addr64(instr, 0);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                }
            }
            // Tier-2 address-dump callback for atomic globals / shared.
            // ATOM / ATOMG / RED target global memory; ATOMS targets
            // shared. The op_kind is dispatched via
            // tier2_atomic_op_kind(instr): ATOM / ATOMG / ATOMS map
            // to 5 (atomic_rmw, returns prior value); RED / REDG /
            // REDAS map to 4 (atomic_store, no return). The Tier-1
            // sector / bank-conflict callbacks above still fire for
            // these instructions because the GLOBAL / SHARED outer
            // filters do not exclude atomics; only the regular
            // Tier-2 install was gated on `!is_atomic`. cache_modifier
            // is 0 (atomic SASS opcodes do not carry .cg-style cache
            // suffixes).
            if (!is_tma && is_atomic &&
                (instr->isLoad() || instr->isStore())) {
                if (tier2_address_dump_enabled()) {
                    const uint32_t atomic_addr_space =
                        atomic_address_space(instr);
                    const uint32_t atomic_op_kind =
                        tier2_atomic_op_kind(instr);
                    const int bytes_per_access_atomic = instr->getSize();
                    const uint32_t tier2_instruction_id_atomic =
                        static_cast<uint32_t>(instr->getIdx());
                    nvbit_insert_call(instr,
                                      "count_tier2_global_mem_access",
                                      IPOINT_BEFORE);
                    nvbit_add_call_arg_guard_pred_val(instr);
                    nvbit_add_call_arg_const_val32(instr, atomic_op_kind);
                    nvbit_add_call_arg_const_val32(instr,
                                                   atomic_addr_space);
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_const_val32(
                        instr,
                        static_cast<uint32_t>(bytes_per_access_atomic));
                    nvbit_add_call_arg_const_val32(
                        instr, tier2_instruction_id_atomic);
                    // l1_access_policy = l1_unknown (atomic
                    // routing is governed by op_kind, not by
                    // the global-load policy field).
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_mref_addr64(instr, 0);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                }
            }
            // Tier-2 address-dump callback for `cp.async` source-side
            // accesses. cp.async reads global memory and writes shared
            // memory; the source-side records use op_kind=6 (cp_async)
            // and address_space=1 (global). NVBit reports cp.async
            // instructions with memory space `GLOBAL_TO_SHARED`.
            // cache_modifier is 0 (cp.async carries an `.E` extender
            // but not a `.cg`-style cache hint at the SASS layer).
            //
            // MREF operand index discipline: LDGSTS carries two
            // memory references in SASS order — `[shared_dst]`
            // first, `[global_src.64]` second. To record GLOBAL
            // source sectors (not shared destination), the address
            // arg uses MREF id=1 (the global source). The runtime
            // test on calib_cpasync_global_to_shared validates this
            // by comparing captured sectors against the kernel's
            // d_in address; a future SASS form that swaps operand
            // ordering breaks the test loudly.
            if (!is_tma && !is_atomic &&
                instr->getMemorySpace() ==
                    InstrType::MemorySpace::GLOBAL_TO_SHARED &&
                (instr->isLoad() || instr->isStore())) {
                if (tier2_address_dump_enabled()) {
                    const int bytes_per_access_cpasync = instr->getSize();
                    const uint32_t tier2_instruction_id_cpasync =
                        static_cast<uint32_t>(instr->getIdx());
                    nvbit_insert_call(instr,
                                      "count_tier2_global_mem_access",
                                      IPOINT_BEFORE);
                    nvbit_add_call_arg_guard_pred_val(instr);
                    nvbit_add_call_arg_const_val32(instr, 6u);
                    nvbit_add_call_arg_const_val32(instr, 1u);
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_const_val32(
                        instr,
                        static_cast<uint32_t>(bytes_per_access_cpasync));
                    nvbit_add_call_arg_const_val32(
                        instr, tier2_instruction_id_cpasync);
                    // l1_access_policy = l1_unknown (cp.async
                    // routing is governed by op_kind=6, not by
                    // the global-load policy field).
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_mref_addr64(instr, 1);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                }
            }
            // Tier-2 address-dump callback for distributed shared
            // memory (DSMEM). DSMEM SASS instructions are emitted only
            // on sm_90+ hardware (B200 sm_100a in particular). On
            // sm_86 this branch is unreachable at runtime but the
            // recognition stub still compiles, so the deployed B200
            // build covers DSMEM by construction. op_kind=1/2,
            // address_space=4 (dsmem), cache_modifier=0.
            if (!is_tma && !is_atomic &&
                is_distributed_shared(instr) &&
                (instr->isLoad() || instr->isStore())) {
                if (tier2_address_dump_enabled()) {
                    const int bytes_per_access_dsmem = instr->getSize();
                    const uint32_t tier2_op_kind_dsmem =
                        instr->isStore() ? 2u : 1u;
                    const uint32_t tier2_instruction_id_dsmem =
                        static_cast<uint32_t>(instr->getIdx());
                    nvbit_insert_call(instr,
                                      "count_tier2_global_mem_access",
                                      IPOINT_BEFORE);
                    nvbit_add_call_arg_guard_pred_val(instr);
                    nvbit_add_call_arg_const_val32(instr,
                                                   tier2_op_kind_dsmem);
                    nvbit_add_call_arg_const_val32(instr, 4u);
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_const_val32(
                        instr,
                        static_cast<uint32_t>(bytes_per_access_dsmem));
                    nvbit_add_call_arg_const_val32(
                        instr, tier2_instruction_id_dsmem);
                    // l1_access_policy = l1_unknown (DSMEM is
                    // governed by op_kind / address_space=4,
                    // not by the global-load policy field).
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_mref_addr64(instr, 0);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                }
            }
            // Tier-2 address-dump callback for constant-memory
            // loads (LDC / LDCU / ULDC). The constant cache sits
            // parallel to L1 between the SM and L2; constant
            // cache misses go to L2 and contribute to the L2
            // read-sector denominator. The producer detects
            // these instructions by SASS opcode prefix because
            // NVBit's ``getMemorySpace()`` does not consistently
            // return ``MemorySpace::CONSTANT`` for uniform LDC
            // variants (ULDC writes to a uniform register; the
            // memory-space classifier may return NONE for the
            // uniform-pipe instruction). Opcode-string detection
            // catches all three SASS forms — `LDC` (per-thread),
            // `LDCU` (per-thread, unconditional?), and `ULDC`
            // (uniform) — regardless of NVBit's memory-space
            // tag. LDC-family opcodes do NOT surface
            // MOV/IMAD-with-CBANK constant operands; those
            // remain unobservable through the NVBit injection
            // surface (the constant operand is encoded inline
            // and the host instruction's primary memory space
            // is NONE/GENERIC). Gated on
            // ``tier2_address_dump_enabled()`` so the Tier-1
            // hot path stays byte-identical with env=0.
            const std::string opcode_str_for_ldc =
                instr->getOpcode();
            const bool is_ldc_family = (
                opcode_str_for_ldc.find("LDC") != std::string::npos
                || opcode_str_for_ldc.find("ULDC") != std::string::npos
            );
            if (!is_tma && !is_atomic && is_ldc_family) {
                if (tier2_address_dump_enabled()) {
                    const int bytes_per_access_constant = instr->getSize();
                    const uint32_t tier2_instruction_id_constant =
                        static_cast<uint32_t>(instr->getIdx());
                    nvbit_insert_call(instr,
                                      "count_tier2_global_mem_access",
                                      IPOINT_BEFORE);
                    nvbit_add_call_arg_guard_pred_val(instr);
                    // op_kind=1 (load), address_space=5
                    // (constant), cache_modifier=0 (LDC has no
                    // .cg-style suffix at the SASS layer),
                    // l1_access_policy=0 (the L1-policy enum is
                    // not meaningful for constant-memory loads
                    // because the constant cache bypasses L1
                    // architecturally).
                    nvbit_add_call_arg_const_val32(instr, 1u);
                    nvbit_add_call_arg_const_val32(instr, 5u);
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    nvbit_add_call_arg_const_val32(
                        instr,
                        static_cast<uint32_t>(bytes_per_access_constant));
                    nvbit_add_call_arg_const_val32(
                        instr, tier2_instruction_id_constant);
                    nvbit_add_call_arg_const_val32(instr, 0u);
                    // LDC-family instructions encode the
                    // constant address as a CBANK operand
                    // rather than a normal MREF; passing the
                    // literal 0 as the address gives every
                    // lane the same address so the sector
                    // derivation collapses to a single
                    // sector_id=0 per LDC instance under the
                    // existing __shfl_sync gather. The audit
                    // diagnostic only cares about sector
                    // COUNT, not sector identity, so this
                    // approximation is acceptable for the
                    // L2 read-denominator coverage metric.
                    nvbit_add_call_arg_const_val64(instr, 0ull);
                    nvbit_add_call_arg_launch_val64(instr, 0);
                }
            }
            // Additional callback for branch-class instructions: per
            // warp-instance, the leader lane computes popcount of the
            // warp-collective predicate mask and atomicAdds 1 into
            // branch_active_lanes_hist[count]. The instrument-class
            // histogram still counts BRA via the count_metric path
            // above; this callback layers a divergence-shape aggregator
            // on top, so the same branch instruction contributes to
            // both the histogram and the branch active-lanes histogram.
            // The classifier's BRA bucket covers BRA / BRX / EXIT /
            // JMP / CALL / RET / etc., so re-using `cls ==
            // INSTR_CLASS_BRA` keeps T7 and T12 anchored to the same
            // bucket set by construction.
            if (!is_tma && cls == INSTR_CLASS_BRA) {
                nvbit_insert_call(instr,
                                  "count_branch_active_mask_cb",
                                  IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_launch_val64(instr, 0);
            }
        }
    }
}

// --------------------------------------------------------------------------
// filter_stats.json writer — invoked from nvbit_at_term(). Hand-rolled
// JSON because the schema is small and the tool has no JSON dep at build
// time. Fields:
//   - schema_version, tool_version (from common.h)
//   - regex_source (the active regex string)
//   - accepted_launches, skipped_launches, module_origin_miss_count
//   - cooperative_launches, graph_node_launches
//   - representative_skipped_paths (capped at 10, deduped)
//   - launch_index (total launches the lifecycle saw)
// --------------------------------------------------------------------------

static const char* DEFAULT_FILTER_STATS_PATH =
    "./metric_extractor_filter_stats.json";

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Helper for the multi-process-safe filter_stats.json writer. Parses
// a non-negative integer counter value out of a previously-written
// JSON-encoded filter_stats body. Returns ``0`` when the field is
// missing or unparseable; the caller takes ``max(parsed, in_process)``
// so an absent / zero / garbage value never overwrites a live
// process's counter state.
static uint64_t parse_existing_filter_stats_uint(
    const std::string& body, const std::string& field_name) {
    // Look for the literal `"field":<digits>` shape the writer emits.
    // We don't need a full JSON parser here — the file's format is
    // entirely under our control and the regex is robust to comma /
    // whitespace.
    const std::string pat = "\"" + field_name + "\":\\s*([0-9]+)";
    std::regex re(pat);
    std::smatch m;
    if (std::regex_search(body, m, re)) {
        try {
            return static_cast<uint64_t>(std::stoull(m[1].str()));
        } catch (...) {
            return 0;
        }
    }
    return 0;
}

// Helper for the multi-process-safe filter_stats.json writer. Parses
// the ``representative_skipped_paths`` array out of a previously-
// written JSON body into a flat vector of strings. Skips empty
// arrays and gracefully tolerates ill-formed input.
static std::vector<std::string> parse_existing_skipped_paths(
    const std::string& body) {
    std::vector<std::string> result;
    std::regex array_re(
        R"DEL("representative_skipped_paths":\s*\[(.*?)\])DEL");
    std::smatch m;
    if (!std::regex_search(body, m, array_re)) return result;
    const std::string inside = m[1].str();
    if (inside.empty()) return result;
    // Walk through the array and pull each `"<path>"` literal. The
    // producer never embeds escaped quotes in these paths (they come
    // from /proc/self/maps), so a non-escaped quote regex is safe.
    std::regex item_re(R"DEL("([^"]*)")DEL");
    auto begin = std::sregex_iterator(inside.begin(), inside.end(), item_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        result.push_back((*it)[1].str());
    }
    return result;
}

// Multi-process-safe filter_stats.json writer.
//
// Codex's Round-5 review showed that the Round-5 writer let a
// zero-counter atexit process overwrite a nonzero file. The cause:
// instrumented eval subprocesses share the same bundle path via the
// ``METRIC_EXTRACTOR_FILTER_STATS_PATH`` env var; whichever process
// exits last overwrites the file unconditionally. The fix is to
// merge: each writer ``flock``s the target file, reads any existing
// content, takes ``max(existing, mine)`` for every integer counter,
// unions ``representative_skipped_paths`` capped at 10, then atomically
// renames a temp file into place.
//
// ``max()`` is the correct merge for monotonic counters — every
// counter in this writer only grows during a process's lifetime, so a
// zero-counter writer can never decrease a non-zero on-disk value
// under the merge.
//
// The flock serialises concurrent writers so the read-modify-write is
// atomic relative to other writers. The atomic rename guarantees that
// a reader never sees a partial write — readers either see the
// previous complete file or the new complete file.
static void write_filter_stats_json() {
    const char* env = getenv("METRIC_EXTRACTOR_FILTER_STATS_PATH");
    const std::string path =
        (env != nullptr && env[0] != '\0') ? env : DEFAULT_FILTER_STATS_PATH;

    // Snapshot in-process state.
    std::string regex_source;
    {
        std::lock_guard<std::mutex> lk(g_regex_mu);
        regex_source = g_user_module_regex_source.empty()
                           ? std::string(DEFAULT_USER_MODULE_REGEX)
                           : g_user_module_regex_source;
    }
    std::vector<std::string> my_skipped_paths;
    {
        std::lock_guard<std::mutex> lk(g_skipped_paths_mu);
        my_skipped_paths = g_representative_skipped_paths;
    }
    const uint64_t my_accepted          = g_accepted_launches.load();
    const uint64_t my_skipped           = g_skipped_launches.load();
    const uint64_t my_origin_misses     = g_module_origin_misses.load();
    const uint64_t my_cooperative       = g_cooperative_launches.load();
    const uint64_t my_graph_nodes       = g_graph_node_launches.load();
    const uint64_t my_graph_accepted    = g_graph_node_accepted.load();
    const uint64_t my_graph_skipped     = g_graph_node_skipped.load();
    const uint64_t my_budget_skipped    = g_per_process_budget_skipped.load();
    const uint64_t my_launch_index      = g_launch_index;
    // Codex's Round-7 review caught that the merge writer was
    // omitting three counters that the per-record snapshot DOES
    // emit: graph_launches_seen, graph_launches_not_observed,
    // and unknown_opcode_count. The terminal file could therefore
    // drift from the in-record state on those three slots even
    // under the lock + merge + atomic-rename discipline. The
    // Round-8 fix is to add them to the snapshot block, the
    // take_max() merge below, and the JSON emit. Every counter
    // the per-record filter_stats_snapshot carries is now covered
    // by the terminal merge.
    const uint64_t my_graph_seen        = g_graph_launches_seen.load();
    const uint64_t my_graph_not_obs     = g_graph_launches_not_observed.load();
    const uint64_t my_unknown_opcode    = g_unknown_opcode_count.load();

    // Acquire an exclusive flock on a STABLE sibling lock file
    // (``<path>.lock``) so the read-modify-write-rename critical
    // section is atomic across concurrent writers — even when one
    // of the writers ``rename()``s the target. Codex's Round-6
    // review caught the previous design where we locked the target
    // path itself: POSIX ``flock`` is associated with the open file
    // description / inode, not the pathname, so after the first
    // ``rename()`` succeeds, new writers lock the new inode while
    // older waiters can still later acquire a lock on the
    // now-unlinked old inode. The two then run concurrently and a
    // newer merge can be lost. The sibling lock file's inode stays
    // stable across target renames; every writer locks the same
    // inode, so the critical section is genuinely serialised.
    const std::string lock_path = path + ".lock";
    int lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
    if (lock_fd < 0) {
        fprintf(stderr,
                "[metric_extractor] could not open filter_stats.json "
                "sibling lock (%s) for merge-locking: %s; counts "
                "dropped on terminate\n",
                lock_path.c_str(), strerror(errno));
        return;
    }
    if (flock(lock_fd, LOCK_EX) != 0) {
        fprintf(stderr,
                "[metric_extractor] flock(LOCK_EX) on filter_stats.json "
                "sibling lock (%s) failed: %s; counts dropped on "
                "terminate\n",
                lock_path.c_str(), strerror(errno));
        close(lock_fd);
        return;
    }

    // Read the existing file content (if any) so the merge knows
    // what state was already on disk. An empty / nonexistent file
    // contributes zero to every max() — the in-process state is
    // taken verbatim.
    std::string existing_body;
    {
        std::ifstream in(path);
        if (in) {
            std::ostringstream buf;
            buf << in.rdbuf();
            existing_body = buf.str();
        }
    }

    // Counter merge: take max(existing, in-process) for each field.
    // ``max()`` is correct because every counter is monotonic in a
    // single process and the merge runs at term, so the disk
    // snapshot from another process is at most as old as the file
    // mtime; merging max-wise can never decrease a live counter.
    auto take_max = [&](uint64_t mine, const std::string& field) {
        return std::max(mine,
                        parse_existing_filter_stats_uint(existing_body, field));
    };
    const uint64_t merged_accepted       = take_max(my_accepted,
                                                    "accepted_launches");
    const uint64_t merged_skipped        = take_max(my_skipped,
                                                    "skipped_launches");
    const uint64_t merged_origin_misses  = take_max(my_origin_misses,
                                                    "module_origin_miss_count");
    const uint64_t merged_cooperative    = take_max(my_cooperative,
                                                    "cooperative_launches");
    const uint64_t merged_graph_nodes    = take_max(my_graph_nodes,
                                                    "graph_node_launches");
    const uint64_t merged_graph_accepted = take_max(my_graph_accepted,
                                                    "graph_node_accepted");
    const uint64_t merged_graph_skipped  = take_max(my_graph_skipped,
                                                    "graph_node_skipped");
    const uint64_t merged_budget_skipped = take_max(my_budget_skipped,
                                                    "per_process_budget_skipped");
    const uint64_t merged_launch_index   = take_max(my_launch_index,
                                                    "launch_index");
    const uint64_t merged_graph_seen     = take_max(my_graph_seen,
                                                    "graph_launches_seen");
    const uint64_t merged_graph_not_obs  = take_max(my_graph_not_obs,
                                                    "graph_launches_not_observed");
    const uint64_t merged_unknown_opcode = take_max(my_unknown_opcode,
                                                    "unknown_opcode_count");

    // Union representative_skipped_paths from the existing file with
    // the in-process list, dedup, cap at MAX_REPRESENTATIVE_SKIPPED_PATHS.
    std::vector<std::string> merged_paths = parse_existing_skipped_paths(
        existing_body);
    {
        std::unordered_set<std::string> seen(merged_paths.begin(),
                                             merged_paths.end());
        for (const auto& p : my_skipped_paths) {
            if (merged_paths.size() >= MAX_REPRESENTATIVE_SKIPPED_PATHS) break;
            if (seen.insert(p).second) {
                merged_paths.push_back(p);
            }
        }
    }

    // Build merged JSON.
    std::ostringstream oss;
    oss << "{";
    oss << "\"schema_version\":" << METRIC_EXTRACTOR_SCHEMA_VERSION << ",";
    oss << "\"tool_version\":\""
        << json_escape(METRIC_EXTRACTOR_TOOL_VERSION) << "\",";
    oss << "\"regex_source\":\"" << json_escape(regex_source) << "\",";
    oss << "\"accepted_launches\":" << merged_accepted << ",";
    oss << "\"skipped_launches\":" << merged_skipped << ",";
    oss << "\"module_origin_miss_count\":" << merged_origin_misses << ",";
    oss << "\"cooperative_launches\":" << merged_cooperative << ",";
    oss << "\"graph_node_launches\":" << merged_graph_nodes << ",";
    oss << "\"graph_node_accepted\":" << merged_graph_accepted << ",";
    oss << "\"graph_node_skipped\":" << merged_graph_skipped << ",";
    oss << "\"graph_launches_seen\":" << merged_graph_seen << ",";
    oss << "\"graph_launches_not_observed\":" << merged_graph_not_obs << ",";
    oss << "\"per_process_budget_skipped\":" << merged_budget_skipped << ",";
    oss << "\"unknown_opcode_count\":" << merged_unknown_opcode << ",";
    oss << "\"launch_index\":" << merged_launch_index << ",";
    oss << "\"representative_skipped_paths\":[";
    for (size_t i = 0; i < merged_paths.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << json_escape(merged_paths[i]) << "\"";
    }
    oss << "]";
    oss << "}\n";

    // Atomic rename: write to ``<path>.tmp.<pid>`` then rename into
    // place. The rename is atomic on POSIX, so readers either see
    // the pre-merge file or the merged file — never a partial write.
    const std::string tmp_path = path + ".tmp." + std::to_string(getpid());
    {
        std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
        if (!out) {
            fprintf(stderr,
                    "[metric_extractor] could not write filter_stats.json "
                    "tmp file (%s): %s; counts dropped on terminate\n",
                    tmp_path.c_str(), strerror(errno));
            flock(lock_fd, LOCK_UN);
            close(lock_fd);
            return;
        }
        out << oss.str();
    }
    if (rename(tmp_path.c_str(), path.c_str()) != 0) {
        fprintf(stderr,
                "[metric_extractor] atomic rename of filter_stats.json "
                "from %s to %s failed: %s; counts dropped on terminate\n",
                tmp_path.c_str(), path.c_str(), strerror(errno));
        unlink(tmp_path.c_str());
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
        return;
    }

    flock(lock_fd, LOCK_UN);
    close(lock_fd);
}

// --------------------------------------------------------------------------
// Driver identities distinguish context lifetimes, recycled stream handles,
// and per-thread default streams. They are local to this process; the
// post-hoc join must not compare them numerically with another run's IDs.
static constexpr int64_t kStreamIdUnresolved = -1;
static constexpr const char* kDriverIdentityNamespace = "cuda_driver";

struct StreamIdentity {
    int64_t context_id = kStreamIdUnresolved;
    int64_t stream_id = kStreamIdUnresolved;
};

static std::atomic<bool> g_identity_failure_reported{false};

// NVBit's vendored headers do not declare every Driver entry point.
extern "C" CUresult cuCtxGetId(CUcontext context, unsigned long long* contextId);
extern "C" CUresult cuStreamGetId(CUstream stream, unsigned long long* streamId);
extern "C" CUresult cuStreamGetId_ptsz(CUstream stream, unsigned long long* streamId);

static StreamIdentity resolve_stream_identity(
    CUcontext context, CUstream stream, bool per_thread_default = false) {
    const CUstream identity_stream =
        per_thread_default && stream == nullptr ? CU_STREAM_PER_THREAD : stream;
    unsigned long long context_id = 0;
    unsigned long long stream_id = 0;
    const CUresult context_rc = cuCtxGetId(context, &context_id);
    const CUresult stream_rc = per_thread_default
        ? cuStreamGetId_ptsz(identity_stream, &stream_id)
        : cuStreamGetId(identity_stream, &stream_id);
    const auto max_id =
        static_cast<unsigned long long>(std::numeric_limits<int64_t>::max());
    if (context_rc != CUDA_SUCCESS || stream_rc != CUDA_SUCCESS ||
        context_id > max_id || stream_id > max_id) {
        bool expected = false;
        if (g_identity_failure_reported.compare_exchange_strong(
                expected, true, std::memory_order_relaxed)) {
            fprintf(stderr,
                    "[metric_extractor] driver identity unresolved: "
                    "context_rc=%d stream_rc=%d context=%p stream=%p "
                    "per_thread_default=%u\n",
                    static_cast<int>(context_rc), static_cast<int>(stream_rc),
                    static_cast<void*>(context), static_cast<void*>(stream),
                    per_thread_default ? 1u : 0u);
        }
        return {};
    }
    StreamIdentity identity;
    identity.context_id = static_cast<int64_t>(context_id);
    identity.stream_id = static_cast<int64_t>(stream_id);
    return identity;
}

// --------------------------------------------------------------------------
// Per-launch JSON-L emitter — writes one record per accepted launch
// (and per graph-launch boundary that was not observed by the per-node
// hook) to the path resolved from METRIC_EXTRACTOR_LAUNCHES_PATH (default
// ./metric_extractor_launches.jsonl). Schema lives in JSONL_SCHEMA.md
// alongside this source. Wall-clock fields are intentionally absent.
// --------------------------------------------------------------------------

namespace {

struct LaunchContext {
    std::string kernel_name_demangled;
    std::string kernel_name_mangled;
    uint64_t    function_uid;
    uint64_t    ctx_id = 0;
    // Raw context pointer for process-local diagnostics. The fields below
    // are driver IDs used to describe this run's stream topology.
    int64_t     context_id = kStreamIdUnresolved;
    // Program-lifetime stream ID from cuStreamGetId. Negative
    // values mean the driver identity query failed; the joiner treats
    // the record as unjoinable.
    int64_t     stream_ordinal = kStreamIdUnresolved;
    // ``launch_index`` is the global observed-launch ordinal (AC-8's
    // ``kernel_ordinal_in_workload`` field) — counts EVERY observed
    // kernel-launch CBID including filter-skipped framework kernels.
    // This is what the CUPTI Activity surface's matched run sees, and
    // is what the post-hoc cross-surface joiner spines against.
    uint64_t    launch_index;
    // ``slot_index`` is the dense accepted-launch counter used as the
    // ring-buffer pool index. Skipped launches NEVER read or write a
    // slot, so the slot pool stays compact. The JSON-L emitter reads
    // device-side counters from ``g_slot_pool[slot_index % g_max_launch_slots]``,
    // NOT from ``launch_index % g_max_launch_slots`` — the two roles
    // (observed-ordinal vs slot-pointer) are now correctly separated
    // post-Round-2 review.
    uint64_t    slot_index;
    uint32_t    grid_dim[3];
    uint32_t    block_dim[3];
    uint32_t    cluster_dim[3];
    uint32_t    dynamic_smem;
    std::string launch_kind;
    bool        is_graph_node;
    bool        is_cooperative;
    std::string waiver_reason;
    // Round 18 (Codex Round-17 review Finding 3): per-launch profiled-
    // binary identity. The launch-enter hook hashes the accepted module
    // origin once per origin path (cached) and stores the 32-byte
    // digest here. ``binary_sha256_valid`` is false when the origin
    // file cannot be hashed (the producer falls back to a 32-zero
    // digest and logs ``tier2_binary_sha256_unresolved``). The host
    // serializer reads these directly so the schema-defined
    // ``LaunchHeaderV1.binary_sha256`` field carries the actual
    // profiled binary image digest rather than the Round-17 source
    // proxy.
    uint8_t     binary_sha256[32];
    bool        binary_sha256_valid;
};

static void set_stream_identity(LaunchContext& lc, CUcontext context,
                                CUstream stream,
                                bool per_thread_default = false) {
    const StreamIdentity identity =
        resolve_stream_identity(context, stream, per_thread_default);
    lc.context_id = identity.context_id;
    lc.stream_ordinal = identity.stream_id;
}

}  // namespace

// Single pending context; the launch mutex serialises enter→exit so a
// single slot is sufficient for non-graph launches. Graph-node launches
// use a per-call local LaunchContext (no enter/exit pairing through the
// launch mutex).
static LaunchContext g_pending_launch_context;

// Forward declarations for the graph-launch substatus helpers (defined
// after nvbit_at_graph_node_launch but referenced from nvbit_at_cuda_event
// dispatch above it).
static bool is_graph_launch_cbid(nvbit_api_cuda_t cbid);
static void handle_graph_launch_event(CUcontext ctx, int is_exit,
                                      nvbit_api_cuda_t cbid,
                                      void* params);

// --------------------------------------------------------------------------
// Per-cuGraphLaunch state, keyed by the stream the graph runs on.
//
// Lives behind its OWN mutex (g_graph_state_mu) — separate from
// g_launch_mutex — so the cuGraphLaunch enter/exit dispatch never holds
// the launch mutex across the asynchronous driver call. Holding the
// launch mutex across the cuGraphLaunch boundary would deadlock against
// nvbit_at_graph_node_launch's own brief acquisition of g_launch_mutex
// for slot allocation. NVBit fires the per-node hook synchronously from
// the same thread that called cuGraphLaunch, so all per-node records
// are appended to pending_node_records before cuGraphLaunch-exit fires.
//
// On cuGraphLaunch-exit:
//   1. Lift state[(context, stream)] out of the map (under g_graph_state_mu).
//   2. cudaStreamSynchronize(stream) to wait for the graph's kernels to
//      finish — slot reads and any future metric-counter reads must
//      happen post-sync.
//   3. If pending_node_records is empty: emit one
//      graph_launch_not_observed JSON-L record.
//   4. Otherwise: emit one graph_node_traced JSON-L record per pending
//      entry (the per-node hook deferred the emit so the records are
//      flushed only after the graph kernels actually completed).
struct GraphLaunchState {
    uint64_t                   snapshot_node_count;
    CUcontext                  ctx;
    CUstream                   stream;
    std::vector<LaunchContext> pending_node_records;
};

struct GraphLaunchKey {
    CUcontext context;
    CUstream stream;

    bool operator==(const GraphLaunchKey& other) const {
        return context == other.context && stream == other.stream;
    }
};

struct GraphLaunchKeyHash {
    size_t operator()(const GraphLaunchKey& key) const {
        const size_t context_hash =
            std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(key.context));
        const size_t stream_hash =
            std::hash<uintptr_t>()(reinterpret_cast<uintptr_t>(key.stream));
        return context_hash ^ (stream_hash + static_cast<size_t>(0x9e3779b9u) +
                               (context_hash << 6) + (context_hash >> 2));
    }
};

static std::mutex g_graph_state_mu;
static std::unordered_map<GraphLaunchKey, GraphLaunchState, GraphLaunchKeyHash>
    g_pending_graph_states;

static const char* DEFAULT_LAUNCHES_PATH =
    "./metric_extractor_launches.jsonl";
static std::mutex   g_launches_file_mu;
static std::ofstream g_launches_file;
static bool          g_launches_file_open_attempted = false;
static std::string   g_launches_file_path_resolved;

// graph_launches_seen / graph_launches_not_observed / unknown_opcode_count
// moved up to the main counter declaration block earlier in this file
// so write_filter_stats_json() (defined before this point but called at
// nvbit_at_term) can snapshot them. See the comment up there.

// Host snapshot for the JSON-L `host` field. Resolved once on first emit.
static std::mutex   g_host_snapshot_mu;
static std::string  g_host_compute_capability;
static int32_t      g_host_cuda_runtime_version = 0;
static std::string  g_host_arch_label;
static bool         g_host_snapshot_resolved = false;

static void resolve_host_snapshot_locked() {
    if (g_host_snapshot_resolved) return;
    int driver_runtime_version = 0;
    if (cudaRuntimeGetVersion(&driver_runtime_version) == cudaSuccess) {
        g_host_cuda_runtime_version = driver_runtime_version;
    }
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        char ccbuf[16];
        std::snprintf(ccbuf, sizeof(ccbuf), "%d.%d", prop.major, prop.minor);
        g_host_compute_capability = ccbuf;
        char archbuf[32];
        // The arch_label mirrors the harness manifest's evidence-label
        // family ("local_sm_<CC>" or "b200_sm_100a"). Detect the B200
        // family by compute_cap.major == 10.
        if (prop.major == 10) {
            std::snprintf(archbuf, sizeof(archbuf), "b200_sm_100a");
        } else {
            std::snprintf(archbuf, sizeof(archbuf), "local_sm_%d%d",
                          prop.major, prop.minor);
        }
        g_host_arch_label = archbuf;
    }
    g_host_snapshot_resolved = true;
}

static void open_launches_file_if_needed_locked() {
    if (g_launches_file_open_attempted) return;
    g_launches_file_open_attempted = true;
    const char* env = getenv("METRIC_EXTRACTOR_LAUNCHES_PATH");
    g_launches_file_path_resolved =
        (env != nullptr && env[0] != '\0') ? env : DEFAULT_LAUNCHES_PATH;
    g_launches_file.open(g_launches_file_path_resolved,
                         std::ios::out | std::ios::trunc);
    if (!g_launches_file) {
        fprintf(stderr,
                "[metric_extractor] could not open launches JSON-L at %s; "
                "per-launch records dropped\n",
                g_launches_file_path_resolved.c_str());
    }
}

static void append_filter_stats_snapshot(std::ostringstream& oss) {
    // Snapshot the in-process representative_skipped_paths list under
    // g_skipped_paths_mu so an asynchronous record_skipped_path()
    // call can't race the iteration. The array is capped at
    // MAX_REPRESENTATIVE_SKIPPED_PATHS at insertion, so the snapshot
    // length is bounded by construction.
    std::vector<std::string> skipped_paths_snapshot;
    {
        std::lock_guard<std::mutex> lk(g_skipped_paths_mu);
        skipped_paths_snapshot = g_representative_skipped_paths;
    }
    oss << "\"filter_stats_snapshot\":{";
    oss << "\"accepted_launches\":" << g_accepted_launches.load() << ",";
    oss << "\"skipped_launches\":" << g_skipped_launches.load() << ",";
    oss << "\"module_origin_miss_count\":"
        << g_module_origin_misses.load() << ",";
    oss << "\"cooperative_launches\":"
        << g_cooperative_launches.load() << ",";
    oss << "\"graph_node_launches\":"
        << g_graph_node_launches.load() << ",";
    oss << "\"graph_node_accepted\":"
        << g_graph_node_accepted.load() << ",";
    oss << "\"graph_node_skipped\":"
        << g_graph_node_skipped.load() << ",";
    oss << "\"graph_launches_seen\":"
        << g_graph_launches_seen.load() << ",";
    oss << "\"graph_launches_not_observed\":"
        << g_graph_launches_not_observed.load() << ",";
    oss << "\"per_process_budget_skipped\":"
        << g_per_process_budget_skipped.load() << ",";
    oss << "\"unknown_opcode_count\":"
        << g_unknown_opcode_count.load() << ",";
    // Embed the representative_skipped_paths array directly in the
    // per-launch snapshot so the surface-side reconciler can recover
    // module attribution when ``nvbit_at_term`` misses the terminal
    // write. Codex's Round-6 review flagged that the Python
    // reconciler can only copy what the in-record snapshot carries;
    // the previous shape gave it counters only.
    oss << "\"representative_skipped_paths\":[";
    for (size_t i = 0; i < skipped_paths_snapshot.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "\"" << json_escape(skipped_paths_snapshot[i]) << "\"";
    }
    oss << "]";
    oss << "}";
}

static void append_host_snapshot(std::ostringstream& oss) {
    std::string cc, arch;
    int32_t rt = 0;
    {
        std::lock_guard<std::mutex> lk(g_host_snapshot_mu);
        resolve_host_snapshot_locked();
        cc = g_host_compute_capability;
        arch = g_host_arch_label;
        rt = g_host_cuda_runtime_version;
    }
    oss << "\"host\":{";
    oss << "\"compute_capability\":\""
        << json_escape(cc) << "\",";
    oss << "\"cuda_runtime_version\":" << rt << ",";
    oss << "\"nvbit_tool_version\":\""
        << json_escape(METRIC_EXTRACTOR_TOOL_VERSION) << "\",";
    oss << "\"arch_label\":\""
        << json_escape(arch) << "\"";
    oss << "}";
}

static void append_xyz(std::ostringstream& oss, const char* name,
                       const uint32_t v[3]) {
    oss << "\"" << name << "\":[" << v[0] << "," << v[1] << "," << v[2]
        << "]";
}

// Map an InstructionClass bucket id to its canonical JSON-L sparse-map key.
// Keeping the names in a single switch keeps the JSON-L emit, the
// JSONL_SCHEMA.md table, and the bitlesson-flagged contract test in lockstep
// — any new bucket id added to the enum in common.h must land its
// canonical name here in the same commit.
static const char* instruction_class_name(int cls) {
    switch (cls) {
        case INSTR_CLASS_FFMA:     return "ffma";
        case INSTR_CLASS_MUFU:     return "mufu";
        case INSTR_CLASS_LDG:      return "ldg";
        case INSTR_CLASS_STG:      return "stg";
        case INSTR_CLASS_LDS:      return "lds";
        case INSTR_CLASS_STS:      return "sts";
        case INSTR_CLASS_HMMA:     return "hmma";
        case INSTR_CLASS_WGMMA:    return "wgmma";
        case INSTR_CLASS_TCGEN05:  return "tcgen05";
        case INSTR_CLASS_TMA:      return "tma";
        case INSTR_CLASS_DSMEM:    return "dsmem";
        case INSTR_CLASS_BRA:      return "bra";
        case INSTR_CLASS_BARRIER:  return "barrier";
        case INSTR_CLASS_WARPSYNC: return "warpsync";
        case INSTR_CLASS_UNKNOWN:  return "unknown";
        default:                   return "unknown";
    }
}

// Read the per-launch instruction-class histogram out of the slot pool
// (post-cudaDeviceSynchronize for regular / cooperative launches, post-
// cudaStreamSynchronize for graph-node launches) and serialise it as a
// sparse JSON object keyed by canonical class name. Zero-count buckets are
// omitted per the JSONL_SCHEMA.md sparse-map contract. The not-observed
// graph-launch synthetic record carries no slot — it lands an empty object
// so the field shape stays stable across record kinds.
static void append_instruction_class_histogram(std::ostringstream& oss,
                                               const LaunchContext& lc) {
    oss << "\"instruction_class_histogram\":{";
    if (lc.launch_kind == "graph_launch_not_observed" ||
        g_slot_pool == nullptr || g_max_launch_slots == 0) {
        oss << "}";
        return;
    }
    const uint64_t slot_index = lc.slot_index % g_max_launch_slots;
    const LaunchMetrics& slot = g_slot_pool[slot_index];
    bool first = true;
    for (int i = 0; i < NUM_INSTRUCTION_CLASSES; ++i) {
        const uint64_t count = slot.instruction_class_histogram[i];
        if (count == 0) continue;
        if (!first) oss << ",";
        first = false;
        oss << "\"" << instruction_class_name(i) << "\":" << count;
    }
    oss << "}";
}

// Map InstrType::MemorySpace enum values to short JSON-L tags. Kept in a
// helper rather than inlined into the descriptor emitter so the schema doc
// can pin against a stable string set.
static const char* tma_memspace_name(InstrType::MemorySpace ms) {
    switch (ms) {
        case InstrType::MemorySpace::NONE:                return "none";
        case InstrType::MemorySpace::LOCAL:               return "local";
        case InstrType::MemorySpace::GENERIC:             return "generic";
        case InstrType::MemorySpace::GLOBAL:              return "global";
        case InstrType::MemorySpace::SHARED:              return "shared";
        case InstrType::MemorySpace::CONSTANT:            return "constant";
        case InstrType::MemorySpace::GLOBAL_TO_SHARED:    return "global_to_shared";
        case InstrType::MemorySpace::SURFACE:             return "surface";
        case InstrType::MemorySpace::TEXTURE:             return "texture";
        case InstrType::MemorySpace::DISTRIBUTED_SHARED:  return "distributed_shared";
        default:                                          return "unknown";
    }
}

// Decode and emit the first-sample TMA descriptor for this launch. When no
// TMA instruction fired (the common case for non-TMA kernels), the field
// is `null` and `confidence.tma_descriptor` is `"absent"` (NOT
// "placeholder" — the absence is a true negative observation, not a
// counter that hasn't landed yet). When a sample is present, the host
// calls `nvbit_parse_tma_transfer_info` to decode the captured handle and
// emits the compact descriptor summary. The synthetic
// graph_launch_not_observed record always lands `null`.
static void append_tma_descriptor(std::ostringstream& oss,
                                  const LaunchContext& lc) {
    if (lc.launch_kind == "graph_launch_not_observed" ||
        g_slot_pool == nullptr || g_max_launch_slots == 0) {
        oss << "\"tma_descriptor\":null";
        return;
    }
    const uint64_t slot_index = lc.slot_index % g_max_launch_slots;
    LaunchMetrics& slot = g_slot_pool[slot_index];
    if (slot.tma_sample_valid == 0u || slot.tma_param_handle_size == 0u) {
        oss << "\"tma_descriptor\":null";
        return;
    }
    const std::string mnemonic =
        lookup_opcode_by_id(slot.tma_sample_opcode_id);
    if (mnemonic.empty()) {
        // Defensive: opcode-id resolution miss is rare (would imply the
        // device wrote an id the host never allocated), but the emit path
        // must stay safe.
        oss << "\"tma_descriptor\":null";
        return;
    }
    CUcontext ctx = reinterpret_cast<CUcontext>(
        static_cast<uintptr_t>(lc.ctx_id));
    TMATransferInfo_t info = nvbit_parse_tma_transfer_info(
        ctx, mnemonic.c_str(), slot.tma_sample_handle,
        slot.tma_param_handle_size);
    oss << "\"tma_descriptor\":{";
    oss << "\"sample_count\":" << slot.tma_count << ",";
    oss << "\"sample_opcode\":\"" << json_escape(mnemonic) << "\",";
    oss << "\"sample_handle_size\":" << slot.tma_param_handle_size << ",";
    oss << "\"is_tensor\":"     << (info.is_tensor ? "true" : "false") << ",";
    oss << "\"is_bulk\":"       << (info.is_bulk ? "true" : "false") << ",";
    oss << "\"is_prefetch\":"   << (info.is_prefetch ? "true" : "false") << ",";
    oss << "\"is_multicast\":"  << (info.is_multicast ? "true" : "false") << ",";
    oss << "\"src_memspace\":\""
        << json_escape(tma_memspace_name(info.src_memspace)) << "\",";
    oss << "\"dst_memspace\":\""
        << json_escape(tma_memspace_name(info.dst_memspace)) << "\",";
    oss << "\"transfer_count\":" << info.transfer_count << ",";
    oss << "\"transfer_size\":"  << info.transfer_size << ",";
    oss << "\"byte_count\":"     << info.byte_count << ",";
    oss << "\"multicast_cta_mask\":" << info.multicast_cta_mask;
    oss << "}";
}

// True iff ``lc`` references a real aggregator slot the JSON-L emitter
// can read counter fields from. Synthetic ``launch_kind`` values
// (``graph_launch_not_observed``, ``per_process_budget_exceeded``) skip
// the slot read because no aggregator was wired up for them. The pool /
// max-slots guard catches the rare case of the slot pool failing to
// allocate at ``nvbit_tool_init`` time.
static bool has_aggregator_slot(const LaunchContext& lc) {
    if (lc.launch_kind == "graph_launch_not_observed") return false;
    if (lc.launch_kind == "per_process_budget_exceeded") return false;
    return g_slot_pool != nullptr && g_max_launch_slots > 0;
}

// Emit a bounded replacement JSON-L record carrying identity fields
// only — no metric arrays, no slot reads. Used for the two AC-15 skip
// paths:
//
// * ``per_launch_artifact_cap_exceeded`` — the full record would have
//   exceeded ``METRIC_EXTRACTOR_PER_LAUNCH_ARTIFACT_CAP_BYTES``. The
//   metric payload is discarded; this replacement is emitted instead.
// * ``per_process_budget_exceeded`` — the accepted-launch count exceeded
//   the slot pool budget. The kernel ran uninstrumented; this record
//   surfaces the launch with explicit skipped status so the surface
//   parser can roll it into the manifest substatus.
//
// The replacement is built into a separate ostringstream so the
// failed full-record buffer can be discarded cleanly. A final size
// assertion before flush guarantees the replacement itself is under
// cap; if not (which is impossible in practice — identity blocks land
// under ~1 KiB), the record is dropped with a stderr diagnostic
// instead of emitting a malformed oversize line.
static void write_bounded_replacement_record_locked(
    const LaunchContext& lc,
    const char* waiver_reason,
    bool is_truncated,
    bool is_skipped,
    uint64_t saturated_counters) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"schema_version\":" << METRIC_EXTRACTOR_SCHEMA_VERSION << ",";
    oss << "\"tool_version\":\""
        << json_escape(METRIC_EXTRACTOR_TOOL_VERSION) << "\",";
    oss << "\"identity_namespace\":\""
        << kDriverIdentityNamespace << "\",";
    oss << "\"launch_kind\":\"" << json_escape(lc.launch_kind) << "\",";
    if (lc.launch_kind != "graph_launch_not_observed") {
        oss << "\"kernel_name\":\""
            << json_escape(lc.kernel_name_demangled) << "\",";
        oss << "\"kernel_name_mangled\":\""
            << json_escape(lc.kernel_name_mangled) << "\",";
        oss << "\"function_uid\":" << lc.function_uid << ",";
    }
    oss << "\"ctx_id\":" << lc.ctx_id << ",";
    oss << "\"context_id\":" << lc.context_id << ",";
    oss << "\"stream_id\":" << lc.stream_ordinal << ",";
    oss << "\"launch_index\":" << lc.launch_index << ",";
    append_xyz(oss, "grid_dim", lc.grid_dim); oss << ",";
    append_xyz(oss, "block_dim", lc.block_dim); oss << ",";
    append_xyz(oss, "cluster_dim", lc.cluster_dim); oss << ",";
    oss << "\"dynamic_smem\":" << lc.dynamic_smem << ",";
    // Metric values: null / empty. The bounded replacement preserves
    // the canonical field shape (a strict parser walking these slots
    // sees the same key set as a full record) but with no metric
    // observations — the cap or budget skip prevented aggregation, so
    // there is nothing to report.
    oss << "\"instruction_class_histogram\":{},";
    oss << "\"tma_descriptor\":null,";
    oss << "\"sector_efficiency_global_load\":null,";
    oss << "\"sector_efficiency_global_store\":null,";
    // Bank conflicts: 0 (parser-required non-bool integer; the
    // replacement record has no aggregator slot to read from, so 0 is
    // the correct sentinel "no observation" value the schema also
    // documents for the normal-record path when no LDS/STS fired).
    oss << "\"bank_conflicts\":0,";
    // Branch active-lanes histogram: exactly 33 zero buckets (parser
    // requires the array length to equal METRIC_EXTRACTOR_BRANCH_ACTIVE_LANES_BUCKETS).
    // The Round-5 regression that emitted `[]` here made every
    // truncated / budget-skipped record fail the surface parse —
    // Codex's Round-5 review's blocking finding.
    oss << "\"branch_active_lanes_hist\":[";
    for (int i = 0; i < (int)METRIC_EXTRACTOR_BRANCH_ACTIVE_LANES_BUCKETS;
         ++i) {
        if (i > 0) oss << ",";
        oss << "0";
    }
    oss << "],";
    oss << "\"confidence\":{";
    oss << "\"instruction_class_histogram\":\"absent\",";
    oss << "\"tma_descriptor\":\"absent\",";
    oss << "\"sector_efficiency_global_load\":\"absent\",";
    oss << "\"sector_efficiency_global_store\":\"absent\",";
    oss << "\"bank_conflicts\":\"absent\",";
    oss << "\"branch_active_lanes_hist\":\"absent\",";
    oss << "\"static_sidecar\":\"absent\"";
    oss << "},";
    oss << "\"static_sidecar\":null,";
    oss << "\"cross_surface_join_key\":{";
    oss << "\"identity_namespace\":\""
        << kDriverIdentityNamespace << "\",";
    oss << "\"workload_index\":null,";
    oss << "\"evaluator_config_hash\":null,";
    if (lc.launch_kind != "graph_launch_not_observed") {
        oss << "\"demangled_kernel_name\":\""
            << json_escape(lc.kernel_name_demangled) << "\",";
    } else {
        oss << "\"demangled_kernel_name\":null,";
    }
    oss << "\"kernel_ordinal_in_workload\":" << lc.launch_index << ",";
    append_xyz(oss, "grid_xyz", lc.grid_dim); oss << ",";
    append_xyz(oss, "block_xyz", lc.block_dim); oss << ",";
    oss << "\"dynamic_smem\":" << lc.dynamic_smem << ",";
    oss << "\"stream_ordinal\":" << lc.stream_ordinal;
    oss << ",\"context_id\":" << lc.context_id;
    oss << "},";
    oss << "\"substatus\":{";
    oss << "\"truncated\":" << (is_truncated ? "true" : "false") << ",";
    oss << "\"skipped\":" << (is_skipped ? "true" : "false") << ",";
    oss << "\"saturated_counters\":" << saturated_counters << ",";
    oss << "\"is_graph_node\":" << (lc.is_graph_node ? "true" : "false")
        << ",";
    oss << "\"is_cooperative\":"
        << (lc.is_cooperative ? "true" : "false") << ",";
    oss << "\"waiver_reason\":\"" << json_escape(waiver_reason) << "\"";
    oss << "},";
    append_filter_stats_snapshot(oss); oss << ",";
    append_host_snapshot(oss);
    oss << "}\n";

    const std::string out = oss.str();
    const size_t cap =
        static_cast<size_t>(METRIC_EXTRACTOR_PER_LAUNCH_ARTIFACT_CAP_BYTES);
    if (out.size() > cap) {
        // Catastrophic: an identity-only record should never approach
        // the cap. Drop the record rather than emit a malformed
        // oversize line; the operator-side run log carries enough
        // signal that a record was dropped, and the strict parser
        // would reject the oversize line anyway.
        fprintf(stderr,
                "[metric_extractor] FATAL: bounded replacement record "
                "(%zu bytes) exceeds per-launch artifact cap (%zu bytes); "
                "dropping record\n",
                out.size(), cap);
        return;
    }
    g_launches_file << out;
    g_launches_file.flush();
}

// --------------------------------------------------------------------------
// Tier-2 address-dump per-launch sidecar binary writer.
//
// At env=1 each accepted launch produces one LaunchFrame in
// nvbit_metrics/tier2_address_trace.t2a:
//
//   LaunchFrame := LaunchHeaderV1 (160 B) AccessRecord* TruncationRecord?
//                  LaunchTrailerV1 (80 B)
//
// The writer streams record bytes through a Tier-2 SHA-256 context to
// compute the per-launch sidecar digest covering the bytes after the
// 160-byte header through the final trailer byte (excluding the
// digest field placeholder in the header itself). After the trailer
// is written the writer seek-patches the header's
// access_record_count / budget_bytes_used / trailer_file_offset /
// sidecar_sha256 fields, then fsyncs.
//
// Frame-delimiter magics distinct from records, per Tier-2 schema
// Section 1: header carries 'TI2A' as a frame-start marker; trailer
// carries 'T2AT' as a frame-end marker. These are NOT record kinds.
// --------------------------------------------------------------------------

static std::mutex   g_tier2_sidecar_mu;
static FILE*        g_tier2_sidecar_file = nullptr;
static std::string  g_tier2_sidecar_path_resolved;

// Tier2EmitResult is the per-launch projection of the emission run that
// feeds the JSONL emitter. The serializer in tier2_frame_serializer.h
// fills in the digest/typed-reason/trace_bytes/marker-emitted fields;
// write_tier2_sidecar_frame layers launch_uid + sidecar_path on top.
struct Tier2EmitResult {
    bool emitted = false;
    uint64_t launch_uid = 0;
    char sha256_hex[65] = {0};
    uint64_t trace_bytes = 0;  // budget_bytes_used = records + truncation
    bool truncated = false;
    std::string typed_reason;  // empty means null
    std::string sidecar_path;
};

static std::string resolve_tier2_sidecar_path_locked() {
    if (!g_tier2_sidecar_path_resolved.empty()) {
        return g_tier2_sidecar_path_resolved;
    }
    const char* launches_env = getenv("METRIC_EXTRACTOR_LAUNCHES_PATH");
    std::string dir = ".";
    if (launches_env != nullptr && launches_env[0] != '\0') {
        std::string p = launches_env;
        auto slash = p.find_last_of('/');
        if (slash != std::string::npos) {
            dir = p.substr(0, slash);
        }
    }
    g_tier2_sidecar_path_resolved = dir + "/tier2_address_trace.t2a";
    return g_tier2_sidecar_path_resolved;
}

// Compute the deterministic per-launch UID per the Tier-2 schema
// Section 2: launch_uid = first 64 bits little-endian of SHA-256
// over ingredient stream including binary_sha256, launch_index,
// stream_ordinal, length-prefixed mangled kernel name, grid/block
// dims, dynamic smem.
static uint64_t compute_tier2_launch_uid(
    const LaunchContext& lc, const uint8_t binary_sha256[32]) {
    tier2::Sha256Ctx ctx;
    static const char prefix[] = "solbench-tier2-launch-v1";
    ctx.Update(prefix, sizeof(prefix));  // includes trailing NUL
    ctx.Update(binary_sha256, 32);
    auto put_u64 = [&](uint64_t v) {
        uint8_t buf[8];
        for (int i = 0; i < 8; ++i) buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
        ctx.Update(buf, 8);
    };
    auto put_u32 = [&](uint32_t v) {
        uint8_t buf[4];
        for (int i = 0; i < 4; ++i) buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
        ctx.Update(buf, 4);
    };
    put_u64(lc.launch_index);
    put_u32(static_cast<uint32_t>(lc.stream_ordinal));
    const uint64_t name_len = static_cast<uint64_t>(lc.kernel_name_mangled.size());
    put_u64(name_len);
    if (name_len > 0) {
        ctx.Update(lc.kernel_name_mangled.data(), lc.kernel_name_mangled.size());
    }
    put_u32(lc.grid_dim[0]);
    put_u32(lc.grid_dim[1]);
    put_u32(lc.grid_dim[2]);
    put_u32(lc.block_dim[0]);
    put_u32(lc.block_dim[1]);
    put_u32(lc.block_dim[2]);
    put_u64(static_cast<uint64_t>(lc.dynamic_smem));
    uint8_t digest[32];
    ctx.Finalize(digest);
    uint64_t uid = 0;
    for (int i = 0; i < 8; ++i) {
        uid |= (static_cast<uint64_t>(digest[i]) << (i * 8));
    }
    return uid;
}

// The Tier-2 frame-write helpers (put_*_le / write_tier2_compact_record /
// write_tier2_per_lane_record / tier2_serialize_frame_body) plus the
// TmaExpansionStatus enum and TmaExpansionResult struct live in
// tier2_frame_serializer.h so the standalone test binary can drive the
// same serialization path with synthetic inputs.

// State passed to the NVBit TMA address-expansion callback. Collects
// the global-side TMA element addresses into a std::vector so the
// caller can derive ordered 32 B sector IDs.
struct TMAAddressCollector {
    std::vector<uint64_t> addresses;
};

static void tma_addr_collector_cb(const TMAElementAddress_t* addrs,
                                  size_t count,
                                  void* user_data) {
    TMAAddressCollector* c =
        static_cast<TMAAddressCollector*>(user_data);
    c->addresses.reserve(c->addresses.size() + count);
    for (size_t i = 0; i < count; ++i) {
        if (!addrs[i].is_oob) {
            c->addresses.push_back(addrs[i].address);
        }
    }
}

// TmaExpansionStatus + TmaExpansionResult moved to tier2_frame_serializer.h
// so the standalone test binary can construct synthetic expansions.

// Per-event TMA descriptor expansion. Reads the captured handle
// from src.tma_handle / src.tma_handle_size, calls
// nvbit_parse_tma_transfer_info to decide src/dst global side, calls
// nvbit_parse_tma_src_addrs_with_flags or _dst_addrs_with_flags with
// ignore_shared=true to collect per-element global addresses, and
// derives ordered 32 B sectors with first-observed de-duplication
// (mirroring the device-side compact path). Returns the expansion
// result with an explicit status enum; callers cache the result
// during the pre-pass and emit from the cache to avoid running the
// expensive NVBit API twice per record.
//
// Status discipline:
//   - empty handle / lookup miss / empty collector with global side
//     expected -> TMA_EXPANSION_FAILED (trace loss; launch is marked
//     truncated by the host serializer)
//   - neither src nor dst is GLOBAL -> TMA_EXPANSION_NO_MODELED_GLOBAL_SIDE
//     (honest clean-omission; the descriptor models a transfer the
//     predictor does not care about)
//   - new sector would push n_sectors past the cap ->
//     TMA_EXPANSION_SECTOR_OVERFLOW (trace loss; the v1.0 wire format
//     cannot honestly represent this many sectors; sector_ids carries
//     the sectors collected before the cap as a diagnostic)
//   - else -> TMA_EXPANSION_OK
static TmaExpansionResult expand_tma_record(
    const LaunchContext& lc, const CompactAccessRecord& src) {
    TmaExpansionResult out = {};
    out.status = TMA_EXPANSION_FAILED;
    if (src.tma_handle_size == 0u) {
        // Device captured the event but the descriptor handle is
        // empty -- the sample-time capture failed. This is trace
        // loss, not a benign omission.
        return out;
    }
    const std::string mnemonic = lookup_opcode_by_id(
        static_cast<uint32_t>(src.pc_or_instruction_id));
    if (mnemonic.empty()) {
        // Opcode-id resolution miss; the host cannot dispatch the
        // descriptor-expansion APIs without the mnemonic.
        return out;
    }
    CUcontext ctx = reinterpret_cast<CUcontext>(
        static_cast<uintptr_t>(lc.ctx_id));
    // NVBit's parse APIs take a non-const handle pointer; the handle
    // bytes in the managed buffer are stable by the time we reach
    // this host-side serializer (post-cudaDeviceSynchronize).
    uint8_t* handle = const_cast<uint8_t*>(src.tma_handle);
    TMATransferInfo_t info = nvbit_parse_tma_transfer_info(
        ctx, mnemonic.c_str(), handle, src.tma_handle_size);
    const bool src_is_global =
        (info.src_memspace == InstrType::MemorySpace::GLOBAL);
    const bool dst_is_global =
        (info.dst_memspace == InstrType::MemorySpace::GLOBAL);
    if (!src_is_global && !dst_is_global) {
        // Honest clean-omission: the descriptor has no global side
        // for the predictor to model (e.g. shared-to-shared bulk
        // copy in a future TMA variant). The host serializer skips
        // this record without marking the launch truncated.
        out.status = TMA_EXPANSION_NO_MODELED_GLOBAL_SIDE;
        return out;
    }
    TMAAddressCollector collector;
    if (src_is_global) {
        nvbit_parse_tma_src_addrs_with_flags(
            ctx, mnemonic.c_str(), handle, src.tma_handle_size,
            tma_addr_collector_cb, &collector,
            /*ignore_shared=*/true, /*ignore_global=*/false);
    }
    if (dst_is_global) {
        nvbit_parse_tma_dst_addrs_with_flags(
            ctx, mnemonic.c_str(), handle, src.tma_handle_size,
            tma_addr_collector_cb, &collector,
            /*ignore_shared=*/true, /*ignore_global=*/false);
    }
    if (collector.addresses.empty()) {
        // The descriptor reported a global side but the
        // address-expansion API returned no addresses. Treat as
        // trace loss rather than benign omission.
        out.status = TMA_EXPANSION_FAILED;
        return out;
    }
    const uint32_t bytes_per_element =
        info.transfer_size > 0
            ? static_cast<uint32_t>(info.transfer_size)
            : 32u;
    out.bytes_per_access = static_cast<uint16_t>(bytes_per_element);
    uint32_t n_sectors = 0;
    bool overflow_seen = false;
    for (uint64_t addr : collector.addresses) {
        const uint64_t first_sec = addr >> 5;
        const uint64_t last_sec =
            (addr + (uint64_t)bytes_per_element - 1ULL) >> 5;
        for (uint64_t s = first_sec; s <= last_sec; ++s) {
            bool found = false;
            for (uint32_t k = 0; k < n_sectors; ++k) {
                if (out.sector_ids[k] == s) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (n_sectors >=
                    METRIC_EXTRACTOR_TIER2_SECTORS_PER_RECORD) {
                    // A new sector would push past the v1.0 wire-
                    // format cap. Mark the record as overflowing so
                    // the host serializer treats it as trace loss
                    // (truncation) rather than silently clamping.
                    overflow_seen = true;
                    break;
                }
                out.sector_ids[n_sectors++] = s;
            }
        }
        if (overflow_seen) break;
    }
    out.sector_count = static_cast<uint16_t>(n_sectors);
    out.status = overflow_seen
                     ? TMA_EXPANSION_SECTOR_OVERFLOW
                     : TMA_EXPANSION_OK;
    return out;
}

static Tier2EmitResult write_tier2_sidecar_frame(
    const LaunchContext& lc, const TierTwoBuffer& tier2_buf) {
    Tier2EmitResult result;
    if (!tier2_address_dump_enabled() || g_tier2_pool == nullptr) {
        return result;
    }
    std::lock_guard<std::mutex> lk(g_tier2_sidecar_mu);
    const std::string path = resolve_tier2_sidecar_path_locked();
    if (g_tier2_sidecar_file == nullptr) {
        g_tier2_sidecar_file = fopen(path.c_str(), "wb");
        if (g_tier2_sidecar_file == nullptr) {
            result.typed_reason = "tier2_sidecar_open_failed";
            result.sidecar_path = path;
            return result;
        }
    }
    FILE* f = g_tier2_sidecar_file;

    // Per-Tier-2 schema: binary_sha256 is the SHA-256 of the profiled
    // binary image as seen by the harness. Round 18 sources this
    // directly from LaunchContext, which the launch-enter hook
    // populated by hashing the accepted module origin (cached per
    // origin path). The Round-17 env-var path
    // (METRIC_EXTRACTOR_TIER2_BINARY_SHA256_HEX) still wins when
    // explicitly set, but the launch-enter hook is what decides
    // between the env-override and the origin hash; this site just
    // copies the final per-launch value.
    uint8_t binary_sha256[32];
    memcpy(binary_sha256, lc.binary_sha256, 32);
    const uint64_t launch_uid =
        compute_tier2_launch_uid(lc, binary_sha256);

    const uint64_t device_seq_count = tier2_buf.sequence_id_counter;
    const uint64_t capacity =
        (uint64_t)METRIC_EXTRACTOR_TIER2_RECORDS_PER_LAUNCH;
    const uint64_t prefix_records =
        device_seq_count < capacity ? device_seq_count : capacity;

    // Expand every op_kind=7 record's descriptor handle ONCE via the
    // NVBit-bound expander. The host-only serializer in
    // tier2_frame_serializer.h reads from this precomputed array; the
    // expander itself stays here because it calls NVBit APIs
    // (nvbit_parse_tma_transfer_info / _*_addrs_with_flags).
    std::vector<TmaExpansionResult> tma_expansions(prefix_records);
    for (uint64_t i = 0; i < prefix_records; ++i) {
        if (tier2_buf.records[i].op_kind == 7u) {
            tma_expansions[i] =
                expand_tma_record(lc, tier2_buf.records[i]);
        }
    }

    const uint64_t budget_limit = tier2_trace_budget_bytes();
    const bool per_lane_on = tier2_per_lane_enabled();
    Tier2FrameSerializeOutput body;
    tier2_serialize_frame_body(
        f,
        launch_uid,
        binary_sha256,
        lc.stream_ordinal,
        lc.grid_dim,
        tier2_buf.records,
        prefix_records,
        tma_expansions.data(),
        tier2_buf.dropped_count,
        tier2_buf.truncated != 0,
        budget_limit,
        per_lane_on,
        &body);

    result.emitted = body.emitted;
    result.launch_uid = launch_uid;
    memcpy(result.sha256_hex, body.sha256_hex, sizeof(result.sha256_hex));
    result.trace_bytes = body.trace_bytes;
    result.truncated = body.truncated;
    if (body.typed_reason != nullptr) {
        result.typed_reason = body.typed_reason;
    }
    result.sidecar_path = path;
    return result;
}

static void close_tier2_sidecar_locked() {
    if (g_tier2_sidecar_file != nullptr) {
        fclose(g_tier2_sidecar_file);
        g_tier2_sidecar_file = nullptr;
    }
}

static void write_launch_jsonl_record(const LaunchContext& lc) {
    std::lock_guard<std::mutex> lk(g_launches_file_mu);
    open_launches_file_if_needed_locked();
    if (!g_launches_file) return;

    // Read the slot's saturated_counters first so both the full record
    // and the bounded-replacement fallback carry the correct value
    // through. has_aggregator_slot() returns false for the two skip
    // paths (graph_launch_not_observed, per_process_budget_exceeded);
    // those records always report saturated_counters = 0.
    uint64_t saturated_counters = 0;
    if (has_aggregator_slot(lc)) {
        const uint64_t sat_slot_idx =
            lc.slot_index % g_max_launch_slots;
        saturated_counters =
            g_slot_pool[sat_slot_idx].saturated_counters;
    }

    // Per-process budget skip path. The launch hook tagged the launch
    // with this kind WHEN g_accepted_slot_index already reached
    // g_max_launch_slots and refused to instrument; the kernel ran
    // uninstrumented. Emit a bounded replacement with skipped status
    // so the surface parser sees the launch as explicitly skipped,
    // not silently dropped.
    if (lc.launch_kind == "per_process_budget_exceeded") {
        write_bounded_replacement_record_locked(
            lc, "per_process_budget_exceeded",
            /*is_truncated=*/false, /*is_skipped=*/true,
            saturated_counters);
        return;
    }

    // Tier-2 address-dump sidecar emission. When the env gate is on
    // AND this launch has a Tier-2 buffer wired (i.e. it was accepted
    // through the instrumentation path), serialize one launch frame
    // into nvbit_metrics/tier2_address_trace.t2a before building the
    // JSON-L line so the tier2_* fields (digest, byte count,
    // truncation, sidecar path, launch uid) can be appended to the
    // per-launch record. When the gate is off (or this launch does
    // not have an aggregator slot), the result is empty and no
    // tier2_* fields are appended — Tier-1 byte identity holds by
    // construction.
    Tier2EmitResult tier2_result;
    if (tier2_address_dump_enabled() && has_aggregator_slot(lc) &&
        g_tier2_pool != nullptr) {
        const uint64_t t2_slot_idx =
            lc.slot_index % g_max_launch_slots;
        const LaunchMetrics& t2_slot = g_slot_pool[t2_slot_idx];
        if (t2_slot.tier2_buffer_ptr != 0) {
            const TierTwoBuffer& t2_buf =
                *reinterpret_cast<const TierTwoBuffer*>(
                    t2_slot.tier2_buffer_ptr);
            tier2_result = write_tier2_sidecar_frame(lc, t2_buf);
        }
    }

    std::ostringstream oss;
    oss << "{";
    oss << "\"schema_version\":" << METRIC_EXTRACTOR_SCHEMA_VERSION << ",";
    oss << "\"tool_version\":\""
        << json_escape(METRIC_EXTRACTOR_TOOL_VERSION) << "\",";
    oss << "\"identity_namespace\":\""
        << kDriverIdentityNamespace << "\",";
    oss << "\"launch_kind\":\"" << json_escape(lc.launch_kind) << "\",";
    if (lc.launch_kind != "graph_launch_not_observed") {
        oss << "\"kernel_name\":\""
            << json_escape(lc.kernel_name_demangled) << "\",";
        oss << "\"kernel_name_mangled\":\""
            << json_escape(lc.kernel_name_mangled) << "\",";
        oss << "\"function_uid\":" << lc.function_uid << ",";
    }
    oss << "\"ctx_id\":" << lc.ctx_id << ",";
    oss << "\"context_id\":" << lc.context_id << ",";
    oss << "\"stream_id\":" << lc.stream_ordinal << ",";
    oss << "\"launch_index\":" << lc.launch_index << ",";
    append_xyz(oss, "grid_dim", lc.grid_dim); oss << ",";
    append_xyz(oss, "block_dim", lc.block_dim); oss << ",";
    append_xyz(oss, "cluster_dim", lc.cluster_dim); oss << ",";
    oss << "\"dynamic_smem\":" << lc.dynamic_smem << ",";

    // Metric value fields. instruction_class_histogram + tma_descriptor
    // + sector_efficiency_global_{load,store} are populated from the
    // per-launch slot's leader-lane atomicAdd aggregation; histogram and
    // tma_descriptor ship with confidence=high, sector_efficiency stays
    // at confidence=placeholder until the synthetic calibration corpus
    // and tolerance gate land (T23-T25 in the plan). The remaining
    // Tier-1 metrics (bank_conflicts, branch_active_lanes_hist) keep
    // their placeholder shape until their per-metric instrumentation
    // sites land in subsequent rounds.
    append_instruction_class_histogram(oss, lc); oss << ",";
    append_tma_descriptor(oss, lc); oss << ",";
    // Compute sector efficiency ratios from the per-launch accumulators.
    // useful_bytes / transferred_bytes lands in [0.0, 1.0] by
    // construction post-Round-6: the device-side count_distinct_sectors
    // helper iterates the full byte interval [addr, addr+bytes) per lane,
    // so every useful byte is accounted for in transferred_bytes and the
    // ratio cannot exceed 1.0. NO upper clamp here — if a future
    // producer-side regression breaks the useful <= transferred
    // invariant, an out-of-range value reaches the strict parser, which
    // rejects the record with a [0.0, 1.0] ParseError. Silent flattening
    // to 1.0 (which masked the Round-5 single-sector-only bug Codex
    // caught in review) is what this round explicitly removes.
    // The lower 0.0 clamp stays — useful and transferred are unsigned
    // counters, so the ratio is always non-negative by type, but
    // -1.0 is the explicit sentinel for "no LDG / STG observed" so the
    // null emit path can disambiguate from a real 0.0 ratio.
    double sector_load_ratio = -1.0;
    double sector_store_ratio = -1.0;
    if (lc.launch_kind != "graph_launch_not_observed" &&
        g_slot_pool != nullptr && g_max_launch_slots > 0) {
        const uint64_t sec_slot_idx =
            lc.slot_index % g_max_launch_slots;
        const LaunchMetrics& sec_slot = g_slot_pool[sec_slot_idx];
        if (sec_slot.sector_load_transferred_bytes > 0) {
            sector_load_ratio =
                (double)sec_slot.sector_load_useful_bytes /
                (double)sec_slot.sector_load_transferred_bytes;
        }
        if (sec_slot.sector_store_transferred_bytes > 0) {
            sector_store_ratio =
                (double)sec_slot.sector_store_useful_bytes /
                (double)sec_slot.sector_store_transferred_bytes;
        }
    }
    oss << "\"sector_efficiency_global_load\":";
    if (sector_load_ratio < 0.0) {
        oss << "null";
    } else {
        // Six-decimal-place format keeps the ratio human-readable while
        // staying well inside JSON's float precision. The sprintf trip
        // avoids ostream locale-dependence on the decimal separator.
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f", sector_load_ratio);
        oss << buf;
    }
    oss << ",";
    oss << "\"sector_efficiency_global_store\":";
    if (sector_store_ratio < 0.0) {
        oss << "null";
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f", sector_store_ratio);
        oss << buf;
    }
    oss << ",";
    // bank_conflicts: aggregate count of per-warp 32-bank shared-memory
    // conflicts across all instrumented plain-shared LDS / STS warp
    // instances. DSMEM is excluded by construction (the instrument-pass
    // filter is `== SHARED`, not `== DISTRIBUTED_SHARED`). 0 is a true
    // observation meaning "no conflicts observed" (or no LDS/STS fired),
    // not a placeholder.
    uint64_t bank_conflicts_value = 0;
    if (lc.launch_kind != "graph_launch_not_observed" &&
        g_slot_pool != nullptr && g_max_launch_slots > 0) {
        const uint64_t bc_slot_idx =
            lc.slot_index % g_max_launch_slots;
        bank_conflicts_value =
            g_slot_pool[bc_slot_idx].bank_conflict_total;
    }
    oss << "\"bank_conflicts\":" << bank_conflicts_value << ",";
    // branch_active_lanes_hist: per-bucket counts of warp-instances of
    // every instrumented INSTR_CLASS_BRA instruction, indexed by the
    // popcount of the warp-collective predicate mask. Bucket 0 counts
    // predicate-false-everywhere instances; bucket 32 counts
    // fully-active branches; bucket 16 is the AC-4 canonical
    // 16-active divergent branch. For graph_launch_not_observed the
    // array stays all-zero (no slot to read from), matching the
    // existing pattern for bank_conflicts / sector_efficiency.
    oss << "\"branch_active_lanes_hist\":[";
    if (lc.launch_kind != "graph_launch_not_observed" &&
        g_slot_pool != nullptr && g_max_launch_slots > 0) {
        const uint64_t br_slot_idx =
            lc.slot_index % g_max_launch_slots;
        const LaunchMetrics& br_slot = g_slot_pool[br_slot_idx];
        for (int i = 0;
             i < (int)METRIC_EXTRACTOR_BRANCH_ACTIVE_LANES_BUCKETS;
             ++i) {
            if (i > 0) oss << ",";
            oss << br_slot.branch_active_lanes_hist[i];
        }
    } else {
        for (int i = 0;
             i < (int)METRIC_EXTRACTOR_BRANCH_ACTIVE_LANES_BUCKETS;
             ++i) {
            if (i > 0) oss << ",";
            oss << "0";
        }
    }
    oss << "],";
    // Compute the tma_descriptor confidence label based on whether the
    // slot's sample-valid flag fired this launch. The non-fired path
    // intentionally uses "absent" (the field is null, but the absence is
    // a true negative observation — no TMA instruction was instrumented)
    // rather than "placeholder" (which would mean "the field shape is in
    // place but the counter pathway hasn't been wired yet"). For the
    // graph_launch_not_observed synthetic record the field is null and
    // the confidence stays "absent" too.
    const char* tma_confidence = "absent";
    if (lc.launch_kind != "graph_launch_not_observed" &&
        g_slot_pool != nullptr && g_max_launch_slots > 0) {
        const uint64_t tma_slot_idx =
            lc.slot_index % g_max_launch_slots;
        if (g_slot_pool[tma_slot_idx].tma_sample_valid != 0u &&
            g_slot_pool[tma_slot_idx].tma_param_handle_size != 0u) {
            tma_confidence = "high";
        }
    }
    // Post-T25 (Round-6) confidence labels. The synthetic
    // Tier-1 calibration corpus (T23/T24) was run end-to-end against
    // the producer's emit path; all six AC-4 positive tests
    // (coalesced 32x4B, stride-2 32x4B, 32-way bank conflict,
    // broadcast, predicated branch 16-active, vector 32x16B) pass
    // every per-metric verdict. The four Tier-1 metric fields are
    // therefore promoted from "placeholder" → "calibrated":
    // sector_efficiency_global_load (positive tests 1, 2, 6;
    // negative tests confirmed via launch-shape mismatch), the same
    // for _global_store, bank_conflicts (positive tests 3, 4),
    // branch_active_lanes_hist (positive test 5). The
    // instruction_class_histogram bucket counts stay "high" — they
    // are direct counters without a calibration step. The
    // static_sidecar label stays "placeholder" because the harness-
    // side join populates the four sub-fields independently and
    // they have no calibration corpus of their own. Calibration
    // report preserved in the round-6 evidence bundle.
    oss << "\"confidence\":{";
    oss << "\"instruction_class_histogram\":\"high\",";
    oss << "\"tma_descriptor\":\"" << tma_confidence << "\",";
    oss << "\"sector_efficiency_global_load\":\"calibrated\",";
    oss << "\"sector_efficiency_global_store\":\"calibrated\",";
    oss << "\"bank_conflicts\":\"calibrated\",";
    oss << "\"branch_active_lanes_hist\":\"calibrated\",";
    oss << "\"static_sidecar\":\"placeholder\"";
    oss << "},";
    oss << "\"static_sidecar\":null,";

    // Cross-surface CUPTI join key. workload_index and
    // evaluator_config_hash are populated by the harness surface plugin
    // when it lands; the tool emits null for those slots so the field
    // shape stays stable.
    oss << "\"cross_surface_join_key\":{";
    oss << "\"identity_namespace\":\""
        << kDriverIdentityNamespace << "\",";
    oss << "\"workload_index\":null,";
    oss << "\"evaluator_config_hash\":null,";
    if (lc.launch_kind != "graph_launch_not_observed") {
        oss << "\"demangled_kernel_name\":\""
            << json_escape(lc.kernel_name_demangled) << "\",";
    } else {
        oss << "\"demangled_kernel_name\":null,";
    }
    oss << "\"kernel_ordinal_in_workload\":" << lc.launch_index << ",";
    append_xyz(oss, "grid_xyz", lc.grid_dim); oss << ",";
    append_xyz(oss, "block_xyz", lc.block_dim); oss << ",";
    oss << "\"dynamic_smem\":" << lc.dynamic_smem << ",";
    oss << "\"stream_ordinal\":" << lc.stream_ordinal << ",";
    oss << "\"context_id\":" << lc.context_id;
    oss << "},";

    // Build the substatus block with the REAL saturated_counters value
    // read off the slot at the top of this function. The truncated /
    // skipped flags are false here — the actual truncation path
    // (full-record-over-cap) is handled AFTER the full record is
    // built, by discarding ``oss`` and emitting the bounded
    // replacement. The per-process-budget skip path is handled at the
    // top of this function before any of the histogram / sector /
    // bank / branch slot reads, so this code only runs for normally
    // aggregated launches.
    oss << "\"substatus\":{";
    oss << "\"truncated\":false,";
    oss << "\"skipped\":false,";
    oss << "\"saturated_counters\":" << saturated_counters << ",";
    oss << "\"is_graph_node\":" << (lc.is_graph_node ? "true" : "false")
        << ",";
    oss << "\"is_cooperative\":"
        << (lc.is_cooperative ? "true" : "false") << ",";
    oss << "\"waiver_reason\":";
    if (lc.waiver_reason.empty()) {
        oss << "null";
    } else {
        oss << "\"" << json_escape(lc.waiver_reason) << "\"";
    }
    oss << "},";

    append_filter_stats_snapshot(oss); oss << ",";
    append_host_snapshot(oss);

    // Tier-2 address-dump appended fields. Emitted only when the
    // sidecar writer ran for this launch, i.e. env=1 AND the launch
    // had an aggregator slot AND the buffer pointer was wired. At
    // env=0 these fields are omitted; the existing field order
    // preceding the closing brace stays byte-identical to the
    // pre-Tier-2 baseline so AC-13 byte-identity holds by
    // construction.
    if (tier2_result.emitted) {
        oss << ",";
        oss << "\"tier2_sidecar_path\":\""
            << json_escape(tier2_result.sidecar_path) << "\",";
        oss << "\"tier2_launch_uid\":" << tier2_result.launch_uid << ",";
        oss << "\"tier2_sidecar_sha256\":\""
            << tier2_result.sha256_hex << "\",";
        oss << "\"tier2_trace_bytes\":" << tier2_result.trace_bytes << ",";
        oss << "\"tier2_trace_truncated\":"
            << (tier2_result.truncated ? "true" : "false") << ",";
        oss << "\"tier2_trace_typed_reason\":";
        if (tier2_result.typed_reason.empty()) {
            oss << "null";
        } else {
            oss << "\"" << json_escape(tier2_result.typed_reason) << "\"";
        }
    }

    oss << "}\n";

    // Final size check. If the full record exceeds the per-launch
    // artifact cap, DISCARD it and emit a bounded replacement with
    // identity fields + null metric values + truncated flag +
    // per_launch_artifact_cap_exceeded waiver. The plan's AC-15
    // negative test explicitly rejects flushing the oversize payload
    // with the flag flipped — the cap must actually bound the line.
    const std::string full_record = oss.str();
    const size_t per_launch_cap =
        static_cast<size_t>(METRIC_EXTRACTOR_PER_LAUNCH_ARTIFACT_CAP_BYTES);
    if (full_record.size() > per_launch_cap) {
        write_bounded_replacement_record_locked(
            lc, "per_launch_artifact_cap_exceeded",
            /*is_truncated=*/true, /*is_skipped=*/false,
            saturated_counters);
        return;
    }

    g_launches_file << full_record;
    g_launches_file.flush();
}

static void write_graph_launch_not_observed_record(CUcontext ctx,
                                                    CUstream stream,
                                                    uint64_t launch_index,
                                                    bool per_thread_default) {
    LaunchContext lc;
    lc.function_uid = 0;
    lc.ctx_id = reinterpret_cast<uintptr_t>(ctx);
    set_stream_identity(lc, ctx, stream, per_thread_default);
    lc.launch_index = launch_index;
    // graph_launch_not_observed records never read a slot — the synthesised
    // record carries placeholder metric values. Pin slot_index to 0 so
    // the JSON-L emitter's modulo lookup lands safely on slot 0 even
    // though it never dereferences any counter there.
    lc.slot_index = 0;
    for (int i = 0; i < 3; ++i) {
        lc.grid_dim[i] = 0;
        lc.block_dim[i] = 0;
        lc.cluster_dim[i] = 0;
    }
    lc.dynamic_smem = 0;
    lc.launch_kind = "graph_launch_not_observed";
    lc.is_graph_node = false;
    lc.is_cooperative = false;
    lc.waiver_reason = "graph launches not enumerated by tool";
    write_launch_jsonl_record(lc);
}

// --------------------------------------------------------------------------
// NVBit lifecycle
// --------------------------------------------------------------------------

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    fprintf(stderr,
            "[metric_extractor] loaded (schema_version=%d tool_version=%s)\n",
            METRIC_EXTRACTOR_SCHEMA_VERSION,
            METRIC_EXTRACTOR_TOOL_VERSION);
}

void nvbit_tool_init(CUcontext /*ctx*/) {
    pthread_mutex_lock(&g_launch_mutex);
    if (g_slot_pool != nullptr) {
        pthread_mutex_unlock(&g_launch_mutex);
        return;
    }
    g_max_launch_slots = resolve_max_launch_slots();
    // The combined Tier-1 + Tier-2 per-slot footprint must fit under
    // the per-process budget. When the Tier-2 address-dump gate is
    // on, every slot pays both sizeof(LaunchMetrics) (Tier-1) and
    // sizeof(TierTwoBuffer) (Tier-2; ~478 MiB per slot at the bumped
    // TIER2_RECORDS_PER_LAUNCH=524288 cap). Clamping only
    // sizeof(LaunchMetrics) would let the Tier-2 pool blow through
    // ~1.87 TiB of managed memory at the default 4096-slot cap.
    const bool tier2_on = tier2_address_dump_enabled();
    const size_t per_slot_bytes =
        sizeof(LaunchMetrics) +
        (tier2_on ? sizeof(TierTwoBuffer) : (size_t)0);
    size_t bytes = per_slot_bytes * g_max_launch_slots;
    // When the operator configures METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS
    // large enough that the slot pool would exceed
    // METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES (default 512 MiB),
    // clamp the slot count down to the largest count that fits and
    // emit a stderr warning. The slot pool stays compact and the
    // tool keeps running; without the clamp a misconfigured env value
    // would silently OOM the device.
    //
    // Guard against the zero-slot edge case: if an operator
    // explicitly sets METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES below
    // ``per_slot_bytes`` (e.g. 64 MiB while Tier-2 is on and a slot
    // costs ~478 MiB), ``budget / per_slot_bytes`` truncates to zero
    // and the slot pool would be unallocatable -- every launch
    // would route to per_process_budget_exceeded. Fail closed to at
    // least one valid slot and emit a stronger warning so the
    // operator sees the under-sized override.
    const size_t budget = resolve_per_process_budget_bytes();
    if (bytes > budget) {
        uint32_t clamped_slots = static_cast<uint32_t>(
            budget / per_slot_bytes
        );
        if (clamped_slots == 0u) {
            clamped_slots = 1u;
            fprintf(stderr,
                    "[metric_extractor] WARNING: per-process budget "
                    "%zu bytes is smaller than the per-slot footprint "
                    "(%zu bytes; Tier-2 address-dump %s). Clamping the "
                    "slot pool to 1 slot so the tool still allocates a "
                    "usable buffer. Operator should raise "
                    "METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES if more "
                    "concurrent instrumented launches are needed.\n",
                    budget,
                    per_slot_bytes,
                    tier2_on ? "on" : "off");
        }
        fprintf(stderr,
                "[metric_extractor] WARNING: requested slot pool "
                "(%u slots x %zu bytes = %zu bytes) exceeds the "
                "per-process budget cap (%zu bytes). Clamping to %u "
                "slots; kernels accepted beyond that count will run "
                "uninstrumented and emit "
                "substatus.skipped=true with "
                "waiver_reason=per_process_budget_exceeded so the "
                "operator sees the skipped cohort explicitly.\n",
                (unsigned)g_max_launch_slots,
                per_slot_bytes,
                bytes,
                budget,
                (unsigned)clamped_slots);
        g_max_launch_slots = clamped_slots;
    }
    bytes = sizeof(LaunchMetrics) * g_max_launch_slots;
    ME_CUDA_CHECK(cudaMallocManaged(
        reinterpret_cast<void**>(&g_slot_pool), bytes));
    if (g_slot_pool != nullptr) {
        memset(g_slot_pool, 0, bytes);
    }
    fprintf(stderr,
            "[metric_extractor] slot pool allocated: %u slots x "
            "%zu bytes = %zu bytes total (per-process budget %zu bytes)\n",
            (unsigned)g_max_launch_slots,
            sizeof(LaunchMetrics),
            bytes,
            budget);
    // Pre-warm the regex so an invalid env var surfaces at init time
    // rather than on the first launch.
    {
        std::lock_guard<std::mutex> lk(g_regex_mu);
        init_user_module_regex_locked();
    }
    // Allocate the Tier-2 per-launch buffer pool ONLY when the
    // address-dump env gate is on. The Tier-2 pool is parallel to
    // ``g_slot_pool`` with one ``TierTwoBuffer`` per slot; the host
    // attaches the per-slot buffer pointer into
    // ``LaunchMetrics::tier2_buffer_ptr`` at launch-enter and the
    // device callback in ``inject_funcs.cu`` follows that pointer.
    // When the gate is off the pool stays NULL and zero managed
    // memory is consumed by the Tier-2 producer path.
    if (tier2_address_dump_enabled()) {
        const size_t tier2_bytes =
            sizeof(TierTwoBuffer) *
            static_cast<size_t>(g_max_launch_slots);
        ME_CUDA_CHECK(cudaMallocManaged(
            reinterpret_cast<void**>(&g_tier2_pool), tier2_bytes));
        if (g_tier2_pool != nullptr) {
            memset(g_tier2_pool, 0, tier2_bytes);
        }
        fprintf(stderr,
                "[metric_extractor] tier2 pool allocated: %u slots x "
                "%zu bytes = %zu bytes (METRIC_EXTRACTOR_TIER2_"
                "ADDRESS_DUMP=1)\n",
                (unsigned)g_max_launch_slots,
                sizeof(TierTwoBuffer),
                tier2_bytes);
    }
    pthread_mutex_unlock(&g_launch_mutex);
}

void nvbit_at_term() {
    // Close the Tier-2 sidecar file (when env=1 opened it). The
    // sidecar is opened lazily on the first accepted launch and
    // remains open across the process lifetime so per-launch frames
    // append efficiently; at process termination we fsync + close
    // so consumers see a complete file.
    {
        std::lock_guard<std::mutex> lk(g_tier2_sidecar_mu);
        close_tier2_sidecar_locked();
    }
    fprintf(stderr,
            "[metric_extractor] terminating: launches_seen=%llu "
            "accepted=%llu skipped=%llu origin_misses=%llu "
            "coop=%llu graph_nodes=%llu (accepted=%llu skipped=%llu) "
            "graph_launches=%llu (not_observed=%llu) "
            "per_process_budget_skipped=%llu pool_slots=%u\n",
            (unsigned long long)g_launch_index,
            (unsigned long long)g_accepted_launches.load(),
            (unsigned long long)g_skipped_launches.load(),
            (unsigned long long)g_module_origin_misses.load(),
            (unsigned long long)g_cooperative_launches.load(),
            (unsigned long long)g_graph_node_launches.load(),
            (unsigned long long)g_graph_node_accepted.load(),
            (unsigned long long)g_graph_node_skipped.load(),
            (unsigned long long)g_graph_launches_seen.load(),
            (unsigned long long)g_graph_launches_not_observed.load(),
            (unsigned long long)g_per_process_budget_skipped.load(),
            (unsigned)g_max_launch_slots);
    write_filter_stats_json();
    // The launches JSON-L file contains ONE record per accepted
    // launch (regular / cooperative / graph_node_traced) plus one
    // graph_launch_not_observed record per missed graph boundary.
    // Run-end summary lives in filter_stats.json; do NOT write a
    // synthetic terminal record into the per-launch file.
    {
        std::lock_guard<std::mutex> lk(g_launches_file_mu);
        if (g_launches_file.is_open()) {
            g_launches_file.close();
        }
    }
}

// --------------------------------------------------------------------------
// Launch-event hook
// --------------------------------------------------------------------------

static bool is_module_load_event(nvbit_api_cuda_t cbid) {
    return (cbid == API_CUDA_cuModuleLoad) ||
           (cbid == API_CUDA_cuModuleLoadData) ||
           (cbid == API_CUDA_cuModuleLoadDataEx) ||
           (cbid == API_CUDA_cuModuleLoadFatBinary) ||
           (cbid == API_CUDA_cuLibraryLoadData) ||
           (cbid == API_CUDA_cuLibraryLoadFromFile) ||
           (cbid == API_CUDA_cuModuleGetFunction) ||
           (cbid == API_CUDA_cuKernelGetFunction) ||
           (cbid == API_CUDA_cuModuleUnload) ||
           (cbid == API_CUDA_cuLibraryUnload);
}

static void handle_module_load_event(CUcontext ctx, int is_exit,
                                     nvbit_api_cuda_t cbid, void* params,
                                     CUresult* pStatus) {
    // Only act on success-exit so failed load attempts do not pollute the
    // registry.
    if (!is_exit) return;
    if (pStatus == nullptr || *pStatus != CUDA_SUCCESS) return;

    // Resolve any /proc/self/maps lookups BEFORE acquiring g_origin_mu so
    // the inner critical section does not perform host-side I/O.
    std::string staged_origin;
    switch (cbid) {
        case API_CUDA_cuModuleLoadData: {
            auto* p = static_cast<cuModuleLoadData_params*>(params);
            if (p->module && *p->module) {
                staged_origin = host_addr_to_so_path(p->image);
            }
            break;
        }
        case API_CUDA_cuModuleLoadDataEx: {
            auto* p = static_cast<cuModuleLoadDataEx_params*>(params);
            if (p->module && *p->module) {
                staged_origin = host_addr_to_so_path(p->image);
            }
            break;
        }
        case API_CUDA_cuModuleLoadFatBinary: {
            auto* p = static_cast<cuModuleLoadFatBinary_params*>(params);
            if (p->module && *p->module) {
                staged_origin = host_addr_to_so_path(p->fatCubin);
            }
            break;
        }
        case API_CUDA_cuLibraryLoadData: {
            auto* p = static_cast<cuLibraryLoadData_params*>(params);
            if (p->library && *p->library) {
                staged_origin = host_addr_to_so_path(p->code);
            }
            break;
        }
        default:
            break;
    }

    std::lock_guard<std::mutex> lk(g_origin_mu);
    switch (cbid) {
        case API_CUDA_cuModuleLoad: {
            // File-path module load: the kernel records the on-disk path
            // verbatim, no /proc/self/maps lookup needed.
            auto* p = static_cast<cuModuleLoad_params*>(params);
            if (p->module && *p->module && p->fname) {
                g_module_origin[ModuleKey{ctx, *p->module}] = p->fname;
                populate_module_func_origins_locked(ctx, *p->module);
            }
            break;
        }
        case API_CUDA_cuModuleLoadData: {
            auto* p = static_cast<cuModuleLoadData_params*>(params);
            if (p->module && *p->module) {
                g_module_origin[ModuleKey{ctx, *p->module}] = staged_origin;
                populate_module_func_origins_locked(ctx, *p->module);
            }
            break;
        }
        case API_CUDA_cuModuleLoadDataEx: {
            auto* p = static_cast<cuModuleLoadDataEx_params*>(params);
            if (p->module && *p->module) {
                g_module_origin[ModuleKey{ctx, *p->module}] = staged_origin;
                populate_module_func_origins_locked(ctx, *p->module);
            }
            break;
        }
        case API_CUDA_cuModuleLoadFatBinary: {
            auto* p = static_cast<cuModuleLoadFatBinary_params*>(params);
            if (p->module && *p->module) {
                g_module_origin[ModuleKey{ctx, *p->module}] = staged_origin;
                populate_module_func_origins_locked(ctx, *p->module);
            }
            break;
        }
        case API_CUDA_cuLibraryLoadData: {
            auto* p = static_cast<cuLibraryLoadData_params*>(params);
            if (p->library && *p->library) {
                g_library_origin[LibraryKey{ctx, *p->library}] =
                    staged_origin;
                populate_library_func_origins_locked(
                    ctx, *p->library, staged_origin);
            }
            break;
        }
        case API_CUDA_cuLibraryLoadFromFile: {
            auto* p = static_cast<cuLibraryLoadFromFile_params*>(params);
            if (p->library && *p->library && p->fileName) {
                const std::string origin(p->fileName);
                g_library_origin[LibraryKey{ctx, *p->library}] = origin;
                populate_library_func_origins_locked(
                    ctx, *p->library, origin);
            }
            break;
        }
        case API_CUDA_cuModuleGetFunction: {
            auto* p = static_cast<cuModuleGetFunction_params*>(params);
            if (p->hfunc && *p->hfunc && p->hmod) {
                auto it = g_module_origin.find(ModuleKey{ctx, p->hmod});
                if (it != g_module_origin.end()) {
                    register_func_and_related_origin_locked(
                        ctx, *p->hfunc, it->second);
                }
            }
            break;
        }
        case API_CUDA_cuKernelGetFunction: {
            // The cuLibrary -> CUkernel -> CUfunction chain finalises
            // here. params->kernel is the CUkernel; resolve its owning
            // CUlibrary via cuKernelGetLibrary (driver API; pure
            // driver call, no NVBit re-entry) and look up the
            // previously-captured library origin under an exact
            // LibraryKey. This is the deterministic replacement
            // for an earlier draft that scanned g_library_origin
            // for any context-matching entry — that scan was wrong
            // because g_library_origin is unordered_map and "the last
            // match" is not load order, so a user kernel could inherit
            // a framework library origin (or vice versa) and break
            // the user-module filter guarantee.
            auto* p = static_cast<cuKernelGetFunction_params*>(params);
            if (p->pFunc == nullptr || *p->pFunc == nullptr ||
                p->kernel == nullptr) {
                break;
            }
            CUlibrary lib = nullptr;
            const CUresult lib_res = cuKernelGetLibrary(&lib, p->kernel);
            if (lib_res != CUDA_SUCCESS || lib == nullptr) {
                break;
            }
            auto orig_it = g_library_origin.find(LibraryKey{ctx, lib});
            if (orig_it == g_library_origin.end()) {
                break;
            }
            register_func_and_related_origin_locked(
                ctx, *p->pFunc, orig_it->second);
            break;
        }
        case API_CUDA_cuModuleUnload: {
            auto* p = static_cast<cuModuleUnload_params*>(params);
            if (p->hmod) {
                g_module_origin.erase(ModuleKey{ctx, p->hmod});
                // Func-keyed entries from this module become stale; the
                // next launch on a stale func handle will fall through
                // to the mangled-name fallback and, failing that, be
                // reported as an origin miss harmlessly.
            }
            break;
        }
        case API_CUDA_cuLibraryUnload: {
            auto* p = static_cast<cuLibraryUnload_params*>(params);
            if (p->library) {
                g_library_origin.erase(LibraryKey{ctx, p->library});
            }
            break;
        }
        default:
            break;
    }
}

static bool is_launch_cbid(nvbit_api_cuda_t cbid) {
    return (cbid == API_CUDA_cuLaunchKernel) ||
           (cbid == API_CUDA_cuLaunchKernel_ptsz) ||
           (cbid == API_CUDA_cuLaunchKernelEx) ||
           (cbid == API_CUDA_cuLaunchKernelEx_ptsz) ||
           (cbid == API_CUDA_cuLaunchCooperativeKernel) ||
           (cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz);
}

static bool is_per_thread_default_launch(nvbit_api_cuda_t cbid) {
    return (cbid == API_CUDA_cuLaunchKernel_ptsz) ||
           (cbid == API_CUDA_cuLaunchKernelEx_ptsz) ||
           (cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz);
}

static CUfunction extract_func_from_launch_params(nvbit_api_cuda_t cbid,
                                                   void* params) {
    if (cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        auto* p = static_cast<cuLaunchKernelEx_params*>(params);
        return p->f;
    }
    if (cbid == API_CUDA_cuLaunchCooperativeKernel ||
        cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
        auto* p = static_cast<cuLaunchCooperativeKernel_params*>(params);
        return p->f;
    }
    auto* p = static_cast<cuLaunchKernel_params*>(params);
    return p->f;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* /*name*/, void* params,
                         CUresult* pStatus) {
    if (is_module_load_event(cbid)) {
        handle_module_load_event(ctx, is_exit, cbid, params, pStatus);
        return;
    }

    if (is_graph_launch_cbid(cbid)) {
        handle_graph_launch_event(ctx, is_exit, cbid, params);
        return;
    }

    if (!is_launch_cbid(cbid)) return;

    if (!is_exit) {
        pthread_mutex_lock(&g_launch_mutex);
        g_last_launch_accepted = false;

        // EVERY observed kernel-launch CBID burns one ``g_launch_index``
        // ordinal regardless of whether the user-module filter ends up
        // accepting it. This is the AC-8 invariant the matched
        // ``cupti_activity`` Surface 2 run depends on: CUPTI sees all
        // launches (filtered framework kernels included), so to keep
        // the two surfaces' ordinals in lockstep the producer must
        // also count filtered launches. The ordinal is captured into
        // ``observed_launch_index`` BEFORE the early-return filter
        // branches so a skipped launch still consumes its slot in the
        // ordinal sequence.
        const uint64_t observed_launch_index = g_launch_index;
        g_launch_index++;

        if (cbid == API_CUDA_cuLaunchCooperativeKernel ||
            cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
            g_cooperative_launches.fetch_add(1, std::memory_order_relaxed);
        }

        if (g_slot_pool == nullptr || g_max_launch_slots == 0) {
            // Pool allocation failed during nvbit_tool_init(); skip slot
            // bookkeeping but keep the mutex symmetric so launch-exit
            // unlock pairs cleanly. The launch is treated as skipped,
            // but the observed ordinal was already consumed at the top
            // of this branch.
            g_skipped_launches.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        CUfunction func = extract_func_from_launch_params(cbid, params);
        if (func == nullptr) {
            g_skipped_launches.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Resolve the mangled name OUTSIDE the registry lock so the
        // NVBit call (which is its own thread-safe API) does not
        // enter g_origin_mu transitively.
        const char* mangled_c =
            nvbit_get_func_name(ctx, func, /*mangled=*/true);
        const std::string mangled_name =
            (mangled_c != nullptr) ? std::string(mangled_c) : std::string();
        std::string origin;
        {
            std::lock_guard<std::mutex> lk(g_origin_mu);
            origin = lookup_func_origin_locked(ctx, func, mangled_name);
        }

        const bool origin_resolved =
            !origin.empty() && origin.front() != '<';
        if (!origin_resolved) {
            g_module_origin_misses.fetch_add(1, std::memory_order_relaxed);
            g_skipped_launches.fetch_add(1, std::memory_order_relaxed);
            record_skipped_path(origin.empty() ? "<origin-miss>" : origin);
            nvbit_enable_instrumented(ctx, func, false);
            return;
        }

        if (!should_accept_origin(origin)) {
            g_skipped_launches.fetch_add(1, std::memory_order_relaxed);
            record_skipped_path(origin);
            nvbit_enable_instrumented(ctx, func, false);
            return;
        }

        // Kernel-name regex gate (sibling to module-origin gate above).
        // When METRIC_EXTRACTOR_KERNEL_NAME_REGEX is unset, this admits
        // every launch (byte-stable with the pre-filter producer). When
        // set, only launches whose demangled name matches the regex are
        // accepted; non-matching launches are skipped without consuming
        // a slot, preserving the Tier-2 buffer pool for the target
        // kernel even when prior PyTorch RNG / curand launches would
        // otherwise grab the only slot under the per-process budget cap.
        const char* demangled_c =
            nvbit_get_func_name(ctx, func, /*mangled=*/false);
        const std::string demangled_name =
            (demangled_c != nullptr) ? std::string(demangled_c)
                                     : std::string();
        if (!should_accept_kernel_name(demangled_name)) {
            g_skipped_launches.fetch_add(1, std::memory_order_relaxed);
            nvbit_enable_instrumented(ctx, func, false);
            return;
        }

        // AC-15 per-process budget skip path. Once the dense accepted
        // counter reaches ``g_max_launch_slots`` the slot pool is fully
        // consumed and we have nothing to point ``nvbit_set_at_launch``
        // at; instead of reusing slot 0 (which would corrupt an
        // already-flushed launch's data), run the kernel uninstrumented
        // and queue a bounded JSON-L record so the operator sees the
        // skipped launch explicitly. The kernel keeps running because
        // the user app would otherwise observe a missing kernel; the
        // metric_extractor's contract is to never break the host
        // application.
        if (g_accepted_slot_index >= g_max_launch_slots) {
            g_per_process_budget_skipped.fetch_add(
                1, std::memory_order_relaxed);
            nvbit_enable_instrumented(ctx, func, false);
            g_pending_launch_context = LaunchContext{};
            const char* demangled_skip =
                nvbit_get_func_name(ctx, func, /*mangled=*/false);
            g_pending_launch_context.kernel_name_demangled =
                (demangled_skip != nullptr) ? std::string(demangled_skip)
                                            : std::string();
            g_pending_launch_context.kernel_name_mangled = mangled_name;
            g_pending_launch_context.function_uid =
                nvbit_get_func_addr(ctx, func);
            g_pending_launch_context.ctx_id =
                reinterpret_cast<uintptr_t>(ctx);
            g_pending_launch_context.launch_index = observed_launch_index;
            g_pending_launch_context.slot_index = 0;
            g_pending_launch_context.launch_kind =
                "per_process_budget_exceeded";
            g_pending_launch_context.waiver_reason =
                "per_process_budget_exceeded";
            g_pending_launch_context.is_graph_node = false;
            g_pending_launch_context.is_cooperative =
                (cbid == API_CUDA_cuLaunchCooperativeKernel ||
                 cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz);
            for (int i = 0; i < 3; ++i) {
                g_pending_launch_context.grid_dim[i] = 0;
                g_pending_launch_context.block_dim[i] = 0;
                g_pending_launch_context.cluster_dim[i] = 1;
            }
            g_pending_launch_context.dynamic_smem = 0;
            // Fill grid/block/dynamic_smem/stream from params so the
            // identity slot still pins the launch the same way an
            // instrumented record would. This matters for AC-8: the
            // composite-join key still includes the launch shape on
            // the skipped record, so a downstream cross-surface
            // consumer can correlate the skipped record back to the
            // CUPTI Activity row it was supposed to match.
            if (cbid == API_CUDA_cuLaunchKernelEx ||
                cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
                auto* p = static_cast<cuLaunchKernelEx_params*>(params);
                if (p->config != nullptr) {
                    g_pending_launch_context.grid_dim[0] =
                        p->config->gridDimX;
                    g_pending_launch_context.grid_dim[1] =
                        p->config->gridDimY;
                    g_pending_launch_context.grid_dim[2] =
                        p->config->gridDimZ;
                    g_pending_launch_context.block_dim[0] =
                        p->config->blockDimX;
                    g_pending_launch_context.block_dim[1] =
                        p->config->blockDimY;
                    g_pending_launch_context.block_dim[2] =
                        p->config->blockDimZ;
                    g_pending_launch_context.dynamic_smem =
                        p->config->sharedMemBytes;
                    set_stream_identity(
                        g_pending_launch_context, ctx, p->config->hStream,
                        is_per_thread_default_launch(cbid));
                }
            } else if (cbid == API_CUDA_cuLaunchCooperativeKernel ||
                       cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
                auto* p = static_cast<
                    cuLaunchCooperativeKernel_params*>(params);
                g_pending_launch_context.grid_dim[0]  = p->gridDimX;
                g_pending_launch_context.grid_dim[1]  = p->gridDimY;
                g_pending_launch_context.grid_dim[2]  = p->gridDimZ;
                g_pending_launch_context.block_dim[0] = p->blockDimX;
                g_pending_launch_context.block_dim[1] = p->blockDimY;
                g_pending_launch_context.block_dim[2] = p->blockDimZ;
                g_pending_launch_context.dynamic_smem = p->sharedMemBytes;
                set_stream_identity(
                    g_pending_launch_context, ctx, p->hStream,
                    is_per_thread_default_launch(cbid));
            } else {
                auto* p = static_cast<cuLaunchKernel_params*>(params);
                g_pending_launch_context.grid_dim[0]  = p->gridDimX;
                g_pending_launch_context.grid_dim[1]  = p->gridDimY;
                g_pending_launch_context.grid_dim[2]  = p->gridDimZ;
                g_pending_launch_context.block_dim[0] = p->blockDimX;
                g_pending_launch_context.block_dim[1] = p->blockDimY;
                g_pending_launch_context.block_dim[2] = p->blockDimZ;
                g_pending_launch_context.dynamic_smem = p->sharedMemBytes;
                set_stream_identity(
                    g_pending_launch_context, ctx, p->hStream,
                    is_per_thread_default_launch(cbid));
            }
            g_last_launch_accepted = false;
            g_pending_budget_skipped_record = true;
            return;
        }

        // Accepted user kernel: instrument once, allocate slot, expose
        // slot pointer to NVBit's per-launch state machine, enable
        // instrumentation for this CUfunction. ORDER MATTERS: set the
        // launch value BEFORE re-enabling instrumentation so the
        // per-launch arg store is in place when NVBit promotes the
        // function to its instrumented variant; the upstream mem_trace
        // sample sets the launch value last with no explicit enable call.
        //
        // Slot indexing uses ``g_accepted_slot_index`` — the dense
        // accepted-only counter — so the slot pool stays compact
        // regardless of how many skipped launches we observed. The
        // dense slot index is also what the JSON-L emitter reads from
        // when it dereferences the per-launch device counters; see
        // ``LaunchContext::slot_index``.
        const uint64_t accepted_slot_index = g_accepted_slot_index;
        LaunchMetrics* slot = &g_slot_pool[accepted_slot_index];
        memset(slot, 0, sizeof(LaunchMetrics));
        // Wire the per-launch Tier-2 buffer pointer into the slot when
        // the env gate is on AND the parallel pool was successfully
        // allocated by nvbit_tool_init. Zeroing the buffer's header
        // counters here (rather than relying on a global memset) keeps
        // each launch's record sequence starting at zero independent
        // of how the previous launch terminated.
        if (g_tier2_pool != nullptr) {
            TierTwoBuffer* tier2 = &g_tier2_pool[accepted_slot_index];
            tier2->record_count = 0;
            tier2->sequence_id_counter = 0;
            tier2->dropped_count = 0;
            tier2->truncated = 0;
            slot->tier2_buffer_ptr = reinterpret_cast<uint64_t>(tier2);
        }

        instrument_function_if_needed(ctx, func);
        nvbit_set_at_launch(ctx, func, reinterpret_cast<uint64_t>(slot));
        nvbit_enable_instrumented(ctx, func, true);

        // Capture per-launch identity for the JSON-L emitter; the
        // matched launch-exit branch consumes this. Single static is
        // safe because the launch mutex serialises enter→exit.
        g_pending_launch_context = LaunchContext{};
        // Round 18 (Codex Round-17 review Finding 3): populate the
        // profiled-binary digest from the accepted origin. Env-var
        // override wins when explicitly set (test path); otherwise
        // hash the loaded .so once per origin (production path,
        // amortized via the per-origin cache).
        if (tier2_binary_sha256_env_explicitly_set()) {
            tier2_load_binary_sha256(g_pending_launch_context.binary_sha256);
            g_pending_launch_context.binary_sha256_valid = true;
        } else {
            g_pending_launch_context.binary_sha256_valid =
                tier2_load_binary_sha256_for_origin(
                    origin, g_pending_launch_context.binary_sha256);
        }
        const char* demangled =
            nvbit_get_func_name(ctx, func, /*mangled=*/false);
        g_pending_launch_context.kernel_name_demangled =
            (demangled != nullptr) ? std::string(demangled) : std::string();
        g_pending_launch_context.kernel_name_mangled = mangled_name;
        g_pending_launch_context.function_uid =
            nvbit_get_func_addr(ctx, func);
        g_pending_launch_context.ctx_id =
            reinterpret_cast<uintptr_t>(ctx);
        // The AC-8 ``kernel_ordinal_in_workload`` field uses the
        // observed-launch index captured at the top of this branch
        // (before the filter); the dense slot index is recorded
        // separately for the slot-pool lookups on launch-exit.
        g_pending_launch_context.launch_index = observed_launch_index;
        g_pending_launch_context.slot_index = g_accepted_slot_index;
        g_pending_launch_context.is_graph_node = false;
        g_pending_launch_context.is_cooperative =
            (cbid == API_CUDA_cuLaunchCooperativeKernel ||
             cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz);
        g_pending_launch_context.launch_kind =
            g_pending_launch_context.is_cooperative ? "cooperative"
                                                    : "regular";
        g_pending_launch_context.waiver_reason.clear();
        for (int i = 0; i < 3; ++i) {
            g_pending_launch_context.cluster_dim[i] = 1;
        }
        // Extract grid/block/sharedMem/stream from the params struct;
        // the cuLaunchKernelEx variant carries them inside CUlaunchConfig
        // (and may include cluster dims via the attributes array).
        if (cbid == API_CUDA_cuLaunchKernelEx ||
            cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
            auto* p = static_cast<cuLaunchKernelEx_params*>(params);
            if (p->config != nullptr) {
                g_pending_launch_context.grid_dim[0] = p->config->gridDimX;
                g_pending_launch_context.grid_dim[1] = p->config->gridDimY;
                g_pending_launch_context.grid_dim[2] = p->config->gridDimZ;
                g_pending_launch_context.block_dim[0] =
                    p->config->blockDimX;
                g_pending_launch_context.block_dim[1] =
                    p->config->blockDimY;
                g_pending_launch_context.block_dim[2] =
                    p->config->blockDimZ;
                g_pending_launch_context.dynamic_smem =
                    p->config->sharedMemBytes;
                set_stream_identity(
                    g_pending_launch_context, ctx, p->config->hStream,
                    is_per_thread_default_launch(cbid));
                if (p->config->attrs != nullptr) {
                    for (unsigned i = 0; i < p->config->numAttrs; ++i) {
                        if (p->config->attrs[i].id ==
                            CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
                            const auto& cd =
                                p->config->attrs[i].value.clusterDim;
                            g_pending_launch_context.cluster_dim[0] = cd.x;
                            g_pending_launch_context.cluster_dim[1] = cd.y;
                            g_pending_launch_context.cluster_dim[2] = cd.z;
                        }
                    }
                }
            }
        } else if (cbid == API_CUDA_cuLaunchCooperativeKernel ||
                   cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
            auto* p =
                static_cast<cuLaunchCooperativeKernel_params*>(params);
            g_pending_launch_context.grid_dim[0] = p->gridDimX;
            g_pending_launch_context.grid_dim[1] = p->gridDimY;
            g_pending_launch_context.grid_dim[2] = p->gridDimZ;
            g_pending_launch_context.block_dim[0] = p->blockDimX;
            g_pending_launch_context.block_dim[1] = p->blockDimY;
            g_pending_launch_context.block_dim[2] = p->blockDimZ;
            g_pending_launch_context.dynamic_smem = p->sharedMemBytes;
            set_stream_identity(
                g_pending_launch_context, ctx, p->hStream,
                is_per_thread_default_launch(cbid));
        } else {
            auto* p = static_cast<cuLaunchKernel_params*>(params);
            g_pending_launch_context.grid_dim[0] = p->gridDimX;
            g_pending_launch_context.grid_dim[1] = p->gridDimY;
            g_pending_launch_context.grid_dim[2] = p->gridDimZ;
            g_pending_launch_context.block_dim[0] = p->blockDimX;
            g_pending_launch_context.block_dim[1] = p->blockDimY;
            g_pending_launch_context.block_dim[2] = p->blockDimZ;
            g_pending_launch_context.dynamic_smem = p->sharedMemBytes;
            set_stream_identity(
                g_pending_launch_context, ctx, p->hStream,
                is_per_thread_default_launch(cbid));
        }

        g_accepted_launches.fetch_add(1, std::memory_order_relaxed);
        // Bump the dense accepted-slot counter so the next accepted
        // launch lands on a fresh slot. The observed-launch counter
        // ``g_launch_index`` was already advanced at the top of this
        // branch when the CBID first fired, regardless of whether
        // the filter eventually accepted the kernel.
        g_accepted_slot_index++;
        g_last_launch_accepted = true;
    } else {
        // Wait for any injected atomicAdds to land before any host-side
        // slot read. The slot read itself lands when LaunchMetrics has
        // counter fields; until then there is nothing to read but the
        // synchronisation barrier sits in source order so a future
        // counter-emit pass can read safely. Skip the sync when the
        // launch was filtered out — there is nothing in flight that the
        // host could race against.
        if (g_last_launch_accepted) {
            ME_CUDA_CHECK(cudaDeviceSynchronize());
            // Emit one JSON-L record per accepted launch. The metric
            // value fields are placeholders until SASS-walk counters
            // land; the identity, dims, status, and host-snapshot
            // fields are populated for real.
            write_launch_jsonl_record(g_pending_launch_context);
        } else if (g_pending_budget_skipped_record) {
            // Budget-skipped kernels ran uninstrumented (no
            // instrumentation arg was set), so no cudaDeviceSynchronize
            // is needed before emitting — there are no injected
            // atomicAdds in flight to race against. Emit the bounded
            // replacement record so the operator sees the skipped
            // launch explicitly.
            write_launch_jsonl_record(g_pending_launch_context);
            g_pending_budget_skipped_record = false;
        }
        pthread_mutex_unlock(&g_launch_mutex);
    }
}

// --------------------------------------------------------------------------
// Graph-node launch hook — fires per node when CUDA Graphs replay. Same
// accept/skip discipline as cuLaunchKernel but the per-node
// nvbit_set_at_launch overload takes the stream + launch_handle so
// instrumentation receives the right slot pointer for the right node.
// --------------------------------------------------------------------------

void nvbit_at_graph_node_launch(CUcontext ctx, CUfunction func,
                                CUstream stream, uint64_t launch_handle) {
    g_graph_node_launches.fetch_add(1, std::memory_order_relaxed);

    if (func == nullptr) {
        g_skipped_launches.fetch_add(1, std::memory_order_relaxed);
        g_graph_node_skipped.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const char* mangled_c =
        nvbit_get_func_name(ctx, func, /*mangled=*/true);
    const std::string mangled_name =
        (mangled_c != nullptr) ? std::string(mangled_c) : std::string();
    std::string origin;
    {
        std::lock_guard<std::mutex> lk(g_origin_mu);
        origin = lookup_func_origin_locked(ctx, func, mangled_name);
    }
    const bool origin_resolved = !origin.empty() && origin.front() != '<';
    if (!origin_resolved) {
        g_module_origin_misses.fetch_add(1, std::memory_order_relaxed);
        g_skipped_launches.fetch_add(1, std::memory_order_relaxed);
        g_graph_node_skipped.fetch_add(1, std::memory_order_relaxed);
        record_skipped_path(origin.empty() ? "<origin-miss>" : origin);
        nvbit_enable_instrumented(ctx, func, false);
        return;
    }
    if (!should_accept_origin(origin)) {
        g_skipped_launches.fetch_add(1, std::memory_order_relaxed);
        g_graph_node_skipped.fetch_add(1, std::memory_order_relaxed);
        record_skipped_path(origin);
        nvbit_enable_instrumented(ctx, func, false);
        return;
    }

    // Slot allocation + instrumentation pass needs g_launch_mutex
    // briefly because g_launch_index is shared with regular launches.
    // CRITICAL: g_launch_mutex MUST NOT be held across the
    // cuGraphLaunch boundary — handle_graph_launch_event() does NOT
    // acquire it. Holding it here briefly is safe because the per-node
    // hook fires synchronously on the calling thread; the cuGraphLaunch
    // dispatch itself does not acquire g_launch_mutex.
    LaunchMetrics* slot = nullptr;
    uint64_t my_launch_index = 0;
    if (g_slot_pool == nullptr || g_max_launch_slots == 0) {
        g_skipped_launches.fetch_add(1, std::memory_order_relaxed);
        g_graph_node_skipped.fetch_add(1, std::memory_order_relaxed);
        nvbit_enable_instrumented(ctx, func, false);
        return;
    }
    pthread_mutex_lock(&g_launch_mutex);
    // Graph-node accepted launches consume one observed-launch ordinal
    // AND one dense accepted-slot index, matching the regular-launch
    // accounting. The observed ordinal is recorded on the per-node
    // LaunchContext as ``launch_index`` (AC-8's
    // ``kernel_ordinal_in_workload``); the dense slot index is the
    // ring-buffer pointer the JSON-L emitter uses to read the
    // device-side counters back. Pre-Round-2 code reused
    // ``g_launch_index`` for both roles, which broke the AC-8
    // composite-key invariant against filtered framework kernels.
    //
    // AC-15 per-process budget skip path: once the slot pool is fully
    // consumed, accept the kernel without instrumenting it and queue
    // a bounded skipped record for the cuGraphLaunch-exit drain. The
    // emit deliberately routes through ``pending_node_records`` so
    // the existing graph-launch flush still owns the record's
    // ordering relative to other node launches on the same stream.
    if (g_accepted_slot_index >= g_max_launch_slots) {
        g_per_process_budget_skipped.fetch_add(
            1, std::memory_order_relaxed);
        my_launch_index = g_launch_index;
        g_launch_index++;
        pthread_mutex_unlock(&g_launch_mutex);
        nvbit_enable_instrumented(ctx, func, false);
        // Round 19 (Codex Round-18-review Finding 4): value-init the
        // skip-path LaunchContext so newly added fields (e.g.
        // binary_sha256, binary_sha256_valid) can't carry indeterminate
        // bytes if this record shape ever serializes a Tier-2 frame.
        // The branch currently emits only a bounded JSONL skipped
        // record, but the C++ hygiene is the same regardless.
        LaunchContext skip_lc{};
        const char* demangled_skip =
            nvbit_get_func_name(ctx, func, /*mangled=*/false);
        skip_lc.kernel_name_demangled =
            (demangled_skip != nullptr) ? std::string(demangled_skip)
                                        : std::string();
        skip_lc.kernel_name_mangled = mangled_name;
        skip_lc.function_uid = nvbit_get_func_addr(ctx, func);
        skip_lc.ctx_id = reinterpret_cast<uintptr_t>(ctx);
        set_stream_identity(skip_lc, ctx, stream);
        skip_lc.launch_index = my_launch_index;
        skip_lc.slot_index = 0;
        for (int i = 0; i < 3; ++i) {
            skip_lc.grid_dim[i] = 0;
            skip_lc.block_dim[i] = 0;
            skip_lc.cluster_dim[i] = 1;
        }
        skip_lc.dynamic_smem = 0;
        skip_lc.launch_kind = "per_process_budget_exceeded";
        skip_lc.is_graph_node = true;
        skip_lc.is_cooperative = false;
        skip_lc.waiver_reason = "per_process_budget_exceeded";
        {
            std::lock_guard<std::mutex> lk(g_graph_state_mu);
            auto it = g_pending_graph_states.find(GraphLaunchKey{ctx, stream});
            if (it != g_pending_graph_states.end()) {
                it->second.pending_node_records.push_back(
                    std::move(skip_lc));
            } else {
                write_launch_jsonl_record(skip_lc);
            }
        }
        return;
    }
    const uint64_t slot_index = g_accepted_slot_index;
    slot = &g_slot_pool[slot_index];
    memset(slot, 0, sizeof(LaunchMetrics));
    instrument_function_if_needed(ctx, func);
    my_launch_index = g_launch_index;
    g_launch_index++;
    const uint64_t my_slot_index = g_accepted_slot_index;
    g_accepted_slot_index++;
    pthread_mutex_unlock(&g_launch_mutex);

    // Set per-node launch arg BEFORE enabling instrumentation so the
    // per-launch arg store is in place when NVBit promotes the function
    // to its instrumented variant. The reverse order corrupts the
    // launch_val64 the device reads, which the regular-launch path
    // surfaced as a misaligned-address fault at sync time.
    nvbit_set_at_launch(ctx, func, reinterpret_cast<uint64_t>(slot),
                        stream, launch_handle);
    nvbit_enable_instrumented(ctx, func, true);

    g_accepted_launches.fetch_add(1, std::memory_order_relaxed);
    g_graph_node_accepted.fetch_add(1, std::memory_order_relaxed);

    // Build the LaunchContext but DEFER the JSON-L emit until
    // cuGraphLaunch-exit drains the per-stream pending queue. The
    // graph kernels have not actually executed yet — emitting the
    // record now would write before any future slot read could land
    // post-sync.
    LaunchContext lc{};
    // Round 18 (Codex Round-17 review Finding 3): same per-launch
    // binary digest population as the regular launch path. Graph-
    // node launches also need the schema-defined binary identity
    // since they emit the same Tier-2 frame shape downstream.
    if (tier2_binary_sha256_env_explicitly_set()) {
        tier2_load_binary_sha256(lc.binary_sha256);
        lc.binary_sha256_valid = true;
    } else {
        lc.binary_sha256_valid =
            tier2_load_binary_sha256_for_origin(origin, lc.binary_sha256);
    }
    const char* demangled =
        nvbit_get_func_name(ctx, func, /*mangled=*/false);
    lc.kernel_name_demangled =
        (demangled != nullptr) ? std::string(demangled) : std::string();
    lc.kernel_name_mangled = mangled_name;
    lc.function_uid = nvbit_get_func_addr(ctx, func);
    lc.ctx_id = reinterpret_cast<uintptr_t>(ctx);
    set_stream_identity(lc, ctx, stream);
    lc.launch_index = my_launch_index;
    lc.slot_index = my_slot_index;
    for (int i = 0; i < 3; ++i) {
        lc.grid_dim[i] = 0;
        lc.block_dim[i] = 0;
        lc.cluster_dim[i] = 1;
    }
    lc.dynamic_smem = 0;
    lc.launch_kind = "graph_node_traced";
    lc.is_graph_node = true;
    lc.is_cooperative = false;
    lc.waiver_reason.clear();

    {
        std::lock_guard<std::mutex> lk(g_graph_state_mu);
        auto it = g_pending_graph_states.find(GraphLaunchKey{ctx, stream});
        if (it != g_pending_graph_states.end()) {
            it->second.pending_node_records.push_back(std::move(lc));
        } else {
            // No active cuGraphLaunch boundary on this stream — the
            // per-node hook fired outside an enter/exit window we
            // observed. Fall back to immediate emit so the record is
            // not dropped on the floor; the graph_launch_not_observed
            // path won't fire in this case because the boundary is
            // missing entirely.
            write_launch_jsonl_record(lc);
        }
    }
}

// --------------------------------------------------------------------------
// Graph-launch substatus hooks — track per-cuGraphLaunch boundaries so a
// graph launch that completes WITHOUT any per-node hook firing produces
// a single JSON-L record marking the gap. Counterpart to the per-node
// hook above; together they cover both halves of the graph substatus
// contract (`traced` vs `not_observed`).
// --------------------------------------------------------------------------

static bool is_graph_launch_cbid(nvbit_api_cuda_t cbid) {
    return (cbid == API_CUDA_cuGraphLaunch) ||
           (cbid == API_CUDA_cuGraphLaunch_ptsz);
}

static void handle_graph_launch_event(CUcontext ctx, int is_exit,
                                      nvbit_api_cuda_t cbid,
                                      void* params) {
    // Extract the stream from the params struct once; both enter and
    // exit need it.
    CUstream stream = nullptr;
    if (cbid == API_CUDA_cuGraphLaunch ||
        cbid == API_CUDA_cuGraphLaunch_ptsz) {
        auto* p = static_cast<cuGraphLaunch_params*>(params);
        if (p != nullptr) {
            stream = p->hStream;
        }
    }

    if (!is_exit) {
        // CRITICAL: do NOT lock g_launch_mutex here. The per-node
        // hook (nvbit_at_graph_node_launch) fires from inside the
        // cuGraphLaunch driver call on the same thread; if we held
        // g_launch_mutex across the driver call, the per-node hook's
        // own brief acquisition of g_launch_mutex would deadlock
        // (the mutex is non-recursive). Use the dedicated
        // g_graph_state_mu for the per-(context, stream) state map.
        g_graph_launches_seen.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(g_graph_state_mu);
        GraphLaunchState state;
        state.snapshot_node_count =
            g_graph_node_launches.load(std::memory_order_relaxed);
        state.ctx = ctx;
        state.stream = stream;
        // Overwrite any stale entry on this stream (the prior
        // cuGraphLaunch on this stream should have drained at its
        // exit; if it didn't, dropping the stale snapshot is the
        // right recovery — we never hold pending records across
        // boundaries).
        g_pending_graph_states[GraphLaunchKey{ctx, stream}] = std::move(state);
        return;
    }

    // is_exit == true: drain the per-(context, stream) state. Take ownership of
    // the GraphLaunchState (lift it out of the map) before any
    // synchronization or file I/O so the next cuGraphLaunch on the
    // same stream sees a clean slot.
    GraphLaunchState lifted;
    bool had_state = false;
    {
        std::lock_guard<std::mutex> lk(g_graph_state_mu);
        auto it = g_pending_graph_states.find(GraphLaunchKey{ctx, stream});
        if (it != g_pending_graph_states.end()) {
            lifted = std::move(it->second);
            g_pending_graph_states.erase(it);
            had_state = true;
        }
    }
    if (!had_state) {
        // cuGraphLaunch exit fired without a matching enter — likely
        // a synthetic / out-of-band call. Nothing to drain.
        return;
    }

    // Synchronize the graph's stream BEFORE reading the per-node
    // delta or writing any per-node JSON-L record. cuGraphLaunch is
    // asynchronous; without this barrier a valid graph replay can
    // exit before the node-launch hooks have all completed and the
    // per-node counter delta would falsely look like 0. The legacy
    // nvbit_release/tools/mem_read_shared sample uses the same
    // discipline.
    if (stream != nullptr) {
        ME_CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        // Default-stream replays sync against the device.
        ME_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // The "graph launch not observed" condition is "NVBit's per-node
    // hook never fired during this cuGraphLaunch boundary" — NOT
    // "no node records were accepted". A graph that fired the per-
    // node hook for kernels that turned out to be filtered (origin
    // miss, regex non-match, slot pool unavailable) WAS observed by
    // NVBit; reporting it as `not_observed` would be wrong because
    // those nodes already counted toward g_skipped_launches /
    // g_graph_node_skipped via the per-node hook's filter path.
    //
    // The correct predicate is the per-node-counter delta against
    // the snapshot we captured on cuGraphLaunch-enter. The per-node
    // hook bumps g_graph_node_launches BEFORE any filter check, so
    // a non-zero delta means "NVBit observed at least one node",
    // independent of whether that node passed the filter.
    const uint64_t observed_delta =
        g_graph_node_launches.load(std::memory_order_relaxed) -
        lifted.snapshot_node_count;

    if (observed_delta == 0) {
        // Graph replay produced no per-node hook fires at all — emit
        // the documented `not_observed` JSON-L record with the
        // canonical waiver string. Allocate a launch_index for the
        // boundary record itself (under g_launch_mutex briefly so
        // the counter stays coherent with regular launches).
        g_graph_launches_not_observed.fetch_add(
            1, std::memory_order_relaxed);
        uint64_t my_launch_index = 0;
        pthread_mutex_lock(&g_launch_mutex);
        my_launch_index = g_launch_index;
        g_launch_index++;
        pthread_mutex_unlock(&g_launch_mutex);
        write_graph_launch_not_observed_record(
            ctx, stream, my_launch_index,
            cbid == API_CUDA_cuGraphLaunch_ptsz);
        return;
    }

    // observed_delta > 0: drain whatever per-node records were
    // accepted into pending_node_records. The empty-pending case here
    // is "NVBit observed graph nodes but every one was filtered out"
    // — emit nothing for the boundary record (the skipped counters in
    // filter_stats.json already account for the state). Otherwise,
    // each pending record becomes a graph_node_traced JSON-L line;
    // the cudaStreamSynchronize above guarantees any future metric-
    // counter reads from the slot see consistent values.
    for (const auto& lc : lifted.pending_node_records) {
        write_launch_jsonl_record(lc);
    }
}
