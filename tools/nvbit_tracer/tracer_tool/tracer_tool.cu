// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <algorithm>
#include <assert.h>
#include <atomic>
#include <bitset>
#include <climits>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <sched.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <regex>
#include <set>
#include <string_view>

// MOD. Begin. Enhanced Tracer
#include <execinfo.h> /* backtrace, backtrace_symbols_fd */
#include <unistd.h> /* STDOUT_FILENO */
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <sys/wait.h>
#include <signal.h>
#include <errno.h>

#include "../traces_enhanced/src/traced_execution.h"
#include "../traces_enhanced/src/string_utilities.h"
#include "../isa_def/trace_opcode.h"
#include "path_resolver.h"
#include "phase_record_writer.h"
#include "active_launch_gate.h"
#include "module_origin_resolver.h"
// MOD. End. Enhanced Tracer

#include "../traces_enhanced/pb_trace/include/trace.pb.h"
#include "../traces_enhanced/pb_trace/include/gpu_device.pb.h"
#include "../traces_enhanced/pb_trace/include/cuda_stream.pb.h"
#include "../traces_enhanced/pb_trace/include/kernel.pb.h"
#include "../traces_enhanced/pb_trace/include/threadblock.pb.h"
#include "../traces_enhanced/pb_trace/include/warp.pb.h"
#include "../traces_enhanced/pb_trace/include/instruction.pb.h"
#include "../traces_enhanced/pb_trace/include/address.pb.h"
#include "../traces_enhanced/pb_trace/include/dim3d.pb.h"

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* per-context allocation registry (host-only translation unit) */
#include "allocation_registry.h"

/* typed-exclusion aggregator for task11b producer-side sidecars */
#include "exclusion_aggregator.h"

/* contains definition of the inst_trace_t structure */
#include "common.h"

// CUDA 12+ assigns one program-unique stream ID to every driver/runtime
// stream. Keep this declaration explicit because NVBit's vendored CUDA
// headers do not expose every driver entry point on all toolkit builds.
extern "C" CUresult cuStreamGetId(CUstream hStream,
                                  unsigned long long *streamId);
extern "C" CUresult cuStreamGetId_ptsz(CUstream hStream,
                                        unsigned long long *streamId);

/* Header-only short-opcode classifiers for the phase-injection block. */
#include "phase_classifier.h"

/* Header-only SASS-opcode -> memory-operation-class classifier for
 * the per-instruction trace contract. The mirror enum's numeric
 * values must stay 1:1 with `dynamic_trace::MemoryOpClass` in
 * `instruction.proto`; the static_asserts below pin that mapping at
 * compile time so a future proto-enum reorder cannot silently shift
 * the wire encoding. */
#include "memory_op_class.h"
#include "memory_op_access_kind.h"

static_assert(
    static_cast<int>(memory_op_class_classifier::kUnspecified) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_UNSPECIFIED),
    "memory_op_class_classifier::kUnspecified must mirror MEMORY_OP_CLASS_UNSPECIFIED");
static_assert(
    static_cast<int>(memory_op_class_classifier::kGlobal) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_GLOBAL),
    "memory_op_class_classifier::kGlobal must mirror MEMORY_OP_CLASS_GLOBAL");
static_assert(
    static_cast<int>(memory_op_class_classifier::kShared) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_SHARED),
    "memory_op_class_classifier::kShared must mirror MEMORY_OP_CLASS_SHARED");
static_assert(
    static_cast<int>(memory_op_class_classifier::kLocal) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_LOCAL),
    "memory_op_class_classifier::kLocal must mirror MEMORY_OP_CLASS_LOCAL");
static_assert(
    static_cast<int>(memory_op_class_classifier::kTexture) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_TEXTURE),
    "memory_op_class_classifier::kTexture must mirror MEMORY_OP_CLASS_TEXTURE");
static_assert(
    static_cast<int>(memory_op_class_classifier::kSurface) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_SURFACE),
    "memory_op_class_classifier::kSurface must mirror MEMORY_OP_CLASS_SURFACE");
static_assert(
    static_cast<int>(memory_op_class_classifier::kAtomic) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ATOMIC),
    "memory_op_class_classifier::kAtomic must mirror MEMORY_OP_CLASS_ATOMIC");
static_assert(
    static_cast<int>(memory_op_class_classifier::kTma) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_TMA),
    "memory_op_class_classifier::kTma must mirror MEMORY_OP_CLASS_TMA");
static_assert(
    static_cast<int>(memory_op_class_classifier::kAsyncCopy) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_ASYNC_COPY),
    "memory_op_class_classifier::kAsyncCopy must mirror MEMORY_OP_CLASS_ASYNC_COPY");
static_assert(
    static_cast<int>(memory_op_class_classifier::kClusterDsmem) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_CLUSTER_DSMEM),
    "memory_op_class_classifier::kClusterDsmem must mirror MEMORY_OP_CLASS_CLUSTER_DSMEM");
static_assert(
    static_cast<int>(memory_op_class_classifier::kTmem) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_CLASS_TMEM),
    "memory_op_class_classifier::kTmem must mirror MEMORY_OP_CLASS_TMEM");

/* Header-only SASS-opcode -> access-kind classifier orthogonal to
 * the memory-op-class classifier. The kind dimension tags the
 * direction (load / store / atomic / async-copy); consumers use it
 * with the class dimension to distinguish write-allocate paths
 * (global / local store + atomic + load) from no-write-allocate
 * paths (shared store, async-copy destination side). The mirror
 * enum's numeric values must stay 1:1 with
 * `dynamic_trace::MemoryOpAccessKind` in `instruction.proto`; the
 * static_asserts below pin the mapping at compile time. */
static_assert(
    static_cast<int>(memory_op_access_kind_classifier::kUnspecified) ==
        static_cast<int>(
            dynamic_trace::MEMORY_OP_ACCESS_KIND_UNSPECIFIED),
    "memory_op_access_kind_classifier::kUnspecified must mirror MEMORY_OP_ACCESS_KIND_UNSPECIFIED");
static_assert(
    static_cast<int>(memory_op_access_kind_classifier::kLoad) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_ACCESS_KIND_LOAD),
    "memory_op_access_kind_classifier::kLoad must mirror MEMORY_OP_ACCESS_KIND_LOAD");
static_assert(
    static_cast<int>(memory_op_access_kind_classifier::kStore) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_ACCESS_KIND_STORE),
    "memory_op_access_kind_classifier::kStore must mirror MEMORY_OP_ACCESS_KIND_STORE");
static_assert(
    static_cast<int>(memory_op_access_kind_classifier::kAtomic) ==
        static_cast<int>(dynamic_trace::MEMORY_OP_ACCESS_KIND_ATOMIC),
    "memory_op_access_kind_classifier::kAtomic must mirror MEMORY_OP_ACCESS_KIND_ATOMIC");
static_assert(
    static_cast<int>(memory_op_access_kind_classifier::kAsyncCopy) ==
        static_cast<int>(
            dynamic_trace::MEMORY_OP_ACCESS_KIND_ASYNC_COPY),
    "memory_op_access_kind_classifier::kAsyncCopy must mirror MEMORY_OP_ACCESS_KIND_ASYNC_COPY");

#define TRACER_VERSION 5

/* Version of the trace-sampling counting/selective-dump capture passes; stamped
 * into trace_sampling_capture.json so a consumer never mixes pre/post-change
 * counting stats within a cell. */
#define TRACE_SAMPLING_PASS_VERSION 2

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static constexpr size_t MAX_PHASE_RECORD_BUFFER_BYTES = 64ULL * 1024ULL * 1024ULL;

int num_devices = 0;

// Fixed upper bound for all per-device process state. Every device identity is
// status-checked and range-checked before it can index these arrays.
static constexpr int TRACER_MAX_DEVICES = 32;

enum class RecvThreadState {
  WORKING,
  STOP,
  FINISHED,
};

struct CTXstate {
  /* context id */
  int id;

  /* Checked CUDA device ordinal that owns this context. Set exactly once in
   * init_context_state after cudaGetDeviceCount and cuCtxGetDevice both succeed;
   * every per-device array access derives from this immutable identity. */
  int device_id = -1;

  /* Channel used to communicate from GPU to CPU receiving thread */
  ChannelDev* channel_dev = nullptr;
  ChannelHost channel_host;
  bool instruction_channel_initialized = false;

  // Recv-thread lifecycle state:
  //   - main thread stores WORKING in nvbit_tool_init → init_context_state,
  //     BEFORE channel_host.init() spawns the recv thread, so the recv thread
  //     observes WORKING on entry to its loop;
  //   - main thread stores STOP in nvbit_at_ctx_term to request shutdown;
  //   - recv thread stores FINISHED at the bottom of recv_thread_fun after
  //     its per-context cleanup, signaling that main thread may destroy the
  //     channel and free the CTXstate.
  //
  // Previously declared as `volatile RecvThreadState`. `volatile` does not
  // provide inter-thread memory ordering or atomicity in C++: it only tells
  // the compiler not to cache the value in a register. Modern cores can
  // reorder loads and stores around `volatile` reads/writes, so the handshake
  // could be observed out of order, leading to hangs or use-after-free on
  // the CTXstate during termination.
  //
  // std::atomic with explicit acquire/release semantics gives a proper
  // synchronizes-with edge between the publishing store and the observing
  // load, which is what this handshake actually needs.
  std::atomic<RecvThreadState> recv_thread_done{RecvThreadState::STOP};

  // Per-context flush-handshake flag:
  //   - main thread stores true before launching flush_channel kernel to
  //     request the recv thread drain pending channel data and acknowledge;
  //   - recv thread stores false after it has drained the flush-sentinel
  //     record (cta_id_x == -1) from the channel;
  //   - main thread spin-loads until it sees false, then proceeds to
  //     serialize per-threadblock output.
  //
  // Previously lived in a global std::map<CUcontext, bool>, which is NOT
  // thread-safe for concurrent read/write, and the bool itself had no
  // atomicity guarantee. Moving it into CTXstate and making it atomic is
  // the minimal correct fix: every call site already has `ctx_state` in
  // scope (we look it up under mutex before any access), so no extra
  // indirection is needed.
  std::atomic<bool> recv_thread_receiving{false};

  // Identity of the accepted launch currently producing channel records.
  // These values are per-context because receiver threads can drain launches
  // from different contexts on the same device concurrently; a device-global
  // stream/kernel snapshot would let one context label another context's
  // records.
  std::atomic<int64_t> active_stream_id{-1};
  std::atomic<int> active_kernel_id{-1};

  // nvbit_tool_init can be entered by the first graph kernel while its stream
  // is still being captured. CUDA rejects the channel allocations in that
  // state, so initialization is deferred until the matching end-capture
  // callback succeeds.
  bool tool_init_deferred = false;
  bool tool_init_complete = false;

  // Immutable allocation view for the accepted launch. Channel delivery can
  // lag host API callbacks, so resolving addresses against the live registry
  // at receive time would let an async free/reuse relabel an older kernel.
  nvbit_tracer::AllocationSnapshot active_allocation_snapshot;
  bool active_allocation_snapshot_valid = false;

  // Next opt-in asynchronous-copy instrumentation-order token for the accepted
  // launch. The main callback resets it to one before publishing
  // recv_thread_receiving=true; the single receive thread then advances it only
  // when a selected packet creates a new protobuf instruction. The existing
  // release/acquire launch handshake orders reset before consumption.
  uint64_t next_async_instrumentation_order = 1;

  // Per-context device binary version (= `major*10 + minor` for the CUDA
  // device that owns this context). Resolved unconditionally in
  // nvbit_tool_init(ctx) so every context gets its own value, and
  // updated in the per-kernel-launch hook if cuFuncGetAttribute yields
  // a positive BINARY_VERSION. The process-global `binary_version` is
  // kept as a scratch value aligned with whichever context is currently
  // being consumed (synchronized at ctx-init, per-launch, and ctx-term
  // just before enhanced_tracer() reads it). Keeping the per-context
  // value here is what lets multi-context / multi-device traces honor
  // each context's own arch: without this, the first context's value
  // would be pinned for the whole process by the ctx-init's
  // `<= 0` guard.
  int binary_version = 0;

  /* Closure-plan NVBit phase-injection contract: per-context phase
   * channel for `record_phase_clock` records pushed by the injected
   * function in `inject_funcs.cu`. Allocated only when
   * `phase_injection_enabled` is true (env-gated activation switch);
   * stays nullptr otherwise so the existing instruction-trace
   * pipeline is unchanged. The phase channel is independent from
   * the protobuf instruction-trace channel above so the two byte
   * streams do not corrupt each other.
   */
  ChannelDev* phase_channel_dev = nullptr;
  ChannelHost phase_channel_host;
  std::atomic<RecvThreadState> phase_recv_thread_done{RecvThreadState::STOP};
  /* Generation counter for diagnostics. Main thread stores
   * `this_launch_id` here BEFORE pushing the launch-indexed flush
   * sentinel; the recv thread compares each sentinel's launch_index
   * against this value to diagnose any unexpected earlier-generation
   * sentinel. The actual
   * post-launch handshake uses `phase_completed_flush_launch_index`
   * below — making the *waited-on* state itself generation-valued
   * means a stale sentinel writing `completed = N` cannot satisfy
   * the host's wait for `completed == N+1` even under arbitrary
   * scheduling of the recv thread.
   */
  std::atomic<uint32_t> phase_expected_flush_launch_index{0};
  /* Per-context completed-flush handshake. Recv thread stores the
   * sentinel's `launch_index` into this atomic on every sentinel
   * it processes; main thread post-launch path waits for
   * `completed == this_launch_id`. Initial sentinel value
   * `UINT32_MAX` means "no flush completed yet"; that value is
   * outside the launch-index space and never equals a real
   * `this_launch_id`, so the first launch's wait does not satisfy
   * spuriously.
   */
  std::atomic<uint32_t> phase_completed_flush_launch_index{UINT32_MAX};
  /* Counters for diagnostics: how many stale (pre-current-launch)
   * sentinels and quarantined records the recv thread / snapshot
   * path has dropped. Incremented under no lock; readers tolerate
   * racy reads since the values are only used in WARNING logs.
   */
  std::atomic<uint64_t> phase_stale_sentinel_count{0};
  std::atomic<uint64_t> phase_quarantined_record_count{0};
  /* Accumulated phase_clock_record_t bytes for the current launch.
   * The phase recv thread appends drained bytes here; the main
   * thread snapshots and clears at each per-launch flush boundary
   * to write `phase_records_<repeat>_<launch>.bin` files into the
   * traces folder. Records are filtered by `launch_index` at
   * snapshot time so an unexpected earlier-generation record cannot
   * contaminate the current launch's evidence.
   */
  std::vector<char> phase_records_buffer;
  std::mutex phase_records_buffer_mutex;
};

[[noreturn]] static void fail_device_identity(const char *stage,
                                              const std::string &detail) {
  fprintf(stderr,
          "FATAL: checked CUDA device identity failed at %s: %s; refusing "
          "unchecked per-device state access.\n",
          stage, detail.c_str());
  fflush(stderr);
  _exit(EXIT_FAILURE);
}

static int checked_device_id_value(int device_id, const char *stage) {
  if (num_devices <= 0 || num_devices > TRACER_MAX_DEVICES || device_id < 0 ||
      device_id >= num_devices || device_id >= TRACER_MAX_DEVICES) {
    std::ostringstream detail;
    detail << "device_id=" << device_id << " num_devices=" << num_devices
           << " TRACER_MAX_DEVICES=" << TRACER_MAX_DEVICES;
    fail_device_identity(stage, detail.str());
  }
  return device_id;
}

static void initialize_or_validate_device_count(const char *stage) {
  int observed_num_devices = 0;
  const CUresult status = cuDeviceGetCount(&observed_num_devices);
  if (status != CUDA_SUCCESS || observed_num_devices <= 0 ||
      observed_num_devices > TRACER_MAX_DEVICES) {
    std::ostringstream detail;
    detail << "cuDeviceGetCount returned CUresult=" << static_cast<int>(status)
           << " observed_num_devices=" << observed_num_devices
           << " TRACER_MAX_DEVICES=" << TRACER_MAX_DEVICES;
    fail_device_identity(stage, detail.str());
  }
  if (num_devices == 0) {
    num_devices = observed_num_devices;
  } else if (num_devices != observed_num_devices) {
    std::ostringstream detail;
    detail << "observed_num_devices=" << observed_num_devices
           << " differs from retained num_devices=" << num_devices;
    fail_device_identity(stage, detail.str());
  }
}

static int checked_current_context_device_id(const CTXstate *ctx_state,
                                             const char *stage) {
  CUdevice observed_device = -1;
  const CUresult status = cuCtxGetDevice(&observed_device);
  if (status != CUDA_SUCCESS) {
    std::ostringstream detail;
    detail << "cuCtxGetDevice returned CUresult=" << static_cast<int>(status);
    fail_device_identity(stage, detail.str());
  }
  const int device_id =
      checked_device_id_value(static_cast<int>(observed_device), stage);
  if (ctx_state != nullptr && ctx_state->device_id >= 0 &&
      ctx_state->device_id != device_id) {
    std::ostringstream detail;
    detail << "current device_id=" << device_id
           << " differs from stored CTXstate::device_id="
           << ctx_state->device_id;
    fail_device_identity(stage, detail.str());
  }
  return device_id;
}

static int checked_stored_context_device_id(const CTXstate *ctx_state,
                                            const char *stage) {
  if (ctx_state == nullptr) {
    fail_device_identity(stage, "CTXstate is null");
  }
  return checked_device_id_value(ctx_state->device_id, stage);
}

/* Host-managed launch_index counter that the injected
 * `record_phase_clock` function reads at call time. Updated by the
 * main thread in the per-launch nvbit callback before the
 * receiving-flag release; the device-side function reads the
 * pointee with a volatile qualifier so per-launch updates are
 * observed correctly. Kept as a __managed__ scalar so device
 * loads are coherent across the host-device boundary without
 * explicit cudaMemcpy synchronization.
 */
__managed__ uint32_t simbench_phase_launch_index_for_inject = 0;

/* lock */
pthread_mutex_t mutex;

// Separate lock that specifically protects cross-thread writes to the
// process-global trace containers: `dyn_trace` (protobuf), `threadblocks`
// (unordered_map), `remaining_injects_to_current_instruction`
// (unordered_map), and `pc_to_opcode` (map).
//
// Why a SECOND mutex instead of reusing `mutex`:
//   - The main thread holds `mutex` for the duration of nvbit_at_cuda_event,
//     including the spin-wait on recv_thread_receiving=false (see
//     nvbit_at_cuda_event at :1276-1280 and nvbit_at_ctx_term similarly).
//     If the recv thread tried to acquire `mutex` to touch containers, the
//     main thread would still be holding it, so the recv thread would block
//     forever and the handshake would deadlock.
//   - `recv_container_mutex` is acquired ONLY in short, bounded critical
//     sections: around the recv thread's batch-processing body (released
//     during its own wait-for-receiving-true spin) and around each of the
//     main thread's discrete container-write call sites. The main thread
//     is NEVER spinning on a condition while holding this mutex, so the
//     recv thread's requests always eventually succeed.
//
// Lock order (to prevent inversion): `mutex` is always acquired BEFORE
// `recv_container_mutex` when the main thread takes both. The recv thread
// only ever takes `recv_container_mutex` (it does not take `mutex` after
// its initial ctx_state lookup at :1421), so the order is consistent.
//
// This fixes the multi-context races Codex flagged:
//   - Two different contexts' recv threads both writing to `dyn_trace` /
//     `threadblocks` concurrently.
//   - A recv thread of one context writing to `dyn_trace` while another
//     context's main thread is also writing (the existing `mutex` only
//     serializes main-thread callbacks amongst themselves, not against
//     recv threads from other contexts).
std::mutex recv_container_mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

// Number of active stream captures per context. The callback map lets
// nvbit_tool_init defer CUDA allocations without guessing which stream is
// current; nested captures are released one-for-one at successful end-capture.
std::unordered_map<CUcontext, unsigned> active_capture_counts;

// Pool handles are opaque process-local objects. Cache their location at
// creation/query time so an allocation made while a stream is being captured
// can be attributed without issuing capture-invalidating pool queries.
struct MemoryPoolIdentity {
  CUmemLocationType location_type = CU_MEM_LOCATION_TYPE_DEVICE;
  int location_id = -1;
};
std::unordered_map<CUmemoryPool, MemoryPoolIdentity> memory_pool_identity_map;

// Resolve the driver-assigned stream ID carried on the trace wire. Resolve on
// every accepted launch rather than caching by the
// opaque handle: CUDA may recycle a destroyed handle, and per-thread default
// streams can share a sentinel handle while representing different streams.
extern bool skip_callback_flag;

static int64_t stream_ordinal_for(CUstream stream, bool per_thread_default) {
  const CUstream identity_stream =
      per_thread_default && stream == nullptr ? CU_STREAM_PER_THREAD : stream;
  unsigned long long driver_stream_id = 0;
  const bool previous_skip = skip_callback_flag;
  skip_callback_flag = true;
  const CUresult stream_id_status = per_thread_default
      ? cuStreamGetId_ptsz(identity_stream, &driver_stream_id)
      : cuStreamGetId(identity_stream, &driver_stream_id);
  skip_callback_flag = previous_skip;

  if (stream_id_status != CUDA_SUCCESS ||
      driver_stream_id >
          static_cast<unsigned long long>(std::numeric_limits<int64_t>::max())) {
    std::ostringstream detail;
    detail << "cuStreamGetId status=" << static_cast<int>(stream_id_status)
           << " stream_id=" << driver_stream_id;
    fail_device_identity("stream identity", detail.str());
  }
  const int64_t ordinal = static_cast<int64_t>(driver_stream_id);
  return ordinal;
}

// Stream-ordered pool pointers are process-wide under the 64-bit unified
// addressing contract. Verify that the device identity used by the registry
// is suitable for this process before publishing an allocation.
static int checked_uva_device_ordinal(int device_ordinal, const char *stage) {
  if (sizeof(void *) < sizeof(uint64_t)) {
    fail_device_identity(stage,
                         "context-independent allocations require a 64-bit process");
  }
  const int checked_ordinal =
      checked_device_id_value(device_ordinal, stage);
  int unified_addressing = 0;
  const bool skip_before_attribute = skip_callback_flag;
  skip_callback_flag = true;
  const CUresult attribute_status = cuDeviceGetAttribute(
      &unified_addressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
      checked_ordinal);
  skip_callback_flag = skip_before_attribute;
  if (attribute_status != CUDA_SUCCESS || unified_addressing == 0) {
    std::ostringstream detail;
    detail << "device " << checked_ordinal
           << " does not report unified addressing (CUresult="
           << static_cast<int>(attribute_status)
           << ", value=" << unified_addressing << ")";
    fail_device_identity(stage, detail.str());
  }
  return checked_ordinal;
}

// CUDA graph capture returns a pointer before the allocation node executes,
// but pointer attributes for that graph-owned pointer are invalid until a graph
// launch. Querying cuStreamIsCapturing is a non-enqueueing operation and is
// safe from this callback when re-entry is suppressed.
static bool stream_capture_active(CUstream stream, bool per_thread_default) {
  const CUstream identity_stream =
      per_thread_default && stream == nullptr ? CU_STREAM_PER_THREAD : stream;
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  const bool previous_skip = skip_callback_flag;
  skip_callback_flag = true;
  const CUresult status =
      cuStreamIsCapturing(identity_stream, &capture_status);
  skip_callback_flag = previous_skip;
  return status == CUDA_SUCCESS &&
         capture_status == CU_STREAM_CAPTURE_STATUS_ACTIVE;
}

// Resolve a captured pool allocation from metadata cached when its pool was
// created or returned by a device-pool query. cuMemPoolGetAttribute is not
// capture-safe (on CUDA 13.3 it returns CUDA_ERROR_NOT_PERMITTED and
// invalidates the capture), so unknown pools deliberately remain unresolved.
static bool cached_pool_device_ordinal(CUmemoryPool pool, int *device_ordinal) {
  if (pool == nullptr || device_ordinal == nullptr) return false;
  auto it = memory_pool_identity_map.find(pool);
  if (it == memory_pool_identity_map.end() ||
      it->second.location_type != CU_MEM_LOCATION_TYPE_DEVICE ||
      it->second.location_id < 0) {
    return false;
  }
  *device_ordinal = it->second.location_id;
  return true;
}

static int allocation_device_ordinal_or_fail(
    CUdeviceptr ptr, CUmemoryPool pool, const CTXstate *ctx_state,
    CUstream stream, bool per_thread_default, const char *stage) {
  if (stream_capture_active(stream, per_thread_default)) {
    int cached_device = -1;
    if (pool != nullptr && cached_pool_device_ordinal(pool, &cached_device)) {
      return checked_uva_device_ordinal(cached_device, stage);
    }
    // cuMemAllocAsync uses the current pool of the submitting stream's device.
    // A captured default allocation therefore has a checked context identity;
    // for an explicit unknown pool we fail closed at the caller instead of
    // assigning a potentially wrong cross-device identity.
    if (pool == nullptr) {
      return checked_uva_device_ordinal(
          checked_stored_context_device_id(ctx_state, stage), stage);
    }
    return -1;
  }

  // Outside capture the driver has materialized the allocation, and its
  // pointer attribute is the authoritative device even for a pool on a
  // different device than the submitting stream.
  if (sizeof(void *) < sizeof(uint64_t)) {
    fail_device_identity(stage,
                         "context-independent allocations require a 64-bit process");
  }
  int device_ordinal = -1;
  const bool previous_skip = skip_callback_flag;
  skip_callback_flag = true;
  const CUresult status = cuPointerGetAttribute(
      &device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr);
  skip_callback_flag = previous_skip;
  if (status != CUDA_SUCCESS) {
    std::ostringstream detail;
    detail << "cuPointerGetAttribute(device ordinal) returned CUresult="
           << static_cast<int>(status) << " ptr=0x" << std::hex
           << static_cast<uint64_t>(ptr);
    fail_device_identity(stage, detail.str());
  }
  return checked_uva_device_ordinal(device_ordinal, stage);
}

static CTXstate *lookup_context_state_or_fail(CUcontext ctx,
                                               const char *stage) {
  auto it = ctx_state_map.find(ctx);
  if (it == ctx_state_map.end() || it->second == nullptr) {
    std::ostringstream detail;
    detail << "context handle " << ctx << " is not registered";
    fail_device_identity(stage, detail.str());
  }
  return it->second;
}

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call.
 *
 * skip_callback_flag is accessed only under the global recursive mutex above,
 * so a plain bool is sufficient — the mutex provides the memory ordering.
 *
 * is_first_init_context_call is also accessed only under the mutex
 * (via init_context_state called from nvbit_tool_init at :1599-1602).
 *
 * The old global `std::map<CUcontext, bool> recv_thread_receiving` has been
 * removed: the handshake flag is now `std::atomic<bool> recv_thread_receiving`
 * inside CTXstate, so each context owns its own atomic and there is no
 * shared-map contention between main and recv threads. */
bool skip_callback_flag = false;
bool is_first_init_context_call = true;
bool *stop_report;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int intermediate_extra_files_persistance = 0;
// Skip enhanced_tracer() (the post-execution cuobjdump/nvdisasm/parse_sass
// pass that produces extra_info/enhanced_execution_info.json + extra_info/cubin/
// + extra_info/sass/ + extra_info/register_usage/). Set SKIP_ENHANCED_TRACER=1
// to bypass this pass entirely; the primary trace artifacts
// (dynamic_trace.pb, threadblocks/device_*/stream_*/kernel_*/d_*.pb,
// stats.csv) are unaffected because they are produced before this point.
// Useful when the host process otherwise hangs in cubin parsing on a
// large multi-kernel CUTLASS-instantiated cubin (~169 KB / ~2,200 SASS
// lines vs. typical inline-PTX cubins of <5 KB / ~30 SASS lines). The
// downstream trace artifact verifier does not consult any extra_info/
// outputs, so skipping the pass is contract-safe.
int skip_enhanced_tracer = 0;
// Per-subprocess deadline (seconds) for the cuobjdump / nvdisasm shell-outs
// inside the enhanced_tracer per-cubin loop. The fork+exec helper polls
// waitpid(WNOHANG) against a monotonic deadline; on timeout it SIGTERMs the
// child process group, sleeps briefly, then SIGKILLs the group as a
// fallback. No global signal handler, no longjmp.
//
// Default 30 seconds — generous for typical inline-PTX cubins (<200ms in
// practice) and tight enough to abandon a misbehaving shell child without
// pinning the host process. Set to 0 to disable the timeout (the helper
// then waits indefinitely; not recommended for suites that contain
// CUTLASS-instantiated kernels).
int enhanced_tracer_per_cubin_timeout_s = 30;
// Maximum size (in KB) of the per-cubin .sass / .rfu files that
// parse_sass() / parse_rfu() will accept. When either generated file
// exceeds the threshold (or fails to be produced), both parsers are
// skipped for that cubin and a typed
// `enhanced_tracer_cubin_parse_timeout` diagnostic is emitted. The
// inline-PTX kernels in the LCD suite produce <10 KB SASS / <50 KB RFU;
// CUTLASS-instantiated Conv2dFprop produces ~170 KB SASS, well above
// the default 64 KB budget. The threshold gives a deterministic
// normal-return code path without any signal-driven control flow.
int enhanced_tracer_max_parser_input_kb = 64;
int gather_registers = 0;
int enable_compress = 1;
int print_core_id = 0;
int exclude_pred_off = 1;
int active_from_start = 1;
/* used to select region of interest when active from start is 0 */
bool active_region = true;

/* Optional exact-cardinality capture contract.  When
 * MAX_ACCEPTED_ACTIVE_LAUNCHES=1, only a launch whose entry occurs inside the
 * cuProfilerStart/cuProfilerStop region can be accepted, later launches in
 * that region run uninstrumented, and finalization rejects a zero-launch or
 * multi-accepted capture. The default (0 / unset) preserves legacy
 * multi-launch behavior. */
uint64_t max_accepted_active_launches = 0;
nvbit_tracer::ActiveLaunchGate active_launch_gate;
std::atomic<uint64_t> inactive_channel_records_dropped{0};
std::atomic<uint64_t> accepted_channel_records{0};

struct PendingAcceptedLaunch {
  bool accepted = false;
  uint64_t total_insts_at_entry = 0;
  uint64_t reported_insts_at_entry = 0;
  int device_id = -1;
  uint32_t trace_launch_id = 0;
};

/* CUDA launch entry and exit callbacks run on the calling host thread.  Keep
 * the pairing state thread-local so an inactive launch's exit cannot consume
 * another thread's accepted state. The strict exact-one gate runs later
 * process-wide active launches uninstrumented, so their capture paths never
 * overlap the selected launch. */
thread_local PendingAcceptedLaunch pending_accepted_launch;

namespace {

struct ModuleKey {
  CUcontext ctx;
  CUmodule module;
  bool operator==(const ModuleKey& other) const {
    return ctx == other.ctx && module == other.module;
  }
};

struct PointerPairHash {
  template <typename Key>
  size_t operator()(const Key& key) const noexcept {
    return std::hash<void*>()(key.ctx) ^
           (std::hash<void*>()(key.module) << 1);
  }
};

struct AcceptedModuleMapping {
  uint64_t accepted_launch_index = 0;
  int device_id = -1;
  int64_t stream_id = -1;
  uint32_t trace_launch_id = 0;
  std::string kernel_name;
  std::string resolution_source;
  nvbit_tracer::ModuleOriginEvidence origin;
};

std::unordered_map<ModuleKey, std::string, PointerPairHash>
    loaded_module_origins;
// CUDA libraries are context-independent. Their lifetime ends at library
// unload, not at the destruction of the context that observed the load.
std::unordered_map<CUlibrary, std::string> loaded_library_origins;
// A caller may convert CUkernel to a context-bound CUfunction before launch.
// Retain the observed exact module/library association, never a symbol name.
std::unordered_map<ModuleKey, CUlibrary, PointerPairHash> loaded_library_modules;
std::vector<AcceptedModuleMapping> accepted_module_mappings;

}  // namespace

/* Allocation/range identity registry. The actual data
 * structures and lookup/insert/free helpers live in
 * `allocation_registry.h` / `allocation_registry.cc` (a
 * host-only translation unit with no NVBit / CUDA
 * dependency, so the cross-context isolation invariants are
 * unit-testable on hosts without a GPU toolchain). The
 * registry is keyed per-context: live intervals form a
 * nested `std::map<CUcontext, std::map<uint64_t, uint32_t>>`
 * so two contexts with overlapping virtual address ranges
 * cannot alias and a free in one context cannot perturb
 * another's intervals. Allocation IDs are globally
 * monotonic across all contexts. Context IDs are also
 * globally monotonic, assigned at first registration on a
 * previously-unseen context handle. CUcontext is a pointer
 * typedef so the implicit pointer-to-`void *` conversion
 * carries it through the registry's API safely.
 *
 * Wrapper at the bottom of this block (`write_allocation_sidecar`)
 * opens `<traces_path>/extra_info/allocations.json`,
 * delegates to `nvbit_tracer::serialize_history_json` for
 * the JSON payload, and logs the entry count. Kept here
 * (rather than in the registry .cc) because the sidecar
 * directory creation + log line are NVBit-tracer-side
 * concerns the registry should not know about. */
void create_folder(const char *folder_path);  // forward decl
static bool write_allocation_sidecar(
    const std::string &dir_path) {
  create_folder(dir_path.c_str());
  std::string filename = dir_path + "/allocations.json";
  std::ofstream ofs(filename);
  if (!ofs) {
    std::cerr << "Failed to open " << filename
              << " for writing." << std::endl;
    return false;
  }
  nvbit_tracer::serialize_history_json(ofs);
  ofs.flush();
  if (!ofs) {
    std::cerr << "Failed while writing " << filename << std::endl;
    return false;
  }
  ofs.close();
  if (!ofs) {
    std::cerr << "Failed to close " << filename << std::endl;
    return false;
  }
  std::size_t count =
      nvbit_tracer::snapshot_history().size();
  std::cout << "Serialized allocation registry to " << filename
            << " (" << count << " entries)"
            << std::endl;
  return true;
}

/* Closure-plan NVBit phase-injection contract: env-gated activation
 * switch (read once from `SIMBENCH_NVBIT_PHASE_INJECT` in
 * `nvbit_at_init`). False by default; when true the tracer
 * initializes a per-context phase-channel and injects %clock64
 * reads at semantic phase boundaries. The actual injection-call
 * insertion at boundary classes (function entry/exit, basic-block
 * entry, bar.sync) sequences in a subsequent round so the existing
 * instruction-trace pipeline is not perturbed by foundational
 * wiring alone.
 */
bool phase_injection_enabled = false;
// Opt-in selected asynchronous-copy instruction numbering. This is a host-side
// observation of accepted main-channel instruction creation order only: it adds
// no device call, counter, clock read, atomic, or channel record.
bool async_order_enabled = false;
/* Zero-based A/B repeat identity supplied by the retained Python orchestrator.
 * Each injected subprocess is a fresh process, so this immutable process-local
 * value prevents later repeats from overwriting phase_records_0_<launch>.bin. */
uint32_t phase_repeat_index = 0;

/* Descriptor-relative authority for phase-record leaves. Opened once from the
 * resolved TRACES_FOLDER before the first launch can be admitted. The helper
 * owns and revalidates the retained directory inode, enforces a single GPU
 * device for the unqualified canonical filename namespace, and performs the
 * host-only exclusive/durable publication contract. */
simbench_nvbit::PhaseRecordWriter phase_record_writer;

/* Iteration cap for the per-launch phase-channel sentinel spin and the
 * per-context phase teardown spin. Configurable via env-var
 * `SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS` (default 1e7). The returned
 * value is always strictly positive: empty, non-numeric, negative,
 * and zero values warn and fall back to the default. This guarantees
 * both spin sites stay bounded — a stalled receive thread cannot
 * wedge the host indefinitely regardless of misconfiguration. The
 * function is cached on first call so multi-launch programs do not
 * re-parse the env-var on every per-launch boundary.
 */
static long long phase_flush_timeout_iters() {
  static long long val = []() {
    const long long kDefault = 10000000;
    const char *env = std::getenv("SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS");
    if (env != nullptr && env[0] != '\0') {
      char *endp = nullptr;
      errno = 0;
      long long parsed = std::strtoll(env, &endp, 10);
      if (errno == 0 && endp != env && *endp == '\0' && parsed > 0) {
        return parsed;
      }
      fprintf(stderr,
              "WARNING: SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS=%s is not a "
              "strictly positive integer; falling back to default "
              "%lld to keep the bounded-spin guarantee intact.\n",
              env, kDefault);
    }
    return kDefault;
  }();
  return val;
}

static int checked_sampling_scope_int(uint64_t value, const char *field) {
  if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    fprintf(stderr,
            "FATAL: sampling scope %s=%" PRIu64
            " exceeds the signed metadata domain\n",
            field, value);
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

static int checked_kernel_launch_id(int value) {
  if (value <= 0 || value >= INT_MAX) {
    fprintf(stderr,
            "FATAL: per-device kernel launch id %d cannot be advanced "
            "without leaving the positive metadata domain\n",
            value);
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
  return value;
}

static int next_candidate_function_id_or_fail(int current) {
  if (current < 0 || current >= INT_MAX - 1) {
    fprintf(stderr,
            "FATAL: static function identity counter %d cannot be advanced "
            "without leaving the positive int32 metadata domain\n",
            current);
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
  return current + 1;
}

static void require_record_alignment(uint32_t byte_count,
                                     size_t record_size,
                                     const char *channel) {
  if (record_size == 0 || byte_count % record_size != 0) {
    fprintf(stderr,
            "FATAL: %s channel returned %" PRIu32
            " bytes, not a whole-record multiple of %zu\n",
            channel, byte_count, record_size);
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
}

/* Should we terminate the program once we are done tracing? */
int terminate_after_limit_number_of_kernels_reached = 0;
int user_defined_folders = 0;

/* Trace-sampling capture modes for budget-breaching cells: 0=off, 1=counting-
 * only (per-block stats, no address payload), 2=selective-dump (only the chosen
 * whole blocks are written; counting stays active for EVERY block so the exact
 * auxiliary totals survive). The selective-dump gate never rewrites gridDim. */
int trace_sampling_mode = 0;
std::string trace_sampling_sidecar_path;
std::unordered_set<long> trace_sampling_selected_blocks;

struct TsBlockStat {
  long records = 0;
  int cta_x = 0, cta_y = 0, cta_z = 0;
  int sm_id = -1;
  int max_warpid_sm = -1;
  long atomic_records = 0;
  uint64_t sector_accesses = 0;
  uint64_t serialized_bytes = 0;
  /* The channel emits one record per injected operand, while the protobuf
   * merges all of an instruction's operand records into one instruction row.
   * Keep the same per-warp countdown here so `records` has identical semantics
   * in counting-only/selective and full-trace decoding. */
  std::map<int, uint32_t> remaining_injects_by_warp;
  std::map<int, bool> atomic_seen_by_warp;
};
/* Per-block counting stats (keyed by the threadblock string id) + a bounded
 * atomic-sector hot-line sketch. Guarded by recv_container_mutex like the
 * threadblocks map they parallel. */
std::map<std::string, TsBlockStat> ts_block_stats;
std::map<uint64_t, uint64_t> ts_atomic_sectors;
bool ts_atomic_sectors_overflow = false;
static const size_t TS_ATOMIC_SECTORS_CAP = 1u << 20;
/* A malformed width or arithmetic overflow must not be serialized as a
 * plausible zero-sector result.  The v2 consumer requires this marker to be
 * true, so an invalid producer observation fails closed. */
bool ts_sector_accesses_valid = true;
bool ts_serialized_bytes_valid = true;
/* Trace-sampling scoping: the label profiles ONE launch (the first), so the
 * counting + selective-dump modes are scoped to the FIRST (device, stream,
 * kernel) launch seen by the receiver — every later launch is ignored so the
 * gate totals + selected payload describe exactly that launch, mirroring the
 * consumers' first_launch_dir convention. -1 == unset. */
int ts_scope_device = -1;
int64_t ts_scope_stream = -1;
int ts_scope_kernel = -1;
/* The scoped launch's grid/block dims, captured at its launch entry (where the
 * dims are authoritative from the launch params) so the sidecar is shape-bound;
 * a consumer rejects a sidecar whose shape != dynamic_trace.pb's first launch. */
int ts_scope_grid_x = -1, ts_scope_grid_y = -1, ts_scope_grid_z = -1;
int ts_scope_block_x = -1, ts_scope_block_y = -1, ts_scope_block_z = -1;
bool ts_scope_dims_captured = false;

void create_folder(const char *folder_path);  // forward decl

/* Trace-sampling capture sidecar: per-block counting stats + the bounded atomic
 * hot-line sketch + the selected-block set (selective-dump). Modeled on
 * write_allocation_sidecar; a no-op when trace sampling is off. Block cta coords
 * are emitted verbatim (the consumer flattens with the true grid_dim from the
 * launch record). */
static bool write_trace_sampling_sidecar(const std::string &dir_path) {
  if (trace_sampling_mode == 0) return true;
  create_folder(dir_path.c_str());
  std::string filename = trace_sampling_sidecar_path.empty()
                             ? dir_path + "/trace_sampling_capture.json"
                             : trace_sampling_sidecar_path;
  std::ofstream ofs(filename);
  if (!ofs) {
    std::cerr << "Failed to open " << filename << " for writing." << std::endl;
    return false;
  }
  uint64_t atomic_total = 0;
  std::vector<uint64_t> atomic_counts;
  atomic_counts.reserve(ts_atomic_sectors.size());
  for (const auto &kv : ts_atomic_sectors) {
    const uint64_t count = kv.second;
    if (atomic_total > std::numeric_limits<uint64_t>::max() - count) {
      std::cerr << "Trace-sampling atomic-sector total overflowed"
                << std::endl;
      return false;
    }
    atomic_total += count;
    atomic_counts.push_back(count);
  }
  std::sort(atomic_counts.rbegin(), atomic_counts.rend());
  uint64_t atomic_top256 = 0;
  for (size_t i = 0; i < atomic_counts.size() && i < 256; ++i) {
    if (atomic_top256 > std::numeric_limits<uint64_t>::max() - atomic_counts[i]) {
      std::cerr << "Trace-sampling atomic top-256 total overflowed"
                << std::endl;
      return false;
    }
    atomic_top256 += atomic_counts[i];
  }
  if (trace_sampling_mode == 2) {
    for (const auto &kv : ts_block_stats) {
      if (kv.second.records > 0 && kv.second.serialized_bytes == 0) {
        ts_serialized_bytes_valid = false;
        break;
      }
    }
  }
  for (const auto &kv : ts_block_stats) {
    for (const auto &warp : kv.second.remaining_injects_by_warp) {
      if (warp.second != 0) {
        ts_sector_accesses_valid = false;
        break;
      }
    }
    if (!ts_sector_accesses_valid) break;
  }
  ofs << "{\n";
  ofs << "  \"mode\": " << trace_sampling_mode << ",\n";
  ofs << "  \"tracer_version\": " << TRACER_VERSION << ",\n";
  ofs << "  \"stats_pass_version\": " << TRACE_SAMPLING_PASS_VERSION << ",\n";
  ofs << "  \"launch\": {\"device\": " << ts_scope_device << ", \"stream\": "
      << ts_scope_stream << ", \"kernel\": " << ts_scope_kernel << "},\n";
  ofs << "  \"grid_dim\": {\"x\": " << ts_scope_grid_x << ", \"y\": " << ts_scope_grid_y
      << ", \"z\": " << ts_scope_grid_z << "},\n";
  ofs << "  \"block_dim\": {\"x\": " << ts_scope_block_x << ", \"y\": " << ts_scope_block_y
      << ", \"z\": " << ts_scope_block_z << "},\n";
  ofs << "  \"atomic_sector_total\": " << atomic_total << ",\n";
  ofs << "  \"atomic_top256_touches\": " << atomic_top256 << ",\n";
  ofs << "  \"atomic_sectors_overflow\": "
      << (ts_atomic_sectors_overflow ? "true" : "false") << ",\n";
  ofs << "  \"sector_accesses_valid\": "
      << (ts_sector_accesses_valid ? "true" : "false") << ",\n";
  ofs << "  \"serialized_bytes_exact\": "
      << ((trace_sampling_mode == 2 && ts_serialized_bytes_valid) ? "true" : "false")
      << ",\n";
  ofs << "  \"blocks\": [";
  bool first = true;
  for (const auto &kv : ts_block_stats) {
    const TsBlockStat &s = kv.second;
    if (!first) ofs << ",";
    first = false;
    ofs << "\n    {\"cta\": [" << s.cta_x << ", " << s.cta_y << ", " << s.cta_z
        << "], \"records\": " << s.records << ", \"sm_id\": " << s.sm_id
        << ", \"max_warpid_sm\": " << s.max_warpid_sm
        << ", \"atomic_records\": " << s.atomic_records
        << ", \"sector_accesses\": " << s.sector_accesses
        << ", \"serialized_bytes\": " << s.serialized_bytes << "}";
  }
  ofs << (ts_block_stats.empty() ? "" : "\n  ") << "],\n";
  ofs << "  \"selected_block_ids\": [";
  {
    std::vector<long> sel(trace_sampling_selected_blocks.begin(),
                          trace_sampling_selected_blocks.end());
    std::sort(sel.begin(), sel.end());
    for (size_t i = 0; i < sel.size(); ++i) {
      if (i) ofs << ", ";
      ofs << sel[i];
    }
  }
  ofs << "]\n}\n";
  ofs.flush();
  if (!ofs) {
    std::cerr << "Failed while writing " << filename << std::endl;
    return false;
  }
  ofs.close();
  if (!ofs) {
    std::cerr << "Failed to close " << filename << std::endl;
    return false;
  }
  std::cout << "Wrote trace-sampling capture sidecar to " << filename
            << " (mode " << trace_sampling_mode << ", " << ts_block_stats.size()
            << " blocks)" << std::endl;
  return true;
}

/* Deterministic accepted-launch -> loaded-module mapping.  Consumers use the
 * exact accepted ordinal instead of guessing from the benchmark executable or
 * from a framework's first process launch. */
static bool active_capture_contract_satisfied() {
  return active_launch_gate.ExactContractSatisfied() &&
         accepted_module_mappings.size() ==
             active_launch_gate.accepted_launches() &&
         inactive_channel_records_dropped.load(std::memory_order_acquire) == 0 &&
         (!active_launch_gate.strict_limit_enabled() ||
          accepted_channel_records.load(std::memory_order_acquire) > 0);
}

static bool write_module_origin_sidecar(const std::string& dir_path) {
  create_folder(dir_path.c_str());
  const std::string filename = dir_path + "/traced_kernel_modules.json";
  std::ofstream output(filename, std::ios::out | std::ios::trunc);
  if (!output) {
    std::cerr << "Failed to open " << filename << " for writing." << std::endl;
    return false;
  }
  output << "{\n";
  output << "  \"schema_version\": 1,\n";
  const char* alpha_env = std::getenv("NVBIT_ALPHA_NO_HASH");
  const bool alpha = alpha_env != nullptr && std::string(alpha_env) == "1";
  if (alpha) {
    output << "  \"module_identity\": {\"stable_key\": null, "
              "\"path_semantics\": \"path_at_capture\", "
              "\"rebind_contract\": \"alpha_path_only\"},\n";
  } else {
    output << "  \"module_identity\": {\"stable_key\": \"sha256\", "
              "\"path_semantics\": \"path_at_capture\", "
              "\"rebind_contract\": \"sha256_required_basename_hint\"},\n";
  }
  output << "  \"selection\": {\"active_from_start\": "
         << (active_from_start ? "true" : "false")
         << ", \"max_accepted_active_launches\": "
         << max_accepted_active_launches
         << ", \"accepted_active_launches\": "
         << active_launch_gate.accepted_launches()
         << ", \"active_launch_attempts\": "
         << active_launch_gate.active_launch_attempts()
         << ", \"active_launches_skipped_after_limit\": "
         << active_launch_gate.active_launches_skipped_after_limit()
         << ", \"concurrent_limit_violations\": "
         << active_launch_gate.concurrent_limit_violations()
         << ", \"exact_contract_satisfied\": "
         << (active_capture_contract_satisfied() ? "true" : "false")
         << ", \"module_mapping_count\": "
         << accepted_module_mappings.size()
         << ", \"inactive_channel_records_dropped\": "
         << inactive_channel_records_dropped.load(std::memory_order_acquire)
         << ", \"accepted_channel_records\": "
         << accepted_channel_records.load(std::memory_order_acquire)
         << "},\n";
  output << "  \"launches\": [";
  for (size_t index = 0; index < accepted_module_mappings.size(); ++index) {
    const AcceptedModuleMapping& mapping = accepted_module_mappings[index];
    if (index != 0) output << ",";
    output << "\n    {\"accepted_launch_index\": "
           << mapping.accepted_launch_index
           << ", \"device_id\": " << mapping.device_id
           << ", \"stream_id\": " << mapping.stream_id
           << ", \"trace_launch_id\": " << mapping.trace_launch_id
           << ", \"kernel_name\": \""
           << nvbit_tracer::JsonEscape(mapping.kernel_name)
           << "\", \"resolution_source\": \""
           << nvbit_tracer::JsonEscape(mapping.resolution_source)
           << "\", \"module\": {\"status\": \""
           << nvbit_tracer::JsonEscape(mapping.origin.status)
           << "\", \"reason\": \""
           << nvbit_tracer::JsonEscape(mapping.origin.reason)
           << "\", \"path_at_capture\": \""
           << nvbit_tracer::JsonEscape(mapping.origin.path)
           << "\", \"basename\": \""
           << nvbit_tracer::JsonEscape(mapping.origin.basename) << "\"";
    if (!alpha) output << ", \"sha256\": \"" << mapping.origin.sha256 << "\"";
    output << "}}";
  }
  output << (accepted_module_mappings.empty() ? "" : "\n  ") << "]\n";
  output << "}\n";
  output.flush();
  output.close();
  if (!output.good()) {
    std::cerr << "Failed while writing " << filename << "." << std::endl;
    return false;
  }
  std::cout << "Wrote accepted-launch module mapping to " << filename
            << " (" << accepted_module_mappings.size() << " launch(es))"
            << std::endl;
  return true;
}

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

std::string cwd = getcwd(NULL,0);
std::string traces_location = cwd + "/traces/";
std::string stats_location = cwd + "/traces/stats.csv";

/* kernel instruction counter, updated by the GPU */
int dynamic_kernel_limit_start =
    0;                                 // 0 means start from the begging kernel
int dynamic_kernel_limit_end = 0; // 0 means no limit

enum address_format { list_all = 0, base_stride = 1, base_delta = 2 };

// Sentinel value the `list_all` compression format writes at every
// inactive-lane position so each `address.addrs` vector on the wire
// carries exactly 32 entries (one slot per lane). Per the
// per-record contract the offline `import_trace_contract` validator
// enforces, the int64 sentinel -1 is unambiguous because real GPU
// virtual addresses are non-negative `uint64` values that never
// collide with the negative reinterpretation. The compressed
// `base_stride` and `base_delta` formats keep the existing sparse
// semantics — consumers reconstruct the 32-lane vector from
// `base_address + stride/deltas + (active_mask & predicate_mask)`
// downstream; the per-record validator currently enforces the
// 32-entry shape only on `list_all` records, with a reconstruction
// round-trip clause queued for a later refinement.
static constexpr int64_t LANE_INACTIVE_SENTINEL = -1;

int binary_version;

// Resolve binary_version (= `major*10 + minor` for the current CUDA device,
// e.g. compute 12.0 -> 120 matching nvcc's `sm_120` arch target) so
// enhanced_tracer()'s `cuobjdump -arch=sm_<binary_version>` shell-out has a
// valid arch even when NVBit never hooks a CUDA kernel launch.
//
// This helper MUST be called at context-init time (nvbit_at_ctx_init),
// not only from the per-kernel-launch hook, because cluster kernels
// launched via `cudaLaunchKernelEx` may never trigger the per-launch
// instrumentation code path on this CUDA 13.2 / SM120 stack. In that
// scenario the per-launch fallback (see `instrument_function_if_needed`
// callers below) is never reached, `binary_version` stays at its initial
// uninitialised value, and enhanced_tracer() ends up emitting
// `cuobjdump ... -arch=sm_0` which fatals with
// "Value 'sm_0' is not defined for option 'gpu-architecture'".
//
// Per `NVIDIA_Manuals/cuda_guide/01-03-04-01_binary-compatibility.md`
// cubins are NEVER binary-compatible across major versions, so on
// sm_120 hardware the running cubin is always native sm_120 (binary
// version 120) or JIT-compiled from PTX to sm_120 (also 120). The
// `major*10 + minor` formula is therefore unambiguous on this device.
//
// Returns the resolved version on success; on hard failure, prints a
// FATAL message to stderr and calls `_exit(EXIT_FAILURE)`. The caller
// should treat this as a terminal failure rather than retrying.
static int resolve_binary_version_for_current_device() {
  // Establish a checked driver identity first. The runtime path is used only
  // when it independently reports that same in-range device ordinal.
  const int checked_device = checked_current_context_device_id(
      nullptr, "resolve_binary_version_for_current_device");
  int runtime_device = -1;
  const cudaError_t dev_rc = cudaGetDevice(&runtime_device);
  cudaDeviceProp prop{};
  cudaError_t prop_rc = cudaErrorInvalidDevice;
  if (dev_rc == cudaSuccess && runtime_device == checked_device) {
    prop_rc = cudaGetDeviceProperties(&prop, runtime_device);
  }
  if (dev_rc == cudaSuccess && runtime_device == checked_device &&
      prop_rc == cudaSuccess && (prop.major * 10 + prop.minor) > 0) {
    return prop.major * 10 + prop.minor;
  }
  // Driver-API fallback uses the already status/range-checked context device.
  const CUdevice cdev = static_cast<CUdevice>(checked_device);
  int major = 0, minor = 0;
  const CUresult maj_res = cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cdev);
  const CUresult min_res = cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cdev);
  if (maj_res == CUDA_SUCCESS && min_res == CUDA_SUCCESS &&
      (major * 10 + minor) > 0) {
    return major * 10 + minor;
  }
  fprintf(stderr,
          "FATAL: resolve_binary_version_for_current_device cannot "
          "determine binary_version. dev_rc=%d runtime_device=%d "
          "checked_device=%d prop_rc=%d maj_res=%d min_res=%d major=%d "
          "minor=%d. Aborting.\n",
          (int)dev_rc, runtime_device, checked_device, (int)prop_rc,
          (int)maj_res, (int)min_res, major, minor);
  fflush(stderr);
  _exit(EXIT_FAILURE);
}

// Per-device kernel launch counter. The in-flight stream/kernel pair lives in
// each context's CTXstate; keeping it there prevents cross-context receiver
// misattribution when two contexts share a device.
//
// The device launch counter is written by the main thread under the global
// mutex and is used only to allocate globally unique launch IDs. The receiver
// reads the active launch pair from its own CTXstate, so contexts sharing a
// device cannot overwrite one another's stream/kernel snapshot.
//
// Fixed-size atomic arrays give us a proper acquire/release handshake:
// the main thread's atomic store at :1176 is release-ordered, and the
// recv thread's atomic load on `recv_thread_receiving` acquires that ordering
// transitively, making the per-context active launch pair visible.
//
// Device identities are checked against both the runtime-reported device count
// and TRACER_MAX_DEVICES before any access reaches these arrays.
std::atomic<int> kernel_id[TRACER_MAX_DEVICES]{};

// MOD. Begin. Enhanced Tracer
int next_candidate_unique_function_id = 0;
std::set<int> opcodes_id_ldgsts;
// opcode_id -> per-instruction memory space (dynamic_trace::MemorySpace as int),
// captured at instrument time from InstrType::getMemorySpace() and read by the
// receiver to (a) set the per-address `address_space` and (b) decide whether the
// global allocation-identity gate applies. Written under recv_container_mutex
// during instrumentation and read lock-free by recv threads (same discipline as
// pc_to_opcode; instrumentation of a static instruction completes before its
// dynamic records are received). See docs/STATUS.md.
std::map<int, int> opcode_id_to_address_space;
int threshold_unique_kernel_checking;
std::string variant_delimiter_str = "___";

struct traced_operand_instrument {
  traced_operand_instrument() = default; // Add this default constructor
  traced_operand_instrument(TRACED_REG_TYPE reg_type, int num_regs, int first_reg_id) {
    this->reg_type = reg_type;
    this->num_regs = num_regs;
    this->first_reg_id = first_reg_id;
  }
  TRACED_REG_TYPE reg_type;
  int num_regs;
  int first_reg_id;
};

struct traced_kernel_id {
  traced_kernel_id(std::string kernel_name, unsigned int variant_id, unsigned int candidate_unique_function_id,std::map<int, std::string> key_instructions_by_pc, std::map<int, std::string> call_or_ret_by_pc, uint64_t func_addr) {
    this->original_kernel_name = kernel_name;
    this->variant_id = variant_id;
    this->unique_function_id = candidate_unique_function_id;
    this->key_instructions_by_pc = key_instructions_by_pc;
    this->call_or_ret_by_pc = call_or_ret_by_pc;
    this->func_addr = func_addr;
    sass_has_been_parsed = false;
    rfu_has_been_parsed = false;
  }
  std::string original_kernel_name;
  unsigned int unique_function_id; // ID to identify statically the code of the kernel. It should be repetead over all the kernels with the same name and variant
  unsigned int variant_id;
  bool sass_has_been_parsed;
  bool rfu_has_been_parsed;
  std::map<int, std::string> key_instructions_by_pc;
  std::map<int, std::string> call_or_ret_by_pc;
  uint64_t func_addr;
};

// This structures are used to uniquely identify kernels. Some of them have the same name, even that they have different code
std::map<std::string,std::vector<traced_kernel_id>> all_kernels_key_instructions_by_pc; // Stores all the kernels informations ofthe execution
std::map<int, std::string> current_kernel_key_instructions_by_pc; // Only stores the information of the current analyzed kernel.
std::map<int, std::string> current_kernel_call_or_ret_by_pc; // Only stores the information of the current analyzed kernel.
std::map<CUfunction, std::tuple<std::string,int>> map_function_to_kernel_name_and_variant_id; // This is used to store the kernel name and variant id of the kernels that are being instrumented
std::map<std::string, int> map_function_name_to_unique_function_id_with_variant; // This is used to store the unique function identifier of the kernels that are being instrumented
std::map<std::string, int> map_function_name_to_unique_function_id_without_variant;
std::map<uint64_t, std::string> map_func_addr_to_kernel_name;

/* Map a memory_op_class_classifier::Value to its schema-string
 * form for the typed-exclusion sidecar. Mirrors the enum in
 * `workloads/simbench_bw/schemas/nvbit_unsupported_memory_path.schema.json`
 * (UPPER_CASE with underscores). The R23 producer hook
 * routes the 7 excluded classes (TEXTURE, SURFACE, ATOMIC, TMA,
 * ASYNC_COPY, CLUSTER_DSMEM, TMEM) to `unsupported_memory_path.json`.
 * TMEM was added for Blackwell (sm_100) tensor-memory accesses via
 * the `tcgen05.{ld,st}` family (mnemonics `LDT`/`STT`). */
static const char *op_class_to_schema_string(
    memory_op_class_classifier::Value cls) {
  switch (cls) {
    case memory_op_class_classifier::kGlobal:       return "GLOBAL";
    case memory_op_class_classifier::kShared:       return "SHARED";
    case memory_op_class_classifier::kLocal:        return "LOCAL";
    case memory_op_class_classifier::kTexture:      return "TEXTURE";
    case memory_op_class_classifier::kSurface:      return "SURFACE";
    case memory_op_class_classifier::kAtomic:       return "ATOMIC";
    case memory_op_class_classifier::kTma:          return "TMA";
    case memory_op_class_classifier::kAsyncCopy:    return "ASYNC_COPY";
    case memory_op_class_classifier::kClusterDsmem: return "CLUSTER_DSMEM";
    case memory_op_class_classifier::kTmem:         return "TMEM";
    case memory_op_class_classifier::kUnspecified:  return "UNSPECIFIED";
  }
  return "UNSPECIFIED";
}

/* True when the classifier value is one of the excluded memory-op
 * classes. Records of these classes route to the
 * `unsupported_memory_path` typed-exclusion sidecar instead of
 * receiving a normal memory payload on the wire (fail closed).
 * TMEM is excluded because tensor-memory is a Blackwell-new
 * storage class for tensor-core operand staging — the address-
 * space allocator path does not track tensor-memory backing,
 * so emitting a normal memory payload would mis-classify the
 * access.
 *
 * ATOMIC and ASYNC_COPY were removed from this set by the
 * address-carrying contract transition. A global atomic emits a
 * normal address record and resolves its allocation_id like an
 * ordinary global access; a shared atomic (ATOMS) is admitted but
 * exempt from the global allocation-id rule. Destination-bearing V3
 * ASYNC_COPY records both LDGSTS MREFs and uses their structural ordinal to
 * stamp the GLOBAL source and SHARED destination independently. See
 * docs/STATUS.md. */
static bool is_excluded_op_class(
    memory_op_class_classifier::Value cls) {
  return cls == memory_op_class_classifier::kTexture ||
         cls == memory_op_class_classifier::kSurface ||
         cls == memory_op_class_classifier::kTma ||
         cls == memory_op_class_classifier::kClusterDsmem ||
         cls == memory_op_class_classifier::kTmem;
}

/* Map an NVBit per-instruction memory space to the wire `address_space`
 * (dynamic_trace::MemorySpace). This is the default discriminator the
 * allocation-identity gate keys on. GLOBAL_TO_SHARED is instruction-level
 * provenance and therefore maps to GLOBAL here; destination-bearing LDGSTS
 * records are refined per MREF ordinal by operand_address_space() below.
 * DISTRIBUTED_SHARED collapses to SHARED (intra-cluster shared, not a global
 * allocation). CONSTANT/TEXTURE/SURFACE/NONE are not reached on the payload
 * path (their classes are excluded) and map to UNSPECIFIED. */
static dynamic_trace::MemorySpace nvbit_memspace_to_proto(
    InstrType::MemorySpace s) {
  switch (s) {
    case InstrType::MemorySpace::GLOBAL:
    case InstrType::MemorySpace::GLOBAL_TO_SHARED:
      return dynamic_trace::MEMORY_SPACE_GLOBAL;
    case InstrType::MemorySpace::SHARED:
    case InstrType::MemorySpace::DISTRIBUTED_SHARED:
      return dynamic_trace::MEMORY_SPACE_SHARED;
    case InstrType::MemorySpace::LOCAL:
      return dynamic_trace::MEMORY_SPACE_LOCAL;
    case InstrType::MemorySpace::GENERIC:
      return dynamic_trace::MEMORY_SPACE_GENERIC;
    default:
      return dynamic_trace::MEMORY_SPACE_UNSPECIFIED;
  }
}

/* Refine instruction-level memory space to an exact operand space. NVBit
 * exposes LDGSTS as GLOBAL_TO_SHARED at instruction granularity, while its
 * decoded MREFs are [shared destination, global source]. The injected channel
 * carries that ordinal in reg_id: 0 -> SHARED, 1 -> GLOBAL. Any other ordinal
 * stays UNSPECIFIED and fails the V3 ordered-pair contract instead of being
 * guessed from address values or allocation lookup results. */
static dynamic_trace::MemorySpace operand_address_space(
    memory_op_class_classifier::Value cls,
    dynamic_trace::MemorySpace instruction_space,
    uint32_t mref_ordinal) {
  if (cls != memory_op_class_classifier::kAsyncCopy) {
    return instruction_space;
  }
  if (mref_ordinal == 0) return dynamic_trace::MEMORY_SPACE_SHARED;
  if (mref_ordinal == 1) return dynamic_trace::MEMORY_SPACE_GLOBAL;
  return dynamic_trace::MEMORY_SPACE_UNSPECIFIED;
}

/* Whether a wire `address_space` must resolve a global allocation_id and fail
 * closed (unmapped_memory_range) when it cannot. GLOBAL requires it; GENERIC
 * fails closed to requiring it (a generic access that lands in shared resolves
 * to 0 and is dropped, typed — a flagged residual for generic-atomic-into-shared
 * kernels). SHARED/LOCAL/UNSPECIFIED are exempt: allocation_id 0 is legitimate. */
static bool space_requires_global_identity(dynamic_trace::MemorySpace space) {
  return space == dynamic_trace::MEMORY_SPACE_GLOBAL ||
         space == dynamic_trace::MEMORY_SPACE_GENERIC;
}

static uint32_t effective_address_lane_mask(const inst_trace_t &record) {
  const uint32_t instruction_mask =
      record.active_mask & record.predicate_mask;
  return record.address_lane_mask_present
             ? record.address_lane_mask & instruction_mask
             : instruction_mask;
}

/* Count the 32-byte L2 sectors touched by one address-carrying channel record.
 * Each active+predicated lane contributes the byte interval [addr, addr+width),
 * then overlapping sector intervals are merged.  This deduplicates sectors
 * within this instruction/operand record without coalescing equal sectors from
 * distinct operand records.  The admitted classes mirror the allocation-
 * identity gate: GLOBAL, addressed GLOBAL/GENERIC ATOMIC, and the structurally
 * GLOBAL source record of ASYNC_COPY. */
static bool trace_sampling_sector_accesses(
    const inst_trace_t &record,
    memory_op_class_classifier::Value cls,
    dynamic_trace::MemorySpace address_space,
    uint64_t *count) {
  *count = 0;
  const bool global_space = space_requires_global_identity(address_space);
  /* GLOBAL has the same documented legacy/current fallback as the protobuf
   * consumer: absent/UNSPECIFIED space is still unambiguously global from the
   * instruction class. An explicitly non-global space is contradictory.
   * ASYNC_COPY admits its exact GLOBAL source and SHARED destination, but only
   * the source contributes L2 sectors. */
  const bool global_class_fallback =
      cls == memory_op_class_classifier::kGlobal &&
      address_space == dynamic_trace::MEMORY_SPACE_UNSPECIFIED;
  if (cls == memory_op_class_classifier::kGlobal &&
      !global_space && !global_class_fallback)
    return false;
  if (cls == memory_op_class_classifier::kAsyncCopy &&
      !global_space &&
      address_space != dynamic_trace::MEMORY_SPACE_SHARED)
    return false;
  if (cls == memory_op_class_classifier::kAtomic && !global_space &&
      address_space != dynamic_trace::MEMORY_SPACE_SHARED &&
      address_space != dynamic_trace::MEMORY_SPACE_LOCAL)
    return false;
  const bool l2_relevant =
      (cls == memory_op_class_classifier::kGlobal &&
       (global_space || global_class_fallback)) ||
      (cls == memory_op_class_classifier::kAtomic && global_space) ||
      (cls == memory_op_class_classifier::kAsyncCopy && global_space);
  if (!l2_relevant) return true;
  if (record.width <= 0) return false;

  const uint32_t mask = effective_address_lane_mask(record);
  if (mask == 0) return true;
  const uint64_t width_minus_one =
      static_cast<uint64_t>(record.width - 1);
  std::vector<std::pair<uint64_t, uint64_t>> intervals;
  intervals.reserve(32);
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (((mask >> lane) & 1u) == 0) continue;
    const uint64_t address = record.addrs_or_reg_val_0[lane];
    if (address > std::numeric_limits<uint64_t>::max() - width_minus_one)
      return false;
    intervals.emplace_back(address >> 5,
                           (address + width_minus_one) >> 5);
  }
  std::sort(intervals.begin(), intervals.end());

  uint64_t total = 0;
  uint64_t first = intervals.front().first;
  uint64_t last = intervals.front().second;
  for (size_t i = 1; i < intervals.size(); ++i) {
    const uint64_t next_first = intervals[i].first;
    const uint64_t next_last = intervals[i].second;
    const bool adjacent =
        last != std::numeric_limits<uint64_t>::max() &&
        next_first == last + 1;
    if (next_first <= last || adjacent) {
      if (next_last > last) last = next_last;
      continue;
    }
    const uint64_t span = last - first + 1;
    if (total > std::numeric_limits<uint64_t>::max() - span) return false;
    total += span;
    first = next_first;
    last = next_last;
  }
  const uint64_t span = last - first + 1;
  if (total > std::numeric_limits<uint64_t>::max() - span) return false;
  *count = total + span;
  return true;
}

struct TraceSamplingIdentityResult {
  bool required = false;
  bool any_zero = false;
  bool multiple_non_zero = false;
  uint32_t allocation_id = 0;
};

static uint32_t lookup_launch_allocation_id(
    const CTXstate *ctx_state, CUcontext ctx, uint64_t address) {
  if (ctx_state != nullptr && ctx_state->active_allocation_snapshot_valid) {
    return nvbit_tracer::lookup_allocation_id_in_snapshot(
        ctx_state->active_allocation_snapshot, address);
  }
  return nvbit_tracer::lookup_allocation_id(ctx, address);
}

/* Run the exact allocation-identity decision used by payload emission early
 * enough that counting-only mode sees it too.  A record excluded from the full
 * address payload must invalidate v2 sector evidence rather than remain a
 * plausible capture-only sector count. */
static TraceSamplingIdentityResult trace_sampling_identity_result(
    CUcontext ctx,
    const inst_trace_t &record,
    memory_op_class_classifier::Value cls,
    dynamic_trace::MemorySpace address_space,
    const CTXstate *ctx_state = nullptr) {
  TraceSamplingIdentityResult result;
  result.required =
      cls == memory_op_class_classifier::kGlobal ||
      (cls == memory_op_class_classifier::kAtomic &&
       space_requires_global_identity(address_space)) ||
      (cls == memory_op_class_classifier::kAsyncCopy &&
       space_requires_global_identity(address_space));
  const uint32_t mask = effective_address_lane_mask(record);
  if (!result.required || mask == 0) return result;

  uint32_t first_non_zero = 0;
  for (unsigned lane = 0; lane < 32; ++lane) {
    if (((mask >> lane) & 1u) == 0) continue;
    const uint32_t lane_id = lookup_launch_allocation_id(
        ctx_state, ctx, record.addrs_or_reg_val_0[lane]);
    if (lane_id == 0) {
      result.any_zero = true;
    } else if (first_non_zero == 0) {
      first_non_zero = lane_id;
    } else if (lane_id != first_non_zero) {
      result.multiple_non_zero = true;
    }
  }
  result.allocation_id = first_non_zero;
  return result;
}

/* Reverse-lookup: given a tracer-assigned function_unique_id,
 * return the kernel name (without variant suffix). The forward
 * map (`map_function_name_to_unique_function_id_without_variant`,
 * declared above) is name -> id; reverse direction is rare
 * (only at trace finalization when serializing the typed-
 * exclusion sidecar) so an O(n) scan is fine. Returns "unknown"
 * on miss so the schema's `slug.minLength: 1` constraint is
 * always satisfied. */
static std::string slug_of_function_unique_id(uint32_t fid) {
  for (const auto &kv :
       map_function_name_to_unique_function_id_without_variant) {
    if (static_cast<uint32_t>(kv.second) == fid) {
      return kv.first;
    }
  }
  return std::string("unknown");
}
std::map<uint64_t, std::map<int,std::string>> map_func_addr_to_pc_to_sass_instr; // Used for kernels that does not appear in the binary
std::map<std::string, uint64_t> map_kernel_name_to_func_addr;
// First element of tuple is kernel name and second the variant id

traced_execution *m_enhanced_traced_execution;
std::string traces_path = "traces";
std::string extrainfo_path = traces_path + "/extra_info";
std::string cubin_path = extrainfo_path + "/cubin";
std::string sass_path = extrainfo_path + "/sass";
std::string register_usage_path = extrainfo_path + "/register_usage";
std::string threadblock_trace_path = traces_path + "/threadblocks";
std::string threadblock_register_values_path = traces_path + "/threadblocks/register_values";

std::string get_program_path() {
    size_t size;
    enum Constexpr { MAX_SIZE = 1024 };
    void *array[MAX_SIZE];
    size = backtrace(array, MAX_SIZE);
    if (size == 0) {
        return {};
    }
    char **calls = backtrace_symbols(array, size);
    if (calls == nullptr || calls[size - 1] == nullptr) {
        free(calls);
        return {};
    }
    std::string program_caller = calls[size - 1];
    free(calls);
    const std::size_t open_paren = program_caller.find_last_of('(');
    if (open_paren == std::string::npos) {
        return program_caller;
    }
    return program_caller.substr(0, open_paren);
}

dynamic_trace::Trace dyn_trace;
// Key: d_{device}_s_{stream}_k_{kernel}_{cta_id_x},{cta_id_y},{cta_id_z}
std::unordered_map<std::string, dynamic_trace::threadblock> threadblocks;
//TB key, warp id, int remaining injects
std::unordered_map<std::string, std::map<unsigned int, unsigned int>> remaining_injects_to_current_instruction;
const std::unordered_map<std::string, OpcodeChar> *OpcodeMap = nullptr;

dynamic_trace::threadblock& get_threadblock(std::string tb_key) {
  if(threadblocks.find(tb_key) == threadblocks.end()) {
    threadblocks[tb_key] = dynamic_trace::threadblock();
  }
  return threadblocks[tb_key];
}

// Get by reference the bool of is_last_warp_inst_ldgsts_half
unsigned int& get_remaining_injects_to_current_instruction(std::string tb_key, unsigned int warp_id) {
  if(remaining_injects_to_current_instruction.find(tb_key) == remaining_injects_to_current_instruction.end()) {
    remaining_injects_to_current_instruction[tb_key] = std::map<unsigned int, unsigned int>();
  }
  if(remaining_injects_to_current_instruction[tb_key].find(warp_id) == remaining_injects_to_current_instruction[tb_key].end()) {
    remaining_injects_to_current_instruction[tb_key][warp_id] = 0;
  }
  return remaining_injects_to_current_instruction[tb_key][warp_id];
}

void create_folder(const char * folder_path) {
  if (mkdir(folder_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    if (errno == EEXIST) {
      // alredy exists
    } else {
      // something else
      std::cout << "cannot create folder error:" << strerror(errno)
                << std::endl;
      return;
    }
  }
}

void remove_folder(const char * folder_path) {
  std::filesystem::remove_all(folder_path);
}

std::string read_stripped_line(std::ifstream &ifs) {
  std::string line;
  std::getline(ifs, line);
  std::string clear_line = ReplaceAll(line, "/*", " ");
  clear_line = ReplaceAll(clear_line, "*/", " ");
  clear_line = ReplaceAll(clear_line, ",", " ");
  clear_line = ReplaceAll(clear_line, ";", " ");
  return strip_string(clear_line);
}

std::string getEnclosedSubstring(std::string str) {
  size_t start_pos = str.find('(');
  if(start_pos == std::string::npos)
      return "";
  size_t end_pos = str.find(')', start_pos);
  if(end_pos == std::string::npos)
      return "";
  return str.substr(start_pos + 1, end_pos - start_pos - 1);
}

// Regex to remove dependency requirements, e.g. " &req={1}"
std::regex reqRegex("\\s*&req=\\{[^}]+\\}");
// Regex to remove write-slot annotations, e.g. " &wr=0x4"
std::regex wrRegex("\\s*&wr=0x[0-9A-Fa-f]+");
// Regex to remove write-slot annotations, e.g. " &wr=0x4"
std::regex rdRegex("\\s*&rd=0x[0-9A-Fa-f]+");
// Regex to remove transaction/synchronization annotations, e.g. " ?trans1;" or " ?WAIT4_END_GROUP;"
std::regex transRegex("\\s*\\?[A-Za-z0-9_]+");

std::string replaceInstructionNewExtraInformation(std::string original_sass_string) {
  std::string transformed = original_sass_string;
  transformed = std::regex_replace(transformed, reqRegex, "");
  transformed = std::regex_replace(transformed, wrRegex, "");
  transformed = std::regex_replace(transformed, rdRegex, "");
  transformed = std::regex_replace(transformed, transRegex, "");
  return transformed;
}

void print_map(const std::map<int, std::string> &map_to_print) {
  std::cout << "Number of elements in the map: " << map_to_print.size() << std::endl;
  for(auto it = map_to_print.begin(); it != map_to_print.end(); ++it) {
    std::cout << std::hex << it->first << std::dec << " " << it->second << std::endl;
  }
  std::cout << std::endl;
}

void print_all_traced_kernels() {
  for(auto it = all_kernels_key_instructions_by_pc.begin(); it != all_kernels_key_instructions_by_pc.end(); ++it) {
    for(unsigned int i = 0; i < it->second.size(); i++) {
      std::cout << "Kernel " << it->second[i].original_kernel_name << " variant " << i << " has been traced. SASS parsed? " <<  it->second[i].sass_has_been_parsed
        << " RFU parsed? " << it->second[i].rfu_has_been_parsed << std::endl;
    }
  }
}

void erase_not_tracked_call_or_rets(std::map<int, std::string> &kernel_call_or_ret_by_pc, std::set<unsigned int> &call_or_ret_pcs_not_to_consider) {
  for(auto it = call_or_ret_pcs_not_to_consider.begin(); it != call_or_ret_pcs_not_to_consider.end(); ++it) {
    auto it_candidate = kernel_call_or_ret_by_pc.find(*it);
    if(it_candidate != kernel_call_or_ret_by_pc.end()) {
      kernel_call_or_ret_by_pc.erase(it_candidate);
    }
  }
}

bool are_two_kernels_equal(traced_kernel_id &kernel_to_be_compared, std::map<int, std::string> &candidate_key_instructions_by_pc, std::map<int, std::string> &candidate_call_or_ret_by_pc, bool compare_call_or_ret) {
  bool are_same_key_instructions = (kernel_to_be_compared.key_instructions_by_pc == candidate_key_instructions_by_pc);
  bool are_same_call_or_ret = true;
  if(compare_call_or_ret) {
    are_same_call_or_ret = (kernel_to_be_compared.call_or_ret_by_pc == candidate_call_or_ret_by_pc);
  }
  return are_same_key_instructions && are_same_call_or_ret;
}

bool has_the_kernel_been_traced(std::string kernel_name, std::map<int, std::string> candidate_parsed_kernel, std::map<int, std::string> &candidate_call_or_ret_by_pc, unsigned int &variant_id, bool is_rfu_parsing, bool compare_call_or_ret, std::set<unsigned int> &call_or_ret_pcs_not_to_consider) {
  bool has_been_traced = false;
  auto it_already_traced = all_kernels_key_instructions_by_pc.find(kernel_name);
  if(it_already_traced != all_kernels_key_instructions_by_pc.end()) {
    for(unsigned int i = 0; !has_been_traced && (i < all_kernels_key_instructions_by_pc[kernel_name].size()); i++) {
      bool is_alread_parsed = is_rfu_parsing ? it_already_traced->second[i].rfu_has_been_parsed : it_already_traced->second[i].sass_has_been_parsed;
      if(compare_call_or_ret) {
        erase_not_tracked_call_or_rets(it_already_traced->second[i].call_or_ret_by_pc, call_or_ret_pcs_not_to_consider);
      }
      if( are_two_kernels_equal(it_already_traced->second[i], candidate_parsed_kernel, candidate_call_or_ret_by_pc, compare_call_or_ret ) && !is_alread_parsed) {
        has_been_traced = true;
        variant_id = it_already_traced->second[i].variant_id;
      }
    }
  }
  return has_been_traced;
}

bool parse_sass(int binary_version, const std::filesystem::directory_entry &entry) {
  std::string absolute_sass_path = resolve_output_path(cwd, sass_path);
  std::string sass_file = absolute_sass_path + "/" + entry.path().stem().string() + ".sass";
  std::ifstream ifs_sass; // input file stream
  ifs_sass.open(sass_file, std::ios::in);
  if (!ifs_sass)
  {
    std::cerr << "Error opening file " << sass_file << std::endl;
    fflush(stderr);
    return false;
  }
  
  bool is_starting_reading_kernel = false;
  std::string kernel_name = "";
  traced_kernel *current_traced_kernel = nullptr;
  std::map<int, std::string> full_inst_call_or_ret_by_pc;
  std::set<unsigned int> call_or_ret_pcs_not_to_consider;

  while (!ifs_sass.eof())
  {
    std::string stripped_line = read_stripped_line(ifs_sass);

    if (!is_starting_reading_kernel)
    {
      if ((stripped_line.find("Function") != std::string::npos))
      {
        std::vector<std::string> aux_splitted = split_string(stripped_line, ' ');
        if (aux_splitted.size() != 3) {
          std::cerr << "Malformed cuobjdump Function line in " << sass_file
                    << std::endl;
          return false;
        }
        kernel_name = aux_splitted[2];
        current_traced_kernel = new traced_kernel(kernel_name, static_cast<unsigned>(binary_version));
        is_starting_reading_kernel = true;
      }
    }
    else
    {
      if (stripped_line.find("..........") != std::string::npos)
      {
        is_starting_reading_kernel = false;

        // Check if the kernel was captured during the execution
        unsigned int variant_id = 0;
        bool has_been_traced = has_the_kernel_been_traced(kernel_name, current_traced_kernel->get_key_instructions_pcs(), full_inst_call_or_ret_by_pc, variant_id, false, false, call_or_ret_pcs_not_to_consider);

        if(has_been_traced) {
          std::string final_kernel_name = kernel_name + variant_delimiter_str + std::to_string(variant_id);
          auto it_search_funct = map_kernel_name_to_func_addr.find(kernel_name);
          if (it_search_funct == map_kernel_name_to_func_addr.end()) {
            std::cerr << "No traced function address for SASS kernel "
                      << kernel_name << std::endl;
            delete current_traced_kernel;
            return false;
          }
          all_kernels_key_instructions_by_pc[kernel_name][variant_id].sass_has_been_parsed = true;
          current_traced_kernel->set_kernel_name(final_kernel_name);
          if (!m_enhanced_traced_execution->add_traced_kernel(
                  final_kernel_name,
                  all_kernels_key_instructions_by_pc[kernel_name][variant_id]
                      .unique_function_id,
                  current_traced_kernel, it_search_funct->second, true)) {
            current_traced_kernel = nullptr;
            return false;
          }
          current_traced_kernel = nullptr;
        }else {
          delete current_traced_kernel;
          current_traced_kernel = nullptr;
        }
      }
      else if (stripped_line.find("headerflags") == std::string::npos)
      {
        stripped_line = replaceInstructionNewExtraInformation(stripped_line);
        std::vector<std::string> aux_list = split_string(stripped_line, ' ');
        std::vector<std::string> aux_list2;
        if (binary_version >= 70)
        {
          if (ifs_sass.eof()) {
            delete current_traced_kernel;
            return false;
          }
          aux_list2 = split_string(read_stripped_line(ifs_sass), ' ');
          if (aux_list2.empty()) {
            delete current_traced_kernel;
            return false;
          }
        }
        if (aux_list.empty()) {
          delete current_traced_kernel;
          return false;
        }
        try {
          if (!current_traced_kernel->add_instruction(
                  aux_list, aux_list2, threshold_unique_kernel_checking,
                  stripped_line)) {
            delete current_traced_kernel;
            return false;
          }
        } catch (const std::exception &error) {
          std::cerr << "Malformed SASS instruction in " << sass_file << ": "
                    << error.what() << std::endl;
          delete current_traced_kernel;
          return false;
        }
      }
    }
  }
  if (current_traced_kernel != nullptr) {
    delete current_traced_kernel;
    return false;
  }
  if (ifs_sass.bad()) {
    return false;
  }
  ifs_sass.close();
  return true;
}

std::string replaceEnclosedSubstring(std::string str, const std::string &replacement)
{
  size_t start_pos = str.find('`');
  if (start_pos == std::string::npos)
    return str;
  size_t end_pos = str.find(')', start_pos);
  if (end_pos == std::string::npos)
    return str;
  str.replace(start_pos, end_pos - start_pos + 1, replacement);
  return str;
}

bool parse_rfu_instruction_info(std::vector<std::string> splitted_text, std::vector<std::string> reg_order, std::string full_instruction_str, std::map<int, std::string> &key_instructions_by_pc,
  std::map<unsigned int, std::map<std::string, unsigned int>> &reg_usage_by_pc, std::map<unsigned int, std::tuple<std::string, int>> &call_target_by_pc, std::map<int, std::string> &full_inst_call_or_ret_by_pc, std::set<unsigned int> &call_or_ret_pcs_not_to_consider)
{
  if (splitted_text.empty() || reg_order.empty()) return false;
  unsigned pc = 0;
  try {
    std::size_t parsed = 0;
    const unsigned long parsed_pc = std::stoul(splitted_text[0], &parsed, 16);
    if (parsed != splitted_text[0].size() ||
        parsed_pc > static_cast<unsigned long>(std::numeric_limits<int>::max())) {
      return false;
    }
    pc = static_cast<unsigned>(parsed_pc);
  } catch (const std::exception &) {
    return false;
  }
  bool is_all_reg_use_gathered = false;
  int num_already_visited_reg_files = 0;
  std::string instruction_str = full_instruction_str;
  instruction_str = replaceInstructionNewExtraInformation(instruction_str);
  std::string sass_string = create_sass_instr(instruction_str, true, "//");
  if (sass_string.empty()) {
    return false;
  }
  if(track_this_instruction(reg_usage_by_pc.size(), threshold_unique_kernel_checking, instruction_str)){
    key_instructions_by_pc[pc] = sass_string;
  }

  if((sass_string.find("CALL")!= std::string::npos) || (sass_string.find("RET")!= std::string::npos)) {
    std::string funct_name = getEnclosedSubstring(sass_string);
    auto it_search_funct = map_kernel_name_to_func_addr.find(funct_name);
    bool is_rel_type = (sass_string.find("REL") != std::string::npos);
    if(!is_rel_type && (it_search_funct != map_kernel_name_to_func_addr.end())) {
      uint64_t func_addr = it_search_funct->second;
      std::stringstream ss;
      ss << "0x" << std::hex << func_addr;
      std::string new_sass_string = replaceEnclosedSubstring(sass_string, ss.str());
      full_inst_call_or_ret_by_pc[pc] = new_sass_string;
    }else if(!is_rel_type && ((sass_string.find("printf") != std::string::npos) || (sass_string.find("assert") != std::string::npos) ) ) {
      call_or_ret_pcs_not_to_consider.insert(pc);
    }else if(!is_rel_type){
      full_inst_call_or_ret_by_pc[pc] = sass_string;
    }
  }

  if(sass_string.find("CALL") != std::string::npos) {
    std::size_t first_char =  sass_string.find("(");
    std::size_t last_char =  sass_string.find(")");
    if((first_char != std::string::npos) && (last_char != std::string::npos)) {
      std::string call_target = sass_string.substr(first_char+1, last_char - first_char - 1);
      auto search_unique_function_id = map_function_name_to_unique_function_id_without_variant.find(call_target);
      if(search_unique_function_id != map_function_name_to_unique_function_id_without_variant.end()) {
        int candidate_unique_function_id = search_unique_function_id->second;
        call_target_by_pc[pc]= std::make_tuple(call_target, candidate_unique_function_id);
      }
    }
  }

  for (unsigned int i = (splitted_text.size() - 1); (i > 0) && !is_all_reg_use_gathered; --i)
  {
    if (splitted_text[i] == "//")
    {
      is_all_reg_use_gathered = true;
    }
    else if (splitted_text[i] == "|")
    {
      num_already_visited_reg_files++;
    }
    else
    {
      if (num_already_visited_reg_files <= 0 ||
          static_cast<std::size_t>(num_already_visited_reg_files) >
              reg_order.size()) {
        return false;
      }
      const char *first = splitted_text[i].data();
      const char *last = first + splitted_text[i].size();
      unsigned long parsed_regs = 0;
      const auto result = std::from_chars(first, last, parsed_regs, 10);
      if (result.ec != std::errc{} || result.ptr != last ||
          parsed_regs > std::numeric_limits<unsigned>::max()) {
        return false;
      }
      unsigned num_regs = static_cast<unsigned>(parsed_regs);
      std::string reg_file_name =
          reg_order[reg_order.size() - num_already_visited_reg_files];
      reg_usage_by_pc[pc][reg_file_name] = num_regs;
    }
  }
  return true;
}

bool parse_rfu(const std::filesystem::directory_entry &entry) {
  std::string absolute_rfu_path = resolve_output_path(cwd, register_usage_path);
  std::string rfu_file = absolute_rfu_path + "/" + entry.path().stem().string() + ".rfu";
  std::ifstream ifs_rfu; // input file stream
  ifs_rfu.open(rfu_file, std::ios::in);
  if (!ifs_rfu)
  {
    std::cerr << "Error opening file " << rfu_file << std::endl;
    fflush(stderr);
    return false;
  }
  
  bool is_starting_reading_kernel = false;
  bool is_code_region_started = false;
  std::string kernel_name = "";
  std::vector<std::string> reg_order;
  std::map<int, std::string> key_instructions_by_pc;
  std::map<unsigned int, std::map<std::string, unsigned int>> reg_usage_by_pc;
  std::map<unsigned int, std::tuple<std::string, int>> call_target_by_pc;
  std::map<int, std::string> full_call_or_ret_inst_by_pc;
  std::set<unsigned int> call_or_ret_pcs_not_to_consider;
  key_instructions_by_pc.clear();
  reg_usage_by_pc.clear();
  call_target_by_pc.clear();

  while (!ifs_rfu.eof())
  {
    std::string stripped_line = read_stripped_line(ifs_rfu);
    
    if (!is_starting_reading_kernel)
    {
      if ((stripped_line.find("GPR") != std::string::npos))
      {
        stripped_line = ReplaceAll(stripped_line, "|", "");
        std::vector<std::string> aux_splitted = split_string(stripped_line, ' ');
        aux_splitted.erase(aux_splitted.begin());
        reg_order = aux_splitted;
        is_starting_reading_kernel = true;
      }
    }
    else
    {
      if (stripped_line.find("Legend:") != std::string::npos)
      {
        is_starting_reading_kernel = false;
        is_code_region_started = false;
        unsigned int variant_id = 0;
        bool has_been_traced = has_the_kernel_been_traced(kernel_name, key_instructions_by_pc, full_call_or_ret_inst_by_pc, variant_id, true, true, call_or_ret_pcs_not_to_consider);
        if(has_been_traced) {
          std::string final_kernel_name = kernel_name + variant_delimiter_str + std::to_string(variant_id);
          if (!m_enhanced_traced_execution->add_register_usage_to_a_kernel(
                  final_kernel_name, reg_usage_by_pc, call_target_by_pc)) {
            return false;
          }
          all_kernels_key_instructions_by_pc[kernel_name][variant_id].rfu_has_been_parsed = true;
        }
        key_instructions_by_pc.clear();
        reg_usage_by_pc.clear();
        call_target_by_pc.clear();
        full_call_or_ret_inst_by_pc.clear();
        call_or_ret_pcs_not_to_consider.clear();
      }else {
        std::vector<std::string> splitted_text = split_string(stripped_line, ' ');
        if(!splitted_text.empty()) {
          if(is_code_region_started && (stripped_line.find(":") == std::string::npos) &&
            (stripped_line.find(".weak") == std::string::npos) && (stripped_line.find(".type") == std::string::npos)
            && (stripped_line.find(".size") == std::string::npos)) {
            if (!parse_rfu_instruction_info(splitted_text, reg_order, stripped_line,
                                            key_instructions_by_pc, reg_usage_by_pc,
                                            call_target_by_pc,
                                            full_call_or_ret_inst_by_pc,
                                            call_or_ret_pcs_not_to_consider)) {
              return false;
            }
          }else if( (splitted_text[0].find("0000") != std::string::npos) && (stripped_line.find("//") != std::string::npos) && (stripped_line.find(":") == std::string::npos) &&
            (stripped_line.find(".weak") == std::string::npos) && (stripped_line.find(".type") == std::string::npos)
            && (stripped_line.find(".size") == std::string::npos)) {
            is_code_region_started = true;
            if (!parse_rfu_instruction_info(splitted_text, reg_order, stripped_line,
                                            key_instructions_by_pc, reg_usage_by_pc,
                                            call_target_by_pc,
                                            full_call_or_ret_inst_by_pc,
                                            call_or_ret_pcs_not_to_consider)) {
              return false;
            }
          }else if(splitted_text[0].find(".text") != std::string::npos) {
            if (splitted_text[0].size() <= 6 ||
                splitted_text[0].back() != ':') {
              return false;
            }
            kernel_name = splitted_text[0].substr(6);
            kernel_name.pop_back(); // Remove the last  : char
          }
        }
      }
    }
  }
  if (is_starting_reading_kernel || ifs_rfu.bad()) {
    return false;
  }
  ifs_rfu.close();
  return true;
}
// MOD. End. Enhanced Tracer

static uint64_t parse_max_accepted_active_launches() {
  const char* value = std::getenv("MAX_ACCEPTED_ACTIVE_LAUNCHES");
  if (value == nullptr || value[0] == '\0') return 0;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed != 1) {
    fprintf(stderr,
            "FATAL: MAX_ACCEPTED_ACTIVE_LAUNCHES must be exactly 1 when "
            "set (got '%s'). The exact-launch capture contract does not "
            "silently accept malformed or broader limits.\n",
            value);
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
  return 1;
}

static bool is_module_origin_event(nvbit_api_cuda_t cbid) {
  return cbid == API_CUDA_cuModuleLoad ||
         cbid == API_CUDA_cuModuleLoadData ||
         cbid == API_CUDA_cuModuleLoadDataEx ||
         cbid == API_CUDA_cuModuleLoadFatBinary ||
         cbid == API_CUDA_cuModuleGetFunction ||
         cbid == API_CUDA_cuModuleUnload ||
         cbid == API_CUDA_cuLibraryLoadData ||
         cbid == API_CUDA_cuLibraryLoadFromFile ||
         cbid == API_CUDA_cuKernelGetFunction ||
         cbid == API_CUDA_cuLibraryUnload;
}

/* Retain observed load origins, never a guessed benchmark filename. There
 * is deliberately no function-name registry: symbol names are module-local,
 * so two live modules may contain the same kernel name. Query the owner of
 * the exact launch handle instead, including CUDA's context-less CUkernel
 * launch convention. No eager function enumeration or disassembly is needed. */
static void handle_module_origin_event(CUcontext ctx, int is_exit,
                                       nvbit_api_cuda_t cbid, void* params,
                                       CUresult* status) {
  if (!is_exit || status == nullptr || *status != CUDA_SUCCESS) return;
  switch (cbid) {
    case API_CUDA_cuModuleLoad: {
      auto* p = static_cast<cuModuleLoad_params*>(params);
      if (p->module != nullptr && *p->module != nullptr && p->fname != nullptr)
        loaded_module_origins[ModuleKey{ctx, *p->module}] = p->fname;
      break;
    }
    case API_CUDA_cuModuleLoadData: {
      auto* p = static_cast<cuModuleLoadData_params*>(params);
      if (p->module != nullptr && *p->module != nullptr)
        loaded_module_origins[ModuleKey{ctx, *p->module}] =
            nvbit_tracer::LookupAddressInProcessMaps(p->image).path;
      break;
    }
    case API_CUDA_cuModuleLoadDataEx: {
      auto* p = static_cast<cuModuleLoadDataEx_params*>(params);
      if (p->module != nullptr && *p->module != nullptr)
        loaded_module_origins[ModuleKey{ctx, *p->module}] =
            nvbit_tracer::LookupAddressInProcessMaps(p->image).path;
      break;
    }
    case API_CUDA_cuModuleLoadFatBinary: {
      auto* p = static_cast<cuModuleLoadFatBinary_params*>(params);
      if (p->module != nullptr && *p->module != nullptr)
        loaded_module_origins[ModuleKey{ctx, *p->module}] =
            nvbit_tracer::LookupAddressInProcessMaps(p->fatCubin).path;
      break;
    }
    case API_CUDA_cuModuleUnload: {
      auto* p = static_cast<cuModuleUnload_params*>(params);
      if (p->hmod != nullptr) {
        loaded_module_origins.erase(ModuleKey{ctx, p->hmod});
        loaded_library_modules.erase(ModuleKey{ctx, p->hmod});
      }
      break;
    }
    case API_CUDA_cuLibraryLoadData: {
      auto* p = static_cast<cuLibraryLoadData_params*>(params);
      if (p->library != nullptr && *p->library != nullptr)
        loaded_library_origins[*p->library] =
            nvbit_tracer::LookupAddressInProcessMaps(p->code).path;
      break;
    }
    case API_CUDA_cuLibraryLoadFromFile: {
      auto* p = static_cast<cuLibraryLoadFromFile_params*>(params);
      if (p->library != nullptr && *p->library != nullptr &&
          p->fileName != nullptr)
        loaded_library_origins[*p->library] = p->fileName;
      break;
    }
    case API_CUDA_cuLibraryUnload: {
      auto* p = static_cast<cuLibraryUnload_params*>(params);
      if (p->library != nullptr) {
        loaded_library_origins.erase(p->library);
        for (auto it = loaded_library_modules.begin();
             it != loaded_library_modules.end();) {
          if (it->second == p->library) it = loaded_library_modules.erase(it);
          else ++it;
        }
      }
      break;
    }
    case API_CUDA_cuKernelGetFunction: {
      auto* p = static_cast<cuKernelGetFunction_params*>(params);
      if (p->pFunc == nullptr || *p->pFunc == nullptr || p->kernel == nullptr)
        break;
      const bool previous_skip = skip_callback_flag;
      skip_callback_flag = true;
      CUlibrary library = nullptr;
      CUmodule module = nullptr;
      if (cuKernelGetLibrary(&library, p->kernel) == CUDA_SUCCESS &&
          library != nullptr &&
          cuFuncGetModule(&module, *p->pFunc) == CUDA_SUCCESS &&
          module != nullptr) {
        loaded_library_modules[ModuleKey{ctx, module}] = library;
      }
      skip_callback_flag = previous_skip;
      break;
    }
    default:
      break;
  }
}

static AcceptedModuleMapping resolve_accepted_module_mapping(
    CUcontext ctx, CUfunction function, int device_id, int64_t stream_id,
    uint32_t trace_launch_id) {
  AcceptedModuleMapping mapping;
  mapping.accepted_launch_index = active_launch_gate.accepted_launches() - 1;
  mapping.device_id = device_id;
  mapping.stream_id = stream_id;
  mapping.trace_launch_id = trace_launch_id;
  const char* name = nvbit_get_func_name(ctx, function, true);
  if (name != nullptr) mapping.kernel_name = name;

  std::string raw_origin;
  const bool previous_skip = skip_callback_flag;
  skip_callback_flag = true;
  CUmodule module = nullptr;
  if (cuFuncGetModule(&module, function) == CUDA_SUCCESS && module != nullptr) {
    auto origin = loaded_module_origins.find(ModuleKey{ctx, module});
    if (origin != loaded_module_origins.end()) {
      raw_origin = origin->second;
      mapping.resolution_source = "module_registry";
    } else {
      mapping.resolution_source = "module_registry_miss";
      auto library_owner = loaded_library_modules.find(ModuleKey{ctx, module});
      if (library_owner != loaded_library_modules.end()) {
        auto library_origin = loaded_library_origins.find(library_owner->second);
        if (library_origin != loaded_library_origins.end()) {
          raw_origin = library_origin->second;
          mapping.resolution_source = "library_function_registry";
        }
      }
    }
  }
  if (raw_origin.empty()) {
    // cuLaunchKernel/cuLaunchKernelEx also accept CUkernel cast to CUfunction.
    // Such handles reject cuFuncGetModule in the usual case; retrying the
    // exact owner query after a module-registry miss also covers lazy library
    // loading without falling back to a same-named function.
    CUlibrary library = nullptr;
    if (cuKernelGetLibrary(&library, reinterpret_cast<CUkernel>(function)) ==
            CUDA_SUCCESS && library != nullptr) {
      auto origin = loaded_library_origins.find(library);
      if (origin != loaded_library_origins.end()) {
        raw_origin = origin->second;
        mapping.resolution_source = "library_registry";
      } else {
        mapping.resolution_source = "library_registry_miss";
      }
    }
  }
  skip_callback_flag = previous_skip;
  if (mapping.resolution_source.empty())
    mapping.resolution_source = "origin_unresolved";
  mapping.origin = nvbit_tracer::EvaluateModuleOrigin(raw_origin);
  return mapping;
}

/* Resolve every output global from TRACES_FOLDER during nvbit_at_init, before
 * any CUDA callback can trigger a launch or an application thread can close an
 * inherited /proc/self/fd directory authority. The first-event path still
 * creates the ordinary trace tree, but it no longer establishes phase
 * authority from a mutable lexical path. */
static void configure_trace_output_paths_from_environment() {
  if (user_defined_folders != 1) return;

  const char *usr_folder_c = std::getenv("TRACES_FOLDER");
  if (usr_folder_c == nullptr || usr_folder_c[0] == '\0') return;

  std::string usr_folder = usr_folder_c;
  while (!usr_folder.empty() && usr_folder.back() == '/') {
    usr_folder.pop_back();
  }
  traces_path = usr_folder;
  extrainfo_path = traces_path + "/extra_info";
  cubin_path = extrainfo_path + "/cubin";
  sass_path = extrainfo_path + "/sass";
  register_usage_path = extrainfo_path + "/register_usage";
  threadblock_trace_path = traces_path + "/threadblocks";
  threadblock_register_values_path =
      traces_path + "/threadblocks/register_values";
  traces_location = traces_path + "/";
  stats_location = traces_path + "/stats.csv";
  printf("\n Traces location is %s \n", traces_location.c_str());
  printf("Stats location is %s \n", stats_location.c_str());
}

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
              "Exclude predicated off instruction from count");
  GET_VAR_INT(dynamic_kernel_limit_end, "DYNAMIC_KERNEL_LIMIT_END", 0,
              "Limit of the number kernel to be printed, 0 means no limit");
  GET_VAR_INT(dynamic_kernel_limit_start, "DYNAMIC_KERNEL_LIMIT_START", 0,
              "start to report kernel from this kernel id, 0 means starts from "
              "the beginning, i.e. first kernel");
  GET_VAR_INT(
         active_from_start, "ACTIVE_FROM_START", 1,
         "Start instruction tracing from start or wait for cuProfilerStart "
         "and cuProfilerStop. If set to 0, DYNAMIC_KERNEL_LIMIT options have no effect");
  max_accepted_active_launches = parse_max_accepted_active_launches();
  if (max_accepted_active_launches == 1 && active_from_start != 0) {
    fprintf(stderr,
            "FATAL: MAX_ACCEPTED_ACTIVE_LAUNCHES=1 requires "
            "ACTIVE_FROM_START=0 so cuProfilerStart/cuProfilerStop, not the "
            "first process launch, select the capture target.\n");
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
  active_launch_gate.Configure(active_from_start != 0,
                               max_accepted_active_launches);
  if (max_accepted_active_launches == 1) {
    fprintf(stderr,
            "[tracer_tool] exact active-launch contract enabled: "
            "MAX_ACCEPTED_ACTIVE_LAUNCHES=1 ACTIVE_FROM_START=%d\n",
            active_from_start);
  }
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(intermediate_extra_files_persistance, "INTERMEDIATE_EXTRA_FILES_PERSISTANCE", 0, "Enable to not delete intermediate files from the extra trace generation");
  GET_VAR_INT(skip_enhanced_tracer, "SKIP_ENHANCED_TRACER", 0, "Skip the post-execution enhanced_tracer pass (cubin extraction + parse_sass + parse_rfu). Primary trace artifacts (dynamic_trace.pb, per-CTA pb files, stats.csv) are unaffected.");
  GET_VAR_INT(enhanced_tracer_per_cubin_timeout_s, "ENHANCED_TRACER_PER_CUBIN_TIMEOUT_S", 30, "Per-subprocess deadline (seconds) for the cuobjdump/nvdisasm shell-outs inside enhanced_tracer. On timeout the child process group is SIGTERM'd then SIGKILL'd; the parent always returns normally. Set to 0 to wait indefinitely (legacy behavior).");
  GET_VAR_INT(enhanced_tracer_max_parser_input_kb, "ENHANCED_TRACER_MAX_PARSER_INPUT_KB", 64, "Maximum size (KB) of the .sass / .rfu files passed to parse_sass / parse_rfu. Cubins whose generated parser inputs exceed the threshold (or whose subprocess timed out) are skipped with a typed enhanced_tracer_cubin_parse_timeout diagnostic. Set to 0 to disable the size pre-skip.");
  GET_VAR_INT(print_core_id, "TOOL_TRACE_CORE", 0,
              "write the core id in the traces");
  GET_VAR_INT(terminate_after_limit_number_of_kernels_reached, "TERMINATE_UPON_LIMIT", 0, 
              "Stop the process once the current kernel > DYNAMIC_KERNEL_LIMIT_END");
  GET_VAR_INT(user_defined_folders, "USER_DEFINED_FOLDERS", 0, "Uses the user defined "
              "folder TRACES_FOLDER path environment");
  configure_trace_output_paths_from_environment();
  GET_VAR_INT(trace_sampling_mode, "TRACE_SAMPLING_MODE", 0,
              "Trace-sampling capture mode: 0=off, 1=counting-only (per-block "
              "stats, no address payload), 2=selective-dump (write only the "
              "TRACE_SAMPLING_SELECTED_BLOCKS whole blocks; counting stays on)");
  {
    const char *ts_sel = std::getenv("TRACE_SAMPLING_SELECTED_BLOCKS");
    if (ts_sel != nullptr) {
      std::string spec = ts_sel;
      std::string body;
      if (!spec.empty() && spec[0] == '@') {
        std::ifstream f(spec.substr(1));
        std::stringstream ss;
        ss << f.rdbuf();
        body = ss.str();
      } else {
        body = spec;
      }
      for (char &c : body) {
        if (c == ',' || c == '\n' || c == '\t') c = ' ';
      }
      std::stringstream ss(body);
      long id;
      while (ss >> id) trace_sampling_selected_blocks.insert(id);
    }
    const char *ts_sc = std::getenv("TRACE_SAMPLING_SIDECAR");
    if (ts_sc != nullptr) trace_sampling_sidecar_path = ts_sc;
  }
  GET_VAR_INT(threshold_unique_kernel_checking, "THRESHOLD_UNIQUE_KERNEL_CHECKING", 10,
              "Number of instructions used to check if kernels with the same name are different");
  GET_VAR_INT(gather_registers, "GATHER_REGISTERS", 0, "Compatibility knob; protobuf register snapshots are not emitted by this tracer.");

  const char *async_order_env = std::getenv("SIMBENCH_NVBIT_ASYNC_ORDER");
  if (async_order_env == nullptr || std::string_view(async_order_env) == "0") {
    async_order_enabled = false;
  } else if (std::string_view(async_order_env) == "1") {
    async_order_enabled = true;
    fprintf(stderr,
            "[tracer_tool] async_order_enabled=1: selected LDGSTS / "
            "LDGDEPBAR / DEPBAR.LE SB0 instruction creations will receive "
            "dense host main-channel order tokens.\n");
  } else {
    fprintf(stderr,
            "FATAL: SIMBENCH_NVBIT_ASYNC_ORDER must be exactly 0, 1, or "
            "unset; got %s.\n",
            async_order_env);
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }

  /* Non-root measurement NVBit phase-injection contract: env-gated
   * activation switch for the consolidated %clock64 phase-clock
   * tracer extension. When `SIMBENCH_NVBIT_PHASE_INJECT=1`, the
   * tracer initializes a per-context phase-channel (separate from
   * the existing instruction-trace channel so the protobuf trace
   * is not corrupted) and injects `%clock64` reads at semantic
   * phase boundaries (function entry/exit, basic-block entry,
   * `bar.sync`). When unset (default) or `0`, the tracer behaves
   * identically to the pre-extension instruction-trace pipeline so
   * the existing dual-source measurement baseline is unaffected.
   * Foundational wiring is in place; the actual injection-call
   * insertion at boundary classes is sequenced for a subsequent
   * implementation round under the current measurement contract.
   */
  const char *phase_inject_env = std::getenv("SIMBENCH_NVBIT_PHASE_INJECT");
  phase_injection_enabled = (phase_inject_env != nullptr &&
                             phase_inject_env[0] == '1');
  if (phase_injection_enabled) {
    const char *repeat_env =
        std::getenv("SIMBENCH_NVBIT_PHASE_REPEAT_INDEX");
    if (repeat_env != nullptr) {
      bool decimal = repeat_env[0] != '\0';
      for (const char *p = repeat_env; decimal && *p != '\0'; ++p) {
        decimal = *p >= '0' && *p <= '9';
      }
      errno = 0;
      char *endp = nullptr;
      const unsigned long long parsed = std::strtoull(repeat_env, &endp, 10);
      if (!decimal || errno != 0 || endp == repeat_env || *endp != '\0' ||
          parsed > UINT32_MAX) {
        fprintf(stderr,
                "ERROR: SIMBENCH_NVBIT_PHASE_REPEAT_INDEX=%s is not a "
                "uint32 decimal repeat index.\n",
                repeat_env);
        std::exit(EXIT_FAILURE);
      }
      phase_repeat_index = static_cast<uint32_t>(parsed);
    }
    fprintf(stderr,
            "[tracer_tool] phase_injection_enabled=1 "
            "repeat_index=%" PRIu32 " "
            "(SIMBENCH_NVBIT_PHASE_INJECT=1); phase-injection "
            "foundation wired (phase_clock_record_t struct + "
            "record_phase_clock injected function defined). "
            "Injection-call insertion at function-entry / "
            "function-exit / basic-block-entry / bar.sync "
            "boundaries is sequenced for a subsequent round; no "
            "per-launch phase_cycle_ledger sidecars are emitted "
            "yet.\n",
            phase_repeat_index);

    create_folder(traces_path.c_str());
    std::string phase_authority_error;
    if (!phase_record_writer.OpenDirectory(traces_path,
                                           &phase_authority_error)) {
      simbench_nvbit::FailPhaseRecordAuthority("early TRACES_FOLDER open/snapshot",
                                  phase_authority_error);
    }
  }

  std::string pad(100, '-');
  printf("%s\n", pad.c_str());

  if (active_from_start == 0) {
    active_region = false;
    active_launch_gate.ProfilerStop();
  }

  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&mutex, &attr);
}

// Key is unique function ID. The next key is PC and finally the Value is the datawidh of the opcode ID of the instruction.
std::map<int,std::map<int, int>>  pc_to_opcode;
// Static identities selected for opt-in host-side instrumentation ordering.
// The pair is (function_unique_id, function-relative vpc). Written during
// instrumentation and read by the receive thread under recv_container_mutex.
std::set<std::pair<uint32_t, uint32_t>> async_order_selected_pcs;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void instrument_function_if_needed(CUcontext ctx, CUfunction func, int device_id) {
  CTXstate* ctx_state = lookup_context_state_or_fail(
      ctx, "instrument_function_if_needed/context");
  device_id = checked_device_id_value(
      device_id, "instrument_function_if_needed/stop_report");
  if (device_id != ctx_state->device_id) {
    std::ostringstream detail;
    detail << "instrumentation device_id=" << device_id
           << " differs from CTXstate::device_id=" << ctx_state->device_id;
    fail_device_identity("instrument_function_if_needed/stop_report",
                         detail.str());
  }

  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    current_kernel_key_instructions_by_pc.clear();
    current_kernel_call_or_ret_by_pc.clear();
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second) {
      continue;
    }

    std::string current_kernel_name(nvbit_get_func_name(ctx, f, true));
    uint64_t addr_funct = (uint64_t)nvbit_get_func_addr(ctx, f);
    
    next_candidate_unique_function_id =
        next_candidate_function_id_or_fail(next_candidate_unique_function_id);

    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose) {
      printf("Inspecting function %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), addr_funct);
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {

      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }
      if (verbose) {
        instr->printDecoded();
      }

      const uint32_t raw_vpc = instr->getOffset();
      if (raw_vpc > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr,
                "FATAL: static SASS PC offset 0x%" PRIx32
                " exceeds the signed map-key domain\n",
                raw_vpc);
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }

      if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = instr->getOpcode();
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];
      int vpc = static_cast<int>(raw_vpc);
      const bool async_order_selected =
          is_async_order_selected_instruction(instr->getOpcodeShort(),
                                              instr->getSass());
      // pc_to_opcode is also read (without `mutex`) by recv threads at
      // recv_thread_fun:1503. Serialize writes/reads with recv_container_mutex
      // so multi-context recv threads don't race with concurrent
      // instrumentation on a different context.
      {
        std::lock_guard<std::mutex> rcm_lock(recv_container_mutex);
        pc_to_opcode[next_candidate_unique_function_id][vpc] = opcode_id;
        // Capture the per-instruction memory space so the receiver can stamp
        // `address_space` + key the allocation-identity gate on it (the
        // address-carrying contract discriminator). Keyed by opcode_id; the
        // space is a function of the opcode, so a repeat write is idempotent.
        opcode_id_to_address_space[opcode_id] =
            nvbit_memspace_to_proto(instr->getMemorySpace());
        if (async_order_enabled && async_order_selected) {
          async_order_selected_pcs.emplace(
              static_cast<uint32_t>(next_candidate_unique_function_id),
              static_cast<uint32_t>(vpc));
        }
      }

      std::string inst_str = ReplaceAll(instr->getSass(), ",", " ");
      inst_str = replaceInstructionNewExtraInformation(inst_str);
      inst_str = ReplaceAll(inst_str, ";", "");
      inst_str = strip_string(inst_str);

      if(std::string(instr->getOpcode()).find("LDGSTS") != std::string::npos) {
        opcodes_id_ldgsts.insert(opcode_id);
      }

      bool is_call_or_ret = (std::string(instr->getOpcode()).find("CALL") != std::string::npos) || (std::string(instr->getOpcode()).find("RET") != std::string::npos);
      bool is_rel_type = (std::string(instr->getOpcode()).find("REL") != std::string::npos);
      bool is_call_or_ret_with_reg = false;

      if(track_this_instruction(cnt, threshold_unique_kernel_checking ,instr->getOpcode())) {
        current_kernel_key_instructions_by_pc[vpc] = inst_str;
      }

      std::shared_ptr<traced_instruction> inst_parsed = create_no_binay_instruction(vpc, inst_str);
      if (inst_parsed == nullptr) {
        fprintf(stderr,
                "FATAL: failed to parse static SASS instruction at PC 0x%x: %s\n",
                static_cast<unsigned int>(vpc), inst_str.c_str());
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      inst_parsed->set_simulation_opcode(OpcodeMap, inst_parsed->get_op_code());

      if(is_call_or_ret && !is_rel_type) {
        current_kernel_call_or_ret_by_pc[vpc] = inst_str;
      }

      map_func_addr_to_pc_to_sass_instr[addr_funct][vpc] = inst_str;
      
      /* We only report memory addresses */
      int mem_oper_idx = -1;
      int num_mref = 0;

      bool has_ldc_with_reg = false;
      bool has_ldc_with_ureg = false;
      std::vector<uint32_t> desc_ureg_ids;
      std::map<uint32_t, uint32_t> memref_idx_with_desc;
      std::map<uint32_t, traced_operand_instrument> per_operand_type;
      int ldc_reg_id = -1;

      std::vector<int> aux_reg_ids;
      uint64_t call_ret_imm = 0;
      uint32_t num_of_injects = 0;

      int ldgsts_ignore_src_predicate = -1;
      bool ldgsts_ignore_src_predicate_is_not = false;
      if (std::string(instr->getOpcode()).find("LDGSTS") !=
          std::string::npos) {
        for (int operand = 0; operand < instr->getNumOperands(); ++operand) {
          const InstrType::operand_t *candidate = instr->getOperand(operand);
          if (candidate->type == InstrType::OperandType::PRED) {
            ldgsts_ignore_src_predicate = candidate->u.pred.num;
            ldgsts_ignore_src_predicate_is_not = candidate->is_not;
          }
        }
      }

      if(inst_parsed->is_tensor_core_op()) {
        if (!inst_parsed->set_tensor_core_instruction_info()) {
          fprintf(stderr,
                  "FATAL: unsupported tensor-core instruction shape at PC "
                  "0x%x; refusing to publish incomplete register metadata.\n",
                  instr->getOffset());
          fflush(stderr);
          _exit(EXIT_FAILURE);
        }
      }
      for(int i = 0; i < instr->getNumOperands(); ++i){
        const InstrType::operand_t *op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::MREF) {
          mem_oper_idx++;
          // LDGSTS carries [shared_dst] at MREF ordinal 0 and
          // [global_src.64] at ordinal 1. Instrument both. Calls are emitted in
          // reverse MREF order below, yielding the V3 wire order
          // (GLOBAL/GENERIC source, SHARED destination); the ordinal is carried
          // independently through inst_trace_t::reg_id so the receiver never
          // infers operand space from an allocation lookup.
          num_mref++;
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          if(op->u.mref.has_desc) {
            memref_idx_with_desc[mem_oper_idx] = op->u.mref.desc_ureg_num;
          }
          num_of_injects++;
        }else if( (op->type == InstrType::OperandType::CBANK) && (instr->getOperand(i)->u.cbank.has_reg_offset || (std::string(op->str).find("UR") != std::string::npos) ) ) {
          if(std::string(op->str).find("UR") != std::string::npos) {
            has_ldc_with_ureg = true;
            unsigned int uniform_register_id = 0;
            if (!get_ur_register(std::string(op->str), uniform_register_id) ||
                uniform_register_id > static_cast<unsigned int>(URZ)) {
              fprintf(stderr,
                      "FATAL: malformed uniform register in constant-bank "
                      "operand at PC 0x%x: %s\n",
                      static_cast<unsigned int>(vpc), op->str);
              fflush(stderr);
              _exit(EXIT_FAILURE);
            }
            ldc_reg_id = static_cast<int>(uniform_register_id);
          }else {
            has_ldc_with_reg = true;
            ldc_reg_id = instr->getOperand(i)->u.cbank.reg_offset;
          }
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::IMM_UINT64 && is_call_or_ret) {
          call_ret_imm = op->u.imm_uint64.value;
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::REG && is_call_or_ret) {
          aux_reg_ids.push_back(op->u.reg.num);
          aux_reg_ids.push_back(op->u.reg.num + 1);
          is_call_or_ret_with_reg = true;
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::REG) {
          TraceEnhancedOperandType reg_type = TraceEnhancedOperandType::REG;
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, reg_type);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::REGULAR, num_uses, op->u.reg.num);
          if(num_uses == 2) {
            per_operand_type[num_of_injects].reg_type = TRACED_REG_TYPE::REGULAR_2_REGS;
          }else if(num_uses == 4) {
            per_operand_type[num_of_injects].reg_type = TRACED_REG_TYPE::REGULAR_4_REGS;
          }
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::UREG) {
          TraceEnhancedOperandType reg_type = TraceEnhancedOperandType::UREG;
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, reg_type);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::UNIFORM, num_uses, op->u.reg.num);
          if(num_uses == 2) {
            per_operand_type[num_of_injects].reg_type = TRACED_REG_TYPE::UNIFORM_2_REGS;
          }
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::PRED) {
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, TraceEnhancedOperandType::PRED);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::PREDICATE, num_uses, op->u.reg.num);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::UPRED) {
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, TraceEnhancedOperandType::UPRED);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::UNIFORM_PREDICATE, num_uses, op->u.reg.num);
          num_of_injects++;
        }
      }

      if(num_of_injects == 0) {
        per_operand_type[0] = traced_operand_instrument(TRACED_REG_TYPE::NO_REGS, 0, 0);
        num_of_injects++;
      }
      if (num_mref > 2) {
        fprintf(stderr,
                "FATAL: instruction at PC 0x%x has %d memory references; "
                "the tracer supports at most two and will not publish "
                "ambiguous operand attribution.\n",
                instr->getOffset(), num_mref);
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }

      for(unsigned int i = 0; i < num_of_injects; i++) {
          /* insert call to the instrumentation function with its arguments */
          nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);
          /* pass predicate value */
          nvbit_add_call_arg_guard_pred_val(instr);
          /* send Unique Function Identifier and PC */
          nvbit_add_call_arg_const_val32(instr, next_candidate_unique_function_id);
          nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
          /* send number of injects and type of operand */
          nvbit_add_call_arg_const_val32(instr, num_of_injects);
          nvbit_add_call_arg_const_val32(instr, per_operand_type[i].reg_type);

          if(per_operand_type[i].reg_type == TRACED_REG_TYPE::MEMORY_REF) {
              if (mem_oper_idx >= 0) {
                nvbit_add_call_arg_const_val32(instr, MEM_TYPE::STANDARD_MEM);
              nvbit_add_call_arg_mref_addr64(instr, mem_oper_idx);
              nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());
              if(memref_idx_with_desc.find(mem_oper_idx) != memref_idx_with_desc.end()) {
                nvbit_add_call_arg_ureg_val(instr, memref_idx_with_desc[mem_oper_idx]);
              }else {
                nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              }
              // Carry the structural MREF ordinal plus LDGSTS ignore-src
              // metadata in op_reg_val_2. Only the GLOBAL source operand
              // (ordinal 1) consumes the extra predicate; the SHARED
              // destination remains active so zero-fill writes are preserved.
              uint32_t mref_metadata =
                  static_cast<uint32_t>(mem_oper_idx) & MREF_ORDINAL_MASK;
              const bool carries_ignore_src =
                  mem_oper_idx == 1 && ldgsts_ignore_src_predicate >= 0;
              if (carries_ignore_src) {
                mref_metadata |= MREF_IGNORE_SRC_PRESENT;
                if (ldgsts_ignore_src_predicate_is_not) {
                  mref_metadata |= MREF_IGNORE_SRC_NEGATED;
                }
              }
              nvbit_add_call_arg_const_val32(instr, mref_metadata);
              if (carries_ignore_src) {
                nvbit_add_call_arg_pred_val_at(
                    instr, ldgsts_ignore_src_predicate);
              } else {
                nvbit_add_call_arg_const_val32(
                    instr, SECRET_UREG_DESC_NOT_USED);
              }
              mem_oper_idx--;
            }else if(has_ldc_with_reg || has_ldc_with_ureg) {
              nvbit_add_call_arg_const_val32(instr, MEM_TYPE::CONSTANT_MEM);
              nvbit_add_call_arg_const_val64(instr, 0);
              if(has_ldc_with_ureg) {
                nvbit_add_call_arg_ureg_val(instr, ldc_reg_id);
              }else {
                nvbit_add_call_arg_reg_val(instr, ldc_reg_id);
              }
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            }else if(is_call_or_ret) {
              nvbit_add_call_arg_const_val32(instr, MEM_TYPE::CALL_OR_RET);
              nvbit_add_call_arg_const_val64(instr, call_ret_imm);
              if(is_call_or_ret_with_reg) {
                nvbit_add_call_arg_reg_val(instr, aux_reg_ids[0]);
                nvbit_add_call_arg_reg_val(instr, aux_reg_ids[1]);
              }else {
                nvbit_add_call_arg_const_val32(instr, 0);
                nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              }
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            }
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::REGULAR) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id); 
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::REGULAR_2_REGS) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 1);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::REGULAR_4_REGS) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 1);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 2);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 3);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::UNIFORM) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_ureg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::UNIFORM_2_REGS) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_ureg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_ureg_val(instr, per_operand_type[i].first_reg_id + 1);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::PREDICATE) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_pred_reg(instr);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::UNIFORM_PREDICATE) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_upred_reg(instr);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else {
            if (per_operand_type[i].reg_type != TRACED_REG_TYPE::NO_REGS ||
                num_of_injects != 1) {
              fprintf(stderr,
                      "FATAL: unsupported instrumentation operand layout at "
                      "PC 0x%x; refusing to publish a malformed record.\n",
                      instr->getOffset());
              fflush(stderr);
              _exit(EXIT_FAILURE);
            }
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }

          /* add pointer to channel_dev and other counters*/
          nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
          nvbit_add_call_arg_const_val64(instr,
                                        (uint64_t)&total_dynamic_instr_counter);
          nvbit_add_call_arg_const_val64(instr,
                                        (uint64_t)&reported_dynamic_instr_counter);
          nvbit_add_call_arg_const_val64(instr, (uint64_t)&stop_report[device_id]);
      }
      cnt++;
    }

    // BEGIN_PHASE_INJECTION_BLOCK
    /* Closure-plan NVBit phase-injection contract: second pass over
     * the function's instructions that inserts paired open/close
     * calls to `record_phase_clock` at the four boundary classes
     * named in `phase_cycle_ledger.schema.json`'s `phase_kind` enum
     * (function_entry, function_exit, basic_block_entry, bar_sync).
     *
     * phase ID assignment is deterministic per function:
     *   - phase_id `next_phase_id_counter++` is the function_entry
     *     pair (open at the first instruction, close at every
     *     function-terminator instruction). A thread executes exactly
     *     one terminator in a function, so the paired close fires
     *     exactly once per warp per kernel run.
     *   - Each subsequent boundary location (first BAR, first BB-entry
     *     following a branch, first function-terminator for
     *     function_exit) gets its own `next_phase_id_counter++` so the
     *     per-warp pairing key in the host aggregator does not collide.
     *
     * Only the FIRST occurrence of each marker boundary is
     * instrumented (warp barrier, basic-block-entry, function_exit).
     * Multi-execution boundaries (e.g., a BAR.SYNC inside a loop)
     * would otherwise produce multiple opens with the same key,
     * which the host aggregator's PhaseRecordDuplicateOpen
     * diagnostic correctly rejects. Robust loop-aware insertion is
     * a follow-up refinement once a basic-block CFG analysis pass
     * is wired in; the first-occurrence rule is conservative but
     * produces correct (if sparse) records for an MVP producer.
     *
     * --- Mnemonic classification contract (exact-match, short form) ---
     * The three predicates below consume `instr->getOpcodeShort()` and
     * exact-match against canonical short mnemonics. The legacy
     * implementation used substring search over the full mnemonic
     * returned by `getOpcode()`; that search over-counted on every
     * compiled kernel because a bare BAR substring matched the
     * memory-fence family (MEMBAR / DEPBAR / LDGDEPBAR / ERRBAR /
     * CGAERRBAR), and a bare RET substring could match Pascal PRET.
     * The contract is:
     *   - Warp barrier (phase_kind=3) is `getOpcodeShort() == "BAR"`
     *     only. This catches the entire BAR.SYNC.* family because NVBit
     *     normalizes them all to short form `BAR`. Memory-fence /
     *     dependency-barrier opcodes (`MEMBAR`, `DEPBAR`, `LDGDEPBAR`,
     *     `ERRBAR`, `CGAERRBAR`) are NOT instrumented as warp barriers.
     *     Newer-architecture barrier mnemonics (`SYNCS`, `WARPGROUP`,
     *     `WARPGROUPSET`, `UCGABAR.*`) are also NOT instrumented as
     *     warp barriers in this MVP; if a future ledger consumer needs
     *     them, extend `is_warp_barrier_opcode_short` and add coverage
     *     to `phase_classifier_test.cc`.
     *   - Function terminator (phase_kind=0 close + phase_kind=1
     *     open/close pair) covers `RET` and `EXIT`. A device function
     *     can return via either; both must close the function-entry
     *     pair AND emit the function-exit open/close pair. The shared
     *     `is_function_terminator_opcode_short` predicate is used at
     *     this single conditional, which contains both call sites.
     *   - Branch (drives basic_block_entry phase_kind=2 on the
     *     instruction following a branch) covers
     *     `{BRA, BRX, JMP, BSSY, BSYNC, EXIT, RET}`. `CALL` is NOT
     *     classified as a branch terminator here: it is routed through
     *     the existing CALL_OPS tracing path, and treating it as a
     *     basic-block terminator would double-count.
     *   - The branch heuristic is a linear MVP. `BSSY` / `BSYNC` are
     *     recorded as branch-terminator candidates because compiled
     *     sm_86+ code uses them for divergent control flow; they are
     *     NOT true CFG-determined basic-block boundaries. The
     *     structural CFG-based replacement using `nvbit_get_CFG()` is
     *     a documented follow-up (canonical example at
     *     `tools/nvbit_tracer/nvbit_release/tools/instr_count_bb/`
     *     `instr_count.cu:157-171`). Until that lands the predicates
     *     here remain heuristic and over-cover slightly rather than
     *     under-cover.
     */
    if (phase_injection_enabled && ctx_state->phase_channel_dev != nullptr) {
      static std::atomic<uint32_t> next_phase_id_counter{0};
      uint32_t function_phase_id =
          next_phase_id_counter.fetch_add(1, std::memory_order_relaxed);
      bool is_first_instr = true;
      bool prev_was_branch = false;
      bool inserted_first_bar_sync = false;
      bool inserted_first_bb_entry = false;
      bool function_exit_phase_assigned = false;
      uint32_t function_exit_phase_id = 0;
      uint32_t cnt2 = 0;
      for (auto instr : instrs) {
        if (cnt2 < instr_begin_interval || cnt2 >= instr_end_interval) {
          cnt2++;
          continue;
        }
        std::string_view opcode_short(instr->getOpcodeShort());
        bool is_function_terminator =
            is_function_terminator_opcode_short(opcode_short);
        bool is_bar = is_warp_barrier_opcode_short(opcode_short);
        bool is_branch = is_branch_opcode_short(opcode_short);

        /* Helper macro mirroring the boundary-call argument list:
         * 5 args per call match `record_phase_clock`'s signature
         * in `inject_funcs.cu`. Inlined here to avoid an extra
         * helper symbol in this translation unit. */
        #define INSERT_PHASE_CALL(p_id, p_kind, p_role) do {                  \
          nvbit_insert_call(instr, "record_phase_clock", IPOINT_BEFORE);      \
          nvbit_add_call_arg_const_val32(instr, (p_id));                      \
          nvbit_add_call_arg_const_val32(instr, (p_kind));                    \
          nvbit_add_call_arg_const_val32(instr, (p_role));                    \
          nvbit_add_call_arg_const_val64(instr,                               \
              (uint64_t)&simbench_phase_launch_index_for_inject);             \
          nvbit_add_call_arg_const_val64(instr,                               \
              (uint64_t)ctx_state->phase_channel_dev);                        \
        } while (0)

        /* function_entry kind (0): open at the first instruction. */
        if (is_first_instr) {
          INSERT_PHASE_CALL(function_phase_id,
                            /*phase_kind=*/0u, /*clock_role=open*/0u);
        }

        /* function_entry kind (0): close at every function-terminator
         * (RET or EXIT). Same conditional also emits the function_exit
         * kind (1) paired open/close marker so EXIT-ending kernels and
         * RET-ending kernels behave identically (one shared predicate
         * applied at both call sites). */
        if (is_function_terminator) {
          INSERT_PHASE_CALL(function_phase_id,
                            /*phase_kind=*/0u, /*clock_role=close*/1u);
          /* function_exit kind (1): paired open/close marker at
           * the same terminator instruction. Cycle delta will be
           * small (~one %clock64 read latency), but the boundary
           * IS recorded. */
          if (!function_exit_phase_assigned) {
            function_exit_phase_id =
                next_phase_id_counter.fetch_add(1,
                                                std::memory_order_relaxed);
            function_exit_phase_assigned = true;
          }
          INSERT_PHASE_CALL(function_exit_phase_id,
                            /*phase_kind=*/1u, /*clock_role=open*/0u);
          INSERT_PHASE_CALL(function_exit_phase_id,
                            /*phase_kind=*/1u, /*clock_role=close*/1u);
        }

        /* bar_sync kind (3): paired marker at the first warp barrier. */
        if (is_bar && !inserted_first_bar_sync) {
          uint32_t bar_phase_id =
              next_phase_id_counter.fetch_add(1, std::memory_order_relaxed);
          INSERT_PHASE_CALL(bar_phase_id,
                            /*phase_kind=*/3u, /*clock_role=open*/0u);
          INSERT_PHASE_CALL(bar_phase_id,
                            /*phase_kind=*/3u, /*clock_role=close*/1u);
          inserted_first_bar_sync = true;
        }

        /* basic_block_entry kind (2): paired marker at the first
         * basic-block-entry instruction (the first instruction
         * following a branch instruction, excluding the very first
         * instruction of the function which is already covered by
         * function_entry). */
        if (prev_was_branch && !is_first_instr && !inserted_first_bb_entry) {
          uint32_t bb_phase_id =
              next_phase_id_counter.fetch_add(1, std::memory_order_relaxed);
          INSERT_PHASE_CALL(bb_phase_id,
                            /*phase_kind=*/2u, /*clock_role=open*/0u);
          INSERT_PHASE_CALL(bb_phase_id,
                            /*phase_kind=*/2u, /*clock_role=close*/1u);
          inserted_first_bb_entry = true;
        }

        #undef INSERT_PHASE_CALL

        is_first_instr = false;
        prev_was_branch = is_branch;
        cnt2++;
      }
    }
    // END_PHASE_INJECTION_BLOCK

    pthread_mutex_lock(&mutex);
    int variant_id = 0;
    auto it_map_already_instrumented = map_function_to_kernel_name_and_variant_id.find(f);
    if (it_map_already_instrumented !=
        map_function_to_kernel_name_and_variant_id.end()) {
      fprintf(stderr,
              "FATAL: function was already assigned an instrumentation "
              "identity before registration completed; refusing to overwrite "
              "the existing launch mapping.\n");
      fflush(stderr);
      pthread_mutex_unlock(&mutex);
      _exit(EXIT_FAILURE);
    }

    auto it_already_traced = all_kernels_key_instructions_by_pc.find(current_kernel_name);
    if (it_already_traced == all_kernels_key_instructions_by_pc.end())
    {
      all_kernels_key_instructions_by_pc[current_kernel_name].push_back(traced_kernel_id(current_kernel_name, 0, next_candidate_unique_function_id, current_kernel_key_instructions_by_pc, current_kernel_call_or_ret_by_pc, addr_funct));
    }
    else
    {
      bool is_already_traced = false;
      for (unsigned int i = 0; !is_already_traced && (i < it_already_traced->second.size()); i++)
      {
        if (are_two_kernels_equal(it_already_traced->second[i], current_kernel_key_instructions_by_pc, current_kernel_call_or_ret_by_pc, true)) // CAMBIAR A FUNCION
        {
          is_already_traced = true;
          variant_id = it_already_traced->second[i].variant_id;
        }
      }

      if (!is_already_traced)
      {
        variant_id = it_already_traced->second.size();
        all_kernels_key_instructions_by_pc[current_kernel_name].push_back(traced_kernel_id(current_kernel_name, it_already_traced->second.size(), next_candidate_unique_function_id, current_kernel_key_instructions_by_pc, current_kernel_call_or_ret_by_pc, addr_funct));
      }
    }
    
    std::string candidate_final_kernel_name = current_kernel_name + variant_delimiter_str + std::to_string(variant_id);
    map_function_to_kernel_name_and_variant_id[f] = std::make_tuple(current_kernel_name, variant_id);
    map_function_name_to_unique_function_id_with_variant[candidate_final_kernel_name] = next_candidate_unique_function_id;
    map_function_name_to_unique_function_id_without_variant[current_kernel_name] = next_candidate_unique_function_id;
    map_func_addr_to_kernel_name[addr_funct] = current_kernel_name;
    map_kernel_name_to_func_addr[current_kernel_name] = addr_funct;
    pthread_mutex_unlock(&mutex);
  }
}

__global__ void flush_channel(ChannelDev* ch_dev) {
  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  inst_trace_t ma;
  ma.cta_id_x = -1;
  ch_dev->push(&ma, sizeof(inst_trace_t));

  /* flush channel */
  ch_dev->flush();
}

/* phase-channel flush kernel for the non-root measurement NVBit
 * phase-injection contract. Pushes a sentinel
 * `phase_clock_record_t` with `cta_id_x = -1` so the phase recv
 * thread can detect the per-launch drain boundary the same way
 * `recv_thread_fun` detects the instruction-trace drain boundary.
 *
 * The sentinel's `launch_index` field carries the host-side
 * `this_launch_id` of the kernel that just finished. The recv
 * thread always stores `launch_index` into the per-context
 * `phase_completed_flush_launch_index` atomic; the host post-launch
 * path waits for `completed == this_launch_id`. An unexpected stale sentinel
 * publishes its own generation and cannot spuriously satisfy a different
 * launch's wait. The recv thread also compares `launch_index` against
 * `phase_expected_flush_launch_index` and emits a stale-sentinel
 * `WARNING:` when the values disagree; that diagnostic is not
 * load-bearing for safety.
 */
__global__ void flush_phase_channel(ChannelDev* ch_dev,
                                    uint32_t launch_index) {
  phase_clock_record_t pcr;
  pcr.cta_id_x = -1;
  pcr.cta_id_y = 0;
  pcr.cta_id_z = 0;
  pcr.sm_id = 0;
  pcr.warpid_tb = 0;
  pcr.phase_id = 0;
  pcr.phase_kind = 0;
  pcr.clock_role = 0;
  pcr.clock64 = 0;
  pcr.active_mask = 0;
  pcr.launch_index = launch_index;
  ch_dev->push(&pcr, sizeof(phase_clock_record_t));
  ch_dev->flush();
}

/* phase-channel receive thread. Mirrors `recv_thread_fun`'s
 * lifecycle (WORKING / STOP / FINISHED handshake + drain on
 * sentinel) but consumes `phase_clock_record_t` records and
 * appends them to the per-context `phase_records_buffer`. The host
 * main thread snapshots-and-clears that buffer at every per-launch
 * flush boundary so the buffer represents exactly one launch's
 * phase records, ready to be written as a per-launch
 * `phase_records_<repeat>_<launch>.bin` file. */
/* Test-only synthetic-stall hook: when a sentinel for launch N is
 * about to acknowledge, sched_yield K times before doing so. Set
 * `SIMBENCH_PHASE_TEST_STALL_LAUNCH=N` to pick the target launch and
 * `SIMBENCH_PHASE_TEST_STALL_ITERS=K` (default 1e8) to choose the
 * stall length. With a small `SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS`
 * cap, this forces the post-launch timeout branch deterministically so a live
 * smoke run can verify fatal rejection and noncanonical partial retention.
 *
 * Disabled (no-op) unless the env var is set. Reads cached on first
 * call. A negative or zero target launch disables the hook entirely.
 */
static int phase_test_stall_target_launch() {
  static int val = []() {
    const char *env = std::getenv("SIMBENCH_PHASE_TEST_STALL_LAUNCH");
    if (env != nullptr && env[0] != '\0') {
      char *endp = nullptr;
      long parsed = std::strtol(env, &endp, 10);
      if (endp != env && *endp == '\0' && parsed > 0 && parsed <= INT_MAX) {
        return (int)parsed;
      }
    }
    return -1;  // disabled
  }();
  return val;
}

static long long phase_test_stall_iters() {
  static long long val = []() {
    const char *env = std::getenv("SIMBENCH_PHASE_TEST_STALL_ITERS");
    if (env != nullptr && env[0] != '\0') {
      char *endp = nullptr;
      long long parsed = std::strtoll(env, &endp, 10);
      if (endp != env && *endp == '\0' && parsed >= 0) {
        return parsed;
      }
    }
    return (long long)100000000;  // 1e8 default; long enough to
                                   // exhaust a small flush-timeout cap
  }();
  return val;
}

/* Test-only teardown stall: sched_yield K times after STOP is
 * signaled but before setting FINISHED. Set
 * `SIMBENCH_PHASE_TEST_STALL_TEARDOWN_ITERS=K` to enable. With a
 * small `SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS` cap, this forces the
 * teardown timeout branch to fire deterministically and prove the
 * intentional-leak diagnostic path. Reads cached on first call.
 */
static long long phase_test_stall_teardown_iters() {
  static long long val = []() {
    const char *env = std::getenv("SIMBENCH_PHASE_TEST_STALL_TEARDOWN_ITERS");
    if (env != nullptr && env[0] != '\0') {
      char *endp = nullptr;
      long long parsed = std::strtoll(env, &endp, 10);
      if (endp != env && *endp == '\0' && parsed >= 0) {
        return parsed;
      }
    }
    return (long long)0;  // disabled
  }();
  return val;
}

void *phase_recv_thread_fun(void *args) {
  CUcontext ctx = (CUcontext)args;
  pthread_mutex_lock(&mutex);
  CTXstate* ctx_state = lookup_context_state_or_fail(
      ctx, "phase_recv_thread_fun/context");
  checked_stored_context_device_id(ctx_state,
                                   "phase_recv_thread_fun/context device");
  ChannelHost* ch_host = &ctx_state->phase_channel_host;
  pthread_mutex_unlock(&mutex);

  char *recv_buffer = static_cast<char *>(malloc(CHANNEL_SIZE));
  if (recv_buffer == nullptr) {
    fail_device_identity("phase_recv_thread_fun/receive buffer allocation",
                         "malloc(CHANNEL_SIZE) returned null");
  }
  while (ctx_state->phase_recv_thread_done.load(std::memory_order_acquire) ==
         RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
    if (num_recv_bytes > 0) {
      require_record_alignment(num_recv_bytes, sizeof(phase_clock_record_t),
                               "phase");
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        phase_clock_record_t *pcr =
            (phase_clock_record_t *)&recv_buffer[num_processed_bytes];
        if (pcr->cta_id_x == -1) {
          /* Per-launch flush sentinel. Always publish the
           * sentinel's launch_index into the per-context completed
           * counter; the host post-launch path waits for
           * `completed == this_launch_id`; an unexpected stale sentinel
           * stores only its own generation and cannot acknowledge a different
           * launch. The generation-valued counter makes the handshake
           * race-free under arbitrary recv-thread scheduling.
           */
          uint32_t sentinel_launch = pcr->launch_index;
          /* Synthetic-stall hook: deterministic test injection that
           * delays the acknowledgment of a chosen launch's sentinel
           * long enough for the host's bounded-spin cap to expire.
           * Default-OFF (negative target). */
          int stall_target = phase_test_stall_target_launch();
          if (stall_target > 0 &&
              (uint32_t)stall_target == sentinel_launch) {
            long long stall = phase_test_stall_iters();
            for (long long i = 0; i < stall; ++i) {
              sched_yield();
            }
          }
          /* Reload `expected` AFTER the stall so the stale-sentinel
           * diagnostic compares against the current host expectation,
           * not a cached value from before the stall. */
          uint32_t expected_launch =
              ctx_state->phase_expected_flush_launch_index.load(
                  std::memory_order_acquire);
          if (sentinel_launch != expected_launch) {
            ctx_state->phase_stale_sentinel_count.fetch_add(
                1, std::memory_order_relaxed);
            fprintf(stderr,
                    "WARNING: stale phase flush sentinel "
                    "launch_index=%u (expected=%u); the generation-valued "
                    "completed counter cannot acknowledge the wrong launch. "
                    "If the expected sentinel does not arrive within the "
                    "configured bound, capture exits fatally before canonical "
                    "phase publication.\n",
                    (unsigned)sentinel_launch, (unsigned)expected_launch);
          }
          ctx_state->phase_completed_flush_launch_index.store(
              sentinel_launch, std::memory_order_release);
          break;
        }
        /* Append the phase record bytes to the per-context buffer
         * under the buffer mutex so the main thread's snapshot is
         * coherent at the per-launch boundary. */
        {
          std::lock_guard<std::mutex> blk(ctx_state->phase_records_buffer_mutex);
          if (ctx_state->phase_records_buffer.size() >
              MAX_PHASE_RECORD_BUFFER_BYTES - sizeof(phase_clock_record_t)) {
            fprintf(stderr,
                    "FATAL: phase-record buffer exceeded %zu-byte bound; "
                    "refusing an unbounded host allocation\n",
                    MAX_PHASE_RECORD_BUFFER_BYTES);
            fflush(stderr);
            _exit(EXIT_FAILURE);
          }
          try {
            ctx_state->phase_records_buffer.insert(
                ctx_state->phase_records_buffer.end(),
                (char*)pcr,
                (char*)pcr + sizeof(phase_clock_record_t));
          } catch (const std::bad_alloc &) {
            fprintf(stderr,
                    "FATAL: phase-record buffer allocation failed at %zu "
                    "bytes\n",
                    ctx_state->phase_records_buffer.size());
            fflush(stderr);
            _exit(EXIT_FAILURE);
          }
        }
        num_processed_bytes += sizeof(phase_clock_record_t);
      }
    }
  }
  free(recv_buffer);
  /* Synthetic teardown-stall hook: sched_yield K times before
   * setting FINISHED. Used by the teardown-timeout regression test. */
  long long teardown_stall = phase_test_stall_teardown_iters();
  for (long long i = 0; i < teardown_stall; ++i) {
    sched_yield();
  }
  ctx_state->phase_recv_thread_done.store(RecvThreadState::FINISHED,
                                          std::memory_order_release);
  return NULL;
}

static FILE *statsFile = NULL;
static bool first_call = true;
static unsigned int pending_devices_to_finish = 0;

static FILE *open_stats_file(const char *mode) {
  FILE *file = fopen(stats_location.c_str(), mode);
  if (file == nullptr) {
    fprintf(stderr, "FATAL: could not open tracer stats file %s with mode %s: %s\n",
            stats_location.c_str(), mode, strerror(errno));
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
  return file;
}

static void fatal_stats_io(const char *operation) {
  const char *detail = errno != 0 ? strerror(errno) : "unknown I/O error";
  fprintf(stderr, "FATAL: tracer stats I/O failed during %s for %s: %s\n",
          operation, stats_location.c_str(), detail);
  fflush(stderr);
  _exit(EXIT_FAILURE);
}

static void checked_stats_write(int result, const char *operation) {
  if (result < 0) fatal_stats_io(operation);
}

static void checked_stats_close(FILE *file) {
  if (fclose(file) != 0) fatal_stats_io("close");
}

static uint64_t checked_dimension_product(uint64_t lhs, uint64_t rhs,
                                          const char *label) {
  if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
    fprintf(stderr,
            "FATAL: launch %s product overflows the uint64 stats domain "
            "(lhs=%" PRIu64 ", rhs=%" PRIu64 ")\n",
            label, lhs, rhs);
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
  return lhs * rhs;
}

static void init_context_state(CUcontext ctx);
static int resolve_binary_version_for_current_device();

static void checked_cuda_runtime_status(cudaError_t status,
                                        const char *operation) {
  if (status != cudaSuccess) {
    fprintf(stderr, "FATAL: CUDA runtime call %s failed: %s\n", operation,
            cudaGetErrorString(status));
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
}

/* The instruction tracer's launch state is scoped to one ordinary
 * cuLaunchKernel callback: one pending launch, one per-context channel
 * handshake, and one active stream/kernel identity.  CUDA graph replay does
 * not provide that entry/exit pair for each node; NVBit exposes the separate
 * nvbit_at_graph_node_launch() callback instead.  Until this tracer has a
 * graph-specific queue and per-node channel identities, accepting a graph
 * launch would produce an apparently successful but empty (or misattributed)
 * dynamic_trace.pb.  Fail at the graph boundary so callers cannot mistake
 * that artifact for a valid capture.
 */
static bool is_graph_launch_callback(nvbit_api_cuda_t cbid) {
  return cbid == API_CUDA_cuGraphLaunch ||
         cbid == API_CUDA_cuGraphLaunch_ptsz;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  // std::cout << "nvbit_at_cuda_event: "  << cbid << std::endl; fflush(stdout);
  pthread_mutex_lock(&mutex);
  /* we prevent re-entry on this callback when issuing CUDA functions inside
    * this function */
  if (skip_callback_flag) {
      pthread_mutex_unlock(&mutex);
      return;
  }

  // cuLibrary* is context-independent and may run before any context exists.
  if (is_module_origin_event(cbid)) {
    handle_module_origin_event(ctx, is_exit, cbid, params, pStatus);
    pthread_mutex_unlock(&mutex);
    return;
  }

  CTXstate* ctx_state = lookup_context_state_or_fail(
      ctx, "nvbit_at_cuda_event/context");

  // A graph used while the profiler is inactive is outside the requested
  // capture region and must not change application behavior. Reject only a
  // graph launch that would otherwise be part of the active trace.
  if (is_graph_launch_callback(cbid) && active_region) {
    fprintf(stderr,
            "FATAL: graph_node_capture_unsupported: %s observed, but "
            "tracer_tool has no per-node channel identity/flush path; "
            "refusing to publish an empty or misattributed graph capture.\n",
            is_exit ? "cuGraphLaunch exit" : "cuGraphLaunch entry");
    fflush(stderr);
    pthread_mutex_unlock(&mutex);
    _exit(EXIT_FAILURE);
  }

  const bool capture_begin =
      cbid == API_CUDA_cuStreamBeginCapture ||
      cbid == API_CUDA_cuStreamBeginCapture_ptsz ||
      cbid == API_CUDA_cuStreamBeginCapture_v2 ||
      cbid == API_CUDA_cuStreamBeginCapture_v2_ptsz;
  const bool capture_end = cbid == API_CUDA_cuStreamEndCapture ||
                           cbid == API_CUDA_cuStreamEndCapture_ptsz;
  // Commit capture bookkeeping only after the API succeeds.  An entry-side
  // increment would leave a stale count when cuStreamBeginCapture rejects the
  // request, causing later ordinary launches to defer initialization forever.
  if (capture_begin && is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
    ++active_capture_counts[ctx];
  } else if (capture_end && is_exit && pStatus &&
             *pStatus == CUDA_SUCCESS) {
    auto capture_it = active_capture_counts.find(ctx);
    if (capture_it != active_capture_counts.end()) {
      if (capture_it->second <= 1) {
        active_capture_counts.erase(capture_it);
      } else {
        --capture_it->second;
      }
    }
    // A deferred tool initialization is completed only after every capture
    // owned by this context has ended, so its CUDA allocations are legal.
    auto state_it = ctx_state_map.find(ctx);
    if (state_it != ctx_state_map.end() && state_it->second != nullptr &&
        state_it->second->tool_init_deferred &&
        active_capture_counts.find(ctx) == active_capture_counts.end()) {
      CTXstate *deferred_state = state_it->second;
      skip_callback_flag = true;
      init_context_state(ctx);
      deferred_state->binary_version = resolve_binary_version_for_current_device();
      binary_version = deferred_state->binary_version;
      deferred_state->tool_init_deferred = false;
      deferred_state->tool_init_complete = true;
      skip_callback_flag = false;
    }
  }

  if (first_call == true) {

    first_call = false;

    // Output globals were resolved in nvbit_at_init. In phase mode that hook
    // also opened and snapshotted TRACES_FOLDER before any CUDA callback.
    create_folder(traces_path.c_str());

    if (active_from_start &&
        (!dynamic_kernel_limit_start || dynamic_kernel_limit_start == 1)) {
      active_region = true;
      active_launch_gate.ProfilerStart();
    } else {
      if (active_from_start) {
        active_region = false;
        active_launch_gate.ProfilerStop();
      }
    }

    statsFile = open_stats_file("w");
    const int header_written = fprintf(statsFile,
            "device_id, stream_id, kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, "
            "#blocks, block_dimX, block_dimY, block_dimZ, #threads, "
            "total_insts, total_reported_insts\n");
    checked_stats_write(header_written, "header write");
    checked_stats_close(statsFile);
  }

  const bool is_h2d_sync =
      cbid == API_CUDA_cuMemcpyHtoD ||
      cbid == API_CUDA_cuMemcpyHtoD_v2 ||
      cbid == API_CUDA_cuMemcpyHtoD_v2_ptds;
  const bool is_h2d_async =
      cbid == API_CUDA_cuMemcpyHtoDAsync ||
      cbid == API_CUDA_cuMemcpyHtoDAsync_v2 ||
      cbid == API_CUDA_cuMemcpyHtoDAsync_v2_ptsz;
  if (is_h2d_sync || is_h2d_async) {
    if (!is_exit && active_launch_gate.region_active()) {
      // Synchronous copies target either the legacy or per-thread default
      // stream, while asynchronous copies carry an explicit stream. The
      // trace wire carries driver-assigned IDs, so resolve the matching
      // default-stream variant rather than assuming stream zero.
      const bool per_thread_default_memcpy =
          cbid == API_CUDA_cuMemcpyHtoD_v2_ptds ||
          cbid == API_CUDA_cuMemcpyHtoDAsync_v2_ptsz;
      CUdeviceptr dst_device = 0;
      size_t byte_count = 0;
      CUstream copy_stream = nullptr;
      if (cbid == API_CUDA_cuMemcpyHtoD) {
        const auto *p = static_cast<const cuMemcpyHtoD_params *>(params);
        dst_device = static_cast<CUdeviceptr>(p->dstDevice);
        byte_count = static_cast<size_t>(p->ByteCount);
      } else if (cbid == API_CUDA_cuMemcpyHtoD_v2) {
        const auto *p = static_cast<const cuMemcpyHtoD_v2_params *>(params);
        dst_device = p->dstDevice;
        byte_count = p->ByteCount;
      } else if (cbid == API_CUDA_cuMemcpyHtoD_v2_ptds) {
        const auto *p =
            static_cast<const cuMemcpyHtoD_v2_ptds_params *>(params);
        dst_device = p->dstDevice;
        byte_count = p->ByteCount;
      } else if (cbid == API_CUDA_cuMemcpyHtoDAsync) {
        const auto *p =
            static_cast<const cuMemcpyHtoDAsync_params *>(params);
        dst_device = static_cast<CUdeviceptr>(p->dstDevice);
        byte_count = static_cast<size_t>(p->ByteCount);
        copy_stream = p->hStream;
      } else if (cbid == API_CUDA_cuMemcpyHtoDAsync_v2) {
        const auto *p =
            static_cast<const cuMemcpyHtoDAsync_v2_params *>(params);
        dst_device = p->dstDevice;
        byte_count = p->ByteCount;
        copy_stream = p->hStream;
      } else {
        const auto *p = static_cast<const cuMemcpyHtoDAsync_v2_ptsz_params *>(
            params);
        dst_device = p->dstDevice;
        byte_count = p->ByteCount;
        copy_stream = p->hStream;
      }
      const int64_t stream_key =
          stream_ordinal_for(copy_stream, per_thread_default_memcpy);
      char buffer[128];
      const int device_id = checked_current_context_device_id(
          ctx_state, "nvbit_at_cuda_event/cuMemcpyHtoD_v2");
      const int written = snprintf(
          buffer, sizeof(buffer), "MemcpyHtoD,0x%016" PRIx64 ",%zu",
          static_cast<uint64_t>(dst_device), byte_count);
      if (written < 0 || static_cast<size_t>(written) >= sizeof(buffer)) {
        fprintf(stderr, "FATAL: CUDA memcpy event description exceeded its buffer\n");
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      // Serialize dyn_trace mutation against any concurrent recv-thread
      // writes on other contexts.
      {
        std::lock_guard<std::mutex> rcm_lock(recv_container_mutex);
        dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[device_id];
        dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
        stream.set_id(stream_key);
        stream.add_ordered_cuda_events(buffer);
      }
    }

  } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
             cbid == API_CUDA_cuLaunchKernel ||
             cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
             cbid == API_CUDA_cuLaunchKernelEx) {
    // Normalize the cuLaunchKernel / cuLaunchKernelEx parameter surfaces
    // into a single set of locals. cuLaunchKernelEx is the ONLY launch path
    // that honors cluster-dimension attributes (cudaLaunchAttributeClusterDimension),
    // so without this branch every thread-block-cluster kernel in the
    // SM120 cluster microbenchmark suite bypasses instrument_function_if_needed
    // and produces zero-kernel traces.
    //
    // Parameter layouts:
    //   nvbit_release/core/generated_cuda_meta.h:3590-3604  (struct fields)
    //   nvbit_release/core/cuda.h:2480-2491                 (CUlaunchConfig)
    // Reference pattern (switch on cbid, extract func + dims):
    //   nvbit_release/tools/instr_count/instr_count.cu:215-225,269-277
    CUfunction kernel_func;
    CUstream kernel_stream;
    unsigned grid_dim_x, grid_dim_y, grid_dim_z;
    unsigned block_dim_x, block_dim_y, block_dim_z;
    unsigned shared_mem_bytes;
    int shared_memory_carveout_preference_pct = -1;
    // Cluster dimension from cudaLaunchAttributeClusterDimension /
    // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION (id=4 per
    // nvbit_release/core/cuda.h:2207). Only cuLaunchKernelEx
    // carries launch attributes; plain cuLaunchKernel has no cluster
    // support. Default (1,1,1) represents a non-cluster launch so the
    // captured trace can distinguish a 4-block cluster from a plain
    // 4-block grid without cluster topology.
    unsigned cluster_dim_x = 1, cluster_dim_y = 1, cluster_dim_z = 1;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params *p_ex = (cuLaunchKernelEx_params *)params;
      kernel_func      = p_ex->f;
      kernel_stream    = p_ex->config->hStream;
      grid_dim_x       = p_ex->config->gridDimX;
      grid_dim_y       = p_ex->config->gridDimY;
      grid_dim_z       = p_ex->config->gridDimZ;
      block_dim_x      = p_ex->config->blockDimX;
      block_dim_y      = p_ex->config->blockDimY;
      block_dim_z      = p_ex->config->blockDimZ;
      shared_mem_bytes = p_ex->config->sharedMemBytes;
      // Walk the launch-attribute array for CLUSTER_DIMENSION (id=4).
      // See nvbit_release/core/cuda.h:2480-2491 for the
      // CUlaunchConfig struct and :2374-2398 for
      // CUlaunchAttributeValue::clusterDim.
      if (p_ex->config->attrs != nullptr) {
        for (unsigned i = 0; i < p_ex->config->numAttrs; ++i) {
          if (p_ex->config->attrs[i].id ==
              CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
            cluster_dim_x = p_ex->config->attrs[i].value.clusterDim.x;
            cluster_dim_y = p_ex->config->attrs[i].value.clusterDim.y;
            cluster_dim_z = p_ex->config->attrs[i].value.clusterDim.z;
          }
          if (p_ex->config->attrs[i].id ==
              CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT) {
            shared_memory_carveout_preference_pct = static_cast<int>(
                p_ex->config->attrs[i].value.sharedMemCarveout);
          }
        }
      }
    } else {
      cuLaunchKernel_params *p_ker = (cuLaunchKernel_params *)params;
      kernel_func      = p_ker->f;
      kernel_stream    = p_ker->hStream;
      grid_dim_x       = p_ker->gridDimX;
      grid_dim_y       = p_ker->gridDimY;
      grid_dim_z       = p_ker->gridDimZ;
      block_dim_x      = p_ker->blockDimX;
      block_dim_y      = p_ker->blockDimY;
      block_dim_z      = p_ker->blockDimZ;
      shared_mem_bytes = p_ker->sharedMemBytes;
      // cluster_dim_* already initialized to (1,1,1) — plain
      // cuLaunchKernel has no cluster support.
    }
    // A launch attribute overrides the function preference. For ordinary
    // launches, retain the function-level preference when the driver exposes
    // it; absence is a typed trace metadata gap, not zero shared memory.
    if (shared_memory_carveout_preference_pct < 0) {
      int function_carveout = -1;
      const CUresult carveout_res = cuFuncGetAttribute(
          &function_carveout,
          CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
          kernel_func);
      if (carveout_res == CUDA_SUCCESS) {
        shared_memory_carveout_preference_pct = function_carveout;
      }
    }
    if (shared_memory_carveout_preference_pct < -1 ||
        shared_memory_carveout_preference_pct > 100) {
      fprintf(stderr,
              "FATAL: shared_memory_carveout_attribute_invalid: %d\n",
              shared_memory_carveout_preference_pct);
      fflush(stderr);
      _exit(EXIT_FAILURE);
    }
    // cuLaunchKernel/cuLaunchKernelEx require a current context. Query it only
    // after CBID dispatch, then status/bounds-check and compare it with the
    // immutable CTXstate identity before any per-device array access.
    const int launch_device_id = checked_current_context_device_id(
        ctx_state, "nvbit_at_cuda_event/kernel launch");
    if (!is_exit) {
      const int device_id = launch_device_id;
      pending_accepted_launch = PendingAcceptedLaunch{};
      if (active_from_start && dynamic_kernel_limit_start &&
          kernel_id[device_id].load(std::memory_order_acquire) ==
              dynamic_kernel_limit_start) {
        active_region = true;
        active_launch_gate.ProfilerStart();
      }

      if (terminate_after_limit_number_of_kernels_reached && dynamic_kernel_limit_end != 0 && kernel_id[device_id].load(std::memory_order_acquire) > dynamic_kernel_limit_end)
      {
        pthread_mutex_unlock(&mutex);
        exit(0);
      }

      const nvbit_tracer::ActiveLaunchDecision launch_decision =
          active_launch_gate.BeginLaunch();
      if (launch_decision ==
          nvbit_tracer::ActiveLaunchDecision::kConcurrentLimitViolation) {
        fprintf(stderr,
                "FATAL: a second active launch overlapped the exact selected "
                "launch (kernel='%s'); refusing a potentially truncated "
                "capture.\n",
                nvbit_get_func_name(ctx, kernel_func, true));
        fflush(stderr);
        pthread_mutex_unlock(&mutex);
        _exit(EXIT_FAILURE);
      }
      if (launch_decision == nvbit_tracer::ActiveLaunchDecision::kLimitReached) {
        fprintf(stderr,
                "[tracer_tool] MAX_ACCEPTED_ACTIVE_LAUNCHES=1: running "
                "additional active launch attempt %llu uninstrumented "
                "(kernel='%s').\n",
                (unsigned long long)active_launch_gate.active_launch_attempts(),
                nvbit_get_func_name(ctx, kernel_func, true));
        fflush(stderr);
        nvbit_enable_instrumented(ctx, kernel_func, false);
        if (stop_report != nullptr) stop_report[device_id] = true;
        pthread_mutex_unlock(&mutex);
        return;
      }
      if (launch_decision == nvbit_tracer::ActiveLaunchDecision::kInactive) {
        nvbit_enable_instrumented(ctx, kernel_func, false);
        if (stop_report != nullptr) stop_report[device_id] = true;
        pthread_mutex_unlock(&mutex);
        return;
      }

      if (ctx_state->recv_thread_receiving.load(std::memory_order_acquire)) {
        fprintf(stderr,
                "FATAL: overlapping accepted launches in CUDA context %p "
                "cannot share one receiver launch identity; refusing to "
                "merge their channel records.\n",
                static_cast<void *>(ctx));
        fflush(stderr);
        pthread_mutex_unlock(&mutex);
        _exit(EXIT_FAILURE);
      }

      /* Canonical phase leaves intentionally retain the admitted
       * phase_records_<repeat>_<launch>.bin spelling. Because kernel_id is a
       * per-device counter, two producing devices would otherwise both claim
       * the same launch numbers. Admit exactly one GPU device before phase
       * instrumentation is enabled; the host-only writer also rechecks this at
       * publication. */
      if (phase_injection_enabled) {
        std::string phase_device_error;
        if (!phase_record_writer.AdmitDevice(device_id,
                                             &phase_device_error)) {
          simbench_nvbit::FailPhaseRecordAuthority("single-device admission",
                                     phase_device_error);
        }
      }

      const bool per_thread_default_launch =
          cbid == API_CUDA_cuLaunchKernel_ptsz ||
          cbid == API_CUDA_cuLaunchKernelEx_ptsz;
      const int64_t stream_key =
          stream_ordinal_for(kernel_stream, per_thread_default_launch);

      // Some NVBit releases omit nvbit_tool_init for a graph node launched
      // after stream capture. Initialize lazily at the first post-capture
      // kernel callback; CUDA channel allocations are legal once capture is
      // inactive. If this callback is itself part of capture, leave the node
      // uninstrumented and let the later graph-launch callback retry.
      if (!is_exit && !ctx_state->tool_init_complete) {
        const bool capture_active =
            stream_capture_active(kernel_stream, per_thread_default_launch);
        if (!capture_active) {
          skip_callback_flag = true;
          init_context_state(ctx);
          ctx_state->binary_version = resolve_binary_version_for_current_device();
          binary_version = ctx_state->binary_version;
          ctx_state->tool_init_deferred = false;
          ctx_state->tool_init_complete = true;
          skip_callback_flag = false;
        } else {
          ctx_state->tool_init_deferred = true;
          int expected_counter = 0;
          kernel_id[launch_device_id].compare_exchange_strong(
              expected_counter, 1, std::memory_order_relaxed,
              std::memory_order_relaxed);
          pthread_mutex_unlock(&mutex);
          return;
        }
      }
      int expected_counter = 0;
      kernel_id[launch_device_id].compare_exchange_strong(
          expected_counter, 1, std::memory_order_relaxed,
          std::memory_order_relaxed);

      const int launch_kernel_id = checked_kernel_launch_id(
          kernel_id[device_id].load(std::memory_order_acquire));
      pending_accepted_launch.accepted = true;
      pending_accepted_launch.device_id = device_id;
      pending_accepted_launch.trace_launch_id = static_cast<uint32_t>(
          launch_kernel_id);
      pending_accepted_launch.total_insts_at_entry =
          total_dynamic_instr_counter;
      pending_accepted_launch.reported_insts_at_entry =
          reported_dynamic_instr_counter;

      // Claim the single process-global managed launch identity before any
      // instrumentation can publish a new value. Same-device overlap is fatal:
      // an unqualified managed scalar cannot represent two in-flight launches.
      if (phase_injection_enabled) {
        std::string phase_launch_error;
        if (!phase_record_writer.BeginLaunch(
                device_id, pending_accepted_launch.trace_launch_id,
                &phase_launch_error)) {
          simbench_nvbit::FailPhaseRecordAuthority(
              "exclusive phase-launch admission", phase_launch_error);
        }
      }

      // Sampling scope dimensions belong to the accepted profiler-bounded
      // launch.  Warm-up / reference launches outside the active range cannot
      // claim the shape slot.
      if (trace_sampling_mode != 0 && !ts_scope_dims_captured) {
        ts_scope_device = device_id;
        ts_scope_stream = stream_key;
        ts_scope_kernel = checked_sampling_scope_int(
            pending_accepted_launch.trace_launch_id, "kernel");
        ts_scope_grid_x = checked_sampling_scope_int(grid_dim_x, "grid.x");
        ts_scope_grid_y = checked_sampling_scope_int(grid_dim_y, "grid.y");
        ts_scope_grid_z = checked_sampling_scope_int(grid_dim_z, "grid.z");
        ts_scope_block_x = checked_sampling_scope_int(block_dim_x, "block.x");
        ts_scope_block_y = checked_sampling_scope_int(block_dim_y, "block.y");
        ts_scope_block_z = checked_sampling_scope_int(block_dim_z, "block.z");
        ts_scope_dims_captured = true;
      }
      
      // The launch callback's CUfunction is NVBit-owned metadata. Querying it
      // through a directly linked libcuda can return CUDA_ERROR_INVALID_HANDLE
      // even though NVBit can inspect and instrument the function. Use NVBit's
      // function-configuration API, which authenticates these fields from the
      // same function object used by the instrumentation path.
      func_config_t function_config;
      memset(&function_config, 0xff, sizeof(function_config));
      nvbit_get_func_config(ctx, kernel_func, &function_config);
      if (function_config.num_registers ==
          std::numeric_limits<uint32_t>::max()) {
        fprintf(stderr,
                "FATAL: register_attribute_unavailable: "
                "nvbit_get_func_config did not initialize num_registers; "
                "refusing to serialize an unauthenticated value.\n");
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      if (function_config.shmem_static_nbytes ==
          std::numeric_limits<uint32_t>::max()) {
        fprintf(stderr,
                "FATAL: static_shared_memory_attribute_unavailable: "
                "nvbit_get_func_config did not initialize "
                "shmem_static_nbytes; refusing to serialize an "
                "unauthenticated value.\n");
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      const uint32_t nregs = function_config.num_registers;
      const uint32_t shmem_static_nbytes =
          function_config.shmem_static_nbytes;
      if (nregs > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        fprintf(stderr,
                "FATAL: register_attribute_out_of_range: %u exceeds the "
                "int32 protobuf field. Aborting.\n",
                nregs);
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      const uint64_t shmem_combined_nbytes =
          static_cast<uint64_t>(shmem_static_nbytes) +
          static_cast<uint64_t>(shared_mem_bytes);
      if (shmem_combined_nbytes >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        fprintf(stderr,
                "FATAL: combined shared-memory byte count %" PRIu64
                " exceeds the legacy int32 protobuf field. Aborting.\n",
                shmem_combined_nbytes);
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      int driver_occupancy_limit_blocks = -1;
      const uint64_t block_threads = static_cast<uint64_t>(block_dim_x) *
                                     static_cast<uint64_t>(block_dim_y) *
                                     static_cast<uint64_t>(block_dim_z);
      if (block_threads <= static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        int occupancy_blocks = 0;
        const CUresult occupancy_res = cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &occupancy_blocks,
            kernel_func,
            static_cast<int>(block_threads),
            static_cast<size_t>(shared_mem_bytes));
        if (occupancy_res == CUDA_SUCCESS && occupancy_blocks > 0) {
          driver_occupancy_limit_blocks = occupancy_blocks;
        } else if (verbose) {
          fprintf(stderr,
                  "INFO: cuOccupancyMaxActiveBlocksPerMultiprocessor failed: result=%d block_threads=%llu shared=%u\n",
                  static_cast<int>(occupancy_res),
                  static_cast<unsigned long long>(block_threads),
                  shared_mem_bytes);
        }
      }

      {
        // NVBit's CUDA_SAFECALL macro uses cudaGetLastError() (runtime API) to
        // check cuFuncGetAttribute() (driver API), silently dropping the
        // CUresult. On SM120 (Blackwell, compute 12.0) the driver returns
        // CUDA_ERROR_INVALID_VALUE for CU_FUNC_ATTRIBUTE_BINARY_VERSION
        // on the cloud Blackwell target; attr_bv stays zero. (On the local
        // RTX 3090 Ti / sm_86 host this branch builds against, the call
        // succeeds and the fallback below is unused.)
        //
        // The fallback derives binary_version from device compute capability
        // using `major*10 + minor` — e.g. compute 12.0 → binary_version=120,
        // which matches the `sm_120` nvcc arch target. Per
        // NVIDIA_Manuals/cuda_guide/01-03-04-01_binary-compatibility.md, cubins
        // are NEVER binary-compatible across major versions, so on sm_120
        // hardware the running cubin is always either native sm_120 (binary
        // version 120) or JIT-compiled from PTX to sm_120 (also 120). There is
        // no edge case on this hardware where this formula produces a wrong
        // value.
        int attr_bv = 0;
        CUresult bv_res = cuFuncGetAttribute(
            &attr_bv, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kernel_func);
        if (bv_res == CUDA_SUCCESS && attr_bv > 0) {
          binary_version = attr_bv;
          ctx_state->binary_version = attr_bv;
        } else {
          const CUdevice dev = static_cast<CUdevice>(device_id);
          int major = 0, minor = 0;
          const CUresult maj_res = cuDeviceGetAttribute(
              &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
          const CUresult min_res = cuDeviceGetAttribute(
              &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
          if (maj_res == CUDA_SUCCESS && min_res == CUDA_SUCCESS &&
              (major * 10 + minor) > 0) {
            binary_version = major * 10 + minor;
            ctx_state->binary_version = binary_version;
            fprintf(stderr,
                    "INFO: cuFuncGetAttribute(BINARY_VERSION) returned "
                    "attr_bv=%d (CUresult=%d); falling back to device "
                    "compute capability %d.%d -> binary_version=%d\n",
                    attr_bv, (int)bv_res, major, minor, binary_version);
          } else {
            fprintf(stderr,
                    "FATAL: Cannot determine binary_version. "
                    "cuFuncGetAttribute returned attr_bv=%d (CUresult=%d), "
                    "checked device fallback failed (device_id=%d, "
                    "maj_res=%d, min_res=%d, major=%d, minor=%d). "
                    "Aborting.\n",
                    attr_bv, (int)bv_res, device_id, (int)maj_res,
                    (int)min_res, major, minor);
            fflush(stderr);
            _exit(EXIT_FAILURE);
          }
        }
      }

      if (!get_opcode_map(OpcodeMap, binary_version)) {
        fprintf(stderr,
                "FATAL: unsupported binary_version=%d; cannot map traced opcodes\n",
                binary_version);
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      instrument_function_if_needed(ctx, kernel_func, device_id);
      accepted_module_mappings.push_back(resolve_accepted_module_mapping(
          ctx, kernel_func, device_id, stream_key,
          pending_accepted_launch.trace_launch_id));

      nvbit_enable_instrumented(ctx, kernel_func, true);
      stop_report[device_id] = false;

      const std::string kernel_trace_name =
          "kernel-" +
          std::to_string(launch_kernel_id) +
          ".trace";

      // Serialize the entire pre-launch dyn_trace mutation block against
      // any concurrent recv-thread writes from other contexts. We look up
      // `gpu_dev` INSIDE the lock so the reference is valid for the
      // duration of the critical section. The lock is released before we
      // touch the atomic kernel_id / receiving flags below, because those
      // don't require container protection.
      {
        std::lock_guard<std::mutex> rcm_lock(recv_container_mutex);
        dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[device_id];

        if (!stop_report[device_id]) {
          int variant_id = 0;
          auto it_map_already_instrumented = map_function_to_kernel_name_and_variant_id.find(kernel_func);
          if (it_map_already_instrumented ==
              map_function_to_kernel_name_and_variant_id.end()) {
            fprintf(stderr,
                    "FATAL: accepted kernel has no instrumentation identity; "
                    "refusing to publish an unbound trace launch.\n");
            fflush(stderr);
            _exit(EXIT_FAILURE);
          }
          variant_id = std::get<1>(it_map_already_instrumented->second);
          std::string kernel_name(nvbit_get_func_name(ctx, kernel_func, true));

          // std::vector<int> kernel_argument_sizes = nvbit_get_kernel_argument_sizes(kernel_func);

          std::string final_kernel_name = kernel_name + variant_delimiter_str + std::to_string(variant_id);

          auto it_map_unique_function_id = map_function_name_to_unique_function_id_with_variant.find(final_kernel_name);
          if (it_map_unique_function_id ==
              map_function_name_to_unique_function_id_with_variant.end()) {
            fprintf(stderr,
                    "FATAL: accepted kernel has no function-unique identity; "
                    "refusing to publish an unbound trace launch.\n");
            fflush(stderr);
            _exit(EXIT_FAILURE);
          }
          dyn_trace.set_binary_version(binary_version);
          dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
          stream.set_id(stream_key);
          dynamic_trace::kernel *ker = stream.add_kernels();
          ker->set_id(launch_kernel_id);
          ker->set_name(final_kernel_name);
          ker->set_function_unique_id(it_map_unique_function_id->second);
          ker->set_size_shared_memory(
              static_cast<int32_t>(shmem_combined_nbytes));
          ker->set_shared_memory_static_bytes(
              static_cast<uint32_t>(shmem_static_nbytes));
          ker->set_shared_memory_dynamic_bytes(
              static_cast<uint32_t>(shared_mem_bytes));
          if (shared_memory_carveout_preference_pct >= 0) {
            ker->set_shared_memory_carveout_preference_pct(
                shared_memory_carveout_preference_pct);
          }
          if (driver_occupancy_limit_blocks >= 0) {
            ker->set_driver_occupancy_limit_blocks(
                static_cast<uint32_t>(driver_occupancy_limit_blocks));
          }
          ker->set_number_of_registers(nregs);
          ker->set_shared_memory_base_address((uint64_t)nvbit_get_shmem_base_addr(ctx));
          ker->set_local_memory_base_address((uint64_t)nvbit_get_local_mem_base_addr(ctx));
          dynamic_trace::dim3d *grid_dim = ker->mutable_grid_dim();
          grid_dim->set_x(grid_dim_x);
          grid_dim->set_y(grid_dim_y);
          grid_dim->set_z(grid_dim_z);
          dynamic_trace::dim3d *block_dim = ker->mutable_block_dim();
          block_dim->set_x(block_dim_x);
          block_dim->set_y(block_dim_y);
          block_dim->set_z(block_dim_z);
          // Persist cluster_dim so downstream analysis can reconstruct
          // cluster topology. (1,1,1) for non-cluster launches; extracted
          // from CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION above for
          // cuLaunchKernelEx.
          dynamic_trace::dim3d *cluster_dim_pb = ker->mutable_cluster_dim();
          cluster_dim_pb->set_x(cluster_dim_x);
          cluster_dim_pb->set_y(cluster_dim_y);
          cluster_dim_pb->set_z(cluster_dim_z);
        }

        // This will be a relative path to the traces file
        if (!stop_report[device_id]) {
          dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
          stream.add_ordered_cuda_events(kernel_trace_name);
        }
      }

      // Publish this context's launch identity before arming its receiver.
      // The receiver acquires recv_thread_receiving below, so these stores are
      // visible even when another context on the same device is active.
      ctx_state->active_stream_id.store(stream_key, std::memory_order_release);
      ctx_state->active_kernel_id.store(launch_kernel_id,
                                        std::memory_order_release);

      statsFile = open_stats_file("a");
      // Keep launch-shape products in the same 64-bit domain consumed by
      // the stats readers.  Multiplying as `unsigned` silently wrapped large
      // legal grids before they reached `stats.csv`.
      const uint64_t blocks = checked_dimension_product(
          checked_dimension_product(static_cast<uint64_t>(grid_dim_x),
                                    static_cast<uint64_t>(grid_dim_y),
                                    "grid plane"),
          static_cast<uint64_t>(grid_dim_z), "grid volume");
      const uint64_t threads = checked_dimension_product(
          checked_dimension_product(static_cast<uint64_t>(block_dim_x),
                                    static_cast<uint64_t>(block_dim_y),
                                    "block plane"),
          static_cast<uint64_t>(block_dim_z), "block volume");

      const int launch_row_written = fprintf(
          statsFile,
          "%d, %" PRId64 ", %s, %s, %u, %u, %u, %" PRIu64
          ", %u, %u, %u, %" PRIu64 ", ",
          device_id,
          stream_key,
          kernel_trace_name.c_str(), nvbit_get_func_name(ctx, kernel_func, true),
          grid_dim_x, grid_dim_y, grid_dim_z, blocks, block_dim_x, block_dim_y,
          block_dim_z, threads);
      checked_stats_write(launch_row_written, "launch row write");
      checked_stats_close(statsFile);

      const uint32_t this_launch_id =
          pending_accepted_launch.trace_launch_id;
      kernel_id[device_id].fetch_add(1, std::memory_order_acq_rel);
      // Snapshot allocation generations before releasing the launch handshake.
      // The receiver may process channel records after an async free/reuse API
      // callback has run, so resolving against the live registry at receive
      // time would relabel an older kernel's addresses.
      ctx_state->active_allocation_snapshot =
          nvbit_tracer::snapshot_live_allocations(ctx);
      ctx_state->active_allocation_snapshot_valid = true;
      // Release-store publishes the active launch identity and every other
      // main-thread mutation (pc_to_opcode, threadblocks, etc.) to the
      // recv thread, which acquire-loads this same flag at :1402 before
      // touching any of that state.
      ctx_state->next_async_instrumentation_order = 1;
      ctx_state->recv_thread_receiving.store(true, std::memory_order_release);

      /* Closure-plan NVBit phase-injection contract: publish the
       * current launch_index to the device-readable __managed__
       * counter so the injected `record_phase_clock` function reads
       * the correct value during this launch. Also publish the
       * expected flush-sentinel launch_index; the recv thread
       * compares each sentinel's launch_index against this value
       * for diagnostics, but the actual generation-safe handshake
       * is the completed-counter wait below.
       */
      if (phase_injection_enabled &&
          ctx_state->phase_channel_dev != nullptr) {
        simbench_phase_launch_index_for_inject = this_launch_id;
        ctx_state->phase_expected_flush_launch_index.store(
            this_launch_id, std::memory_order_release);
      }

    } else {
      // The matching launch entry was outside the active profiler region.
      // It never armed instrumentation or the receive channel, so its exit
      // must not flush channels, append the stats suffix, serialize blocks, or
      // mutate sampling state.
      if (!pending_accepted_launch.accepted) {
        pthread_mutex_unlock(&mutex);
        return;
      }
      const int device_id = launch_device_id;
      if (device_id != pending_accepted_launch.device_id) {
        fprintf(stderr,
                "FATAL: accepted launch exit changed devices (entry=%d, "
                "exit=%d); refusing a cross-device trace suffix.\n",
                pending_accepted_launch.device_id, device_id);
        fflush(stderr);
        pthread_mutex_unlock(&mutex);
        _exit(EXIT_FAILURE);
      }
      /* make sure current kernel is completed */
      checked_cuda_runtime_status(
          cudaDeviceSynchronize(), "cudaDeviceSynchronize(kernel completion)");
      checked_cuda_runtime_status(cudaGetLastError(),
                                  "cudaGetLastError(kernel completion)");

      /* make sure we prevent re-entry on the nvbit_callback when issuing
       * the flush_channel kernel */
      skip_callback_flag = true;

      /* issue flush of channel so we are sure all the memory accesses
       * have been pushed */
      flush_channel<<<1, 1>>>(ctx_state->channel_dev);
      checked_cuda_runtime_status(cudaDeviceSynchronize(),
                                  "cudaDeviceSynchronize(trace flush)");
      checked_cuda_runtime_status(cudaGetLastError(),
                                  "cudaGetLastError(trace flush)");

      /* Closure-plan NVBit phase-injection contract: per-launch
       * phase-channel flush + drain. Only when env-gated activation
       * is on and the phase channel was allocated. The flush kernel
       * pushes a sentinel record (cta_id_x = -1); the phase recv
       * thread releases its receiving flag on the sentinel so we
       * can spin on the post-flush boundary the same way we do for
       * the instruction-trace channel.
       *
       * The sentinel spin is bounded by `SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS`
       * (default 1e7; the env-var is parsed by `phase_flush_timeout_iters`
       * which forces every runtime value to be strictly positive). A
       * stalled receive thread (e.g., backpressured device-side push()
       * that never completes) would otherwise wedge the host
       * indefinitely. On exhaustion the host snapshots any bytes already
       * received, optionally retains them under the explicitly noncanonical
       * partial-evidence filename, and then exits fatally before canonical
       * publication. A timed-out flush can never authorize a capture.
       */
      bool phase_flush_timed_out = false;
      const uint32_t phase_this_launch_id =
          pending_accepted_launch.trace_launch_id;
      if (phase_injection_enabled &&
          ctx_state->phase_channel_dev != nullptr) {
        flush_phase_channel<<<1, 1>>>(ctx_state->phase_channel_dev,
                                      phase_this_launch_id);
        checked_cuda_runtime_status(
            cudaDeviceSynchronize(), "cudaDeviceSynchronize(phase flush)");
        checked_cuda_runtime_status(cudaGetLastError(),
                                    "cudaGetLastError(phase flush)");
        /* Wait for the recv thread to publish
         * `phase_completed_flush_launch_index == phase_this_launch_id`.
         * The generation-valued counter makes the wait race-free: an
         * unexpected stale sentinel publishes only its own launch_index and
         * cannot spuriously satisfy this launch's wait. Bounded by
         * SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS.
         */
        long long iters_cap = phase_flush_timeout_iters();
        long long iters_remaining = iters_cap;
        while (ctx_state->phase_completed_flush_launch_index.load(
                   std::memory_order_acquire) != phase_this_launch_id) {
          if (iters_remaining == 0) {
            phase_flush_timed_out = true;
            break;
          }
          iters_remaining--;
          sched_yield();
        }
      }

      /* unset the skip flag */
      skip_callback_flag = false;

      /* wait here until the receiving thread has not finished with the
       * current kernel */
      while (ctx_state->recv_thread_receiving.load(std::memory_order_acquire)) {
        sched_yield();
      }

      /* Closure-plan NVBit phase-injection contract: snapshot the
       * per-launch phase records buffer and write it to disk so the
       * Python orchestrator can decode + aggregate it into the
       * per-launch `phase_cycle_ledger_<repeat>_<launch>.json`. The
       * filename uses the launch_index from before the per-launch
       * fetch_add, mirroring the kernel-trace filename convention.
       *
       * Records are filtered by `launch_index`: only records whose
       * `launch_index` equals `phase_this_launch_id` enter this launch's
       * evidence. Earlier-generation records are quarantined as defense in
       * depth; the timeout path now exits fatally, so they cannot be carried
       * into a successfully continuing capture. Future-generation records are
       * retained for their own snapshot.
       */
      uint64_t phase_quarantined_this_launch = 0;
      if (phase_injection_enabled &&
          ctx_state->phase_channel_dev != nullptr) {
        std::vector<char> snapshot_for_launch;
        std::vector<char> retained;
        {
          std::lock_guard<std::mutex> blk(
              ctx_state->phase_records_buffer_mutex);
          const size_t REC_SZ = sizeof(phase_clock_record_t);
          const size_t total = ctx_state->phase_records_buffer.size();
          if (REC_SZ == 0 || total % REC_SZ != 0) {
            fprintf(stderr,
                    "FATAL: phase-record buffer has %zu bytes, not a "
                    "whole-record multiple of %zu\n",
                    total, REC_SZ);
            fflush(stderr);
            _exit(EXIT_FAILURE);
          }
          for (size_t off = 0; off < total; off += REC_SZ) {
            const phase_clock_record_t *r =
                reinterpret_cast<const phase_clock_record_t *>(
                    ctx_state->phase_records_buffer.data() + off);
            if (r->launch_index == phase_this_launch_id) {
              snapshot_for_launch.insert(
                  snapshot_for_launch.end(),
                  ctx_state->phase_records_buffer.data() + off,
                  ctx_state->phase_records_buffer.data() + off + REC_SZ);
            } else if (r->launch_index < phase_this_launch_id) {
              phase_quarantined_this_launch++;
            } else {
              retained.insert(
                  retained.end(),
                  ctx_state->phase_records_buffer.data() + off,
                  ctx_state->phase_records_buffer.data() + off + REC_SZ);
            }
          }
          ctx_state->phase_records_buffer.swap(retained);
        }
        if (phase_quarantined_this_launch > 0) {
          ctx_state->phase_quarantined_record_count.fetch_add(
              phase_quarantined_this_launch, std::memory_order_relaxed);
          fprintf(stderr,
                  "WARNING: quarantined %llu earlier-generation phase "
                  "record(s) while snapshotting kernel='%s' "
                  "launch_index=%u; they cannot contaminate this launch's "
                  "phase evidence.\n",
                  (unsigned long long)phase_quarantined_this_launch,
                  nvbit_get_func_name(ctx, kernel_func, true),
                  (unsigned)phase_this_launch_id);
        }
        if (phase_flush_timed_out) {
          std::ostringstream timeout_detail;
          timeout_detail
              << "SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS="
              << phase_flush_timeout_iters()
              << " exhausted draining the phase channel for kernel='"
              << nvbit_get_func_name(ctx, kernel_func, true)
              << "' launch_index=" << phase_this_launch_id
              << "; canonical phase publication is forbidden";
          if (!snapshot_for_launch.empty()) {
            std::string partial_publication_error;
            if (!phase_record_writer.WritePartialRecord(
                    device_id, phase_repeat_index, phase_this_launch_id,
                    snapshot_for_launch.data(), snapshot_for_launch.size(),
                    &partial_publication_error)) {
              timeout_detail
                  << "; additionally, noncanonical partial-evidence "
                     "publication failed: "
                  << partial_publication_error;
            } else {
              timeout_detail
                  << "; retained " << snapshot_for_launch.size()
                  << " partial byte(s) only as "
                     ".partial-phase-records_<repeat>_<launch>.bin";
            }
          } else {
            timeout_detail << "; no partial record bytes were available";
          }
          simbench_nvbit::FailPhaseRecordAuthority(
              "phase-channel flush timeout", timeout_detail.str());
        }
        if (!snapshot_for_launch.empty()) {
          simbench_nvbit::PublishCanonicalRecordOrFail(
              phase_record_writer, device_id, phase_repeat_index,
              phase_this_launch_id, snapshot_for_launch.data(),
              snapshot_for_launch.size(), "exclusive durable publication");
        }
      }

      if (phase_injection_enabled) {
        std::string phase_launch_error;
        if (!phase_record_writer.EndLaunch(device_id, phase_this_launch_id,
                                           &phase_launch_error)) {
          simbench_nvbit::FailPhaseRecordAuthority(
              "exclusive phase-launch completion", phase_launch_error);
        }
      }

      const uint64_t total_insts_per_kernel = total_dynamic_instr_counter -
          pending_accepted_launch.total_insts_at_entry;
      const uint64_t reported_insts_per_kernel =
          reported_dynamic_instr_counter -
          pending_accepted_launch.reported_insts_at_entry;

      statsFile = open_stats_file("a");
      const int totals_written = fprintf(
          statsFile, "%llu,%llu",
          (unsigned long long)total_insts_per_kernel,
          (unsigned long long)reported_insts_per_kernel);
      checked_stats_write(totals_written, "kernel totals write");
      checked_stats_write(fprintf(statsFile, "\n"), "kernel totals newline");
      checked_stats_close(statsFile);
      
      if (!stop_report[device_id]) {
        create_folder(threadblock_trace_path.c_str());
        create_folder(threadblock_register_values_path.c_str());
        // Serialize the threadblocks iteration + erase block against any
        // concurrent recv-thread writes on other contexts. We have already
        // cleared recv_thread_receiving=false above, so our own recv thread
        // will not be writing to threadblocks for THIS context's current
        // kernel, but a *different* context's recv thread could still be
        // writing to the same global map.
        std::lock_guard<std::mutex> rcm_lock(recv_container_mutex);
        auto it_tb = threadblocks.begin();
        while(it_tb != threadblocks.end()) {
          std::string tb_string_id = it_tb->first;
          ThreadblockStringParseInfo tb_info = parse_tb_string_id(tb_string_id);
          if((tb_info.device_id == device_id) &&
             (tb_info.stream_id == ctx_state->active_stream_id.load(std::memory_order_acquire)) &&
             (tb_info.kernel_id == ctx_state->active_kernel_id.load(std::memory_order_acquire))) {
            // Create hierarchical folder structure
            std::string device_folder = threadblock_trace_path + "/device_" + std::to_string(tb_info.device_id);
            std::string stream_folder = device_folder + "/stream_" + std::to_string(tb_info.stream_id);
            std::string kernel_folder = stream_folder + "/kernel_" + std::to_string(tb_info.kernel_id);

            create_folder(device_folder.c_str());
            create_folder(stream_folder.c_str());
            create_folder(kernel_folder.c_str());

            dynamic_trace::threadblock &tb = it_tb->second;
            // Selective-dump: write only the chosen whole blocks (flattened
            // from the true block_id + the real grid_dim — gridDim is NEVER
            // rewritten, so the reader still sees the full grid and can detect
            // the sampled-out blocks). All other modes write every block.
            bool ts_do_write = true;
            if (trace_sampling_mode == 2) {
              long ts_flat = (long)tb.block_id().z() * grid_dim_y * grid_dim_x
                           + (long)tb.block_id().y() * grid_dim_x
                           + (long)tb.block_id().x();
              ts_do_write = trace_sampling_selected_blocks.find(ts_flat) !=
                            trace_sampling_selected_blocks.end();
            }
            if (trace_sampling_mode == 2) {
              auto it_stat = ts_block_stats.find(tb_string_id);
              if (it_stat == ts_block_stats.end()) {
                ts_serialized_bytes_valid = false;
              } else {
                it_stat->second.serialized_bytes =
                    static_cast<uint64_t>(tb.ByteSizeLong());
              }
            }
            if (ts_do_write) {
              std::string tb_file = kernel_folder + "/" + tb_string_id + ".pb";
              std::ofstream ofs_tb(tb_file, std::ios::out | std::ios::binary);
              if (!ofs_tb) {
                fprintf(stderr,
                        "FATAL: failed to open threadblock trace %s for writing\n",
                        tb_file.c_str());
                fflush(stderr);
                _exit(EXIT_FAILURE);
              }
              if (!tb.SerializeToOstream(&ofs_tb)) {
                fprintf(stderr, "FATAL: failed to serialize threadblock trace %s\n",
                        tb_file.c_str());
                fflush(stderr);
                _exit(EXIT_FAILURE);
              }
              ofs_tb.flush();
              if (!ofs_tb) {
                fprintf(stderr, "FATAL: failed while writing threadblock trace %s\n",
                        tb_file.c_str());
                fflush(stderr);
                _exit(EXIT_FAILURE);
              }
              ofs_tb.close();
              if (!ofs_tb) {
                fprintf(stderr, "FATAL: failed to close threadblock trace %s\n",
                        tb_file.c_str());
                fflush(stderr);
                _exit(EXIT_FAILURE);
              }
            }
            tb.Clear();
            it_tb = threadblocks.erase(it_tb);
          }else {
            it_tb++;
          }
        }
      }
      if (!active_launch_gate.FinishAcceptedLaunch()) {
        fprintf(stderr,
                "FATAL: accepted launch exit has no matching gate entry.\n");
        fflush(stderr);
        pthread_mutex_unlock(&mutex);
        _exit(EXIT_FAILURE);
      }
      pending_accepted_launch = PendingAcceptedLaunch{};
      if (active_from_start && dynamic_kernel_limit_end &&
          kernel_id[device_id].load(std::memory_order_acquire) >
              dynamic_kernel_limit_end) {
        active_region = false;
        active_launch_gate.ProfilerStop();
      }
    }
  } else if (cbid == API_CUDA_cuMemPoolCreate) {
    // Cache pool location before a stream can enter capture.  Querying pool
    // attributes from an active capture is not permitted on current drivers,
    // while the create parameters are the authoritative location contract.
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemPoolCreate_params *p = (cuMemPoolCreate_params *)params;
      if (p->pool != nullptr && *p->pool != nullptr &&
          p->poolProps != nullptr) {
        memory_pool_identity_map[*p->pool] = MemoryPoolIdentity{
            p->poolProps->location.type, p->poolProps->location.id};
      }
    }
  } else if (cbid == API_CUDA_cuMemPoolDestroy) {
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemPoolDestroy_params *p = (cuMemPoolDestroy_params *)params;
      memory_pool_identity_map.erase(p->pool);
    }
  } else if (cbid == API_CUDA_cuDeviceGetDefaultMemPool) {
    // Default device pools are not created through cuMemPoolCreate, so seed
    // their identities from the device argument when the handle is returned.
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuDeviceGetDefaultMemPool_params *p =
          (cuDeviceGetDefaultMemPool_params *)params;
      if (p->pool_out != nullptr && *p->pool_out != nullptr) {
        memory_pool_identity_map[*p->pool_out] = MemoryPoolIdentity{
            CU_MEM_LOCATION_TYPE_DEVICE, static_cast<int>(p->dev)};
      }
    }
  } else if (cbid == API_CUDA_cuDeviceGetMemPool) {
    // cuDeviceGetMemPool returns the current device pool.  Device pools are
    // the only pools accepted by this API, so the device argument identifies
    // the location without another capture-unsafe query.
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuDeviceGetMemPool_params *p = (cuDeviceGetMemPool_params *)params;
      if (p->pool != nullptr && *p->pool != nullptr) {
        memory_pool_identity_map[*p->pool] = MemoryPoolIdentity{
            CU_MEM_LOCATION_TYPE_DEVICE, static_cast<int>(p->dev)};
      }
    }
  } else if (cbid == API_CUDA_cuMemGetMemPool) {
    // The location argument is supplied by the caller.  Cache only concrete
    // device locations; host/managed pools have no fixed backing ordinal.
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemGetMemPool_params *p = (cuMemGetMemPool_params *)params;
      if (p->pool != nullptr && *p->pool != nullptr && p->location != nullptr &&
          p->location->type == CU_MEM_LOCATION_TYPE_DEVICE) {
        memory_pool_identity_map[*p->pool] = MemoryPoolIdentity{
            p->location->type, p->location->id};
      }
    }
  } else if (cbid == API_CUDA_cuMemAlloc_v2) {
    /* Synchronous device-memory allocation. The runtime API
     * cudaMalloc translates to this driver-API call, so this
     * single hook covers the common synchronous path. The
     * dptr is OUT, so we read *p->dptr only on is_exit + on
     * CUDA_SUCCESS (after the driver has populated it and the
     * call actually succeeded). Async / pitched / managed
     * variants are separate driver-API entry points and are
     * deferred to follow-up rounds. The registry is keyed by
     * `ctx` so two contexts cannot alias on overlapping
     * device-pointer ranges. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *)params;
      if (p->dptr != nullptr) {
        uint64_t base = static_cast<uint64_t>(*p->dptr);
        uint64_t size = static_cast<uint64_t>(p->bytesize);
        if (base != 0 && size > 0) {
          const int device_id = checked_stored_context_device_id(
              ctx_state, "nvbit_at_cuda_event/allocation registry");
          nvbit_tracer::register_allocation(
              ctx, base, size, device_id);
        }
      }
    }
  } else if (cbid == API_CUDA_cuMemFree_v2) {
    /* Synchronous device-memory free. p->dptr is a value
     * parameter the driver reads to decide what to free, so
     * it stays valid on both entry and exit. Mirror the
     * cuMemAlloc_v2 hook's symmetry: fire on is_exit only
     * when *pStatus == CUDA_SUCCESS, so a failed free does
     * not falsely mark the allocation dead in the registry
     * (the GPU still owns the buffer when the driver returns
     * a non-success status). The registry is keyed by `ctx`
     * so a free in one context cannot perturb another's
     * intervals. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemFree_v2_params *p = (cuMemFree_v2_params *)params;
      uint64_t base = static_cast<uint64_t>(p->dptr);
      if (base != 0) {
        nvbit_tracer::mark_allocation_freed(ctx, base);
      }
    }
  } else if (cbid == API_CUDA_cuMemAlloc) {
    /* Legacy synchronous device-memory allocation (cbid 29).
     * Param shape: `CUdeviceptr_v1 *dptr; unsigned int
     * bytesize;`. v1 device pointers are 32-bit but the
     * driver widens transparently; we cast through uint64_t
     * uniformly with the v2 path. Same is_exit + success
     * gating as the v2 hook. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemAlloc_params *p = (cuMemAlloc_params *)params;
      if (p->dptr != nullptr) {
        uint64_t base = static_cast<uint64_t>(*p->dptr);
        uint64_t size = static_cast<uint64_t>(p->bytesize);
        if (base != 0 && size > 0) {
          const int device_id = checked_stored_context_device_id(
              ctx_state, "nvbit_at_cuda_event/allocation registry");
          nvbit_tracer::register_allocation(
              ctx, base, size, device_id);
        }
      }
    }
  } else if (cbid == API_CUDA_cuMemFree) {
    /* Legacy synchronous device-memory free (cbid 33). Param
     * shape: `CUdeviceptr_v1 dptr;`. Same success-gated exit
     * semantics as cuMemFree_v2. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemFree_params *p = (cuMemFree_params *)params;
      uint64_t base = static_cast<uint64_t>(p->dptr);
      if (base != 0) {
        nvbit_tracer::mark_allocation_freed(ctx, base);
      }
    }
  } else if (cbid == API_CUDA_cuMemAllocPitch_v2) {
    /* Pitched device-memory allocation (cbid 244). Param
     * shape: `CUdeviceptr *dptr; size_t *pPitch; size_t
     * WidthInBytes; size_t Height; unsigned int
     * ElementSizeBytes;`. The driver writes the actual pitch
     * (rounded up from WidthInBytes for hardware-friendly
     * alignment) through `*pPitch`, so the producer-faithful
     * byte size is `*pPitch * Height`. Both `dptr` and
     * `pPitch` are OUT, so we read on is_exit + success. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemAllocPitch_v2_params *p =
          (cuMemAllocPitch_v2_params *)params;
      if (p->dptr != nullptr && p->pPitch != nullptr) {
        uint64_t base = static_cast<uint64_t>(*p->dptr);
        uint64_t size =
            static_cast<uint64_t>(*p->pPitch) *
            static_cast<uint64_t>(p->Height);
        if (base != 0 && size > 0) {
          const int device_id = checked_stored_context_device_id(
              ctx_state, "nvbit_at_cuda_event/allocation registry");
          nvbit_tracer::register_allocation(
              ctx, base, size, device_id);
        }
      }
    }
  } else if (cbid == API_CUDA_cuMemAllocPitch) {
    /* Legacy pitched device-memory allocation (cbid 31).
     * Same param-struct shape as the v2 variant but with
     * `unsigned int` (32-bit) pitch / width / height fields
     * via CUdeviceptr_v1. Cast through uint64_t. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemAllocPitch_params *p =
          (cuMemAllocPitch_params *)params;
      if (p->dptr != nullptr && p->pPitch != nullptr) {
        uint64_t base = static_cast<uint64_t>(*p->dptr);
        uint64_t size =
            static_cast<uint64_t>(*p->pPitch) *
            static_cast<uint64_t>(p->Height);
        if (base != 0 && size > 0) {
          const int device_id = checked_stored_context_device_id(
              ctx_state, "nvbit_at_cuda_event/allocation registry");
          nvbit_tracer::register_allocation(
              ctx, base, size, device_id);
        }
      }
    }
  } else if (cbid == API_CUDA_cuMemAllocManaged) {
    /* Managed (unified) memory allocation (cbid 371). Param
     * shape: `CUdeviceptr *dptr; size_t bytesize; unsigned
     * int flags;`. Identical to cuMemAlloc_v2 for our
     * purposes; the flags only affect kernel/host migration
     * policy and do not change the device-pointer base or
     * byte size. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemAllocManaged_params *p =
          (cuMemAllocManaged_params *)params;
      if (p->dptr != nullptr) {
        uint64_t base = static_cast<uint64_t>(*p->dptr);
        uint64_t size = static_cast<uint64_t>(p->bytesize);
        if (base != 0 && size > 0) {
          const int device_id = checked_stored_context_device_id(
              ctx_state, "nvbit_at_cuda_event/allocation registry");
          nvbit_tracer::register_allocation(
              ctx, base, size, device_id);
        }
      }
    }
  } else if (cbid == API_CUDA_cuMemAllocAsync ||
             cbid == API_CUDA_cuMemAllocAsync_ptsz) {
    /* Stream-ordered async device-memory allocation (cbid 598
     * default-stream variant; 599 _ptsz). Param shape: `CUdeviceptr
     * *dptr; size_t bytesize; CUstream hStream;`. dptr is OUT, so
     * read on is_exit + success. The `bytesize` is the requested
     * allocation size; the driver may round up internally but the
     * registry tracks the requested range, mirroring the
     * synchronous cuMemAlloc_v2 hook's behavior.
     *
     * Async allocations are context-independent under CUDA's unified
     * addressing contract. Outside graph capture the returned pointer gives
     * the authoritative device; capture uses the checked submitting context
     * because graph-owned pointer attributes are not valid yet. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemAllocAsync_params *p = (cuMemAllocAsync_params *)params;
      if (p->dptr != nullptr) {
        uint64_t base = static_cast<uint64_t>(*p->dptr);
        uint64_t size = static_cast<uint64_t>(p->bytesize);
        if (base != 0 && size > 0) {
          const bool per_thread_default_alloc =
              cbid == API_CUDA_cuMemAllocAsync_ptsz;
          const int device_id = allocation_device_ordinal_or_fail(
              static_cast<CUdeviceptr>(base), nullptr, ctx_state, p->hStream,
              per_thread_default_alloc,
              "nvbit_at_cuda_event/async allocation device");
          if (device_id >= 0) {
            nvbit_tracer::register_context_independent_allocation(
                ctx, base, size, device_id);
          }
        }
      }
    }
  } else if (cbid == API_CUDA_cuMemAllocFromPoolAsync ||
             cbid == API_CUDA_cuMemAllocFromPoolAsync_ptsz) {
    /* Stream-ordered async allocation from a specific memory pool
     * (cbid 611 default-stream variant; 612 _ptsz). Param shape:
     * `CUdeviceptr *dptr; size_t bytesize; CUmemoryPool pool;
     * CUstream hStream;`. The pool selector does not change the
     * registry's view of the allocation: the device pointer is in
     * the same address space as cuMemAllocAsync allocations and is
     * tracked the same way. Outside capture the returned pointer identifies
     * the backing device; capture uses the cached pool location. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemAllocFromPoolAsync_params *p =
          (cuMemAllocFromPoolAsync_params *)params;
      if (p->dptr != nullptr) {
        uint64_t base = static_cast<uint64_t>(*p->dptr);
        uint64_t size = static_cast<uint64_t>(p->bytesize);
        if (base != 0 && size > 0) {
          const bool per_thread_default_alloc =
              cbid == API_CUDA_cuMemAllocFromPoolAsync_ptsz;
          const int device_id = allocation_device_ordinal_or_fail(
              static_cast<CUdeviceptr>(base), p->pool, ctx_state, p->hStream,
              per_thread_default_alloc,
              "nvbit_at_cuda_event/pool allocation device");
          if (device_id >= 0) {
            nvbit_tracer::register_context_independent_allocation(
                ctx, base, size, device_id);
          }
        }
      }
    }
  } else if (cbid == API_CUDA_cuMemFreeAsync ||
             cbid == API_CUDA_cuMemFreeAsync_ptsz) {
    /* Stream-ordered async free (cbid 600 default-stream variant;
     * 601 _ptsz). Param shape: `CUdeviceptr dptr; CUstream
     * hStream;`. CRITICAL semantic difference from cuMemFree_v2:
     * the API returns CUDA_SUCCESS as soon as the free is queued
     * onto the stream, NOT when the free has actually executed.
     * Kernels already enqueued before this free can still
     * legitimately access (ctx, base) AFTER the API returns.
     * Marking the allocation dead at API exit would create false
     * `unmapped_memory_range` exclusions for those still-running
     * kernels' lane-address records. Conservative semantics:
     * queue the (ctx, stream_id, base) into the registry's
     * pending-free list and keep alive=true; the matching
     * cuStreamSynchronize / cuStreamQuery (success-only) /
     * cuCtxSynchronize / cuCtxSynchronize_v2 hook below retires
     * the pending free at the proven completion boundary.
     *
     * Stream identity is resolved with the matching regular or ptsz driver
     * stream-ID entry point, so NULL and per-thread-default streams cannot
     * alias by raw handle. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemFreeAsync_params *p = (cuMemFreeAsync_params *)params;
      uint64_t base = static_cast<uint64_t>(p->dptr);
      const bool per_thread_default_free =
          cbid == API_CUDA_cuMemFreeAsync_ptsz;
      if (base != 0) {
        const int64_t stream_id =
            stream_ordinal_for(p->hStream, per_thread_default_free);
        nvbit_tracer::register_pending_async_free(
            ctx, stream_id, base);
      }
    }
  } else if (cbid == API_CUDA_cuStreamSynchronize ||
             cbid == API_CUDA_cuStreamSynchronize_ptsz) {
    /* Stream synchronization (cbid 126 default-stream variant; 440
     * _ptsz). Param shape: `CUstream hStream;`. CUDA_SUCCESS on
     * exit proves every operation enqueued on hStream before this
     * call has completed; that is the boundary where an earlier
     * cuMemFreeAsync on the same stream is guaranteed to have
     * executed. Retire any pending async frees on (ctx, stream_id)
     * here. cuStreamWaitEvent does NOT prove completion (it only
     * proves the wait operation was queued), so retirement is
     * deliberately NOT wired into that hook.
     *
     * Stream identity is resolved with the matching regular or ptsz driver
     * stream-ID entry point. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuStreamSynchronize_params *p =
          (cuStreamSynchronize_params *)params;
      const bool per_thread_default_sync =
          cbid == API_CUDA_cuStreamSynchronize_ptsz;
      const int64_t stream_id =
          stream_ordinal_for(p->hStream, per_thread_default_sync);
      nvbit_tracer::retire_pending_frees_for_stream(ctx, stream_id);
    }
  } else if (cbid == API_CUDA_cuStreamQuery ||
             cbid == API_CUDA_cuStreamQuery_ptsz) {
    /* Stream query (cbid 125 default-stream variant; 439 _ptsz).
     * Param shape: `CUstream hStream;`. CUDA_SUCCESS on exit
     * proves every stream-queued operation has completed (same
     * proof shape as cuStreamSynchronize); CUDA_ERROR_NOT_READY
     * means work is still in flight and pending frees must NOT
     * be retired. The is_exit + *pStatus == CUDA_SUCCESS gate
     * cleanly rejects the not-ready case. Stream identity is resolved with the
     * matching regular or ptsz driver stream-ID entry point. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuStreamQuery_params *p = (cuStreamQuery_params *)params;
      const bool per_thread_default_query =
          cbid == API_CUDA_cuStreamQuery_ptsz;
      const int64_t stream_id =
          stream_ordinal_for(p->hStream, per_thread_default_query);
      nvbit_tracer::retire_pending_frees_for_stream(ctx, stream_id);
    }
  } else if (cbid == API_CUDA_cuCtxSynchronize) {
    /* Context synchronization (cbid 17). No params: a successful
     * exit proves every stream in the current context has drained.
     * Retire pending async frees across all streams under `ctx`.
     * cuDeviceSynchronize is not separately hooked because, for
     * the trace's per-context registry semantics, draining all
     * streams in one context is the relevant proof; a
     * device-level synchronize that crosses contexts would still
     * leave each context's pending frees retired by the next
     * cuStreamSynchronize / cuCtxSynchronize on that context. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      nvbit_tracer::retire_pending_frees_for_context(ctx);
    }
  } else if (cbid == API_CUDA_cuCtxSynchronize_v2) {
    /* Context synchronization v2 (cbid 800). Param shape:
     * `CUcontext ctx;`. Same proven-completion semantics as
     * cuCtxSynchronize. A non-NULL parameter synchronizes that
     * context; CUDA treats NULL as the current context. Retire
     * pending async frees across the resolved context. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuCtxSynchronize_v2_params *p =
          (cuCtxSynchronize_v2_params *)params;
      // CUDA treats a NULL context as the current context for
      // cuCtxSynchronize_v2.  NVBit supplies that current context in
      // the callback's `ctx` argument, so use it instead of silently
      // leaving successful stream-ordered frees pending forever.
      void *explicit_ctx = p->ctx != nullptr
                               ? static_cast<void *>(p->ctx)
                               : static_cast<void *>(ctx);
      if (explicit_ctx != nullptr) {
        nvbit_tracer::retire_pending_frees_for_context(explicit_ctx);
      }
    }
  } else if (cbid == API_CUDA_cuMemAddressReserve) {
    /* VMM virtual address range reservation (cbid 547). Param
     * shape: `CUdeviceptr *ptr; size_t size; size_t alignment;
     * CUdeviceptr addr; unsigned long long flags;`. The driver
     * writes the reserved virtual address through `*p->ptr` on
     * success; `size` is the reservation length. A reservation
     * is NOT a live allocation — kernels would page-fault if
     * they accessed the range without a subsequent cuMemMap.
     * Track it in the registry's reservation set so a future
     * cuMemAddressFree on a still-mapped range can defensively
     * clean up the leaked mapping. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemAddressReserve_params *p =
          (cuMemAddressReserve_params *)params;
      if (p->ptr != nullptr) {
        uint64_t reserved = static_cast<uint64_t>(*p->ptr);
        uint64_t size = static_cast<uint64_t>(p->size);
        if (reserved != 0 && size > 0) {
          nvbit_tracer::register_vmm_reservation(
              ctx, reserved, size);
        }
      }
    }
  } else if (cbid == API_CUDA_cuMemMap) {
    /* VMM mapping bind (cbid 551). Param shape: `CUdeviceptr ptr;
     * size_t size; size_t offset; CUmemGenericAllocationHandle
     * handle; unsigned long long flags;`. Successful exit binds
     * physical backing into a previously-reserved virtual range,
     * making the [ptr, ptr+size) range kernel-accessible. THIS
     * is when a live allocation is registered. The reservation
     * tracker keeps the parent reservation entry because a
     * single reservation can host multiple partial mappings at
     * different offsets. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemMap_params *p = (cuMemMap_params *)params;
      uint64_t base = static_cast<uint64_t>(p->ptr);
      uint64_t size = static_cast<uint64_t>(p->size);
      if (base != 0 && size > 0) {
        const int device_id = checked_stored_context_device_id(
            ctx_state, "nvbit_at_cuda_event/cuMemMap registry");
        nvbit_tracer::register_allocation(ctx, base, size, device_id);
      }
    }
  } else if (cbid == API_CUDA_cuMemUnmap) {
    /* VMM mapping unbind (cbid 552). Param shape: `CUdeviceptr
     * ptr; size_t size;`. Successful exit removes the binding;
     * the virtual range becomes kernel-inaccessible (kernels
     * would page-fault). Mark the live allocation at `ptr`
     * freed; the reservation tracker keeps the parent reservation
     * entry because a future cuMemMap can bind a different
     * physical handle into the same virtual range. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemUnmap_params *p = (cuMemUnmap_params *)params;
      uint64_t base = static_cast<uint64_t>(p->ptr);
      uint64_t size = static_cast<uint64_t>(p->size);
      if (base != 0 && size > 0) {
        // cuMemUnmap can unmap any sub-range of a previously
        // cuMemMap-bound region. Use the range-aware primitive so
        // interior partial unmaps split the live allocation and
        // base-partial unmaps shrink instead of dropping the
        // entire mapping.
        nvbit_tracer::mark_allocation_freed_range(ctx, base, size);
      }
    }
  } else if (cbid == API_CUDA_cuMemAddressFree) {
    /* VMM virtual address range release (cbid 548). Param shape:
     * `CUdeviceptr ptr; size_t size;`. Successful exit releases
     * the reservation. A clean cuMemUnmap-before-cuMemAddressFree
     * sequence already cleared any live allocation in the range;
     * if a mapping survives past its containing reservation's
     * release (producer/driver oddity), `release_vmm_reservation`
     * defensively marks any intersecting allocation freed before
     * removing the reservation entry. */
    if (is_exit && pStatus && *pStatus == CUDA_SUCCESS) {
      cuMemAddressFree_params *p =
          (cuMemAddressFree_params *)params;
      uint64_t base = static_cast<uint64_t>(p->ptr);
      if (base != 0) {
        nvbit_tracer::release_vmm_reservation(ctx, base);
      }
    }
  } else if (cbid == API_CUDA_cuProfilerStart && is_exit && pStatus != nullptr &&
             *pStatus == CUDA_SUCCESS) {
      if (!active_from_start) {
        active_region = true;
        active_launch_gate.ProfilerStart();
      }
  } else if (cbid == API_CUDA_cuProfilerStop && is_exit && pStatus != nullptr &&
             *pStatus == CUDA_SUCCESS) {
      if (!active_from_start) {
        active_region = false;
        active_launch_gate.ProfilerStop();
      }
  }

  pthread_mutex_unlock(&mutex);
}

bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() &&
         std::isdigit(static_cast<unsigned char>(*it)))
    ++it;
  return !s.empty() && it == s.end();
}

unsigned get_datawidth_from_opcode(const std::vector<std::string> &opcode) {
  for (unsigned i = 0; i < opcode.size(); ++i) {
    if (opcode[i].empty()) {
      continue;
    }
    if (is_number(opcode[i])) {
      errno = 0;
      char *end = nullptr;
      const unsigned long bits = std::strtoul(opcode[i].c_str(), &end, 10);
      if (errno == 0 && end != opcode[i].c_str() && *end == '\0' &&
          bits <= std::numeric_limits<unsigned>::max()) {
        return static_cast<unsigned>(bits) / 8;
      }
    } else if (opcode[i][0] == 'U' && opcode[i].size() > 1 &&
               is_number(opcode[i].substr(1))) {
      // handle the U* case
      errno = 0;
      char *end = nullptr;
      const unsigned long bits = std::strtoul(opcode[i].c_str() + 1, &end, 10);
      if (errno == 0 && end != opcode[i].c_str() + 1 && *end == '\0' &&
          bits <= std::numeric_limits<unsigned>::max()) {
        return static_cast<unsigned>(bits) / 8;
      }
    }
  }

  return 4; // default is 4 bytes
}

bool check_opcode_contain(const std::vector<std::string> &opcode,
                          std::string param) {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param)
      return true;

  return false;
}

static bool checked_signed_difference_i32(uint64_t current, uint64_t previous,
                                          int &difference) {
  if (current >= previous) {
    const uint64_t delta = current - previous;
    if (delta > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
      return false;
    difference = static_cast<int>(delta);
    return true;
  }
  const uint64_t delta = previous - current;
  constexpr uint64_t INT32_MIN_MAGNITUDE =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1u;
  if (delta > INT32_MIN_MAGNITUDE)
    return false;
  if (delta == INT32_MIN_MAGNITUDE) {
    difference = std::numeric_limits<int32_t>::min();
  } else {
    difference = -static_cast<int>(delta);
  }
  return true;
}

static bool checked_signed_difference_i64(uint64_t current, uint64_t previous,
                                          int64_t &difference) {
  if (current >= previous) {
    const uint64_t delta = current - previous;
    if (delta > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return false;
    difference = static_cast<int64_t>(delta);
    return true;
  }
  const uint64_t delta = previous - current;
  constexpr uint64_t INT64_MIN_MAGNITUDE =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1u;
  if (delta > INT64_MIN_MAGNITUDE)
    return false;
  if (delta == INT64_MIN_MAGNITUDE) {
    difference = std::numeric_limits<int64_t>::min();
  } else {
    difference = -static_cast<int64_t>(delta);
  }
  return true;
}

bool base_stride_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                          uint64_t &base_addr, int &stride) {

  // calulcate the difference between addresses
  // write cosnsctive addresses with constant stride in a more
  // compressed way (i.e. start adress and stride)
  bool const_stride = true;
  bool first_bit1_found = false;
  bool last_bit1_found = false;

  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      first_bit1_found = true;
      base_addr = addrs[s];
      if (s < 31 && mask.test(s + 1)) {
        if (!checked_signed_difference_i32(addrs[s + 1], addrs[s], stride))
          return false;
      } else {
        const_stride = false;
        break;
      }
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask.test(s)) {
        int diff_addr = 0;
        if (!checked_signed_difference_i32(addrs[s], addrs[s - 1], diff_addr))
          return false;
        if (stride != diff_addr) {
          const_stride = false;
          break;
        }
      } else
        last_bit1_found = true;
    } else if (last_bit1_found) {
      if (mask.test(s)) {
        const_stride = false;
        break;
      }
    }
  }

  return const_stride;
}

bool base_delta_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                         uint64_t &base_addr, std::vector<long long> &deltas) {

  // save the delta from the previous address
  bool first_bit1_found = false;
  uint64_t last_address = 0;
  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      base_addr = addrs[s];
      first_bit1_found = true;
      last_address = addrs[s];
    } else if (mask.test(s) && first_bit1_found) {
      int64_t delta = 0;
      if (!checked_signed_difference_i64(addrs[s], last_address, delta))
        return false;
      deltas.push_back(static_cast<long long>(delta));
      last_address = addrs[s];
    }
  }
  return true;
}

void *recv_thread_fun(void *args) {
  CUcontext ctx = (CUcontext)args;
  pthread_mutex_lock(&mutex);
  /* get context state from map */
  CTXstate* ctx_state = lookup_context_state_or_fail(
      ctx, "recv_thread_fun/context");
  const int device_id = checked_stored_context_device_id(
      ctx_state, "recv_thread_fun/per-device counters");
  ChannelHost* ch_host = &ctx_state->channel_host;
  pthread_mutex_unlock(&mutex);

  char *recv_buffer = static_cast<char *>(malloc(CHANNEL_SIZE));
  if (recv_buffer == nullptr) {
    fail_device_identity("recv_thread_fun/receive buffer allocation",
                         "malloc(CHANNEL_SIZE) returned null");
  }

  while (ctx_state->recv_thread_done.load(std::memory_order_acquire) == RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
    if (num_recv_bytes > 0) {
      require_record_alignment(num_recv_bytes, sizeof(inst_trace_t), "trace");
      uint32_t num_processed_bytes = 0;
      // The receiving flag is the consumer-path gate.  An inactive launch
      // never arms it; any unexpected channel record from that period is
      // dropped and counted rather than waiting until a later active launch
      // and contaminating that launch's protobuf/sampling output.
      const bool batch_is_accepted =
          ctx_state->recv_thread_receiving.load(std::memory_order_acquire);
      // Acquire-load the active launch pair only after the accepted-flag
      // acquire. These values belong to this context, so another context on
      // the same device cannot relabel this batch.
      //
      // IMPORTANT: these acquire-loads are LOAD-BEARING and must not be
      // "optimized" away as redundant with the receiving-flag handshake.
      // The NVBit channel (nvbit_release/core/utils/channel.hpp) only
      // provides GPU→host synchronization via `volatile int* doorbell` +
      // `__threadfence_system()` on the device side. It does NOT establish
      // any C++ happens-before between the main THREAD's CPU-side writes
      // (kernel_id++, pc_to_opcode[…]=…, dyn_trace.mutable_*) and the recv
      // THREAD's CPU-side reads. The only happens-before chain is via the
      // atomic `recv_thread_receiving` release/acquire pair.
      const int64_t stream_id_snap = batch_is_accepted
          ? ctx_state->active_stream_id.load(std::memory_order_acquire)
          : -1;
      const int kernel_id_snap = batch_is_accepted
          ? ctx_state->active_kernel_id.load(std::memory_order_acquire)
          : -1;
      // Acquire the container lock for the entire batch. This serializes
      // the recv thread's writes to `dyn_trace`, `threadblocks`,
      // `remaining_injects_to_current_instruction`, and `pc_to_opcode`
      // against any concurrent writes from other contexts' recv threads
      // and from main-thread callback handlers.
      std::unique_lock<std::mutex> rcm_lock(recv_container_mutex);
      while (num_processed_bytes < num_recv_bytes) {
        inst_trace_t *ma = (inst_trace_t *)&recv_buffer[num_processed_bytes];

        /* when we receive a CTA_id_x it means all the kernels
        * completed, this is the special token we receive from the
        * flush channel kernel that is issues at the end of the
        * context */
        if (ma->cta_id_x == -1) {
          // Release-store so the main thread's acquire-load at :1209 sees
          // the 'drain complete' signal with proper happens-before ordering.
          ctx_state->recv_thread_receiving.store(false, std::memory_order_release);
          break;
        }
        if (!batch_is_accepted) {
          inactive_channel_records_dropped.fetch_add(1,
                                                      std::memory_order_acq_rel);
          num_processed_bytes += sizeof(inst_trace_t);
          continue;
        }
        accepted_channel_records.fetch_add(1, std::memory_order_relaxed);
        // Key: d_{device}_s_{stream}_k_{kernel}_{cta_id_x},{cta_id_y},{cta_id_z}
        std::string tb_string_id = "d_" + std::to_string(device_id) + "_s_" + std::to_string(stream_id_snap) + "_k_" + std::to_string(kernel_id_snap) + "_" + std::to_string(ma->cta_id_x) + "," +
                                   std::to_string(ma->cta_id_y) + "," +
                                   std::to_string(ma->cta_id_z);
        TraceSamplingIdentityResult ts_identity;
        bool ts_identity_available = false;
        // Trace-sampling counting is scoped to the profiler-selected launch
        // fixed by the launch-entry callback: accumulate per-block stats for
        // both capture modes (so
        // selective-dump keeps exact all-block auxiliary totals for that launch);
        // counting-only then skips the payload build, and any later launch is
        // skipped entirely so it never pollutes the gate totals or the payload.
        if (trace_sampling_mode != 0) {
          int ts_launch_kernel = kernel_id_snap;
          const bool ts_scope_valid = ts_scope_dims_captured &&
                                      ts_scope_device >= 0 &&
                                      ts_scope_stream >= 0 &&
                                      ts_scope_kernel >= 0;
          if (!ts_scope_valid) ts_sector_accesses_valid = false;
          bool ts_in_scope = (ts_scope_valid &&
                              device_id == ts_scope_device &&
                              stream_id_snap == ts_scope_stream &&
                              ts_launch_kernel == ts_scope_kernel);
          if (ts_in_scope) {
            TsBlockStat &ts = ts_block_stats[tb_string_id];
            ts.cta_x = ma->cta_id_x;
            ts.cta_y = ma->cta_id_y;
            ts.cta_z = ma->cta_id_z;
            ts.sm_id = ma->sm_id;
            if (ma->warpid_sm > ts.max_warpid_sm) ts.max_warpid_sm = ma->warpid_sm;
            uint32_t &ts_remaining =
                ts.remaining_injects_by_warp[ma->warpid_tb];
            if (ts_remaining == 0) {
              if (ma->num_of_injects == 0) {
                ts_sector_accesses_valid = false;
              } else {
                ts.records++;
                ts_remaining = ma->num_of_injects;
                ts.atomic_seen_by_warp[ma->warpid_tb] = false;
              }
            }
            if (ma->mem_type == MEM_TYPE::STANDARD_MEM) {
              int ts_opcode_id = pc_to_opcode[ma->unique_function_id][ma->vpc];
              auto ts_cls = memory_op_class_classifier::classify(
                  id_to_opcode_map[ts_opcode_id]);
              dynamic_trace::MemorySpace ts_address_space =
                  dynamic_trace::MEMORY_SPACE_UNSPECIFIED;
              auto it_space = opcode_id_to_address_space.find(ts_opcode_id);
              if (it_space != opcode_id_to_address_space.end()) {
                ts_address_space = static_cast<dynamic_trace::MemorySpace>(
                    it_space->second);
              }
              ts_address_space = operand_address_space(
                  ts_cls, ts_address_space, ma->reg_id);
              uint64_t record_sector_accesses = 0;
              if (!trace_sampling_sector_accesses(
                      *ma, ts_cls, ts_address_space,
                      &record_sector_accesses) ||
                  ts.sector_accesses >
                      std::numeric_limits<uint64_t>::max() -
                          record_sector_accesses) {
                ts_sector_accesses_valid = false;
              } else {
                ts.sector_accesses += record_sector_accesses;
              }
              ts_identity = trace_sampling_identity_result(
                  ctx, *ma, ts_cls, ts_address_space, ctx_state);
              ts_identity_available = true;
              if (ts_identity.required &&
                  effective_address_lane_mask(*ma) != 0 &&
                  (ts_identity.any_zero ||
                   ts_identity.multiple_non_zero)) {
                ts_sector_accesses_valid = false;
              }
              if (ts_cls == memory_op_class_classifier::kAtomic) {
                bool &atomic_seen =
                    ts.atomic_seen_by_warp[ma->warpid_tb];
                if (!atomic_seen) {
                  ts.atomic_records++;
                  atomic_seen = true;
                }
                /* Preserve the existing hot-line sketch semantics: every
                 * atomic operand record contributes each active lane's starting
                 * sector, without width expansion or deduplication. */
                uint32_t ts_mask = effective_address_lane_mask(*ma);
                for (int lane = 0; lane < 32; ++lane) {
                  if (!((ts_mask >> lane) & 1u)) continue;
                  uint64_t sector = ma->addrs_or_reg_val_0[lane] >> 5;
                  auto atomic_sector = ts_atomic_sectors.find(sector);
                  if (atomic_sector != ts_atomic_sectors.end()) {
                    if (atomic_sector->second ==
                        std::numeric_limits<uint64_t>::max()) {
                      ts_atomic_sectors_overflow = true;
                    } else {
                      atomic_sector->second++;
                    }
                  } else if (ts_atomic_sectors.size() < TS_ATOMIC_SECTORS_CAP) {
                    ts_atomic_sectors.emplace(sector, 1);
                  } else {
                    ts_atomic_sectors_overflow = true;
                  }
                }
              }
            }
            if (ts_remaining == 0) {
              ts_sector_accesses_valid = false;
            } else {
              ts_remaining--;
            }
          }
          if (trace_sampling_mode == 1 || !ts_in_scope) {
            num_processed_bytes += sizeof(inst_trace_t);
            continue;
          }
        }
        dynamic_trace::threadblock &tb = get_threadblock(tb_string_id);
        dynamic_trace::dim3d *cta_id = tb.mutable_block_id();
        cta_id->set_x(ma->cta_id_x);
        cta_id->set_y(ma->cta_id_y);
        cta_id->set_z(ma->cta_id_z);
        int warp_id_tb = ma->warpid_tb;
        dynamic_trace::warp &wp = (*tb.mutable_warps())[warp_id_tb];
        wp.set_id(warp_id_tb);
        // if (print_core_id) {
        //   tb.set_sm_id(ma->sm_id);
        //   wp.set_warp_id_in_sm(ma->warpid_sm);
        // }
        int opcode_id = pc_to_opcode[ma->unique_function_id][ma->vpc];
        unsigned int &remaining_injects = get_remaining_injects_to_current_instruction(tb_string_id, warp_id_tb);
        dynamic_trace::instruction *inst;
        if(remaining_injects == 0) {
          inst = wp.add_instructions();
          inst->set_pc(ma->vpc);
          inst->set_active_mask(ma->active_mask);
          inst->set_predicate_mask(ma->predicate_mask);
          inst->set_function_unique_id(ma->unique_function_id);
          // SIMT scheduling identity + virtual PC. Set once when the
          // protobuf inst is first created; subsequent operand records
          // for the same static instruction take the reuse branch
          // below and inherit these values (the warp did not migrate
          // mid-instruction, and vpc is by definition the same static
          // PC).
          //
          //   - warpid_tb / warpid_sm / sm_id: lane-grouping identity
          //     the downstream coalescer / HRD profiler needs.
          //   - vpc: full 64-bit virtual PC. Currently sign-extended
          //     from the uint32_t on the device->host channel;
          //     widening the channel itself is queued for a follow-up
          //     round (touches inst_trace_t + inject_funcs + the
          //     nvbit_add_call_arg_const_val32 call sites).
          //
          // Memory-only fields (data_width, memory_op_class) are set
          // separately, after the branch, so a multi-operand SASS
          // instruction whose first operand is non-memory still gets
          // those fields populated when a later MEMORY_REF operand
          // arrives.
          inst->set_warpid_tb(ma->warpid_tb);
          inst->set_warpid_sm(ma->warpid_sm);
          inst->set_sm_id(ma->sm_id);
          inst->set_vpc(static_cast<uint64_t>(ma->vpc));
          if (async_order_enabled &&
              async_order_selected_pcs.find(
                  {ma->unique_function_id, ma->vpc}) !=
                  async_order_selected_pcs.end()) {
            const uint64_t order =
                ctx_state->next_async_instrumentation_order;
            if (order == 0 || order == UINT64_MAX) {
              fprintf(stderr,
                      "FATAL: selected asynchronous-copy instrumentation "
                      "order exhausted for function_id=%u vpc=0x%x.\n",
                      ma->unique_function_id, ma->vpc);
              fflush(stderr);
              _exit(EXIT_FAILURE);
            }
            inst->set_instrumentation_order(order);
            ctx_state->next_async_instrumentation_order = order + 1;
          }
          remaining_injects = ma->num_of_injects;
        }else {
          inst = wp.mutable_instructions(wp.instructions_size()-1);
        }
        remaining_injects--;
        std::bitset<32> mask(effective_address_lane_mask(*ma));
        // Memory-class and per-lane data width: applied whenever the
        // arriving operand record is memory (mem_type != NONE),
        // regardless of whether this is the first operand for the
        // static instruction or a later one taking the reuse branch.
        // Multi-operand SASS instructions (e.g. STG with a register
        // data source AND a memory address operand) may emit a
        // non-memory operand record FIRST, which then takes the
        // creation branch and would leave the inst with no
        // data_width / memory_op_class. A later MEMORY_REF operand
        // for the same static instruction lands here, on the reuse
        // branch, and must populate those fields. The setters are
        // idempotent under repeat memory operands (same opcode →
        // same classifier output, same width on the channel) so
        // first-operand-was-memory paths are also safe.
        // The downstream per-instruction trace contract enumerates
        // exactly nine memory-operation classes (global, shared,
        // local, texture, surface, atomic, tma, async_copy,
        // cluster_dsmem). The instrumentation pipeline also emits
        // operand records for two adjacent kinds — `CONSTANT_MEM`
        // (LDC-family constant-bank reads) and `CALL_OR_RET`
        // (call/return target operands) — that the device-side stub
        // labels with `MEMORY_REF` so the host channel carries them,
        // but which are NOT memory operations under the contract.
        //
        // Producer policy: only `STANDARD_MEM` records contribute to
        // the per-instruction memory surface. Their operand records
        // populate the instruction-level `data_width` and
        // `memory_op_class` fields and add an `address` record.
        // `CONSTANT_MEM` and `CALL_OR_RET` records create the
        // protobuf instruction (so the warp's `instructions` list
        // still counts them) but contribute no `data_width`,
        // `memory_op_class`, or `address` payload — keeping the
        // wire surface aligned with the offline contract validator.
        if (ma->mem_type == MEM_TYPE::STANDARD_MEM) {
          const std::string &opcode_full =
              id_to_opcode_map[opcode_id];
          auto cls =
              memory_op_class_classifier::classify(opcode_full);
          /* Orthogonal direction classifier. The kind dimension
           * tags load / store / atomic / async-copy independent of
           * the memory space. Both classifiers run on the same
           * opcode head; both share the kUnspecified fail-closed
           * sentinel, which falls into the typed-exclusion path
           * below alongside the class-side fail-closed cases. */
          auto kind =
              memory_op_access_kind_classifier::classify(opcode_full);

          /* Per-operand memory space. The instruction-level map supplies the
           * ordinary default; LDGSTS is refined from its channel-carried MREF
           * ordinal so GLOBAL source and SHARED destination cannot be conflated.
           * The resulting wire discriminator decides whether allocation identity
           * applies to this record. */
          dynamic_trace::MemorySpace addr_space =
              dynamic_trace::MEMORY_SPACE_UNSPECIFIED;
          {
            auto it_space = opcode_id_to_address_space.find(opcode_id);
            if (it_space != opcode_id_to_address_space.end())
              addr_space =
                  static_cast<dynamic_trace::MemorySpace>(it_space->second);
          }
          addr_space = operand_address_space(cls, addr_space, ma->reg_id);

          /* task11b producer-side typed exclusions: route both
           * the 6 explicit excluded classes (TEXTURE / SURFACE /
           * ATOMIC / TMA / ASYNC_COPY / CLUSTER_DSMEM) AND the
           * kUnspecified sentinel through `unsupported_memory_path.json`
           * and fail closed — skip the entire memory-payload
           * emission so the wire-side instruction record has no
           * addresses, no memory_op_class, and no data_width.
           * The validator's `looks_like_memory` check returns
           * false on such records and admits them as non-memory
           * instructions; the sidecar carries the typed
           * exclusion explanation downstream feature extraction
           * needs.
           *
           * The kUnspecified path is the R24 addition. The
           * classifier returns kUnspecified for any opcode whose
           * head doesn't match a known prefix; the schema enum
           * carries `UNSPECIFIED` to surface this case through
           * the same sidecar without bumping the schema version
           * (an additive enum widening). Without this hook,
           * kUnspecified records would fall through and be
           * rejected by the validator as
           * `memory_op_class_missing_on_memory_instruction`
           * since they would have addresses but no
           * memory_op_class. The supported classes (GLOBAL /
           * SHARED / LOCAL) continue through the normal payload
           * path below.
           *
           * Access-kind UNSPECIFIED also routes through the same
           * exclusion sidecar: an opcode the class-side
           * classifier accepts but the kind-side classifier does
           * not is still a producer-side fail-closed event.
           * Keeping the two classifiers in lockstep here means a
           * future opcode known to one but not the other surfaces
           * as a typed exclusion rather than a silent half-tagged
           * wire record. */
          if (is_excluded_op_class(cls) ||
              cls == memory_op_class_classifier::kUnspecified ||
              kind == memory_op_access_kind_classifier::kUnspecified) {
            /* AC-T5.2 path_resolved arm:
             *   true  — both classifiers recognized the opcode
             *           and assigned it a memory class in the
             *           excluded-but-recognized set (one of the
             *           7 routed-to-exclusion classes per
             *           is_excluded_op_class). Downstream
             *           consumers promote these entries to
             *           UNSUPPORTED_BLACKWELL_MEMORY_PATH in the
             *           headline-validity gate.
             *   false — at least one classifier returned
             *           kUnspecified (unknown opcode head OR
             *           known opcode head with unknown access
             *           direction). Downstream consumers keep
             *           these on TRACE_OPCODE_COVERAGE_INCOMPLETE
             *           because the kernel mix contains an
             *           opcode the tracer cannot classify. */
            const bool path_resolved =
                is_excluded_op_class(cls) &&
                kind != memory_op_access_kind_classifier::kUnspecified;
            nvbit_tracer::aggregate(
                static_cast<uint32_t>(inst->function_unique_id()),
                std::string(op_class_to_schema_string(cls)),
                static_cast<uint64_t>(ma->vpc),
                std::string("unsupported_memory_path"),
                path_resolved);
            num_processed_bytes += sizeof(inst_trace_t);
            continue;
          }

          /* Per-lane allocation resolution gate (GLOBAL records
           * only). The wire `allocation_id` represents the
           * WHOLE warp instruction, not the lowest-numbered
           * lane. Resolve every active+predicated lane against
           * the per-context allocator registry and decide one
           * of three outcomes:
           *   - all lanes resolve to the same non-zero id: stamp
           *     it on the wire below and emit normally.
           *   - some lane resolves to 0 (address outside any
           *     live allocation): aggregate
           *     `unmapped_memory_range` and skip the rest of
           *     memory-payload emission for this record.
           *   - lanes resolve to two or more distinct non-zero
           *     ids (single warp instruction spanning multiple
           *     allocations): aggregate
           *     `cross_allocation_lane_span` and skip the rest
           *     of memory-payload emission.
           * Skipping payload emission keeps the wire-side record
           * free of partial / first-lane-biased identity in the
           * exclusion paths. SHARED / LOCAL records have no
           * allocation_id requirement (the registry tracks
           * GLOBAL device address spaces only) and continue
           * through the normal payload path with
           * resolved_allocation_id == 0. */
          uint32_t resolved_allocation_id = 0;
          // Allocation-identity gate. Plain global loads/stores resolve as
          // before (kGlobal, unconditional — preserves the frozen M1 anchor).
          // A GLOBAL/GENERIC-space atomic resolves the same way; a SHARED
          // atomic (ATOMS) is exempt and keeps resolved_allocation_id == 0.
          // ASYNC_COPY follows its per-operand space: the LDGSTS global source
          // resolves exactly like a global access, while its shared destination
          // is exempt and contributes no allocation identity. An unmapped source
          // remains a typed `unmapped_memory_range`, never a silent skip.
          TraceSamplingIdentityResult identity =
              ts_identity_available
                  ? ts_identity
                  : trace_sampling_identity_result(
                        ctx, *ma, cls, addr_space, ctx_state);
          if (identity.required && mask.any()) {
            if (identity.any_zero) {
              nvbit_tracer::aggregate(
                  static_cast<uint32_t>(
                      inst->function_unique_id()),
                  std::string("GLOBAL"),
                  static_cast<uint64_t>(ma->vpc),
                  std::string("unmapped_memory_range"));
              num_processed_bytes += sizeof(inst_trace_t);
              continue;
            }
            if (identity.multiple_non_zero) {
              nvbit_tracer::aggregate(
                  static_cast<uint32_t>(
                      inst->function_unique_id()),
                  std::string("GLOBAL"),
                  static_cast<uint64_t>(ma->vpc),
                  std::string("cross_allocation_lane_span"));
              num_processed_bytes += sizeof(inst_trace_t);
              continue;
            }
            resolved_allocation_id = identity.allocation_id;
          }

          inst->set_data_width(ma->width);
          inst->set_memory_op_class(
              static_cast<dynamic_trace::MemoryOpClass>(cls));
          inst->set_memory_op_access_kind(
              static_cast<dynamic_trace::MemoryOpAccessKind>(kind));
          dynamic_trace::address *addr = inst->add_addresses();
          // Per-address memory space (the address-carrying contract
          // discriminator). Stamped whenever known so the consumer identity
          // gate keys on it; absent (UNSPECIFIED) on records whose space the
          // instrument-time map did not capture, where consumers fall back to
          // the instruction-level MemoryOpClass.
          if (addr_space != dynamic_trace::MEMORY_SPACE_UNSPECIFIED) {
            addr->set_address_space(addr_space);
          }
          if (ma->address_lane_mask_present != 0) {
            addr->set_lane_mask(ma->address_lane_mask);
          }
          std::istringstream iss(id_to_opcode_map[opcode_id]);
          std::vector<std::string> tokens;
          std::string token;
          addr->set_udesc_value(ma->ureg_desc_value);
          while (std::getline(iss, token, '.'))
          {
            if (!token.empty())
              tokens.push_back(token);
          }
          addr->set_data_width(get_datawidth_from_opcode(tokens));

          bool base_stride_success = false;
          bool base_delta_success = false;
          uint64_t base_addr = 0;
          int stride = 0;
          std::vector<long long> deltas;

          if (enable_compress) {
            // try base+stride format
            base_stride_success =
                base_stride_compress(ma->addrs_or_reg_val_0, mask, base_addr, stride);
            if (!base_stride_success) {
              // if base+stride fails, try base+delta format
              base_delta_success = base_delta_compress(
                  ma->addrs_or_reg_val_0, mask, base_addr, deltas);
            }
          }

          if (base_stride_success && enable_compress) {
            // base + stride format
            addr->set_compression_format(address_format::base_stride);
            addr->set_base_address(base_addr);
            addr->set_stride(stride);
          } else if (!base_stride_success && base_delta_success && enable_compress) {
            // base + delta format
            addr->set_compression_format(address_format::base_delta);
            addr->set_base_address(base_addr);
            for (unsigned int s = 0; s < deltas.size(); s++) {
              addr->add_addrs(deltas[s]);
            }
          } else {
            // list_all compression format: emit exactly 32 entries
            // (one per lane). Active+predicated lanes carry the
            // real address; inactive lanes carry
            // LANE_INACTIVE_SENTINEL (-1). The 32-entry shape lets
            // consumers index lanes positionally without a separate
            // active-mask join, and the offline validator enforces
            // it via the `lane_vector_length_invalid` typed-reason
            // rejection.
            addr->set_compression_format(address_format::list_all);
            for (int s = 0; s < 32; s++) {
              if (mask.test(s)) {
                addr->add_addrs(
                    static_cast<int64_t>(ma->addrs_or_reg_val_0[s]));
              } else {
                addr->add_addrs(LANE_INACTIVE_SENTINEL);
              }
            }
          }

          /* Allocation/range identity attachment. The per-lane
           * resolution above (GLOBAL gate) decided whether every
           * active+predicated lane resolves to the same non-zero
           * allocation_id. If so, stamp it on the wire here.
           * Otherwise (SHARED / LOCAL, or the GLOBAL all-zero /
           * multi-non-zero cases that already issued typed
           * exclusions and `continue`'d) `resolved_allocation_id`
           * is 0 and the field stays absent — proto3
           * explicit-presence (HasField=false) is the wire signal
           * the validator's allocation/range identity check uses
           * to skip the per-record range verification. */
          if (resolved_allocation_id != 0) {
            addr->set_allocation_id(resolved_allocation_id);
          }
        }
        num_processed_bytes += sizeof(inst_trace_t);
      }
    }
  }
  free(recv_buffer);
  // Release-store so the main thread's acquire-load spin at :1643 sees
  // FINISHED with proper happens-before ordering; everything the recv
  // thread wrote to shared state above is now visible to main.
  ctx_state->recv_thread_done.store(RecvThreadState::FINISHED, std::memory_order_release);
  return NULL;
}

int check_system_call(int system_res, const char* syscall) {
  if(system_res != 0) {
    std::cout << "Error. System call failed. System call: " << syscall << std::endl;
    fflush(stdout);
    std::string str_error(syscall);
    if(str_error.find("nvdisasm -lrm count") == std::string::npos) {
      abort();
    }
  }
  return system_res;
}

// Outcome of a bounded subprocess invocation. `kExited` carries the child's
// exit status (0 on success); `kTimedOut` means the parent killed the child
// process group after the configured deadline elapsed; `kForkFailed` /
// `kExecFailed` cover the rare host-side error paths.
enum class SubprocessOutcome { kExited, kTimedOut, kForkFailed, kExecFailed };

struct SubprocessResult {
  SubprocessOutcome outcome;
  int exit_status; // valid only when outcome == kExited
};

// Run `cmd` via `/bin/sh -c` with a normal-return monotonic-deadline timeout.
// Forks a child placed in its own process group via setpgid(0, 0) so the
// parent can clean up the entire shell+tool subtree atomically on timeout.
// Polls waitpid(child, ..., WNOHANG) every 50 ms; on deadline expiry the
// parent sends SIGTERM to the child's pgid, sleeps 250 ms, then sends
// SIGKILL as a fallback before reaping. No SIGALRM, no longjmp, no global
// signal handler — the whole sequence stays inside the calling thread's
// normal control flow.
//
// timeout_s == 0 disables the deadline (the parent waits indefinitely).
//
// Exec-failure detection uses a close-on-exec pipe: pipe2(O_CLOEXEC)
// gives both ends the FD_CLOEXEC flag. The child closes the read end
// explicitly and writes errno to the write end if execl() returns
// (i.e., the shell could not be started). On a successful execl() the
// kernel closes the write end via CLOEXEC without anything having been
// written, so the parent's post-waitpid read on the read end returns 0
// (EOF). A non-empty parent read therefore unambiguously signals
// SubprocessOutcome::kExecFailed.
static SubprocessResult run_subprocess_with_timeout(const std::string &cmd,
                                                   int timeout_s) {
  int err_pipe[2];
  if (pipe2(err_pipe, O_CLOEXEC) < 0) {
    return {SubprocessOutcome::kForkFailed, 0};
  }
  pid_t child = fork();
  if (child < 0) {
    close(err_pipe[0]);
    close(err_pipe[1]);
    return {SubprocessOutcome::kForkFailed, 0};
  }
  if (child == 0) {
    // Child: close read end, become a process-group leader, then exec.
    close(err_pipe[0]);
    setpgid(0, 0);
    execl("/bin/sh", "sh", "-c", cmd.c_str(), (char *)nullptr);
    // execl only returns on failure: stash errno for the parent and
    // exit with a sentinel status. The write end is closed implicitly
    // by _exit(); a partial / failed write is acceptable because the
    // parent read length distinguishes "wrote sizeof(int)" from "EOF".
    int e = errno;
    ssize_t written = write(err_pipe[1], &e, sizeof(e));
    (void)written;
    _exit(127);
  }
  // Parent: close the write end so the post-waitpid read sees EOF on
  // the success path; idempotently confirm the child's pgid.
  close(err_pipe[1]);
  setpgid(child, child);

  using clock = std::chrono::steady_clock;
  auto deadline = (timeout_s > 0)
                      ? clock::now() + std::chrono::seconds(timeout_s)
                      : clock::time_point::max();

  SubprocessResult result{SubprocessOutcome::kExited, 0};
  bool reaped = false;
  for (;;) {
    int status = 0;
    pid_t r = waitpid(child, &status, WNOHANG);
    if (r == child) {
      if (WIFEXITED(status)) {
        result = {SubprocessOutcome::kExited, WEXITSTATUS(status)};
      } else if (WIFSIGNALED(status)) {
        result = {SubprocessOutcome::kExited, 128 + WTERMSIG(status)};
      } else {
        result = {SubprocessOutcome::kExited, 0};
      }
      reaped = true;
      break;
    }
    if (r < 0 && errno != EINTR) {
      // Should not happen for a child we own; surface as exec-fail-like.
      close(err_pipe[0]);
      return {SubprocessOutcome::kExecFailed, errno};
    }
    if (timeout_s > 0 && clock::now() >= deadline) {
      // Deadline expired: tear down the child's process group.
      kill(-child, SIGTERM);
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      // If SIGTERM did not finish the job, escalate.
      pid_t reap = waitpid(child, &status, WNOHANG);
      if (reap == 0) {
        kill(-child, SIGKILL);
      }
      // Block-wait so the child slot is reaped before we return.
      waitpid(child, &status, 0);
      close(err_pipe[0]);
      return {SubprocessOutcome::kTimedOut, 0};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Child reaped: drain the error pipe. read returns 0 (EOF) when
  // execl() succeeded (write end closed by CLOEXEC); read returns
  // sizeof(int) when execl() failed and the child wrote errno before
  // _exit(127).
  if (reaped) {
    int child_errno = 0;
    ssize_t n = read(err_pipe[0], &child_errno, sizeof(child_errno));
    close(err_pipe[0]);
    if (n == (ssize_t)sizeof(child_errno)) {
      return {SubprocessOutcome::kExecFailed, child_errno};
    }
    return result;
  }
  close(err_pipe[0]);
  return result;
}

// Snapshot of a generated parser-input file (the .sass / .rfu side
// outputs of nvdisasm / cuobjdump). `exists` is false when the file
// is missing, not a regular file, or otherwise unreadable. The split
// of existence vs. byte count lets the per-cubin skip-parsers gate
// distinguish "size threshold tripped" from "input never produced",
// each of which gets its own marker in the typed diagnostic.
struct ParserInputStat {
  bool exists;
  uintmax_t bytes;
};

static ParserInputStat stat_parser_input(const std::filesystem::path &path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    return {false, 0};
  }
  uintmax_t sz = std::filesystem::file_size(path, ec);
  if (ec) {
    return {false, 0};
  }
  return {true, sz};
}

static std::string shell_quote(const std::string &value) {
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('\'');
  for (char c : value) {
    if (c == '\'') {
      quoted.append("'\\''");
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

void enhanced_tracer() {
  create_folder(extrainfo_path.c_str());
  create_folder(cubin_path.c_str());
  create_folder(sass_path.c_str());
  create_folder(register_usage_path.c_str());
  std::string program_path = get_program_path();
  if (!program_path.empty() && program_path[0] != '/') {
    program_path = cwd + "/" + program_path;
  }
  std::size_t found = program_path.find_last_of("/");
  std::string program_name = program_path.substr(found + 1);
  if (binary_version <= 0) {
    // Hard-fail guard: enhanced_tracer() must never invoke
    // `cuobjdump -arch=sm_0`. If we reach this point with an
    // uninitialised binary_version, both the ctx-init fallback and the
    // per-launch fallback have failed, so the only honest thing to do is
    // abort loudly with the diagnostic the operator needs.
    fprintf(stderr,
            "FATAL: enhanced_tracer reached with binary_version=%d. "
            "Neither the ctx-init resolve nor the per-kernel-launch hook "
            "set a valid value. Refusing to run `cuobjdump -arch=sm_0`.\n",
            binary_version);
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
  std::string command_get_cubin =
      "cd " + shell_quote(cubin_path) + " && cuobjdump " +
      shell_quote(program_path) + " -xelf all -arch=sm_" +
      std::to_string(binary_version);
  std::cout << "Generating extra information for the enhanced traces of benchmark: " << program_name << std::endl;
  int cubin_extract_result = system(command_get_cubin.c_str());
  bool cubin_extraction_succeeded = (cubin_extract_result == 0);
  if (!cubin_extraction_succeeded) {
    std::cout << "Warning: cuobjdump failed (exit " << cubin_extract_result
              << "). JIT-compiled kernels will use NVBit-captured SASS metadata "
              << "(control bits unavailable)." << std::endl;
  }
  m_enhanced_traced_execution = new traced_execution(program_name);
  std::string absolute_cubin_path = resolve_output_path(cwd, cubin_path);
  if (cubin_extraction_succeeded) {
  // Per-cubin loop. Each iteration runs the two external producers
  // (nvdisasm -lrm count, cuobjdump -sass) under a normal-return
  // subprocess deadline, then size-gates the in-process parsers
  // (parse_sass / parse_rfu) against ENHANCED_TRACER_MAX_PARSER_INPUT_KB.
  // Both bounds use ordinary control flow — no SIGALRM, no longjmp.
  for (const auto &entry : std::filesystem::directory_iterator(absolute_cubin_path))
  {
    std::string aux_cubin = entry.path().filename().string();
    std::string base_name = entry.path().stem().string();
    std::string command_get_register_usage =
        "cd " + shell_quote(register_usage_path) +
        " && nvdisasm -lrm count ../cubin/" + shell_quote(aux_cubin) +
        " > " + shell_quote(base_name + ".rfu");
    std::string command_get_sass =
        "cd " + shell_quote(sass_path) + " && cuobjdump -sass ../cubin/" +
        shell_quote(aux_cubin) + " > " + shell_quote(base_name + ".sass");
    std::cout << "Parsing cubin: " << aux_cubin << std::endl;

    SubprocessResult rfu_res = run_subprocess_with_timeout(
        command_get_register_usage, enhanced_tracer_per_cubin_timeout_s);
    SubprocessResult sass_res = run_subprocess_with_timeout(
        command_get_sass, enhanced_tracer_per_cubin_timeout_s);

    // Restore the legacy abort policy on every non-timeout subprocess
    // failure: kForkFailed and kExecFailed surface as a synthesised
    // exit status of 127 (the conventional shell "command not found"
    // value), and an ordinary kExited result with non-zero status is
    // forwarded verbatim. check_system_call() aborts on every code
    // EXCEPT the legacy-tolerated `nvdisasm -lrm count` exception
    // (matching the pre-Round-22 policy at the top of this file).
    // kTimedOut is the only soft-failure path; it falls through to
    // the typed-diagnostic skip-parsers gate below.
    auto enforce_failure_policy = [](const SubprocessResult &r,
                                     const char *cmd) {
      switch (r.outcome) {
        case SubprocessOutcome::kForkFailed:
        case SubprocessOutcome::kExecFailed:
          check_system_call(127, cmd);
          break;
        case SubprocessOutcome::kExited:
          if (r.exit_status != 0) {
            check_system_call(r.exit_status, cmd);
          }
          break;
        case SubprocessOutcome::kTimedOut:
          break;
      }
    };
    enforce_failure_policy(rfu_res, command_get_register_usage.c_str());
    enforce_failure_policy(sass_res, command_get_sass.c_str());

    // Skip-parsers gate: skip both parse_sass and parse_rfu when the
    // parser input is missing OR exceeds the size threshold OR a
    // producer timed out. Treating "missing" as a skip reason (rather
    // than as a valid 0-byte input) closes the case where the legacy
    // nvdisasm-tolerance branch lets the loop reach this gate without
    // an .rfu file on disk; it also covers any future producer that
    // exits 0 but writes no output. Threshold == 0 disables the size
    // sub-check; the existence sub-check is always active.
    std::filesystem::path sass_file =
        std::filesystem::path(cwd) / sass_path / (base_name + ".sass");
    std::filesystem::path rfu_file = std::filesystem::path(cwd) /
                                     register_usage_path /
                                     (base_name + ".rfu");
    ParserInputStat sass_stat = stat_parser_input(sass_file);
    ParserInputStat rfu_stat = stat_parser_input(rfu_file);
    uintmax_t threshold_bytes =
        (uintmax_t)std::max(0, enhanced_tracer_max_parser_input_kb) * 1024UL;
    bool missing_input = !sass_stat.exists || !rfu_stat.exists;
    bool over_size = (enhanced_tracer_max_parser_input_kb > 0) &&
                     (sass_stat.bytes > threshold_bytes ||
                      rfu_stat.bytes > threshold_bytes);
    bool producer_timed_out =
        (sass_res.outcome == SubprocessOutcome::kTimedOut) ||
        (rfu_res.outcome == SubprocessOutcome::kTimedOut);
    bool skip_parsers = missing_input || over_size || producer_timed_out;

    if (skip_parsers) {
      std::cout << "enhanced_tracer_cubin_parse_timeout: " << aux_cubin
                << " (sass=" << (sass_stat.bytes / 1024UL) << "KB"
                << (sass_stat.exists ? "" : " (missing)")
                << " rfu=" << (rfu_stat.bytes / 1024UL) << "KB"
                << (rfu_stat.exists ? "" : " (missing)")
                << " threshold=" << enhanced_tracer_max_parser_input_kb
                << "KB; producer="
                << (producer_timed_out ? "timeout" : "exited")
                << "); skipping auxiliary extra_info for this cubin "
                << "(primary trace artifacts unaffected)."
                << std::endl;
      continue;
    }

    // Both producers exited 0 (or returned via the tolerated nvdisasm
    // branch with the .rfu input present), both files exist, and both
    // are within the size budget. Run the in-process parsers.
    if (!parse_sass(binary_version, entry)) {
      std::cerr << "Warning: failed to parse SASS output "
                << entry.path().string() << std::endl;
    }
    if (!parse_rfu(entry)) {
      std::cerr << "Warning: failed to parse register-usage output "
                << entry.path().string() << std::endl;
    }
  }
  } // end if (cubin_extraction_succeeded)
  for(auto kernel_name : all_kernels_key_instructions_by_pc) {
    for(auto variant : kernel_name.second) {
      if(!variant.sass_has_been_parsed || !variant.rfu_has_been_parsed) {
        std::cout << "Error. Kernel " << kernel_name.first << " variant " << variant.variant_id << " has not been parsed." << std::endl;
        std::cout << "SASS parsed: " << variant.sass_has_been_parsed << std::endl;
        std::cout << "RFU parsed: " << variant.rfu_has_been_parsed << std::endl;
        std::cout << "Traced instructions: " << std::endl;
        print_map(variant.key_instructions_by_pc);
        auto it_already_captured_instr = map_func_addr_to_pc_to_sass_instr.find(variant.func_addr);
        if (it_already_captured_instr == map_func_addr_to_pc_to_sass_instr.end()) {
          fprintf(stderr,
                  "FATAL: parsed-kernel fallback has no captured SASS map for "
                  "function address 0x%" PRIx64 "\n",
                  static_cast<uint64_t>(variant.func_addr));
          fflush(stderr);
          _exit(EXIT_FAILURE);
        }
        std::string kernel_name_to_add = variant.original_kernel_name + variant_delimiter_str + std::to_string(variant.variant_id);
        if (m_enhanced_traced_execution->has_kernel(kernel_name_to_add) &&
            !m_enhanced_traced_execution->remove_specific_kernel(
                kernel_name_to_add)) {
          fprintf(stderr,
                  "FATAL: failed to discard partial auxiliary metadata for "
                  "kernel %s before fallback reconstruction\n",
                  kernel_name_to_add.c_str());
          fflush(stderr);
          _exit(EXIT_FAILURE);
        }
        if (!m_enhanced_traced_execution->add_no_binary_kernel(
                kernel_name_to_add, variant.unique_function_id,
                variant.func_addr, binary_version,
                it_already_captured_instr->second, false)) {
          fprintf(stderr,
                  "FATAL: failed to construct no-binary fallback metadata "
                  "for kernel %s\n",
                  kernel_name_to_add.c_str());
          fflush(stderr);
          _exit(EXIT_FAILURE);
        }
      }
    }
  }
  std::cout << "Enhanced tracer has parsed " << all_kernels_key_instructions_by_pc.size() << "/" << already_instrumented.size() << " kernels" << std::endl;
  if(!intermediate_extra_files_persistance) {
    remove_folder(cubin_path.c_str());
    remove_folder(sass_path.c_str());
    remove_folder(register_usage_path.c_str());
  }
  m_enhanced_traced_execution->remove_useless_kernels();
  const std::string execution_info_path =
      extrainfo_path + "/enhanced_execution_info.json";
  if (!m_enhanced_traced_execution->SerializeToFile(execution_info_path)) {
    fprintf(stderr,
            "FATAL: failed to publish enhanced execution metadata %s\n",
            execution_info_path.c_str());
    fflush(stderr);
    _exit(EXIT_FAILURE);
  }
}

void nvbit_at_ctx_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  pending_devices_to_finish++;
  if (verbose) {
      printf("Tracer: Starting context %p\n", ctx);
  }
  if (ctx_state_map.find(ctx) != ctx_state_map.end()) {
    fail_device_identity("nvbit_at_ctx_init/duplicate context",
                         "context handle is already registered");
  }
  CTXstate* ctx_state = new CTXstate;
  ctx_state_map[ctx] = ctx_state;
  // Application allocations may occur before NVBit calls nvbit_tool_init at
  // the first kernel launch. Establish checked non-allocating driver identity
  // now so those callbacks cannot observe the CTXstate sentinel values. The
  // guard suppresses the driver queries' own recursive callbacks.
  skip_callback_flag = true;
  initialize_or_validate_device_count("nvbit_at_ctx_init/device count");
  ctx_state->device_id = checked_current_context_device_id(
      nullptr, "nvbit_at_ctx_init/context device");
  skip_callback_flag = false;
  pthread_mutex_unlock(&mutex);
}

static void init_context_state(CUcontext ctx) {
  CTXstate* ctx_state = lookup_context_state_or_fail(
      ctx, "init_context_state/context");
  if (ctx_state->instruction_channel_initialized) {
    return;
  }
  if (stop_report == nullptr) {
    is_first_init_context_call = false;
    const cudaError_t stop_report_status =
        cudaMallocManaged(&stop_report, num_devices * sizeof(bool));
    if (stop_report_status != cudaSuccess || stop_report == nullptr) {
      std::ostringstream detail;
      detail << "cudaMallocManaged(stop_report) status="
             << static_cast<int>(stop_report_status)
             << " num_devices=" << num_devices;
      fail_device_identity("init_context_state/stop_report allocation",
                           detail.str());
    }
    for (int i = 0; i < num_devices; ++i) stop_report[i] = false;
  }
  // A context may have deferred tool initialization during graph capture, or
  // NVBit may have initialized a context that never reached nvbit_tool_init.
  // In either case the global counter must be seeded before the first launch;
  // zero is reserved for the uninitialized state and cannot be advanced.
  for (int i = 0; i < num_devices; ++i) {
    int expected = 0;
    kernel_id[i].compare_exchange_strong(
        expected, 1, std::memory_order_relaxed,
        std::memory_order_relaxed);
  }
  ctx_state->device_id = checked_current_context_device_id(
      ctx_state, "init_context_state/context device");
  // Release-store: the recv thread's acquire-load at :1385 must observe
  // WORKING on its very first iteration, which happens strictly after
  // channel_host.init() spawns it below.
  ctx_state->recv_thread_done.store(RecvThreadState::WORKING, std::memory_order_release);
  ctx_state->channel_dev = nullptr;
  const cudaError_t channel_status =
      cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
  if (channel_status != cudaSuccess || ctx_state->channel_dev == nullptr) {
    std::ostringstream detail;
    detail << "cudaMallocManaged(channel_dev) status="
           << static_cast<int>(channel_status);
    fail_device_identity("init_context_state/channel allocation", detail.str());
  }
  ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                               ctx_state->channel_dev, recv_thread_fun, ctx);
  ctx_state->instruction_channel_initialized = true;
  nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());

  /* Closure-plan NVBit phase-injection contract: when the env-gated
   * activation switch is on, allocate a per-context phase channel
   * (separate from the instruction-trace channel above) and spawn
   * the phase recv thread. The phase channel is non-allocated when
   * the env var is unset so the existing instruction-trace pipeline
   * is unchanged.
   */
  if (phase_injection_enabled) {
    ctx_state->phase_recv_thread_done.store(RecvThreadState::WORKING,
                                            std::memory_order_release);
    const cudaError_t phase_channel_status =
        cudaMallocManaged(&ctx_state->phase_channel_dev, sizeof(ChannelDev));
    if (phase_channel_status != cudaSuccess ||
        ctx_state->phase_channel_dev == nullptr) {
      std::ostringstream detail;
      detail << "cudaMallocManaged(phase_channel_dev) status="
             << static_cast<int>(phase_channel_status);
      fail_device_identity("init_context_state/phase channel allocation",
                           detail.str());
    }
    ctx_state->phase_channel_host.init((int)ctx_state_map.size() + 1024,
                                       CHANNEL_SIZE,
                                       ctx_state->phase_channel_dev,
                                       phase_recv_thread_fun,
                                       ctx);
  }
}

void nvbit_tool_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  CTXstate* ctx_state = lookup_context_state_or_fail(
      ctx, "nvbit_tool_init/context");
  if (ctx_state->tool_init_complete) {
    pthread_mutex_unlock(&mutex);
    return;
  }
  if (active_capture_counts.find(ctx) != active_capture_counts.end()) {
    // CUDA graph capture disallows the managed allocations used by the NVBit
    // channel. End-capture will complete initialization before graph launch.
    ctx_state->tool_init_deferred = true;
    pthread_mutex_unlock(&mutex);
    return;
  }
  // Tool-owned CUDA setup calls re-enter nvbit_at_cuda_event. Suppress those
  // callbacks so the allocation registry never consumes partially initialized
  // device identity or records tracer-owned managed allocations.
  skip_callback_flag = true;
  init_context_state(ctx);
  ctx_state = lookup_context_state_or_fail(ctx, "nvbit_tool_init/context");
  // Resolve binary_version per-context from the current CUDA device's
  // compute capability, unconditionally. nvbit_tool_init is the
  // correct hook per the upstream NVBit warning: "Do not call CUDA memory
  // allocation in nvbit_at_ctx_init(). It will cause deadlocks. Do them
  // in nvbit_tool_init()." Doing it here guarantees enhanced_tracer()
  // has a valid `-arch=sm_<N>` argument for its cuobjdump shell-out even
  // if NVBit never hooks a kernel launch under this context.
  //
  // The resolution is keyed to `ctx` so multi-context / multi-device
  // traces see each context's own arch — the previous
  // `if (binary_version <= 0)` guard pinned the whole process to the
  // first-initialized context's value, which silently used the wrong
  // arch when a later context on a different GPU only exercised the
  // cudaLaunchKernelEx path (no per-launch hook to refresh the global).
  ctx_state->binary_version = resolve_binary_version_for_current_device();
  // Keep the process-global `binary_version` in sync with the most
  // recently initialized context so downstream consumers that still read
  // the global (enhanced_tracer/parse_sass) see a current value when
  // they run.
  binary_version = ctx_state->binary_version;
  ctx_state->tool_init_complete = true;
  fprintf(stderr,
          "INFO: tracer binary_version initialised per-context from "
          "device compute capability at tool-init: binary_version=%d "
          "(ctx=%p)\n",
          ctx_state->binary_version, (void*)ctx);
  fflush(stderr);
  skip_callback_flag = false;
  pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  bool exact_capture_contract_failed = false;
  skip_callback_flag = true;
  if (verbose) {
      printf("Tracer: Terminating context %p\n", ctx);
  }
  /* get context state from map */
  CTXstate* ctx_state = lookup_context_state_or_fail(
      ctx, "nvbit_at_ctx_term/context");
  pending_devices_to_finish--;
  // Context destruction releases context-bound allocations regardless of
  // whether another CUDA context remains active. Pool allocations are kept in
  // the separate process-wide map and are retired only by their async free.
  nvbit_tracer::retire_context_allocations(ctx);
  if(pending_devices_to_finish == 0) {
    // Final dyn_trace finalization + serialization. Take recv_container_mutex
    // around the writes and the SerializeToOstream call. At this point any
    // other contexts' recv threads are still alive (we haven't set STOP for
    // this context yet, and we don't touch other contexts here), so their
    // concurrent writes would race with our finalize. The mutex prevents that.
    {
      std::lock_guard<std::mutex> rcm_lock(recv_container_mutex);
      dyn_trace.set_nvbit_version(NVBIT_VERSION);
      dyn_trace.set_accelsim_version(TRACER_VERSION);
      // GATHER_REGISTERS is retained as a compatibility knob, but this
      // tracer does not serialize register snapshots into the protobuf
      // instruction stream.  Never advertise the capability merely because
      // the environment variable was set; the separate record_reg_vals NVBit
      // example has its own text artifact and is normalized by
      // register_value_adapter.py with an explicit partial-identity claim.
      if (gather_registers) {
        fprintf(stderr,
                "WARNING: GATHER_REGISTERS requested, but enhanced protobuf "
                "register snapshots are not emitted; keeping "
                "is_gathered_registers_values=false.\n");
      }
      dyn_trace.set_is_gathered_registers_values(false);
      // Trace-contract identity + capability. The v2 contract preserves the
      // address-carrying payload classes and additionally requires every kernel
      // to carry presence-bearing static and dynamic shared-memory components
      // whose sum equals the legacy combined field.
      dyn_trace.set_trace_contract_id("sm86_addr_carry_ldgsts_dst_v3");
      const dynamic_trace::MemoryOpClass retained_capabilities[] = {
          dynamic_trace::MEMORY_OP_CLASS_GLOBAL,
          dynamic_trace::MEMORY_OP_CLASS_SHARED,
          dynamic_trace::MEMORY_OP_CLASS_LOCAL,
          dynamic_trace::MEMORY_OP_CLASS_ATOMIC,
          dynamic_trace::MEMORY_OP_CLASS_ASYNC_COPY};
      for (const auto capability : retained_capabilities) {
        bool present = false;
        for (int i = 0; i < dyn_trace.memory_payload_classes_size(); ++i) {
          if (dyn_trace.memory_payload_classes(i) == capability) {
            present = true;
            break;
          }
        }
        if (!present) dyn_trace.add_memory_payload_classes(capability);
      }
      // Ensure the serialized trace reports a device binary_version that
      // actually corresponds to one of the contexts that contributed the
      // recorded kernels. In mixed-architecture multi-context runs the
      // terminating context may be a device that never launched any of
      // the kernels present in `dyn_trace`, so blindly overwriting with
      // `ctx_state->binary_version` here would mistag every recorded
      // kernel and make `enhanced_tracer()`'s `cuobjdump -arch=sm_<N>`
      // shell-out extract with the wrong arch or skip the kernels
      // entirely.
      //
      // Authoritative value:
      //   The per-kernel-launch hook at :1385 calls
      //   `dyn_trace.set_binary_version(binary_version)` with the
      //   contributing context's arch for every kernel actually added
      //   to dyn_trace. Whichever context launched the last kernel wins
      //   in the single-`binary_version`-per-trace schema (multi-arch
      //   cubin extraction is explicitly out of phase 1 scope).
      //
      // Rules applied at this final-serialize path:
      //   (a) If `dyn_trace.binary_version() > 0`, at least one kernel
      //       was recorded and the value came from its contributing
      //       context. Preserve it; only sync the process-global scratch
      //       so `enhanced_tracer()` / `parse_sass()` below see the
      //       contributing context's arch, not this terminating
      //       context's (possibly different) arch.
      //   (b) Else, no kernel launch ever wrote `dyn_trace.binary_version`
      //       (the zero-kernel-trace fallback when NVBit misses the
      //       launch-hook surface). Fall back to the terminating
      //       context's device-resolved value so the serialized trace
      //       still carries a valid arch tag for downstream consumers.
      if (dyn_trace.binary_version() > 0) {
        binary_version = dyn_trace.binary_version();
      } else if (ctx_state->binary_version > 0) {
        dyn_trace.set_binary_version(ctx_state->binary_version);
        binary_version = ctx_state->binary_version;
      }
      std::string program_name = get_program_path();
      std::size_t found = program_name.find_last_of("/");
      if(found != std::string::npos) {
        program_name = program_name.substr(found + 1);
      }
      dyn_trace.set_name(program_name);
      std::string filename = traces_path + "/dynamic_trace.pb";
      std::ofstream ofs(filename, std::ios::binary);
      if (!ofs) {
        fprintf(stderr, "FATAL: failed to open dynamic trace %s for writing\n",
                filename.c_str());
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      if (!dyn_trace.SerializeToOstream(&ofs)) {
        fprintf(stderr, "FATAL: failed to serialize dynamic trace %s\n",
                filename.c_str());
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      ofs.flush();
      if (!ofs) {
        fprintf(stderr, "FATAL: failed while writing dynamic trace %s\n",
                filename.c_str());
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      ofs.close();
      if (!ofs) {
        fprintf(stderr, "FATAL: failed to close dynamic trace %s\n",
                filename.c_str());
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
      std::cout << "Serialized protocol buffer written to " << filename << std::endl;
      // Keep the process-global trace accumulated across sequential CUDA
      // contexts. NVBit may destroy one context before creating the next;
      // clearing here would make the final dynamic_trace.pb silently omit
      // every kernel emitted by earlier contexts.
    }
    /* Allocation registry sidecar. Always written (regardless
     * of skip_enhanced_tracer) because the allocations.json
     * is consumed by the validator allocation/range identity
     * check, not by enhanced_tracer's cuobjdump pass.
     * The path mirrors the global `extrainfo_path` so it
     * lives under the same `traces/extra_info/` tree as
     * `enhanced_execution_info.json`. */
    const bool allocation_sidecar_written =
        write_allocation_sidecar(extrainfo_path);
    /* Trace-sampling capture sidecar (per-block counts / selected-block set);
     * a no-op unless a trace-sampling capture mode is active. */
    const bool sampling_sidecar_written =
        write_trace_sampling_sidecar(extrainfo_path);
    if (!allocation_sidecar_written || !sampling_sidecar_written) {
      fprintf(stderr,
              "FATAL: required trace sidecar publication failed; refusing "
              "to publish an incomplete capture\n");
      fflush(stderr);
      _exit(EXIT_FAILURE);
    }
    const bool module_origin_sidecar_written =
        write_module_origin_sidecar(extrainfo_path);
    exact_capture_contract_failed =
        max_accepted_active_launches != 0 &&
        (!active_capture_contract_satisfied() ||
         !module_origin_sidecar_written);
    if (exact_capture_contract_failed) {
      fprintf(stderr,
              "FATAL: exact active-launch capture contract was not "
              "satisfied at finalization (accepted=%llu, mappings=%zu, "
              "in_flight=%llu, skipped_after_limit=%llu, concurrent=%llu, "
              "inactive_records=%llu, "
              "accepted_records=%llu). "
              "Evidence was written "
              "with exact_contract_satisfied=false; the process will fail.\n",
              (unsigned long long)active_launch_gate.accepted_launches(),
              accepted_module_mappings.size(),
              (unsigned long long)
                  active_launch_gate.in_flight_accepted_launches(),
              (unsigned long long)
                  active_launch_gate.active_launches_skipped_after_limit(),
              (unsigned long long)
                  active_launch_gate.concurrent_limit_violations(),
              (unsigned long long)inactive_channel_records_dropped.load(
                  std::memory_order_acquire),
              (unsigned long long)accepted_channel_records.load(
                  std::memory_order_acquire));
      fflush(stderr);
    }

    /* R23 producer-side typed-exclusion sidecars
     * (`unsupported_memory_path.json` / `unmapped_memory_range.json`).
     * Always written for the same reason as the allocation
     * sidecar; an empty list `[]` is emitted for whichever
     * typed reason has zero accumulated entries so consumers
     * never see "file missing" as a degenerate case. The
     * device tag is built from the captured `binary_version`
     * formatted as `sm_<n>` to match the schema's compute-
     * capability convention. */
    {
      const uint32_t bin_ver =
          dyn_trace.binary_version() > 0
              ? dyn_trace.binary_version()
              : (binary_version > 0 ? binary_version : 0);
      const std::string device_tag =
          "sm_" + std::to_string(bin_ver);
      if (!nvbit_tracer::serialize_exclusion_sidecars(
              extrainfo_path, device_tag,
              slug_of_function_unique_id)) {
        fprintf(stderr,
                "FATAL: typed-exclusion sidecar publication failed; "
                "refusing to publish an incomplete trace\n");
        fflush(stderr);
        _exit(EXIT_FAILURE);
      }
    }
    if (skip_enhanced_tracer) {
      std::cout << "Skipping the enhanced tracer (SKIP_ENHANCED_TRACER=1)" << std::endl;
    } else {
      std::cout << "Starting the enhanced tracer" << std::endl;
      enhanced_tracer();
      std::cout << "Terminated the enhanced tracer" << std::endl;
    }
  }
  // A context can be created and destroyed without ever reaching
  // nvbit_tool_init (for example, an allocation-only CUDA probe). In that
  // path no receiver thread or channel exists, so waiting for FINISHED would
  // deadlock forever. Only perform the stop/join/free sequence after channel
  // initialization has actually published a receiver.
  if (ctx_state->instruction_channel_initialized) {
    // Release-store STOP so the recv thread's acquire-load at :1385 observes
    // it on its next iteration and exits the WORKING loop. Then acquire-load
    // FINISHED in a sched_yield spin so we don't burn a whole CPU core.
    ctx_state->recv_thread_done.store(RecvThreadState::STOP,
                                      std::memory_order_release);
    while (ctx_state->recv_thread_done.load(std::memory_order_acquire) !=
           RecvThreadState::FINISHED) {
      sched_yield();
    }
    ctx_state->channel_host.destroy(false);
  }
  if (ctx_state->channel_dev != nullptr) {
    cudaFree(ctx_state->channel_dev);
    ctx_state->channel_dev = nullptr;
  }
  /* Closure-plan NVBit phase-injection contract: tear down the
   * per-context phase channel symmetrically with its allocation in
   * init_context_state. Only allocated when phase_injection_enabled
   * is true, so the destroy/free pair is gated on the same flag.
   */
  bool phase_teardown_timed_out = false;
  if (phase_injection_enabled && ctx_state->phase_channel_dev != nullptr) {
    ctx_state->phase_recv_thread_done.store(RecvThreadState::STOP,
                                            std::memory_order_release);
    /* Bounded FINISHED-flag spin. The same env-var that bounds the
     * per-launch flush spin also bounds this teardown spin so a
     * permanently-stuck recv thread cannot wedge process termination.
     * On exhaustion we deliberately do NOT call
     * `phase_channel_host.destroy(false)` (which internally
     * pthread_join's the recv thread) and do NOT call
     * `cudaFree(phase_channel_dev)`: both are unsafe if the recv
     * thread is still potentially live and may still be touching
     * the device-side phase buffer. We also intentionally retain
     * `CTXstate` itself: the still-live recv thread holds
     * `CTXstate*` and `ChannelHost*` pointers (see
     * `phase_recv_thread_fun` :1427) that it dereferences on every
     * iteration of its WORKING loop (atomic flags, the phase records
     * mutex, the records buffer). `delete ctx_state` here would
     * race that thread and produce a use-after-free. Process
     * teardown reclaims the allocations as the OS tears the address
     * space down. The leak is bounded (one CTXstate + one phase
     * channel per terminated context).
     */
    long long iters_cap = phase_flush_timeout_iters();
    long long iters_remaining = iters_cap;
    while (ctx_state->phase_recv_thread_done.load(
               std::memory_order_acquire) != RecvThreadState::FINISHED) {
      if (iters_remaining == 0) {
        phase_teardown_timed_out = true;
        break;
      }
      iters_remaining--;
      sched_yield();
    }
    if (!phase_teardown_timed_out) {
      ctx_state->phase_channel_host.destroy(false);
      cudaFree(ctx_state->phase_channel_dev);
      ctx_state->phase_channel_dev = nullptr;
    } else {
      fprintf(stderr,
              "WARNING: SIMBENCH_PHASE_FLUSH_TIMEOUT_ITERS=%lld "
              "exhausted during phase channel teardown "
              "(timed_join_attempted=false); phase channel and "
              "CTXstate intentionally leaked to avoid use-after-free "
              "by the still-live phase recv thread; process teardown "
              "will reclaim the allocations.\n",
              (long long)iters_cap);
    }
  }
  skip_callback_flag = false;
  // All receiver threads that consult ctx_state_map have finished before this
  // point. Remove the entry before freeing the state so a later CUDA context
  // handle cannot resolve to a dangling CTXstate pointer. On the timeout path
  // the phase receiver may still be live, but it already holds its CTXstate
  // pointer and never consults the map again, so the lookup can be removed
  // there as well while the state itself remains intentionally leaked until
  // process teardown.
  // Capture counts are keyed by the same opaque context handle. Clear them
  // before a handle can be recycled for a later context, otherwise a stale
  // count would incorrectly defer tool initialization.
  active_capture_counts.erase(ctx);
  // Context-bound module handles can be recycled after destruction. Libraries
  // are intentionally retained until their separate cuLibraryUnload event.
  for (auto it = loaded_module_origins.begin(); it != loaded_module_origins.end();) {
    if (it->first.ctx == ctx) it = loaded_module_origins.erase(it);
    else ++it;
  }
  for (auto it = loaded_library_modules.begin(); it != loaded_library_modules.end();) {
    if (it->first.ctx == ctx) it = loaded_library_modules.erase(it);
    else ++it;
  }
  ctx_state_map.erase(ctx);
  if (!phase_teardown_timed_out) {
    delete ctx_state;
  }
  pthread_mutex_unlock(&mutex);
  if (exact_capture_contract_failed) _exit(EXIT_FAILURE);
}
