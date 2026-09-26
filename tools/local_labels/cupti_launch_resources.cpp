// Minimal CUPTI resource-only injector.
//
// It records the driver's selected shared-memory configuration
// (CUpti_ActivityKernel11::sharedMemoryExecuted), which is not recoverable
// from a CUDA carveout preference alone. It deliberately enables only kernel
// activity records: no timing, metric, or replay target is collected.

#include <cupti.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace {
FILE* output = nullptr;
std::mutex output_mutex;
uint64_t launches = 0;
uint64_t dropped = 0;

void check(CUptiResult result) {
  if (result == CUPTI_SUCCESS) return;
  const char* text = nullptr;
  cuptiGetResultString(result, &text);
  std::fprintf(stderr, "cupti_launch_resources: %s\n", text ? text : "CUPTI failure");
  std::exit(2);
}

std::string json_string(const char* value) {
  std::string result = "\"";
  for (const unsigned char* cursor = reinterpret_cast<const unsigned char*>(value);
       cursor != nullptr && *cursor != 0; ++cursor) {
    switch (*cursor) {
      case '"': case '\\': result.push_back('\\'); result.push_back(static_cast<char>(*cursor)); break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      case '\b': result += "\\b"; break;
      case '\f': result += "\\f"; break;
      default: result.push_back(static_cast<char>(*cursor)); break;
    }
  }
  result.push_back('"');
  return result;
}

void CUPTIAPI request_buffer(uint8_t** buffer, size_t* size, size_t* max_records) {
  *size = 8 * 1024 * 1024;
  *max_records = 0;
  void* allocation = nullptr;
  if (posix_memalign(&allocation, 8, *size) != 0) std::abort();
  *buffer = static_cast<uint8_t*>(allocation);
}

void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream, uint8_t* buffer,
                              size_t, size_t valid_size) {
  std::lock_guard<std::mutex> lock(output_mutex);
  CUpti_Activity* activity = nullptr;
  CUptiResult result;
  while ((result = cuptiActivityGetNextRecord(buffer, valid_size, &activity)) ==
         CUPTI_SUCCESS) {
    if (activity->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) continue;
    const auto* kernel = reinterpret_cast<const CUpti_ActivityKernel11*>(activity);
    std::fprintf(
        output,
        "{\"record\":\"launch_resources\",\"kernel_name\":%s,"
        "\"device\":%u,\"stream\":%u,\"correlation_id\":%u,"
        "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
        "\"registers_per_thread\":%u,\"static_shared_memory_bytes\":%u,"
        "\"dynamic_shared_memory_bytes\":%u,\"shared_memory_config_bytes\":%u,"
        "\"carveout_requested\":%u,\"carveout_pct\":%u}\n",
        json_string(kernel->name).c_str(), kernel->deviceId, kernel->streamId,
        kernel->correlationId, static_cast<unsigned>(kernel->gridX),
        static_cast<unsigned>(kernel->gridY), static_cast<unsigned>(kernel->gridZ),
        static_cast<unsigned>(kernel->blockX), static_cast<unsigned>(kernel->blockY),
        static_cast<unsigned>(kernel->blockZ), kernel->registersPerThread,
        static_cast<unsigned>(kernel->staticSharedMemory),
        static_cast<unsigned>(kernel->dynamicSharedMemory),
        kernel->sharedMemoryExecuted, kernel->isSharedMemoryCarveoutRequested,
        kernel->sharedMemoryCarveoutRequested);
    ++launches;
  }
  if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) check(result);
  size_t lost = 0;
  check(cuptiActivityGetNumDroppedRecords(context, stream, &lost));
  dropped += lost;
  std::free(buffer);
}

void finish() {
  check(cuptiActivityFlushAll(1));
  std::lock_guard<std::mutex> lock(output_mutex);
  std::fprintf(output, "{\"record\":\"summary\",\"launches\":%llu,\"dropped\":%llu}\n",
               static_cast<unsigned long long>(launches),
               static_cast<unsigned long long>(dropped));
  std::fflush(output);
  std::fclose(output);
}
}  // namespace

extern "C" int InitializeInjection() {
  const char* path = std::getenv("SIM_RESOURCE_OUTPUT");
  if (path == nullptr) return 0;
  output = std::fopen(path, "wx");
  if (output == nullptr) {
    std::perror(path);
    std::exit(2);
  }
  uint32_t version = 0;
  check(cuptiGetVersion(&version));
  std::fprintf(output, "{\"record\":\"tool\",\"cupti_version\":%u,\"kind\":\"resource_only\"}\n",
               version);
  check(cuptiActivityRegisterCallbacks(request_buffer, complete_buffer));
  check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  std::atexit(finish);
  return 1;
}
