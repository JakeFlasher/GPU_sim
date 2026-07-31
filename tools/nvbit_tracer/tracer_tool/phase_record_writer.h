#ifndef NVBIT_TRACER_TOOL_PHASE_RECORD_WRITER_H
#define NVBIT_TRACER_TOOL_PHASE_RECORD_WRITER_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <sys/types.h>

namespace simbench_nvbit {

// Host-only phase-record publisher. The live tracer opens TRACES_FOLDER once,
// retains this object's private directory descriptor, and publishes leaves
// relative to a descriptor duplicate whose device/inode identity is revalidated
// before use. The helper deliberately has no CUDA or NVBit surface so its
// authority and durability contract can be compiled and executed on a host.
class PhaseRecordWriter {
 public:
  PhaseRecordWriter();
  ~PhaseRecordWriter();

  PhaseRecordWriter(const PhaseRecordWriter&) = delete;
  PhaseRecordWriter& operator=(const PhaseRecordWriter&) = delete;

  // Opens and snapshots the phase-output directory. An exact inherited
  // /proc/self/fd/<decimal> authority is duplicated directly; otherwise the
  // final path component must itself be a directory, not a symlink. This method
  // may be called once.
  bool OpenDirectory(const std::string& directory_path, std::string* error);

  // Establishes a process-wide, single-GPU phase authority. Re-admitting the
  // same device is idempotent; any other device fails closed before its
  // independent launch counter can enter the unqualified filename namespace.
  bool AdmitDevice(int device_id, std::string* error);

  // Claims the process-global managed launch identity for one phase-enabled
  // launch. A second in-flight launch, including one on the same device, poisons
  // the writer and fails closed before it can overwrite that identity.
  bool BeginLaunch(int device_id, std::uint32_t launch_index,
                   std::string* error);
  bool EndLaunch(int device_id, std::uint32_t launch_index,
                 std::string* error);

  // Exclusively creates phase_records_<repeat>_<launch>.bin relative to the
  // retained directory authority and durably writes exactly payload_size bytes.
  // Any failure poisons this writer; the live tracer treats false as fatal.
  bool WriteRecord(int device_id, std::uint32_t repeat_index,
                   std::uint32_t launch_index, const void* payload,
                   std::size_t payload_size, std::string* error);

  // Retains bytes from an incomplete flush under the explicitly noncanonical
  // .partial-phase-records_<repeat>_<launch>.bin namespace. This has the same
  // exclusive/durable filesystem contract but can never authorize a canonical
  // phase record.
  bool WritePartialRecord(int device_id, std::uint32_t repeat_index,
                          std::uint32_t launch_index, const void* payload,
                          std::size_t payload_size, std::string* error);

#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
  using WriteHook = ssize_t (*)(int, const void*, std::size_t, void*);
  using CurrentPidHook = pid_t (*)(void*);
  using PostCreateHook = void (*)(int, int, const char*, void*);
  using FsyncHook = int (*)(int, bool, void*);
  using CloseHook = int (*)(int, const char*, void*);

  // Host-regression hooks only. Production translation units do not expose
  // these methods or carry hook state.
  void SetWriteHookForTesting(WriteHook hook, void* opaque);
  void SetCurrentPidHookForTesting(CurrentPidHook hook, void* opaque);
  void SetPostCreateHookForTesting(PostCreateHook hook, void* opaque);
  void SetFsyncHookForTesting(FsyncHook hook, void* opaque);
  void SetCloseHookForTesting(CloseHook hook, void* opaque);
  int DirectoryDescriptorForTesting() const;
#endif

 private:
  enum class LeafKind { kCanonical, kPartial };

  pid_t CurrentPidBeforeLock() const;
  bool CheckOwnerPidBeforeLock(std::string* error) const;
  bool SetFailureLocked(const std::string& message, std::string* error);
  bool CheckUsableLocked(std::string* error);
  bool AdmitDeviceLocked(int device_id, std::string* error);
  bool VerifyDirectoryIdentityLocked(int descriptor,
                                     const char* descriptor_role,
                                     std::string* error);
  int AcquireOperationDirectoryLocked(std::string* error);
  ssize_t WriteOnceLocked(int descriptor, const void* payload,
                          std::size_t payload_size);
  bool FsyncDescriptorLocked(int descriptor, bool is_directory);
  bool CloseDescriptorLocked(int* descriptor, const char* role,
                             std::string* message);
  void InvokePostCreateHookLocked(int operation_directory,
                                  int record_descriptor, const char* leaf);
  bool WriteRecordLocked(LeafKind leaf_kind, int device_id,
                         std::uint32_t repeat_index,
                         std::uint32_t launch_index, const void* payload,
                         std::size_t payload_size, std::string* error);

  mutable std::mutex mutex_;
  const pid_t owner_pid_;
  int directory_descriptor_;
  dev_t directory_device_;
  ino_t directory_inode_;
  bool opened_;
  bool poisoned_;
  int admitted_device_id_;
  bool launch_active_;
  int active_launch_device_id_;
  std::uint32_t active_launch_index_;

#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
  WriteHook write_hook_;
  void* write_hook_opaque_;
  CurrentPidHook current_pid_hook_;
  void* current_pid_hook_opaque_;
  PostCreateHook post_create_hook_;
  void* post_create_hook_opaque_;
  FsyncHook fsync_hook_;
  void* fsync_hook_opaque_;
  CloseHook close_hook_;
  void* close_hook_opaque_;
#endif
};

// Shared host-testable fatal path used by the live tracer. Canonical publication
// failures never return to the application as a successful capture.
[[noreturn]] void FailPhaseRecordAuthority(const char* stage,
                                           const std::string& detail);
void PublishCanonicalRecordOrFail(PhaseRecordWriter& writer, int device_id,
                                  std::uint32_t repeat_index,
                                  std::uint32_t launch_index,
                                  const void* payload,
                                  std::size_t payload_size,
                                  const char* stage);

#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
using PhaseRecordFatalHook =
    void (*)(const char*, const std::string&, void*);
void SetPhaseRecordFatalHookForTesting(PhaseRecordFatalHook hook, void* opaque);
#endif

}  // namespace simbench_nvbit

#endif  // NVBIT_TRACER_TOOL_PHASE_RECORD_WRITER_H
