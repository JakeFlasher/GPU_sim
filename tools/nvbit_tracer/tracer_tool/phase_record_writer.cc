#include "phase_record_writer.h"

#include "phase_record_path.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>

namespace simbench_nvbit {
namespace {

constexpr std::size_t kWriteChunkBytes = 1U << 20;
constexpr unsigned kInterruptedRetryLimit = 1024;

#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
PhaseRecordFatalHook g_fatal_hook = nullptr;
void* g_fatal_hook_opaque = nullptr;
#endif

std::string ErrnoMessage(const char* operation, int error_number) {
  std::ostringstream out;
  out << operation << " failed: errno=" << error_number << " ("
      << std::error_code(error_number, std::generic_category()).message()
      << ")";
  return out.str();
}

void AppendFailure(std::string* message, const std::string& addition) {
  if (message->empty()) {
    *message = addition;
  } else {
    *message += "; additionally, " + addition;
  }
}

int OpenDirectoryNoFollow(const char* path) {
  unsigned interrupted = 0;
  for (;;) {
    const int descriptor =
        ::open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor >= 0 || errno != EINTR) return descriptor;
    if (++interrupted > kInterruptedRetryLimit) return -1;
  }
}

int OpenLeafExclusive(int directory_descriptor, const char* leaf) {
  unsigned interrupted = 0;
  for (;;) {
    const int descriptor =
        ::openat(directory_descriptor, leaf,
                 O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                 S_IRUSR | S_IWUSR);
    if (descriptor >= 0 || errno != EINTR) return descriptor;
    if (++interrupted > kInterruptedRetryLimit) return -1;
  }
}

bool FstatChecked(int descriptor, struct stat* result) {
  unsigned interrupted = 0;
  for (;;) {
    if (::fstat(descriptor, result) == 0) return true;
    if (errno != EINTR || ++interrupted > kInterruptedRetryLimit) return false;
  }
}

bool FstatatNoFollowChecked(int directory_descriptor, const char* leaf,
                            struct stat* result) {
  unsigned interrupted = 0;
  for (;;) {
    if (::fstatat(directory_descriptor, leaf, result, AT_SYMLINK_NOFOLLOW) ==
        0) {
      return true;
    }
    if (errno != EINTR || ++interrupted > kInterruptedRetryLimit) return false;
  }
}

int DuplicateCloseOnExec(int descriptor) {
  unsigned interrupted = 0;
  for (;;) {
    const int duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 3);
    if (duplicate >= 0 || errno != EINTR) return duplicate;
    if (++interrupted > kInterruptedRetryLimit) return -1;
  }
}

enum class ProcSelfFdParse { kNotProcSelfFd, kValid, kInvalid };

ProcSelfFdParse ParseProcSelfFdPath(const std::string& path,
                                    int* inherited_descriptor) {
  static constexpr char kPrefix[] = "/proc/self/fd/";
  const std::string prefix(kPrefix);
  if (path.compare(0, prefix.size(), prefix) != 0) {
    return ProcSelfFdParse::kNotProcSelfFd;
  }
  if (path.size() == prefix.size()) return ProcSelfFdParse::kInvalid;

  int value = 0;
  for (std::size_t index = prefix.size(); index < path.size(); ++index) {
    const char character = path[index];
    if (character < '0' || character > '9') {
      return ProcSelfFdParse::kInvalid;
    }
    const int digit = character - '0';
    if (value > (INT_MAX - digit) / 10) {
      return ProcSelfFdParse::kInvalid;
    }
    value = value * 10 + digit;
  }
  *inherited_descriptor = value;
  return ProcSelfFdParse::kValid;
}

bool SameFileIdentity(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino;
}

std::string DirectoryMismatchMessage(const char* role, dev_t expected_device,
                                     ino_t expected_inode,
                                     const struct stat& observed) {
  std::ostringstream out;
  out << role << " no longer names the retained TRACES_FOLDER authority: "
      << "expected_dev=" << static_cast<std::uintmax_t>(expected_device)
      << " expected_ino=" << static_cast<std::uintmax_t>(expected_inode)
      << " observed_dev=" << static_cast<std::uintmax_t>(observed.st_dev)
      << " observed_ino=" << static_cast<std::uintmax_t>(observed.st_ino)
      << " observed_mode=" << static_cast<std::uintmax_t>(observed.st_mode)
      << " observed_nlink=" << static_cast<std::uintmax_t>(observed.st_nlink);
  return out.str();
}

bool ValidateSingleLinkRegular(const struct stat& observed,
                               const struct stat* expected_identity,
                               std::size_t expected_size, bool require_size,
                               const char* role, std::string* message) {
  if (!S_ISREG(observed.st_mode) || observed.st_nlink != 1 ||
      (expected_identity != nullptr &&
       !SameFileIdentity(observed, *expected_identity)) ||
      (require_size &&
       static_cast<std::uintmax_t>(observed.st_size) !=
           static_cast<std::uintmax_t>(expected_size))) {
    std::ostringstream out;
    out << role << " failed single-link regular-file verification: dev="
        << static_cast<std::uintmax_t>(observed.st_dev)
        << " ino=" << static_cast<std::uintmax_t>(observed.st_ino)
        << " mode=" << static_cast<std::uintmax_t>(observed.st_mode)
        << " nlink=" << static_cast<std::uintmax_t>(observed.st_nlink)
        << " size=" << static_cast<std::uintmax_t>(observed.st_size);
    if (expected_identity != nullptr) {
      out << " expected_dev="
          << static_cast<std::uintmax_t>(expected_identity->st_dev)
          << " expected_ino="
          << static_cast<std::uintmax_t>(expected_identity->st_ino);
    }
    if (require_size) out << " expected_size=" << expected_size;
    *message = out.str();
    return false;
  }
  return true;
}

}  // namespace

PhaseRecordWriter::PhaseRecordWriter()
    : owner_pid_(::getpid()),
      directory_descriptor_(-1),
      directory_device_(0),
      directory_inode_(0),
      opened_(false),
      poisoned_(false),
      admitted_device_id_(-1),
      launch_active_(false),
      active_launch_device_id_(-1),
      active_launch_index_(0)
#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
      ,
      write_hook_(nullptr),
      write_hook_opaque_(nullptr),
      current_pid_hook_(nullptr),
      current_pid_hook_opaque_(nullptr),
      post_create_hook_(nullptr),
      post_create_hook_opaque_(nullptr),
      fsync_hook_(nullptr),
      fsync_hook_opaque_(nullptr),
      close_hook_(nullptr),
      close_hook_opaque_(nullptr)
#endif
{
}

pid_t PhaseRecordWriter::CurrentPidBeforeLock() const {
#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
  if (current_pid_hook_ != nullptr) {
    return current_pid_hook_(current_pid_hook_opaque_);
  }
#endif
  return ::getpid();
}

bool PhaseRecordWriter::CheckOwnerPidBeforeLock(std::string* error) const {
  const pid_t current_pid = CurrentPidBeforeLock();
  if (current_pid == owner_pid_) return true;
  if (error != nullptr) {
    std::ostringstream out;
    out << "phase-record writer belongs to owner_pid=" << owner_pid_
        << " but current_pid=" << current_pid
        << "; rejecting inherited post-fork state before acquiring its mutex";
    *error = out.str();
  }
  return false;
}

PhaseRecordWriter::~PhaseRecordWriter() {
  // A child may inherit this object while another parent thread owns mutex_. Do
  // not touch that inherited mutex, counter state, or possibly reused descriptor.
  if (!CheckOwnerPidBeforeLock(nullptr)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (directory_descriptor_ >= 0) {
    // A destructor cannot report a close error. Every publication descriptor is
    // closed explicitly and checked during publication; this is only the retained
    // process-lifetime directory authority fallback.
    ::close(directory_descriptor_);
    directory_descriptor_ = -1;
  }
}

bool PhaseRecordWriter::SetFailureLocked(const std::string& message,
                                         std::string* error) {
  poisoned_ = true;
  if (error != nullptr) *error = message;
  return false;
}

bool PhaseRecordWriter::CheckUsableLocked(std::string* error) {
  if (poisoned_) {
    if (error != nullptr) {
      *error = "phase-record writer is poisoned by an earlier authority or "
               "publication failure";
    }
    return false;
  }
  if (!opened_ || directory_descriptor_ < 0) {
    return SetFailureLocked(
        "phase-record TRACES_FOLDER authority has not been opened", error);
  }
  return true;
}

bool PhaseRecordWriter::VerifyDirectoryIdentityLocked(
    int descriptor, const char* descriptor_role, std::string* error) {
  struct stat observed {};
  if (!FstatChecked(descriptor, &observed)) {
    const int saved_errno = errno;
    if (error != nullptr) {
      *error = ErrnoMessage(descriptor_role, saved_errno);
    }
    return false;
  }
  if (!S_ISDIR(observed.st_mode) || observed.st_nlink == 0 ||
      observed.st_dev != directory_device_ ||
      observed.st_ino != directory_inode_) {
    if (error != nullptr) {
      *error = DirectoryMismatchMessage(descriptor_role, directory_device_,
                                        directory_inode_, observed);
    }
    return false;
  }
  return true;
}

bool PhaseRecordWriter::CloseDescriptorLocked(int* descriptor, const char* role,
                                              std::string* message) {
  if (*descriptor < 0) return true;
  const int closing = *descriptor;
  *descriptor = -1;
  int result = 0;
#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
  if (close_hook_ != nullptr) {
    result = close_hook_(closing, role, close_hook_opaque_);
  } else {
    result = ::close(closing);
  }
#else
  result = ::close(closing);
#endif
  if (result == 0) return true;
  const int saved_errno = errno;
  AppendFailure(message, ErrnoMessage(role, saved_errno));
  return false;
}

bool PhaseRecordWriter::OpenDirectory(const std::string& directory_path,
                                      std::string* error) {
  if (!CheckOwnerPidBeforeLock(error)) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  if (opened_ || poisoned_) {
    return SetFailureLocked(
        "phase-record TRACES_FOLDER authority may be opened exactly once",
        error);
  }
  if (directory_path.empty()) {
    return SetFailureLocked(
        "phase-record TRACES_FOLDER authority path is empty", error);
  }

  int inherited_descriptor = -1;
  const ProcSelfFdParse proc_fd_parse =
      ParseProcSelfFdPath(directory_path, &inherited_descriptor);
  if (proc_fd_parse == ProcSelfFdParse::kInvalid) {
    return SetFailureLocked(
        "TRACES_FOLDER /proc/self/fd authority must end in one exact "
        "non-negative decimal descriptor number",
        error);
  }

  int descriptor = -1;
  const char* open_operation = nullptr;
  if (proc_fd_parse == ProcSelfFdParse::kValid) {
    descriptor = DuplicateCloseOnExec(inherited_descriptor);
    open_operation =
        "fcntl(F_DUPFD_CLOEXEC, inherited TRACES_FOLDER descriptor)";
  } else {
    descriptor = OpenDirectoryNoFollow(directory_path.c_str());
    open_operation = "open(TRACES_FOLDER, O_DIRECTORY|O_NOFOLLOW)";
  }
  if (descriptor < 0) {
    const int saved_errno = errno;
    return SetFailureLocked(ErrnoMessage(open_operation, saved_errno), error);
  }

  struct stat observed {};
  if (!FstatChecked(descriptor, &observed)) {
    const int saved_errno = errno;
    std::string message =
        ErrnoMessage("fstat(retained TRACES_FOLDER descriptor)", saved_errno);
    CloseDescriptorLocked(&descriptor,
                          "close(retained TRACES_FOLDER descriptor)",
                          &message);
    return SetFailureLocked(message, error);
  }
  if (!S_ISDIR(observed.st_mode) || observed.st_nlink == 0) {
    std::ostringstream out;
    out << "opened TRACES_FOLDER authority is not a linked directory: mode="
        << static_cast<std::uintmax_t>(observed.st_mode)
        << " nlink=" << static_cast<std::uintmax_t>(observed.st_nlink);
    std::string message = out.str();
    CloseDescriptorLocked(&descriptor,
                          "close(retained TRACES_FOLDER descriptor)",
                          &message);
    return SetFailureLocked(message, error);
  }

  directory_descriptor_ = descriptor;
  directory_device_ = observed.st_dev;
  directory_inode_ = observed.st_ino;
  opened_ = true;
  return true;
}

bool PhaseRecordWriter::AdmitDeviceLocked(int device_id, std::string* error) {
  if (device_id < 0) {
    return SetFailureLocked(
        "phase-record producer device id must be non-negative", error);
  }
  if (admitted_device_id_ < 0) {
    admitted_device_id_ = device_id;
    return true;
  }
  if (admitted_device_id_ != device_id) {
    std::ostringstream out;
    out << "second GPU device rejected by single-device phase authority: "
        << "admitted_device_id=" << admitted_device_id_
        << " requested_device_id=" << device_id
        << "; canonical phase_records_<repeat>_<launch>.bin names are not "
           "device-qualified";
    return SetFailureLocked(out.str(), error);
  }
  return true;
}

bool PhaseRecordWriter::AdmitDevice(int device_id, std::string* error) {
  if (!CheckOwnerPidBeforeLock(error)) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!CheckUsableLocked(error)) return false;
  if (!AdmitDeviceLocked(device_id, error)) return false;

  std::string identity_error;
  if (!VerifyDirectoryIdentityLocked(directory_descriptor_,
                                     "retained TRACES_FOLDER descriptor",
                                     &identity_error)) {
    return SetFailureLocked(identity_error, error);
  }
  return true;
}

bool PhaseRecordWriter::BeginLaunch(int device_id,
                                    std::uint32_t launch_index,
                                    std::string* error) {
  if (!CheckOwnerPidBeforeLock(error)) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!CheckUsableLocked(error)) return false;
  if (!AdmitDeviceLocked(device_id, error)) return false;
  if (launch_active_) {
    std::ostringstream out;
    out << "overlapping phase-enabled launch rejected before managed launch "
           "identity overwrite: active_device_id="
        << active_launch_device_id_
        << " active_launch_index=" << active_launch_index_
        << " requested_device_id=" << device_id
        << " requested_launch_index=" << launch_index;
    return SetFailureLocked(out.str(), error);
  }
  std::string identity_error;
  if (!VerifyDirectoryIdentityLocked(directory_descriptor_,
                                     "retained TRACES_FOLDER descriptor",
                                     &identity_error)) {
    return SetFailureLocked(identity_error, error);
  }
  launch_active_ = true;
  active_launch_device_id_ = device_id;
  active_launch_index_ = launch_index;
  return true;
}

bool PhaseRecordWriter::EndLaunch(int device_id, std::uint32_t launch_index,
                                  std::string* error) {
  if (!CheckOwnerPidBeforeLock(error)) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!CheckUsableLocked(error)) return false;
  if (!launch_active_) {
    return SetFailureLocked(
        "phase-enabled launch end has no matching active launch", error);
  }
  if (active_launch_device_id_ != device_id ||
      active_launch_index_ != launch_index) {
    std::ostringstream out;
    out << "phase-enabled launch end identity mismatch: active_device_id="
        << active_launch_device_id_
        << " active_launch_index=" << active_launch_index_
        << " ending_device_id=" << device_id
        << " ending_launch_index=" << launch_index;
    return SetFailureLocked(out.str(), error);
  }
  std::string identity_error;
  if (!VerifyDirectoryIdentityLocked(directory_descriptor_,
                                     "retained TRACES_FOLDER descriptor",
                                     &identity_error)) {
    return SetFailureLocked(identity_error, error);
  }
  launch_active_ = false;
  active_launch_device_id_ = -1;
  active_launch_index_ = 0;
  return true;
}

int PhaseRecordWriter::AcquireOperationDirectoryLocked(std::string* error) {
  std::string identity_error;
  if (!VerifyDirectoryIdentityLocked(directory_descriptor_,
                                     "retained TRACES_FOLDER descriptor",
                                     &identity_error)) {
    SetFailureLocked(identity_error, error);
    return -1;
  }

  int operation_descriptor = DuplicateCloseOnExec(directory_descriptor_);
  if (operation_descriptor < 0) {
    const int saved_errno = errno;
    SetFailureLocked(
        ErrnoMessage("fcntl(F_DUPFD_CLOEXEC, TRACES_FOLDER)", saved_errno),
        error);
    return -1;
  }

  if (!VerifyDirectoryIdentityLocked(operation_descriptor,
                                     "operation TRACES_FOLDER descriptor",
                                     &identity_error)) {
    std::string message = identity_error;
    CloseDescriptorLocked(&operation_descriptor,
                          "close(operation TRACES_FOLDER descriptor)",
                          &message);
    SetFailureLocked(message, error);
    return -1;
  }
  return operation_descriptor;
}

ssize_t PhaseRecordWriter::WriteOnceLocked(int descriptor,
                                           const void* payload,
                                           std::size_t payload_size) {
#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
  if (write_hook_ != nullptr) {
    return write_hook_(descriptor, payload, payload_size, write_hook_opaque_);
  }
#endif
  return ::write(descriptor, payload, payload_size);
}

bool PhaseRecordWriter::FsyncDescriptorLocked(int descriptor,
                                              bool is_directory) {
#ifndef SIMBENCH_PHASE_RECORD_WRITER_TESTING
  (void)is_directory;
#endif
  unsigned interrupted = 0;
  for (;;) {
    int result = 0;
#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
    if (fsync_hook_ != nullptr) {
      result = fsync_hook_(descriptor, is_directory, fsync_hook_opaque_);
    } else {
      result = ::fsync(descriptor);
    }
#else
    result = ::fsync(descriptor);
#endif
    if (result == 0) return true;
    if (errno != EINTR || ++interrupted > kInterruptedRetryLimit) return false;
  }
}

void PhaseRecordWriter::InvokePostCreateHookLocked(int operation_directory,
                                                   int record_descriptor,
                                                   const char* leaf) {
#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
  if (post_create_hook_ != nullptr) {
    post_create_hook_(operation_directory, record_descriptor, leaf,
                      post_create_hook_opaque_);
  }
#else
  (void)operation_directory;
  (void)record_descriptor;
  (void)leaf;
#endif
}

bool PhaseRecordWriter::WriteRecordLocked(
    LeafKind leaf_kind, int device_id, std::uint32_t repeat_index,
    std::uint32_t launch_index, const void* payload, std::size_t payload_size,
    std::string* error) {
  if (!CheckUsableLocked(error)) return false;
  if (!AdmitDeviceLocked(device_id, error)) return false;
  if (launch_active_ &&
      (active_launch_device_id_ != device_id ||
       active_launch_index_ != launch_index)) {
    std::ostringstream out;
    out << "phase-record publication identity does not match the immutable "
           "active launch: active_device_id="
        << active_launch_device_id_
        << " active_launch_index=" << active_launch_index_
        << " publication_device_id=" << device_id
        << " publication_launch_index=" << launch_index;
    return SetFailureLocked(out.str(), error);
  }
  if (payload_size > 0 && payload == nullptr) {
    return SetFailureLocked(
        "phase-record payload is null with a non-zero byte count", error);
  }
  if (payload_size > static_cast<std::size_t>(SSIZE_MAX)) {
    return SetFailureLocked(
        "phase-record payload exceeds the bounded ssize_t publication range",
        error);
  }

  char leaf[96];
  const bool formatted =
      leaf_kind == LeafKind::kCanonical
          ? format_phase_record_leaf(leaf, sizeof(leaf), repeat_index,
                                     launch_index)
          : format_partial_phase_record_leaf(leaf, sizeof(leaf), repeat_index,
                                             launch_index);
  if (!formatted) {
    return SetFailureLocked(
        leaf_kind == LeafKind::kCanonical
            ? "canonical phase-record leaf exceeds its bounded name buffer"
            : "partial phase-record leaf exceeds its bounded name buffer",
        error);
  }

  int operation_directory = AcquireOperationDirectoryLocked(error);
  if (operation_directory < 0) return false;
  int record_descriptor = -1;

  auto fail_with_cleanup = [&](std::string message) {
    CloseDescriptorLocked(&record_descriptor, "close(phase-record leaf)",
                          &message);
    CloseDescriptorLocked(&operation_directory,
                          "close(operation TRACES_FOLDER descriptor)",
                          &message);
    return SetFailureLocked(message, error);
  };

  std::string identity_error;
  if (!VerifyDirectoryIdentityLocked(operation_directory,
                                     "operation TRACES_FOLDER descriptor",
                                     &identity_error)) {
    return fail_with_cleanup(identity_error);
  }

  record_descriptor = OpenLeafExclusive(operation_directory, leaf);
  if (record_descriptor < 0) {
    const int saved_errno = errno;
    std::ostringstream operation;
    operation << "openat(" << leaf << ", O_CREAT|O_EXCL|O_NOFOLLOW)";
    return fail_with_cleanup(
        ErrnoMessage(operation.str().c_str(), saved_errno));
  }

  // This test seam runs at the only point where a hostile actor can add a link
  // after exclusive creation but before the first descriptor/link verification.
  InvokePostCreateHookLocked(operation_directory, record_descriptor, leaf);

  // Revalidate the duplicated directory after openat. Even if another thread
  // closed/reused the private retained descriptor concurrently, this operation
  // descriptor pins the already-authenticated inode for the whole publication.
  if (!VerifyDirectoryIdentityLocked(operation_directory,
                                     "operation TRACES_FOLDER descriptor",
                                     &identity_error)) {
    return fail_with_cleanup(identity_error);
  }

  struct stat opened_record {};
  if (!FstatChecked(record_descriptor, &opened_record)) {
    const int saved_errno = errno;
    return fail_with_cleanup(
        ErrnoMessage("fstat(new phase-record leaf)", saved_errno));
  }
  std::string verification_error;
  if (!ValidateSingleLinkRegular(opened_record, nullptr, 0, false,
                                 "new phase-record descriptor",
                                 &verification_error)) {
    return fail_with_cleanup(verification_error);
  }

  if (!VerifyDirectoryIdentityLocked(operation_directory,
                                     "operation TRACES_FOLDER descriptor",
                                     &identity_error)) {
    return fail_with_cleanup(identity_error);
  }
  struct stat named_record {};
  if (!FstatatNoFollowChecked(operation_directory, leaf, &named_record)) {
    const int saved_errno = errno;
    return fail_with_cleanup(
        ErrnoMessage("fstatat(new phase-record leaf, AT_SYMLINK_NOFOLLOW)",
                     saved_errno));
  }
  if (!ValidateSingleLinkRegular(named_record, &opened_record, 0, false,
                                 "new phase-record locator",
                                 &verification_error)) {
    return fail_with_cleanup(verification_error);
  }

  const auto* bytes = static_cast<const unsigned char*>(payload);
  std::size_t written = 0;
  unsigned interrupted = 0;
  while (written < payload_size) {
    const std::size_t request =
        std::min(kWriteChunkBytes, payload_size - written);
    const ssize_t result =
        WriteOnceLocked(record_descriptor, bytes + written, request);
    if (result < 0) {
      const int saved_errno = errno;
      if (saved_errno == EINTR && ++interrupted <= kInterruptedRetryLimit) {
        continue;
      }
      return fail_with_cleanup(
          ErrnoMessage("write(phase-record leaf)", saved_errno));
    }
    if (result == 0) {
      return fail_with_cleanup(
          "write(phase-record leaf) made zero progress before the exact "
          "payload was published");
    }
    if (static_cast<std::size_t>(result) > request) {
      return fail_with_cleanup(
          "write(phase-record leaf) reported more bytes than requested");
    }
    interrupted = 0;
    written += static_cast<std::size_t>(result);
  }

  struct stat completed_record {};
  if (!FstatChecked(record_descriptor, &completed_record)) {
    const int saved_errno = errno;
    return fail_with_cleanup(
        ErrnoMessage("fstat(completed phase-record leaf)", saved_errno));
  }
  if (!ValidateSingleLinkRegular(completed_record, &opened_record, payload_size,
                                 true, "completed phase-record descriptor",
                                 &verification_error)) {
    return fail_with_cleanup(verification_error);
  }

  if (!FsyncDescriptorLocked(record_descriptor, false)) {
    const int saved_errno = errno;
    return fail_with_cleanup(
        ErrnoMessage("fsync(phase-record leaf)", saved_errno));
  }

  // Catch a link-count or identity mutation that raced the data fsync before
  // accepting close as the publication boundary.
  struct stat durable_record {};
  if (!FstatChecked(record_descriptor, &durable_record)) {
    const int saved_errno = errno;
    return fail_with_cleanup(
        ErrnoMessage("fstat(durable phase-record leaf)", saved_errno));
  }
  if (!ValidateSingleLinkRegular(durable_record, &opened_record, payload_size,
                                 true, "durable phase-record descriptor",
                                 &verification_error)) {
    return fail_with_cleanup(verification_error);
  }

  std::string close_error;
  if (!CloseDescriptorLocked(&record_descriptor, "close(phase-record leaf)",
                             &close_error)) {
    return fail_with_cleanup(close_error);
  }

  if (!VerifyDirectoryIdentityLocked(operation_directory,
                                     "operation TRACES_FOLDER descriptor",
                                     &identity_error)) {
    return fail_with_cleanup(identity_error);
  }
  struct stat closed_named_record {};
  if (!FstatatNoFollowChecked(operation_directory, leaf,
                              &closed_named_record)) {
    const int saved_errno = errno;
    return fail_with_cleanup(
        ErrnoMessage("fstatat(closed phase-record leaf, AT_SYMLINK_NOFOLLOW)",
                     saved_errno));
  }
  if (!ValidateSingleLinkRegular(closed_named_record, &opened_record,
                                 payload_size, true,
                                 "closed phase-record locator",
                                 &verification_error)) {
    return fail_with_cleanup(verification_error);
  }

  if (!VerifyDirectoryIdentityLocked(operation_directory,
                                     "operation TRACES_FOLDER descriptor",
                                     &identity_error)) {
    return fail_with_cleanup(identity_error);
  }
  if (!FsyncDescriptorLocked(operation_directory, true)) {
    const int saved_errno = errno;
    return fail_with_cleanup(
        ErrnoMessage("fsync(TRACES_FOLDER directory)", saved_errno));
  }

  // Final locator check after directory durability and immediately before the
  // operation descriptor is released.
  if (!VerifyDirectoryIdentityLocked(operation_directory,
                                     "operation TRACES_FOLDER descriptor",
                                     &identity_error)) {
    return fail_with_cleanup(identity_error);
  }
  struct stat final_named_record {};
  if (!FstatatNoFollowChecked(operation_directory, leaf, &final_named_record)) {
    const int saved_errno = errno;
    return fail_with_cleanup(
        ErrnoMessage("fstatat(final phase-record leaf, AT_SYMLINK_NOFOLLOW)",
                     saved_errno));
  }
  if (!ValidateSingleLinkRegular(final_named_record, &opened_record,
                                 payload_size, true,
                                 "final phase-record locator",
                                 &verification_error)) {
    return fail_with_cleanup(verification_error);
  }

  std::string operation_close_error;
  if (!CloseDescriptorLocked(&operation_directory,
                             "close(operation TRACES_FOLDER descriptor)",
                             &operation_close_error)) {
    return SetFailureLocked(operation_close_error, error);
  }
  return true;
}

bool PhaseRecordWriter::WriteRecord(int device_id,
                                    std::uint32_t repeat_index,
                                    std::uint32_t launch_index,
                                    const void* payload,
                                    std::size_t payload_size,
                                    std::string* error) {
  if (!CheckOwnerPidBeforeLock(error)) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  return WriteRecordLocked(LeafKind::kCanonical, device_id, repeat_index,
                           launch_index, payload, payload_size, error);
}

bool PhaseRecordWriter::WritePartialRecord(int device_id,
                                           std::uint32_t repeat_index,
                                           std::uint32_t launch_index,
                                           const void* payload,
                                           std::size_t payload_size,
                                           std::string* error) {
  if (!CheckOwnerPidBeforeLock(error)) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  return WriteRecordLocked(LeafKind::kPartial, device_id, repeat_index,
                           launch_index, payload, payload_size, error);
}

[[noreturn]] void FailPhaseRecordAuthority(const char* stage,
                                           const std::string& detail) {
  const char* safe_stage = stage == nullptr ? "unspecified operation" : stage;
#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
  if (g_fatal_hook != nullptr) {
    g_fatal_hook(safe_stage, detail, g_fatal_hook_opaque);
  }
#endif
  std::fprintf(stderr,
               "FATAL: phase-record %s failed; refusing to continue with "
               "incomplete or redirected phase evidence: %s\n",
               safe_stage, detail.c_str());
  std::fflush(stderr);
  ::_exit(EXIT_FAILURE);
}

void PublishCanonicalRecordOrFail(PhaseRecordWriter& writer, int device_id,
                                  std::uint32_t repeat_index,
                                  std::uint32_t launch_index,
                                  const void* payload,
                                  std::size_t payload_size,
                                  const char* stage) {
  std::string error;
  if (!writer.WriteRecord(device_id, repeat_index, launch_index, payload,
                          payload_size, &error)) {
    if (error.empty()) error = "canonical phase-record publication failed";
    FailPhaseRecordAuthority(stage, error);
  }
}

#ifdef SIMBENCH_PHASE_RECORD_WRITER_TESTING
void PhaseRecordWriter::SetWriteHookForTesting(WriteHook hook, void* opaque) {
  if (!CheckOwnerPidBeforeLock(nullptr)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  write_hook_ = hook;
  write_hook_opaque_ = opaque;
}

void PhaseRecordWriter::SetCurrentPidHookForTesting(CurrentPidHook hook,
                                                    void* opaque) {
  if (!CheckOwnerPidBeforeLock(nullptr)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  current_pid_hook_ = hook;
  current_pid_hook_opaque_ = opaque;
}

void PhaseRecordWriter::SetPostCreateHookForTesting(PostCreateHook hook,
                                                    void* opaque) {
  if (!CheckOwnerPidBeforeLock(nullptr)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  post_create_hook_ = hook;
  post_create_hook_opaque_ = opaque;
}

void PhaseRecordWriter::SetFsyncHookForTesting(FsyncHook hook, void* opaque) {
  if (!CheckOwnerPidBeforeLock(nullptr)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  fsync_hook_ = hook;
  fsync_hook_opaque_ = opaque;
}

void PhaseRecordWriter::SetCloseHookForTesting(CloseHook hook, void* opaque) {
  if (!CheckOwnerPidBeforeLock(nullptr)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  close_hook_ = hook;
  close_hook_opaque_ = opaque;
}

int PhaseRecordWriter::DirectoryDescriptorForTesting() const {
  if (!CheckOwnerPidBeforeLock(nullptr)) return -1;
  std::lock_guard<std::mutex> lock(mutex_);
  return directory_descriptor_;
}

void SetPhaseRecordFatalHookForTesting(PhaseRecordFatalHook hook,
                                       void* opaque) {
  g_fatal_hook = hook;
  g_fatal_hook_opaque = opaque;
}
#endif

}  // namespace simbench_nvbit
