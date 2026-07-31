#ifndef SIMBENCH_PHASE_RECORD_WRITER_TESTING
#error "phase_record_writer_test requires SIMBENCH_PHASE_RECORD_WRITER_TESTING"
#endif

#include "../phase_record_path.h"
#include "../phase_record_writer.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    char pattern[] = "/tmp/simbench-phase-record-writer-XXXXXX";
    char* created = ::mkdtemp(pattern);
    if (created == nullptr) {
      throw std::system_error(errno, std::generic_category(), "mkdtemp");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

std::string CanonicalLeaf(std::uint32_t repeat_index,
                          std::uint32_t launch_index) {
  char leaf[96];
  Require(simbench_nvbit::format_phase_record_leaf(
              leaf, sizeof(leaf), repeat_index, launch_index),
          "format_phase_record_leaf unexpectedly failed");
  return leaf;
}

std::string PartialLeaf(std::uint32_t repeat_index,
                        std::uint32_t launch_index) {
  char leaf[96];
  Require(simbench_nvbit::format_partial_phase_record_leaf(
              leaf, sizeof(leaf), repeat_index, launch_index),
          "format_partial_phase_record_leaf unexpectedly failed");
  return leaf;
}

void WriteFixture(const std::filesystem::path& path,
                  const std::string& payload) {
  std::ofstream output(path, std::ios::out | std::ios::binary | std::ios::trunc);
  Require(static_cast<bool>(output), "failed to open fixture " + path.string());
  output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  output.close();
  Require(!output.fail(), "failed to write fixture " + path.string());
}

std::string ReadFixture(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  Require(static_cast<bool>(input), "failed to open result " + path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

struct stat Lstat(const std::filesystem::path& path) {
  struct stat result {};
  Require(::lstat(path.c_str(), &result) == 0,
          "lstat failed for " + path.string());
  return result;
}

void TestSuccessfulCanonicalPublication() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);

  const std::string payload("phase\0bytes", 11);
  Require(writer.WriteRecord(3, 12, 34, payload.data(), payload.size(), &error),
          error);

  const std::string leaf = CanonicalLeaf(12, 34);
  Require(leaf == "phase_records_12_34.bin",
          "canonical phase-record filename changed");
  const auto published = directory.path() / leaf;
  const struct stat observed = Lstat(published);
  Require(S_ISREG(observed.st_mode), "published phase leaf is not regular");
  Require(observed.st_nlink == 1,
          "published phase leaf does not have exactly one link");
  Require(ReadFixture(published) == payload,
          "published phase payload differs from exact input bytes");
}

void TestProcSelfFdAuthorityIsPrivatelyRetained() {
  TemporaryDirectory directory;
  int inherited_descriptor =
      ::open(directory.path().c_str(),
             O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  Require(inherited_descriptor >= 0,
          "failed to open inherited TRACES_FOLDER authority");

  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  const std::string proc_path =
      "/proc/self/fd/" + std::to_string(inherited_descriptor);
  Require(writer.OpenDirectory(proc_path, &error), error);
  Require(::close(inherited_descriptor) == 0,
          "failed to close caller-owned inherited directory descriptor");

  const std::string payload = "privately-retained-authority";
  Require(writer.WriteRecord(4, 13, 35, payload.data(), payload.size(), &error),
          error);
  Require(ReadFixture(directory.path() / CanonicalLeaf(13, 35)) == payload,
          "private duplicate did not retain inherited directory authority");
}

void TestExistingRegularFileRefused() {
  TemporaryDirectory directory;
  const auto leaf = directory.path() / CanonicalLeaf(1, 2);
  WriteFixture(leaf, "retained-regular");

  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  const std::string replacement = "replacement";
  Require(!writer.WriteRecord(0, 1, 2, replacement.data(),
                              replacement.size(), &error),
          "existing regular phase leaf was overwritten");
  Require(error.find("O_CREAT|O_EXCL|O_NOFOLLOW") != std::string::npos,
          "regular-file refusal did not report exclusive open");
  Require(ReadFixture(leaf) == "retained-regular",
          "existing regular phase leaf bytes changed");
}

void TestSymlinkRefused() {
  TemporaryDirectory directory;
  const auto victim = directory.path() / "victim.bin";
  const auto leaf = directory.path() / CanonicalLeaf(2, 3);
  WriteFixture(victim, "victim-retained");
  Require(::symlink("victim.bin", leaf.c_str()) == 0,
          "failed to create hostile symlink");

  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  const std::string payload = "must-not-follow";
  Require(!writer.WriteRecord(0, 2, 3, payload.data(), payload.size(), &error),
          "symlink phase leaf was followed");
  Require(S_ISLNK(Lstat(leaf).st_mode), "hostile symlink was replaced");
  Require(ReadFixture(victim) == "victim-retained",
          "symlink victim bytes changed");
}

void TestHardlinkRefused() {
  TemporaryDirectory directory;
  const auto source = directory.path() / "hardlink-source.bin";
  const auto leaf = directory.path() / CanonicalLeaf(3, 4);
  WriteFixture(source, "hardlink-retained");
  Require(::link(source.c_str(), leaf.c_str()) == 0,
          "failed to create hostile hardlink");

  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  const std::string payload = "must-not-overwrite";
  Require(!writer.WriteRecord(0, 3, 4, payload.data(), payload.size(), &error),
          "hardlink phase leaf was overwritten");
  const struct stat source_stat = Lstat(source);
  const struct stat leaf_stat = Lstat(leaf);
  Require(source_stat.st_dev == leaf_stat.st_dev &&
              source_stat.st_ino == leaf_stat.st_ino &&
              source_stat.st_nlink == 2,
          "hostile hardlink identity changed");
  Require(ReadFixture(source) == "hardlink-retained",
          "hardlink source bytes changed");
}

void TestFifoRefusedWithoutBlocking() {
  TemporaryDirectory directory;
  const auto leaf = directory.path() / CanonicalLeaf(4, 5);
  Require(::mkfifo(leaf.c_str(), S_IRUSR | S_IWUSR) == 0,
          "failed to create hostile FIFO");

  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  const std::string payload = "must-not-open-fifo";
  Require(!writer.WriteRecord(0, 4, 5, payload.data(), payload.size(), &error),
          "FIFO phase leaf was opened");
  Require(S_ISFIFO(Lstat(leaf).st_mode), "hostile FIFO was replaced");
}

struct ShortWriteFailureState {
  int calls = 0;
};

ssize_t ShortWriteThenFail(int descriptor, const void* payload,
                           std::size_t payload_size, void* opaque) {
  auto* state = static_cast<ShortWriteFailureState*>(opaque);
  ++state->calls;
  if (state->calls == 1) {
    const std::size_t short_size = std::min<std::size_t>(2, payload_size);
    return ::write(descriptor, payload, short_size);
  }
  errno = ENOSPC;
  return -1;
}

void TestInjectedShortWriteFailureRetainsEvidence() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  ShortWriteFailureState state;
  writer.SetWriteHookForTesting(&ShortWriteThenFail, &state);

  const std::string payload = "abcdef";
  Require(!writer.WriteRecord(0, 5, 6, payload.data(), payload.size(), &error),
          "injected short-write failure was accepted");
  Require(state.calls == 2,
          "short-write regression did not reach the injected failure");
  Require(error.find("write(phase-record leaf)") != std::string::npos &&
              error.find("errno=" + std::to_string(ENOSPC)) !=
                  std::string::npos,
          "short-write failure did not preserve the ENOSPC diagnostic");

  const auto partial = directory.path() / CanonicalLeaf(5, 6);
  Require(S_ISREG(Lstat(partial).st_mode),
          "partial publication evidence was removed");
  Require(ReadFixture(partial) == "ab",
          "partial publication evidence does not preserve exact written bytes");
}

void TestRetainedDescriptorReuseRejected() {
  TemporaryDirectory authority;
  TemporaryDirectory replacement;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(authority.path().string(), &error), error);

  const int authority_descriptor = writer.DirectoryDescriptorForTesting();
  Require(authority_descriptor >= 3,
          "retained authority descriptor is unexpectedly a stdio descriptor");
  Require(::close(authority_descriptor) == 0,
          "failed to close retained authority descriptor for hostile test");

  int replacement_descriptor =
      ::open(replacement.path().c_str(),
             O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  Require(replacement_descriptor >= 0,
          "failed to open replacement directory for descriptor reuse test");
  if (replacement_descriptor != authority_descriptor) {
    Require(::dup2(replacement_descriptor, authority_descriptor) ==
                authority_descriptor,
            "failed to reuse retained authority descriptor number");
    Require(::close(replacement_descriptor) == 0,
            "failed to close extra replacement descriptor");
  }
  // The writer owns the reused descriptor number from here; its destructor
  // closes it. The publication must reject the replacement before openat.

  const std::string payload = "must-not-redirect";
  Require(!writer.WriteRecord(0, 6, 7, payload.data(), payload.size(), &error),
          "reused directory descriptor redirected phase publication");
  Require(error.find("no longer names the retained TRACES_FOLDER authority") !=
              std::string::npos,
          "descriptor-reuse rejection did not report identity mismatch");
  Require(!std::filesystem::exists(authority.path() / CanonicalLeaf(6, 7)),
          "descriptor-reuse test wrote into original authority unexpectedly");
  Require(!std::filesystem::exists(replacement.path() / CanonicalLeaf(6, 7)),
          "descriptor-reuse test redirected bytes into replacement directory");
}

void TestSecondDeviceRejectedBeforeFilenameCollision() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);

  const std::string payload = "device-zero";
  Require(writer.WriteRecord(0, 7, 1, payload.data(), payload.size(), &error),
          error);
  const std::string second_payload = "device-one";
  Require(!writer.WriteRecord(1, 7, 2, second_payload.data(),
                              second_payload.size(), &error),
          "second GPU device entered unqualified phase filename namespace");
  Require(error.find("second GPU device rejected") != std::string::npos &&
              error.find("admitted_device_id=0") != std::string::npos &&
              error.find("requested_device_id=1") != std::string::npos,
          "second-device rejection omitted device identities");
  Require(ReadFixture(directory.path() / CanonicalLeaf(7, 1)) == payload,
          "first device's durable phase leaf changed");
  Require(!std::filesystem::exists(directory.path() / CanonicalLeaf(7, 2)),
          "second device created a colliding phase leaf");
}

void TestPartialPublicationUsesNoncanonicalNamespace() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  Require(writer.BeginLaunch(0, 81, &error), error);

  const std::string payload("partial\0phase", 13);
  Require(writer.WritePartialRecord(0, 8, 81, payload.data(), payload.size(),
                                    &error),
          error);
  Require(writer.EndLaunch(0, 81, &error), error);

  const std::string leaf = PartialLeaf(8, 81);
  Require(leaf == ".partial-phase-records_8_81.bin",
          "partial phase-record filename changed");
  Require(ReadFixture(directory.path() / leaf) == payload,
          "partial evidence did not retain exact bytes");
  Require(!std::filesystem::exists(directory.path() / CanonicalLeaf(8, 81)),
          "partial publication created a canonical phase-record leaf");
}

void TestOverlappingSameDeviceLaunchRejected() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  Require(writer.BeginLaunch(0, 91, &error), error);
  Require(!writer.BeginLaunch(0, 92, &error),
          "overlapping same-device launch was admitted");
  Require(error.find("overlapping phase-enabled launch rejected") !=
                  std::string::npos &&
              error.find("active_launch_index=91") != std::string::npos &&
              error.find("requested_launch_index=92") != std::string::npos,
          "overlap rejection omitted immutable launch identities");
}

struct OneShotPidMismatchState {
  int calls = 0;
  pid_t owner_pid = -1;
};

pid_t MismatchPidOnce(void* opaque) {
  auto* state = static_cast<OneShotPidMismatchState*>(opaque);
  ++state->calls;
  if (state->calls == 1) return state->owner_pid + 1;
  return ::getpid();
}

void TestOwnerPidMismatchRejectedBeforeWriterStateMutation() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);

  OneShotPidMismatchState state;
  state.owner_pid = ::getpid();
  writer.SetCurrentPidHookForTesting(&MismatchPidOnce, &state);
  Require(!writer.AdmitDevice(0, &error),
          "post-fork owner-PID mismatch was accepted");
  Require(state.calls == 1 &&
              error.find("rejecting inherited post-fork state before acquiring "
                         "its mutex") != std::string::npos,
          "owner-PID rejection did not occur at the pre-lock gate");

  // The PID mismatch is deliberately non-poisoning: it represents a child
  // process rejecting inherited state, not a mutation of the parent's writer.
  Require(writer.AdmitDevice(0, &error),
          "pre-lock PID rejection unexpectedly mutated writer state: " + error);
  writer.SetCurrentPidHookForTesting(nullptr, nullptr);
}

pid_t AlwaysMismatchedPid(void* opaque) {
  const auto* state = static_cast<const OneShotPidMismatchState*>(opaque);
  return state->owner_pid + 1;
}

void TestPostForkDestructorAvoidsInheritedMutexAndDescriptor() {
  TemporaryDirectory directory;
  auto* writer = new simbench_nvbit::PhaseRecordWriter;
  std::string error;
  Require(writer->OpenDirectory(directory.path().string(), &error), error);
  const int inherited_descriptor = writer->DirectoryDescriptorForTesting();
  Require(inherited_descriptor >= 0,
          "destructor PID regression did not retain a directory descriptor");

  OneShotPidMismatchState state;
  state.owner_pid = ::getpid();
  writer->SetCurrentPidHookForTesting(&AlwaysMismatchedPid, &state);
  delete writer;

  struct stat observed {};
  Require(::fstat(inherited_descriptor, &observed) == 0 &&
              S_ISDIR(observed.st_mode),
          "post-fork destructor touched its inherited descriptor");
  Require(::close(inherited_descriptor) == 0,
          "failed to close descriptor intentionally retained by PID guard");
}

struct PostCreateHardlinkState {
  int calls = 0;
  int result = -1;
  int saved_errno = 0;
};

void AddPostCreateHardlink(int operation_directory, int record_descriptor,
                           const char* leaf, void* opaque) {
  (void)record_descriptor;
  auto* state = static_cast<PostCreateHardlinkState*>(opaque);
  ++state->calls;
  state->result = ::linkat(operation_directory, leaf, operation_directory,
                           "post-create-hardlink.bin", 0);
  state->saved_errno = state->result == 0 ? 0 : errno;
}

void TestPostCreateHardlinkFailsSingleLinkVerification() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  PostCreateHardlinkState state;
  writer.SetPostCreateHookForTesting(&AddPostCreateHardlink, &state);

  const std::string payload = "must-not-authorize";
  Require(!writer.WriteRecord(0, 9, 101, payload.data(), payload.size(), &error),
          "post-create hardlink race was accepted");
  Require(state.calls == 1 && state.result == 0,
          "post-create hook failed to install hardlink: errno=" +
              std::to_string(state.saved_errno));
  Require(error.find("single-link regular-file verification") !=
              std::string::npos,
          "post-create hardlink did not reach single-link verification");

  const struct stat canonical =
      Lstat(directory.path() / CanonicalLeaf(9, 101));
  const struct stat hardlink =
      Lstat(directory.path() / "post-create-hardlink.bin");
  Require(canonical.st_dev == hardlink.st_dev &&
              canonical.st_ino == hardlink.st_ino &&
              canonical.st_nlink == 2,
          "failed post-create hardlink evidence was not retained exactly");
}

struct FsyncFailureState {
  bool fail_directory = false;
  int leaf_calls = 0;
  int directory_calls = 0;
};

int FailSelectedFsync(int descriptor, bool is_directory, void* opaque) {
  auto* state = static_cast<FsyncFailureState*>(opaque);
  if (is_directory) {
    ++state->directory_calls;
  } else {
    ++state->leaf_calls;
  }
  if (is_directory == state->fail_directory) {
    errno = EIO;
    return -1;
  }
  return ::fsync(descriptor);
}

void TestLeafFsyncFailureRetainsBytes() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  FsyncFailureState state;
  state.fail_directory = false;
  writer.SetFsyncHookForTesting(&FailSelectedFsync, &state);

  const std::string payload = "leaf-fsync-evidence";
  Require(!writer.WriteRecord(0, 10, 111, payload.data(), payload.size(),
                              &error),
          "injected leaf fsync failure was accepted");
  Require(state.leaf_calls == 1 && state.directory_calls == 0,
          "leaf fsync injector reached the wrong durability boundary");
  Require(error.find("fsync(phase-record leaf)") != std::string::npos &&
              error.find("errno=" + std::to_string(EIO)) != std::string::npos,
          "leaf fsync failure lost its primary diagnostic");
  Require(ReadFixture(directory.path() / CanonicalLeaf(10, 111)) == payload,
          "leaf fsync failure discarded publication bytes");
}

void TestDirectoryFsyncFailureRetainsLeaf() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  FsyncFailureState state;
  state.fail_directory = true;
  writer.SetFsyncHookForTesting(&FailSelectedFsync, &state);

  const std::string payload = "directory-fsync-evidence";
  Require(!writer.WriteRecord(0, 11, 121, payload.data(), payload.size(),
                              &error),
          "injected directory fsync failure was accepted");
  Require(state.leaf_calls == 1 && state.directory_calls == 1,
          "directory fsync injector did not follow durable leaf fsync");
  Require(error.find("fsync(TRACES_FOLDER directory)") != std::string::npos &&
              error.find("errno=" + std::to_string(EIO)) != std::string::npos,
          "directory fsync failure lost its primary diagnostic");
  Require(ReadFixture(directory.path() / CanonicalLeaf(11, 121)) == payload,
          "directory fsync failure discarded the durable leaf");
}

ssize_t FailWriteWithEnospc(int descriptor, const void* payload,
                            std::size_t payload_size, void* opaque) {
  (void)descriptor;
  (void)payload;
  (void)payload_size;
  (void)opaque;
  errno = ENOSPC;
  return -1;
}

struct CloseFailureState {
  bool fail_leaf = false;
  bool fail_directory = false;
  int leaf_calls = 0;
  int directory_calls = 0;
};

int CloseThenReportSelectedFailure(int descriptor, const char* role,
                                   void* opaque) {
  auto* state = static_cast<CloseFailureState*>(opaque);
  const std::string role_string(role);
  const bool is_leaf = role_string == "close(phase-record leaf)";
  const bool is_directory =
      role_string == "close(operation TRACES_FOLDER descriptor)";
  if (is_leaf) ++state->leaf_calls;
  if (is_directory) ++state->directory_calls;

  const int actual_result = ::close(descriptor);
  if (actual_result != 0) return actual_result;
  if (is_leaf && state->fail_leaf) {
    errno = EIO;
    return -1;
  }
  if (is_directory && state->fail_directory) {
    errno = EBUSY;
    return -1;
  }
  return 0;
}

void TestWritePrimaryPrecedesCloseCleanupFailures() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  writer.SetWriteHookForTesting(&FailWriteWithEnospc, nullptr);
  CloseFailureState state;
  state.fail_leaf = true;
  state.fail_directory = true;
  writer.SetCloseHookForTesting(&CloseThenReportSelectedFailure, &state);

  const std::string payload = "never-written";
  Require(!writer.WriteRecord(0, 12, 131, payload.data(), payload.size(),
                              &error),
          "write and cleanup failures were accepted");
  const std::size_t write_pos = error.find("write(phase-record leaf)");
  const std::size_t leaf_close_pos = error.find("close(phase-record leaf)");
  const std::size_t directory_close_pos =
      error.find("close(operation TRACES_FOLDER descriptor)");
  Require(write_pos != std::string::npos &&
              leaf_close_pos != std::string::npos &&
              directory_close_pos != std::string::npos &&
              write_pos < leaf_close_pos &&
              leaf_close_pos < directory_close_pos,
          "cleanup failures replaced or reordered the primary write failure: " +
              error);
  Require(state.leaf_calls == 1 && state.directory_calls == 1,
          "write failure did not execute both descriptor cleanups");
}

void TestLeafClosePrimaryStillCleansOperationDirectory() {
  TemporaryDirectory directory;
  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  CloseFailureState state;
  state.fail_leaf = true;
  state.fail_directory = true;
  writer.SetCloseHookForTesting(&CloseThenReportSelectedFailure, &state);

  const std::string payload = "close-failure-evidence";
  Require(!writer.WriteRecord(0, 13, 141, payload.data(), payload.size(),
                              &error),
          "injected leaf close failure was accepted");
  const std::size_t leaf_close_pos = error.find("close(phase-record leaf)");
  const std::size_t directory_close_pos =
      error.find("close(operation TRACES_FOLDER descriptor)");
  Require(leaf_close_pos == 0 && directory_close_pos != std::string::npos &&
              leaf_close_pos < directory_close_pos,
          "leaf close did not remain primary over directory cleanup: " + error);
  Require(state.leaf_calls == 1 && state.directory_calls == 1,
          "leaf close failure skipped operation-directory cleanup");
  Require(ReadFixture(directory.path() / CanonicalLeaf(13, 141)) == payload,
          "close failure discarded exact leaf bytes");
}

struct FatalIntercept {};

struct FatalState {
  int calls = 0;
  std::string stage;
  std::string detail;
};

void ThrowOnFatal(const char* stage, const std::string& detail, void* opaque) {
  auto* state = static_cast<FatalState*>(opaque);
  ++state->calls;
  state->stage = stage;
  state->detail = detail;
  throw FatalIntercept{};
}

void TestCanonicalPublicationFalseInvokesFatalIntegration() {
  TemporaryDirectory directory;
  const auto leaf = directory.path() / CanonicalLeaf(14, 151);
  WriteFixture(leaf, "existing-authority");

  simbench_nvbit::PhaseRecordWriter writer;
  std::string error;
  Require(writer.OpenDirectory(directory.path().string(), &error), error);
  FatalState state;
  simbench_nvbit::SetPhaseRecordFatalHookForTesting(&ThrowOnFatal, &state);
  bool intercepted = false;
  const std::string payload = "must-not-replace";
  try {
    simbench_nvbit::PublishCanonicalRecordOrFail(
        writer, 0, 14, 151, payload.data(), payload.size(),
        "host fatal integration regression");
  } catch (const FatalIntercept&) {
    intercepted = true;
  } catch (...) {
    simbench_nvbit::SetPhaseRecordFatalHookForTesting(nullptr, nullptr);
    throw;
  }
  simbench_nvbit::SetPhaseRecordFatalHookForTesting(nullptr, nullptr);

  Require(intercepted && state.calls == 1,
          "false WriteRecord did not invoke the shared fatal path");
  Require(state.stage == "host fatal integration regression" &&
              state.detail.find("O_CREAT|O_EXCL|O_NOFOLLOW") !=
                  std::string::npos,
          "fatal integration lost publication stage or primary detail");
  Require(ReadFixture(leaf) == "existing-authority",
          "fatal integration altered existing canonical evidence");
}

}  // namespace

int main() {
  try {
    TestSuccessfulCanonicalPublication();
    TestProcSelfFdAuthorityIsPrivatelyRetained();
    TestExistingRegularFileRefused();
    TestSymlinkRefused();
    TestHardlinkRefused();
    TestFifoRefusedWithoutBlocking();
    TestInjectedShortWriteFailureRetainsEvidence();
    TestRetainedDescriptorReuseRejected();
    TestSecondDeviceRejectedBeforeFilenameCollision();
    TestPartialPublicationUsesNoncanonicalNamespace();
    TestOverlappingSameDeviceLaunchRejected();
    TestOwnerPidMismatchRejectedBeforeWriterStateMutation();
    TestPostForkDestructorAvoidsInheritedMutexAndDescriptor();
    TestPostCreateHardlinkFailsSingleLinkVerification();
    TestLeafFsyncFailureRetainsBytes();
    TestDirectoryFsyncFailureRetainsLeaf();
    TestWritePrimaryPrecedesCloseCleanupFailures();
    TestLeafClosePrimaryStillCleansOperationDirectory();
    TestCanonicalPublicationFalseInvokesFatalIntegration();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "phase_record_writer_test: %s\n", error.what());
    return 1;
  }
  std::printf("phase_record_writer_test: all assertions passed\n");
  return 0;
}
