#ifndef NVBIT_TRACER_ACTIVE_LAUNCH_GATE_H_
#define NVBIT_TRACER_ACTIVE_LAUNCH_GATE_H_

#include <cstdint>

namespace nvbit_tracer {

enum class ActiveLaunchDecision {
  kInactive,
  kAccepted,
  kLimitReached,
  kConcurrentLimitViolation,
};

// Host-only state machine for profiler-delimited trace capture.  The CUDA
// callback owns the synchronization; keeping this class free of CUDA/NVBit
// types makes the selection contract directly unit-testable.
class ActiveLaunchGate {
 public:
  void Configure(bool active_from_start, uint64_t max_accepted_launches) {
    region_active_ = active_from_start;
    max_accepted_launches_ = max_accepted_launches;
    accepted_launches_ = 0;
    active_launch_attempts_ = 0;
    in_flight_accepted_launches_ = 0;
    active_launches_skipped_after_limit_ = 0;
    concurrent_limit_violations_ = 0;
  }

  void ProfilerStart() { region_active_ = true; }
  void ProfilerStop() { region_active_ = false; }

  ActiveLaunchDecision BeginLaunch() {
    if (!region_active_) return ActiveLaunchDecision::kInactive;
    ++active_launch_attempts_;
    if (max_accepted_launches_ != 0 &&
        accepted_launches_ >= max_accepted_launches_) {
      if (in_flight_accepted_launches_ != 0) {
        ++concurrent_limit_violations_;
        return ActiveLaunchDecision::kConcurrentLimitViolation;
      }
      ++active_launches_skipped_after_limit_;
      return ActiveLaunchDecision::kLimitReached;
    }
    ++accepted_launches_;
    ++in_flight_accepted_launches_;
    return ActiveLaunchDecision::kAccepted;
  }

  bool FinishAcceptedLaunch() {
    if (in_flight_accepted_launches_ == 0) return false;
    --in_flight_accepted_launches_;
    return true;
  }

  bool region_active() const { return region_active_; }
  bool strict_limit_enabled() const { return max_accepted_launches_ != 0; }
  uint64_t max_accepted_launches() const { return max_accepted_launches_; }
  uint64_t accepted_launches() const { return accepted_launches_; }
  uint64_t active_launch_attempts() const { return active_launch_attempts_; }
  uint64_t in_flight_accepted_launches() const {
    return in_flight_accepted_launches_;
  }
  uint64_t active_launches_skipped_after_limit() const {
    return active_launches_skipped_after_limit_;
  }
  uint64_t concurrent_limit_violations() const {
    return concurrent_limit_violations_;
  }

  // An opt-in maximum is also an exact-cardinality contract at finalization:
  // zero accepted launches must not vacuously produce a usable capture.
  bool ExactContractSatisfied() const {
    if (!strict_limit_enabled()) return true;
    return concurrent_limit_violations_ == 0 &&
           in_flight_accepted_launches_ == 0 &&
           accepted_launches_ == max_accepted_launches_;
  }

 private:
  bool region_active_ = true;
  uint64_t max_accepted_launches_ = 0;
  uint64_t accepted_launches_ = 0;
  uint64_t active_launch_attempts_ = 0;
  uint64_t in_flight_accepted_launches_ = 0;
  uint64_t active_launches_skipped_after_limit_ = 0;
  uint64_t concurrent_limit_violations_ = 0;
};

}  // namespace nvbit_tracer

#endif  // NVBIT_TRACER_ACTIVE_LAUNCH_GATE_H_
