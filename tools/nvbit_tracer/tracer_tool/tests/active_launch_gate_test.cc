#include <cstdio>

#include "../active_launch_gate.h"

using nvbit_tracer::ActiveLaunchDecision;
using nvbit_tracer::ActiveLaunchGate;

static bool expect(bool value, const char* message) {
  if (value) return true;
  std::fprintf(stderr, "FAIL: %s\n", message);
  return false;
}

int main() {
  bool ok = true;

  ActiveLaunchGate exact;
  exact.Configure(false, 1);
  ok &= expect(exact.BeginLaunch() == ActiveLaunchDecision::kInactive,
               "launch before profiler start must be inactive");
  ok &= expect(exact.accepted_launches() == 0,
               "inactive launch must not consume accepted cardinality");
  exact.ProfilerStart();
  ok &= expect(exact.BeginLaunch() == ActiveLaunchDecision::kAccepted,
               "first profiler-bounded launch must be accepted");
  ok &= expect(exact.FinishAcceptedLaunch(),
               "accepted launch exit must pair with entry");
  exact.ProfilerStop();
  ok &= expect(exact.BeginLaunch() == ActiveLaunchDecision::kInactive,
               "launch after profiler stop must be inactive");
  ok &= expect(exact.ExactContractSatisfied(),
               "one completed accepted launch must satisfy exact-one contract");

  ActiveLaunchGate overflow;
  overflow.Configure(false, 1);
  overflow.ProfilerStart();
  ok &= expect(overflow.BeginLaunch() == ActiveLaunchDecision::kAccepted,
               "overflow setup first launch");
  ok &= expect(overflow.FinishAcceptedLaunch(), "overflow setup finish");
  ok &= expect(overflow.BeginLaunch() == ActiveLaunchDecision::kLimitReached,
               "second active launch must run outside capture");
  ok &= expect(overflow.accepted_launches() == 1,
               "limit-reached launch must never become accepted");
  ok &= expect(overflow.active_launches_skipped_after_limit() == 1,
               "limit-reached launch must be accounted");
  ok &= expect(overflow.ExactContractSatisfied(),
               "one accepted launch remains exact despite later skipped work");

  ActiveLaunchGate concurrent;
  concurrent.Configure(false, 1);
  concurrent.ProfilerStart();
  ok &= expect(concurrent.BeginLaunch() == ActiveLaunchDecision::kAccepted,
               "concurrent setup first launch");
  ok &= expect(concurrent.BeginLaunch() ==
                   ActiveLaunchDecision::kConcurrentLimitViolation,
               "overlapping active launch must fail instead of truncating target");
  ok &= expect(!concurrent.ExactContractSatisfied(),
               "concurrent cap violation invalidates exact capture");
  ok &= expect(concurrent.FinishAcceptedLaunch(), "concurrent setup finish");

  ActiveLaunchGate zero;
  zero.Configure(false, 1);
  ok &= expect(!zero.ExactContractSatisfied(),
               "zero accepted launches must not vacuously satisfy exact-one");

  ActiveLaunchGate legacy;
  legacy.Configure(true, 0);
  ok &= expect(legacy.BeginLaunch() == ActiveLaunchDecision::kAccepted,
               "disabled limit preserves active-from-start behavior");
  ok &= expect(legacy.FinishAcceptedLaunch(), "legacy first finish");
  ok &= expect(legacy.BeginLaunch() == ActiveLaunchDecision::kAccepted,
               "disabled limit admits multiple launches");
  ok &= expect(legacy.FinishAcceptedLaunch(), "legacy second finish");
  ok &= expect(legacy.ExactContractSatisfied(),
               "disabled limit has no exact-cardinality requirement");

  if (!ok) return 1;
  std::puts("active_launch_gate_test: PASS");
  return 0;
}
