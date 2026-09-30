# NCU-local versus external clock boundary (2026-09-30)

A bounded single-kernel probe ran the existing HMMA binary under
`ncu --clock-control boost` while an independent `nvidia-smi --loop-ms=5`
sampler collected utilization/clock samples.

* External sampler: 155 samples, all **1350 MHz** SM clock, no nonzero GPU
  utilization samples observed at the 5-ms sample instants.
* NCU kernel-local metrics: approximately **1.870 GHz** SM/GPC frequency for
  the profiled kernel.

This is a direct current-host demonstration that NCU-local frequency metrics
and asynchronous host clock sampling are distinct observations. It does not
prove either clock is physically wrong; it prevents using NCU frequency as an
external regime witness. The regime router therefore requires an independent
active-clock observation and otherwise selects its neutral route.

Artifacts: `results/alpha/sm86_active_clock_probe_5ms_20260930/`.

For longer campaigns, `tools/sm86_clock_sample_window_observation.py` now
retains both median and high-burst fraction. This avoids treating idle-diluted
host samples as a launch-local clock while still allowing an explicit
high-regime hypothesis when the burst fraction is substantial.
