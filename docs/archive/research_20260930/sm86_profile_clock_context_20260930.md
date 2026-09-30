# NCU local clock-control context (2026-09-30)

The prior full campaigns used a 100-ms `nvidia-smi` sampler around the NCU
process. That sampler is not launch-local: it can observe clocks before/after
the profiled kernel and miss the microsecond kernel interval. A new interleaved
single-kernel probe therefore requested NCU `--clock-control base` and `boost`
and collected NCU per-kernel frequency metrics alongside duration and GPC
cycles.

Artifact: `results/alpha/sm86_profile_clock_context_20260930/summary.json`.

| NCU request | SM/GPC frequency median | DRAM frequency median | duration median | GPC median |
|---|---:|---:|---:|---:|
| base | 1.546875 GHz | 10.435556 GHz | 9.600 us | 14,857 cycles |
| boost | 1.867973 GHz | 10.463878 GHz | 8.416 us | 15,748 cycles |

This is stronger endpoint-local clock evidence than the external sampler for
this kernel, but it is still a profiler-domain counter, not a synchronized
device-timer witness. It also does not prove that the full campaigns shared the
same launch-local regime. The result corrects the earlier wording: the broad
campaign sampler did not establish a base/boost regime, while the dedicated
NCU frequency metrics show that the requested controls can produce distinct
per-kernel frequencies in this probe.

No NCU target labels were used to fit a simulator model, and the default replay
is unchanged. No tests, hashing, or verification workflow was run.
