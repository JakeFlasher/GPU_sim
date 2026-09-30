# Same-body raw-QMD versus NCU crosswalk

The retained independent Mesa NAK `shader_2` body was run through two launch
paths on the fixed RTX 3090 Ti (1350/10501 MHz):

1. direct raw RM Ampere QMD V03 with four-word nanosecond reports and WFI;
2. the existing CUDA-ELF bridge and CUDA Driver API under Nsight Compute.

Both paths used the same 14-instruction body and the same static operation
population (10 aggregate ALU instructions and 3 aggregate LSU instructions),
but they were **not a literal launch-equivalence pair**: the raw QMD used the
NAK QMD's local size `8x1x1`, whereas the CUDA-ELF launcher used a
`32x1x1` block (one warp in both cases), and the NCU run requested
`--cache-control all` while the raw path did not perform that profiler cache
intervention. Raw QMD
repeats (32 launches) had GPU timestamp deltas with median **4,048 ns** and
range **3,072--5,120 ns**.

Three NCU profiles of the bridged body (`--cache-control all`,
`--clock-control none`) had median duration **2,496 ns**, median **3,310 GPC
cycles**, and stable 14 / 10 ALU / 3 LSU counts.

The raw-QMD/NCU duration ratio is about **1.62** for this body-matched,
launch-confounded comparison. It is not a conversion factor: raw QMD excludes
CUDA-ELF loading, profiler replay, and NCU's endpoint envelope, while NCU
includes its own WaitForIdle/replay semantics; block geometry and cache
intervention also differ. The comparison therefore establishes a bounded
endpoint/launch-envelope discrepancy, not a clean causal separation of the
timer domains. A future closure experiment must hold block geometry, cache
state, metadata, and body constant while changing only the endpoint.

## Block-size control

As a bounded follow-up, the retained CUDA-ELF body was launched with a
`8x1x1` block to match the raw QMD local size. One NCU profile (`--cache-control
all --clock-control none`) returned **2,496 ns**, **3,296 GPC cycles**, and the
same 14/10-ALU/3-LSU population; the earlier `32x1x1` profile median was
2,496 ns / 3,310 cycles. This makes block size an unlikely explanation for
the 1.62 ratio, but it does not remove the remaining cache-intervention,
metadata/QMD, and endpoint-envelope differences. The one-shot control is
supporting performance evidence, not a unified-timer proof.

Artifacts: `results/alpha/sm86_nak_direct_qmd_goal_20260930/timestamp_shader2/`.
