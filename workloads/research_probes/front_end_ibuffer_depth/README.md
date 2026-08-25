# sm_86 branch-free IBuffer-depth probe

This successor to `fixed_latency_front_end_footprint` reconstructs the
independent-instruction experiment used by Huerta et al. to infer a three-entry
per-warp Instruction Buffer. The measured branch-free sequence contains eight
groups of one CS2R plus three NOPs, giving the CLOCK resource enough separation
to keep the warp otherwise eligible. A ninth group keeps compiler epilogue work
out of the first 32 instructions; its timestamp is returned only as a boundary
check. Every warp returns all nine CLOCK values, directly exposing its issue
timeline.

Patched calibration variants use an encoded stall-1 zero-bubble baseline and
place either stall 4 or yield with stall 1 at each of the first four SASS
instructions. This maps the paper's logical “second instruction” onto the
CS2R/NOP group before selecting the final campaign pair. Seven same-context
launches expose cold/warm timeline stability; NCU profiles the first launch
with direct `no_instruction` and branch-resolving counters.

The completed matrix contains 32 cells, 224 correctness executions, 144 NCU
profiles, seven SourceCounters profiles, 32 NVBit captures, and 21,358 replay
records. Both CLOCK objectives select capacity two after in-flight fetches
reserve entries, but incompatible instruction-cache fits and under-15% exact
event-order accuracy reject the complete candidate. No IBuffer timing is
active. See
`results/captures/sm86_front_end_ibuffer_depth_20260818_v1/analysis/README.md`.
