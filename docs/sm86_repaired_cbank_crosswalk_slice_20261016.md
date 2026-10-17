# SM86 repaired dynamic-CBANK crosswalk — H89 — 2026-10-16

## Capture and replay

The repaired external CUDA13.2 NVBit tracer was run on the H85 matrix binary
under the locked RTX 3090 Ti environment. Real mode-0 and mode-1 captures
completed with 84 threadblocks each at `/tmp/h89_gcc_h85_mode{0,1}_v6/`; the
enhanced protobuf carries dynamic CBANK payloads. Both captures replayed
successfully through the active adapter at
`.../gcc_demand_h85/replay_h89_v6/full.jsonl.zst`.

Mode 0 reports 5,376 ULDC instructions, 21,504 dynamic constant references,
688,128 addresses, and 3 unique sectors. Mode 1 reports 344,064 LDC, 5,376
ULDC, 712,320 references, 22,794,240 addresses, and 147 unique sectors; no
metadata-only records were observed.

## Crosswalk boundary

The matched H85 NCU mode-1 row reports 590,041 GCC constant requests and 2,240
`lts__t_sectors_srcunit_gcc` sectors, while the repaired trace reports 344,064
LDC instructions and 147 unique bank-local sectors. These domains are
different: dynamic trace population/address reuse is not equivalent to GCC
request or L2-sector traffic. The repaired transport is therefore promoted as
a typed observer only; no GCC/L2/DRAM timing or request conversion is enabled.

This closes the prior missing-field transport diagnosis, but not P4 source
client completion ownership. A future promotion needs an independently
validated mapping across a held-out matrix, not a same-workload NCU fit.
