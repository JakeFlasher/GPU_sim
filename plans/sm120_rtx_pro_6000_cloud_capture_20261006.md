# SM120 RTX PRO 6000 Workstation cloud-capture plan

Updated 2026-10-06. The active SM120 target is the **NVIDIA RTX PRO 6000
Blackwell Workstation Edition** on a full Ubuntu cloud VM, not the historical
RTX 5060 laptop. Cloud collection and local replay are intentionally separate.
The operator manual and small scripts live outside the repository at
`/home/jakeshea/sm120_rtx_pro_6000_capture/MANUAL.md`.

This plan records technical milestones only. Captures, NCU reports, and trace
inventories are ordinary run outputs; none is an approval, admission, or
execution gate.

| Phase | Milestone | Completion bar | State |
|---|---|---|---|
| P0 | Source and hardware review | Review project memory/status, existing SM120/5060 card, SimBench/SOL populations, NVBit/NCU seams, CUDA/Blackwell references; distinguish vendor facts from live gaps | complete |
| P1 | Independent collection workspace | Create `/home/jakeshea/sm120_rtx_pro_6000_capture`, target/config files, 235-row SOL + 92-row SimBench catalog, small cloud/local scripts, and a step-by-step manual | complete |
| P2 | Pro 6000 architecture identity | Add `rtx_pro_6000_ws` card, aliases, schema/Rust SKU, manifest schema branches, and active Pro routing without changing historical 5060 labels | complete — schema checks, `sim-arch` 77 tests, and focused Python compilation pass |
| P3 | Cloud image preflight | Run `cloud/00_check_prereqs.sh`, optional user-dependency setup, host metadata, and NCU metric query on the actual Pro 6000; stop on wrong GPU/MIG | pending cloud |
| P4 | Tracer/toolchain build | Build `tracer_tool.so` and metric extractor for `ARCH=sm_120`; retain compiler/protobuf/driver errors and do not use an SM86 tracer | pending cloud |
| P5 | First bounded vertical | Same-binary correctness → Compute Sanitizer → static SASS → NVBit trace → full-set NCU report/raw CSV for one scalar workload | pending cloud |
| P6 | Mechanism matrix | Expand scalar, memory/cache, tensor/quant, async/TMA, cluster/DSMEM, and vendor controls one group at a time; preserve unsupported/timeouts | pending cloud |
| P7 | SOLBench crosswalk | Stage all 26 FIB, representative then complete L1/L2, and all 33 Quant source/build/correctness rows; add Pro-only SimBench counterparts without calling them score replacements | pending cloud |
| P8 | Local import/replay | Pull captures non-destructively, inspect wire/static artifacts, replay with the Pro card on the local host, and add only evidence-supported SM120 simulator seams | pending local + cloud |
| P9 | Held-out validation | Keep NCU timing, issue/model, completion, typed tails, and accuracy frontiers separate; document residual gaps without broad hardware claims | pending local + cloud |

## Current next action

P3 is the only active step. Execute the cloud manual through the first host
metadata capture. Do not start P6 fan-out before P5 has one complete vertical.

P2 local closeout checks: all three cards pass JSON schema validation; `sim-arch`
77 tests, `sim-sm` 496 tests, `sim-run` 66 tests, and focused Python/schema/
NVBit/reference checks (44 passed / 703 subtests) are green. Shell syntax checks
pass, and the local 3090 Ti correctly exits 3 from the Pro-only cloud preflight.
These are software/configuration checks, not SM120 hardware evidence.

## Active target facts

Vendor sources state CC 12.0, GB202, 188 SMs, 96 GB ECC GDDR7, 1,792 GB/s
specified bandwidth, 48 resident warps/SM, 32 blocks/SM, and 128 KiB shared
memory per SM for CC 12.0. These facts seed the dedicated card only where the
card labels them as vendor information. Cache slices/replacement, load policy,
latencies, issue topology, clock crosswalk, and completion semantics remain
research gaps until measured on this exact workstation SKU.

## Historical separation

`sm_120-laptop` / `rtx_5060_laptop` remains valid for retained historical
artifacts and explicit laptop-only tests. The bare active `sm_120` loader path
now selects the Pro 6000 card; callers that need the old laptop must request
`sm_120-laptop` or `rtx_5060_laptop` explicitly.
