# Blackwell SM120 target notes

These notes route exact-target questions; they are not a substitute for the
active plan or a claim that hidden hardware topology is known. The active
Blackwell device is the GeForce RTX 5090 (`sm_120`, CC 12.0, GB202). The
SM86/RTX 3090 Ti lane remains a separate compatibility baseline.

## Tracked sources

- `references/NVIDIA_Manuals/blackwell_tuning/`
- `references/NVIDIA_Manuals/blackwell_compatibility/`
- `references/NVIDIA_Manuals/ptx_isa/`
- `arch_profiles/cards/sm_120-rtx-5090.json`
- `/dev/shm/sm120_rtx_5090_work` for measured event/NCU/NVBit evidence

Optional `.claude/knowledge/` cards are not present in this checkout; do not
loop on missing cards or recreate them as prerequisites.

## Current target facts and limits

- The retained RTX 5090 card records 48 resident warps/SM, 24 device-reported
  max blocks/SM, 170 SMs, 96 MiB vendor L2 capacity, and unresolved cache/issue/
  timing fields. These are card inputs and measured runtime attributes, not a
  complete physical model.
- Q10–Q15 observations establish legal target execution, route/placement
  responses, a post-sync TMA payload marker, and cluster ranks. They do not
  expose native completion timestamps, issue-port wiring, cache hash, MSHR
  count, or queue topology.
- FP4/MX and other architecture-specific instructions require an explicit
  `sm_120a`/`sm_120f` build only when the source and compiler support that path;
  a compile token is not evidence of runtime capability.
- Cluster/TMA behavior must be measured on the selected device. Do not infer
  SM120 behavior from SM100/Hopper documentation or from another SKU.

## Routing rules

1. Identify the exact device and compute capability before making a host claim.
2. Inspect generated SASS/cubin and the active trace contract; static mnemonic
   names do not reveal physical scheduler assignment.
3. Keep CUDA events, CUPTI, NCU, NVBit, and simulator frontiers as distinct
   observations.
4. Use the full replay first and the fast diagnostic second; never use NCU
   targets to tune the same scored replay.
