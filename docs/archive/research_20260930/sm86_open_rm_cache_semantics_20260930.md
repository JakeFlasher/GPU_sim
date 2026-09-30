# SM86 open-RM cache semantics probe

This report records driver/RM query and control observations on the local
RTX 3090 Ti, plus inspection of the installed 615.71.09 open kernel source.
No line-level cache or DRAM telemetry was obtained.

## Host

* GPU: NVIDIA GeForce RTX 3090 Ti, SM86, UUID
  `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`
* Driver/KMD: 615.71.09
* Clock regime during the probe: graphics/SM 1350 MHz, memory 10501 MHz

## Public RM observations

Using the open-driver SDK headers and the existing RM ioctl envelope:

* `NV2080_CTRL_CMD_FB_GET_GPU_CACHE_INFO` returned `NV_OK` and reported
  `powerState=ENABLED`, `writeMode=WRITETHROUGH`, `bypassMode=DISABLED`, and
  `rcmState=FULL`. These are driver-reported configuration values, not live
  line-state readings: the implementation hardcodes power, bypass, and RCM
  values and reads write mode from a chip-initialized software field.
* `NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE` full-cache operations returned `NV_OK`
  for video-memory invalidate/evict, system-memory writeback/invalidate/evict,
  dirty-memory writeback/invalidate/evict, compression-tag operations, and the
  `FB_FLUSH` variants. Video-memory writeback-only returned `NV_ERR_NOT_SUPPORTED`.
* `NV0080_CTRL_CMD_DMA_FLUSH` returned `NV_OK` for the L2, compression-tag,
  FB, and combined broadcast target masks.
* The internal-only controls
  `NV2080_CTRL_CMD_INTERNAL_MEMSYS_L2_INVALIDATE_EVICT`,
  `NV2080_CTRL_CMD_INTERNAL_MEMSYS_FLUSH_L2_ALL_RAMS_AND_CACHES`,
  `NV2080_CTRL_CMD_INTERNAL_MEMSYS_CLEAN_LTC_PROBE_FILTER`, and the IRQL
  flush command returned `NV_ERR_NOT_SUPPORTED` from the user RM client.
  The source calls the internal commands with an internal client through
  physical RM. User-client refusal does not establish hardware absence.

## Open-driver interpretation

The installed NVIDIA open kernel source clarifies the semantic mapping:

* `src/nvidia/src/kernel/gpu/mem_sys/kern_mem_sys_ctrl.c` maps the public
  flags to `FB_CACHE_INVALIDATE`, `FB_CACHE_WRITEBACK`, or `FB_CACHE_EVICT`.
  Address-array mode is currently rejected with `NV_ERR_GENERIC`; full-cache
  mode calls the architecture cache-op HAL.
* GA102 selects the GM200 cache-op HAL. Dirty-memory writeback/evict uses
  `NV_UFLUSH_L2_FLUSH_DIRTY`; system-memory evict also invalidates clean lines.
  Video-memory invalidate/evict forwards to physical RM, whose implementation
  is not shown by this CPU-RM source.
* A successful return is not uniformly an operation witness: dirty-memory
  invalidate-only and compression-tag invalidate-only explicitly return
  `NV_OK` without the requested register operation. Dirty-memory evict uses
  the dirty flush path, not an additional full invalidate.
* `kern_mem_sys_gm107.c` writes the flush register and polls the pending and
  outstanding bits until completion (or timeout) before returning to the
  caller. This supports synchronous protocol-completion semantics for the
  paths that actually reach that helper on a healthy attached GPU. It has
  successful-return exceptions for reset/detach and virtual-device cases;
  it is not an unconditional per-line completion witness.
* `g_kern_mem_sys_nvoc.c` classifies GA102's L2 write mode as
  `FB_CACHE_WRITE_MODE_WRITETHROUGH`, consistent with the live control query.

## Boundary update

This improves the interpretation of the existing public flush controls; it
does not close the native cache blocker. Keep the successful API return, the
specific aperture/operation, and the source-derived completion contract
separate. Do not turn a no-op success into `FlushProtocolComplete`, or treat
the static write-mode enum as a complete GPU-store policy. Line owner,
read-from source, dirty-sector residency, and per-line DRAM acknowledgement
remain unknown.

The direct source/probe paths are retained outside the simulator implementation
so the native replay contract remains unchanged. Public reference:
[NVIDIA 615.71.09 cache-operation source](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.71.09/src/nvidia/src/kernel/gpu/mem_sys/arch/maxwell/kern_mem_sys_gm200.c).

## Per-surface follow-up

A separate 4096-byte `NV01_MEMORY_LOCAL_USER` allocation was used with
`NV0041_CTRL_CMD_SURFACE_FLUSH_GPU_CACHE`. Invalidate (`flags=0x2`) and evict
(`flags=0x3`) returned `NV_OK`; writeback-only (`0x1`) returned
`NV_ERR_NOT_SUPPORTED`; an empty operation returned `NV_ERR_INVALID_ARGUMENT`.
`NV0041_CTRL_CMD_GET_SURFACE_PHYS_ATTR` returned `NVOS_STATUS_NOT_SUPPORTED`
for this user client. The retained output is
`results/alpha/sm86_rm_surface_flush_probe_20260930/`.

This adds per-allocation control evidence but still returns no physical cache
attribute, owner, read-from source, or DRAM acknowledgement.
