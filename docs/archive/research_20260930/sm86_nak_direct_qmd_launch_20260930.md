# Independent NAK body through an Ampere QMD V03/raw-RM launch

This report records a new compiler/launch seam exercised on the local RTX 3090
Ti. It is a hardware execution observation, not a test, hash, or verification
workflow.

## External reference and host adaptation

The experiment used the open-source `libmc` Ampere GA10x backend (QMD V03,
raw RM/UVM compute channel, no `libcuda`, no CUDA ELF loader) from
[libmc](https://github.com/maxime-peim/libmc). Its QMD builder follows Mesa
NVK's [`Qmd3_0` field layout](https://github.com/chaotic-cx/mesa-mirror/blob/main/src/nouveau/compiler/nak/qmd.rs)
and its GA10x class path uses `AMPERE_COMPUTE_B`
(`0xc7c0`). The clone was built against the installed open NVIDIA
615.71.09 headers; the older `cla0c0` cache-method include was mapped to the
current `clb0c0` header for this source-only probe.

The host was the locked RTX 3090 Ti/SM86:

* UUID `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`
* graphics/SM 1350 MHz, memory 10501 MHz
* driver/KMD 615.71.09

## Direct independent-codegen launch

The normal libmc demo first launched its embedded `nvcc` SM86 body through the
raw QMD path and wrote `0xdeadbeef` to the BAR1-visible scratch cell. I then
replaced only the GA10x body with the 160-byte code section from the
independently generated Mesa NAK shader (`shader_1.bin`, 10 instructions),
remapping its NAK constant references `c0[0]/c0[4]` to the libmc Ampere
argument slots `c0[0x160]/c0[0x164]`. The QMD was adapted to the NAK shader
metadata (`local_size=8`, `global_size=8`, `num_gprs=19`, QMD group/cache
invalidations, constant-buffer invalidation, prefetch address, SM86 SASS
version, and shared-memory config). The direct RM launch completed with
`launch_status=0`; the captured BAR1 scratch range contained `1,2,3,4,5` in
dwords 1--5, matching the NAK body’s thread-derived vector-store payload
(dword 0 remains a separate visibility/cache corner of this BAR1 mapping).
`nvdisasm` decoded the loaded body as
native SM86 `S2R`, `SHF`, `LEA`, `PRMT`, `LEA.HI.X`, and `STG.E.128`
instructions.

The NAK body uses a Vulkan/NVK-style GPU-scope store. The direct observation is
not being promoted to a system-scope publication claim: a deliberately
system-scope body or a device-memory consumer is still needed for that. Native
code generation and QMD execution are independent of CUDA ELF, but memory-scope
and publication semantics remain producer/runtime contracts. The reporting
semaphore is emitted only after an explicit compute `WAIT_FOR_IDLE`; submission
consumption and shader completion are therefore not collapsed into one event.

## Boundary update

This materially narrows the compiler blocker:

* independent SM86 instruction bytes can now execute under an independently
  constructed Ampere QMD V03 and raw RM submission path;
* the launch no longer depends on NVIDIA-generated ELF sections, CUDA module
  loading, or the two-parameter CUDA template;
* the remaining gap is unrestricted semantic/ABI coverage: arbitrary NAK
  bodies, relocations, launch descriptors, system-scope/publication semantics,
  and general CUDA parameter layouts are not yet unified.

Therefore this is a bounded `IndependentISA + DirectQMDLaunch` capability,
not unrestricted CUDA ABI/ISA closure.

An Nsight Compute attempt with `--target-processes all` around the same raw-RM
binary returned the body’s host output but reported **no kernels profiled** and
did not create an NCU report. This is a concrete endpoint boundary: direct
GPFIFO/QMD launches are not automatically visible to the CUDA profiler, so the
direct body cannot be promoted to an NCU pipe/timing witness.
