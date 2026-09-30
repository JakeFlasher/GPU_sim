# SM86 deep research continuation

Updated 2026-09-29. This report records research and bounded local
microbenchmarks only. It is not a correctness, hashing, verification, or
silicon-certification gate.

## Pipe and issue observability

NVIDIA's Nsight Compute guide describes pipeline metrics as candidate/logical
pipeline information; dynamic multi-pipeline assignment is not included in
`sass__inst_executed_per_opcode_pipeline`. CUPTI PC sampling periodically
selects a random active warp; the sampled warp may differ from the scheduler's
issued warp and has no issue-cycle resolution. The local campaign
`results/alpha/full_3090ti_20260928_110445_sampling/` collected these fields
for all 15 workloads at 2,048-cycle PC-sampling intervals. They remain
statistical evidence, not `(PC, warp, ordinal, physical pipe, issue cycle)`
witnesses.

The new local isolated-class probe at
`results/alpha/sm86_isolated_pipe_probe_20260928/` narrows this boundary:
static SASS counts and aggregate NCU counters agree for IADD3→ALU (128),
FFMA→FMA-lite (256), MUFU.RCP→XU (256), and the target LDG population within
LSU (32 of 34). This is an exact population/class witness for deliberately
isolated kernels, not a per-instruction route tag or issue-cycle witness for
arbitrary mixed workloads.

The current Nsight Compute surface audit at
`results/alpha/sm86_pipe_metric_surface_20260929/` makes the boundary explicit:
`ncu --query-metrics` lists aggregate `smsp__inst_executed_pipe_*` counters,
while `--page source` emits SASS PCs, instruction counts, and stall columns but
no pipe value per address. The raw report likewise retains kernel-level pipe
totals only. No public CUPTI/NCU field in this toolchain provides the missing
mixed-kernel physical-route or issue-cycle tag.

The current-clock expansion at
`results/alpha/sm86_isolated_pipe_classes_20260929/` profiles eight additional
known-population classes. Static-to-aggregate rows are IADD3/ALU 34→36,
FFMA/FMA 64→65, MUFU.RCP/XU 64→64, LDG/LSU 1→3, SHF/ALU 64→67, PRMT/ALU
64→67, SHFL.BFLY/LSU aggregate 64→66, and BAR.SYNC/ADU 64→64. The SHFL row
demonstrates that a public aggregate counter label need not match the SASS
opcode's intuitive functional name; these remain class-local correspondences,
not dynamic physical-route or issue-cycle traces.

The direct RM/PMA follow-up at
`results/alpha/sm86_nak_pma_probe_20260929/` captured 52 Ampere PC/stall
records from independently generated NAK code, including active/warp and
GPC/TPC/SM fields. The decoded stream has no physical-pipe or issue-slot
attribute, providing an independent telemetry corroboration rather than a new
route witness.

The current NVIDIA forum microarchitecture discussion likewise describes
GA10x pipes at a high level and explicitly notes that the available public
detail is limited; it lists multiple INT32 functional classes without exposing
the scheduler's dynamic sub-unit selection. The active `sass-king` project
provides controlled SASS corpus/pattern evidence and keeps control-code gaps
open, but it is an audit/disassembly corpus rather than a physical issue-trace
source. These independent historical/current lanes converge on the same
fail-closed boundary.

Primary sources:

- https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- https://docs.nvidia.com/cupti/main/main.html
- https://arxiv.org/abs/2503.20481
- https://patents.google.com/patent/US20110072244A1/en
- https://forums.developer.nvidia.com/t/mapping-of-pipelines-to-functional-units/315200
- https://github.com/florianmattana/sass-king

## Compiler and ISA

The in-tree `tools/nvptx_compile_sm86/` adapter successfully lowers PTX through
NVIDIA's `nvPTXCompiler` API to an SM86 cubin. This is the strongest
available practical compiler path, but it is proprietary PTX lowering. Public
`nvdisasm`/`cuobjdump` are disassemblers. A bounded probe of
`redplait/denvdis` (`ina` with generated SM86 tables) emitted a 16-byte
`IADD3 R1,R2,R3,R4` word that `nvdisasm -b SM86` decoded correctly; the raw
word is retained at `results/alpha/sm86_sass_encoder_probe_20260928/`. This is
stronger than a hypothetical encoder, but `ina` is not a cubin linker/compiler:
it does not generate current CUDA ELF/EIATTR/relocation ABI, scheduler-safe
control metadata, or arbitrary PTX lowering. CuAssembler and NBAssembler remain
bounded alternatives with incomplete SM86 metadata/coverage. The current CUDA
13.x cubin ELF metadata is not a stable public ABI for independent assemblers.

The official linker seam is now exercised as well:
`results/alpha/sm86_nvjitlink_probe_20260929/` contains a 3,496-byte SM86
ET_EXEC cubin emitted by the installed `nvJitLink` (runtime-reported version
13.4 under the local `cuda-13.2` path) from a representative PTX input. `cuobjdump`
decodes the linked image and its generated CUDA/ELF sections. This removes a
practical PTX-to-linked-cubin gap, but it remains NVIDIA's proprietary lowering
and linker; it supplies no unrestricted independent SASS/control-word encoder
or public CUDA ABI.
The restart matrix at `results/alpha/sm86_research_restart_20260929/` corrects
the earlier input-surface inference: the old adapter omitted PTX's terminating
NUL. With the corrected adapter, `mul.wide.u32` and `cvt.u64.u32` + `shl.b64`
forms link under both installed libraries. The earlier grammar-restriction
claim is superseded; independent SASS/control-word/CUDA-ABI generation remains
open.

The companion
`results/alpha/sm86_nvptx_compiler_surface_20260929/` sends the same
lane-addressing PTX through NVIDIA's `nvPTXCompiler` API instead. It emits a
3,240-byte SM86 cubin and executes 1,000 Driver API launches in 1.659872 ms,
with lane-indexed output 7,8,9,10. Both official NVIDIA paths remain
proprietary PTX paths; neither supplies independent SASS, scheduler/control-
word, or CUDA ABI generation.

The linked cubin was also loaded and launched through the CUDA Driver API by
`tools/nvptx_compile_sm86/nvjitlink_launch_sm86.cpp`. The local locked card
completed 1,000 launches in a 1.707776-ms CUDA-event span, and the first
generated output lane was 7. This closes the practical PTX-to-native-execution
path while leaving independent SASS/control-word/ABI generation unresolved.

An independent native-code lane is now exercised at
`results/alpha/sm86_nak_independent_codegen_20260929/`. Mesa NAK through
tinygrad/tinymesa targeted SM86 and executed an add kernel on the local card;
its `nvdisasm` output contains `IADD3`, `LDG.E.128.CONSTANT`,
`STG.E.128.STRONG.GPU`, `PRMT`, `LEA`, and `SHF`, with observed output 7..14.
This materially narrows independent code generation, but its Mesa/tinygrad
shader ABI and QMD runtime are not an unrestricted CUDA PTX/ELF/kernel ABI or
complete SASS/control-word implementation.

The operation-family follow-up at
`results/alpha/sm86_nak_operation_family_20260929/` compiles and executes
integer add/multiply/multiply-add, reciprocal, shift, xor, and maximum families
through NAK. Retained disassembly shows IADD3, IMAD.U32, MUFU.RCP, SHF,
LOP3.LUT, LDG/STG, and LEA. This strengthens independent native SM86 coverage,
but remains a Mesa/tinygrad shader ABI rather than unrestricted CUDA PTX/ELF.

The NAK/CUDA-ELF bridge at
`results/alpha/sm86_nak_cuda_elf_bridge_20260929/` goes one step further:
NAK's native body is inserted into NVIDIA-generated two-parameter SM86 ELF
metadata, its constant-buffer references are retargeted to CUDA parameter
slots, and the resulting cubin executes through the CUDA Driver API with
7,8,9,10 output. This demonstrates a real independent body-to-CUDA-loadable
bridge, while preserving the remaining general-linker/ABI limitation because
metadata and one parameter layout are hand reused.
The checked reusable implementation is
`tools/nvptx_compile_sm86/nak_cuda_bridge.py`; it is intentionally template-
based and bounded rather than a general CUDA linker.
It has now been exercised across NAK multiply, multiply-add, reciprocal, shift,
xor, and maximum bodies in
`results/alpha/sm86_nak_cuda_elf_bridge_20260929/family/`; each launches with
the CUDA template parameters. The remaining general metadata/relocation/
control-word limitation is unchanged.

The latest 2026 `nvsass-disassembler` release associated with arXiv:2609.18662
is useful corroboration, not closure: it extracts NVIDIA's generated SIDL text
and decodes/re-encodes 128-bit words, but explicitly requires CUDA 12.8 for
extraction and lists tested architectures `sm_80`, `sm_89`, `sm_90`, `sm_100`,
and `sm_120`—not `sm_86`. It is a specification/disassembler, not a complete
SM86 compiler, linker, scheduler/control-code generator, or CUDA ABI. Its
documentation also states that the source descriptions are generated on the
user's machine rather than shipped as open instruction data.
The active `sass2mlir-bench` public benchmark is likewise explicitly scoped to
SM87 Jetson and SM120 desktop paths; it does not supply an SM86 compiler/ABI
or a GA102 physical-pipe oracle.

The all-form local expansion at
`results/alpha/sm86_sass_encoder_form_coverage_20260929/` selected all 1,271
generated SM86 denvdis forms and decoded 1,162 emitted words with `nvdisasm`.
This materially strengthens the instruction-form lane, while the remaining
non-default/pseudo forms and missing ELF/ABI/scheduler generation preserve the
unrestricted compiler blocker.

Primary sources:

- https://docs.nvidia.com/cuda/cuda-binary-utilities/
- https://docs.nvidia.com/cuda/pdf/PTX_Compiler_API_User_Guide.pdf
- https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/
- https://github.com/gpuocelot/cuasm
- https://github.com/QianyanTech/NBAssembler
- https://github.com/redplait/denvdis
- https://arxiv.org/abs/2609.18662
- https://github.com/reoLantern/nvsass-disassembler
- https://github.com/mbuchel/sass2mlir-bench

## Cache ownership and flush

The local policy campaign and the copied `arabel1a/gpu-micro-bench` run provide
GA102 observations: non-inclusive behavior, write-through and L2 write-
allocate outcomes, context-switch L1 autoflush, strong cache-state effects,
and 6 MiB L2 on this card. The fixed-SM L1-allocation probe is unstable across
bounded repeats (both outcomes observed), so it is retained as unknown. NVIDIA's PTX documentation defines
scope points of coherency and cache operators as hints; `discard.global.L2`
invalidates without writeback and leaves the value undetermined. Public NVIDIA
patents describe generic coherence/flush mechanisms and point-of-coherence
vocabulary, not the RTX 3090 Ti's physical owner/dirty/read-from state.
Recent NVIDIA forum reports also show that discard/store-to-load outcomes can
change with control dependencies and forwarding, so a simple discard/reload
result is not a physical ownership witness.
NVIDIA's open hardware documentation exposes a lower-level Ampere host-channel
method `MEM_OP_D_OPERATION_L2_FLUSH_DIRTY` (`0x10`) on GPFIFO classes, including
the Ampere B/GA10x family. A corrected authorized submission now reaches that
method through a current 615.71.09 RM/UVM C56F channel. The setup includes the
secondary `UVM_MM_INITIALIZE` FD, current 376-byte channel allocation record,
image pushbuffer, notifier, CUDA-captured UVM channel envelope, RM token,
schedule, UserD GP_PUT, and Turing usermode doorbell. The corrected semaphore
probe at `results/alpha/rm_mem_op_d_semaphore_3090ti_20260928/` uses DRF-packed
GET/LENGTH fields and a post-packet timestamp: the 112-byte A/B/C/D packet
(`D = 0x80000000`) has a 480-tick completion span versus 64 ticks for a
corrected NOP baseline, GP_GET=1, and zero error status.

This closes the *privileged method reachability and packet-consumption* seam,
not the physical cache semantic seam: no independent dirty-line producer and
bypass consumer were coupled to the flush, so owner, dirty payload, read-from,
and “every dirty line written back” claims remain unobservable.
The first GP_GET-only hand-built entry was subsequently audited: its GET field
was not DRF-packed and its length expression was wrong. It is superseded by
the corrected semaphore probe at
`results/alpha/rm_mem_op_d_semaphore_3090ti_20260928/`, which uses C56F
semaphore timestamps and a corrected NOP baseline.
The independent-endpoint audit at
`results/alpha/rm_memop_independent_endpoint_audit_20260928/README.md` records
why CUDA sync/events/copies, virtual C7B5 DMA, BAR1 reads, and RM/UVM remaps
cannot be treated as a guaranteed L2-bypass oracle on GA102.
The new internal-control audit at
`results/alpha/rm_l2_internal_control_probe_20260929/` queried the open-driver
`NV2080_CTRL_CMD_INTERNAL_MEMSYS_L2_INVALIDATE_EVICT`, flush-all-RAMs/caches,
and clean-probe-filter commands through a user RM subdevice. Each returned
`0x56`, defined by the installed `nvstatuscodes.h` as `NV_ERR_NOT_SUPPORTED`.
The source-visible internal mechanisms are therefore not a user-space
independent bypass/flush oracle on this GA102 setup.
The public RM flush audit at
`results/alpha/rm_public_l2_flush_control_probe_20260929/` found and corrected
an additional driver-version issue: 615.71.09 uses writeback/invalidate/full-
cache bits 3/4/5. Full-cache writeback+invalidate controls for video memory
(`0x38`) and dirty memory (`0x3b`) returned success, while the old bit layout
returned `NV_ERR_INVALID_ARGUMENT`. This provides an accepted public flush
control, but not a dirty-line owner/read-from/DRAM writeback oracle.
The end-to-end artifact
`results/alpha/nak_shader_public_flush_coupling_20260929/` then uses a real NAK
shader to write a normal tinygrad RM image allocation, reads 7..14 through its
CPU-visible mapping, issues public full-cache writeback+invalidate (`0x38`),
and reads 7..14 afterward. This couples the producer and accepted flush path,
but does not identify the pre-flush read source or prove every dirty line was
written to DRAM.
Five-repeat host-I/O timing in the same artifact measured median control times
of 225.750 microseconds (`0x38`) and 141.152 microseconds (`0x3b`); these are
RM/ioctl synchronization observations, not L2-service latency.
The separate
`results/alpha/rm_image_uvm_external_map_probe_20260929/` probe then accepted
an `NVOS32_TYPE_IMAGE` allocation through UVM external-range creation and GPU
mapping while retaining its CPU BAR1-visible mapping. This establishes a viable
address-sharing seam for a future dirty producer/consumer comparison, but no
GPU method was submitted and BAR1 bypass/full writeback semantics remain
unproven.
The follow-on
`results/alpha/rm_image_uvm_cuda_endpoint_probe_20260929/` couples that image
VA to a CUDA Driver API context: copy-engine reads/writes and CPU BAR1 reads
agree, but an SM86 shader-store launch leaves the value unchanged. This
distinguishes the copy-engine/mapping path from shader visibility, and leaves
the cache-owner/flush comparison unresolved.
The installed 615.71.09 `nvidia-push` source confirms why this is not a
standalone CUDA command: channel allocation requires an RM import table with
`rmApiAlloc`, `rmApiControl`, DMA mapping, notifier, and kickoff callbacks
supplied by the host driver. The current direct probe reconstructs those
callbacks only for this bounded local witness; it is not a supported public
CUDA interface. Earlier allocation-only and empty-schedule artifacts remain
under `results/alpha/rm_channel_alloc_only_uvm_3090ti_20260928/` and
`results/alpha/rm_channel_schedule_empty_3090ti_20260928/`.
Artifacts:

- `results/alpha/cache_policy_sm86_20260928_104229/`
- `results/alpha/cache_microbench_3090ti_20260928_1254/`

Primary sources:

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
- https://yananguo.com/files/usenix24_1.pdf
- https://patents.google.com/patent/US9563561B2/en
- https://patents.google.com/patent/DE102025112846A1/en
- https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/ampere/ga100/dev_pbdma.ref.txt
- https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/README.txt
- https://forums.developer.nvidia.com/t/flushing-dirty-l2-cache-lines/258812
- https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/turing/tu104/dev_pbdma.ref.txt

## Timing domains

`results/alpha/timing_crosswalk_3090ti_20260928_1256/` compares locked-clock
in-kernel timers, CUPTI activity, and CUDA events. One-block spans agree closely;
a multi-block kernel demonstrates that block-0 in-kernel time undercounts grid
completion. Nsight Compute's WaitForIdle/semaphore endpoint and CUPTI's
prolog/scheduler endpoint are intentionally different. Stable external clocks
therefore close clock drift for the campaign, but not a single identical
NCU/CUPTI/simulator endpoint.

The current-driver rerun is
`results/alpha/timing_crosswalk_3090ti_20260929_current/` (driver 615.71.09,
P0/1350/10501, five repetitions). Its 108-block medians were 35,792,641 ns
(CUDA event), 35,789,261 ns (CUPTI), and 26,928,416 ns (block-0
`%globaltimer`); the one-block medians were 212,985,184, 213,175,851, and
212,888,096 ns respectively. CUPTI and CUDA event therefore provide a
current-driver matched grid-span pair within about 3.4 microseconds, while the
single-thread device timer undercounts grid completion by about 8.864 ms. This
is stronger timing-domain evidence, not closure of the NCU/simulator endpoint
boundary.

The fresh full campaign at
`results/alpha/full_3090ti_20260929_022713_pidfd/` also removes a host-side
measurement confounder: all 375 build/native/trace/replay/NCU process waits
used Linux pidfd readiness rather than Python's coarse timeout polling. Its
replay-wall median is 0.631848 s. That result is a cleaner host-duration
observation, not proof that host replay, CUPTI, NCU, and simulator completion
share one endpoint.
The paired warm-cache run at
`results/alpha/full_3090ti_20260929_023820_pidfd_warm/` completed 15/15 with
the same pidfd timing path and NCU `cache-control=none`; its median NCU span
was 8,576 ns and replay wall was 0.634228 s. The cold/warm contrast is a
cache-domain observation, not a matched timing score.

Primary sources:

- https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html
- https://forums.developer.nvidia.com/t/question-about-nsight-tools-timestamping-mechanism-and-clock-sources/323887
- https://arxiv.org/abs/2208.11174

## Closure boundary

The research closes practical PTX compilation, bounded GA102 cache-policy
observations, privileged C56F method reachability/retirement, isolated
instruction-class pipe populations, stable-clock campaign operation, and
statistical pipe evidence. It does not close universal exact physical SM pipe
route witnesses, independent unrestricted SM86 SASS/ABI
generation, proprietary cache owner/dirty/read-from state, semantic proof of
complete host flush, or identical public timing endpoints. Those remain
external blockers rather than gaps that can be honestly filled by a larger
heuristic.
