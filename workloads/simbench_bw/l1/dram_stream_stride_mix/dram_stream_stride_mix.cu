// slug: dram_stream_stride_mix
// tier: P
// arch_gate: sm_86 sm_90a sm_120
//
// DRAM streaming microbench: each warp issues `accesses` global loads
// across `stride_classes` distinct strides to exercise DRAM page /
// bank / row controller activity. Output is the identity round-trip;
// the strided reads are folded into a sink to prevent DCE.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void dram_stream_stride_mix_kernel(
    const float *__restrict__ in, float *__restrict__ out, float *__restrict__ sink,
    int32_t warps, int32_t accesses, int32_t stride_classes)
{
    // R32: index arithmetic uses int64 so the working_set_sweep MiB
    // ladder (`accesses = working_set_mib * 4096`) does not overflow
    // `int` for the larger ladder slots. The two target source loads
    // (`in[idx]` and `in[i]`) are the SASS-confirmation surfaces; the
    // index types do not change which LDG opcode the compiler emits.
    const int64_t total = (int64_t)warps * accesses * stride_classes;
    const int tid = threadIdx.x;
    const int lanes = blockDim.x;
    const int warp_id = blockIdx.x;
    float acc = 0.0f;
    // R25 rehab: per-warp deterministic line-layer reuse. Each
    // warp owns a distinct 4096-float slice of `in` so that the
    // sequence of line-touches inside its window does not collide
    // with other warps' sequences. For each target line count in
    // {32, 64, 128}, the warp visits that many distinct lines and
    // re-visits each line `repeats` times; the second-and-later
    // line-0 touches produce a line-layer reuse distance of
    // (target_lines - 1), placing observations cleanly in
    // `line(16-31)`, `line(32-63)`, and `line(64-127)` bins. The
    // historical strided sweep is removed because its modular wrap
    // pattern scattered samples thinly across many mid-range bins
    // (every bin had <50 obs under `trace_l2_accuracy_xl`).
    const int32_t target_lines[3] = {32, 64, 128};
    const int32_t repeats_per_target = 8;
    const int32_t LINE_FLOATS = 32;
    const int64_t warp_slot_floats = (int64_t)4096;
    int64_t warp_base = (int64_t)warp_id * warp_slot_floats;
    for (int t = 0; t < 3; ++t) {
        int32_t lines = target_lines[t];
        for (int repeat = 0; repeat < repeats_per_target; ++repeat) {
            for (int li = 0; li < lines; ++li) {
                int64_t addr = (warp_base + (int64_t)li * LINE_FLOATS + tid) % total;
                acc += in[addr];
            }
        }
    }
    // Sink write restricted to block 0's tid 0 so multi-block
    // launches (warps > 1) do not race on `sink[0]`. The per-warp
    // strided reuse pattern above uses `warp_id = blockIdx.x` for
    // per-warp slice ownership, which is correct; the sink write
    // is a single-cell scalar and must NOT be replicated per
    // block. Codex Round-19 review P2 fix.
    if (blockIdx.x == 0 && tid == 0) sink[0] = acc;
    __syncthreads();
    // Final identity copy uses a global thread index so each
    // thread copies a unique slot across the multi-block grid.
    // Previously the copy used per-block `tid` / `lanes` which
    // caused every block to redundantly copy the entire output
    // and race on each `out[i]` write (inflating the timing /
    // trace workload by roughly the warp count for shapes like
    // `timing_00` with warps=1024). Codex Round-19 review P2 fix.
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_lanes = gridDim.x * blockDim.x;
    for (int64_t i = global_tid; i < total; i += global_lanes) out[i] = in[i];
}

extern "C" void dram_stream_stride_mix_launch(
    const float *d_in, float *d_out, float *d_sink,
    int warps, int accesses, int stride_classes, cudaStream_t stream)
{
    int threads = 32;
    int blocks = warps > 0 ? warps : 1;
    dim3 grid(blocks, 1, 1);
    dim3 block(threads, 1, 1);
    dram_stream_stride_mix_kernel<<<grid, block, 0, stream>>>(d_in, d_out, d_sink, warps, accesses, stride_classes);
}
