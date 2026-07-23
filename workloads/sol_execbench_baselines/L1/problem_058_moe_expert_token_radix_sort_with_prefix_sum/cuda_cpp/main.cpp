#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <string>

// MoE token sorting + expert bucketing via stable sort + histogram prefix sum.
//
// Reference (problem_058_moe_expert_token_radix_sort_with_prefix_sum):
//     num_experts = 256
//     flat = topk_idx.reshape(-1)
//     # Stable sort on expert IDs = counting-sort permutation.
//     _, sorted_token_indices = flat.sort(stable=True)
//     expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
//     expert_offsets[1:] = torch.bincount(flat.long(), minlength=num_experts).cumsum(0).to(torch.int32)
//     return sorted_token_indices.to(torch.int32), expert_offsets
//
// Per definition.json (insertion order):
//   topk_idx              : [batch_size, seq_len, num_experts_per_tok]   int32
//   sorted_token_indices  : [num_tokens]                                 int32  (DPS)
//   expert_offsets        : [num_experts_plus_one]                       int32  (DPS)
//
//   num_tokens = batch_size * seq_len * num_experts_per_tok
//   num_experts = 256 (const axis); num_experts_plus_one = 257.
//
// Semantics worth pinning:
//   * `flat.sort(stable=True)` returns (sorted_values, sort_indices). The
//     reference discards the values and keeps the INDICES -- i.e.
//     sorted_token_indices is the stable ARGSORT permutation of the flattened
//     topk_idx (positions into the flattened [num_tokens] array), NOT the
//     expert ids themselves. We reproduce that with at::argsort(stable=true).
//   * Stability is contractual: among equal expert ids the original flattened
//     order is preserved. We pass stable=true so the tie-break matches torch's
//     stable sort exactly (an unstable sort would reorder ties arbitrarily and
//     fail correctness).
//   * bincount(flat.long(), minlength=num_experts) histograms expert ids into
//     `num_experts` bins; cumsum(0) is the inclusive prefix sum; the leading 0
//     is supplied by zeroing expert_offsets and writing the cumsum into [1:].
//   * Output dtype is int32 for both; argsort yields int64 indices and the
//     cumsum is int64, so both are cast back to int32.
//
// num_experts is derived from the pre-allocated expert_offsets shape
// (num_experts_plus_one - 1) so the histogram width and DPS output stay in
// lockstep; it equals the reference's hard-coded 256 for every workload.
//
// Implementation: pure ATen dispatch -- NO custom CUDA kernels. at::argsort /
// at::bincount / at::cumsum each launch their own CUDA kernels on the current
// stream; outputs are written in place into the pre-allocated destinations
// (destination-passing style).

static void check_int32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kInt32, name, " must be int32, got ", t.dtype());
}

void run(
    const torch::Tensor& topk_idx,
    torch::Tensor sorted_token_indices,
    torch::Tensor expert_offsets
) {
    check_int32_cuda(topk_idx, "topk_idx");
    check_int32_cuda(sorted_token_indices, "sorted_token_indices");
    check_int32_cuda(expert_offsets, "expert_offsets");
    TORCH_CHECK(expert_offsets.dim() == 1,
                "expert_offsets must be 1D [num_experts+1], got ", expert_offsets.dim(), "D");
    TORCH_CHECK(expert_offsets.size(0) >= 1,
                "expert_offsets must have at least 1 element");

    const c10::cuda::CUDAGuard device_guard(topk_idx.device());

    // num_experts derived from the pre-allocated offsets array (== 256).
    const int64_t num_experts = expert_offsets.size(0) - 1;

    // flat = topk_idx.reshape(-1)
    const torch::Tensor flat = topk_idx.reshape({-1});

    // _, sorted_token_indices = flat.sort(stable=True)  -> keep the INDICES, the
    // stable argsort permutation. at::argsort over an int32 1-D tensor works
    // directly; pass stable=true for the same tie-break as torch's stable sort.
    torch::Tensor sort_idx = at::argsort(flat, /*stable=*/true, /*dim=*/-1, /*descending=*/false);
    // sorted_token_indices.to(torch.int32)
    sort_idx = sort_idx.to(torch::kInt32);

    // expert_offsets[1:] = bincount(flat.long(), minlength=num_experts).cumsum(0).to(int32)
    const torch::Tensor flat_long = flat.to(torch::kInt64);
    const torch::Tensor counts = at::bincount(flat_long, /*weights=*/{}, /*minlength=*/num_experts);
    const torch::Tensor cumsum = at::cumsum(counts, /*dim=*/0).to(torch::kInt32);

    // expert_offsets = [0, cumsum...]: zero the destination (supplies the
    // leading 0 at index 0) then write the cumsum into [1:].
    expert_offsets.zero_();
    using torch::indexing::Slice;
    expert_offsets.index_put_({Slice(1, c10::nullopt)}, cumsum);

    TORCH_CHECK(sort_idx.sizes() == sorted_token_indices.sizes(),
                "argsort output shape ", sort_idx.sizes(),
                " does not match pre-allocated sorted_token_indices ",
                sorted_token_indices.sizes());
    TORCH_CHECK(cumsum.size(0) == num_experts,
                "bincount/cumsum length ", cumsum.size(0),
                " does not match num_experts ", num_experts);

    sorted_token_indices.copy_(sort_idx);
    // expert_offsets already written in place above.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "MoE expert token sort + prefix sum via at::argsort(stable) + "
          "at::bincount + at::cumsum: stable argsort permutation into "
          "sorted_token_indices, [0, cumsum(histogram)] into expert_offsets; "
          "int32; destination-passing style.");
}
