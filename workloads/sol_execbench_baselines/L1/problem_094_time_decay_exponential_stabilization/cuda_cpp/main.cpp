#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <string>

// RWKV time-decay exponential with numerical stabilization (WKV recurrence).
//
// Reference (problem_094_time_decay_exponential_stabilization), reproduced
// op-for-op:
//   batch_size, seq_len, hidden_size = key.size()
//   max_state = max_state.clone().float()
//   num_state = num_state.clone().float()
//   den_state = den_state.clone().float()
//   time_decay_exp = -exp(time_decay.float())          # [hidden_size]
//   output = zeros_like(key, dtype=float32)
//   for t in range(seq_len):
//       current_key   = key[:, t].float()              # [B, hidden_size]
//       current_value = value[:, t].float()            # [B, hidden_size]
//       # --- output computation with stabilization ---
//       max_for_output = maximum(max_state, current_key + time_first)
//       e1_output = exp(max_state - max_for_output)
//       e2_output = exp(current_key + time_first - max_for_output)
//       numerator   = e1_output * num_state + e2_output * current_value
//       denominator = e1_output * den_state + e2_output
//       output[:, t] = numerator / denominator
//       # --- state update with stabilization ---
//       max_for_state = maximum(max_state + time_decay_exp, current_key)
//       e1_state = exp(max_state + time_decay_exp - max_for_state)
//       e2_state = exp(current_key - max_for_state)
//       num_state = e1_state * num_state + e2_state * current_value
//       den_state = e1_state * den_state + e2_state
//       max_state = max_for_state
//   return output, max_state, num_state, den_state
//
// Per definition.json (insertion order):
//   time_decay    : [attention_hidden_size]                 fp32
//   key           : [batch_size, seq_len, hidden_size]      fp32
//   time_first    : [attention_hidden_size]                 fp32
//   value         : [batch_size, seq_len, hidden_size]      fp32
//   max_state     : [batch_size, attention_hidden_size]     fp32
//   num_state     : [batch_size, attention_hidden_size]     fp32
//   den_state     : [batch_size, attention_hidden_size]     fp32
//   output        : [batch_size, seq_len, hidden_size]      fp32  (DPS, pre-alloc)
//   max_state_out : [batch_size, attention_hidden_size]     fp32  (DPS, pre-alloc)
//   num_state_out : [batch_size, attention_hidden_size]     fp32  (DPS, pre-alloc)
//   den_state_out : [batch_size, attention_hidden_size]     fp32  (DPS, pre-alloc)
//
// Implementation: pure ATen / cuBLAS dispatch -- NO custom CUDA kernels. This
// is an inherently sequential linear-attention scan, so it runs as a host-side
// loop over the timestep axis; each iteration issues a handful of elementwise
// ATen ops on the current stream. The loop-carried max / num / den states are
// seeded from the input tensors (cloned so the inputs stay untouched, matching
// the reference's .clone().float()) and threaded across iterations. The
// per-timestep result is written into output[:, t] in place via select+copy_,
// and the three final states are written into their pre-allocated destinations.
// All compute is fp32; the reference's .float() casts are no-ops at fp32 but
// reproduced for faithfulness. Many tiny launches are expected -- correctness,
// not throughput, is the contract here. Outputs are destination-passing style.

static void check_fp32_cuda(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32, got ", t.dtype());
}

void run(
    const torch::Tensor& time_decay,
    const torch::Tensor& key,
    const torch::Tensor& time_first,
    const torch::Tensor& value,
    const torch::Tensor& max_state,
    const torch::Tensor& num_state,
    const torch::Tensor& den_state,
    torch::Tensor output,
    torch::Tensor max_state_out,
    torch::Tensor num_state_out,
    torch::Tensor den_state_out
) {
    check_fp32_cuda(time_decay, "time_decay");
    check_fp32_cuda(key, "key");
    check_fp32_cuda(time_first, "time_first");
    check_fp32_cuda(value, "value");
    check_fp32_cuda(max_state, "max_state");
    check_fp32_cuda(num_state, "num_state");
    check_fp32_cuda(den_state, "den_state");
    check_fp32_cuda(output, "output");
    check_fp32_cuda(max_state_out, "max_state_out");
    check_fp32_cuda(num_state_out, "num_state_out");
    check_fp32_cuda(den_state_out, "den_state_out");
    TORCH_CHECK(key.dim() == 3,
                "key must be 3D [batch, seq_len, hidden], got ", key.dim(), "D");
    TORCH_CHECK(value.sizes() == key.sizes(),
                "value shape ", value.sizes(), " must match key ", key.sizes());

    const c10::cuda::CUDAGuard device_guard(key.device());

    const int64_t seq_len = key.size(1);

    // Loop-carried states, seeded from the inputs (cloned so the input tensors
    // are not mutated, matching the reference's .clone().float()).
    auto max_running = max_state.clone().to(torch::kFloat32);
    auto num_running = num_state.clone().to(torch::kFloat32);
    auto den_running = den_state.clone().to(torch::kFloat32);

    // Exponential decay (negated to ensure decay).
    auto time_decay_exp = -at::exp(time_decay.to(torch::kFloat32));  // [hidden]

    // Sequential scan over the timestep axis.
    for (int64_t t = 0; t < seq_len; ++t) {
        auto current_key = key.select(1, t).to(torch::kFloat32);    // [B, hidden]
        auto current_value = value.select(1, t).to(torch::kFloat32);  // [B, hidden]

        // --- Output computation with stabilization. ---
        auto key_plus_first = current_key + time_first;
        auto max_for_output = at::maximum(max_running, key_plus_first);
        auto e1_output = at::exp(max_running - max_for_output);
        auto e2_output = at::exp(key_plus_first - max_for_output);
        auto numerator = e1_output * num_running + e2_output * current_value;
        auto denominator = e1_output * den_running + e2_output;
        output.select(1, t).copy_(numerator / denominator);

        // --- State update with stabilization. ---
        auto max_plus_decay = max_running + time_decay_exp;
        auto max_for_state = at::maximum(max_plus_decay, current_key);
        auto e1_state = at::exp(max_plus_decay - max_for_state);
        auto e2_state = at::exp(current_key - max_for_state);
        num_running = e1_state * num_running + e2_state * current_value;
        den_running = e1_state * den_running + e2_state;
        max_running = max_for_state;
    }

    // --- Destination-passing-style writes (final loop-carried states). ---
    TORCH_CHECK(max_running.sizes() == max_state_out.sizes(),
                "max_state_out shape ", max_running.sizes(),
                " does not match pre-allocated ", max_state_out.sizes());
    TORCH_CHECK(num_running.sizes() == num_state_out.sizes(),
                "num_state_out shape ", num_running.sizes(),
                " does not match pre-allocated ", num_state_out.sizes());
    TORCH_CHECK(den_running.sizes() == den_state_out.sizes(),
                "den_state_out shape ", den_running.sizes(),
                " does not match pre-allocated ", den_state_out.sizes());
    max_state_out.copy_(max_running);
    num_state_out.copy_(num_running);
    den_state_out.copy_(den_running);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "RWKV WKV time-decay recurrence with log-sum-exp-style stabilization. "
          "Pure ATen dispatch: a host-side sequential scan over the timestep "
          "axis threading loop-carried max / num / den states; per timestep does "
          "the stabilized WKV output (maximum, exp, normalized numerator / "
          "denominator) and the stabilized state update. fp32; destination-"
          "passing style (output, max_state_out, num_state_out, den_state_out).");
}
