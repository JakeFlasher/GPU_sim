"""Bounded SM86 reference control for variable-length attention backward.

The original SOL reference loops over chunks and uses production dimensions.
This control keeps the same Q/K/V projection and chunk-isolated attention
backward dataflow, but builds one block-diagonal attention mask so NVBit sees a
small, deterministic set of GEMMs on GA102.
"""
import torch


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    s = int(axes_and_scalars["total_seq_len"])
    chunks = int(axes_and_scalars["num_chunks"])
    d_model = int(axes_and_scalars["d_model"])
    heads = int(axes_and_scalars["num_heads"])
    head_dim = int(axes_and_scalars["head_dim"])
    qkv = heads * head_dim
    if chunks < 1 or s < chunks:
        raise ValueError("SM86 control requires nonempty attention chunks")
    lengths = [s // chunks + (i < s % chunks) for i in range(chunks)]
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return {
        "grad_output": torch.randn(s, d_model, dtype=torch.bfloat16, device=device),
        "hidden_states": torch.randn(s, d_model, dtype=torch.bfloat16, device=device),
        "query_states": torch.randn(1, heads, s, head_dim, dtype=torch.bfloat16, device=device),
        "key_states": torch.randn(1, heads, s, head_dim, dtype=torch.bfloat16, device=device),
        "value_states": torch.randn(1, heads, s, head_dim, dtype=torch.bfloat16, device=device),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "q_weight": torch.randn(qkv, d_model, dtype=torch.bfloat16, device=device),
        "k_weight": torch.randn(qkv, d_model, dtype=torch.bfloat16, device=device),
        "v_weight": torch.randn(qkv, d_model, dtype=torch.bfloat16, device=device),
        "out_weight": torch.randn(d_model, qkv, dtype=torch.bfloat16, device=device),
    }


def _flat(x: torch.Tensor) -> torch.Tensor:
    # [1, H, S, D] -> [S, H*D]
    return x.squeeze(0).transpose(0, 1).reshape(x.shape[2], -1).contiguous()


@torch.no_grad()
def run(grad_output, hidden_states, query_states, key_states, value_states,
        cu_seqlens, q_weight, k_weight, v_weight, out_weight):
    go = grad_output.float()
    hs = hidden_states.float()
    q = query_states.float()
    k = key_states.float()
    v = value_states.float()
    qw, kw, vw, ow = (q_weight.float(), k_weight.float(), v_weight.float(), out_weight.float())
    s, heads, d = q.shape[2], q.shape[1], q.shape[3]
    scale = d ** -0.5

    # Build a block-diagonal mask entirely on-device; setup happens inside the
    # profiled region but is tiny for the bounded control dimensions.
    pos = torch.arange(s, device=q.device, dtype=torch.int64)
    bounds = cu_seqlens[1:-1].to(torch.int64)
    chunk_id = torch.searchsorted(bounds, pos, right=False)
    same_chunk = chunk_id[:, None].eq(chunk_id[None, :])[None, None]

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~same_chunk, -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    attn = torch.matmul(probs, v)

    grad_attn = torch.matmul(go, ow).reshape(s, heads, d).transpose(0, 1).unsqueeze(0)
    grad_v = torch.matmul(probs.transpose(-2, -1), grad_attn)
    grad_probs = torch.matmul(grad_attn, v.transpose(-2, -1))
    grad_scores = probs * (grad_probs - (probs * grad_probs).sum(dim=-1, keepdim=True))
    grad_q = torch.matmul(grad_scores, k) * scale
    grad_k = torch.matmul(grad_scores.transpose(-2, -1), q) * scale

    qf, kf, vf = _flat(grad_q), _flat(grad_k), _flat(grad_v)
    af = _flat(attn)
    grad_out_weight = torch.matmul(go.transpose(0, 1), af)
    grad_out_bias = go.sum(dim=0)
    grad_q_weight = torch.matmul(qf.transpose(0, 1), hs)
    grad_q_bias = qf.sum(dim=0)
    grad_k_weight = torch.matmul(kf.transpose(0, 1), hs)
    grad_k_bias = kf.sum(dim=0)
    grad_v_weight = torch.matmul(vf.transpose(0, 1), hs)
    grad_v_bias = vf.sum(dim=0)
    grad_hidden = torch.matmul(qf, qw) + torch.matmul(kf, kw) + torch.matmul(vf, vw)
    return tuple(x.to(torch.bfloat16) for x in (
        grad_hidden, grad_q_weight, grad_q_bias, grad_k_weight, grad_k_bias,
        grad_v_weight, grad_v_bias, grad_out_weight, grad_out_bias,
    ))
