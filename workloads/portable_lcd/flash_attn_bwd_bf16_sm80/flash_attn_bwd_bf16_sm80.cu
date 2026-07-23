// slug: flash_attn_bwd_bf16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// FA-2 backward (softmax-recompute) using the LCD-glue header at
// ../flash_attn/lcd_glue/bwd_sm80.cuh. Verbatim upstream Tri Dao
// FA v2 source lives alongside under ../flash_attn/upstream/.

#include "../flash_attn/lcd_glue/bwd_sm80.cuh"

extern "C" void flash_attn_bwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V_T, const __nv_bfloat16 *d_dO,
    __nv_bfloat16 *d_dQ, __nv_bfloat16 *d_dK, __nv_bfloat16 *d_dV,
    int BH, int Sq, int Sk, float scale, cudaStream_t stream)
{
    flash_attn_bwd::flash_attn_bwd_bf16_sm80_launch(
        d_Q, d_K, d_V_T, d_dO, d_dQ, d_dK, d_dV, BH, Sq, Sk, scale, stream);
}
