// slug: flash_attn_fwd_bf16_sm80
// tier: P (LCD-portable)
// arch_gate: sm_86 sm_90 sm_120
//
// FlashAttention v2 forward kernel using the LCD-portable glue
// header at workloads/portable_lcd/flash_attn/lcd_glue/fwd_sm80.cuh.
// Verbatim upstream Tri Dao FA v2 source lives alongside under
// `../flash_attn/upstream/` (see ../flash_attn/{VERSION, NOTES.md}
// for the vendoring split). The lcd_glue header is structurally
// faithful to upstream FA v2 forward but compiles cleanly under
// the LCD `compute_80` PTX baseline without CUTLASS / CUTE.

#include "../flash_attn/lcd_glue/fwd_sm80.cuh"

extern "C" void flash_attn_fwd_bf16_sm80_launch(
    const __nv_bfloat16 *d_Q,
    const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V_T,
    __nv_bfloat16 *d_O,
    int BH, int Sq, int Sk, float scale, cudaStream_t stream)
{
    flash_attn::flash_attn_fwd_bf16_sm80_launch(
        d_Q, d_K, d_V_T, d_O, BH, Sq, Sk, scale, stream);
}
