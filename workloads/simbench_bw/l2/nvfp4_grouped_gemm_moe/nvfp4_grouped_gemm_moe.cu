// SM120 routed NVFP4 expert projection. Replaces the previous BF16 stand-in.
// Device block16 E2M1 quantization with nonnegative E4M3 scales (tensor scale 1),
// stable top-k softmax routing, per-expert token lists, warp-level block-scaled
// MMA, then weighted combine. No WGMMA, TMEM, tcgen05, or remote DSMEM.
// Fragment layout follows l1/nvfp4_blockscale_gemm and CUDA 13.2 PTX 9.2
// mma.sync.m16n8k64.kind::mxf4nvf4.scale_vec::4X. Not a CUTLASS/TMA kernel.
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

static void cuda_ok(cudaError_t status, const char *stage) {
    if (status != cudaSuccess) {
        fprintf(stderr, "nvfp4_grouped_gemm_moe: %s: %s\n", stage, cudaGetErrorString(status));
        exit(1);
    }
}

// One thread per K-block; weights change layout [expert,K,N] -> [expert,N,K].
__global__ void nvfp4_pack_block16(const __nv_bfloat16 *src, uint8_t *packed,
                                  uint8_t *scales, size_t rows, int K, int N,
                                  bool weight_layout) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int kb = K / 16;
    if (idx >= rows * kb) return;
    size_t row = idx / kb;
    int k = int(idx % kb) * 16;
    float v[16], amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        size_t pos = weight_layout ? (row / N) * size_t(K) * N + (k + j) * size_t(N) + row % N
                                   : row * K + k + j;
        v[j] = __bfloat162float(src[pos]);
        amax = fmaxf(amax, fabsf(v[j]));
    }
    // E4M3 min subnormal prevents zero division for tiny blocks. Zero blocks
    // use scale=1 and quantize to zero. Decode the rounded scale before FP4.
    uint8_t scale = __nv_cvt_float_to_fp8(amax == 0 ? 1.0f : fmaxf(amax / 6.0f, 0.001953125f),
                                         __NV_SATFINITE, __NV_E4M3);
    float s = __half2float(__nv_cvt_fp8_to_halfraw(scale, __NV_E4M3));
    scales[idx] = scale;
    #pragma unroll
    for (int j = 0; j < 16; j += 2) {
        uint8_t lo = __nv_cvt_float_to_fp4(v[j] / s, __NV_E2M1, cudaRoundNearest);
        uint8_t hi = __nv_cvt_float_to_fp4(v[j+1] / s, __NV_E2M1, cudaRoundNearest);
        packed[row * (K/2) + (k+j)/2] = (lo & 15) | (hi << 4);
    }
}

__global__ void nvfp4_route_topk(const float *logits, int *ids, float *weights,
                                int E, int topk) {
    if (threadIdx.x != 0) return;
    int t = blockIdx.x;
    float maxlog = -FLT_MAX, denom = 0;
    for (int e=0; e<E; ++e) maxlog = fmaxf(maxlog, logits[t*E+e]);
    for (int e=0; e<E; ++e) denom += expf(logits[t*E+e]-maxlog);
    float selected_sum = 0;
    for (int r=0; r<topk; ++r) {
        int best = -1;
        float bestp = -1;
        for (int e=0; e<E; ++e) {
            bool chosen = false;
            for (int j=0; j<r; ++j) chosen |= ids[t*topk+j] == e;
            float prob = expf(logits[t*E+e]-maxlog) / denom;
            if (!chosen && (prob > bestp || (prob == bestp && e < best))) {
                best=e; bestp=prob;
            }
        }
        ids[t*topk+r]=best; weights[t*topk+r]=bestp; selected_sum+=bestp;
    }
    for (int r=0; r<topk; ++r) weights[t*topk+r] /= selected_sum;
}

// Deterministic token order without overallocating by a capacity factor or
// dropping overloaded experts. At most T tokens can route to a given expert.
__global__ void nvfp4_dispatch_tokens(const int *ids, int *tokens_by_expert,
                                     int *slot_by_route, int *counts, int T, int topk) {
    if (threadIdx.x != 0) return;
    int e=blockIdx.x, count=0;
    for (int t=0; t<T; ++t) for (int r=0; r<topk; ++r) {
        if (ids[t*topk+r] == e) {
            tokens_by_expert[e*T+count]=t;
            slot_by_route[t*topk+r]=count++;
        }
    }
    counts[e]=count;
}

__global__ void nvfp4_routed_expert_mma(
    const uint8_t *A, const uint8_t *B, const uint8_t *SA, const uint8_t *SB,
    const int *tokens_by_expert, const int *counts, float *partials,
    int T, int N, int K) {
    int e=blockIdx.z, tile=blockIdx.x*16, col=blockIdx.y*8;
    if (tile >= counts[e]) return; // uniform whole-CTA skip, no MMA for empty tiles
    int lane=threadIdx.x, group=lane>>2, tidg=lane&3;
    int slot0=tile+group, slot1=slot0+8;
    int tok0=slot0<counts[e] ? tokens_by_expert[e*T+slot0] : -1;
    int tok1=slot1<counts[e] ? tokens_by_expert[e*T+slot1] : -1;
    size_t brow=size_t(e)*N+col+group;
    int kh=K/2, kb=K/16;
    float d0=0,d1=0,d2=0,d3=0;
    for (int k=0; k<K; k+=64) {
        int off=k/2+tidg*4;
        uint32_t a0=0,a1=0,a2=0,a3=0;
        if (tok0>=0) {
            const uint8_t *p=A+size_t(tok0)*kh+off;
            a0=*reinterpret_cast<const uint32_t*>(p);
            a2=*reinterpret_cast<const uint32_t*>(p+16);
        }
        if (tok1>=0) {
            const uint8_t *p=A+size_t(tok1)*kh+off;
            a1=*reinterpret_cast<const uint32_t*>(p);
            a3=*reinterpret_cast<const uint32_t*>(p+16);
        }
        const uint8_t *q=B+brow*kh+off;
        uint32_t b0=*reinterpret_cast<const uint32_t*>(q);
        uint32_t b1=*reinterpret_cast<const uint32_t*>(q+16);
        uint32_t sa=0, sb=0;
        if (tidg==0) {
            if(tok0>=0) sa=*reinterpret_cast<const uint32_t*>(SA+size_t(tok0)*kb+k/16);
            sb=*reinterpret_cast<const uint32_t*>(SB+brow*kb+k/16);
        } else if(tidg==1 && tok1>=0) {
            sa=*reinterpret_cast<const uint32_t*>(SA+size_t(tok1)*kb+k/16);
        }
        asm volatile(
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
          : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    }
    col+=tidg*2;
    if (tok0>=0) { partials[(size_t(e)*T+slot0)*N+col]=d0; partials[(size_t(e)*T+slot0)*N+col+1]=d1; }
    if (tok1>=0) { partials[(size_t(e)*T+slot1)*N+col]=d2; partials[(size_t(e)*T+slot1)*N+col+1]=d3; }
}

__global__ void nvfp4_combine_experts(const float *partials, const int *ids,
    const int *slots, const float *weights, __nv_bfloat16 *out, int T,int N,int topk) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(T)*N) return;
    int t=i/N, col=i%N;
    float value=0;
    for(int r=0; r<topk; ++r) {
        int e=ids[t*topk+r], s=slots[t*topk+r];
        value+=weights[t*topk+r]*partials[(size_t(e)*T+s)*N+col];
    }
    out[i]=__float2bfloat16(value);
}

extern "C" void nvfp4_grouped_gemm_moe_launch(const float *logits,
    const __nv_bfloat16 *x, const __nv_bfloat16 *w, __nv_bfloat16 *out,
    int T, int E, int topk, int K, int N, cudaStream_t stream) {
    if(T<=0 || T>65535 || E<=0 || E>65535 || topk<=0 || topk>E || K<=0 || K%64 || N<=0 || N%8) {
        fprintf(stderr,"nvfp4_grouped_gemm_moe: require 0<T,E<=65535; 0<topk<=E; K%%64=N%%8=0\n");
        exit(2);
    }
    uint8_t *a,*b,*sa,*sb; int *ids,*slots,*tokenlist,*counts; float *weights,*partials;
    cuda_ok(cudaMalloc(&a,size_t(T)*K/2),"packed A");
    cuda_ok(cudaMalloc(&b,size_t(E)*N*K/2),"packed B");
    cuda_ok(cudaMalloc(&sa,size_t(T)*K/16),"scales A");
    cuda_ok(cudaMalloc(&sb,size_t(E)*N*K/16),"scales B");
    cuda_ok(cudaMalloc(&ids,size_t(T)*topk*sizeof(int)),"route IDs");
    cuda_ok(cudaMalloc(&slots,size_t(T)*topk*sizeof(int)),"route slots");
    cuda_ok(cudaMalloc(&weights,size_t(T)*topk*sizeof(float)),"route weights");
    cuda_ok(cudaMalloc(&tokenlist,size_t(E)*T*sizeof(int)),"expert tokens");
    cuda_ok(cudaMalloc(&counts,size_t(E)*sizeof(int)),"expert counts");
    cuda_ok(cudaMalloc(&partials,size_t(E)*T*N*sizeof(float)),"expert outputs");
    nvfp4_pack_block16<<<(size_t(T)*(K/16)+127)/128,128,0,stream>>>(x,a,sa,T,K,N,false);
    cuda_ok(cudaGetLastError(),"pack A");
    nvfp4_pack_block16<<<(size_t(E)*N*(K/16)+127)/128,128,0,stream>>>(w,b,sb,size_t(E)*N,K,N,true);
    cuda_ok(cudaGetLastError(),"pack B");
    nvfp4_route_topk<<<T,32,0,stream>>>(logits,ids,weights,E,topk);
    cuda_ok(cudaGetLastError(),"route");
    nvfp4_dispatch_tokens<<<E,32,0,stream>>>(ids,tokenlist,slots,counts,T,topk);
    cuda_ok(cudaGetLastError(),"dispatch");
    nvfp4_routed_expert_mma<<<dim3((T+15)/16,N/8,E),32,0,stream>>>(a,b,sa,sb,tokenlist,counts,partials,T,N,K);
    cuda_ok(cudaGetLastError(),"expert MMA");
    nvfp4_combine_experts<<<(size_t(T)*N+127)/128,128,0,stream>>>(partials,ids,slots,weights,out,T,N,topk);
    cuda_ok(cudaGetLastError(),"combine");
    cuda_ok(cudaStreamSynchronize(stream),"routed NVFP4 completion");
    cudaFree(a);cudaFree(b);cudaFree(sa);cudaFree(sb);cudaFree(ids);cudaFree(slots);
    cudaFree(weights);cudaFree(tokenlist);cudaFree(counts);cudaFree(partials);
}
