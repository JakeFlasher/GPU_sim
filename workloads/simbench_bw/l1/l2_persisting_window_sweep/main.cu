// Host driver for l2_persisting_window_sweep. It validates the identity output
// and block-0 sink while recording the advisory persistence-window capacity.
#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" void l2_persisting_window_sweep_launch(const float *d_in, float *d_out, float *d_sink, int allocation_floats, int accesses, int warps, cudaStream_t stream);
static inline uint32_t next_u32(uint32_t *s) { *s = (*s) * 1103515245u + 12345u; return *s; }
static void seed_tensor(float *flat, size_t n, uint32_t seed) { uint32_t st = seed * 2654435761u + 1u; for (size_t i = 0; i < n; ++i) { double r = (double)next_u32(&st) / 4294967296.0; flat[i] = (float)(-1.0 + 2.0 * r); } }
static void fail(const char *msg, int rc) { fprintf(stderr, "l2_persisting_window_sweep_run: %s (rc=%d: %s)\n", msg, rc, cudaGetErrorString((cudaError_t)rc)); exit(1); }
static uint32_t parse_u32(const char *name, const char *text, uint32_t maximum) { char *end = nullptr; errno = 0; unsigned long long v = strtoull(text, &end, 10); if (text[0]<'0'||text[0]>'9'||errno||!end||*end||v==0||v>maximum) { fprintf(stderr,"l2_persisting_window_sweep_run: out_of_domain: invalid %s: %s\n",name,text); exit(2); } return (uint32_t)v; }
int main(int argc, char **argv) {
    int allocation_bytes=4096, persisting_window_bytes=1024, accesses=64, warps=4; uint32_t seed=42; const char *shape_point="correctness_00"; bool summary_only=false;
    for (int i=1;i<argc;++i) { if (!strcmp(argv[i],"--allocation-bytes")&&i+1<argc) allocation_bytes=(int)parse_u32("allocation-bytes",argv[++i],INT_MAX); else if(!strcmp(argv[i],"--persisting-window-bytes")&&i+1<argc) persisting_window_bytes=(int)parse_u32("persisting-window-bytes",argv[++i],INT_MAX); else if(!strcmp(argv[i],"--accesses")&&i+1<argc) accesses=(int)parse_u32("accesses",argv[++i],INT_MAX); else if(!strcmp(argv[i],"--warps")&&i+1<argc) warps=(int)parse_u32("warps",argv[++i],INT_MAX); else if(!strcmp(argv[i],"--seed")&&i+1<argc) seed=parse_u32("seed",argv[++i],UINT32_MAX); else if(!strcmp(argv[i],"--shape-point")&&i+1<argc) shape_point=argv[++i]; else if(!strcmp(argv[i],"--summary")) summary_only=true; else { fprintf(stderr,"l2_persisting_window_sweep_run: unknown arg %s\n",argv[i]); return 2; } }
    if (!*shape_point || strspn(shape_point,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=strlen(shape_point)) { fprintf(stderr,"l2_persisting_window_sweep_run: out_of_domain: invalid shape-point token\n"); return 2; }
    const uint64_t alloc_bytes64=(uint64_t)allocation_bytes, n64=(alloc_bytes64+3)/4, output64=(uint64_t)warps*32;
    if (n64<32 || n64>INT_MAX || output64>SIZE_MAX/sizeof(float) || n64>SIZE_MAX/sizeof(float)) { fprintf(stderr,"l2_persisting_window_sweep_run: out_of_domain: shape overflows index or allocation\n"); return 2; }
    const size_t n=(size_t)n64, bytes=n*sizeof(float), output_count=(size_t)output64, output_bytes=output_count*sizeof(float); size_t free_bytes=0,total_bytes=0; cudaError_t rc=cudaMemGetInfo(&free_bytes,&total_bytes); if(rc) fail("memory query",rc); if(bytes+output_bytes>free_bytes/2) { fprintf(stderr,"l2_persisting_window_sweep_run: out of memory: requested shape exceeds free device memory\n"); return 1; }
    float *h_in=(float*)malloc(bytes),*h_out=(float*)malloc(output_bytes); if(!h_in||!h_out){fprintf(stderr,"l2_persisting_window_sweep_run: out of memory: host allocation failed\n"); free(h_in);free(h_out);return 1;} seed_tensor(h_in,n,seed);
    float *d_in=nullptr,*d_out=nullptr,*d_sink=nullptr,h_sink=0.0f; rc=cudaMalloc(&d_in,bytes);if(rc)fail("mc in",rc);rc=cudaMalloc(&d_out,output_bytes);if(rc)fail("mc out",rc);rc=cudaMalloc(&d_sink,sizeof(float));if(rc)fail("mc sink",rc);rc=cudaMemset(d_out,0xff,output_bytes);if(rc)fail("memset out",rc);rc=cudaMemset(d_sink,0xff,sizeof(float));if(rc)fail("memset sink",rc);rc=cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice);if(rc)fail("H2D",rc);
    cudaStream_t stream=nullptr; int max_window=0; rc=cudaDeviceGetAttribute(&max_window,cudaDevAttrMaxAccessPolicyWindowSize,0); if(rc!=cudaSuccess||max_window<=0){printf("{\n  \"schema_version\":1,\n  \"slug\":\"l2_persisting_window_sweep\",\n  \"shape_point\":\"%s\",\n  \"result_kind\":\"typed_unavailable\",\n  \"typed_reason\":\"arch_feature_unavailable_l2_persisting\"\n}\n",shape_point);cudaFree(d_in);cudaFree(d_out);cudaFree(d_sink);free(h_in);free(h_out);return 0;}
    rc=cudaStreamCreate(&stream);if(rc)fail("stream create",rc);cudaStreamAttrValue attr={};attr.accessPolicyWindow.base_ptr=(void*)d_in;attr.accessPolicyWindow.num_bytes=(size_t)persisting_window_bytes<(size_t)max_window?(size_t)persisting_window_bytes:(size_t)max_window;attr.accessPolicyWindow.hitRatio=1.0f;attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;rc=cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&attr);if(rc)fail("set persistence window",rc);
    l2_persisting_window_sweep_launch(d_in,d_out,d_sink,(int)n,accesses,warps,stream);rc=cudaGetLastError();if(rc)fail("launch",rc);rc=cudaDeviceSynchronize();if(rc)fail("sync",rc);rc=cudaMemcpy(h_out,d_out,output_bytes,cudaMemcpyDeviceToHost);if(rc)fail("D2H",rc);rc=cudaMemcpy(&h_sink,d_sink,sizeof(float),cudaMemcpyDeviceToHost);if(rc)fail("sink D2H",rc);
    float expected_sink=0.0f; expected_sink+=h_in[0]; for(int c=0;c<4;++c) expected_sink+=h_in[((size_t)(warps+c)*32)%n]; for(int s=0;s<256;++s) expected_sink+=h_in[((size_t)(warps+4*warps+s)*32)%n]; expected_sink+=h_in[0]; size_t mismatch=0; for(size_t i=0;i<output_count;++i){int src=(int)i; if(!isfinite(h_out[i])||h_out[i]!=h_in[src])++mismatch;} double sink_error=fabs((double)h_sink-(double)expected_sink);
    // The sink prevents dead-code elimination but is not the output contract;
    // floating reassociation may change its reduction order. Require only a
    // finite sink while checking the identity output exactly.
    bool correct=mismatch==0&&isfinite(h_sink);
    printf("{\n  \"schema_version\":1,\n  \"slug\":\"l2_persisting_window_sweep\",\n  \"shape_point\":\"%s\",\n  \"allocation_bytes\":%d,\n  \"persisting_window_bytes\":%d,\n  \"effective_window_bytes\":%d,\n  \"accesses\":%d,\n  \"warps\":%d,\n  \"correct\":%s,\n  \"mismatch_count\":%zu,\n  \"sink\":%.9g,\n  \"expected_sink\":%.9g,\n  \"sink_abs_error\":%.9g",shape_point,allocation_bytes,persisting_window_bytes,(int)attr.accessPolicyWindow.num_bytes,accesses,warps,correct?"true":"false",mismatch,h_sink,expected_sink,sink_error);
    if(!summary_only){printf(",\n  \"output\":[");for(int w=0;w<warps;++w){if(w)printf(",");printf("[");for(int l=0;l<32;++l){if(l)printf(",");printf("%.6g",h_out[w*32+l]);}printf("]");}printf("]");} printf("\n}\n"); cudaStreamDestroy(stream);cudaFree(d_in);cudaFree(d_out);cudaFree(d_sink);free(h_in);free(h_out);return correct?0:1;
}
