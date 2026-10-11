// SM120 timing-only interventions; output values are observed, not asserted.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));exit(2);} } while(0)
struct Stamp { unsigned long long start, stop; unsigned sm, slot, end_sm, end_slot, sink; };
// Modes: fma, mad, add, within-warp alternating fma/mad, distinct fma/mad warps, barrier.
template<int Mode, int K, int U>
__global__ void scheduler_probe(Stamp* stamps, unsigned rounds, unsigned active_mask, unsigned seed) {
  unsigned w=threadIdx.x/32, lane=threadIdx.x%32;
  unsigned v[K]; float f[K];
  #pragma unroll
  for(int j=0;j<K;++j){v[j]=seed+threadIdx.x*37+j*17; f[j]=1.0f+0.00001f*(threadIdx.x+j);}
  __syncthreads(); // All allocated warps participate, including inactive controls.
  if(!(active_mask & (1u<<w))) return;
  Stamp s;
  asm volatile("mov.u32 %0,%%smid; mov.u32 %1,%%warpid;" : "=r"(s.sm),"=r"(s.slot));
  asm volatile("mov.u64 %0,%%clock64;" : "=l"(s.start) :: "memory");
  #pragma unroll
  for(int j=0;j<K;++j){v[j]^=unsigned(s.start); f[j]=__uint_as_float(0x3f800000u|((v[j])&0x007fffffu));}
  #pragma unroll 1
  for(unsigned r=0;r<rounds;++r){
    #pragma unroll
    for(int u=0;u<U;++u){
      #pragma unroll
      for(int j=0;j<K;++j){
        if constexpr(Mode==0) asm volatile("fma.rn.f32 %0,%0,0f3f7fffff,0f33800000;" : "+f"(f[j]));
        if constexpr(Mode==1) asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;" : "+r"(v[j]));
        if constexpr(Mode==2) asm volatile("add.u32 %0,%0,1013904223;" : "+r"(v[j]));
        if constexpr(Mode==3){
          asm volatile("fma.rn.f32 %0,%0,0f3f7fffff,0f33800000;" : "+f"(f[j]));
          asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;" : "+r"(v[j]));
        }
        if constexpr(Mode==4){
          if((w&1u)==0) asm volatile("fma.rn.f32 %0,%0,0f3f7fffff,0f33800000;" : "+f"(f[j]));
          else asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;" : "+r"(v[j]));
        }
        if constexpr(Mode==5) asm volatile("add.u32 %0,%0,1013904223;" : "+r"(v[j]));
      }
      if constexpr(Mode==5) __syncthreads();
    }
  }
  unsigned sink=0;
  #pragma unroll
  for(int j=0;j<K;++j){
    if constexpr(Mode==0) sink+=__float_as_uint(f[j]);
    else if constexpr(Mode==3) sink+=v[j]+__float_as_uint(f[j]);
    else if constexpr(Mode==4) sink+=((w&1u)==0?__float_as_uint(f[j]):v[j]);
    else sink+=v[j];
  }
  // Force both timer arms to wait on all consumed accumulators. Fixed tail
  // overhead is not subtracted; slopes use multiple work lengths.
  asm volatile("{.reg .pred p; setp.ne.u32 p,%1,0; @p mov.u64 %0,%%clock64; @!p mov.u64 %0,%%clock64;}"
               : "=l"(s.stop) : "r"(sink) : "memory");
  asm volatile("mov.u32 %0,%%smid; mov.u32 %1,%%warpid;" : "=r"(s.end_sm),"=r"(s.end_slot));
  s.sink=sink;
  if(lane==0) stamps[blockIdx.x*(blockDim.x/32)+w]=s;
}
struct Args {const char* mode; unsigned rounds,warps,mask,blocks,reps,warmups;};
template<int M,int K,int U> void run(Args a){
  Stamp* ptr; size_t size=a.warps*a.blocks;
  CUDA(cudaMalloc(&ptr,size*sizeof(Stamp)));CUDA(cudaMemset(ptr,0,size*sizeof(Stamp)));
  cudaFuncAttributes attr;CUDA(cudaFuncGetAttributes(&attr,scheduler_probe<M,K,U>));
  int occ;CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,scheduler_probe<M,K,U>,a.warps*32,0));
  for(unsigned i=0;i<a.warmups;++i) scheduler_probe<M,K,U><<<a.blocks,a.warps*32>>>(ptr,a.rounds,a.mask,17);
  CUDA(cudaDeviceSynchronize());
  cudaEvent_t begin,end;CUDA(cudaEventCreate(&begin));CUDA(cudaEventCreate(&end));
  std::vector<Stamp> h(size);
  for(unsigned rep=0;rep<a.reps;++rep){
    CUDA(cudaProfilerStart());
    CUDA(cudaEventRecord(begin));
    scheduler_probe<M,K,U><<<a.blocks,a.warps*32>>>(ptr,a.rounds,a.mask,17);
    CUDA(cudaGetLastError());CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));
    CUDA(cudaProfilerStop());float ms;CUDA(cudaEventElapsedTime(&ms,begin,end));
    CUDA(cudaMemcpy(h.data(),ptr,size*sizeof(Stamp),cudaMemcpyDeviceToHost));
    printf("{\"mode\":\"%s\",\"chains\":%d,\"unroll\":%d,\"rounds\":%u,\"warps\":%u,\"active_mask\":%u,\"blocks\":%u,\"rep\":%u,\"registers\":%d,\"local_bytes\":%zu,\"driver_occupancy_blocks\":%d,\"event_ms\":%.9f,\"body_ops_per_active_warp\":%llu,\"barriers_per_active_warp\":%llu,\"warps_observed\":[",
      a.mode,K,U,a.rounds,a.warps,a.mask,a.blocks,rep,attr.numRegs,attr.localSizeBytes,occ,ms,
      (unsigned long long)a.rounds*K*U*(M==3?2:1),(unsigned long long)(M==5?a.rounds*U:0));
    bool comma=false;
    for(unsigned b=0;b<a.blocks;++b) for(unsigned w=0;w<a.warps;++w) if(a.mask&(1u<<w)){
      auto s=h[b*a.warps+w];
      printf("%s{\"block\":%u,\"logical_warp\":%u,\"start\":%llu,\"stop\":%llu,\"elapsed_cycles\":%llu,\"sm\":%u,\"slot\":%u,\"end_sm\":%u,\"end_slot\":%u,\"sink\":%u}",comma?",":"",b,w,s.start,s.stop,s.stop-s.start,s.sm,s.slot,s.end_sm,s.end_slot,s.sink);comma=true;
    }printf("]}\n");
  }
  CUDA(cudaEventDestroy(begin));CUDA(cudaEventDestroy(end));CUDA(cudaFree(ptr));
}
template<int M,int U> void chains(int k,Args a){switch(k){
#define C(k) case k:run<M,k,U>(a);break;
C(1) C(2) C(4) C(8) C(16)
#undef C
 default:fprintf(stderr,"unsupported chain count\n");exit(2);}}
int main(int argc,char**argv){
  if(argc!=10){fprintf(stderr,"usage: scheduler_probe fma|mad|add|mix|warp_mix|barrier chains unroll rounds allocated_warps active_mask blocks reps warmups\n");return 2;}
  unsigned k=strtoul(argv[2],nullptr,0),u=strtoul(argv[3],nullptr,0);
  Args a{argv[1],(unsigned)strtoul(argv[4],nullptr,0),(unsigned)strtoul(argv[5],nullptr,0),(unsigned)strtoul(argv[6],nullptr,0),(unsigned)strtoul(argv[7],nullptr,0),(unsigned)strtoul(argv[8],nullptr,0),(unsigned)strtoul(argv[9],nullptr,0)};
  if(!a.rounds||a.rounds>1048576||!a.warps||a.warps>32||!a.mask|| (a.warps<32 && (a.mask>>a.warps))||!a.blocks||a.blocks>170||!a.reps||a.reps>31||a.warmups>9) return 2;
  if(!strcmp(a.mode,"barrier") && a.mask!=(a.warps==32?0xffffffffu:((1u<<a.warps)-1))) return 2;
  if(u==16){
    if(!strcmp(a.mode,"fma")) chains<0,16>(k,a);
    else if(!strcmp(a.mode,"mad")) chains<1,16>(k,a);
    else if(!strcmp(a.mode,"add")) chains<2,16>(k,a);
    else if(!strcmp(a.mode,"mix")) chains<3,16>(k,a);
    else if(!strcmp(a.mode,"warp_mix")) chains<4,16>(k,a);
    else if(!strcmp(a.mode,"barrier")) chains<5,16>(k,a);
    else return 2;
  }else if(!strcmp(a.mode,"fma") && k==4){
    switch(u){case 64:run<0,4,64>(a);break;case 256:run<0,4,256>(a);break;case 1024:run<0,4,1024>(a);break;default:return 2;}
  }else return 2;
}
